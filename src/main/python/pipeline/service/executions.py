'''
Jobs (design/pipeline-service.md §5): a unit of queued work -- a scan, a run of a selection through a stage, or a bulk
accept. One runs at a time, because the index, the review queue and the git working trees each have one writer; the
others wait their turn in the order they came.

A running job is cancelled cooperatively, between titles (`run_stages(should_cancel=)`), so it leaves only whole titles
done. What it reports as it goes -- progress, and the execution events the work list shows as Details -- is kept, redacted,
in a bounded buffer per job and passed to listeners (the HTTP event stream). The last `history_limit` jobs are written to
`<state_dir>/jobs.json`, atomically, as each changes state; a job found queued or running there at start-up is recorded
`interrupted` and not resumed (a run is idempotent: the next one does what is still needed).

A job may be given an active window (design/job-windows.md): the times of day it may run, else the default window's. A
queued job whose window is closed waits while others run; a running run whose window closes stops taking titles, as a
cancel does, and is `paused` until its window opens, when it carries on with the titles it had not started (`parts`). A
paused job, and a queued one waiting for its window, is kept across a restart.

What a job *does* is the `execute` callable (`pipeline.service.work.execute` in the service), so this module knows nothing
of profiles or indexes and is tested with a fake.
'''
import json
import logging
import os
import tempfile
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from datetime import tzinfo
from typing import Any, Callable, Deque, Dict, List, Optional

from model.execution_events import ExecutionEvent, event_text, redact_text, redacted_event
from pipeline.library.bulk import DEFAULT_ACCEPT_THRESHOLD
from pipeline.library.selection import THROUGH, Selection
from pipeline.library.stages import FfmpegProgress, Progress
from pipeline.service.window import ActiveWindow, time_zone_name

logger = logging.getLogger('service_jobs')

KINDS = ('scan', 'run', 'accept')
ORIGINS = ('api', 'schedule')
STATES = ('queued', 'running', 'paused', 'succeeded', 'failed', 'cancelled', 'interrupted')
# a run's execution events that put a title in its hands, and those that end its part in the run (pipeline.library.stages)
IN_HAND_FROM = ('queued', 'stage_queued', 'stage_started')
IN_HAND_UNTIL = ('title_completed', 'failed', 'skipped', 'cancelled')
FINISHED = ('succeeded', 'failed', 'cancelled', 'interrupted')
MAX_JOB_EVENTS = 500
FFMPEG_EVENT_INTERVAL = 5.0   # seconds between kept ffmpeg updates for a title: ffmpeg sends one every 0.5s
HISTORY_FILE = 'jobs.json'
WINDOW_FILE = 'job-window.json'   # the default window
WINDOW_RECHECK_SECONDS = 30.0    # the worker looks again this often while every queued job waits for its window


# --- what a job is asked to do ------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ScanRequest:
    sources: tuple = ()          # names of the profile's sources to list again; empty is all of them
    allow_empty: bool = False    # a source that now lists nothing replaces what it listed (an unmounted share otherwise keeps it)


# how many titles one redesign job may send back: enough for a whole retry phase, few enough to read in its log
MAX_REDESIGN_IDS = 500


def redesign_problem(selection: Selection, through: str, *, raise_it: bool = False) -> str:
    '''
    Why a redesign cannot take this selection, or '': through design, and either 1 to MAX_REDESIGN_IDS ids and nothing
    else, or the titles out of date (design/design-version.md) with any other filter but ids.
    :raises ValueError: with the reason, if `raise_it`.
    '''
    by_ids = 1 <= len(selection.ids) <= MAX_REDESIGN_IDS and selection == Selection(ids=selection.ids)
    by_version = selection.out_of_date is True and not selection.ids
    problem = '' if through == 'design' and (by_ids or by_version) else \
        f'redesign requires through=design and either 1 to {MAX_REDESIGN_IDS} title ids and no other filter, or ' \
        'out_of_date=true and no ids'
    if problem and raise_it:
        raise ValueError(problem)
    return problem


@dataclass(frozen=True)
class RunRequest:
    selection: Selection = Selection()
    through: str = 'design'
    scan_first: bool = True      # list the sources before selecting, so new titles are seen
    retry_failed: bool = False
    unattended: bool = False     # the schedule's: a failed extraction is not tried again until its source or settings change

    redesign: bool = False      # explicitly invalidate the titles' existing designs before running (one run for them all)
    revise_to: str = 'design'   # with redesign: how far back each is sent first ('design', or 'extract' to extract again)
    audio_stream: Optional[int] = None   # with redesign to extract, one title: the audio stream (zero-based) a person chose

    def __post_init__(self):
        if self.redesign:
            redesign_problem(self.selection, self.through, raise_it=True)
        if self.revise_to not in ('design', 'extract'):
            raise ValueError(f"revise_to must be design or extract; got {self.revise_to!r}")
        if self.audio_stream is not None and not (self.redesign and self.revise_to == 'extract'
                                                  and len(self.selection.ids) == 1 and self.audio_stream >= 0):
            raise ValueError('an audio stream (0 or more) is chosen for one title, with redesign and revise_to=extract')
        if self.through not in THROUGH:
            raise ValueError(f"through must be one of {', '.join(THROUGH)}; got {self.through!r}")


@dataclass(frozen=True)
class AcceptRequest:
    selection: Selection = Selection()
    threshold: float = DEFAULT_ACCEPT_THRESHOLD
    dry_run: bool = False

    def __post_init__(self):
        if not 0 <= self.threshold <= 1:
            raise ValueError('threshold must be between 0 and 1')


_REQUEST_KINDS = {ScanRequest: 'scan', RunRequest: 'run', AcceptRequest: 'accept'}


def writes_repositories(request: Any) -> bool:
    ''' True if the request would write to (or push) the catalogue repositories, or accept for a person. '''
    if isinstance(request, RunRequest):
        return request.through in ('publish', 'commit')
    return isinstance(request, AcceptRequest) and not request.dry_run


class RepositoryWritesRefused(PermissionError):
    ''' The service config does not allow publish, commit or bulk accept (allow_repository_writes). '''


class ExecutionNotFound(KeyError):
    pass


class ExecutionFinished(ValueError):
    ''' The job has already finished: there is nothing to cancel. '''


def resumed(request: RunRequest, remaining: tuple) -> RunRequest:
    '''
    What a paused run carries on with: the titles it had not started, through the same stage, without scanning first. A
    redesign sent every title back before it designed any, so it carries on as a plain run (design/job-windows.md §3).
    '''
    return replace(request, selection=Selection(ids=tuple(remaining)), scan_first=False, redesign=False,
                   revise_to='design', audio_stream=None)


def _selection_from_dict(data: Dict[str, Any]) -> Selection:
    ''' A request's selection read back from the history; what a redesign is made with is asked again when it runs. '''
    known = {f.name for f in fields(Selection)} - {'current'}
    values = {k: tuple(v) if isinstance(v, list) else v for k, v in data.items() if k in known}
    return Selection(**values)


def request_from_dict(kind: str, data: Dict[str, Any]) -> Any:
    ''' A job's request read back from the history (a job kept across a restart). :raises: for one that is not. '''
    if kind == 'scan':
        return ScanRequest(sources=tuple(data.get('sources') or ()), allow_empty=bool(data.get('allow_empty')))
    values = dict(data, selection=_selection_from_dict(data.get('selection') or {}))
    if kind == 'run':
        return RunRequest(**values)
    if kind == 'accept':
        return AcceptRequest(**values)
    raise ValueError(f'not a job kind: {kind!r}')


# --- a job -------------------------------------------------------------------------------------------------------------

@dataclass
class Execution:
    id: str
    kind: str
    origin: str
    request: Any                         # ScanRequest | RunRequest | AcceptRequest (a plain dict for one read from history)
    state: str = 'queued'
    submitted_at: float = 0.0
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    progress: Optional[Progress] = None
    result: Any = None
    error: str = ''
    events: Deque[dict] = field(default_factory=lambda: deque(maxlen=MAX_JOB_EVENTS), repr=False)
    cancel_requested: bool = False
    joined_to: Optional[str] = None      # the run job whose extract/design phase this one joined (F5): it ends with it
    ffmpeg_at: Dict[str, float] = field(default_factory=dict, repr=False)   # when each title's last ffmpeg update was kept
    in_hand: Dict[str, str] = field(default_factory=dict, repr=False)   # titles the run has queued or started and not
                                                                         # finished: {id: stage}; not decided on meanwhile
    # design/job-windows.md: the ranges it may run in (None: the default window's; []: any time)
    window: Optional[List[str]] = None
    parts: int = 0                       # how many times it has been started (more than one once paused and resumed)
    paused_at: List[float] = field(default_factory=list)
    part_request: Any = None             # what the next part does, once paused: the titles left (None: the request)
    earlier: List[Any] = field(default_factory=list, repr=False)   # the results of the parts before, as plain data
    done_before: int = 0                 # the progress the parts before made, so the bar carries on from there
    order: int = 0                       # its place in the queue, which a paused job goes back to
    pausing: bool = field(default=False, repr=False)   # its window closed while it ran: it pauses when the part ends

    @property
    def finished(self) -> bool:
        return self.state in FINISHED

    @property
    def current_request(self) -> Any:
        ''' What the part now running is to do. '''
        return self.part_request if self.part_request is not None else self.request


def _plain(value: Any) -> Any:
    ''' A dataclass (a request, a report) as JSON-ready data; anything else as it is. '''
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return value


def execution_to_dict(job: Execution) -> Dict[str, Any]:
    ''' What the history file keeps of a job (its events are not kept). '''
    return {'id': job.id, 'kind': job.kind, 'origin': job.origin, 'state': job.state, 'submitted_at': job.submitted_at,
            'started_at': job.started_at, 'finished_at': job.finished_at, 'request': _plain(job.request),
            'result': _plain(job.result), 'error': job.error, 'progress': _plain(job.progress),
            'joined_to': job.joined_to, 'window': job.window, 'parts': job.parts, 'paused_at': job.paused_at,
            'part_request': _plain(job.part_request), 'earlier': [_plain(r) for r in job.earlier],
            'done_before': job.done_before, 'order': job.order}


def execution_from_dict(data: Dict[str, Any]) -> Execution:
    progress = data.get('progress')
    return Execution(id=data['id'], kind=data['kind'], origin=data.get('origin', 'api'), request=data.get('request'),
               state=data['state'], submitted_at=data.get('submitted_at') or 0.0, started_at=data.get('started_at'),
               finished_at=data.get('finished_at'), progress=Progress(**progress) if progress else None,
               result=data.get('result'), error=data.get('error') or '', joined_to=data.get('joined_to'),
               window=data.get('window'), parts=data.get('parts') or 0, paused_at=list(data.get('paused_at') or ()),
               part_request=data.get('part_request'), earlier=list(data.get('earlier') or ()),
               done_before=data.get('done_before') or 0, order=data.get('order') or 0)


class ExecutionControl:
    ''' What the running job's `execute` is given: whether to stop, and where to report. '''

    def __init__(self, manager: 'ExecutionQueue', job: Execution):
        self.__manager = manager
        self.__job = job

    def cancelled(self) -> bool:
        return self.__job.cancel_requested

    def paused(self) -> bool:
        ''' Whether a run is to stop taking titles because its window has closed (it pauses, and carries on later). '''
        return self.__manager is not None and self.__manager._window_closed(self.__job)

    def should_stop(self) -> bool:
        ''' Cancelled or paused: what a run polls between titles. '''
        return self.cancelled() or self.paused()

    def progress(self, progress) -> None:
        self.__manager._on_progress(self.__job, progress)

    def event(self, event) -> None:
        self.__manager._on_event(self.__job, event)

    def accept_joins(self, join) -> None:
        ''' The run's JoinQueue: a run job submitted while it lasts joins this run instead of waiting for it (F5). '''
        self.__manager._on_join_queue(self.__job, join)


def _cancelled(result: Any) -> bool:
    ''' Whether a result (an object, or a merged result's plain data) says its run was cancelled. '''
    if isinstance(result, dict):
        return bool((result.get('report') or {}).get('cancelled'))
    return bool(getattr(result, 'cancelled', False))


Execute = Callable[[Execution, ExecutionControl], Any]
Listener = Callable[[Execution, dict], None]


class ExecutionQueue:
    '''
    :param execute: does a job: returns its result, or raises (the job then fails with the redacted message).
    :param state_dir: where the history is kept; None keeps it in memory only.
    :param allow_repository_writes: False refuses publish, commit and bulk accept (RepositoryWritesRefused).
    :param failed: says whether a finished job's result means it failed (a title failed, a publish was refused).
    :param remaining: the titles a run's result says it had not started (not_run), so a paused run carries on with them;
        () for none left, None to do the part again as it was (it stopped before it planned any).
    :param merge: one result from the results of a paused job's parts, as plain data, oldest first.
    :param tz: the zone windows are in; None is the service's local time.
    '''

    def __init__(self, execute: Execute, *, state_dir: Optional[str] = None, history_limit: int = 200,
                 allow_repository_writes: bool = False, failed: Callable[[Any], bool] = lambda result: False,
                 clock: Callable[[], float] = time.time,
                 remaining: Callable[[Any], Optional[tuple]] = lambda result: None,
                 merge: Callable[[List[Any]], Any] = lambda parts: parts[-1], tz: Optional[tzinfo] = None):
        self.__execute = execute
        self.__remaining = remaining
        self.__merge = merge
        self.__tz = tz
        self.__state_dir = state_dir
        self.__history_limit = history_limit
        self.__allow_writes = allow_repository_writes
        self.__failed = failed
        self.__clock = clock
        self.__lock = threading.RLock()
        self.__wake = threading.Condition(self.__lock)
        self.__jobs: Dict[str, Execution] = {}      # every job known, oldest first
        self.__queue: Deque[str] = deque()
        self.__listeners: List[Listener] = []
        self.__completed: List[Callable[[Execution], None]] = []
        self.__stopping = False
        self.__seq = 0
        self.__executing: Optional[str] = None           # the job `execute` is doing (joined jobs are running too)
        self.__join_queues: Dict[str, Any] = {}           # executing run job id -> its JoinQueue
        self.__joined: Dict[str, List[str]] = {}          # host run job id -> the jobs that joined it, in order
        self.__order = 0
        self.__default_window: List[str] = self.__load_default_window()
        self.__load_history()
        self.__worker = threading.Thread(target=self.__run, name='pipeline-service-jobs', daemon=True)
        self.__worker.start()

    # --- asking --------------------------------------------------------------------------------------------------------

    def submit(self, request: Any, origin: str = 'api', *, if_idle: bool = False,
               window: Optional[List[str]] = None) -> Optional[Execution]:
        '''
        Queues the job the request describes.
        :param window: the ranges it may run in; None for the default window's, [] for any time.
        :raises ValueError: for a range that is not one.
        :raises RepositoryWritesRefused: for publish, commit or a bulk accept when the config does not allow them.
        :raises RuntimeError: while the service is stopping.
        '''
        kind = _REQUEST_KINDS.get(type(request))
        if kind is None:
            raise TypeError(f'not a job request: {request!r}')
        if origin not in ORIGINS:
            raise ValueError(f'origin must be one of {", ".join(ORIGINS)}')
        if writes_repositories(request) and not self.__allow_writes:
            raise RepositoryWritesRefused('publish, commit and bulk accept are switched off: set allow_repository_writes '
                                          'in the service config to allow them')
        with self.__lock:
            if self.__stopping:
                raise RuntimeError('the service is stopping')
            if if_idle and (self.current is not None or any(self.__runnable(self.__jobs[i]) for i in self.__queue)):
                return None   # a job waiting for its window, or paused, does not count: it is not in the way
            self.__order += 1
            job = Execution(id=str(uuid.uuid4()), kind=kind, origin=origin, request=request, submitted_at=self.__clock(),
                      window=None if window is None else ActiveWindow.of(window, self.__tz).texts(), order=self.__order)
            self.__jobs[job.id] = job
            if self.__join(job):
                self.__save()
                return job
            self.__queue.append(job.id)
            self.__record(job, {'type': 'state', 'state': 'queued'})
            self.__save()
            self.__wake.notify_all()
            return job

    def get(self, job_id: str) -> Execution:
        with self.__lock:
            if job_id not in self.__jobs:
                raise ExecutionNotFound(job_id)
            return self.__jobs[job_id]

    def jobs(self) -> List[Execution]:
        ''' Every job known, newest first. '''
        with self.__lock:
            return list(reversed(self.__jobs.values()))

    @property
    def current(self) -> Optional[Execution]:
        ''' The job being done (a job that joined it is running too, but is done by it). '''
        with self.__lock:
            return self.__jobs.get(self.__executing) if self.__executing else None

    @property
    def queued(self) -> List[Execution]:
        with self.__lock:
            return [self.__jobs[job_id] for job_id in self.__queue]

    @property
    def busy(self) -> bool:
        ''' True while a job runs or waits. '''
        with self.__lock:
            return bool(self.__queue) or self.current is not None

    # --- windows (design/job-windows.md) --------------------------------------------------------------------------------

    def now(self) -> float:
        return self.__clock()

    def window(self, ranges: Optional[List[str]] = None) -> ActiveWindow:
        ''' These ranges (else the default's) as a window in the service's time zone. :raises ValueError: '''
        with self.__lock:
            return ActiveWindow.of(self.__default_window if ranges is None else ranges, self.__tz)

    @property
    def time_zone(self) -> str:
        ''' The name of the zone every window is in: the container's TZ, else what the clock calls it. '''
        return time_zone_name(self.__tz)

    def window_of(self, job: Execution) -> ActiveWindow:
        ''' The job's window, or the default's for a job given none. '''
        with self.__lock:
            return ActiveWindow.of(job.window if job.window is not None else self.__default_window, self.__tz)

    def waiting_until(self, job: Execution) -> Optional[float]:
        ''' When a queued or paused job's closed window next opens; None if it is open, or the job is not waiting. '''
        with self.__lock:
            if job.state not in ('queued', 'paused'):
                return None
            window, now = self.window_of(job), self.__clock()
            return None if window.is_open(now) else window.next_open(now)

    @property
    def default_window(self) -> List[str]:
        with self.__lock:
            return list(self.__default_window)

    def set_default_window(self, ranges: List[str]) -> List[str]:
        ''' The window of every job given none, kept in the state directory. :raises ValueError: for a bad range. '''
        texts = ActiveWindow.of(ranges, self.__tz).texts()
        with self.__lock:
            self.__write_json(WINDOW_FILE, texts)
            self.__default_window = texts
            self.__wake.notify_all()
            return list(texts)

    def set_window(self, job_id: str, ranges: Optional[List[str]]) -> Execution:
        '''
        Changes a queued, running or paused job's window (None: the default's; []: any time). A running run whose new
        window is closed pauses before its next title; a waiting job whose new window is open is taken when the worker is free.
        :raises ExecutionNotFound, ExecutionFinished, ValueError: for a bad range.
        '''
        texts = None if ranges is None else ActiveWindow.of(ranges, self.__tz).texts()
        with self.__lock:
            job = self.get(job_id)
            if job.finished:
                raise ExecutionFinished(f'job {job_id} has already {job.state}')
            job.window = texts
            self.__record(job, {'type': 'window', 'window': texts})
            self.__save()
            self.__wake.notify_all()
            return job

    def __runnable(self, job: Execution) -> bool:
        return self.window_of(job).is_open(self.__clock())

    def _window_closed(self, job: Execution) -> bool:
        ''' ExecutionControl.paused(): a run whose window has closed stops taking titles, and pauses when its part ends. '''
        with self.__lock:
            if job.kind != 'run' or job.cancel_requested or self.__stopping or self.__runnable(job):
                return False
            if not job.pausing:
                job.pausing = True
                self.__record(job, {'type': 'state', 'state': 'pausing'})
            return True

    def cancel(self, job_id: str) -> Execution:
        '''
        A queued job is dropped; a running one stops before its next title.
        :raises ExecutionNotFound: for an unknown id.
        :raises ExecutionFinished: for a job that has already finished.
        '''
        with self.__lock:
            job = self.get(job_id)
            if job.finished:
                raise ExecutionFinished(f'job {job_id} has already {job.state}')
            job.cancel_requested = True
            if job.state in ('queued', 'paused'):
                self.__queue.remove(job_id)
                if job.earlier:   # what its parts did, as a cancel of a running job reports it
                    job.result = self.__merge(list(job.earlier))
                self.__finish(job, 'cancelled')
            elif job.joined_to:   # its titles are in the host's run now: it is recorded cancelled when that ends
                self.__record(job, {'type': 'state', 'state': 'cancelling'})
            else:
                self.__record(job, {'type': 'state', 'state': 'cancelling'})
            return job

    def events(self, job_id: str, after: int = 0) -> List[dict]:
        ''' The job's kept events whose `seq` is above `after`. '''
        with self.__lock:
            return [event for event in self.get(job_id).events if event['seq'] > after]

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        '''
        `listener(job, event)` for every event of every job, on the thread that makes it and under the manager's lock, so it
        must hand the event on and return (the HTTP stream puts it on its event loop); returns an unsubscribe.
        '''
        with self.__lock:
            self.__listeners.append(listener)

        def unsubscribe():
            with self.__lock:
                if listener in self.__listeners:
                    self.__listeners.remove(listener)
        return unsubscribe

    def subscribe_completed(self, listener: Callable[[Execution], None]) -> Callable[[], None]:
        '''Called after a finished job has been saved; listeners should hand work to another thread.'''
        with self.__lock:
            self.__completed.append(listener)

        def unsubscribe():
            with self.__lock:
                if listener in self.__completed:
                    self.__completed.remove(listener)
        return unsubscribe

    def stop(self, grace_seconds: float = 120.0) -> None:
        '''
        No more jobs are taken; the running one is cancelled and waited for up to the grace; queued ones are cancelled,
        except one paused or waiting for its window, which is kept for the next start.
        '''
        with self.__lock:
            self.__stopping = True
            for job_id in list(self.__queue):
                job = self.__jobs[job_id]
                if job.state == 'queued' and self.__runnable(job):
                    self.cancel(job_id)
            running = self.current
            if running is not None:
                running.cancel_requested = True
            self.__wake.notify_all()
        self.__worker.join(grace_seconds)

    # --- the worker -----------------------------------------------------------------------------------------------------

    def __next(self) -> Optional[Execution]:
        ''' The first job in the queue whose window is open now: one waiting for its window keeps its place. '''
        for job_id in self.__queue:
            job = self.__jobs[job_id]
            if self.__runnable(job):
                return job
        return None

    def __wait_seconds(self) -> Optional[float]:
        ''' How long the worker may sleep: until the first waiting window opens (or a little while); None while nothing waits. '''
        if not self.__queue:
            return None
        now = self.__clock()
        opens = [self.window_of(self.__jobs[i]).next_open(now) for i in self.__queue]
        soonest = min((t for t in opens if t is not None), default=now + WINDOW_RECHECK_SECONDS)
        return max(0.05, min(soonest - now, WINDOW_RECHECK_SECONDS))

    def __run(self) -> None:
        while True:
            with self.__lock:
                while not self.__stopping:
                    job = self.__next()
                    if job is not None:
                        break
                    self.__wake.wait(self.__wait_seconds())
                if self.__stopping:
                    return   # anything left in the queue is paused or waiting for its window, and is kept
                self.__queue.remove(job.id)
                job.parts += 1
                job.state, job.pausing = 'running', False
                job.started_at = job.started_at or self.__clock()
                self.__executing = job.id
                self.__record(job, {'type': 'state', 'state': 'running', **({'part': job.parts} if job.parts > 1 else {})})
                self.__save()
            try:
                result = self.__execute(job, ExecutionControl(self, job))
            except Exception as error:   # the job fails; the service does not
                logger.exception('job %s (%s) failed', job.id, job.kind)
                with self.__lock:
                    job.error = redact_text(f'{type(error).__name__}: {error}')
                    if job.earlier:
                        job.result = self.__merge(list(job.earlier))
                    self.__end_joins(job, None, 'failed')
                    self.__finish(job, 'failed')
                continue
            with self.__lock:
                if self.__pause(job, result):
                    continue
                if job.earlier:
                    result = self.__merge(list(job.earlier) + [_plain(result)])
                job.result = result
                cancelled = job.cancel_requested or _cancelled(result)
                state = 'cancelled' if cancelled else 'failed' if self.__failed(result) else 'succeeded'
                self.__end_joins(job, result, state)
                self.__finish(job, state)

    def __pause(self, job: Execution, result: Any) -> bool:
        '''
        A part that stopped because its window closed: the job is paused with the titles it had not started, in its place
        in the queue. :return: False if it is to finish instead (cancelled meanwhile, or nothing left to do).
        '''
        pausing, job.pausing = job.pausing, False
        if not pausing or job.cancel_requested or self.__stopping:
            return False
        remaining = self.__remaining(result)
        if remaining is not None and not remaining:
            return False   # it had started every title before its window closed
        if result is not None:
            job.earlier.append(_plain(result))
            job.done_before = job.progress.done if job.progress else job.done_before
        if remaining is not None:
            job.part_request = resumed(job.current_request, tuple(remaining))
        job.state = 'paused'
        job.paused_at.append(self.__clock())
        job.in_hand.clear()
        self.__end_joins(job, None, 'paused')
        self.__record(job, {'type': 'state', 'state': 'paused', 'resumes_at': self.window_of(job).next_open(self.__clock())})
        self.__requeue(job)
        self.__save()
        return True

    def __requeue(self, job: Execution) -> None:
        ''' Back to its place: before the first job queued after it. '''
        place = next((n for n, i in enumerate(self.__queue) if self.__jobs[i].order > job.order), len(self.__queue))
        self.__queue.insert(place, job.id)
        self.__wake.notify_all()

    # --- joining a run in progress (design/archive/library-sync/worklist-feedback.md F5) ------------------------------

    def _on_join_queue(self, job: Execution, join) -> None:
        with self.__lock:
            self.__join_queues[job.id] = join

    def __join(self, job: Execution) -> bool:
        '''
        A run job through extract or design joins the executing run job's machine phase while it lasts: it is running at
        once, and ends when that run does. :return: False if there is nothing it can join (it is then queued).
        '''
        from pipeline.library.join import JOINABLE, JoinRequest
        host = self.__jobs.get(self.__executing) if self.__executing else None
        join = self.__join_queues.get(host.id) if host is not None else None
        if join is None or not isinstance(job.request, RunRequest) or job.request.through not in JOINABLE or job.request.redesign:
            return False
        if not self.__runnable(job):
            return False   # it waits for its own window
        request = job.request
        if not join.offer(JoinRequest(request.selection, request.through, request.retry_failed, id=job.id)):
            return False
        job.state, job.started_at, job.joined_to = 'running', self.__clock(), host.id
        self.__joined.setdefault(host.id, []).append(job.id)
        self.__record(job, {'type': 'state', 'state': 'running', 'joined': host.id})
        self.__record(host, {'type': 'joined', 'job': job.id})
        return True

    def __end_joins(self, host: Execution, result: Any, state: str) -> None:
        '''
        The jobs that joined `host` end with it, sharing its result -- except one its run did not take (offered once it had
        stopped taking more), which goes back to the front of the queue, unless the host was cancelled. After a failure,
        or a pause, each goes back: a title the host's run did finish is planned out when it runs.
        '''
        self.__join_queues.pop(host.id, None)
        self.__executing = None
        joined = [self.__jobs[i] for i in self.__joined.pop(host.id, []) if i in self.__jobs]
        report = getattr(result, 'report', None)
        not_joined = set(getattr(report, 'not_joined', ()) or ())
        requeue = []
        for job in joined:
            if state == 'paused' and not job.cancel_requested:
                job.state, job.started_at, job.joined_to = 'queued', None, None
                self.__record(job, {'type': 'state', 'state': 'queued'})
                requeue.append(job.id)
            elif job.cancel_requested or (state == 'cancelled'):
                job.result = result
                self.__finish(job, 'cancelled')
            elif result is None or job.id in not_joined:
                job.state, job.started_at, job.joined_to = 'queued', None, None
                self.__record(job, {'type': 'state', 'state': 'queued'})
                requeue.append(job.id)
            else:
                job.result = result
                self.__finish(job, state)
        self.__queue.extendleft(reversed(requeue))
        if requeue:
            self.__wake.notify_all()

    def __finish(self, job: Execution, state: str) -> None:
        job.state, job.finished_at = state, self.__clock()
        self.__record(job, {'type': 'state', 'state': state})
        self.__trim()
        self.__save()
        for listener in list(self.__completed):
            try:
                listener(job)
            except Exception:
                logger.exception('a job completion listener failed')

    def _on_progress(self, job: Execution, progress) -> None:
        with self.__lock:
            if isinstance(progress, Progress):
                if job.done_before:   # a resumed part counts on from where the parts before stopped
                    progress = replace(progress, done=progress.done + job.done_before,
                                       total=progress.total + job.done_before)
                job.progress = progress
                if progress.stage and progress.id:
                    job.in_hand[progress.id] = progress.stage
                self.__record(job, {'type': 'run_progress', 'done': progress.done, 'total': progress.total,
                                    **({'stage': progress.stage, 'title': progress.title, 'title_id': progress.id}
                                       if progress.stage else {})})
            elif isinstance(progress, FfmpegProgress) and self.__keep_ffmpeg(job, progress):
                total = progress.total_micros // 1000 if progress.total_micros > 0 else None
                done = progress.out_time_micros // 1000
                self.__record(job, {'type': 'extract_progress', 'title': progress.title, 'title_id': progress.id,
                                    'done_ms': done, 'total_ms': total,
                                    'percent': min(max(done * 100 // total, 0), 100) if total else None})

    def __keep_ffmpeg(self, job: Execution, progress: FfmpegProgress) -> bool:
        '''
        A title's first and last ffmpeg update, and one every FFMPEG_EVENT_INTERVAL between: every one would flood the
        stream and push the title's own events out of the bounded buffer within minutes.
        '''
        now, last = self.__clock(), job.ffmpeg_at.get(progress.id)
        done = progress.total_micros > 0 and progress.out_time_micros >= progress.total_micros
        if last is not None and not done and now - last < FFMPEG_EVENT_INTERVAL:
            return False
        job.ffmpeg_at[progress.id] = now
        return True

    def _on_event(self, job: Execution, event) -> None:
        if not isinstance(event, ExecutionEvent) or event.kind == 'progress':
            return   # an ffmpeg progress event repeats the ffmpeg update _on_progress already kept (or let go)
        safe = redacted_event(event)
        with self.__lock:
            if event.title_id and event.kind in IN_HAND_FROM:
                job.in_hand[event.title_id] = event.stage or job.in_hand.get(event.title_id, '')
            elif event.title_id and event.kind in IN_HAND_UNTIL:
                job.in_hand.pop(event.title_id, None)
            self.__record(job, {'type': 'event', 'title_id': safe.title_id, 'stage': safe.stage, 'kind': safe.kind,
                                'text': event_text(safe)})

    def titles_in_hand(self) -> Dict[str, str]:
        '''
        The titles the running job (and any joined to it) has queued or started and not finished, {id: stage}: a person's
        decision on one would be overwritten by the design in flight (design/web-app.md §3).
        '''
        with self.__lock:
            result: Dict[str, str] = {}
            for job in self.__jobs.values():
                if job.state == 'running':
                    result.update(job.in_hand)
            return result

    def has_job(self, job_id: str) -> bool:
        ''' Whether `job_id` is one of this service's jobs (a lease holder that is not is the work list or the CLI). '''
        with self.__lock:
            return job_id in self.__jobs

    def __record(self, job: Execution, event: dict) -> None:
        self.__seq += 1
        event = {'seq': self.__seq, 'at': self.__clock(), **event}
        job.events.append(event)
        for listener in list(self.__listeners):
            try:
                listener(job, event)
            except Exception:
                logger.exception('a job listener failed')

    # --- history --------------------------------------------------------------------------------------------------------

    def __trim(self) -> None:
        finished = [job_id for job_id, job in self.__jobs.items() if job.finished]
        for job_id in finished[:max(0, len(finished) - self.__history_limit)]:
            del self.__jobs[job_id]

    def __write_json(self, name: str, value: Any) -> None:
        ''' Atomically, in the state directory (nowhere without one). :raises OSError: '''
        if not self.__state_dir:
            return
        os.makedirs(self.__state_dir, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f'.{name.split(".")[0]}-', suffix='.json', dir=self.__state_dir)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(value, f, default=str)
            os.replace(temporary, os.path.join(self.__state_dir, name))
        except BaseException:
            if os.path.exists(temporary):
                os.unlink(temporary)
            raise

    def __save(self) -> None:
        try:
            self.__write_json(HISTORY_FILE, [execution_to_dict(job) for job in self.__jobs.values()])
        except OSError:
            logger.exception('the job history could not be written to %s', self.__state_dir)

    def __load_default_window(self) -> List[str]:
        if not self.__state_dir:
            return []
        path = os.path.join(self.__state_dir, WINDOW_FILE)
        try:
            with open(path, encoding='utf-8') as f:
                return ActiveWindow.of(json.load(f), self.__tz).texts()
        except FileNotFoundError:
            return []
        except (OSError, ValueError, TypeError):
            logger.exception('the default job window at %s cannot be read; jobs may run any time', path)
            return []

    def __keep(self, job: Execution) -> bool:
        '''
        Whether a job the service stopped before it finished carries on after the restart: one paused, or queued while its
        window was closed, whose request can be read back. A job that was running is not (the next run redoes its title).
        '''
        if job.state == 'paused' or (job.state == 'queued' and not self.window_of(job).is_open(self.__clock())):
            try:
                job.request = request_from_dict(job.kind, job.request or {})
                if job.part_request is not None:
                    job.part_request = request_from_dict(job.kind, job.part_request)
                return True
            except (TypeError, ValueError, KeyError, AttributeError):
                logger.warning('job %s cannot be read back to carry on with, and is recorded interrupted', job.id)
        return False

    def __load_history(self) -> None:
        if not self.__state_dir:
            return
        path = os.path.join(self.__state_dir, HISTORY_FILE)
        try:
            with open(path, encoding='utf-8') as f:
                saved = json.load(f)
        except FileNotFoundError:
            return
        except (OSError, ValueError):
            logger.exception('the job history at %s cannot be read; starting without it', path)
            return
        now = self.__clock()
        for data in saved if isinstance(saved, list) else []:
            try:
                job = execution_from_dict(data)
            except (KeyError, TypeError):
                logger.warning('a job in %s could not be read, and is left out', path)
                continue
            if not job.finished and not self.__keep(job):   # the service stopped while it waited or ran
                job.error = f'the service stopped while the job was {job.state}'
                job.state, job.finished_at = 'interrupted', now
                if job.earlier:
                    job.result = self.__merge(list(job.earlier))
            self.__jobs[job.id] = job
        self.__order = max((job.order for job in self.__jobs.values()), default=0)
        for job in sorted((j for j in self.__jobs.values() if not j.finished), key=lambda j: j.order):
            self.__queue.append(job.id)
        self.__trim()
        self.__save()
