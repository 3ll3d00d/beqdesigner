'''
Jobs (design/pipeline-service.md §5): a unit of queued work -- a scan, a run of a selection through a stage, or a bulk
accept. One runs at a time, because the index, the review queue and the git working trees each have one writer; the
others wait their turn in the order they came.

A running job is cancelled cooperatively, between titles (`run_stages(should_cancel=)`), so it leaves only whole titles
done. What it reports as it goes -- progress, and the execution events the work list shows as Details -- is kept, redacted,
in a bounded buffer per job and passed to listeners (the HTTP event stream). The last `history_limit` jobs are written to
`<state_dir>/jobs.json`, atomically, as each changes state; a job found queued or running there at start-up is recorded
`interrupted` and not resumed (a run is idempotent: the next one does what is still needed).

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
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Callable, Deque, Dict, List, Optional

from model.execution_events import ExecutionEvent, event_text, redact_text, redacted_event
from pipeline.library.bulk import DEFAULT_ACCEPT_THRESHOLD
from pipeline.library.selection import THROUGH, Selection
from pipeline.library.stages import FfmpegProgress, Progress

logger = logging.getLogger('service_jobs')

KINDS = ('scan', 'run', 'accept')
ORIGINS = ('api', 'schedule')
STATES = ('queued', 'running', 'succeeded', 'failed', 'cancelled', 'interrupted')
FINISHED = ('succeeded', 'failed', 'cancelled', 'interrupted')
MAX_JOB_EVENTS = 500
HISTORY_FILE = 'jobs.json'


# --- what a job is asked to do ------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ScanRequest:
    sources: tuple = ()          # names of the profile's sources to list again; empty is all of them
    allow_empty: bool = False    # a source that now lists nothing replaces what it listed (an unmounted share otherwise keeps it)


@dataclass(frozen=True)
class RunRequest:
    selection: Selection = Selection()
    through: str = 'design'
    scan_first: bool = True      # list the sources before selecting, so new titles are seen
    retry_failed: bool = False

    def __post_init__(self):
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


class JobNotFound(KeyError):
    pass


class JobFinished(ValueError):
    ''' The job has already finished: there is nothing to cancel. '''


# --- a job -------------------------------------------------------------------------------------------------------------

@dataclass
class Job:
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

    @property
    def finished(self) -> bool:
        return self.state in FINISHED


def _plain(value: Any) -> Any:
    ''' A dataclass (a request, a report) as JSON-ready data; anything else as it is. '''
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return value


def job_to_dict(job: Job) -> Dict[str, Any]:
    ''' What the history file keeps of a job (its events are not kept). '''
    return {'id': job.id, 'kind': job.kind, 'origin': job.origin, 'state': job.state, 'submitted_at': job.submitted_at,
            'started_at': job.started_at, 'finished_at': job.finished_at, 'request': _plain(job.request),
            'result': _plain(job.result), 'error': job.error, 'progress': _plain(job.progress)}


def job_from_dict(data: Dict[str, Any]) -> Job:
    progress = data.get('progress')
    return Job(id=data['id'], kind=data['kind'], origin=data.get('origin', 'api'), request=data.get('request'),
               state=data['state'], submitted_at=data.get('submitted_at') or 0.0, started_at=data.get('started_at'),
               finished_at=data.get('finished_at'), progress=Progress(**progress) if progress else None,
               result=data.get('result'), error=data.get('error') or '')


class JobControl:
    ''' What the running job's `execute` is given: whether to stop, and where to report. '''

    def __init__(self, manager: 'JobManager', job: Job):
        self.__manager = manager
        self.__job = job

    def cancelled(self) -> bool:
        return self.__job.cancel_requested

    def progress(self, progress) -> None:
        self.__manager._on_progress(self.__job, progress)

    def event(self, event) -> None:
        self.__manager._on_event(self.__job, event)


Execute = Callable[[Job, JobControl], Any]
Listener = Callable[[Job, dict], None]


class JobManager:
    '''
    :param execute: does a job: returns its result, or raises (the job then fails with the redacted message).
    :param state_dir: where the history is kept; None keeps it in memory only.
    :param allow_repository_writes: False refuses publish, commit and bulk accept (RepositoryWritesRefused).
    :param failed: says whether a finished job's result means it failed (a title failed, a publish was refused).
    '''

    def __init__(self, execute: Execute, *, state_dir: Optional[str] = None, history_limit: int = 200,
                 allow_repository_writes: bool = False, failed: Callable[[Any], bool] = lambda result: False,
                 clock: Callable[[], float] = time.time):
        self.__execute = execute
        self.__state_dir = state_dir
        self.__history_limit = history_limit
        self.__allow_writes = allow_repository_writes
        self.__failed = failed
        self.__clock = clock
        self.__lock = threading.RLock()
        self.__wake = threading.Condition(self.__lock)
        self.__jobs: Dict[str, Job] = {}      # every job known, oldest first
        self.__queue: Deque[str] = deque()
        self.__listeners: List[Listener] = []
        self.__stopping = False
        self.__seq = 0
        self.__load_history()
        self.__worker = threading.Thread(target=self.__run, name='pipeline-service-jobs', daemon=True)
        self.__worker.start()

    # --- asking --------------------------------------------------------------------------------------------------------

    def submit(self, request: Any, origin: str = 'api') -> Job:
        '''
        Queues the job the request describes.
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
            job = Job(id=str(uuid.uuid4()), kind=kind, origin=origin, request=request, submitted_at=self.__clock())
            self.__jobs[job.id] = job
            self.__queue.append(job.id)
            self.__record(job, {'type': 'state', 'state': 'queued'})
            self.__save()
            self.__wake.notify_all()
            return job

    def get(self, job_id: str) -> Job:
        with self.__lock:
            if job_id not in self.__jobs:
                raise JobNotFound(job_id)
            return self.__jobs[job_id]

    def jobs(self) -> List[Job]:
        ''' Every job known, newest first. '''
        with self.__lock:
            return list(reversed(self.__jobs.values()))

    @property
    def current(self) -> Optional[Job]:
        with self.__lock:
            return next((job for job in self.__jobs.values() if job.state == 'running'), None)

    @property
    def queued(self) -> List[Job]:
        with self.__lock:
            return [self.__jobs[job_id] for job_id in self.__queue]

    @property
    def busy(self) -> bool:
        ''' True while a job runs or waits. '''
        with self.__lock:
            return bool(self.__queue) or self.current is not None

    def cancel(self, job_id: str) -> Job:
        '''
        A queued job is dropped; a running one stops before its next title.
        :raises JobNotFound: for an unknown id.
        :raises JobFinished: for a job that has already finished.
        '''
        with self.__lock:
            job = self.get(job_id)
            if job.finished:
                raise JobFinished(f'job {job_id} has already {job.state}')
            job.cancel_requested = True
            if job.state == 'queued':
                self.__queue.remove(job_id)
                self.__finish(job, 'cancelled')
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

    def stop(self, grace_seconds: float = 120.0) -> None:
        ''' No more jobs are taken; the running one is cancelled and waited for up to the grace; queued ones are cancelled. '''
        with self.__lock:
            self.__stopping = True
            for job_id in list(self.__queue):
                self.cancel(job_id)
            running = self.current
            if running is not None:
                running.cancel_requested = True
            self.__wake.notify_all()
        self.__worker.join(grace_seconds)

    # --- the worker -----------------------------------------------------------------------------------------------------

    def __run(self) -> None:
        while True:
            with self.__lock:
                while not self.__queue and not self.__stopping:
                    self.__wake.wait()
                if self.__stopping and not self.__queue:
                    return
                job = self.__jobs[self.__queue.popleft()]
                job.state, job.started_at = 'running', self.__clock()
                self.__record(job, {'type': 'state', 'state': 'running'})
                self.__save()
            try:
                result = self.__execute(job, JobControl(self, job))
            except Exception as error:   # the job fails; the service does not
                logger.exception('job %s (%s) failed', job.id, job.kind)
                with self.__lock:
                    job.error = redact_text(f'{type(error).__name__}: {error}')
                    self.__finish(job, 'failed')
                continue
            with self.__lock:
                job.result = result
                cancelled = job.cancel_requested or bool(getattr(result, 'cancelled', False))
                self.__finish(job, 'cancelled' if cancelled else 'failed' if self.__failed(result) else 'succeeded')

    def __finish(self, job: Job, state: str) -> None:
        job.state, job.finished_at = state, self.__clock()
        self.__record(job, {'type': 'state', 'state': state})
        self.__trim()
        self.__save()

    def _on_progress(self, job: Job, progress) -> None:
        with self.__lock:
            if isinstance(progress, Progress):
                job.progress = progress
                self.__record(job, {'type': 'progress', **asdict(progress)})
            elif isinstance(progress, FfmpegProgress):
                self.__record(job, {'type': 'ffmpeg', **asdict(progress)})

    def _on_event(self, job: Job, event) -> None:
        if not isinstance(event, ExecutionEvent):
            return
        safe = redacted_event(event)
        with self.__lock:
            self.__record(job, {'type': 'event', 'title_id': safe.title_id, 'stage': safe.stage, 'kind': safe.kind,
                                'text': event_text(safe)})

    def __record(self, job: Job, event: dict) -> None:
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

    def __save(self) -> None:
        if not self.__state_dir:
            return
        try:
            os.makedirs(self.__state_dir, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix='.jobs-', suffix='.json', dir=self.__state_dir)
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump([job_to_dict(job) for job in self.__jobs.values()], f, default=str)
                os.replace(temporary, os.path.join(self.__state_dir, HISTORY_FILE))
            except BaseException:
                if os.path.exists(temporary):
                    os.unlink(temporary)
                raise
        except OSError:
            logger.exception('the job history could not be written to %s', self.__state_dir)

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
                job = job_from_dict(data)
            except (KeyError, TypeError):
                logger.warning('a job in %s could not be read, and is left out', path)
                continue
            if not job.finished:   # the service stopped while it waited or ran
                job.error = f'the service stopped while the job was {job.state}'
                job.state, job.finished_at = 'interrupted', now
            self.__jobs[job.id] = job
        self.__trim()
        self.__save()
