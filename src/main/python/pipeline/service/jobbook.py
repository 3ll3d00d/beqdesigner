'''
Jobs (design/jobs-and-executions.md §2.1): the definition of some work -- a scan, a run of a selection through a stage,
or a bulk accept -- done once or every N minutes, within its window, until it is removed. Each continuous run that does
it is an execution (pipeline.service.executions), which the book submits to the ExecutionQueue when the job is due.

- A **one-off** job is executed until it has done everything, each execution started while its window is open: one that
  stops (its window closed, or the job was paused) leaves the titles it had not started in `remaining`, and the next
  execution does those. It is `done` when an
  execution ends any other way, and `removed` when a person cancels its execution.
- A **recurring** job (a run through extract or design, `every` N minutes) is executed that long after its last execution
  finished, within its window. A tick due while another execution runs or waits is skipped (`last_skip` busy), and one
  that designs is skipped while the designer does not answer (tried again within DESIGNER_RETRY_MINUTES, one
  notification per outage). One whose filter lists ids is `done` once an execution leaves none of them to do.

Jobs are kept in `<state_dir>/jobs.json`, atomically. `migrate()` turns an older service's `schedule.json` and job
history into a recurring job and executions, once, before the queue reads its history.
'''
import json
import logging
import os
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from queue import SimpleQueue
from typing import Any, Callable, Dict, List, Optional, Tuple

from pipeline.service.executions import Execution, ExecutionFinished, ExecutionQueue, RunRequest, _plain, \
    request_from_dict, resumed

logger = logging.getLogger('service_jobbook')

JOBS_FILE = 'jobs.json'
EXECUTIONS_FILE = 'executions.json'
SCHEDULE_FILE = 'schedule.json'
FORMAT = 2                     # jobs.json as a book; an older service kept its job history there, as a list
JOB_STATES = ('active', 'paused', 'done', 'removed')
MIN_EVERY_MINUTES = 5
DESIGNER_RETRY_MINUTES = 5     # a tick skipped because the designer did not answer is tried again this soon (or sooner)
SCHEDULE_NAME = 'Extract & design new titles'
UNSET: Any = object()


class JobNotFound(KeyError):
    pass


class JobRefused(ValueError):
    ''' What was asked cannot be done to the job in the state it is in (HTTP 409). '''


class JobBusy(RuntimeError):
    ''' Run now: another execution is running or waiting to run. '''


@dataclass
class Job:
    id: str
    name: str
    kind: str
    request: Any                              # ScanRequest | RunRequest | AcceptRequest
    every: Optional[int] = None               # minutes between executions; None: once
    window: Optional[List[str]] = None        # None: the default window's; []: any time
    state: str = 'active'
    created_at: float = 0.0
    remaining: Optional[Tuple[str, ...]] = None   # a one-off run's titles not started yet; None: its request as it is
    executions: List[str] = field(default_factory=list)   # its executions' ids, oldest first
    pending: Optional[str] = None             # its execution queued or running
    due_at: Optional[float] = None            # a recurring job: when its next execution is due (its window aside)
    last_finished_at: Optional[float] = None
    done: int = 0                             # title-stages its finished executions did
    total: Optional[int] = None               # a one-off: how many its first execution set out to do
    last_skip: str = ''
    ended: str = ''                           # why it is done, when it ended by itself
    removing: bool = field(default=False, repr=False)

    @property
    def recurring(self) -> bool:
        return self.every is not None

    @property
    def origin(self) -> str:
        ''' Its executions' origin, which notification targets choose by: the schedule's, or a person's. '''
        return 'schedule' if self.recurring else 'api'

    def next_request(self) -> Any:
        ''' What its next execution does: a one-off run that stopped carries on with what it had not started. '''
        if not self.recurring and self.remaining is not None and isinstance(self.request, RunRequest):
            return resumed(self.request, self.remaining)
        return self.request


def default_name(kind: str, request: Any, every: Optional[int] = None) -> str:
    ''' What a job is called when nobody names it: what it does. '''
    if kind == 'scan':
        sources = getattr(request, 'sources', ())
        return f'List {", ".join(sources)} again' if sources else 'List the sources again'
    if kind == 'accept':
        return 'Accept the top picks' + (' (dry run)' if getattr(request, 'dry_run', False) else '')
    selection = request.selection
    if request.redesign:
        count = len(selection.ids)
        return f'Redesign {count} title{"s" if count != 1 else ""}' if count else 'Redesign the out-of-date designs'
    what = f'{len(selection.ids)} titles' if selection.ids else 'titles'
    return f'{"Every " + str(every) + " min: " if every else ""}Run {what} through {request.through}'


def left_to_do(execution: Execution, retry_failed: bool) -> Optional[int]:
    '''
    How many of a finished run's titles it left for a later one: unavailable, not reached, or (when failures are retried)
    failed. None if it did not finish its selection: it was cancelled or stopped, or it raised.
    '''
    from pipeline.service.models import run_result
    if execution.result is None or execution.state in ('cancelled', 'stopped', 'interrupted'):
        return None
    result = run_result(execution.result)
    if result.cancelled or result.stopped:
        return None
    return len(result.unavailable) + len(result.not_run) + (len(result.failed) if retry_failed else 0)


def recurring_problem(kind: str, request: Any) -> str:
    ''' Why a job cannot recur, or '': only a run through extract or design does (nothing is published by itself). '''
    if kind != 'run' or request.through not in ('extract', 'design') or request.redesign:
        return 'a recurring job is a run through extract or design (not a redesign, a scan or an accept)'
    return ''


def job_to_dict(job: Job) -> Dict[str, Any]:
    return {'id': job.id, 'name': job.name, 'kind': job.kind, 'request': _plain(job.request), 'every': job.every,
            'window': job.window, 'state': job.state, 'created_at': job.created_at,
            'remaining': list(job.remaining) if job.remaining is not None else None, 'executions': job.executions,
            'pending': job.pending, 'due_at': job.due_at, 'last_finished_at': job.last_finished_at, 'done': job.done,
            'total': job.total, 'last_skip': job.last_skip, 'ended': job.ended}


def job_from_dict(data: Dict[str, Any]) -> Job:
    remaining = data.get('remaining')
    return Job(id=data['id'], name=data.get('name') or '', kind=data['kind'],
               request=request_from_dict(data['kind'], data.get('request') or {}), every=data.get('every'),
               window=data.get('window'), state=data.get('state') or 'active', created_at=data.get('created_at') or 0.0,
               remaining=tuple(remaining) if remaining is not None else None,
               executions=list(data.get('executions') or ()), pending=data.get('pending'), due_at=data.get('due_at'),
               last_finished_at=data.get('last_finished_at'), done=data.get('done') or 0, total=data.get('total'),
               last_skip=data.get('last_skip') or '', ended=data.get('ended') or '')


def _write_json(state_dir: str, name: str, value: Any) -> None:
    ''' Atomically, in the state directory. :raises OSError: '''
    os.makedirs(state_dir, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{name.split(".")[0]}-', suffix='.json', dir=state_dir)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(value, f, default=str)
        os.replace(temporary, os.path.join(state_dir, name))
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _read_json(path: str) -> Any:
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


class JobBook:
    '''
    :param queue: where executions are submitted.
    :param designer: asked before a recurring run that designs: why the designer cannot be used, or '' if it can.
    :param on_designer_down: told the reason once when the designer stops answering, not at every skipped tick.
    :param start: False for tests, which call tick().
    '''

    def __init__(self, queue: ExecutionQueue, state_dir: Optional[str] = None, *, clock: Callable[[], float] = time.time,
                 start: bool = True, designer: Optional[Callable[[], str]] = None,
                 on_designer_down: Optional[Callable[[str], None]] = None):
        self.queue, self.state_dir, self.clock = queue, state_dir, clock
        self.designer, self.on_designer_down = designer, on_designer_down
        self.designer_down = ''
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.stopped = False
        self.finished: SimpleQueue = SimpleQueue()
        self.jobs: Dict[str, Job] = {}
        self.__load()
        self.unsubscribe = queue.subscribe_completed(self.__on_finished)
        self.thread: Optional[threading.Thread] = None
        if start:
            self.thread = threading.Thread(target=self.__loop, name='pipeline-service-jobbook', daemon=True)
            self.thread.start()

    # --- asking ---------------------------------------------------------------------------------------------------------

    def create(self, kind: str, request: Any, *, name: str = '', every: Optional[int] = None,
               window: Optional[List[str]] = None) -> Job:
        '''
        A new job; a one-off is queued at once while its window is open (else when it opens).
        :raises ValueError: a bad window, interval, or a job that cannot recur.
        :raises RepositoryWritesRefused: for publish, commit or a bulk accept when the config does not allow them.
        '''
        window = self.__window_texts(window)
        if every is not None:
            if every < MIN_EVERY_MINUTES:
                raise ValueError(f'a recurring job runs at most every {MIN_EVERY_MINUTES} minutes')
            problem = recurring_problem(kind, request)
            if problem:
                raise ValueError(problem)
        with self.lock:
            now = self.clock()
            job = Job(id=str(uuid.uuid4()), name=name.strip() or default_name(kind, request, every), kind=kind,
                      request=request, every=every, window=window, created_at=now,
                      due_at=now + every * 60 if every is not None else None)
            self.queue.check(request)   # refused here (repository writes) before the job is kept
            if every is None and self.__window_open(job):
                self.__submit(job)
            self.jobs[job.id] = job
            self.__save()
            self.wake.set()
            return job

    def get(self, job_id: str) -> Job:
        with self.lock:
            self.__drain()   # what it says reflects every execution that has ended
            if job_id not in self.jobs:
                raise JobNotFound(job_id)
            return self.jobs[job_id]

    def all(self) -> List[Job]:
        ''' Every job, newest first. '''
        with self.lock:
            self.__drain()
            return sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)

    def pause(self, job_id: str) -> Job:
        ''' Nothing more starts; its running execution stops before its next title. :raises JobRefused: '''
        with self.lock:
            self.__drain()
            job = self.__live(job_id)
            if job.state == 'paused':
                return job
            job.state = 'paused'
            self.__halt(job, 'paused')
            self.__save()
            return job

    def resume(self, job_id: str) -> Job:
        ''' Active again: a one-off carries on with what it had left; a recurring job is due at once. '''
        with self.lock:
            self.__drain()
            job = self.__live(job_id)
            if job.state == 'active':
                return job
            job.state, job.last_skip = 'active', ''
            if job.recurring:
                job.due_at = self.clock()
            self.__save()
            self.wake.set()
            return job

    def run_now(self, job_id: str) -> Execution:
        ''' One execution of a recurring job now. :raises JobRefused, JobBusy: if another is running or waiting. '''
        with self.lock:
            self.__drain()
            job = self.__live(job_id)
            if not job.recurring:
                raise JobRefused('only a recurring job is run now: a one-off runs by itself')
            if job.pending is not None:
                raise JobBusy('it is running, or waiting to run, already')
            execution = self.__submit(job, if_idle=True)
            if execution is None:
                raise JobBusy('another execution is running or waiting to run')
            job.last_skip = ''
            self.__save()
            return execution

    def edit(self, job_id: str, *, name: Any = UNSET, window: Any = UNSET, every: Any = UNSET,
             request: Any = UNSET) -> Job:
        '''
        The name and window of any job not done; a recurring job's interval and request too, from its next execution (a
        one-off's request is fixed). A running execution follows a window change at its next title.
        :raises JobRefused, ValueError:
        '''
        with self.lock:
            self.__drain()
            job = self.__live(job_id)
            texts = self.__window_texts(window) if window is not UNSET else UNSET
            if every is not UNSET or request is not UNSET:
                if not job.recurring:
                    raise JobRefused("a one-off job's request is fixed: make another job")
                if every is not UNSET and (every is None or every < MIN_EVERY_MINUTES):
                    raise ValueError(f'a recurring job runs at most every {MIN_EVERY_MINUTES} minutes')
                if request is not UNSET:
                    problem = recurring_problem(job.kind, request)
                    if problem:
                        raise ValueError(problem)
            if name is not UNSET:
                job.name = (name or '').strip() or default_name(job.kind, job.request, job.every)
            if texts is not UNSET:
                job.window = texts
                if job.pending is not None:
                    try:
                        self.queue.set_window(job.pending, texts)
                    except ExecutionFinished:
                        pass
            if request is not UNSET:
                job.request = request
            if every is not UNSET:
                job.every = every
                job.due_at = (job.last_finished_at or job.created_at) + every * 60
            self.__save()
            self.wake.set()
            return job

    def remove(self, job_id: str) -> Job:
        ''' Removed: nothing more starts, and its running execution stops before its next title. '''
        with self.lock:
            self.__drain()
            job = self.get(job_id)
            if job.state == 'removed':
                return job
            job.state, job.removing = 'removed', True
            self.__halt(job, 'removed')
            self.__save()
            return job

    def next_at(self, job: Job) -> Optional[float]:
        '''
        When its next execution may start: a recurring job's next due time, moved to its window's next opening; a waiting
        one-off's window opening (now, if open). None for a job paused, done or removed, or one already running.
        '''
        with self.lock:
            if job.state != 'active':
                return None
            now = self.clock()
            if job.pending is not None:
                try:
                    execution = self.queue.get(job.pending)
                except KeyError:
                    return None
                return self.queue.waiting_until(execution) if execution.state == 'queued' else None
            at = max(job.due_at or now, now) if job.recurring else now
            window = self.queue.window(job.window)
            return at if window.is_open(at) else window.next_open(at)

    def progress(self, job: Job) -> Tuple[int, Optional[int]]:
        ''' A one-off's title-stages done over every execution so far, and how many it set out to do. '''
        with self.lock:
            done = job.done
            if job.pending is not None:
                try:
                    running = self.queue.get(job.pending)
                except KeyError:
                    running = None
                if running is not None and running.progress is not None:
                    done += running.progress.done
            return done, job.total

    def stop(self) -> None:
        self.stopped = True
        self.wake.set()
        if self.thread:
            self.thread.join(timeout=2)
        self.unsubscribe()

    # --- when each job is due -------------------------------------------------------------------------------------------

    def tick(self) -> None:
        ''' Starts what is due: a one-off with nothing running, a recurring job whose time has come within its window. '''
        with self.lock:
            self.__drain()
            for job in list(self.jobs.values()):
                if job.state == 'active' and job.pending is None and not job.recurring and self.__window_open(job):
                    try:
                        self.__submit(job)
                    except Exception as error:   # refused since it was made (the service is stopping): it waits
                        logger.warning('job %s could not be started: %s', job.id, error)
            due = [job for job in self.jobs.values() if self.__due(job)]
        for job in due:
            self.__tick_recurring(job)

    def __due(self, job: Job) -> bool:
        if job.state != 'active' or job.pending is not None or not job.recurring:
            return False
        return job.due_at is not None and self.clock() >= job.due_at and self.__window_open(job)

    def __window_open(self, job: Job) -> bool:
        return self.queue.window(job.window).is_open(self.clock())

    def __tick_recurring(self, job: Job) -> None:
        ask = self.designer is not None and job.request.through != 'extract'
        reason = self.designer() if ask else ''   # not holding the lock: it may take the designer's timeout
        down = ''
        with self.lock:
            if not self.__due(job):
                return
            now = self.clock()
            if reason:
                job.last_skip = f'designer unavailable: {reason}'
                job.due_at = now + min(job.every, DESIGNER_RETRY_MINUTES) * 60
                down = reason if not self.designer_down else ''
                self.designer_down = reason
            else:
                if ask:
                    self.designer_down = ''
                if self.__submit(job, if_idle=True) is None:
                    job.last_skip = 'busy'
                    job.due_at = now + job.every * 60
                else:
                    job.last_skip = ''
            self.__save()
        if down and self.on_designer_down is not None:
            self.on_designer_down(down)

    def __submit(self, job: Job, *, if_idle: bool = False) -> Optional[Execution]:
        execution = self.queue.submit(job.next_request(), job.origin, if_idle=if_idle, window=job.window,
                                      job_id=job.id, number=len(job.executions) + 1)
        if execution is not None:
            job.executions.append(execution.id)
            job.pending = execution.id
        return execution

    def __halt(self, job: Job, because: str) -> None:
        if job.pending is None:
            return
        try:
            self.queue.halt(job.pending, because)
        except (ExecutionFinished, KeyError):
            pass

    # --- what an execution's end means for its job ----------------------------------------------------------------------

    def __on_finished(self, execution: Execution) -> None:
        ''' The queue's completion listener: called under its lock, so it only hands the execution on. '''
        if execution.job_id:
            self.finished.put(execution)
            self.wake.set()

    def __drain(self) -> None:
        changed = False
        while not self.finished.empty():
            execution = self.finished.get_nowait()
            job = self.jobs.get(execution.job_id)
            if job is None:
                continue
            self.__ended(job, execution)
            changed = True
        if changed:
            self.__save()

    def __ended(self, job: Job, execution: Execution) -> None:
        if job.pending == execution.id:
            job.pending = None
        job.last_finished_at = execution.finished_at or self.clock()
        if execution.progress is not None:
            if job.total is None and not job.recurring and execution.number == 1:
                job.total = execution.progress.total
            if not job.recurring:
                job.done += execution.progress.done
        if job.state == 'removed':
            return
        if job.recurring:
            job.due_at = job.last_finished_at + job.every * 60
            if self.__all_done(job, execution):
                job.state = 'done'
                job.ended = (f'all {len(job.request.selection.ids)} listed titles are done: nothing they need up to '
                             f'{job.request.through} is left')
            return
        state = execution.state
        if state == 'cancelled':
            job.state = 'removed'   # cancelling a one-off's execution ends the job (the user's decision, 2026-10-28)
        elif state in ('stopped', 'interrupted'):   # it carries on with what was not started (all of it, if unknown)
            if execution.remaining is not None:
                job.remaining = tuple(execution.remaining)
            if execution.stopped_because == 'removed':
                job.state = 'removed'
        else:
            job.state, job.remaining = 'done', ()

    @staticmethod
    def __all_done(job: Job, execution: Execution) -> bool:
        ''' A recurring job naming its titles (filter ids), whose execution of those same titles left none to do. '''
        ids = tuple(job.request.selection.ids)
        if not ids or tuple(getattr(getattr(execution.request, 'selection', None), 'ids', ())) != ids:
            return False
        try:
            return left_to_do(execution, job.request.retry_failed) == 0
        except (KeyError, TypeError, ValueError):
            return False

    # --- helpers --------------------------------------------------------------------------------------------------------

    def __live(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job.state in ('done', 'removed'):
            raise JobRefused(f'job {job_id} is {job.state}')
        return job

    def __window_texts(self, window: Any) -> Optional[List[str]]:
        ''' :raises ValueError: for a range that is not one. '''
        return None if window is None else self.queue.window(window).texts()

    def __loop(self) -> None:
        while not self.stopped:
            try:
                self.tick()
            except Exception:
                logger.exception('the job book could not start what was due')
            self.wake.wait(1)
            self.wake.clear()

    def __save(self) -> None:
        self.__trim()
        if not self.state_dir:
            return
        try:
            _write_json(self.state_dir, JOBS_FILE, {'format': FORMAT,
                                                    'jobs': [job_to_dict(j) for j in self.jobs.values()]})
        except OSError:
            logger.exception('the jobs could not be written to %s', self.state_dir)

    def __trim(self) -> None:
        ''' A job done or removed is kept while the queue still has one of its executions, for the log. '''
        for job_id, job in list(self.jobs.items()):
            if job.state in ('done', 'removed') and job.pending is None and \
                    not any(self.queue.has_job(e) for e in job.executions):
                del self.jobs[job_id]

    def __load(self) -> None:
        if not self.state_dir:
            return
        path = os.path.join(self.state_dir, JOBS_FILE)
        try:
            saved = _read_json(path)
        except (OSError, ValueError):
            logger.exception('the jobs at %s cannot be read; starting without them', path)
            return
        if not isinstance(saved, dict):
            return
        for data in saved.get('jobs') or ():
            try:
                job = job_from_dict(data)
            except (KeyError, TypeError, ValueError, AttributeError):
                logger.warning('a job in %s could not be read, and is left out', path)
                continue
            job.pending = None   # the queue recorded what was queued or running interrupted: a one-off starts again
            self.jobs[job.id] = job


# --- from an older service ------------------------------------------------------------------------------------------------

def schedule_request(values: Dict[str, Any]) -> Tuple[RunRequest, int, bool]:
    ''' An older service's schedule (schedule.json, or service.yaml's defaults): its run, interval and whether it was on. '''
    from pipeline.service.models import ScheduleUpdate, TitleFilter
    settings = ScheduleUpdate.model_validate(values or {})
    data = settings.filter.model_dump()
    data.update(needs=['extract', 'design'], new_since_scan=False)
    selection = TitleFilter.model_validate(data).to_selection()
    request = RunRequest(selection, settings.through.value, scan_first=True, retry_failed=settings.retry_failed,
                         unattended=True)
    return request, settings.interval_minutes, settings.enabled


def migrate(state_dir: Optional[str], schedule_defaults: Optional[Dict[str, Any]] = None,
            clock: Callable[[], float] = time.time) -> bool:
    '''
    Once, at the first start of a release with jobs and executions: the schedule becomes a recurring job (active if it
    was enabled, paused if not), and each job of the old history an execution -- the schedule's of that recurring job,
    each other one of a one-off job made for it, `done` once it had finished, else `active` with the titles it had left,
    so a windowed run in progress carries on. Run before the ExecutionQueue reads its history.
    :return: whether anything was migrated (False once jobs.json is a book, or with no state directory).
    '''
    if not state_dir:
        return False
    jobs_path = os.path.join(state_dir, JOBS_FILE)
    saved = _read_json(jobs_path)
    if isinstance(saved, dict) and saved.get('format') == FORMAT:
        return False
    old = saved if isinstance(saved, list) else []
    schedule_path = os.path.join(state_dir, SCHEDULE_FILE)
    values = _read_json(schedule_path) if os.path.isfile(schedule_path) else schedule_defaults
    now = clock()
    jobs: List[Job] = []
    request, every, enabled = schedule_request(values or {})
    schedule = Job(id=str(uuid.uuid4()), name=SCHEDULE_NAME, kind='run', request=request, every=every,
                   state='active' if enabled else 'paused', created_at=now, due_at=now + every * 60)
    jobs.append(schedule)
    executions = []
    for record in old:
        if not isinstance(record, dict) or 'id' not in record or 'kind' not in record:
            continue
        record = dict(record)
        state = record.get('state')
        if record.get('origin') == 'schedule':
            schedule.executions.append(record['id'])
            record.update(job_id=schedule.id, number=len(schedule.executions))
            if record.get('finished_at'):
                schedule.last_finished_at = max(schedule.last_finished_at or 0, record['finished_at'])
        else:
            finished = state in ('succeeded', 'failed', 'cancelled', 'interrupted')
            try:
                job_request = request_from_dict(record['kind'], record.get('request') or {})
            except (TypeError, ValueError, KeyError, AttributeError):
                logger.warning('job %s of the old history cannot be read back, and is kept as a record only',
                               record['id'])
                continue
            part = record.get('part_request')
            remaining = tuple(((part or {}).get('selection') or {}).get('ids') or ()) or None if part else None
            job = Job(id=str(uuid.uuid4()), name=default_name(record['kind'], job_request), kind=record['kind'],
                      request=job_request, window=record.get('window'), created_at=record.get('submitted_at') or now,
                      state='done' if finished else 'active', remaining=remaining, executions=[record['id']],
                      last_finished_at=record.get('finished_at'))
            jobs.append(job)
            record.update(job_id=job.id, number=1)
        if state == 'paused':
            record.update(state='stopped', stopped_because='window', finished_at=record.get('finished_at') or now)
        executions.append(record)
    if schedule.last_finished_at:
        schedule.due_at = schedule.last_finished_at + every * 60
    if not os.path.isfile(os.path.join(state_dir, EXECUTIONS_FILE)):
        _write_json(state_dir, EXECUTIONS_FILE, executions)
    _write_json(state_dir, JOBS_FILE, {'format': FORMAT, 'jobs': [job_to_dict(j) for j in jobs]})
    if os.path.isfile(schedule_path):
        os.replace(schedule_path, schedule_path + '.migrated')
    logger.info('migrated the schedule and %d jobs of the old history to jobs and executions', len(executions))
    return True

