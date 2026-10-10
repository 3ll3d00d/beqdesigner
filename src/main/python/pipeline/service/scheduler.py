'''Automatic extract/design jobs for the pipeline service.'''
import json
import os
import tempfile
import threading
import time
from queue import SimpleQueue
from dataclasses import dataclass
from typing import Callable, Optional

from pipeline.service.jobs import JobManager, RunRequest
from pipeline.service.models import ScheduleUpdate, TitleFilter, run_result

DESIGNER_RETRY_MINUTES = 5   # a tick skipped because the designer did not answer is tried again this soon (or sooner)


@dataclass(frozen=True)
class LastRun:
    job_id: str
    state: str
    finished_at: Optional[float]


def left_to_do(job, retry_failed: bool) -> Optional[int]:
    '''
    How many of a finished scheduled run's titles it left for a later run: unavailable, not reached, or (when failures are
    retried) failed. None if it did not finish its selection: it was cancelled or stopped, or it raised.
    '''
    if job.result is None or job.state in ('cancelled', 'interrupted'):
        return None
    result = run_result(job.result)
    if result.cancelled or result.stopped:
        return None
    return len(result.unavailable) + len(result.not_run) + (len(result.failed) if retry_failed else 0)


class AutoScheduler:
    '''
    A single timer. `tick()` and the clock injection also allow deterministic tests.

    :param designer: asked before a tick that designs: why the designer cannot be used, or '' if it can. A tick it
        refuses is skipped (`last_skip`) and tried again within DESIGNER_RETRY_MINUTES.
    :param on_designer_down: told the reason once when the designer stops answering, not at every skipped tick.
    '''

    def __init__(self, manager: JobManager, state_dir: Optional[str], defaults: dict,
                 clock: Callable[[], float] = time.time, *, start: bool = True,
                 designer: Optional[Callable[[], str]] = None,
                 on_designer_down: Optional[Callable[[str], None]] = None):
        self.manager, self.clock = manager, clock
        self.designer, self.on_designer_down = designer, on_designer_down
        self.designer_down = ''   # why the designer did not answer at the last tick that asked
        self.path = os.path.join(state_dir, 'schedule.json') if state_dir else None
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.stopped = False
        values = defaults
        if self.path and os.path.isfile(self.path):
            with open(self.path, encoding='utf-8') as stream:
                values = json.load(stream)
        self.settings = ScheduleUpdate.model_validate(values)
        self.next_run_at = clock() + self.settings.interval_minutes * 60 if self.settings.enabled else None
        self.last_run: Optional[LastRun] = None
        self.last_skip: Optional[str] = None
        self.ended: Optional[str] = None   # why the schedule turned itself off
        self.finished = SimpleQueue()
        self.unsubscribe = manager.subscribe(self._job_event)
        self.thread: Optional[threading.Thread] = None
        if start:
            self.thread = threading.Thread(target=self._loop, name='pipeline-service-schedule', daemon=True)
            self.thread.start()

    def snapshot(self) -> dict:
        with self.lock:
            self._drain()
            return {'enabled': self.settings.enabled, 'interval_minutes': self.settings.interval_minutes,
                    'filter': self.settings.filter, 'through': self.settings.through,
                    'retry_failed': self.settings.retry_failed, 'next_run_at': self.next_run_at,
                    'last_run': vars(self.last_run) if self.last_run else None, 'last_skip': self.last_skip,
                    'ended': self.ended}

    def update(self, settings: ScheduleUpdate) -> dict:
        with self.lock:
            self._drain()
            self._persist(settings)
            self.settings = settings
            self.next_run_at = self.clock() + settings.interval_minutes * 60 if settings.enabled else None
            self.last_skip = None
            self.ended = None
            self.wake.set()
            return self.snapshot()

    def _persist(self, settings: ScheduleUpdate) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        name = None
        try:
            with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=os.path.dirname(self.path),
                                             prefix='.schedule-', delete=False) as stream:
                name = stream.name
                json.dump(settings.model_dump(mode='json'), stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.path)
        finally:
            if name and os.path.exists(name):
                os.unlink(name)

    def _request(self) -> RunRequest:
        data = self.settings.filter.model_dump()
        data.update(needs=['extract', 'design'], new_since_scan=False)
        selection = TitleFilter.model_validate(data).to_selection()
        return RunRequest(selection, self.settings.through.value, scan_first=True,
                          retry_failed=self.settings.retry_failed, unattended=True)

    def trigger(self):
        '''Submit one scheduled run now, or return None if another job is active or queued.'''
        with self.lock:
            self._drain()
            job = self.manager.submit(self._request(), origin='schedule', if_idle=True)
            if job is not None:
                self.next_run_at = None
                self.last_skip = None
            return job

    def _due(self) -> bool:
        return self.settings.enabled and self.next_run_at is not None and self.clock() >= self.next_run_at

    def tick(self) -> None:
        with self.lock:
            self._drain()
            if not self._due():
                return
            ask = self.designer is not None and self.settings.through.value != 'extract'
        reason = self.designer() if ask else ''   # not holding the lock: it may take the designer's timeout
        down = ''
        with self.lock:
            if not self._due():
                return
            if reason:
                self.last_skip = f'designer unavailable: {reason}'
                self.next_run_at = self.clock() + min(self.settings.interval_minutes, DESIGNER_RETRY_MINUTES) * 60
                down = reason if not self.designer_down else ''
                self.designer_down = reason
            else:
                if ask:
                    self.designer_down = ''
                if self.trigger() is None:
                    self.last_skip = 'busy'
                    self.next_run_at = self.clock() + self.settings.interval_minutes * 60
        if down and self.on_designer_down is not None:
            self.on_designer_down(down)

    def _job_event(self, job, event: dict) -> None:
        if job.origin != 'schedule' or event.get('type') != 'state' or not job.finished:
            return
        self.finished.put((LastRun(job.id, job.state, job.finished_at), job))
        self.wake.set()

    def _drain(self) -> None:
        '''Called with the scheduler lock; manager callbacks only queue data, avoiding reversed lock order.'''
        while not self.finished.empty():
            run, job = self.finished.get_nowait()
            self.last_run = run
            if self.settings.enabled:
                if self._all_done(job):
                    self._end(f'all {len(self.settings.filter.ids)} listed titles are done: '
                              f'nothing they need up to {self.settings.through.value} is left')
                else:
                    self.next_run_at = (run.finished_at or self.clock()) + self.settings.interval_minutes * 60

    def _all_done(self, job) -> bool:
        '''
        Whether the schedule names its titles (filter.ids), so none can join them, and this run of those same titles left
        none for a later one. A filter without ids may match titles found later, so it keeps ticking.
        '''
        ids = tuple(self.settings.filter.ids)
        if not ids or tuple(job.request.selection.ids) != ids:
            return False
        try:
            return left_to_do(job, self.settings.retry_failed) == 0
        except (KeyError, TypeError, ValueError):
            return False

    def _end(self, why: str) -> None:
        settings = self.settings.model_copy(update={'enabled': False})
        self._persist(settings)
        self.settings, self.next_run_at, self.ended = settings, None, why

    def _loop(self) -> None:
        while not self.stopped:
            self.tick()
            with self.lock:
                remaining = self.next_run_at - self.clock() if self.next_run_at is not None else 1
            self.wake.wait(max(0.05, min(remaining, 1)))
            self.wake.clear()

    def stop(self) -> None:
        self.stopped = True
        self.wake.set()
        if self.thread:
            self.thread.join(timeout=2)
        self.unsubscribe()
