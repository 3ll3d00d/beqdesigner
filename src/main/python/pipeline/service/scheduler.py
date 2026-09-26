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
from pipeline.service.models import ScheduleUpdate, TitleFilter


@dataclass(frozen=True)
class LastRun:
    job_id: str
    state: str
    finished_at: Optional[float]


class AutoScheduler:
    '''A single timer. `tick()` and the clock injection also allow deterministic tests.'''

    def __init__(self, manager: JobManager, state_dir: Optional[str], defaults: dict,
                 clock: Callable[[], float] = time.time, *, start: bool = True):
        self.manager, self.clock = manager, clock
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
                    'last_run': vars(self.last_run) if self.last_run else None, 'last_skip': self.last_skip}

    def update(self, settings: ScheduleUpdate) -> dict:
        with self.lock:
            self._drain()
            self._persist(settings)
            self.settings = settings
            self.next_run_at = self.clock() + settings.interval_minutes * 60 if settings.enabled else None
            self.last_skip = None
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
                          retry_failed=self.settings.retry_failed)

    def trigger(self):
        '''Submit one scheduled run now, or return None if another job is active or queued.'''
        with self.lock:
            self._drain()
            job = self.manager.submit(self._request(), origin='schedule', if_idle=True)
            if job is not None:
                self.next_run_at = None
                self.last_skip = None
            return job

    def tick(self) -> None:
        with self.lock:
            self._drain()
            if not self.settings.enabled or self.next_run_at is None or self.clock() < self.next_run_at:
                return
            if self.trigger() is None:
                self.last_skip = 'busy'
                self.next_run_at = self.clock() + self.settings.interval_minutes * 60

    def _job_event(self, job, event: dict) -> None:
        if job.origin != 'schedule' or event.get('type') != 'state' or not job.finished:
            return
        self.finished.put(LastRun(job.id, job.state, job.finished_at))
        self.wake.set()

    def _drain(self) -> None:
        '''Called with the scheduler lock; manager callbacks only queue data, avoiding reversed lock order.'''
        while not self.finished.empty():
            run = self.finished.get_nowait()
            self.last_run = run
            if self.settings.enabled:
                self.next_run_at = (run.finished_at or self.clock()) + self.settings.interval_minutes * 60

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
