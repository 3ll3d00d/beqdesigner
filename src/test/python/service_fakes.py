'''
Fakes for the service's executions and jobs (pipeline/service/executions.py, jobbook.py): a clock to move, and a run that
does its titles in turn as run_stages does, polling should_stop() before each.
'''
import threading
import time
from datetime import datetime, timezone

from pipeline.library.selection import Selection
from pipeline.library.stages import Progress
from pipeline.service.executions import RunRequest, ScanRequest

UTC = timezone.utc
NIGHT = ['01:00-02:00']


def at(text: str) -> float:
    return datetime.strptime(text, '%Y-%m-%d %H:%M').replace(tzinfo=UTC).timestamp()


def until(condition, timeout=5.0):
    deadline = time.time() + timeout
    while not condition():
        assert time.time() < deadline, 'timed out'
        time.sleep(0.005)


class Clock:
    def __init__(self, text):
        self.t = at(text)
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            return self.t

    def set(self, text):
        with self.lock:
            self.t = at(text)

    def advance(self, minutes):
        with self.lock:
            self.t += minutes * 60


class Titles:
    '''
    A run, as run_stages does one: its titles in turn, polling should_stop() before each; each takes `minutes` on the clock.
    Stopped, it reports the titles it had not started (not_run) and that it was cancelled. `hold` holds a scan, and
    `gate`, if set, holds each title until it is set.
    '''

    def __init__(self, clock, minutes=20):
        self.clock, self.minutes = clock, minutes
        self.done, self.requests, self.hold, self.gate = [], [], None, None

    def __call__(self, execution, control):
        request = execution.request
        self.requests.append((execution.id, request))
        if isinstance(request, ScanRequest):
            if self.hold:
                assert self.hold.wait(5)
            return {'scan': True}
        ids = list(request.selection.ids)
        attempted = []
        for n, title in enumerate(ids):
            if self.gate is not None:
                assert self.gate.wait(5)
            if control.should_stop():   # run_stages' last progress, when stopped, is how far it got
                control.progress(Progress(n, len(ids), '', ''))
                return {'report': {'attempted': attempted, 'not_run': ids[n:], 'cancelled': True}}
            control.progress(Progress(n, len(ids), title, 'design', title))
            self.clock.advance(self.minutes)
            self.done.append(title)
            attempted.append(title)
        control.progress(Progress(len(ids), len(ids), '', ''))
        return {'report': {'attempted': attempted, 'not_run': [], 'cancelled': False}}


def remaining(result):
    return None if result is None or 'report' not in result else tuple(result['report']['not_run'])


def run(*ids):
    return RunRequest(Selection(ids=ids))
