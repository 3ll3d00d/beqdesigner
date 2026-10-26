'''
pipeline/service/jobs.py with active windows (design/job-windows.md §3): a job waits for its window while others run, a
run whose window closes pauses between titles and carries on when it opens, as one job, and a paused or waiting job is
kept across a restart. Driven by a fake run and a fake clock.
'''
import json
import threading
import time
from datetime import datetime, timezone

import pytest

import pipeline.service.jobs as jobs
from pipeline.library.selection import Selection
from pipeline.library.stages import Progress
from pipeline.service.jobs import JobFinished, JobManager, RunRequest, ScanRequest, resumed

UTC = timezone.utc
NIGHT = ['01:00-02:00']


def at(text: str) -> float:
    return datetime.strptime(text, '%Y-%m-%d %H:%M').replace(tzinfo=UTC).timestamp()


def _until(condition, timeout=5.0):
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
    Stopped, it reports the titles it had not started (not_run) and that it was cancelled.
    '''

    def __init__(self, clock, minutes=20):
        self.clock, self.minutes = clock, minutes
        self.done, self.parts, self.hold = [], [], None

    def __call__(self, job, control):
        request = job.current_request
        self.parts.append((job.id, request))
        if isinstance(request, ScanRequest):
            if self.hold:
                assert self.hold.wait(5)
            return {'scan': True}
        ids = list(request.selection.ids)
        attempted = []
        for n, title in enumerate(ids):
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


def merge(parts):
    return {'report': {'attempted': [t for p in parts for t in p['report']['attempted']],
                       'not_run': parts[-1]['report']['not_run'], 'cancelled': parts[-1]['report']['cancelled']}}


@pytest.fixture(autouse=True)
def quick_recheck(monkeypatch):
    monkeypatch.setattr(jobs, 'WINDOW_RECHECK_SECONDS', 0.01)


@pytest.fixture
def managers():
    made = []

    def make(execute, clock, **kwargs):
        manager = JobManager(execute, clock=clock, tz=UTC, remaining=remaining, merge=merge, **kwargs)
        made.append(manager)
        return manager
    yield make
    for manager in made:
        manager.stop(grace_seconds=5)


def run(*ids):
    return RunRequest(Selection(ids=ids))


def test_a_job_waits_for_its_window_while_others_run_and_then_keeps_its_place(managers):
    clock = Clock('2026-10-26 00:30')
    titles = Titles(clock, minutes=0)
    manager = managers(titles, clock)
    waiting = manager.submit(run('a'), window=NIGHT)
    other = manager.submit(run('b'))
    _until(lambda: other.finished)
    assert waiting.state == 'queued' and manager.waiting_until(waiting) == at('2026-10-26 01:00')
    assert titles.done == ['b'] and manager.busy

    later = manager.submit(run('c'), window=['00:00-24:00'])   # open, but queued after the waiting job
    _until(lambda: later.finished)
    clock.set('2026-10-26 01:00')
    _until(lambda: waiting.finished)
    assert waiting.state == 'succeeded' and titles.done == ['b', 'c', 'a'] and manager.waiting_until(waiting) is None


def test_a_run_whose_window_closes_pauses_between_titles_and_carries_on_as_one_job(managers):
    clock = Clock('2026-10-26 01:00')
    titles = Titles(clock, minutes=20)
    manager = managers(titles, clock)
    job = manager.submit(run('a', 'b', 'c', 'd', 'e'), window=NIGHT)
    _until(lambda: job.state == 'paused')

    assert titles.done == ['a', 'b', 'c']   # 01:00, 01:20, 01:40 began inside; at 02:00 it stopped before d
    assert job.paused_at == [at('2026-10-26 02:00')] and job.parts == 1 and not job.finished
    assert job.part_request.selection.ids == ('d', 'e') and manager.current is None
    assert manager.waiting_until(job) == at('2026-10-27 01:00')
    states = [e['state'] for e in job.events if e['type'] == 'state']
    assert states[-2:] == ['pausing', 'paused']

    other = manager.submit(run('x'), window=[])   # the worker and the lease are free for another job meanwhile
    _until(lambda: other.finished)
    clock.set('2026-10-27 01:00')
    _until(lambda: job.finished)

    assert job.state == 'succeeded' and job.parts == 2
    assert titles.done == ['a', 'b', 'c', 'x', 'd', 'e']   # every title once
    assert job.result['report']['attempted'] == ['a', 'b', 'c', 'd', 'e']
    assert (job.progress.done, job.progress.total) == (5, 5)   # the bar carried on from where it paused
    assert {'type': 'state', 'state': 'running', 'part': 2}.items() <= \
        next(e for e in job.events if e.get('part') == 2).items()


def test_a_scan_is_not_paused_when_its_window_closes(managers):
    clock = Clock('2026-10-26 01:59')
    titles = Titles(clock)
    titles.hold = threading.Event()
    manager = managers(titles, clock)
    job = manager.submit(ScanRequest(), window=NIGHT)
    _until(lambda: job.state == 'running')
    clock.set('2026-10-26 03:00')
    titles.hold.set()
    _until(lambda: job.finished)
    assert job.state == 'succeeded' and job.parts == 1 and not job.paused_at


def test_the_default_window_applies_to_a_job_given_none_and_any_time_overrides_it(managers, tmp_path):
    clock = Clock('2026-10-26 12:00')
    titles = Titles(clock, minutes=0)
    manager = managers(titles, clock, state_dir=str(tmp_path))
    assert manager.default_window == []
    assert manager.set_default_window([' 1:00 - 2:00']) == NIGHT
    defaulted = manager.submit(run('a'))
    anytime = manager.submit(run('b'), window=[])
    _until(lambda: anytime.finished)
    assert defaulted.state == 'queued' and defaulted.window is None

    manager.set_default_window(['11:00-13:00'])   # a job given none follows the default as it is when checked
    _until(lambda: defaulted.finished)
    assert json.loads((tmp_path / 'job-window.json').read_text()) == ['11:00-13:00']
    with pytest.raises(ValueError, match='HH:MM'):
        manager.set_default_window(['night'])
    with pytest.raises(ValueError, match='HH:MM'):
        manager.submit(run('c'), window=['night'])


def test_a_waiting_jobs_window_can_be_changed_but_not_a_finished_ones(managers):
    clock = Clock('2026-10-26 12:00')
    titles = Titles(clock, minutes=0)
    manager = managers(titles, clock)
    job = manager.submit(run('a'), window=NIGHT)
    time.sleep(0.05)
    assert job.state == 'queued'
    assert manager.set_window(job.id, ['11:00-13:00']) is job and job.window == ['11:00-13:00']
    _until(lambda: job.finished)
    assert [e['window'] for e in job.events if e['type'] == 'window'] == [['11:00-13:00']]
    with pytest.raises(JobFinished):
        manager.set_window(job.id, [])


def test_a_running_run_given_a_closed_window_pauses_before_its_next_title(managers):
    clock = Clock('2026-10-26 12:00')
    started, release = threading.Event(), threading.Event()

    def execute(job, control):
        request = job.current_request
        if request.selection.ids == ('a', 'b'):
            started.set()
            assert release.wait(5)
            if control.should_stop():
                return {'report': {'attempted': ['a'], 'not_run': ['b'], 'cancelled': True}}
        return {'report': {'attempted': list(request.selection.ids), 'not_run': [], 'cancelled': False}}

    manager = managers(execute, clock)
    job = manager.submit(run('a', 'b'), window=[])
    assert started.wait(5)
    manager.set_window(job.id, NIGHT)
    release.set()
    _until(lambda: job.state == 'paused')
    manager.set_window(job.id, None)   # the default: any time
    _until(lambda: job.finished)
    assert job.state == 'succeeded' and job.result['report']['attempted'] == ['a', 'b']


def test_a_paused_job_can_be_cancelled_and_reports_what_it_did(managers):
    clock = Clock('2026-10-26 01:40')
    manager = managers(Titles(clock), clock)
    job = manager.submit(run('a', 'b', 'c'), window=NIGHT)
    _until(lambda: job.state == 'paused')
    manager.cancel(job.id)
    assert job.state == 'cancelled' and job.result['report']['attempted'] == ['a']


def test_a_window_closing_after_the_last_title_started_finishes_the_job(managers):
    clock = Clock('2026-10-26 01:40')
    manager = managers(Titles(clock), clock)
    job = manager.submit(run('a'), window=NIGHT)
    _until(lambda: job.finished)
    assert job.state == 'succeeded' and not job.paused_at


def test_a_job_waiting_for_its_window_does_not_hold_up_the_schedule(managers):
    clock = Clock('2026-10-26 12:00')
    gate = threading.Event()

    def execute(job, control):
        assert gate.wait(5)
        return {'report': {'attempted': [], 'not_run': [], 'cancelled': False}}

    manager = managers(execute, clock)
    manager.submit(run('a'), window=NIGHT)
    scheduled = manager.submit(run('b'), origin='schedule', if_idle=True)
    assert scheduled is not None
    assert manager.submit(run('c'), origin='schedule', if_idle=True) is None   # but one that can run now does
    gate.set()


def test_paused_and_waiting_jobs_carry_on_after_a_restart(tmp_path):
    clock = Clock('2026-10-26 01:40')
    titles = Titles(clock)
    first = JobManager(titles, clock=clock, tz=UTC, remaining=remaining, merge=merge, state_dir=str(tmp_path))
    paused = first.submit(run('a', 'b', 'c'), window=NIGHT)
    _until(lambda: paused.state == 'paused')
    waiting = first.submit(ScanRequest(sources=('jriver',)), window=['05:00-06:00'])
    open_now = first.submit(run('z'), window=['02:00-03:00'])
    _until(lambda: open_now.finished)
    first.stop(grace_seconds=5)
    saved = {j['id']: j['state'] for j in json.loads((tmp_path / 'jobs.json').read_text())}
    assert saved[paused.id] == 'paused' and saved[waiting.id] == 'queued'

    clock.set('2026-10-27 01:00')
    second = JobManager(titles, clock=clock, tz=UTC, remaining=remaining, merge=merge, state_dir=str(tmp_path))
    try:
        again = second.get(paused.id)
        _until(lambda: again.finished)
        assert again.state == 'succeeded' and again.result['report']['attempted'] == ['a', 'b', 'c']
        assert again.parts == 2 and titles.done.count('b') == 1
        kept = second.get(waiting.id)
        assert kept.state == 'queued' and kept.request == ScanRequest(sources=('jriver',))
    finally:
        second.stop(grace_seconds=5)


def test_a_job_running_when_the_service_stopped_is_still_interrupted(tmp_path):
    (tmp_path / 'jobs.json').write_text(json.dumps([{'id': 'r', 'kind': 'run', 'state': 'running', 'window': NIGHT,
                                                     'request': {'selection': {'ids': ['a']}}}]))
    manager = JobManager(lambda job, control: None, clock=Clock('2026-10-26 12:00'), tz=UTC, state_dir=str(tmp_path))
    try:
        assert manager.get('r').state == 'interrupted'
    finally:
        manager.stop(grace_seconds=5)


def test_a_redesign_carries_on_as_a_plain_run_of_what_it_had_not_designed():
    request = RunRequest(Selection(ids=('a', 'b', 'c')), redesign=True, revise_to='extract', scan_first=True)
    carried = resumed(request, ('c',))
    assert (carried.selection, carried.redesign, carried.revise_to, carried.scan_first, carried.through) == \
        (Selection(ids=('c',)), False, 'design', False, 'design')
