'''
pipeline/service/executions.py with active windows (design/job-windows.md, design/jobs-and-executions.md §2.2): an
execution waits for its window while others run; a run whose window closes, or whose job is paused or removed (`halt`),
stops between titles and ends `stopped`, saying why and what it had not started; and the service stopping leaves what
was queued or running `interrupted`. Driven by a fake run and a fake clock.
'''
import json
import threading
import time

import pytest

import pipeline.service.executions as executions
from pipeline.library.selection import Selection
from pipeline.service.executions import ExecutionFinished, ExecutionQueue, RunRequest, ScanRequest, resumed
from service_fakes import NIGHT, UTC, Clock, Titles, at, remaining, run, until


@pytest.fixture(autouse=True)
def quick_recheck(monkeypatch):
    monkeypatch.setattr(executions, 'WINDOW_RECHECK_SECONDS', 0.01)


@pytest.fixture
def queues():
    made = []

    def make(execute, clock, **kwargs):
        queue = ExecutionQueue(execute, clock=clock, tz=UTC, remaining=remaining, **kwargs)
        made.append(queue)
        return queue
    yield make
    for queue in made:
        queue.stop(grace_seconds=5)


def test_an_execution_waits_for_its_window_while_others_run_and_then_keeps_its_place(queues):
    clock = Clock('2026-10-26 00:30')
    titles = Titles(clock, minutes=0)
    queue = queues(titles, clock)
    waiting = queue.submit(run('a'), window=NIGHT)
    other = queue.submit(run('b'))
    until(lambda: other.finished)
    assert waiting.state == 'queued' and queue.waiting_until(waiting) == at('2026-10-26 01:00')
    assert titles.done == ['b'] and queue.busy

    later = queue.submit(run('c'), window=['00:00-24:00'])   # open, but queued after the waiting one
    until(lambda: later.finished)
    clock.set('2026-10-26 01:00')
    until(lambda: waiting.finished)
    assert waiting.state == 'succeeded' and titles.done == ['b', 'c', 'a'] and queue.waiting_until(waiting) is None


def test_a_run_whose_window_closes_stops_between_titles_saying_what_it_had_not_started(queues):
    clock = Clock('2026-10-26 01:00')
    titles = Titles(clock, minutes=20)
    queue = queues(titles, clock)
    execution = queue.submit(run('a', 'b', 'c', 'd', 'e'), window=NIGHT, job_id='job-1', number=1)
    until(lambda: execution.finished)

    assert titles.done == ['a', 'b', 'c']   # 01:00, 01:20, 01:40 began inside; at 02:00 it stopped before d
    assert execution.state == 'stopped' and execution.stopped_because == 'window'
    assert execution.remaining == ('d', 'e') and (execution.job_id, execution.number) == ('job-1', 1)
    states = [(e['state'], e.get('because')) for e in execution.events if e['type'] == 'state']
    assert states[-2:] == [('stopping', 'window'), ('stopped', None)]
    assert queue.current is None and not queue.busy


def test_a_window_closing_after_the_last_title_started_is_not_a_stop(queues):
    clock = Clock('2026-10-26 01:40')
    queue = queues(Titles(clock), clock)
    execution = queue.submit(run('a'), window=NIGHT)
    until(lambda: execution.finished)
    assert execution.state == 'succeeded' and not execution.stopped_because


def test_a_scan_is_not_stopped_when_its_window_closes(queues):
    clock = Clock('2026-10-26 01:59')
    titles = Titles(clock)
    titles.hold = threading.Event()
    queue = queues(titles, clock)
    execution = queue.submit(ScanRequest(), window=NIGHT)
    until(lambda: execution.state == 'running')
    clock.set('2026-10-26 03:00')
    titles.hold.set()
    until(lambda: execution.finished)
    assert execution.state == 'succeeded'


def test_the_default_window_applies_to_an_execution_given_none_and_any_time_overrides_it(queues, tmp_path):
    clock = Clock('2026-10-26 12:00')
    titles = Titles(clock, minutes=0)
    queue = queues(titles, clock, state_dir=str(tmp_path))
    assert queue.default_window == []
    assert queue.set_default_window([' 1:00 - 2:00']) == NIGHT
    defaulted = queue.submit(run('a'))
    anytime = queue.submit(run('b'), window=[])
    until(lambda: anytime.finished)
    assert defaulted.state == 'queued' and defaulted.window is None

    queue.set_default_window(['11:00-13:00'])   # one given none follows the default as it is when checked
    until(lambda: defaulted.finished)
    assert json.loads((tmp_path / 'job-window.json').read_text()) == ['11:00-13:00']
    with pytest.raises(ValueError, match='HH:MM'):
        queue.set_default_window(['night'])
    with pytest.raises(ValueError, match='HH:MM'):
        queue.submit(run('c'), window=['night'])


def test_a_waiting_executions_window_can_be_changed_but_not_a_finished_ones(queues):
    clock = Clock('2026-10-26 12:00')
    queue = queues(Titles(clock, minutes=0), clock)
    execution = queue.submit(run('a'), window=NIGHT)
    time.sleep(0.05)
    assert execution.state == 'queued'
    assert queue.set_window(execution.id, ['11:00-13:00']) is execution and execution.window == ['11:00-13:00']
    until(lambda: execution.finished)
    assert [e['window'] for e in execution.events if e['type'] == 'window'] == [['11:00-13:00']]
    with pytest.raises(ExecutionFinished):
        queue.set_window(execution.id, [])


def test_a_running_run_given_a_closed_window_stops_before_its_next_title(queues):
    clock = Clock('2026-10-26 12:00')
    titles = Titles(clock, minutes=0)
    titles.gate = threading.Event()
    queue = queues(titles, clock)
    execution = queue.submit(run('a', 'b'), window=[])
    until(lambda: execution.state == 'running')
    queue.set_window(execution.id, NIGHT)
    titles.gate.set()
    until(lambda: execution.finished)
    assert (execution.state, execution.stopped_because, execution.remaining) == ('stopped', 'window', ('a', 'b'))


@pytest.mark.parametrize('because', ['paused', 'removed'])
def test_a_halted_run_stops_before_its_next_title_and_a_queued_one_at_once(queues, because):
    clock = Clock('2026-10-26 12:00')
    titles = Titles(clock, minutes=0)
    titles.gate = threading.Event()
    queue = queues(titles, clock)
    running = queue.submit(run('a', 'b'))
    waiting = queue.submit(run('c'))
    until(lambda: running.state == 'running')

    assert queue.halt(waiting.id, because) is waiting
    assert (waiting.state, waiting.stopped_because, waiting.remaining) == ('stopped', because, None)
    queue.halt(running.id, because)
    titles.gate.set()
    until(lambda: running.finished)
    assert (running.state, running.stopped_because, running.remaining) == ('stopped', because, ('a', 'b'))
    assert titles.done == []
    with pytest.raises(ExecutionFinished):
        queue.halt(running.id, because)
    with pytest.raises(ValueError):
        queue.halt(queue.submit(run('d'), window=NIGHT).id, 'bored')


def test_a_halted_run_that_had_started_everything_ends_as_its_result_says(queues):
    clock = Clock('2026-10-26 12:00')
    started, release = threading.Event(), threading.Event()

    def execute(execution, control):
        started.set()
        assert release.wait(5)
        return {'report': {'attempted': ['a'], 'not_run': [], 'cancelled': False}}

    queue = queues(execute, clock)
    execution = queue.submit(run('a'))
    assert started.wait(5)
    queue.halt(execution.id, 'paused')
    release.set()
    until(lambda: execution.finished)
    assert execution.state == 'succeeded' and not execution.stopped_because


def test_one_waiting_for_its_window_does_not_hold_up_a_tick_that_asks_for_an_idle_queue(queues):
    clock = Clock('2026-10-26 12:00')
    gate = threading.Event()

    def execute(execution, control):
        assert gate.wait(5)
        return {'report': {'attempted': [], 'not_run': [], 'cancelled': False}}

    queue = queues(execute, clock)
    queue.submit(run('a'), window=NIGHT)
    scheduled = queue.submit(run('b'), origin='schedule', if_idle=True)
    assert scheduled is not None
    assert queue.submit(run('c'), origin='schedule', if_idle=True) is None   # but one that can run now does
    gate.set()


def test_stopping_the_service_interrupts_what_runs_and_waits_and_a_restart_keeps_it_interrupted(tmp_path):
    clock = Clock('2026-10-26 12:00')
    titles = Titles(clock, minutes=0)
    titles.gate = threading.Event()
    first = ExecutionQueue(titles, clock=clock, tz=UTC, remaining=remaining, state_dir=str(tmp_path))
    running = first.submit(run('a', 'b'))
    waiting = first.submit(run('c'), window=NIGHT)
    until(lambda: running.state == 'running')
    stopper = threading.Thread(target=first.stop, kwargs={'grace_seconds': 5})
    stopper.start()
    until(lambda: waiting.finished)
    titles.gate.set()
    stopper.join(5)
    assert running.state == 'interrupted' and waiting.state == 'interrupted'
    saved = {e['id']: e['state'] for e in json.loads((tmp_path / 'executions.json').read_text())}
    assert saved == {running.id: 'interrupted', waiting.id: 'interrupted'}

    (tmp_path / 'executions.json').write_text(json.dumps([{'id': 'r', 'kind': 'run', 'state': 'running',
                                                           'request': {'selection': {'ids': ['a']}}}]))
    second = ExecutionQueue(lambda e, c: None, clock=clock, tz=UTC, state_dir=str(tmp_path))
    try:
        assert second.get('r').state == 'interrupted' and not second.queued
    finally:
        second.stop(grace_seconds=5)


def test_a_redesign_carries_on_as_a_plain_run_of_what_it_had_not_designed():
    request = RunRequest(Selection(ids=('a', 'b', 'c')), redesign=True, revise_to='extract', scan_first=True)
    carried = resumed(request, ('c',))
    assert (carried.selection, carried.redesign, carried.revise_to, carried.scan_first, carried.through) == \
        (Selection(ids=('c',)), False, 'design', False, 'design')
