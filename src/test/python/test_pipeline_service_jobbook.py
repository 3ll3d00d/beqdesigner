'''
pipeline/service/jobbook.py (design/jobs-and-executions.md §2.1, §3): a job is the definition, done once or every N
minutes within its window; each continuous run is an execution. A one-off carries on with what a stopped execution had
not started, and is removed when a person cancels its execution; a recurring job skips a busy tick or one the designer
cannot take, and ends itself once the titles it names are done. Jobs are kept across a restart, and an older service's
schedule and job history are migrated once. Driven by a fake run and a fake clock.
'''
import json
import threading

import pytest

import pipeline.service.executions as executions
from pipeline.library.selection import Selection
from pipeline.service.executions import AcceptRequest, ExecutionQueue, RepositoryWritesRefused, RunRequest, ScanRequest
from pipeline.service.jobbook import DESIGNER_RETRY_MINUTES, SCHEDULE_NAME, JobBook, JobBusy, JobNotFound, JobRefused, \
    migrate, schedule_request
from service_fakes import NIGHT, UTC, Clock, Titles, at, remaining, run, until


@pytest.fixture(autouse=True)
def quick_recheck(monkeypatch):
    monkeypatch.setattr(executions, 'WINDOW_RECHECK_SECONDS', 0.01)


class Service:
    ''' A queue and a book over it, ticked by hand. '''

    def __init__(self, execute, clock, state_dir=None, history_limit=200, **book):
        self.clock = clock
        self.queue = ExecutionQueue(execute, clock=clock, tz=UTC, remaining=remaining, state_dir=state_dir,
                                    history_limit=history_limit)
        self.book = JobBook(self.queue, state_dir, clock=clock, start=False, **book)

    def settle(self, job, condition, ticks=200):
        ''' Ticks until `condition(job)` holds. '''
        for _ in range(ticks):
            self.book.tick()
            if condition(job):
                return
            threading.Event().wait(0.01)
        raise AssertionError(f'job {job.name} never reached the state asked for: {job}')

    def executions_of(self, job):
        return [self.queue.get(e) for e in job.executions]

    def stop(self):
        self.book.stop()
        self.queue.stop(grace_seconds=5)


@pytest.fixture
def services():
    made = []

    def make(execute, clock, **kwargs):
        service = Service(execute, clock, **kwargs)
        made.append(service)
        return service
    yield make
    for service in made:
        service.stop()


def recurring(*ids, through='design'):
    return RunRequest(Selection(ids=ids, needs=('extract', 'design')), through, unattended=True)


# --- one-off jobs ---------------------------------------------------------------------------------------------------

def test_a_one_off_job_is_queued_at_once_and_done_when_its_execution_succeeds(services):
    clock = Clock('2026-10-26 12:00')
    titles = Titles(clock, minutes=0)
    service = services(titles, clock)
    job = service.book.create('run', run('a', 'b'))
    assert job.name == 'Run 2 titles through design' and job.pending is not None and job.origin == 'api'
    service.settle(job, lambda j: j.state == 'done')

    [execution] = service.executions_of(job)
    assert (execution.job_id, execution.number, execution.state) == (job.id, 1, 'succeeded')
    assert titles.done == ['a', 'b'] and service.book.progress(job) == (2, 2)
    assert service.book.next_at(job) is None and job.pending is None


def test_a_one_off_stopped_by_its_window_carries_on_with_what_it_had_not_started_next_night(services):
    clock = Clock('2026-10-26 01:00')
    titles = Titles(clock, minutes=20)
    service = services(titles, clock)
    job = service.book.create('run', RunRequest(Selection(ids=('a', 'b', 'c', 'd', 'e')), redesign=True),
                              window=NIGHT)
    service.settle(job, lambda j: j.pending is None)

    first = service.executions_of(job)[0]
    assert (first.state, first.stopped_because, first.remaining) == ('stopped', 'window', ('d', 'e'))
    assert job.state == 'active' and job.remaining == ('d', 'e')
    assert service.book.next_at(job) == at('2026-10-27 01:00') and service.book.progress(job) == (3, 5)

    clock.set('2026-10-27 01:00')
    service.settle(job, lambda j: j.state == 'done')
    second = service.executions_of(job)[1]
    assert second.number == 2 and second.state == 'succeeded'
    assert second.request.selection.ids == ('d', 'e') and not second.request.redesign   # sent back once, by the first
    assert titles.done == ['a', 'b', 'c', 'd', 'e'] and service.book.progress(job) == (5, 5)


def test_cancelling_a_one_offs_execution_removes_the_job(services):
    clock = Clock('2026-10-26 12:00')
    titles = Titles(clock, minutes=0)
    titles.gate = threading.Event()
    service = services(titles, clock)
    job = service.book.create('run', run('a', 'b'))
    until(lambda: service.queue.get(job.pending).state == 'running')
    service.queue.cancel(job.pending)
    titles.gate.set()
    service.settle(job, lambda j: j.state == 'removed')
    for _ in range(5):
        service.book.tick()
    assert len(job.executions) == 1 and titles.done == []


def test_pausing_stops_the_running_execution_and_resuming_carries_on(services):
    clock = Clock('2026-10-26 12:00')
    titles = Titles(clock, minutes=0)
    titles.gate = threading.Event()
    service = services(titles, clock)
    job = service.book.create('run', run('a', 'b'))
    until(lambda: service.queue.get(job.pending).state == 'running')
    assert service.book.pause(job.id) is job and job.state == 'paused'
    titles.gate.set()
    service.settle(job, lambda j: j.pending is None)
    for _ in range(5):
        service.book.tick()
    stopped = service.executions_of(job)[0]
    assert (stopped.state, stopped.stopped_because) == ('stopped', 'paused') and len(job.executions) == 1
    assert service.book.next_at(job) is None

    service.book.resume(job.id)
    service.settle(job, lambda j: j.state == 'done')
    assert titles.done == ['a', 'b'] and len(job.executions) == 2


def test_removing_a_job_stops_its_execution_and_nothing_more_starts(services):
    clock = Clock('2026-10-26 12:00')
    titles = Titles(clock, minutes=0)
    titles.gate = threading.Event()
    service = services(titles, clock)
    job = service.book.create('run', run('a'))
    until(lambda: service.queue.get(job.pending).state == 'running')
    service.book.remove(job.id)
    titles.gate.set()
    service.settle(job, lambda j: j.pending is None)
    assert job.state == 'removed' and service.executions_of(job)[0].stopped_because == 'removed'
    with pytest.raises(JobRefused):
        service.book.resume(job.id)


def test_a_one_off_that_would_write_the_repositories_is_refused_before_it_is_kept(services):
    clock = Clock('2026-10-26 12:00')
    service = services(Titles(clock), clock)
    with pytest.raises(RepositoryWritesRefused):
        service.book.create('accept', AcceptRequest(Selection()))
    assert service.book.all() == []


def test_a_one_offs_request_is_fixed_but_its_name_and_window_are_not(services):
    clock = Clock('2026-10-26 12:00')
    service = services(Titles(clock), clock)
    job = service.book.create('scan', ScanRequest(), window=NIGHT)
    assert job.name == 'List the sources again'
    assert job.pending is None and service.book.next_at(job) == at('2026-10-27 01:00')   # started when it opens
    with pytest.raises(JobRefused):
        service.book.edit(job.id, request=ScanRequest(sources=('x',)))
    with pytest.raises(JobRefused):
        service.book.edit(job.id, every=10)
    service.book.edit(job.id, name='  Nightly scan ', window=[])
    assert job.name == 'Nightly scan' and job.window == []
    service.settle(job, lambda j: j.state == 'done')
    assert service.executions_of(job)[0].window == []
    with pytest.raises(JobRefused):
        service.book.edit(job.id, name='again')
    with pytest.raises(JobNotFound):
        service.book.get('nope')


# --- recurring jobs -------------------------------------------------------------------------------------------------

def test_a_recurring_job_runs_every_interval_after_its_last_execution_finished(services):
    clock = Clock('2026-10-26 12:00')
    titles = Titles(clock, minutes=0)
    service = services(titles, clock)
    job = service.book.create('run', recurring(), every=5)
    assert job.origin == 'schedule' and service.book.next_at(job) == at('2026-10-26 12:05')
    service.book.tick()
    assert job.executions == []

    clock.set('2026-10-26 12:05')
    service.settle(job, lambda j: len(j.executions) == 1 and j.pending is None)
    assert job.state == 'active' and job.due_at == at('2026-10-26 12:10')
    clock.set('2026-10-26 12:09')
    service.book.tick()
    assert len(job.executions) == 1
    clock.set('2026-10-26 12:10')
    service.settle(job, lambda j: len(j.executions) == 2)


def test_a_recurring_job_waits_for_its_window(services):
    clock = Clock('2026-10-26 12:00')
    service = services(Titles(clock, minutes=0), clock)
    job = service.book.create('run', recurring(), every=5, window=NIGHT)
    clock.set('2026-10-26 12:30')
    service.book.tick()
    assert job.executions == [] and service.book.next_at(job) == at('2026-10-27 01:00')
    clock.set('2026-10-27 01:00')
    service.settle(job, lambda j: len(j.executions) == 1)


def test_a_busy_tick_is_skipped_and_run_now_refuses_while_another_runs(services):
    clock = Clock('2026-10-26 12:00')
    release = threading.Event()
    service = services(lambda e, c: release.wait(5) and None, clock)
    job = service.book.create('run', recurring(), every=5)
    other = service.queue.submit(ScanRequest())
    until(lambda: other.state == 'running')
    clock.set('2026-10-26 12:05')
    service.book.tick()
    assert job.last_skip == 'busy' and job.due_at == at('2026-10-26 12:10') and job.executions == []
    with pytest.raises(JobBusy):
        service.book.run_now(job.id)
    release.set()
    until(lambda: other.finished)
    execution = service.book.run_now(job.id)
    assert execution.job_id == job.id and job.last_skip == ''
    with pytest.raises(JobBusy):
        service.book.run_now(job.id)   # it is running already
    one_off = service.book.create('scan', ScanRequest())
    with pytest.raises(JobRefused):
        service.book.run_now(one_off.id)


def test_a_tick_is_skipped_while_the_designer_is_down_notified_once_and_resumed_when_it_is_back(services):
    clock = Clock('2026-10-26 12:00')
    down, told = ['HTTP 503'], []
    service = services(Titles(clock, minutes=0), clock, designer=lambda: down[0], on_designer_down=told.append)
    job = service.book.create('run', recurring(), every=60)
    clock.advance(60)
    service.book.tick()
    assert job.executions == [] and job.last_skip == 'designer unavailable: HTTP 503' and told == ['HTTP 503']
    assert job.due_at == clock() + DESIGNER_RETRY_MINUTES * 60   # sooner than the hour
    clock.advance(DESIGNER_RETRY_MINUTES)
    service.book.tick()                      # still down: skipped again, not notified again
    assert job.executions == [] and len(told) == 1
    down[0] = ''
    clock.advance(DESIGNER_RETRY_MINUTES)
    service.settle(job, lambda j: len(j.executions) == 1)
    assert job.last_skip == '' and service.book.designer_down == ''


def test_a_new_outage_is_notified_again(services):
    clock = Clock('2026-10-26 12:00')
    down, told = [''], []
    service = services(Titles(clock, minutes=0), clock, designer=lambda: down[0], on_designer_down=told.append)
    job = service.book.create('run', recurring(), every=5)
    for reason in ('HTTP 503', '', 'refused'):
        down[0] = reason
        clock.advance(60)
        service.book.tick()
        until(lambda: job.pending is None or service.queue.get(job.pending).finished)
        service.book.tick()
    assert told == ['HTTP 503', 'refused']


def test_a_recurring_run_through_extract_does_not_ask_the_designer(services):
    clock = Clock('2026-10-26 12:00')
    asked = []
    service = services(Titles(clock, minutes=0), clock, designer=lambda: asked.append(1) or 'down')
    job = service.book.create('run', recurring(through='extract'), every=5)
    clock.advance(5)
    service.settle(job, lambda j: len(j.executions) == 1)
    assert asked == []


@pytest.mark.parametrize('kind, request_', [('scan', ScanRequest()), ('accept', AcceptRequest(Selection())),
                                            ('run', RunRequest(Selection(), 'publish')),
                                            ('run', RunRequest(Selection(ids=('a',)), redesign=True))])
def test_only_a_run_through_extract_or_design_recurs(services, kind, request_):
    clock = Clock('2026-10-26 12:00')
    service = services(Titles(clock), clock)
    with pytest.raises(ValueError, match='recurring'):
        service.book.create(kind, request_, every=5)
    with pytest.raises(ValueError, match='every 5 minutes'):
        service.book.create('run', recurring(), every=4)


def test_a_recurring_jobs_interval_and_run_change_from_its_next_execution(services):
    clock = Clock('2026-10-26 12:00')
    service = services(Titles(clock, minutes=0), clock)
    job = service.book.create('run', recurring(), every=5)
    service.book.edit(job.id, every=30, request=recurring('x'))
    assert job.every == 30 and job.due_at == at('2026-10-26 12:30') and job.request.selection.ids == ('x',)
    with pytest.raises(ValueError):
        service.book.edit(job.id, request=RunRequest(Selection(), 'commit'))


def _ran_once(services, ids, run_report, retry_failed=False, **report):
    clock = Clock('2026-10-26 12:00')
    result = {'report': {'through': 'design', 'selected': 2, 'run': run_report, **report}}
    service = services(lambda e, c: result, clock)
    job = service.book.create('run', RunRequest(Selection(ids=ids), retry_failed=retry_failed), every=5)
    clock.advance(5)
    service.settle(job, lambda j: len(j.executions) == 1 and j.pending is None)
    return job


def test_a_recurring_job_of_listed_titles_ends_itself_once_none_is_left_to_do(services):
    job = _ran_once(services, ('a', 'b'), {'designed': ['a'], 'design_cached': ['b']})
    assert job.state == 'done' and job.ended == 'all 2 listed titles are done: nothing they need up to design is left'


@pytest.mark.parametrize('ids, run_report, retry, report', [
    ((), {}, False, {}),                                                                 # no ids: titles found later
    (('a', 'b'), {'unavailable': [{'id': 'b', 'message': 'designer down'}]}, False, {}),
    (('a', 'b'), {}, False, {'stopped': 'stopped after 3 titles in a row', 'not_run': ['b']}),
    (('a', 'b'), {'failed': [{'id': 'b', 'message': 'bad rip'}]}, True, {}),
])
def test_a_recurring_job_keeps_going_while_something_may_be_left(services, ids, run_report, retry, report):
    job = _ran_once(services, ids, run_report, retry, **report)
    assert job.state == 'active' and not job.ended


# --- kept across a restart, and migrated once -----------------------------------------------------------------------

def test_jobs_are_kept_and_a_one_off_interrupted_by_a_stop_starts_again(tmp_path):
    clock = Clock('2026-10-26 12:00')
    titles = Titles(clock, minutes=0)
    titles.gate = threading.Event()
    first = Service(titles, clock, state_dir=str(tmp_path))
    one_off = first.book.create('run', run('a', 'b'), name='Two titles')
    schedule = first.book.create('run', recurring(), every=30, window=NIGHT)
    first.book.pause(schedule.id)
    until(lambda: first.queue.get(one_off.pending).state == 'running')
    stopper = threading.Thread(target=first.stop)
    stopper.start()
    titles.gate.set()
    stopper.join(10)
    assert first.queue.get(one_off.executions[0]).state == 'interrupted'

    second = Service(titles, clock, state_dir=str(tmp_path))
    try:
        again, kept = second.book.get(one_off.id), second.book.get(schedule.id)
        assert (kept.state, kept.every, kept.window, kept.name) == ('paused', 30, NIGHT, schedule.name)
        assert again.name == 'Two titles' and again.request == run('a', 'b') and again.pending is None
        second.settle(again, lambda j: j.state == 'done')
        assert len(again.executions) == 2 and second.queue.get(again.executions[1]).number == 2
    finally:
        second.stop()


def test_a_finished_job_is_forgotten_once_its_executions_leave_the_history(tmp_path):
    clock = Clock('2026-10-26 12:00')
    service = Service(Titles(clock, minutes=0), clock, state_dir=str(tmp_path), history_limit=1)
    try:
        old = service.book.create('scan', ScanRequest())
        service.settle(old, lambda j: j.state == 'done')
        new = service.book.create('scan', ScanRequest(sources=('x',)))
        service.settle(new, lambda j: j.state == 'done')
        service.book.tick()
        assert [j.id for j in service.book.all()] == [new.id]
    finally:
        service.stop()


def test_an_older_services_schedule_and_history_are_migrated_once(tmp_path):
    (tmp_path / 'schedule.json').write_text(json.dumps({'enabled': True, 'interval_minutes': 30,
                                                        'filter': {'kind': 'movie'}, 'through': 'extract'}))
    (tmp_path / 'jobs.json').write_text(json.dumps([
        {'id': 'tick', 'kind': 'run', 'origin': 'schedule', 'state': 'succeeded', 'submitted_at': 100.0,
         'finished_at': 200.0, 'request': {'selection': {'needs': ['extract', 'design']}, 'through': 'extract'}},
        {'id': 'scan', 'kind': 'scan', 'origin': 'api', 'state': 'failed', 'submitted_at': 300.0, 'finished_at': 310.0,
         'request': {'sources': []}},
        {'id': 'night', 'kind': 'run', 'origin': 'api', 'state': 'paused', 'submitted_at': 400.0, 'window': NIGHT,
         'request': {'selection': {'ids': ['a', 'b', 'c']}, 'redesign': True},
         'part_request': {'selection': {'ids': ['c']}, 'scan_first': False}},
    ]))
    assert migrate(str(tmp_path), {}, clock=lambda: 1000.0)
    assert not (tmp_path / 'schedule.json').exists() and (tmp_path / 'schedule.json.migrated').exists()
    assert not migrate(str(tmp_path), {})   # once

    clock = Clock('2026-10-26 12:00')
    titles = Titles(clock, minutes=0)
    service = Service(titles, clock, state_dir=str(tmp_path))
    try:
        jobs = {j.name: j for j in service.book.all()}
        schedule = jobs[SCHEDULE_NAME]
        assert (schedule.state, schedule.every, schedule.executions) == ('active', 30, ['tick'])
        assert schedule.request.through == 'extract' and schedule.request.selection.kind == 'movie'
        assert schedule.due_at == 200.0 + 30 * 60
        assert service.queue.get('tick').job_id == schedule.id
        assert jobs['List the sources again'].state == 'done' and service.queue.get('scan').state == 'failed'

        night = jobs['Redesign 3 titles']
        stopped = service.queue.get('night')
        assert (stopped.state, stopped.stopped_because, stopped.job_id) == ('stopped', 'window', night.id)
        assert night.state == 'active' and night.remaining == ('c',) and night.window == NIGHT
        clock.set('2026-10-27 01:00')
        service.settle(night, lambda j: j.state == 'done')
        assert titles.done == ['c']
    finally:
        service.stop()


def test_a_first_start_with_no_schedule_file_makes_the_recurring_job_from_the_service_config(tmp_path):
    assert migrate(str(tmp_path), {'enabled': False, 'interval_minutes': 45}, clock=lambda: 1000.0)
    saved = json.loads((tmp_path / 'jobs.json').read_text())
    assert saved['format'] == 2
    [schedule] = saved['jobs']
    assert (schedule['name'], schedule['state'], schedule['every']) == (SCHEDULE_NAME, 'paused', 45)
    assert json.loads((tmp_path / 'executions.json').read_text()) == []
    request, every, enabled = schedule_request({'filter': {'ids': ['a']}})
    assert request.selection.needs == ('extract', 'design') and request.unattended and every == 60 and not enabled
