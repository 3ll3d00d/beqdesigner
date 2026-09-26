'''
pipeline/service/jobs.py (design/pipeline-service.md §5): one job at a time, in the order asked, cancelled between titles,
with its progress and redacted events kept, and a history that survives a restart. Driven by a fake `execute`.
'''
import json
import threading
import time

import pytest

from model.execution_events import ExecutionEvent
from pipeline.library.selection import Selection
from pipeline.library.stages import FfmpegProgress, Progress
from pipeline.service.jobs import AcceptRequest, JobFinished, JobManager, JobNotFound, RepositoryWritesRefused, \
    RunRequest, ScanRequest, writes_repositories


def _until(condition, timeout=5.0):
    deadline = time.time() + timeout
    while not condition():
        assert time.time() < deadline, 'timed out'
        time.sleep(0.005)


class Gate:
    ''' An execute that records the order jobs ran in and holds each until released. '''

    def __init__(self):
        self.order, self.entered, self.release = [], threading.Event(), threading.Event()

    def __call__(self, job, control):
        self.order.append(job.id)
        self.entered.set()
        assert self.release.wait(5)
        return f'done {job.kind}'


@pytest.fixture
def managers():
    made = []

    def make(execute, **kwargs):
        manager = JobManager(execute, **kwargs)
        made.append(manager)
        return manager
    yield make
    for manager in made:
        manager.stop(grace_seconds=5)


def test_jobs_run_one_at_a_time_in_the_order_they_were_asked(managers):
    gate = Gate()
    manager = managers(gate)
    first, second = manager.submit(ScanRequest()), manager.submit(RunRequest(Selection(kind='movie')))
    _until(gate.entered.is_set)

    assert (first.state, second.state) == ('running', 'queued') and manager.busy
    assert manager.current is first and manager.queued == [second]
    gate.release.set()
    _until(lambda: second.finished)
    assert gate.order == [first.id, second.id]
    assert (first.state, first.result, second.result) == ('succeeded', 'done scan', 'done run')
    assert first.started_at >= first.submitted_at and first.finished_at >= first.started_at and not manager.busy
    assert [job.id for job in manager.jobs()] == [second.id, first.id]   # newest first


def test_a_queued_job_is_dropped_and_a_running_one_stops_before_its_next_title(managers):
    titles_done = []

    def execute(job, control):
        for title in range(1000):
            if control.cancelled():
                return 'stopped'
            titles_done.append(title)
            time.sleep(0.001)
        return 'all'
    manager = managers(execute)
    running, waiting = manager.submit(ScanRequest()), manager.submit(ScanRequest())
    _until(lambda: titles_done)

    manager.cancel(waiting.id)
    manager.cancel(running.id)
    _until(lambda: running.finished)

    assert waiting.state == 'cancelled' and waiting.started_at is None
    assert running.state == 'cancelled' and running.result == 'stopped' and len(titles_done) < 1000
    with pytest.raises(JobFinished):
        manager.cancel(running.id)
    with pytest.raises(JobNotFound):
        manager.cancel('nope')


def test_a_job_that_raises_fails_with_the_message_redacted_and_the_next_one_still_runs(managers):
    def execute(job, control):
        if job.kind == 'scan':
            raise ValueError('designer at https://me:hunter2@host refused, api_key=abc123')
        return 'ok'
    manager = managers(execute)
    bad, good = manager.submit(ScanRequest()), manager.submit(AcceptRequest(dry_run=True))
    _until(lambda: good.finished)

    assert bad.state == 'failed' and bad.error.startswith('ValueError: designer at https://[REDACTED]@host')
    assert 'hunter2' not in bad.error and 'abc123' not in bad.error
    assert good.state == 'succeeded'


def test_a_result_that_says_it_failed_or_was_cancelled_is_recorded_so(managers):
    class Report:
        def __init__(self, failed=False, cancelled=False):
            self.failed, self.cancelled = failed, cancelled
    results = iter([Report(failed=True), Report(cancelled=True), Report()])
    manager = managers(lambda job, control: next(results), failed=lambda result: result.failed)
    jobs = [manager.submit(ScanRequest()) for _ in range(3)]
    _until(lambda: jobs[-1].finished)

    assert [job.state for job in jobs] == ['failed', 'cancelled', 'succeeded']


def test_progress_and_events_are_kept_redacted_and_passed_to_listeners(managers):
    def execute(job, control):
        control.progress(Progress(0, 2, 'Alien', 'extract', 'a'))
        control.progress(FfmpegProgress('Alien', 'a', 50, 100))
        control.event(ExecutionEvent('r', 'a', 'design', 'command_finished', 0.0, 'ran', ('d', '--password', 'pw'),
                                     stdout='Authorization: Bearer secret-token'))
        control.event('not an event')   # ignored
        return 'ok'
    heard = []
    manager = managers(execute)
    unsubscribe = manager.subscribe(lambda job, event: heard.append(event['type']))
    job = manager.submit(ScanRequest())
    _until(lambda: job.finished)
    unsubscribe()

    events = manager.events(job.id)
    assert [e['type'] for e in events] == ['state', 'state', 'progress', 'ffmpeg', 'event', 'state'] == heard
    assert [e['seq'] for e in events] == sorted(e['seq'] for e in events)
    assert job.progress == Progress(0, 2, 'Alien', 'extract', 'a')
    text = events[4]['text']
    assert 'secret-token' not in text and "Command: d --password '[REDACTED]'" in text and ' pw' not in text
    assert manager.events(job.id, after=events[3]['seq']) == events[4:]


def test_a_listener_that_raises_does_not_stop_the_job(managers):
    manager = managers(lambda job, control: 'ok')
    manager.subscribe(lambda job, event: 1 / 0)
    job = manager.submit(ScanRequest())
    _until(lambda: job.finished)

    assert job.state == 'succeeded'


@pytest.mark.parametrize('request_, writes', [
    (RunRequest(through='design'), False), (RunRequest(through='publish'), True), (RunRequest(through='commit'), True),
    (AcceptRequest(), True), (AcceptRequest(dry_run=True), False), (ScanRequest(), False)])
def test_publish_commit_and_accept_are_refused_unless_the_config_allows_them(managers, request_, writes):
    assert writes_repositories(request_) is writes
    refusing, allowing = managers(lambda job, control: 'ok'), managers(lambda job, control: 'ok', allow_repository_writes=True)

    if writes:
        with pytest.raises(RepositoryWritesRefused, match='allow_repository_writes'):
            refusing.submit(request_)
    else:
        refusing.submit(request_)
    allowing.submit(request_)


def test_requests_check_themselves():
    with pytest.raises(ValueError, match='through must be one of'):
        RunRequest(through='review')
    with pytest.raises(ValueError, match='threshold must be between 0 and 1'):
        AcceptRequest(threshold=1.5)


def test_the_history_is_kept_on_disk_and_a_job_cut_off_by_a_stop_is_interrupted(managers, tmp_path):
    gate = Gate()
    first = managers(gate, state_dir=str(tmp_path))
    done = first.submit(ScanRequest(sources=('films',)))
    gate.release.set()
    _until(lambda: done.finished)
    gate.release.clear()
    cut_off = first.submit(RunRequest(Selection(year='2026'), through='extract'))
    _until(lambda: cut_off.state == 'running')
    saved = json.loads((tmp_path / 'jobs.json').read_text())
    assert [(j['id'], j['state']) for j in saved] == [(done.id, 'succeeded'), (cut_off.id, 'running')]
    assert saved[1]['request']['selection']['year'] == '2026' and saved[1]['request']['through'] == 'extract'

    restarted = managers(lambda job, control: 'ok', state_dir=str(tmp_path))   # as if the process had died

    back = restarted.get(cut_off.id)
    assert back.state == 'interrupted' and 'stopped while the job was running' in back.error
    assert restarted.get(done.id).state == 'succeeded' and restarted.get(done.id).request == {'sources': ['films'],
                                                                                            'allow_empty': False}
    gate.release.set()


def test_only_the_latest_finished_jobs_are_kept(managers, tmp_path):
    manager = managers(lambda job, control: 'ok', state_dir=str(tmp_path), history_limit=2)
    jobs = [manager.submit(ScanRequest()) for _ in range(4)]
    _until(lambda: jobs[-1].finished)

    assert [job.id for job in manager.jobs()] == [jobs[3].id, jobs[2].id]
    assert len(json.loads((tmp_path / 'jobs.json').read_text())) == 2


def test_a_damaged_history_is_ignored(managers, tmp_path):
    (tmp_path / 'jobs.json').write_text('{not json')

    assert managers(lambda job, control: 'ok', state_dir=str(tmp_path)).jobs() == []


def test_stopping_cancels_what_waits_and_refuses_new_work(managers):
    gate = Gate()
    manager = managers(gate)
    running, waiting = manager.submit(ScanRequest()), manager.submit(ScanRequest())
    _until(gate.entered.is_set)
    gate.release.set()

    manager.stop(grace_seconds=5)

    assert waiting.state == 'cancelled' and running.finished
    with pytest.raises(RuntimeError, match='stopping'):
        manager.submit(ScanRequest())
