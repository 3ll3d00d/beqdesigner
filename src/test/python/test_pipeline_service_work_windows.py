'''
pipeline/service/work.py with job windows (design/job-windows.md §3): a run's window closing stops `run_stages` between
titles, a redesign is sent back once and carries on as a plain run, the parts' reports merge into one result the API reads,
and a run waiting for the work directory pauses rather than holding the worker.
'''
import json
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone

import pytest

import pipeline.service.executions as jobs
from pipeline.library.run import LibraryRunReport
from pipeline.library.selection import Selection
from pipeline.library.stages import Progress, StagesReport
from pipeline.service import work
from pipeline.service.executions import ExecutionQueue, RunRequest, ScanRequest
from pipeline.service.models import run_result
from pipeline.service.work import RunOutcome, executor, job_failed, merge_outcomes, remaining_titles

UTC = timezone.utc


def at(text):
    return datetime.strptime(text, '%Y-%m-%d %H:%M').replace(tzinfo=UTC).timestamp()


def _until(condition, timeout=30.0):
    deadline = time.time() + timeout
    while not condition():
        assert time.time() < deadline, 'timed out'
        time.sleep(0.005)


class Clock:
    def __init__(self, text):
        self.t, self.lock = at(text), threading.Lock()

    def __call__(self):
        with self.lock:
            return self.t

    def set(self, text):
        with self.lock:
            self.t = at(text)

    def advance(self, minutes):
        with self.lock:
            self.t += minutes * 60


@pytest.fixture(autouse=True)
def designer_answers(monkeypatch):
    monkeypatch.setattr('pipeline.service.context.check_designer', lambda *args, **kwargs: None)
    monkeypatch.setattr(jobs, 'WINDOW_RECHECK_SECONDS', 0.01)


@pytest.fixture
def profile(tmp_path):
    media = tmp_path / 'media'
    media.mkdir()
    for name in ('Alien (1979).mkv', 'Dune (2021).mkv', 'Heat (1995).mkv'):
        (media / name).write_bytes(b'')
    config = {'sources': [{'name': 'disk', 'kind': 'filesystem', 'globs': [str(media)]}],
              'designers': {'rolloff': 'http://127.0.0.1:9/design'},
              'run': {'work_dir': str(tmp_path / 'work'), 'queue_dir': str(tmp_path / 'queue'), 'designer': 'rolloff',
                      'target_fs': 500},
              'sync': {'filter_repo': str(tmp_path / 'filters')}}
    path = tmp_path / 'profile.json'
    path.write_text(json.dumps(config))
    return path


@pytest.fixture
def clock():
    return Clock('2026-10-26 01:00')


@pytest.fixture
def manager(profile, clock):
    made = ExecutionQueue(executor(str(profile), env={}, poll_seconds=0.01), failed=job_failed, clock=clock, tz=UTC,
                      remaining=remaining_titles, merge=merge_outcomes)
    yield made
    made.stop(grace_seconds=5)


def _scanned(manager, profile):
    from pipeline.library.index import LibraryIndex, index_path
    from pipeline.review import update_entry
    from pipeline.service.context import load_context
    from review_entry_fixture import write_entry
    job = manager.submit(ScanRequest(), window=[])
    _until(lambda: job.finished)
    ctx = load_context(str(profile), {})
    with LibraryIndex(index_path(ctx.work_dir)) as index:
        ids = sorted(row.id for row in index.titles())
    queue = ctx.scan_settings().queue_dir
    for title_id in ids:
        write_entry(queue, title_id)
        update_entry(queue, title_id, design_fingerprint='old')
    return ids


def stages_on(clock, runs, minutes=40):
    ''' run_stages as it treats should_cancel: polled before each title; each title takes `minutes` on the clock. '''
    def stages(profile_, selection, through, **kwargs):
        runs.append(selection.ids)
        report = StagesReport(through, len(selection.ids))
        for n, title_id in enumerate(selection.ids):
            if kwargs['should_cancel']():
                report.cancelled, report.not_run = True, list(selection.ids[n:])
                break
            kwargs['on_progress'](Progress(n, len(selection.ids), title_id, 'design', title_id))
            clock.advance(minutes)
            report.run.designed.append(title_id)
            report.attempted.append(title_id)
        kwargs['on_progress'](Progress(len(report.attempted), len(selection.ids), '', ''))
        return report
    return stages


def test_a_redesign_whose_window_closes_is_sent_back_once_and_carries_on_as_a_plain_run(manager, profile, clock,
                                                                                        monkeypatch):
    ids = _scanned(manager, profile)
    runs, sent_back = [], []
    monkeypatch.setattr(work, 'run_stages', stages_on(clock, runs))
    revise = work.revise_entry
    monkeypatch.setattr(work, 'revise_entry', lambda queue, title_id, *a, **k: (sent_back.append(title_id),
                                                                                revise(queue, title_id, *a, **k))[1])

    job = manager.submit(RunRequest(Selection(ids=tuple(ids)), scan_first=False, redesign=True), window=['01:00-02:00'])
    _until(lambda: job.state == 'paused')
    assert runs == [tuple(ids)] and sent_back == ids   # all three sent back; two designed by 01:00 + 2 x 40 min
    assert job.part_request.selection.ids == (ids[2],) and not job.part_request.redesign

    clock.set('2026-10-27 01:00')
    _until(lambda: job.finished)
    assert job.state == 'succeeded', job.error
    assert runs == [tuple(ids), (ids[2],)] and sent_back == ids   # not sent back again
    result = run_result(job.result)
    assert result.designed == ids and result.attempted == ids and not result.cancelled and result.not_run == []
    assert result.selected == 3 and (job.progress.done, job.progress.total) == (3, 3)


def test_remaining_titles_are_what_a_run_stopped_before_and_none_once_it_stopped_for_another_reason():
    stopped = StagesReport('design', 3, cancelled=True, attempted=['a'], not_run=['b', 'c'])
    assert remaining_titles(RunOutcome(None, stopped)) == ('b', 'c')
    assert remaining_titles(RunOutcome(None, StagesReport('design', 3, attempted=['a', 'b', 'c']))) == ()
    streak = StagesReport('design', 3, cancelled=True, stopped='5 unavailable in a row', not_run=['c'])
    assert remaining_titles(RunOutcome(None, streak)) == ()   # it ends as it would have: failed
    assert remaining_titles(None) is None   # it stopped before it planned anything: do it again


def test_the_parts_merge_into_one_result_that_reads_and_fails_as_one_run():
    first = StagesReport('design', 3, LibraryRunReport(designed=['a'], seasons={'s1': ['a']}), cancelled=True,
                         attempted=['a'], not_run=['b', 'c'], counts={'design': 2})
    second = StagesReport('design', 2, LibraryRunReport(designed=['b'], failed=[('c', 'boom')], seasons={'s2': ['b']}),
                          attempted=['b', 'c'], counts={'design': 0})
    merged = merge_outcomes([asdict(RunOutcome(None, first)), asdict(RunOutcome(None, second))])
    result = run_result(merged)
    assert (result.selected, result.designed, result.attempted) == (3, ['a', 'b'], ['a', 'b', 'c'])
    assert (result.cancelled, result.not_run, result.counts, result.seasons) == \
        (False, [], {'design': 0}, {'s1': ['a'], 's2': ['b']})
    assert [p.id for p in result.failed] == ['c'] and job_failed(merged)
    assert not job_failed(merge_outcomes([asdict(RunOutcome(None, first)),
                                          asdict(RunOutcome(None, StagesReport('design', 2, attempted=['b', 'c'])))]))


def test_a_run_waiting_for_the_work_directory_pauses_when_its_window_closes(manager, profile, clock, tmp_path,
                                                                            monkeypatch):
    from pipeline.service.lease import WorkDirLease
    ids = _scanned(manager, profile)
    runs = []
    monkeypatch.setattr(work, 'run_stages', stages_on(clock, runs, minutes=0))
    lease = WorkDirLease(str(tmp_path / 'work'), 'worklist-1', host='desk', pid=1).__enter__()
    try:   # a redesign does not join the work list's run: it waits for the lease
        job = manager.submit(RunRequest(Selection(ids=tuple(ids)), scan_first=False, redesign=True),
                             window=['01:00-02:00'])
        _until(lambda: job.state == 'running')
        clock.set('2026-10-26 02:00')
        _until(lambda: job.state == 'paused')
        assert manager.current is None and job.part_request is None   # it carries on with all of it
    finally:
        lease.__exit__(None, None, None)
    clock.set('2026-10-27 01:00')
    _until(lambda: job.finished)
    assert job.state == 'succeeded', job.error
    assert runs == [tuple(ids)]
