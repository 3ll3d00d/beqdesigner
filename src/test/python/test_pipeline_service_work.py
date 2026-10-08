'''
pipeline/service/work.py: what a scan, run and accept job does -- the command line's own calls, on a profile read again for
each job. The parity test holds a service run to exactly what `run --profile FILE` does with the same file.
'''
import json
import time

import pytest

from pipeline.library import cli
from pipeline.library.index import ScanResult
from pipeline.library.selection import Selection
from pipeline.library.stages import StagesReport
from pipeline.service import work
from pipeline.service.jobs import AcceptRequest, JobManager, RunRequest, ScanRequest
from pipeline.service.work import RunOutcome, executor, job_failed


def _until(condition, timeout=30.0):   # a slow Windows runner took over 10 s
    deadline = time.time() + timeout
    while not condition():
        assert time.time() < deadline, 'timed out'
        time.sleep(0.005)


@pytest.fixture(autouse=True)
def designer_answers(monkeypatch):
    ''' The profile's designer (port 9, nothing there) answers /health: these tests are not about the designer. '''
    monkeypatch.setattr('pipeline.service.context.check_designer', lambda *args, **kwargs: None)


@pytest.fixture
def profile(tmp_path):
    media = tmp_path / 'media'
    media.mkdir()
    for name in ('Alien (1979).mkv', 'Dune (2021).mkv'):
        (media / name).write_bytes(b'')
    config = {'sources': [{'name': 'disk', 'kind': 'filesystem', 'globs': [str(media)]}],
              'designers': {'rolloff': 'http://127.0.0.1:9/design'},
              'run': {'work_dir': str(tmp_path / 'work'), 'queue_dir': str(tmp_path / 'queue'), 'designer': 'rolloff',
                      'parallelism': {'extract': 2}, 'target_fs': 500},
              'sync': {'filter_repo': str(tmp_path / 'filters')}}
    path = tmp_path / 'profile.json'
    path.write_text(json.dumps(config))
    return path


@pytest.fixture
def manager(profile):
    made = JobManager(executor(str(profile), env={}), failed=job_failed, allow_repository_writes=True)
    yield made
    made.stop(grace_seconds=5)


def _done(manager, request):
    job = manager.submit(request)
    _until(lambda: job.finished)
    return job


def test_a_scan_job_lists_the_profiles_sources_into_the_index(manager):
    job = _done(manager, ScanRequest())

    assert job.state == 'succeeded' and isinstance(job.result, ScanResult)
    assert job.result.titles == 2 and job.result.errors == {}


def test_a_scan_of_a_source_the_profile_does_not_have_fails_the_job_and_names_the_ones_it_has(manager):
    job = _done(manager, ScanRequest(sources=('films',)))

    assert job.state == 'failed' and 'no such source: films (the profile has: disk)' in job.error


def test_a_run_scans_first_then_runs_the_stages_on_the_selection(manager, monkeypatch):
    seen = {}

    def fake_run_stages(profile, selection, through, **kwargs):
        seen.update(selection=selection, through=through, titles=len(kwargs['index'].titles()), **kwargs)
        kwargs['on_progress'](type('P', (), {})())   # ignored: not a Progress
        return StagesReport(through, 2)
    monkeypatch.setattr(work, 'run_stages', fake_run_stages)

    job = _done(manager, RunRequest(Selection(needs=('extract',), year='>=2000'), through='extract', retry_failed=True))

    assert job.state == 'succeeded' and isinstance(job.result, RunOutcome) and job.result.scan.titles == 2
    assert seen['titles'] == 2                        # the scan came first
    assert seen['selection'] == Selection(needs=('extract',), year='>=2000') and seen['through'] == 'extract'
    assert seen['retry_failed'] is True and seen['publish'] is None and seen['should_cancel']() is False
    assert seen['unattended'] is False    # an API run is a person's: a failed extraction is tried again
    from pipeline.library.join import JoinQueue
    # design/archive/library-sync/worklist-feedback.md F5: other runs hand it their work
    assert isinstance(seen['join'], JoinQueue)

    _done(manager, RunRequest(scan_first=False, unattended=True))   # the schedule's
    assert seen['unattended'] is True
    assert seen['run_config'].designer == 'rolloff' and seen['run_config'].extract_parallelism == 2


def test_a_run_that_does_not_scan_first_still_scans_an_index_that_has_never_been(manager, monkeypatch):
    monkeypatch.setattr(work, 'run_stages', lambda *a, **k: StagesReport('design', 0))

    first = _done(manager, RunRequest(scan_first=False))
    second = _done(manager, RunRequest(scan_first=False))

    assert first.result.scan is not None and second.result.scan is None


def test_the_service_runs_what_the_command_line_runs_for_the_same_profile(manager, profile, monkeypatch, capsys):
    ''' Parity (design/pipeline-service.md §4): the same file is the same run from either. '''
    calls = []

    def capture(profile_, selection, through, **kwargs):
        calls.append(dict(profile=profile_, selection=selection, through=through, run_config=kwargs['run_config'],
                          settings=kwargs['settings'], publish=kwargs['publish'], retry_failed=kwargs['retry_failed'],
                          unattended=kwargs['unattended']))
        return StagesReport(through, 0)
    monkeypatch.setattr(cli, 'run_stages', capture)
    monkeypatch.setattr(work, 'run_stages', capture)

    for through in ('design', 'commit'):
        assert cli.main(['run', '--profile', str(profile), '--needs', 'extract', '--kind', 'movie', '--through', through]) == 0
        assert _done(manager, RunRequest(Selection(needs=('extract',), kind='movie'), through=through,
                                         scan_first=False)).state == 'succeeded'

    by_cli, by_service = calls[0::2], calls[1::2]
    assert by_cli == by_service
    assert by_service[1]['publish'] is not None and by_service[1]['publish'].xml_repo.local_path.endswith('filters')


def test_a_run_with_a_designer_the_profile_does_not_declare_fails_before_anything_runs(tmp_path, profile, monkeypatch):
    config = json.loads(profile.read_text())
    config['run']['designer'] = 'nobody'
    profile.write_text(json.dumps(config))
    monkeypatch.setattr(work, 'run_stages', lambda *a, **k: pytest.fail('ran'))
    manager = JobManager(executor(str(profile), env={}), failed=job_failed)
    try:
        job = _done(manager, RunRequest())
    finally:
        manager.stop(grace_seconds=5)

    assert job.state == 'failed' and "designer 'nobody' is not registered" in job.error


def test_accept_needs_a_scanned_index_and_a_dry_run_changes_nothing(manager):
    before = _done(manager, AcceptRequest(dry_run=True))
    _done(manager, ScanRequest())
    after = _done(manager, AcceptRequest(dry_run=True))

    assert before.state == 'failed' and 'no index at' in before.error
    assert after.state == 'succeeded' and after.result.eligible == [] and after.result.not_for_review == 2


def test_a_job_failed_when_a_title_failed_or_a_source_could_not_be_listed():
    assert job_failed(ScanResult(1, 1, errors={'disk': 'gone'})) and not job_failed(ScanResult(1, 1))
    assert job_failed(RunOutcome(ScanResult(1, 1, errors={'disk': 'down'}), StagesReport('design', 0)))
    assert not job_failed(RunOutcome(None, StagesReport('design', 0))) and not job_failed('anything else')


def test_a_run_job_while_the_work_list_holds_the_work_directory_hands_its_titles_to_that_run(manager, tmp_path, monkeypatch):
    ''' design/archive/library-sync/worklist-feedback.md F5: not refused, not run alongside: joined, and reported from
        the index. '''
    import sqlite3
    import threading
    from pipeline.library.inbox import WorkDirInbox
    from pipeline.library.index import index_path
    from pipeline.service.lease import WorkDirLease
    work_dir = str(tmp_path / 'work')
    assert _done(manager, ScanRequest()).state == 'succeeded'
    monkeypatch.setattr(work, 'run_stages', lambda *a, **k: pytest.fail('ran alongside the run in progress'))
    lease = WorkDirLease(work_dir, 'worklist-1', host='desk', pid=1).__enter__()

    def worklist_run():   # the work list's run takes them, extracts one, fails the other, and ends
        inbox = WorkDirInbox(work_dir)
        while not (taken := inbox.claim()):
            time.sleep(0.02)
        ids = sorted(taken[0].selection.ids)
        db = sqlite3.connect(index_path(work_dir))
        with db:
            db.execute("UPDATE titles SET needs = 'design', extract_state = 'current' WHERE id = ?", (ids[0],))
            db.execute("UPDATE titles SET extract_state = 'failed', detail = 'extract failed: boom' WHERE id = ?",
                       (ids[1],))
        db.close()
        lease.__exit__(None, None, None)

    thread = threading.Thread(target=worklist_run)
    thread.start()
    job = _done(manager, RunRequest(Selection(needs=('extract',)), through='extract', scan_first=False))
    thread.join()

    assert job.state == 'failed' and isinstance(job.result, RunOutcome)   # one of its titles failed
    report = job.result.report
    assert len(report.attempted) == 2 and [message for _, message in report.run.failed] == ['extract failed: boom']
    assert any(e.get('kind') == 'handed_off' for e in job.events)
