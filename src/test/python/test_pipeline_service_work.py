'''
pipeline/service/work.py: what a scan, run and accept job does -- the command line's own calls, on a profile read again for
each job. The parity test holds a service run to exactly what `run --profile FILE` does with the same file.
'''
import json
import os
import time

import pytest

from pipeline.library import cli
from pipeline.library.index import ScanResult
from pipeline.library.selection import Selection
from pipeline.library.stages import StagesReport
from pipeline.service import work
from pipeline.service.executions import AcceptRequest, ExecutionQueue, RunRequest, ScanRequest
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
    made = ExecutionQueue(executor(str(profile), env={}), failed=job_failed, allow_repository_writes=True)
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
    manager = ExecutionQueue(executor(str(profile), env={}), failed=job_failed)
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


@pytest.mark.parametrize('status', ['pending', 'accepted', 'rejected'])
def test_redesign_reopens_only_the_requested_title_before_running(manager, profile, monkeypatch, status):
    from pipeline.library.index import LibraryIndex, index_path
    from pipeline.review import read_entry, update_entry
    from pipeline.service.context import load_context
    from review_entry_fixture import write_entry

    _done(manager, ScanRequest())
    ctx = load_context(str(profile), {})
    with LibraryIndex(index_path(ctx.work_dir)) as index:
        ids = [row.id for row in index.titles()]
    queue = ctx.scan_settings().queue_dir
    for title_id in ids:
        write_entry(queue, title_id, status=status)
        update_entry(queue, title_id, design_fingerprint='old')
    seen = []

    def stages(profile_, selection, through, **kwargs):
        entry = read_entry(queue, ids[0])
        seen.append(selection.ids)
        assert entry.status == 'pending' and entry.chosen_candidate_index is None
        assert entry.design_fingerprint is None
        assert entry.meta['title'] == ids[0] and len(entry.candidates) == 2
        assert read_entry(queue, ids[1]).design_fingerprint == 'old'
        return StagesReport(through, 1)

    monkeypatch.setattr(work, 'run_stages', stages)
    job = _done(manager, RunRequest(Selection(ids=(ids[0],)), scan_first=False, redesign=True))
    assert job.state == 'succeeded', job.error
    assert seen == [(ids[0],)]
    assert 'Sent back for redesign' in read_entry(queue, ids[0]).reviewer_note


def _scanned_with_entries(manager, profile, status='pending'):
    from pipeline.library.index import LibraryIndex, index_path
    from pipeline.review import update_entry
    from pipeline.service.context import load_context
    from review_entry_fixture import write_entry

    _done(manager, ScanRequest())
    ctx = load_context(str(profile), {})
    with LibraryIndex(index_path(ctx.work_dir)) as index:
        ids = [row.id for row in index.titles()]
    queue = ctx.scan_settings().queue_dir
    for title_id in ids:
        write_entry(queue, title_id, status=status)
        update_entry(queue, title_id, design_fingerprint='old')
    return ids, queue


def test_redesign_sends_several_titles_back_and_designs_them_in_one_run(manager, profile, monkeypatch):
    ''' A retry phase: one job, so one run takes them all and the whole designer pool is used. '''
    from pipeline.review import read_entry
    ids, queue = _scanned_with_entries(manager, profile)
    assert len(ids) >= 2
    runs = []

    def stages(profile_, selection, through, **kwargs):
        runs.append(selection.ids)
        assert all(read_entry(queue, t).design_fingerprint is None for t in selection.ids)
        return StagesReport(through, len(selection.ids))

    monkeypatch.setattr(work, 'run_stages', stages)
    job = _done(manager, RunRequest(Selection(ids=tuple(ids[:2])), scan_first=False, redesign=True))

    assert job.state == 'succeeded', job.error
    assert runs == [tuple(ids[:2])]
    assert all('Sent back for redesign' in read_entry(queue, t).reviewer_note for t in ids[:2])
    assert all(read_entry(queue, t).design_fingerprint == 'old' for t in ids[2:])


@pytest.mark.parametrize('problem, message', [('unknown', 'no such title: nope'),
                                              ('published', 'allow_repository_writes')])
def test_one_bad_title_fails_a_redesign_before_any_is_sent_back(manager, profile, monkeypatch, problem, message):
    from pipeline.review import read_entry, update_entry
    ids, queue = _scanned_with_entries(manager, profile)
    wanted = (ids[0], 'nope') if problem == 'unknown' else (ids[0], ids[1])
    if problem == 'published':
        update_entry(queue, ids[1], status='published')
    monkeypatch.setattr(work, 'run_stages', lambda *args, **kwargs: pytest.fail('must not run'))

    job = _done(manager, RunRequest(Selection(ids=wanted), scan_first=False, redesign=True))

    assert job.state == 'failed' and message in job.error
    assert read_entry(queue, ids[0]).design_fingerprint == 'old'   # nothing was sent back


def test_redesign_refuses_published_titles_without_repository_permission(manager, profile, monkeypatch):
    from pipeline.library.index import LibraryIndex, index_path
    from pipeline.review import read_entry
    from pipeline.service.context import load_context
    from review_entry_fixture import write_entry

    _done(manager, ScanRequest())
    ctx = load_context(str(profile), {})
    with LibraryIndex(index_path(ctx.work_dir)) as index:
        title_id = index.titles()[0].id
    queue = ctx.scan_settings().queue_dir
    write_entry(queue, title_id, status='published')
    monkeypatch.setattr(work, 'run_stages', lambda *args, **kwargs: pytest.fail('must not run'))
    job = _done(manager, RunRequest(Selection(ids=(title_id,)), scan_first=False, redesign=True))
    assert job.state == 'failed' and 'allow_repository_writes' in job.error
    assert read_entry(queue, title_id).status == 'published'


# --- a redesign by design version (design/design-version.md) --------------------------------------------------------

def test_redesign_by_version_sends_back_the_designs_out_of_date_that_nobody_accepted(manager, profile, monkeypatch):
    from pipeline.review import read_entry, update_entry
    from review_entry_fixture import write_entry
    monkeypatch.setattr('pipeline.service.context.check_designer', lambda *args, **kwargs: {
        'design_version': 4, 'design_settings': 'd0', 'design_settings_detail': {}})
    ids, queue = _scanned_with_entries(manager, profile)
    write_entry(queue, ids[1], status='accepted')
    for title_id in ids:
        update_entry(queue, title_id, design_version=3, design_settings='d0')
    runs = []

    def stages(profile_, selection, through, **kwargs):
        runs.append(selection)
        return StagesReport(through, len(selection.ids))

    monkeypatch.setattr(work, 'run_stages', stages)
    job = _done(manager, RunRequest(Selection(out_of_date=True), redesign=True))

    assert job.state == 'succeeded', job.error
    assert [s.ids for s in runs] == [(ids[0],)] and runs[0].out_of_date is None   # the run designs them by id
    assert 'Sent back for redesign' in read_entry(queue, ids[0]).reviewer_note
    assert read_entry(queue, ids[1]).status == 'accepted'   # a person's decision stands


def test_redesign_by_version_with_nothing_out_of_date_runs_nothing(manager, profile, monkeypatch):
    from pipeline.review import update_entry
    monkeypatch.setattr('pipeline.service.context.check_designer', lambda *args, **kwargs: {'design_version': 4})
    ids, queue = _scanned_with_entries(manager, profile)
    for title_id in ids:
        update_entry(queue, title_id, design_version=4)
    monkeypatch.setattr(work, 'run_stages', lambda *args, **kwargs: pytest.fail('nothing to run'))

    job = _done(manager, RunRequest(Selection(out_of_date=True), redesign=True))

    assert job.state == 'succeeded', job.error


def test_out_of_date_fails_a_job_while_no_designer_says_its_version(manager, profile, monkeypatch):
    _scanned_with_entries(manager, profile)   # the designer answers /health, but says no design version
    monkeypatch.setattr(work, 'run_stages', lambda *args, **kwargs: pytest.fail('nothing to run'))

    job = _done(manager, RunRequest(Selection(out_of_date=True), redesign=True))

    assert job.state == 'failed' and 'no designer that is up says its design version' in job.error


def test_a_redesign_takes_ids_or_out_of_date_but_not_both():
    RunRequest(Selection(out_of_date=True, needs=('review',)), redesign=True)
    for selection in (Selection(out_of_date=True, ids=('a',)), Selection(out_of_date=False), Selection()):
        with pytest.raises(ValueError, match='redesign requires'):
            RunRequest(selection, redesign=True)
    with pytest.raises(ValueError, match='redesign requires'):
        RunRequest(Selection(out_of_date=True), through='publish', redesign=True)


# --- revise to extract, and a person's audio stream (design/web-parity.md WP5) -----------------------------------------

_STREAMS = ({'codec': 'LPCM', 'channels': '2', 'audio_types': ('LPCM 2.0',)},
            {'codec': 'DTS-HD MA', 'channels': '6', 'audio_types': ('DTS-HD MA 5.1',)})


def test_a_redesign_to_extract_forgets_the_extraction_before_running(manager, profile, monkeypatch):
    from pipeline.library.extract_cache import _write_manifest, read_manifest
    from pipeline.library.workdir import entry_directory
    from pipeline.review import read_entry
    from pipeline.service.context import load_context
    ids, queue = _scanned_with_entries(manager, profile)
    work_dir = load_context(str(profile), {}).work_dir
    folder = entry_directory(work_dir, ids[0])
    _write_manifest(folder, {'mono_wav': 'mono.wav', 'source_channels': 6})
    monkeypatch.setattr(work, 'run_stages', lambda profile_, selection, through, **kw: StagesReport(through, 1))

    job = _done(manager, RunRequest(Selection(ids=(ids[0],)), scan_first=False, redesign=True, revise_to='extract'))

    assert job.state == 'succeeded', job.error
    assert read_manifest(folder) == {'source_channels': 6}
    assert 'Sent back for re-extraction' in read_entry(queue, ids[0]).reviewer_note


@pytest.mark.parametrize('designed', [True, False])
def test_a_persons_audio_stream_is_kept_in_the_index_and_the_entry_before_running(manager, profile, monkeypatch,
                                                                                 designed):
    from pipeline.library.index import LibraryIndex, index_path
    from pipeline.review import read_entry
    from pipeline.service.context import load_context
    ids, queue = _scanned_with_entries(manager, profile)
    ctx = load_context(str(profile), {})
    with LibraryIndex(index_path(ctx.work_dir)) as index:
        index.set_audio_stream_details(ids[0], _STREAMS)
    if not designed:
        os.remove(os.path.join(queue, f'{ids[0]}.json'))
    seen = []
    monkeypatch.setattr(work, 'run_stages',
                        lambda profile_, selection, through, **kw: seen.append(selection.ids) or StagesReport(through, 1))

    job = _done(manager, RunRequest(Selection(ids=(ids[0],)), scan_first=False, redesign=True, revise_to='extract',
                                    audio_stream=0))

    assert job.state == 'succeeded', job.error
    assert seen == [(ids[0],)]
    with LibraryIndex(index_path(ctx.work_dir), readonly=True) as index:
        item = index.units([ids[0]])[ids[0]]
    assert (item.audio_stream, item.audio_stream_source) == (0, 'manual')
    if designed:
        entry = read_entry(queue, ids[0])
        assert entry.audio_stream == 0 and 'Audio stream 1 selected' in entry.reviewer_note


@pytest.mark.parametrize('fields, says', [(dict(revise_to='review'), 'revise_to must be design or extract'),
                                          (dict(audio_stream=0), 'an audio stream'),
                                          (dict(revise_to='extract', audio_stream=0, ids=('a', 'b')), 'an audio stream'),
                                          (dict(revise_to='extract', audio_stream=-1), 'an audio stream')])
def test_a_revise_job_is_checked_when_it_is_asked_for(fields, says):
    ids = fields.pop('ids', ('a',))
    with pytest.raises(ValueError, match=says):
        RunRequest(Selection(ids=ids), scan_first=False, redesign=True, **fields)


def test_a_stream_chosen_from_streams_read_from_the_file_keeps_them_first(manager, profile, monkeypatch):
    from pipeline.library.index import LibraryIndex, index_path
    from pipeline.service.context import load_context
    ids, queue = _scanned_with_entries(manager, profile)
    ctx = load_context(str(profile), {})
    probed = []
    monkeypatch.setattr(work, 'probe_stream_details', lambda item: probed.append(item.id) or _STREAMS)
    monkeypatch.setattr(work, 'run_stages', lambda profile_, selection, through, **kw: StagesReport(through, 1))

    job = _done(manager, RunRequest(Selection(ids=(ids[0],)), scan_first=False, redesign=True, revise_to='extract',
                                    audio_stream=1))

    assert job.state == 'succeeded', job.error
    assert probed == [ids[0]]
    with LibraryIndex(index_path(ctx.work_dir), readonly=True) as index:
        item = index.units([ids[0]])[ids[0]]
    assert len(item.audio_stream_details) == 2 and (item.audio_stream, item.audio_stream_source) == (1, 'manual')
