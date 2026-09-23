'''
run_stages() -- doing the work for a selection of titles from the discovery index (design.md §12.7): `through`
semantics, cancel, retry of remembered failures, and the publish / republish / commit stages against real temp git repos.
Extraction and design themselves are faked (they are covered by their own tests) but leave the real outputs a scan reads,
so what is asserted is the index afterwards, as a user would see it.
'''
import os
import subprocess
import threading
import time

import pytest

from pipeline.library.run import LibraryRunConfig, LibraryRunReport, UnitWork
from pipeline.library.selection import Selection
from pipeline.library.stages import Progress, PublishSettings, run_stages
from pipeline.library.status import ScanSettings, failure_key
from pipeline.review import read_entry, update_entry
from pipeline.publish.git import RepoTarget
from test_pipeline_library_commit import IMAGES_NAME, OWNER, _commits, _on_remote, repos  # noqa: F401 (a fixture)
from test_pipeline_library_index import (CONFIG, DESIGNER, FakeSource, _entry, _extracted, _item,  # noqa: F401
                                         _needs, _profile, _published, _ready, _row, _scan, env)


@pytest.fixture
def work(env, monkeypatch):
    ''' Fakes extraction and design, recording what was asked, and leaving the outputs a scan reads. '''

    class Work:
        calls = []
        fail = {}          # (stage, item id) -> exception

    def extract(session, item, item_dir, config, mono_mix=True, force=False, on_progress=None):
        Work.calls.append(('extract', item.id))
        if ('extract', item.id) in Work.fail:
            raise Work.fail[('extract', item.id)]
        _extracted(env, item)
        return os.path.join(item_dir, 'mono.wav'), False

    def design(session, item, wav_path, designer, queue_dir, config, **kwargs):
        Work.calls.append(('design', item.id))
        if ('design', item.id) in Work.fail:
            raise Work.fail[('design', item.id)]
        if not os.path.isfile(os.path.join(queue_dir, f'{item.id}.json')):
            _entry(env, item, confidence=0.95)
        return type('R', (), {'designed': True, 'project_edit_preserved': False})()

    monkeypatch.setattr('pipeline.library.run.extract_if_needed', extract)
    monkeypatch.setattr('pipeline.library.run.design_if_needed', design)
    Work.calls = []
    Work.fail = {}
    return Work


def _run_config(env, **kwargs):
    return LibraryRunConfig(work_dir=env.work, queue_dir=env.queue, designer=DESIGNER, **kwargs)


def _go(env, selection, through, settings=None, **kwargs):
    settings = settings or env.settings
    return run_stages(_profile(env), selection, through, run_config=_run_config(env), index=env.index,
                      settings=settings, **kwargs)


# --- through: extract / design -------------------------------------------------------------------------------------

def test_through_extract_only_extracts_and_leaves_the_title_needing_design(env, work):
    _scan(env, _item('a'))

    report = _go(env, Selection(), 'extract')

    assert work.calls == [('extract', 'fs-a')]
    assert report.run.extracted == ['fs-a'] and report.run.designed == []
    assert _needs(env, 'fs-a')[0] == 'design' and report.counts['design'] == 1


def test_through_design_extracts_first_when_the_title_has_not_been_and_then_designs(env, work):
    _scan(env, _item('a'))

    report = _go(env, Selection(), 'design')

    assert work.calls == [('extract', 'fs-a'), ('design', 'fs-a')]
    assert report.run.designed == ['fs-a']
    assert _needs(env, 'fs-a')[0] == 'review'  # review is a person's: the machine stops here


def test_through_design_on_an_extracted_title_only_designs(env, work):
    item = _item('a')
    _extracted(env, item)
    _scan(env, item)

    _go(env, Selection(), 'design')

    assert [stage for stage, _ in work.calls if stage == 'design'] == ['design']
    assert _needs(env, 'fs-a')[0] == 'review'


def test_only_machine_titles_run_and_the_rest_are_skipped_with_the_reason(env, work):
    todo, waiting, skipped_by_person = _item('todo'), _item('waiting'), _item('rejected')
    for item in (waiting, skipped_by_person):
        _extracted(env, item)
    _entry(env, waiting)
    _entry(env, skipped_by_person, status='rejected')
    _scan(env, todo, waiting, skipped_by_person)

    report = _go(env, Selection(), 'design')

    assert {c[1] for c in work.calls} == {'fs-todo'}
    reasons = {s.id: s.reason for s in report.skipped}
    assert reasons['fs-waiting'] == 'waiting for a person to review it'
    assert reasons['fs-rejected'] == 'rejected'
    assert report.selected == 3 and report.attempted == ['fs-todo']


def test_a_selection_narrows_what_runs(env, work):
    _scan(env, _item('alpha'), _item('beta'), _item('gamma'))

    _go(env, Selection(match='beta'), 'extract')
    assert work.calls == [('extract', 'fs-beta')]

    work.calls.clear()
    _go(env, Selection(ids=('fs-gamma',)), 'extract')
    assert work.calls == [('extract', 'fs-gamma')]

    work.calls.clear()
    _go(env, Selection(needs=('design',)), 'extract')  # nothing needs design yet, whatever the stage asked
    assert work.calls == []


def test_a_selection_by_source_runs_only_that_sources_titles(env, work):
    sources = {'films': FakeSource([_item('a')]), 'disk': FakeSource([_item('b', source_path='/disk/b.mkv')])}
    _scan(env, sources=sources)

    _go(env, Selection(source='disk'), 'extract')

    assert work.calls == [('extract', 'fs-b')]


def test_new_since_scan_selects_only_the_newest_titles(env, work):
    _scan(env, _item('old'))
    _scan(env, _item('old'), _item('fresh'))

    _go(env, Selection(new_since_scan=True), 'extract')

    assert work.calls == [('extract', 'fs-fresh')]


def test_a_protected_accepted_title_is_never_redesigned_by_a_run_through_design(env, work):
    item = _item('a')
    _extracted(env, item)
    _entry(env, item, status='accepted')
    _scan(env, item)
    before = read_entry(env.queue, 'fs-a')

    report = _go(env, Selection(ids=('fs-a',)), 'design')

    assert work.calls == [] and read_entry(env.queue, 'fs-a') == before
    assert report.skipped[0].reason == 'publishing needs --through publish (or commit)'


def test_the_index_is_refreshed_after_a_run_without_counting_as_a_scan(env, work):
    _scan(env, _item('a'))
    generation, last_scan = env.index.generation, env.index.summary().last_scan_at

    _scan(env, _item('a'), _item('b'))  # a second scan: b is new
    generation, last_scan = env.index.generation, env.index.summary().last_scan_at
    _go(env, Selection(ids=('fs-a',)), 'design')

    summary = env.index.summary()
    assert (summary.generation, summary.last_scan_at) == (generation, last_scan)
    assert summary.new == 1 and env.index.title('fs-b').is_new  # the new marker survives the run
    assert _needs(env, 'fs-a')[0] == 'review'


# --- progress and cancel -------------------------------------------------------------------------------------------

def test_progress_is_determinate_and_names_the_title_and_stage(env, work):
    _scan(env, _item('a'), _item('b'))
    seen = []

    _go(env, Selection(), 'design', on_progress=seen.append)

    assert all(isinstance(p, Progress) and p.total == 2 for p in seen)
    assert {(p.stage, p.id) for p in seen[:-1]} == {
        ('extract', 'fs-a'), ('design', 'fs-a'), ('extract', 'fs-b'), ('design', 'fs-b')}
    assert all(0 <= p.done <= p.total for p in seen)
    assert (seen[-1].done, seen[-1].stage) == (2, '')


def test_run_stages_emits_structured_title_and_stage_events(env, work):
    _scan(env, _item('a'))
    seen = []

    _go(env, Selection(), 'design', on_event=seen.append)

    assert [event.kind for event in seen] == [
        'queued', 'stage_started', 'stage_completed', 'stage_queued', 'stage_started', 'stage_completed',
        'title_completed']
    assert {event.run_id for event in seen} and len({event.run_id for event in seen}) == 1
    assert {event.title_id for event in seen} == {'fs-a'}
    assert [event.stage for event in seen if event.kind.startswith('stage_')] == [
        'extract', 'extract', 'design', 'design', 'design']


def test_extract_and_design_stages_overlap_without_holding_each_others_capacity(env, monkeypatch):
    from pipeline.library import stages

    _scan(env, _item('a'), _item('b'))
    design_started = threading.Event()
    extraction_overlapped_design = threading.Event()

    def extract(session, unit, config, local_report, index, **kwargs):
        item = unit.item if hasattr(unit, 'item') else unit
        if item.id == 'fs-b':
            assert design_started.wait(5), 'design stage did not start while extraction capacity was free'
            extraction_overlapped_design.set()
        return UnitWork(unit, item, 'mono.wav', item.id)

    def design(work, config, index, on_stage=None):
        if on_stage:
            on_stage(work.item.id, 'design')
        if work.item.id == 'fs-a':
            design_started.set()
            assert extraction_overlapped_design.wait(5), 'extraction did not overlap the running design'
        return LibraryRunReport(designed=[work.item.id])

    monkeypatch.setattr(stages, 'run_unit', extract)
    monkeypatch.setattr(stages, 'design_unit_work', design)
    config = _run_config(env, extract_parallelism=1, design_parallelism=1)

    report = run_stages(_profile(env), Selection(ids=('fs-a', 'fs-b')), 'design', run_config=config,
                        index=env.index, settings=env.settings)

    assert extraction_overlapped_design.is_set()
    assert sorted(report.run.designed) == ['fs-a', 'fs-b']


def test_extract_and_design_worker_counts_obey_separate_limits(env, monkeypatch):
    from pipeline.library import stages

    _scan(env, *[_item(letter) for letter in 'abcd'])
    lock = threading.Lock()
    active = {'extract': 0, 'design': 0, 'max_extract': 0, 'max_design': 0}
    pair = threading.Barrier(2)

    def extract(session, unit, config, local_report, index, **kwargs):
        item = unit.item if hasattr(unit, 'item') else unit
        with lock:
            active['extract'] += 1
            active['max_extract'] = max(active['max_extract'], active['extract'])
        pair.wait(timeout=5)
        time.sleep(0.005)
        with lock:
            active['extract'] -= 1
        return UnitWork(unit, item, 'mono.wav', item.id)

    def design(work, config, index, on_stage=None):
        with lock:
            active['design'] += 1
            active['max_design'] = max(active['max_design'], active['design'])
        time.sleep(0.005)
        with lock:
            active['design'] -= 1
        return LibraryRunReport(designed=[work.item.id])

    monkeypatch.setattr(stages, 'run_unit', extract)
    monkeypatch.setattr(stages, 'design_unit_work', design)
    config = _run_config(env, extract_parallelism=2, design_parallelism=1)
    events = []

    report = run_stages(_profile(env), Selection(ids=tuple(f'fs-{letter}' for letter in 'abcd')), 'design',
                        run_config=config, index=env.index, settings=env.settings, on_event=events.append)

    assert sorted(report.run.designed) == [f'fs-{letter}' for letter in 'abcd']
    assert active['max_extract'] == 2
    assert active['max_design'] == 1
    queued_design_ids = {event.title_id for event in events
                         if event.kind == 'stage_queued' and event.stage == 'design'}
    assert queued_design_ids == {f'fs-{letter}' for letter in 'abcd'}


def test_real_run_units_produce_the_same_outputs_with_independent_stage_pools(env, work):
    serial = [_item(f'serial-{n}') for n in range(3)]
    parallel = [_item(f'parallel-{n}') for n in range(3)]
    _scan(env, *(serial + parallel))

    serial_report = run_stages(_profile(env), Selection(ids=tuple(item.id for item in serial)), 'design',
                               run_config=_run_config(env, extract_parallelism=1, design_parallelism=1),
                               index=env.index, settings=env.settings)
    parallel_report = run_stages(_profile(env), Selection(ids=tuple(item.id for item in parallel)), 'design',
                                 run_config=_run_config(env, extract_parallelism=2, design_parallelism=2),
                                 index=env.index, settings=env.settings)

    assert sorted(serial_report.run.designed) == [item.id for item in serial]
    assert sorted(parallel_report.run.designed) == [item.id for item in parallel]
    for serial_item, parallel_item in zip(serial, parallel):
        assert _needs(env, serial_item.id)[0] == _needs(env, parallel_item.id)[0] == 'review'
        serial_entry = read_entry(env.queue, serial_item.id)
        parallel_entry = read_entry(env.queue, parallel_item.id)
        assert [c.filters for c in serial_entry.candidates] == [c.filters for c in parallel_entry.candidates]
        assert os.path.isfile(os.path.join(env.work, serial_item.id, 'mono.wav'))
        assert os.path.isfile(os.path.join(env.work, parallel_item.id, 'mono.wav'))


def test_a_selected_season_and_its_member_are_rejected_before_either_runs(env, monkeypatch):
    from pipeline.library.season import SeasonGroup

    season_row, member = _item('season'), _item('member')
    _scan(env, season_row, member)
    other_member = _item('other-episode')
    group = SeasonGroup(season_row, (member, other_member))
    monkeypatch.setattr('pipeline.library.stages._units_by_title',
                        lambda index, ids: ({season_row.id: group, member.id: member}, {}))
    calls = []
    monkeypatch.setattr('pipeline.library.stages.run_unit', lambda *args, **kwargs: calls.append(args))

    report = run_stages(_profile(env), Selection(ids=(season_row.id, member.id)), 'design',
                        run_config=_run_config(env), index=env.index, settings=env.settings)

    assert calls == []
    assert {title_id for title_id, _ in report.run.failed} == {season_row.id, member.id}
    assert all('shared by selected titles' in reason for _, reason in report.run.failed)


def test_cancel_between_titles_leaves_only_whole_titles_done_and_says_what_was_not(env, work):
    _scan(env, _item('a'), _item('b'), _item('c'))
    def cancel():
        return len({call[1] for call in work.calls}) >= 1  # after the first title has been worked on

    report = _go(env, Selection(), 'design', should_cancel=cancel)

    assert report.cancelled and report.attempted == ['fs-a'] and sorted(report.not_run) == ['fs-b', 'fs-c']
    assert work.calls == [('extract', 'fs-a'), ('design', 'fs-a')]  # the title in hand finished both stages
    assert _needs(env, 'fs-a')[0] == 'review' and _needs(env, 'fs-b')[0] == 'extract'  # consistent, and refreshed
    assert report.counts['extract'] == 2

    _go(env, Selection(), 'design')  # a later run picks up where it stopped
    assert _needs(env, 'fs-c')[0] == 'review' and [c for c in work.calls if c[1] == 'fs-a'] == [
        ('extract', 'fs-a'), ('design', 'fs-a')]


def test_cancelled_before_anything_runs_does_nothing(env, work):
    _scan(env, _item('a'))

    report = _go(env, Selection(), 'design', should_cancel=lambda: True)

    assert report.cancelled and work.calls == [] and report.not_run == ['fs-a']


# --- failures and retry --------------------------------------------------------------------------------------------

def test_a_failed_title_is_not_retried_until_asked_or_until_its_source_changes(env, work):
    item = _item('a')
    _scan(env, item)
    work.fail[('extract', 'fs-a')] = RuntimeError('ffmpeg exploded')

    first = _go(env, Selection(), 'design')
    assert first.run.failed == [('fs-a', 'RuntimeError: ffmpeg exploded')] and first.failed
    assert _needs(env, 'fs-a')[0] == 'attention'

    work.calls.clear()
    again = _go(env, Selection(), 'design')  # the same source and settings: not tried again
    assert work.calls == [] and not again.run.failed
    assert 'retry failed' in again.skipped[0].reason and 'ffmpeg exploded' in again.skipped[0].reason

    work.fail.clear()
    retry = _go(env, Selection(), 'design', retry_failed=True)
    assert retry.run.designed == ['fs-a'] and _needs(env, 'fs-a')[0] == 'review'
    assert env.index.failures() == {}


def test_a_failed_title_whose_source_changed_is_retried_without_asking(env, work):
    _scan(env, _item('a'))
    work.fail[('extract', 'fs-a')] = RuntimeError('boom')
    _go(env, Selection(), 'design')

    work.fail.clear()
    _scan(env, _item('a', fingerprint='fp-a-2'))  # re-ripped: the failure no longer applies
    report = _go(env, Selection(), 'design')

    assert report.run.designed == ['fs-a']


def test_retry_failed_reruns_a_design_failure_from_its_own_stage(env, work):
    _scan(env, _item('a'))
    work.fail[('design', 'fs-a')] = RuntimeError('designer said no')
    _go(env, Selection(), 'design')
    assert _row(env, 'fs-a').design_state == 'failed'

    work.fail.clear()
    work.calls.clear()
    _go(env, Selection(ids=('fs-a',)), 'design', retry_failed=True)

    assert ('design', 'fs-a') in work.calls and _needs(env, 'fs-a')[0] == 'review'


def test_one_failing_title_does_not_stop_the_others(env, work):
    _scan(env, _item('a'), _item('b'))
    work.fail[('extract', 'fs-a')] = RuntimeError('boom')

    report = _go(env, Selection(), 'design')

    assert [f[0] for f in report.run.failed] == ['fs-a'] and report.run.designed == ['fs-b']


def test_a_failure_recorded_by_run_stages_uses_the_same_key_as_discovery(env, work):
    item = _item('a')
    _scan(env, item)
    work.fail[('extract', 'fs-a')] = RuntimeError('boom')
    _go(env, Selection(), 'design')

    memory = env.index.failures()['fs-a']

    assert memory.key == failure_key('extract', item, config=CONFIG, designer=DESIGNER,
                                     coverage='complete_programme', keep_multichannel=False)


# --- publish, republish, commit -----------------------------------------------------------------------------------

def _publish_settings(repos, **kwargs):
    xml, _, images, _ = repos
    return PublishSettings(xml, images, image_owner='3ll3d00d', image_repo_name='beq-images', xml_dir='xml',
                           image_dir='img', **kwargs)


def _real_wav(env, item):
    ''' Publishing with a work directory writes the title's .beq projects, which load the mono wav for real. '''
    import numpy as np
    import soundfile as sf
    sf.write(os.path.join(env.work, item.id, 'mono.wav'), np.random.default_rng(1).normal(0, 0.1, 4000), 1000)


def _accepted(env, repos, *names):
    ''' Titles a person accepted but nobody has published: extracted, designed, accepted. '''
    xml, _, images, _ = repos
    items = [_item(n) for n in names]
    for item in items:
        _extracted(env, item)
        _real_wav(env, item)
        _entry(env, item, status='accepted')
    settings = ScanSettings(work_dir=env.work, queue_dir=env.queue, designer=DESIGNER, xml_repo=xml.local_path,
                            xml_dir='xml', images_repo=images.local_path, image_dir='img', image_owner=OWNER,
                            image_repo_name=IMAGES_NAME)   # what publish is given: they are in the digest
    return items, settings


def test_through_publish_writes_accepted_titles_and_not_commits(env, work, repos):
    xml, xml_bare, images, _ = repos
    (a, b), settings = _accepted(env, repos, 'a', 'b')
    _entry(env, _item('c'), status='pending')  # waiting for review: never published
    _scan(env, a, b, _item('c'), settings=settings)
    assert _needs(env, 'fs-a')[0] == 'publish'

    report = _go(env, Selection(needs=('publish',)), 'publish', settings=settings, publish=_publish_settings(repos))

    assert sorted(r['id'] for r in report.published) == ['fs-a', 'fs-b'] and not report.failed
    assert (xml.local_path and os.path.isfile(os.path.join(xml.local_path, 'xml', 'fs-a.json')))
    assert read_entry(env.queue, 'fs-a').status == 'published' and read_entry(env.queue, 'fs-c').status == 'pending'
    assert _commits(xml) == []  # written, not committed
    assert _needs(env, 'fs-a') == ('commit', 'written, not committed')


def test_through_commit_publishes_then_commits_one_commit_per_repo_images_first(env, work, repos):
    xml, xml_bare, images, images_bare = repos
    (a, b), settings = _accepted(env, repos, 'a', 'b')
    _scan(env, a, b, settings=settings)

    events = []
    report = _go(env, Selection(needs=('publish',)), 'commit', settings=settings, publish=_publish_settings(repos),
                 on_event=events.append)

    assert len(report.published) == 2 and report.committed is not None and not report.commit_error
    assert len(_commits(xml)) == 1 and len(_commits(images)) == 1  # one commit per repo for the whole selection
    assert b'Film a' in _on_remote(xml_bare, 'xml/fs-a.json') and b'Film b' in _on_remote(xml_bare, 'xml/fs-b.json')
    assert _needs(env, 'fs-a') == ('done', 'pushed')
    for kind in ('command_started', 'command_finished'):
        commit_events = [event for event in events if event.stage == 'commit' and event.kind == kind]
        assert {'fs-a', 'fs-b'} <= {event.title_id for event in commit_events}


def test_through_commit_commits_only_the_selection(env, work, repos):
    xml, xml_bare, images, _ = repos
    (a, b), settings = _accepted(env, repos, 'a', 'b')
    _scan(env, a, b, settings=settings)
    _go(env, Selection(), 'publish', settings=settings, publish=_publish_settings(repos))

    report = _go(env, Selection(ids=('fs-a',)), 'commit', settings=settings, publish=_publish_settings(repos))

    assert report.committed.xml.paths == ['xml/fs-a.json', 'xml/database.json']
    assert _needs(env, 'fs-a')[0] == 'done' and _needs(env, 'fs-b')[0] == 'commit'


def test_a_title_needing_commit_is_committed_by_a_run_through_commit(env, work, repos):
    (a,), settings = _published(env, repos, 'a')
    _real_wav(env, a)
    _ready(repos)
    _scan(env, a, settings=settings)
    assert _needs(env, 'fs-a')[0] == 'commit'

    report = _go(env, Selection(needs=('commit',)), 'commit', settings=settings, publish=_publish_settings(repos))

    assert report.published == [] and report.committed.xml.paths == ['xml/fs-a.json', 'xml/database.json']
    assert _needs(env, 'fs-a')[0] == 'done'


def test_a_run_through_design_does_not_touch_accepted_or_written_titles(env, work, repos):
    (a,), settings = _accepted(env, repos, 'a')
    _scan(env, a, settings=settings)

    _go(env, Selection(), 'design', settings=settings)

    assert read_entry(env.queue, 'fs-a').status == 'accepted'
    assert not os.path.exists(os.path.join(repos[0].local_path, 'xml', 'fs-a.json'))


def test_publishing_without_repository_settings_is_refused_before_anything_runs(env, work, repos):
    (a,), settings = _accepted(env, repos, 'a')
    _scan(env, a, settings=settings)

    with pytest.raises(ValueError, match='xml-repo'):
        _go(env, Selection(), 'publish', settings=settings)
    assert read_entry(env.queue, 'fs-a').status == 'accepted'


def test_a_metadata_typo_on_a_published_title_is_republished_at_the_same_path_without_a_review(env, work, repos):
    xml, xml_bare, images, _ = repos
    _ready(repos)
    (a,), settings = _published(env, repos, 'a')
    _real_wav(env, a)
    from pipeline.library.commit import commit_catalogue
    commit_catalogue(env.queue, xml, images, xml_dir='xml', image_dir='img')
    _scan(env, a, settings=settings)
    assert _needs(env, 'fs-a')[0] == 'done'
    published = read_entry(env.queue, 'fs-a')
    update_entry(env.queue, 'fs-a', meta={**published.meta, 'title': 'Film a (fixed)'})
    _scan(env, a, settings=settings)
    assert _needs(env, 'fs-a') == ('publish', 'changed since it was published')

    report = _go(env, Selection(needs=('publish',)), 'publish', settings=settings, publish=_publish_settings(repos))

    assert [(r['id'], r['republished']) for r in report.published] == [('fs-a', True)]
    entry = read_entry(env.queue, 'fs-a')
    assert entry.status == 'published' and entry.published_digest != published.published_digest
    assert entry.revision == published.revision + 1  # a republish over a committed, clean XML begins a revision
    with open(os.path.join(xml.local_path, 'xml', 'fs-a.json'), encoding='utf-8') as f:
        assert 'Film a (fixed)' in f.read()
    assert _needs(env, 'fs-a') == ('commit', 'written, not committed')

    committed = _go(env, Selection(needs=('commit',)), 'commit', settings=settings, publish=_publish_settings(repos))

    assert committed.committed.xml.paths == ['xml/fs-a.json', 'xml/database.json'] and len(_commits(xml)) == 3  # README, the first publish, and this revision of the same path
    assert b'Film a (fixed)' in _on_remote(xml_bare, 'xml/fs-a.json')
    assert _needs(env, 'fs-a') == ('done', 'pushed')


def test_a_republish_of_an_unchanged_published_title_does_nothing(env, work, repos):
    _ready(repos)
    (a,), settings = _published(env, repos, 'a')
    _real_wav(env, a)
    _scan(env, a, settings=settings)
    before = read_entry(env.queue, 'fs-a')

    report = _go(env, Selection(ids=('fs-a',)), 'publish', settings=settings, publish=_publish_settings(repos))

    assert report.skipped and report.published == [] and read_entry(env.queue, 'fs-a') == before


def test_an_accepted_title_with_incomplete_metadata_waits_for_a_person_and_the_rest_are_published(env, work, repos):
    (a, b), settings = _accepted(env, repos, 'a', 'b')
    update_entry(env.queue, 'fs-a', meta={'title': '', 'year': '2018', 'audio_types': ['Atmos']})
    _scan(env, a, b, settings=settings)
    assert _needs(env, 'fs-a')[0] == 'review'  # the index sends it to a person; publish is never even asked

    report = _go(env, Selection(ids=('fs-a', 'fs-b')), 'publish', settings=settings,
                 publish=_publish_settings(repos))

    assert [r['id'] for r in report.published] == ['fs-b'] and not report.failed
    assert [s.id for s in report.skipped] == ['fs-a'] and 'metadata incomplete' in report.skipped[0].reason
    assert read_entry(env.queue, 'fs-a').status == 'accepted'


def test_publish_refusal_for_one_title_does_not_discard_another_titles_write(env, work, repos):
    (a, b), settings = _accepted(env, repos, 'a', 'b')
    _scan(env, a, b, settings=settings)
    # The index still says both are ready, but the first entry changed after the scan.
    update_entry(env.queue, 'fs-a', meta={'title': '', 'year': '2018', 'audio_types': ['Atmos']})
    events = []

    report = _go(env, Selection(ids=('fs-a', 'fs-b')), 'publish', settings=settings,
                 publish=_publish_settings(repos), on_event=events.append)

    assert [result['id'] for result in report.published] == ['fs-b']
    assert [result['id'] for result in report.publish_errors] == ['fs-a']
    assert read_entry(env.queue, 'fs-a').status == 'accepted'
    assert read_entry(env.queue, 'fs-b').status == 'published'
    assert ('fs-a', 'failed') in [(event.title_id, event.kind) for event in events]
    assert ('fs-b', 'title_completed') in [(event.title_id, event.kind) for event in events]


def test_cancel_between_published_entries_stops_cleanly_and_leaves_no_commit(env, work, repos):
    xml, _, images, _ = repos
    (a, b, c), settings = _accepted(env, repos, 'a', 'b', 'c')
    _scan(env, a, b, c, settings=settings)
    seen = []

    report = _go(env, Selection(), 'commit', settings=settings, publish=_publish_settings(repos),
                 on_progress=seen.append, should_cancel=lambda: len([p for p in seen if p.stage == 'publish']) >= 1)

    assert report.cancelled and report.committed is None  # a cancelled run never goes on to commit
    assert [r['id'] for r in report.published] == ['fs-a'] and sorted(report.not_run) == ['fs-b', 'fs-c']
    assert [read_entry(env.queue, i).status for i in ('fs-a', 'fs-b', 'fs-c')] == ['published', 'accepted', 'accepted']
    assert _commits(xml) == []
    assert _needs(env, 'fs-a')[0] == 'commit' and _needs(env, 'fs-b')[0] == 'publish'


def test_a_rejected_push_is_reported_and_what_was_committed_stays(env, work, repos):
    xml, xml_bare, images, images_bare = repos
    (a,), settings = _accepted(env, repos, 'a')
    _scan(env, a, settings=settings)
    subprocess.run(['rm', '-rf', str(images_bare)], check=True)  # the remote has gone away

    report = _go(env, Selection(), 'commit', settings=settings, publish=_publish_settings(repos))

    assert report.commit_error.startswith('git failed') and report.failed
    assert read_entry(env.queue, 'fs-a').status == 'published'
    assert len(_commits(images)) == 1  # committed locally; the next run pushes it


def test_run_stages_accepts_settings_built_from_a_scan_settings(repos, env):
    xml, _, images, _ = repos
    settings = ScanSettings(work_dir=env.work, queue_dir=env.queue, xml_repo=xml.local_path, xml_dir='x',
                            images_repo=images.local_path, image_dir='i', meta_defaults={'source': 'Disc'})

    publish = PublishSettings.from_scan_settings(settings, image_owner='o', push=False)

    assert publish == PublishSettings(RepoTarget(xml.local_path), RepoTarget(images.local_path), 'o', None, 'x', 'i',
                                      {'source': 'Disc'}, push=False)
    with pytest.raises(ValueError, match='xml-repo'):
        PublishSettings.from_scan_settings(ScanSettings(work_dir='w', queue_dir='q'))
