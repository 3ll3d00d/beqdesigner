'''
Fixes from the independent review of the discovery index (chunk 24, and 25's use of it): locking, an empty listing, a
per-title ignore of a season, a change of tv_mode, unknown sources, season source changes, rebuilds, search parity,
refresh with another profile, and the small ones. Each test failed before its fix.
'''
import json
import os
import sqlite3
import threading
import time

import pytest

from pipeline.library import cli
from pipeline.library.index import INDEX_FILE_NAME, IndexFileError, LibraryIndex, index_path
from pipeline.library.season import plan_units, track_fingerprint
from pipeline.library.status import ScanSettings, season_source_fingerprint
from pipeline.review import update_entry
from test_pipeline_library_index import (DESIGNER, FakeSource, _entry, _episode, _extracted, _item, _needs, _profile,  # noqa
                                         _row, _scan, env)
from test_pipeline_library_index import IMAGES_NAME, OWNER, _published, _ready, repos  # noqa: F401 (a fixture)


# --- 7: a slow source does not block a reader ---------------------------------------------------------------------------

class _BlockingSource:
    def __init__(self, items):
        self.items, self.entered, self.release = list(items), threading.Event(), threading.Event()

    def list_items(self, **query):
        self.entered.set()
        assert self.release.wait(30)
        return list(self.items)


def _within(seconds, call):
    ''' True if `call()` returns within `seconds`, on another thread (so a blocked call fails the test, not hangs it). '''
    done = threading.Event()
    threading.Thread(target=lambda: (call(), done.set()), daemon=True).start()
    return done.wait(seconds)


def test_readers_are_not_blocked_while_a_source_is_being_listed(env):
    _scan(env, _item('a'))                                   # the last scan: what a reader must go on seeing
    slow = _BlockingSource([_item('a'), _item('b')])
    scanning = threading.Thread(target=lambda: env.index.scan(_profile(env), env.settings, sources={'films': slow}))
    scanning.start()
    try:
        assert slow.entered.wait(10)
        assert _within(2, env.index.titles), 'titles() waited for the scan'
        assert _within(2, env.index.summary), 'summary() waited for the scan'
        assert _within(2, env.index.sources), 'sources() waited for the scan'
        assert _within(2, lambda: env.index.record_failure('fs-a', 'design', 'boom', 'fp', 'k')), \
            'record_failure() waited for the scan'
        assert [r.id for r in env.index.titles()] == ['fs-a']  # the previous scan, until this one finishes
    finally:
        slow.release.set()
        scanning.join(30)
    assert [r.id for r in env.index.titles()] == ['fs-a', 'fs-b']


# --- 8: an empty listing is not the truth -------------------------------------------------------------------------------

def test_a_source_that_lists_nothing_after_listing_something_keeps_its_previous_listing(env):
    _scan(env, _item('a'), _item('b'))

    result = _scan(env)   # an unmounted share: glob matches nothing and raises nothing

    assert 'listed 0 items (previously 2)' in result.errors['films']
    assert result.gone == [] and result.dropped == []
    assert sorted(r.id for r in env.index.titles()) == ['fs-a', 'fs-b'] and _row(env, 'fs-a').flags == []
    (source,) = env.index.sources()
    assert 'listed 0 items' in source.last_error and source.item_count == 2
    assert _scan(env, _item('a'), _item('b')).errors == {}          # the share is back: fine again


def test_an_empty_listing_is_accepted_when_the_caller_says_so(env):
    _scan(env, _item('a'))

    result = _scan(env, allow_empty=True)

    assert result.errors == {} and result.dropped == ['fs-a'] and env.index.titles() == []


def test_a_source_that_never_listed_anything_may_list_nothing(env):
    assert _scan(env).errors == {}


def test_the_cli_scan_reports_an_empty_listing_and_has_a_flag_for_a_really_empty_library(tmp_path, capsys):
    import yaml
    media = tmp_path / 'films'
    media.mkdir()
    (media / 'a.mkv').write_bytes(b'x')
    config = tmp_path / 'p.yaml'
    config.write_text(yaml.safe_dump({'sources': [{'name': 'disk', 'kind': 'filesystem', 'globs': [str(media)]}],
                                      'run': {'work_dir': str(tmp_path / 'work'), 'queue_dir': str(tmp_path / 'q')}}))
    assert cli.main(['scan', '--profile', str(config)]) == 0
    capsys.readouterr()
    (media / 'a.mkv').unlink()

    assert cli.main(['scan', '--profile', str(config)]) == 1
    assert 'listed 0 items' in json.loads(capsys.readouterr().out)['errors']['disk']
    assert cli.main(['scan', '--profile', str(config), '--allow-empty']) == 0
    assert json.loads(capsys.readouterr().out)['dropped']


# --- 9: a per-title ignore of a season --------------------------------------------------------------------------------

def test_a_per_title_ignore_of_a_season_id_ignores_the_season_row(env):
    episodes = [_episode(1), _episode(2)]
    (group,) = plan_units(episodes, 'season')
    settings = ScanSettings(work_dir=env.work, queue_dir=env.queue, designer=DESIGNER, tv_mode='season')

    _scan(env, *episodes, settings=settings, profile=_profile(env, ignored_titles={group.item.id: 'not this one'}))

    row = _row(env, group.item.id)
    assert (row.needs, row.tier, row.flags, row.detail) == ('done', 'done', ['Ignored'], 'ignored by you: not this one')
    assert len(env.index.titles()) == 1


# --- 10: a change of tv_mode does not turn pending reviews into "gone" -------------------------------------------------

def _season_settings(env, mode):
    return ScanSettings(work_dir=env.work, queue_dir=env.queue, designer=DESIGNER, tv_mode=mode)


def test_switching_to_season_mode_keeps_episode_rows_with_a_review_and_says_what_replaced_them(env):
    episodes = [_episode(1), _episode(2)]
    (group,) = plan_units(episodes, 'season')
    for episode in episodes:
        _extracted(env, episode)
        _entry(env, episode)                                  # both awaiting review
    _scan(env, *episodes, settings=_season_settings(env, 'episode'))
    assert {_needs(env, e.id)[0] for e in episodes} == {'review'}

    result = _scan(env, *episodes, settings=_season_settings(env, 'season'))

    for episode in episodes:
        row = _row(env, episode.id)
        assert (row.needs, row.tier, row.flags) == ('review', 'human', [])       # not gone, not hidden
        assert row.detail.endswith(f'superseded by {group.item.id}')
    assert sorted(result.superseded) == sorted(e.id for e in episodes) and result.gone == []
    assert _row(env, group.item.id).unit == 'season'


def test_switching_to_season_mode_drops_episode_rows_that_have_nothing_to_review(env):
    episodes = [_episode(1), _episode(2)]
    (group,) = plan_units(episodes, 'season')
    _scan(env, *episodes, settings=_season_settings(env, 'episode'))

    result = _scan(env, *episodes, settings=_season_settings(env, 'season'))

    assert sorted(result.dropped) == sorted(e.id for e in episodes) and result.gone == []
    assert [r.id for r in env.index.titles()] == [group.item.id]


def test_switching_back_to_episode_mode_keeps_a_season_row_awaiting_review(env):
    episodes = [_episode(1), _episode(2)]
    (group,) = plan_units(episodes, 'season')
    for episode in episodes:
        _extracted(env, episode)
    wavs = [(e.episodes[0], os.path.join(env.work, e.id, 'mono.wav')) for e in episodes]
    _entry(env, group.item, fingerprint=track_fingerprint(wavs))
    _scan(env, *episodes, settings=_season_settings(env, 'season'))
    assert _needs(env, group.item.id)[0] == 'review'

    result = _scan(env, *episodes, settings=_season_settings(env, 'episode'))

    row = _row(env, group.item.id)
    assert (row.needs, row.tier, row.flags) == ('review', 'human', [])
    assert 'superseded by fs-show-e1' in row.detail and result.superseded == [group.item.id] and result.gone == []
    assert {r.unit for r in env.index.titles()} == {'item', 'season'}


# --- 11: an unknown source fingerprint is not compared, for design as for extract ------------------------------------------

def test_a_title_whose_media_cannot_be_read_keeps_its_design_current(env):
    item = _item('a', fingerprint='', source_path='/offline-nas/a.mkv')       # no fingerprint and cannot be stat'd
    _extracted(env, item, fingerprint='fp-recorded')
    _entry(env, item, fingerprint='fp-recorded')

    _scan(env, item)

    assert (_row(env, 'fs-a').extract_state, _row(env, 'fs-a').design_state) == ('current', 'current')
    assert _needs(env, 'fs-a')[0] == 'review'


def test_an_unknown_source_still_notices_a_changed_designer(env):
    item = _item('a', fingerprint='', source_path='/offline-nas/a.mkv')
    _extracted(env, item, fingerprint='fp-recorded')
    _entry(env, item, fingerprint='fp-recorded')

    _scan(env, item, settings=ScanSettings(work_dir=env.work, queue_dir=env.queue, designer='another'))

    assert _row(env, 'fs-a').design_state == 'stale'


# --- 12: a re-ripped episode of an accepted season is seen at scan time ---------------------------------------------------

def _accepted_season(env, status='accepted'):
    episodes = [_episode(1), _episode(2)]
    (group,) = plan_units(episodes, 'season')
    for episode in episodes:
        _extracted(env, episode)
    wavs = [(e.episodes[0], os.path.join(env.work, e.id, 'mono.wav')) for e in episodes]
    _entry(env, group.item, status=status, fingerprint=track_fingerprint(wavs))
    return episodes, group, wavs


def test_a_reripped_episode_of_an_accepted_season_is_source_changed_without_re_extracting(env):
    episodes, group, _ = _accepted_season(env)
    update_entry(env.queue, group.item.id, source_fingerprint=season_source_fingerprint(group))  # as design records it
    settings = _season_settings(env, 'season')
    _scan(env, *episodes, settings=settings)
    assert _needs(env, group.item.id) == ('publish', 'accepted, not written to the repository')

    _scan(env, _episode(1), _episode(2, fingerprint='fp-reripped'), settings=settings)

    assert _needs(env, group.item.id) == ('attention', 'source changed since accepted')


def test_a_season_designed_before_seasons_had_their_own_fingerprint_raises_no_false_alarm(env):
    episodes, group, wavs = _accepted_season(env)       # its source_fingerprint is the joined track's, as it was
    _scan(env, *episodes, settings=_season_settings(env, 'season'))
    assert _needs(env, group.item.id)[0] == 'publish'


def test_design_records_the_seasons_own_fingerprint(env, monkeypatch):
    from pipeline.library import run as run_module
    episodes = [_episode(1), _episode(2)]
    (group,) = plan_units(episodes, 'season')
    seen = {}
    monkeypatch.setattr(run_module, '_extract_season', lambda *a, **k: ('/t.wav', 'trackfp', group.item, '/g'))
    monkeypatch.setattr(run_module, '_design', lambda *a, **k: seen.update(k))
    run_module._run_season(None, group, None, None)
    assert seen['recorded_source'] == season_source_fingerprint(group) != ''


# --- 13: a rebuild of a live index -------------------------------------------------------------------------------------

def test_a_rebuild_of_a_live_index_resets_the_generation_and_the_recorded_sources(env):
    item = _item('a')
    _extracted(env, item)
    _entry(env, item)
    _scan(env, item)
    assert env.index.generation == 1 and env.index.sources()

    env.index.rebuild_from_outputs(env.settings)

    assert env.index.generation == 0                 # so the next selector run scans first
    assert env.index.sources() == [] and env.index.summary().last_scan_at is None


def test_scan_from_outputs_on_a_live_index_leaves_status_saying_scan_first(tmp_path, capsys):
    import yaml
    media = tmp_path / 'films'
    media.mkdir()
    (media / 'a.mkv').write_bytes(b'x')
    config = tmp_path / 'p.yaml'
    config.write_text(yaml.safe_dump({'sources': [{'name': 'disk', 'kind': 'filesystem', 'globs': [str(media)]}],
                                      'run': {'work_dir': str(tmp_path / 'work'), 'queue_dir': str(tmp_path / 'q')}}))
    assert cli.main(['scan', '--profile', str(config)]) == 0
    assert cli.main(['scan', '--profile', str(config), '--from-outputs']) == 0
    capsys.readouterr()

    assert cli.main(['status', '--profile', str(config)]) == 1
    assert 'never been scanned' in capsys.readouterr().out


# --- 14: search parity with the work list -------------------------------------------------------------------------------

def test_match_folds_case_beyond_ascii_as_the_work_list_does(env):
    _scan(env, _item('a', title='Élite', display_name='Élite (2018)'), _item('b', title='Other'))

    assert [r.id for r in env.index.titles(match='élite')] == ['fs-a']
    assert [r.id for r in env.index.titles(match='ÉLITE')] == ['fs-a']
    assert [r.id for r in env.index.titles(match='Straße'.upper())] == []
    _scan(env, _item('c', title='Straße'))
    assert [r.id for r in env.index.titles(match='STRASSE')] == ['fs-c']      # casefold: ss == ß


def test_match_treats_percent_and_underscore_literally(env):
    _scan(env, _item('a', title='100% Wolf'), _item('b', title='100 Wolf'))
    assert [r.id for r in env.index.titles(match='100%')] == ['fs-a']


# --- 15: refresh is not a scan of another profile ---------------------------------------------------------------------

def test_refresh_with_a_profile_whose_source_names_differ_changes_nothing(env):
    item = _item('a')
    _extracted(env, item)
    _entry(env, item)
    _scan(env, item)
    before = [(r.id, r.needs, r.flags, r.source) for r in env.index.titles()]

    result = env.index.refresh(_profile(env, 'filesystem'), env.settings)     # an ad-hoc profile named by its kind

    assert [(r.id, r.needs, r.flags, r.source) for r in env.index.titles()] == before
    assert [s.name for s in env.index.sources()] == ['films'] and result.gone == [] and result.dropped == []


def test_refresh_of_an_index_that_recorded_no_sources_is_left_alone(env):
    item = _item('a')
    _extracted(env, item)
    _entry(env, item)
    env.index.rebuild_from_outputs(env.settings)

    env.index.refresh(_profile(env, 'films'), env.settings)

    assert [r.id for r in env.index.titles()] == ['fs-a'] and _row(env, 'fs-a').flags == []
    assert env.index.generation == 0 and env.index.sources() == []


def test_a_legacy_selector_run_does_not_wipe_the_index_a_profile_scan_wrote(tmp_path, capsys):
    import yaml
    from pipeline.designer.registry import register_designer, unregister_designer
    media = tmp_path / 'films'
    media.mkdir()
    (media / 'a.mkv').write_bytes(b'x')
    common = {'work_dir': str(tmp_path / 'work'), 'queue_dir': str(tmp_path / 'q'), 'designer': 'rolloff'}
    profile = tmp_path / 'profile.yaml'
    profile.write_text(yaml.safe_dump({'sources': [{'name': 'disk', 'kind': 'filesystem', 'globs': [str(media)]}],
                                       'run': common}))
    legacy = tmp_path / 'legacy.yaml'
    legacy.write_text(yaml.safe_dump({'sources': {'filesystem': {'globs': [str(media)]}},
                                      'run': {**common, 'source': 'filesystem'}}))
    assert cli.main(['scan', '--profile', str(profile)]) == 0
    register_designer('rolloff', lambda request: None)
    try:
        cli.main(['--config', str(legacy), 'run', '--source', 'filesystem', '--needs', 'review'])
    finally:
        unregister_designer('rolloff')
    capsys.readouterr()

    with LibraryIndex(index_path(str(tmp_path / 'work'))) as index:
        assert [r.flags for r in index.titles()] == [[]] and len(index.titles()) == 1
        assert [s.name for s in index.sources()] == ['disk']


# --- 16: the small ones ----------------------------------------------------------------------------------------------

def test_a_corrupt_entry_summary_is_a_cache_miss_not_a_failed_scan(env):
    item = _item('a')
    _extracted(env, item)
    _entry(env, item)
    _scan(env, item)
    with sqlite3.connect(index_path(env.work)) as raw:
        raw.execute("UPDATE titles SET entry_summary = '{not json'")
    env.index.close()
    env.index = LibraryIndex(index_path(env.work))

    _scan(env, item)

    assert _needs(env, 'fs-a')[0] == 'review'
    with sqlite3.connect(index_path(env.work)) as raw:
        raw.execute("UPDATE titles SET entry_summary = '{\"unexpected\": 1}'")
    _scan(env, item)
    assert _needs(env, 'fs-a')[0] == 'review'


def test_an_entry_rewritten_with_the_same_size_and_mtime_is_still_noticed(env):
    ''' skipped -> pending is the same length; a filesystem with a coarse mtime would show one signature for both. '''
    item = _item('a')
    _extracted(env, item)
    _entry(env, item, status='skipped')
    _scan(env, item)
    path = os.path.join(env.queue, 'fs-a.json')
    before = os.stat(path)

    update_entry(env.queue, 'fs-a', status='pending')
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))     # as if within one mtime tick
    assert os.stat(path).st_size == before.st_size and os.stat(path).st_mtime_ns == before.st_mtime_ns
    _scan(env, item)

    assert _row(env, 'fs-a').review_state == 'pending'


def test_a_catalogue_record_that_will_not_parse_is_read_once_not_every_scan(tmp_path, monkeypatch):
    from pipeline.library import catalogue_scan
    repo = tmp_path / 'xml'
    repo.mkdir()
    (repo / 'bad.json').write_text('{not-valid-json')
    first = catalogue_scan.scan_xml_repo(str(repo))
    calls = []
    real = catalogue_scan.parse_record
    monkeypatch.setattr(catalogue_scan, 'parse_record', lambda path: calls.append(path) or real(path))

    again = catalogue_scan.scan_xml_repo(str(repo), first)

    assert set(first) == {'bad.json'}
    assert calls == [] and again == first and catalogue_scan.tmdb_index(again) == {}


def test_a_published_title_with_no_xml_repo_configured_does_not_need_commit_for_ever(env, repos):
    _ready(repos)
    (item,), settings = _published(env, repos, 'a')
    from dataclasses import replace
    no_repo = replace(settings, xml_repo='')      # everything else as published, so the digest still matches

    _scan(env, item, settings=no_repo)

    assert _row(env, 'fs-a').publish_state == 'written'
    assert _needs(env, 'fs-a')[0] != 'commit' and _row(env, 'fs-a').commit_state == 'none'


def test_a_foreign_sqlite_file_is_refused_not_dropped(tmp_path):
    path = str(tmp_path / INDEX_FILE_NAME)
    with sqlite3.connect(path) as raw:
        raw.execute('CREATE TABLE precious (x)')
        raw.execute('INSERT INTO precious VALUES (1)')
        raw.execute('PRAGMA user_version = 7')

    with pytest.raises(IndexFileError, match='not a library index'):
        LibraryIndex(path)

    with sqlite3.connect(path) as raw:
        assert raw.execute('SELECT x FROM precious').fetchall() == [(1,)]


def test_an_index_of_another_version_drops_only_its_own_tables(tmp_path):
    path = str(tmp_path / INDEX_FILE_NAME)
    LibraryIndex(path).close()
    with sqlite3.connect(path) as raw:
        raw.execute('CREATE TABLE mine (x)')
        raw.execute('PRAGMA user_version = 99')

    with LibraryIndex(path) as index:
        assert index.generation == 0
    with sqlite3.connect(path) as raw:
        assert raw.execute('SELECT name FROM sqlite_master WHERE name = "mine"').fetchall() == [('mine',)]


def test_status_opens_read_only_and_never_drops_an_old_version(tmp_path, capsys):
    import yaml
    work = tmp_path / 'work'
    with LibraryIndex(index_path(str(work))) as index:
        pass
    with sqlite3.connect(index_path(str(work))) as raw:
        raw.execute('PRAGMA user_version = 99')
    config = tmp_path / 'p.yaml'
    config.write_text(yaml.safe_dump({'run': {'work_dir': str(work)}}))

    assert cli.main(['status', '--profile', str(config)]) == 1

    assert 'run `scan`' in capsys.readouterr().out
    with sqlite3.connect(index_path(str(work))) as raw:     # untouched: still version 99, tables still there
        assert raw.execute('PRAGMA user_version').fetchone()[0] == 99
        assert raw.execute('SELECT COUNT(*) FROM titles').fetchone()[0] == 0


def test_a_readonly_index_reads_and_refuses_to_write(tmp_path):
    path = index_path(str(tmp_path / 'work'))
    LibraryIndex(path).close()
    with LibraryIndex(path, readonly=True) as index:
        assert index.summary().titles == 0
        with pytest.raises(sqlite3.OperationalError):
            index.clear_failure()


def test_titles_by_id_are_looked_up_in_batches_and_come_back_in_work_list_order(env, monkeypatch):
    from pipeline.library import index as index_module
    monkeypatch.setattr(index_module, '_ID_BATCH', 100)
    items = [_item(f'n{n:04d}') for n in range(250)]
    _scan(env, *items)

    found = env.index.titles(ids=[i.id for i in reversed(items)])

    assert len(found) == 250 and [r.id for r in found] == [r.id for r in env.index.titles()]   # one order, as unbatched
    assert env.index.titles(ids=[]) == [] and len(env.index.titles(ids=['fs-n0007', 'fs-n0007'])) == 1


def test_last_seen_moves_only_for_sources_that_were_actually_listed(env):
    a, b = _item('a', source_path='/one/a.mkv'), _item('b', source_path='/two/b.mkv')
    sources = {'one': FakeSource([a]), 'two': FakeSource([b])}
    _scan(env, sources=sources, now=100.0)

    sources['two'].error = OSError('down')
    _scan(env, sources=sources, now=200.0)
    assert (_row(env, 'fs-a').last_seen, _row(env, 'fs-b').last_seen) == (200.0, 100.0)

    env.index.refresh(_profile(env, 'one', 'two'), env.settings, now=300.0)
    assert (_row(env, 'fs-a').last_seen, _row(env, 'fs-b').last_seen) == (200.0, 100.0)


def test_a_second_copy_of_an_episode_is_named_on_the_season_row_not_silently_lost(env):
    first = _episode(1)
    copy = _item('show-e1-copy', title='Some Show', kind='tv', season='1', episodes=(1,), source_path='/tv/copy/e1.mkv')
    (group,) = plan_units([first, copy], 'season')

    _scan(env, first, copy, settings=_season_settings(env, 'season'))

    row = _row(env, group.item.id)
    assert 'left out of the season' in row.detail and 'fs-show-e1-copy' in row.detail
    assert 'fs-show-e1-copy' not in row.duplicates


def test_a_claimed_copy_shadowed_by_another_says_it_has_outputs_of_its_own(env):
    a, b = _item('a', source_path='/m/heat.mkv'), _item('b', source_path='/m/heat.mkv')
    _extracted(env, a)
    _extracted(env, b)

    _scan(env, a, b)

    assert _row(env, 'fs-b').flags == ['Shadowed'] and 'own review or outputs' in _row(env, 'fs-b').detail
    assert _row(env, 'fs-a').duplicates == ('fs-b',)


# --- 18: the published digest includes what publish is given -------------------------------------------------------------

def test_the_owner_repo_and_report_spec_are_in_the_scan_settings_and_mark_a_published_title_out_of_date(env, repos):
    from pipeline.publish.report import ReportSpec
    _ready(repos)
    (item,), settings = _published(env, repos, 'a')
    _scan(env, item, settings=settings)
    assert _row(env, 'fs-a').publish_state == 'written'

    from dataclasses import replace
    for changed in (replace(settings, image_owner='someone-else'), replace(settings, image_repo_name='other-repo'),
                    replace(settings, report_spec=ReportSpec(width_px=2000))):
        _scan(env, item, settings=changed)
        assert _row(env, 'fs-a').publish_state == 'out_of_date', changed
    _scan(env, item, settings=settings)
    assert _row(env, 'fs-a').publish_state == 'written'


def test_unset_owner_repo_and_report_spec_keep_a_digest_recorded_without_them_valid(env, repos):
    from dataclasses import replace
    from pipeline.publish.report import ReportSpec
    from pipeline.review import current_publish_digest, read_entry
    _ready(repos)
    (item,), settings = _published(env, repos, 'a')
    # a digest recorded before owner/repo were counted (or with publish working them out from the remote)
    entry = read_entry(env.queue, 'fs-a')
    update_entry(env.queue, 'fs-a', published_digest=current_publish_digest(
        entry, work_dir=env.work, has_image=True))
    unset = replace(settings, image_owner='', image_repo_name='', report_spec=None)

    _scan(env, item, settings=unset)
    assert _row(env, 'fs-a').publish_state == 'written'
    _scan(env, item, settings=replace(unset, report_spec=ReportSpec()))         # the default spelled out: the same
    assert _row(env, 'fs-a').publish_state == 'written'
    _scan(env, item, settings=settings)                                         # now they are given: a change
    assert _row(env, 'fs-a').publish_state == 'out_of_date'


def test_scan_settings_read_owner_repo_and_report_spec_from_the_profile_file():
    from pipeline.library.profile import profile_from_config
    profile = profile_from_config({'sync': {'image_owner': 'me', 'image_repo_name': 'imgs',
                                            'report_spec': {'width_px': 1200}}})

    settings = ScanSettings.from_profile(profile)

    assert (settings.image_owner, settings.image_repo_name, settings.report_spec.width_px) == ('me', 'imgs', 1200)
    with pytest.raises(ValueError, match='report_spec is not valid'):
        ScanSettings.from_values({'report_spec': {'nonsense': 1}})
