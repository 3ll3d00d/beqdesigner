'''
pipeline/library/drift.py (chunk 27c): which accepted or published titles were designed under other settings than the ones
in force now -- the count behind the work list's "settings changed since N titles were designed" banner. Real scans over
real outputs (`test_pipeline_library_index.py`'s helpers); nothing is faked.
'''
import dataclasses
import json
import os

from pipeline.config import AnalysisConfig
from pipeline.library.drift import designed_under_other_settings
from pipeline.review import update_entry
from test_pipeline_library_index import CONFIG, DESIGNER, _entry, _extracted, _item, _profile, _scan, env  # noqa: F401 (a fixture)


def _accepted(env, *names, **fields):
    items = [_item(n) for n in names]
    for item in items:
        _extracted(env, item)
        _entry(env, item, status='accepted', **fields)
    _scan(env, *items)
    return items


def _drift(env, **changes):
    settings = dataclasses.replace(env.settings, **changes)
    return designed_under_other_settings(env.index, settings)


def test_titles_designed_under_the_current_settings_are_not_counted(env):
    _accepted(env, 'a', 'b')
    assert _drift(env) == []


def test_a_changed_designer_counts_every_accepted_title(env):
    _accepted(env, 'a', 'b')
    assert _drift(env, designer='another.designer') == ['fs-a', 'fs-b']


def test_a_changed_analysis_setting_or_coverage_counts(env):
    _accepted(env, 'a')
    assert _drift(env, config=AnalysisConfig(target_fs=CONFIG.target_fs * 2)) == ['fs-a']
    assert _drift(env, coverage='mostly_complete') == ['fs-a']


def test_published_titles_count_too_and_pending_or_skipped_ones_do_not(env):
    a, b, c = (_item(n) for n in 'abc')
    for item, status in ((a, 'published'), (b, 'pending'), (c, 'skipped')):
        _extracted(env, item)
        _entry(env, item, status=status)
    _scan(env, a, b, c)
    assert _drift(env, designer='another.designer') == ['fs-a']     # the others are redesigned by the next run anyway


def test_a_changed_source_is_not_the_settings_and_is_not_counted(env):
    ''' The title is `attention` ("source changed since accepted"); that is a different banner-less state. '''
    item = _item('a')
    _extracted(env, item)
    _entry(env, item, status='accepted')
    reripped = _item('a', fingerprint='fp-a-2')
    _scan(env, reripped)
    row = env.index.title('fs-a')
    assert row.needs == 'attention' and 'source changed' in row.detail
    assert _drift(env) == []
    assert _drift(env, designer='another.designer') == ['fs-a']     # ... but a settings change still is


def test_an_entry_with_no_design_or_source_fingerprint_is_never_counted(env):
    ''' Designed before they were recorded: the index treats these as current, so there are no false alarms. '''
    items = [_item(n) for n in 'ab']
    for item in items:
        _extracted(env, item)
    _entry(env, items[0], status='accepted')
    update_entry(env.queue, items[0].id, design_fingerprint=None)
    _entry(env, items[1], status='accepted')
    update_entry(env.queue, items[1].id, source_fingerprint=None)
    _scan(env, *items)
    assert _drift(env, designer='another.designer') == []


def test_a_multichannel_design_is_compared_as_a_multichannel_design(env):
    item = _item('a')
    _extracted(env, item, channels=6)
    manifest_path = os.path.join(env.work, item.id, 'manifest.json')
    with open(manifest_path) as f:
        manifest = json.load(f)
    manifest['multichannel_source_fingerprint'] = item.fingerprint
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f)
    from pipeline.library.design_cache import design_fingerprint
    _entry(env, item, status='accepted')
    update_entry(env.queue, item.id, design_fingerprint=design_fingerprint(
        item, DESIGNER, CONFIG, 'complete_programme', multichannel=True, source=item.fingerprint))
    _scan(env, item)
    keep = dataclasses.replace(env.settings, keep_multichannel=True)
    assert designed_under_other_settings(env.index, keep) == []
    # keeping the multichannel track was switched off: a mono design would now be made, so this one is out of step
    assert designed_under_other_settings(env.index, env.settings) == ['fs-a']


def test_a_season_is_not_counted(env):
    ''' Its design came from the joined track, whose fingerprint is not kept with the entry: it is not claimed either way. '''
    from pipeline.library.season import plan_units
    from test_pipeline_library_index import _episode
    episodes = [_episode(1), _episode(2)]
    (group,) = plan_units(episodes, 'season')
    settings = dataclasses.replace(env.settings, tv_mode='season')
    for episode in episodes:
        _extracted(env, episode)
    _entry(env, group.item, status='accepted')      # a season's own entry, protected, with fingerprints that cannot be redone here
    update_entry(env.queue, group.item.id, design_fingerprint='0' * 64, source_fingerprint='season:abc')
    _scan(env, *episodes, settings=settings)
    row = env.index.title(group.item.id)
    assert (row.unit, row.review_state, row.design_state) == ('season', 'accepted', 'protected')
    assert designed_under_other_settings(env.index, dataclasses.replace(settings, designer='another.designer')) == []


def test_ignored_shadowed_and_gone_titles_are_not_counted(env):
    items = _accepted(env, 'a', 'b')
    _scan(env, items[0])           # b has left its source: `gone`, with outputs so it is kept
    assert env.index.title('fs-b').gone
    assert _drift(env, designer='another.designer') == ['fs-a']


def test_nothing_is_read_without_a_queue_directory(env):
    _accepted(env, 'a')
    assert designed_under_other_settings(env.index, dataclasses.replace(env.settings, queue_dir='')) == []


def test_rows_may_be_given_and_are_the_ones_examined(env):
    _accepted(env, 'a', 'b')
    only_a = [r for r in env.index.titles() if r.id == 'fs-a']
    assert designed_under_other_settings(env.index, dataclasses.replace(env.settings, designer='x'), only_a) == ['fs-a']



# --- the independent review of chunk 27c (2026-09-21) ---------------------------------------------------------------------------

def _reference_drift(index, settings, rows=None):
    ''' The check as it was written before it was made cheaper: every entry read in full, every manifest read. '''
    from pipeline.library.design_cache import PROTECTED_STATUSES, design_fingerprint
    from pipeline.library.extract_cache import read_manifest
    from pipeline.library.status import read_entry_facts
    rows = list(index.titles() if rows is None else rows)
    candidates = [row for row in rows if row.review_state == 'accepted' and row.design_state == 'protected'
                  and row.unit == 'item' and not (row.ignored or row.shadowed_by or row.gone)]
    units = index.units([row.id for row in candidates])
    found = []
    for row in candidates:
        item, facts = units.get(row.id), read_entry_facts(settings.queue_dir, row.id)
        if item is None or facts is None or facts.status not in PROTECTED_STATUSES or not facts.design_fingerprint \
                or not facts.source_fingerprint:
            continue
        item_dir = os.path.join(settings.work_dir, row.id)
        manifest = read_manifest(item_dir) if os.path.isdir(item_dir) else {}
        count = manifest.get('source_channel_count')
        kept = bool(settings.keep_multichannel and 'multichannel_source_fingerprint' in manifest and count != 1)
        variants = [kept] + ([not kept] if count is None and kept else [])
        now = {design_fingerprint(item, settings.designer, settings.config, settings.coverage, multichannel=mc,
                                  source=facts.source_fingerprint) for mc in variants}
        if facts.design_fingerprint not in now:
            found.append(row.id)
    return found


def _mixed_library(env):
    ''' Mono and multichannel extractions, some with the multichannel manifest marker, one with no channel count. '''
    from pipeline.library.design_cache import design_fingerprint
    items = [_item(n) for n in 'abcdef']
    for item, channels in zip(items, (2, 6, 6, 1, None, 6)):
        _extracted(env, item, channels=channels)
    for item in items[1:3] + items[4:]:
        path = os.path.join(env.work, item.id, 'manifest.json')
        with open(path) as f:
            manifest = json.load(f)
        if item.id != 'fs-e':
            manifest['multichannel_source_fingerprint'] = item.fingerprint
        else:
            manifest.pop('source_channel_count', None)
            manifest['multichannel_source_fingerprint'] = item.fingerprint
        with open(path, 'w') as f:
            json.dump(manifest, f)
    for item in items:
        _entry(env, item, status='accepted')
    for item, multichannel in zip(items, (False, True, False, False, True, True)):    # some designed multichannel, some mono
        update_entry(env.queue, item.id, design_fingerprint=design_fingerprint(
            item, DESIGNER, CONFIG, 'complete_programme', multichannel=multichannel, source=item.fingerprint))
    _scan(env, *items)
    return items


def test_the_cheaper_check_gives_exactly_the_old_answer_for_every_setting(env):
    _mixed_library(env)
    for changes in ({}, {'keep_multichannel': True}, {'keep_multichannel': False}, {'designer': 'x'},
                    {'keep_multichannel': True, 'designer': 'x'}, {'coverage': 'mostly_complete'},
                    {'config': AnalysisConfig(target_fs=CONFIG.target_fs * 2), 'keep_multichannel': True}):
        settings = dataclasses.replace(env.settings, **changes)
        assert designed_under_other_settings(env.index, settings) == _reference_drift(env.index, settings), changes
    # (the library really does differ between the settings, so the comparison is not of two empty lists)
    assert designed_under_other_settings(env.index, env.settings) != \
        designed_under_other_settings(env.index, dataclasses.replace(env.settings, keep_multichannel=True))


def test_an_unchanged_entry_is_not_parsed_again_and_the_manifest_is_not_read_when_multichannel_is_not_kept(env, monkeypatch):
    import pipeline.library.drift as drift_module
    import pipeline.library.status as status_module
    _accepted(env, 'a', 'b', 'c')
    reads, manifests = [], []
    real_entry, real_manifest = status_module.read_entry, drift_module.read_manifest
    monkeypatch.setattr(status_module, 'read_entry', lambda *a: reads.append(a) or real_entry(*a))
    monkeypatch.setattr(drift_module, 'read_manifest', lambda *a: manifests.append(a) or real_manifest(*a))

    assert designed_under_other_settings(env.index, env.settings) == []
    assert reads == [] and manifests == []              # the index's summaries and file signatures were enough

    update_entry(env.queue, 'fs-b', reviewer_note='touched')                 # the file changed: this one is read in full
    assert designed_under_other_settings(env.index, env.settings) == []
    assert [args[1] for args in reads] == ['fs-b']

    manifests.clear()
    assert designed_under_other_settings(env.index, dataclasses.replace(env.settings, keep_multichannel=True)) == []
    assert len(manifests) == 3                                                # kept: it matters, so it is read


def test_two_thousand_protected_entries_are_checked_well_inside_a_budget(env):
    import time
    items = [_item(f'{n:04d}') for n in range(2000)]
    for item in items:
        _extracted(env, item)
        _entry(env, item, status='accepted')
    _scan(env, *items)

    started = time.perf_counter()
    unchanged = designed_under_other_settings(env.index, env.settings)
    took = time.perf_counter() - started
    started = time.perf_counter()
    changed = designed_under_other_settings(env.index, dataclasses.replace(env.settings, designer='another.designer'))
    took_changed = time.perf_counter() - started

    assert unchanged == [] and len(changed) == 2000
    assert took < 8 and took_changed < 8, (took, took_changed)


def test_a_title_designed_by_the_real_design_path_agrees_with_the_check_until_the_settings_change(env, monkeypatch):
    ''' Not a fingerprint built by the check's own function: the entry is what `design_if_needed()` records. '''
    from pipeline.library.design_cache import design_if_needed
    from pipeline.review import CandidateSummary, QueueEntry, write_queue_entry

    def fake_design(session, entry_id, wav_path, designer, queue_dir, **kwargs):
        entry = QueueEntry(id=entry_id, fs=1000, meta=kwargs.get('meta') or {}, curve={},
                           candidates=[CandidateSummary(filters={}, confidence=0.9, method='fitted', mv_adjust_db=0.0)])
        write_queue_entry(queue_dir, entry)
        return entry

    monkeypatch.setattr('pipeline.library.design_cache.design_and_queue', fake_design)
    mono, multichannel = _item('m'), _item('c')
    for item in (mono, multichannel):
        _extracted(env, item, channels=6)
    path = os.path.join(env.work, multichannel.id, 'manifest.json')
    with open(path) as f:
        manifest = json.load(f)
    manifest['multichannel_source_fingerprint'] = multichannel.fingerprint
    with open(path, 'w') as f:
        json.dump(manifest, f)

    designed = [design_if_needed(None, mono, '/w/mono.wav', DESIGNER, env.queue, CONFIG),
                design_if_needed(None, multichannel, '/w/mono.wav', DESIGNER, env.queue, CONFIG,
                                 multichannel_wav_path='/w/multichannel.wav')]
    assert all(r.designed for r in designed)
    for item in (mono, multichannel):
        update_entry(env.queue, item.id, status='accepted', chosen_candidate_index=0)
    _scan(env, mono, multichannel)
    keep = dataclasses.replace(env.settings, keep_multichannel=True)

    assert designed_under_other_settings(env.index, keep) == []                        # a clean design: no drift
    assert sorted(designed_under_other_settings(env.index, dataclasses.replace(keep, designer='another.designer'))) == \
        ['fs-c', 'fs-m']
    assert sorted(designed_under_other_settings(env.index, dataclasses.replace(keep, coverage='mostly_complete'))) == \
        ['fs-c', 'fs-m']
    assert designed_under_other_settings(env.index, env.settings) == ['fs-c']          # multichannel no longer kept: c is out of step
