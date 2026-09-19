'''Tests for library-stage idempotent design caching.'''
from dataclasses import replace

from pipeline.config import AnalysisConfig
from pipeline.library.design_cache import design_fingerprint, design_if_needed
from pipeline.library.source import LibraryItem
from pipeline.review import QueueEntry, read_entry, update_entry, write_queue_entry


def _item(tmp_path, fingerprint='source-1'):
    source = tmp_path / 'source.wav'
    source.write_bytes(b'placeholder')
    return LibraryItem(id='title-1', source_path=str(source), display_name='Title One', fingerprint=fingerprint)


def _entry(fingerprint=None, status='pending'):
    return QueueEntry(id='title-1', fs=1000, meta={}, curve={}, status=status,
                      design_fingerprint=fingerprint)


def _install_fake_design(monkeypatch, calls):
    def fake_design(session, entry_id, wav_path, designer, queue_dir, **kwargs):
        calls.append((entry_id, wav_path, designer, kwargs))
        entry = replace(_entry(), meta=kwargs.get('meta') or {})  # like the real design_and_queue()
        write_queue_entry(queue_dir, entry)
        return entry

    monkeypatch.setattr('pipeline.library.design_cache.design_and_queue', fake_design)


def test_first_run_designs_and_records_fingerprint(tmp_path, monkeypatch):
    calls = []
    _install_fake_design(monkeypatch, calls)
    item = _item(tmp_path)
    queue_dir = str(tmp_path / 'queue')

    result = design_if_needed(None, item, '/work/mono.wav', 'designer.v1', queue_dir, AnalysisConfig())

    assert result.designed is True
    assert result.protected is False
    assert len(calls) == 1
    assert result.entry.design_fingerprint == design_fingerprint(item, 'designer.v1', AnalysisConfig(),
                                                                 'complete_programme')
    assert read_entry(queue_dir, item.id).design_fingerprint == result.entry.design_fingerprint


def test_matching_pending_entry_skips_design(tmp_path, monkeypatch):
    calls = []
    _install_fake_design(monkeypatch, calls)
    item = _item(tmp_path)
    queue_dir = str(tmp_path / 'queue')
    fingerprint = design_fingerprint(item, 'designer.v1', AnalysisConfig(), 'complete_programme')
    write_queue_entry(queue_dir, _entry(fingerprint))

    result = design_if_needed(None, item, '/work/mono.wav', 'designer.v1', queue_dir, AnalysisConfig())

    assert result.designed is False
    assert result.protected is False
    assert calls == []


def test_changed_inputs_or_force_redesign_a_nonfinal_entry(tmp_path, monkeypatch):
    calls = []
    _install_fake_design(monkeypatch, calls)
    item = _item(tmp_path)
    queue_dir = str(tmp_path / 'queue')
    write_queue_entry(queue_dir, _entry('stale'))

    changed = design_if_needed(None, item, '/work/mono.wav', 'designer.v2', queue_dir, AnalysisConfig())
    forced = design_if_needed(None, item, '/work/mono.wav', 'designer.v2', queue_dir, AnalysisConfig(), force=True)

    assert changed.designed is True
    assert forced.designed is True
    assert len(calls) == 2


def test_accepted_and_published_entries_are_never_redesigned(tmp_path, monkeypatch):
    calls = []
    _install_fake_design(monkeypatch, calls)
    item = _item(tmp_path)
    queue_dir = str(tmp_path / 'queue')

    for status in ('accepted', 'published'):
        from pipeline.review import CandidateSummary
        entry = replace(_entry('stale'), status=status,
                        candidates=[CandidateSummary(filters={}, confidence=1.0, method='exact', mv_adjust_db=0.0)],
                        chosen_candidate_index=0)
        write_queue_entry(queue_dir, entry)
        result = design_if_needed(None, item, '/work/mono.wav', 'designer.v1', queue_dir, AnalysisConfig(),
                                  force=True)

        assert result.designed is False
        assert result.protected is True
        assert result.entry.status == status

    assert calls == []


def test_design_if_needed_threads_project_arguments_on_a_real_run(tmp_path, monkeypatch):
    calls = []
    _install_fake_design(monkeypatch, calls)
    item = _item(tmp_path)
    queue_dir = str(tmp_path / 'queue')

    design_if_needed(None, item, '/work/mono.wav', 'designer.v1', queue_dir, AnalysisConfig(),
                     meta={'title': 'Title One'}, multichannel_wav_path='/work/multichannel.wav',
                     channel_layout_name='5.1', project_dir='/work/title-1')

    assert calls[0][3]['meta'] == {'title': 'Title One'}
    assert calls[0][3]['multichannel_wav_path'] == '/work/multichannel.wav'
    assert calls[0][3]['channel_layout_name'] == '5.1'
    assert calls[0][3]['project_dir'] == '/work/title-1'


def test_meta_callable_is_only_resolved_when_a_design_runs(tmp_path, monkeypatch):
    calls = []
    _install_fake_design(monkeypatch, calls)
    item = _item(tmp_path)
    queue_dir = str(tmp_path / 'queue')
    resolved = []

    def resolve():
        resolved.append(item.id)
        return {'title': 'Title One'}

    design_if_needed(None, item, '/work/mono.wav', 'designer.v1', queue_dir, AnalysisConfig(), meta=resolve)
    assert resolved == [item.id]
    assert calls[0][3]['meta'] == {'title': 'Title One'}

    # second run is a cache hit: the callable must not be invoked
    result = design_if_needed(None, item, '/work/mono.wav', 'designer.v1', queue_dir, AnalysisConfig(), meta=resolve)
    assert result.designed is False
    assert resolved == [item.id]


def test_fingerprint_changes_for_every_input_that_alters_the_design(tmp_path):
    item = _item(tmp_path)
    base = design_fingerprint(item, 'designer.v1', AnalysisConfig(), 'complete_programme')

    assert design_fingerprint(item, 'designer.v1', AnalysisConfig(), 'complete_programme', multichannel=True) != base
    assert design_fingerprint(replace(item, audio_stream=1), 'designer.v1', AnalysisConfig(),
                              'complete_programme') != base
    assert design_fingerprint(replace(item, playlist_name='00800'), 'designer.v1', AnalysisConfig(),
                              'complete_programme') != base


def test_fingerprint_ignores_metadata_and_keeps_the_pre_existing_default_value(tmp_path):
    item = _item(tmp_path)
    base = design_fingerprint(item, 'designer.v1', AnalysisConfig(), 'complete_programme')

    retagged = replace(item, title='Corrected', year='1999', external_ids={'tmdb': '1'}, meta={'edition': 'x'})
    assert design_fingerprint(retagged, 'designer.v1', AnalysisConfig(), 'complete_programme') == base
    # the payload for a default run is unchanged from before these inputs were added, so existing pending
    # entries are not redesigned (and a reviewer's edits not lost) by upgrading
    # (value computed by the implementation as it stood before those inputs were added)
    assert base == '234311ea0f896eb96cd4536999e4c74099930ee06fdaaec005d5b343b7240300'


def test_enabling_multichannel_redesigns_a_pending_entry_and_writes_its_projects(tmp_path, monkeypatch):
    calls = []
    _install_fake_design(monkeypatch, calls)
    item = _item(tmp_path)
    queue_dir = str(tmp_path / 'queue')

    design_if_needed(None, item, '/work/mono.wav', 'designer.v1', queue_dir, AnalysisConfig())
    result = design_if_needed(None, item, '/work/mono.wav', 'designer.v1', queue_dir, AnalysisConfig(),
                              multichannel_wav_path='/work/multichannel.wav')

    assert result.designed is True
    assert len(calls) == 2
    assert calls[1][3]['multichannel_wav_path'] == '/work/multichannel.wav'


def test_redesigning_a_pending_entry_keeps_a_reviewers_metadata_artwork_and_note(tmp_path, monkeypatch):
    calls = []
    _install_fake_design(monkeypatch, calls)
    item = _item(tmp_path)
    queue_dir = str(tmp_path / 'queue')
    write_queue_entry(queue_dir, replace(_entry('stale'), meta={'title': 'Reviewer Title', 'edition': 'Extended'},
                                         art_path='/art/poster.jpg', art_overridden=True,
                                         reviewer_note='check the LFE'))

    result = design_if_needed(None, item, '/work/mono.wav', 'designer.v2', queue_dir, AnalysisConfig(),
                              meta={'title': 'Library Title', 'year': '1999'})

    assert result.designed is True
    stored = read_entry(queue_dir, item.id)
    assert stored.meta == {'title': 'Reviewer Title', 'year': '1999', 'edition': 'Extended'}
    assert stored.art_path == '/art/poster.jpg'
    assert stored.art_overridden is True
    assert stored.reviewer_note == 'check the LFE'
    assert stored.design_fingerprint == result.entry.design_fingerprint
    assert stored.design_fingerprint != 'stale'
    assert calls[0][3]['meta'] == {'title': 'Reviewer Title', 'year': '1999', 'edition': 'Extended'}


def test_a_first_design_has_nothing_to_preserve(tmp_path, monkeypatch):
    calls = []
    _install_fake_design(monkeypatch, calls)
    item = _item(tmp_path)

    result = design_if_needed(None, item, '/work/mono.wav', 'designer.v1', str(tmp_path / 'queue'),
                              AnalysisConfig(), meta={'title': 'Title One'})

    assert calls[0][3]['meta'] == {'title': 'Title One'}
    assert result.entry.art_path is None
    assert result.entry.art_overridden is False
    assert result.entry.reviewer_note is None


def _stub_art(monkeypatch, result):
    calls = []
    monkeypatch.setattr('pipeline.library.design_cache.resolve_art',
                        lambda item, meta, art_dir: calls.append((item.id, dict(meta), art_dir)) or result)
    return calls


def test_a_design_stores_the_resolved_artwork_on_the_entry(tmp_path, monkeypatch):
    _install_fake_design(monkeypatch, [])
    art_calls = _stub_art(monkeypatch, '/work/title-1/poster.jpg')
    queue_dir = str(tmp_path / 'queue')

    result = design_if_needed(None, _item(tmp_path), '/work/mono.wav', 'designer.v1', queue_dir, AnalysisConfig(),
                              meta={'title': 'T', 'poster': '/p.jpg'}, project_dir='/work/title-1')

    assert result.entry.art_path == '/work/title-1/poster.jpg'
    assert result.entry.art_overridden is False
    assert read_entry(queue_dir, 'title-1').art_path == '/work/title-1/poster.jpg'
    assert art_calls == [('title-1', {'title': 'T', 'poster': '/p.jpg'}, '/work/title-1')]


def test_a_reviewers_artwork_is_never_replaced_by_a_redesign(tmp_path, monkeypatch):
    _install_fake_design(monkeypatch, [])
    art_calls = _stub_art(monkeypatch, '/auto.jpg')
    queue_dir = str(tmp_path / 'queue')
    # the file is deliberately absent: a human's choice is kept even if it has since gone missing
    write_queue_entry(queue_dir, replace(_entry('stale'), art_path='/mine/poster.jpg', art_overridden=True))

    result = design_if_needed(None, _item(tmp_path), '/work/mono.wav', 'designer.v2', queue_dir, AnalysisConfig())

    assert result.entry.art_path == '/mine/poster.jpg'
    assert result.entry.art_overridden is True
    assert art_calls == []


def test_existing_automatic_artwork_is_kept_while_its_file_exists_and_replaced_when_it_does_not(tmp_path,
                                                                                                monkeypatch):
    _install_fake_design(monkeypatch, [])
    art_calls = _stub_art(monkeypatch, '/fresh.jpg')
    queue_dir = str(tmp_path / 'queue')
    existing = tmp_path / 'poster.jpg'
    existing.write_bytes(b'x')
    write_queue_entry(queue_dir, replace(_entry('stale'), art_path=str(existing)))

    kept = design_if_needed(None, _item(tmp_path), '/work/mono.wav', 'designer.v2', queue_dir, AnalysisConfig())
    assert kept.entry.art_path == str(existing)
    assert art_calls == []

    existing.unlink()
    write_queue_entry(queue_dir, replace(_entry('stale'), art_path=str(existing)))
    replaced = design_if_needed(None, _item(tmp_path), '/work/mono.wav', 'designer.v2', queue_dir, AnalysisConfig())
    assert replaced.entry.art_path == '/fresh.jpg'


def test_a_cache_hit_does_not_resolve_artwork(tmp_path, monkeypatch):
    _install_fake_design(monkeypatch, [])
    art_calls = _stub_art(monkeypatch, '/auto.jpg')
    item = _item(tmp_path)
    queue_dir = str(tmp_path / 'queue')
    write_queue_entry(queue_dir, _entry(design_fingerprint(item, 'designer.v1', AnalysisConfig(),
                                                           'complete_programme')))

    result = design_if_needed(None, item, '/work/mono.wav', 'designer.v1', queue_dir, AnalysisConfig())

    assert result.designed is False
    assert result.entry.art_path is None
    assert art_calls == []
