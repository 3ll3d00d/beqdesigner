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
        entry = _entry()
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
