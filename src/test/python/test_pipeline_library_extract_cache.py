'''
Coverage for pipeline.library.extract_cache.extract_if_needed() -- design/library-sync-pipeline-plan.md
§4.1 / Appendix D. Uses the same synthetic-wav-fixture pattern test_pipeline_review.py's
_write_synthetic_wav() already uses for real ffmpeg extraction: a plain wav file with N channels of
constant-per-channel amplitude, which ffmpeg can probe/extract without needing a real video/audio codec
fixture on disk.
'''
import dataclasses
import os
import wave

import numpy as np
import pytest

from pipeline.config import AnalysisConfig
from pipeline.library.extract_cache import extract_if_needed
from pipeline.library.source import LibraryItem
from pipeline.orchestrate import Session


def _write_synthetic_wav(path, fs=48000, duration_s=0.25, channel_values=(1000, 2000, 3000, 4000, 5000, 6000)):
    ''' Defaults to a 6-channel (5.1) source, matching test_pipeline_review.py's _write_synthetic_wav() --
    a genuinely mono (1-channel) source hits a pre-existing, unrelated ffmpeg pan-filter bug in
    model.ffmpeg.Executor's mono_mix path (Session.extract()/extract_with_layout() with mono_mix=True),
    which nothing in the existing suite exercises either since every existing extract() caller's fixture
    is multi-channel too. Out of scope for this chunk. '''
    n_frames = int(fs * duration_s)
    frame = np.array(channel_values, dtype=np.int16)
    data = np.tile(frame, (n_frames, 1)).astype('<i2').tobytes()
    with wave.open(path, 'wb') as w:
        w.setnchannels(len(channel_values))
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(data)


def _mono_item(source_path, fingerprint='fp1'):
    return LibraryItem(id='title-1', source_path=source_path, display_name='Title One', fingerprint=fingerprint)


def test_extract_if_needed_runs_ffmpeg_on_first_call(tmp_path):
    source = str(tmp_path / 'source.wav')
    _write_synthetic_wav(source)
    target_dir = str(tmp_path / 'work')
    session = Session(AnalysisConfig())
    item = _mono_item(source)

    wav_path, cached = extract_if_needed(session, item, target_dir, AnalysisConfig(), mono_mix=True)

    assert cached is False
    assert wav_path == os.path.join(target_dir, 'mono.wav')
    assert os.path.isfile(wav_path)
    manifest_path = os.path.join(target_dir, 'manifest.json')
    assert os.path.isfile(manifest_path)
    import json
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert 'mono_source_fingerprint' in manifest
    assert 'mono_params_hash' in manifest


def test_extract_if_needed_skips_ffmpeg_on_a_repeat_call(tmp_path, monkeypatch):
    source = str(tmp_path / 'source.wav')
    _write_synthetic_wav(source)
    target_dir = str(tmp_path / 'work')
    session = Session(AnalysisConfig())
    item = _mono_item(source)
    config = AnalysisConfig()

    wav_path, cached = extract_if_needed(session, item, target_dir, config, mono_mix=True)
    assert cached is False

    calls = []
    orig = Session.extract_with_layout

    def spy(self, *args, **kwargs):
        calls.append((args, kwargs))
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(Session, 'extract_with_layout', spy)

    wav_path_2, cached_2 = extract_if_needed(session, item, target_dir, config, mono_mix=True)

    assert cached_2 is True
    assert wav_path_2 == wav_path
    assert calls == []  # ffmpeg genuinely didn't run again


def test_extract_if_needed_reextracts_when_params_hash_changes(tmp_path):
    source = str(tmp_path / 'source.wav')
    _write_synthetic_wav(source)
    target_dir = str(tmp_path / 'work')
    session = Session(AnalysisConfig())
    item = _mono_item(source)

    _, cached_1 = extract_if_needed(session, item, target_dir, AnalysisConfig(target_fs=1000), mono_mix=True)
    _, cached_2 = extract_if_needed(session, item, target_dir, AnalysisConfig(target_fs=500), mono_mix=True)

    assert cached_1 is False
    assert cached_2 is False


def test_extract_if_needed_reextracts_when_source_fingerprint_changes(tmp_path):
    source = str(tmp_path / 'source.wav')
    _write_synthetic_wav(source)
    target_dir = str(tmp_path / 'work')
    session = Session(AnalysisConfig())
    item = _mono_item(source, fingerprint='fp1')
    config = AnalysisConfig()

    _, cached_1 = extract_if_needed(session, item, target_dir, config, mono_mix=True)
    changed_item = dataclasses.replace(item, fingerprint='fp2')
    _, cached_2 = extract_if_needed(session, changed_item, target_dir, config, mono_mix=True)

    assert cached_1 is False
    assert cached_2 is False


def test_extract_if_needed_force_always_reextracts(tmp_path):
    source = str(tmp_path / 'source.wav')
    _write_synthetic_wav(source)
    target_dir = str(tmp_path / 'work')
    session = Session(AnalysisConfig())
    item = _mono_item(source)
    config = AnalysisConfig()

    _, cached_1 = extract_if_needed(session, item, target_dir, config, mono_mix=True)
    _, cached_2 = extract_if_needed(session, item, target_dir, config, mono_mix=True, force=True)

    assert cached_1 is False
    assert cached_2 is False


def test_extract_if_needed_mono_and_multichannel_are_independently_cached(tmp_path, monkeypatch):
    source = str(tmp_path / 'source.wav')
    _write_synthetic_wav(source, channel_values=(1000, 2000, 3000, 4000, 5000, 6000))
    target_dir = str(tmp_path / 'work')
    session = Session(AnalysisConfig())
    item = _mono_item(source)
    config = AnalysisConfig()

    mono_path, mono_cached = extract_if_needed(session, item, target_dir, config, mono_mix=True)
    mc_path, mc_cached = extract_if_needed(session, item, target_dir, config, mono_mix=False)
    assert mono_cached is False
    assert mc_cached is False

    mc_mtime_before = os.path.getmtime(mc_path)
    import json
    manifest_path = os.path.join(target_dir, 'manifest.json')
    with open(manifest_path) as f:
        manifest_before = json.load(f)

    calls = []
    orig = Session.extract_with_layout

    def spy(self, *args, **kwargs):
        calls.append((args, kwargs))
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(Session, 'extract_with_layout', spy)

    mono_path_2, mono_cached_2 = extract_if_needed(session, item, target_dir, config, mono_mix=True)

    assert mono_cached_2 is True
    assert calls == []
    assert os.path.getmtime(mc_path) == mc_mtime_before
    with open(manifest_path) as f:
        manifest_after = json.load(f)
    assert manifest_after['multichannel_source_fingerprint'] == manifest_before['multichannel_source_fingerprint']
    assert manifest_after['multichannel_params_hash'] == manifest_before['multichannel_params_hash']
    assert manifest_after['channel_layout_name'] == manifest_before['channel_layout_name']


def test_extract_if_needed_multichannel_records_channel_layout_name(tmp_path):
    source = str(tmp_path / 'source.wav')
    _write_synthetic_wav(source, channel_values=(1000, 2000, 3000, 4000, 5000, 6000))
    target_dir = str(tmp_path / 'work')
    session = Session(AnalysisConfig())
    item = _mono_item(source)
    config = AnalysisConfig()

    extract_if_needed(session, item, target_dir, config, mono_mix=False)

    import json
    with open(os.path.join(target_dir, 'manifest.json')) as f:
        manifest = json.load(f)
    assert manifest['channel_layout_name'] == '5.1'

    # a mono call, on its own (fresh target_dir), must never write the channel_layout_name key at all
    mono_dir = str(tmp_path / 'mono_only')
    extract_if_needed(session, item, mono_dir, config, mono_mix=True)
    with open(os.path.join(mono_dir, 'manifest.json')) as f:
        mono_manifest = json.load(f)
    assert 'channel_layout_name' not in mono_manifest


def test_extract_if_needed_multichannel_does_not_decimate(tmp_path):
    source = str(tmp_path / 'source.wav')
    _write_synthetic_wav(source, fs=48000, channel_values=(1000, 2000, 3000, 4000, 5000, 6000))
    target_dir = str(tmp_path / 'work')
    session = Session(AnalysisConfig(target_fs=1000))
    item = _mono_item(source)
    config = AnalysisConfig(target_fs=1000)

    mono_path, _ = extract_if_needed(session, item, target_dir, config, mono_mix=True)
    mc_path, _ = extract_if_needed(session, item, target_dir, config, mono_mix=False)

    with wave.open(mono_path) as w:
        mono_rate = w.getframerate()
    with wave.open(mc_path) as w:
        mc_rate = w.getframerate()

    assert mono_rate == 1000
    assert mc_rate == 48000


def test_extract_with_layout_is_behaviourally_identical_to_extract(tmp_path):
    source = str(tmp_path / 'source.wav')
    _write_synthetic_wav(source, channel_values=(1000, 2000, 3000, 4000, 5000, 6000))
    session = Session(AnalysisConfig())
    dir_a = str(tmp_path / 'a')
    dir_b = str(tmp_path / 'b')

    result = session.extract_with_layout(source, dir_a)
    extract_path = session.extract(source, dir_b)

    # same source + params -> ffmpeg derives the same output basename regardless of target_dir
    expected_path = os.path.join(dir_b, os.path.basename(result.wav_path))
    assert extract_path == expected_path

    with open(result.wav_path, 'rb') as f:
        content_a = f.read()
    with open(extract_path, 'rb') as f:
        content_b = f.read()
    assert content_a == content_b


def test_pipeline_library_extract_cache_module_has_no_qtpy_import():
    import ast
    import pathlib
    source = (pathlib.Path(__file__).parent / '..' / '..' / 'main' / 'python' / 'pipeline' / 'library' /
             'extract_cache.py').resolve()
    tree = ast.parse(source.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(n.name.startswith('qtpy') for n in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith('qtpy')
