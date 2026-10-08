'''
TODO R5: a published title's kept multichannel audio is compressed losslessly and restored when read; a run stops before
extraction fills the disk.
'''
import os

import numpy as np
import pytest
import soundfile as sf

from pipeline.library import retention
from pipeline.library.extract_cache import extract_params_hash, extract_status, read_manifest
from pipeline.library.retention import FLAC, WAV, check_free_space, compress_multichannel, compress_published, \
    min_free_gb, restore_multichannel
from pipeline.library.selection import Selection
from pipeline.library.source import LibraryItem
from pipeline.config import AnalysisConfig
from test_pipeline_library_commit import repos  # noqa: F401 (a fixture)
from test_pipeline_library_index import _item, _needs, _scan, env  # noqa: F401
from test_pipeline_library_stages import _accepted, _go, _publish_settings, work  # noqa: F401


def _kept(directory, frames=5000, channels=6, subtype='PCM_24', seed=3):
    ''' A kept multichannel extraction: the wav and the manifest that says it is current. '''
    os.makedirs(directory, exist_ok=True)
    data = np.random.default_rng(seed).integers(-2 ** 23, 2 ** 23, (frames, channels)).astype(np.int32) << 8
    sf.write(os.path.join(directory, WAV), data, 1000, subtype=subtype)
    return data


def test_compressing_then_restoring_gives_back_every_sample_and_saves_space(tmp_path):
    original = _kept(tmp_path)
    before = os.path.getsize(tmp_path / WAV)

    saved = compress_multichannel(str(tmp_path))

    assert not (tmp_path / WAV).exists() and (tmp_path / FLAC).is_file()
    assert saved == before - os.path.getsize(tmp_path / FLAC) and read_manifest(str(tmp_path))['multichannel_compressed'] == 'PCM_24'
    assert restore_multichannel(str(tmp_path)) is True
    restored, fs = sf.read(str(tmp_path / WAV), dtype='int32')
    assert fs == 1000 and sf.info(str(tmp_path / WAV)).subtype == 'PCM_24'
    assert np.array_equal(restored, original)
    assert not (tmp_path / FLAC).exists() and 'multichannel_compressed' not in read_manifest(str(tmp_path))
    assert restore_multichannel(str(tmp_path)) is False   # nothing to do


def test_a_float_wav_is_left_as_it_is(tmp_path):
    os.makedirs(tmp_path, exist_ok=True)
    sf.write(str(tmp_path / WAV), np.zeros((100, 2)), 1000, subtype='FLOAT')
    assert compress_multichannel(str(tmp_path)) == 0 and (tmp_path / WAV).is_file()


def test_a_compressed_extraction_is_still_current(tmp_path):
    item = LibraryItem(id='t', source_path='/films/t.mkv', display_name='t', fingerprint='fp')
    config = AnalysisConfig()
    _kept(tmp_path)
    import json
    (tmp_path / 'manifest.json').write_text(json.dumps({
        'multichannel_source_fingerprint': 'fp', 'multichannel_params_hash': extract_params_hash(item, config, False)}))
    assert extract_status(item, str(tmp_path), config, False).current

    compress_multichannel(str(tmp_path))

    assert extract_status(item, str(tmp_path), config, False).current


def test_publishing_compresses_the_published_titles_audio_without_making_it_need_publishing_again(env, work, repos):
    (a,), settings = _accepted(env, repos, 'a')
    original = _kept(os.path.join(env.work, a.id))
    _scan(env, a, settings=settings)

    report = _go(env, Selection(needs=('publish',)), 'publish', settings=settings, publish=_publish_settings(repos))

    assert [r['id'] for r in report.published] == ['fs-a']
    folder = os.path.join(env.work, a.id)
    assert os.path.isfile(os.path.join(folder, FLAC)) and not os.path.isfile(os.path.join(folder, WAV))
    assert _needs(env, 'fs-a') == ('commit', 'written, not committed')   # its digest did not change
    restore_multichannel(folder)
    assert np.array_equal(sf.read(os.path.join(folder, WAV), dtype='int32')[0], original)


def test_a_failure_to_compress_never_fails_the_publish(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(retention, 'compress_multichannel', lambda folder: (_ for _ in ()).throw(OSError('disk')))
    assert compress_published(str(tmp_path), ['x']) == 0
    assert 'Could not compress the multichannel audio of x' in caplog.text


def test_a_design_from_compressed_audio_restores_it_first(tmp_path):
    from pipeline.library.run import LibraryRunConfig, LibraryRunReport, UnitWork, _design_work

    folder = tmp_path / 'work' / 't'
    _kept(folder, channels=2)
    compress_multichannel(str(folder))
    seen = []

    class Session:
        def load_channels(self, path, layout):
            seen.append(os.path.isfile(path))
            return {}

    item = LibraryItem(id='t', source_path='/films/t.mkv', display_name='t', fingerprint='fp')
    work_ = UnitWork(item, item, str(folder / 'mono.wav'), str(folder), str(folder / WAV), 'stereo')
    config = LibraryRunConfig(work_dir=str(tmp_path / 'work'), queue_dir=str(tmp_path / 'q'), designer='none')
    import pipeline.library.run as run_module
    original_design = run_module._design
    run_module._design = lambda *args, **kwargs: None
    try:
        _design_work(Session(), work_, config, LibraryRunReport())
    finally:
        run_module._design = original_design

    assert seen == [True] and (folder / WAV).is_file()


# --- room for the next extraction ------------------------------------------------------------------------------------

def test_below_the_floor_no_extraction_starts(tmp_path, monkeypatch):
    monkeypatch.setattr(retention.shutil, 'disk_usage', lambda path: type('U', (), {'free': 4e9})())
    with pytest.raises(retention.OutOfSpace, match='only 4.0 GB is free'):
        check_free_space(str(tmp_path / 'not yet made'), 10)
    check_free_space(str(tmp_path), 3)
    check_free_space(str(tmp_path), 0)   # 0 turns it off


def test_a_full_disk_stops_the_run_at_once_and_remembers_nothing(env, work, monkeypatch):
    _scan(env, *(_item(n) for n in 'abc'))
    monkeypatch.setattr(retention.shutil, 'disk_usage', lambda path: type('U', (), {'free': 1e9})())

    report = _go(env, Selection(), 'design', unattended=True)

    assert work.calls == [] and [i for i, _ in report.run.unavailable] == ['fs-a']
    assert report.stopped.startswith('stopped: only 1.0 GB is free') and report.not_run == ['fs-b', 'fs-c']
    assert env.index.failures() == {}


def test_min_free_gb_must_be_a_number_of_gigabytes():
    assert min_free_gb() == 10 and min_free_gb(0) == 0 and min_free_gb(2.5) == 2.5
    for bad in (-1, '10', True):
        with pytest.raises(ValueError, match='run.min_free_gb'):
            min_free_gb(bad)
