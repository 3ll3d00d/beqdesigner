'''
Unit-level coverage of pipeline.orchestrate.Session, complementing
test_pipeline_acceptance.py's full Ready Player One run: the manual
set_filters(FilterSpec list) path, tmdb()/report() as thin wrappers, and
publish() without an image (the XML-only path).
'''
import subprocess
import wave

import numpy as np
import pytest

from model.iir import CompleteFilter, LowShelf, PeakingEQ
from pipeline.config import AnalysisConfig
from pipeline.filters import FilterSpec
from pipeline.metadata import BeqMetadata
from pipeline.orchestrate import Session
from pipeline.publish.git import RepoTarget


def _write_mono_wav(path, fs=48000, duration_s=0.5):
    n_frames = int(fs * duration_s)
    samples = (0.2 * np.sin(2 * np.pi * 40 * np.linspace(0, duration_s, n_frames, endpoint=False))).astype(np.float32)
    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes((samples * 32767).astype('<i2').tobytes())


@pytest.fixture
def loaded_signal(tmp_path):
    session = Session(AnalysisConfig())
    path = str(tmp_path / 'mono.wav')
    _write_mono_wav(path)
    return session, session.load(path, decimate=False)


def test_set_filters_accepts_a_filter_spec_list(loaded_signal):
    session, sig = loaded_signal
    specs = [FilterSpec(type='low_shelf', freq=18.0, gain=4.5, q=0.7, count=5),
            FilterSpec(type='peaking_eq', freq=40.0, gain=-3.0, q=2.0)]

    complete_filter = session.set_filters(sig, specs)

    assert isinstance(complete_filter, CompleteFilter)
    assert sig.filter is complete_filter
    filters = list(complete_filter)
    assert any(isinstance(f, LowShelf) and f.count == 5 for f in filters)
    assert any(isinstance(f, PeakingEQ) for f in filters)


def test_set_filters_accepts_a_complete_filter_directly(loaded_signal):
    session, sig = loaded_signal
    fs = sig.signal.fs
    complete_filter = CompleteFilter(fs=fs, filters=[PeakingEQ(fs, 40.0, 2.0, -3.0)])

    result = session.set_filters(sig, complete_filter)

    assert result is complete_filter
    assert sig.filter is complete_filter


def test_stats_reflects_filtering(loaded_signal):
    session, sig = loaded_signal
    session.set_filters(sig, [FilterSpec(type='peaking_eq', freq=40.0, gain=-3.0, q=2.0)])

    unfiltered = session.stats(sig, filtered=False)
    filtered = session.stats(sig, filtered=True)

    assert unfiltered.fs == sig.signal.fs
    assert filtered.fs == sig.signal.fs
    assert unfiltered.peak != filtered.peak


def test_tmdb_is_a_thin_wrapper_over_pipeline_metadata(monkeypatch, loaded_signal):
    session, _ = loaded_signal
    calls = []

    def fake_tmdb_lookup(title, year, api_key, kind='movie', audio_types=None):
        calls.append((title, year, api_key, kind, audio_types))
        return BeqMetadata(title=title, year=year, audio_types=audio_types or [])

    monkeypatch.setattr('pipeline.orchestrate.tmdb_lookup', fake_tmdb_lookup)

    meta = session.tmdb('Ready Player One', '2018', api_key='dummy', audio_types=['Atmos'])

    assert meta.title == 'Ready Player One'
    assert calls == [('Ready Player One', '2018', 'dummy', 'movie', ['Atmos'])]


def test_publish_without_an_image_only_pushes_xml(tmp_path, loaded_signal):
    session, sig = loaded_signal
    session.set_filters(sig, [FilterSpec(type='peaking_eq', freq=40.0, gain=-3.0, q=2.0)])
    meta = BeqMetadata(title='Some Film', year='2020', audio_types=['Atmos'])

    bare = tmp_path / 'xml.git'
    subprocess.run(['git', 'init', '--bare', '-q', str(bare)], check=True, capture_output=True)
    work = tmp_path / 'xml_work'
    work.mkdir()
    subprocess.run(['git', 'init', '-q', str(work)], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(work), 'config', 'user.email', 'test@example.com'], check=True,
                   capture_output=True)
    subprocess.run(['git', '-C', str(work), 'config', 'user.name', 'Test'], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(work), 'remote', 'add', 'origin', str(bare)], check=True, capture_output=True)
    xml_repo = RepoTarget(local_path=str(work))

    result = session.publish(sig.filter, meta, xml_repo, 'xml/some-film.xml')

    assert 'image_url' not in result
    assert meta.spectrum_url == ''
    assert '<beq_title>Some Film</beq_title>' in result['xml']


def test_publish_raises_when_image_png_given_without_a_repo(loaded_signal):
    session, sig = loaded_signal
    session.set_filters(sig, [FilterSpec(type='peaking_eq', freq=40.0, gain=-3.0, q=2.0)])
    meta = BeqMetadata(title='Some Film', year='2020', audio_types=['Atmos'])
    xml_repo = RepoTarget(local_path='/nonexistent')

    with pytest.raises(ValueError, match='image_png given without'):
        session.publish(sig.filter, meta, xml_repo, 'xml/some-film.xml', image_png=b'data')


def test_pipeline_orchestrate_module_has_no_qtpy_import():
    import ast
    import pathlib
    source = (pathlib.Path(__file__).parent / '..' / '..' / 'main' / 'python' / 'pipeline' / 'orchestrate.py').resolve()
    tree = ast.parse(source.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(n.name.startswith('qtpy') for n in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith('qtpy')
