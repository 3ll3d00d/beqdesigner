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
from pipeline.orchestrate import Applied, Session
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


def test_design_threads_bass_management_and_gain_reduction_db(loaded_signal):
    '''
    designer-interface-feedback.md #2/#1: DesignRequest.bass_management is
    passed straight through, and a candidate's gain_reduction_db (the
    actual clipping-cost metric, distinct from mv_adjust_db) survives onto
    the Applied outcome.
    '''
    from pipeline.designer.contract import BiquadSpec, DesignCandidate, DesignResponse
    from pipeline.designer.registry import register_designer, unregister_designer

    session, sig = loaded_signal
    bm_config = {'lpf_fs': 80.0, 'lpf_position': 'Before', 'headroom_type': 'WCS',
                'clip_before': False, 'clip_after': False}
    seen_requests = []

    def fake_designer(request):
        seen_requests.append(request)
        return DesignResponse(contract_version='1.0', candidates=[
            DesignCandidate(filters=[BiquadSpec(type='low_shelf', freq_hz=18.0, gain_db=4.5, q=0.7)],
                            confidence=0.9, mv_adjust_db=22.5, gain_reduction_db=-4.37, method='fitted'),
        ])

    register_designer('test.bm', fake_designer)
    try:
        outcome = session.design(sig, 'test.bm', bass_management=bm_config)
    finally:
        unregister_designer('test.bm')

    assert seen_requests[0].bass_management == bm_config
    assert outcome.mv_adjust_db == 22.5
    assert outcome.gain_reduction_db == -4.37


def test_design_bass_management_defaults_to_none(loaded_signal):
    from pipeline.designer.contract import BiquadSpec, DesignCandidate, DesignResponse
    from pipeline.designer.registry import register_designer, unregister_designer

    session, sig = loaded_signal
    seen_requests = []

    def fake_designer(request):
        seen_requests.append(request)
        return DesignResponse(contract_version='1.0', candidates=[
            DesignCandidate(filters=[BiquadSpec(type='low_shelf', freq_hz=18.0, gain_db=4.5, q=0.7)],
                            confidence=0.9, mv_adjust_db=22.5, method='fitted'),
        ])

    register_designer('test.bm-none', fake_designer)
    try:
        outcome = session.design(sig, 'test.bm-none')
    finally:
        unregister_designer('test.bm-none')

    assert seen_requests[0].bass_management is None
    assert outcome.gain_reduction_db is None


def test_manual_designer_returns_an_editable_flat_filter(loaded_signal):
    from pipeline.designer.manual import MANUAL_DESIGNER

    session, sig = loaded_signal
    outcome = session.design(sig, MANUAL_DESIGNER)

    assert outcome.confidence == 0.0 and outcome.commentary['Manual filter'].startswith('No automatic')
    assert len(outcome.filters.filters) == 1 and outcome.filters.filters[0].gain == 0.0


def _write_multichannel_wav(path, channel_values, fs=48000, duration_s=0.1):
    n_frames = int(fs * duration_s)
    frame = np.array(channel_values, dtype=np.int16)
    data = np.tile(frame, (n_frames, 1)).astype('<i2').tobytes()
    with wave.open(path, 'wb') as w:
        w.setnchannels(len(channel_values))
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(data)


def test_load_channels_decomposes_a_multichannel_wav_with_layout_labels(tmp_path):
    '''
    Pure decomposition, no mixing -- design/designer-interface.md §2's DesignRequest.channels input.
    Labels come from model.ffmpeg's layout tables, same as the GUI already uses for a multichannel
    extraction's per-channel signal names.
    '''
    session = Session(AnalysisConfig())
    path = str(tmp_path / 'multi.wav')
    _write_multichannel_wav(path, (1000, 2000, 3000, 4000, 5000, 6000))

    channels = session.load_channels(path, channel_layout_name='5.1', decimate=False)

    assert set(channels.keys()) == {'FL', 'FR', 'FC', 'LFE', 'BL', 'BR'}
    # constant per-channel amplitude in the source wav -- ordering survives normalisation/resampling
    ordered = [channels[label][0] for label in ('FL', 'FR', 'FC', 'LFE', 'BL', 'BR')]
    assert ordered == sorted(ordered)
    assert all(v > 0 for v in ordered)


def test_load_channels_of_a_mono_wav_is_empty(tmp_path):
    session = Session(AnalysisConfig())
    path = str(tmp_path / 'mono.wav')
    _write_mono_wav(path)

    assert session.load_channels(path) == {}


def test_design_threads_channels_through_to_the_request(loaded_signal, tmp_path):
    ''' DesignRequest.channels -- the per-channel diagnostic a designer may use for channel_scope. '''
    from pipeline.designer.contract import BiquadSpec, DesignCandidate, DesignResponse
    from pipeline.designer.registry import register_designer, unregister_designer

    session, sig = loaded_signal
    channels = {'FL': np.zeros(len(sig.signal.samples)), 'FR': np.ones(len(sig.signal.samples))}
    seen_requests = []

    def fake_designer(request):
        seen_requests.append(request)
        return DesignResponse(contract_version='1.0', candidates=[
            DesignCandidate(filters=[BiquadSpec(type='low_shelf', freq_hz=18.0, gain_db=4.5, q=0.7)],
                            confidence=0.9, mv_adjust_db=22.5, method='fitted'),
        ])

    register_designer('test.channels', fake_designer)
    try:
        session.design(sig, 'test.channels', channels=channels)
    finally:
        unregister_designer('test.channels')

    assert seen_requests[0].channels is channels


def test_design_rejects_misaligned_channel_diagnostics_before_calling_designer(loaded_signal):
    '''Mismatched extraction outputs are an error to fix, not optional diagnostics to silently discard.'''
    import pytest
    from pipeline.designer.registry import register_designer, unregister_designer

    session, sig = loaded_signal
    seen_requests = []

    def fake_designer(request):
        seen_requests.append(request)
        return None

    register_designer('test.misaligned-channels', fake_designer)
    try:
        with pytest.raises(ValueError, match='channels must be one-dimensional and match mono_mix length'):
            session.design(sig, 'test.misaligned-channels', channels={'FL': np.zeros(10), 'FR': np.ones(10)})
    finally:
        unregister_designer('test.misaligned-channels')

    assert seen_requests == []


def test_design_threads_excerpt_coverage_through_to_the_request(loaded_signal):
    ''' design/designer-interface.md §2's Coverage -- every other test here only ever exercises
    the default 'complete_programme'; this confirms 'excerpt' actually reaches the request too. '''
    from pipeline.designer.contract import BiquadSpec, DesignCandidate, DesignResponse
    from pipeline.designer.registry import register_designer, unregister_designer

    session, sig = loaded_signal
    seen_requests = []

    def fake_designer(request):
        seen_requests.append(request)
        return DesignResponse(contract_version='1.0', candidates=[
            DesignCandidate(filters=[BiquadSpec(type='low_shelf', freq_hz=18.0, gain_db=4.5, q=0.7)],
                            confidence=0.9, mv_adjust_db=22.5, method='fitted'),
        ])

    register_designer('test.excerpt', fake_designer)
    try:
        session.design(sig, 'test.excerpt', coverage='excerpt')
    finally:
        unregister_designer('test.excerpt')

    assert seen_requests[0].coverage == 'excerpt'


def test_design_builds_alternatives_from_ranked_candidates_best_first(loaded_signal):
    '''
    design/designer-interface.md §3: a designer may return several ranked candidates.
    Session.design() must reflect only candidates[0] in Applied's own fields (the only one ever
    simulated/published) and carry the rest as Applied.alternatives, in the same best-first
    order -- pipeline.designer.convert.alternative_filters() is already tested against this rule
    in isolation (test_pipeline_designer_contract.py); this is the untested wiring one level up,
    where orchestrate.py zips candidates[1:] against alternative_filters()'s output itself.
    '''
    from model.iir import LowShelf, PeakingEQ
    from pipeline.designer.contract import BiquadSpec, DesignCandidate, DesignResponse
    from pipeline.designer.registry import register_designer, unregister_designer

    session, sig = loaded_signal

    def fake_designer(request):
        return DesignResponse(contract_version='1.0', candidates=[
            DesignCandidate(filters=[BiquadSpec(type='low_shelf', freq_hz=18.0, gain_db=4.5, q=0.7)],
                            confidence=0.9, mv_adjust_db=18.0, method='exact', commentary={'note': 'best'}),
            DesignCandidate(filters=[BiquadSpec(type='peaking_eq', freq_hz=40.0, gain_db=-3.0, q=2.0)],
                            confidence=0.6, mv_adjust_db=3.0, method='fitted', commentary={'note': 'second'}),
            DesignCandidate(filters=[BiquadSpec(type='low_shelf', freq_hz=20.0, gain_db=2.0, q=0.7)],
                            confidence=0.3, mv_adjust_db=2.0, method='non_parametric', commentary={'note': 'third'}),
        ])

    register_designer('test.alternatives', fake_designer)
    try:
        outcome = session.design(sig, 'test.alternatives')
    finally:
        unregister_designer('test.alternatives')

    assert isinstance(outcome, Applied)
    assert isinstance(outcome.filters.filters[0], LowShelf) and outcome.confidence == 0.9

    assert len(outcome.alternatives) == 2
    first, second = outcome.alternatives
    assert isinstance(first.filters.filters[0], PeakingEQ)
    assert first.confidence == 0.6 and first.commentary == {'note': 'second'}
    assert isinstance(second.filters.filters[0], LowShelf)
    assert second.confidence == 0.3 and second.commentary == {'note': 'third'}


def test_design_threads_fc_slope_and_uncertainties_onto_applied(loaded_signal):
    ''' Diagnostic-only fields (design/designer-interface.md §3) -- never published to the XML,
    but must still survive from the winning candidate onto Applied for a report to show. '''
    from pipeline.designer.contract import BiquadSpec, DesignCandidate, DesignResponse
    from pipeline.designer.registry import register_designer, unregister_designer

    session, sig = loaded_signal

    def fake_designer(request):
        return DesignResponse(contract_version='1.0', candidates=[
            DesignCandidate(filters=[BiquadSpec(type='low_shelf', freq_hz=15.81, gain_db=15.918, q=0.7071)],
                            confidence=0.94, mv_adjust_db=15.918, method='exact',
                            fc_hz=25.0, slope=24.0, fc_uncertainty_hz=0.6, slope_uncertainty=1.1),
        ])

    register_designer('test.fc', fake_designer)
    try:
        outcome = session.design(sig, 'test.fc')
    finally:
        unregister_designer('test.fc')

    assert isinstance(outcome, Applied)
    assert outcome.fc_hz == 25.0
    assert outcome.slope == 24.0
    assert outcome.fc_uncertainty_hz == 0.6
    assert outcome.slope_uncertainty == 1.1


def test_design_threads_channel_scope_onto_applied_and_alternatives(loaded_signal):
    '''
    design/designer-interface.md §3's channel_scope -- "worth populating whenever you have an
    answer, since it's otherwise invisible to the caller". Applied/AlternativeDesign previously
    had no field for it at all, so a fully conformant designer response populating channel_scope
    was silently dropped between validate_response() and the outcome a report would show.
    '''
    from pipeline.designer.contract import BiquadSpec, DesignCandidate, DesignResponse
    from pipeline.designer.registry import register_designer, unregister_designer

    session, sig = loaded_signal

    def fake_designer(request):
        return DesignResponse(contract_version='1.0', candidates=[
            DesignCandidate(filters=[BiquadSpec(type='low_shelf', freq_hz=18.0, gain_db=4.5, q=0.7)],
                            confidence=0.9, mv_adjust_db=18.0, method='exact', channel_scope='all_channels'),
            DesignCandidate(filters=[BiquadSpec(type='peaking_eq', freq_hz=40.0, gain_db=-3.0, q=2.0)],
                            confidence=0.5, mv_adjust_db=3.0, method='fitted', channel_scope='lfe_only'),
        ])

    register_designer('test.channel-scope', fake_designer)
    try:
        outcome = session.design(sig, 'test.channel-scope')
    finally:
        unregister_designer('test.channel-scope')

    assert isinstance(outcome, Applied)
    assert outcome.channel_scope == 'all_channels'
    assert len(outcome.alternatives) == 1
    assert outcome.alternatives[0].channel_scope == 'lfe_only'


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

    result = session.publish(sig.filter, meta, xml_repo, 'xml/some-film.json')

    assert 'image_url' not in result
    assert meta.spectrum_url == ''
    assert result['record']['title'] == 'Some Film'


def test_publish_raises_when_image_png_given_without_a_repo(loaded_signal):
    session, sig = loaded_signal
    session.set_filters(sig, [FilterSpec(type='peaking_eq', freq=40.0, gain=-3.0, q=2.0)])
    meta = BeqMetadata(title='Some Film', year='2020', audio_types=['Atmos'])
    xml_repo = RepoTarget(local_path='/nonexistent')

    with pytest.raises(ValueError, match='image_png given without'):
        session.publish(sig.filter, meta, xml_repo, 'xml/some-film.json', image_png=b'data')


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
