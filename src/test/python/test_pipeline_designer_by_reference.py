'''
Arrays by reference (designer-interface.md §7.1, 1.2; design/designer-by-reference.md D1.2): http_designer(url,
shared_root=...) sends an array whose source WAV is under the shared root as a `file` plus the SHA-256 of the samples it
would otherwise have carried, sends everything else inline, checks the designer's /health once before the first request
by reference, and reports a 422 as a failure without retrying inline. Session.design() passes sources only to a
designer registered as taking them.

The designer here is a fake on a real socket that does what §7.1 asks of one: resolves the path under its own root,
refuses one that leaves it, decodes the column and checks the digest before using it.
'''
import base64
import hashlib
import http.server
import json
import os
import threading
from urllib.parse import urlsplit

import numpy as np
import pytest
import soundfile as sf

from pipeline.designer.contract import DesignResponse, build_request
from pipeline.designer.http_binding import HttpDesignerError, _request_to_json, health_url, http_designer, sha256_of
from pipeline.designer.registry import register_designer, takes_sources, unregister_designer
from pipeline.designer.sources import ArraySource, AudioSources

FS = 1000


def _write_wav(path, columns, fs=FS):
    sf.write(path, np.column_stack(columns), fs, subtype='PCM_24')
    data, _ = sf.read(path, dtype='float64', always_2d=True)
    return [data[:, i].copy() for i in range(data.shape[1])]


def _decode(array, root):
    ''' §7.1's decoding, as a designer does it; raises ValueError (-> 422) for anything it may not honour. '''
    if 'data_base64' in array:
        return np.frombuffer(base64.b64decode(array['data_base64']), dtype='<f8')
    if root is None:
        raise ValueError('no shared root configured')
    real_root = os.path.realpath(root)
    real_path = os.path.realpath(os.path.join(real_root, array['file']['path']))
    if os.path.commonpath([real_root, real_path]) != real_root:
        raise ValueError('path is outside the shared root')
    data, fs = sf.read(real_path, dtype='float64', always_2d=True)
    column = np.ascontiguousarray(data[:, array['file']['channel']], dtype='<f8')
    if hashlib.sha256(column.tobytes()).hexdigest() != array['sha256']:
        raise ValueError('digest differs')
    return column


class _Designer(http.server.BaseHTTPRequestHandler):
    root = None
    health = None
    bodies = None
    gets = None

    def do_GET(self):
        self.gets.append(self.path)
        self._send(200, self.health)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.bodies.append(body)
        try:
            mono = _decode(body['mono_mix'], self.root)
            channels = {k: _decode(v, self.root) for k, v in (body.get('channels') or {}).items()}
        except (ValueError, IndexError, RuntimeError) as e:
            self._send(422, {'error': f'mono_mix: {e}'})
            return
        digests = {'mono_mix': sha256_of(mono), **{k: sha256_of(v) for k, v in channels.items()}}
        self._send(200, {'contract_version': body['contract_version'], 'candidates': [{
            'filters': [{'type': 'low_shelf', 'freq_hz': 20.0, 'gain_db': 3.0, 'q': 0.7}],
            'confidence': 0.9, 'mv_adjust_db': 0.0, 'method': 'fitted', 'commentary': digests}]})

    def _send(self, status, payload):
        out = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args):
        pass


@pytest.fixture
def designer(tmp_path):
    ''' (url, handler class) of a by-reference designer whose root is tmp_path/'shared'. '''
    shared = tmp_path / 'shared'
    shared.mkdir()
    handler = type('Handler', (_Designer,), {'root': str(shared), 'bodies': [], 'gets': [],
                                             'health': {'contract_version': '1.2', 'shared_root': True}})
    server = http.server.HTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f'http://127.0.0.1:{server.server_address[1]}/design', handler
    server.shutdown()
    server.server_close()


@pytest.fixture
def title(tmp_path):
    ''' A title extracted under the shared root: its request, its sources and the root. '''
    shared = tmp_path / 'shared'
    shared.mkdir(exist_ok=True)
    (shared / 't_1').mkdir()
    rng = np.random.default_rng(1)
    mono_path = str(shared / 't_1' / 'mono.wav')
    mc_path = str(shared / 't_1' / 'multichannel.wav')
    [mono] = _write_wav(mono_path, [rng.uniform(-0.5, 0.5, 2000)])
    left, right, lfe = _write_wav(mc_path, [rng.uniform(-0.5, 0.5, 2000) for _ in range(3)])
    request = build_request(mono, FS, channels={'L': left, 'R': right, 'LFE': lfe})
    sources = AudioSources.from_wavs(mono_path, mc_path, ['L', 'R', 'LFE'])
    return request, sources, str(shared)


def _inline_digests(request):
    ''' SHA-256 of the bytes each array's inline encoding carries. '''
    body = _request_to_json(request)
    arrays = {'mono_mix': body['mono_mix'], **body['channels']}
    return {k: hashlib.sha256(base64.b64decode(v['data_base64'])).hexdigest() for k, v in arrays.items()}


def test_arrays_under_the_shared_root_go_by_reference_with_the_digest_of_what_would_have_gone_inline(designer, title):
    url, handler = designer
    request, sources, root = title

    response = http_designer(url, shared_root=root)(request, sources=sources)

    [body] = handler.bodies
    assert body['contract_version'] == '1.2'
    assert body['mono_mix']['file'] == {'path': 't_1/mono.wav', 'channel': 0}
    assert {k: v['file'] for k, v in body['channels'].items()} == {
        'L': {'path': 't_1/multichannel.wav', 'channel': 0}, 'R': {'path': 't_1/multichannel.wav', 'channel': 1},
        'LFE': {'path': 't_1/multichannel.wav', 'channel': 2}}
    assert not any('data_base64' in v for v in [body['mono_mix'], *body['channels'].values()])
    inline = _inline_digests(request)
    assert {'mono_mix': body['mono_mix']['sha256'], **{k: v['sha256'] for k, v in body['channels'].items()}} == inline
    # what the designer decoded from the files is byte for byte what the inline form carries
    assert response.candidates[0].commentary == inline


def test_the_health_is_asked_once_on_the_design_endpoints_origin(designer, title):
    url, handler = designer
    request, sources, root = title
    call = http_designer(url, shared_root=root)

    call(request, sources=sources)
    call(request, sources=sources)

    assert handler.gets == ['/health']
    assert health_url('http://host:8420/beq/design?x=1') == 'http://host:8420/health'


def test_a_file_outside_the_shared_root_goes_inline(designer, title, tmp_path):
    url, handler = designer
    request, _, root = title
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    [mono] = _write_wav(str(elsewhere / 'mono.wav'), [np.asarray(request.mono_mix)])

    http_designer(url, shared_root=root)(build_request(mono, FS), sources=AudioSources.from_wavs(str(elsewhere / 'mono.wav')))

    [body] = handler.bodies
    assert body['contract_version'] == '1.1' and 'data_base64' in body['mono_mix']
    assert handler.gets == []  # nothing by reference, so no need to ask


def test_a_symlink_under_the_root_that_leads_out_of_it_goes_inline(designer, title, tmp_path):
    url, handler = designer
    request, _, root = title
    outside = tmp_path / 'outside.wav'
    [mono] = _write_wav(str(outside), [np.asarray(request.mono_mix)])
    link = os.path.join(root, 't_1', 'link.wav')
    os.symlink(outside, link)

    http_designer(url, shared_root=root)(build_request(mono, FS), sources=AudioSources.from_wavs(link))

    assert 'data_base64' in handler.bodies[0]['mono_mix']


def test_a_wav_that_cannot_be_the_array_unchanged_goes_inline(designer, title, tmp_path):
    ''' Another rate (the array was resampled), another length (trimmed), a channel it does not have. '''
    url, handler = designer
    request, sources, root = title
    other_rate = os.path.join(root, 't_1', 'other_rate.wav')
    _write_wav(other_rate, [np.asarray(request.mono_mix)], fs=2 * FS)
    call = http_designer(url, shared_root=root)

    call(request, sources=AudioSources(mono=ArraySource(other_rate, 0)))
    call(build_request(request.mono_mix[:-1], FS), sources=AudioSources(mono=sources.mono))
    call(request, sources=AudioSources(mono=ArraySource(sources.mono.path, 1)))

    assert all('data_base64' in body['mono_mix'] for body in handler.bodies)


def test_a_wave_format_extensible_file_goes_by_reference(designer, title):
    ''' What ffmpeg writes for more than two channels -- libsndfile calls it WAVEX -- is a WAV like any other. '''
    url, handler = designer
    request, sources, root = title
    path = os.path.join(root, 't_1', 'extensible.wav')
    data, _ = sf.read(sources.channels['L'].path, dtype='float64', always_2d=True)
    sf.write(path, data, FS, subtype='PCM_24', format='WAVEX')
    assert sf.info(path).format == 'WAVEX'
    extensible = AudioSources.from_wavs(sources.mono.path, path, ['L', 'R', 'LFE'])

    response = http_designer(url, shared_root=root)(request, sources=extensible)

    assert all('file' in v for v in handler.bodies[0]['channels'].values())
    assert response.candidates[0].commentary == _inline_digests(request)


def test_a_request_mixes_forms_when_only_some_arrays_have_a_source(designer, title):
    url, handler = designer
    request, sources, root = title

    response = http_designer(url, shared_root=root)(request, sources=AudioSources(channels={'LFE': sources.channels['LFE']}))

    [body] = handler.bodies
    assert body['contract_version'] == '1.2'
    assert 'data_base64' in body['mono_mix'] and 'data_base64' in body['channels']['L']
    assert 'file' in body['channels']['LFE']
    assert response.candidates[0].commentary == _inline_digests(request)


def test_without_a_shared_root_every_array_goes_inline_and_the_callable_takes_the_request_alone(designer, title):
    url, handler = designer
    request, sources, _ = title
    call = http_designer(url)

    call(request)

    assert handler.bodies[0]['contract_version'] == '1.1' and 'data_base64' in handler.bodies[0]['mono_mix']
    with pytest.raises(TypeError):
        call(request, sources=sources)


@pytest.mark.parametrize('health', [{'contract_version': '1.1', 'shared_root': True},
                                    {'contract_version': '1.2', 'shared_root': False},
                                    {'status': 'ok'}])
def test_a_designer_whose_health_cannot_take_references_is_sent_nothing(designer, title, health):
    url, handler = designer
    request, sources, root = title
    handler.health = health

    with pytest.raises(HttpDesignerError, match='cannot take arrays by reference') as raised:
        http_designer(url, shared_root=root)(request, sources=sources)

    assert handler.bodies == []
    assert repr(health) in str(raised.value)


def test_a_422_is_a_failure_with_the_designers_reason_and_is_not_retried_inline(designer, title):
    url, handler = designer
    request, sources, root = title
    with open(sources.mono.path, 'r+b') as f:  # the file changes after the caller loaded it
        f.seek(-3, os.SEEK_END)
        f.write(b'\x7f\x7f\x7f')

    with pytest.raises(HttpDesignerError, match='422.*mono_mix: digest differs'):
        http_designer(url, shared_root=root)(request, sources=sources)

    assert len(handler.bodies) == 1


def test_session_design_passes_sources_only_to_a_designer_registered_as_taking_them(title):
    from pipeline.config import AnalysisConfig
    from pipeline.orchestrate import Session
    from model.signal import Signal, SingleChannelSignalData
    request, sources, _ = title
    seen = []

    def decline():
        return DesignResponse(contract_version='1.1', decline_reason='no_rolloff_detected', decline_message='-')

    def with_sources(req, sources=None):
        seen.append(('with', sources))
        return decline()

    def request_only(req):
        seen.append(('only', None))
        return decline()

    register_designer('test.with_sources', with_sources, takes_sources=True)
    register_designer('test.request_only', request_only)
    try:
        session = Session(AnalysisConfig())
        sig = SingleChannelSignalData(name='t', signal=Signal('t', request.mono_mix, fs=FS))
        session.design(sig, 'test.with_sources', sources=sources)
        session.design(sig, 'test.request_only', sources=sources)
        session.design(sig, 'test.with_sources')
    finally:
        unregister_designer('test.with_sources')
        unregister_designer('test.request_only')
    assert seen == [('with', sources), ('only', None), ('with', None)]


def test_the_registry_forgets_that_a_designer_takes_sources_when_it_is_replaced_or_removed():
    register_designer('test.flag', lambda r, sources=None: None, takes_sources=True)
    assert takes_sources('test.flag')
    register_designer('test.flag', lambda r: None)
    assert not takes_sources('test.flag')
    register_designer('test.flag', lambda r, sources=None: None, takes_sources=True)
    unregister_designer('test.flag')
    assert not takes_sources('test.flag')


def test_sources_from_wavs_name_each_label_by_its_column():
    sources = AudioSources.from_wavs('/w/mono.wav', '/w/mc.wav', ['L', 'R', 'LFE'])
    assert sources.mono == ArraySource('/w/mono.wav', 0)
    assert sources.channels == {'L': ArraySource('/w/mc.wav', 0), 'R': ArraySource('/w/mc.wav', 1),
                                'LFE': ArraySource('/w/mc.wav', 2)}
    assert AudioSources.from_wavs('/w/mono.wav', None, ['L']).channels == {}
    assert AudioSources.from_wavs(None).mono is None


def _spy_by_reference(name, url, root, requests_seen):
    ''' Registers `name` as the real by-reference binding, remembering each request and its sources on the way. '''
    inner = http_designer(url, shared_root=root)

    def spy(request, sources=None):
        requests_seen.append((request, sources))
        return inner(request, sources=sources)

    register_designer(name, spy, takes_sources=True)


class _Decoding(_Designer):
    ''' Keeps what it decoded from each file, for a test to compare with the request. '''
    decoded = None

    def do_POST(self):
        length = int(self.headers['Content-Length'])
        raw = self.rfile.read(length)
        body = json.loads(raw)
        arrays = {'mono_mix': body['mono_mix'], **(body.get('channels') or {})}
        self.decoded.append({k: (_decode(v, self.root), v.get('file')) for k, v in arrays.items()})
        self.bodies.append(body)
        self._send(200, {'contract_version': body['contract_version'], 'decline_reason': 'no_rolloff_detected',
                         'decline_message': 'decoded'})


@pytest.mark.requires_ffmpeg
def test_the_library_path_sends_its_extraction_by_reference_and_the_designer_decodes_what_inline_would_carry(tmp_path):
    '''
    AGENTS.md's extraction/design parity rule, for arrays by reference: a short synthetic 5.1 source through
    run_library's extraction and design, to a by-reference designer sharing the work directory. Every array is sent by
    reference, and decodes byte-identical to the request's own (so to its inline encoding), at the same rate and frame
    count for mono and every channel.
    '''
    from test_pipeline_library_extract_cache import _mono_item, _write_synthetic_wav
    from pipeline.config import AnalysisConfig
    from pipeline.library.run import LibraryRunConfig, run_library
    work = tmp_path / 'work'
    work.mkdir()
    handler = type('Handler', (_Decoding,), {'root': str(work), 'bodies': [], 'gets': [], 'decoded': [],
                                             'health': {'contract_version': '1.2', 'shared_root': True}})
    server = http.server.HTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    source_path = str(tmp_path / 'source.wav')
    _write_synthetic_wav(source_path, fs=48000, duration_s=0.5)
    item = _mono_item(source_path)

    class Source:
        def list_items(self, **query):
            return [item]

    seen = []
    _spy_by_reference('test.library_by_reference', f'http://127.0.0.1:{server.server_address[1]}/design', str(work), seen)
    try:
        report = run_library(Source(), LibraryRunConfig(
            work_dir=str(work), queue_dir=str(tmp_path / 'queue'), designer='test.library_by_reference',
            config=AnalysisConfig(target_fs=1000), keep_multichannel=True))
    finally:
        unregister_designer('test.library_by_reference')
        server.shutdown()
        server.server_close()

    assert report.failed == [] and report.designed == ['title-1']
    [(request, _)] = seen
    [decoded] = handler.decoded
    [body] = handler.bodies
    assert body['contract_version'] == '1.2'
    assert decoded['mono_mix'][1] == {'path': 'title-1/mono.wav', 'channel': 0}
    assert len(request.channels) == 6
    expected = {'mono_mix': request.mono_mix, **request.channels}
    assert set(decoded) == set(expected)
    for name, (samples, file) in decoded.items():
        assert file is not None, f'{name} went inline'
        assert samples.tobytes() == np.ascontiguousarray(expected[name], dtype='<f8').tobytes(), name
        assert len(samples) == len(request.mono_mix)
    assert {v[1]['path'] for k, v in decoded.items() if k != 'mono_mix'} == {'title-1/multichannel.wav'}
    assert request.fs == 1000
    assert sf.info(str(work / 'title-1' / 'mono.wav')).samplerate == \
        sf.info(str(work / 'title-1' / 'multichannel.wav')).samplerate == request.fs


def test_design_and_queue_names_the_wavs_its_arrays_came_from(title, tmp_path):
    from pipeline.config import AnalysisConfig
    from pipeline.orchestrate import Session
    from pipeline.review import design_and_queue
    request, sources, root = title
    mc_path = sources.channels['L'].path
    channels = Session(AnalysisConfig(target_fs=FS)).load_channels(mc_path)
    seen = []

    def designer(req, sources=None):
        seen.append(sources)
        return DesignResponse(contract_version='1.1', decline_reason='no_rolloff_detected', decline_message='-')

    register_designer('test.names_wavs', designer, takes_sources=True)
    try:
        design_and_queue(Session(AnalysisConfig(target_fs=FS)), 't_1', sources.mono.path, 'test.names_wavs',
                         str(tmp_path / 'queue'), channels=channels, multichannel_wav_path=mc_path)
        design_and_queue(Session(AnalysisConfig(target_fs=FS)), 't_1', sources.mono.path, 'test.names_wavs',
                         str(tmp_path / 'queue'), channels=dict(list(channels.items())[:2]), multichannel_wav_path=mc_path)
    finally:
        unregister_designer('test.names_wavs')

    assert seen[0] == AudioSources.from_wavs(sources.mono.path, mc_path, list(channels))
    # two labels for a three-column file: no label may name another column, so the channels have no sources
    assert seen[1] == AudioSources.from_wavs(sources.mono.path)
