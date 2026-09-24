'''
Phase 1 of design/http-designer-binding-plan.md: http_designer()'s wire
format and error handling. One real round trip against an actual HTTP
server (proves the JSON+base64 encode/decode genuinely works, not just
mocked); the rest of the error paths are exercised via a monkeypatched
requests.post, which is faster and deterministic.
'''
import base64
import http.server
import json
import threading

import numpy as np
import pytest

from pipeline.designer.contract import DesignRequest, build_request
from pipeline.designer.http_binding import HttpDesignerError, http_designer

_DECLINE_SENTINEL_FS = 999


class _EchoHandler(http.server.BaseHTTPRequestHandler):
    ''' Decodes the request body and replies with a response shaped to prove it was decoded correctly. '''

    def do_POST(self):
        length = int(self.headers['Content-Length'])
        body = json.loads(self.rfile.read(length))
        mono_mix = np.frombuffer(base64.b64decode(body['mono_mix']['data_base64']), dtype='<f8')
        if body['fs'] == _DECLINE_SENTINEL_FS:
            response = {'contract_version': '1.0', 'decline_reason': 'no_rolloff_detected',
                       'decline_message': 'nothing to correct'}
        else:
            response = {
                'contract_version': '1.0',
                'candidates': [{
                    'filters': [{'type': 'low_shelf', 'freq_hz': 18.0, 'gain_db': 4.5, 'q': 0.7}],
                    'confidence': 0.9,
                    'mv_adjust_db': float(mono_mix.max()),  # proves mono_mix round-tripped intact
                    'method': 'fitted',
                    'gain_reduction_db': -1.5,
                    'residual_band_hz': [5.0, 200.0],
                    'commentary': {'received_channels': 'yes' if body.get('channels') else 'no',
                                  'received_bass_management': 'yes' if body.get('bass_management') else 'no'},
                }],
            }
        payload = json.dumps(response).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass  # silence default request logging to stderr


@pytest.fixture
def echo_server():
    server = http.server.HTTPServer(('127.0.0.1', 0), _EchoHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/design"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _request(mono_mix=None, channels=None, bass_management=None, fs=1000):
    mono_mix = np.array([0.1, -0.9, 0.5, 0.25], dtype=np.float64) if mono_mix is None else mono_mix
    return build_request(mono_mix=mono_mix, fs=fs, channels=channels, bass_management=bass_management)


def test_round_trip_against_a_real_server(echo_server):
    response = http_designer(echo_server)(_request())

    assert response.contract_version == '1.0'
    assert len(response.candidates) == 1
    candidate = response.candidates[0]
    assert candidate.mv_adjust_db == 0.5  # max of the mono_mix sent
    assert candidate.gain_reduction_db == -1.5
    assert candidate.residual_band_hz == (5.0, 200.0)
    assert candidate.commentary == {'received_channels': 'no', 'received_bass_management': 'no'}
    assert candidate.filters[0].type == 'low_shelf'
    assert candidate.filters[0].freq_hz == 18.0


def test_channels_and_bass_management_are_sent(echo_server):
    request = _request(channels={'L': np.array([1.0, 2.0, 3.0, 4.0]),
                                 'LFE': np.array([3.0, 4.0, 5.0, 6.0])},
                       bass_management={'lpf_fs': 80.0, 'lpf_position': 'Before', 'headroom_type': 'WCS',
                                       'clip_before': False, 'clip_after': False})

    response = http_designer(echo_server)(request)

    assert response.candidates[0].commentary == {'received_channels': 'yes', 'received_bass_management': 'yes'}


def test_decline_round_trips(echo_server):
    request = _request(fs=_DECLINE_SENTINEL_FS)

    response = http_designer(echo_server)(request)

    assert response.decline_reason == 'no_rolloff_detected'
    assert response.decline_message == 'nothing to correct'
    assert response.candidates is None


# --- error paths (monkeypatched requests.post) -----------------------------

class _FakeResponse:
    def __init__(self, status_ok=True, json_value=None, json_raises=False):
        self.__status_ok = status_ok
        self.__json_value = json_value
        self.__json_raises = json_raises

    def raise_for_status(self):
        if not self.__status_ok:
            import requests
            raise requests.HTTPError('500 Server Error')

    def json(self):
        if self.__json_raises:
            raise ValueError('not json')
        return self.__json_value


def test_connection_failure_raises_http_designer_error(monkeypatch):
    import requests

    def fake_post(*args, **kwargs):
        raise requests.ConnectionError('refused')

    monkeypatch.setattr('pipeline.designer.http_binding.requests.post', fake_post)

    with pytest.raises(HttpDesignerError, match='POST'):
        http_designer('http://127.0.0.1:1/design')(_request())


def test_non_2xx_status_raises_http_designer_error(monkeypatch):
    monkeypatch.setattr('pipeline.designer.http_binding.requests.post',
                        lambda *a, **k: _FakeResponse(status_ok=False))

    with pytest.raises(HttpDesignerError):
        http_designer('http://example.invalid/design')(_request())


def test_malformed_json_body_raises_http_designer_error(monkeypatch):
    monkeypatch.setattr('pipeline.designer.http_binding.requests.post',
                        lambda *a, **k: _FakeResponse(json_raises=True))

    with pytest.raises(HttpDesignerError, match='JSON'):
        http_designer('http://example.invalid/design')(_request())


def test_malformed_candidate_shape_raises_http_designer_error(monkeypatch):
    body = {'contract_version': '1.0', 'candidates': [{'filters': []}]}  # missing confidence/mv_adjust_db/method
    monkeypatch.setattr('pipeline.designer.http_binding.requests.post',
                        lambda *a, **k: _FakeResponse(json_value=body))

    with pytest.raises(HttpDesignerError, match="DesignResponse's shape"):
        http_designer('http://example.invalid/design')(_request())


def test_unsupported_array_dtype_rejected():
    from pipeline.designer.http_binding import _ndarray_from_json
    with pytest.raises(HttpDesignerError, match='dtype'):
        _ndarray_from_json({'dtype': 'float32', 'shape': [1], 'data_base64': 'AAAAAA=='})


def test_pipeline_designer_http_binding_has_no_qtpy_import():
    import ast
    import pathlib
    source = (pathlib.Path(__file__).parent / '..' / '..' / 'main' / 'python' / 'pipeline' / 'designer' /
             'http_binding.py').resolve()
    tree = ast.parse(source.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(n.name.startswith('qtpy') for n in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith('qtpy')


def test_declared_designers_are_registered_from_a_configs_designers_section(monkeypatch):
    from pipeline.designer import http_binding
    from pipeline.designer.registry import get_designer, unregister_designer
    made = []
    monkeypatch.setattr(http_binding, 'http_designer',
                        lambda url, timeout, headers: made.append((url, timeout, headers)) or (lambda request: None))

    names = http_binding.register_declared_designers({'plain.d': 'http://a/d', 'full.d': {
        'url': 'http://b/d', 'timeout': 30, 'headers': {'X': 'y'}}})
    try:
        assert names == ['plain.d', 'full.d']
        assert made == [('http://a/d', 300.0, None), ('http://b/d', 30.0, {'X': 'y'})]
        assert get_designer('plain.d') is not None and get_designer('full.d') is not None
    finally:
        unregister_designer('plain.d')
        unregister_designer('full.d')
    assert http_binding.register_declared_designers(None) == []


def test_a_declared_designer_without_a_url_is_refused():
    from pipeline.designer import http_binding
    with pytest.raises(ValueError, match="designer 'bad' needs a url"):
        http_binding.register_declared_designers({'bad': {'timeout': 3}})
