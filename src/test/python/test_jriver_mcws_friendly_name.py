'''MediaServer reads the FriendlyName /Alive reports -- against a local fake, never a real server.'''
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from model.jriver.mcws import MCWSError, MediaServer


@contextmanager
def _fake_mc(alive_items):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith('/MCWS/v1/Authenticate'):
                body = '<Response Status="OK"><Item Name="Token">tok</Item></Response>'
            elif self.path.startswith('/MCWS/v1/Alive'):
                items = ''.join(f'<Item Name="{k}">{v}</Item>' for k, v in alive_items.items())
                body = f'<Response Status="OK">{items}</Response>'
            else:
                self.send_error(404)
                return
            data = body.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/xml')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'127.0.0.1:{server.server_port}'
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_the_friendly_name_is_read_from_alive():
    with _fake_mc({'ProgramVersion': '35.0.39', 'FriendlyName': '  Cinema PC '}) as endpoint:
        server = MediaServer(endpoint)
        assert server.friendly_name is None  # nothing is known before authenticating

        server.authenticate()

    assert server.friendly_name == 'Cinema PC'


@pytest.mark.parametrize('items', [
    {'ProgramVersion': '35.0.39'},
    {'ProgramVersion': '35.0.39', 'FriendlyName': ''},
    {'ProgramVersion': '35.0.39', 'FriendlyName': '   '},
])
def test_a_server_without_a_usable_name_has_none(items):
    with _fake_mc(items) as endpoint:
        server = MediaServer(endpoint)
        server.authenticate()

    assert server.friendly_name is None


def test_authentication_still_needs_a_program_version():
    with _fake_mc({'FriendlyName': 'Cinema PC'}) as endpoint:
        with pytest.raises(MCWSError, match='No version'):
            MediaServer(endpoint).authenticate()
