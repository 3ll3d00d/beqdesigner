'''Tests for the Qt-free JRiver browse-node library source.'''
import asyncio
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from pipeline.library.jriver import JRiverLibrarySource


def _source(**kwargs):
    return JRiverLibrarySource('JRiver.local', 52199, 42, **kwargs)


def _row(**overrides):
    row = {
        'Key': 1234,
        'Filename': '/media/Example.mkv',
        'Name': 'Example',
        'Year': '2024',
        'Media Type': 'Video',
        'Date Modified': 1720000000,
        'File Size': 123456,
        'IMDB': 'tt1234567',
        'TheMovieDB': '123',
        'Image File': 'INTERNAL',
    }
    row.update(overrides)
    return row


@contextmanager
def _browse_server(rows):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(urlparse(self.path))
            if self.path.startswith('/MCWS/v1/Browse/Files'):
                body = json.dumps(rows).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port, requests
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_maps_a_browse_file_row():
    item = _source()._map_rows([_row()])[0]

    assert item.id.startswith('jriver-')
    assert item.id.endswith('-1234')
    assert item.source_path == '/media/Example.mkv'
    assert item.display_name == item.title == 'Example'
    assert item.year == '2024'
    assert item.kind == 'movie'
    assert item.external_ids == {'imdb': 'tt1234567', 'tmdb': '123'}
    assert item.art_path is None
    assert item.fingerprint == '{"date_modified": "1720000000", "file_size": "123456"}'


def test_lists_files_from_the_configured_browse_node():
    with _browse_server([_row()]) as (port, requests):
        items = list(JRiverLibrarySource('127.0.0.1', port, 42).list_items())

    assert len(items) == 1
    assert items[0].title == 'Example'
    assert len(requests) == 1
    request = requests[0]
    query = parse_qs(request.query)
    assert request.path == '/MCWS/v1/Browse/Files'
    assert query['ID'] == ['42']
    assert query['Action'] == ['JSON']
    fields = query['Fields'][0].split(',')
    assert set(JRiverLibrarySource.FIELDS).issubset(fields)
    assert {'Key', 'Name', 'Media Type', 'Series', 'Season', 'Episode'}.issubset(fields)


def test_maps_optional_fields_and_tv_metadata(tmp_path):
    artwork = tmp_path / 'poster.jpg'
    artwork.write_bytes(b'image')

    item = _source()._map_rows([_row(Name='', Year=None, **{
        'Series': 'Example Show', 'Season': '2', 'Episode': '3', 'Image File': str(artwork),
        'Date Modified': None, 'File Size': None, 'IMDB': '', 'TheMovieDB': '',
    })])[0]

    assert item.display_name == 'Example.mkv'
    assert item.title is None
    assert item.year is None
    assert item.kind == 'tv'
    assert item.meta == {'season': '2'}
    assert item.art_path == str(artwork)
    assert item.external_ids == {}
    assert item.fingerprint == ''


def test_id_is_stable_across_filename_changes_and_distinct_per_server():
    first = _source()._map_rows([_row()])[0]
    renamed = _source()._map_rows([_row(Filename='/media/Renamed.mkv')])[0]
    other_server = JRiverLibrarySource('other.local', 52199, 42)._map_rows([_row()])[0]

    assert renamed.id == first.id
    assert other_server.id != first.id


def test_duplicate_key_is_deduplicated_but_conflicting_path_fails():
    source = _source()
    assert len(source._map_rows([_row(), _row()])) == 1

    with pytest.raises(ValueError, match='conflicting filenames'):
        source._map_rows([_row(), _row(Filename='/media/Other.mkv')])


def test_rejects_source_query_and_active_event_loop():
    source = _source()
    with pytest.raises(TypeError, match='accepts no query'):
        source.list_items(name='Example')

    async def call_from_loop():
        with pytest.raises(RuntimeError, match='active event loop'):
            source.list_items()

    asyncio.run(call_from_loop())
