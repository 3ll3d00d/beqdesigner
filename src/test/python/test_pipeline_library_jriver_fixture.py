'''
TODO E1: the JRiver MCWS responses of a real server (captured and sanitised by fixtures/jriver/capture.py, see
fixtures/jriver/README.md) replayed by a fake server, so the field aliases, the browse tree, FriendlyName and item identity
are checked against what MC actually says rather than what a test assumed it says.
'''
import json
import pathlib
import re
import threading
from collections import Counter
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest

from pipeline.library.jriver import BrowseNode, JRiverLibrarySource, list_browse_children
from pipeline.library.pathmap import PathMapping

FIXTURE = pathlib.Path(__file__).parent / 'fixtures' / 'jriver'
CAPTURE = json.loads((FIXTURE / 'capture.json').read_text())
ROWS = json.loads((FIXTURE / 'files.json').read_text(encoding='utf-8'))
NODE = CAPTURE['request']['browse_node_id']
TMDB = {'movie': {'tmdb': ['TheMovieDB Movie ID']}}   # as the captured profile configured it
MAPPINGS = (PathMapping('W:\\', '/media/films'),)


@contextmanager
def _replay(children=None):
    ''' A fake MCWS answering from the fixture: (host, port). `children` replaces a Browse/Children body by node. '''
    asked = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlsplit(self.path)
            query = {k: v[0] for k, v in parse_qs(url.query).items()}
            asked.append((url.path, query))
            if url.path == '/MCWS/v1/Alive':
                body, kind = (FIXTURE / 'alive.xml').read_bytes(), 'text/xml'
            elif url.path == '/MCWS/v1/Authenticate':
                body, kind = b'<Response Status="OK"><Item Name="Token">token</Item></Response>', 'text/xml'
            elif url.path == '/MCWS/v1/Library/Fields':
                body, kind = (FIXTURE / 'fields.xml').read_bytes(), 'text/xml'
            elif url.path == '/MCWS/v1/Browse/Children':
                node = query.get('ID')
                if children and node in children:
                    body = children[node].encode('utf-8')
                else:
                    path = FIXTURE / f'children_{node}.xml'
                    body = path.read_bytes() if path.is_file() else b'<Response Status="OK"/>'
                kind = 'text/xml'
            elif url.path == '/MCWS/v1/Browse/Files' and query.get('ID') == str(NODE):
                body, kind = (FIXTURE / 'files.json').read_bytes(), 'application/json'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.02}, daemon=True)
    thread.start()
    try:
        yield '127.0.0.1', server.server_port, asked
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture(scope='module')
def items():
    with _replay() as (host, port, asked):
        listed = JRiverLibrarySource(host, port, NODE, external_id_fields=TMDB, path_mappings=MAPPINGS).list_items()
        files = [query for path, query in asked if path == '/MCWS/v1/Browse/Files']
    return listed, files


def test_the_fixture_is_sanitised():
    text = ''.join(p.read_text(encoding='utf-8') for p in FIXTURE.iterdir() if p.suffix in ('.xml', '.json'))
    assert 'MEDIA-SERVER' in text and 'ACCESSKEY' in text
    assert 'Description' not in json.dumps(ROWS)
    assert 'DISPLAY#DEVICE' in text and not re.search(r'&UID\d', text)   # a display's hardware id
    assert all(row['Filename'][:3].upper() == 'W:\\' for row in ROWS)   # nothing outside the media share


def test_one_request_asks_for_every_field_the_source_reads(items):
    _, files = items
    assert len(files) == 1 and files[0]['Action'] == 'JSON'
    assert files[0]['Fields'].split(',') == CAPTURE['request']['fields']


def test_every_captured_row_becomes_an_item_with_a_stable_id(items):
    listed, _ = items
    keys = [str(row['Key']) for row in ROWS]
    assert len(listed) == len(ROWS) == len(set(keys))
    assert [item.id.rpartition('-')[2] for item in listed] == keys   # the id is the server's Key: stable across scans
    assert len({item.id.split('-')[1] for item in listed}) == 1      # and scoped to this server


def test_year_is_read_from_the_name_mc_answers_with(items):
    ''' Asked for `Year`, MC answers `Date (year)` (and an unset one is absent, not 0). '''
    listed, _ = items
    assert 'Year' not in set().union(*ROWS)
    by_key = {item.id.rpartition('-')[2]: item for item in listed}
    for row in ROWS:
        year = row.get('Date (year)')
        assert by_key[str(row['Key'])].year == (str(year) if year else None)


def test_external_ids_come_from_the_configured_fields_and_absent_ones_are_left_out(items):
    listed, _ = items
    by_key = {item.id.rpartition('-')[2]: item for item in listed}
    with_tmdb = [row for row in ROWS if row.get('TheMovieDB Movie ID')]
    assert with_tmdb and len(with_tmdb) < len(ROWS)
    for row in ROWS:
        ids = by_key[str(row['Key'])].external_ids
        assert ids.get('tmdb') == (str(row['TheMovieDB Movie ID']) if row.get('TheMovieDB Movie ID') else None)


def test_paths_map_whatever_the_case_of_the_drive_and_discs_are_their_folders(items):
    listed, _ = items
    by_key = {item.id.rpartition('-')[2]: item for item in listed}
    assert any(row['Filename'].startswith('w:\\') for row in ROWS)   # MC reports some in lower case
    for row in ROWS:
        item = by_key[str(row['Key'])]
        assert item.source_path.startswith('/media/films/') and item.source_path_problem is None
        name = row['Filename'].lower()
        if name.endswith('index.bdmv') or (';' in name and '\\playlist\\' not in name):
            assert not item.source_path.lower().endswith(('index.bdmv', '.bluray;1'))   # the disc's own folder


def test_artwork_is_a_file_name_beside_the_media(items):
    ''' No `INTERNAL` artwork in the captured library; a bare name is looked for beside the media file. '''
    listed, _ = items
    by_key = {item.id.rpartition('-')[2]: item for item in listed}
    for row in ROWS:
        candidates = by_key[str(row['Key'])].art_candidates
        if row.get('Image File'):
            assert any(c.endswith('/' + row['Image File']) for c in candidates)
        else:
            assert candidates == ()


def test_every_audio_stream_mc_lists_is_a_choice(items):
    listed, _ = items
    by_key = {item.id.rpartition('-')[2]: item for item in listed}
    multi = [row for row in ROWS if int(row.get('Audio Streams') or 0) > 1]
    assert multi
    for row in ROWS:
        details = by_key[str(row['Key'])].audio_stream_details
        assert len(details) == int(row.get('Audio Streams') or 0)
        assert [d['codec'] for d in details] == ([c.strip() for c in row['Audio Codec'].split(';')]
                                                if row.get('Audio Codec') else [])


def test_the_browse_tree_is_read_as_mc_sends_it():
    with _replay() as (host, port, _):
        root = list_browse_children(host, port)
        video = next(node for node in root if node.name == 'Video')
        below = list_browse_children(host, port, video.id)
    assert [node.name for node in root][:3] == ['Player', 'Audio', 'Video']
    assert BrowseNode(NODE, 'Movies') in below and len(below) == len({n.id for n in below})


def test_two_nodes_with_the_same_name_keep_their_own_ids():
    ''' None in the captured library, so made by repeating one of its own entries under another id. '''
    video = (FIXTURE / 'children_3.xml').read_text(encoding='utf-8')
    entry = next(line for line in video.splitlines() if '>1004<' in line)
    duplicated = video.replace(entry, entry + '\n' + entry.replace('>1004<', '>2004<'))
    with _replay(children={'3': duplicated}) as (host, port, _):
        below = list_browse_children(host, port, 3)
    movies = [node for node in below if node.name == 'Movies']
    assert movies == [BrowseNode(1004, 'Movies'), BrowseNode(2004, 'Movies')]
    assert Counter(n.name for n in below)['Movies'] == 2


def test_the_server_names_itself_by_its_friendly_name():
    from model.jriver.mcws import MediaServer
    with _replay() as (host, port, _):
        server = MediaServer(f'{host}:{port}')
        server.authenticate()
    assert server.friendly_name == 'MEDIA-SERVER'
