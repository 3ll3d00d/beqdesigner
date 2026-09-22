'''Tests for the Qt-free JRiver browse-node library source.'''
import asyncio
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from pipeline.library.jriver import BrowseNode, JRiverLibrarySource, LibraryField, list_browse_children, \
    list_library_fields, normalise_external_id_fields
from pipeline.library.artwork import resolve_art
from pipeline.library.pathmap import PathMapping


def _source(**kwargs):
    return JRiverLibrarySource('JRiver.local', 52199, 42, **kwargs)


def _art(source, row):
    ''' The poster the item would get at design time (a listing only names candidates; it touches no file). '''
    return resolve_art(source._map_row(_row(**row)), {}, None)


def _row(**overrides):
    row = {
        'Key': 1234,
        'Filename': '/media/Example.mkv',
        'Name': 'Example',
        'Year': '2024',
        'Media Type': 'Video',
        'Date Modified': 1720000000,
        'File Size': 123456,
        'IMDb ID': 'tt1234567',
        'TheMovieDB Movie ID': '123',
        'Image File': 'INTERNAL',
    }
    row.update(overrides)
    return row


@contextmanager
def _browse_server(rows, children=None, fields_xml=None):
    ''' children: {parent node id: {name: child id}} served as MCWS's XML for Browse/Children. '''
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
            elif self.path.startswith('/MCWS/v1/Library/Fields') and fields_xml is not None:
                body = fields_xml.encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'text/xml')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path.startswith('/MCWS/v1/Browse/Children') and children is not None:
                node = int(parse_qs(urlparse(self.path).query)['ID'][0])
                items = ''.join(f'<Item Name="{name}">{child}</Item>' for name, child in children.get(node, {}).items())
                body = f'<Response Status="OK">{items}</Response>'.encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'text/xml')
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


def test_an_unmapped_windows_path_carries_a_non_sensitive_mapping_diagnostic(monkeypatch):
    monkeypatch.setattr('pipeline.library.pathmap.os.name', 'posix')
    item = _source()._map_row(_row(Filename='W:\\Films\\Example.mkv'))
    assert item.source_path_problem == 'JRiver reported a Windows path with no local mapping; add one in Preferences > JRiver.'


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
    assert {'IMDb ID', 'TheMovieDB Movie ID', 'TMDb ID', 'IMDb Series ID', 'TheMovieDB Series ID'}.issubset(fields)
    assert {'Key', 'Name', 'Media Type', 'Series', 'Season', 'Episode'}.issubset(fields)


def test_maps_optional_fields_and_tv_metadata(tmp_path):
    artwork = tmp_path / 'poster.jpg'
    artwork.write_bytes(b'image')

    item = _source()._map_rows([_row(Name='', Year=None, **{
        'Series': 'Example Show', 'Season': '2', 'Episode': '3', 'Image File': str(artwork),
        'Date Modified': None, 'File Size': None, 'IMDb ID': '', 'TheMovieDB Movie ID': '',
    })])[0]

    assert item.display_name == 'Example Show S02E03'
    assert item.title == 'Example Show'  # the series, not the episode
    assert item.year is None
    assert item.kind == 'tv'
    assert (item.season, item.episodes) == ('2', (3,))
    assert item.meta == {}
    assert resolve_art(item, {}, None) == str(artwork)
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


def test_lists_the_children_of_a_browse_node():
    children = {-1: {'Video': '2', 'Audio': '1'}, 2: {'Movies': '21', 'Needs BEQ': '22'}}
    with _browse_server([], children) as (port, requests):
        root = list_browse_children('127.0.0.1', port)
        below_video = list_browse_children('127.0.0.1', port, 2)

    assert root == [BrowseNode(2, 'Video'), BrowseNode(1, 'Audio')]
    assert below_video == [BrowseNode(21, 'Movies'), BrowseNode(22, 'Needs BEQ')]
    assert requests[0].path == '/MCWS/v1/Browse/Children'
    assert parse_qs(requests[0].query)['ID'] == ['-1']
    assert parse_qs(requests[1].query)['ID'] == ['2']


def test_a_node_with_no_children_is_an_empty_list():
    with _browse_server([], {}) as (port, _):
        assert list_browse_children('127.0.0.1', port, 99) == []


def test_children_that_are_not_integer_ids_are_skipped_rather_than_failing():
    with _browse_server([], {-1: {'Video': '2', 'Odd': 'not-an-id', 'Empty': ''}}) as (port, _):
        assert list_browse_children('127.0.0.1', port) == [BrowseNode(2, 'Video')]


def test_listing_children_rejects_a_running_event_loop():
    async def call_from_loop():
        with pytest.raises(RuntimeError, match='active event loop'):
            list_browse_children('127.0.0.1', 1)

    asyncio.run(call_from_loop())


def test_the_year_is_read_from_the_display_name_jriver_actually_returns():
    # a live server answers a request for 'Year' with the key 'Date (year)'
    live_style = _row(**{'Date (year)': '2019'})
    del live_style['Year']

    assert _source()._map_row(live_style).year == '2019'
    assert _source()._map_row(_row(Year='2001', **{'Date (year)': '2019'})).year == '2001'  # requested name wins
    without = _row()
    del without['Year']
    assert _source()._map_row(without).year is None


@pytest.mark.parametrize('sub_type, extra, expected', [
    ('Movie', {}, 'movie'),
    ('TV Show', {'Series': 'S', 'Season': '1', 'Episode': '2'}, 'tv'),
    ('TV Show', {}, 'tv'),  # the sub type is authoritative even if the episode fields are blank
    ('Movie', {'Series': 'Lord of the Rings'}, 'movie'),  # a film franchise
    ('Movie', {'Series': 'Film Clips', 'Season': 'Bass', 'Episode': '1'}, 'movie'),  # stray episodic values
    ('Home Video', {}, 'movie'),
    ('', {'Series': 'S', 'Season': '1', 'Episode': '2'}, 'tv'),  # no sub type: fall back, needing all three
    ('', {'Series': 'S'}, 'movie'),
    ('', {'Season': '1', 'Episode': '2'}, 'movie'),
    ('', {}, 'movie'),
])
def test_kind_follows_the_media_sub_type_before_the_episodic_fields(sub_type, extra, expected):
    row = _row(**{'Media Sub Type': sub_type, **extra})

    assert _source()._map_row(row).kind == expected


def test_season_and_episode_are_only_kept_for_tv():
    tv = _source()._map_row(_row(**{'Media Sub Type': 'TV Show', 'Series': 'S', 'Season': '2', 'Episode': '3'}))
    film = _source()._map_row(_row(**{'Media Sub Type': 'Movie', 'Series': 'Film Clips', 'Season': 'Bass'}))

    assert (tv.season, tv.episodes) == ('2', (3,))
    assert (film.season, film.episodes) == (None, ())
    assert film.title == 'Example'  # a film keeps its own name, not its Series


@pytest.mark.parametrize('extra, expected_title, season, episodes', [
    ({'Series': 'Show', 'Name': 'The Sofa'}, 'Show', '1', (2,)),
    ({'Series': '', 'Name': 'The Sofa'}, 'The Sofa', '1', (2,)),  # no series: fall back to the name
    ({'Series': 'Show', 'Season': 'Bass'}, 'Show', None, ()),  # a non-numeric season is not a season
    ({'Series': 'Show', 'Episode': 'Pilot'}, 'Show', '1', ()),  # nor an episode
    ({'Series': 'Show', 'Season': ''}, 'Show', None, ()),
])
def test_tv_title_season_and_episodes_come_from_the_series_and_numeric_fields(extra, expected_title, season, episodes):
    row = _row(**{'Media Sub Type': 'TV Show', 'Season': '1', 'Episode': '2', **extra})

    item = _source()._map_row(row)

    assert item.title == expected_title
    assert item.season == season
    assert item.episodes == episodes


def test_an_episode_is_displayed_with_its_series_season_and_number():
    row = _row(**{'Media Sub Type': 'TV Show', 'Series': 'Eastbound', 'Season': '1', 'Episode': '11', 'Name': 'Chapter 11'})

    assert _source()._map_row(row).display_name == 'Eastbound S01E11 Chapter 11'


# --- path mapping: JRiver reports the *server's* (Windows) paths ---------------------------------------------

WINDOWS_ROW = {'Filename': 'W:\\Films\\Action\\Die Hard (1988).mkv', 'Name': 'Die Hard', 'Image File': 'Die_Hard.jpg'}


def test_a_windows_path_is_translated_to_the_local_one(tmp_path):
    source = _source(path_mappings=[PathMapping('W:\\Films', str(tmp_path))])

    item = source._map_row(_row(**WINDOWS_ROW))

    assert item.source_path == str(tmp_path / 'Action' / 'Die Hard (1988).mkv')


def test_paths_are_passed_through_when_no_mapping_applies():
    item = _source(path_mappings=[PathMapping('X:\\Other', '/mnt/other')])._map_row(_row(**WINDOWS_ROW))

    assert item.source_path == WINDOWS_ROW['Filename']
    assert _source()._map_row(_row(**WINDOWS_ROW)).source_path == WINDOWS_ROW['Filename']


def test_the_display_name_comes_from_a_windows_path_too():
    item = _source()._map_row(_row(**{**WINDOWS_ROW, 'Name': ''}))

    assert item.display_name == 'Die Hard (1988).mkv'


def test_the_id_does_not_depend_on_the_mapping():
    mapped = _source(path_mappings=[PathMapping('W:\\Films', '/mnt/films')])._map_row(_row(**WINDOWS_ROW))
    unmapped = _source()._map_row(_row(**WINDOWS_ROW))

    assert mapped.id == unmapped.id
    assert mapped.source_path != unmapped.source_path


def test_a_bare_image_file_name_is_found_beside_the_translated_media_file(tmp_path):
    (tmp_path / 'Action').mkdir()
    cover = tmp_path / 'Action' / 'Die_Hard.jpg'
    cover.write_bytes(b'image')
    source = _source(path_mappings=[PathMapping('W:\\Films', str(tmp_path))])

    assert _art(source, WINDOWS_ROW) == str(cover)
    assert _art(source, {**WINDOWS_ROW, 'Image File': 'missing.jpg'}) is None


def test_a_bare_image_file_name_is_not_found_without_a_mapping():
    assert _art(_source(), WINDOWS_ROW) is None


def test_an_absolute_windows_image_path_is_mapped_too(tmp_path):
    (tmp_path / 'art').mkdir()
    cover = tmp_path / 'art' / 'cover.jpg'
    cover.write_bytes(b'image')
    source = _source(path_mappings=[PathMapping('W:\\Films', str(tmp_path))])

    assert _art(source, {**WINDOWS_ROW, 'Image File': 'W:\\Films\\art\\cover.jpg'}) == str(cover)


def test_an_internal_image_is_never_a_file(tmp_path):
    (tmp_path / 'Action').mkdir()
    (tmp_path / 'Action' / 'INTERNAL').write_bytes(b'x')  # even a file that happens to share the marker's name
    source = _source(path_mappings=[PathMapping('W:\\Films', str(tmp_path))])

    assert _art(source, {**WINDOWS_ROW, 'Image File': 'INTERNAL'}) is None


# --- which fields hold the external ids (they differ between libraries) ----------------------------------------

def test_the_defaults_are_the_field_names_a_real_library_uses():
    ids = normalise_external_id_fields(None)

    assert ids['movie'] == {'imdb': ('IMDb ID',), 'tmdb': ('TheMovieDB Movie ID', 'TMDb ID'), 'tvdb': ()}
    assert ids['tv'] == {'imdb': ('IMDb Series ID',), 'tmdb': ('TheMovieDB Series ID',), 'tvdb': ()}


def test_a_tv_title_reads_its_series_ids_not_the_episodes_own():
    row = _row(**{'Media Sub Type': 'TV Show', 'IMDb ID': 'tt-episode', 'IMDb Series ID': 'tt-series',
                  'TheMovieDB Series ID': '66292'})

    assert _source()._map_row(row).external_ids == {'imdb': 'tt-series', 'tmdb': '66292'}


def test_the_first_field_with_a_value_wins_and_an_unset_numeric_id_does_not_count():
    row = _row(**{'TheMovieDB Movie ID': '0', 'TMDb ID': '702'})

    assert _source()._map_row(row).external_ids['tmdb'] == '702'
    assert 'tmdb' not in _source()._map_row(_row(**{'TheMovieDB Movie ID': '0'})).external_ids


def test_fields_can_be_overridden_per_kind_keeping_the_other_defaults():
    source = _source(external_id_fields={'movie': {'imdb': ['My IMDb'], 'tmdb': 'My TMDb'}})
    row = _row(**{'My IMDb': 'tt9', 'My TMDb': '9', 'IMDb ID': 'tt-default'})

    assert source._map_row(row).external_ids == {'imdb': 'tt9', 'tmdb': '9'}
    assert source.external_id_fields['tv'] == normalise_external_id_fields(None)['tv']


def test_a_flat_mapping_applies_to_both_kinds():
    source = _source(external_id_fields={'imdb': ['Custom IMDb']})
    film = source._map_row(_row(**{'Media Sub Type': 'Movie', 'Custom IMDb': 'tt1'}))
    show = source._map_row(_row(**{'Media Sub Type': 'TV Show', 'Custom IMDb': 'tt2'}))

    assert film.external_ids['imdb'] == 'tt1'
    assert show.external_ids['imdb'] == 'tt2'


def test_an_empty_list_switches_an_identifier_off():
    source = _source(external_id_fields={'movie': {'tmdb': []}})

    assert 'tmdb' not in source._map_row(_row(**{'TheMovieDB Movie ID': '123'})).external_ids
    assert 'imdb' in source._map_row(_row()).external_ids


def test_an_unknown_identifier_is_rejected():
    with pytest.raises(ValueError, match="unknown external id 'other'"):
        normalise_external_id_fields({'movie': {'other': ['Some ID']}})


def test_the_configured_fields_are_what_gets_requested_without_repeats():
    source = _source(external_id_fields={'movie': {'imdb': ['My IMDb', 'IMDb ID']}})

    fields = source.requested_fields

    assert 'My IMDb' in fields
    assert len(fields) == len(set(fields))
    assert set(JRiverLibrarySource.FIELDS).issubset(fields)


def test_lists_the_library_fields_the_server_defines():
    xml = ('<Response Status="OK"><Fields>'
           '<Field Name="IMDb ID" DataType="String" EditType="Standard" DisplayName="IMDb ID"/>'
           '<Field Name="Date (year)" DataType="Integer" EditType="Standard" DisplayName="Year"/>'
           '</Fields></Response>')
    with _browse_server([], fields_xml=xml) as (port, requests):
        fields = list_library_fields('127.0.0.1', port)

    assert fields == [LibraryField('IMDb ID', 'IMDb ID', 'String'), LibraryField('Date (year)', 'Year', 'Integer')]
    assert requests[0].path == '/MCWS/v1/Library/Fields'


def test_listing_library_fields_rejects_a_running_event_loop():
    async def call_from_loop():
        with pytest.raises(RuntimeError, match='active event loop'):
            list_library_fields('127.0.0.1', 1)

    asyncio.run(call_from_loop())


# --- optical discs: JRiver reports a pseudo-file, not the disc folder --------------------------------------------

@pytest.mark.parametrize('filename, root', [
    ('W:\\28_Years_Later\\BDMV\\index.bluray;1', 'W:\\28_Years_Later'),
    ('W:\\Films\\48 Hrs. (Remastered)\\BDMV\\index.bluray;2', 'W:\\Films\\48 Hrs. (Remastered)'),
    ('/mnt/films/Disc/BDMV/INDEX.BLURAY;1', '/mnt/films/Disc'),
    ('W:\\BDMV\\index.bluray;1', 'W:\\'),  # a disc at the drive's top level
])
def test_a_bluray_pseudo_file_becomes_the_disc_folder(filename, root):
    assert _source()._map_row(_row(Filename=filename)).source_path == root


def test_the_disc_folder_is_then_translated_like_any_other_path(tmp_path):
    source = _source(path_mappings=[PathMapping('W:\\', str(tmp_path))])

    item = source._map_row(_row(Filename='W:\\28_Years_Later\\BDMV\\index.bluray;1'))

    assert item.source_path == str(tmp_path / '28_Years_Later')


@pytest.mark.parametrize('filename', [
    'W:\\Films\\a.mkv',
    'W:\\Disc\\BDMV\\PLAYLIST\\index.bluray;1',  # names a playlist on the disc: which title is unknown
    '\\BDMV\\index.bluray;1',  # nothing above BDMV to be the disc
    'W:\\Disc\\BDMV\\STREAM\\00000.m2ts',
])
def test_anything_else_is_left_as_reported(filename):
    assert _source()._map_row(_row(Filename=filename)).source_path == filename


# --- a listing never touches the media (design.md §12.5) ---------------------------------------------------------

def _fail_on_disk(monkeypatch, only_under=None):
    import os
    real_stat, real_isfile = os.stat, os.path.isfile

    def guard(real):
        def checked(path, *args, **kwargs):
            if only_under is None or str(path).startswith(only_under):
                raise AssertionError(f'a JRiver listing touched the disk: {path!r}')
            return real(path, *args, **kwargs)
        return checked

    monkeypatch.setattr(os, 'stat', guard(real_stat))
    monkeypatch.setattr(os.path, 'isfile', guard(real_isfile))
    monkeypatch.setattr(os.path, 'isdir', guard(os.path.isdir))
    monkeypatch.setattr(os.path, 'exists', guard(os.path.exists))


def test_mapping_rows_performs_no_stat_or_isfile_on_media_or_artwork(monkeypatch):
    rows = [_row(Key=i, Filename=f'W:\\Films\\{i}.mkv', **{'Image File': f'{i}.jpg'}) for i in range(50)]
    rows.append(_row(Key=99, Filename='W:\\Films\\x\\BDMV\\index.bluray;1', **{'Image File': 'W:\\Films\\art.jpg'}))
    source = _source(path_mappings=[PathMapping('W:\\Films', '/mnt/films')])
    _fail_on_disk(monkeypatch)

    items = source._map_rows(rows)

    assert len(items) == 51
    assert items[0].art_path is None
    assert items[0].art_candidates == ('/mnt/films/0.jpg',)


def test_listing_a_library_over_http_touches_no_media_file(monkeypatch):
    rows = [_row(Key=i, Filename=f'/mnt/films/{i}.mkv', **{'Image File': f'{i}.jpg'}) for i in range(5)]
    with _browse_server(rows) as (port, _):
        _fail_on_disk(monkeypatch, only_under='/mnt/films')
        items = list(JRiverLibrarySource('127.0.0.1', port, 42).list_items())

    assert len(items) == 5


def test_artwork_is_resolved_at_design_time_from_the_candidates(tmp_path):
    cover = tmp_path / 'a.jpg'
    cover.write_bytes(b'image')
    item = _source()._map_row(_row(Filename=str(tmp_path / 'a.mkv'), **{'Image File': 'a.jpg'}))

    assert item.art_path is None and item.art_candidates == (str(cover),)
    assert resolve_art(item, {}, None) == str(cover)
    cover.unlink()
    assert resolve_art(item, {}, None) is None
