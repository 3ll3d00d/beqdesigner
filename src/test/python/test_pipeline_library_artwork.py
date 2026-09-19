'''Tests for automatic library-run artwork resolution (plan §3.1.3, tiers 2 and 3).'''
import os

import requests

from pipeline.library.artwork import resolve_art
from pipeline.library.source import LibraryItem


def _item(art_path=None):
    return LibraryItem(id='title-1', source_path='/media/title.mkv', display_name='Title', art_path=art_path)


def _fake_fetch(monkeypatch, calls, content=b'poster'):
    def fetch(poster_path, dest_dir=None, **kwargs):
        calls.append((poster_path, dest_dir))
        path = os.path.join(dest_dir, 'random123.jpg')
        with open(path, 'wb') as f:
            f.write(content)
        return path

    monkeypatch.setattr('pipeline.library.artwork.fetch_poster', fetch)


def test_library_art_is_used_in_place_and_wins_over_tmdb(tmp_path, monkeypatch):
    calls = []
    _fake_fetch(monkeypatch, calls)
    art = tmp_path / 'cover.jpg'
    art.write_bytes(b'x')

    assert resolve_art(_item(str(art)), {'poster': '/p.jpg'}, str(tmp_path / 'art')) == str(art)
    assert calls == []


def test_tmdb_poster_is_downloaded_to_a_fixed_name_in_the_item_dir(tmp_path, monkeypatch):
    calls = []
    _fake_fetch(monkeypatch, calls)
    art_dir = str(tmp_path / 'item')

    result = resolve_art(_item(), {'poster': '/p.jpg'}, art_dir)

    assert result == os.path.join(art_dir, 'poster.jpg')
    assert open(result, 'rb').read() == b'poster'
    assert os.listdir(art_dir) == ['poster.jpg']
    assert calls == [('/p.jpg', art_dir)]


def test_a_missing_library_art_file_falls_through_to_tmdb(tmp_path, monkeypatch):
    calls = []
    _fake_fetch(monkeypatch, calls)

    result = resolve_art(_item(str(tmp_path / 'gone.jpg')), {'poster': '/p.jpg'}, str(tmp_path / 'item'))

    assert result.endswith('poster.jpg')


def test_no_poster_or_no_art_dir_means_no_artwork(tmp_path, monkeypatch):
    calls = []
    _fake_fetch(monkeypatch, calls)

    assert resolve_art(_item(), {}, str(tmp_path)) is None
    assert resolve_art(_item(), {'poster': '/p.jpg'}, None) is None
    assert calls == []


def test_a_failed_download_is_no_artwork_not_an_error(tmp_path, monkeypatch):
    def failing(poster_path, dest_dir=None, **kwargs):
        raise requests.HTTPError('404')

    monkeypatch.setattr('pipeline.library.artwork.fetch_poster', failing)

    assert resolve_art(_item(), {'poster': '/p.jpg'}, str(tmp_path)) is None
