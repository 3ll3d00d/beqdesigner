'''
Phase 3 (item 8) of design/pipeline-implementation-plan.md: poster fetch.
The TMDB HTTP call is mocked -- no network access required.
'''
import os

import pytest
import requests

from pipeline.publish.art import TMDB_IMAGE_BASE_URL, fetch_poster, poster_url


def test_poster_url_combines_base_size_and_fragment():
    assert poster_url('/abc.jpg') == f"{TMDB_IMAGE_BASE_URL}original/abc.jpg"
    assert poster_url('/abc.jpg', size='w500') == f"{TMDB_IMAGE_BASE_URL}w500/abc.jpg"


def test_poster_url_tolerates_slash_variations():
    assert poster_url('abc.jpg', base_url='https://example.test/p') == 'https://example.test/p/original/abc.jpg'


class _FakeResponse:
    def __init__(self, content, status_code=200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


def test_fetch_poster_writes_downloaded_bytes(monkeypatch, tmp_path):
    monkeypatch.setattr('requests.get', lambda url: _FakeResponse(b'fake-jpeg-bytes'))

    path = fetch_poster('/abc.jpg', dest_dir=str(tmp_path))

    assert os.path.dirname(path) == str(tmp_path)
    assert path.endswith('.jpg')
    with open(path, 'rb') as f:
        assert f.read() == b'fake-jpeg-bytes'


def test_fetch_poster_raises_and_cleans_up_on_http_error(monkeypatch, tmp_path):
    monkeypatch.setattr('requests.get', lambda url: _FakeResponse(b'', status_code=404))

    with pytest.raises(requests.HTTPError):
        fetch_poster('/missing.jpg', dest_dir=str(tmp_path))

    assert os.listdir(str(tmp_path)) == []


def test_pipeline_publish_art_module_has_no_qtpy_import():
    import ast
    import pathlib
    source = (pathlib.Path(__file__).parent / '..' / '..' / 'main' / 'python' / 'pipeline' / 'publish' / 'art.py').resolve()
    tree = ast.parse(source.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(n.name.startswith('qtpy') for n in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith('qtpy')
