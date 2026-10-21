'''
pipeline/library/artwork_edit.py -- a title's artwork chosen by a person, shared by the title page and the service
(design/web-parity.md WP4): only a complete png or jpeg is used, and a download or upload lands only in the art cache.
'''
import io
import os

import pytest
from PIL import Image

from pipeline.library import artwork_edit as art
from pipeline.review import read_entry
from review_entry_fixture import write_entry


def _image(kind='PNG', size=(8, 8)) -> bytes:
    out = io.BytesIO()
    Image.new('RGB', size, 'blue').save(out, format=kind)
    return out.getvalue()


@pytest.mark.parametrize('kind, extension', [('PNG', '.png'), ('JPEG', '.jpg')])
def test_a_png_or_jpeg_is_used(kind, extension):
    assert art.image_extension(_image(kind)) == extension


@pytest.mark.parametrize('data, says', [(_image('GIF'), 'not a png or jpeg'), (b'<html></html>', 'not a png or jpeg'),
                                        (_image('PNG', (64, 64))[:-40], 'damaged or incomplete')])
def test_anything_else_is_refused_with_a_sentence(data, says):
    with pytest.raises(art.ArtworkError, match=says):
        art.image_extension(data)


def test_a_local_file_is_checked_the_same_way(tmp_path):
    good, bad = tmp_path / 'p.jpg', tmp_path / 'x.png'
    good.write_bytes(_image('JPEG'))
    bad.write_bytes(b'nope')
    art.check_local_image(str(good))
    with pytest.raises(art.ArtworkError, match='that file is not a png or jpeg'):
        art.check_local_image(str(bad))
    with pytest.raises(art.ArtworkError, match='is not a file'):
        art.check_local_image(str(tmp_path / 'missing.png'))


def test_an_image_is_stored_in_the_art_cache_replacing_the_last(tmp_path):
    first = art.store_image(_image('PNG'), str(tmp_path), '../a b')
    assert os.path.dirname(first) == str(tmp_path / art.ART_CACHE) and os.path.basename(first) == '_a_b.png'
    again = art.store_image(_image('PNG', (2, 2)), str(tmp_path), '../a b')
    assert again == first and Image.open(again).size == (2, 2)


def test_an_image_too_big_is_refused_before_it_is_read(tmp_path, monkeypatch):
    monkeypatch.setattr(art, 'MAX_DOWNLOAD_BYTES', 10)
    with pytest.raises(art.ArtworkError, match='bigger than'):
        art.store_image(_image(), str(tmp_path), 'a')
    assert not (tmp_path / art.ART_CACHE).exists()


def test_setting_marks_it_a_persons_choice_and_clearing_forgets_it(tmp_path):
    write_entry(str(tmp_path), 'a')
    path = art.store_image(_image(), str(tmp_path), 'a')
    entry = art.set_artwork(str(tmp_path), 'a', path)
    assert (entry.art_path, entry.art_overridden) == (path, True)
    art.set_artwork(str(tmp_path), 'a', None)
    assert (read_entry(str(tmp_path), 'a').art_path, read_entry(str(tmp_path), 'a').art_overridden) == (None, False)
