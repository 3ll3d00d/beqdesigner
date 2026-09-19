'''Tests for the filesystem LibrarySource (plan §11.2).'''
import os

import pytest

from pipeline.library.extract_cache import source_fingerprint
from pipeline.library.filesystem import FilesystemLibrarySource


def _touch(path, content=b'x'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(content)
    return str(path)


def _disc(root):
    _touch(os.path.join(root, 'BDMV', 'index.bdmv'))
    _touch(os.path.join(root, 'BDMV', 'STREAM', '00000.m2ts'))
    _touch(os.path.join(root, 'BDMV', 'PLAYLIST', '00800.mpls'))
    return str(root)


def test_a_directory_yields_its_media_files_only(tmp_path):
    film = _touch(tmp_path / 'lib' / 'Film One (2001).mkv')
    _touch(tmp_path / 'lib' / 'Film One (2001).nfo')
    _touch(tmp_path / 'lib' / 'poster.jpg')

    items = list(FilesystemLibrarySource([str(tmp_path / 'lib')]).list_items())

    assert [i.source_path for i in items] == [film]
    assert items[0].display_name == 'Film One (2001)'
    assert items[0].kind == 'movie'
    assert items[0].title is None  # not guessed from the filename
    assert items[0].fingerprint == ''  # the extract cache's mtime/size fallback applies
    assert source_fingerprint(items[0])  # ...and works


def test_a_directory_is_not_recursive_but_a_double_star_glob_is(tmp_path):
    top = _touch(tmp_path / 'lib' / 'a.mkv')
    deep = _touch(tmp_path / 'lib' / 'sub' / 'deeper' / 'b.mkv')

    assert [i.source_path for i in FilesystemLibrarySource([str(tmp_path / 'lib')]).list_items()] == [top]
    found = FilesystemLibrarySource([str(tmp_path / 'lib' / '**' / '*.mkv')]).list_items()
    assert sorted(i.source_path for i in found) == sorted([top, deep])


def test_extensions_can_be_widened_or_disabled(tmp_path):
    odd = _touch(tmp_path / 'lib' / 'film.xyz')

    assert list(FilesystemLibrarySource([str(tmp_path / 'lib')]).list_items()) == []
    assert [i.source_path for i in
            FilesystemLibrarySource([str(tmp_path / 'lib')], extensions={'.XYZ'}).list_items()] == [odd]
    assert [i.source_path for i in
            FilesystemLibrarySource([str(tmp_path / 'lib')], extensions=None).list_items()] == [odd]


def test_a_disc_rip_is_one_item_and_its_clips_are_not(tmp_path):
    disc = _disc(tmp_path / 'lib' / 'Some Disc')
    _touch(tmp_path / 'lib' / 'plain' / 'a.mkv')

    items = list(FilesystemLibrarySource([str(tmp_path / 'lib' / '**' / '*')]).list_items())

    assert sorted(i.source_path for i in items) == sorted([disc, str(tmp_path / 'lib' / 'plain' / 'a.mkv')])
    disc_item = next(i for i in items if i.source_path == disc)
    assert disc_item.display_name == 'Some Disc'
    assert disc_item.fingerprint  # stat of BDMV/index.bdmv -- a directory's own stat is not meaningful


def test_a_plain_folder_that_is_not_a_disc_is_ignored(tmp_path):
    os.makedirs(tmp_path / 'lib' / 'empty folder')

    assert list(FilesystemLibrarySource([str(tmp_path / 'lib')]).list_items()) == []


def test_ids_are_stable_distinct_and_deduplicated_across_overlapping_globs(tmp_path):
    a = _touch(tmp_path / 'lib' / 'a.mkv')
    _touch(tmp_path / 'lib' / 'b.mkv')

    source = FilesystemLibrarySource([str(tmp_path / 'lib'), str(tmp_path / 'lib' / '*.mkv')])
    first = list(source.list_items())
    second = list(source.list_items())

    assert len(first) == 2
    assert [i.id for i in first] == [i.id for i in second]
    assert len({i.id for i in first}) == 2
    assert all(i.id.startswith('fs-') for i in first)
    assert first[0].source_path == a


def test_the_same_file_reached_through_a_symlink_is_one_title(tmp_path):
    real = _touch(tmp_path / 'real' / 'a.mkv')
    os.makedirs(tmp_path / 'lib')
    try:
        os.symlink(real, tmp_path / 'lib' / 'link.mkv')
    except (OSError, NotImplementedError):
        pytest.skip('symlinks unavailable')

    items = list(FilesystemLibrarySource([str(tmp_path / 'real'), str(tmp_path / 'lib')]).list_items())

    assert len(items) == 1


def test_nothing_matching_is_an_empty_library_not_an_error(tmp_path):
    assert list(FilesystemLibrarySource([str(tmp_path / 'missing' / '*.mkv')]).list_items()) == []


def test_it_needs_a_glob_and_takes_no_query():
    with pytest.raises(ValueError):
        FilesystemLibrarySource([])
    with pytest.raises(TypeError):
        FilesystemLibrarySource(['/x']).list_items(title='x')
