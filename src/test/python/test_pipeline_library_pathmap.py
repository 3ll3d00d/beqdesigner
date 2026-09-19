'''Translating server (Windows) paths to local ones -- plan §11.6.'''
import os

import pytest

from pipeline.library.pathmap import PathMapping, mappings_from_config, translate_path

FILMS = PathMapping('W:\\Films', '/mnt/films')


def _t(path, *mappings):
    return translate_path(path, mappings)


def test_a_windows_path_under_a_mapping_becomes_a_local_one():
    assert _t('W:\\Films\\Action\\Die Hard (1988).mkv', FILMS) == os.path.join('/mnt/films', 'Action',
                                                                               'Die Hard (1988).mkv')


def test_matching_ignores_case_and_treats_both_separators_alike():
    assert _t('w:/films/a.mkv', FILMS) == os.path.join('/mnt/films', 'a.mkv')
    assert _t('W:\\FILMS\\a.mkv', PathMapping('w:/films', '/mnt/films')) == os.path.join('/mnt/films', 'a.mkv')


def test_a_prefix_only_matches_whole_path_components():
    assert _t('W:\\Filmsback\\a.mkv', FILMS) == 'W:\\Filmsback\\a.mkv'
    assert _t('W:\\Film\\a.mkv', PathMapping('W:\\Films', '/x')) == 'W:\\Film\\a.mkv'


def test_a_trailing_separator_on_either_side_makes_no_difference():
    for source in ('W:\\Films', 'W:\\Films\\', 'W:/Films/'):
        assert _t('W:\\Films\\a.mkv', PathMapping(source, '/mnt/films/')) == os.path.join('/mnt/films/', 'a.mkv')


def test_the_folder_itself_maps_to_the_target():
    assert _t('W:\\Films', FILMS) == '/mnt/films'
    assert _t('W:\\Films\\', FILMS) == '/mnt/films'


def test_a_drive_root_can_be_mapped_whole():
    assert _t('W:\\Films\\a.mkv', PathMapping('W:\\', '/mnt/w')) == os.path.join('/mnt/w', 'Films', 'a.mkv')
    assert _t('W:\\a.mkv', PathMapping('W:', '/mnt/w')) == os.path.join('/mnt/w', 'a.mkv')


def test_the_longest_matching_prefix_wins_whatever_the_order():
    broad = PathMapping('W:\\', '/mnt/w')
    narrow = PathMapping('W:\\Films\\4K', '/mnt/uhd')
    expected = os.path.join('/mnt/uhd', 'a.mkv')

    assert _t('W:\\Films\\4K\\a.mkv', broad, narrow) == expected
    assert _t('W:\\Films\\4K\\a.mkv', narrow, broad) == expected
    assert _t('W:\\Films\\HD\\a.mkv', broad, narrow) == os.path.join('/mnt/w', 'Films', 'HD', 'a.mkv')


def test_unc_paths_map_too():
    mapping = PathMapping('\\\\nas\\media', '/mnt/nas')

    assert _t('\\\\NAS\\Media\\Films\\a.mkv', mapping) == os.path.join('/mnt/nas', 'Films', 'a.mkv')


def test_an_unmapped_path_is_returned_untouched():
    assert _t('D:\\Other\\a.mkv', FILMS) == 'D:\\Other\\a.mkv'
    assert _t('/already/local/a.mkv', FILMS) == '/already/local/a.mkv'
    assert _t('W:\\Films\\a.mkv') == 'W:\\Films\\a.mkv'  # no mappings at all


def test_a_blank_source_never_matches():
    assert _t('W:\\Films\\a.mkv', PathMapping('', '/mnt')) == 'W:\\Films\\a.mkv'


@pytest.mark.parametrize('entries, expected', [
    ([{'from': 'W:\\Films', 'to': '/mnt/films'}], [PathMapping('W:\\Films', '/mnt/films')]),
    ([['W:\\', '/mnt/w']], [PathMapping('W:\\', '/mnt/w')]),
    (['W:\\Films=/mnt/films'], [PathMapping('W:\\Films', '/mnt/films')]),
    ([PathMapping('a', 'b')], [PathMapping('a', 'b')]),
    (None, []),
])
def test_config_entries_in_every_accepted_shape(entries, expected):
    assert mappings_from_config(entries) == expected


@pytest.mark.parametrize('bad', [[{'from': 'only-one'}], ['no-equals'], [['one']], [3]])
def test_a_malformed_config_entry_is_rejected(bad):
    with pytest.raises(ValueError, match='path mapping'):
        mappings_from_config(bad)
