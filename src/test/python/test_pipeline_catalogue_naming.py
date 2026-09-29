'''
Where a published title lives: a readable name (`Title (Year) (Edition) Audio`, the way beqcatalogue named its files), under
configurable movies/tv folders, on by default, then a folder per first letter -- pipeline.publish.catalogue.
'''
import pytest

from pipeline.publish.catalogue import CategoryFolders, FolderName, aggregate_path, catalogue_paths, catalogue_stem, \
    category_folders_from_values, category_for_metadata, heatmap_path, letter_folder, lettered_stem, record_folder, \
    unique_stem
from pipeline.publish.git import RAW_CONTENT_TEMPLATE


def test_the_stem_is_the_title_year_edition_and_audio():
    meta = {'title': '1917', 'year': '2019', 'edition': 'Amazon', 'audio_types': ['DD+']}
    assert catalogue_stem(meta) == '1917 (2019) (Amazon) DD+'


def test_the_stem_omits_what_a_title_lacks_and_joins_several_audio_types():
    assert catalogue_stem({'title': 'Heat', 'year': '1995', 'audio_types': ['Atmos', 'TrueHD 7.1']}) == \
           'Heat (1995) Atmos + TrueHD 7.1'
    assert catalogue_stem({'title': 'Heat'}) == 'Heat'


def test_a_season_is_in_the_stem():
    assert catalogue_stem({'title': 'The Expanse', 'year': '2015', 'season': '1', 'audio_types': ['DD+']}) == \
           'The Expanse (2015) S01 DD+'
    assert catalogue_stem({'title': 'Show', 'season': 'Specials'}) == 'Show Specials'


def test_the_stem_is_safe_as_a_file_name():
    stem = catalogue_stem({'title': 'Face/Off: Redux?', 'year': '1997', 'audio_types': ['DTS-HD MA 5.1']})
    assert stem == 'Face_Off_ Redux_ (1997) DTS-HD MA 5.1'


def test_a_title_less_entry_falls_back_to_its_id():
    assert catalogue_stem({'title': ''}, fallback='jriver-1-2') == 'jriver-1-2'


def test_the_master_volume_is_not_in_the_stem_so_a_revision_keeps_the_path():
    assert '+' not in catalogue_stem({'title': 'A', 'year': '2000', 'mv': '+3', 'gain': '+3'})


def test_unique_stem_numbers_a_taken_name():
    taken = {'A (2000) Atmos', 'A (2000) Atmos (2)'}
    assert unique_stem('A (2000) Atmos', taken.__contains__) == 'A (2000) Atmos (3)'
    assert unique_stem('B', taken.__contains__) == 'B'


@pytest.mark.parametrize('name, folder', [('Heat (1995) Atmos', 'H'), ('alien', 'A'), ('Élite', 'E'), ('Ærø', '#'),
                                          ('1917 (2019) DD+', '0-9'), ('[REC] (2007)', '#'), ('_Untitled', '#'),
                                          ('Шрек', '#'), ('', '#')])
def test_the_letter_folder_is_the_first_letter_upper_case_and_unaccented(name, folder):
    assert letter_folder(name) == folder


def test_a_lettered_stem_puts_the_files_of_both_repositories_in_the_letter_folder():
    stem = lettered_stem('Heat (1995) Atmos')
    assert stem == 'H/Heat (1995) Atmos'
    assert catalogue_paths('id', 'xml', 'img', category='film', stem=stem) == \
           ('xml/movies/H/Heat (1995) Atmos.json', 'img/movies/H/Heat (1995) Atmos.png')
    assert unique_stem(stem, lambda name: name == stem) == 'H/Heat (1995) Atmos (2)'


def test_the_aggregate_is_in_the_category_folder_above_the_letter_folders():
    assert record_folder('xml', category='film') == 'xml/movies'
    assert aggregate_path('xml', category='film') == 'xml/movies/database.json'
    assert aggregate_path('xml') == 'xml/database.json'


def test_paths_use_the_stem_else_the_id():
    assert catalogue_paths('id', 'f', 'i', stem='Heat (1995) Atmos') == ('f/Heat (1995) Atmos.json',
                                                                          'i/Heat (1995) Atmos.png')
    assert catalogue_paths('id', 'f', 'i') == ('f/id.json', 'i/id.png')


def test_the_heatmap_is_beside_the_report_image():
    assert heatmap_path('img/movies/Heat (1995) Atmos.png') == 'img/movies/Heat (1995) Atmos heatmap.png'


def test_category_folders_are_on_unless_a_profile_says_off():
    assert category_folders_from_values({}) == CategoryFolders('movies', 'tv')
    assert category_folders_from_values({'category_folders': False}) is False


def test_the_folder_names_are_configurable():
    folders = category_folders_from_values({'movies_dir': 'Movie BEQs/', 'tv_dir': 'TV Shows BEQ'})
    assert folders == CategoryFolders('Movie BEQs', 'TV Shows BEQ')
    film = category_for_metadata({}, None, folders)
    show = category_for_metadata({'season': '1'}, None, folders)
    assert isinstance(film, FolderName)
    assert catalogue_paths('x', 'f', 'i', category=film, stem='A')[0] == 'f/Movie BEQs/A.json'
    assert catalogue_paths('x', 'f', 'i', category=show, stem='A')[1] == 'i/TV Shows BEQ/A.png'


def test_a_custom_folder_called_film_or_tv_is_not_mistaken_for_the_legacy_marker():
    folders = category_folders_from_values({'movies_dir': 'film', 'tv_dir': 'TV'})
    assert catalogue_paths('x', category=category_for_metadata({}, None, folders), stem='A')[0] == 'film/A.json'
    assert catalogue_paths('x', category=category_for_metadata({'season': 1}, None, folders), stem='A')[0] == 'TV/A.json'


@pytest.mark.parametrize('bad', ['..', 'a/../b', '../x'])
def test_a_folder_that_climbs_out_of_the_repository_is_refused(bad):
    with pytest.raises(ValueError):
        category_folders_from_values({'movies_dir': bad})


def test_true_still_means_the_default_folders():
    assert category_for_metadata({}, None, True) == 'film'
    assert catalogue_paths('x', category='film')[0] == 'movies/x.json'
    assert catalogue_paths('x', category='TV')[0] == 'tv/x.json'


def test_a_raw_url_quotes_a_readable_name():
    from urllib.parse import quote
    url = RAW_CONTENT_TEMPLATE.format(owner='o', repo='r', branch='main',
                                      path=quote('movies/1917 (2019) DD+ heatmap.png', safe='/'))
    assert ' ' not in url and '%20' in url and '%2B' in url
