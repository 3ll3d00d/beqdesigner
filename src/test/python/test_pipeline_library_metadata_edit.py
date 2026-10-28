'''
pipeline/library/metadata_edit.py -- the metadata edit rules the desktop title page and the service share (design/web-parity.md
WP0). The page's behaviour over them is tested in gui/test_worklist_metadata*.py; these are the rules alone, with no widgets.
'''
import pytest
import requests

from pipeline.library import metadata_edit as edit
from pipeline.metadata import BeqMetadata
from pipeline.review import read_entry
from review_entry_fixture import write_entry


def test_texts_become_changes_with_blank_unsetting_and_blank_episodes_meaning_none():
    changes = edit.changes_from_texts({'title': ' Alien ', 'note': '  ', 'audio_types': 'Atmos, DTS:X,', 'episodes': ''})
    assert changes == {'title': 'Alien', 'note': edit.REMOVE, 'audio_types': ['Atmos', 'DTS:X'], 'episodes': []}
    assert edit.changes_from_texts({'audio_types': ' , '}) == {'audio_types': edit.REMOVE}
    assert edit.changes_from_texts({'episodes': '1-3, 5'}) == {'episodes': [1, 2, 3, 5]}


@pytest.mark.parametrize('texts, extras, says', [({'episodes': 'one'}, None, 'Episodes:'),
                                                 ({'colour': 'red'}, None, 'colour is not a metadata field'),
                                                 ({}, {'title': 'x'}, 'title is not something a reload fills')])
def test_what_cannot_be_a_change_is_refused_with_a_sentence(texts, extras, says):
    with pytest.raises(ValueError, match=says):
        edit.changes_from_texts(texts, extras)


def test_extras_from_a_reload_are_taken_as_they_are():
    genres = [{'id': 1, 'name': 'Horror'}]
    assert edit.changes_from_texts({}, {'genres': genres}) == {'genres': genres}


def test_every_text_field_round_trips_through_its_text():
    meta = {'title': 'Alien', 'year': '1979', 'audio_types': ['Atmos', 'DD+'], 'episodes': [1, 2, 3, 5], 'gain': '-1'}
    texts = edit.field_texts(meta)
    assert set(texts) == set(edit.TEXT_FIELDS)
    assert edit.apply_changes({}, edit.changes_from_texts({k: texts[k] for k in meta})) == meta


def test_a_save_merges_into_the_entry_on_disk_and_says_whether_it_wrote(tmp_path):
    write_entry(str(tmp_path), 'r-alien')
    entry, wrote = edit.save_meta_changes(str(tmp_path), 'r-alien', {'note': 'kept'})
    assert wrote and entry.meta['note'] == 'kept' and read_entry(str(tmp_path), 'r-alien').meta['note'] == 'kept'
    assert edit.save_meta_changes(str(tmp_path), 'r-alien', {'note': 'kept'})[1] is False
    with pytest.raises(FileNotFoundError):
        edit.save_meta_changes(str(tmp_path), 'missing', {'note': 'x'})


def test_a_reload_fills_only_what_tmdb_knows():
    found = BeqMetadata(title='Alien', year='1979', the_movie_db='348', overview='In space...', genres=[{'name': 'Horror'}])
    texts, extras = edit.tmdb_fill(found)
    assert texts == {'title': 'Alien', 'year': '1979', 'the_movie_db': '348'}
    assert extras == {'overview': 'In space...', 'genres': [{'name': 'Horror'}]}
    assert edit.tmdb_fill(BeqMetadata(title='Alien', year='')) == ({}, {})


@pytest.mark.parametrize('error', [requests.ConnectionError('https://api.themoviedb.org/3/movie/1?api_key=SECRET down'),
                                   KeyError('api_key=SECRET')])
def test_a_failed_reload_raises_its_sentence_without_the_key(monkeypatch, error):
    def fail(*_, **__):
        raise error
    monkeypatch.setattr(edit, 'tmdb_details_by_id', fail)
    with pytest.raises(edit.TmdbError) as raised:
        edit.tmdb_reload('SECRET', '1', '', '', 'movie')
    assert 'TMDB lookup failed' in str(raised.value) and 'SECRET' not in str(raised.value)


def test_a_reload_looks_up_by_id_else_by_title_and_year(monkeypatch):
    calls = []
    monkeypatch.setattr(edit, 'tmdb_details_by_id', lambda i, k, kind: calls.append(('id', i, kind)) or BeqMetadata('', ''))
    monkeypatch.setattr(edit, 'tmdb_lookup', lambda t, y, k, kind: calls.append(('title', t, y, kind)) or BeqMetadata('', ''))
    edit.tmdb_reload('key', '348', 'Alien', '1979', 'movie')
    edit.tmdb_reload('key', '', 'Alien', '1979', 'tv')
    assert calls == [('id', '348', 'movie'), ('title', 'Alien', '1979', 'tv')]


def test_lpcm_2_0_is_an_audio_type_a_person_can_pick_and_it_is_kept_as_typed():
    assert 'LPCM 2.0' in edit.AUDIO_TYPES
    changes = edit.changes_from_texts({'audio_types': 'LPCM 2.0, DTS-HD MA 5.1'})
    assert changes['audio_types'] == ['LPCM 2.0', 'DTS-HD MA 5.1']
