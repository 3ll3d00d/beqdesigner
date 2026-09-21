'''
Reading the local JSON record repo for TMDB ids (design.md §12.5 "Repo awareness").
'''
import json
import os

from pipeline.library.catalogue_scan import parse_record, scan_xml_repo, tmdb_index


def _write(repo, name, **meta):
    fields = dict(the_movie_db='', content_type='film')
    fields.update(meta)
    path = os.path.join(repo, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'theMovieDB': fields['the_movie_db'], 'content_type': fields['content_type']}, f)
    return path


def test_a_film_has_its_tmdb_id_and_is_not_tv(tmp_path):
    assert parse_record(_write(str(tmp_path), 'film.json', the_movie_db='603')) == ('603', False)


def test_a_plain_season_or_a_structured_one_makes_it_tv(tmp_path):
    plain = _write(str(tmp_path), 'plain.json', the_movie_db='1396', content_type='TV')
    structured = _write(str(tmp_path), 'structured.json', the_movie_db='1396', content_type='TV')

    assert parse_record(plain) == ('1396', True)
    assert parse_record(structured) == ('1396', True)


def test_an_unresolved_id_is_empty_and_an_unreadable_file_is_none(tmp_path):
    assert parse_record(_write(str(tmp_path), 'blank.json')) == ('', False)
    broken = tmp_path / 'broken.json'
    broken.write_text('{')
    assert parse_record(str(broken)) is None
    assert parse_record(str(tmp_path / 'missing.json')) is None


def test_scan_finds_every_record_below_the_repo_but_not_inside_dot_git(tmp_path):
    repo = str(tmp_path)
    _write(repo, 'a.json', the_movie_db='1')
    _write(repo, os.path.join('deep', 'er', 'b.JSON'), the_movie_db='2')
    _write(repo, os.path.join('.git', 'c.json'), the_movie_db='3')
    (tmp_path / 'notes.txt').write_text('x')

    records = scan_xml_repo(repo)

    assert sorted(records) == ['a.json', 'deep/er/b.JSON']
    assert tmdb_index(records) == {('1', False): {'a'}, ('2', False): {'b'}}


def test_a_missing_repo_is_empty_and_unchanged_files_are_reused(tmp_path):
    assert scan_xml_repo(str(tmp_path / 'nowhere')) == {} and scan_xml_repo('') == {}
    _write(str(tmp_path), 'a.json', the_movie_db='1')
    first = scan_xml_repo(str(tmp_path))

    assert scan_xml_repo(str(tmp_path), first)['a.json'] is first['a.json']  # not parsed again
