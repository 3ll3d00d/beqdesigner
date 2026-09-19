'''
Reading the local XML repo for TMDB ids (design.md §12.5 "Repo awareness"). Uses the real XML writer, so the element
names the scan looks for are the ones a publish actually emits (§12.14).
'''
import os

from model.iir import CompleteFilter, LowShelf
from pipeline.library.catalogue_scan import parse_xml, scan_xml_repo, tmdb_index
from pipeline.metadata import BeqMetadata
from pipeline.publish.xml import to_beq_xml

FILTER = CompleteFilter(fs=48000, filters=[LowShelf(20, 6, 0.7, 4.0)])


def _write(repo, name, **meta):
    fields = dict(title='T', year='2020', audio_types=['Atmos'], gain='-3')
    fields.update(meta)
    path = os.path.join(repo, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(to_beq_xml(FILTER, BeqMetadata(**fields)))
    return path


def test_a_film_has_its_tmdb_id_and_is_not_tv(tmp_path):
    assert parse_xml(_write(str(tmp_path), 'film.xml', the_movie_db='603')) == ('603', False)


def test_a_plain_season_or_a_structured_one_makes_it_tv(tmp_path):
    plain = _write(str(tmp_path), 'plain.xml', the_movie_db='1396', season='1')
    structured = _write(str(tmp_path), 'structured.xml', the_movie_db='1396', season='1', season_id='3572',
                        season_episode_count=7, episodes=[1, 2])

    assert parse_xml(plain) == ('1396', True)
    assert parse_xml(structured) == ('1396', True)


def test_an_unresolved_id_is_empty_and_an_unreadable_file_is_none(tmp_path):
    assert parse_xml(_write(str(tmp_path), 'blank.xml')) == ('', False)
    broken = tmp_path / 'broken.xml'
    broken.write_text('<a><b>')
    assert parse_xml(str(broken)) is None
    assert parse_xml(str(tmp_path / 'missing.xml')) is None


def test_scan_finds_every_xml_below_the_repo_but_not_inside_dot_git(tmp_path):
    repo = str(tmp_path)
    _write(repo, 'a.xml', the_movie_db='1')
    _write(repo, os.path.join('deep', 'er', 'b.XML'), the_movie_db='2')
    _write(repo, os.path.join('.git', 'c.xml'), the_movie_db='3')
    (tmp_path / 'notes.txt').write_text('x')

    records = scan_xml_repo(repo)

    assert sorted(records) == ['a.xml', 'deep/er/b.XML']
    assert tmdb_index(records) == {('1', False): {'a'}, ('2', False): {'b'}}


def test_a_missing_repo_is_empty_and_unchanged_files_are_reused(tmp_path):
    assert scan_xml_repo(str(tmp_path / 'nowhere')) == {} and scan_xml_repo('') == {}
    _write(str(tmp_path), 'a.xml', the_movie_db='1')
    first = scan_xml_repo(str(tmp_path))

    assert scan_xml_repo(str(tmp_path), first)['a.xml'] is first['a.xml']  # not parsed again
