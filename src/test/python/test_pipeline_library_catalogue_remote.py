'''
pipeline/library/catalogue_remote.py -- published BEQs read on demand from beqcatalogue's comparison files
(design/web-parity.md WP3): cached a day, asked again conditionally, the cached copy used when the catalogue is down.
'''
import pytest

from catalogue_fixture import FakeCatalogue
from pipeline.library.catalogue_match import CatalogueTarget
from pipeline.library.catalogue_remote import CatalogueCache, CatalogueUnavailable, same_track


class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


@pytest.fixture
def fake():
    return FakeCatalogue()


def _cache(tmp_path, fake, clock=None):
    return CatalogueCache(str(tmp_path), 'https://example.org/latest', fake, clock or Clock())


def test_a_title_is_found_by_tmdb_id_and_its_entries_read(tmp_path, fake):
    found = _cache(tmp_path, fake).find(CatalogueTarget(title='Whatever', year='1990', tmdb='348'))
    assert (found.key, found.matched_by, found.title) == ('film-alien_348', 'tmdb', 'Alien')
    assert [e.author for e in found.entries] == ['aron7awol', 'mobe1969']
    assert found.entries[0].audio_types == ('DTS-HD MA 5.1',) and found.entries[0].catalogue_url.endswith('#alien')
    assert [name for name, _ in fake.asked] == ['titles.json', 'film-alien_348.json']


def test_else_by_title_and_year_and_never_across_film_and_tv(tmp_path, fake):
    cache = _cache(tmp_path, fake)
    assert cache.find(CatalogueTarget(title='ALIEN', year='1979')).matched_by == 'title'
    assert cache.find(CatalogueTarget(title='Alien', year='1979', is_tv=True)).key == 'tv-alien_9999'
    assert cache.find(CatalogueTarget(title='Alien', year='1980')) is None
    assert cache.find(CatalogueTarget(title='Alien')) is None          # not usable without a year or an id
    assert cache.entries('tv-alien_9999') == []                        # listed, but no comparison file


def test_a_file_is_asked_for_at_most_once_a_day_then_conditionally(tmp_path, fake):
    clock = Clock()
    cache = _cache(tmp_path, fake, clock)
    cache.titles()
    cache.titles()
    assert len(fake.asked) == 1
    clock.now += 25 * 3600
    assert cache.titles()[0]['key'] == 'film-alien_348'
    assert fake.asked[-1] == ('titles.json', {'If-None-Match': '"titles.json-v1"'})
    clock.now += 3600
    cache.titles()
    assert len(fake.asked) == 2                                         # the 304 counted as fresh


def test_the_cached_copy_is_used_when_the_catalogue_is_down_and_without_one_it_says_so(tmp_path, fake):
    clock = Clock()
    cache = _cache(tmp_path, fake, clock)
    cache.titles()
    fake.down = True
    clock.now += 25 * 3600
    assert cache.titles()[0]['title'] == 'Alien'
    with pytest.raises(CatalogueUnavailable, match='no route to host'):
        _cache(tmp_path / 'empty', fake).titles()


def test_a_page_that_is_not_json_is_never_kept(tmp_path):
    fake = FakeCatalogue({'titles.json': b'<html>maintenance</html>'})
    with pytest.raises(CatalogueUnavailable):
        _cache(tmp_path, fake).titles()
    assert not (tmp_path / 'titles.json').exists()


def test_a_key_that_could_name_another_file_is_refused(tmp_path, fake):
    with pytest.raises(ValueError, match='not a catalogue key'):
        _cache(tmp_path, fake).entries('../secrets')


def test_same_track_is_the_title_pages_test(tmp_path, fake):
    entries = _cache(tmp_path, fake).entries('film-alien_348')
    target = CatalogueTarget(title='Alien', year='1979', tmdb='348', audio_types=('DTS-HD MA 5.1',))
    assert [same_track(target, e) for e in entries] == [True, False]
