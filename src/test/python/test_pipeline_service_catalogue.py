'''
Published BEQs for a title over HTTP (design/web-parity.md WP3), against a fake catalogue: matched by TMDB id, each
marked whether it is for the same track, and any one charted against the title's measured curves.
'''
import pytest
from fastapi.testclient import TestClient

from catalogue_fixture import FakeCatalogue
from pipeline.review import read_entry, update_entry
from pipeline.service.api import create_app
from pipeline.service.config import ServiceConfig
from pipeline.service.jobs import JobManager
from pipeline.service.models import Check
from test_pipeline_service_review import AUTH, TOKEN, Gate, setup  # noqa: F401 (fixtures)


@pytest.fixture
def catalogue(setup):
    profile, _, queue = setup
    fake = FakeCatalogue()
    manager = JobManager(Gate())
    app = create_app(manager, ServiceConfig(profile_path=str(profile), token=TOKEN), env={},
                     checks=lambda: [Check(name='profile', ok=True)], catalogue_url='https://example.org/latest',
                     catalogue_get=fake)
    entry = read_entry(str(queue), 'a')
    update_entry(str(queue), 'a', meta={**entry.meta, 'title': 'Alien', 'year': '1979', 'the_movie_db': '348',
                                        'audio_types': ['DTS-HD MA 5.1']})
    with TestClient(app) as client:
        yield client, fake
    manager.stop(grace_seconds=5)


def test_a_titles_published_beqs_are_listed_each_marked_whether_it_is_the_same_track(catalogue):
    client, fake = catalogue
    body = client.get('/v1/titles/a/catalogue', headers=AUTH).json()
    assert (body['key'], body['matched_by']) == ('film-alien_348', 'tmdb')
    assert body['compare_url'] == 'https://example.org/latest/compare/?t=film-alien_348'
    assert [(e['author'], e['same_track'], e['filter_count']) for e in body['entries']] == \
        [('aron7awol', True, 1), ('mobe1969', False, 1)]
    assert body['entries'][1]['edition'] == "Director's Cut" and body['entries'][1]['catalogue_url'] is None


def test_one_is_charted_as_its_response_and_the_track_under_it(catalogue):
    client, _ = catalogue
    chart = client.get('/v1/titles/a/catalogue/0/chart', headers=AUTH).json()
    assert [(s['kind'], s['name']) for s in chart['series']] == [
        ('filter', "aron7awol's BEQ"),
        ('average', "Average audio track (all channels mixed) under aron7awol's BEQ"),
        ('peak', "Peak audio track (all channels mixed) under aron7awol's BEQ")]
    response = chart['series'][0]
    assert response['y'][0] == pytest.approx(10.0, abs=0.5)              # two +5 dB shelves (count 2)
    assert client.get('/v1/titles/a/catalogue/5/chart', headers=AUTH).status_code == 404
    assert client.get('/v1/titles/a/catalogue/0/chart', params={'smoothing': 'loads'}, headers=AUTH).status_code == 422


def test_a_title_the_catalogue_does_not_have_says_so(catalogue):
    client, _ = catalogue
    body = client.get('/v1/titles/b/catalogue', headers=AUTH).json()
    assert body['entries'] == [] and body['key'] is None and 'no BEQ for this title' in body['message']


def test_a_catalogue_that_cannot_be_reached_is_a_502(catalogue):
    client, fake = catalogue
    fake.down = True
    answer = client.get('/v1/titles/a/catalogue', headers=AUTH)
    assert (answer.status_code, answer.json()['title']) == (502, 'Catalogue unavailable')
