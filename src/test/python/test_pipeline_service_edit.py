'''
Editing a title over HTTP (design/web-parity.md WP4): its metadata, a TMDB reload and its artwork, with the title page's
rules (pipeline.library.metadata_edit and artwork_edit, tested on their own) and the review routes' refusals: a title a
run has in hand, and an edit made against metadata that has changed since.
'''
import base64
import io
import json

import pytest
import requests
from PIL import Image

from pipeline.metadata import BeqMetadata
from pipeline.review import read_entry, update_entry
from pipeline.service.edits import RUNNING
from test_pipeline_service_review import AUTH, Gate, _event, _review, service, setup  # noqa: F401 (fixtures)


def _png(size=(4, 4)) -> bytes:
    out = io.BytesIO()
    Image.new('RGB', size, 'red').save(out, format='PNG')
    return out.getvalue()


def _edit(client, title_id='a', **body):
    return client.patch(f'/v1/titles/{title_id}/metadata', json=body, headers=AUTH)


def test_the_fields_are_the_title_pages_in_order_with_their_choices(service):
    client, *_ = service
    fields = client.get('/v1/metadata/fields', headers=AUTH).json()
    assert [f['key'] for f in fields][:3] == ['title', 'year', 'audio_types']
    audio = fields[2]
    assert audio['kind'] == 'choices' and 'Atmos' in audio['choices'] and audio['group'] == 'essentials'
    assert next(f for f in fields if f['key'] == 'gain')['group'] == 'more'


def test_a_review_carries_the_texts_and_the_digest_an_edit_is_made_against(service):
    client, *_ = service
    review = _review(client)
    assert review['metadata_texts']['title'] == review['metadata']['title']
    assert len(review['metadata_digest']) == 32
    assert review['artwork'] == {'present': False, 'overridden': False, 'name': None}


def test_an_edit_is_merged_into_the_entry_and_answered_with_the_review(service):
    client, _, queue, _ = service
    review = _review(client)
    answer = _edit(client, digest=review['metadata_digest'], texts={'note': 'Kept', 'edition': '  ', 'episodes': ''})
    assert answer.status_code == 200, answer.text
    meta = read_entry(queue, 'a').meta
    assert meta['note'] == 'Kept' and 'edition' not in meta and meta['episodes'] == []
    assert answer.json()['metadata_texts']['note'] == 'Kept'
    assert answer.json()['metadata_digest'] != review['metadata_digest']


def test_blanking_a_required_field_is_saved_and_holds_accept_back(service):
    client, *_ = service
    answer = _edit(client, digest=_review(client)['metadata_digest'], texts={'year': ''}).json()
    assert answer['metadata_problems'] and 'year' in answer['metadata_problems'][0].lower()
    assert answer['blocked']['accept']


def test_an_edit_made_against_older_metadata_is_refused(service):
    client, _, queue, _ = service
    digest = _review(client)['metadata_digest']
    update_entry(queue, 'a', meta={**read_entry(queue, 'a').meta, 'note': 'a run wrote this'})
    refused = _edit(client, digest=digest, texts={'note': 'mine'})
    assert (refused.status_code, refused.json()['title']) == (409, 'Changed since it was read')
    assert read_entry(queue, 'a').meta['note'] == 'a run wrote this'


@pytest.mark.parametrize('texts, extras, says', [({'episodes': 'one'}, {}, 'Episodes:'),
                                                 ({'colour': 'red'}, {}, 'not a metadata field'),
                                                 ({}, {'year': '1979'}, 'not something a reload fills')])
def test_what_cannot_be_an_edit_is_a_422(service, texts, extras, says):
    client, *_ = service
    refused = _edit(client, digest=_review(client)['metadata_digest'], texts=texts, extras=extras)
    assert refused.status_code == 422 and says in refused.json()['detail']


def test_nothing_is_edited_on_a_title_a_run_has_in_hand(service):
    client, gate, *_ = service
    gate.report = [_event('a', 'queued', '')]
    review = _review(client)
    client.post('/v1/jobs/scan', json={}, headers=AUTH)
    assert gate.entered.wait(10)
    for refused in (_edit(client, digest=review['metadata_digest'], texts={'note': 'x'}),
                    client.put('/v1/titles/a/artwork', json={'digest': review['metadata_digest'],
                                                             'data_base64': base64.b64encode(_png()).decode()},
                               headers=AUTH),
                    client.delete('/v1/titles/a/artwork', params={'digest': review['metadata_digest']}, headers=AUTH)):
        assert (refused.status_code, refused.json()['title']) == (409, 'In flight')


# --- TMDB ---------------------------------------------------------------------------------------------------------------

def test_a_reload_without_a_key_says_so(service):
    client, *_ = service
    refused = client.post('/v1/titles/a/metadata/tmdb', json={}, headers=AUTH)
    assert (refused.status_code, refused.json()['title']) == (409, 'No TMDB key')


@pytest.fixture
def keyed(service):
    ''' The profile names a TMDB key (as TMDB_API_KEY does, through run.tmdb_api_key). '''
    profile = service[3]
    config = json.loads(profile.read_text())
    config['run']['tmdb_api_key'] = 'SECRET'
    profile.write_text(json.dumps(config))


def test_a_reload_proposes_what_tmdb_knows_and_saves_nothing(keyed, service, monkeypatch):
    client, *_ = service
    asked = []

    def lookup(title, year, key, kind):
        asked.append((title, year, key, kind))
        return BeqMetadata('Alien', '1979', the_movie_db='348', overview='In space', genres=[{'name': 'Horror'}])
    monkeypatch.setattr('pipeline.library.metadata_edit.tmdb_lookup', lookup)
    before = _review(client)
    proposal = client.post('/v1/titles/a/metadata/tmdb', json={'title': 'Alien', 'year': '1979', 'tmdb_id': ''},
                           headers=AUTH)
    assert proposal.status_code == 200, proposal.text
    body = proposal.json()
    assert body['found'] and body['texts']['the_movie_db'] == '348' and body['extras']['overview'] == 'In space'
    assert asked == [('Alien', '1979', 'SECRET', 'movie')]
    assert _review(client)['metadata_digest'] == before['metadata_digest']


def test_a_failed_reload_is_a_502_without_the_key(keyed, service, monkeypatch):
    client, *_ = service

    def down(*_, **__):
        raise requests.ConnectionError('https://api.themoviedb.org/3/movie/348?api_key=SECRET refused')
    monkeypatch.setattr('pipeline.library.metadata_edit.tmdb_details_by_id', down)
    failed = client.post('/v1/titles/a/metadata/tmdb', json={'tmdb_id': '348'}, headers=AUTH)
    assert failed.status_code == 502 and 'SECRET' not in failed.text


def test_no_match_is_said(keyed, service, monkeypatch):
    client, *_ = service
    monkeypatch.setattr('pipeline.library.metadata_edit.tmdb_lookup', lambda *a, **k: BeqMetadata('', ''))
    body = client.post('/v1/titles/a/metadata/tmdb', json={'title': 'Nope', 'tmdb_id': ''}, headers=AUTH).json()
    assert body['found'] is False and 'no match for "Nope"' in body['message']


# --- artwork ------------------------------------------------------------------------------------------------------------

def test_an_image_sent_is_kept_in_the_art_cache_and_served(service):
    client, _, queue, _ = service
    review = _review(client)
    answer = client.put('/v1/titles/a/artwork', json={'digest': review['metadata_digest'],
                                                      'data_base64': base64.b64encode(_png()).decode()}, headers=AUTH)
    assert answer.status_code == 200, answer.text
    assert answer.json()['artwork'] == {'present': True, 'overridden': True, 'name': 'a.png'}
    assert read_entry(queue, 'a').art_path.endswith('_art_cache/a.png')
    served = client.get('/v1/titles/a/artwork', headers=AUTH)
    assert served.status_code == 200 and served.headers['content-type'] == 'image/png' and served.content == _png()


def test_an_image_is_downloaded_from_a_url(service, monkeypatch):
    client, _, queue, _ = service

    class Answer:
        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield _png()
    monkeypatch.setattr('pipeline.library.artwork_edit.requests.get', lambda url, **kw: Answer())
    answer = client.put('/v1/titles/a/artwork', json={'digest': _review(client)['metadata_digest'],
                                                      'url': 'https://example.org/poster.png'}, headers=AUTH)
    assert answer.status_code == 200 and answer.json()['artwork']['present']


@pytest.mark.parametrize('body, says', [({'data_base64': base64.b64encode(b'<html>').decode()}, 'not a png or jpeg'),
                                        ({'data_base64': '***'}, 'not base64'),
                                        ({'url': 'ftp://example.org/a.png'}, 'http:// or https://'),
                                        ({}, 'not both or neither'),
                                        ({'url': 'https://x', 'data_base64': ''}, 'not both or neither')])
def test_what_is_not_an_image_is_refused_and_nothing_is_written(service, body, says):
    client, _, queue, _ = service
    refused = client.put('/v1/titles/a/artwork', json={'digest': _review(client)['metadata_digest'], **body}, headers=AUTH)
    assert refused.status_code == 422 and says in refused.json()['detail']
    assert read_entry(queue, 'a').art_path is None


def test_clearing_forgets_the_poster(service):
    client, _, queue, _ = service
    digest = client.put('/v1/titles/a/artwork', json={'digest': _review(client)['metadata_digest'],
                                                      'data_base64': base64.b64encode(_png()).decode()},
                        headers=AUTH).json()['metadata_digest']
    cleared = client.delete('/v1/titles/a/artwork', params={'digest': digest}, headers=AUTH)
    assert cleared.status_code == 200 and cleared.json()['artwork']['present'] is False
    assert read_entry(queue, 'a').art_path is None
    assert client.get('/v1/titles/a/artwork', headers=AUTH).status_code == 404


# --- revise and the audio stream (WP5) ------------------------------------------------------------------------------------

def _revise(client, **body):
    return client.post('/v1/revise', json=body, headers=AUTH)


def test_titles_are_sent_back_now_each_with_its_outcome(service):
    client, _, queue, _ = service
    update_entry(queue, 'b', status='accepted', chosen_candidate_index=0)
    answer = _revise(client, ids=['b', 'a', 'zz'], to='review', reason='a second look')
    assert answer.status_code == 200, answer.text
    body = answer.json()
    assert [r['id'] for r in body['revised']] == ['b'] and body['job'] is None
    assert {f['id']: f['reason'] for f in body['failed']} == {'a': "'a' is already pending review", 'zz': 'no such title'}
    entry = read_entry(queue, 'b')
    assert entry.status == 'pending' and 'Reopened for review: a second look' in entry.reviewer_note


def test_a_title_sent_back_to_design_waits_for_the_next_run(service):
    client, _, queue, _ = service
    update_entry(queue, 'a', design_fingerprint='old')
    body = _revise(client, ids=['a'], to='design').json()
    assert body['revised'] == [{'id': 'a', 'reverted': [], 'extract_invalidated': False}]
    assert read_entry(queue, 'a').design_fingerprint is None


def test_a_published_title_needs_repository_writes_and_an_undesigned_one_has_nothing_to_send_back(service):
    client, _, queue, _ = service
    update_entry(queue, 'a', status='published', chosen_candidate_index=0)
    import os
    os.remove(os.path.join(queue, 'b.json'))
    body = _revise(client, ids=['a', 'b'], to='design').json()
    reasons = {f['id']: f['reason'] for f in body['failed']}
    assert 'allow_repository_writes' in reasons['a'] and reasons['b'] == 'not designed yet: nothing to send back'
    assert read_entry(queue, 'a').status == 'published'


def test_a_title_a_run_has_in_hand_is_not_sent_back(service):
    client, gate, *_ = service
    gate.report = [_event('a', 'queued', '')]
    client.post('/v1/jobs/scan', json={}, headers=AUTH)
    assert gate.entered.wait(10)
    body = _revise(client, ids=['a'], to='design').json()
    assert body['failed'] == [{'id': 'a', 'reason': RUNNING}]


def test_sending_back_with_a_run_is_one_redesign_job(service):
    client, *_ = service
    body = _revise(client, ids=['a', 'b'], to='extract', run=True).json()
    assert body['revised'] == [] and body['job']['request']['redesign'] is True
    assert body['job']['request']['revise_to'] == 'extract' and body['job']['request']['filter']['ids'] == ['a', 'b']
    refused = _revise(client, ids=['a'], to='review', run=True)
    assert (refused.status_code, refused.json()['title']) == (422, 'Nothing to run')


@pytest.fixture
def streams(setup):
    import sqlite3
    from pipeline.library.index import LibraryIndex, index_path, item_to_json
    from pipeline.library.source import LibraryItem
    _, work, _ = setup
    db = sqlite3.connect(index_path(str(work)))   # the fixture's rows carry no item: give 'a' its one file
    with db:
        db.execute("UPDATE titles SET unit = 'item', items = ? WHERE id = 'a'",
                   (json.dumps([item_to_json(LibraryItem('a', '/m/a.mkv', 'Alien'))]),))
    db.close()
    with LibraryIndex(index_path(str(work))) as index:
        index.set_audio_stream_details('a', ({'codec': 'LPCM', 'channels': '2', 'audio_types': ('LPCM 2.0',)},
                                             {'codec': 'DTS-HD MA', 'channels': '6', 'language': 'eng',
                                              'audio_types': ('DTS-HD MA 5.1',)}))


def test_a_titles_streams_say_which_is_extracted_and_which_is_preferred(streams, service):
    client, *_ = service
    body = client.get('/v1/titles/a/streams', headers=AUTH).json()
    assert body['chosen'] == 1 and body['source'] == ''
    assert [(s['index'], s['preferred']) for s in body['streams']] == [(0, False), (1, True)]
    assert body['streams'][1]['text'].startswith('2: DTS-HD MA 5.1')


def test_choosing_a_stream_is_a_run_job_and_one_not_listed_is_refused(streams, service):
    client, *_ = service
    answer = client.put('/v1/titles/a/stream', json={'audio_stream': 0}, headers=AUTH)
    assert answer.status_code == 202, answer.text
    request = answer.json()['request']
    assert (request['audio_stream'], request['revise_to'], request['filter']['ids']) == (0, 'extract', ['a'])
    refused = client.put('/v1/titles/a/stream', json={'audio_stream': 5}, headers=AUTH)
    assert (refused.status_code, refused.json()['title']) == (422, 'No such stream')
