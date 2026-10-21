'''
Editing a title over HTTP (design/web-parity.md WP4): its metadata, a TMDB reload and its artwork, with the title page's
rules (pipeline.library.metadata_edit and artwork_edit, tested on their own) and the review routes' refusals: a title a
run has in hand, and an edit made against metadata that has changed since.
'''
import base64
import io
import os
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
    kept = read_entry(queue, 'a').art_path
    assert os.path.basename(kept) == 'a.png' and os.path.basename(os.path.dirname(kept)) == '_art_cache'   # any OS's separator
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


# --- adjustments (WP6) ----------------------------------------------------------------------------------------------------

ASKS = [{'kind': 'exclude_band', 'params': {'low_hz': 28, 'high_hz': 34}}, {'kind': 'note', 'text': 'the logo is not the film'}]


def test_sending_back_to_design_keeps_what_the_designer_is_asked_to_change(service):
    client, _, queue, _ = service
    body = _revise(client, ids=['a'], to='design', reason='hum at 30 Hz', adjustments=ASKS).json()
    assert [r['id'] for r in body['revised']] == ['a']
    kept = read_entry(queue, 'a').adjustments
    assert [(a['id'], a['kind'], a['params'], a['text'], a['reason']) for a in kept] == [
        ('a1', 'exclude_band', {'low_hz': 28, 'high_hz': 34}, None, 'hum at 30 Hz'),
        ('a2', 'note', {}, 'the logo is not the film', 'hum at 30 Hz')]
    assert all(a['by'] == 'service' and a['at'] for a in kept)
    assert _review(client)['adjustments'] == kept and _review(client)['adjustment_outcomes'] == []
    _revise(client, ids=['a'], to='design', adjustments=[])
    assert read_entry(queue, 'a').adjustments == []                       # [] asks for none
    _revise(client, ids=['a'], to='design', adjustments=ASKS)
    _revise(client, ids=['a'], to='design')
    assert len(read_entry(queue, 'a').adjustments) == 2                   # absent: kept


def test_with_a_run_they_are_kept_before_the_job_sends_the_titles_back(service):
    client, _, queue, _ = service
    body = _revise(client, ids=['a', 'zz'], to='extract', run=True, adjustments=ASKS[:1]).json()
    assert body['job']['request']['filter']['ids'] == ['a'] and body['failed'] == [{'id': 'zz', 'reason': 'no such title'}]
    assert read_entry(queue, 'a').adjustments[0]['kind'] == 'exclude_band'


@pytest.mark.parametrize('body', [dict(to='review', adjustments=ASKS), dict(to='design', adjustments=[{'kind': ''}]),
                                  dict(to='design', adjustments=[{'params': {}}])])
def test_adjustments_that_cannot_be_asked_are_a_422(service, body):
    client, *_ = service
    assert _revise(client, ids=['a'], **body).status_code == 422


# --- variants (WP7) --------------------------------------------------------------------------------------------------------

@pytest.fixture
def two_runs(service):
    ''' Title a's current design is v2 (asked to cap the boost); v1 before it asked for nothing. '''
    from dataclasses import asdict
    client, _, queue, _ = service
    entry = read_entry(queue, 'a')
    v1 = {name: value for name, value in asdict(entry).items() if name in (
        'fs', 'curve', 'peak_curve', 'audio_stream', 'candidates', 'rejected', 'decline_reason', 'decline_message',
        'design_fingerprint', 'source_fingerprint', 'bass_management', 'designer', 'designer_build', 'design_version',
        'design_settings', 'adjusted_with', 'adjustment_outcomes')}
    v1.update(variant_id='v1', designed_at='2026-10-01T10:00:00+00:00', candidates=v1['candidates'][:1])
    asked = [{'id': 'a1', 'kind': 'max_boost', 'params': {'db': 10}}]
    update_entry(queue, 'a', variant_id='v2', variants=[v1], adjustments=asked, adjusted_with=asked,
                 adjustment_outcomes=[{'id': 'a1', 'outcome': 'honoured', 'applied': None, 'reason': None}])
    return client, queue


def test_a_review_lists_the_other_design_runs_and_what_each_was_asked(two_runs):
    client, _ = two_runs
    review = _review(client)
    assert review['variant_id'] == 'v2' and review['adjusted_with'][0]['kind'] == 'max_boost'
    assert [(v['variant_id'], v['designs'], v['adjusted_with']) for v in review['variants']] == [('v1', 2, [])]   # a candidate and a rejected design


def test_another_run_is_charted_by_name(two_runs):
    client, _ = two_runs
    chart = client.get('/v1/titles/a/chart', params={'variant': 'v1', 'candidates': 'all'}, headers=AUTH).json()
    assert chart['candidates'] == [0, 1] and {s['variant'] for s in chart['series'] if s['filtered']} == {'v1'}
    assert chart['series'][2]['name'].endswith('(v1)')
    assert client.get('/v1/titles/a/chart', params={'variant': 'v9'}, headers=AUTH).status_code == 404


def test_an_earlier_run_is_used_again_and_the_current_one_kept(two_runs):
    client, queue = two_runs
    digest = _review(client)['digest']
    answer = client.post('/v1/titles/a/variants/v1/use', json={'digest': digest}, headers=AUTH)
    assert answer.status_code == 200, answer.text
    body = answer.json()
    assert body['variant_id'] == 'v1' and body['adjustments'] == [] and len(body['candidates']) == 1
    assert [v['variant_id'] for v in body['variants']] == ['v2']
    stale = client.post('/v1/titles/a/variants/v2/use', json={'digest': digest}, headers=AUTH)
    assert (stale.status_code, stale.json()['title']) == (409, 'Changed since it was read')


def test_a_run_is_discarded_but_never_the_current_one(two_runs):
    client, queue = two_runs
    digest = _review(client)['digest']
    refused = client.delete('/v1/titles/a/variants/v2', params={'digest': digest}, headers=AUTH)
    assert (refused.status_code, refused.json()['title']) == (409, 'Not now')
    gone = client.delete('/v1/titles/a/variants/v1', params={'digest': digest}, headers=AUTH)
    assert gone.status_code == 200 and gone.json()['variants'] == [] and read_entry(queue, 'a').variants == []


# --- editing a design (WP8) ----------------------------------------------------------------------------------------------

SHELF = {'type': 'low_shelf', 'freq_hz': 20.0, 'gain_db': 5.0, 'q': 0.7, 'count': 2}


def _edit_design(client, **body):
    return client.put('/v1/titles/a/edited', json=body, headers=AUTH)


def test_an_edit_is_a_new_design_run_and_the_current_one_is_kept(service):
    client, _, queue, _ = service
    before = _review(client)
    answer = _edit_design(client, digest=before['digest'], source=0, filters=[SHELF, {
        'type': 'peaking_eq', 'freq_hz': 35.0, 'gain_db': -2.0, 'q': 3.0}])
    assert answer.status_code == 200, answer.text
    body = answer.json()
    assert body['variant_id'] == 'v2' and [v['variant_id'] for v in body['variants']] == ['v1']
    assert len(body['candidates']) == 1 and body['rejected'] == [] and body['candidates'][0]['method'] == 'edited'
    assert body['candidates'][0]['commentary'] == {'edited_from': 'v1 design 1'}
    entry = read_entry(queue, 'a')
    shelf = entry.candidates[0].filters['filters'][0]
    assert (shelf['_type'], shelf['fc'], shelf['gain'], shelf.get('count')) == ('LowShelf', 20.0, 5.0, 2)
    assert entry.status == 'pending' and entry.fs == before_fs(queue)


def before_fs(queue):
    return read_entry(queue, 'a').variants[0]['fs']


@pytest.mark.parametrize('filters, says', [([dict(SHELF, freq_hz=0)], 'freq_hz must be > 0'),
                                           ([dict(SHELF, q=-1)], 'q must be > 0'),
                                           ([dict(SHELF, freq_hz=600000)], 'below'),
                                           ([dict(SHELF, count=6), dict(SHELF, count=5)], 'exceeds the budget')])
def test_filters_a_designer_could_not_return_are_refused(service, filters, says):
    client, *_ = service
    answer = _edit_design(client, digest=_review(client)['digest'], source=0, filters=filters)
    assert answer.status_code == 422 and says in answer.json()['detail']


def test_an_edit_needs_a_design_to_edit_an_open_title_and_what_was_looked_at(service):
    client, _, queue, _ = service
    digest = _review(client)['digest']
    assert _edit_design(client, digest=digest, source=9, filters=[SHELF]).status_code == 409
    assert _edit_design(client, digest='old', source=0, filters=[SHELF]).json()['title'] == 'Changed since it was read'
    update_entry(queue, 'a', status='accepted', chosen_candidate_index=0)
    refused = _edit_design(client, digest=_review(client)['digest'], source=0, filters=[SHELF])
    assert refused.status_code == 409 and 'reopen it' in refused.json()['detail']
    assert _edit_design(client, digest=digest, source=0, filters=[dict(SHELF, type='all_pass')]).status_code == 422


# --- streams read from the file when the source lists none ------------------------------------------------------------------

PROBED = ({'codec': 'TrueHD', 'channels': '8', 'audio_types': ('TrueHD 7.1',)},
          {'codec': 'AC3', 'channels': '6', 'audio_types': ('DD 5.1',)})


@pytest.fixture
def unlisted(setup):
    ''' Title a's one file, listed with no audio streams (a filesystem source lists none). '''
    import sqlite3
    from pipeline.library.index import index_path, item_to_json
    from pipeline.library.source import LibraryItem
    _, work, _ = setup
    db = sqlite3.connect(index_path(str(work)))
    with db:
        db.execute("UPDATE titles SET unit = 'item', items = ? WHERE id = 'a'",
                   (json.dumps([item_to_json(LibraryItem('a', '/m/a.mkv', 'Alien'))]),))
    db.close()


def test_streams_the_source_did_not_list_are_read_from_the_file(unlisted, service, monkeypatch):
    client, *_ = service
    monkeypatch.setattr('pipeline.service.edits.probe_stream_details', lambda item: PROBED)
    body = client.get('/v1/titles/a/streams', headers=AUTH).json()
    assert body['probed'] is True and [s['text'][:12] for s in body['streams']] == ['1: TrueHD 7.', '2: AC3 5.1']
    answer = client.put('/v1/titles/a/stream', json={'audio_stream': 1}, headers=AUTH)
    assert answer.status_code == 202 and answer.json()['request']['audio_stream'] == 1
    assert client.put('/v1/titles/a/stream', json={'audio_stream': 2}, headers=AUTH).status_code == 422


def test_a_file_that_cannot_be_read_says_why(unlisted, service, monkeypatch):
    client, *_ = service

    def fail(item):
        raise ValueError('/m/a.mkv cannot be found (is the media share mounted here?)')
    monkeypatch.setattr('pipeline.service.edits.probe_stream_details', fail)
    answer = client.get('/v1/titles/a/streams', headers=AUTH)
    assert (answer.status_code, answer.json()['title']) == (409, 'Streams not read')
    assert 'media share mounted' in answer.json()['detail']


# --- the title's projects ---------------------------------------------------------------------------------------------------

def test_the_projects_say_whether_a_person_edited_them_since_the_design(setup, service):
    import os
    from model.iir import CompleteFilter, LowShelf
    from pipeline.config import AnalysisConfig
    from pipeline.library.workdir import entry_directory
    from pipeline.orchestrate import Session
    from pipeline.publish.project import write_mono_project
    from test_pipeline_publish_project import _HUMAN_FILTER, _hand_edit_filter, _write_mono_wav
    client, *_ = service
    _, work, _ = setup
    assert client.get('/v1/titles/a/projects', headers=AUTH).json()['summary'].startswith('No project yet')
    folder = entry_directory(str(work), 'a')
    os.makedirs(folder, exist_ok=True)
    wav = os.path.join(folder, 'mono.wav')
    _write_mono_wav(wav)
    mono = os.path.join(folder, 'a.mono.beq')
    write_mono_project(Session(AnalysisConfig()), wav, CompleteFilter(fs=1000, filters=[LowShelf(1000, 20, 0.7, 4)]), mono)
    body = client.get('/v1/titles/a/projects', headers=AUTH).json()
    assert (body['summary'], body['warn']) == ('Projects as designed', False)
    assert body['projects'] == [{'kind': 'mono', 'name': 'a.mono.beq', 'exists': True, 'edited': False, 'flat': False,
                                 'error': ''}]
    _hand_edit_filter(mono, _HUMAN_FILTER)
    body = client.get('/v1/titles/a/projects', headers=AUTH).json()
    assert body['warn'] and body['summary'] == 'Modified since design: mono project' and body['projects'][0]['edited']
    assert 'what gets published' in body['note']
