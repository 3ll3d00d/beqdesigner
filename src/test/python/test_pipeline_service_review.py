'''
Reviewing a title over HTTP (design/web-app.md §3): the review, chart, decision and next routes, over a fixture index and
real queue entries, and the in-flight rule that keeps a decision off a title a run is working on. The decision itself is
`pipeline.library.decide`, covered on its own in test_pipeline_library_decide.py; these tests hold the routes to it.
'''
import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from model.execution_events import ExecutionEvent
from pipeline.review import read_entry
from pipeline.service.api import create_app
from pipeline.service.config import ServiceConfig
from pipeline.service.jobs import JobManager
from pipeline.service.lease import lease_path
from pipeline.service.models import Check
from review_entry_fixture import write_entry
from worklist_fixture import make_index, title_row
from test_pipeline_library_index import env as scanned  # noqa: F401 (a fixture: an index a scan can fill)

TOKEN = 'test-token'
AUTH = {'Authorization': f'Bearer {TOKEN}'}


def _waiting(title_id, title='', **fields):
    ''' A row the index says is waiting for a person: designed, and its entry pending. '''
    return title_row(title_id, title, needs='review', extract_state='done', design_state='done', review_state='pending',
                     **fields)


@pytest.fixture
def setup(tmp_path):
    ''' (profile path, work dir, queue dir), with an index of four titles: a, b and d waiting, c needing design. '''
    work, queue, media = tmp_path / 'work', tmp_path / 'queue', tmp_path / 'media'
    for folder in (work, queue, media):
        folder.mkdir()
    config = {'sources': [{'name': 'films', 'kind': 'filesystem', 'globs': [str(media)]}],
              'designers': {'rolloff': 'http://127.0.0.1:9/design'},
              'run': {'work_dir': str(work), 'queue_dir': str(queue), 'designer': 'rolloff'}}
    profile = tmp_path / 'profile.json'
    profile.write_text(json.dumps(config))
    make_index(work, [_waiting('a', 'Alien'), _waiting('b', 'Brazil'),
                      title_row('c', 'Cube', needs='design', extract_state='done', design_state='stale',
                                review_state='pending', detail='settings changed'),
                      _waiting('d', 'Dune')])
    for title_id in 'abcd':
        write_entry(str(queue), title_id, rejected=1 if title_id == 'a' else 0)
    return profile, work, queue


class Gate:
    ''' A job that reports what it is told to, then waits to be let go: a run with titles in hand. '''

    def __init__(self):
        self.entered, self.release, self.report = threading.Event(), threading.Event(), []

    def __call__(self, job, control):
        for item in self.report:
            (control.event if isinstance(item, ExecutionEvent) else control.progress)(item)
        self.entered.set()
        assert self.release.wait(10)


@pytest.fixture
def service(setup):
    ''' (client, gate, queue dir, profile path). '''
    profile, _, queue = setup
    gate = Gate()
    manager = JobManager(gate)
    app = create_app(manager, ServiceConfig(profile_path=str(profile), token=TOKEN), env={},
                     checks=lambda: [Check(name='profile', ok=True)])
    with TestClient(app) as client:
        yield client, gate, str(queue), profile
    gate.release.set()
    manager.stop(grace_seconds=5)


def _event(title_id, kind, stage='design'):
    return ExecutionEvent('run', title_id, stage, kind, time.time(), message=title_id)


def _review(client, title_id='a'):
    response = client.get(f'/v1/titles/{title_id}/review', headers=AUTH)
    assert response.status_code == 200, response.text
    return response.json()


def _decide(client, title_id='a', **body):
    return client.post(f'/v1/titles/{title_id}/decision', json=body, headers=AUTH)


# --- reading a review ---------------------------------------------------------------------------------------------------

def test_a_review_shows_the_designs_the_rejected_ones_after_them_and_what_is_offered(service):
    client, *_ = service

    review = _review(client)

    assert (review['title'], review['year'], review['status'], review['status_text']) == \
        ('Alien', '2001', 'pending', 'Waiting for a decision')
    assert [(c['index'], c['rejected'], c['method']) for c in review['candidates']] == [(0, False, 'fitted'), (1, False, 'fitted')]
    assert [(c['index'], c['rejected']) for c in review['rejected']] == [(2, True)]
    assert review['rejected'][0]['rejection_reasons'][0].startswith('introduces a cliff')
    assert review['candidates'][0]['commentary'] == {'note': 'top pick'}
    assert review['candidates'][0]['filters']['filters']     # the .filter document
    assert review['blocked'] == {'accept': '', 'reject': ''} and review['in_flight'] is False
    assert review['metadata_problems'] == [] and review['metadata']['title'] == 'a'
    assert review['playback'].startswith('none sent') and review['declined'] is None and len(review['digest']) == 32


def test_a_title_that_needs_design_offers_reject_but_not_accept_and_says_where_to_run_it(service):
    client, *_ = service
    review = _review(client, 'c')
    assert review['blocked']['accept'] == ('Not offered: this title needs design first (settings changed). '
                                           'Run it through design first (POST /v1/jobs/run).')
    assert review['blocked']['reject'] == ''


def test_incomplete_metadata_is_listed_and_holds_accept_back(service):
    client, _, queue, _ = service
    write_entry(queue, 'b', meta={'title': 'b'})

    review = _review(client, 'b')

    assert review['metadata_problems']
    assert review['blocked']['accept'].startswith('Fill in the missing metadata first (in the BEQDesigner app): ')
    refused = _decide(client, 'b', decision='accept', candidate=0, digest=review['digest'])
    assert (refused.status_code, refused.json()['title']) == (409, 'Metadata incomplete')
    assert read_entry(queue, 'b').status == 'pending'


def test_a_decided_title_says_so_and_offers_nothing(service):
    client, _, queue, _ = service
    write_entry(queue, 'b', status='accepted', chosen=1)

    review = _review(client, 'b')

    assert (review['status'], review['chosen_index']) == ('accepted', 1)
    assert review['blocked'] == {'accept': 'This title is accepted.', 'reject': 'This title is accepted.'}


def test_a_decline_is_shown_as_its_reason_and_one_flat_design(service):
    client, _, queue, _ = service
    write_entry(queue, 'b', decline=True)
    review = _review(client, 'b')
    assert review['declined'] == {'reason': 'no_rolloff_detected', 'message': 'nothing found'}
    assert [c['confidence'] for c in review['candidates']] == [None]


def test_an_unknown_title_is_404_and_one_with_no_design_is_409(service, setup):
    client, _, queue, _ = service
    assert client.get('/v1/titles/nope/review', headers=AUTH).status_code == 404
    (setup[2] / 'b.json').unlink()
    response = client.get('/v1/titles/b/review', headers=AUTH)
    assert (response.status_code, response.json()['title']) == (409, 'Not designed')
    assert _decide(client, 'b', decision='reject', digest='x').status_code == 409


@pytest.mark.parametrize('method, path', [('get', '/v1/titles/a/review'), ('get', '/v1/titles/a/chart'),
                                          ('post', '/v1/titles/a/decision'), ('get', '/v1/review/next')])
def test_the_review_routes_need_the_token(service, method, path):
    client, *_ = service
    assert getattr(client, method)(path).status_code == 401


# --- deciding -----------------------------------------------------------------------------------------------------------

def test_accepting_writes_the_design_and_answers_with_the_review_as_it_now_is(service):
    client, _, queue, _ = service
    digest = _review(client)['digest']

    response = _decide(client, decision='accept', candidate=1, digest=digest)

    assert response.status_code == 200, response.text
    assert (response.json()['status'], response.json()['chosen_index']) == ('accepted', 1)
    assert response.json()['blocked']['reject'] == 'This title is accepted.'
    entry = read_entry(queue, 'a')
    assert (entry.status, entry.chosen_candidate_index) == ('accepted', 1)
    again = _decide(client, decision='reject', digest=digest)
    assert (again.status_code, again.json()['title'], again.json()['detail']) == \
        (409, 'Changed since it was read', 'Not changed: this title is accepted now.')


def test_rejecting_needs_no_design(service):
    client, _, queue, _ = service
    response = _decide(client, decision='reject', digest=_review(client)['digest'])
    assert response.status_code == 200 and read_entry(queue, 'a').status == 'rejected'


def test_a_redesign_after_the_review_was_read_refuses_the_decision(service):
    client, _, queue, _ = service
    digest = _review(client, 'b')['digest']
    write_entry(queue, 'b', reverse=True)

    response = _decide(client, 'b', decision='accept', candidate=0, digest=digest)

    assert (response.status_code, response.json()['title']) == (409, 'Changed since it was read')
    assert 'changed while it was open' in response.json()['detail']
    assert read_entry(queue, 'b').status == 'pending'


def test_a_rejected_design_is_accepted_only_with_the_override(service):
    client, _, queue, _ = service
    digest = _review(client)['digest']

    refused = _decide(client, decision='accept', candidate=2, digest=digest)
    assert (refused.status_code, refused.json()['title']) == (409, 'The designer rejected this design')

    accepted = _decide(client, decision='accept', candidate=2, digest=digest, override_rejection=True)
    assert accepted.status_code == 200 and read_entry(queue, 'a').overrides_rejection


@pytest.mark.parametrize('body', [
    {'decision': 'accept', 'digest': 'x'},                                  # no design chosen
    {'decision': 'accept', 'candidate': 9, 'digest': 'DIGEST'},             # no such design
    {'decision': 'approve', 'digest': 'x'},                                 # no such decision
    {'decision': 'reject', 'digest': 'x', 'candidte': 1},                   # misspelt: never silently ignored
    {'decision': 'accept', 'candidate': -1, 'digest': 'x'},
])
def test_an_invalid_decision_is_a_422_and_writes_nothing(service, body):
    client, _, queue, _ = service
    if body.get('digest') == 'DIGEST':
        body = dict(body, digest=_review(client)['digest'])
    response = _decide(client, **body)
    assert response.status_code == 422, response.text
    assert response.headers['content-type'] == 'application/problem+json'
    assert read_entry(queue, 'a').status == 'pending'


# --- a run going --------------------------------------------------------------------------------------------------------

def test_nothing_is_decided_on_a_title_a_run_has_in_hand_until_its_part_ends(service):
    client, gate, queue, _ = service
    gate.report = [_event('a', 'queued', ''), _event('b', 'queued', ''), _event('b', 'title_completed', '')]
    digest = _review(client)['digest']
    job = client.post('/v1/jobs/scan', json={}, headers=AUTH).json()
    assert gate.entered.wait(10)

    review = _review(client)
    assert review['in_flight'] is True
    assert review['blocked']['reject'] == 'A run is working on this title now: wait for it to finish.'
    refused = _decide(client, decision='reject', digest=digest)
    assert (refused.status_code, refused.json()['title']) == (409, 'Not offered now')
    assert _review(client, 'b')['in_flight'] is False      # its part of the run is over
    assert client.get('/v1/review/next', params={'after': 'd'}, headers=AUTH).json()['id'] == 'b'   # a is not next

    gate.release.set()
    deadline = time.time() + 10
    while client.get(f"/v1/jobs/{job['id']}", headers=AUTH).json()['state'] == 'running':
        assert time.time() < deadline
        time.sleep(0.01)
    assert _decide(client, decision='reject', digest=digest).status_code == 200


def test_a_progress_report_naming_a_title_puts_it_in_hand(service):
    from pipeline.library.stages import Progress
    client, gate, *_ = service
    gate.report = [Progress(0, 2, 'Dune', 'design', 'd')]
    client.post('/v1/jobs/scan', json={}, headers=AUTH)
    assert gate.entered.wait(10)
    assert _review(client, 'd')['in_flight'] is True and _review(client, 'b')['in_flight'] is False


def test_a_run_elsewhere_holding_the_lease_keeps_decisions_off_titles_that_need_work(service, setup):
    client, *_ = service
    work = setup[1]
    (work / 'service').mkdir(exist_ok=True)
    pathlib_lease = work / 'service' / 'lease.json'
    assert str(pathlib_lease) == lease_path(str(work))
    pathlib_lease.write_text(json.dumps({'host': 'desk', 'pid': 1, 'job_id': 'worklist-1', 'heartbeat_at': time.time()}))

    assert _review(client, 'c')['in_flight'] is True       # needs design: the work list's run may be designing it
    assert _review(client, 'a')['in_flight'] is False      # needs review: no run touches it


# --- the chart ----------------------------------------------------------------------------------------------------------

def test_the_chart_is_the_measured_curves_then_each_after_the_design_asked_for(service):
    client, *_ = service

    measured = client.get('/v1/titles/a/chart', headers=AUTH).json()
    designed = client.get('/v1/titles/a/chart', params={'candidate': 0}, headers=AUTH).json()

    assert measured['candidate'] is None
    assert [(s['kind'], s['filtered']) for s in measured['series']] == [('average', False), ('peak', False)]
    assert [(s['kind'], s['filtered']) for s in designed['series']] == \
        [('average', False), ('peak', False), ('average', True), ('peak', True)]
    assert designed['series'][2]['name'] == 'Filtered average audio track (all channels mixed)'
    average = designed['series'][0]
    assert len(average['x']) == len(average['y']) == 50 and set(average['y']) == {0.0}
    assert max(designed['series'][2]['y']) > 0                # the low shelf lifts the bottom
    assert client.get('/v1/titles/a/chart', params={'candidate': 3}, headers=AUTH).status_code == 422


# --- the next title -----------------------------------------------------------------------------------------------------

def test_next_is_the_next_waiting_title_in_the_filtered_list_round_to_the_top(service):
    client, _, queue, _ = service
    write_entry(queue, 'b', status='rejected')       # decided since the index was read: the entry has the last word

    def next_after(after=None, **params):
        response = client.get('/v1/review/next', params={**({'after': after} if after else {}), **params}, headers=AUTH)
        return response.json()['id'] if response.status_code == 200 else response.status_code

    assert next_after() == 'a'
    assert next_after('a') == 'd'                     # not b (decided) nor c (needs design)
    assert next_after('d') == 'a'                     # round to the top
    assert next_after('a', match='alien') == 204      # nothing else in this list
    assert client.get('/v1/review/next', params={'source': 'nope'}, headers=AUTH).status_code == 422


# --- what the app can offer ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('writes', [False, True])
def test_the_status_says_whether_publish_and_commit_can_be_offered(setup, writes):
    profile = setup[0]
    manager = JobManager(lambda job, control: None)
    try:
        app = create_app(manager, ServiceConfig(profile_path=str(profile), token=TOKEN, allow_repository_writes=writes),
                         env={}, checks=lambda: [])
        with TestClient(app) as client:
            status = client.get('/v1/status', headers=AUTH).json()
            assert (status['repository_writes'], status['repositories_configured']) == (writes, False)
            config = json.loads(profile.read_text())
            config['sync'] = {'xml_repo': str(profile.parent / 'xml')}
            profile.write_text(json.dumps(config))
            assert client.get('/v1/status', headers=AUTH).json()['repositories_configured'] is True
    finally:
        manager.stop(grace_seconds=5)


# --- the index learns of a decision -------------------------------------------------------------------------------------

class FakeRefresher:
    def __init__(self):
        self.requests = 0

    def request(self):
        self.requests += 1


def test_a_written_decision_asks_for_the_index_to_be_refreshed_and_a_refused_one_does_not(setup):
    profile = setup[0]
    refresher, manager = FakeRefresher(), JobManager(lambda job, control: None)
    try:
        app = create_app(manager, ServiceConfig(profile_path=str(profile), token=TOKEN), env={}, checks=lambda: [],
                         refresher=refresher)
        with TestClient(app) as client:
            digest = _review(client)['digest']
            assert _decide(client, decision='reject', digest='stale').status_code == 409
            assert refresher.requests == 0
            assert _decide(client, decision='reject', digest=digest).status_code == 200
            assert refresher.requests == 1
    finally:
        manager.stop(grace_seconds=5)


@pytest.fixture
def refreshes(setup, monkeypatch):
    ''' (refresher, calls, busy): an IndexRefresher over the fixture profile whose LibraryIndex.refresh is recorded. '''
    from pipeline.library.index import LibraryIndex
    from pipeline.service.context import load_context
    from pipeline.service.refresh import IndexRefresher
    calls, busy = [], {'now': False}

    def record(index, profile, settings=None):
        calls.append(settings.queue_dir)
        time.sleep(0.05)      # long enough for requests to arrive while it goes
    monkeypatch.setattr(LibraryIndex, 'refresh', record)
    refresher = IndexRefresher(lambda: load_context(str(setup[0]), {}), busy=lambda: busy['now'])
    return refresher, calls, busy


def test_the_index_is_refreshed_off_the_request_and_requests_meanwhile_fold_into_one(refreshes, setup):
    refresher, calls, _ = refreshes

    for _ in range(5):
        refresher.request()

    assert refresher.wait_idle(10)
    assert calls[0] == str(setup[2]) and 1 <= len(calls) <= 2 and refresher.refreshed == len(calls)


def test_a_refresh_is_left_to_a_run_going_here_or_elsewhere(refreshes, setup):
    refresher, calls, busy = refreshes
    busy['now'] = True
    assert refresher.refresh_now() is False

    busy['now'] = False
    work = setup[1]
    (work / 'service').mkdir(exist_ok=True)
    (work / 'service' / 'lease.json').write_text(
        json.dumps({'host': 'desk', 'pid': 1, 'job_id': 'cli-1', 'heartbeat_at': time.time()}))
    assert refresher.refresh_now() is False

    (work / 'service' / 'lease.json').unlink()
    assert refresher.refresh_now() is True
    assert (len(calls), refresher.skipped) == (1, 2)



def test_a_decision_reaches_a_scanned_index_through_the_refresher(scanned):
    ''' Through a real scan (the fixture index above has no listing to refresh from): accepted, the title needs publish. '''
    from types import SimpleNamespace

    from pipeline.library.decide import decide, offered_digest
    from pipeline.service.refresh import IndexRefresher
    from test_pipeline_library_index import _entry, _extracted, _item, _needs, _profile, _scan
    env = scanned
    item = _item('a')
    _scan(env, item)
    _extracted(env, item)
    _entry(env, item)
    env.index.refresh(_profile(env), env.settings)
    assert _needs(env, item.id)[0] == 'review'

    decide(env.queue, item.id, 'accept', picked=0, seen_digest=offered_digest(read_entry(env.queue, item.id)))
    ctx = SimpleNamespace(work_dir=env.work, profile=_profile(env), scan_settings=lambda: env.settings)
    assert IndexRefresher(lambda: ctx, busy=lambda: False).refresh_now() is True

    assert _needs(env, item.id)[0] == 'publish'
