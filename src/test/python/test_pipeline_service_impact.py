'''
pipeline/service/impact.py -- which out-of-date designs to redesign, over HTTP (contract 1.6; design/redesign-scope.md
step 4): `POST /v1/impact` asks the designer about the titles out of date, `GET /v1/impact` follows it, and
`POST /v1/reviewed` keeps the ones a person chooses not to redesign. Against a fake 1.6 designer over real HTTP.
'''
import json
import time

import pytest
from fastapi.testclient import TestClient

from fake_impact_designer import CHANGE, FakeImpactDesigner
from pipeline.designer.identity import CurrentIdentity
from pipeline.service.api import create_app
from pipeline.service.config import ServiceConfig
from pipeline.service.impact import ImpactChecker
from pipeline.service.executions import ExecutionQueue
from pipeline.service.models import Check
from pipeline.service.work import executor, job_failed

TOKEN = 'test-token'
AUTH = {'Authorization': f'Bearer {TOKEN}'}
RECORD = 'records/{}-v6.run.json.gz'


def _until(condition, timeout=10.0):
    deadline = time.time() + timeout
    while not condition():
        assert time.time() < deadline, 'timed out'
        time.sleep(0.005)


@pytest.fixture
def designer():
    with FakeImpactDesigner(design_version=7, design_settings='s0',
                            tiers={RECORD.format('alien'): 'likely'}) as fake:
        yield fake


@pytest.fixture
def library(tmp_path, designer, monkeypatch):
    '''
    (client, queue, ids by title, the checker): three films scanned and designed at version 6, Alien and Dune with run
    records, Heat without; the designer now designs at 7.
    '''
    from pipeline.library import index as index_module
    from pipeline.review import update_entry
    from review_entry_fixture import write_entry
    from test_pipeline_library_index import FakeSource, _item
    films = FakeSource([_item('alien', title='Alien', year='1979'), _item('dune', title='Dune', year='2021'),
                        _item('heat', title='Heat', year='1995')])
    monkeypatch.setattr(index_module, 'build_source', lambda kind, settings: films)
    queue = tmp_path / 'queue'
    profile = tmp_path / 'profile.json'
    profile.write_text(json.dumps({
        'sources': [{'name': 'disk', 'kind': 'filesystem', 'globs': [str(tmp_path)]}],
        'designers': {'beqforge': designer.url},
        'run': {'work_dir': str(tmp_path / 'work'), 'queue_dir': str(queue), 'designer': 'beqforge'}}))
    manager = ExecutionQueue(executor(str(profile), env={}), failed=job_failed)
    checker = ImpactChecker(str(tmp_path / 'state'))
    app = create_app(manager, ServiceConfig(profile_path=str(profile), token=TOKEN), env={},
                     checks=lambda: [Check(name='profile', ok=True)], impact_checker=checker)
    with TestClient(app) as client:
        _scan(client)
        ids = {t['title']: t['id'] for t in client.get('/v1/titles', headers=AUTH).json()['titles']}
        for title, title_id in ids.items():
            write_entry(str(queue), title_id)
            update_entry(str(queue), title_id, design_version=6, design_settings='s0',
                         run_record=None if title == 'Heat' else RECORD.format(title.lower()))
        _scan(client)
        yield client, str(queue), ids, checker
    manager.stop(grace_seconds=5)


def _scan(client):
    response = client.post('/v1/jobs/scan', json={}, headers=AUTH)
    assert response.status_code == 202, response.text
    location = response.headers['location']
    _until(lambda: client.get(location, headers=AUTH).json()['state'] not in ('queued', 'running'))


def _checked(client, body=None):
    response = client.post('/v1/impact', json=body or {}, headers=AUTH)
    assert response.status_code == 202, response.text
    _until(lambda: client.get('/v1/impact', headers=AUTH).json()['state'] != 'running')
    return client.get('/v1/impact', headers=AUTH).json()


def _out_of_date(client):
    response = client.get('/v1/titles', params={'out_of_date': True}, headers=AUTH)
    assert response.status_code == 200, response.text
    return sorted(t['id'] for t in response.json()['titles'])


def test_before_any_check_there_is_none(library):
    client, *_ = library
    assert client.get('/v1/impact', headers=AUTH).json()['state'] == 'idle'


def test_the_out_of_date_designs_are_checked_and_read_most_likely_to_move_first(library, designer):
    client, _, ids, _ = library
    check = _checked(client)

    assert check['state'] == 'succeeded', check['error']
    assert (check['design_version'], check['design_settings'], check['done'], check['total']) == (7, 's0', 3, 3)
    assert [(t['id'], t['tier']) for t in check['titles']] == [(ids['Heat'], 'unknown'), (ids['Alien'], 'likely'),
                                                                (ids['Dune'], 'unaffected')]
    assert check['titles'][0]['title']   # named, for display
    assert check['titles'][0]['reasons'][0]['detail'] == 'the designer named no run record for this design'
    assert check['titles'][1]['reasons'][0]['change'] == 'fit-choice' and check['changes'][0]['id'] == 'fit-choice'
    assert check['designer_url'] == designer.url
    sent = designer.asked[0]['records']   # Heat has no record: not sent
    assert sorted(r['path'] for r in sent) == [RECORD.format('alien'), RECORD.format('dune')]
    assert {r['id'] for r in sent} == {ids['Alien'], ids['Dune']}


def test_the_answer_keeps_what_counts_as_moved_its_basis_and_why_a_title_is_unknown(library):
    client, _, ids, checker = library
    check = _checked(client)
    change = check['changes'][0]
    assert change['moved'] == CHANGE['moved'] and change['basis'] == CHANGE['basis']   # 'class', as on the wire
    assert [(t['id'], t['unknown_reason']) for t in check['titles']] == [(ids['Heat'], 'no_record'),
                                                                         (ids['Alien'], None), (ids['Dune'], None)]
    assert ImpactChecker(checker.state_dir).snapshot()['changes'][0]['basis'] == CHANGE['basis']   # kept across a restart


def test_named_titles_alone_are_checked(library, designer):
    client, _, ids, _ = library
    check = _checked(client, {'ids': [ids['Dune']]})
    assert [t['id'] for t in check['titles']] == [ids['Dune']] and len(designer.asked[0]['records']) == 1


def test_a_title_that_does_not_exist_is_a_422(library):
    client, *_ = library
    response = client.post('/v1/impact', json={'ids': ['nope']}, headers=AUTH)
    assert response.status_code == 422 and 'nope' in response.json()['detail']


def test_a_kept_design_is_no_longer_out_of_date_and_a_second_keep_says_why(library):
    client, queue, ids, _ = library
    from pipeline.review import read_entry
    assert _out_of_date(client) == sorted(ids.values())

    response = client.post('/v1/reviewed', json={'ids': [ids['Dune'], 'nope']}, headers=AUTH)

    assert response.status_code == 200, response.text
    outcome = response.json()
    assert (outcome['design_version'], outcome['design_settings']) == (7, 's0')
    assert [(t['reviewed'], t['reason']) for t in outcome['titles']] == [(True, ''), (False, 'no such title')]
    assert read_entry(queue, ids['Dune']).reviewed_at[0]['design_version'] == 7
    _scan(client)
    assert _out_of_date(client) == sorted([ids['Alien'], ids['Heat']])
    dune = client.get(f"/v1/titles/{ids['Dune']}", headers=AUTH).json()
    assert dune['reviewed_at'] == [{'design_version': 7, 'design_settings': 's0'}]
    again = client.post('/v1/reviewed', json={'ids': [ids['Dune']]}, headers=AUTH).json()['titles'][0]
    assert again['reviewed'] is False and 'not out of date' in again['reason']


def test_a_designer_that_does_not_answer_impact_is_a_409_and_nothing_is_asked(library, designer):
    client, *_ = library
    designer.health['impact'] = False
    response = client.post('/v1/impact', json={}, headers=AUTH)
    assert response.status_code == 409 and 'answers /impact' in response.json()['detail']
    assert designer.asked == []


def test_with_nothing_out_of_date_there_is_nothing_to_check(library, designer):
    client, *_ = library
    designer.health['design_version'] = 6
    response = client.post('/v1/impact', json={}, headers=AUTH)
    assert response.status_code == 409 and 'out of date' in response.json()['detail']
    assert client.post('/v1/reviewed', json={'ids': ['x']}, headers=AUTH).json()['titles'][0]['reviewed'] is False


def test_an_answer_out_of_shape_fails_the_check_and_says_why(library, designer):
    client, *_ = library
    designer.mangle = lambda answer: {**answer, 'results': answer['results'][::-1]}
    check = _checked(client)
    assert check['state'] == 'failed' and 'order asked' in check['error'] and check['titles'] == []


def test_an_answer_for_another_identity_than_the_one_a_redesign_would_use_fails(library, designer):
    client, *_ = library
    designer.mangle = lambda answer: {**answer, 'design_settings': 's9'}
    check = _checked(client)
    assert check['state'] == 'failed' and 's9' in check['error']


def test_one_check_at_a_time(library):
    client, _, _, checker = library
    checker.state = {'state': 'running', 'started_at': 0}
    response = client.post('/v1/impact', json={}, headers=AUTH)
    assert response.status_code == 409 and 'running already' in response.json()['detail']


def test_the_last_check_is_kept_across_a_restart_and_one_cut_short_says_so(tmp_path):
    def ask(url, records, **kwargs):
        from pipeline.designer.impact import ImpactAnswer, ImpactResult
        return ImpactAnswer(7, 's0', 'b', (), tuple(ImpactResult(i, 'possible', 0.3, 6) for i, _ in records))
    state = str(tmp_path)
    first = ImpactChecker(state, ask=ask, run=lambda work: work())
    first.start('u', None, CurrentIdentity(7, 's0'), [('a', 'records/a')], ['b'], {'a': ('A', '2000')})

    again = ImpactChecker(state).snapshot()
    assert again['state'] == 'succeeded' and [t['id'] for t in again['titles']] == ['b', 'a']
    assert again['titles'][1]['title'] == 'A'

    cut = ImpactChecker(state, run=lambda work: None)   # never finishes: the service stops part way
    cut.start('u', None, CurrentIdentity(7, 's0'), [('a', 'records/a')], [], {})
    restarted = ImpactChecker(state).snapshot()
    assert restarted['state'] == 'failed' and 'restarted' in restarted['error']
