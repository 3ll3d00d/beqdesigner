'''
pipeline/service/api.py (design/pipeline-service.md §6): the HTTP interface, driven through FastAPI's TestClient over a real
profile, index and ExecutionQueue (run_stages faked), and its published OpenAPI document.
'''
import json
import pathlib
import re
import threading
import time

import pytest
from fastapi.testclient import TestClient

from pipeline.library.stages import Progress, StagesReport
from pipeline.service import work
from pipeline.service.api import create_app, openapi_document
from pipeline.service.config import ServiceConfig
from pipeline.service.executions import ExecutionQueue
from pipeline.service.models import Check
from pipeline.service.work import executor, job_failed
from service_http import first_execution, make_job

TOKEN = 'test-token'
AUTH = {'Authorization': f'Bearer {TOKEN}'}
PUBLISHED = pathlib.Path(__file__).parents[3] / 'docs' / 'schema' / 'service.openapi.json'


def _until(condition, timeout=10.0):
    deadline = time.time() + timeout
    while not condition():
        assert time.time() < deadline, 'timed out'
        time.sleep(0.005)


@pytest.fixture(autouse=True)
def designer_answers(monkeypatch):
    ''' The profile's designer (port 9, nothing there) answers /health: these tests are not about the designer. '''
    monkeypatch.setattr('pipeline.service.context.check_designer', lambda *args, **kwargs: None)


@pytest.fixture
def profile(tmp_path):
    media = tmp_path / 'media'
    media.mkdir()
    for name in ('Alien (1979).mkv', 'Dune (2021).mkv', 'Heat (1995).mkv'):
        (media / name).write_bytes(b'')
    config = {'sources': [{'name': 'disk', 'kind': 'filesystem', 'globs': [str(media)]}],
              'designers': {'rolloff': 'http://127.0.0.1:9/design'},
              'run': {'work_dir': str(tmp_path / 'work'), 'queue_dir': str(tmp_path / 'queue'), 'designer': 'rolloff'}}
    path = tmp_path / 'profile.json'
    path.write_text(json.dumps(config))
    return path


@pytest.fixture
def service(profile, monkeypatch):
    '''
    (client, manager, calls): the app over a real ExecutionQueue; run_stages is recorded, not run. The profile's source lists
    three films with years (a filesystem source reports none: a title has one from JRiver, or once TMDB has named it).
    '''
    from pipeline.library import index as index_module
    from test_pipeline_library_index import FakeSource, _item
    films = FakeSource([_item('alien', title='Alien', year='1979'), _item('dune', title='Dune', year='2021'),
                        _item('heat', title='Heat', year='1995')])
    monkeypatch.setattr(index_module, 'build_source', lambda kind, settings: films)
    calls = []

    def fake_run_stages(profile_, selection, through, **kwargs):
        calls.append((selection, through))
        kwargs['on_progress'](Progress(1, 1, 'Alien', 'design', 'x'))
        return StagesReport(through, 1, attempted=['x'])
    monkeypatch.setattr(work, 'run_stages', fake_run_stages)
    manager = ExecutionQueue(executor(str(profile), env={}), failed=job_failed)
    app = create_app(manager, ServiceConfig(profile_path=str(profile), token=TOKEN), env={},
                     checks=lambda: [Check(name='profile', ok=True)])
    with TestClient(app) as client:
        yield client, manager, calls
    manager.stop(grace_seconds=5)


def _finish(client, response):
    ''' The first execution of the job the response made, once it has finished. '''
    return first_execution(client, AUTH, response)


def _scanned(client):
    job = _finish(client, make_job(client, AUTH, 'scan', {}))
    assert job['state'] == 'succeeded' and job['result']['titles'] == 3
    return job


def test_a_recurring_job_is_made_edited_paused_and_run_now(service):
    client, manager, calls = service
    invalid = make_job(client, AUTH, 'run', {'through': 'publish'}, every=5)
    assert invalid.status_code == 422 and 'recurring' in invalid.json()['detail']
    assert make_job(client, AUTH, 'run', {}, every=4).status_code == 422
    made = make_job(client, AUTH, 'run', {'filter': {'kind': 'movie', 'needs': ['extract', 'design']}}, every=30,
                    name='Nightly')
    assert made.status_code == 201 and made.headers['location'] == f"/v1/jobs/{made.json()['id']}"
    job = made.json()
    assert (job['name'], job['every'], job['state'], job['executions']) == ('Nightly', 30, 'active', 0)
    assert job['next_at'] and job['progress'] is None and job['running_execution'] is None

    edited = client.patch(f"/v1/jobs/{job['id']}", json={'every': 60, 'window': ['01:00-02:00']}, headers=AUTH).json()
    assert (edited['every'], edited['window'], edited['name']) == (60, ['01:00-02:00'], 'Nightly')
    assert client.patch(f"/v1/jobs/{job['id']}", json={'window': None}, headers=AUTH).json()['window'] is None
    assert client.post(f"/v1/jobs/{job['id']}/pause", headers=AUTH).json()['state'] == 'paused'
    assert client.post(f"/v1/jobs/{job['id']}/resume", headers=AUTH).json()['state'] == 'active'

    ran = client.post(f"/v1/jobs/{job['id']}/run", headers=AUTH)
    assert ran.status_code == 202 and ran.headers['location'] == f"/v1/executions/{ran.json()['id']}"
    _until(lambda: client.get(ran.headers['location'], headers=AUTH).json()['state'] == 'succeeded')
    execution = client.get(ran.headers['location'], headers=AUTH).json()
    assert (execution['job_id'], execution['job_name'], execution['number'], execution['origin']) == \
        (job['id'], 'Nightly', 1, 'schedule')
    assert execution['request']['filter']['needs'] == ['extract', 'design'] and calls[-1][1] == 'design'
    assert client.get(f"/v1/jobs/{job['id']}", headers=AUTH).json()['last_execution'] == execution['id']

    removed = client.delete(f"/v1/jobs/{job['id']}", headers=AUTH)
    assert removed.status_code == 200 and removed.json()['state'] == 'removed'
    refused = client.post(f"/v1/jobs/{job['id']}/resume", headers=AUTH)
    assert refused.status_code == 409 and refused.json()['title'] == 'Not now'
    assert client.get('/v1/jobs/nope', headers=AUTH).status_code == 404


def test_run_now_is_409_while_another_execution_runs_and_a_one_offs_request_is_fixed(profile):
    release = threading.Event()
    running = threading.Event()

    def execute(job, control):
        running.set()
        release.wait(5)

    manager = ExecutionQueue(execute)
    app = create_app(manager, ServiceConfig(profile_path=str(profile), token=TOKEN), env={})
    try:
        with TestClient(app) as client:
            schedule = make_job(client, AUTH, 'run', {}, every=5).json()
            one_off = make_job(client, AUTH, 'scan', {})
            assert one_off.status_code == 201 and running.wait(5)
            busy = client.post(f"/v1/jobs/{schedule['id']}/run", headers=AUTH)
            assert busy.status_code == 409 and busy.json()['title'] == 'Service busy'
            fixed = client.patch(f"/v1/jobs/{one_off.json()['id']}", json={'every': 10}, headers=AUTH)
            assert fixed.status_code == 409 and 'fixed' in fixed.json()['detail']
            status = client.get('/v1/status', headers=AUTH).json()
            assert status['running_execution']['job_id'] == one_off.json()['id'] and status['waiting'] == 0
    finally:
        release.set()
        manager.stop(5)


# --- the unauthenticated routes -----------------------------------------------------------------------------------------

def test_health_and_readiness_need_no_token(service):
    client, _, _ = service

    assert client.get('/health').json()['status'] == 'ok'
    assert client.get('/ready').json() == {'ready': True, 'checks': [{'name': 'profile', 'ok': True, 'detail': '', 'required': True}]}


def test_not_ready_is_a_503_that_says_why(profile):
    manager = ExecutionQueue(lambda job, control: None)
    app = create_app(manager, ServiceConfig(profile_path=str(profile), token=TOKEN),
                     checks=lambda: [Check(name='ffmpeg', ok=False, detail='not found: ffmpeg')])
    try:
        response = TestClient(app).get('/ready')
    finally:
        manager.stop(1)

    assert response.status_code == 503 and response.json()['checks'][0]['detail'] == 'not found: ffmpeg'


def test_the_default_readiness_checks_the_profile_work_dir_ffmpeg_and_designer(profile):
    manager = ExecutionQueue(lambda job, control: None)
    app = create_app(manager, ServiceConfig(profile_path=str(profile), token=TOKEN), env={})
    try:
        checks = {c['name']: c for c in TestClient(app).get('/ready').json()['checks']}
    finally:
        manager.stop(1)

    assert list(checks) == ['profile', 'work_dir', 'ffmpeg', 'designer', 'tmdb', 'designer_reachable']
    assert checks['tmdb'] == {'name': 'tmdb', 'ok': False, 'required': False,
                              'detail': "TMDB_API_KEY is not set: titles are designed with the library's own metadata only"}
    assert checks['profile']['ok'] and checks['work_dir']['ok'] and checks['designer'] == \
        {'name': 'designer', 'ok': True, 'detail': 'rolloff', 'required': True}
    assert checks['designer_reachable'] == {'name': 'designer_reachable', 'ok': True,
                                            'detail': 'http://127.0.0.1:9/design', 'required': False}


def test_the_docs_and_the_document_are_open_and_try_it_out_is_on(service):
    client, _, _ = service

    page = client.get('/docs')
    assert page.status_code == 200 and '"tryItOutEnabled": true' in page.text and 'persistAuthorization' in page.text
    assert client.get('/redoc').status_code == 200
    assert client.get('/openapi.json').json()['openapi'] == '3.1.0'


def test_the_docs_use_a_local_copy_of_their_scripts_when_there_is_one(tmp_path, profile):
    (tmp_path / 'static').mkdir()
    (tmp_path / 'static' / 'swagger-ui-bundle.js').write_text('// swagger')
    manager = ExecutionQueue(lambda job, control: None)
    app = create_app(manager, ServiceConfig(profile_path=str(profile), token=TOKEN), static_dir=str(tmp_path / 'static'))
    try:
        client = TestClient(app)
        page, script = client.get('/docs').text, client.get('/static/swagger-ui-bundle.js')
    finally:
        manager.stop(1)

    assert '/static/swagger-ui-bundle.js' in page and 'cdn.jsdelivr' not in page and script.text == '// swagger'


# --- authentication -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('headers', [{}, {'Authorization': 'Bearer wrong'}, {'Authorization': f'Basic {TOKEN}'}])
def test_every_v1_route_needs_the_token(service, headers):
    client, _, _ = service

    response = client.get('/v1/status', headers=headers)

    assert response.status_code == 401 and response.headers['www-authenticate'] == 'Bearer'
    assert response.headers['content-type'] == 'application/problem+json' and response.json()['status'] == 401


def test_a_token_is_required_unless_started_without_auth(profile):
    manager = ExecutionQueue(lambda job, control: None)
    try:
        with pytest.raises(ValueError, match='BEQ_SERVICE_TOKEN'):
            create_app(manager, ServiceConfig(profile_path=str(profile)))
        open_app = create_app(manager, ServiceConfig(profile_path=str(profile)), require_token=False, env={})
        assert TestClient(open_app).get('/v1/status').status_code == 200
    finally:
        manager.stop(1)


# --- titles and plans ---------------------------------------------------------------------------------------------------

def test_before_a_scan_titles_say_to_scan_first(service):
    client, _, _ = service

    response = client.get('/v1/titles', headers=AUTH)

    assert response.status_code == 409 and 'POST /v1/jobs' in response.json()['detail']
    assert client.get('/v1/status', headers=AUTH).json()['index'] is None


def test_titles_are_filtered_as_the_work_list_filters_them_and_paged(service):
    client, _, _ = service
    _scanned(client)

    def titles(**params):
        page = client.get('/v1/titles', params=params, headers=AUTH).json()
        return sorted(t['title'] for t in page['titles'])

    assert titles() == ['Alien', 'Dune', 'Heat']
    assert titles(year='>=1990') == ['Dune', 'Heat'] and titles(year='1979') == ['Alien'] and titles(kind='tv') == []
    assert titles(needs=['extract'], kind='movie', year='1990-2020') == ['Heat']
    assert titles(match='dun') == ['Dune']
    page = client.get('/v1/titles', params={'limit': 2, 'offset': 2}, headers=AUTH).json()
    assert (page['total'], len(page['titles']), page['offset']) == (3, 1, 2)
    one = page['titles'][0]
    assert client.get(f"/v1/titles/{one['id']}", headers=AUTH).json() == one
    assert client.get('/v1/titles/nope', headers=AUTH).status_code == 404


def test_titles_are_sorted_by_any_column_and_filtered_on_detail_and_flags(service):
    client, _, _ = service
    _scanned(client)

    def titles(**params):
        return [t['title'] for t in client.get('/v1/titles', params=params, headers=AUTH).json()['titles']]

    assert titles(sort='title') == ['Alien', 'Dune', 'Heat']
    assert titles(sort='title', descending=True) == ['Heat', 'Dune', 'Alien']
    years = [t['year'] for t in client.get('/v1/titles', params={'sort': 'year'}, headers=AUTH).json()['titles']]
    assert years == sorted(years)
    # paging is over the sorted list, not each page sorted on its own
    assert titles(sort='title', descending=True, limit=1, offset=1) == ['Dune']
    every = client.get('/v1/titles', headers=AUTH).json()['titles']
    word = every[0]['detail'].split()[0]
    assert titles(detail=word.upper()) == [t['title'] for t in every if word.casefold() in t['detail'].casefold()]
    assert titles(detail='no such words') == []
    assert titles(flags=['Gone']) == []
    assert titles(declined=True) == []   # the fixture's titles are not designed yet
    assert sorted(titles(declined=False)) == ['Alien', 'Dune', 'Heat']
    out_of_date = client.get('/v1/titles', params={'out_of_date': True}, headers=AUTH)
    assert out_of_date.status_code == 409 and 'design version' in out_of_date.json()['detail']   # nobody says one
    assert client.get('/v1/titles', params={'sort': 'colour'}, headers=AUTH).status_code == 422
    assert client.get('/v1/titles', params={'flags': 'purple'}, headers=AUTH).status_code == 422


@pytest.mark.parametrize('params, where', [({'year': '2026-2020'}, 'year'), ({'year': '20s'}, 'year'),
                                           ({'kind': 'film'}, 'kind'), ({'needs': 'nothing'}, 'needs'),
                                           ({'limit': 0}, 'limit'), ({'colour': 'red'}, 'colour')])
def test_a_bad_filter_is_a_422_problem_naming_the_field(service, params, where):
    client, _, _ = service
    _scanned(client)

    response = client.get('/v1/titles', params=params, headers=AUTH)

    assert response.status_code == 422 and response.headers['content-type'] == 'application/problem+json'
    assert any(where in e['loc'] for e in response.json()['errors'])


def test_an_unknown_source_is_a_422_naming_the_profiles_sources(service):
    client, _, _ = service
    _scanned(client)

    response = client.get('/v1/titles', params={'source': 'films'}, headers=AUTH)

    assert response.status_code == 422 and response.json()['detail'] == 'no such source: films (the profile has: disk)'
    assert make_job(client, AUTH, 'scan', {'sources': ['films']}).status_code == 422


def test_a_plan_says_what_a_run_would_do_and_changes_nothing(service):
    client, _, calls = service
    _scanned(client)

    plan = client.post('/v1/plan', json={'filter': {'year': '>=1990'}, 'through': 'design'}, headers=AUTH).json()

    assert plan['label'] == 'Extract & design 2' and plan['skipped'] == []
    assert all(p['stages'] == ['extract', 'design'] for p in plan['planned']) and calls == []


# --- jobs ---------------------------------------------------------------------------------------------------------------

def test_a_run_job_takes_the_filter_and_reports_a_typed_result(service):
    client, _, calls = service

    job = _finish(client, make_job(client, AUTH, 'run', {'filter': {'kind': 'movie', 'year': '2026'}}))

    assert job['kind'] == 'run' and job['state'] == 'succeeded' and job['origin'] == 'api'
    assert job['request'] == {'filter': {'needs': [], 'new_since_scan': False, 'source': None, 'match': None, 'ids': [],
                                         'kind': 'movie', 'year': '2026', 'detail': None, 'flags': [],
                                         'declined': None, 'out_of_date': None, 'adjusted': None, 'unmet': None,
                                         'several_runs': None},
                              'through': 'design', 'scan_first': True, 'retry_failed': False, 'redesign': False,
                              'revise_to': 'design', 'audio_stream': None}
    assert job['result']['scan']['titles'] == 3 and job['result']['attempted'] == ['x'] and job['result']['failed'] == []
    assert job['progress'] == {'done': 1, 'total': 1, 'title': 'Alien', 'stage': 'design', 'id': 'x',
                               'per_hour': None, 'remaining_seconds': None, 'estimated_finish': None}   # finished: no estimate
    assert calls[0][0].kind == 'movie' and calls[0][0].year == '2026' and calls[0][1] == 'design'


def test_a_misspelt_field_is_refused_not_a_wider_selection(service):
    client, _, calls = service

    response = make_job(client, AUTH, 'run', {'filter': {'yeer': '2026'}})

    assert response.status_code == 422 and 'yeer' in response.json()['detail'] and calls == []


def test_publish_commit_and_accept_are_403_unless_the_config_allows_them(service):
    client, _, _ = service

    for body, kind in (({'through': 'publish'}, 'run'), ({'through': 'commit'}, 'run'), ({}, 'accept')):
        response = make_job(client, AUTH, kind, body)
        assert response.status_code == 403 and 'allow_repository_writes' in response.json()['detail']
    assert client.get('/v1/jobs', headers=AUTH).json() == {'jobs': []}   # refused before it was kept
    _scanned(client)
    dry = _finish(client, make_job(client, AUTH, 'accept', {'dry_run': True}))
    assert dry['state'] == 'succeeded' and dry['result']['dry_run'] and dry['result']['not_for_review'] == 3


def test_jobs_and_executions_are_listed_newest_first_filtered_and_a_finished_one_cannot_be_cancelled(service):
    client, _, _ = service
    scan = _scanned(client)
    run = _finish(client, make_job(client, AUTH, 'run', {}))

    jobs = client.get('/v1/jobs', headers=AUTH).json()['jobs']
    assert [j['id'] for j in jobs] == [run['job_id'], scan['job_id']] and {j['state'] for j in jobs} == {'done'}
    assert jobs[0]['progress'] == {'done': 1, 'total': 1} and jobs[0]['next_at'] is None
    assert [j['id'] for j in client.get('/v1/jobs', params={'kind': 'scan'}, headers=AUTH).json()['jobs']] == \
        [scan['job_id']]
    assert client.get('/v1/jobs', params={'state': 'active'}, headers=AUTH).json() == {'jobs': []}
    log = client.get('/v1/executions', headers=AUTH).json()
    assert [e['id'] for e in log['executions']] == [run['id'], scan['id']] and log['running'] is None
    assert log['next'] == []
    assert [e['id'] for e in client.get('/v1/executions', params={'job': scan['job_id']},
                                        headers=AUTH).json()['executions']] == [scan['id']]
    assert client.get('/v1/executions', params={'state': 'failed'}, headers=AUTH).json()['executions'] == []
    response = client.post(f"/v1/executions/{scan['id']}/cancel", headers=AUTH)
    assert response.status_code == 409 and response.json()['title'] == 'Execution finished'
    assert client.get('/v1/executions/nope', headers=AUTH).status_code == 404


def test_the_status_counts_the_index_and_names_the_running_job(service):
    client, _, _ = service
    _scanned(client)

    status = client.get('/v1/status', headers=AUTH).json()

    assert status['index']['titles'] == 3 and status['index']['counts']['extract'] == 3
    assert status['index']['sources'][0]['name'] == 'disk' and status['running_execution'] is None
    assert status['waiting'] == 0 and 'schedule' not in status


# --- following a job ----------------------------------------------------------------------------------------------------

@pytest.fixture
def gated(profile):
    '''
    (client, entered, release) over an app whose jobs wait for `release`, so a stream can be opened on a running one. The
    client talks to a real server on a free port: TestClient hands back a streamed body only once it has ended.
    '''
    import httpx
    import uvicorn
    release, entered = threading.Event(), threading.Event()

    def execute(job, control):
        entered.set()
        assert release.wait(10)
        control.progress(Progress(1, 1, 'Heat', 'extract', 'h'))
        return None
    manager = ExecutionQueue(execute)
    app = create_app(manager, ServiceConfig(profile_path=str(profile), token=TOKEN), env={})
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=0, log_level='warning'))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _until(lambda: server.started)
    port = server.servers[0].sockets[0].getsockname()[1]
    with httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=10) as client:
        yield client, entered, release
    release.set()
    server.should_exit = True
    thread.join(10)
    manager.stop(grace_seconds=5)


def _read_stream(client, execution_id, into):
    with client.stream('GET', f'/v1/executions/{execution_id}/events', headers=AUTH) as response:
        assert response.headers['content-type'].startswith('text/event-stream')
        assert response.headers['x-accel-buffering'] == 'no'   # a proxy passes each event on as it comes
        for line in response.iter_lines():
            if line.startswith('data: '):
                into.append(json.loads(line[len('data: '):]))


def test_the_event_stream_sends_what_happened_then_follows_the_job_to_its_end(gated):
    client, entered, release = gated
    job = make_job(client, AUTH, 'scan', {}).json()
    assert entered.wait(10)
    seen = []
    execution_id = job['last_execution']
    reader = threading.Thread(target=_read_stream, args=(client, execution_id, seen))
    reader.start()
    _until(lambda: len(seen) >= 2)   # the backlog: queued, running

    release.set()
    reader.join(10)

    assert not reader.is_alive()
    assert [(e['type'], e.get('state')) for e in seen] == [('state', 'queued'), ('state', 'running'),
                                                           ('run_progress', None), ('state', 'succeeded')]
    assert seen[2]['title'] == 'Heat' and [e['seq'] for e in seen] == sorted(e['seq'] for e in seen)
    assert seen[0]['text'] == 'execution queued' and seen[-1]['text'] == 'execution succeeded'
    assert all(None not in e.values() for e in seen)   # only the fields its type has
    log = client.get(f"/v1/executions/{execution_id}/log", headers=AUTH).json()
    assert log == seen
    assert client.get(f"/v1/executions/{execution_id}/log", params={'after': seen[1]['seq']},
                      headers=AUTH).json() == seen[2:]


def test_the_stream_of_a_finished_job_is_its_history(gated):
    client, entered, release = gated
    release.set()
    job = make_job(client, AUTH, 'scan', {}).json()
    execution_id = job['last_execution']
    _until(lambda: client.get(f"/v1/executions/{execution_id}", headers=AUTH).json()['state'] == 'succeeded')
    seen = []

    _read_stream(client, execution_id, seen)

    assert seen[-1]['state'] == 'succeeded' and len(seen) == 4


# --- the published interface --------------------------------------------------------------------------------------------

def test_the_published_document_is_the_interface():
    ''' Regenerate with: PYTHONPATH=src/main/python python -m pipeline.service.api > docs/schema/service.openapi.json '''
    assert json.loads(PUBLISHED.read_text()) == json.loads(json.dumps(openapi_document()))


def test_the_document_is_valid_openapi_with_every_body_typed():
    from openapi_spec_validator import validate
    document = openapi_document()

    validate(document)
    schemas = document['components']['schemas']
    assert {'TitleFilter', 'RunJobRequest', 'RunJob', 'RunResult', 'Title', 'JobEvent', 'Problem',
            'Notification', 'NotifyOutcome'} <= set(schemas)
    assert 'notification' in document['webhooks']
    assert schemas['TitleFilter']['additionalProperties'] is False
    year = schemas['TitleFilter']['properties']['year']['anyOf'][0]
    assert year['pattern'] and '>=2020' in year['examples']
    assert document['components']['securitySchemes']['HTTPBearer']['scheme'] == 'bearer'
    assert 'HTTPValidationError' not in json.dumps(document)


def test_the_user_guide_names_every_route():
    ''' docs/library/service.md is how a person meets the interface: a route it does not mention is one nobody finds. '''
    guide = (PUBLISHED.parents[1] / 'library' / 'service.md').read_text()
    routes = {re.sub(r'\{[a-z_]+\}', '{id}', path) for path in openapi_document()['paths']}

    assert sorted(route for route in routes if f'`{route}`' not in guide and f'{route}`' not in guide) == []


def test_each_type_of_event_has_its_own_fields_and_says_in_words_what_they_mean():
    """An ffmpeg event was microseconds and a page of nulls: each type now has a fixed payload, and text."""
    from pipeline.service.models import JobEvent
    dumped = lambda event: JobEvent.of(event).model_dump(mode='json', exclude_none=True)   # noqa: E731

    assert dumped({'seq': 1, 'at': 0.0, 'type': 'extract_progress', 'title': 'The Godfather', 'title_id': 'jriver-1',
                   'done_ms': 3405475, 'total_ms': 10528059, 'percent': 32}) == {
        'seq': 1, 'at': '1970-01-01T00:00:00Z', 'type': 'extract_progress', 'title': 'The Godfather',
        'title_id': 'jriver-1', 'done_ms': 3405475, 'total_ms': 10528059, 'percent': 32,
        'text': 'The Godfather: extracted 0:56:45 of 2:55:28 (32%)'}
    assert dumped({'seq': 2, 'at': 0.0, 'type': 'extract_progress', 'title': 'Heat', 'title_id': 'h', 'done_ms': 61000,
                   'total_ms': None, 'percent': None})['text'] == 'Heat: extracted 0:01:01'
    assert dumped({'seq': 3, 'at': 0.0, 'type': 'run_progress', 'done': 1, 'total': 5, 'title': 'Heat',
                   'stage': 'design', 'title_id': 'x'})['text'] == '1 of 5 done; design: Heat'
    assert dumped({'seq': 4, 'at': 0.0, 'type': 'run_progress', 'done': 5, 'total': 5}) == {
        'seq': 4, 'at': '1970-01-01T00:00:00Z', 'type': 'run_progress', 'done': 5, 'total': 5,
        'text': '5 of 5 done; finished'}
    assert dumped({'seq': 5, 'at': 0.0, 'type': 'state', 'state': 'running'})['text'] == 'execution running'
    assert dumped({'seq': 6, 'at': 0.0, 'type': 'event', 'title_id': 'h', 'stage': 'extract', 'kind': 'finished',
                   'text': '[12:00 extract] done'})['text'] == '[12:00 extract] done'
    with pytest.raises(ValueError):   # a type's fields are required, not optional
        JobEvent.of({'seq': 7, 'at': 0.0, 'type': 'extract_progress', 'title': 'Heat'})


def test_the_published_event_schema_is_one_shape_per_type():
    schemas = openapi_document()['components']['schemas']
    assert schemas['JobEvent']['discriminator']['propertyName'] == 'type'
    assert set(schemas['JobEvent']['discriminator']['mapping']) == {'state', 'run_progress', 'extract_progress', 'event',
                                                                    'window'}
    assert {'done_ms', 'total_ms', 'percent'} <= set(schemas['ExtractProgressEvent']['properties'])


@pytest.mark.parametrize('body', [
    {}, {'filter': {'ids': [f't{i}' for i in range(501)]}}, {'filter': {'ids': ['a'], 'needs': ['review']}},
    {'filter': {'ids': ['a']}, 'through': 'publish'},
])
def test_redesign_takes_1_to_500_ids_and_design_only(service, body):
    client, _, calls = service
    response = make_job(client, AUTH, 'run', {**body, 'redesign': True})
    assert response.status_code == 422
    assert calls == []


def test_titles_out_of_date_against_what_the_designer_up_now_says(service, monkeypatch):
    client, _, calls = service
    monkeypatch.setattr('pipeline.service.context.check_designer', lambda *args, **kwargs: {'design_version': 4})
    _scanned(client)

    def titles(**params):
        response = client.get('/v1/titles', params=params, headers=AUTH)
        assert response.status_code == 200, response.text
        return sorted(t['title'] for t in response.json()['titles'])

    assert titles(out_of_date=True) == []   # nothing designed yet: nothing out of date
    assert titles(out_of_date=False) == ['Alien', 'Dune', 'Heat']
    assert {t['design_version'] for t in client.get('/v1/titles', headers=AUTH).json()['titles']} == {None}
    job = _finish(client, make_job(client, AUTH, 'run', {'filter': {'out_of_date': True}, 'redesign': True}))
    assert job['request']['filter']['out_of_date'] is True and job['request']['redesign'] is True


@pytest.mark.parametrize('body', [{'filter': {'out_of_date': True, 'ids': ['a']}},
                                  {'filter': {'out_of_date': True}, 'through': 'publish'},
                                  {'filter': {'out_of_date': False}}])
def test_redesign_by_version_takes_out_of_date_true_without_ids(service, body):
    client, _, calls = service
    assert make_job(client, AUTH, 'run', {**body, 'redesign': True}).status_code == 422
    assert calls == []
