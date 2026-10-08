'''
pipeline/service/api.py (design/pipeline-service.md §6): the HTTP interface, driven through FastAPI's TestClient over a real
profile, index and JobManager (run_stages faked), and its published OpenAPI document.
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
from pipeline.service.jobs import JobManager
from pipeline.service.models import Check
from pipeline.service.work import executor, job_failed

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
    (client, manager, calls): the app over a real JobManager; run_stages is recorded, not run. The profile's source lists
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
    manager = JobManager(executor(str(profile), env={}), failed=job_failed)
    app = create_app(manager, ServiceConfig(profile_path=str(profile), token=TOKEN), env={},
                     checks=lambda: [Check(name='profile', ok=True)])
    with TestClient(app) as client:
        yield client, manager, calls
    manager.stop(grace_seconds=5)


def _finish(client, response):
    assert response.status_code == 202, response.text
    location = response.headers['location']
    _until(lambda: client.get(location, headers=AUTH).json()['state'] not in ('queued', 'running'))
    return client.get(location, headers=AUTH).json()


def _scanned(client):
    job = _finish(client, client.post('/v1/jobs/scan', json={}, headers=AUTH))
    assert job['state'] == 'succeeded' and job['result']['titles'] == 3
    return job


def test_schedule_routes_validate_persist_and_trigger(service):
    client, manager, calls = service
    assert client.get('/v1/schedule', headers=AUTH).json()['enabled'] is False
    invalid = client.put('/v1/schedule', json={'enabled': True, 'through': 'publish'}, headers=AUTH)
    assert invalid.status_code == 422 and invalid.json()['status'] == 422
    saved = client.put('/v1/schedule', json={'enabled': True, 'interval_minutes': 5,
                                             'filter': {'kind': 'movie', 'needs': ['publish']}}, headers=AUTH)
    assert saved.status_code == 200 and saved.json()['next_run_at']
    assert client.get('/v1/status', headers=AUTH).json()['schedule']['enabled']
    job = _finish(client, client.post('/v1/schedule/trigger', headers=AUTH))
    assert job['origin'] == 'schedule' and job['request']['filter']['needs'] == ['extract', 'design']
    assert calls[-1][1] == 'design'


def test_schedule_trigger_is_409_while_a_job_runs(profile):
    release = threading.Event()
    running = threading.Event()

    def execute(job, control):
        running.set()
        release.wait(5)

    manager = JobManager(execute)
    app = create_app(manager, ServiceConfig(profile_path=str(profile), token=TOKEN), env={})
    try:
        with TestClient(app) as client:
            first = client.post('/v1/schedule/trigger', headers=AUTH)
            assert first.status_code == 202
            assert running.wait(5)
            busy = client.post('/v1/schedule/trigger', headers=AUTH)
            assert busy.status_code == 409 and busy.json()['title'] == 'Service busy'
    finally:
        release.set()
        manager.stop(5)


# --- the unauthenticated routes -----------------------------------------------------------------------------------------

def test_health_and_readiness_need_no_token(service):
    client, _, _ = service

    assert client.get('/health').json()['status'] == 'ok'
    assert client.get('/ready').json() == {'ready': True, 'checks': [{'name': 'profile', 'ok': True, 'detail': '', 'required': True}]}


def test_not_ready_is_a_503_that_says_why(profile):
    manager = JobManager(lambda job, control: None)
    app = create_app(manager, ServiceConfig(profile_path=str(profile), token=TOKEN),
                     checks=lambda: [Check(name='ffmpeg', ok=False, detail='not found: ffmpeg')])
    try:
        response = TestClient(app).get('/ready')
    finally:
        manager.stop(1)

    assert response.status_code == 503 and response.json()['checks'][0]['detail'] == 'not found: ffmpeg'


def test_the_default_readiness_checks_the_profile_work_dir_ffmpeg_and_designer(profile):
    manager = JobManager(lambda job, control: None)
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
    manager = JobManager(lambda job, control: None)
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
    manager = JobManager(lambda job, control: None)
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

    assert response.status_code == 409 and 'POST /v1/jobs/scan' in response.json()['detail']
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
    assert client.post('/v1/jobs/scan', json={'sources': ['films']}, headers=AUTH).status_code == 422


def test_a_plan_says_what_a_run_would_do_and_changes_nothing(service):
    client, _, calls = service
    _scanned(client)

    plan = client.post('/v1/plan', json={'filter': {'year': '>=1990'}, 'through': 'design'}, headers=AUTH).json()

    assert plan['label'] == 'Extract & design 2' and plan['skipped'] == []
    assert all(p['stages'] == ['extract', 'design'] for p in plan['planned']) and calls == []


# --- jobs ---------------------------------------------------------------------------------------------------------------

def test_a_run_job_takes_the_filter_and_reports_a_typed_result(service):
    client, _, calls = service

    job = _finish(client, client.post('/v1/jobs/run', json={'filter': {'kind': 'movie', 'year': '2026'}}, headers=AUTH))

    assert job['kind'] == 'run' and job['state'] == 'succeeded' and job['origin'] == 'api'
    assert job['request'] == {'filter': {'needs': [], 'new_since_scan': False, 'source': None, 'match': None, 'ids': [],
                                         'kind': 'movie', 'year': '2026'},
                              'through': 'design', 'scan_first': True, 'retry_failed': False}
    assert job['result']['scan']['titles'] == 3 and job['result']['attempted'] == ['x'] and job['result']['failed'] == []
    assert job['progress'] == {'done': 1, 'total': 1, 'title': 'Alien', 'stage': 'design', 'id': 'x',
                               'per_hour': None, 'remaining_seconds': None, 'estimated_finish': None}   # finished: no estimate
    assert calls[0][0].kind == 'movie' and calls[0][0].year == '2026' and calls[0][1] == 'design'


def test_a_misspelt_field_is_refused_not_a_wider_selection(service):
    client, _, calls = service

    response = client.post('/v1/jobs/run', json={'filter': {'yeer': '2026'}}, headers=AUTH)

    assert response.status_code == 422 and 'yeer' in response.json()['detail'] and calls == []


def test_publish_commit_and_accept_are_403_unless_the_config_allows_them(service):
    client, _, _ = service

    for body, path in (({'through': 'publish'}, '/v1/jobs/run'), ({'through': 'commit'}, '/v1/jobs/run'),
                       ({}, '/v1/jobs/accept')):
        response = client.post(path, json=body, headers=AUTH)
        assert response.status_code == 403 and 'allow_repository_writes' in response.json()['detail']
    _scanned(client)
    dry = _finish(client, client.post('/v1/jobs/accept', json={'dry_run': True}, headers=AUTH))
    assert dry['state'] == 'succeeded' and dry['result']['dry_run'] and dry['result']['not_for_review'] == 3


def test_jobs_are_listed_newest_first_filtered_and_a_finished_one_cannot_be_cancelled(service):
    client, _, _ = service
    scan = _scanned(client)
    run = _finish(client, client.post('/v1/jobs/run', json={}, headers=AUTH))

    assert [j['id'] for j in client.get('/v1/jobs', headers=AUTH).json()['jobs']] == [run['id'], scan['id']]
    assert [j['id'] for j in client.get('/v1/jobs', params={'kind': 'scan'}, headers=AUTH).json()['jobs']] == [scan['id']]
    assert client.get('/v1/jobs', params={'state': 'failed'}, headers=AUTH).json() == {'jobs': []}
    response = client.post(f"/v1/jobs/{scan['id']}/cancel", headers=AUTH)
    assert response.status_code == 409 and response.json()['title'] == 'Job finished'
    assert client.get('/v1/jobs/nope', headers=AUTH).status_code == 404


def test_the_status_counts_the_index_and_names_the_running_job(service):
    client, _, _ = service
    _scanned(client)

    status = client.get('/v1/status', headers=AUTH).json()

    assert status['index']['titles'] == 3 and status['index']['counts']['extract'] == 3
    assert status['index']['sources'][0]['name'] == 'disk' and status['current_job'] is None and status['queued'] == 0


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
    manager = JobManager(execute)
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


def _read_stream(client, job_id, into):
    with client.stream('GET', f'/v1/jobs/{job_id}/events', headers=AUTH) as response:
        assert response.headers['content-type'].startswith('text/event-stream')
        for line in response.iter_lines():
            if line.startswith('data: '):
                into.append(json.loads(line[len('data: '):]))


def test_the_event_stream_sends_what_happened_then_follows_the_job_to_its_end(gated):
    client, entered, release = gated
    job = client.post('/v1/jobs/scan', json={}, headers=AUTH).json()
    assert entered.wait(10)
    seen = []
    reader = threading.Thread(target=_read_stream, args=(client, job['id'], seen))
    reader.start()
    _until(lambda: len(seen) >= 2)   # the backlog: queued, running

    release.set()
    reader.join(10)

    assert not reader.is_alive()
    assert [(e['type'], e.get('state')) for e in seen] == [('state', 'queued'), ('state', 'running'),
                                                           ('progress', None), ('state', 'succeeded')]
    assert seen[2]['title'] == 'Heat' and [e['seq'] for e in seen] == sorted(e['seq'] for e in seen)
    log = client.get(f"/v1/jobs/{job['id']}/log", headers=AUTH).json()
    assert log == seen
    assert client.get(f"/v1/jobs/{job['id']}/log", params={'after': seen[1]['seq']}, headers=AUTH).json() == seen[2:]


def test_the_stream_of_a_finished_job_is_its_history(gated):
    client, entered, release = gated
    release.set()
    job = client.post('/v1/jobs/scan', json={}, headers=AUTH).json()
    _until(lambda: client.get(f"/v1/jobs/{job['id']}", headers=AUTH).json()['state'] == 'succeeded')
    seen = []

    _read_stream(client, job['id'], seen)

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
