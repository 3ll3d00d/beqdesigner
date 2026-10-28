'''
The service's job windows over HTTP (design/job-windows.md §4, design/jobs-and-executions.md §4): a window given with a
job, changed on a job, and the default, through FastAPI's TestClient over a real ExecutionQueue and JobBook with a fake
clock (and a fake execute: these are about the interface).
'''
import json
import time
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

import pipeline.service.executions as executions
from pipeline.service.api import create_app
from pipeline.service.config import ServiceConfig
from pipeline.service.executions import ExecutionQueue
from pipeline.service.jobbook import JobBook
from pipeline.service.models import JobEvent
from service_http import make_job

TOKEN = 'test-token'
AUTH = {'Authorization': f'Bearer {TOKEN}'}
UTC = timezone.utc


def at(text):
    return datetime.strptime(text, '%Y-%m-%d %H:%M').replace(tzinfo=UTC).timestamp()


def iso(text):
    return datetime.fromtimestamp(at(text), UTC).isoformat().replace('+00:00', 'Z')


def _until(condition, timeout=10.0):
    deadline = time.time() + timeout
    while not condition():
        assert time.time() < deadline, 'timed out'
        time.sleep(0.005)


@pytest.fixture(autouse=True)
def quick_recheck(monkeypatch):
    monkeypatch.setattr(executions, 'WINDOW_RECHECK_SECONDS', 0.01)
    monkeypatch.setattr('pipeline.library.setup.check_designer', lambda *args, **kwargs: None)


@pytest.fixture
def profile(tmp_path):
    config = {'sources': [{'name': 'disk', 'kind': 'filesystem', 'globs': [str(tmp_path / 'media')]}],
              'designers': {'rolloff': 'http://127.0.0.1:9/design'},
              'run': {'work_dir': str(tmp_path / 'work'), 'queue_dir': str(tmp_path / 'queue'), 'designer': 'rolloff'}}
    path = tmp_path / 'profile.json'
    path.write_text(json.dumps(config))
    return path


@pytest.fixture
def service(profile, tmp_path):
    ''' (client, queue, book, ran): executions record that they ran and return nothing; the clock reads 12:00 UTC. '''
    now = [at('2026-10-26 12:00')]
    ran = []
    queue = ExecutionQueue(lambda execution, control: ran.append(execution.id), clock=lambda: now[0], tz=UTC,
                           state_dir=str(tmp_path / 'state'))
    book = JobBook(queue, str(tmp_path / 'state'), clock=lambda: now[0], start=False)
    app = create_app(queue, ServiceConfig(profile_path=str(profile), token=TOKEN), env={}, book=book)
    with TestClient(app) as client:
        yield client, queue, book, ran
    book.stop()
    queue.stop(grace_seconds=5)


def test_a_job_given_a_closed_window_waits_and_says_when_it_will_start(service):
    client, queue, book, ran = service
    response = make_job(client, AUTH, 'scan', window=[' 1:00 - 2:00'])
    assert response.status_code == 201, response.text
    job = response.json()
    assert (job['state'], job['window'], job['executions'], job['running_execution']) == \
        ('active', ['01:00-02:00'], 0, None)
    assert job['next_at'] == iso('2026-10-27 01:00')
    log = client.get('/v1/executions', headers=AUTH).json()
    assert log['next'] == [{'job_id': job['id'], 'job_name': 'List the sources again', 'at': iso('2026-10-27 01:00')}]
    time.sleep(0.05)
    assert ran == []


@pytest.mark.parametrize('route, body', [
    ('/v1/jobs', {'kind': 'scan', 'window': ['night']}),
    ('/v1/jobs', {'kind': 'run', 'window': ['01:00-01:00']}),
    ('/v1/jobs', {'kind': 'accept', 'window': ['25:00-01:00'], 'request': {'dry_run': True}}),
    ('/v1/jobs', {'kind': 'scan', 'window': [f'{h:02d}:00-{h:02d}:30' for h in range(25)]}),   # more than 24 ranges
    ('/v1/revise', {'ids': ['a'], 'to': 'design', 'window': []}),   # a window is for the run job
])
def test_a_window_that_is_not_one_is_refused_before_anything_is_kept(service, route, body):
    client, queue, book, ran = service
    response = client.post(route, json=body, headers=AUTH)
    assert response.status_code == 422, response.text
    assert queue.jobs() == [] and book.all() == []


def test_a_jobs_window_is_changed_and_its_execution_takes_it(service):
    client, queue, book, ran = service
    job = make_job(client, AUTH, 'scan', window=['01:00-02:00']).json()

    response = client.patch(f"/v1/jobs/{job['id']}", json={'window': []}, headers=AUTH)
    assert response.status_code == 200 and response.json()['window'] == []
    book.tick()   # its window is open now: it starts
    _until(lambda: client.get(f"/v1/jobs/{job['id']}", headers=AUTH).json()['state'] == 'done')
    execution = client.get(f"/v1/jobs/{job['id']}", headers=AUTH).json()['last_execution']
    assert client.get(f'/v1/executions/{execution}', headers=AUTH).json()['window'] == []

    assert client.patch(f"/v1/jobs/{job['id']}", json={'window': None}, headers=AUTH).status_code == 409
    assert client.patch('/v1/jobs/nope', json={'window': None}, headers=AUTH).status_code == 404
    assert client.patch(f"/v1/jobs/{job['id']}", json={'window': ['night']}, headers=AUTH).status_code == 422


def test_the_default_window_is_read_and_changed_and_says_when_it_next_opens(service, tmp_path):
    client, queue, book, ran = service
    default = client.get('/v1/job-window', headers=AUTH).json()
    assert (default['window'], default['open_now'], default['next_change']) == ([], True, None)
    assert default['time_zone'] == 'UTC'

    changed = client.put('/v1/job-window', json={'window': ['01:00-02:00']}, headers=AUTH).json()
    assert (changed['window'], changed['open_now'], changed['next_change']) == \
        (['01:00-02:00'], False, iso('2026-10-27 01:00'))
    assert json.loads((tmp_path / 'state' / 'job-window.json').read_text()) == ['01:00-02:00']
    job = make_job(client, AUTH, 'scan').json()   # given none: it follows the default
    assert (job['window'], job['executions'], job['next_at']) == (None, 0, iso('2026-10-27 01:00'))

    open_now = client.put('/v1/job-window', json={'window': ['11:00-13:00']}, headers=AUTH).json()
    assert open_now['open_now'] and open_now['next_change'] == iso('2026-10-26 13:00')   # open: when it closes
    book.tick()
    _until(lambda: client.get(f"/v1/jobs/{job['id']}", headers=AUTH).json()['state'] == 'done')
    assert client.put('/v1/job-window', json={'window': ['night']}, headers=AUTH).status_code == 422
    assert client.put('/v1/job-window', json={'window': None}, headers=AUTH).status_code == 422


def test_a_stop_says_in_words_why():
    stopping = JobEvent.of({'seq': 1, 'at': at('2026-10-26 02:00'), 'type': 'state', 'state': 'stopping',
                            'because': 'window'})
    assert stopping.root.because == 'window'
    assert stopping.root.text == 'stopping: its window closed; it stops before its next title'
    assert JobEvent.of({'seq': 2, 'at': at('2026-10-26 02:01'), 'type': 'state', 'state': 'stopped'}).root.text == \
        'execution stopped'
