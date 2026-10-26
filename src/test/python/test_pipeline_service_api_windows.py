'''
The service's job windows over HTTP (design/job-windows.md §4): a window given with a job, changed on a job, and the default,
through FastAPI's TestClient over a real JobManager with a fake clock (and a fake execute: these are about the interface).
'''
import json
import threading
import time
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

import pipeline.service.jobs as jobs
from pipeline.service.api import create_app
from pipeline.service.config import ServiceConfig
from pipeline.service.jobs import JobManager
from pipeline.service.models import JobEvent

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
    monkeypatch.setattr(jobs, 'WINDOW_RECHECK_SECONDS', 0.01)
    monkeypatch.setattr('pipeline.service.context.check_designer', lambda *args, **kwargs: None)


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
    ''' (client, manager, ran): jobs record that they ran and return nothing; the clock reads 12:00 UTC. '''
    now = [at('2026-10-26 12:00')]
    ran = []
    manager = JobManager(lambda job, control: ran.append(job.id), clock=lambda: now[0], tz=UTC,
                         state_dir=str(tmp_path / 'state'))
    app = create_app(manager, ServiceConfig(profile_path=str(profile), token=TOKEN), env={})
    with TestClient(app) as client:
        yield client, manager, ran
    manager.stop(grace_seconds=5)


def test_a_job_given_a_closed_window_is_queued_and_says_when_it_will_be_taken(service):
    client, manager, ran = service
    response = client.post('/v1/jobs/scan', json={'window': [' 1:00 - 2:00']}, headers=AUTH)
    assert response.status_code == 202, response.text
    job = response.json()
    assert (job['state'], job['window'], job['window_open']) == ('queued', ['01:00-02:00'], False)
    assert job['waiting_until'] == iso('2026-10-27 01:00') and job['request']['window'] == ['01:00-02:00']
    assert job['parts'] == 0 and job['paused_at'] == []
    time.sleep(0.05)
    assert ran == []


@pytest.mark.parametrize('route, body', [
    ('/v1/jobs/scan', {'window': ['night']}),
    ('/v1/jobs/run', {'window': ['01:00-01:00']}),
    ('/v1/jobs/accept', {'window': ['25:00-01:00'], 'dry_run': True}),
    ('/v1/jobs/scan', {'window': [f'{h:02d}:00-{h:02d}:30' for h in range(25)]}),   # more than 24 ranges
    ('/v1/revise', {'ids': ['a'], 'to': 'design', 'window': []}),   # a window is for the run job
])
def test_a_window_that_is_not_one_is_refused_before_anything_is_queued(service, route, body):
    client, manager, ran = service
    response = client.post(route, json=body, headers=AUTH)
    assert response.status_code == 422, response.text
    assert manager.jobs() == []


def test_a_jobs_window_is_changed_until_it_finishes(service):
    client, manager, ran = service
    job = client.post('/v1/jobs/scan', json={'window': ['01:00-02:00']}, headers=AUTH).json()

    response = client.put(f"/v1/jobs/{job['id']}/window", json={'window': []}, headers=AUTH)
    assert response.status_code == 200 and response.json()['window'] == []
    _until(lambda: client.get(f"/v1/jobs/{job['id']}", headers=AUTH).json()['state'] == 'succeeded')
    log = client.get(f"/v1/jobs/{job['id']}/log", headers=AUTH).json()
    assert {'type': 'window', 'text': 'window: any time', 'window': []}.items() <= \
        next(e for e in log if e['type'] == 'window').items()

    assert client.put(f"/v1/jobs/{job['id']}/window", json={'window': None}, headers=AUTH).status_code == 409
    assert client.put('/v1/jobs/nope/window', json={'window': None}, headers=AUTH).status_code == 404
    assert client.put(f"/v1/jobs/{job['id']}/window", json={}, headers=AUTH).status_code == 422   # say which window


def test_the_default_window_is_read_and_changed_and_says_when_it_next_opens(service, tmp_path):
    client, manager, ran = service
    default = client.get('/v1/job-window', headers=AUTH).json()
    assert (default['window'], default['open_now'], default['next_change']) == ([], True, None)
    assert default['time_zone'] == 'UTC'

    changed = client.put('/v1/job-window', json={'window': ['01:00-02:00']}, headers=AUTH).json()
    assert (changed['window'], changed['open_now'], changed['next_change']) == \
        (['01:00-02:00'], False, iso('2026-10-27 01:00'))
    assert json.loads((tmp_path / 'state' / 'job-window.json').read_text()) == ['01:00-02:00']
    job = client.post('/v1/jobs/scan', json={}, headers=AUTH).json()   # given none: it follows the default
    assert (job['window'], job['window_open'], job['state']) == (None, False, 'queued')

    open_now = client.put('/v1/job-window', json={'window': ['11:00-13:00']}, headers=AUTH).json()
    assert open_now['open_now'] and open_now['next_change'] == iso('2026-10-26 13:00')   # open: when it closes
    _until(lambda: client.get(f"/v1/jobs/{job['id']}", headers=AUTH).json()['state'] == 'succeeded')
    assert client.put('/v1/job-window', json={'window': ['night']}, headers=AUTH).status_code == 422
    assert client.put('/v1/job-window', json={'window': None}, headers=AUTH).status_code == 422


def test_a_pause_and_a_resume_say_in_words_when_it_carries_on():
    paused = JobEvent.of({'seq': 1, 'at': at('2026-10-26 02:00'), 'type': 'state', 'state': 'paused',
                          'resumes_at': at('2026-10-27 01:00')}).model_dump(mode='json')
    assert paused['resumes_at'] == iso('2026-10-27 01:00') and paused['text'].startswith(
        'job paused: its window is closed; it carries on at ')
    again = JobEvent.of({'seq': 2, 'at': at('2026-10-27 01:00'), 'type': 'state', 'state': 'running', 'part': 2})
    assert again.root.text == 'job running again (part 2)'
