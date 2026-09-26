'''Completed-job events, webhook formats, retries, secrets and the HTTP test route.'''
import json
import threading
import time
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

from pipeline.library.index import ScanResult
from pipeline.library.run import LibraryRunReport
from pipeline.library.stages import StagesReport
from pipeline.service.api import create_app
from pipeline.service.config import ServiceConfig
from pipeline.service.jobs import Job, JobManager, RunRequest, ScanRequest
from pipeline.service.models import NotifyEvent
from pipeline.service.notify import Notifier, Target, _body, _send, events_for, notification, targets_from_config
from pipeline.service.work import RunOutcome


def _job(*, designed=(), failed=(), earlier=(), publish_errors=(), commit_error='', scan_errors=None,
         state='succeeded', origin='schedule', error=''):
    report = StagesReport('design', 1, run=LibraryRunReport(designed=list(designed), failed=list(failed),
                                                            failed_earlier=list(earlier)),
                          publish_errors=list(publish_errors), commit_error=commit_error, counts={'review': 3})
    scanned = ScanResult(1, 1, errors=scan_errors or {}) if scan_errors is not None else None
    return Job(id='job-1', kind='run', origin=origin, request=RunRequest(), state=state,
               submitted_at=1000, started_at=1001, finished_at=1002,
               result=RunOutcome(scanned, report) if not error else None, error=error)


@pytest.mark.parametrize('kwargs, expected', [
    ({'designed': ['a']}, ['review_waiting', 'job_finished']),
    ({'designed': ['a'], 'failed': [('b', 'bad audio')]}, ['review_waiting', 'failed', 'job_finished']),
    ({'earlier': [('a', 'old failure')]}, ['job_finished']),
    ({'publish_errors': [{'id': 'a', 'error': 'invalid_metadata'}]}, ['failed', 'job_finished']),
    ({'commit_error': 'git refused'}, ['failed', 'job_finished']),
    ({'scan_errors': {'disk': 'offline'}}, ['failed', 'job_finished']),
    ({'state': 'cancelled'}, ['job_finished']),
    ({'state': 'failed', 'error': 'profile unreadable'}, ['failed', 'job_finished']),
])
def test_events_for_job_result_shapes(kwargs, expected):
    assert [event.value for event in events_for(_job(**kwargs))] == expected


def test_payload_has_title_details_and_redacted_new_failures():
    class Row:
        title, display_name, year, kind, confidence = 'Alien', 'Alien file', '1979', 'movie', 0.93
    job = _job(designed=['a'], failed=[('a', 'api_key=secret123')], earlier=[('b', 'old failure')])
    note = notification(job, NotifyEvent.failed, lambda ids: ({'a': Row()}, 4))
    assert note.job.id == 'job-1' and note.review_waiting == 4
    assert note.designed[0].title == 'Alien' and note.designed[0].confidence == 0.93
    assert len(note.failed) == 1 and 'secret123' not in note.failed[0].message
    assert note.links.job == '/v1/jobs/job-1'


def test_target_defaults_filters_and_environment_secrets(tmp_path):
    url_file = tmp_path / 'url'
    url_file.write_text('https://secret.example/token-123\n')
    headers_file = tmp_path / 'headers'
    headers_file.write_text('{"Authorization":"Bearer secret-456"}')
    targets = targets_from_config([{'name': 'phone', 'url': 'https://ignored.example', 'format': 'json'}],
                                  {'BEQ_NOTIFY_URL_PHONE_FILE': str(url_file),
                                   'BEQ_NOTIFY_HEADERS_PHONE_FILE': str(headers_file)})
    target = targets[0]
    assert target.url == 'https://secret.example/token-123'
    assert target.headers['Authorization'] == 'Bearer secret-456'
    assert target.events == {'review_waiting', 'failed'} and target.origins == {'schedule'}
    assert 'secret-456' not in repr(target) and 'token-123' not in repr(target)
    with pytest.raises(ValueError, match='duplicate environment name'):
        targets_from_config([{'name': 'a-b', 'url': 'https://example.com'},
                             {'name': 'a_b', 'url': 'https://example.com'}], {})


class Recorder(BaseHTTPRequestHandler):
    posts = []
    gets = []
    redirect = False
    codes = []

    def do_POST(self):
        body = self.rfile.read(int(self.headers['Content-Length']))
        self.posts.append((self.headers['Content-Type'], body, self.headers.get('Authorization')))
        self.send_response(self.codes.pop(0) if self.codes else 302 if self.redirect else 200)
        if self.redirect:
            self.send_header('Location', '/other')
        self.end_headers()

    def do_GET(self):
        self.gets.append(self.path)
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture
def receiver():
    Recorder.posts = []
    Recorder.gets = []
    Recorder.redirect = False
    Recorder.codes = []
    server = ThreadingHTTPServer(('127.0.0.1', 0), Recorder)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/hook'
    finally:
        server.shutdown()
        thread.join(5)


@pytest.mark.parametrize('format_, content_type, key', [
    ('json', 'application/json', 'event'),
    ('text', 'text/plain; charset=utf-8', None),
    ('slack', 'application/json', 'text'),
    ('discord', 'application/json', 'content'),
])
def test_each_format_posts_to_a_real_http_server(receiver, format_, content_type, key):
    note = notification(_job(designed=['a']), NotifyEvent.review_waiting, lambda ids: ({}, 3))
    target = Target('phone', receiver, format_, headers={'Authorization': 'Bearer configured'})
    body, media_type = _body(target, note)
    _send(target, body, media_type)
    seen_type, seen_body, auth = Recorder.posts[-1]
    assert seen_type == content_type and auth == 'Bearer configured'
    if key:
        assert json.loads(seen_body)[key]
    else:
        assert seen_body == b'3 titles waiting for review, 0 failed'


def test_redirect_does_not_forward_the_payload_to_another_url(receiver):
    Recorder.redirect = True
    with pytest.raises(urllib.error.HTTPError) as error:
        _send(Target('phone', receiver), b'sensitive title', 'text/plain')
    assert error.value.code == 302 and Recorder.gets == []


def test_delivery_retries_then_succeeds_and_gives_up_without_exposing_secrets():
    manager = JobManager(lambda job, control: None)
    calls, delays = [], []

    def send(target, body, content_type):
        calls.append(1)
        if len(calls) < 3:
            raise urllib.error.HTTPError('https://secret.example/token', 503, 'bad', {}, None)
    notifier = Notifier(manager, 'p', [{'name': 'phone', 'url': 'https://secret.example/token'}],
                        send=send, sleep=delays.append, start=False)
    try:
        target = notifier.targets[0]
        note = notification(_job(designed=['a']), NotifyEvent.review_waiting, lambda ids: ({}, 3))
        result = notifier._deliver(target, note)
        assert result.ok and len(calls) == 3 and delays == [1, 2]
        notifier.send = lambda *args: (_ for _ in ()).throw(TimeoutError('secret-token'))
        failed = notifier._deliver(target, note)
        assert failed.ok is False and 'secret' not in failed.message
        assert 'secret.example' not in str(notifier.outcomes())
    finally:
        notifier.stop()
        manager.stop(5)


def test_delivery_retries_real_http_failures(receiver):
    Recorder.codes = [503, 503, 200]
    manager = JobManager(lambda job, control: None)
    waits = []
    notifier = Notifier(manager, 'p', [{'name': 'phone', 'url': receiver}], sleep=waits.append, start=False)
    try:
        target = notifier.targets[0]
        note = notification(_job(designed=['a']), NotifyEvent.review_waiting, lambda ids: ({}, 3))
        assert notifier._deliver(target, note).ok
        assert len(Recorder.posts) == 3 and waits == [1, 2]
    finally:
        notifier.stop()
        manager.stop(5)


def test_a_job_is_saved_before_async_delivery_and_origin_filters_apply(tmp_path):
    recorded = []
    manager = JobManager(lambda job, control: ScanResult(1, 1, errors={'disk': 'offline'}), state_dir=str(tmp_path),
                         failed=lambda result: bool(result.errors))
    notifier = Notifier(manager, 'unused', [{'name': 'phone', 'url': 'http://example.invalid',
                                             'events': ['failed', 'job_finished'], 'origins': ['schedule']}],
                        resolve=lambda ids: ({}, 0),
                        send=lambda target, body, content_type: recorded.append(
                            (json.loads((tmp_path / 'jobs.json').read_text())[-1]['state'], json.loads(body)['event'])))
    try:
        manager.submit(ScanRequest(), origin='api')
        scheduled = manager.submit(ScanRequest(), origin='schedule')
        deadline = time.time() + 5
        while len(recorded) < 2:
            assert time.time() < deadline
            time.sleep(0.005)
        assert scheduled.finished and recorded == [('failed', 'failed'), ('failed', 'job_finished')]
    finally:
        notifier.stop()
        manager.stop(5)


def test_notification_test_route_and_status_never_return_target_secrets(tmp_path):
    profile = tmp_path / 'profile.json'
    (tmp_path / 'media').mkdir()
    profile.write_text(json.dumps({'sources': [{'name': 'disk', 'kind': 'filesystem', 'globs': [str(tmp_path / 'media')]}],
                                  'run': {'work_dir': str(tmp_path / 'work'), 'queue_dir': str(tmp_path / 'queue'),
                                          'designer': 'manual'}}))
    calls = []
    manager = JobManager(lambda job, control: None)
    config = ServiceConfig(profile_path=str(profile), token='api-token',
                           notify=({'name': 'phone', 'url': 'https://secret.example/token',
                                    'events': ['review_waiting', 'failed', 'job_finished']},))
    notifier = Notifier(manager, str(profile), config.notify, env={'BEQ_NOTIFY_HEADERS_PHONE':
                         '{"Authorization":"Bearer secret-header"}'},
                        send=lambda target, body, content_type: calls.append(json.loads(body)['event']), start=False)
    app = create_app(manager, config, notifier=notifier)
    try:
        client = TestClient(app)
        auth = {'Authorization': 'Bearer api-token'}
        assert client.post('/v1/notify/test', json={'target': 'missing'}, headers=auth).status_code == 404
        response = client.post('/v1/notify/test', json={'target': 'phone'}, headers=auth)
        assert response.status_code == 200 and response.json()['ok']
        assert calls == ['review_waiting', 'failed', 'job_finished']
        status = client.get('/v1/status', headers=auth).text
        assert 'phone' in status and 'secret.example' not in status and 'secret-header' not in status
    finally:
        notifier.stop()
        manager.stop(5)
