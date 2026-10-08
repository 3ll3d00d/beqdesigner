'''
The designer is asked whether it is up before a scheduled tick and a run job through design (TODO R2): a tick is skipped
while it is down, with the reason, `failed` is notified once per outage, and the tick runs again once it is back.
`/ready` and `/v1/status` say which, and a designer outage does not make the service unready.
'''
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

from pipeline.designer.http_binding import DesignerUnavailable, check_designer
from pipeline.service import work
from pipeline.service.api import create_app
from pipeline.service.config import ServiceConfig
from pipeline.service.context import load_context
from pipeline.service.designer import DesignerProbe
from pipeline.service.jobs import JobManager, RunRequest
from pipeline.service.notify import Notifier
from pipeline.service.scheduler import DESIGNER_RETRY_MINUTES, AutoScheduler
from pipeline.library.selection import Selection

TOKEN = 'test-token'


class StubDesigner:
    ''' A designer's HTTP server: `health` is what GET /health answers (a status, or a JSON body for 200). '''

    def __init__(self):
        stub = self
        self.health = {'status': 'ok'}
        self.asked = 0

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                stub.asked += 1
                if isinstance(stub.health, int):
                    self.send_response(stub.health)
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                    return
                body = json.dumps(stub.health).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.url = f'http://127.0.0.1:{self.server.server_address[1]}/design'
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': 0.02}, daemon=True)
        self.thread.start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def designer():
    stub = StubDesigner()
    yield stub
    stub.stop()


def _nothing_listening() -> str:
    server = ThreadingHTTPServer(('127.0.0.1', 0), BaseHTTPRequestHandler)
    port = server.server_address[1]
    server.server_close()
    return f'http://127.0.0.1:{port}/design'


def _profile(tmp_path, url, by_reference=False):
    config = {'sources': [{'name': 'disk', 'kind': 'filesystem', 'globs': [str(tmp_path / 'media')]}],
              'designers': {'rolloff': {'url': url, 'by_reference': by_reference} if by_reference else url},
              'run': {'work_dir': str(tmp_path / 'work'), 'queue_dir': str(tmp_path / 'queue'), 'designer': 'rolloff'}}
    path = tmp_path / 'profile.json'
    path.write_text(json.dumps(config))
    return str(path)


# --- asking the designer ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('health', [{'status': 'ok'}, 404, 501, 500])
def test_any_answer_from_a_designer_is_up_since_only_a_1_2_designer_must_serve_health(designer, health):
    designer.health = health
    check_designer(designer.url)


@pytest.mark.parametrize('health', [502, 503, 504])
def test_a_gateway_that_cannot_reach_the_designer_is_down(designer, health):
    designer.health = health
    with pytest.raises(DesignerUnavailable, match=f'HTTP {health}'):
        check_designer(designer.url)


def test_nothing_listening_is_down():
    with pytest.raises(DesignerUnavailable, match='/health failed'):
        check_designer(_nothing_listening(), timeout=2)


def test_a_by_reference_designer_must_say_it_takes_references(designer):
    designer.health = {'contract_version': '1.2', 'shared_root': True}
    check_designer(designer.url, by_reference=True)
    designer.health = {'contract_version': '1.1'}
    with pytest.raises(DesignerUnavailable, match='cannot take arrays by reference'):
        check_designer(designer.url, by_reference=True)


def test_the_profiles_designer_is_the_one_asked(tmp_path, designer):
    ctx = load_context(_profile(tmp_path, designer.url), {})

    assert ctx.designer_unavailable() == '' and designer.asked == 1
    designer.health = 503
    assert 'HTTP 503' in ctx.designer_unavailable()


def test_the_manual_designer_has_nothing_to_ask(tmp_path):
    path = tmp_path / 'profile.json'
    path.write_text(json.dumps({'sources': [{'name': 'disk', 'kind': 'filesystem', 'globs': ['x']}],
                                'run': {'work_dir': str(tmp_path / 'w'), 'queue_dir': str(tmp_path / 'q'),
                                        'designer': 'manual'}}))
    state = DesignerProbe(str(path), {}).check()

    assert state.reachable is None and state.name == 'manual'


def test_the_probe_reuses_a_recent_answer(tmp_path, designer):
    now = [1000.0]
    probe = DesignerProbe(_profile(tmp_path, designer.url), {}, clock=lambda: now[0], max_age=30)

    assert probe.current().reachable and designer.asked == 1
    now[0] += 29
    assert probe.current().reachable and designer.asked == 1
    now[0] += 2
    designer.health = 503
    assert probe.current().reachable is False and designer.asked == 2


# --- the schedule ----------------------------------------------------------------------------------------------------

def test_a_tick_is_skipped_while_the_designer_is_down_notified_once_and_resumed_when_it_is_back(tmp_path, designer):
    now = [1000.0]
    submitted, told = [], []
    manager = JobManager(lambda job, control: submitted.append(job), clock=lambda: now[0])
    probe = DesignerProbe(_profile(tmp_path, designer.url), {})
    scheduler = AutoScheduler(manager, str(tmp_path), {'enabled': True, 'interval_minutes': 60},
                              clock=lambda: now[0], start=False, designer=probe.unavailable,
                              on_designer_down=told.append)
    try:
        designer.health = 503
        now[0] += 3600
        scheduler.tick()
        snapshot = scheduler.snapshot()
        assert manager.jobs() == [] and snapshot['last_skip'].startswith('designer unavailable: ')
        assert 'HTTP 503' in snapshot['last_skip'] and len(told) == 1
        assert snapshot['next_run_at'] == now[0] + DESIGNER_RETRY_MINUTES * 60   # sooner than the hour

        now[0] += DESIGNER_RETRY_MINUTES * 60
        scheduler.tick()                       # still down: skipped again, not notified again
        assert manager.jobs() == [] and len(told) == 1

        designer.health = {'status': 'ok'}
        now[0] += DESIGNER_RETRY_MINUTES * 60
        scheduler.tick()
        assert len(manager.jobs()) == 1 and manager.jobs()[0].origin == 'schedule'
        assert scheduler.snapshot()['last_skip'] is None and scheduler.designer_down == ''
    finally:
        scheduler.stop()
        manager.stop(5)


def test_a_new_outage_is_notified_again(tmp_path, designer):
    now = [1000.0]
    told = []
    manager = JobManager(lambda job, control: None, clock=lambda: now[0])
    probe = DesignerProbe(_profile(tmp_path, designer.url), {})
    scheduler = AutoScheduler(manager, str(tmp_path), {'enabled': True, 'interval_minutes': 5},
                              clock=lambda: now[0], start=False, designer=probe.unavailable,
                              on_designer_down=told.append)
    try:
        for health in (503, {'status': 'ok'}, 503):
            designer.health = health
            now[0] += 3600
            scheduler.tick()
            _wait_idle(manager)
        assert len(told) == 2
    finally:
        scheduler.stop()
        manager.stop(5)


def _wait_idle(manager):
    import time
    deadline = time.time() + 5
    while manager.current is not None or manager.queued:
        assert time.time() < deadline
        time.sleep(0.005)


def test_a_tick_through_extract_does_not_ask_the_designer(tmp_path, designer):
    now = [1000.0]
    manager = JobManager(lambda job, control: None, clock=lambda: now[0])
    asked = []
    scheduler = AutoScheduler(manager, str(tmp_path), {'enabled': True, 'interval_minutes': 5, 'through': 'extract'},
                              clock=lambda: now[0], start=False, designer=lambda: asked.append(1) or 'down')
    try:
        now[0] += 600
        scheduler.tick()
        assert asked == [] and len(manager.jobs()) == 1
    finally:
        scheduler.stop()
        manager.stop(5)


# --- a run job --------------------------------------------------------------------------------------------------------

def test_a_run_job_through_design_is_refused_before_anything_runs_while_the_designer_is_down(tmp_path, designer,
                                                                                           monkeypatch):
    ctx = load_context(_profile(tmp_path, designer.url), {})
    ran = []
    monkeypatch.setattr(work, 'run_stages', lambda *args, **kwargs: ran.append(1))
    designer.health = 503

    with pytest.raises(DesignerUnavailable, match='nothing was run: .*HTTP 503'):
        work.run(ctx, RunRequest(Selection(), 'design'), control=None)
    assert ran == []


# --- what the service reports ----------------------------------------------------------------------------------------

def test_ready_and_status_say_whether_the_designer_answers_and_an_outage_is_not_unready(tmp_path, designer):
    (tmp_path / 'work').mkdir()
    now = [1000.0]
    profile = _profile(tmp_path, designer.url)
    manager = JobManager(lambda job, control: None)
    probe = DesignerProbe(profile, {}, clock=lambda: now[0])
    app = create_app(manager, ServiceConfig(profile_path=profile, token=TOKEN), env={}, designer=probe)
    client = TestClient(app)
    auth = {'Authorization': f'Bearer {TOKEN}'}
    try:
        status = client.get('/v1/status', headers=auth).json()['designer']
        assert status['name'] == 'rolloff' and status['reachable'] is True and status['detail'] == designer.url

        designer.health = 503
        now[0] += 60
        ready = client.get('/ready')
        checks = {c['name']: c for c in ready.json()['checks']}
        assert checks['designer_reachable']['ok'] is False and checks['designer_reachable']['required'] is False
        assert 'HTTP 503' in checks['designer_reachable']['detail']
        assert checks['designer']['ok'] is True
        # unready only for what is the service's own: the designer is not (ffmpeg may be absent on this machine)
        assert ready.json()['ready'] == all(c['ok'] for c in checks.values() if c['required'])
        assert client.get('/v1/status', headers=auth).json()['designer']['reachable'] is False
    finally:
        manager.stop(1)


def test_an_outage_is_notified_as_failed_to_the_schedules_targets(tmp_path):
    sent = []
    manager = JobManager(lambda job, control: None)
    notifier = Notifier(manager, '', [{'name': 'ops', 'url': 'http://example.invalid/hook'},
                                      {'name': 'api-only', 'url': 'http://example.invalid/other', 'origins': ['api']}],
                        env={}, send=lambda target, body, kind: sent.append((target.name, json.loads(body))),
                        start=False)
    try:
        notifier.designer_unavailable('rolloff', 'GET http://designer/health failed: refused')
        notifier.queue.put(None)
        notifier._loop()

        assert [name for name, _ in sent] == ['ops']
        note = sent[0][1]
        assert note['event'] == 'failed' and note['job'] is None
        assert note['failed'] == [{'id': 'designer', 'title': 'rolloff',
                                   'message': 'scheduled run skipped: GET http://designer/health failed: refused'}]
    finally:
        manager.stop(1)
