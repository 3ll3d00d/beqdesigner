'''
TODO R4: a designer that answers one request at a time (beqforge's) queues the rest, so with `run.parallelism.design` above
1 a request waits behind the others before its own design starts. Each declared designer's timeout is multiplied by the
design parallelism, so that wait is not taken for a designer that stopped answering.
'''
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from pipeline.designer.contract import DesignRequest
from pipeline.designer.http_binding import HttpDesignerError
from pipeline.designer.registry import get_designer
from pipeline.library.setup import run_config_from_values

DESIGN_SECONDS = 0.5
TIMEOUT = 0.8   # one design fits; one design and a wait behind another does not


class _SlowDesigner(BaseHTTPRequestHandler):
    ''' One design at a time (a plain HTTPServer, as beqforge's), each taking DESIGN_SECONDS. '''

    def do_POST(self):
        self.rfile.read(int(self.headers['Content-Length']))
        time.sleep(DESIGN_SECONDS)
        body = json.dumps({'contract_version': '1.0', 'decline_reason': 'no_rolloff_detected',
                           'decline_message': 'nothing to restore'}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def designer():
    server = HTTPServer(('127.0.0.1', 0), _SlowDesigner)
    server.request_queue_size = 8
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.02}, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{server.server_address[1]}/design'
    server.shutdown()
    server.server_close()


def _registered(tmp_path, url, design_parallelism):
    values = {'work_dir': str(tmp_path / 'w'), 'queue_dir': str(tmp_path / 'q'), 'designer': 'slow',
              'parallelism': {'design': design_parallelism}}
    run_config_from_values(values, {'designers': {'slow': {'url': url, 'timeout': TIMEOUT}}})
    return get_designer('slow')


def _two_at_once(design):
    request = DesignRequest('1.0', 1000, np.zeros(100), 'complete_programme')
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(design, request) for _ in range(2)]
        outcomes = []
        for future in futures:
            try:
                outcomes.append(future.result())
            except HttpDesignerError as error:
                outcomes.append(error)
    return outcomes


def test_with_two_designs_in_flight_the_one_that_waits_does_not_time_out(tmp_path, designer):
    outcomes = _two_at_once(_registered(tmp_path, designer, design_parallelism=2))

    assert [o.decline_reason for o in outcomes] == ['no_rolloff_detected'] * 2


def test_one_at_a_time_the_second_waits_in_the_pool_not_at_the_designer(tmp_path, designer):
    '''
    A designer declared for one design at a time is never sent a second while the first runs: the pool holds it until the
    first returns, so the wait is ours and its timeout covers only its own design (pipeline.designer.pool). Before the
    pool, it queued at the designer and timed out there.
    '''
    outcomes = _two_at_once(_registered(tmp_path, designer, design_parallelism=1))

    assert [getattr(o, 'decline_reason', o) for o in outcomes] == ['no_rolloff_detected'] * 2


def test_the_work_list_scales_the_profiles_designers_the_same_way(tmp_path, designer):
    from pipeline.library.profile import profile_from_config
    from model.worklist_profile import register_profile_designers

    config = {'sources': [{'name': 'disk', 'kind': 'filesystem', 'globs': ['x']}],
              'designers': {'slow': {'url': designer, 'timeout': TIMEOUT}},
              'run': {'work_dir': str(tmp_path / 'w'), 'queue_dir': str(tmp_path / 'q'),
                      'parallelism': {'design': 2}}}
    assert register_profile_designers(profile_from_config(config)) == []

    outcomes = _two_at_once(get_designer('slow'))

    assert [o.decline_reason for o in outcomes] == ['no_rolloff_detected'] * 2
