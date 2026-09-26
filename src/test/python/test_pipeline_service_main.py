'''
python -m pipeline.service (pipeline/service/__main__.py): what it refuses to start with, where it keeps its jobs, and a real
process started, answering and stopped on SIGTERM.
'''
import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

from pipeline.service.__main__ import build

SRC = pathlib.Path(__file__).parents[2] / 'main' / 'python'


@pytest.fixture
def profile(tmp_path):
    (tmp_path / 'media').mkdir()
    config = {'sources': [{'name': 'disk', 'kind': 'filesystem', 'globs': [str(tmp_path / 'media')]}],
              'run': {'work_dir': str(tmp_path / 'work'), 'queue_dir': str(tmp_path / 'queue'), 'designer': 'manual'}}
    path = tmp_path / 'profile.json'
    path.write_text(json.dumps(config))
    return path


@pytest.fixture
def built():
    made = []

    def make(argv, env):
        app, manager, config = build(argv, env)
        made.append(manager)
        return app, manager, config
    yield make
    for manager in made:
        manager.stop(grace_seconds=5)


def test_it_keeps_its_jobs_beside_the_work_directory_and_listens_where_told(built, profile, tmp_path):
    _, _, config = built(['--profile', str(profile), '--port', '9999'], {'BEQ_SERVICE_TOKEN': 't'})

    assert config.state_dir == str(tmp_path / 'work' / 'service') and config.port == 9999 and config.token == 't'


@pytest.mark.parametrize('argv, env, message', [
    ([], {}, '--profile .* is required'),
    (['--profile', 'missing.json'], {}, 'missing.json'),
    (['--profile', '{profile}'], {}, 'no token: set BEQ_SERVICE_TOKEN'),
    (['--profile', '{profile}', '--no-auth'], {}, '--no-auth is only allowed on a loopback address, not 0.0.0.0'),
])
def test_it_refuses_to_start_without_what_it_needs(profile, capsys, argv, env, message):
    with pytest.raises(SystemExit) as exit_:
        build([arg.format(profile=profile) for arg in argv], env)

    assert exit_.value.code == 2
    import re
    assert re.search(message, capsys.readouterr().err)


def test_no_auth_is_allowed_on_loopback(built, profile):
    _, _, config = built(['--profile', str(profile), '--no-auth', '--host', '127.0.0.1'], {})

    assert config.token is None and config.host == '127.0.0.1'


def test_a_service_config_that_is_wrong_stops_it(profile, tmp_path, capsys):
    (tmp_path / 'service.json').write_text('{"lisen": {}}')

    with pytest.raises(SystemExit):
        build(['--profile', str(profile), '--service-config', str(tmp_path / 'service.json')], {'BEQ_SERVICE_TOKEN': 't'})
    assert 'unknown key lisen' in capsys.readouterr().err


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


@pytest.mark.skipif(sys.platform == 'win32', reason='SIGTERM')
def test_a_real_process_serves_and_stops_cleanly_on_sigterm(profile, tmp_path):
    port = _free_port()
    env = {**os.environ, 'PYTHONPATH': str(SRC), 'BEQ_SERVICE_TOKEN': 'secret-token'}
    process = subprocess.Popen([sys.executable, '-m', 'pipeline.service', '--profile', str(profile), '--host', '127.0.0.1',
                                '--port', str(port)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.time() + 30
        while True:
            try:
                health = json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=1))
                break
            except OSError:
                assert process.poll() is None and time.time() < deadline, process.stdout.read() if process.poll() else 'timeout'
                time.sleep(0.1)
        assert health['status'] == 'ok'
        request = urllib.request.Request(f'http://127.0.0.1:{port}/v1/status', headers={'Authorization': 'Bearer secret-token'})
        assert json.load(urllib.request.urlopen(request, timeout=5))['queued'] == 0
    finally:
        process.send_signal(signal.SIGTERM)
        output = process.communicate(timeout=30)[0]
    assert process.returncode == 0, output
    assert 'secret-token' not in output
