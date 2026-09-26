'''The image's installed dependency set and the fixture CI runs against it.'''
import importlib.util
import json
import pathlib
import shutil
import threading
import time
import tomllib
import wave

import pytest
from fastapi.testclient import TestClient

from http.server import ThreadingHTTPServer
from pipeline.service.api import create_app
from pipeline.service.config import ServiceConfig
from pipeline.service.jobs import JobManager
from pipeline.service.work import executor, job_failed

ROOT = pathlib.Path(__file__).parents[3]


def _load_smoke():
    ''' docker/smoke.py by path: `docker/` is a folder at the repo root, not a package on the test path. '''
    spec = importlib.util.spec_from_file_location('beq_docker_smoke', ROOT / 'docker' / 'smoke.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_smoke = _load_smoke()
Designer, fixture = _smoke.Designer, _smoke.fixture


def test_service_lock_closure_has_no_qt(tmp_path):
    config = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    lock = tomllib.loads((ROOT / 'uv.lock').read_text())
    packages = {package['name']: package for package in lock['package']}
    project = packages['beqdesigner']
    pending = [item['name'] for item in project['dependencies'] + project['dev-dependencies']['service']]
    found = set()
    while pending:
        name = pending.pop()
        if name in found:
            continue
        found.add(name)
        pending.extend(item['name'] for item in packages[name].get('dependencies', []))
    assert not found.intersection({'pyqt6', 'qtpy', 'pyqtgraph', 'qtawesome'})
    assert set(config['tool']['uv']['default-groups']) == {'dev', 'desktop'}
    assert {'pyqt6', 'qtpy', 'pyqtgraph', 'qtawesome'} <= set(config['dependency-groups']['desktop'])
    dockerfile = (ROOT / 'docker' / 'Dockerfile').read_text()
    assert 'uv sync --frozen --no-default-groups --group service' in dockerfile
    assert 'USER beq' in dockerfile and 'dvdvideo' in dockerfile


def test_image_smoke_fixture_is_real_multichannel_audio_and_a_profile(tmp_path):
    fixture(tmp_path, 4321)
    with wave.open(str(tmp_path / 'media' / 'Smoke Movie.wav')) as source:
        assert source.getnchannels() == 6 and source.getframerate() == 48000 and source.getnframes() == 48000
    profile = json.loads((tmp_path / 'config' / 'profile.yaml').read_text())
    assert profile['sources'][0]['globs'] == ['/media']
    assert profile['designers']['smoke'] == 'http://host.docker.internal:4321/design'
    assert profile['run']['queue_dir'] == '/queue'


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='ffmpeg and ffprobe are optional')
def test_image_smoke_fixture_extracts_and_designs_through_the_real_service(tmp_path):
    server = ThreadingHTTPServer(('127.0.0.1', 0), Designer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    fixture(tmp_path, server.server_port)
    profile_path = tmp_path / 'config' / 'profile.yaml'
    profile = json.loads(profile_path.read_text())
    profile['sources'][0]['globs'] = [str(tmp_path / 'media')]
    profile['designers']['smoke'] = f'http://127.0.0.1:{server.server_port}/design'
    profile['run'].update(work_dir=str(tmp_path / 'work'), queue_dir=str(tmp_path / 'queue'))
    profile_path.write_text(json.dumps(profile))
    manager = JobManager(executor(str(profile_path), env={}), failed=job_failed)
    try:
        app = create_app(manager, ServiceConfig(profile_path=str(profile_path), token='smoke-token'), env={})
        with TestClient(app) as client:
            auth = {'Authorization': 'Bearer smoke-token'}
            response = client.post('/v1/jobs/run', json={'filter': {'match': 'Smoke Movie'}, 'through': 'design'},
                                   headers=auth)
            assert response.status_code == 202, response.text
            deadline = time.time() + 30
            while True:
                job = client.get(response.headers['Location'], headers=auth).json()
                if job['state'] not in ('queued', 'running'):
                    break
                assert time.time() < deadline, job
                time.sleep(0.02)
            assert job['state'] == 'succeeded', job
            assert list((tmp_path / 'queue').glob('*.json'))
    finally:
        manager.stop(5)
        server.shutdown()
        thread.join(5)


def test_ci_smokes_before_tag_publish_and_builds_both_platforms():
    ci = (ROOT / '.github' / 'workflows' / 'test.yaml').read_text()
    release = (ROOT / '.github' / 'workflows' / 'create-image.yaml').read_text()
    assert 'docker build -f docker/Dockerfile' in ci and 'python3 docker/smoke.py' in ci
    assert release.index('python3 docker/smoke.py') < release.index('docker/build-push-action')
    assert 'linux/amd64,linux/arm64' in release
    assert 'alpha|beta|rc' in release and 'packages: write' in release
