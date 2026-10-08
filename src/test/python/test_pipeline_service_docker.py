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
Designer, fixture, by_reference = _smoke.Designer, _smoke.fixture, _smoke.by_reference


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


def test_with_a_real_designer_the_fixture_shares_the_work_directory_and_is_long_enough_to_tell(tmp_path):
    fixture(tmp_path, designer_url='http://beq-smoke-designer-1:8420/design', seconds=30)
    profile = json.loads((tmp_path / 'config' / 'profile.yaml').read_text())
    assert profile['designers']['smoke'] == {'url': 'http://beq-smoke-designer-1:8420/design', 'by_reference': True}
    with wave.open(str(tmp_path / 'media' / 'Smoke Movie.wav')) as source:
        # inline, even the mono mix at the 1 kHz analysis rate would be 0.3 MB of base64: by reference it is none
        assert source.getnframes() == 30 * 48000


def test_by_reference_is_read_from_the_designers_own_request_log():
    def log(*sizes):
        return '\n'.join(f'2026-10-08 request timing: body {size} MB, read 0.00 s' for size in sizes)

    assert by_reference(log('0.0', '0.0'))
    assert not by_reference(log('0.0', '2.4'))   # one request carried its audio
    assert not by_reference('beqforge designer server: listening')   # no request reached it


def test_the_compose_example_runs_a_pinned_designer_sharing_the_pipelines_work_directory():
    import yaml
    compose = yaml.safe_load((ROOT / 'docker' / 'compose.example.yaml').read_text())
    pipeline, designer = compose['services']['pipeline'], compose['services']['designer']
    image, _, version = designer['image'].rpartition(':')
    assert image == 'ghcr.io/3ll3d00d/beqforge-designer' and version not in ('', 'latest')
    assert './work:/work' in pipeline['volumes'] and './work:/work' in designer['volumes']
    assert 'designer-cache:/cache' in designer['volumes'] and 'designer-cache' in compose['volumes']
    assert designer['user'] == pipeline['user'] and 'designer' in pipeline['depends_on']
    assert "url: 'http://designer:8420/design', by_reference: true" in (ROOT / 'docs' / 'library' / 'service.md').read_text()


def test_ci_designs_by_reference_through_the_real_designer_on_every_push_and_before_a_release():
    ci = (ROOT / '.github' / 'workflows' / 'test.yaml').read_text()
    release = (ROOT / '.github' / 'workflows' / 'create-image.yaml').read_text()
    assert 'repository: 3ll3d00d/beqforge' in ci and 'beqforge/packaging/designer/Dockerfile' in ci
    assert '--designer-image beqforge-designer:smoke' in ci
    # the release is smoked against the designer the compose example names, before it is published
    assert 'beqforge-designer:[^[:space:]]+\' docker/compose.example.yaml' in release
    assert release.index('--designer-image') < release.index('docker/build-push-action')


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
