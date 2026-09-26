'''The image's installed dependency set and the fixture CI runs against it.'''
import json
import pathlib
import tomllib
import wave

from docker.smoke import fixture

ROOT = pathlib.Path(__file__).parents[3]


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


def test_ci_smokes_before_tag_publish_and_builds_both_platforms():
    ci = (ROOT / '.github' / 'workflows' / 'test.yaml').read_text()
    release = (ROOT / '.github' / 'workflows' / 'create-image.yaml').read_text()
    assert 'docker build -f docker/Dockerfile' in ci and 'python3 docker/smoke.py' in ci
    assert release.index('python3 docker/smoke.py') < release.index('docker/build-push-action')
    assert 'linux/amd64,linux/arm64' in release
    assert 'alpha|beta|rc' in release and 'packages: write' in release
