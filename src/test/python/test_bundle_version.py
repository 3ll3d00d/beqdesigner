'''
The macOS .app's Info.plist versions, from the VERSION file the release workflow writes (issue #525: it said 0.0.0).
'''
import importlib.util
import os

import pytest

SCRIPTS = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.github', 'scripts')
SPEC = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'beqdesigner.spec')


@pytest.fixture(scope='module')
def versions():
    spec = importlib.util.spec_from_file_location('bundle_version', os.path.join(SCRIPTS, 'bundle_version.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('version, expected', [
    ('2.2.0-alpha.10', ('2.2.0', '2.2.0-alpha.10')),
    ('2.2.0', ('2.2.0', '2.2.0')),
    ('v1.4.2', ('1.4.2', '1.4.2')),
    ('3.1', ('3.1.0', '3.1')),
    ('7', ('7.0.0', '7')),
    ('  2.2.0-beta.1\n', ('2.2.0', '2.2.0-beta.1')),
    ('abc1234', ('0.0.0', 'abc1234')),   # test.yaml's builds write the short commit sha
    ('1234abc', ('0.0.0', '1234abc')),   # ... which may start with digits
])
def test_the_short_version_is_the_leading_numbers_and_the_bundle_version_is_the_whole(versions, version, expected):
    assert versions.bundle_versions(version) == expected


def test_a_version_file_is_read(versions, tmp_path):
    path = tmp_path / 'VERSION'
    path.write_text('2.2.0-alpha.11\n')
    assert versions.bundle_versions_from(str(path)) == ('2.2.0', '2.2.0-alpha.11')


def test_without_a_version_file_the_app_s_own_default_is_used(versions, tmp_path):
    assert versions.bundle_versions_from(str(tmp_path / 'missing')) == ('0.0.0', '0.0.0-alpha.1')
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'main', 'python', 'app.py')) as app:
        assert f"'{versions.DEFAULT_VERSION}'" in app.read()


def test_the_spec_gives_the_bundle_both_versions():
    with open(SPEC) as handle:
        spec = handle.read()
    bundle = spec[spec.index('app = BUNDLE('):]
    assert 'version=short_version' in bundle
    assert "'CFBundleVersion': full_version" in bundle
    assert "bundle_versions_from(" in spec
