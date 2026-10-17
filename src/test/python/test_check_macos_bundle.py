'''
The release workflow's check that a zipped macOS .app keeps its frameworks' symlinks (issue #525).
'''
import importlib.util
import os
import stat
import zipfile

import pytest

SCRIPT = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.github', 'scripts', 'check_macos_bundle.py')


@pytest.fixture(scope='module')
def check():
    spec = importlib.util.spec_from_file_location('check_macos_bundle', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FRAMEWORK = 'beqdesigner.app/Contents/Frameworks/PyQt6/Qt6/lib/QtCore.framework'


def _link(archive: zipfile.ZipFile, name: str, target: str) -> None:
    info = zipfile.ZipInfo(name)
    info.external_attr = (stat.S_IFLNK | 0o755) << 16
    archive.writestr(info, target)


def _file(archive: zipfile.ZipFile, name: str) -> None:
    info = zipfile.ZipInfo(name)
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    archive.writestr(info, b'x')


def _dir(archive: zipfile.ZipFile, name: str) -> None:
    info = zipfile.ZipInfo(name.rstrip('/') + '/')
    info.external_attr = (stat.S_IFDIR | 0o755) << 16 | 0x10
    archive.writestr(info, b'')


def _as_ditto_stores_it(path: str) -> str:
    with zipfile.ZipFile(path, 'w') as archive:
        _dir(archive, f'{FRAMEWORK}/Versions/A/Resources')
        _file(archive, f'{FRAMEWORK}/Versions/A/QtCore')
        _file(archive, f'{FRAMEWORK}/Versions/A/Resources/Info.plist')
        _link(archive, f'{FRAMEWORK}/Versions/Current', 'A')
        _link(archive, f'{FRAMEWORK}/QtCore', 'Versions/Current/QtCore')
        _link(archive, f'{FRAMEWORK}/Resources', 'Versions/Current/Resources')
    return path


def _as_zip_r_stores_it(path: str) -> str:
    # zip -r follows every link: Versions/Current arrives as a second real copy of Versions/A
    with zipfile.ZipFile(path, 'w') as archive:
        for version in ('A', 'Current'):
            _dir(archive, f'{FRAMEWORK}/Versions/{version}')
            _dir(archive, f'{FRAMEWORK}/Versions/{version}/Resources')
            _file(archive, f'{FRAMEWORK}/Versions/{version}/QtCore')
            _file(archive, f'{FRAMEWORK}/Versions/{version}/Resources/Info.plist')
        _file(archive, f'{FRAMEWORK}/QtCore')
        _dir(archive, f'{FRAMEWORK}/Resources')
        _file(archive, f'{FRAMEWORK}/Resources/Info.plist')
    return path


def test_a_bundle_zipped_with_its_symlinks_passes(check, tmp_path):
    path = _as_ditto_stores_it(str(tmp_path / 'ok.zip'))
    assert check.problems(path) == []
    assert check.main([path]) == 0


def test_a_bundle_whose_symlinks_were_followed_fails_naming_the_framework(check, tmp_path, capsys):
    path = _as_zip_r_stores_it(str(tmp_path / 'copied.zip'))
    assert check.problems(path) == [f'{FRAMEWORK}: Versions/Current is a copy, not a symlink']
    assert check.main([path]) == 1
    assert 'QtCore.framework' in capsys.readouterr().err


def test_a_current_version_present_only_as_its_contents_is_a_copy(check, tmp_path):
    path = str(tmp_path / 'contents-only.zip')
    with zipfile.ZipFile(path, 'w') as archive:
        _file(archive, f'{FRAMEWORK}/Versions/Current/QtCore')
    assert check.problems(path) == [f'{FRAMEWORK}: Versions/Current is a copy, not a symlink']


def test_a_framework_that_is_itself_a_link_to_one_elsewhere_is_left_to_its_target(check, tmp_path):
    # PyInstaller links Contents/Resources/Python.framework to the framework in Contents/Frameworks
    path = _as_ditto_stores_it(str(tmp_path / 'linked.zip'))
    with zipfile.ZipFile(path, 'a') as archive:
        _link(archive, 'beqdesigner.app/Contents/Resources/Python.framework', '../Frameworks/Python.framework')
        _link(archive, 'beqdesigner.app/Contents/Resources/PyQt6', '../Frameworks/PyQt6')
    assert check.problems(path) == []


def test_a_framework_with_no_current_version_fails(check, tmp_path):
    path = str(tmp_path / 'partial.zip')
    with zipfile.ZipFile(path, 'w') as archive:
        _file(archive, f'{FRAMEWORK}/Versions/A/QtCore')
    assert check.problems(path) == [f'{FRAMEWORK}: no Versions/Current']


def test_a_zip_with_no_frameworks_is_not_taken_for_a_good_app(check, tmp_path):
    path = str(tmp_path / 'empty.zip')
    with zipfile.ZipFile(path, 'w') as archive:
        _file(archive, 'beqdesigner.app/Contents/MacOS/beqdesigner')
    assert len(check.problems(path)) == 1
    assert check.main([path]) == 1


def test_no_arguments_is_a_usage_error(check):
    assert check.main([]) == 2
