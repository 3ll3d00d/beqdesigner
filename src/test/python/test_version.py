'''
model/version.py: the app's and the service's version, from the VERSION file CI writes. From source there is none, which is
expected: the default is used and nothing is logged as an error (it was a traceback at every start).
From source, it is the checkout's last tag and commit (`git describe`).
'''
import logging
import re
import shutil
import subprocess

import pytest

import model.version
from model.version import DEFAULT_VERSION, read_version


def test_the_version_file_is_read(tmp_path):
    (tmp_path / 'VERSION').write_text('2.2.0-alpha.36\n')
    assert read_version(str(tmp_path)) == '2.2.0-alpha.36'


def test_an_empty_version_file_gives_the_default(tmp_path):
    (tmp_path / 'VERSION').write_text('\n')
    assert read_version(str(tmp_path)) == DEFAULT_VERSION


def test_no_version_file_outside_a_checkout_gives_the_default_without_a_warning_or_traceback(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    assert read_version(str(tmp_path)) == DEFAULT_VERSION
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
    assert all(r.exc_info is None for r in caplog.records)


def test_a_version_file_that_cannot_be_read_is_a_warning(tmp_path, caplog):
    (tmp_path / 'VERSION').mkdir()   # there, but not a file
    assert read_version(str(tmp_path)) == DEFAULT_VERSION
    assert [r.levelno for r in caplog.records] == [logging.WARNING]
    assert all(r.exc_info is None for r in caplog.records)


def _git(cwd, *args):
    subprocess.run(['git', '-c', 'user.name=t', '-c', 'user.email=t@t', '-c', 'commit.gpgsign=false',
                    '-c', 'tag.gpgsign=false', *args], cwd=cwd, check=True, capture_output=True)


@pytest.mark.skipif(shutil.which('git') is None, reason='needs git')
def test_no_version_file_in_a_checkout_is_the_last_tag_and_the_commit(tmp_path):
    root = tmp_path / 'src' / 'main' / 'python'
    root.mkdir(parents=True)
    (root / 'app.py').write_text('one\n')
    _git(tmp_path, 'init', '-q')
    _git(tmp_path, 'add', '.')
    _git(tmp_path, 'commit', '-q', '-m', 'one')
    _git(tmp_path, 'tag', '2.2.0-alpha.35')
    assert read_version(str(root)) == '2.2.0-alpha.35'   # on the tag: the tag, as a release is called

    (root / 'app.py').write_text('two\n')
    _git(tmp_path, 'commit', '-q', '-am', 'two')
    assert re.fullmatch(r'2\.2\.0-alpha\.35-1-g[0-9a-f]{7,}', read_version(str(root)))

    (root / 'app.py').write_text('three\n')
    assert read_version(str(root)).endswith('-dirty')


@pytest.mark.skipif(shutil.which('git') is None, reason='needs git')
def test_a_checkout_with_no_tag_is_the_commit(tmp_path):
    (tmp_path / 'a').write_text('a')
    _git(tmp_path, 'init', '-q')
    _git(tmp_path, 'add', '.')
    _git(tmp_path, 'commit', '-q', '-m', 'a')
    assert re.fullmatch(r'[0-9a-f]{7,}', read_version(str(tmp_path)))


def test_without_git_it_is_the_default(tmp_path, monkeypatch):
    def no_git(*args, **kwargs):
        raise FileNotFoundError('git')
    monkeypatch.setattr(model.version.subprocess, 'run', no_git)
    assert read_version(str(tmp_path)) == DEFAULT_VERSION


def test_a_version_file_wins_over_the_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(model.version, 'describe_checkout', lambda root: pytest.fail('git asked'))
    (tmp_path / 'VERSION').write_text('2.2.0\n')
    assert read_version(str(tmp_path)) == '2.2.0'
