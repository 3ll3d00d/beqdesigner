'''
Review fixes for the git layer: paths in a repo are always git's own `/`-separated spelling (a Windows path with
backslashes means the same), a subdirectory clone works, a glob character in a name is literal, and a failed git
command says what git said. Real temp git repos.
'''
import os
import subprocess

import pytest

from pipeline.publish.catalogue import catalogue_paths
from pipeline.publish.git import (GitError, RepoTarget, commit_and_push, commit_paths, discard_changes, image_url,
                                  is_committed, join_posix, posix_path, push, repo_state, write_files)
from test_pipeline_publish_git import _init_repo_with_remote, _run


def test_catalogue_paths_are_posix_whatever_the_separator_used():
    assert catalogue_paths('x', 'a/b', 'c') == ('a/b/x.json', 'c/x.png')
    assert catalogue_paths('x', 'a\\b', 'c\\') == ('a/b/x.json', 'c/x.png')  # what os.path.join builds on Windows
    assert catalogue_paths('x') == ('x.json', 'x.png')
    assert join_posix('', 'a\\b', 'c.xml') == 'a/b/c.xml' and posix_path('.') == ''


def test_backslash_paths_mean_the_same_as_slash_paths_in_every_entry_point(tmp_path):
    target, bare = _init_repo_with_remote(tmp_path)
    write_files(target, {'xml\\one.xml': b'1'})
    assert (tmp_path / 'work' / 'xml' / 'one.xml').read_bytes() == b'1'
    assert repo_state(target).uncommitted == {'xml/one.xml'}

    assert commit_paths(target, ['xml\\one.xml'], 'add one')  # a sha: it was found and committed
    assert is_committed(target, 'xml\\one.xml') and is_committed(target, 'xml/one.xml')

    (tmp_path / 'work' / 'xml' / 'one.xml').write_bytes(b'edited')
    assert discard_changes(target, ['xml\\one.xml']) == ['xml/one.xml']
    # a committed file is restored, never deleted
    assert (tmp_path / 'work' / 'xml' / 'one.xml').read_bytes() == b'1'

    write_files(target, {'xml\\two.xml': b'2'})
    assert discard_changes(target, ['xml\\two.xml']) == ['xml/two.xml']
    assert not (tmp_path / 'work' / 'xml' / 'two.xml').exists()  # never committed: deleted

    url = image_url(target, 'img\\one.png', owner='o', repo_name='r')
    assert url.endswith('/img/one.png')
    assert commit_and_push(target, 'a\\b\\c.xml', b'c', 'nested')
    assert is_committed(target, 'a/b/c.xml')


def test_a_glob_character_in_a_name_is_literal(tmp_path):
    target, _ = _init_repo_with_remote(tmp_path)
    write_files(target, {'xml/st*r.xml': b'star', 'xml/stXr.xml': b'x', 'xml/st?r.xml': b'q'})

    commit_paths(target, ['xml/st*r.xml'], 'just the star')

    assert is_committed(target, 'xml/st*r.xml')
    assert not is_committed(target, 'xml/stXr.xml')
    assert repo_state(target).uncommitted == {'xml/stXr.xml', 'xml/st?r.xml'}
    assert discard_changes(target, ['xml/st?r.xml']) == ['xml/st?r.xml']
    assert (tmp_path / 'work' / 'xml' / 'stXr.xml').exists()  # `?` did not match it


def test_a_subdirectory_clone_reports_paths_relative_to_itself(tmp_path):
    root, bare = _init_repo_with_remote(tmp_path)
    sub = RepoTarget(str(tmp_path / 'work' / 'cat'))
    os.makedirs(sub.local_path)
    write_files(root, {'outside.xml': b'o'})
    write_files(sub, {'x.xml': b'x'})

    assert repo_state(sub).uncommitted == {'x.xml'}   # not `cat/x.xml`, and not the file outside

    assert commit_paths(sub, ['x.xml'], 'add x')
    assert is_committed(sub, 'x.xml')
    assert repo_state(sub).uncommitted == frozenset()
    push(sub)
    assert repo_state(sub).unpushed == frozenset()
    (tmp_path / 'work' / 'cat' / 'x.xml').write_bytes(b'y')
    write_files(sub, {'x2.xml': b'2'})
    commit_paths(sub, ['x2.xml'], 'x2')
    assert repo_state(sub).unpushed == {'x2.xml'}
    # the raw URL is relative to the repo's root
    assert image_url(sub, 'img/a.png', owner='o', repo_name='r').endswith('/cat/img/a.png')


def test_a_failed_git_command_says_what_git_said(tmp_path):
    target, _ = _init_repo_with_remote(tmp_path)
    _run('git', '-C', target.local_path, 'remote', 'set-url', 'origin', str(tmp_path / 'nowhere.git'))
    write_files(target, {'a.xml': b'a'})
    commit_paths(target, ['a.xml'], 'a')

    with pytest.raises(subprocess.CalledProcessError) as error:   # still catchable as before
        push(target)

    assert isinstance(error.value, GitError)
    assert 'nowhere.git' in str(error.value) and 'push' in str(error.value)
    assert 'nowhere.git' in error.value.stderr
