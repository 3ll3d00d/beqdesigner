'''
Phase 4 (item 10) of design/pipeline-implementation-plan.md: commit + push
mechanics for the XML and images repos. Exercises real local git repos
(a bare "remote" + a working clone) rather than mocking subprocess, so the
add/commit/push sequence is genuinely verified -- no network access
required, everything stays on the local filesystem.
'''
import subprocess

import pytest

from pipeline.publish.git import (RepoTarget, commit_and_push, current_branch, parse_github_remote, push_image,
                                  push_xml)


def _run(*args):
    subprocess.run(list(args), check=True, capture_output=True)


def _init_repo_with_remote(tmp_path):
    bare = tmp_path / 'remote.git'
    _run('git', 'init', '--bare', '-q', str(bare))
    work = tmp_path / 'work'
    work.mkdir()
    _run('git', 'init', '-q', str(work))
    _run('git', '-C', str(work), 'config', 'user.email', 'test@example.com')
    _run('git', '-C', str(work), 'config', 'user.name', 'Test')
    _run('git', '-C', str(work), 'remote', 'add', 'origin', str(bare))
    return RepoTarget(local_path=str(work)), bare


def _content_on_remote(bare, sha, relative_path):
    result = subprocess.run(['git', '-C', str(bare), 'cat-file', '-p', f'{sha}:{relative_path}'],
                            check=True, capture_output=True)
    return result.stdout


def test_commit_and_push_writes_commits_and_pushes_to_the_remote(tmp_path):
    target, bare = _init_repo_with_remote(tmp_path)

    sha = commit_and_push(target, 'xml/rp1.xml', b'<beq/>', 'Add RP1')

    assert len(sha) == 40
    assert _content_on_remote(bare, sha, 'xml/rp1.xml') == b'<beq/>'


def test_commit_and_push_creates_nested_directories(tmp_path):
    target, bare = _init_repo_with_remote(tmp_path)

    sha = commit_and_push(target, 'a/b/c/rp1.xml', b'nested', 'Add nested')

    assert _content_on_remote(bare, sha, 'a/b/c/rp1.xml') == b'nested'


def test_push_xml_writes_utf8_and_returns_commit_sha(tmp_path):
    target, bare = _init_repo_with_remote(tmp_path)

    sha = push_xml('<beq_title>Ready Player One</beq_title>', target, 'xml/rp1.xml')

    assert _content_on_remote(bare, sha, 'xml/rp1.xml') == b'<beq_title>Ready Player One</beq_title>'


def test_push_image_pushes_and_returns_raw_content_url_with_explicit_owner(tmp_path):
    target, bare = _init_repo_with_remote(tmp_path)

    url = push_image(b'\x89PNG...', target, 'img/rp1.png', owner='3ll3d00d', repo_name='beq-images')

    branch = current_branch(target)
    assert url == f"https://raw.githubusercontent.com/3ll3d00d/beq-images/{branch}/img/rp1.png"
    sha = subprocess.run(['git', '-C', str(target.local_path), 'rev-parse', 'HEAD'],
                         check=True, capture_output=True, text=True).stdout.strip()
    assert _content_on_remote(bare, sha, 'img/rp1.png') == b'\x89PNG...'


def test_push_image_falls_back_to_parsing_the_remote_when_owner_not_given(tmp_path):
    target, bare = _init_repo_with_remote(tmp_path)

    # the local bare-repo remote isn't a github.com URL, so the fallback parse
    # should fail -- proving push_image actually attempts it rather than
    # silently accepting owner=None/repo_name=None
    with pytest.raises(ValueError, match='Unrecognised'):
        push_image(b'\x89PNG...', target, 'img/rp1.png')

    # the content still landed on the remote -- the push itself happens
    # before the URL is built
    sha = subprocess.run(['git', '-C', str(target.local_path), 'rev-parse', 'HEAD'],
                         check=True, capture_output=True, text=True).stdout.strip()
    assert _content_on_remote(bare, sha, 'img/rp1.png') == b'\x89PNG...'


def test_parse_github_remote_handles_https_and_ssh_forms(tmp_path):
    work = tmp_path / 'work'
    work.mkdir()
    _run('git', 'init', '-q', str(work))
    _run('git', '-C', str(work), 'remote', 'add', 'origin', 'https://github.com/3ll3d00d/beq-images.git')
    target = RepoTarget(local_path=str(work))
    assert parse_github_remote(target) == ('3ll3d00d', 'beq-images')

    _run('git', '-C', str(work), 'remote', 'set-url', 'origin', 'git@github.com:3ll3d00d/beq-images.git')
    assert parse_github_remote(target) == ('3ll3d00d', 'beq-images')


def test_parse_github_remote_raises_on_unrecognised_url(tmp_path):
    work = tmp_path / 'work'
    work.mkdir()
    _run('git', 'init', '-q', str(work))
    _run('git', '-C', str(work), 'remote', 'add', 'origin', 'https://gitlab.com/owner/repo.git')
    target = RepoTarget(local_path=str(work))
    with pytest.raises(ValueError, match='Unrecognised'):
        parse_github_remote(target)


def test_pipeline_publish_git_module_has_no_qtpy_import():
    import ast
    import pathlib
    source = (pathlib.Path(__file__).parent / '..' / '..' / 'main' / 'python' / 'pipeline' / 'publish' / 'git.py').resolve()
    tree = ast.parse(source.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(n.name.startswith('qtpy') for n in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith('qtpy')
