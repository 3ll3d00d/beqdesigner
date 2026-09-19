'''
Phase 4 (item 10) of design/pipeline-implementation-plan.md: commit + push
mechanics for the XML and images repos. Exercises real local git repos
(a bare "remote" + a working clone) rather than mocking subprocess, so the
add/commit/push sequence is genuinely verified -- no network access
required, everything stays on the local filesystem.
'''
import os
import subprocess

import pytest

from pipeline.publish.git import (RepoState, RepoTarget, commit_and_push, commit_paths, current_branch, discard_changes,
                                  image_url, is_committed, parse_github_remote, push, push_image, push_xml, repo_state, write_files)


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


def test_rewriting_a_published_path_with_changed_content_is_a_new_commit_at_the_same_path(tmp_path):
    ''' A revision (design/library-sync/workflow-rework §12.7): same catalogue path, new commit -- works today. '''
    target, bare = _init_repo_with_remote(tmp_path)
    first = commit_and_push(target, 'xml/rp1.xml', b'v1', 'Add RP1')

    second = commit_and_push(target, 'xml/rp1.xml', b'v2', 'Revise RP1')

    assert second != first
    assert _content_on_remote(bare, second, 'xml/rp1.xml') == b'v2'
    assert _content_on_remote(bare, first, 'xml/rp1.xml') == b'v1'


def test_committing_unchanged_content_is_a_no_op_not_an_error(tmp_path):
    target, bare = _init_repo_with_remote(tmp_path)
    first = commit_and_push(target, 'xml/rp1.xml', b'same', 'Add RP1')

    again = commit_and_push(target, 'xml/rp1.xml', b'same', 'Add RP1 again')

    assert again == first


def test_a_commit_contains_only_the_published_path(tmp_path):
    target, bare = _init_repo_with_remote(tmp_path)
    (tmp_path / 'work' / 'foreign.txt').write_text('not ours')
    _run('git', '-C', target.local_path, 'add', 'foreign.txt')

    sha = commit_and_push(target, 'xml/rp1.xml', b'<beq/>', 'Add RP1')

    changed = subprocess.run(['git', '-C', target.local_path, 'show', '--name-only', '--format=', sha],
                             check=True, capture_output=True, text=True).stdout.split()
    assert changed == ['xml/rp1.xml']


def _head(target):
    return subprocess.run(['git', '-C', target.local_path, 'rev-parse', 'HEAD'], check=True, capture_output=True,
                          text=True).stdout.strip()


def _files_in(target, sha):
    return subprocess.run(['git', '-C', target.local_path, 'show', '--name-only', '--format=', sha], check=True,
                          capture_output=True, text=True).stdout.split()


def _remote_head(bare, branch):
    return subprocess.run(['git', '-C', str(bare), 'rev-parse', branch], check=True, capture_output=True,
                          text=True).stdout.strip()


def test_write_files_only_touches_the_working_tree(tmp_path):
    target, bare = _init_repo_with_remote(tmp_path)

    write_files(target, {'xml/a.xml': b'a', 'img/deep/b.png': b'b'})

    assert (tmp_path / 'work' / 'xml' / 'a.xml').read_bytes() == b'a'
    assert (tmp_path / 'work' / 'img' / 'deep' / 'b.png').read_bytes() == b'b'
    assert repo_state(target).uncommitted == {'xml/a.xml', 'img/deep/b.png'}
    assert subprocess.run(['git', '-C', target.local_path, 'log'], capture_output=True).returncode != 0  # no commits


def test_commit_paths_commits_the_first_commit_of_an_empty_repo_and_returns_its_sha(tmp_path):
    target, bare = _init_repo_with_remote(tmp_path)
    write_files(target, {'xml/a.xml': b'a'})

    sha = commit_paths(target, ['xml/a.xml'], 'Add a')

    assert sha == _head(target)
    assert _files_in(target, sha) == ['xml/a.xml']


def test_commit_paths_commits_several_paths_as_one_commit_and_leaves_other_changes_alone(tmp_path):
    target, bare = _init_repo_with_remote(tmp_path)
    write_files(target, {'keep.txt': b'k'})
    commit_paths(target, ['keep.txt'], 'Add keep')
    write_files(target, {'xml/a.xml': b'a', 'xml/b.xml': b'b', 'keep.txt': b'edited by someone', 'new.txt': b'n'})

    sha = commit_paths(target, ['xml/a.xml', 'xml/b.xml'], 'Add two')

    assert sorted(_files_in(target, sha)) == ['xml/a.xml', 'xml/b.xml']
    assert repo_state(target).uncommitted == {'keep.txt', 'new.txt'}  # not swept in, not lost


def test_commit_paths_returns_none_when_the_paths_already_match_head(tmp_path):
    target, bare = _init_repo_with_remote(tmp_path)
    write_files(target, {'a.xml': b'a'})
    first = commit_paths(target, ['a.xml'], 'Add a')

    assert commit_paths(target, ['a.xml'], 'again') is None
    assert commit_paths(target, [], 'nothing') is None
    assert _head(target) == first


def test_push_publishes_the_branch_without_setting_an_upstream(tmp_path):
    target, bare = _init_repo_with_remote(tmp_path)
    write_files(target, {'a.xml': b'a'})
    sha = commit_paths(target, ['a.xml'], 'Add a')

    push(target)

    assert _remote_head(bare, current_branch(target)) == sha
    assert repo_state(target).unpushed is None  # still no upstream configured: unknown, not "nothing"


def test_current_branch_works_before_the_first_commit(tmp_path):
    target, bare = _init_repo_with_remote(tmp_path)

    assert current_branch(target) in ('main', 'master')


def test_image_url_is_known_before_anything_is_committed(tmp_path):
    target, bare = _init_repo_with_remote(tmp_path)

    url = image_url(target, os.path.join('img', 'rp1.png'), owner='3ll3d00d', repo_name='beq-images')

    assert url == f"https://raw.githubusercontent.com/3ll3d00d/beq-images/{current_branch(target)}/img/rp1.png"
    assert not (tmp_path / 'work' / 'img').exists()


def _track_remote(target):
    _run('git', '-C', target.local_path, 'branch', f'--set-upstream-to=origin/{current_branch(target)}')


def test_repo_state_classifies_uncommitted_unpushed_and_pushed_paths(tmp_path):
    target, bare = _init_repo_with_remote(tmp_path)
    commit_and_push(target, 'pushed.xml', b'p', 'Add pushed')
    _track_remote(target)
    write_files(target, {'committed.xml': b'c'})
    commit_paths(target, ['committed.xml'], 'Add committed')
    write_files(target, {'dir/untracked.xml': b'u'})
    write_files(target, {'pushed.xml': b'edited'})

    state = repo_state(target)

    assert state == RepoState(uncommitted=frozenset({'dir/untracked.xml', 'pushed.xml'}),
                              unpushed=frozenset({'committed.xml'}))


def test_repo_state_is_clean_after_everything_is_pushed(tmp_path):
    target, bare = _init_repo_with_remote(tmp_path)
    commit_and_push(target, 'a.xml', b'a', 'Add a')
    _track_remote(target)

    assert repo_state(target) == RepoState(frozenset(), frozenset())


def test_repo_state_follows_a_hand_made_commit(tmp_path):
    target, bare = _init_repo_with_remote(tmp_path)
    commit_and_push(target, 'a.xml', b'a', 'Add a')
    _track_remote(target)
    write_files(target, {'b.xml': b'b'})
    _run('git', '-C', target.local_path, 'add', 'b.xml')
    _run('git', '-C', target.local_path, 'commit', '-q', '-m', 'by hand')

    assert repo_state(target) == RepoState(frozenset(), frozenset({'b.xml'}))


def test_repo_state_reports_a_rename_by_its_new_path_only(tmp_path):
    target, bare = _init_repo_with_remote(tmp_path)
    commit_and_push(target, 'old.xml', b'a long enough body to be detected as a rename', 'Add old')
    _run('git', '-C', target.local_path, 'mv', 'old.xml', 'new.xml')

    assert repo_state(target).uncommitted == {'new.xml'}


def test_repo_state_degrades_to_unknown_outside_a_repo_instead_of_raising(tmp_path):
    state = repo_state(RepoTarget(local_path=str(tmp_path / 'not-a-repo')))

    assert state == RepoState(None, None)


def test_repo_state_survives_a_detached_head_and_a_repo_with_no_upstream(tmp_path):
    target, bare = _init_repo_with_remote(tmp_path)
    commit_and_push(target, 'a.xml', b'a', 'Add a')
    _run('git', '-C', target.local_path, 'checkout', '-q', '--detach')

    assert repo_state(target) == RepoState(frozenset(), None)


def test_is_committed_reads_head_not_the_working_tree(tmp_path):
    target, bare = _init_repo_with_remote(tmp_path)
    assert not is_committed(target, 'a.xml')  # no commits at all yet
    write_files(target, {'a.xml': b'a', 'b.xml': b'b'})
    commit_paths(target, ['a.xml'], 'Add a')

    assert is_committed(target, 'a.xml') and not is_committed(target, 'b.xml')
    (tmp_path / 'work' / 'a.xml').unlink()
    assert is_committed(target, 'a.xml')  # deleted in the tree, still in HEAD


def test_discard_changes_restores_committed_files_deletes_uncommitted_ones_and_ignores_the_rest(tmp_path):
    target, bare = _init_repo_with_remote(tmp_path)
    write_files(target, {'tracked.xml': b'v1', 'clean.xml': b'c', 'other.txt': b'o'})
    commit_paths(target, ['tracked.xml', 'clean.xml', 'other.txt'], 'Add')
    write_files(target, {'tracked.xml': b'v2', 'other.txt': b'edited', 'new.xml': b'n'})

    discarded = discard_changes(target, ['tracked.xml', 'clean.xml', 'new.xml', 'never-existed.xml'])

    assert sorted(discarded) == ['new.xml', 'tracked.xml']
    assert (tmp_path / 'work' / 'tracked.xml').read_bytes() == b'v1'
    assert not (tmp_path / 'work' / 'new.xml').exists()
    assert (tmp_path / 'work' / 'other.txt').read_bytes() == b'edited'  # not named, so not touched
    assert repo_state(target).uncommitted == {'other.txt'}


def test_discard_changes_removes_a_staged_new_file_from_the_index_too(tmp_path):
    target, bare = _init_repo_with_remote(tmp_path)
    write_files(target, {'new.xml': b'n'})
    _run('git', '-C', target.local_path, 'add', 'new.xml')

    assert discard_changes(target, ['new.xml']) == ['new.xml']

    assert repo_state(target).uncommitted == frozenset()


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
