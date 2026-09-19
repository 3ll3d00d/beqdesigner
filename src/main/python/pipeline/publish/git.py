'''
pipeline/publish/git.py: commit + push the report image and the beqcatalogue
XML into their target repos -- design/pipeline-implementation-plan.md phase
4 (item 10).

Per D3's resolution (design/api-headless-pipeline.md §6/§14): two separate
repos -- a small XML-only repo beqcatalogue itself clones, and a second
repo for report images referenced by GitHub raw-content URL. No PR step;
beqcatalogue is triggered by a push (via a repository_dispatch fired by a
copy of its trigger.yaml workflow in the XML repo, set up once, out of
band -- not this module's job). This module only does repo mechanics:
write a file, commit, push, and (for images) hand back the URL that
BeqMetadata.spectrum_url/.pva_url must be set to.

D5 (CLI vs service) is resolved here as: this pipeline runs as the
invoking user, using whatever git credentials/SSH agent/credential helper
are already configured for them on this machine -- shells out to the
user's own `git`, the same as they'd use interactively, rather than
managing any credential of its own. A service deployment would need a
different identity story for this module specifically; no earlier phase
is affected by that choice.

Sequencing note: the image push has to happen *before* to_beq_xml() runs
(the resulting raw URL is beq_spectrumURL/beq_pvaURL) -- that ordering is
the caller's job (pipeline.orchestrate, phase 5), not this module's.
'''
import os
import re
import subprocess
from dataclasses import dataclass
from typing import FrozenSet, List, Mapping, Optional, Sequence, Tuple

RAW_CONTENT_TEMPLATE = 'https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}'

_SSH_REMOTE = re.compile(r'^git@github\.com:(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$')
_HTTPS_REMOTE = re.compile(r'^https://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$')


@dataclass(frozen=True)
class RepoTarget:
    ''' A local clone of a git repo to publish into. Pushes whatever branch is currently checked out. '''
    local_path: str
    remote: str = 'origin'


def _git_raw(target: RepoTarget, *args: str) -> str:
    result = subprocess.run(['git', '-C', target.local_path, *args], check=True, capture_output=True, text=True)
    return result.stdout


def _git(target: RepoTarget, *args: str) -> str:
    return _git_raw(target, *args).strip()


def current_branch(target: RepoTarget) -> str:
    ''' The checked-out branch's name -- also on a repo with no commits yet, where `rev-parse` has nothing to read. '''
    try:
        return _git(target, 'symbolic-ref', '--short', 'HEAD')
    except subprocess.CalledProcessError:  # detached HEAD
        return _git(target, 'rev-parse', '--abbrev-ref', 'HEAD')


def parse_github_remote(target: RepoTarget) -> Tuple[str, str]:
    '''
    :return: (owner, repo_name) parsed from the remote's configured URL.
    :raises ValueError: if the remote isn't a recognised github.com URL --
        pass owner/repo_name explicitly to push_image() instead if the
        remote uses an SSH host alias or anything else this can't parse.
    '''
    url = _git(target, 'remote', 'get-url', target.remote)
    for pattern in (_SSH_REMOTE, _HTTPS_REMOTE):
        m = pattern.match(url)
        if m:
            return m.group('owner'), m.group('repo')
    raise ValueError(f"Unrecognised GitHub remote URL: {url!r}")


def write_files(target: RepoTarget, files: Mapping[str, bytes]) -> None:
    ''' Writes each {relative_path: content} into the repo's working tree. Touches nothing in git. '''
    for relative_path, content in files.items():
        file_path = os.path.join(target.local_path, relative_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            f.write(content)


def commit_paths(target: RepoTarget, relative_paths: Sequence[str], commit_message: str) -> Optional[str]:
    '''
    Commits exactly `relative_paths` -- not whatever else happens to be staged in the working tree (a pathspec
    commit) -- and nothing more.
    :return: the new commit's sha, or None if those paths already match HEAD ("nothing to commit" is a success:
        re-publishing an unchanged file must not be an error).
    '''
    paths = list(relative_paths)
    if not paths:
        return None
    _git(target, 'add', '--', *paths)
    if not _git(target, 'status', '--porcelain', '--', *paths):
        return None
    _git(target, 'commit', '-m', commit_message, '--', *paths)
    return _git(target, 'rev-parse', 'HEAD')


def push(target: RepoTarget) -> None:
    ''' Pushes the checked-out branch to the remote, without touching the repo's upstream configuration. '''
    _git(target, 'push', target.remote, f"HEAD:{current_branch(target)}")


def commit_and_push(target: RepoTarget, relative_path: str, content: bytes, commit_message: str) -> str:
    '''
    Writes content to relative_path within the repo, commits just that path, and pushes the current branch.
    :return: the sha of the commit at the branch's tip -- this one, or the existing tip if the content was unchanged.
    '''
    write_files(target, {relative_path: content})
    commit_paths(target, [relative_path], commit_message)
    push(target)
    return _git(target, 'rev-parse', 'HEAD')


def image_url(target: RepoTarget, relative_path: str, owner: Optional[str] = None,
              repo_name: Optional[str] = None) -> str:
    '''
    :return: the GitHub raw-content URL `relative_path` will have once pushed. It needs only the owner, repo and
        branch, so it is known before anything is committed or pushed -- which is what lets the XML be written
        (with the image's URL in it) ahead of the push.
    :param owner/repo_name: parsed from the remote if not given.
    '''
    if owner is None or repo_name is None:
        parsed_owner, parsed_repo_name = parse_github_remote(target)
        owner = owner or parsed_owner
        repo_name = repo_name or parsed_repo_name
    return RAW_CONTENT_TEMPLATE.format(owner=owner, repo=repo_name, branch=current_branch(target),
                                       path=relative_path.replace(os.sep, '/'))


@dataclass(frozen=True)
class RepoState:
    '''
    Where a repo's files stand relative to git, read from git itself so a commit made by hand is respected.
    Paths are relative to the repo root. Either set is None when it could not be determined (not a repo, or --
    for `unpushed` -- the branch has neither an upstream nor a remote-tracking ref of its own name to compare with);
    callers must treat None as "unknown", never as "nothing".
    '''
    uncommitted: Optional[FrozenSet[str]]  # changed, staged or untracked in the working tree
    unpushed: Optional[FrozenSet[str]]     # differing between the upstream branch and HEAD


def _porcelain_paths(status: str) -> FrozenSet[str]:
    ''' Paths from `git status --porcelain=v1 -z`: `XY path\\0`, and for a rename/copy a second `orig\\0`. '''
    paths, fields = set(), iter(status.split('\0'))
    for field in fields:
        if len(field) < 4:
            continue
        paths.add(field[3:])
        if field[0] in 'RC' or field[1] in 'RC':
            next(fields, None)  # the rename's source path
    return frozenset(paths)


def _remote_tracking_ref(target: RepoTarget) -> Optional[str]:
    '''
    `<remote>/<branch>` for the checked-out branch, if that remote-tracking ref exists -- what `push()` updates, and so
    what "pushed" means for a clone whose branch has no `@{upstream}` configured. None on a detached HEAD (there is no
    branch to compare) or when the branch was never pushed.
    '''
    try:
        branch = _git(target, 'symbolic-ref', '--short', '-q', 'HEAD')
        ref = f'refs/remotes/{target.remote}/{branch}'
        _git(target, 'rev-parse', '--verify', '--quiet', ref)
    except (subprocess.CalledProcessError, OSError):
        return None
    return ref


def repo_state(target: RepoTarget) -> RepoState:
    '''
    One `git status` and one `git diff` for the whole repo, however many titles it holds. Never raises. What is
    unpushed is measured against the branch's upstream, or -- where none is configured (a repo made with `git init`
    and `git remote add`, pushed with `push()`, which sets none) -- against the remote-tracking ref of the same name,
    if there is one; only a branch that was never pushed, or a detached HEAD, is unknown.
    '''
    try:
        uncommitted = _porcelain_paths(_git_raw(target, 'status', '--porcelain=v1', '-z', '--untracked-files=all'))
    except (subprocess.CalledProcessError, OSError):
        uncommitted = None
    unpushed = None
    try:
        unpushed = frozenset(p for p in _git_raw(target, 'diff', '--name-only', '-z', '@{upstream}..HEAD').split('\0')
                             if p)
    except (subprocess.CalledProcessError, OSError):
        fallback = _remote_tracking_ref(target)
        if fallback is not None:
            try:
                unpushed = frozenset(p for p in _git_raw(target, 'diff', '--name-only', '-z',
                                                         f'{fallback}..HEAD').split('\0') if p)
            except (subprocess.CalledProcessError, OSError):
                unpushed = None
    return RepoState(uncommitted, unpushed)


def is_committed(target: RepoTarget, relative_path: str) -> bool:
    ''' True if HEAD contains `relative_path` -- whatever the working tree says about it now. '''
    return subprocess.run(['git', '-C', target.local_path, 'cat-file', '-e', f'HEAD:{relative_path}'],
                          capture_output=True).returncode == 0


def discard_changes(target: RepoTarget, relative_paths: Sequence[str]) -> List[str]:
    '''
    Puts each path back as HEAD has it: a committed file is restored to its committed content, one that was never
    committed (written by a publish and not yet committed) is deleted. Paths that already match HEAD are left
    alone, and nothing outside `relative_paths` is touched.
    :return: the paths that had changes to discard.
    '''
    discarded = []
    for path in relative_paths:
        if not _git_raw(target, 'status', '--porcelain=v1', '-z', '--untracked-files=all', '--', path):
            continue
        if is_committed(target, path):
            _git(target, 'checkout', 'HEAD', '--', path)
        else:
            _git(target, 'rm', '-q', '-f', '--cached', '--ignore-unmatch', '--', path)  # in case it was staged
            full_path = os.path.join(target.local_path, path)
            if os.path.isfile(full_path):
                os.remove(full_path)
        discarded.append(path)
    return discarded


def push_image(png_bytes: bytes, target: RepoTarget, relative_path: str,
               commit_message: str = 'Add report image', owner: Optional[str] = None,
               repo_name: Optional[str] = None) -> str:
    '''
    Commits + pushes a report image to the images repo.
    :param owner: the GitHub owner to build the raw URL from; parsed from
        the remote if not given.
    :param repo_name: the GitHub repo name to build the raw URL from;
        parsed from the remote if not given.
    :return: the pushed file's GitHub raw-content URL -- the value
        BeqMetadata.spectrum_url/.pva_url must be set to before
        pipeline.publish.xml.to_beq_xml() runs.
    '''
    commit_and_push(target, relative_path, png_bytes, commit_message)
    return image_url(target, relative_path, owner, repo_name)


def push_xml(xml: str, target: RepoTarget, relative_path: str, commit_message: str = 'Add BEQ filter') -> str:
    '''
    Commits + pushes the beqcatalogue XML to the XML repo. relative_path
    only needs to land somewhere under that repo's configured subdirectory
    -- beqcatalogue globs **/*.xml recursively, so naming/nesting within it
    is this caller's own choice, not a beqcatalogue constraint.
    :return: the pushed commit's sha.
    '''
    return commit_and_push(target, relative_path, xml.encode('utf-8'), commit_message)
