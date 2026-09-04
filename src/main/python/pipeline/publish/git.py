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
from typing import Optional, Tuple

RAW_CONTENT_TEMPLATE = 'https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}'

_SSH_REMOTE = re.compile(r'^git@github\.com:(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$')
_HTTPS_REMOTE = re.compile(r'^https://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$')


@dataclass(frozen=True)
class RepoTarget:
    ''' A local clone of a git repo to publish into. Pushes whatever branch is currently checked out. '''
    local_path: str
    remote: str = 'origin'


def _git(target: RepoTarget, *args: str) -> str:
    result = subprocess.run(['git', '-C', target.local_path, *args], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def current_branch(target: RepoTarget) -> str:
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


def commit_and_push(target: RepoTarget, relative_path: str, content: bytes, commit_message: str) -> str:
    '''
    Writes content to relative_path within the repo, commits, and pushes
    the current branch.
    :return: the pushed commit's sha.
    '''
    file_path = os.path.join(target.local_path, relative_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        f.write(content)
    _git(target, 'add', relative_path)
    _git(target, 'commit', '-m', commit_message)
    branch = current_branch(target)
    _git(target, 'push', target.remote, f"HEAD:{branch}")
    return _git(target, 'rev-parse', 'HEAD')


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
    if owner is None or repo_name is None:
        parsed_owner, parsed_repo_name = parse_github_remote(target)
        owner = owner or parsed_owner
        repo_name = repo_name or parsed_repo_name
    branch = current_branch(target)
    return RAW_CONTENT_TEMPLATE.format(owner=owner, repo=repo_name, branch=branch, path=relative_path)


def push_xml(xml: str, target: RepoTarget, relative_path: str, commit_message: str = 'Add BEQ filter') -> str:
    '''
    Commits + pushes the beqcatalogue XML to the XML repo. relative_path
    only needs to land somewhere under that repo's configured subdirectory
    -- beqcatalogue globs **/*.xml recursively, so naming/nesting within it
    is this caller's own choice, not a beqcatalogue constraint.
    :return: the pushed commit's sha.
    '''
    return commit_and_push(target, relative_path, xml.encode('utf-8'), commit_message)
