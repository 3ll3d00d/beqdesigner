'''
Commits and pushes what pipeline.review.publish_reviewed_queue(push=False) wrote into the catalogue repos --
design/library-sync/workflow-rework/design.md §12.7.

Publishing and committing are separate so a batch costs one commit and one push per repo rather than two pushes per
title, and so the human can look at the working trees in between. Whether a title is committed or pushed is never
stored: it is read back from git (`repo_state()`), so a commit made by hand is respected and a failed push is simply
retried by running this again.
'''
import logging
import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Collection, List, Optional, Sequence

from pipeline.publish.catalogue import aggregate_path, catalogue_paths
from pipeline.publish.git import RepoState, RepoTarget, fs_path, commit_paths, committed_paths, push, repo_state
from pipeline.review import QueueEntry, read_entry, read_queue

logger = logging.getLogger('library_commit')

_TITLES_IN_MESSAGE = 8


@dataclass(frozen=True)
class RepoCommit:
    repo: str                       # the repo's local path
    paths: List[str]                # the published files that were not yet committed or pushed, and so were handled
    commit: Optional[str] = None    # the new commit's sha; None if everything was already committed
    pushed: bool = False


@dataclass(frozen=True)
class CatalogueCommit:
    xml: RepoCommit
    images: Optional[RepoCommit] = None
    missing: List[str] = field(default_factory=list)  # published entries whose file is not in the repo tree
    # published entries whose file is in the tree but not in HEAD even after committing -- git is ignoring it
    # (a .gitignore rule), so it will never reach the catalogue
    not_committed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)  # things a person should look at; nothing was blocked


def _message(what: str, entries: Sequence[QueueEntry]) -> str:
    titles = [e.meta.get('title') or e.id for e in dict((e.id, e) for e in entries).values()]
    shown = ', '.join(titles[:_TITLES_IN_MESSAGE])
    if len(titles) > _TITLES_IN_MESSAGE:
        shown += f' and {len(titles) - _TITLES_IN_MESSAGE} more'
    return f"Publish {len(titles)} {what}{'s' if len(titles) != 1 else ''}: {shown}"


def _commit_repo(target: RepoTarget, what: str, wanted: dict, push_changes: bool, missing: List[str],
                 not_committed: List[str]) -> RepoCommit:
    '''
    :param wanted: {relative_path: entry} for every published entry, whatever its git state
    :raises subprocess.CalledProcessError: as commit_catalogue(); the error's `repo_commit` is what this repo got
        done before git refused (a commit whose push was rejected).
    '''
    present = {path: entry for path, entry in wanted.items() if os.path.isfile(fs_path(target, path))}
    missing.extend(path for path in wanted if path not in present)
    state: RepoState = repo_state(target)
    # unknown (None) means "cannot tell", so every present file is a candidate: committing an unchanged one is a no-op
    candidates = [p for p in present if state.uncommitted is None or p in state.uncommitted]
    commit = commit_paths(target, candidates, _message(what, [present[p] for p in candidates])) if candidates else None
    # a file that is still not in HEAD was never committed, whatever `git status` said (a .gitignore rule hides it
    # from status, so it was never a candidate): the entry is 'published' but will never reach the catalogue
    held = committed_paths(target, list(present))
    not_committed.extend(p for p in present if p not in held)
    # nothing published means nothing to push (and possibly no commit to push: a brand new clone has no HEAD)
    unpushed = bool(present) and (state.unpushed is None or commit is not None
                                  or any(p in state.unpushed for p in present))
    handled = [p for p in present if p in candidates or (state.unpushed is not None and p in state.unpushed)]
    if push_changes and unpushed:
        try:
            push(target)
        except subprocess.CalledProcessError as error:
            error.repo_commit = RepoCommit(target.local_path, handled, commit, pushed=False)
            raise
    return RepoCommit(target.local_path, handled, commit, pushed=push_changes and unpushed)


def _image_url_warnings(xml_repo: RepoTarget, xml_paths: Sequence[str]) -> List[str]:
    '''
    A committed record that names an image URL, committed without an images repo: the image is not committed or pushed
    here, so if it is not on the remote yet the XML will point at nothing.
    '''
    warnings = []
    for path in xml_paths:
        try:
            with open(fs_path(xml_repo, path), 'r', encoding='utf-8') as f:
                record = json.load(f)
        except (OSError, ValueError):
            continue
        if isinstance(record, dict) and record.get('images'):
            warnings.append(f"{path} names a report image but no images repository was given, so the image is not "
                            f"committed or pushed with it: give --images-repo, or push the image first, or the "
                            f"record will point at nothing")
    return warnings


def commit_catalogue(queue_dir: str, xml_repo: RepoTarget, images_repo: Optional[RepoTarget] = None, *,
                     xml_dir: str = '', image_dir: str = '', push: bool = True,
                     ids: Optional[Collection[str]] = None) -> CatalogueCommit:
    '''
    Commits every 'published' entry's files that git does not already have, one commit per repo containing exactly
    those paths (nothing else staged or changed in the tree is touched), then pushes each repo once. **The images
    repo goes first**: a filter record must never reach the remote before the image its URL points at.

    Safe to repeat: a file already committed is skipped, one already pushed is not pushed again, and a failed push
    is simply retried. If the images push fails the XML repo is not touched, so the next run resumes in order.

    :param push: False commits locally only, to inspect before anything leaves the machine.
    :param ids: commit only these titles' files (a selection); None is every published entry. What else is
        uncommitted in the repos is left for a later commit, and a push still sends whatever the branch has ahead.
    :raises subprocess.CalledProcessError: if git refuses (a rejected push, say) -- what was committed stays committed.
        It is a pipeline.publish.git.GitError, whose message says what git said, and it carries `partial`: the
        CatalogueCommit of what was done before the failure (a repo not reached is an empty RepoCommit).

    Two things are reported rather than left silent: `not_committed` (a published file that git ignores never gets
    committed, whatever this does) and `warnings` (an XML committed with an image URL but no images repo to commit
    the image to). Neither stops the commit.
    '''
    if ids is None:
        published = [e for e in read_queue(queue_dir) if e.status == 'published']
    else:
        found = [read_entry(queue_dir, i) for i in dict.fromkeys(ids) if os.path.isfile(os.path.join(queue_dir, f'{i}.json'))]
        published = [e for e in found if e.status == 'published']
    missing: List[str] = []
    not_committed: List[str] = []
    images: Optional[RepoCommit] = None
    xml: Optional[RepoCommit] = None
    try:
        if images_repo is not None:
            images = _commit_repo(images_repo, 'report image',
                                  {catalogue_paths(e.id, xml_dir, image_dir)[1]: e for e in published}, push, missing,
                                  not_committed)
        filter_files = {catalogue_paths(e.id, xml_dir, image_dir)[0]: e for e in published}
        # Every publish regenerates this derived file.  Commit it with the
        # individual records so a consumer never sees a fresh record with a
        # stale repository aggregate.
        if published:
            filter_files[aggregate_path(xml_dir)] = published[0]
        xml = _commit_repo(xml_repo, 'BEQ filter', filter_files, push, missing,
                           not_committed)
    except subprocess.CalledProcessError as error:
        failed = getattr(error, 'repo_commit', None)   # a commit made, whose push was refused
        if failed is not None:
            if images_repo is not None and images is None:
                images = failed
            else:
                xml = failed
        error.partial = CatalogueCommit(xml=xml or RepoCommit(xml_repo.local_path, []), images=images, missing=missing,
                                        not_committed=not_committed)
        raise
    warnings = _image_url_warnings(xml_repo, xml.paths) if images_repo is None else []
    if missing:
        logger.warning('published entries with no file in their repo: %s', ', '.join(missing))
    if not_committed:
        logger.warning('published files git will not commit (ignored by a .gitignore rule?): %s',
                       ', '.join(not_committed))
    for warning in warnings:
        logger.warning(warning)
    return CatalogueCommit(xml=xml, images=images, missing=missing, not_committed=not_committed, warnings=warnings)
