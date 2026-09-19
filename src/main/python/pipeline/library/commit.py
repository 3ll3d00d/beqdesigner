'''
Commits and pushes what pipeline.review.publish_reviewed_queue(push=False) wrote into the catalogue repos --
design/library-sync/workflow-rework/design.md §12.7.

Publishing and committing are separate so a batch costs one commit and one push per repo rather than two pushes per
title, and so the human can look at the working trees in between. Whether a title is committed or pushed is never
stored: it is read back from git (`repo_state()`), so a commit made by hand is respected and a failed push is simply
retried by running this again.
'''
import logging
import os
from dataclasses import dataclass, field
from typing import Collection, List, Optional, Sequence

from pipeline.publish.catalogue import catalogue_paths
from pipeline.publish.git import RepoState, RepoTarget, commit_paths, push, repo_state
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


def _message(what: str, entries: Sequence[QueueEntry]) -> str:
    titles = [e.meta.get('title') or e.id for e in entries]
    shown = ', '.join(titles[:_TITLES_IN_MESSAGE])
    if len(titles) > _TITLES_IN_MESSAGE:
        shown += f' and {len(titles) - _TITLES_IN_MESSAGE} more'
    return f"Publish {len(titles)} {what}{'s' if len(titles) != 1 else ''}: {shown}"


def _commit_repo(target: RepoTarget, what: str, wanted: dict, push_changes: bool, missing: List[str]) -> RepoCommit:
    '''
    :param wanted: {relative_path: entry} for every published entry, whatever its git state
    '''
    present = {path: entry for path, entry in wanted.items()
               if os.path.isfile(os.path.join(target.local_path, path))}
    missing.extend(path for path in wanted if path not in present)
    state: RepoState = repo_state(target)
    # unknown (None) means "cannot tell", so every present file is a candidate: committing an unchanged one is a no-op
    candidates = [p for p in present if state.uncommitted is None or p in state.uncommitted]
    commit = commit_paths(target, candidates, _message(what, [present[p] for p in candidates])) if candidates else None
    # nothing published means nothing to push (and possibly no commit to push: a brand new clone has no HEAD)
    unpushed = bool(present) and (state.unpushed is None or commit is not None
                                  or any(p in state.unpushed for p in present))
    if push_changes and unpushed:
        push(target)
    handled = [p for p in present if p in candidates or (state.unpushed is not None and p in state.unpushed)]
    return RepoCommit(target.local_path, handled, commit, pushed=push_changes and unpushed)


def commit_catalogue(queue_dir: str, xml_repo: RepoTarget, images_repo: Optional[RepoTarget] = None, *,
                     xml_dir: str = '', image_dir: str = '', push: bool = True,
                     ids: Optional[Collection[str]] = None) -> CatalogueCommit:
    '''
    Commits every 'published' entry's files that git does not already have, one commit per repo containing exactly
    those paths (nothing else staged or changed in the tree is touched), then pushes each repo once. **The images
    repo goes first**: an XML must never reach the remote before the image its URL points at.

    Safe to repeat: a file already committed is skipped, one already pushed is not pushed again, and a failed push
    is simply retried. If the images push fails the XML repo is not touched, so the next run resumes in order.

    :param push: False commits locally only, to inspect before anything leaves the machine.
    :param ids: commit only these titles' files (a selection); None is every published entry. What else is
        uncommitted in the repos is left for a later commit, and a push still sends whatever the branch has ahead.
    :raises subprocess.CalledProcessError: if git refuses (a rejected push, say) -- what was committed stays committed.
    '''
    if ids is None:
        published = [e for e in read_queue(queue_dir) if e.status == 'published']
    else:
        found = [read_entry(queue_dir, i) for i in dict.fromkeys(ids) if os.path.isfile(os.path.join(queue_dir, f'{i}.json'))]
        published = [e for e in found if e.status == 'published']
    missing: List[str] = []
    images = None
    if images_repo is not None:
        images = _commit_repo(images_repo, 'report image',
                              {catalogue_paths(e.id, xml_dir, image_dir)[1]: e for e in published}, push, missing)
    xml = _commit_repo(xml_repo, 'BEQ filter', {catalogue_paths(e.id, xml_dir, image_dir)[0]: e for e in published},
                       push, missing)
    if missing:
        logger.warning('published entries with no file in their repo: %s', ', '.join(missing))
    return CatalogueCommit(xml=xml, images=images, missing=missing)
