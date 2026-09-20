'''
Sending a title back into the work list -- design/library-sync/workflow-rework/design.md §12.8.

Three depths, each a superset of the one before:

- **review**: the status goes back to pending, so a person can pick again. Candidates, metadata, artwork and the .beq
  projects are kept.
- **design**: as review, and the design is marked stale so the next run redesigns it. Protection (accepted or
  published) is dropped on purpose; a hand edit to a .beq project still survives, because the projects are written
  behind the hash gate of §3.3.1 whatever asks for the design.
- **extract**: as design, and the recorded extraction is forgotten so the next run runs ffmpeg again.

These only change state. The work itself -- redesigning, re-extracting, republishing -- happens in the next `run` and
`publish`/`commit`, like any other work; nothing here is slow.

What a title's catalogue files need depends on how far it got (§12.7): *written but not committed* means they were
never published, so they are put back as git has them (deleted, or restored to the previous revision); *committed or
pushed* means the catalogue already holds them, so they are left in place and the title becomes a **revision**: the
same catalogue path is rewritten and committed again.
'''
import os
from dataclasses import dataclass, field
from typing import List, Optional

from pipeline.library.extract_cache import invalidate_extract
from pipeline.library.season import invalidate_season_track
from pipeline.publish.catalogue import catalogue_paths
from pipeline.publish.git import RepoTarget, discard_changes, is_committed, is_repo
from pipeline.review import QueueEntry, read_entry, update_entry

REVISE_TARGETS = ('review', 'design', 'extract')

_NOTES = {'review': 'Reopened for review', 'design': 'Sent back for redesign', 'extract': 'Sent back for re-extraction'}


@dataclass(frozen=True)
class ReviseResult:
    entry: QueueEntry
    reverted: List[str] = field(default_factory=list)  # catalogue files whose unpublished changes were discarded
    extract_invalidated: bool = False


def _with_note(existing: Optional[str], text: str) -> str:
    return f'{existing}\n{text}' if existing else text


def _check_repo(target: RepoTarget, name: str) -> None:
    if not is_repo(target):
        raise ValueError(f"{name} {target.local_path!r} is not a git repository (or does not exist)")


def _send_back(queue_dir: str, entry_id: str, to: str, reason: str, *, xml_repo: Optional[RepoTarget],
               images_repo: Optional[RepoTarget], xml_dir: str, image_dir: str, **fields) -> ReviseResult:
    entry = read_entry(queue_dir, entry_id)
    reverted: List[str] = []
    revision = entry.revision
    if entry.status == 'published':
        if xml_repo is None:
            raise ValueError(f"{entry_id!r} is published, so its files are in the catalogue repos: give xml_repo "
                             f"(and images_repo) to say where")
        # Everything that can be checked is checked before anything changes ("ValueError before anything changes")
        _check_repo(xml_repo, 'xml_repo')
        if images_repo is not None:
            _check_repo(images_repo, 'images_repo')
        xml_path, image_path = catalogue_paths(entry_id, xml_dir, image_dir)
        committed = is_committed(xml_repo, xml_path)
        # The order makes a failure part-way retryable: the image goes first and the XML -- whose state decides the
        # revision count -- last, and the entry is written after both, so until it is written it is still 'published'
        # and running this again does the rest (a discard of a file that already matches HEAD does nothing).
        # The one window left is a failure writing the entry itself after the XML was restored: the retry then
        # finds the XML clean and counts a revision that the first attempt would not have.
        reverted_images = discard_changes(images_repo, [image_path]) if images_repo is not None else []
        reverted = discard_changes(xml_repo, [xml_path]) + reverted_images
        if committed and xml_path not in reverted:
            revision += 1  # the catalogue holds this version, and nothing was written over it: a new revision begins
        # (a dirty committed XML is a revision already begun -- by an earlier reopen, or by a republish, which counts
        # it -- so it is not counted again)
    note = _NOTES[to] + (f': {reason}' if reason else '')
    updated = update_entry(queue_dir, entry_id, status='pending', chosen_candidate_index=None,
                           published_digest=None, published_at=None, revision=revision,
                           reviewer_note=_with_note(entry.reviewer_note, note), **fields)
    return ReviseResult(updated, reverted)


def reopen_entry(queue_dir: str, entry_id: str, reason: str = '', *, xml_repo: Optional[RepoTarget] = None,
                 images_repo: Optional[RepoTarget] = None, xml_dir: str = '', image_dir: str = '') -> ReviseResult:
    '''
    Back to pending for a person to decide again: an accepted, skipped, rejected or published entry. Nothing is
    redesigned.
    :param xml_repo/images_repo/xml_dir/image_dir: where a *published* entry's files are; required for one.
    :raises ValueError: if the entry is already pending, or is published and no xml_repo was given.
    :raises FileNotFoundError: if there is no such entry.
    '''
    if read_entry(queue_dir, entry_id).status == 'pending':
        raise ValueError(f'{entry_id!r} is already pending review')
    return _send_back(queue_dir, entry_id, 'review', reason, xml_repo=xml_repo, images_repo=images_repo,
                      xml_dir=xml_dir, image_dir=image_dir)


def redesign_entry(queue_dir: str, entry_id: str, reason: str = '', *, xml_repo: Optional[RepoTarget] = None,
                   images_repo: Optional[RepoTarget] = None, xml_dir: str = '', image_dir: str = '') -> ReviseResult:
    '''
    As reopen_entry(), and the design is marked stale so the next `run` designs it again -- which it never does for
    an accepted or published entry. Metadata, artwork and the reviewer note carry over to the new design, and a
    hand-edited .beq project is kept.
    '''
    return _send_back(queue_dir, entry_id, 'design', reason, xml_repo=xml_repo, images_repo=images_repo,
                      xml_dir=xml_dir, image_dir=image_dir, design_fingerprint=None)


def revise_entry(queue_dir: str, entry_id: str, to: str, reason: str = '', *, work_dir: Optional[str] = None,
                 xml_repo: Optional[RepoTarget] = None, images_repo: Optional[RepoTarget] = None, xml_dir: str = '',
                 image_dir: str = '') -> ReviseResult:
    '''
    :param to: how far back to send it -- one of REVISE_TARGETS.
    :param work_dir: the run's work directory; required for `extract`, whose recorded extraction lives there. (A TV
        season's member episodes are extracted into their own directories and are not forgotten; the joined
        season track is.)
    '''
    if to not in REVISE_TARGETS:
        raise ValueError(f"to must be one of {REVISE_TARGETS}, got {to!r}")
    if to == 'extract' and not work_dir:
        raise ValueError('work_dir is required to re-extract')
    where = dict(xml_repo=xml_repo, images_repo=images_repo, xml_dir=xml_dir, image_dir=image_dir)
    if to == 'review':
        return reopen_entry(queue_dir, entry_id, reason, **where)
    if to == 'design':
        return redesign_entry(queue_dir, entry_id, reason, **where)
    result = _send_back(queue_dir, entry_id, 'extract', reason, design_fingerprint=None, **where)  # raises before we forget anything
    item_dir = os.path.join(work_dir, entry_id)
    invalidated = invalidate_extract(item_dir) | invalidate_season_track(item_dir)
    return ReviseResult(result.entry, result.reverted, extract_invalidated=invalidated)
