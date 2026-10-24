'''
Keeping an out-of-date design rather than redesigning it (contract 1.6, design/designer-interface.md §3 "Reviewed
designs"; design/redesign-scope.md step 2). A person who has read what a redesign now would change -- usually the
designer's `/impact` answer -- may keep the design: it is stamped *reviewed at* the current identity, and no longer counts
as out of date against it. Nothing else about the design changes: its filters, its rejected designs and the identity it
was designed at stay as they were, and are what get published.
'''
from dataclasses import replace
from datetime import datetime, timezone
from typing import Callable, Optional

from pipeline.designer.identity import CurrentIdentity, is_out_of_date, reviewed_stamp
from pipeline.library.variants import designed
from pipeline.review import QueueEntry, read_entry, write_queue_entry


class ReviewRefused(ValueError):
    ''' Why a design cannot be kept at the current identity, in words. '''


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def review_refused(entry: Optional[QueueEntry], current: Optional[CurrentIdentity]) -> Optional[str]:
    ''' Why `entry`'s design cannot be stamped reviewed at `current`, in words; None when it can. '''
    if current is None:
        return 'no designer that is up says its design version'
    if current.disagreement:
        return 'the designers that are up would design differently: ' + '; '.join(current.disagreement)
    if entry is None or not designed(entry):
        return 'it has no design to keep'
    if not is_out_of_date(entry.design_identity, current):
        return f'its design is not out of date against {current.describe()}'
    return None


def mark_reviewed(queue_dir: str, entry_id: str, current: Optional[CurrentIdentity], by: Optional[str] = None,
                  now: Callable[[], str] = _now) -> QueueEntry:
    '''
    Stamps `entry_id`'s design reviewed at `current` and writes it.
    :param by: who decided, for the record (None: not known).
    :raises ReviewRefused: as `review_refused` says.
    :raises FileNotFoundError: no such entry.
    '''
    entry = read_entry(queue_dir, entry_id)
    refused = review_refused(entry, current)
    if refused:
        raise ReviewRefused(f'{entry_id}: {refused}')
    kept = replace(entry, reviewed_at=[*entry.reviewed_at, reviewed_stamp(current, by, now())])
    write_queue_entry(queue_dir, kept)
    return kept
