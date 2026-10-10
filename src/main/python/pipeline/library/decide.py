'''
Deciding a title's review -- the rules the desktop title page (`model.worklist_title_decide`) and the pipeline service's review
routes share (design/web-review.md §2): which statuses a decision applies to, why a decision is held back, the next title
waiting for one, and the write itself.

The queue entry is the truth for a decision: `decide()` reads it again before writing, and writes nothing if its status no
longer allows the decision, its designs are not the ones the person saw (`offered_digest`), the decision is held back
(`decision_blocked`), the metadata is incomplete (accept), or the pick is a design the designer rejected and the person did
not say they override it. Qt-free: the service imports it.
'''
import hashlib
import json
from dataclasses import asdict
from typing import Callable, Mapping, Optional, Sequence

from pipeline.library.index import TitleRow
from pipeline.library.status import metadata_problems
from pipeline.review import QueueEntry, read_entry, update_entry

STATUS_WORDS = {'pending': 'Waiting for a decision', 'accepted': 'Accepted', 'skipped': 'Skipped',
                'rejected': 'Rejected', 'published': 'Published'}

# what each decision may be applied to: a title decided one way can be decided another only by reopening it
ACCEPTABLE = ('pending', 'skipped')
REJECTABLE = ('pending', 'skipped')

DECISION_STATUS = {'accept': 'accepted', 'reject': 'rejected'}
DECISION_FROM = {'accept': ACCEPTABLE, 'reject': REJECTABLE}

SENT_BACK = {'design': 'sent back for redesign', 'extract': 'sent back for re-extraction'}

# What has to be done for a title sent back to be designed again, in the work list and in the Review folder window (which has
# no index and no runs: a Batch Extract & Design run writes the new entry there and Refresh shows it)
REDO_IN_WORK_LIST = 'run Extract & design from the work list'
REDO_IN_FOLDER = 'run Batch Extract & Design on it again, then press Refresh'
RUN_IN_WORK_LIST = 'Run it from the work list.'


def next_waiting_id(ids: Sequence[str], current: str, waiting: Callable[[str], bool]) -> Optional[str]:
    '''
    The first title after `current` in `ids` that `waiting` says needs a decision, going on from the end round to the
    start (a person working down a list is not sent back to the top, but does not miss what is above), and never
    `current` itself. None if there is no other.
    '''
    try:
        start = list(ids).index(current)
    except ValueError:
        start = -1
    for title_id in list(ids)[start + 1:] + list(ids)[:start + 1]:
        if title_id != current and waiting(title_id):
            return title_id
    return None


def sent_back(revised: str) -> bool:
    ''' Whether `revised` (how far a title was sent back on the page) means its design was cleared: it is not waiting for a decision yet. '''
    return revised in SENT_BACK


def decision_blocked(decision: str, entry: Optional[QueueEntry], row: Optional[TitleRow], running: bool,
                     problems: Sequence[str] = (), revised: str = '', redo: str = REDO_IN_WORK_LIST,
                     run_hint: str = RUN_IN_WORK_LIST) -> str:
    '''
    Why a decision is not offered even though the entry's status allows it; empty if it is. **Nothing is decided on a
    title a run is working on** (a design in flight writes a new pending entry over whatever is there), and a pending
    entry whose row says the design is out of date is not accepted blind: that is what `derive_needs` keeps a person from
    missing, and once accepted a title is never redesigned; Reject is still offered then. **Nothing is decided on a title
    whose last extraction or design failed**: the candidates on screen (if any) are an older design's, so there is nothing
    to accept or reject -- Retry, or another audio stream, is what it needs.

    **Accept is not offered while the metadata is incomplete** (`problems`, from `validate()`): the index would call the
    accepted title `review` again and publish would refuse it. Skip and Reject are not held up by metadata.

    `revised` is how far this title was sent back on the page since the rows were read (`design` or `extract`): its design is
    cleared, so -- as for a pending entry whose row says the design is out of date -- it is not accepted until it has been
    designed again (the row is stale, and cannot say so yet). `redo` says how that is done where the page is (`REDO_IN_*`),
    and `run_hint` how a title that needs a run gets one there.
    '''
    if running:
        return 'A run is working on this title now: wait for it to finish.'
    if row is not None and 'failed' in (row.extract_state, row.design_state):
        stage = 'extraction' if row.extract_state == 'failed' else 'design'
        return (f'Nothing to decide: the last {stage} failed ({row.detail}). Retry it, or choose another audio stream '
                f'from Revise.')
    if decision == 'accept' and revised in SENT_BACK:
        return f'Not offered: this title was {SENT_BACK[revised]}. {redo[0].upper() + redo[1:]} first.'
    if decision == 'accept' and entry is not None and entry.status == 'pending' and row is not None \
            and row.needs in ('extract', 'design', 'attention'):
        return f'Not offered: this title needs {row.needs} first ({row.detail}). {run_hint}'
    if decision == 'accept' and problems:
        return 'Fill in the missing metadata first (Metadata tab): ' + '; '.join(problems) + '.'
    return ''


def offered_digest(entry: QueueEntry) -> str:
    '''
    A stable fingerprint of what a person chooses from -- the entry's `fs` and its `offered` designs, in order -- so a decision
    made on what was seen is refused once a redesign has replaced it. Status, metadata and notes are not part of it.
    '''
    content = {'fs': entry.fs, 'offered': [asdict(candidate) for candidate in entry.offered]}
    return hashlib.sha256(json.dumps(content, sort_keys=True, default=list).encode('utf-8')).hexdigest()[:32]


class DecisionRefused(Exception):
    '''
    A decision `decide()` did not write. `kind` says why: `changed` (the entry is not what was seen: another status, another
    design, metadata no longer complete), `blocked` (`decision_blocked`'s reason), `override` (the pick is a rejected design and
    the override was not given) or `invalid` (the request itself: an unknown decision, a pick out of range).
    '''

    def __init__(self, kind: str, reason: str):
        super().__init__(reason)
        self.kind, self.reason = kind, reason


def decide(queue_dir: str, title_id: str, decision: str, *, seen_digest: str, picked: Optional[int] = None,
           row: Optional[TitleRow] = None, running: bool = False, revised: str = '', redo: str = REDO_IN_WORK_LIST,
           run_hint: str = RUN_IN_WORK_LIST, meta_defaults: Optional[Mapping] = None,
           override_rejection: bool = False) -> QueueEntry:
    '''
    Writes `decision` (`accept` with the `picked` index into `offered`, or `reject`) on the entry, as it is now.
    :param seen_digest: `offered_digest()` of the entry the person decided on.
    :param row: the title's index row, for `decision_blocked` (None in a folder with no index).
    :param running: a run is working on the title.
    :return: the entry as written.
    :raises DecisionRefused: nothing was written, and why.
    :raises FileNotFoundError: there is no entry; OSError: it could not be read or written.
    '''
    if decision not in DECISION_STATUS:
        raise DecisionRefused('invalid', f'unknown decision {decision!r}: it is accept or reject')
    word = DECISION_STATUS[decision]
    if decision == 'accept' and picked is None:
        raise DecisionRefused('invalid', 'Not accepted: no design was chosen.')
    fresh = read_entry(queue_dir, title_id)
    if fresh.status not in DECISION_FROM[decision]:
        raise DecisionRefused('changed', f'Not changed: this title is {fresh.status} now.')
    if offered_digest(fresh) != seen_digest:
        raise DecisionRefused('changed', f'Not {word}: the design of this title changed while it was open. Look at it again.')
    if decision == 'accept' and not 0 <= picked < len(fresh.offered):
        raise DecisionRefused('invalid', f'Not accepted: there is no design {picked + 1}.')
    blocked = decision_blocked(decision, fresh, row, running, revised=revised, redo=redo, run_hint=run_hint)
    if blocked:
        raise DecisionRefused('blocked', blocked)
    fields = {'status': word}
    if decision == 'accept':
        problems = metadata_problems(fresh.meta, meta_defaults)
        if problems:
            raise DecisionRefused('changed', 'Not accepted: the metadata is not complete now: ' + '; '.join(problems) + '.')
        if picked >= len(fresh.candidates) and not override_rejection:
            raise DecisionRefused('override', 'Not accepted: the designer rejected this design, and it was not overridden.')
        fields['chosen_candidate_index'] = picked
    return update_entry(queue_dir, title_id, **fields)
