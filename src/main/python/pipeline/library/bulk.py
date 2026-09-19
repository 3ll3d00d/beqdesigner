'''
Bulk accept -- design/library-sync/workflow-rework/design.md §12.9.

Accepting is the human gate, so bulk accept is deliberately narrow: from a selection it takes only the titles that are
*waiting for review*, whose designer's top pick is at least `threshold` confident, and leaves out -- and reports --
any of those that a person should still look at:

- **incomplete metadata** (pipeline.metadata.validate(): no title, year or audio type), because the accepted title could
  not be published;
- **a decline**: the designer offered nothing, so there is no top pick to accept;
- **an edited project**: someone changed the filter in the title's `.beq` project since it was designed, so the top
  pick is no longer the filter that would ship.

Each title accepted gets the top candidate chosen and a reviewer note, "bulk accepted, confidence >= 0.90".
A title the index calls `review` whose confidence is *below* the threshold is not a candidate at all (it is counted, not
reported as an exclusion). `plan_accept()` is the confirmation's content -- how many, and which are left out -- and
`accept_top_pick()` does it.

Each title is judged from its queue entry, not from the index row, so a stale index cannot accept what is no longer
pending.
'''
from dataclasses import dataclass, field
from typing import List, Optional

from pipeline.library.index import LibraryIndex
from pipeline.library.selection import Selection
from pipeline.library.status import metadata_problems
from pipeline.review import QueueEntry, project_paths, read_entry, update_entry

DEFAULT_ACCEPT_THRESHOLD = 0.90   # the caller's preference (the GUI stores it); this is the default


@dataclass(frozen=True)
class Exclusion:
    id: str
    title: str
    reason: str


@dataclass(frozen=True)
class AcceptPlan:
    threshold: float
    eligible: List[str] = field(default_factory=list)          # ids that would be accepted
    excluded: List[Exclusion] = field(default_factory=list)    # confident enough, but not for bulk accept, and why
    below_threshold: int = 0                                   # waiting for review, top pick less confident
    not_for_review: int = 0                                    # in the selection but not waiting for review


@dataclass(frozen=True)
class AcceptReport:
    threshold: float
    note: str
    accepted: List[str] = field(default_factory=list)
    excluded: List[Exclusion] = field(default_factory=list)
    below_threshold: int = 0
    not_for_review: int = 0


def accept_note(threshold: float) -> str:
    return f'bulk accepted, confidence >= {threshold:.2f}'


def _reasons(entry: QueueEntry, meta_defaults: Optional[dict], work_dir: Optional[str]) -> List[str]:
    from pipeline.publish.project import edited_projects
    reasons = []
    if entry.status != 'pending':
        reasons.append(f'already {entry.status}')
    if not entry.candidates:
        reasons.append(f"designer declined: {entry.decline_message or entry.decline_reason or 'no reason given'}")
    problems = metadata_problems(entry.meta, meta_defaults)
    if problems:
        reasons.append('metadata incomplete: ' + '; '.join(problems))
    if work_dir and entry.candidates:
        _, mono, multichannel, _ = project_paths(work_dir, entry.id)
        try:
            edited = edited_projects(mono, multichannel)
        except (OSError, ValueError, KeyError) as error:
            reasons.append(f'a project cannot be read ({type(error).__name__}): check it before accepting')
        else:
            if edited:
                reasons.append(f'the {edited} project was edited since it was designed: the top pick is not what '
                               f'would be published')
    return reasons


def plan_accept(index: LibraryIndex, selection: Selection, threshold: float = DEFAULT_ACCEPT_THRESHOLD, *,
                queue_dir: str, meta_defaults: Optional[dict] = None, work_dir: Optional[str] = None) -> AcceptPlan:
    '''
    What bulk accept would do to `selection`, without changing anything (the confirmation dialog's content).
    :param meta_defaults: what publish is given (metadata a title lacks is filled from it before it is validated).
    :param work_dir: the run's work directory, where the `.beq` projects are; without it edited projects are not looked for.
    '''
    eligible: List[str] = []
    excluded: List[Exclusion] = []
    below = not_review = 0
    for row in selection.rows(index):
        if row.needs != 'review':
            not_review += 1
            continue
        title = row.title or row.display_name or row.id
        try:
            entry = read_entry(queue_dir, row.id)
        except FileNotFoundError:
            excluded.append(Exclusion(row.id, title, 'it has no queue entry'))
            continue
        if entry.candidates and entry.candidates[0].confidence < threshold:
            below += 1
            continue
        reasons = _reasons(entry, meta_defaults, work_dir)
        if reasons:
            excluded.append(Exclusion(row.id, title, '; '.join(reasons)))
        else:
            eligible.append(row.id)
    return AcceptPlan(threshold, eligible, excluded, below, not_review)


def accept_top_pick(index: LibraryIndex, selection: Selection, threshold: float = DEFAULT_ACCEPT_THRESHOLD, *,
                    queue_dir: str, meta_defaults: Optional[dict] = None, work_dir: Optional[str] = None
                    ) -> AcceptReport:
    '''
    Accepts the designer's top pick for every title plan_accept() finds eligible: status `accepted`, candidate 0 chosen, and
    a reviewer note (added to any note already there). It does not publish; the titles now need *publish*, which
    `run --through publish` or `publish` does. The index is not refreshed: call `index.refresh(profile, settings)` afterwards.
    :param threshold: the smallest top-pick confidence accepted (compared with >=), default 0.90.
    '''
    plan = plan_accept(index, selection, threshold, queue_dir=queue_dir, meta_defaults=meta_defaults, work_dir=work_dir)
    note = accept_note(threshold)
    accepted: List[str] = []
    for entry_id in plan.eligible:
        entry = read_entry(queue_dir, entry_id)
        update_entry(queue_dir, entry_id, status='accepted', chosen_candidate_index=0,
                     reviewer_note=f'{entry.reviewer_note}\n{note}' if entry.reviewer_note else note)
        accepted.append(entry_id)
    return AcceptReport(threshold, note, accepted, plan.excluded, plan.below_threshold, plan.not_for_review)
