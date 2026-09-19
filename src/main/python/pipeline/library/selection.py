'''
Which titles an action applies to, and what "run through stage X" does to each -- design/library-sync/workflow-rework/
design.md §12.7.

There is one selection vocabulary, shared by the CLI and the work list (and the documentation):

    --needs {attention,extract,design,review,publish,commit,done}    a strip chip in the GUI
    --source NAME     --match TEXT     --id ID     --new-since-scan
    --through {extract,design,publish,commit}                        the action button in the GUI

A `Selection` is those filters as a value, answered by the discovery index (LibraryIndex.titles()); every field that is
given must hold (they are ANDed), and an empty Selection is every title. `selection_from_chip()` is what the strip's
chips select, and `plan_stages()` says, for a selection and a `through`, what would run and what would be skipped and
why -- the GUI's action button label ("Extract & design 120", "3 of 125 skipped") and the CLI's `run` both come from it.

Nothing here reads a source, a queue entry or git: it works from index rows, and is pure apart from `Selection.rows()`.
'''
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

from pipeline.library.index import LibraryIndex, TitleRow
from pipeline.library.state import NEEDS

THROUGH = ('extract', 'design', 'publish', 'commit')   # the --through vocabulary, in pipeline order
MACHINE_STAGES = ('extract', 'design')                 # what the run loop does (run.run_unit)

# The pipeline strip's chips, by the label the work list shows, and the Selection each one applies. A chip is one
# `needs` value, except *New*, which is the titles first seen by the latest scan whatever they need.
CHIP_NEW = 'New'
CHIPS = ('Attention', CHIP_NEW, 'Extract', 'Design', 'Review', 'Publish', 'Commit', 'Done')


def _unique(values: Iterable[str]) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(values))


@dataclass(frozen=True)
class Selection:
    '''
    :param needs: titles whose next need is one of these (state.NEEDS); empty means any.
    :param source: only titles owned by this source of the profile, by name.
    :param match: case-insensitive text found in the title, display name, id or path.
    :param ids: only these titles (catalogue ids).
    :param new_since_scan: only titles first seen by the latest scan.
    '''
    needs: Tuple[str, ...] = ()
    source: Optional[str] = None
    match: Optional[str] = None
    ids: Tuple[str, ...] = ()
    new_since_scan: bool = False

    def __post_init__(self):
        object.__setattr__(self, 'needs', _unique(self.needs))
        object.__setattr__(self, 'ids', _unique(self.ids))
        object.__setattr__(self, 'source', self.source or None)
        object.__setattr__(self, 'match', self.match or None)
        unknown = [n for n in self.needs if n not in NEEDS]
        if unknown:
            raise ValueError(f"needs must be from {', '.join(NEEDS)}; got {', '.join(unknown)}")

    @property
    def is_empty(self) -> bool:
        ''' True if nothing narrows it: every title. '''
        return self == Selection()

    def rows(self, index: LibraryIndex) -> List[TitleRow]:
        ''' The matching titles, in work-list order (tier, then oldest waiting first). '''
        return index.titles(needs=list(self.needs) or None, source=self.source, match=self.match,
                            ids=list(self.ids) if self.ids else None, new_only=self.new_since_scan)

    def describe(self) -> str:
        ''' The selection in words, for a log line or a confirmation. '''
        parts = [f"needs {' or '.join(self.needs)}"] if self.needs else []
        if self.source:
            parts.append(f'from {self.source}')
        if self.match:
            parts.append(f'matching "{self.match}"')
        if self.ids:
            parts.append(f'{len(self.ids)} named title{"s" if len(self.ids) != 1 else ""}')
        if self.new_since_scan:
            parts.append('new since the last scan')
        return ', '.join(parts) or 'every title'


def selection_from_chip(chip: str, *, source: Optional[str] = None, match: Optional[str] = None,
                        ids: Sequence[str] = ()) -> Selection:
    '''
    What a strip chip selects, on top of the source combo and the search box (which narrow it, as `--source` and
    `--match` do). *New* is `--new-since-scan`; every other chip is `--needs` with its own name.
    :raises ValueError: for a chip that is not in CHIPS.
    '''
    if chip not in CHIPS:
        raise ValueError(f"unknown chip {chip!r}; the chips are {', '.join(CHIPS)}")
    if chip == CHIP_NEW:
        return Selection(source=source, match=match, ids=tuple(ids), new_since_scan=True)
    return Selection(needs=(chip.lower(),), source=source, match=match, ids=tuple(ids))


# --- what runs ---------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Skipped:
    id: str
    title: str
    reason: str


@dataclass(frozen=True)
class Planned:
    ''' A title the action will work on, and the stages it will run (a subset of THROUGH, in order). '''
    row: TitleRow
    stages: Tuple[str, ...]


@dataclass(frozen=True)
class StagePlan:
    through: str
    planned: List[Planned] = field(default_factory=list)
    skipped: List[Skipped] = field(default_factory=list)

    def with_stage(self, stage: str) -> List[Planned]:
        return [p for p in self.planned if stage in p.stages]

    @property
    def label(self) -> str:
        '''
        What the action button says, from the counts alone: "Extract & design 120", plus what it leaves out.
        '''
        verb = {'extract': 'Extract', 'design': 'Extract & design', 'publish': 'Publish', 'commit': 'Commit'}[
            self.through]
        total = len(self.planned) + len(self.skipped)
        text = f'{verb} {len(self.planned)}'
        return f'{text} ({len(self.skipped)} of {total} skipped)' if self.skipped else text


def _title(row: TitleRow) -> str:
    return row.title or row.display_name or row.id


def _failed_stage(row: TitleRow) -> Optional[str]:
    return 'extract' if row.extract_state == 'failed' else 'design' if row.design_state == 'failed' else None


def _upto(stages: Iterable[str], through: str) -> Tuple[str, ...]:
    limit = THROUGH.index(through)
    return tuple(s for s in stages if THROUGH.index(s) <= limit)


def plan_stages(rows: Iterable[TitleRow], through: str, *, retry_failed: bool = False) -> StagePlan:
    '''
    What "run through `through`" does to each title, by what it needs next (design.md §12.7):

    - **needs extract** -- extract, and design too if `through` reaches it (the user never picks prerequisites);
    - **needs design** -- design;
    - **needs publish** -- publish (a first publish, or a republish of a published title that is out of date), only
      when `through` is publish or commit;
    - **needs commit** -- commit, only when `through` is commit; a title that is published *by this run* is also
      committed by it;
    - **needs attention** because extract or design failed -- nothing, unless `retry_failed`, which runs it again;
    - **needs review** is a person's, so a title never goes past design to publish on its own (only an *accepted*
      title needs publish), and everything else -- done, a project conflict, a changed source -- is skipped, with
      the reason.
    :raises ValueError: for a `through` that is not in THROUGH.
    '''
    if through not in THROUGH:
        raise ValueError(f"through must be one of {', '.join(THROUGH)}; got {through!r}")
    plan = StagePlan(through)
    for row in rows:
        stages: Tuple[str, ...] = ()
        reason = ''
        if row.needs == 'extract':
            stages = _upto(('extract', 'design'), through)
        elif row.needs == 'design':
            stages = _upto(('design',), through)
            reason = 'already extracted'
        elif row.needs == 'publish':
            stages = _upto(('publish', 'commit'), through)
            reason = 'publishing needs --through publish (or commit)'
        elif row.needs == 'commit':
            stages = _upto(('commit',), through)
            reason = 'committing needs --through commit'
        elif row.needs == 'attention':
            failed = _failed_stage(row)
            if failed is None:
                reason = row.detail or 'needs attention'
            elif retry_failed:
                stages = _upto(('extract', 'design') if failed == 'extract' else ('design',), through)
                reason = 'already extracted'
            else:
                reason = f'{row.detail} -- unchanged since; retry failed to try again'
        elif row.needs == 'review':
            reason = 'waiting for a person to review it'
        else:
            reason = row.detail or 'done'
        if stages:
            plan.planned.append(Planned(row, stages))
        else:
            plan.skipped.append(Skipped(row.id, _title(row), reason or 'nothing to do'))
    return plan
