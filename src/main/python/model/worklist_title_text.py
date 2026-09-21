'''
What the title page says, and the data it draws, as plain functions with no widgets (split out of `model.worklist_title`,
which re-exports them): the position "12 of 37", the next title waiting for a decision, a candidate's line, the state line
and notice under the title, the chart's curves, and why a decision is not offered (`decision_blocked`, which since chunk 27b
also holds Accept back while the metadata is incomplete).
'''
from typing import Callable, Optional, Sequence

from model.codec import filter_from_json, xydata_from_json
from pipeline.library.index import TitleRow
from pipeline.review import QueueEntry

STATUS_WORDS = {'pending': 'Waiting for a decision', 'accepted': 'Accepted', 'skipped': 'Skipped',
                'rejected': 'Rejected', 'published': 'Published'}

# what each decision may be applied to: a title decided one way can be decided another only by reopening it
ACCEPTABLE = ('pending', 'skipped')
SKIPPABLE = ('pending',)
REJECTABLE = ('pending', 'skipped')


# --- what is said (no widgets) -----------------------------------------------------------------------------------------

def position_text(position: int, total: int) -> str:
    ''' "12 of 37" for the zero-based `position` in a list of `total`; empty when there is no list. '''
    return f'{position + 1:,} of {total:,}' if 0 <= position < total else ''


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


def candidate_text(index: int, candidate) -> str:
    gain = candidate.gain_reduction_db if candidate.gain_reduction_db is not None else 'n/a'
    return (f'{index + 1}: confidence={candidate.confidence:.2f} method={candidate.method} '
            f'mv_adjust_db={candidate.mv_adjust_db:+.1f} gain_reduction_db={gain}')


def entry_title(entry: Optional[QueueEntry], row: Optional[TitleRow], title_id: str, prefer_entry: bool = False) -> str:
    '''
    What to call a title: the index's title, else the entry's, else its id. `prefer_entry` puts the entry first: the row is
    stale once the title was edited on this page.
    '''
    from_entry = str(entry.meta.get('title') or '') if entry else ''
    from_row = row.title if row is not None and row.title else ''
    return (from_entry or from_row if prefer_entry else from_row or from_entry) or title_id


def entry_year(entry: Optional[QueueEntry], row: Optional[TitleRow], prefer_entry: bool = False) -> str:
    from_entry = str(entry.meta.get('year') or '') if entry else ''
    from_row = row.year if row is not None and row.year else ''
    return from_entry or from_row if prefer_entry else from_row or from_entry


def state_text(entry: Optional[QueueEntry], row: Optional[TitleRow], stale: bool = False) -> str:
    '''
    One line: what the title is waiting for. The index's own detail ("conf 0.62 - 3 candidates", "metadata incomplete:
    ...") is added only while the row agrees with the entry -- after a decision made on this page the row is stale, and
    so it is after an edit (`stale`: the metadata it may be talking about has changed).
    '''
    if entry is None:
        if row is None:
            return ''
        return f'{row.needs.capitalize()}: {row.detail}' if row.detail else row.needs.capitalize()
    words = STATUS_WORDS.get(entry.status, entry.status)
    if row is not None and row.detail and not stale and row.review_state == entry.status \
            and row.detail.lower() != entry.status:
        return f'{words}. {row.detail}'
    return words


def notice_text(entry: Optional[QueueEntry], row: Optional[TitleRow], queue_dir: str, error: str) -> str:
    ''' Why there is nothing to choose between, or what the designer said; empty when there is nothing to add. '''
    if error:
        return f'The queue entry could not be read: {error}'
    if entry is not None:
        if entry.decline_reason:
            return f'Declined: {entry.decline_reason} -- {entry.decline_message or ""}'.rstrip(' -')
        return '' if entry.candidates else 'The designer offered no candidates.'
    if not queue_dir:
        return 'No review queue directory is set (Settings > Locations).'
    if row is None:
        return 'This title is not in the index.'
    if row.needs == 'attention':
        return f'Nothing to review: {row.detail}'
    if row.needs in ('extract', 'design'):
        return (f'Nothing to review yet: this title still has to be {"extracted" if row.needs == "extract" else "designed"}. '
                f'Run it from the work list, and it appears here when the design is done.')
    return 'There is no queue entry for this title.'


def chart_data(entry: Optional[QueueEntry], picked: int) -> list:
    '''
    The curves for the chart: the signal as it is (grey) and, over it, the signal with the picked candidate's filter
    applied (red). Nothing for a title with no candidates.
    '''
    if entry is None or not entry.candidates:
        return []
    unfiltered = xydata_from_json(entry.curve)
    unfiltered.colour = 'grey'
    result = [unfiltered]
    if 0 <= picked < len(entry.candidates):
        complete_filter = filter_from_json(entry.candidates[picked].filters)
        filtered = unfiltered.filter(complete_filter.get_transfer_function().get_magnitude())
        filtered.colour = 'red'
        result.append(filtered)
    return result


_SENT_BACK = {'design': 'sent back for redesign', 'extract': 'sent back for re-extraction'}

# What has to be done for a title sent back to be designed again, in the work list and in the Review folder window (which has
# no index and no runs: a Batch Extract & Design run writes the new entry there and Refresh shows it)
REDO_IN_WORK_LIST = 'run Extract & design from the work list'
REDO_IN_FOLDER = 'run Batch Extract & Design on it again, then press Refresh'


def sent_back(revised: str) -> bool:
    ''' Whether `revised` (how far a title was sent back on the page) means its design was cleared: it is not waiting for a decision yet. '''
    return revised in _SENT_BACK


def revised_note(to: str, redo: str = REDO_IN_WORK_LIST) -> str:
    ''' What to add to the state line of a title sent back on this page (its row still says what it was): '' for a plain reopen. '''
    if to not in _SENT_BACK:
        return ''
    return f'It was {_SENT_BACK[to]}: {redo}, and it comes back here for review.'


def decision_blocked(decision: str, entry: Optional[QueueEntry], row: Optional[TitleRow], running: bool,
                     problems: Sequence[str] = (), revised: str = '', redo: str = REDO_IN_WORK_LIST) -> str:
    '''
    Why a decision is not offered even though the entry's status allows it; empty if it is. **Nothing is decided on a
    title a run is working on** (a design in flight writes a new pending entry over whatever is there), and a pending
    entry whose row says the design is out of date or failed is not accepted blind: that is what `derive_needs` keeps a
    person from missing, and once accepted a title is never redesigned. Skip and Reject are still offered then.

    **Accept is not offered while the metadata is incomplete** (`problems`, from `validate()`): the index would call the
    accepted title `review` again and publish would refuse it. Skip and Reject are not held up by metadata.

    `revised` is how far this title was sent back on the page since the rows were read (`design` or `extract`): its design is
    cleared, so -- as for a pending entry whose row says the design is out of date -- it is not accepted until it has been
    designed again (the row is stale, and cannot say so yet). `redo` says how that is done where the page is (`REDO_IN_*`).
    '''
    if running:
        return 'A run is working on this title now: wait for it to finish.'
    if decision == 'accept' and revised in _SENT_BACK:
        return f'Not offered: this title was {_SENT_BACK[revised]}. {redo[0].upper() + redo[1:]} first.'
    if decision == 'accept' and entry is not None and entry.status == 'pending' and row is not None \
            and row.needs in ('extract', 'design', 'attention'):
        return f'Not offered: this title needs {row.needs} first ({row.detail}). Run it from the work list.'
    if decision == 'accept' and problems:
        return 'Fill in the missing metadata first (Metadata tab): ' + '; '.join(problems) + '.'
    return ''
