'''
What the title page says, and the data it draws, as plain functions with no widgets (split out of `model.worklist_title`,
which re-exports them): the position "12 of 37", the next title waiting for a decision, a candidate's line, the state line
and notice under the title, the chart's curves, and why a decision is not offered (`decision_blocked`, which since chunk 27b
also holds Accept back while the metadata is incomplete).
'''
import html
import re
from typing import Mapping, Optional

from model.preferences import AVG_SPECLAB_COLOURS, PEAK_SPECLAB_COLOURS
from pipeline.library.bass import describe as describe_bass_management
from pipeline.library.decide import ACCEPTABLE, REDO_IN_FOLDER, REDO_IN_WORK_LIST, REJECTABLE, SENT_BACK, STATUS_WORDS, \
    decision_blocked, next_waiting_id, sent_back  # noqa: F401 (moved to the pipeline, re-exported for the app's imports)
from pipeline.library.index import TitleRow
from pipeline.library.review_chart import chart_curves
from pipeline.review import DECLINED_METHOD, QueueEntry


# --- what is said (no widgets) -----------------------------------------------------------------------------------------

def position_text(position: int, total: int) -> str:
    ''' "12 of 37" for the zero-based `position` in a list of `total`; empty when there is no list. '''
    return f'{position + 1:,} of {total:,}' if 0 <= position < total else ''


def candidate_text(index: int, candidate) -> str:
    '''
    One line of the candidate list. `index` counts through `QueueEntry.offered`, so a design the designer rejected keeps
    its number after the candidates (the digit keys pick by it) and says it was rejected, and how many reasons were given.
    '''
    if candidate.method == DECLINED_METHOD:
        return f'{index + 1}: no filter -- the designer declined (does not require BEQ)'
    gain = candidate.gain_reduction_db if candidate.gain_reduction_db is not None else 'n/a'
    line = (f'{index + 1}: confidence={candidate.confidence:.2f} method={candidate.method} '
            f'mv_adjust_db={candidate.mv_adjust_db:+.1f} gain_reduction_db={gain}')
    if candidate.rejection_reasons:
        count = len(candidate.rejection_reasons)
        line = f'{index + 1}: REJECTED ({count} reason{"" if count == 1 else "s"}) {line.partition(": ")[2]}'
    return line


def rejected_heading(count: int) -> str:
    ''' The line between the candidates and the designs the designer rejected. '''
    return f'Rejected by the designer ({count:,}) -- for review; accepting one overrides the designer'


def rejection_html(candidate) -> str:
    '''
    A rejected design's reasons first (the one thing to know about it), one bullet each and **never split**: a reason may
    itself contain `; ` (beqforge's "corrected only down to 24.7 Hz; content continues to 16.7 Hz"), which
    commentary_html() would take for two notes. Then the design's own commentary, as any candidate's.
    '''
    reasons = ''.join(f'<li>{html.escape(reason)}</li>' for reason in candidate.rejection_reasons or [])
    return (f'<p style="margin-bottom:2px"><b>Why the designer rejected it</b></p><ul style="margin-top:0">{reasons}</ul>'
            + commentary_html(candidate.commentary))


def playback_html(entry: Optional[QueueEntry]) -> str:
    '''
    What the designer was told the playback chain is (R8), below its commentary: its clipping figures are for that chain,
    or for one it assumed when none was sent (beqforge's `clipping` note says which).
    '''
    if entry is None:
        return ''
    said = describe_bass_management(entry.bass_management)
    designed = ''
    if entry.designer or entry.designer_build:   # D4: which designer, and which build of it, made this design
        build = entry.designer_build or 'did not say which build'
        designed = (f'<p style="margin-top:8px"><b>Designed by</b><br>'
                    f'{html.escape(entry.designer or "the designer")}, {html.escape(build)}</p>')
    return designed + f'<p style="margin-top:8px"><b>Bass management sent to the designer</b><br>{html.escape(said)}</p>'


def override_question(index: int, candidate) -> str:
    ''' What a person is asked before accepting a design the designer rejected. '''
    reasons = ''.join(f'\n  - {reason}' for reason in candidate.rejection_reasons or [])
    return (f'The designer rejected design {index + 1} as unfit to publish:{reasons}\n\n'
            f'Accept it anyway? It will be published as your override of the designer.')


def decline_commentary(reason: Optional[str], message: Optional[str]) -> dict:
    '''
    A decline laid out like a candidate's commentary, so the page shows it in the same detail. The contract gives only
    `decline_reason` and a free-text `decline_message`, and the message is never machine-parsed: this only splits it for
    reading. A designer that writes `summary | found: a; b [provenance]` (beqforge does) gets a Summary, a heading per
    `label: ...` part and its provenance on its own; any other message is the Summary, whole.
    '''
    commentary = {'decline_reason': reason or ''} if reason else {}
    text = (message or '').strip()
    provenance = re.search(r'\s*\[([^\[\]]+)\]\s*$', text)
    if provenance:
        text = text[:provenance.start()].rstrip()
    for i, part in enumerate(p.strip() for p in text.split(' | ')):
        if not part:
            continue
        label, sep, rest = part.partition(': ')
        if i and sep and re.fullmatch(r'[A-Za-z][A-Za-z _-]{0,30}', label) and label not in commentary:
            commentary[label] = rest
        else:
            key = 'summary' if 'summary' not in commentary else f'detail {i}'
            commentary[key] = part
    if provenance:
        commentary['provenance'] = provenance.group(1)
    return commentary



def commentary_html(commentary: Optional[Mapping]) -> str:
    '''
    A candidate's commentary as wrapping text: each key a bold heading (`target_notes` reads "Target notes") over its
    value, and a value made of `; `-separated notes a bulleted list of them, since the designer's notes run long.
    '''
    parts = []
    for key, value in (commentary or {}).items():
        heading = html.escape(str(key).replace('_', ' ').strip().capitalize())
        notes = [n.strip() for n in str(value).split('; ') if n.strip()]
        if len(notes) > 1:
            body = '<ul style="margin-top:0">' + ''.join(f'<li>{html.escape(n)}</li>' for n in notes) + '</ul>'
        else:
            body = f'<p style="margin-top:0">{html.escape(str(value))}</p>'
        parts.append(f'<p style="margin-bottom:2px"><b>{heading}</b></p>{body}')
    return ''.join(parts)


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
    if entry.status in ('accepted', 'published') and entry.overrides_rejection:
        words += f' design {entry.chosen_candidate_index + 1}, which the designer rejected (your override)'
    if entry.decline_reason:
        return words   # the row's detail is the decline reason, which the commentary already gives in full
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
            return ''   # the commentary says why, in full: saying it here as well only repeated it
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
    The selected track's average and peak mono mix (`pipeline.library.review_chart.chart_curves`), in the main chart's measure
    colours and before/after line styles: dashed before when a design is applied, solid after.
    '''
    curves = chart_curves(entry, picked)
    has_filter = any(curve.filtered for curve in curves)
    result = []
    for curve in curves:
        curve.data.colour = (AVG_SPECLAB_COLOURS if curve.kind == 'average' else PEAK_SPECLAB_COLOURS)[0]
        curve.data.linestyle = '-' if curve.filtered or not has_filter else '--'
        result.append(curve.data)
    return result


def revised_note(to: str, redo: str = REDO_IN_WORK_LIST) -> str:
    ''' What to add to the state line of a title sent back on this page (its row still says what it was): '' for a plain reopen. '''
    if to not in SENT_BACK:
        return ''
    return f'It was {SENT_BACK[to]}: {redo}, and it comes back here for review.'
