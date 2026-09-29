'''
Queue entries for the title page tests: the real `pipeline.review` files, written to a queue directory, so what the page
reads and writes is what the pipeline does.
'''
import numpy as np

from model.codec import xydata_to_json
from model.iir import CompleteFilter, LowShelf, PeakingEQ
from model.xy import MagnitudeData
from pipeline.review import CandidateSummary, QueueEntry, write_queue_entry


def curve_json(kind: str = 'avg') -> dict:
    x = np.linspace(1.0, 500.0, 50)
    return xydata_to_json(MagnitudeData(kind, '', x, np.full_like(x, 10.0 if kind == 'peak' else 0.0)))


def candidates(count: int = 2):
    ''' Up to two candidates: a low shelf (the top pick) and a peaking filter, each with commentary. '''
    low_shelf = CompleteFilter(fs=1000, filters=[LowShelf(1000, 20, 0.7, 4.5)])
    peaking = CompleteFilter(fs=1000, filters=[PeakingEQ(1000, 100, 1, -3.0)])
    return [
        CandidateSummary(filters=low_shelf.to_json(), confidence=0.9, method='fitted', mv_adjust_db=4.0,
                         gain_reduction_db=-1.0, commentary={'note': 'top pick'}),
        CandidateSummary(filters=peaking.to_json(), confidence=0.4, method='fitted', mv_adjust_db=1.0,
                         gain_reduction_db=0.0, commentary={'note': 'alternative', 'extra': 'more'}),
    ][:count]


def rejected_designs(count: int = 1):
    ''' Designs the designer judged unfit to publish (contract 1.1): a steep low shelf, each with its reasons. '''
    return [CandidateSummary(filters=CompleteFilter(fs=1000, filters=[LowShelf(1000, 27 + i, 5.8, 16.0)]).to_json(),
                             confidence=0.95, method='non_parametric', mv_adjust_db=16.0, gain_reduction_db=-2.0,
                             commentary={'strategy': 'flatten'},
                             rejection_reasons=[f'introduces a cliff of {53 + i} dB/oct at 17 Hz',
                                                'corrected only down to 24.7 Hz'])
            for i in range(count)]


def complete_meta(entry_id: str) -> dict:
    ''' Metadata `validate()` accepts: a page offers Accept only for a title whose metadata is complete (chunk 27b). '''
    return {'title': entry_id, 'year': '2001', 'audio_types': ['DD 5.1']}


def write_entry(queue_dir, entry_id: str, status: str = 'pending', count: int = 2, chosen=None, decline: bool = False,
                meta=None, reverse: bool = False, rejected: int = 0) -> QueueEntry:
    '''
    `status='accepted'` needs `chosen` (defaults to the top pick). A decline has no candidates. `reverse` writes the
    candidates in the other order: a redesign that offers the same number of candidates, but not the same ones. `meta` is
    complete metadata unless given (`{'title': id}` is what a library run leaves and is *not* acceptable). `rejected` adds that
    many designs the designer rejected (`chosen` may point at one: `len(candidates) + i`).
    '''
    meta = complete_meta(entry_id) if meta is None else meta
    if status in ('accepted', 'published') and chosen is None:
        chosen = 0
    if decline:
        entry = QueueEntry(id=entry_id, fs=1000, meta=meta, curve=curve_json(), peak_curve=curve_json('peak'), candidates=[],
                           decline_reason='no_rolloff_detected', decline_message='nothing found', status=status,
                           rejected=rejected_designs(rejected),
                           chosen_candidate_index=chosen if status in ('accepted', 'published') else None)
    else:
        entry = QueueEntry(id=entry_id, fs=1000, meta=meta, curve=curve_json(), peak_curve=curve_json('peak'),
                           candidates=candidates(count)[::-1] if reverse else candidates(count), status=status,
                           chosen_candidate_index=chosen if status in ('accepted', 'published') else None,
                           rejected=rejected_designs(rejected))
    write_queue_entry(queue_dir, entry)
    return entry
