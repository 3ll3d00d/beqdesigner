'''
Batch-design driver + on-disk review queue -- design/candidate-review-plan.md
phase 1. A designer's DesignResponse.candidates (design/designer-interface.md
§3) can carry several ranked filters per title; this module is what lets
Session.design() run over many titles unattended and hands the results to a
human to pick from later, rather than requiring the pick to happen inline.

Deliberately Qt-free, like the rest of pipeline/ -- reuses model.iir's
CompleteFilter.to_json()/model.codec.filter_from_json (already the published
.filter format, docs/schema/filter.schema.json) to serialise each
candidate's filters, rather than inventing a parallel format for
BiquadSpec/DesignCandidate.
'''
import json
import os
from dataclasses import asdict, dataclass, field, replace
from typing import List, Optional, Sequence, Tuple

from pipeline.config import AnalysisConfig
from pipeline.designer.contract import Coverage
from pipeline.orchestrate import Applied, Declined, DesignOutcome, Session

VALID_STATUSES = {'pending', 'accepted', 'skipped', 'rejected', 'published'}


@dataclass
class CandidateSummary:
    '''
    One candidate's filters (already realised at the entry's fs, as a
    CompleteFilter.to_json() dict) plus the provenance a reviewer needs to
    judge it. Index 0 in QueueEntry.candidates is always the designer's top
    pick -- same ordering rule as DesignResponse.candidates.
    '''
    filters: dict
    confidence: float
    method: str
    mv_adjust_db: float  # NOT a clipping-cost estimate -- see gain_reduction_db
    gain_reduction_db: Optional[float] = None
    residual_db: Optional[float] = None
    residual_band_hz: Optional[tuple] = None
    commentary: Optional[dict] = None


@dataclass
class QueueEntry:
    '''
    One title pending (or having received) review. `id` is also the
    filename (`<id>.json`) -- callers pick it, batch_design() does not
    invent one, so it should be stable across re-runs of the same title.
    '''
    id: str
    fs: int                          # publish-target fs the candidates were realised at
    meta: dict                       # BeqMetadata.to_dict()-shaped; may be partial/empty if
                                     # title metadata hasn't been resolved yet
    curve: dict                      # one MagnitudeData (avg, unfiltered) via model.codec.xydata_to_json
    candidates: List[CandidateSummary] = field(default_factory=list)  # empty on decline
    decline_reason: Optional[str] = None
    decline_message: Optional[str] = None
    status: str = 'pending'
    chosen_candidate_index: Optional[int] = None
    reviewer_note: Optional[str] = None

    def __post_init__(self):
        if self.status not in VALID_STATUSES:
            raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}, got {self.status!r}")
        if self.status == 'accepted':
            if self.chosen_candidate_index is None:
                raise ValueError("status='accepted' requires chosen_candidate_index")
            if not (0 <= self.chosen_candidate_index < len(self.candidates)):
                raise ValueError(
                    f"chosen_candidate_index {self.chosen_candidate_index} out of range "
                    f"for {len(self.candidates)} candidate(s)")


def _entry_path(queue_dir: str, entry_id: str) -> str:
    return os.path.join(queue_dir, f"{entry_id}.json")


def write_queue_entry(queue_dir: str, entry: QueueEntry) -> None:
    os.makedirs(queue_dir, exist_ok=True)
    with open(_entry_path(queue_dir, entry.id), 'w', encoding='utf-8') as f:
        json.dump(asdict(entry), f)


def _entry_from_dict(d: dict) -> QueueEntry:
    d = dict(d)
    d['candidates'] = [CandidateSummary(**c) for c in d.get('candidates', [])]
    return QueueEntry(**d)


def read_queue(queue_dir: str) -> List[QueueEntry]:
    ''' :return: every entry in queue_dir, sorted pending-first then by id. '''
    entries = []
    for name in sorted(os.listdir(queue_dir)):
        if name.endswith('.json'):
            with open(os.path.join(queue_dir, name), 'r', encoding='utf-8') as f:
                entries.append(_entry_from_dict(json.load(f)))
    entries.sort(key=lambda e: (e.status != 'pending', e.id))
    return entries


def read_entry(queue_dir: str, entry_id: str) -> QueueEntry:
    path = _entry_path(queue_dir, entry_id)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No queue entry {entry_id!r} in {queue_dir}")
    with open(path, 'r', encoding='utf-8') as f:
        return _entry_from_dict(json.load(f))


def update_entry(queue_dir: str, entry_id: str, **fields) -> QueueEntry:
    '''
    Read-modify-write: applies `fields` to the existing entry (validating
    the result the same way construction does -- e.g. status='accepted'
    without chosen_candidate_index is rejected) and rewrites its file.
    :raises FileNotFoundError: if no entry `entry_id` exists yet -- entries
        are only ever created by batch_design(), never by this function.
    '''
    updated = replace(read_entry(queue_dir, entry_id), **fields)
    write_queue_entry(queue_dir, updated)
    return updated


def _outcome_to_entry(entry_id: str, fs: int, meta: dict, curve: dict, outcome: DesignOutcome) -> QueueEntry:
    if isinstance(outcome, Declined):
        return QueueEntry(id=entry_id, fs=fs, meta=meta, curve=curve, candidates=[],
                          decline_reason=outcome.reason, decline_message=outcome.message)
    assert isinstance(outcome, Applied)
    candidates = [CandidateSummary(filters=outcome.filters.to_json(), confidence=outcome.confidence,
                                   method=outcome.method, mv_adjust_db=outcome.mv_adjust_db,
                                   gain_reduction_db=outcome.gain_reduction_db,
                                   residual_db=outcome.residual_db, residual_band_hz=outcome.residual_band_hz,
                                   commentary=outcome.commentary)]
    candidates += [CandidateSummary(filters=alt.filters.to_json(), confidence=alt.confidence, method=alt.method,
                                    mv_adjust_db=alt.mv_adjust_db, gain_reduction_db=alt.gain_reduction_db,
                                    residual_db=alt.residual_db,
                                    residual_band_hz=alt.residual_band_hz, commentary=alt.commentary)
                  for alt in outcome.alternatives]
    return QueueEntry(id=entry_id, fs=fs, meta=meta, curve=curve, candidates=candidates)


def batch_design(items: Sequence[Tuple[str, str, Optional[dict]]], designer: str, queue_dir: str, work_dir: str,
                 config: AnalysisConfig = AnalysisConfig(), coverage: Coverage = 'complete_programme',
                 bass_management: Optional[dict] = None) -> List[str]:
    '''
    Runs Session.extract/load/design for each item and writes one
    QueueEntry (status 'pending') to queue_dir per title -- never calls
    set_filters/publish; that only happens for an entry a human has since
    marked 'accepted' (pipeline.review.apply_reviewed_entry/
    publish_reviewed_queue).
    :param items: (id, source_path, meta) triples. `id` is the queue entry's
        stable id/filename. `meta` is BeqMetadata.to_dict()-shaped, or None
        if title metadata (e.g. a TMDB lookup) hasn't been resolved yet --
        resolving it at review time instead of batch time is deliberately
        supported.
    :param work_dir: scratch directory for extracted audio, one
        subdirectory per item.
    :param bass_management: this batch's bass-management configuration, if
        any -- see Session.design(); the same one is used for every item.
    :return: the ids written, in `items` order.
    '''
    from model.codec import xydata_to_json

    session = Session(config)
    written = []
    for entry_id, source_path, meta in items:
        extracted = session.extract(source_path, os.path.join(work_dir, entry_id))
        sig = session.load(extracted, name=entry_id)
        outcome = session.design(sig, designer, coverage=coverage, bass_management=bass_management)
        curve = xydata_to_json(session.curves(sig, kind='avg', filtered=False))
        entry = _outcome_to_entry(entry_id, sig.signal.fs, meta or {}, curve, outcome)
        write_queue_entry(queue_dir, entry)
        written.append(entry_id)
    return written
