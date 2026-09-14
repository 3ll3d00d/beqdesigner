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
from typing import Callable, List, Optional, Sequence, Tuple

from pipeline.config import AnalysisConfig
from pipeline.designer.contract import Coverage
from pipeline.orchestrate import Applied, Declined, DesignOutcome, Session
from pipeline.publish.git import RepoTarget
from pipeline.publish.report import ReportSpec

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
    meta: dict                       # pipeline.metadata.BeqMetadata *constructor* kwargs (title/year/
                                     # audio_types/genres/...), not its beq_-prefixed to_json() shape --
                                     # BeqMetadata(**meta) must reconstruct it. May be partial/empty if
                                     # title metadata hasn't been resolved yet (see batch_design)
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
    '''
    :return: every entry in queue_dir, sorted pending-first then by id -- [] if queue_dir does not exist yet
        (e.g. a remembered-but-not-yet-written default), rather than raising.
    '''
    if not os.path.isdir(queue_dir):
        return []
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


def design_and_queue(session: Session, entry_id: str, wav_path: str, designer: str, queue_dir: str,
                     meta: Optional[dict] = None, coverage: Coverage = 'complete_programme',
                     bass_management: Optional[dict] = None, channels: Optional[dict] = None) -> QueueEntry:
    '''
    Loads an *already-extracted* wav file, designs it, and writes one
    QueueEntry to queue_dir -- the load+design+curve+write half of
    batch_design(), split out so a caller that has already extracted audio
    itself (e.g. a GUI dialog with its own per-file extraction UI) can
    design+queue without redoing extraction. `session` may be shared/reused
    across many calls -- Session holds no mutable per-call state.
    :param entry_id: the queue entry's stable id/filename (see batch_design).
    :param wav_path: path to an already-extracted (mono) wav file -- the primary signal design() runs
        against (DesignRequest.mono_mix). Must be mono; `channels` is a separate, purely additive input.
    :param meta: BeqMetadata *constructor* kwargs, or None if unresolved yet.
    :param channels: DesignRequest.channels -- see Session.design()/Session.load_channels(). Optional.
    :return: the written QueueEntry.
    '''
    from model.codec import xydata_to_json

    sig = session.load(wav_path, name=entry_id)
    outcome = session.design(sig, designer, coverage=coverage, bass_management=bass_management, channels=channels)
    curve = xydata_to_json(session.curves(sig, kind='avg', filtered=False))
    entry = _outcome_to_entry(entry_id, sig.signal.fs, meta or {}, curve, outcome)
    write_queue_entry(queue_dir, entry)
    return entry


def batch_design(items: Sequence[Tuple[str, str, Optional[dict]]], designer: str, queue_dir: str, work_dir: str,
                 config: AnalysisConfig = AnalysisConfig(), coverage: Coverage = 'complete_programme',
                 bass_management: Optional[dict] = None,
                 on_item_done: Optional[Callable[[str], None]] = None) -> List[str]:
    '''
    Runs Session.extract() then design_and_queue() for each item -- writing
    one QueueEntry (status 'pending') to queue_dir per title. Never calls
    set_filters/publish; that only happens for an entry a human has since
    marked 'accepted' (pipeline.review.apply_reviewed_entry/
    publish_reviewed_queue).
    :param items: (id, source_path, meta) triples. `id` is the queue entry's
        stable id/filename. `meta` is BeqMetadata *constructor* kwargs
        (e.g. {'title': ..., 'year': ...}), or None if title metadata (e.g.
        a TMDB lookup) hasn't been resolved yet -- resolving it at review
        time instead of batch time is deliberately supported.
    :param work_dir: scratch directory for extracted audio, one
        subdirectory per item.
    :param bass_management: this batch's bass-management configuration, if
        any -- see Session.design(); the same one is used for every item.
    :param on_item_done: called with each item's id right after its queue
        entry is written -- a progress hook for a caller that wants one
        (e.g. a GUI driving a progress bar); plain callable, not a Qt
        signal, so this stays Qt-free.
    :return: the ids written, in `items` order.
    '''
    session = Session(config)
    written = []
    for entry_id, source_path, meta in items:
        extracted = session.extract(source_path, os.path.join(work_dir, entry_id))
        design_and_queue(session, entry_id, extracted, designer, queue_dir, meta=meta, coverage=coverage,
                         bass_management=bass_management)
        written.append(entry_id)
        if on_item_done is not None:
            on_item_done(entry_id)
    return written


def _chosen_candidate(entry: QueueEntry) -> CandidateSummary:
    if entry.status != 'accepted':
        raise ValueError(f"entry {entry.id!r} is not accepted (status={entry.status!r})")
    return entry.candidates[entry.chosen_candidate_index]  # QueueEntry.__post_init__ already guarantees this is in range


def apply_reviewed_entry(entry: QueueEntry):
    '''
    entry.status must be 'accepted'. Converts
    entry.candidates[entry.chosen_candidate_index].filters (already a
    realised CompleteFilter.to_json() dict -- see batch_design) back into a
    CompleteFilter via model.codec.filter_from_json -- so nothing
    downstream (set_filters, to_beq_xml, report, publish) can tell a human
    picked this over the top-ranked candidate.
    :return: the chosen candidate's CompleteFilter.
    :raises ValueError: if entry.status != 'accepted'.
    '''
    from model.codec import filter_from_json
    return filter_from_json(_chosen_candidate(entry).filters)


def publish_reviewed_queue(queue_dir: str, xml_repo: RepoTarget, meta_defaults: Optional[dict] = None,
                           images_repo: Optional[RepoTarget] = None, image_owner: Optional[str] = None,
                           image_repo_name: Optional[str] = None, xml_dir: str = '', image_dir: str = '',
                           report_spec: ReportSpec = ReportSpec(),
                           config: AnalysisConfig = AnalysisConfig()) -> List[dict]:
    '''
    Publishes every 'accepted' entry in queue_dir: apply_reviewed_entry()
    for the filters, a fresh report image built from the entry's stored
    curve + chosen filter (Session.report(), when images_repo is given),
    then Session.publish() -- the same image-then-XML sequence
    test_pipeline_acceptance.py exercises for an automatic top-pick, just
    sourced from a human's pick instead. Marks each entry 'published' on
    success. Idempotent -- 'published'/'pending'/'skipped'/'rejected'
    entries are left alone, so re-running after a partial failure only
    retries whatever is still 'accepted'.
    :param meta_defaults: BeqMetadata constructor kwargs used to fill in
        anything entry.meta doesn't supply (e.g. a shared source='Disc');
        entry.meta wins on conflict. If neither supplies `gain`, it
        defaults to the chosen candidate's mv_adjust_db (kept only for this
        catalogue-compatibility purpose -- see designer-interface.md §3).
    :param xml_dir/image_dir: relative directory prefix within each repo;
        each entry publishes to '<xml_dir>/<entry.id>.xml' (and, if
        images_repo is given, '<image_dir>/<entry.id>.png').
    :return: one {'id': entry.id, **Session.publish()'s result} per entry
        actually published this run.
    '''
    from model.codec import xydata_from_json
    from pipeline.metadata import BeqMetadata

    session = Session(config)
    results = []
    for entry in read_queue(queue_dir):
        if entry.status != 'accepted':
            continue
        chosen = _chosen_candidate(entry)
        complete_filter = apply_reviewed_entry(entry)
        meta = BeqMetadata(**{**(meta_defaults or {}), **entry.meta})
        if meta.gain is None:
            meta.gain = f"{chosen.mv_adjust_db:+g}"

        image_png = None
        image_relative_path = None
        if images_repo is not None:
            unfiltered = xydata_from_json(entry.curve)
            filtered = unfiltered.filter(complete_filter.get_transfer_function().get_magnitude())
            image_png = session.report([unfiltered, filtered], complete_filter, meta=meta, spec=report_spec,
                                       mv_offset=chosen.mv_adjust_db)
            image_relative_path = os.path.join(image_dir, f"{entry.id}.png") if image_dir else f"{entry.id}.png"

        xml_relative_path = os.path.join(xml_dir, f"{entry.id}.xml") if xml_dir else f"{entry.id}.xml"
        result = session.publish(complete_filter, meta, xml_repo, xml_relative_path, images_repo=images_repo,
                                 image_relative_path=image_relative_path, image_png=image_png,
                                 image_owner=image_owner, image_repo_name=image_repo_name)
        update_entry(queue_dir, entry.id, status='published')
        results.append({'id': entry.id, **result})
    return results
