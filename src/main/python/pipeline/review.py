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
    art_path: Optional[str] = None   # local image file used as this entry's report poster, if any
    art_overridden: bool = False     # True once a human has explicitly set/cleared art_path; future
                                     # auto-resolution must never overwrite it
    design_fingerprint: Optional[str] = None  # library-run inputs that produced this entry; absent on
                                               # pre-library entries, which must be redesigned once to gain it

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
                     bass_management: Optional[dict] = None, channels: Optional[dict] = None,
                     multichannel_wav_path: Optional[str] = None, channel_layout_name: str = 'unknown',
                     project_dir: Optional[str] = None) -> QueueEntry:
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
        Also the mono `.beq` project's source, when project_dir is given.
    :param meta: BeqMetadata *constructor* kwargs, or None if unresolved yet.
    :param channels: DesignRequest.channels -- see Session.design()/Session.load_channels(). Optional.
    :param multichannel_wav_path: the kept extraction, when it's multichannel (design/library-sync-
        pipeline-plan.md §3.3) -- if given (together with project_dir), a linked multichannel `.beq`
        project is written alongside the mono one.
    :param channel_layout_name: the source's ffmpeg channel layout name, forwarded to
        Session.load_channel_signals() for the multichannel project's channel labels.
    :param project_dir: if given and the outcome was Applied, writes output 1's `.beq` project file(s)
        (design/library-sync-pipeline-plan.md §3.3/Appendix B) -- `<project_dir>/<entry_id>.mono.beq`
        always, plus `<project_dir>/<entry_id>.multichannel.beq` when multichannel_wav_path is also given.
        A Declined outcome has no filter to write, so nothing is written for it (matches "candidates empty
        on decline"). Omitted (the default), no project files are written -- backward compatible.
    :return: the written QueueEntry.
    '''
    from model.codec import xydata_to_json

    sig = session.load(wav_path, name=entry_id)
    outcome = session.design(sig, designer, coverage=coverage, bass_management=bass_management, channels=channels)
    curve = xydata_to_json(session.curves(sig, kind='avg', filtered=False))
    entry = _outcome_to_entry(entry_id, sig.signal.fs, meta or {}, curve, outcome)
    write_queue_entry(queue_dir, entry)
    if project_dir is not None and isinstance(outcome, Applied):
        from pipeline.publish.project import write_title_projects_if_safe
        mono_out = os.path.join(project_dir, f"{entry_id}.mono.beq")
        mc_out = os.path.join(project_dir, f"{entry_id}.multichannel.beq") if multichannel_wav_path else None
        write_title_projects_if_safe(session, wav_path, outcome.filters, mono_out,
                                     multichannel_wav_path=multichannel_wav_path,
                                     channel_layout_name=channel_layout_name, multichannel_out_path=mc_out)
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


def _read_channel_layout_name(project_dir: str) -> str:
    '''
    Reads manifest.json's channel_layout_name key, if the file and key exist -- the extract cache's
    fingerprint record (design/library-sync-pipeline-plan.md §4.1), not yet built (chunk 5) as of this
    chunk. get_channel_name()'s own fallback already handles an unknown layout sanely by channel count,
    so 'unknown' is a safe default here.
    '''
    manifest_path = os.path.join(project_dir, 'manifest.json')
    if os.path.isfile(manifest_path):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        return manifest.get('channel_layout_name', 'unknown')
    return 'unknown'


def publish_reviewed_queue(queue_dir: str, xml_repo: RepoTarget, meta_defaults: Optional[dict] = None,
                           images_repo: Optional[RepoTarget] = None, image_owner: Optional[str] = None,
                           image_repo_name: Optional[str] = None, xml_dir: str = '', image_dir: str = '',
                           report_spec: ReportSpec = ReportSpec(),
                           config: AnalysisConfig = AnalysisConfig(),
                           work_dir: Optional[str] = None) -> List[dict]:
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
    :param work_dir: if given, the published filter is read from the entry's `.beq` project file(s) under
        `<work_dir>/<entry.id>/` (design/library-sync-pipeline-plan.md §3.3.1/Appendix B) rather than from
        apply_reviewed_entry()'s raw chosen candidate -- a human who opened the mono/multichannel project
        and edited the filter directly has that edit published instead. The project(s) are regenerated
        (hash-gated -- an existing human edit is never clobbered) from the just-computed candidate filter
        first, so a never-opened project is created/kept current before being read back. An entry whose
        mono and multichannel projects were independently edited to disagree is *not* published -- its
        result carries an 'error': 'project_conflict' key instead, and its status is left alone so a rerun
        retries it. Omitted (the default), publishing reads apply_reviewed_entry() as before -- backward
        compatible.
    :return: one {'id': entry.id, **Session.publish()'s result} per entry
        actually published this run.
    '''
    from model.codec import xydata_from_json
    from pipeline.metadata import BeqMetadata
    from pipeline.publish.project import ProjectFilterConflict, resolve_published_filter, write_title_projects_if_safe

    session = Session(config)
    results = []
    for entry in read_queue(queue_dir):
        if entry.status != 'accepted':
            continue
        chosen = _chosen_candidate(entry)
        complete_filter = apply_reviewed_entry(entry)  # unchanged -- still drives meta.gain's default below
        meta = BeqMetadata(**{**(meta_defaults or {}), **entry.meta})
        if meta.gain is None:
            meta.gain = f"{chosen.mv_adjust_db:+g}"

        if work_dir is not None:
            project_dir = os.path.join(work_dir, entry.id)
            mono_path = os.path.join(project_dir, f"{entry.id}.mono.beq")
            mc_wav = os.path.join(project_dir, 'multichannel.wav')
            mc_path = os.path.join(project_dir, f"{entry.id}.multichannel.beq") if os.path.isfile(mc_wav) else None
            layout = _read_channel_layout_name(project_dir)
            write_title_projects_if_safe(session, os.path.join(project_dir, 'mono.wav'), complete_filter, mono_path,
                                         multichannel_wav_path=mc_wav if mc_path else None,
                                         channel_layout_name=layout, multichannel_out_path=mc_path)
            try:
                complete_filter, _ = resolve_published_filter(mono_path, mc_path)
            except ProjectFilterConflict:
                results.append({'id': entry.id, 'error': 'project_conflict'})
                continue

        image_png = None
        image_relative_path = None
        if images_repo is not None:
            unfiltered = xydata_from_json(entry.curve)
            filtered = unfiltered.filter(complete_filter.get_transfer_function().get_magnitude())
            image_png = session.report([unfiltered, filtered], complete_filter, meta=meta, poster_path=entry.art_path,
                                       spec=report_spec, mv_offset=chosen.mv_adjust_db)
            image_relative_path = os.path.join(image_dir, f"{entry.id}.png") if image_dir else f"{entry.id}.png"

        xml_relative_path = os.path.join(xml_dir, f"{entry.id}.xml") if xml_dir else f"{entry.id}.xml"
        result = session.publish(complete_filter, meta, xml_repo, xml_relative_path, images_repo=images_repo,
                                 image_relative_path=image_relative_path, image_png=image_png,
                                 image_owner=image_owner, image_repo_name=image_repo_name)
        update_entry(queue_dir, entry.id, status='published')
        results.append({'id': entry.id, **result})
    return results
