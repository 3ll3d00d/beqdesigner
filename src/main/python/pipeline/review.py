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
import logging
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timezone
from typing import Callable, Collection, List, Optional, Sequence, Tuple

from pipeline.config import AnalysisConfig
from pipeline.designer.contract import Coverage
from pipeline.orchestrate import Applied, Declined, DesignOutcome, Session
from pipeline.publish.catalogue import catalogue_paths, publish_digest
from pipeline.publish.git import RepoTarget, fs_path, has_changes, is_committed
from pipeline.publish.report import ReportSpec

logger = logging.getLogger('review_queue')

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
    audio_stream: Optional[int] = None  # zero-based source audio stream; None in queue entries written before this field
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
    source_fingerprint: Optional[str] = None  # the source's own change marker when this was designed (library
                                               # run only); what discovery compares to spot a re-ripped title
    published_digest: Optional[str] = None    # pipeline.publish.catalogue.publish_digest() of what was written to the
                                               # catalogue repos; a different digest now means "out of date"
    published_at: Optional[str] = None        # UTC ISO-8601, when status became 'published'
    revision: int = 0                         # times a title already committed to the catalogue was reopened for
                                               # revision; the same catalogue path is rewritten each time

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
    '''
    Written to a temporary file in the same directory and renamed over the entry, so a reader (the GUI, a concurrent
    `run`) sees the old file or the new one, never a half-written one -- and a crash mid-write cannot lose the entry.
    (The temporary name does not end in `.json`, so read_queue() never mistakes it for an entry.)
    '''
    os.makedirs(queue_dir, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f'.{entry.id}.', suffix='.tmp', dir=queue_dir)
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as f:
            json.dump(asdict(entry), f)
        os.replace(temporary, _entry_path(queue_dir, entry.id))
    except BaseException:
        try:
            os.remove(temporary)
        except OSError:
            pass
        raise


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
                     audio_stream: Optional[int] = None,
                     multichannel_wav_path: Optional[str] = None, channel_layout_name: str = 'unknown',
                     project_dir: Optional[str] = None,
                     on_projects: Optional[Callable[[dict], None]] = None) -> QueueEntry:
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
    :param audio_stream: zero-based source audio stream used for this mono mix, retained for the review chart's
    human-facing legend. None keeps compatibility with callers that cannot know it.
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
    :param on_projects: called with write_title_projects_if_safe()'s result ({'mono': bool, 'multichannel':
        bool|None}; False = an existing human-edited project was left alone) whenever projects were attempted.
    :return: the written QueueEntry.
    '''
    from model.codec import xydata_to_json

    sig = session.load(wav_path, name=entry_id)
    outcome = session.design(sig, designer, coverage=coverage, bass_management=bass_management, channels=channels)
    curve = xydata_to_json(session.curves(sig, kind='avg', filtered=False))
    entry = _outcome_to_entry(entry_id, sig.signal.fs, meta or {}, curve, outcome)
    entry.audio_stream = audio_stream
    write_queue_entry(queue_dir, entry)
    if project_dir is not None and isinstance(outcome, Applied):
        from pipeline.publish.project import write_title_projects_if_safe
        mono_out = os.path.join(project_dir, f"{entry_id}.mono.beq")
        mc_out = os.path.join(project_dir, f"{entry_id}.multichannel.beq") if multichannel_wav_path else None
        written = write_title_projects_if_safe(session, wav_path, outcome.filters, mono_out,
                                               multichannel_wav_path=multichannel_wav_path,
                                               channel_layout_name=channel_layout_name,
                                               multichannel_out_path=mc_out)
        if on_projects is not None:
            on_projects(written)
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


def _project_notes(published, aligned: List[str]) -> dict:
    ''' Result keys saying a human's project edit was what got published; absent when nothing was edited. '''
    notes = {}
    if published.edited_side is not None:
        notes['edited_project'] = published.edited_side
    if aligned:
        notes['projects_aligned'] = aligned
    return notes


_PUBLISH_ERRORS = {
    'project_conflict': 'the mono and multichannel projects were edited independently and now disagree -- '
                        'keep one edit (or re-save one project from the other) and publish again',
    'invalid_metadata': 'the metadata is not complete enough to publish',
    'git_failed': 'git refused',
    'publish_failed': 'publishing failed',
}


def split_publish_results(results: Sequence[dict]) -> Tuple[List[dict], List[dict]]:
    '''
    :return: (published, needs_attention) -- publish_reviewed_queue() reports an entry it refused to publish
        as {'id', 'error'} rather than raising, so a bare len(results) overstates what went out.
    '''
    return [r for r in results if 'error' not in r], [r for r in results if 'error' in r]


def describe_publish_error(result: dict) -> str:
    ''' :return: a one-line, reviewer-facing description of an {'id', 'error'} result. '''
    text = _PUBLISH_ERRORS.get(result['error'], result['error'])
    if result.get('problems'):
        text += ': ' + '; '.join(result['problems'])
    if result.get('message'):
        text += ': ' + result['message']
    return f"{result['id']}: {text}"


class InvalidMetadata(ValueError):
    ''' An entry's metadata cannot even be built into a BeqMetadata; `problems` says why, one line each. '''

    def __init__(self, problems: Sequence[str]):
        super().__init__('; '.join(problems))
        self.problems = list(problems)


def publication_meta(entry: QueueEntry, meta_defaults: Optional[dict] = None):
    '''
    :return: the BeqMetadata an accepted or published entry publishes with: `meta_defaults` under the entry's own
        metadata, and -- if neither says -- `gain` defaulting to the chosen candidate's mv_adjust_db. Before the image
        URLs are filled in, which derive from the repos rather than from the title.
    :raises InvalidMetadata: (a ValueError) if the metadata cannot be built: a field BeqMetadata does not have, or a
        null where a value is needed. A missing or null `title`/`year` is not raised -- it becomes '' so that
        validate() reports it, as the discovery index (status.metadata_problems()) does.
    '''
    from pipeline.metadata import BeqMetadata
    merged = {'title': '', 'year': '', **(meta_defaults or {}), **entry.meta}
    known = {f.name: f for f in fields(BeqMetadata)}
    problems = [f'unknown field {name!r}' for name in merged if name not in known]
    for name in ('title', 'year'):
        if merged.get(name) is None:
            merged[name] = ''
    problems += [f'{name} must not be null' for name, value in merged.items()
                 if value is None and name in known and known[name].default is not None]
    if problems:
        raise InvalidMetadata(problems)
    try:
        meta = BeqMetadata(**merged)
    except (TypeError, AttributeError, ValueError) as error:
        raise InvalidMetadata([f'metadata is not valid: {error}']) from error
    if meta.gain is None:
        meta.gain = f"{entry.candidates[entry.chosen_candidate_index].mv_adjust_db:+g}"
    return meta


def project_paths(work_dir: str, entry_id: str) -> Tuple[str, str, Optional[str], str]:
    ''':return: (project_dir, mono project path, multichannel project path or None, multichannel wav path); the
        multichannel project exists only where a multichannel extraction does.'''
    project_dir = os.path.join(work_dir, entry_id)
    mc_wav = os.path.join(project_dir, 'multichannel.wav')
    return (project_dir, os.path.join(project_dir, f"{entry_id}.mono.beq"),
            os.path.join(project_dir, f"{entry_id}.multichannel.beq") if os.path.isfile(mc_wav) else None, mc_wav)


def current_publish_digest(entry: QueueEntry, *, meta_defaults: Optional[dict] = None,
                           work_dir: Optional[str] = None, has_image: bool = False,
                           report_spec: Optional[ReportSpec] = None, image_owner: Optional[str] = None,
                           image_repo_name: Optional[str] = None) -> str:
    '''
    The digest publish_reviewed_queue() would record for an accepted or published entry *now*, without publishing:
    compare it with `entry.published_digest` to see whether the catalogue's copy is out of date. It writes
    nothing (not the projects, nor the repos); a pipeline-pure project counts as holding the chosen candidate, as
    publishing would rewrite it to, and a hand-edited one as holding its edit.
    :param meta_defaults/work_dir: as publish_reviewed_queue(); the digest depends on them, so pass what publish is
        given.
    :param has_image: whether publish is given an images repo.
    :param report_spec: as publish_reviewed_queue() -- in the digest when it is not the default (see publish_digest()),
        so pass what publish is given.
    :param image_owner/image_repo_name: as publish_reviewed_queue() -- in the digest when given (None: unset).
    :raises ValueError: if the entry has no chosen candidate (it is not accepted or published); InvalidMetadata (one)
        if its metadata cannot be built.
    :raises ProjectFilterConflict: if the mono and multichannel projects were edited independently and disagree.
    '''
    from model.codec import filter_from_json
    from pipeline.publish.project import preview_published_projects
    if entry.status not in ('accepted', 'published') or entry.chosen_candidate_index is None:
        raise ValueError(f"entry {entry.id!r} is not accepted or published (status={entry.status!r})")
    chosen = entry.candidates[entry.chosen_candidate_index]
    complete_filter = filter_from_json(chosen.filters)
    if work_dir is not None:
        _, mono_path, mc_path, _ = project_paths(work_dir, entry.id)
        complete_filter = preview_published_projects(mono_path, mc_path, complete_filter).filter
    return publish_digest(complete_filter.to_json(), publication_meta(entry, meta_defaults), entry.art_path,
                          has_image, chosen.mv_adjust_db, report_spec, image_owner, image_repo_name)


def _needs_republish(entry: QueueEntry, xml_repo: RepoTarget, xml_dir: str, image_dir: str,
                     meta_defaults: Optional[dict], work_dir: Optional[str], has_image: bool,
                     report_spec: Optional[ReportSpec] = None, image_owner: Optional[str] = None,
                     image_repo_name: Optional[str] = None) -> bool:
    '''
    True if a *published* entry's catalogue copy is out of date: its XML is missing from the repo, or the digest of what
    would be published now differs from the one recorded (an entry published before digests were recorded has none,
    and is left alone). A project conflict, or metadata that cannot be built, counts as out of date, so that
    publishing reports it (per entry) rather than the check aborting the batch.
    '''
    from pipeline.publish.project import ProjectFilterConflict
    if not os.path.isfile(fs_path(xml_repo, catalogue_paths(entry.id, xml_dir, image_dir)[0])):
        return True
    if not entry.published_digest:
        return False
    try:
        return current_publish_digest(entry, meta_defaults=meta_defaults, work_dir=work_dir, has_image=has_image,
                                      report_spec=report_spec, image_owner=image_owner,
                                      image_repo_name=image_repo_name) != entry.published_digest
    except (ProjectFilterConflict, InvalidMetadata):
        return True


def publish_reviewed_queue(queue_dir: str, xml_repo: RepoTarget, meta_defaults: Optional[dict] = None,
                           images_repo: Optional[RepoTarget] = None, image_owner: Optional[str] = None,
                           image_repo_name: Optional[str] = None, xml_dir: str = '', image_dir: str = '',
                           report_spec: ReportSpec = ReportSpec(),
                           config: AnalysisConfig = AnalysisConfig(),
                           work_dir: Optional[str] = None, push: bool = True, ids: Optional[Collection[str]] = None,
                           republish: bool = False, on_entry: Optional[Callable[[str], None]] = None,
                           should_cancel: Optional[Callable[[], bool]] = None) -> List[dict]:
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
    :param push: True (the default) commits and pushes each file as it is written, as this always has. False only
        writes the files into the repos' working trees and marks the entry 'published' (meaning *written*), for
        pipeline.library.commit.commit_catalogue() to commit and push the batch -- one commit and one push per repo
        instead of two pushes per title.
    :param ids: only these entries (an id that has no entry is ignored); None is every entry.
    :param republish: also takes each 'published' entry whose catalogue copy is out of date (see _needs_republish():
        a changed metadata field, poster or filter, or a missing XML) and writes it again, **at the same path**, with
        the fresh digest. It stays 'published' and needs no second review, so a typo fixed on a published title
        reaches the catalogue; commit_catalogue() then commits it as a revision. The result carries 'republished': True.
        A republish that rewrites an XML the catalogue holds and that is unchanged in the tree begins a new revision
        (`revision` + 1); over one already rewritten and not yet committed it does not (already counted).
        `xml_dir` is not part of what makes a title out of date: a changed one is a new location, and the file at
        the old one is left behind.
    :param report_spec: what the report image is drawn with; a change of it (from the default) makes the title out of
        date. So does the image's GitHub owner/repo when they are given (`image_owner`/`image_repo_name`).
    :param on_entry: called with an entry's id just before it is published (not for one that is skipped).
    :param should_cancel: checked before each entry; True stops the loop, leaving every entry as it is (each already
        published one is complete).
    :return: one {'id': entry.id, **Session.publish()'s result} per entry
        actually published this run, and one {'id', 'error', ...} per entry that was refused or failed -- one bad
        entry never stops the batch or loses the results before it: 'invalid_metadata' (with 'problems': also a
        field BeqMetadata does not have, or a null), 'project_conflict', 'git_failed' (git refused; 'message' has
        what it said) and 'publish_failed' (anything else, e.g. a poster file that is gone; 'message'). A failed
        entry keeps its status, so a rerun retries it. With work_dir, an entry published from a human's project edit also carries
        'edited_project' ('mono'/'multichannel'/'both') and, if the other project was rewritten to match,
        'projects_aligned' (the names rewritten).
    '''
    from model.codec import filter_from_json, xydata_from_json
    from pipeline.metadata import validate
    from pipeline.publish.project import ProjectFilterConflict, align_projects, resolve_published_projects, \
        write_title_projects_if_safe

    session = Session(config)
    results = []
    if ids is None:
        entries = read_queue(queue_dir)
    else:
        entries = [read_entry(queue_dir, i) for i in dict.fromkeys(ids) if os.path.isfile(_entry_path(queue_dir, i))]

    def metadata_result(entry: QueueEntry, problems: Sequence[str]) -> dict:
        return {'id': entry.id, 'error': 'invalid_metadata', 'problems': list(problems)}

    def publish_one(entry: QueueEntry, republished: bool) -> dict:
        chosen = entry.candidates[entry.chosen_candidate_index]  # accepted and published entries always have one
        complete_filter = filter_from_json(chosen.filters)  # still drives meta.gain's default below
        try:
            meta = publication_meta(entry, meta_defaults)
            problems = validate(meta)
        except InvalidMetadata as error:
            return metadata_result(entry, error.problems)
        except (TypeError, AttributeError, ValueError) as error:  # a value of the wrong kind, e.g. a number for audio_types
            return metadata_result(entry, [f'metadata is not valid: {error}'])
        if problems:  # one incomplete title must not stop the batch, any more than a project conflict does
            return metadata_result(entry, problems)

        if work_dir is not None:
            project_dir, mono_path, mc_path, mc_wav = project_paths(work_dir, entry.id)
            layout = _read_channel_layout_name(project_dir)
            write_title_projects_if_safe(session, os.path.join(project_dir, 'mono.wav'), complete_filter, mono_path,
                                         multichannel_wav_path=mc_wav if mc_path else None,
                                         channel_layout_name=layout, multichannel_out_path=mc_path)
            try:
                published = resolve_published_projects(mono_path, mc_path)
            except ProjectFilterConflict:
                return {'id': entry.id, 'error': 'project_conflict'}
            complete_filter = published.filter
            aligned = align_projects(session, published, mono_path, os.path.join(project_dir, 'mono.wav'),
                                     mc_path, mc_wav if mc_path else None, layout)

        xml_relative_path, image_relative_path = catalogue_paths(entry.id, xml_dir, image_dir)
        image_png = None
        if images_repo is not None:
            unfiltered = xydata_from_json(entry.curve)
            filtered = unfiltered.filter(complete_filter.get_transfer_function().get_magnitude())
            image_png = session.report([unfiltered, filtered], complete_filter, meta=meta, poster_path=entry.art_path,
                                       spec=report_spec, mv_offset=chosen.mv_adjust_db)
        digest = publish_digest(complete_filter.to_json(), meta, entry.art_path, images_repo is not None,
                                chosen.mv_adjust_db, report_spec, image_owner,
                                image_repo_name)  # before publish(), which fills the image URLs into meta
        # A republish over an XML the catalogue holds, and that has not been touched since, begins a revision: the
        # same thing a reopen of it counts (see pipeline.library.revise). Over one already rewritten and not
        # committed it does not, since that revision has been counted -- by the reopen, or by the republish, that
        # wrote it. A republish is a rewrite of a *published* entry, so an accepted one (a first publish, or one
        # after a reopen, which counted) never counts here.
        revision = entry.revision
        if republished and is_committed(xml_repo, xml_relative_path) and not has_changes(xml_repo, xml_relative_path):
            revision += 1
        result = session.publish(complete_filter, meta, xml_repo, xml_relative_path, images_repo=images_repo,
                                 image_relative_path=image_relative_path if images_repo is not None else None,
                                 image_png=image_png, image_owner=image_owner, image_repo_name=image_repo_name, push=push)
        update_entry(queue_dir, entry.id, status='published', published_digest=digest, revision=revision,
                     published_at=datetime.now(timezone.utc).isoformat(timespec='seconds'))
        return {'id': entry.id, **result, **({'republished': True} if republished else {}),
                **(_project_notes(published, aligned) if work_dir else {})}

    def failure(entry: QueueEntry, error: Exception) -> dict:
        ''' What one entry's unexpected failure looks like: it is reported and the rest of the batch goes on. '''
        if isinstance(error, InvalidMetadata):
            return metadata_result(entry, error.problems)
        if isinstance(error, subprocess.CalledProcessError):
            return {'id': entry.id, 'error': 'git_failed', 'message': str(error)}
        return {'id': entry.id, 'error': 'publish_failed', 'message': f'{type(error).__name__}: {error}'}

    for entry in entries:
        republished = False
        if entry.status == 'published' and republish:
            try:
                republished = _needs_republish(entry, xml_repo, xml_dir, image_dir, meta_defaults, work_dir,
                                               images_repo is not None, report_spec, image_owner, image_repo_name)
            except Exception as error:
                results.append(failure(entry, error))
                continue
        if entry.status != 'accepted' and not republished:
            continue
        if should_cancel is not None and should_cancel():
            break
        if on_entry is not None:
            on_entry(entry.id)
        try:
            results.append(publish_one(entry, republished))
        except Exception as error:  # a poster that is gone, a project that will not write, git refusing: this entry only
            logger.warning('could not publish %s: %s', entry.id, error, exc_info=True)
            results.append(failure(entry, error))
    return results
