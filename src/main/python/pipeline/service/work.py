'''
What each kind of job does (design/pipeline-service.md §2): the same calls the command line makes for `scan`, `run` with a
selector, and `accept`, on a profile read again for the job (`pipeline.service.context`). The service adds no pipeline
logic of its own.
'''
import os
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, List, Mapping, Optional

from pipeline.designer.http_binding import DesignerUnavailable
from pipeline.library.bulk import accept_top_pick, plan_accept
from pipeline.library.inbox import WorkDirInbox
from pipeline.library.index import LibraryIndex, ScanResult
from pipeline.library.handoff import STOPPED, WITHDRAWN, hand_off, outcomes as title_outcomes
from pipeline.library.join import JOINABLE, JoinQueue, JoinRequest
from pipeline.library.revise import choose_stream, revise_entry
from pipeline.library.streams import probe_stream_details
from pipeline.publish.git import RepoTarget
from pipeline.review import read_entry
from pipeline.library.selection import Selection, plan_stages
from model.execution_events import ExecutionEvent
from pipeline.library.stages import StagesReport, run_stages
from pipeline.service.context import JobContext, load_context
from pipeline.service.jobs import AcceptRequest, Job, JobControl, RunRequest, ScanRequest
from pipeline.service.lease import LeaseHeld, WorkDirLease, read_lease


@dataclass(frozen=True)
class RunOutcome:
    ''' A run job's result: the scan it took first (if it took one) and what `run_stages` reports. '''
    scan: Optional[ScanResult]
    report: StagesReport

    @property
    def cancelled(self) -> bool:
        return self.report.cancelled


def job_failed(result: Any) -> bool:
    ''' Whether a finished job's result means it failed: a title failed, a publish was refused, git refused, a source was down. '''
    if isinstance(result, RunOutcome):
        return result.report.failed or bool(result.scan and result.scan.errors)
    if isinstance(result, ScanResult):
        return bool(result.errors)
    if isinstance(result, dict) and isinstance(result.get('report'), dict):   # a paused run's parts, merged
        report, run = result['report'], result['report'].get('run') or {}
        return bool(run.get('failed') or run.get('unavailable') or report.get('stopped') or report.get('publish_errors')
                    or report.get('commit_error') or (result.get('scan') or {}).get('errors'))
    return False


def remaining_titles(result: Any) -> Optional[tuple]:
    '''
    The titles a run part stopped before, so a paused run carries on with them (design/job-windows.md §3): its `not_run`
    when it was stopped like a cancel; () when there is nothing to carry on with (it finished, or stopped for another
    reason); None when it stopped before it planned anything (waiting for, or handed to, another process's run).
    '''
    if result is None:
        return None
    if not isinstance(result, RunOutcome):
        return ()
    report = result.report
    return tuple(report.not_run) if report.cancelled and not report.stopped else ()


# a merged report's fields that say how the run ended, or what the index holds after it: the last part's
_LAST = ('cancelled', 'stopped', 'not_run', 'counts', 'halt')
_FIRST = ('through', 'selected')   # what was asked: the first part's (a resumed part selects only what was left)


def _merge(values: List[Any], key: str = '') -> Any:
    if key in _LAST:
        return values[-1]
    if key in _FIRST:
        return values[0]
    if all(isinstance(v, dict) for v in values):
        keys = list(dict.fromkeys(k for v in values for k in v))
        return {k: _merge([v[k] for v in values if k in v], k) for k in keys}
    if all(isinstance(v, list) for v in values):
        joined = [item for v in values for item in v]
        return list(dict.fromkeys(joined)) if key == 'attempted' else joined
    present = [v for v in values if v not in (None, '')]
    return present[-1] if present else values[-1]


def merge_outcomes(parts: List[Any]) -> Any:
    ''' One run job's result from its parts' (plain data, oldest first): lists joined, how it ended from the last. '''
    parts = [p for p in parts if p is not None]
    reports = [p['report'] for p in parts if isinstance(p, dict) and isinstance(p.get('report'), dict)]
    if not reports:
        return parts[-1] if parts else None
    scans = [p.get('scan') for p in parts if isinstance(p, dict) and p.get('scan')]
    seasons = {}
    for report in reports:
        seasons.update((report.get('run') or {}).get('seasons') or {})
    merged = _merge(reports)
    if 'run' in merged:
        merged['run']['seasons'] = seasons
    return {'scan': scans[0] if scans else None, 'report': merged}


def _require_work_dir(context: JobContext) -> str:
    if not context.work_dir:
        raise ValueError('the profile names no work directory (run.work_dir)')
    return context.work_dir


def scan(context: JobContext, request: ScanRequest) -> ScanResult:
    settings = context.scan_settings()
    unknown = set(request.sources) - {spec.name for spec in context.profile.sources}
    if unknown:
        raise ValueError(f"no such source: {', '.join(sorted(unknown))} "
                         f"(the profile has: {', '.join(spec.name for spec in context.profile.sources)})")
    _require_work_dir(context)
    with LibraryIndex(context.index_file) as index:
        return index.scan(context.profile, settings, only=list(request.sources) or None, allow_empty=request.allow_empty)


def with_current_identity(context: JobContext, selection: Selection) -> Selection:
    '''
    The selection, with what a redesign now would be made with when it selects on being out of date: asked of the
    designers as the job starts, not when it was submitted.
    :raises ValueError: if it does and no designer that is up says its design version.
    '''
    if selection.out_of_date is None:
        return selection
    current = context.designer_identity()
    if current is None:
        raise ValueError('no designer that is up says its design version, so no title can be judged out of date')
    return replace(selection, current=current)


def run(context: JobContext, request: RunRequest, control: JobControl, *, allow_repository_writes: bool = False) -> RunOutcome:
    run_config = context.run_config()
    if request.through == 'design':
        # every design would fail: refuse before extracting anything. A run through publish or commit is not refused
        # (it may have only accepted titles to publish); what it designs is reported unavailable instead (R1)
        reason = context.designer_unavailable()
        if reason:
            raise DesignerUnavailable(f'the designer is unavailable, so nothing was run: {reason}')
    join = JoinQueue(sources=[WorkDirInbox(run_config.work_dir).claim])   # other runs' work, and run jobs submitted now
    control.accept_joins(join)
    settings, publish = context.stage_settings(request.through)
    selection = with_current_identity(context, request.selection)
    with LibraryIndex(context.index_file) as index:
        scanned = None
        if request.scan_first or not index.generation:   # never scanned: there is nothing to select from
            scanned = index.scan(context.profile, settings)
        if request.redesign:
            if selection.out_of_date:
                # every designed title out of date that a person has not accepted or published (those stay as they are)
                title_ids = tuple(r.id for r in selection.rows(index)
                                  if r.review_state != 'accepted' and r.design_state != 'protected')
                selection = Selection(ids=title_ids)
                if not title_ids:
                    return RunOutcome(scanned, StagesReport('design', 0))
            title_ids = selection.ids
            # every title is checked before any is sent back: a bad id fails the job with nothing changed
            unknown = [t for t in title_ids if index.title(t) is None]
            if unknown:
                raise ValueError(f'no such title: {", ".join(unknown)}')
            undesigned, published = [], []
            for title_id in title_ids:
                try:
                    entry = read_entry(settings.queue_dir, title_id)
                except FileNotFoundError:
                    undesigned.append(title_id)
                    continue
                if entry.status == 'published':
                    published.append(title_id)
            if undesigned and request.audio_stream is None:   # a stream can be chosen before the first design
                raise ValueError(f'not designed yet, so nothing to redesign: {", ".join(undesigned)}')
            if published and not allow_repository_writes:
                raise ValueError('redesigning a published title requires allow_repository_writes in service config: '
                                 + ', '.join(published))
            for title_id in title_ids:
                # only a cancel stops the sending back, not a window: a paused redesign carries on as a run of what it has
                # not designed (design/job-windows.md §3), which needs every title sent back
                if control.cancelled():   # those sent back so far are designed by the next run that takes them
                    index.refresh(context.profile, settings)
                    return RunOutcome(scanned, StagesReport('design', 0, cancelled=True))
                where = dict(xml_repo=RepoTarget(settings.xml_repo) if settings.xml_repo else None,
                             images_repo=RepoTarget(settings.images_repo) if settings.images_repo else None,
                             xml_dir=settings.xml_dir, image_dir=settings.image_dir,
                             category_folders=settings.category_folders, meta_defaults=settings.meta_defaults)
                if request.audio_stream is not None:
                    unit = index.units([title_id]).get(title_id)
                    if unit is not None and hasattr(unit, 'audio_stream_details') and not unit.audio_stream_details:
                        # the source listed none: read from the file, kept, as the desktop's title page does (W2)
                        index.set_audio_stream_details(title_id, probe_stream_details(unit))
                    choose_stream(index, settings.queue_dir, title_id, request.audio_stream, work_dir=settings.work_dir,
                                  **where)
                else:
                    revise_entry(settings.queue_dir, title_id, request.revise_to, work_dir=settings.work_dir, **where)
            index.refresh(context.profile, settings)
        report = run_stages(context.profile, selection, request.through, run_config=run_config, index=index,
                            publish=publish, settings=settings, retry_failed=request.retry_failed,
                            unattended=request.unattended, join=join, should_cancel=control.should_stop,
                            on_progress=control.progress, on_event=control.event)
    return RunOutcome(scanned, report)


def accept(context: JobContext, request: AcceptRequest):
    settings = context.scan_settings()
    if not settings.work_dir or not settings.queue_dir:
        raise ValueError('the profile names no work or queue directory')
    path = context.index_file
    if not os.path.isfile(path):
        raise ValueError(f'no index at {path}: scan first')
    options = dict(queue_dir=settings.queue_dir, meta_defaults=settings.meta_defaults, work_dir=settings.work_dir)
    selection = with_current_identity(context, request.selection)
    with LibraryIndex(path) as index:
        if request.dry_run:
            return plan_accept(index, selection, request.threshold, **options)
        report = accept_top_pick(index, selection, request.threshold, **options)
        index.refresh(context.profile, settings)
        return report


def executor(profile_path: str, env: Optional[Mapping[str, str]] = None,
             load: Callable[..., JobContext] = load_context, *, poll_seconds: float = 2.0,
             sleep: Callable[[float], None] = time.sleep,
             allow_repository_writes: bool = False) -> Callable[[Job, JobControl], Any]:
    ''' The JobManager's `execute`: reads the profile for each job, then does it holding the work directory's lease. '''
    def execute(job: Job, control: JobControl):
        request = job.current_request   # a paused run carries on with the titles it had left
        context = load(profile_path, env)
        work_dir = _require_work_dir(context)
        while True:
            try:
                lease = WorkDirLease(work_dir, job.id).__enter__()   # one run at a time in a work directory
            except LeaseHeld:
                # the work list or a command-line run has it (design/archive/library-sync/worklist-feedback.md F5):
                # extract and design go to that run; anything else waits for it to end
                holder = read_lease(work_dir)
                if holder is not None and isinstance(request, RunRequest) and request.through in JOINABLE and not request.redesign:
                    outcome = handed_off(context, request, control, holder)
                    if outcome is not WAIT_AGAIN:
                        return outcome
                elif control.should_stop():   # cancelled, or its window closed while it waited: it pauses
                    return None
                else:
                    sleep(poll_seconds)
                continue
            try:
                if isinstance(request, ScanRequest):
                    return scan(context, request)
                if isinstance(request, RunRequest):
                    return run(context, request, control, allow_repository_writes=allow_repository_writes)
                if isinstance(request, AcceptRequest):
                    return accept(context, request)
                raise TypeError(f'not a job request: {request!r}')
            finally:
                lease.__exit__(None, None, None)
    return execute


WAIT_AGAIN = object()   # handed_off(): the run in progress ended without taking the titles, so take the lease and run them


def handed_off(context: JobContext, request: RunRequest, control: JobControl, holder):
    '''
    A run job while another process's run holds the work directory: its titles go to that run, and the job ends when it
    does, with a report of what became of them read from the index. :return: a RunOutcome, None if the job was cancelled
    while it waited, or WAIT_AGAIN.
    '''
    work_dir = _require_work_dir(context)
    with LibraryIndex(context.index_file) as index:
        plan = plan_stages(request.selection.rows(index), request.through, retry_failed=request.retry_failed,
                           unattended=request.unattended)
    ids = [p.row.id for p in plan.planned]
    report = StagesReport(request.through, len(plan.planned) + len(plan.skipped), skipped=list(plan.skipped))
    if not ids:
        return RunOutcome(None, report)
    control.event(ExecutionEvent('', '', '', 'handed_off', time.time(),
                                 f'{holder.who()} is running: handed it {len(ids)} titles'))
    ended = hand_off(work_dir, JoinRequest(Selection(ids=tuple(ids)), request.through, request.retry_failed),
                     should_stop=control.should_stop)
    if ended == STOPPED:
        return None
    if ended == WITHDRAWN:
        return WAIT_AGAIN
    with LibraryIndex(context.index_file) as index:
        for title_id, outcome in title_outcomes(index, ids, request.through).items():
            if outcome.failed:
                report.run.failed.append((title_id, outcome.detail or f'still needs {outcome.needs}'))
            report.attempted.append(title_id)
        report.counts = index.summary().counts
    return RunOutcome(None, report)
