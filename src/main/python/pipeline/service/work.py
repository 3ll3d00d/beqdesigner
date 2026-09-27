'''
What each kind of job does (design/pipeline-service.md §2): the same calls the command line makes for `scan`, `run` with a
selector, and `accept`, on a profile read again for the job (`pipeline.service.context`). The service adds no pipeline
logic of its own.
'''
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from pipeline.library.bulk import accept_top_pick, plan_accept
from pipeline.library.inbox import WorkDirInbox
from pipeline.library.index import LibraryIndex, ScanResult, index_path
from pipeline.library.handoff import STOPPED, WITHDRAWN, hand_off, outcomes as title_outcomes
from pipeline.library.join import JOINABLE, JoinQueue, JoinRequest
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
    return False


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
    with LibraryIndex(index_path(_require_work_dir(context))) as index:
        return index.scan(context.profile, settings, only=list(request.sources) or None, allow_empty=request.allow_empty)


def run(context: JobContext, request: RunRequest, control: JobControl) -> RunOutcome:
    run_config = context.run_config()
    join = JoinQueue(sources=[WorkDirInbox(run_config.work_dir).claim])   # other runs' work, and run jobs submitted now
    control.accept_joins(join)
    settings, publish = context.stage_settings(request.through)
    with LibraryIndex(index_path(run_config.work_dir)) as index:
        scanned = None
        if request.scan_first or not index.generation:   # never scanned: there is nothing to select from
            scanned = index.scan(context.profile, settings)
        report = run_stages(context.profile, request.selection, request.through, run_config=run_config, index=index,
                            publish=publish, settings=settings, retry_failed=request.retry_failed,
                            unattended=request.unattended, join=join, should_cancel=control.cancelled, on_progress=control.progress, on_event=control.event)
    return RunOutcome(scanned, report)


def accept(context: JobContext, request: AcceptRequest):
    settings = context.scan_settings()
    if not settings.work_dir or not settings.queue_dir:
        raise ValueError('the profile names no work or queue directory')
    path = index_path(settings.work_dir)
    if not os.path.isfile(path):
        raise ValueError(f'no index at {path}: scan first')
    options = dict(queue_dir=settings.queue_dir, meta_defaults=settings.meta_defaults, work_dir=settings.work_dir)
    with LibraryIndex(path) as index:
        if request.dry_run:
            return plan_accept(index, request.selection, request.threshold, **options)
        report = accept_top_pick(index, request.selection, request.threshold, **options)
        index.refresh(context.profile, settings)
        return report


def executor(profile_path: str, env: Optional[Mapping[str, str]] = None,
             load: Callable[..., JobContext] = load_context, *, poll_seconds: float = 2.0,
             sleep: Callable[[float], None] = time.sleep) -> Callable[[Job, JobControl], Any]:
    ''' The JobManager's `execute`: reads the profile for each job, then does it holding the work directory's lease. '''
    def execute(job: Job, control: JobControl):
        context = load(profile_path, env)
        work_dir = _require_work_dir(context)
        while True:
            try:
                lease = WorkDirLease(work_dir, job.id).__enter__()   # one run at a time in a work directory
            except LeaseHeld:
                # the work list or a command-line run has it (design/worklist-feedback.md F5): extract and design go to
                # that run; anything else waits for it to end
                holder = read_lease(work_dir)
                if holder is not None and isinstance(job.request, RunRequest) and job.request.through in JOINABLE:
                    outcome = handed_off(context, job.request, control, holder)
                    if outcome is not WAIT_AGAIN:
                        return outcome
                elif control.cancelled():
                    return None
                else:
                    sleep(poll_seconds)
                continue
            try:
                if isinstance(job.request, ScanRequest):
                    return scan(context, job.request)
                if isinstance(job.request, RunRequest):
                    return run(context, job.request, control)
                if isinstance(job.request, AcceptRequest):
                    return accept(context, job.request)
                raise TypeError(f'not a job request: {job.request!r}')
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
    with LibraryIndex(index_path(work_dir)) as index:
        plan = plan_stages(request.selection.rows(index), request.through, retry_failed=request.retry_failed,
                           unattended=request.unattended)
    ids = [p.row.id for p in plan.planned]
    report = StagesReport(request.through, len(plan.planned) + len(plan.skipped), skipped=list(plan.skipped))
    if not ids:
        return RunOutcome(None, report)
    control.event(ExecutionEvent('', '', '', 'handed_off', time.time(),
                                 f'{holder.who()} is running: handed it {len(ids)} titles'))
    ended = hand_off(work_dir, JoinRequest(Selection(ids=tuple(ids)), request.through, request.retry_failed),
                     should_stop=control.cancelled)
    if ended == STOPPED:
        return None
    if ended == WITHDRAWN:
        return WAIT_AGAIN
    with LibraryIndex(index_path(work_dir)) as index:
        for title_id, outcome in title_outcomes(index, ids, request.through).items():
            if outcome.failed:
                report.run.failed.append((title_id, outcome.detail or f'still needs {outcome.needs}'))
            report.attempted.append(title_id)
        report.counts = index.summary().counts
    return RunOutcome(None, report)
