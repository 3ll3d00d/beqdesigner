'''
Doing the work for a selection of titles -- design/library-sync/workflow-rework/design.md §12.7.

    with LibraryIndex(index_path(work_dir)) as index:
        index.scan(profile, settings)
        report = run_stages(profile, Selection(needs=('extract', 'design')), 'design', run_config=..., index=index,
                            should_cancel=lambda: stop.is_set(), on_progress=show)

`through` is "every stage up to and including this one that the title still needs": design extracts first when the
title has not been; extract and design are machine work; review is a person's, so a title never goes past design to
publish on its own -- only a title a person has *accepted* (or a published one whose catalogue copy is out of date)
is published, and only a published one committed. `plan_stages()` (selection.py) says what that means per title.

The titles come from the discovery index, which also remembers what each one was listed as, so nothing here lists a
source again. A title that failed before against the same source and settings is not tried again unless
`retry_failed`. Cancel is cooperative and checked **between titles** (and between entries while publishing): the title
in hand finishes, so a cancelled run leaves only whole titles done, and its report says what was and was not.
'''
import logging
import subprocess
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextvars import copy_context
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from pipeline.library.commit import CatalogueCommit
from pipeline.library.index import LibraryIndex
from pipeline.library.profile import Profile
from pipeline.library.run import LibraryRunConfig, LibraryRunReport, design_unit_work, run_unit
from pipeline.library.selection import Selection, Skipped, plan_stages
from pipeline.library.status import ScanSettings
from pipeline.library.season import conflicting_units
from pipeline.library.sync import commit_library, publish_library
from pipeline.orchestrate import Session
from pipeline.publish.git import RepoTarget
from pipeline.publish.report import ReportSpec
from pipeline.review import split_publish_results
from model.execution_events import emit_execution_event, event_scope, execution_event_context

logger = logging.getLogger('library_stages')


@dataclass(frozen=True)
class Progress:
    '''
    Where a run is. `done` and `total` count title-stages of work -- a title that is extracted and designed counts once,
    one that is published counts once and one that is committed counts once -- so the bar is determinate from the
    start. `title` and `stage` name what is starting; the final report has an empty stage and `done == total` unless the run
    was cancelled, when `done` is how far it got.
    '''
    done: int
    total: int
    title: str
    stage: str      # 'extract' | 'design' | 'publish' | 'commit' | ''
    id: str = ''


@dataclass(frozen=True)
class FfmpegProgress:
    '''One of ffmpeg's real-time extraction updates, in microseconds, for the title currently being extracted.'''
    title: str
    id: str
    out_time_micros: int
    total_micros: int


@dataclass(frozen=True)
class PublishSettings:
    ''' What `publish` and `commit` are given (a scan's ScanSettings must agree with these, see ScanSettings). '''
    xml_repo: RepoTarget
    images_repo: Optional[RepoTarget] = None
    image_owner: Optional[str] = None
    image_repo_name: Optional[str] = None
    xml_dir: str = ''
    image_dir: str = ''
    meta_defaults: Optional[dict] = None
    report_spec: ReportSpec = ReportSpec()
    push: bool = True    # False commits locally only

    @classmethod
    def from_scan_settings(cls, settings: ScanSettings, *, image_owner: Optional[str] = None,
                           image_repo_name: Optional[str] = None, push: bool = True) -> 'PublishSettings':
        '''
        :raises ValueError: if settings names no XML repository.
        '''
        if not settings.xml_repo:
            raise ValueError('xml-repo is required to publish or commit')
        return cls(RepoTarget(settings.xml_repo), RepoTarget(settings.images_repo) if settings.images_repo else None,
                   image_owner or settings.image_owner or None, image_repo_name or settings.image_repo_name or None,
                   settings.xml_dir, settings.image_dir, settings.meta_defaults,
                   report_spec=settings.report_spec or ReportSpec(), push=push)


@dataclass
class StagesReport:
    through: str
    selected: int                                   # titles the selection matched
    run: LibraryRunReport = field(default_factory=LibraryRunReport)   # extract and design, as `run` reports them
    published: List[dict] = field(default_factory=list)         # publish results; a republish carries republished
    publish_errors: List[dict] = field(default_factory=list)    # {'id', 'error'[, 'problems']}: refused, not stopped
    committed: Optional[CatalogueCommit] = None
    commit_error: str = ''                          # git refused (a rejected push): what was committed stays so
    skipped: List[Skipped] = field(default_factory=list)   # not worked on, and why
    cancelled: bool = False
    attempted: List[str] = field(default_factory=list)     # titles whose planned work was started (and finished)
    not_run: List[str] = field(default_factory=list)       # planned but not started, because of the cancel
    counts: Dict[str, int] = field(default_factory=dict)   # titles per needs, after the run

    @property
    def failed(self) -> bool:
        return bool(self.run.failed or self.publish_errors or self.commit_error)


def _title(row) -> str:
    return row.title or row.display_name or row.id


def _units_by_title(index: LibraryIndex, ids: List[str]):
    '''
    :return: ({id: unit}, {id: why it could not be rebuilt}). One query for the lot, and only if that fails
        (a malformed `items` column, a season row that does not plan to one group) one per title, so a single bad
        row fails that title and not the run.
    '''
    try:
        return index.units(ids), {}
    except Exception:
        units, errors = {}, {}
        for title_id in ids:
            try:
                units.update(index.units([title_id]))
            except Exception as error:
                errors[title_id] = f'{type(error).__name__}: {error}'
        return units, errors


def _merge_run_report(target: LibraryRunReport, source: LibraryRunReport, *, include_extract: bool) -> None:
    """Merge one worker's isolated result into the coordinator-owned run report."""
    fields = ('extracted', 'cached', 'seasons') if include_extract else ()
    fields += ('designed', 'design_cached', 'failed', 'failed_earlier', 'meta_unresolved', 'project_edit_preserved')
    for name in fields:
        current, incoming = getattr(target, name), getattr(source, name)
        if isinstance(current, dict):
            current.update(incoming)
        else:
            current.extend(incoming)


def run_stages(profile: Profile, selection: Selection, through: str, *, run_config: LibraryRunConfig,
               index: LibraryIndex, publish: Optional[PublishSettings] = None,
               settings: Optional[ScanSettings] = None, retry_failed: bool = False,
               should_cancel: Optional[Callable[[], bool]] = None,
               on_progress: Optional[Callable[[Progress], None]] = None,
               on_event: Optional[Callable[[object], None]] = None, refresh: bool = True) -> StagesReport:
    '''
    Runs every stage up to and including `through` that each selected title still needs. Never lists a source, never
    reviews (a person's job) and, unless `through` says so, never publishes or commits.

    :param selection: which titles, from the index (its last scan; run `index.scan()` first for a fresh listing).
    :param through: one of selection.THROUGH: `extract`, `design`, `publish` (also writes the accepted -- and out of
        date published -- titles into the repositories' working trees) or `commit` (also commits and pushes them,
        images repository first: one commit and one push per repository for the whole selection).
    :param run_config: designer, analysis, directories; `queue_dir`/`work_dir` are also where publish reads and writes.
    :param publish: required if `through` is publish or commit.
    :param settings: what the index is refreshed with afterwards; default ScanSettings.from_profile(profile). Must match
        what `run` and `publish` are given, as for a scan.
    :param retry_failed: also run titles whose remembered failure still applies (the index keeps it until the source or
        the settings change); their failures are forgotten as they succeed.
    :param should_cancel: polled between titles; True stops before the next one. Titles already done stay done.
    :param on_progress: called from the running thread with a Progress as each title-stage starts, and once at the end.
    :param on_event: structured execution events, including external process commands and responses.
    :param refresh: re-read every title's outputs into the index afterwards (Selection-free and cheap: no source is
        listed), so `needs` is current when this returns -- also after a cancel or a failure.
    :raises ValueError: for an unknown `through`, or publish/commit without `publish` settings.
    '''
    plan = plan_stages(selection.rows(index), through, retry_failed=retry_failed)
    if through in ('publish', 'commit') and publish is None and (plan.with_stage('publish') or plan.with_stage('commit')):
        raise ValueError('xml-repo is required to publish or commit')
    report = StagesReport(through, len(plan.planned) + len(plan.skipped), skipped=list(plan.skipped))

    machine = [p for p in plan.planned if any(s in p.stages for s in ('extract', 'design'))]
    to_publish = plan.with_stage('publish')
    commit_ids = [p.row.id for p in plan.planned if 'commit' in p.stages]
    total = len(machine) + len(to_publish) + len(commit_ids)
    titles = {p.row.id: _title(p.row) for p in plan.planned}
    state = {'done': 0}

    def emit(stage: str, title_id: str = '', title: str = '') -> None:
        if on_progress is not None:
            on_progress(Progress(state['done'], total, title, stage, title_id))

    def cancelled() -> bool:
        return should_cancel is not None and bool(should_cancel())

    run_id = uuid.uuid4().hex
    event_context = execution_event_context(run_id, on_event)
    event_context.__enter__()
    for planned in plan.planned:
        with event_scope(title_id=planned.row.id):
            emit_execution_event('queued', message=titles.get(planned.row.id, planned.row.id))

    try:
        units, unit_errors = _units_by_title(index, [p.row.id for p in machine])
        machine_ids = {p.row.id for p in machine}
        for resource, owners in conflicting_units([units[i] for i in units if i in machine_ids]).items():
            message = f'Work output {resource} is shared by selected titles: {", ".join(owners)}'
            for owner in owners:
                if owner in machine_ids:
                    unit_errors[owner] = message

        eligible = []
        for planned in machine:
            row = planned.row
            if row.id in unit_errors:
                report.run.failed.append((row.id, unit_errors[row.id]))
                report.attempted.append(row.id)
                with event_scope(title_id=row.id):
                    emit_execution_event('failed', message=unit_errors[row.id])
                state['done'] += 1
            elif units.get(row.id) is None:
                report.skipped.append(Skipped(row.id, _title(row), 'not in the last scan: scan again'))
                with event_scope(title_id=row.id):
                    emit_execution_event('skipped', message='Not in the last scan: scan again')
                state['done'] += 1
            else:
                eligible.append((planned, units[row.id]))

        def extract_task(planned, unit):
            local = LibraryRunReport()
            title_id = planned.row.id

            extract_progress = None
            if on_progress is not None or on_event is not None:
                def report_extract_progress(progress_id, out_time, total_time):
                    with event_scope(title_id=progress_id, stage='extract'):
                        emit_execution_event('progress', message='ffmpeg extraction progress', current=out_time,
                                             total=total_time)
                        if on_progress is not None:
                            on_progress(FfmpegProgress(titles.get(progress_id, progress_id), progress_id,
                                                       out_time, total_time))
                extract_progress = report_extract_progress

            with event_scope(title_id=title_id, stage='extract'):
                work = run_unit(Session(run_config.config), unit, run_config, local, index,
                                retry_failed=retry_failed, through='extract',
                                on_stage=lambda progress_id, stage: emit(
                                    stage, progress_id, titles.get(progress_id, progress_id)),
                                on_extract_progress=extract_progress)
            return work, local

        def design_task(planned, work):
            title_id = planned.row.id
            with event_scope(title_id=title_id, stage='design'):
                return design_unit_work(work, run_config, index,
                                        on_stage=lambda progress_id, stage: emit(
                                            stage, progress_id, titles.get(progress_id, progress_id)))

        pending = list(eligible)
        extracting: Dict[Future, object] = {}
        designing: Dict[Future, object] = {}
        with ThreadPoolExecutor(max_workers=run_config.extract_parallelism,
                                thread_name_prefix='library-extract') as extract_pool, \
                ThreadPoolExecutor(max_workers=run_config.design_parallelism,
                                   thread_name_prefix='library-design') as design_pool:
            while pending or extracting or designing:
                cancel_requested = cancelled()
                if not cancel_requested:
                    while pending and len(extracting) < run_config.extract_parallelism:
                        planned, unit = pending.pop(0)
                        future = extract_pool.submit(copy_context().run, extract_task, planned, unit)
                        extracting[future] = planned
                else:
                    report.cancelled = True
                if not extracting and not designing and (not pending or cancelled()):
                    break
                active = set(extracting) | set(designing)
                if not active:
                    break
                completed, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in completed:
                    planned = extracting.pop(future, None)
                    if planned is not None:
                        row_id = planned.row.id
                        try:
                            work, local = future.result()
                        except Exception as error:
                            report.run.failed.append((row_id, f'{type(error).__name__}: {error}'))
                            report.attempted.append(row_id)
                            with event_scope(title_id=row_id):
                                emit_execution_event('failed', message=f'{type(error).__name__}: {error}')
                            state['done'] += 1
                            continue
                        _merge_run_report(report.run, local, include_extract=True)
                        if work is None:
                            if local.failed_earlier:
                                with event_scope(title_id=row_id):
                                    emit_execution_event('skipped', message=local.failed_earlier[-1][1])
                            report.attempted.append(row_id)
                            state['done'] += 1
                            continue
                        if planned.stages[-1] == 'design':
                            design_future = design_pool.submit(copy_context().run, design_task, planned, work)
                            designing[design_future] = planned
                        else:
                            report.attempted.append(row_id)
                            state['done'] += 1
                            with event_scope(title_id=row_id):
                                emit_execution_event('title_completed', message=titles.get(row_id, row_id))
                    else:
                        planned = designing.pop(future)
                        row_id = planned.row.id
                        try:
                            local = future.result()
                        except Exception as error:
                            local = LibraryRunReport(failed=[(row_id, f'{type(error).__name__}: {error}')])
                            with event_scope(title_id=row_id, stage='design'):
                                emit_execution_event('failed', message=f'{type(error).__name__}: {error}')
                        _merge_run_report(report.run, local, include_extract=False)
                        report.attempted.append(row_id)
                        state['done'] += 1
                        if not local.failed:
                            with event_scope(title_id=row_id):
                                emit_execution_event('title_completed', message=titles.get(row_id, row_id))
            if cancelled():
                report.cancelled = True

        if to_publish and not report.cancelled and not cancelled():
            base = state['done']
            wanted = [p.row.id for p in to_publish]
            begun: List[str] = []
            cancel_seen = []

            def before_entry(entry_id: str) -> None:
                state['done'] = base + len(begun)
                begun.append(entry_id)
                with event_scope(title_id=entry_id, stage='publish'):
                    emit_execution_event('stage_started', message='Publishing accepted title')
                emit('publish', entry_id, titles.get(entry_id, entry_id))

            def stop() -> bool:
                if cancelled():
                    cancel_seen.append(True)
                return bool(cancel_seen)

            try:
                # each entry has its own failure boundary inside, so what was published before a failure is kept
                results = publish_library(
                    run_config.queue_dir, publish.xml_repo, meta_defaults=publish.meta_defaults,
                    images_repo=publish.images_repo, image_owner=publish.image_owner,
                    image_repo_name=publish.image_repo_name, xml_dir=publish.xml_dir, image_dir=publish.image_dir,
                    report_spec=publish.report_spec, config=run_config.config, work_dir=run_config.work_dir or None,
                    ids=wanted, republish=True, on_entry=before_entry, should_cancel=stop)
            except Exception as error:  # not one entry's fault (an unreadable queue): report it, do not lose the run
                logger.warning('publish failed: %s', error, exc_info=True)
                results = [{'id': '', 'error': 'publish_failed', 'message': f'{type(error).__name__}: {error}'}]
            report.published, report.publish_errors = split_publish_results(results)
            report.attempted += [r['id'] for r in results if r['id']]
            report.cancelled = bool(cancel_seen)
            # a title publish had nothing to do for (no longer accepted since the scan) is over as well
            state['done'] = base + (len(begun) if report.cancelled else len(wanted))
        elif to_publish:
            report.cancelled = True

        if commit_ids and not report.cancelled and not cancelled():
            with event_scope(stage='commit'):
                emit_execution_event('stage_started', message=f'Committing {len(commit_ids)} titles')
            emit('commit', '', f'{len(commit_ids)} titles')
            try:
                # what the commit takes: the titles that were waiting to be committed and the ones published just now
                with event_scope(stage='commit'):
                    report.committed = commit_library(
                        run_config.queue_dir, publish.xml_repo, images_repo=publish.images_repo,
                        xml_dir=publish.xml_dir, image_dir=publish.image_dir, push=publish.push, ids=commit_ids)
            except subprocess.CalledProcessError as error:
                report.commit_error = f'git failed: {(error.stderr or error.stdout or str(error)).strip()}'
                with event_scope(stage='commit'):
                    emit_execution_event('failed', message=report.commit_error)
                logger.warning('commit failed: %s', report.commit_error)
            else:
                with event_scope(stage='commit'):
                    emit_execution_event('stage_completed', message='Commit/push complete')
                report.attempted += [i for i in commit_ids if i not in report.attempted]
                state['done'] += len(commit_ids)
        elif commit_ids:
            report.cancelled = True
    finally:
        planned_ids = {p.row.id for p in plan.planned}
        report.attempted = list(dict.fromkeys(report.attempted))
        report.not_run = sorted(i for i in planned_ids if i not in report.attempted) if report.cancelled else []
        if refresh:
            try:
                index.refresh(profile, settings or ScanSettings.from_profile(profile))
                report.counts = index.summary().counts
            except Exception as error:  # the work is done and recorded in the outputs; the next scan catches up
                logger.warning('could not refresh the index after the run: %s', error, exc_info=True)
        event_context.__exit__(None, None, None)
    emit('')
    return report
