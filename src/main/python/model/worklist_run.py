'''
Running work from the library work list -- design/library-sync/workflow-rework/design.md §12.7 and §12.10, chunk 26b.

Everything the window's actions need that is not a widget:

* `RunJob`/`RunSignals` -- one run of `pipeline.library.stages.run_stages` on a `QRunnable` in the global thread pool, with
  a connection of its own to the index (as the scan's job has), determinate progress marshalled through a signal and a
  cooperative cancel (a `threading.Event` the pipeline polls between titles);
* `build_run_config()`/`build_publish_settings()` -- the pipeline's settings objects from the window's `WorkListSetup`;
* the words the window shows: `summarise_skipped()` ("3 of 125 skipped: 2 waiting for review, 1 already accepted"),
  `describe_results()` (one line per title: designed, failed, published, refused, committed...), `summarise_report()`
  and `failed_titles()` (the failures panel).

None of this decides what a title needs -- that is the index's, and what a run does to each is `plan_stages()`'s.
'''
import logging
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from qtpy.QtCore import QObject, QRunnable, Signal

from model.preferences import TMDB_API_KEY
from pipeline.library.commit import CatalogueCommit
from pipeline.library.index import LibraryIndex, TitleRow
from pipeline.library.run import LibraryRunConfig
from pipeline.library.selection import Selection, StagePlan
from pipeline.library.stages import PublishSettings, StagesReport, run_stages
from pipeline.publish.catalogue import catalogue_paths
from pipeline.review import describe_publish_error

logger = logging.getLogger('worklist_run')

LEVEL_OK, LEVEL_WARN, LEVEL_ERROR = 'ok', 'warn', 'error'
_LEVEL_ORDER = {LEVEL_ERROR: 0, LEVEL_WARN: 1, LEVEL_OK: 2}


def title_of(row: TitleRow) -> str:
    return row.title or row.display_name or row.id


# --- what a run is ----------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RunRequest:
    '''
    :param through: one of `selection.THROUGH`.
    :param ids: the titles to work on -- never empty (an empty `Selection.ids` would mean every title).
    :param retry_failed: also run titles whose remembered failure still applies.
    :param push: for a commit, push each repository afterwards; False commits locally only.
    '''
    through: str
    ids: Tuple[str, ...]
    retry_failed: bool = False
    push: bool = True

    def __post_init__(self):
        if not self.ids:
            raise ValueError('a run needs at least one title')


class RunSignals(QObject):
    progress = Signal(object)    # pipeline.library.stages.Progress, from the worker thread
    finished = Signal(object)    # StagesReport (also after a cancel)
    errored = Signal(str)


class RunJob(QRunnable):
    '''
    `run_stages()` off the UI thread. `cancel()` is safe to call from any thread: the pipeline polls it between titles (and
    between published entries), finishes the title in hand and reports what it did and did not do. The index is refreshed
    by `run_stages` itself -- after a cancel and after a failure too -- so the window only has to read it again.
    '''

    def __init__(self, index_file: str, profile, settings, run_config: LibraryRunConfig,
                 publish: Optional[PublishSettings], request: RunRequest, runner: Callable = run_stages):
        super().__init__()
        self.signals = RunSignals()
        self.request = request
        self.__index_file, self.__profile, self.__settings = index_file, profile, settings
        self.__run_config, self.__publish, self.__runner = run_config, publish, runner
        self.__cancel = threading.Event()

    def cancel(self) -> None:
        self.__cancel.set()

    @property
    def cancel_requested(self) -> bool:
        return self.__cancel.is_set()

    def run(self):
        try:
            with LibraryIndex(self.__index_file) as index:
                report = self.__runner(
                    self.__profile, Selection(ids=self.request.ids), self.request.through,
                    run_config=self.__run_config, index=index, publish=self.__publish, settings=self.__settings,
                    retry_failed=self.request.retry_failed, should_cancel=self.__cancel.is_set,
                    on_progress=self.signals.progress.emit)
        except Exception as error:
            logger.exception('Library run failed')
            self.signals.errored.emit(f'{type(error).__name__}: {error}')
            return
        self.signals.finished.emit(report)


# --- settings ---------------------------------------------------------------------------------------------------------

def build_run_config(setup, preferences) -> LibraryRunConfig:
    ''' What `run_stages` is given for extract and design: the scan's settings (they must agree) and the TMDB key. '''
    settings = setup.settings
    run = (setup.profile.config.get('run') or {}) if setup.profile else {}
    return LibraryRunConfig(
        work_dir=settings.work_dir, queue_dir=settings.queue_dir, designer=settings.designer, config=settings.config,
        coverage=settings.coverage, keep_multichannel=settings.keep_multichannel, tv_mode=settings.tv_mode,
        tmdb_api_key=preferences.get(TMDB_API_KEY) or None, audio_types=tuple(run.get('audio_types') or ()))


def publish_problem(setup) -> str:
    ''' Why publish and commit cannot run, or '' if they can. '''
    if setup.settings is None or not setup.settings.xml_repo:
        return 'No XML repository is set. Set one in Library Sync, then reopen this window.'
    return ''


def build_publish_settings(setup, push: bool = True) -> PublishSettings:
    '''
    :raises ValueError: if no XML repository is set (see publish_problem()).
    '''
    sync = (setup.profile.config.get('sync') or {}) if setup.profile else {}
    return PublishSettings.from_scan_settings(setup.settings, image_owner=sync.get('image_owner'),
                                              image_repo_name=sync.get('image_repo_name'), push=push)


# --- what a selection would do ----------------------------------------------------------------------------------------

def skip_reason(row: TitleRow, retry_failed: bool = False) -> str:
    ''' Why a run over this title's kind of need does nothing to it, in the words of the work list. '''
    if row.needs == 'review':
        return 'waiting for review'
    if row.needs == 'publish':
        return 'already accepted'
    if row.needs == 'commit':
        return 'already published'
    if row.needs == 'attention':
        if row.extract_state == 'failed' or row.design_state == 'failed':
            return 'failed before' if not retry_failed else 'needs attention'
        return 'needs attention'
    if row.needs == 'done':
        return 'already done'
    return row.detail or 'nothing to do'


def summarise_skipped(plan: StagePlan, rows: Mapping[str, TitleRow], retry_failed: bool = False) -> str:
    '''
    "3 of 125 skipped: 2 waiting for review, 1 already accepted" -- what the action leaves out and why; '' if it leaves
    nothing out.
    '''
    if not plan.skipped:
        return ''
    reasons: Dict[str, int] = {}
    for skipped in plan.skipped:
        row = rows.get(skipped.id)
        reason = skip_reason(row, retry_failed) if row is not None else skipped.reason
        reasons[reason] = reasons.get(reason, 0) + 1
    total = len(plan.planned) + len(plan.skipped)
    parts = ', '.join(f'{count:,} {reason}' for reason, count in sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0])))
    return f'{len(plan.skipped):,} of {total:,} skipped: {parts}'


def plan_label(plan: StagePlan) -> str:
    ''' The action button's text: `StagePlan.label` with thousands separators. '''
    verb = {'extract': 'Extract', 'design': 'Extract & design', 'publish': 'Publish', 'commit': 'Commit'}[plan.through]
    text = f'{verb} {len(plan.planned):,}'
    if plan.skipped:
        text += f' ({len(plan.skipped):,} of {len(plan.planned) + len(plan.skipped):,} skipped)'
    return text


# --- the failures panel -----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class FailedTitle:
    id: str
    title: str
    source: str
    stage: str
    reason: str


def failed_titles(rows: Sequence[TitleRow], failures: Mapping[str, object]) -> List[FailedTitle]:
    '''
    The titles whose extract or design failed (the index's `failed` state, which is a remembered failure that still
    applies), with the reason the index kept, in the work list's order.
    '''
    found = []
    for row in rows:
        stage = 'extract' if row.extract_state == 'failed' else 'design' if row.design_state == 'failed' else ''
        if not stage:
            continue
        memory = failures.get(row.id)
        reason = (getattr(memory, 'message', '') or row.failure or row.detail) if memory or row.failure else row.detail
        found.append(FailedTitle(row.id, title_of(row), row.source, getattr(memory, 'stage', '') or stage, reason))
    return found


# --- the results list -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ResultLine:
    id: str
    title: str
    outcome: str
    detail: str = ''
    level: str = LEVEL_OK


def _refusal(result: dict) -> str:
    text = describe_publish_error(result)
    prefix = f"{result['id']}: "
    return text[len(prefix):] if text.startswith(prefix) else text


def describe_results(report: StagesReport, plan: StagePlan, settings, rows: Optional[Mapping[str, TitleRow]] = None
                     ) -> List[ResultLine]:
    '''
    One line per title the run touched (what happened to it, in words), then one per repository for a commit. Problems
    first: a failure, a refusal or a title that was not run comes before what went well.

    :param plan: what was planned (`plan_stages()` for the request), for the titles' names and what each was to get.
    :param settings: the `ScanSettings` (its `xml_dir`/`image_dir` say where a title's files are, for the commit).
    '''
    titles = {p.row.id: title_of(p.row) for p in plan.planned}
    titles.update({s.id: s.title for s in plan.skipped})
    outcomes: Dict[str, List[str]] = {}
    details: Dict[str, List[str]] = {}
    levels: Dict[str, str] = {}

    def note(title_id: str, outcome: str, level: str = LEVEL_OK, detail: str = '') -> None:
        outcomes.setdefault(title_id, []).append(outcome)
        if detail:
            details.setdefault(title_id, []).append(detail)
        if _LEVEL_ORDER[level] < _LEVEL_ORDER[levels.get(title_id, LEVEL_OK)]:
            levels[title_id] = level

    run = report.run
    failed = dict(run.failed)
    earlier = dict(run.failed_earlier)
    for title_id, reason in failed.items():
        note(title_id, 'failed', LEVEL_ERROR, reason)
    for title_id, reason in earlier.items():
        note(title_id, 'not retried', LEVEL_WARN, f'failed before: {reason}. Retry failed runs it again.')
    handled = set(failed) | set(earlier)
    for title_id in run.designed:
        if title_id not in handled:
            note(title_id, 'designed')
    for title_id in run.design_cached:
        if title_id not in handled and title_id not in run.designed:
            note(title_id, 'already designed')
    for title_id in run.extracted + run.cached:
        if title_id not in handled and title_id not in run.designed and title_id not in run.design_cached:
            note(title_id, 'extracted' if title_id in run.extracted else 'already extracted')
    for result in report.published:
        detail = 'from your project edits' if 'edited_project' in result else ''
        note(result['id'], 'republished' if result.get('republished') else 'published', LEVEL_OK, detail)
        titles.setdefault(result['id'], result['id'])
    for result in report.publish_errors:
        note(result['id'], 'refused', LEVEL_ERROR, _refusal(result))
        titles.setdefault(result['id'], result['id'])

    repo_lines: List[ResultLine] = []
    committed: Optional[CatalogueCommit] = report.committed
    commit_ids = [p.row.id for p in plan.with_stage('commit')]
    published_ids = {r['id'] for r in report.published}
    if committed is not None:
        handled_paths = set(committed.xml.paths)
        for title_id in commit_ids:
            xml_path = catalogue_paths(title_id, settings.xml_dir, settings.image_dir)[0]
            if xml_path in handled_paths:
                note(title_id, 'committed')
                if committed.xml.pushed:
                    note(title_id, 'pushed')
            elif 'refused' in outcomes.get(title_id, ()):
                pass
            elif title_id in published_ids:
                note(title_id, 'nothing new to commit')
            else:
                note(title_id, 'nothing to commit', LEVEL_WARN, 'its file is not in the repository')
        for name, repo in (('Images repository', committed.images), ('XML repository', committed.xml)):
            if repo is None:
                continue
            what = (f'commit {repo.commit[:8]}' if repo.commit else 'nothing new to commit') + \
                   (', pushed' if repo.pushed else ', not pushed')
            repo_lines.append(ResultLine('', name, 'committed' if repo.commit else 'unchanged',
                                         f'{what} ({len(repo.paths)} file{"s" if len(repo.paths) != 1 else ""}) -- '
                                         f'{repo.repo}'))
        if committed.missing:
            repo_lines.append(ResultLine('', 'Published files missing', 'warning',
                                         ', '.join(committed.missing), LEVEL_WARN))
    if report.commit_error:
        repo_lines.append(ResultLine('', 'Commit', 'failed', report.commit_error, LEVEL_ERROR))
        for title_id in commit_ids:
            note(title_id, 'not committed', LEVEL_ERROR, report.commit_error)

    for title_id in report.not_run:
        note(title_id, 'not run', LEVEL_WARN, 'the run was cancelled before it got here')
    for skipped in report.skipped:
        if skipped.id in outcomes:
            continue
        row = (rows or {}).get(skipped.id)
        reason = skipped.reason if row is None or skipped.reason.startswith('not in the last scan') \
            else skip_reason(row)
        note(skipped.id, 'skipped', LEVEL_WARN if skipped.reason.startswith('not in the last scan') else LEVEL_OK,
             reason)
        titles.setdefault(skipped.id, skipped.title)

    lines = []
    for title_id, phrases in outcomes.items():
        text = ', '.join(phrases)
        lines.append(ResultLine(title_id, titles.get(title_id, title_id), text[:1].upper() + text[1:],
                                '; '.join(details.get(title_id, ())), levels.get(title_id, LEVEL_OK)))
    lines.sort(key=lambda line: _LEVEL_ORDER[line.level])   # stable: the work list's order within a level
    problems = [line for line in repo_lines if line.level != LEVEL_OK]
    return lines + [line for line in repo_lines if line.level == LEVEL_OK] + problems


def summarise_report(report: StagesReport, plan: StagePlan) -> Tuple[str, str]:
    '''
    The one-line outcome of a run and its level: "Extract & design finished: 38 designed, 2 failed, 4 skipped" or, after a
    cancel, "Stopped after 2 of 5 titles (3 not run): 2 designed".
    '''
    run = report.run
    parts = []
    for count, word in ((len(run.designed), 'designed'), (len(run.design_cached), 'already designed'),
                        (len([i for i in run.extracted if i not in run.designed]) if plan.through == 'extract'
                         else 0, 'extracted'),
                        (len(report.published), 'published'), (len(report.publish_errors), 'refused'),
                        (len(run.failed), 'failed'), (len(run.failed_earlier), 'not retried')):
        if count:
            parts.append(f'{count:,} {word}')
    committed = report.committed
    if committed is not None and (committed.xml.commit or (committed.images and committed.images.commit)):
        parts.append('committed' + (' and pushed' if committed.xml.pushed else ' (not pushed)'))
    if report.commit_error:
        parts.append('the commit failed')
    if report.skipped:
        parts.append(f'{len(report.skipped):,} skipped')
    detail = ', '.join(parts) or 'nothing to do'
    level = LEVEL_ERROR if report.failed else LEVEL_OK
    if report.cancelled:
        done, planned = len(report.attempted), len(plan.planned)
        return (f'Stopped after {done:,} of {planned:,} title{"" if planned == 1 else "s"} '
                f'({len(report.not_run):,} not run): {detail}', LEVEL_ERROR if report.failed else LEVEL_WARN)
    verb = {'extract': 'Extract', 'design': 'Extract & design', 'publish': 'Publish', 'commit': 'Commit'}[plan.through]
    return f'{verb} finished: {detail}', level
