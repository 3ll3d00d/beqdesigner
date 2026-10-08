'''
The work list's actions -- design/archive/library-sync/workflow-rework/design.md §12.7 and §12.10, chunk 26b. A mixin of
`model.worklist.WorkListWindow` (which owns the widgets, the model, the index and the setup, and declares the signals):

* what the buttons work on (`target_ids()`, `plan_for()`) and their labels (`_refresh_actions()`);
* starting a run -- `run_selected()`, `publish_selected()`, `commit_selected()`, `retry_failed()`, each behind the
  confirmation it needs -- and `cancel_run()`;
* following a run on the UI thread (progress, the running-row marker) and showing its result (the failures panel and the
  *Last run* tab).

The run itself is `model.worklist_run.RunJob`; the words are in `model.worklist_run`; the confirmations in
`model.worklist_confirm`.
'''
import logging
import time
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

from qtpy.QtCore import QTimer, Qt, QThreadPool
from qtpy.QtWidgets import QDialog, QLabel

from model.preferences import WORKLIST_PUSH
from model.execution_events import ExecutionEvent
from model.worklist_failure import retry_label
from model.worklist_confirm import ConfirmDialog, commit_text, machine_text, publish_text, retry_text
from model.worklist_model import warning_colour
from model.worklist_run import LEVEL_ERROR, LEVEL_OK, FailedTitle, ResultLine, RunJob, RunRequest, \
    build_publish_settings, build_run_config, describe_results, plan_label, publish_problem, summarise_report, \
    summarise_skipped
from model.worklist_run_details import EventBuffer, save_run_details
from pipeline.library.inbox import PENDING, WorkDirInbox
from pipeline.library.index import TitleRow
from pipeline.library.join import JOINABLE, JoinRequest
from pipeline.library.selection import Selection, StagePlan, plan_stages
from pipeline.library.stages import FfmpegProgress, Progress, StagesReport
from pipeline.service.lease import read_lease

logger = logging.getLogger('worklist')


def _button_text(text: str) -> str:
    ''' A button's text without Qt reading an "&" as a mnemonic: "Extract & design 3" is shown as written. '''
    return text.replace('&', '&&')



@dataclass
class _RunContext:
    ''' What the window remembers of the run in flight, to word its progress and its result. '''
    request: RunRequest
    plan: StagePlan
    rows: Dict[str, TitleRow]
    skipped_text: str
    commit_ids: List[str]
    stage: str = ''   # what the pipeline last said it was starting
    active: Dict[str, str] = field(default_factory=dict)   # title id -> what it is doing, for the titles in flight


class WorkListActions:
    '''
    The mixin: it uses what `WorkListWindow` sets up (`_model`, `_proxy`, `_setup`, `_index`, `_preferences`, `_job`,
    `_run_context`, `_failed`, `_results`, `_precheck`, `_run_stages`, `_scanning`, the widgets and `_refresh_view()`).
    '''

    def _configure_actions(self) -> None:
        self.workTable.selectionModel().selectionChanged.connect(self._refresh_actions)
        self.runLayout.setStretchFactor(self.runStatusLabel, 1)
        self.runCountsLabel = QLabel(self.runPanel)
        self.runCountsLabel.setObjectName('runCountsLabel')
        self.runCountsLabel.setTextFormat(Qt.TextFormat.PlainText)
        self.runLayout.insertWidget(1, self.runCountsLabel)
        self.runButton.clicked.connect(lambda: self.run_selected())
        self.publishButton.clicked.connect(lambda: self.publish_selected())
        self.commitButton.clicked.connect(lambda: self.commit_selected())
        self.retryButton.clicked.connect(lambda: self.retry_failed())
        self.cancelButton.clicked.connect(self.cancel_run)
        self.selectAllButton.clicked.connect(self.workTable.selectAll)
        self.clearSelectionButton.clicked.connect(self.workTable.clearSelection)
        self.runPanel.setVisible(False)
        self.runProgress.setTextVisible(True)

    @property
    def is_running(self) -> bool:
        return self._job is not None

    @property
    def failed(self) -> List[FailedTitle]:
        ''' The failures panel's titles: extract or design failed, and the failure still applies. '''
        return list(self._failed)

    @property
    def results(self) -> List[ResultLine]:
        ''' What the last run did to each title (the *Last run* tab). '''
        return list(self._results)

    def target_ids(self) -> List[str]:
        ''' What the buttons work on: the selected rows, or -- with none selected -- everything the filters list. '''
        return self.selected_ids() or self.listed_ids()

    def _rows_by_id(self) -> Dict[str, TitleRow]:
        return {row.id: row for row in self._model.rows}

    def plan_for(self, through: str, ids: Optional[List[str]] = None, retry_failed: bool = False) -> StagePlan:
        '''
        What running `through` over these titles (default: `target_ids()`) would do -- the button's label and the run
        itself come from this. Publish and Commit are for the titles that need exactly that, so a title that still needs
        extracting is never published by pressing Publish. While a run is going, extract and design leave out the titles
        already in it: what the button adds to it is only what is new.
        '''
        rows = self._rows_by_id()
        targets = [rows[i] for i in (self.target_ids() if ids is None else ids) if i in rows]
        if through in ('publish', 'commit'):
            targets = [row for row in targets if row.needs == through]
        elif self._run_context is not None:
            in_run = set(self._run_context.request.ids)
            targets = [row for row in targets if row.id not in in_run]
        return plan_stages(targets, through, retry_failed=retry_failed)

    def _busy(self) -> bool:
        return self.is_running or self._blocking()

    def _blocking(self) -> bool:
        '''
        What keeps the run, Publish, Commit and Retry buttons from being used: a scan, an index sync or a bulk job. A run in
        progress does not (design/archive/library-sync/worklist-feedback.md F5): more extract and design work joins it,
        anything else waits for it to end and then starts.
        '''
        return self._scanning or self._syncing or self._bulk_job is not None

    def _refresh_actions(self, *_) -> None:
        ''' The selection text and the four buttons: their labels say how many titles each will work on. '''
        rows = self._rows_by_id()
        selected, listed = self.selected_ids(), self._proxy.rowCount()
        self.selectionLabel.setText(f'{len(selected):,} selected of {listed:,} listed' if selected else
                                    f'None selected: the buttons work on all {listed:,} listed' if listed else '')
        ready = self._setup.ready and self._index is not None and not self._blocking()
        running = self.is_running
        plan = self.plan_for('design')
        self.runButton.setText(_button_text(f'Add {len(plan.planned):,} to the run' if running and plan.planned
                                            else plan_label(plan)))
        self.runButton.setEnabled(ready and bool(plan.planned))
        self.runButton.setToolTip(
            'Extract the audio and design a filter for each title that needs it. A title is never taken past design '
            'without a person: reviewing is done on the title page, and Publish and Commit are the buttons beside this one.')
        self.skippedLabel.setText(summarise_skipped(plan, rows))
        self.skippedLabel.setVisible(bool(plan.skipped))
        problem = publish_problem(self._setup)
        for button, through in ((self.publishButton, 'publish'), (self.commitButton, 'commit')):
            step = self.plan_for(through)
            text = plan_label(step)
            if through == 'commit' and step.planned and not self._uncommitted(step):
                text = 'Push' + text[len('Commit'):]   # every one is committed already: all that is left is pushing
            button.setText(_button_text(f'{text} after the run' if running and step.planned else text))
            button.setEnabled(ready and bool(step.planned) and not problem)
            if problem:
                button.setToolTip(problem)
            elif through == 'publish':
                button.setToolTip('Write the accepted titles (and those published but out of date) into the repositories\' '
                                  'working trees. Nothing is committed or pushed.')
            else:
                button.setToolTip('Commit the written titles: one commit per repository, images first, then push. '
                                  'Titles committed earlier without a push are only pushed.')
        retry = self._retry_ids()
        self.retryButton.setText(_button_text(retry_label([f.stage for f in self._failed if f.id in retry], len(retry))))
        self.retryButton.setVisible(bool(self._failed))
        self.retryButton.setEnabled(ready and bool(retry))
        self.retryButton.setToolTip('Extract and design the failed titles again, even though nothing changed: the selected '
                                    'ones, else every failed title listed. The Attention chip lists them all.')
        self.rescanButton.setEnabled(self._setup.ready and not self._busy())
        self._refresh_open_button()
        self._refresh_bulk_actions()
        self._refresh_settings_state()

    @staticmethod
    def _uncommitted(plan: StagePlan) -> int:
        ''' How many of a commit plan's titles still have to be committed (the others only need pushing). '''
        return sum(1 for p in plan.planned if p.row.commit_state != 'committed')

    def _retry_ids(self) -> List[str]:
        ''' What Retry works on: the selected titles that failed, else every failed title the view lists. '''
        failed = {f.id for f in self._failed}
        return [title_id for title_id in (self.selected_ids() or self.listed_ids()) if title_id in failed]

    def _refresh_failures(self) -> None:
        self._refresh_actions()

    def _refresh_results(self) -> None:
        '''
        The last run's outcomes are on each title's row; what belongs to no title (a repository that could not be committed
        or pushed) is summed up on the status line, and said in full in its tooltip.
        '''
        text = '\n\n'.join(f'{line.title or line.outcome}: {line.full}' for line in self._results if not line.id and line.full)
        self.runStatusLabel.setToolTip(text)

    # --- starting a run -------------------------------------------------------------------------------------------------

    def run_selected(self) -> bool:
        ''' The action button: extract and design what needs it (the machine tier), through design and no further. '''
        return self._begin('design')

    def publish_selected(self) -> bool:
        ''' Publish: write the accepted (and out-of-date published) titles into the repositories, after a confirmation. '''
        return self._begin('publish')

    def commit_selected(self) -> bool:
        ''' Commit: one commit and, unless unticked, one push per repository for the written titles, after a confirmation. '''
        return self._begin('commit')

    def retry_failed(self, ids: Optional[List[str]] = None) -> bool:
        ''' Runs failed titles again even though nothing changed: these, else those Retry works on (`_retry_ids()`). '''
        wanted = ids if ids is not None else self._retry_ids()
        if not wanted:
            self._say('Nothing to retry: no failed title is selected or listed.')
            return False
        # every failed title a view lists can be as big as running the whole view, so that asks; a choice does not
        return self._begin('design', retry_failed=True, ids=list(wanted), confirm=ids is None and not self.selected_ids())

    def _say(self, text: str, level: str = LEVEL_OK) -> None:
        self.runPanel.setVisible(bool(text) or self.is_running)
        self.runStatusLabel.setText(text)
        self.runStatusLabel.setStyleSheet(f'color: {warning_colour().name()}' if level != LEVEL_OK else '')

    def _begin(self, through: str, retry_failed: bool = False, ids: Optional[List[str]] = None,
               confirm: Optional[bool] = None) -> bool:
        '''
        Confirms where that is called for and starts the run. :return: False if nothing was started (a run or a scan is in
        progress, the setup is incomplete, there is nothing to do, or the person declined).

        :param confirm: whether an extract/design run over more than one title asks first; by default it does unless
            the person selected the titles.
        '''
        self._flush_settings()   # a setting edited a moment ago is what this run must use
        if self._blocking() or not self._setup.ready or self._index is None or self._setup.index_file is None:
            return False
        holder = None if self.is_running else read_lease(self._setup.profile.work_dir if self._setup.profile else None)
        if holder is not None and through not in JOINABLE:   # two runs would both write the index and the repositories
            self._say(f'Cannot start: {holder.describe()}.', LEVEL_ERROR)
            return False
        ask = (not (ids is not None or bool(self.selected_ids()))) if confirm is None else confirm
        rows = self._rows_by_id()
        plan = self.plan_for(through, ids, retry_failed)
        if not plan.planned:
            self._say('Nothing to do for this selection.')
            return False
        push = True
        if through in ('publish', 'commit'):
            problem = publish_problem(self._setup)
            if problem:
                self._say(problem, LEVEL_ERROR)
                return False
            settings = build_publish_settings(self._setup, preferences=self._preferences)
            count = len(plan.planned)
            if through == 'publish':
                heading, body = publish_text(count, settings, sum(1 for p in plan.planned
                                                                  if p.row.publish_state == 'out_of_date'))
                dialog = ConfirmDialog(self, heading, body, f'Publish {count:,} title{"" if count == 1 else "s"}')
            else:
                uncommitted = self._uncommitted(plan)
                heading, body = commit_text(count, settings, uncommitted)
                verb = 'Commit' if uncommitted else 'Push'
                dialog = ConfirmDialog(self, heading, body, f'{verb} {count:,} title{"" if count == 1 else "s"}',
                                       'Push each repository after committing (untick to commit locally only)',
                                       bool(self._preferences.get(WORKLIST_PUSH)))
            if dialog.exec() != QDialog.DialogCode.Accepted:
                self._say(f'{through.capitalize()} cancelled: nothing was '
                           f'{"written" if through == "publish" else "committed"}.')
                return False
            if through == 'commit':
                push = dialog.checked
                self._preferences.set(WORKLIST_PUSH, push)
        else:
            # ffmpeg is only needed to extract: a title that is extracted already is only designed
            extracts = any('extract' in p.stages for p in plan.planned)
            if extracts and self._precheck is not None and not self._precheck(through):
                return False
            if ask and len(plan.planned) > 1:
                heading, body = (retry_text(len(plan.planned)) if retry_failed else
                                 machine_text(len(plan.planned), True, self._view_description()))
                dialog = ConfirmDialog(self, heading, body, f'{"Retry" if retry_failed else "Extract & design"} '
                                                            f'{len(plan.planned):,}')
                if dialog.exec() != QDialog.DialogCode.Accepted:
                    self._say('Cancelled: nothing was extracted or designed.')
                    return False
        request = RunRequest(through, tuple(p.row.id for p in plan.planned), retry_failed, push)
        skipped = summarise_skipped(plan, rows, retry_failed)
        if self.is_running:
            return self._add_to_run(request, plan, rows, skipped)
        if holder is not None:   # another process's run: its extract and design work takes these too
            return self._hand_off(holder, request)
        return self._launch(request, plan, rows, skipped)

    # --- work handed to another process's run (design/archive/library-sync/worklist-feedback.md F5) -------------------

    def _hand_off(self, holder, request: RunRequest) -> bool:
        '''
        Posts the titles to the work directory's join inbox, for the run holding the lease (the service, a command-line
        run, another work list), and follows the index until that run ends. If it ends without taking them, they are run
        here. :return: True.
        '''
        work_dir = self._setup.profile.work_dir
        join = JoinRequest(Selection(ids=request.ids), request.through, request.retry_failed)
        WorkDirInbox(work_dir).post(join)
        self._handed_off.append((join.id, request))
        for title_id in request.ids:
            self._model.set_run_state(title_id, active=False, queued=True, stage='', text=f'Handed to {holder.who()}')
        count = len(request.ids)
        self._say(f'Handed {count:,} title{"" if count == 1 else "s"} to {holder.who()}; the list follows it until it ends.')
        if self._handoff_timer is None:
            self._handoff_timer = QTimer(self)
            self._handoff_timer.timeout.connect(self._follow_handed_off)
        self._handoff_timer.start(self.handoff_poll_ms)
        self._refresh_view()
        return True

    def _follow_handed_off(self) -> None:
        ''' Reads the index again while the other run goes on; once it ends, runs here what it did not take. '''
        work_dir = self._setup.profile.work_dir if self._setup.profile else None
        if work_dir is None or self._closed:
            return
        holder = read_lease(work_dir)
        self.refresh_from_index()
        if holder is not None:
            return
        inbox = WorkDirInbox(work_dir)
        for join_id, request in self._handed_off:
            if inbox.state(join_id) == PENDING and inbox.withdraw(join_id):
                self._queued_runs.append(request)   # nobody took them: they are this window's to run
            inbox.forget(join_id)
            for title_id in request.ids:
                self._model.set_run_state(title_id, active=False, queued=False, text='')
        self._handed_off.clear()
        self._handoff_timer.stop()
        self._refresh_view()
        self._start_next_queued_run()

    @property
    def handed_off(self) -> List[str]:
        ''' The titles handed to another process's run and not yet over. '''
        return [title_id for _, request in self._handed_off for title_id in request.ids]

    # --- more work while a run is going (design/archive/library-sync/worklist-feedback.md F5) -------------------------

    def _add_to_run(self, request: RunRequest, plan: StagePlan, rows: Dict[str, TitleRow], skipped_text: str) -> bool:
        '''
        Extract and design work joins the run in progress while its machine phase lasts; anything else -- Publish, Commit,
        or work offered once that phase is over -- waits, and starts when the run ends. :return: True (either way).
        '''
        if request.through in JOINABLE and self._job is not None:
            join = JoinRequest(Selection(ids=request.ids), request.through, request.retry_failed)
            if self._job.join.offer(join):
                self._offered_joins[join.id] = request
                self._joined(request, plan, rows)
                count = len(request.ids)
                self._say(f'Added {count:,} title{"" if count == 1 else "s"} to the run in progress.')
                self._refresh_actions()
                self._refresh_view()
                return True
        self._queued_runs.append(request)
        self._say(f'{plan_label(plan)} starts when the run in progress ends'
                  f' ({len(self._queued_runs):,} waiting).')
        self._refresh_actions()
        return True

    def _joined(self, request: RunRequest, plan: StagePlan, rows: Dict[str, TitleRow]) -> None:
        ''' The run's context, progress and rows take in the titles that joined it. '''
        context = self._run_context
        if context is None:
            return
        new = [p for p in plan.planned if p.row.id not in context.request.ids]
        context.request = replace(context.request, ids=context.request.ids + tuple(p.row.id for p in new))
        context.plan.planned.extend(new)
        context.rows.update(rows)
        for planned in new:
            self._run_outcomes[planned.row.id] = 'queued'
            self._model.set_run_state(planned.row.id, active=False, queued=True, stage='', text='Queued', current=None,
                                      total=None, has_details=False, attempting=True, attempt_detail='',
                                      previous_failure=self._failure_info(planned.row.id))
        self._update_run_progress()
        self._update_run_summary()

    def _requeue_unjoined(self, not_joined: List[str], dropped: bool) -> None:
        ''' What was offered to the run that ended and not taken goes first in line, unless the person cancelled. '''
        unjoined = [self._offered_joins[i] for i in not_joined if i in self._offered_joins]
        self._offered_joins.clear()
        if dropped:
            self._queued_runs.clear()
        else:
            self._queued_runs[:0] = unjoined

    @property
    def queued_runs(self) -> List[RunRequest]:
        ''' The runs waiting for the one in progress to end, in the order they start. '''
        return list(self._queued_runs)

    def _start_next_queued_run(self) -> bool:
        '''
        Starts the first waiting run, planned again (the rows have moved on: a title done by the run that ended is left
        out). A waiting run with nothing left to do is dropped, and the next one tried. :return: True if one started.
        '''
        while self._queued_runs and not self.is_running and not self._blocking():
            request = self._queued_runs.pop(0)
            rows = self._rows_by_id()
            plan = self.plan_for(request.through, list(request.ids), request.retry_failed)
            if not plan.planned:
                continue
            fresh = replace(request, ids=tuple(p.row.id for p in plan.planned))
            if self._launch(fresh, plan, rows, summarise_skipped(plan, rows, request.retry_failed)):
                return True
        return False

    def _view_description(self) -> str:
        parts = [self._proxy.chip]
        if self._proxy.source_filter:
            parts.append(f'source {self._proxy.source_filter}')
        if self._proxy.text:
            parts.append(f'matching "{self._proxy.text}"')
        return ', '.join(parts)

    def _launch(self, request: RunRequest, plan: StagePlan, rows: Dict[str, TitleRow], skipped_text: str) -> bool:
        setup = self._setup
        try:
            run_config = build_run_config(setup, self._preferences)
            publish = build_publish_settings(setup, request.push, self._preferences) if request.through in ('publish', 'commit') else None
        except Exception as error:
            logger.exception('Could not prepare the run')
            self._say(f'Cannot start: {error}', LEVEL_ERROR)
            return False
        job = RunJob(setup.index_file, setup.profile, setup.settings, run_config, publish, request, self._run_stages)
        job.signals.progress.connect(lambda progress, source=job: self._on_run_progress(source, progress))
        job.signals.event.connect(lambda event, source=job: self._on_execution_event(source, event))
        job.signals.finished.connect(lambda report, source=job: self._on_run_finished(source, report))
        job.signals.errored.connect(lambda message, source=job: self._on_run_failed(source, message))
        self._job = job
        # Details is one in-memory generation for the whole window. A new run
        # expires every title's prior history, including titles outside this
        # run's selection.
        for dialog in list(self._detail_dialogs.values()):
            dialog.close()
        self._detail_dialogs.clear()
        self._event_buffers.clear()
        self._saved_details.clear()
        self._save_run_details()
        self._active_run_id = ''
        self._run_outcomes = {title_id: 'queued' for title_id in request.ids}
        self._model.clear_run_states()
        for title_id in request.ids:
            self._model.set_run_state(title_id, active=False, queued=True, stage='', text='Queued', current=None,
                                      total=None, has_details=False, attempting=True, attempt_detail='',
                                      previous_failure=self._failure_info(title_id))
        self._run_context = _RunContext(request, plan, rows, skipped_text,
                                         [p.row.id for p in plan.with_stage('commit')])
        self.cancelButton.setVisible(True)
        self.cancelButton.setEnabled(True)
        self.runProgress.setVisible(True)
        self.runProgress.setRange(0, max(1, len(plan.planned)))
        self.runProgress.setValue(0)
        self.runProgress.setFormat(f'0 / {len(plan.planned)} titles')
        self._update_run_progress()
        self._say(f'Starting: {plan_label(plan)}...')
        self._update_run_summary()
        self._refresh_actions()
        self._refresh_view()
        try:
            QThreadPool.globalInstance().start(job)
        except Exception as error:   # the run never began: do not leave the window "running" for ever
            logger.exception('Could not start the run')
            self._end_run()
            self._model.clear_run_states()
            self._say(f'Cannot start: {type(error).__name__}: {error}', LEVEL_ERROR)
            self._refresh_actions()
            self._refresh_view()
            return False
        self.run_started.emit(request)   # after the job is under way, so nothing a listener does can strand it
        return True

    # --- while it runs --------------------------------------------------------------------------------------------------

    def cancel_run(self) -> bool:
        '''
        Asks the run to stop after the title being worked on (and, in a publish, after the entry in hand); nothing is
        committed by a run that was cancelled. :return: False if no run is in progress.
        '''
        if self._job is None:
            return False
        self._job.cancel()
        self._job.join.close()   # nothing more joins a run that is stopping; what was offered is dropped with it
        dropped = len(self._queued_runs)
        self._queued_runs.clear()
        self.cancelButton.setEnabled(False)
        if self._run_context is not None and self._run_context.stage == 'commit':
            self._say('Cancel requested, but a commit cannot be stopped part way: it finishes, and the run ends after it.')
        else:
            self._say('Cancelling: the title being worked on finishes first...' +
                      (f' {dropped:,} waiting run{"" if dropped == 1 else "s"} will not start.' if dropped else ''))
        return True

    def _on_run_progress(self, source_job, progress: Progress) -> None:
        if self._job is None or source_job is not self._job:
            return
        context = self._run_context
        if context is None:
            return
        title_scoped = isinstance(progress, FfmpegProgress) or bool(progress.id)
        if title_scoped and progress.id not in context.request.ids:
            # Stage progress may be shared (empty id), but title-scoped
            # updates must belong to this run, just as execution events must.
            return
        if isinstance(progress, FfmpegProgress):
            self._on_ffmpeg_progress(progress)
            return
        # Stage starts are activity, not completed titles. The shared bar is
        # driven only by terminal per-title outcomes below.
        if not progress.stage:
            self._model.clear_active_run_states()
            return
        context.stage = progress.stage
        if progress.stage == 'commit':
            for title_id in context.commit_ids:
                self._model.set_run_state(title_id, active=True, queued=False, stage='commit', text='Committing')
            text = f'Committing {progress.title}'
            text += f'  ({min(progress.done + 1, progress.total):,} of {progress.total:,})'
        else:
            word = {'extract': 'Extracting', 'design': 'Designing', 'publish': 'Publishing'}.get(progress.stage,
                                                                                                 progress.stage)
            if progress.id:
                self._model.set_run_state(progress.id, active=True, queued=False, stage=progress.stage, text=word)
                context.active[progress.id] = f'{word} {progress.title}'
            text = self._in_flight_text(context, f'{word} {progress.title}', progress)
        if self._job is not None and self._job.cancel_requested:
            text += ' -- a commit cannot be stopped, it finishes' if progress.stage == 'commit' \
                else ' -- cancelling after this one'
        self._say(text)

    def _in_flight_text(self, context: _RunContext, single: str, progress: Progress) -> str:
        '''
        One line for the status: the title's own words while it is the only one in flight, else a count, so that titles
        running side by side do not take turns to overwrite each other.
        '''
        finished = {title_id for title_id, outcome in self._run_outcomes.items() if outcome in ('succeeded', 'failed', 'cancelled')}
        for title_id in finished:
            context.active.pop(title_id, None)
        if len(context.active) > 1:
            return f'{len(context.active):,} titles in progress  ({len(finished):,} of {progress.total:,} done)'
        return single + f'  ({min(progress.done + 1, progress.total):,} of {progress.total:,})'

    def _on_ffmpeg_progress(self, progress: FfmpegProgress) -> None:
        '''Keep ffmpeg's ``out_time_ms`` percentage in the title's own row.'''
        if progress.total_micros <= 0:
            return
        percent = min(100, max(0, int(progress.out_time_micros * 100 / progress.total_micros)))
        self._model.set_run_state(progress.id, active=True, queued=False, stage='extract',
                                  text=f'Extracting {percent}%', current=percent, total=100)
        context = self._run_context
        if context is not None:
            for title_id in [t for t in context.active if self._run_outcomes.get(t) in ('succeeded', 'failed', 'cancelled')]:
                del context.active[title_id]
        if context is not None and len(context.active) > 1:
            return   # several titles are in flight: the count says it, and each row shows its own percentage
        self._say(f'Extracting {progress.title}  ({percent}%)')

    def _update_run_progress(self, context: Optional[_RunContext] = None) -> None:
        '''Show the viewed track on its title page, or completed-title count on the list.'''
        context = context or self._run_context
        if context is None:
            return
        if self._title_open and self._title_page is not None:
            title_id = self._title_page.current_id
            state = self._model.run_state(title_id)
            outcome = self._run_outcomes.get(title_id)
            self.runProgress.setRange(0, 100)
            if outcome in ('succeeded', 'failed', 'cancelled'):
                self.runProgress.setValue(100)
                self.runProgress.setFormat({'succeeded': 'Complete', 'failed': 'Failed',
                                            'cancelled': 'Cancelled'}[outcome])
            elif state.get('current') is not None and state.get('total'):
                percent = min(100, max(0, int(state['current'] * 100 / state['total'])))
                self.runProgress.setValue(percent)
                self.runProgress.setFormat(f'{percent}% of this track')
            else:
                self.runProgress.setValue(0)
                self.runProgress.setFormat(state.get('text') or ('Queued' if outcome == 'queued' else 'This track'))
            return
        total = max(1, len(context.request.ids))
        completed = sum(outcome in ('succeeded', 'failed', 'cancelled') for outcome in self._run_outcomes.values())
        self.runProgress.setRange(0, total)
        self.runProgress.setValue(min(completed, total))
        self.runProgress.setFormat(f'{min(completed, total)} / {len(context.request.ids)} titles')

    def _end_run(self) -> Optional[_RunContext]:
        context, self._job, self._run_context = self._run_context, None, None
        self._model.clear_active_run_states()
        self.cancelButton.setVisible(False)
        self.runProgress.setVisible(False)
        self._start_auto_publish()   # titles accepted while it went on, which had to wait for the repositories
        return context

    def _finish_attempts(self, context: Optional[_RunContext]) -> None:
        '''Reveal refreshed index details only after this run's result has been read.'''
        if context is not None:
            for title_id in context.request.ids:
                self._model.set_run_state(title_id, attempting=False, attempt_detail='', previous_failure={})
                dialog = self._detail_dialogs.get(title_id)
                if dialog is not None:
                    dialog.set_text(self._run_details_text(title_id))

    def _on_run_finished(self, source_job, report: StagesReport) -> None:
        if self._job is None or source_job is not self._job:
            return
        cancel_asked = self._job is not None and self._job.cancel_requested
        context = self._end_run()
        planned_ids = set(self._run_outcomes)
        cancelled_ids = set(report.not_run) & planned_ids
        failed_ids = set(dict(report.run.failed)) | set(dict(report.run.failed_earlier)) | \
            set(dict(report.run.unavailable))
        for title_id in report.attempted:
            if title_id not in planned_ids:
                continue
            self._run_outcomes[title_id] = 'failed' if title_id in failed_ids else 'succeeded'
        for result in report.published:
            if result.get('id') in planned_ids:
                self._run_outcomes[result['id']] = 'succeeded'
        for result in report.publish_errors:
            if result.get('id') in planned_ids:
                self._run_outcomes[result['id']] = 'failed'
        if report.commit_error:
            for title_id in context.commit_ids if context is not None else ():
                self._run_outcomes[title_id] = 'failed'
        for title_id in cancelled_ids:
            buffer = self._event_buffers.setdefault(title_id, EventBuffer())
            cancelled_event = ExecutionEvent(self._active_run_id, title_id, '', 'cancelled', time.time(),
                                             f'Not started: {report.stopped}' if report.stopped
                                             else 'Cancelled before dispatch')
            buffer.append(cancelled_event)
            self._run_outcomes[title_id] = 'cancelled'
            self._model.set_run_state(title_id, active=False, queued=False, stage='',
                                      text='Not run' if report.stopped else 'Cancelled', has_details=True)
            dialog = self._detail_dialogs.get(title_id)
            if dialog is not None:
                dialog.set_text(self._run_details_text(title_id))
        # The report is authoritative for the final aggregate, including
        # failures and titles cancelled before dispatch.
        for title_id in self._run_outcomes:
            if title_id in cancelled_ids:
                self._run_outcomes[title_id] = 'cancelled'
            elif title_id in failed_ids or any(item.get('id') == title_id for item in report.publish_errors) or \
                    (report.commit_error and context is not None and title_id in context.commit_ids):
                self._run_outcomes[title_id] = 'failed'
            elif title_id in report.attempted or any(item.get('id') == title_id for item in report.published):
                self._run_outcomes[title_id] = 'succeeded'
        for stage, ids in (('extract', report.run.cached), ('design', report.run.design_cached)):
            for title_id in ids:
                if title_id in planned_ids:
                    buffer = self._event_buffers.setdefault(title_id, EventBuffer())
                    buffer.append(ExecutionEvent(self._active_run_id, title_id, stage, 'cache_hit', time.time(),
                                                 f'{stage.capitalize()} cache hit: reused cached result; no command ran.'))
                    self._model.set_run_state(title_id, has_details=True)
        self._remember_run_details()
        self._update_run_progress(context)
        self._update_run_summary()
        self.refresh_from_index()
        self._finish_attempts(context)
        self._sync_index_if_dirty()   # decisions made while it ran may be newer than what it read
        if context is not None:
            self._results = describe_results(report, context.plan, self._setup.settings, context.rows)
            text, level = summarise_report(report, context.plan, self._setup.settings)
            if cancel_asked and not report.cancelled:   # the last title was already in hand, or a commit had begun
                text = f'Cancel requested too late: nothing was left to stop. {text}'
            if context.skipped_text:
                text += f'. {context.skipped_text}'
            self._refresh_results()
            self._say(text, level)
        self._requeue_unjoined(report.not_joined, dropped=cancel_asked)
        self._refresh_view()
        self.run_finished.emit(report)
        self._start_next_queued_run()

    def _on_run_failed(self, source_job, message: str) -> None:
        if self._job is None or source_job is not self._job:
            return
        context = self._end_run()
        self._remember_run_details()
        self.refresh_from_index()   # the pipeline refreshes the index whatever happened, so show what it now says
        self._finish_attempts(context)
        self._sync_index_if_dirty()
        self._say(f'The run failed: {message}. Titles finished before it are kept; see Help > Logs for the details.',
                   LEVEL_ERROR)
        # what joined it may or may not have been done: offered again, a title already done is planned out
        self._requeue_unjoined(list(self._offered_joins), dropped=False)
        self._refresh_view()
        self.run_failed.emit(message)
        self._start_next_queued_run()

    def _save_run_details(self) -> None:
        try:
            save_run_details(self._setup.profile.work_dir if self._setup.profile else None, self._saved_details)
        except OSError:
            logger.exception('Could not save the last run Details')

    def _remember_run_details(self) -> None:
        self._saved_details = {title_id: buffer.text() for title_id, buffer in self._event_buffers.items()}
        self._save_run_details()
