'''
The work list's actions -- design/library-sync/workflow-rework/design.md §12.7 and §12.10, chunk 26b. A mixin of
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
from dataclasses import dataclass
from typing import Dict, List, Optional

from qtpy.QtCore import Qt, QThreadPool
from qtpy.QtGui import QBrush
from qtpy.QtWidgets import QAbstractItemView, QDialog, QLabel, QPushButton, QTableWidgetItem

from model.preferences import WORKLIST_PUSH
from model.execution_events import ExecutionEvent
from model.worklist_confirm import ConfirmDialog, commit_text, machine_text, publish_text, retry_text
from model.worklist_model import warning_colour
from model.worklist_run import LEVEL_ERROR, LEVEL_OK, FailedTitle, ResultLine, RunJob, RunRequest, \
    build_publish_settings, build_run_config, describe_results, plan_label, publish_problem, summarise_report, \
    summarise_skipped
from model.worklist_run_details import EventBuffer
from pipeline.library.index import TitleRow
from pipeline.library.selection import StagePlan, plan_stages
from pipeline.library.stages import FfmpegProgress, Progress, StagesReport

logger = logging.getLogger('worklist')


def _result_tooltip(line: ResultLine) -> str:
    ''' A results row's tooltip: the title, the outcome and the whole detail (a cell shows one line of it). '''
    detail = line.full or line.detail
    heading = line.title or line.outcome
    return f'{heading}\n{line.outcome}: {detail}' if detail else f'{heading}\n{line.outcome}'


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


class WorkListActions:
    '''
    The mixin: it uses what `WorkListWindow` sets up (`_model`, `_proxy`, `_setup`, `_index`, `_preferences`, `_job`,
    `_run_context`, `_failed`, `_results`, `_precheck`, `_run_stages`, `_scanning`, the widgets and `_refresh_view()`).
    '''

    def _configure_actions(self) -> None:
        for table, columns in ((self.failuresTable, ('Title', 'Source', 'Stage', 'Reason')),
                               (self.resultsTable, ('Title', 'Result', 'Detail'))):
            table.setColumnCount(len(columns))
            table.setHorizontalHeaderLabels(list(columns))
            table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            table.setAlternatingRowColors(True)
            table.setShowGrid(False)
            table.setWordWrap(False)
            table.setTextElideMode(Qt.TextElideMode.ElideRight)
            table.verticalHeader().setVisible(False)
            header = table.horizontalHeader()
            header.setStretchLastSection(True)
            header.resizeSection(0, 260)
        self.failuresTable.horizontalHeader().resizeSection(1, 90)
        self.failuresTable.horizontalHeader().resizeSection(2, 70)
        self.resultsTable.horizontalHeader().resizeSection(1, 200)
        self.workTable.selectionModel().selectionChanged.connect(self._refresh_actions)
        self.failuresTable.itemSelectionChanged.connect(self._refresh_actions)
        self.resultsTable.itemSelectionChanged.connect(self._refresh_result_details)
        self.resultDetails.setVisible(False)
        self.runLayout.setStretchFactor(self.runStatusLabel, 1)
        self.runCountsLabel = QLabel(self.runPanel)
        self.runCountsLabel.setObjectName('runCountsLabel')
        self.runCountsLabel.setTextFormat(Qt.TextFormat.PlainText)
        self.runLayout.insertWidget(1, self.runCountsLabel)
        self.runButton.clicked.connect(lambda: self.run_selected())
        self.publishButton.clicked.connect(lambda: self.publish_selected())
        self.commitButton.clicked.connect(lambda: self.commit_selected())
        self.retryButton.clicked.connect(lambda: self.retry_failed())
        self.retryFromResultsButton = QPushButton('Retry failed titles')
        self.retryFromResultsButton.setToolTip('Retry titles that failed in this run.')
        self.retryFromResultsButton.clicked.connect(lambda: self.retry_failed())
        self.resultsLayout.addWidget(self.retryFromResultsButton)
        self.cancelButton.clicked.connect(self.cancel_run)
        self.selectAllButton.clicked.connect(self.workTable.selectAll)
        self.clearSelectionButton.clicked.connect(self.workTable.clearSelection)
        self.detailsTabs.setTabVisible(0, False)
        self.detailsTabs.setTabVisible(1, False)
        self.detailsTabs.setVisible(False)
        self.bodySplitter.setStretchFactor(0, 1)
        self.bodySplitter.setStretchFactor(1, 0)
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
        extracting is never published by pressing Publish.
        '''
        rows = self._rows_by_id()
        targets = [rows[i] for i in (self.target_ids() if ids is None else ids) if i in rows]
        if through in ('publish', 'commit'):
            targets = [row for row in targets if row.needs == through]
        return plan_stages(targets, through, retry_failed=retry_failed)

    def _busy(self) -> bool:
        return self.is_running or self._scanning or self._syncing or self._bulk_job is not None

    def _refresh_actions(self, *_) -> None:
        ''' The selection text and the four buttons: their labels say how many titles each will work on. '''
        rows = self._rows_by_id()
        selected, listed = self.selected_ids(), self._proxy.rowCount()
        self.selectionLabel.setText(f'{len(selected):,} selected of {listed:,} listed' if selected else
                                    f'None selected: the buttons work on all {listed:,} listed' if listed else '')
        ready = self._setup.ready and self._index is not None and not self._busy()
        plan = self.plan_for('design')
        self.runButton.setText(_button_text(plan_label(plan)))
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
            button.setText(_button_text(text))
            button.setEnabled(ready and bool(step.planned) and not problem)
            if problem:
                button.setToolTip(problem)
            elif through == 'publish':
                button.setToolTip('Write the accepted titles (and those published but out of date) into the repositories\' '
                                  'working trees. Nothing is committed or pushed.')
            else:
                button.setToolTip('Commit the written titles: one commit per repository, images first, then push. '
                                  'Titles committed earlier without a push are only pushed.')
        failed = self._selected_failed_ids() or [f.id for f in self._failed]
        self.retryButton.setText(f'Retry {len(failed):,} selected' if self._selected_failed_ids()
                                 else f'Retry {len(failed):,} failed')
        self.retryButton.setEnabled(ready and bool(failed))
        self.retryButton.setToolTip('Run the titles in the failures panel again, even though nothing changed. The panel '
                                    'lists every failed title in the library, not only those in the current view.')
        self.retryFromResultsButton.setVisible(bool(self._failed))
        self.retryFromResultsButton.setEnabled(ready and bool(self._failed))
        self.rescanButton.setEnabled(self._setup.ready and not self._busy())
        self._refresh_open_button()
        self._refresh_bulk_actions()
        self._refresh_settings_state()

    @staticmethod
    def _uncommitted(plan: StagePlan) -> int:
        ''' How many of a commit plan's titles still have to be committed (the others only need pushing). '''
        return sum(1 for p in plan.planned if p.row.commit_state != 'committed')

    def _selected_failed_ids(self) -> List[str]:
        rows = sorted({index.row() for index in self.failuresTable.selectionModel().selectedRows()})
        return [self._failed[row].id for row in rows if row < len(self._failed)]

    def _refresh_failures(self) -> None:
        table = self.failuresTable
        table.setRowCount(len(self._failed))
        for row, failure in enumerate(self._failed):
            for column, text in enumerate((failure.title, failure.source, failure.stage, failure.reason)):
                item = QTableWidgetItem(text)
                item.setToolTip(f'{failure.title} ({failure.id})\n{failure.reason}' if column in (0, 3) else '')
                if column == 3:
                    item.setForeground(QBrush(warning_colour()))
                table.setItem(row, column, item)
        self.detailsTabs.setTabText(0, f'Failures ({len(self._failed):,})')
        self._refresh_details()

    def _refresh_results(self) -> None:
        table = self.resultsTable
        table.setRowCount(len(self._results))
        for row, line in enumerate(self._results):
            for column, text in enumerate((line.title, line.outcome, line.detail)):
                item = QTableWidgetItem(text)
                item.setToolTip(_result_tooltip(line))
                if line.level != LEVEL_OK and column >= 1:
                    item.setForeground(QBrush(warning_colour()))
                    if line.level == LEVEL_ERROR and column == 1:
                        font = item.font()
                        font.setBold(True)
                        item.setFont(font)
                table.setItem(row, column, item)
        self.detailsTabs.setTabText(1, f'Last run ({len(self._results):,})')
        self._refresh_result_details()
        self._refresh_details()

    def _refresh_result_details(self) -> None:
        '''
        The *Details* area under the results: the whole text of the selected line if it was shortened for its cell, else
        -- with nothing selected -- the whole text of every repository-level problem, each once.
        '''
        selected = sorted({index.row() for index in self.resultsTable.selectionModel().selectedRows()})
        lines = [self._results[r] for r in selected if r < len(self._results)]
        if not lines:
            lines = [line for line in self._results if not line.id and line.full]
        text = '\n\n'.join(f'{line.title or line.outcome}: {line.full}' for line in lines if line.full)
        self.resultDetails.setPlainText(text)
        self.resultDetails.setVisible(bool(text))

    def _refresh_details(self) -> None:
        if self._title_open:   # the title page has the window: leaving it shows the panel again
            return
        self.detailsTabs.setTabVisible(0, bool(self._failed))
        self.detailsTabs.setTabVisible(1, bool(self._results))
        show = bool(self._failed or self._results)
        if show and not self.detailsTabs.isVisibleTo(self):
            self.detailsTabs.setVisible(True)
            total = self.bodySplitter.height()
            self.bodySplitter.setSizes([max(total - 280, 200), 280])
        elif not show:
            self.detailsTabs.setVisible(False)

    def show_last_run(self) -> None:
        ''' Brings the *Last run* tab (what the last run did to each title) to the front. '''
        self.detailsTabs.setCurrentWidget(self.resultsTab)

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
        ''' Runs the failed titles again (those selected in the failures panel, else every one) even though nothing changed. '''
        chosen = ids if ids is not None else self._selected_failed_ids()
        wanted = chosen or [f.id for f in self._failed]
        # the failures panel is the whole library's: retrying all of it is as big as running a whole view, so it asks
        return self._begin('design', retry_failed=True, ids=list(wanted), confirm=not chosen)

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
        if self._busy() or not self._setup.ready or self._index is None or self._setup.index_file is None:
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
            settings = build_publish_settings(self._setup)
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
        return self._launch(request, plan, rows, skipped)

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
            publish = build_publish_settings(setup, request.push) if request.through in ('publish', 'commit') else None
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
        self._active_run_id = ''
        self._run_outcomes = {title_id: 'queued' for title_id in request.ids}
        for row in self._model.rows:
            state = self._model.run_state(row.id)
            if state.get('has_details'):
                self._model.set_run_state(row.id, has_details=False)
        for title_id in request.ids:
            self._model.set_run_state(title_id, active=False, queued=True, stage='', text='Queued', current=None,
                                      total=None, has_details=False)
        self._run_context = _RunContext(request, plan, rows, skipped_text,
                                         [p.row.id for p in plan.with_stage('commit')])
        self.cancelButton.setVisible(True)
        self.cancelButton.setEnabled(True)
        self.runProgress.setVisible(True)
        self.runProgress.setRange(0, max(1, len(plan.planned)))
        self.runProgress.setValue(0)
        self.runProgress.setFormat(f'0 / {len(plan.planned)} titles')
        self._say(f'Starting: {plan_label(plan)}...')
        self._update_run_summary()
        self._refresh_actions()
        self._refresh_view()
        try:
            QThreadPool.globalInstance().start(job)
        except Exception as error:   # the run never began: do not leave the window "running" for ever
            logger.exception('Could not start the run')
            self._end_run()
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
        self.cancelButton.setEnabled(False)
        if self._run_context is not None and self._run_context.stage == 'commit':
            self._say('Cancel requested, but a commit cannot be stopped part way: it finishes, and the run ends after it.')
        else:
            self._say('Cancelling: the title being worked on finishes first...')
        return True

    def _on_run_progress(self, source_job, progress: Progress) -> None:
        if self._job is None or source_job is not self._job:
            return
        context = self._run_context
        if context is None:
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
        else:
            if progress.id:
                self._model.set_run_state(progress.id, active=True, queued=False, stage=progress.stage, text=progress.stage)
            word = {'extract': 'Extracting', 'design': 'Designing', 'publish': 'Publishing'}.get(progress.stage,
                                                                                                 progress.stage)
            text = f'{word} {progress.title}'
        text += f'  ({min(progress.done + 1, progress.total):,} of {progress.total:,})'
        if self._job is not None and self._job.cancel_requested:
            text += ' -- a commit cannot be stopped, it finishes' if progress.stage == 'commit' \
                else ' -- cancelling after this one'
        self._say(text)

    def _on_ffmpeg_progress(self, progress: FfmpegProgress) -> None:
        '''Keep ffmpeg's ``out_time_ms`` percentage in the title's own row.'''
        if progress.total_micros <= 0:
            return
        percent = min(100, max(0, int(progress.out_time_micros * 100 / progress.total_micros)))
        self._model.set_run_state(progress.id, active=True, queued=False, stage='extract',
                                  text=f'Extracting {percent}%', current=percent, total=100)
        self._say(f'Extracting {progress.title}  ({percent}%)')

    def _update_run_progress(self, context: Optional[_RunContext] = None) -> None:
        '''Count unique titles with terminal outcomes; stages and ffmpeg packets do not advance this bar.'''
        context = context or self._run_context
        if context is None:
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
        return context

    def _on_run_finished(self, source_job, report: StagesReport) -> None:
        if self._job is None or source_job is not self._job:
            return
        cancel_asked = self._job is not None and self._job.cancel_requested
        context = self._end_run()
        cancelled_ids = set(report.not_run)
        failed_ids = set(dict(report.run.failed)) | set(dict(report.run.failed_earlier))
        for title_id in report.attempted:
            self._run_outcomes[title_id] = 'failed' if title_id in failed_ids else 'succeeded'
        for result in report.published:
            self._run_outcomes[result['id']] = 'succeeded'
        for result in report.publish_errors:
            if result.get('id'):
                self._run_outcomes[result['id']] = 'failed'
        if report.commit_error:
            for title_id in context.commit_ids if context is not None else ():
                self._run_outcomes[title_id] = 'failed'
        for title_id in cancelled_ids:
            buffer = self._event_buffers.setdefault(title_id, EventBuffer())
            cancelled_event = ExecutionEvent(self._active_run_id, title_id, '', 'cancelled', time.time(),
                                             'Cancelled before dispatch')
            buffer.append(cancelled_event)
            self._run_outcomes[title_id] = 'cancelled'
            self._model.set_run_state(title_id, active=False, queued=False, stage='', text='Cancelled',
                                      has_details=True)
            dialog = self._detail_dialogs.get(title_id)
            if dialog is not None:
                dialog.set_text(buffer.text())
        # The report is authoritative for the final aggregate, including
        # failures and titles cancelled before dispatch.
        for title_id in self._run_outcomes:
            if title_id in cancelled_ids:
                self._run_outcomes[title_id] = 'cancelled'
            elif title_id in failed_ids or any(item.get('id') == title_id for item in report.publish_errors):
                self._run_outcomes[title_id] = 'failed'
            elif title_id in report.attempted or any(item.get('id') == title_id for item in report.published):
                self._run_outcomes[title_id] = 'succeeded'
        self._update_run_progress(context)
        self._update_run_summary()
        self.refresh_from_index()
        self._sync_index_if_dirty()   # decisions made while it ran may be newer than what it read
        if context is not None:
            self._results = describe_results(report, context.plan, self._setup.settings, context.rows)
            text, level = summarise_report(report, context.plan, self._setup.settings)
            if cancel_asked and not report.cancelled:   # the last title was already in hand, or a commit had begun
                text = f'Cancel requested too late: nothing was left to stop. {text}'
            if context.skipped_text:
                text += f'. {context.skipped_text}'
            self._refresh_results()
            self.show_last_run()
            self._say(text, level)
        self._refresh_view()
        self.run_finished.emit(report)

    def _on_run_failed(self, source_job, message: str) -> None:
        if self._job is None or source_job is not self._job:
            return
        self._end_run()
        self.refresh_from_index()   # the pipeline refreshes the index whatever happened, so show what it now says
        self._sync_index_if_dirty()
        self._say(f'The run failed: {message}. Titles finished before it are kept; see Help > Logs for the details.',
                   LEVEL_ERROR)
        self._refresh_view()
        self.run_failed.emit(message)
