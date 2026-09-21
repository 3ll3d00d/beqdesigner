'''
The work list's bulk actions -- design/library-sync/workflow-rework/design.md §12.6, §12.8, §12.9, chunk 27c. A mixin of
`model.worklist.WorkListWindow`:

* **Accept top pick** -- `pipeline.library.bulk`: the titles waiting for review whose top pick is at or above the threshold
  (`WORKLIST_ACCEPT_THRESHOLD`, Settings > Locations, default 0.90), on the rows selected -- or, with none selected, the rows
  the filters list, like the other buttons. **Nothing is accepted without the confirmation**, which says how many, at what
  confidence, and lists every title that was left out and why (incomplete metadata, a decline, an edited project); the
  titles it accepts get the reviewer note "bulk accepted, confidence >= 0.90". Working out the plan, and doing it, are two
  jobs on the thread pool with a connection of their own to the index, and the index reads the outputs again at the end.
* **Revise...** -- on the selected rows (only: it is not offered for "everything listed"): `model.worklist_revise`. The
  question says what will happen before it does anything.
* **The "settings changed" banner** -- accepted and published titles are never redesigned by a run, and a change of designer,
  analysis or coverage deliberately does not put them back in the list (it would put every done title there at once), so the
  banner says how many were designed under other settings (`pipeline.library.drift`) and offers *Revise...* over exactly
  those, starting on *Redesign*. *Dismiss* hides it until the set of titles changes. It is worked out on a worker after the
  index is read, and an answer that arrives after a newer question is dropped.

Everything here refuses while a run, a scan or another such job is going (`_busy()`): they all write to the same outputs.
'''
import logging
from typing import Callable, Dict, List, Optional, Tuple

from qtpy.QtCore import QObject, QRunnable, QThreadPool, Signal
from qtpy.QtWidgets import QDialog

from model.preferences import WORKLIST_ACCEPT_THRESHOLD
from model.worklist_confirm import ConfirmDialog, accept_text, drift_text
from model.worklist_model import warning_colour
from model.worklist_revise import ReviseContext, ReviseDialog, ReviseSummary, describe_outcome, revise_context, \
    revise_problem, revise_titles, summarise_outcome
from model.worklist_run import LEVEL_ERROR, LEVEL_OK, LEVEL_WARN, ResultLine
from pipeline.library.bulk import AcceptPlan, AcceptReport, accept_top_pick, plan_accept
from pipeline.library.drift import designed_under_other_settings
from pipeline.library.index import LibraryIndex, TitleRow
from pipeline.library.selection import Selection

logger = logging.getLogger('worklist')

AskRevise = Callable[[ReviseSummary, Optional[ReviseContext], str], Optional[Tuple[str, str]]]


class _IndexSignals(QObject):
    finished = Signal(object)
    errored = Signal(str)


class IndexJob(QRunnable):
    '''
    Runs `work(index)` on a worker thread with a connection of its own to the index file (as `_ScanJob` and `_SyncJob` do), so
    the UI thread is never held up behind it. The result, or the error, comes back through `signals`, which the caller keeps
    a reference to.
    '''

    def __init__(self, path: str, work: Callable[[LibraryIndex], object], what: str):
        super().__init__()
        self.signals = _IndexSignals()
        self._path, self._work, self._what = path, work, what

    def run(self):
        try:
            with LibraryIndex(self._path) as index:
                result = self._work(index)
            self.signals.finished.emit(result)
        except Exception as error:
            logger.exception('%s failed', self._what)
            self.signals.errored.emit(f'{type(error).__name__}: {error}')


def _title(rows: Dict[str, TitleRow], title_id: str) -> str:
    row = rows.get(title_id)
    return (row.title or row.display_name or title_id) if row is not None else title_id


class WorkListBulk:
    '''
    The mixin: it uses `_setup`, `_index`, `_model`, `_preferences`, `_busy()`, `_say()`, `_refresh_actions()`,
    `_refresh_results()`, `show_last_run()`, `refresh_from_index()`, `target_ids()`, `selected_ids()`, `_rows_by_id()`,
    `_results`, `_closed` and the widgets `acceptButton`, `reviseButton`, `driftBanner`, `driftBannerLabel`,
    `driftBannerButton` and `driftDismissButton` of the window.
    '''

    def _init_bulk(self, ask_revise: Optional[AskRevise]) -> None:
        ''' The state, before anything can refresh the buttons (the window calls this first). '''
        self._ask_revise = ask_revise
        self._bulk_job: Optional[IndexJob] = None
        self._bulk_kind = ''
        self._bulk_context = None
        self._drift_ids: List[str] = []
        self._drift_dismissed: frozenset = frozenset()
        self._drift_token = 0
        self._drift_jobs: Dict[int, IndexJob] = {}

    def _configure_bulk(self) -> None:
        self.acceptButton.clicked.connect(lambda: self.accept_selected())
        self.reviseButton.clicked.connect(lambda: self.revise_selected())
        self.driftBannerButton.clicked.connect(lambda: self.revise_drifted())
        self.driftDismissButton.clicked.connect(lambda: self.dismiss_drift())
        self.driftBanner.setObjectName('driftBanner')
        self.driftBanner.setStyleSheet(f'QFrame#driftBanner {{ border: 1px solid {warning_colour().name()}; '
                                       f'border-radius: 4px; }}')
        self.driftBanner.setVisible(False)

    @property
    def bulk_running(self) -> bool:
        ''' Whether a bulk accept or revise (or the check before one) is going. '''
        return self._bulk_job is not None

    @property
    def drift_ids(self) -> List[str]:
        ''' The titles the "settings changed" banner is about. '''
        return list(self._drift_ids)

    def accept_threshold(self) -> float:
        return float(self._preferences.get(WORKLIST_ACCEPT_THRESHOLD))

    # --- the buttons ---------------------------------------------------------------------------------------------------------

    def _refresh_bulk_actions(self) -> None:
        ''' The two buttons' labels and whether they are offered; called whenever the other buttons refresh. '''
        rows, threshold = self._rows_by_id(), self.accept_threshold()
        ready = self._setup.ready and self._index is not None and not self._busy()
        waiting = [rows[i] for i in self.target_ids() if i in rows and rows[i].needs == 'review']
        eligible = [r for r in waiting if r.confidence is not None and r.confidence >= threshold]
        self.acceptButton.setText(f'Accept top pick ({len(eligible):,})')
        self.acceptButton.setEnabled(ready and bool(eligible))
        self.acceptButton.setToolTip(
            f'Accept the designer\'s top pick for the titles waiting for review whose confidence is {threshold:.2f} or more '
            f'({len(waiting):,} waiting in {"the selection" if self.selected_ids() else "the view"}). You are shown what '
            f'will happen, and which titles are left out and why, before anything is accepted. The threshold is in Settings.')
        selected = self.selected_ids()
        self.reviseButton.setText(f'Revise ({len(selected):,})...' if selected else 'Revise...')
        self.reviseButton.setEnabled(ready and bool(selected))
        self.reviseButton.setToolTip('Send the selected titles back: reopen them for review, redesign them or extract them '
                                     'again. Select the titles first; it is not offered for everything listed.')
        self.driftBannerButton.setEnabled(ready)
        self.driftDismissButton.setEnabled(True)

    def _start_bulk(self, kind: str, work: Callable[[LibraryIndex], object], message: str, context=None) -> bool:
        job = IndexJob(self._setup.index_file, work, kind)
        job.signals.finished.connect(self._on_bulk_finished)
        job.signals.errored.connect(self._on_bulk_failed)
        self._bulk_job, self._bulk_kind, self._bulk_context = job, kind, context
        self._say(message)
        self._refresh_actions()
        QThreadPool.globalInstance().start(job)
        return True

    def _can_start(self) -> bool:
        self._flush_settings()
        return not self._busy() and self._setup.ready and self._index is not None and self._setup.index_file is not None

    def _end_bulk(self):
        kind, context = self._bulk_kind, self._bulk_context
        self._bulk_job, self._bulk_kind, self._bulk_context = None, '', None
        return kind, context

    def _on_bulk_finished(self, result) -> None:
        kind, context = self._end_bulk()
        self._refresh_actions()
        if kind == 'plan':
            self._confirm_accept(result, context)
        elif kind == 'accept':
            self._show_accepted(result, context)
        elif kind == 'revise':
            self._show_revised(result, context)
        self._sync_index_if_dirty()   # decisions made on the title page while the job ran are not in what it read

    def _on_bulk_failed(self, message: str) -> None:
        kind, _ = self._end_bulk()
        self.refresh_from_index()
        self._say(f'{"Accepting" if kind in ("plan", "accept") else "Revising"} failed: {message}. See Help > Logs for the '
                  f'details; run the check again with Rescan if the list looks wrong.', LEVEL_ERROR)
        self._sync_index_if_dirty()

    # --- bulk accept -----------------------------------------------------------------------------------------------------------

    def accept_selected(self, ids: Optional[List[str]] = None) -> bool:
        '''
        Works out what accepting the top pick of the selected titles (default: `target_ids()`) would do, asks, and does it.
        :return: False if nothing was started -- something else is going, the setup is incomplete, nothing is eligible, or the
            person declined. Nothing is accepted without the confirmation.
        '''
        if not self._can_start():
            return False
        settings = self._setup.settings
        chosen = tuple(self.target_ids() if ids is None else ids)
        if not chosen or not settings.queue_dir:   # (an empty Selection.ids would mean every title)
            self._say('No review queue directory is set (Settings > Locations).' if chosen else 'No titles to accept.',
                      LEVEL_WARN)
            return False
        threshold = self.accept_threshold()
        selection = Selection(ids=chosen)

        def plan(index: LibraryIndex) -> AcceptPlan:
            return plan_accept(index, selection, threshold, queue_dir=settings.queue_dir,
                               meta_defaults=settings.meta_defaults, work_dir=settings.work_dir or None)

        return self._start_bulk('plan', plan, 'Checking which titles can be accepted...', threshold)

    def _confirm_accept(self, plan: AcceptPlan, threshold: float) -> None:
        if self._closed:   # the window was closed while the plan was worked out: nothing to ask, and nobody to ask
            return
        count = len(plan.eligible)
        if not count:
            reasons = f' {len(plan.excluded):,} at or above the threshold need a person to look.' if plan.excluded else ''
            self._say(f'Nothing to accept: no title waiting for review has a top pick of {threshold:.2f} or more '
                      f'that is ready to accept.{reasons}', LEVEL_WARN)
            if plan.excluded:
                self._results = [ResultLine(e.id, e.title, 'Not accepted', e.reason, LEVEL_WARN) for e in plan.excluded]
                self._refresh_results()
                self.show_last_run()
            return
        heading, body, details = accept_text(plan)
        dialog = ConfirmDialog(self, heading, body, f'Accept {count:,} title{"" if count == 1 else "s"}', details=details)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            self._say('Cancelled: nothing was accepted.')
            return
        if not self._can_start():   # a scan or a run began while the question was open
            self._say('Not accepted: something else started while you were deciding. Try again when it has finished.',
                      LEVEL_WARN)
            return
        settings, setup = self._setup.settings, self._setup
        selection = Selection(ids=tuple(plan.eligible))    # exactly what was confirmed

        def accept(index: LibraryIndex):
            report = accept_top_pick(index, selection, threshold, queue_dir=settings.queue_dir,
                                     meta_defaults=settings.meta_defaults, work_dir=settings.work_dir or None)
            error = ''
            try:
                index.refresh(setup.profile, settings)   # what the titles need now: the pipeline reads the outputs
            except Exception as failure:
                logger.exception('Could not update the library index after accepting')
                error = f'{type(failure).__name__}: {failure}'
            return report, error

        self._start_bulk('accept', accept, f'Accepting {count:,} title{"" if count == 1 else "s"}...', plan)

    def _show_accepted(self, result: Tuple[AcceptReport, str], plan: AcceptPlan) -> None:
        '''
        :param plan: what the person confirmed. The second job **plans again** (`accept_top_pick` judges every title afresh, from
            the entries as they are then), so it accepts a *subset* of the confirmed titles, never more; the titles it does not
            accept are accounted for here, each with why, so a confirmed title never just goes missing:

            * left out **before** the confirmation (`plan.excluded`: a decline, incomplete metadata, an edited project);
            * excluded **by the second plan or the write** (`report.excluded`: it changed after the person confirmed);
            * **dropped by the second plan without a word** (`report.not_for_review` / `below_threshold`, which are counts: the
              index no longer said it was waiting for review, or its top pick fell below the threshold).
        '''
        report, error = result
        rows = self._rows_by_id()
        self.refresh_from_index()
        confirmed = list(plan.eligible)
        accepted, excluded_now = set(report.accepted), {e.id for e in report.excluded}
        dropped = [i for i in confirmed if i not in accepted and i not in excluded_now]
        left_out = list(plan.excluded) + [e for e in report.excluded if e.id not in {p.id for p in plan.excluded}]
        lines = [ResultLine(e.id, e.title, 'Not accepted', e.reason, LEVEL_WARN) for e in left_out]
        lines += [ResultLine(i, _title(rows, i), 'Not accepted', 'it was no longer waiting for review, or no longer confident '
                             'enough, when it came to accepting it: look at it again', LEVEL_WARN) for i in dropped]
        lines += [ResultLine(i, _title(rows, i), 'Accepted', report.note) for i in report.accepted]
        self._results = lines
        self._refresh_results()
        self.show_last_run()
        count = len(report.accepted)
        text = f'Accepted the top pick for {count:,} title{"" if count == 1 else "s"} (confidence {report.threshold:.2f} or more).'
        if plan.excluded:
            text += f' {len(plan.excluded):,} left out: see the Last run tab.'
        not_accepted = len(confirmed) - count
        if not_accepted:
            changed = len([i for i in confirmed if i in excluded_now])
            reasons = []
            if changed:
                reasons.append(f'{changed:,} changed after you confirmed')
            waiting = min(report.not_for_review, len(dropped))
            if waiting:
                reasons.append(f'{waiting:,} no longer waiting for review')
            fell = min(report.below_threshold, len(dropped) - waiting)
            if fell:
                reasons.append(f'{fell:,} below the threshold now')
            rest = len(dropped) - waiting - fell
            if rest:
                reasons.append(f'{rest:,} no longer in the list')
            text += f' {not_accepted:,} of the {len(confirmed):,} you confirmed {"was" if not_accepted == 1 else "were"} not ' \
                    f'accepted ({"; ".join(reasons)}): see the Last run tab.'
        text += ' They now need Publish.' if count else ''
        if error:
            text += f' The list could not be brought up to date ({error}): Rescan.'
        self._say(text, LEVEL_WARN if plan.excluded or not_accepted or error else LEVEL_OK)

    # --- revise --------------------------------------------------------------------------------------------------------------

    def revise_selected(self) -> bool:
        ''' Revise... on the selected rows. '''
        return self.revise_ids(self.selected_ids())

    def revise_drifted(self) -> bool:
        ''' The banner's Revise...: the titles designed under other settings, starting on Redesign. '''
        return self.revise_ids(self._drift_ids, default='design')

    def revise_ids(self, ids: List[str], default: str = 'review', to: Optional[str] = None, reason: str = '') -> bool:
        '''
        Asks what to do with these titles and sends them back (`model.worklist_revise`), on a worker; the index then reads
        the outputs again. **The question says what will happen before anything changes.**
        :param to: answer the question for the caller (a script or a test); the dialog is not shown.
        :return: False if nothing was started -- something else is going, nothing to revise, it cannot be done (the reason is
            in the status line) or the person cancelled.
        '''
        if not ids or not self._can_start():
            return False
        rows = self._rows_by_id()
        context = revise_context(self._setup)
        summary = ReviseSummary.from_rows([rows[i] for i in ids if i in rows])
        if to is None:
            answer = (self._ask_revise or self._show_revise_dialog)(summary, context, default)
            if answer is None:
                self._say('Cancelled: nothing was sent back.')
                return False
            to, reason = answer
        problem = revise_problem(to, summary, context)
        if problem:
            self._say(f'Not changed: {problem}', LEVEL_ERROR)
            return False
        setup = self._setup
        chosen = list(ids)

        def revise(index: LibraryIndex):
            outcome = revise_titles(context, chosen, to, reason)
            error = ''
            try:
                index.refresh(setup.profile, setup.settings)
            except Exception as failure:
                logger.exception('Could not update the library index after revising')
                error = f'{type(failure).__name__}: {failure}'
            return outcome, error

        return self._start_bulk('revise', revise, f'Sending {len(chosen):,} title{"" if len(chosen) == 1 else "s"} back...',
                                {row.id: _title(rows, row.id) for row in (rows.get(i) for i in chosen) if row is not None})

    def _show_revise_dialog(self, summary: ReviseSummary, context: Optional[ReviseContext], default: str
                            ) -> Optional[Tuple[str, str]]:
        dialog = ReviseDialog(self, summary, context, default)
        return (dialog.choice, dialog.reason) if dialog.exec() else None

    def _show_revised(self, result, titles: Dict[str, str]) -> None:
        outcome, error = result
        self.refresh_from_index()
        self._results = describe_outcome(outcome, titles)
        self._refresh_results()
        self.show_last_run()
        text, level = summarise_outcome(outcome)
        if error:
            text += f' The list could not be brought up to date ({error}): Rescan.'
            level = LEVEL_WARN if level == LEVEL_OK else level
        self._say(text, level)

    # --- the "settings changed" banner ---------------------------------------------------------------------------------------

    def dismiss_drift(self) -> None:
        ''' Hides the banner until the set of titles it is about changes. '''
        self._drift_dismissed = frozenset(self._drift_ids)
        self._refresh_drift_banner()

    def _refresh_drift_banner(self) -> None:
        shown = bool(self._drift_ids) and frozenset(self._drift_ids) != self._drift_dismissed
        self.driftBanner.setVisible(shown)
        if shown:
            self.driftBannerLabel.setText(drift_text(len(self._drift_ids)))

    def _refresh_drift(self) -> None:
        '''
        Asks (on a worker) which accepted or published titles were designed under other settings than the ones now in force,
        and shows the banner if there are any. Called whenever the rows are read; an answer to an earlier question, that
        arrives after a newer one, is dropped.
        '''
        self._drift_token += 1
        token, setup, rows = self._drift_token, self._setup, list(self._model.rows)
        protected = any(r.review_state == 'accepted' and r.design_state == 'protected' for r in rows)
        if self._closed or not protected or not setup.ready or setup.index_file is None or setup.settings is None:
            self._set_drift([])
            return
        settings = setup.settings

        def work(index: LibraryIndex):
            return token, designed_under_other_settings(index, settings, rows)

        job = IndexJob(setup.index_file, work, 'drift')
        self._drift_jobs[token] = job
        job.signals.finished.connect(self._on_drift)
        job.signals.errored.connect(self._on_drift_failed)
        QThreadPool.globalInstance().start(job)

    def _on_drift(self, result) -> None:
        token, ids = result
        self._drift_jobs.pop(token, None)
        if token != self._drift_token or self._closed:   # a newer question was asked: this answer is about old rows
            return
        self._set_drift(ids)

    def _on_drift_failed(self, message: str) -> None:
        ''' The check is a courtesy: a failure is logged (by the job) and the banner keeps what it had. '''
        for token in [t for t in self._drift_jobs if t < self._drift_token]:   # (the failed one may be the newest: it stays
            self._drift_jobs.pop(token, None)                                   # referenced until the next question)

    def _set_drift(self, ids: List[str]) -> None:
        self._drift_ids = list(ids)
        self._refresh_drift_banner()
