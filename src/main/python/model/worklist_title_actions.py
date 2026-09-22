'''
What the title page can do to a title besides decide it -- design/library-sync/workflow-rework/design.md §12.10, chunk 27c:

* **Open project** (mono, multichannel): the title's `.beq` projects, opened in the main window through a callable the work
  list is given by `BeqDesigner` (this page does not import `app`: the two import each other at the top level, AGENTS.md
  gotcha 3). A button is enabled only while its file exists, and a badge says whether a project was **modified since
  design** (`model.worklist_projects`; its tooltip says what that means for the title). The badge is read when a title is
  shown, and again whenever the window becomes active -- a person edits in the main window, saves the project, and comes back.
* **Reopen / Revise...** (`model.worklist_revise`): send the title back for review, redesign or re-extraction. It changes
  state only (nothing is redesigned or published now), tells the person what will happen before it does anything, records
  their reason in the reviewer note, and shows what it did on the page. Nothing is revised while a run is working on the
  title (or, for an accepted or published one, while a publish or commit run is going: the repositories are in use).

`TitleActions` is a mixin of `model.worklist_title.TitlePage`, which owns the widgets, the entry and the signals
(`revised`); `TitleHooks` is everything it asks of the window.
'''
import json
import logging
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Tuple

from qtpy.QtCore import Signal
from qtpy.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from model.worklist_model import warning_colour
from model.worklist_projects import EDITED_TOOLTIP, LEVEL_WARN, MONO, MULTICHANNEL, ProjectState, badge, \
    project_states
from model.worklist_title_text import REDO_IN_WORK_LIST
from model.worklist_revise import ReviseContext, ReviseDialog, ReviseSummary, revise_problem, revise_titles, \
    summarise_outcome
from pipeline.review import QueueEntry, read_entry

logger = logging.getLogger('worklist')

AskRevise = Callable[[ReviseSummary, Optional[ReviseContext], str], Optional[Tuple[str, str]]]


@dataclass
class TitleHooks:
    '''
    What the page asks of the window that hosts it. All optional, so a page can be built without a window (the review folder).
    :param open_project: opens a project file in the main window; None when there is no main window to open it in.
    :param work_dir: where the projects are (asked each time: a setting may change while the page is open).
    :param revise_context: the queue directory, work directory and repositories a revise needs; None if there are none.
    :param revise_blocked: `(title id, entry status) -> why not`, or '' when it may be revised now.
    :param ask_revise: `(summary, context, default choice) -> (choice, reason)` or None if the person cancelled; the
        `ReviseDialog` unless given.
    :param redo: how a title sent back for redesign gets designed again, where the page is (`worklist_title_text.REDO_IN_*`).
    :param run_design: starts the Library Work List's Extract & design after a redesign or re-extraction.  The Review
        Folder deliberately leaves this unset because it has no library runner.
    :param commit_state: `title id -> the index's commit state` (`committed`, `uncommitted`, `unknown`, ...), asked when the
        revise question is put, for a page with no index row that could say (the review folder); None to use the row's.
    '''
    open_project: Optional[Callable[[str], Optional[bool]]] = None
    work_dir: Callable[[], str] = lambda: ''
    revise_context: Callable[[], Optional[ReviseContext]] = lambda: None
    revise_blocked: Callable[[str, str], str] = lambda title_id, status: ''
    ask_revise: Optional[AskRevise] = None
    redo: str = REDO_IN_WORK_LIST
    run_design: Optional[Callable[[str], bool]] = None
    commit_state: Optional[Callable[[str], str]] = None
    retry_failed: Optional[Callable[[str], bool]] = None
    open_jriver_preferences: Optional[Callable[[], None]] = None
    choose_audio_stream: Optional[Callable[[str], bool]] = None


class TitleActionsBar(QWidget):
    ''' The row of buttons and the badge: no logic, it reports clicks. '''
    open_requested = Signal(str)     # `mono` or `multichannel`
    revise_requested = Signal()
    retry_requested = Signal()
    jriver_preferences_requested = Signal()
    audio_stream_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self.projectsLabel = QLabel('Projects:')
        self.monoButton = QPushButton('Open mono project')
        self.multichannelButton = QPushButton('Open multichannel project')
        self.projectBadge = QLabel()
        self.projectBadge.setWordWrap(True)
        self.reviseButton = QPushButton('Reopen / Revise...')
        self.retryButton = QPushButton('Retry failed extraction')
        self.jriverPreferencesButton = QPushButton('Open JRiver path mappings')
        self.audioStreamButton = QPushButton('Choose audio stream...')
        for button in (self.monoButton, self.multichannelButton, self.reviseButton, self.retryButton,
                       self.jriverPreferencesButton, self.audioStreamButton):
            button.setAutoDefault(False)   # Enter in a field must never press one (design.md §12.1)
        self.monoButton.clicked.connect(lambda: self.open_requested.emit(MONO))
        self.multichannelButton.clicked.connect(lambda: self.open_requested.emit(MULTICHANNEL))
        self.reviseButton.clicked.connect(lambda: self.revise_requested.emit())
        self.retryButton.clicked.connect(self.retry_requested.emit)
        self.jriverPreferencesButton.clicked.connect(self.jriver_preferences_requested.emit)
        self.audioStreamButton.clicked.connect(self.audio_stream_requested.emit)
        for widget in (self.projectsLabel, self.monoButton, self.multichannelButton):
            row.addWidget(widget)
        row.addWidget(self.projectBadge, 1)
        row.addWidget(self.reviseButton)
        row.addWidget(self.retryButton)
        row.addWidget(self.jriverPreferencesButton)
        row.addWidget(self.audioStreamButton)
        outer.addLayout(row)
        self.messageLabel = QLabel()
        self.messageLabel.setWordWrap(True)
        self.messageLabel.setVisible(False)
        outer.addWidget(self.messageLabel)

    def button_for(self, kind: str) -> QPushButton:
        return self.monoButton if kind == MONO else self.multichannelButton

    def show_projects(self, states: List[ProjectState], can_open: bool, why_not: str) -> None:
        by_kind = {s.kind: s for s in states}
        for kind in (MONO, MULTICHANNEL):
            state, button = by_kind.get(kind), self.button_for(kind)
            button.setVisible(kind == MONO or state is not None)
            enabled = state is not None and state.readable and can_open
            button.setEnabled(enabled)
            button.setToolTip(
                why_not if not can_open else
                f'Open {state.path} in the main window' if enabled else
                f'The {kind} project could not be read: {state.error}' if state is not None and state.error else
                f'There is no {kind} project yet: it is written when the title is designed.' if state is not None else '')
        text, level = badge(states)
        self.projectBadge.setText(text)
        self.projectBadge.setToolTip(EDITED_TOOLTIP if level == LEVEL_WARN and 'Modified' in text else '')
        colour = warning_colour().name()
        self.projectBadge.setStyleSheet(f'color: {colour}' if level == LEVEL_WARN else '')
        self.projectsLabel.setVisible(bool(states))
        self.projectBadge.setVisible(bool(text))

    def show_message(self, text: str, problem: bool = False) -> None:
        self.messageLabel.setText(text)
        self.messageLabel.setVisible(bool(text))
        self.messageLabel.setStyleSheet(f'color: {warning_colour().name()}' if problem else '')


class TitleActions:
    '''
    The mixin: it uses the page's `titleRootLayout`, `noticeLabel`, `_title_id`, `_entry`, `_rows`, `_running`,
    `_decided_here`, `_flush_for_move()`, `_say()`, `reload()`, `notice`, `revised`, `_read_entry()`.
    '''

    def _configure_title_actions(self, hooks: Optional[TitleHooks]) -> None:
        self._hooks = hooks or TitleHooks()
        self._revised_here: Dict[str, str] = {}    # id -> how far it was sent back, on this page since the rows were read
        self._revised_stamp: Dict[str, tuple] = {}   # id -> the design as the revise left it (`_design_stamp`)
        self._bar = TitleActionsBar(self)
        self.titleRootLayout.insertWidget(self.titleRootLayout.indexOf(self.noticeLabel) + 1, self._bar)
        self._bar.open_requested.connect(lambda kind: self.open_project(kind))
        self._bar.revise_requested.connect(lambda: self.revise())
        self._bar.retry_requested.connect(self.retry_failed)
        self._bar.jriver_preferences_requested.connect(self.open_jriver_preferences)
        self._bar.audio_stream_requested.connect(self.choose_audio_stream)

    @property
    def actions_bar(self) -> TitleActionsBar:
        return self._bar

    @property
    def revised_here(self) -> Dict[str, str]:
        ''' The titles sent back on this page since the index last read them, and how far. '''
        return dict(self._revised_here)

    def forget_revised(self) -> None:
        self._revised_here = {}
        self._revised_stamp = {}

    @staticmethod
    def _design_stamp(entry: Optional[QueueEntry]) -> tuple:
        '''
        What a redesign changes and this page does not: the design fingerprint (a library run records one; a revise cleared it),
        the candidates and the reviewer note (Batch Extract & Design writes a fresh entry without the note a revise added).
        Metadata and artwork edits, which the page writes itself, are not in it.
        '''
        if entry is None:
            return ()
        return (entry.design_fingerprint, entry.reviewer_note, json.dumps([asdict(c) for c in entry.candidates], sort_keys=True))

    def _release_if_redesigned(self) -> None:
        '''
        A title sent back for redesign is held back from Accept because the page cannot tell it is designed again (its row is stale,
        or, in the review folder, there is none). Called whenever the entry is read: if what a design writes differs from what the
        revise left, the title *was* designed again, and the hold goes -- otherwise it would stay for as long as the page lives.
        '''
        title_id = self._title_id
        stamp = self._revised_stamp.get(title_id)
        if stamp is not None and self._entry is not None and self._design_stamp(self._entry) != stamp:
            self._revised_here.pop(title_id, None)
            self._revised_stamp.pop(title_id, None)

    # --- projects ----------------------------------------------------------------------------------------------------------

    def project_states(self) -> List[ProjectState]:
        ''' The title's projects as they are on disk now. '''
        return project_states(self._hooks.work_dir(), self._title_id)

    def refresh_projects(self) -> None:
        ''' Reads the projects again (a person may have saved one in the main window since). '''
        can_open = self._hooks.open_project is not None
        self._bar.show_projects(self.project_states() if self._title_id else [], can_open,
                                'Opening a project needs the main window: use the Library Work List from Tools.')

    def open_project(self, kind: str) -> bool:
        '''
        Opens the mono or multichannel project in the main window (what *File > Load Project* does with that file).
        :return: False if it could not be opened -- there is no such project, or nothing to open it with -- and the page says why.
        '''
        opener = self._hooks.open_project
        state = next((s for s in self.project_states() if s.kind == kind), None)
        if opener is None:
            self._bar.show_message('Projects open in the main window, which is not available here.', True)
            return False
        if state is None or not state.exists:
            self._bar.show_message(f'There is no {kind} project yet: it is written when the title is designed.', True)
            return False
        if state.error:
            self._bar.show_message(f'The {kind} project could not be read: {state.error}', True)
            return False
        try:
            opened = opener(state.path)
        except Exception as error:   # the main window refused, or the file is not a project after all
            logger.exception('Could not open the project %s', state.path)
            self._bar.show_message(f'Could not open the {kind} project: {type(error).__name__}: {error}', True)
            return False
        if opened is False:          # the person declined to replace what the main window holds
            self._bar.show_message(f'The {kind} project was not opened.')
            return False
        self._bar.show_message(f'Opened the {kind} project in the main window. Edit the filter there, then save it over '
                               f'the same file (File > Save Project: {state.path}) to keep the change. This page notices '
                               f'when you come back.')
        self.notice.emit(f'Opened {state.path}')
        return True

    # --- revise ------------------------------------------------------------------------------------------------------------

    def _revise_blocked(self) -> str:
        entry = self._entry
        if entry is None:
            return 'Nothing has been designed for this title yet.'
        if self._title_id in self._running():
            return 'A run is working on this title now: wait for it to finish.'
        return self._hooks.revise_blocked(self._title_id, entry.status)

    def _render_actions(self) -> None:
        self.refresh_projects()
        self._render_revise()

    def _render_revise(self) -> None:
        ''' Whether Reopen / Revise is offered now (a run may have started or ended with this title). '''
        blocked = self._revise_blocked()
        self._bar.reviseButton.setEnabled(not blocked)
        self._bar.reviseButton.setToolTip(blocked or 'Send this title back: reopen it for review, redesign it or extract '
                                                       'its audio again. You are asked what will happen first.')
        row = self._rows().get(self._title_id)
        failed = row is not None and (row.extract_state == 'failed' or row.design_state == 'failed')
        can_retry = failed and self._hooks.retry_failed is not None and self._title_id not in self._running()
        self._bar.retryButton.setVisible(failed)
        self._bar.retryButton.setEnabled(can_retry)
        self._bar.retryButton.setToolTip('Run this failed title again.' if can_retry else
                                         'Wait for the current run to finish before retrying.')
        mapping_problem = row is not None and 'Preferences > JRiver' in row.detail
        self._bar.jriverPreferencesButton.setVisible(mapping_problem)
        self._bar.jriverPreferencesButton.setEnabled(mapping_problem and self._hooks.open_jriver_preferences is not None)
        can_choose_stream = self._hooks.choose_audio_stream is not None and self._title_id not in self._running()
        self._bar.audioStreamButton.setVisible(self._hooks.choose_audio_stream is not None)
        self._bar.audioStreamButton.setEnabled(can_choose_stream)
        self._bar.audioStreamButton.setToolTip('Choose the source audio stream, then re-extract and redesign this title.'
                                               if can_choose_stream else 'Wait for the current run to finish before changing stream.')

    def open_jriver_preferences(self) -> None:
        opener = self._hooks.open_jriver_preferences
        if opener is not None:
            opener()

    def retry_failed(self) -> bool:
        retry = self._hooks.retry_failed
        if retry is None:
            return False
        return retry(self._title_id)

    def choose_audio_stream(self) -> bool:
        chooser = self._hooks.choose_audio_stream
        return bool(chooser and chooser(self._title_id))

    def _summary(self) -> ReviseSummary:
        row = self._rows().get(self._title_id)
        commit = row.commit_state if row is not None else ''
        if self._hooks.commit_state is not None:
            commit = self._hooks.commit_state(self._title_id)
        title = (self._entry.meta.get('title') if self._entry is not None else '') or self._title_id
        return ReviseSummary.of([(self._entry.status if self._entry is not None else 'none', commit)], str(title))

    def _ask_revise(self, summary: ReviseSummary, context: Optional[ReviseContext], default: str
                    ) -> Optional[Tuple[str, str]]:
        dialog = ReviseDialog(self, summary, context, default, run_now=self._hooks.run_design is not None)
        return (dialog.choice, dialog.reason) if dialog.exec() else None

    def revise(self, to: Optional[str] = None, reason: str = '') -> bool:
        '''
        Sends this title back. With `to` given, does it (the question is the person's to answer, so a caller that names the
        depth has answered it); without, asks with the dialog, which says what will happen and what could not.
        :return: True if the title was sent back. False if it was refused or cancelled -- and the page says why.
        '''
        blocked = self._revise_blocked()
        if blocked:
            self._say(blocked, problem=True)
            return False
        if not self._flush_for_move():    # what is being typed is saved first: revising rewrites the entry
            return False
        self._read_entry()                # the status it has *now*
        entry = self._entry
        if entry is None:
            self._say('Nothing has been designed for this title yet.', problem=True)
            return False
        context, summary = self._hooks.revise_context(), self._summary()
        if to is None:
            ask = self._hooks.ask_revise or self._ask_revise
            answer = ask(summary, context, 'design' if entry.status == 'pending' else 'review')
            if answer is None:
                return False
            to, reason = answer
        problem = revise_problem(to, summary, context)
        if problem:
            self._say(f'Not changed: {problem}', problem=True)
            return False
        outcome = revise_titles(context, [self._title_id], to, reason)
        if outcome.failed:
            self._say(f'Not changed: {outcome.failed[0][1]}', problem=True)
            return False
        self._decided_here.add(self._title_id)    # the rows still say what it was
        self._revised_here[self._title_id] = to
        try:
            self._revised_stamp[self._title_id] = self._design_stamp(read_entry(self._queue_dir(), self._title_id))
        except Exception:      # unreadable now: nothing to tell a redesign from, so the hold stays until the rows are read
            self._revised_stamp.pop(self._title_id, None)
        self.revised.emit(self._title_id, to)
        started = to in ('design', 'extract') and self._hooks.run_design is not None and self._hooks.run_design(self._title_id)
        self.reload()
        text, _ = summarise_outcome(outcome)
        self._say('Extract & design is starting now.' if started else text)
        return True
