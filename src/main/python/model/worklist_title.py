'''
The title page -- design/archive/library-sync/workflow-rework/design.md §12.10, chunk 27a (core).

One title of the work list, in the window itself (it replaces the table; `model.worklist_titles` is how the window opens
and closes it): what the index says about the title, and -- for one that has been designed -- its candidates, the
commentary on each and the filter it would apply, drawn over the signal. **Accept & next** picks the highlighted
candidate and goes to the next title in the list that is waiting for a decision; **Skip** and **Reject** decide the other
ways and advance the same. Previous / Next walk the list the page was opened over ("12 of 37").

Chunk 27b adds the **metadata and artwork** (`model.worklist_metadata.MetadataPanel`, on the *Metadata* tab beside the
chart): the header badge says whether the metadata is complete enough to publish ("Ready to publish", or what is missing),
and **Accept is not offered while it is not** -- the index calls an accepted title with incomplete metadata `review` again
("metadata incomplete"), and publish refuses it, so accepting it would only put it back in the list. (The index judges the
metadata of pending, accepted and published titles only: for a skipped or rejected one the page lists the gaps without the
"Not ready to publish" alarm.) Skip and Reject are not held up by incomplete metadata. **An edit is never lost**: it is
saved when a field is left, and `flush()` saves it before the page moves (Previous / Next / a decision / back / closing the
window); an edit that cannot be saved blocks Previous / Next and Accept, and for Skip, Reject, back and closing it asks
whether to discard it (Discard / Cancel, `confirm_discard`), so a person is never stuck behind it. A refused Accept (the key, or
Enter on the candidate list) shows the Metadata tab and marks the missing box **without moving the keyboard** into it: the
same key again must not type into a field. The keys A, S, R and 1-9 yield to a text field (it claims printable keys);
Enter decides only on the candidate list; Esc in a field returns to the candidate list, and a second one goes back.

Chunk 27c adds, through `model.worklist_title_actions`, **Open project** (the title's mono and multichannel `.beq` projects, opened
in the main window through a callable the work list is given; a badge says whether a project was *modified since design*) and
**Reopen / Revise...** (`model.worklist_revise`: back to review, redesign or re-extract; state only, with the question first). A
title decided here can be reversed there: Accept, Skip and Reject apply again to a reopened (pending) title, and one sent back for
redesign is not accepted until it has been designed again. The *metadata* of an accepted, published or rejected title can be
edited (it becomes Publish: out of date, or needs review again, on the next read of the index).

Designer contract 1.1 (design/outstanding.md W4): the designs the designer **rejected** are listed after the candidates, under
a heading that cannot be picked, numbered on from them (`QueueEntry.offered`) so the digit keys reach them too; picking one shows
why it was rejected, its commentary and its filter on the chart like any candidate. Accepting one is a person overriding the
designer: the button says so, `confirm_override` asks first (with the reasons), and the entry records it.

The page is three files: this one (the widgets, what is shown, editing), `model.worklist_title_actions` (projects, Reopen / Revise) and
`model.worklist_title_decide` (Accept / Skip / Reject and which title is next), mixins of `TitlePage`.

The queue entry (`pipeline.review.QueueEntry`, one `<id>.json` in the queue directory) is the truth for a decision: the
page reads it when it shows a title and again before it writes one, so a decision is never applied to a design that
has changed since it was on screen. The index is not touched here -- the rows it gave the page are stale after a
decision or an edit -- and the window reads the outputs again when the page is left (see `model.worklist_titles`; the
`decided` and `changed` signals tell it that it must).
'''
import logging
from typing import Callable, List, Mapping, Optional, Sequence, Set

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QKeySequence, QShortcut
from qtpy.QtWidgets import QAbstractSpinBox, QApplication, QComboBox, QLineEdit, QListWidgetItem, QMessageBox, \
    QPlainTextEdit, QTextEdit, QWidget, QHeaderView

from model.magnitude import MagnitudeModel
from model.worklist_spectrum import SpectrumPanel, comparison_request
from model.worklist_catalogue import catalogue_target
from model.worklist_published import PublishedPanel
from model.worklist_metadata import MetadataPanel, badge_alarms, badge_text, ok_colour
from model.worklist_model import warning_colour
from model.worklist_title_actions import TitleActions, TitleHooks
from model.worklist_title_decide import DECISION_FROM, TitleDecisions
from model.worklist_title_text import ACCEPTABLE, REJECTABLE, candidate_text, chart_data, playback_html, \
    commentary_html, decline_commentary, decision_blocked, entry_title, entry_year, next_waiting_id, notice_text, \
    override_question, position_text, rejected_heading, rejection_html, revised_note, \
    state_text  # noqa: F401 (the pure functions are re-exported: tests and callers import them from here)
from pipeline.library.index import TitleRow
from pipeline.review import QueueEntry, read_entry
from ui.worklisttitle import Ui_titlePage

logger = logging.getLogger('worklist')



# --- the page --------------------------------------------------------------------------------------------------------------

class TitlePage(TitleDecisions, TitleActions, QWidget, Ui_titlePage):
    '''
    :param preferences: for the chart and the TMDB key.
    :param queue_dir: where the queue entries are (asked each time: a setting may change while the page is open).
    :param rows: the index's rows by title id, as the window has them now (asked each time).
    :param running: the titles a run is working on now, by id (asked each time).
    :param meta_defaults: the profile's `meta_defaults`, what publish fills in where a title's metadata is silent (asked
        each time: the badge must agree with what the index says of the same metadata).
    :param confirm_discard: asked, with the reason, when an edit cannot be saved and the page is being left anyway; True to
        throw the edit away. A question box unless given.
    :param confirm_override: asked, with the question (`override_question()`), before a design the designer rejected is
        accepted; True to accept it anyway. A question box unless given.
    :param choose_file: asks for an artwork file; the file dialog unless given.
    :param hooks: what the page asks of its window: opening a project, the revise settings (`TitleHooks`).
    '''
    back_requested = Signal()       # the breadcrumb, or Esc
    title_shown = Signal(str)       # the id now on the page
    decided = Signal(str, str)      # a title's id, and the status it was given (accepted, skipped, rejected)
    changed = Signal(str)           # a title's metadata or artwork was written (the index must read it again)
    notice = Signal(str)            # something to tell about a title that is not on the page (for the window's status bar)
    revised = Signal(str, str)      # a title's id, and how far it was sent back (review, design, extract): the index must read it again

    def __init__(self, parent, preferences, queue_dir: Callable[[], str], rows: Callable[[], Mapping[str, TitleRow]],
                 running: Callable[[], Mapping[str, str]] = lambda: {},
                 meta_defaults: Callable[[], Optional[dict]] = lambda: None,
                 confirm_discard: Optional[Callable[[str], bool]] = None,
                 choose_file: Optional[Callable[[], str]] = None, hooks: Optional[TitleHooks] = None,
                 confirm_override: Optional[Callable[[str], bool]] = None):
        super().__init__(parent)
        self.setupUi(self)
        self._configure_title_actions(hooks)
        from model.filter import FilterModel, FilterTableModel
        self._proposed_filters = FilterModel(self.proposedFilterView, preferences)
        self.proposedFilterView.setModel(FilterTableModel(self._proposed_filters, self.proposedFilterView))
        self.proposedFilterView.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.proposedFilterView.horizontalHeader().setStretchLastSection(True)
        self.proposedFilterView.verticalHeader().hide()
        self._queue_dir, self._rows, self._running, self._defaults = queue_dir, rows, running, meta_defaults
        self.confirm_discard = confirm_discard or self._ask_discard
        self.confirm_override = confirm_override or self._ask_override
        self._offered_at_row: List[Optional[int]] = []   # the candidate list's rows: an index into `offered`, None for a heading
        self._ids: List[str] = []
        self._review_session = False
        self._title_id = ''
        self._entry: Optional[QueueEntry] = None
        self._error = ''
        self._picked = 0
        self._decided_here: Set[str] = set()   # decided on this page since it was opened: the rows do not say so yet
        self._edited_here: Set[str] = set()    # edited on this page: nor do they say what its metadata is now
        self._problems: List[str] = []
        font = self.titleLabel.font()
        font.setBold(True)
        self.titleLabel.setFont(font)
        self.crumbLabel.hide()
        self.navLayout.setStretch(2, 1)
        self.headerLayout.setStretch(1, 1)
        self.titleSplitter.setStretchFactor(0, 0)
        self.titleSplitter.setStretchFactor(1, 1)
        self.titleSplitter.setSizes([440, 660])   # a candidate's line is long: give the list room to read it
        self.decisionLayout.setStretch(0, 1)      # the label takes what the buttons leave
        self._magnitude = MagnitudeModel('worklist-title', self.previewChart, preferences, self._chart_data, 'Filter',
                                         fill_primary=False)
        self._metadata = MetadataPanel(self, preferences, queue_dir, running, meta_defaults, choose_file)
        self.metadataTabLayout.addWidget(self._metadata)
        self._metadata.saved.connect(self._on_saved)
        self._metadata.edited.connect(self._on_edited)
        self._metadata.elsewhere.connect(self.notice.emit)
        for button in (self.backButton, self.previousButton, self.nextButton, self.skipButton, self.rejectButton,
                       self.acceptButton):
            button.setAutoDefault(False)
        self.backButton.clicked.connect(self.back_requested.emit)
        self.previousButton.clicked.connect(lambda: self.previous())
        self.nextButton.clicked.connect(lambda: self.next())
        self.acceptButton.clicked.connect(lambda: self.accept())
        self.skipButton.clicked.connect(lambda: self.skip())
        self.rejectButton.clicked.connect(lambda: self.reject())
        self.candidateList.currentRowChanged.connect(self._on_candidate_picked)
        self.spectrumPanel = SpectrumPanel(self)
        self._spectrum_preferences = preferences
        self.rightTabs.addTab(self.spectrumPanel, 'Spectrum comparison')
        self.publishedPanel = PublishedPanel(preferences, self)
        self.rightTabs.addTab(self.publishedPanel, 'Published BEQs')
        self.spectrumPanel.refresh_requested.connect(lambda: self._refresh_spectrum(force=True))
        self.rightTabs.currentChanged.connect(lambda _: (self._refresh_spectrum(), self._refresh_published()))
        self._install_shortcuts()

    def _install_shortcuts(self) -> None:
        '''
        Enter decides only while the candidate list has focus (the lesson of design.md §12.1: a window-wide Enter fired from
        inside a text field). The letters and digits belong to the page, not the window, and need no more scoping than that:
        a text field claims printable keys itself (its `ShortcutOverride`), so A, S, R and 1-9 typed into one reach the
        field (`test_worklist_metadata.py` presses each). Esc in a field returns to the candidate list; there a second one
        goes back. Alt+Left/Right still move between titles from a field: the edit is saved first (`flush()`).
        '''
        # a held key must not decide a title a second time, let alone every title down the list: no auto-repeat on
        # the decisions (and Enter from the table, which opens the page and moves the focus here, may still be down)
        for key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            QShortcut(QKeySequence(key), self.candidateList, activated=lambda: self.accept(),
                      context=Qt.ShortcutContext.WidgetShortcut).setAutoRepeat(False)
        scope = Qt.ShortcutContext.WidgetWithChildrenShortcut
        typing = self._in_text_field
        for keys, action, repeats in (('A', self.accept, False), ('S', self.skip, False), ('R', self.reject, False),
                                      ('Esc', self._escape, False), ('Alt+Left', self.previous, True),
                                      ('Alt+Right', self.next, True)):
            # A, S and R do nothing from a text field: an editable one never sees the shortcut (it claims the key), but a
            # read-only one does not, and the key must not decide a title either. Esc and Alt+arrows are safe from one.
            guarded = keys in ('A', 'S', 'R')
            QShortcut(QKeySequence(keys), self, activated=lambda action=action, guarded=guarded:
                      None if guarded and typing() else action(), context=scope).setAutoRepeat(repeats)
        for i in range(1, 10):
            QShortcut(QKeySequence(str(i)), self, context=scope,
                      activated=lambda index=i - 1: None if typing() else self.pick_candidate(index))

    def _in_text_field(self) -> bool:
        '''
        Whether the keyboard is in a box for text (a read-only one included) inside the page. The commentary is only
        read, so the decision keys still work from it.
        '''
        focus = QApplication.focusWidget()
        return isinstance(focus, (QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QAbstractSpinBox)) \
            and focus is not self.commentaryText and self.isAncestorOf(focus)

    # --- what the page shows ------------------------------------------------------------------------------------------

    @property
    def current_id(self) -> str:
        return self._title_id

    @property
    def entry(self) -> Optional[QueueEntry]:
        ''' The queue entry as it was read when the title was shown (None if it has none, or could not be read). '''
        return self._entry

    @property
    def ids(self) -> List[str]:
        ''' The list Previous and Next walk: the titles the work list showed when the page was opened. '''
        return list(self._ids)

    @property
    def picked(self) -> int:
        ''' The zero-based index of the highlighted candidate. '''
        return self._picked

    def open(self, title_id: str, ids: Sequence[str]) -> None:
        ''' Shows `title_id`, with `ids` (the work list's filtered and sorted titles) to walk through. '''
        self._ids = list(ids) if title_id in ids else [title_id, *ids]
        rows = self._rows()
        self._review_session = bool(self._ids) and all(
            rows.get(item) is not None and rows[item].needs == 'review' for item in self._ids)
        self.previousButton.setVisible(not self._review_session)
        self.nextButton.setVisible(not self._review_session)
        self.show_title(title_id)

    def forget_decisions(self, revised: bool = True) -> None:
        '''
        The rows now say what was decided and edited (the window read the index again): no need to remember it here.
        :param revised: also forget which titles were sent back for redesign, which held Accept back until the rows could say so.
            The review folder's rows are always current but say nothing of a design (they are made from the entry), so it passes
            False: the hold goes when the entry is seen to have been designed again (`_release_if_redesigned`), not before.
        '''
        self._decided_here = set()
        self._edited_here = set()
        if revised:
            self.forget_revised()

    def show_title(self, title_id: str) -> bool:
        '''
        Shows another title from the list (the entry is read again), after saving what was being edited on this one.
        :return: False, and stays where it is, if that edit could not be saved.
        '''
        if not self._flush_for_move():
            return False
        if title_id != self._title_id:
            self._bar.show_message('')   # a note about the title shown before (a project opened for it) is not about this one
        self._title_id = title_id
        self._read_entry()
        self._picked = self._default_pick()
        self._render()
        self.candidateList.setFocus()
        self.title_shown.emit(title_id)
        return True

    def reload(self) -> None:
        '''
        Reads the entry and the rows again -- after the index was refreshed or the settings changed. The highlighted
        candidate is kept if the same candidates are still there.
        '''
        before = self._entry
        self._read_entry()
        same = before is not None and self._entry is not None and before.offered == self._entry.offered
        self._picked = self._picked if same else self._default_pick()
        self._render(keep_edits=True)   # what is being typed is not thrown away by a refresh

    def _read_entry(self) -> None:
        self._entry, self._error = None, ''
        queue_dir = self._queue_dir()
        if not queue_dir:
            return
        try:
            self._entry = read_entry(queue_dir, self._title_id)
        except FileNotFoundError:
            pass    # not designed yet: the notice says what the title needs
        except Exception as error:  # a damaged or newer-format file: the page says so and the rest of the list still works
            logger.exception('Could not read the queue entry %s', self._title_id)
            self._error = f'{type(error).__name__}: {error}'
        self._release_if_redesigned()

    def _default_pick(self) -> int:
        entry = self._entry
        if entry is not None and entry.status == 'accepted' and entry.chosen_candidate_index is not None:
            return entry.chosen_candidate_index
        return 0

    def _render(self, keep_edits: bool = False) -> None:
        rows = self._rows()
        self._render_header(rows)
        self._render_position()
        self._render_candidates()
        self._metadata.show_entry(self._title_id, self._entry, self._kind(rows), keep_edits=keep_edits, error=self._error)
        self._problems = self._metadata.problems()
        self._render_badge()
        self._render_decisions(rows)
        self._render_actions()

    def _kind(self, rows: Mapping[str, TitleRow]) -> str:
        row = rows.get(self._title_id)
        return row.kind if row is not None and row.kind else 'movie'

    def _render_header(self, rows: Mapping[str, TitleRow]) -> None:
        row, entry = rows.get(self._title_id), self._entry
        edited = self._title_id in self._edited_here
        title, year = entry_title(entry, row, self._title_id, edited), entry_year(entry, row, edited)
        heading = f'{title} ({year})' if year else title
        self.crumbLabel.setText(f'› {heading}')
        self.titleLabel.setText(heading)
        self.titleLabel.setToolTip('\n'.join(p for p in (self._title_id, row.path if row else '') if p))
        parts = [row.source, row.kind, *row.flags] if row is not None else []
        self.subtitleLabel.setText(' · '.join(p for p in parts if p))
        self.subtitleLabel.setToolTip(row.path if row is not None and row.path else '')
        revised = self._revised_here.get(self._title_id, '')
        self.stateLabel.setText(' '.join(t for t in (state_text(entry, row, stale=edited or bool(revised)),
                                                     revised_note(revised, self._hooks.redo)) if t))
        notice = notice_text(entry, row, self._queue_dir(), self._error)
        self.noticeLabel.setText(notice)
        self.noticeLabel.setVisible(bool(notice))
        colour = warning_colour().name()
        self.noticeLabel.setStyleSheet(f'color: {colour}' if self._error else '')

    def _render_badge(self) -> None:
        ''' "Ready to publish", or what is missing: `validate()`'s verdict on the metadata as it would be saved now. '''
        entry, problems = self._entry, self._problems
        self.badgeLabel.setVisible(entry is not None)
        alarm = entry is not None and badge_alarms(problems, entry.status)
        if entry is not None:
            self.badgeLabel.setText(badge_text(problems, entry.status, self._metadata.dirty))
            # a skipped or rejected title's gaps are listed, not shouted: the index says nothing of them either
            self.badgeLabel.setStyleSheet('' if problems and not alarm else
                                          f'color: {warning_colour().name() if problems else ok_colour()}')
        self.rightTabs.setTabText(1, 'Metadata !' if alarm else 'Metadata')

    def _render_position(self) -> None:
        position = self._ids.index(self._title_id) if self._title_id in self._ids else -1
        self.positionLabel.setText(position_text(position, len(self._ids)))
        self.previousButton.setEnabled(position > 0)
        self.nextButton.setEnabled(0 <= position < len(self._ids) - 1)

    def _render_candidates(self) -> None:
        entry = self._entry
        offered = entry.offered if entry is not None else []
        if not 0 <= self._picked < len(offered):
            self._picked = 0
        self.candidateList.blockSignals(True)
        self.candidateList.clear()
        self._offered_at_row = []
        for i, candidate in enumerate(offered):
            if i == len(entry.candidates):   # the first rejected design: a heading, which cannot be picked, goes first
                heading = QListWidgetItem(rejected_heading(len(entry.rejected)))
                heading.setFlags(Qt.ItemFlag.NoItemFlags)
                font = heading.font()
                font.setBold(True)
                heading.setFont(font)
                self.candidateList.addItem(heading)
                self._offered_at_row.append(None)
            item = QListWidgetItem(candidate_text(i, candidate))
            if candidate.rejection_reasons:
                font = item.font()
                font.setItalic(True)
                item.setFont(font)
                item.setToolTip('Rejected by the designer:\n' + '\n'.join(f'- {r}' for r in candidate.rejection_reasons))
            self.candidateList.addItem(item)
            self._offered_at_row.append(i)
        if offered:
            self.candidateList.setCurrentRow(self._offered_at_row.index(self._picked))
        self.candidateList.blockSignals(False)
        self._render_commentary()
        self._magnitude.redraw()
        self._refresh_spectrum()
        self._refresh_published()

    def _picked_rejected(self) -> bool:
        ''' The highlighted design is one the designer rejected: accepting it is an override. '''
        entry = self._entry
        return entry is not None and len(entry.candidates) <= self._picked < len(entry.offered)

    def _render_commentary(self) -> None:
        entry = self._entry
        from model.codec import filter_from_json
        selected = entry.offered[self._picked] if entry is not None and 0 <= self._picked < len(entry.offered) else None
        self._proposed_filters.filter = filter_from_json(selected.filters) if selected else None
        self.proposedFiltersHeading.setText(f'Proposed filters — design {self._picked + 1}' if selected else 'No proposed filters')
        commentary = {}
        rejected = self._picked_rejected()
        declined = entry is not None and bool(entry.decline_reason) and not rejected
        if declined:
            commentary = decline_commentary(entry.decline_reason, entry.decline_message)
        elif entry is not None and 0 <= self._picked < len(entry.candidates):
            commentary = entry.candidates[self._picked].commentary or {}
        self.commentaryHeading.setText('Rejected by the designer' if rejected else
                                       'Why the designer declined' if declined else 'Commentary')
        self.commentaryText.setHtml((rejection_html(entry.offered[self._picked]) if rejected else commentary_html(commentary))
                                    + playback_html(entry))

    def _render_decisions(self, rows: Mapping[str, TitleRow]) -> None:
        entry, row = self._entry, rows.get(self._title_id)
        status = entry.status if entry is not None else ''
        has_candidates = entry is not None and bool(entry.offered)
        # accepting a rejected design is an override of the designer: the button says so, and it asks first
        self.acceptButton.setText('Override && accept...' if self._picked_rejected() else 'Accept && next')
        running = self._title_id in self._running()
        blocked = {d: decision_blocked(d, entry, row, running, self._problems, self._revised_here.get(self._title_id, ''),
                                       self._hooks.redo)
                   for d in DECISION_FROM}
        self.acceptButton.setEnabled(has_candidates and status in ACCEPTABLE and not blocked['accept'])
        self.skipButton.setEnabled(self._can_skip(rows))
        self.rejectButton.setEnabled(status in REJECTABLE and not blocked['reject'])
        offered = [d for d, status_set in DECISION_FROM.items() if status in status_set]
        reason = next((blocked[d] for d in offered if blocked[d]), '')
        if reason:
            text = reason
        elif status == 'accepted' and entry.overrides_rejection:
            text = f'Accepted design {entry.chosen_candidate_index + 1}, which the designer rejected: your override.'
        elif status == 'accepted' and entry.chosen_candidate_index is not None:
            text = f'Accepted candidate {entry.chosen_candidate_index + 1}.'
        elif self._picked_rejected() and status in ACCEPTABLE:
            text = 'The designer rejected this design: accepting it overrides the designer, and asks first.'
        else:
            waiting = sum(1 for i in self._ids if self._probably_waiting(i, rows))
            text = f'{waiting:,} in this list waiting for a decision.' if waiting else ''
        self.decisionLabel.setText(text)
        self.decisionLabel.setToolTip(text)
        self.decisionLabel.hide()
        for button, decision in ((self.acceptButton, 'accept'), (self.rejectButton, 'reject')):
            button.setToolTip(blocked[decision] or text)
        self.decisionLabel.setStyleSheet(f'color: {warning_colour().name()}' if reason else '')

    def refresh_decisions(self) -> None:
        ''' Which decisions are offered, again -- a run started or finished with this title. '''
        self._metadata.refresh_enabled()
        self._render_decisions(self._rows())
        self._render_revise()
        self._refresh_spectrum()
        self._refresh_published()

    # --- editing: what the panel tells the page ---------------------------------------------------------------------------

    @property
    def metadata(self) -> MetadataPanel:
        return self._metadata

    def flush(self) -> bool:
        ''' Saves what is being edited. False -- and the reason is on the metadata status line -- if it cannot be. '''
        return self._metadata.flush()

    def _flush_for_move(self) -> bool:
        ''' `flush()` before going to another title or deciding: if it fails, the page stays and says why. '''
        if self.flush():
            return True
        self.rightTabs.setCurrentIndex(1)
        self._say('Not moved: ' + self._metadata.metadataStatusLabel.text(), problem=True)
        return False

    def leave(self) -> bool:
        '''
        Before the page is left (back, or the window closing): saves what is being edited; if that fails, asks whether to
        throw it away. :return: True if it is all right to go.
        '''
        if self._flushed_or_discarded():
            self._metadata.leave()      # a lookup still on its way is no longer wanted
            self.spectrumPanel.unavailable('Open a title to view its spectrum comparison.')
            self.publishedPanel.set_target(None)
            return True
        return False

    def _flushed_or_discarded(self) -> bool:
        '''
        `flush()`; if that fails, asks whether to throw the edit away (`confirm_discard`). :return: False if the edit is still
        there -- it could not be saved and the person did not discard it: the page shows it.
        '''
        if self.flush():
            return True
        if self.confirm_discard(self._metadata.metadataStatusLabel.text()):
            self._metadata.discard()
            return True
        self.rightTabs.setCurrentIndex(1)
        return False

    def _ask_override(self, question: str) -> bool:
        answer = QMessageBox.question(self, 'Override the designer?', question,
                                      QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                                      QMessageBox.StandardButton.Cancel)
        return answer == QMessageBox.StandardButton.Yes

    def _ask_discard(self, reason: str) -> bool:
        answer = QMessageBox.question(
            self, 'Unsaved metadata', f'{reason}\n\nLeave anyway and discard what you typed?',
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Cancel)
        return answer == QMessageBox.StandardButton.Discard

    def _on_edited(self) -> None:
        ''' Something was typed: the badge, the tab and whether Accept is offered follow it, without a save. '''
        problems = self._metadata.problems()
        if problems != self._problems:
            self._problems = problems
            self._render_decisions(self._rows())
        self._render_badge()

    def _on_saved(self, title_id: str) -> None:
        '''
        The panel wrote a title's metadata or artwork (the title it was shown for, which is not this one if a download came
        back after the page moved on): the window must read the index again, and the header follows what is saved now.
        '''
        self._edited_here.add(title_id)
        self.changed.emit(title_id)
        if title_id != self._title_id:
            return
        try:
            fresh = read_entry(self._queue_dir(), title_id)
        except Exception:    # gone, or damaged since: the page shows what is there now
            self.reload()
            return
        if self._entry is not None and fresh.offered == self._entry.offered:
            rows = self._rows()
            self._entry = fresh      # the design on screen is the one decided on, so only the parts an edit touches change
            self._render_header(rows)
            self._refresh_spectrum()
            self._refresh_published()
            self._metadata.show_entry(title_id, fresh, self._kind(rows), keep_edits=True, error='')
            self._problems = self._metadata.problems()
            self._render_badge()
            self._render_decisions(rows)
        else:
            self.reload()            # redesigned while it was open: show it (a decision is refused, see _decide)

    def _escape(self) -> None:
        ''' Esc in a text field goes back to the candidate list (which saves the edit); anywhere else it goes back. '''
        if self._in_text_field():
            self.candidateList.setFocus()
        else:
            self.back_requested.emit()

    def _say(self, text: str, problem: bool = False) -> None:
        self.decisionLabel.setText(text)
        self.decisionLabel.setToolTip(text)
        self.decisionLabel.hide()
        self._bar.show_message(text, problem)
        self.decisionLabel.setStyleSheet(f'color: {warning_colour().name()}' if problem else '')

    def _on_candidate_picked(self, row: int) -> None:
        index = self._offered_at_row[row] if 0 <= row < len(self._offered_at_row) else None
        if index is not None:
            self._picked = index
            self._render_commentary()
            self._render_decisions(self._rows())   # the Accept button says whether it is an override
            self._magnitude.redraw()
        self._refresh_spectrum()
        self._refresh_published()

    def _refresh_published(self) -> None:
        if hasattr(self, 'publishedPanel'):
            self.publishedPanel.set_target(catalogue_target(self._entry, self._rows().get(self._title_id), self._defaults()))
            self.publishedPanel.set_our_filter(self._chart_data(), f'Selected design {self._picked + 1}')

    def _refresh_spectrum(self, force=False) -> None:
        if not hasattr(self, 'spectrumPanel'):
            return
        if self._title_id in self._running():
            self.spectrumPanel.unavailable('A run is updating this title. The comparison is available when it finishes.')
            return
        row = self._rows().get(self._title_id)
        title = entry_title(self._entry, row, self._title_id, prefer_entry=True)
        try:
            request = comparison_request(self._hooks.work_dir(), self._entry, self._picked, title,
                                         self._hooks.analysis_config(), self._spectrum_preferences)
        except (ValueError, OSError) as error:
            self.spectrumPanel.unavailable(str(error))
        else:
            self.spectrumPanel.set_request(request, force=force)

    def _chart_data(self, reference=None) -> list:
        return chart_data(self._entry, self._picked)

    def pick_candidate(self, index: int) -> bool:
        ''' Highlights design `index` (zero-based, through `offered`: a rejected design too), as the digit keys do. '''
        if index in self._offered_at_row:
            self.candidateList.setCurrentRow(self._offered_at_row.index(index))
            return True
        return False

    # --- moving through the list ----------------------------------------------------------------------------------------

    def previous(self) -> bool:
        return self._step(-1)

    def next(self) -> bool:
        return self._step(1)

    def _step(self, by: int) -> bool:
        position = self._ids.index(self._title_id) if self._title_id in self._ids else -1
        if position < 0 or not 0 <= position + by < len(self._ids):
            return False
        return self.show_title(self._ids[position + by])
