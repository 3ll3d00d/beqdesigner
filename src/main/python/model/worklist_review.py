'''
The Review folder window -- design/library-sync/workflow-rework/design.md §12.10, chunk 27c. **Tools > Review Folder...** (it was
"Review Batch Designs", a tab of the Batch Extract dialog) is the work list's *title page* over a queue directory: the
entries a Batch Extract & Design run (or a scripted `pipeline.review.batch_design()`) wrote, worked through one by one with the
same page -- candidates, commentary and chart, Accept & next / Skip / Reject, the metadata and artwork, opening the projects,
Reopen / Revise -- so there is one place to review, one behaviour, and no second, older dialog to keep in step with it.

What it does not have is the discovery index: a queue directory is not a library, so there is no "needs" per title, nothing
scanned and nothing run. The page copes with that (`row is None` is a case it was built for): each entry is shown with a row
made up from its own status (`entry_row()`), which is all the page reads of a row.

**Publishing uses the work list's code, not a second path.** *Publish accepted* calls `pipeline.library.sync.publish_library`
and *Commit published* `commit_library` -- what a run through publish and commit calls (`stages.run_stages`) -- with the
repositories and the work directory of the library profile (Library Work List > Settings; the same `load_setup()` the work
list reads), so the standalone entry point now writes the images repository too, reads the `.beq` projects of an entry that has
them under the work directory (a person's edit is what is published; an entry designed by Batch Extract has none and is published
from its candidates; one whose project is there but whose audio is gone is refused, not published without the edit) and refuses
incomplete metadata per title, which the old XML-only Publish button did not (T9). Publish only
writes into the repositories' working trees; Commit is the separate, confirmed step that commits and pushes (one commit per
repository, images first). Both name the repositories and the count before doing anything, and run on the thread pool -- and
while one runs the window holds the interlocks the work list holds: the titles it is working on are "running" for the page
(no decision, no edit, no revise) and nothing that uses the repositories is revised.

The window reads the library profile again when it is shown or activated, and wherever a decision is made from it (Publish,
Commit, the revise question), so a repository set later in the work list is seen; and it reads where a published title's files
stand in git (`worklist_folder_state.commit_states`), so the revise question tells the truth about what it will do to them.

It is a `QMainWindow`, one per app (`BeqDesigner.showReviewFolderWindow`), remembering the folder in `DESIGNER_QUEUE_DIR`.
'''
import logging
import os
import time
from typing import Callable, Dict, List, Optional

from qtpy.QtCore import QEvent, QObject, QRunnable, Qt, QThreadPool, Signal
from qtpy.QtGui import QKeySequence, QShortcut
from qtpy.QtWidgets import QAbstractItemView, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPlainTextEdit, \
    QPushButton, QSplitter, QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from model.preferences import DESIGNER_QUEUE_DIR, WORKLIST_PUSH
from model.worklist_confirm import ConfirmDialog, commit_text, publish_text
from model.worklist_folder_state import commit_states, entry_row, split_for_publish  # noqa: F401 (entry_row is re-exported)
from model.worklist_model import warning_colour
from model.worklist_profile import WorkListSetup, load_setup
from model.worklist_revise import ReviseContext
from model.worklist_run import build_publish_settings, publish_problem
from model.worklist_title import TitlePage
from model.worklist_title_actions import TitleHooks
from model.worklist_title_text import REDO_IN_FOLDER
from pipeline.library.commit import CatalogueCommit
from pipeline.library.index import TitleRow
from pipeline.library.sync import commit_library, publish_library
from pipeline.review import QueueEntry, describe_publish_error, read_entry, read_queue, split_publish_results

logger = logging.getLogger('worklist')

_COLUMNS = ('Title', 'Status', 'Confidence')


def describe_publish(results: List[dict]) -> str:
    ''' "Published 3 (written to the repositories, not committed); 1 refused: ..." '''
    published, refused = split_publish_results(results)
    text = f'Published {len(published):,} title{"" if len(published) == 1 else "s"}: written into the repositories, not committed yet.'
    if refused:
        text += f' {len(refused):,} could not be published (see below).'
    return text


def describe_commit(commit: CatalogueCommit) -> str:
    parts = []
    for name, repo in (('images', commit.images), ('XML', commit.xml)):
        if repo is None:
            continue
        what = f'commit {repo.commit[:8]}' if repo.commit else 'already committed'
        parts.append(f'{name}: {what}{", pushed" if repo.pushed else ", not pushed"}')
    text = 'Committed. ' + '; '.join(parts) + '.'
    if commit.not_committed:
        text += f' {len(commit.not_committed):,} file(s) could not be committed: git ignores them.'
    return text


class _CallSignals(QObject):
    finished = Signal(object)
    errored = Signal(str)


class _CallJob(QRunnable):
    ''' `work()` on the thread pool; the result or the error comes back through `signals`, which the window keeps. '''

    def __init__(self, work: Callable[[], object], what: str):
        super().__init__()
        self.signals = _CallSignals()
        self._work, self._what = work, what

    def run(self):
        try:
            result = self._work()
            self.signals.finished.emit(result)
        except Exception as error:
            logger.exception('%s failed', self._what)
            self.signals.errored.emit(f'{type(error).__name__}: {error}')


class ReviewFolderWindow(QMainWindow):
    '''
    :param parent: the main window, or None.
    :param preferences: for the remembered folder, the chart, the TMDB key and the library profile's repositories.
    :param open_project: as `WorkListWindow`'s: opens a `.beq` project in the main window; None disables the buttons.
    :param choose_dir: asks for the folder (a directory dialog by default).
    :param ask_revise: as `WorkListWindow`'s.
    :param clock: seconds, monotonic (the tests set it): how old the library profile may be before a keystroke's badge reads it again.
    '''
    published = Signal(object)      # the results of a Publish, once the list shows them
    committed = Signal(object)      # the CatalogueCommit of a Commit
    failed = Signal(str)            # a Publish or Commit raised

    def __init__(self, parent, preferences, *, open_project: Optional[Callable[[str], Optional[bool]]] = None,
                 choose_dir: Optional[Callable[[str], str]] = None, ask_revise: Optional[Callable] = None,
                 auto_load: bool = True, clock: Callable[[], float] = time.monotonic):
        super().__init__(parent)
        self.setWindowTitle('Review Folder')
        self.setMinimumSize(860, 520)
        self.resize(1200, 720)
        self._preferences = preferences
        self._choose_dir = choose_dir or self._ask_dir
        self._queue_dir = ''
        self._entries: Dict[str, QueueEntry] = {}
        self._rows: Dict[str, TitleRow] = {}
        self._order: List[str] = []
        self._job: Optional[_CallJob] = None
        self._working_on: Dict[str, str] = {}      # id -> publish / commit, for the titles the job in flight is working on
        self._loading = False
        self._clock = clock
        self._setup = load_setup(preferences)
        self._setup_read = clock()
        self._build()
        hooks = TitleHooks(open_project=open_project, work_dir=self._work_dir, revise_context=self._revise_context,
                           revise_blocked=self._revise_blocked, ask_revise=ask_revise, redo=REDO_IN_FOLDER,
                           commit_state=self._commit_state)
        self._page = TitlePage(self, preferences, lambda: self._queue_dir, lambda: self._rows, lambda: self._working_on,
                               self._meta_defaults, hooks=hooks)
        self._page.backButton.setVisible(False)          # the list is always on screen: there is nowhere to go back to
        self._page.crumbLabel.setVisible(False)
        self._page.back_requested.connect(self.entryTable.setFocus)
        self._page.title_shown.connect(self._on_page_moved)
        for signal in (self._page.decided, self._page.revised):
            signal.connect(lambda title_id, *_: self._on_page_wrote(title_id))
        self._page.changed.connect(self._on_page_wrote)
        self.stack.addWidget(self._page)
        self.entryTable.itemSelectionChanged.connect(self._on_row_selected)
        for key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):    # Enter on the list moves to the candidates: it decides nothing
            QShortcut(QKeySequence(key), self.entryTable, activated=self._page.candidateList.setFocus,
                      context=Qt.ShortcutContext.WidgetShortcut)
        folder = preferences.get(DESIGNER_QUEUE_DIR)
        if auto_load and folder and os.path.isdir(folder):
            self.load_queue_dir(folder, remember=False)
        else:
            self._refresh_buttons()

    # --- construction --------------------------------------------------------------------------------------------------------

    def _build(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        top = QHBoxLayout()
        top.addWidget(QLabel('Queue folder:'))
        self.folderEdit = QLineEdit()
        self.folderEdit.setReadOnly(True)
        self.folderEdit.setPlaceholderText('Choose the folder a Batch Extract & Design run wrote')
        self.browseButton = QPushButton('Browse...')
        self.refreshButton = QPushButton('Refresh')
        for button in (self.browseButton, self.refreshButton):
            button.setAutoDefault(False)
        self.browseButton.clicked.connect(lambda: self.browse())
        self.refreshButton.clicked.connect(lambda: self.refresh())
        top.addWidget(self.folderEdit, 1)
        top.addWidget(self.browseButton)
        top.addWidget(self.refreshButton)
        layout.addLayout(top)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        self.entryTable = QTableWidget(0, len(_COLUMNS))
        self.entryTable.setHorizontalHeaderLabels(list(_COLUMNS))
        self.entryTable.verticalHeader().setVisible(False)
        self.entryTable.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.entryTable.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.entryTable.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.entryTable.setShowGrid(False)
        self.entryTable.horizontalHeader().setStretchLastSection(True)
        self.entryTable.setColumnWidth(0, 180)
        self.entryTable.setColumnWidth(1, 80)
        self.entryTable.setMinimumWidth(370)
        self.entryTable.setMaximumWidth(460)
        splitter.addWidget(self.entryTable)
        self.stack = QStackedWidget()
        self.emptyLabel = QLabel()
        self.emptyLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.emptyLabel.setWordWrap(True)
        self.stack.addWidget(self.emptyLabel)
        splitter.addWidget(self.stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)
        self.resultsBox = QPlainTextEdit()
        self.resultsBox.setReadOnly(True)
        self.resultsBox.setMaximumHeight(110)
        self.resultsBox.setVisible(False)
        layout.addWidget(self.resultsBox)
        bottom = QHBoxLayout()
        self.statusLabel = QLabel()
        self.statusLabel.setWordWrap(True)
        self.publishButton = QPushButton('Publish accepted')
        self.commitButton = QPushButton('Commit published')
        for button in (self.publishButton, self.commitButton):
            button.setAutoDefault(False)
        self.publishButton.clicked.connect(lambda: self.publish_accepted())
        self.commitButton.clicked.connect(lambda: self.commit_published())
        bottom.addWidget(self.statusLabel, 1)
        bottom.addWidget(self.publishButton)
        bottom.addWidget(self.commitButton)
        layout.addLayout(bottom)
        self.setCentralWidget(root)

    # --- what the page asks for ----------------------------------------------------------------------------------------------

    def _setup_now(self, max_age: float = 0.0) -> WorkListSetup:
        '''
        The library profile as it is now: read again (a repository may have been set, or one removed, in the work list since)
        unless it was read less than `max_age` seconds ago. The page asks for the metadata defaults on every keystroke, so it
        does not read the profile file each time.
        '''
        now = self._clock()
        if now - self._setup_read >= max_age:
            self._setup, self._setup_read = load_setup(self._preferences), now
        return self._setup

    def _work_dir(self) -> str:
        settings = self._setup_now().settings
        return settings.work_dir if settings is not None else ''

    def _meta_defaults(self) -> Optional[dict]:
        settings = self._setup_now(1.0).settings
        return settings.meta_defaults if settings is not None else None

    def _revise_context(self) -> Optional[ReviseContext]:
        ''' The folder is the queue; the repositories and the work directory are the library profile's, read as they are now. '''
        if not self._queue_dir:
            return None
        settings = self._setup_now().settings
        if settings is None:
            return ReviseContext(self._queue_dir)
        from pipeline.publish.git import RepoTarget
        return ReviseContext(self._queue_dir, settings.work_dir, RepoTarget(settings.xml_repo) if settings.xml_repo else None,
                             RepoTarget(settings.images_repo) if settings.images_repo else None, settings.xml_dir,
                             settings.image_dir)

    def _revise_blocked(self, title_id: str, status: str) -> str:
        '''
        The work list's rule: an accepted or published title's files are in the repositories a Publish or Commit is using, so it is
        not revised while one runs. (The titles the job is working on are refused for every purpose: they are "running".)
        '''
        if self._job is not None and status in ('accepted', 'published'):
            return 'A publish or commit is going and it uses the catalogue repositories: revise this after it finishes.'
        return ''

    def _commit_state(self, title_id: str) -> str:
        ''' Where this title's files are in git *now* (the revise question says what will happen to them). '''
        settings = self._setup_now().settings
        if settings is None:
            return 'none'
        return commit_states([title_id], settings.xml_repo or '', settings.xml_dir, settings.image_dir).get(title_id, 'none')

    @property
    def page(self) -> TitlePage:
        return self._page

    @property
    def queue_dir(self) -> str:
        return self._queue_dir

    @property
    def entry_ids(self) -> List[str]:
        ''' The entries listed, in the order shown. '''
        return list(self._order)

    @property
    def is_busy(self) -> bool:
        return self._job is not None

    # --- the folder ------------------------------------------------------------------------------------------------------------

    def _ask_dir(self, start: str) -> str:
        return QFileDialog.getExistingDirectory(self, 'Choose the queue folder', start)

    def browse(self) -> bool:
        ''' Asks for the folder and loads it. :return: False if none was chosen or it could not be loaded. '''
        chosen = self._choose_dir(self._queue_dir or self._preferences.get(DESIGNER_QUEUE_DIR) or '')
        return bool(chosen) and self.load_queue_dir(chosen)

    def load_queue_dir(self, path: str, remember: bool = True) -> bool:
        '''
        Lists the entries of `path` and shows the first that is waiting for a decision (else the first). What was being
        edited on the page is saved first; if it cannot be, the person is asked, and the folder stays as it was if not.
        :return: False if the folder was not loaded -- an unsaved edit was kept, or it is not a folder.
        '''
        if self._job is not None:
            self._say('A publish or commit is going: choose another folder when it has finished.', True)
            return False
        if self._queue_dir and not self._page.leave():
            return False
        if not os.path.isdir(path):
            self._say(f'{path} is not a folder.', True)
            return False
        self._queue_dir = path
        self.folderEdit.setText(path)
        if remember:
            self._preferences.set(DESIGNER_QUEUE_DIR, path)
        self._read_folder()
        self._page.forget_decisions(revised=False)     # (the rows are read from the entries: they are current)
        self._show_first()
        return True

    def refresh(self) -> bool:
        ''' Reads the folder again (a run may have written entries), staying on the same title if it is still there. '''
        if not self._queue_dir or not self._page.leave():
            return False
        current = self._page.current_id
        self._read_folder()
        self._page.forget_decisions(revised=False)
        if current in self._entries:
            self._page.open(current, self._order)
        else:
            self._show_first()
        return True

    def _read_folder(self) -> None:
        try:
            entries = read_queue(self._queue_dir)
        except Exception as error:   # one damaged file: say so, and show what could be read
            logger.exception('Could not read the queue folder %s', self._queue_dir)
            self._say(f'The folder could not be read: {type(error).__name__}: {error}', True)
            entries = self._read_each()
        states = self._commit_states([e.id for e in entries if e.status == 'published'])
        self._entries = {e.id: e for e in entries}
        self._rows = {e.id: entry_row(e, states.get(e.id, 'none')) for e in entries}
        self._order = [e.id for e in entries]
        self._fill_table()

    def _read_each(self) -> List[QueueEntry]:
        entries = []
        for name in sorted(os.listdir(self._queue_dir)):
            if name.endswith('.json'):
                try:
                    entries.append(read_entry(self._queue_dir, name[:-5]))
                except Exception:   # the damaged one is left out; the others are still reviewable
                    logger.warning('Skipping the unreadable queue entry %s', name)
        entries.sort(key=lambda e: (e.status != 'pending', e.id))
        return entries

    def _fill_table(self) -> None:
        self._loading = True
        table = self.entryTable
        table.setRowCount(len(self._order))
        for row, title_id in enumerate(self._order):
            self._fill_row(row, title_id)
        self._loading = False
        self._refresh_buttons()

    def _fill_row(self, row: int, title_id: str) -> None:
        entry, view = self._entries[title_id], self._rows[title_id]
        confidence = f'{entry.candidates[0].confidence:.2f}' if entry.candidates else ''
        for column, text in enumerate((view.title, entry.status, confidence)):
            item = QTableWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, title_id)
            item.setToolTip(f'{view.title} ({title_id})')
            self.entryTable.setItem(row, column, item)

    def _show_first(self) -> None:
        if not self._order:
            self.emptyLabel.setText('This folder has no queue entries.\n\nBatch Extract & Design writes them when its '
                                    'Design filters step is ticked.')
            self.stack.setCurrentWidget(self.emptyLabel)
            self._say('')
            return
        first = next((i for i in self._order if self._entries[i].status == 'pending'), self._order[0])
        self.stack.setCurrentWidget(self._page)
        self._page.open(first, self._order)

    # --- keeping the list and the page in step -------------------------------------------------------------------------------

    def _row_of(self, title_id: str) -> int:
        return self._order.index(title_id) if title_id in self._order else -1

    def _on_page_moved(self, title_id: str) -> None:
        ''' The page went to another title (Next, or Accept & next): the list follows. '''
        row = self._row_of(title_id)
        if row >= 0 and row != self.entryTable.currentRow():
            self._loading = True
            self.entryTable.selectRow(row)
            self._loading = False
        self._refresh_buttons()

    def _on_row_selected(self) -> None:
        if self._loading:
            return
        rows = self.entryTable.selectionModel().selectedRows()
        if not rows:
            return
        title_id = self._order[rows[0].row()]
        if title_id != self._page.current_id and not self._page.show_title(title_id):
            self._on_page_moved(self._page.current_id)   # an edit could not be saved: stay, and show where we are

    def _commit_states(self, published: List[str]) -> Dict[str, str]:
        settings = self._setup_now(1.0).settings
        if settings is None:
            return {i: 'none' for i in published}
        return commit_states(published, settings.xml_repo or '', settings.xml_dir, settings.image_dir)

    def _reload_entries(self, ids: List[str]) -> None:
        ''' What is on disk for these entries is their row now (one look at git for all the published ones). '''
        fresh: Dict[str, QueueEntry] = {}
        for title_id in ids:
            try:
                fresh[title_id] = read_entry(self._queue_dir, title_id)
            except Exception:
                continue
        states = self._commit_states([i for i, e in fresh.items() if e.status == 'published'])
        for title_id, entry in fresh.items():
            self._entries[title_id], self._rows[title_id] = entry, entry_row(entry, states.get(title_id, 'none'))
            row = self._row_of(title_id)
            if row >= 0:
                self._fill_row(row, title_id)
        self._refresh_buttons()

    def _reload_entry(self, title_id: str) -> None:
        ''' A decision, an edit or a revise wrote the entry: its row is what is on disk now. '''
        self._reload_entries([title_id])

    def _on_page_wrote(self, title_id: str) -> None:
        '''
        The page wrote an entry. The row follows, and the page need no longer remember what it decided or edited (the rows here are
        read from the entries, so they are current) -- but it keeps which titles it sent back for redesign: nothing in the entry
        says it has been designed again until it has been, which the page sees for itself when it reads the entry.
        '''
        self._reload_entry(title_id)
        self._page.forget_decisions(revised=False)

    # --- publish and commit --------------------------------------------------------------------------------------------------

    def _ids_with(self, status: str) -> List[str]:
        return [i for i in self._order if self._entries[i].status == status]

    def _say(self, text: str, problem: bool = False) -> None:
        self.statusLabel.setText(text)
        self.statusLabel.setStyleSheet(f'color: {warning_colour().name()}' if problem else '')

    def _refresh_buttons(self) -> None:
        problem = publish_problem(self._setup)
        accepted, published = len(self._ids_with('accepted')), len(self._ids_with('published'))
        idle = self._job is None
        self.publishButton.setText(f'Publish accepted ({accepted:,})')
        self.commitButton.setText(f'Commit published ({published:,})')
        self.publishButton.setEnabled(idle and accepted > 0 and not problem)
        self.commitButton.setEnabled(idle and published > 0 and not problem)
        tip = problem or None
        self.publishButton.setToolTip(tip or 'Write the accepted titles into the catalogue repositories\' working trees '
                                              '(nothing is committed or pushed). Uses the repositories of the Library Work '
                                              'List settings.')
        self.commitButton.setToolTip(tip or 'Commit the written titles: one commit per repository, images first, then push.')
        self.refreshButton.setEnabled(idle and bool(self._queue_dir))

    def _reload_setup(self) -> None:
        '''
        The library profile again, and what depends on it shown again: the buttons, and the page (its badge judges the metadata
        with the profile's defaults). The settings may have changed in the work list since the window opened or was last used.
        '''
        before = self._setup.settings.meta_defaults if self._setup.settings is not None else None
        self._setup_now()
        after = self._setup.settings.meta_defaults if self._setup.settings is not None else None
        self._refresh_buttons()
        if self._page.current_id and after != before:
            self._page.reload()

    def publish_accepted(self) -> bool:
        '''
        Publish (write, not commit) every accepted entry of the folder, after a confirmation naming the repositories.
        :return: False if nothing was started: something is going, nothing is accepted, no repository is set, or the person declined.
        '''
        if self._job is not None or not self._page.flush():
            return False
        self._reload_setup()
        ids = self._ids_with('accepted')
        problem = publish_problem(self._setup)
        if problem or not ids:
            self._say(problem or 'No entry is accepted: there is nothing to publish.', True)
            return False
        settings = build_publish_settings(self._setup)
        heading, body = publish_text(len(ids), settings)
        dialog = ConfirmDialog(self, heading, body, f'Publish {len(ids):,} title{"" if len(ids) == 1 else "s"}')
        if dialog.exec() != ConfirmDialog.DialogCode.Accepted:
            self._say('Publish cancelled: nothing was written.')
            return False
        queue_dir, work_dir = self._queue_dir, self._setup.settings.work_dir or None
        config = self._setup.settings.config     # what a run's publish is given too (`run_config.config`)

        # An entry designed by the library run has its audio and `.beq` projects under the work directory, and publish reads
        # them (a person's edit is what is published). One designed by Batch Extract & Design has none there, and asking
        # publish to read projects that do not exist would fail it: those are published from their candidates. One whose project
        # is there but whose audio is gone is refused, not published from its candidate (which would drop the edit unnoticed).
        with_projects, without, refused = split_for_publish(work_dir, ids)

        def publish():
            results: List[dict] = list(refused)
            for chosen, where in ((with_projects, work_dir), (without, None)):
                if chosen:
                    results += publish_library(
                        queue_dir, settings.xml_repo, meta_defaults=settings.meta_defaults,
                        images_repo=settings.images_repo, image_owner=settings.image_owner,
                        image_repo_name=settings.image_repo_name, xml_dir=settings.xml_dir, image_dir=settings.image_dir,
                        report_spec=settings.report_spec, config=config, work_dir=where, ids=chosen)
            return results

        return self._start(publish, 'Publish', f'Publishing {len(ids):,} title{"" if len(ids) == 1 else "s"}...',
                           self._on_published, {i: 'publish' for i in ids})

    def commit_published(self) -> bool:
        '''
        Commit (and, if the box is ticked, push) the folder's published entries, after a confirmation naming the repositories.
        :return: False if nothing was started.
        '''
        if self._job is not None or not self._page.flush():
            return False
        self._reload_setup()
        ids = self._ids_with('published')
        problem = publish_problem(self._setup)
        if problem or not ids:
            self._say(problem or 'No entry is published: there is nothing to commit.', True)
            return False
        settings = build_publish_settings(self._setup)
        heading, body = commit_text(len(ids), settings)
        dialog = ConfirmDialog(self, heading, body, f'Commit {len(ids):,} title{"" if len(ids) == 1 else "s"}',
                               'Push each repository after committing (untick to commit locally only)',
                               bool(self._preferences.get(WORKLIST_PUSH)))
        if dialog.exec() != ConfirmDialog.DialogCode.Accepted:
            self._say('Commit cancelled: nothing was committed.')
            return False
        push = dialog.checked
        self._preferences.set(WORKLIST_PUSH, push)
        queue_dir = self._queue_dir

        def commit():
            return commit_library(queue_dir, settings.xml_repo, images_repo=settings.images_repo, xml_dir=settings.xml_dir,
                                  image_dir=settings.image_dir, push=push, ids=ids)

        return self._start(commit, 'Commit', f'Committing {len(ids):,} title{"" if len(ids) == 1 else "s"}...',
                           self._on_committed, {i: 'commit' for i in ids})

    def _start(self, work: Callable[[], object], what: str, message: str, done: Callable,
               working_on: Dict[str, str]) -> bool:
        job = _CallJob(work, what)
        job.signals.finished.connect(done)
        job.signals.errored.connect(self._on_failed)
        self._job, self._working_on = job, dict(working_on)
        self._say(message)
        self.resultsBox.setVisible(False)
        self._refresh_buttons()
        if self._page.current_id:
            self._page.refresh_decisions()     # the title on the page may be one the job is working on: no decision, edit or revise
        QThreadPool.globalInstance().start(job)
        return True

    def _end(self) -> None:
        self._job, self._working_on = None, {}
        self._reload_entries(list(self._order))
        if self._queue_dir and self._page.current_id:
            self._page.reload()
        self._refresh_buttons()

    def _on_published(self, results: List[dict]) -> None:
        self._end()
        _, refused = split_publish_results(results)
        self._say(describe_publish(results), bool(refused))
        self.resultsBox.setPlainText('\n'.join(describe_publish_error(r) for r in refused))
        self.resultsBox.setVisible(bool(refused))
        self.published.emit(results)

    def _on_committed(self, commit: CatalogueCommit) -> None:
        self._end()
        self._say(describe_commit(commit), bool(commit.not_committed))
        extras = [f'{name}: {w}' for name, w in (('Warning', w) for w in commit.warnings)] + \
                 [f'Not committed (git ignores it): {p}' for p in commit.not_committed]
        self.resultsBox.setPlainText('\n'.join(extras))
        self.resultsBox.setVisible(bool(extras))
        self.committed.emit(commit)

    def _on_failed(self, message: str) -> None:
        self._end()
        self._say(f'Failed: {message}. What was done before it is kept; see Help > Logs for the details.', True)
        self.failed.emit(message)

    def changeEvent(self, event) -> None:
        '''
        The window becoming active again: the person may have changed the repositories in the work list, or saved a project in the
        main window -- the profile is read again and the page's projects and badge with it.
        '''
        if event.type() == QEvent.Type.ActivationChange and self.isActiveWindow():
            self._reload_setup()
            if self._page.current_id:
                self._page.refresh_projects()
        super().changeEvent(event)

    def showEvent(self, event) -> None:
        self._reload_setup()
        super().showEvent(event)

    def closeEvent(self, event) -> None:
        ''' An edit that cannot be saved is put to the person (Discard / Cancel); a Publish or Commit in flight finishes. '''
        if self._queue_dir and not self._page.leave():
            event.ignore()
            return
        super().closeEvent(event)


def open_review_folder(owner, preferences, queue_dir: str) -> 'ReviewFolderWindow':
    '''
    Opens the Review folder window on `queue_dir` from a dialog that has just written entries there (Batch Extract & Design,
    Extract Audio's design step). Through the main window's one window when `owner`'s parent is the app (it has
    `showReviewFolderWindow`), else -- a dialog on its own, as in the tests -- a window of its own.
    '''
    parent = owner.parent() if owner is not None else None
    show = getattr(parent, 'showReviewFolderWindow', None)
    if callable(show):
        return show(queue_dir)
    window = ReviewFolderWindow(owner, preferences)
    window.load_queue_dir(queue_dir)
    window.show()
    return window
