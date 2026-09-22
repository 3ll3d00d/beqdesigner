'''
Opening the title page from the work list -- design/library-sync/workflow-rework/design.md §12.10, chunks 27a and 27b. A mixin of
`model.worklist.WorkListWindow` (which owns the widgets, the index and the setup):

* **Drill in** -- double-click, Enter on the table, or the *Open* button -- stacks `model.worklist_title.TitlePage` on
  `contentStack`, over the titles the table lists now, and hides the strip, the filters and the action row (they act on
  the table, which is not on screen). The setup and stale banners and the run panel stay: they are about the window, and
  a run can go on while a title is looked at.
* **Back** (Esc, the breadcrumb) returns to the table with the selection, the scroll position and the filters as they
  were. What is being edited on the page is saved first (`TitlePage.leave()`); if it cannot be, the person is asked
  whether to discard it and the page stays if not -- which also stops the window closing (`WorkListWindow.closeEvent`, which
  asks about a run in progress *first*, so that a Discard is only offered once the close is really going ahead).
* **The index is re-read when the page is left.** A decision changes what the title needs (`review` -> `publish`, or
  nothing), and the index only learns that by reading the outputs again (`LibraryIndex.refresh()`), which is a pass over
  every title: too much to do for each Accept & next, so it is done once, on a worker with its own connection, when the
  page is left (or the window closed), and after a scan or run that was going while the decisions were made. The same
  goes for **an edit of a title's metadata or artwork** (`TitlePage.changed`): it changes what the index says of the title
  (metadata incomplete, Publish: out of date). An artwork download that finishes after the page was left starts the read
  at once.
'''
import logging
import os
from typing import Optional

from qtpy.QtCore import QItemSelectionModel, QObject, QRunnable, Qt, QThreadPool, Signal
from qtpy.QtGui import QKeySequence, QShortcut
from qtpy.QtWidgets import QAbstractItemView

from model.worklist_model import ID_ROLE
from model.worklist_revise import revise_context
from model.worklist_title import TitlePage
from model.worklist_title_actions import TitleHooks
from pipeline.library.index import LibraryIndex
from pipeline.library.profile import Profile
from pipeline.library.status import ScanSettings

logger = logging.getLogger('worklist')


class _SyncSignals(QObject):
    finished = Signal()
    errored = Signal(str)


class _SyncJob(QRunnable):
    ''' `LibraryIndex.refresh()` on a worker thread with a connection of its own (as `_ScanJob` does for a scan). '''

    def __init__(self, path: str, profile: Profile, settings: ScanSettings):
        super().__init__()
        self.signals = _SyncSignals()
        self._path, self._profile, self._settings = path, profile, settings

    def run(self):
        try:
            with LibraryIndex(self._path) as index:
                index.refresh(self._profile, self._settings)
            self.signals.finished.emit()
        except Exception as error:
            logger.exception('Could not update the library index after reviewing')
            self.signals.errored.emit(f'{type(error).__name__}: {error}')


class WorkListTitles:
    '''
    The mixin: it uses `contentStack`, `tablePage`, `workTable`, `listHeader`, `listFooter`, `detailsTabs`, `openButton`,
    `_proxy`, `_setup`, `_preferences`, `_rows_by_id()`, `_busy()`, `_refresh_view()`, `_refresh_details()` and
    `refresh_from_index()` of the window, and the attributes `_title_page`, `_title_open`, `_index_dirty` and `_syncing`,
    which the window sets before anything else can call in.
    '''

    def _configure_titles(self) -> None:
        self.workTable.doubleClicked.connect(lambda index: self.open_title(index.data(ID_ROLE)))
        for key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            # scoped to the table: a window-wide Enter would also fire from the search box
            QShortcut(QKeySequence(key), self.workTable, activated=lambda: self.open_title(),
                      context=Qt.ShortcutContext.WidgetShortcut)
        self.openButton.clicked.connect(lambda: self.open_title())
        # a run starting or leaving a title changes which decisions the page may offer for it
        self._model.dataChanged.connect(lambda *_: self._title_open and self._title_page.refresh_decisions())

    @property
    def title_page(self) -> Optional[TitlePage]:
        ''' The page, once it has been opened (it is built the first time: the chart is not free). '''
        return self._title_page

    @property
    def title_page_open(self) -> bool:
        return self._title_open

    def _refresh_open_button(self) -> None:
        self.openButton.setEnabled(self._proxy.rowCount() > 0)

    def open_title(self, title_id: Optional[str] = None) -> bool:
        '''
        Shows the title page over the titles the table lists now.
        :param title_id: the title to open; default the first selected, else the table's current row, else the first
            listed. It must be listed.
        :return: False if there is nothing to open.
        '''
        ids = self.listed_ids()
        if title_id is None:
            selected, current = self.selected_ids(), self.workTable.currentIndex()
            title_id = (selected[0] if selected else None) or (current.data(ID_ROLE) if current.isValid() else None) \
                or (ids[0] if ids else None)
        if title_id is None or title_id not in ids:
            return False
        if self._title_page is None:
            hooks = TitleHooks(open_project=self._open_project, work_dir=self._title_work_dir,
                               revise_context=lambda: revise_context(self._setup), revise_blocked=self._revise_blocked,
                               retry_failed=lambda title_id: self.retry_failed([title_id]),
                               open_jriver_preferences=self.preferences_requested.emit)
            self._title_page = TitlePage(self, self._preferences, self._title_queue_dir, self._rows_by_id,
                                         lambda: self._model.running, self._title_meta_defaults, hooks=hooks)
            self._title_page.back_requested.connect(self.close_title)
            self._title_page.decided.connect(self._on_title_decided)
            self._title_page.revised.connect(self._on_title_revised)
            self._title_page.changed.connect(self._on_title_changed)
            self._title_page.notice.connect(lambda text: self.statusBar.showMessage(text, 15000))
            self.contentStack.addWidget(self._title_page)
        self._title_open = True
        for widget in (self.listHeader, self.listFooter, self.detailsTabs):
            widget.setVisible(False)
        self.contentStack.setCurrentWidget(self._title_page)
        self._title_page.open(title_id, ids)
        self._refresh_open_button()
        return True

    def _title_queue_dir(self) -> str:
        settings = self._setup.settings
        return settings.queue_dir if settings is not None else ''

    def _title_work_dir(self) -> str:
        settings = self._setup.settings
        return settings.work_dir if settings is not None else ''

    def _revise_blocked(self, title_id: str, status: str) -> str:
        '''
        Why a title cannot be sent back now, or ''. Not one a run is working on (a design in flight writes a new entry over
        whatever is there), and -- for an accepted or published one, whose files are in the catalogue repositories -- not
        while a publish or commit run is using them.
        '''
        if title_id in self._model.running:
            return 'A run is working on this title now: wait for it to finish.'
        context = self._run_context
        if self.is_running and context is not None and context.request.through in ('publish', 'commit') \
                and status in ('accepted', 'published'):
            return 'A publish or commit run is going and it uses the catalogue repositories: revise this after it finishes.'
        return ''

    def _title_meta_defaults(self) -> Optional[dict]:
        ''' What publish fills in where a title's metadata is silent: the badge judges the metadata as publish would. '''
        settings = self._setup.settings
        return settings.meta_defaults if settings is not None else None

    def close_title(self, sync: bool = True) -> bool:
        '''
        Back to the table: the selection, scroll position and filters are as they were left; the row of the title last
        looked at is made current (so Enter opens it again) without changing the selection. What was being edited is saved
        first; if it cannot be, the person is asked whether to discard it and the page stays if not. The index is read
        again if anything was decided or edited.
        :param sync: False when the window is closing, which reads the index itself.
        :return: False if the page was not open, or was not left (an edit that could not be saved and was not discarded).
        '''
        if not self._title_open or not self._title_page.leave():
            return False
        viewed = self._title_page.current_id
        self._title_open = False
        self.listHeader.setVisible(True)
        self.listFooter.setVisible(True)
        self._refresh_view()        # puts the table (or the reason there is none) back on the stack
        self._refresh_details()
        for row in range(self._proxy.rowCount()):
            index = self._proxy.index(row, 0)
            if index.data(ID_ROLE) == viewed:
                self.workTable.selectionModel().setCurrentIndex(index, QItemSelectionModel.SelectionFlag.NoUpdate)
                self.workTable.scrollTo(index, QAbstractItemView.ScrollHint.EnsureVisible)
                break
        self.workTable.setFocus()
        if sync:
            self._sync_index_if_dirty()
        return True

    def _on_title_decided(self, title_id: str, status: str) -> None:
        self._index_dirty = True
        self.statusBar.showMessage(f'{title_id}: {status}')

    def _on_title_revised(self, title_id: str, to: str) -> None:
        ''' A title was sent back on the page: what it needs is stale until the index reads it, as after a decision. '''
        self._index_dirty = True
        self.statusBar.showMessage(f'{title_id}: {"reopened for review" if to == "review" else "sent back for " + to}')
        if not self._title_open:
            self._sync_index_if_dirty()

    def _on_title_changed(self, title_id: str) -> None:
        '''
        A title's metadata or artwork was written: what the index says of it (metadata incomplete, Publish: out of date) is
        stale. The read happens when the page is left; if that is already so (a download came back after it), now.
        '''
        self._index_dirty = True
        self.statusBar.showMessage(f'{title_id}: saved')
        if not self._title_open:
            self._sync_index_if_dirty()

    # --- the index, after decisions ---------------------------------------------------------------------------------------

    def _sync_index_if_dirty(self) -> bool:
        '''
        Reads the outputs of every title again (`LibraryIndex.refresh`) on a worker, if a decision was made since the last
        time. Left for later if a scan or a run is going (they read the outputs themselves, and are asked again when they
        end), or another read is: two writers to one index file are not needed.
        :return: True if a job was started.
        '''
        setup = self._setup
        if not self._index_dirty or self._busy() or not setup.ready or setup.index_file is None:
            return False
        if not os.path.isfile(setup.index_file):   # e.g. the work directory was changed: nothing to bring up to date, and
            self._index_dirty = False               # opening a LibraryIndex would create the file
            return False
        job = _SyncJob(setup.index_file, setup.profile, setup.settings)
        job.signals.finished.connect(self._on_index_synced)
        job.signals.errored.connect(self._on_index_sync_failed)
        self._index_dirty, self._syncing = False, True
        self._refresh_actions()
        QThreadPool.globalInstance().start(job)
        return True

    def _on_index_synced(self) -> None:
        self._syncing = False
        self.refresh_from_index()    # also refreshes the buttons, which were disabled while the read went on
        self.index_synced.emit()
        self._sync_index_if_dirty()  # decisions made while it was going

    def _on_index_sync_failed(self, message: str) -> None:
        self._syncing, self._index_dirty = False, True   # what was decided is still not in the index
        self.refresh_from_index()
        self.statusBar.showMessage(f'The list may be out of date: it could not be updated after reviewing ({message}). '
                                   f'Rescan to bring it up to date.')

    def _sync_index_now(self) -> None:
        '''
        On closing: the same read, on this thread, since there is no window left to wait for a worker. Not while a scan, a
        run or a read is going -- they, and the read when they end (which also happens after the window is closed), do it.
        '''
        setup = self._setup
        if not self._index_dirty or self._busy() or not setup.ready or self._index is None:
            return
        self._index_dirty = False
        try:
            self._index.refresh(setup.profile, setup.settings)
        except Exception:
            self._index_dirty = True   # still not in the index: the next page leave or scan tries again
            logger.exception('Could not update the library index after reviewing')
