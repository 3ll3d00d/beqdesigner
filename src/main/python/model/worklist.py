'''
The library work list -- design/library-sync/workflow-rework/design.md §12.10, chunk 26a (read-only).

A top-level window over the discovery index (`pipeline.library.index.LibraryIndex`): the pipeline strip with a count per
kind of work, a searchable, filterable table of every title in the library and what it needs next, and a Rescan that
lists the sources again. It is the intended replacement for `model.library_sync.LibrarySyncDialog`, which stays
untouched and working until chunk 27c. Chunk 26a made it read-only; chunk 26b (below) gives it actions, but **it accepts
nothing**: reviewing is a person's job (the title page, chunk 27).

What the window reads and where it does not think for itself:

* what each title needs, its detail, how long it has waited, whether it is new, its flags -- all straight from the
  index's rows (`TitleRow`), in the index's order (tier, then oldest first); see `model.worklist_model`;
* the profile -- from the `LIBRARY_PROFILE_PATH` file if there is one, else (until the first change is saved) built from
  the Library Sync preferences (`model.worklist_profile`); chunk 26c's **settings drawer** (`model.worklist_settings`, a
  dock on the right: *Settings...*) edits that file, and a banner at the top says what is missing until it is complete;
* the strip's chips are `pipeline.library.selection.CHIPS`, and `current_selection()` is the `Selection` the filters
  amount to, which chunk 26b's action button hands to `plan_stages()`/`run_stages()`.

Actions (chunk 26b; `model.worklist_run` and `model.worklist_confirm` hold what is not a widget):

* the rows selected (or, with none selected, everything the filters list) are what the buttons work on;
* **the action button** is "run through design": `plan_stages()` says what it would do and its label says how many
  titles and what it leaves out; **Publish** and **Commit** are their own buttons, each behind a confirmation that names
  the repositories; **Retry failed** (the failures panel) runs the titles whose remembered failure would otherwise not be
  tried again;
* a run is a `RunJob` on the global thread pool, with a connection of its own to the index, that calls
  `pipeline.library.stages.run_stages()`: progress is determinate, a row shows its stage while it is worked on, **Cancel**
  is cooperative (the title in hand finishes; a cancelled run never commits), the pipeline refreshes the index when it
  ends -- after a cancel or a failure too -- and the window then reads the index again (`refresh_from_index()`), keeping
  the selection, and lists what happened to each title on the *Last run* tab.

The cached index is shown at once (stale-while-revalidate). A rescan runs on a QRunnable in the global thread pool with its
own index connection, so the UI thread never waits for a slow source; the window rescans by itself only when the
index has never been scanned.
'''
import html
import logging
import os
import time
from typing import Callable, Dict, List, Mapping, Optional

from qtpy.QtCore import QItemSelectionModel, QObject, QPoint, QRunnable, Qt, QThreadPool, Signal
from qtpy.QtGui import QKeySequence, QShortcut
from qtpy.QtWidgets import QAbstractItemView, QButtonGroup, QDockWidget, QHeaderView, QInputDialog, QMainWindow, QMenu, \
    QMessageBox, QProgressBar, QSizePolicy

from model.preferences import WORKLIST_GEOMETRY
from model.worklist_actions import WorkListActions
from model.worklist_model import ALL_CHIPS, CHIP_ALL, CHIP_DONE, COL_DETAIL, COL_NEEDS, COL_SOURCE, COL_TITLE, \
    COL_WAITING, COL_YEAR, ID_ROLE, WorkListModel, WorkListProxy, warning_colour
from model.worklist_edit import discovery_changed
from model.worklist_profile import WorkListSetup, load_setup
from model.worklist_run import FailedTitle, ResultLine, RunJob, failed_titles
from model.worklist_settings import SettingsDrawer
from pipeline.library.index import LibraryIndex, ScanResult, SourceRow, TitleRow
from pipeline.library.profile import Profile
from pipeline.library.selection import CHIP_NEW, Selection, selection_from_chip
from pipeline.library.source import LibrarySource
from pipeline.library.stages import run_stages
from pipeline.library.state import NEEDS
from pipeline.library.status import ScanSettings
from ui.worklist import Ui_workListWindow

logger = logging.getLogger('worklist')

DRAWER_WIDTH = 520   # the settings drawer's width, in pixels

# palette() roles, so the strip follows a light or a dark platform theme
_CHIP_STYLE = '''
QPushButton { background: palette(button); color: palette(button-text); border: 1px solid palette(mid);
              border-radius: 4px; padding: 4px 10px; }
QPushButton:hover { border-color: palette(highlight); }
QPushButton:checked { background: palette(highlight); color: palette(highlighted-text); border-color: palette(highlight); }
'''

_CHIP_TIPS = {
    CHIP_ALL: 'Everything except Done',
    'Attention': 'A person must look: extract or design failed, the projects disagree, or the source changed since '
                 'the title was accepted',
    CHIP_NEW: 'Titles first seen by the latest scan',
    'Extract': 'Audio still to extract: new, or the source or the settings changed',
    'Design': 'Extracted, and a filter still to design',
    'Review': 'Designed, and waiting for a person to choose a candidate (or to complete its metadata)',
    'Publish': 'Accepted and not written to the catalogue repository yet, or out of date there',
    'Commit': 'Written to the catalogue repository, and not yet committed and pushed',
    CHIP_DONE: 'Pushed, skipped, rejected, ignored, shadowed by another source or gone from its source. '
               'Hidden from every other view.',
}


def format_time(when: float, now: float) -> str:
    ''' 09:14 for today, else 2026-09-18 09:14, in local time. '''
    moment, today = time.localtime(when), time.localtime(now)
    if moment[:3] == today[:3]:
        return time.strftime('%H:%M', moment)
    return time.strftime('%Y-%m-%d %H:%M', moment)


def describe_sources(sources: List[SourceRow], now: float) -> List[tuple]:
    '''
    What the index recorded for each source, as (text, is_error): when it was last listed, how many titles it gave, or why
    it could not be listed and which earlier listing the titles come from.
    '''
    lines = []
    for source in sources:
        if source.last_error:
            kept = (f'Showing its listing from {format_time(source.last_ok, now)}.' if source.last_ok
                    else 'It has never been listed, so it has no titles here.')
            lines.append((f'{source.name}: could not be listed ({source.last_error}). {kept}', True))
        elif source.last_scanned is None:
            lines.append((f'{source.name}: not scanned yet', False))
        else:
            lines.append((f'{source.name}: scanned {format_time(source.last_scanned, now)}, '
                          f'{source.item_count:,} title{"" if source.item_count == 1 else "s"}', False))
    return lines


class _ScanSignals(QObject):
    finished = Signal(object)  # ScanResult
    errored = Signal(str)


class _ScanJob(QRunnable):
    '''
    Scans on a worker thread with a connection of its own to the index file, so the window's connection (and the UI
    thread) is never held up behind a slow source: a scan only writes at its very end.
    '''

    def __init__(self, path: str, profile: Profile, settings: ScanSettings, only: Optional[List[str]],
                 sources: Optional[Mapping[str, LibrarySource]]):
        super().__init__()
        self.signals = _ScanSignals()
        self._path, self._profile, self._settings = path, profile, settings
        self._only, self._sources = only, sources

    def run(self):
        try:
            with LibraryIndex(self._path) as index:
                result = index.scan(self._profile, self._settings, only=self._only, sources=self._sources)
            self.signals.finished.emit(result)
        except Exception as error:
            logger.exception('Library scan failed')
            self.signals.errored.emit(f'{type(error).__name__}: {error}')


class WorkListWindow(WorkListActions, QMainWindow, Ui_workListWindow):
    '''
    :param parent: the main window, or None.
    :param preferences: `model.preferences.Preferences`; read again by reload().
    :param auto_scan: rescan on opening if the index has never been scanned.
    :param sources: already-built sources by profile name, handed to the scan (tests); by default they are built from the
        profile's settings.
    :param clock: the time source, for "waiting" and "last scan" (tests).
    :param run_stages_fn: what a run calls -- `run_stages` (tests hand in a fake with the same signature).
    :param precheck: called with the `through` of an extract/design run before it starts; False refuses it (the app checks
        that ffmpeg is installed).
    :param choose_profile_path: `(default, overwrite_ok) -> path`, asks where the profile file goes (a file dialog by default).
    :param run_dialog: how the drawer runs its source and ignore-rule dialogs (tests fill them in and accept them).
    :param settings_debounce_ms: how long after an edit the drawer writes the profile file.
    '''
    settings_requested = Signal()      # a Settings... button: the window opens the drawer (open_settings); also emitted, for the app
    preferences_requested = Signal()   # the drawer's link to Preferences (the TMDB key): the app opens them
    settings_saved = Signal(str)       # the drawer wrote the profile file (its path), and the window has read it
    scan_finished = Signal(object)     # a ScanResult, once the list shows it
    scan_failed = Signal(str)
    run_started = Signal(object)       # the RunRequest
    run_finished = Signal(object)      # the StagesReport, once the list shows the result (also after a cancel)
    run_failed = Signal(str)           # the run raised

    def __init__(self, parent, preferences, *, auto_scan: bool = True,
                 sources: Optional[Mapping[str, LibrarySource]] = None, clock=time.time,
                 run_stages_fn: Callable = run_stages, precheck: Optional[Callable[[str], bool]] = None,
                 choose_profile_path: Optional[Callable[[str, bool], str]] = None, run_dialog=None,
                 settings_debounce_ms: int = 400):
        super().__init__(parent)
        self._drawer: Optional[SettingsDrawer] = None
        self._dock: Optional[QDockWidget] = None
        self._stale = False         # the settings changed what a scan says, and none has run since
        self.setupUi(self)
        self._preferences = preferences
        self._auto_scan = auto_scan
        self._sources = sources
        self._clock = clock
        self._run_stages, self._precheck = run_stages_fn, precheck
        self._job: Optional[RunJob] = None
        self._run_context = None   # the run in flight (worklist_actions._RunContext)
        self._failed: List[FailedTitle] = []
        self._results: List[ResultLine] = []
        self._setup: WorkListSetup = WorkListSetup(None, None, 'preferences')
        self._index: Optional[LibraryIndex] = None
        self._index_file: Optional[str] = None
        self._index_error = ''      # why an index file that exists could not be opened or read
        self._closed = False        # closed (hidden) by the person: nothing may reopen the index until it is shown again
        self._sources_seen: List[SourceRow] = []
        self._last_scan_at: Optional[float] = None
        self._generation = 0
        self._scanning = False
        self._active_job: Optional[_ScanJob] = None
        self._sort_column = -1
        self._sort_order = Qt.SortOrder.AscendingOrder

        self._model = WorkListModel(self, clock)
        self._proxy = WorkListProxy(self)
        self._proxy.setSourceModel(self._model)
        self._configure_table()
        self._configure_chips()
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setMaximumWidth(140)
        self._progress.setVisible(False)
        self.statusBar.addPermanentWidget(self._progress)

        self._configure_actions()
        self._configure_settings(choose_profile_path, run_dialog, settings_debounce_ms)
        self.searchEdit.textChanged.connect(self._on_search)
        self.sourceCombo.currentIndexChanged.connect(self._on_source)
        self.rescanButton.clicked.connect(lambda: self.rescan())
        for button in (self.openSettingsButton, self.setupBannerButton, self.settingsButton):
            button.clicked.connect(self.settings_requested.emit)
        self.settings_requested.connect(lambda: self.open_settings())
        self.staleBannerButton.clicked.connect(lambda: self.rescan())
        QShortcut(QKeySequence('Ctrl+F'), self, activated=self.searchEdit.setFocus)
        geometry = preferences.get(WORKLIST_GEOMETRY)
        if geometry is not None:
            self.restoreGeometry(geometry)
        self.reload()

    # --- construction ---------------------------------------------------------------------------------------------

    def _configure_table(self):
        table = self.workTable
        table.setModel(self._proxy)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setAlternatingRowColors(True)
        table.setWordWrap(False)
        table.setTextElideMode(Qt.TextElideMode.ElideRight)
        table.setShowGrid(False)
        table.verticalHeader().setVisible(False)
        header = table.horizontalHeader()
        header.setStretchLastSection(False)
        for column, mode in ((COL_TITLE, QHeaderView.ResizeMode.Interactive),
                             (COL_YEAR, QHeaderView.ResizeMode.ResizeToContents),
                             (COL_SOURCE, QHeaderView.ResizeMode.ResizeToContents),
                             (COL_NEEDS, QHeaderView.ResizeMode.ResizeToContents),
                             (COL_DETAIL, QHeaderView.ResizeMode.Stretch),
                             (COL_WAITING, QHeaderView.ResizeMode.ResizeToContents)):
            header.setSectionResizeMode(column, mode)
        header.resizeSection(COL_TITLE, 300)
        header.setSectionsClickable(True)
        header.setSortIndicatorShown(True)
        header.setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
        header.sectionClicked.connect(self._on_header_clicked)
        self._proxy.modelReset.connect(self._refresh_view)
        self._proxy.layoutChanged.connect(self._refresh_view)

    def _configure_chips(self):
        self._chip_buttons = {
            CHIP_ALL: self.allChip, 'Attention': self.attentionChip, CHIP_NEW: self.newChip,
            'Extract': self.extractChip, 'Design': self.designChip, 'Review': self.reviewChip,
            'Publish': self.publishChip, 'Commit': self.commitChip, CHIP_DONE: self.doneChip}
        assert tuple(self._chip_buttons) == ALL_CHIPS
        self._chip_group = QButtonGroup(self)
        self._chip_group.setExclusive(True)
        for chip, button in self._chip_buttons.items():
            self._chip_group.addButton(button)
            button.setToolTip(_CHIP_TIPS[chip])
            button.setStyleSheet(_CHIP_STYLE)
            sample = f'{chip} 0,000' + (' (hidden)' if chip == CHIP_DONE else '')
            button.setMinimumWidth(button.fontMetrics().horizontalAdvance(sample) + 30)
            button.toggled.connect(lambda checked, name=chip: checked and self.set_chip(name))

    # --- settings ---------------------------------------------------------------------------------------------------

    def _configure_settings(self, choose_profile_path, run_dialog, debounce_ms: int) -> None:
        '''
        The drawer is a dock on the right, closed until asked for, so it never blocks the list. It edits the profile file;
        `_on_settings_saved` reads the file again once it is written.
        '''
        self._drawer = SettingsDrawer(self, self._preferences, rows_provider=lambda: self._model.rows,
                                      choose_path=choose_profile_path, run_dialog=run_dialog, debounce_ms=debounce_ms)
        self._dock = QDockWidget('Settings', self)
        self._dock.setObjectName('settingsDock')
        self._dock.setWidget(self._drawer)
        self._dock.setMinimumWidth(380)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._dock)
        self._dock.setVisible(False)
        self._drawer.saved.connect(self._on_settings_saved)
        self._drawer.profile_file_changed.connect(self._on_profile_file_changed)
        self._drawer.preferences_requested.connect(self.preferences_requested)
        self.setupBanner.setObjectName('setupBanner')
        colour = warning_colour().name()
        self.setupBanner.setStyleSheet(f'QFrame#setupBanner {{ border: 1px solid {colour}; border-radius: 4px; }}')
        self.staleBanner.setObjectName('staleBanner')
        self.staleBanner.setStyleSheet('QFrame#staleBanner { border: 1px solid palette(highlight); border-radius: 4px; }')
        for button in (self.setupBannerButton, self.staleBannerButton):
            button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setupBanner.setVisible(False)
        self.staleBanner.setVisible(False)
        self.ignoreButton.setMenu(self._build_ignore_menu())
        self.workTable.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.workTable.customContextMenuRequested.connect(self._on_table_menu)

    @property
    def drawer(self) -> SettingsDrawer:
        ''' The settings editor (chunk 27 reads the current profile from `drawer.profile`, or `setup.profile`). '''
        return self._drawer

    @property
    def settings_dock(self) -> QDockWidget:
        return self._dock

    @property
    def settings_stale(self) -> bool:
        ''' True if the settings changed what a scan says and none has run since (the banner offers a Rescan). '''
        return self._stale

    def open_settings(self, tab: Optional[str] = None) -> SettingsDrawer:
        '''
        Shows the settings drawer (beside the list: it does not block it). The list needs about 860 pixels to read (the
        strip's chips and the buttons), so where the screen has room the window **grows by the drawer's width** and the
        drawer docks on its right; where it does not (a 1000-pixel window on a small screen) the drawer floats, over the
        window's right edge, and can be moved.
        :param tab: `locations`, `sources` or `ignore`.
        '''
        if not self._dock.isVisible():
            self._place_dock()
        self._dock.show()
        self._dock.raise_()
        if tab:
            self._drawer.select_tab(tab)
        return self._drawer

    def _available_width(self) -> int:
        screen = self.screen()
        return screen.availableGeometry().width() if screen is not None else 0

    def _place_dock(self) -> None:
        if self.width() + DRAWER_WIDTH <= self._available_width():
            self._dock.setFloating(False)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._dock)
            self.resize(self.width() + DRAWER_WIDTH, self.height())
            self.resizeDocks([self._dock], [DRAWER_WIDTH], Qt.Orientation.Horizontal)
        else:
            self._dock.setFloating(True)
            self._dock.resize(DRAWER_WIDTH, min(max(self.height() - 40, 560), 720))
            self._dock.move(self.mapToGlobal(QPoint(max(self.width() - DRAWER_WIDTH, 0), 30)))

    def _flush_settings(self) -> None:
        ''' Writes a setting edited a moment ago now, before something that reads it (a scan, a run, closing). '''
        if self._drawer is not None:
            self._drawer.flush()

    def _on_settings_saved(self, path: str) -> None:
        '''
        The profile file was written: read it back (so what runs is what is on disk), and say when the index no longer
        describes it. **The list is not rescanned by itself**: a JRiver listing can be slow and a person often makes several
        changes in a row, so a banner offers *Rescan now* whenever a source, an ignore rule, an ignored title or a scan
        setting changed (rules and ignores take effect on titles only at a scan; the index is a cache of the last one).
        A new work directory has no index yet, so it is scanned once, as for any library never scanned.
        '''
        before = self._setup
        self._apply_setup(load_drawer=False)
        if discovery_changed(before, self._setup):
            self._stale = True
            self._refresh_banners()
        self.statusBar.showMessage(f'Settings saved to {path}')
        self.settings_saved.emit(path)

    def _on_profile_file_changed(self, path: str) -> None:
        self._stale = False
        self.reload()
        self._stale = self._generation > 0
        self._refresh_banners()

    def _refresh_settings_state(self) -> None:
        ''' Called whenever the buttons refresh: the drawer waits for a run, and the ignore actions follow the selection. '''
        if self._drawer is None:
            return
        self._drawer.set_busy(self._busy())
        selected = self.selected_ids()
        usable = self._setup.profile is not None and not self._busy()
        self.ignoreButton.setEnabled(bool(selected) and usable)
        for action in self.ignoreButton.menu().actions():
            action.setEnabled(usable and (len(selected) == 1 if action is self._ignore_like_action else bool(selected)))

    def _build_ignore_menu(self) -> QMenu:
        menu = QMenu(self)
        self._ignore_like_action = menu.addAction('Ignore titles like this...', lambda: self.ignore_like_selected())
        self._ignore_title_action = menu.addAction('Ignore this title...', lambda: self.ignore_selected_titles())
        self._ignore_like_action.setToolTip('Open the ignore-rule editor filled from this title (its folder, kind, year '
                                            'and title)')
        return menu

    def _on_table_menu(self, position) -> None:
        index = self.workTable.indexAt(position)
        if not index.isValid():
            return
        if index.row() not in {i.row() for i in self.workTable.selectionModel().selectedRows()}:
            self.workTable.selectRow(index.row())
        self._refresh_settings_state()
        self.ignoreButton.menu().exec(self.workTable.viewport().mapToGlobal(position))

    def ignore_like_selected(self) -> bool:
        '''
        "Ignore titles like this...": the rule editor, filled from the one selected row (its folder, kind, year, title and
        source), with "Ignore just this title" as the other choice. Either is written to the profile file.
        :return: False if not exactly one title is selected, or the editor was cancelled.
        '''
        selected = self.selected_ids()
        row = self._rows_by_id().get(selected[0]) if len(selected) == 1 else None
        if row is None or self._drawer.profile is None:
            return False
        self.open_settings('ignore')
        return self._drawer.ignore_like(row)

    def ignore_selected_titles(self, reason: Optional[str] = None) -> bool:
        '''
        Ignores the selected titles one by one (the profile's `ignore_titles`: id -> reason).
        :param reason: why; asked for when None.
        '''
        ids = self.selected_ids()
        if not ids or self._drawer.profile is None:
            return False
        if reason is None:
            reason, accepted = QInputDialog.getText(
                self, 'Ignore titles', f'Why ignore {len(ids):,} title{"" if len(ids) == 1 else "s"}? (optional)')
            if not accepted:
                return False
        self._drawer.ignore_titles(ids, reason.strip())
        return True

    # --- state ------------------------------------------------------------------------------------------------------

    @property
    def setup(self) -> WorkListSetup:
        return self._setup

    @property
    def is_scanning(self) -> bool:
        return self._scanning

    @property
    def chip(self) -> str:
        return self._proxy.chip

    @property
    def proxy(self) -> WorkListProxy:
        return self._proxy

    @property
    def model(self) -> WorkListModel:
        return self._model

    def chip_counts(self) -> Dict[str, int]:
        ''' The number on each chip, under the current source and search. '''
        return self._proxy.counts()

    def listed_ids(self) -> List[str]:
        ''' The titles listed now, in the order shown. '''
        return self._proxy.ids()

    def selected_ids(self) -> List[str]:
        ''' The selected rows' catalogue ids, in the order shown (chunk 26b's actions work on these). '''
        model = self.workTable.selectionModel()
        rows = sorted(index.row() for index in model.selectedRows()) if model else []
        return [self._proxy.id_at(row) for row in rows]

    def select_ids(self, ids) -> None:
        ''' Selects the listed rows with these ids (those not listed are ignored). '''
        wanted = set(ids)
        selection = self.workTable.selectionModel()
        selection.clearSelection()
        for row in range(self._proxy.rowCount()):
            if self._proxy.index(row, 0).data(ID_ROLE) in wanted:
                index = self._proxy.index(row, 0)
                selection.select(index, QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)

    def current_selection(self) -> Selection:
        '''
        The `Selection` the chip, the source combo and the search box amount to -- the same titles the table lists.
        *All* (and *New*) leave Done out, as the table does; a chunk 26b action over "everything in the filter"
        passes this to `plan_stages()`.
        '''
        source, match = self._proxy.source_filter, self._proxy.text or None
        not_done = tuple(n for n in NEEDS if n != 'done')
        chip = self._proxy.chip
        if chip == CHIP_ALL:
            return Selection(needs=not_done, source=source, match=match)
        if chip == CHIP_NEW:
            return Selection(needs=not_done, source=source, match=match, new_since_scan=True)
        return selection_from_chip(chip, source=source, match=match)

    # --- loading ----------------------------------------------------------------------------------------------------

    def reload(self) -> None:
        '''
        Reads the setup again and shows the cached index (no scan, unless it has never been scanned). Called on opening and
        whenever the window is shown again, since the preferences may have changed.
        '''
        self._closed = False
        self._apply_setup(load_drawer=True)

    def _apply_setup(self, load_drawer: bool) -> None:
        '''
        :param load_drawer: show the setup in the drawer too. Not after the drawer itself wrote the file: it already shows
            what it wrote, and reloading would replace a field the person has started typing in.
        '''
        self._setup = load_setup(self._preferences)
        if load_drawer:
            self._drawer.load(self._setup)
        else:
            self._drawer.refresh_designers()
        self._open_index(self._setup.index_file)
        self.refresh_from_index()
        # no index file yet is "never scanned" too: the scan creates it
        if self._auto_scan and self._setup.ready and self._index_error == '' and self._generation == 0 \
                and not self.is_running:
            self.rescan()

    @property
    def has_open_index(self) -> bool:
        ''' Whether the window holds a connection to the index (it does not while closed). '''
        return self._index is not None

    def _open_index(self, path: Optional[str]) -> None:
        '''
        Opens the index for reading. **It never creates the file**: a window that only shows what a scan found has nothing
        to show before the first scan, and opening a `LibraryIndex` creates the file and its schema; the scan does that.
        '''
        if path == self._index_file and self._index is not None:
            return
        self._close_index()
        self._index_file = path
        self._index_error = ''
        if path is None or not os.path.isfile(path):
            return
        try:
            self._index = LibraryIndex(path)
        except Exception as error:
            logger.exception('Could not open the index %s', path)
            self._index = None
            self._index_error = f'{type(error).__name__}: {error}'

    def _close_index(self) -> None:
        if self._index is not None:
            self._index.close()
        self._index, self._index_file = None, None

    def refresh_from_index(self) -> None:
        '''
        Replaces the rows with what the index holds now, keeping the selection, the scroll position and the filters,
        and updates the failures panel and the buttons. It reads the index; it does not scan a source or re-read any
        title's outputs (a run does that itself when it ends, `LibraryIndex.refresh()`).
        '''
        if self._closed:   # a scan or a run that ends after the window was closed must not open the index again
            return
        self._open_index(self._setup.index_file)  # a scan may have just created it
        selected = self.selected_ids()
        scroll = self.workTable.verticalScrollBar().value()
        rows: List[TitleRow] = []
        sources: List[SourceRow] = []
        failures: Mapping = {}
        last_scan, generation = None, 0
        if self._index is not None:
            self._index_error = ''
            try:
                summary = self._index.summary()
                rows, sources = self._index.titles(), summary.sources
                last_scan, generation = summary.last_scan_at, summary.generation
                failures = self._index.failures()
            except Exception as error:
                logger.exception('Could not read the index %s', self._index_file)
                self._index_error = f'{type(error).__name__}: {error}'
        self._sources_seen, self._last_scan_at, self._generation = sources, last_scan, generation
        self._populate_sources(sources)
        self._model.set_rows(rows)  # the proxy's modelReset refreshes the strip and the empty state
        if selected:
            self.select_ids(selected)
        self.workTable.verticalScrollBar().setValue(scroll)
        self._failed = failed_titles(rows, failures)
        self._drawer.set_rows(rows)
        self._refresh_failures()
        self._refresh_actions()

    def _populate_sources(self, sources: List[SourceRow]) -> None:
        wanted = self.sourceCombo.currentData()
        self.sourceCombo.blockSignals(True)
        self.sourceCombo.clear()
        self.sourceCombo.addItem('All sources', None)
        for source in sources:
            self.sourceCombo.addItem(f'{source.name} (!)' if source.last_error else source.name, source.name)
        index = self.sourceCombo.findData(wanted)
        self.sourceCombo.setCurrentIndex(max(index, 0))
        self.sourceCombo.blockSignals(False)
        self._proxy.set_source(self.sourceCombo.currentData())

    # --- filters ----------------------------------------------------------------------------------------------------

    def set_chip(self, chip: str) -> None:
        self._chip_buttons[chip].setChecked(True)
        self._proxy.set_chip(chip)
        self._refresh_view()

    def _on_search(self, text: str) -> None:
        self._proxy.set_text(text)
        self._refresh_view()

    def _on_source(self, _index: int) -> None:
        self._proxy.set_source(self.sourceCombo.currentData())
        self._refresh_view()

    def _on_header_clicked(self, column: int) -> None:
        ''' Ascending, then descending, then back to the index's order (tier, oldest first). '''
        if self._sort_column != column:
            self._sort_column, self._sort_order = column, Qt.SortOrder.AscendingOrder
        elif self._sort_order == Qt.SortOrder.AscendingOrder:
            self._sort_order = Qt.SortOrder.DescendingOrder
        else:
            self._sort_column, self._sort_order = -1, Qt.SortOrder.AscendingOrder
        self.workTable.horizontalHeader().setSortIndicator(self._sort_column, self._sort_order)
        self._proxy.sort_by(self._sort_column, self._sort_order)

    @property
    def sort_column(self) -> int:
        ''' The column the table is sorted by, or -1 for the index's order. '''
        return self._sort_column

    # --- what is shown ----------------------------------------------------------------------------------------------

    def _refresh_view(self, *_) -> None:
        ''' The strip's numbers, the scan text, the source lines, the Rescan button and the empty state. '''
        counts = self._proxy.counts()
        for chip, button in self._chip_buttons.items():
            text = f'{chip} {counts[chip]:,}'
            if chip == CHIP_DONE and not button.isChecked():
                text += ' (hidden)'
            button.setText(text)
            font = button.font()
            font.setBold(chip == 'Attention' and counts[chip] > 0)
            button.setFont(font)
        self._refresh_scan_text()
        self._refresh_source_status()
        self._refresh_empty_state(counts)
        self._refresh_banners()
        self._refresh_actions()

    def _refresh_banners(self) -> None:
        '''
        The banner at the top: what is missing from the setup (or why the profile file cannot be read), with a Settings...
        button; and, once the settings changed what a scan says, one offering Rescan now.
        '''
        setup = self._setup
        if setup.error:
            text = f'<b>The profile file could not be read.</b> {html.escape(setup.error)}'
        elif setup.problems:
            text = '<b>The setup is incomplete.</b> ' + ' '.join(html.escape(p) for p in setup.problems)
        else:
            text = ''
        self.setupBannerLabel.setText(text)
        self.setupBanner.setVisible(bool(text))
        self.staleBanner.setVisible(self._stale and not self._scanning)
        self.staleBannerButton.setEnabled(setup.ready and not self._busy())

    def _refresh_scan_text(self) -> None:
        if self._scanning:
            self.lastScanLabel.setText('scanning...')
        elif self._last_scan_at:
            self.lastScanLabel.setText(f'last scan {format_time(self._last_scan_at, self._clock())}')
        else:
            self.lastScanLabel.setText('never scanned')
        problems = self._setup.problems
        self.rescanButton.setEnabled(self._setup.ready and not self._scanning and not self.is_running)
        self.rescanButton.setText('Scanning...' if self._scanning else 'Rescan')
        self.rescanButton.setToolTip(
            'Cannot scan yet:\n' + '\n'.join(problems) if problems else
            'The profile could not be read.' if self._setup.error else
            'List every library source again and update what each title needs. '
            'Nothing is extracted, designed or published.')

    def _refresh_source_status(self) -> None:
        lines = describe_sources(self._sources_seen, self._clock())
        if not (any(bad for _, bad in lines) or len(lines) > 1):
            self.sourceStatusLabel.setVisible(False)
            return
        colour = warning_colour().name()
        parts = [f'<span style="color:{colour}">{html.escape(text)}</span>' if bad else html.escape(text)
                 for text, bad in lines]
        self.sourceStatusLabel.setText('<br>'.join(parts))
        self.sourceStatusLabel.setVisible(True)

    def _refresh_empty_state(self, counts: Dict[str, int]) -> None:
        setup, listed = self._setup, self._proxy.rowCount()
        title, detail, settings = '', '', False
        if listed:
            self.contentStack.setCurrentWidget(self.tablePage)
            return
        if setup.error:
            title = 'The library profile could not be read'
            detail = f'{setup.path}\n{setup.error}'
        elif setup.problems and not self._model.rowCount():
            title = 'The library is not set up yet'
            detail = '\n'.join(setup.problems) + '\n\nOpen Settings to choose the folders, the library sources and the ' \
                                                  'designer.'
            settings = True
        elif self._index_error and not self._model.rowCount():
            title = 'The library index could not be read'
            detail = f'{self._index_file}\n{self._index_error}\n\nIt may be damaged: delete it, and Rescan builds it again.'
        elif self._scanning and not self._model.rowCount():
            title, detail = 'Scanning the library...', 'The titles appear when the scan finishes.'
        elif self._generation == 0 and not self._model.rowCount():
            title, detail = 'The library has not been scanned yet', 'Press Rescan to list the library sources.'
        elif not self._model.rowCount():
            title = 'The last scan found no titles'
            detail = 'Check the library sources in Settings: the folders they search, or the JRiver browse node.'
        elif self._proxy.text or self._proxy.source_filter:
            title, detail = 'Nothing matches', 'Clear the search box or choose All sources to see more.'
        elif self._proxy.chip == 'Attention':
            title = 'Nothing needs attention'
        else:
            title = f'No titles in {self._proxy.chip}'
        self.emptyTitleLabel.setText(f'<b>{html.escape(title)}</b>')
        self.emptyDetailLabel.setText(html.escape(detail).replace('\n', '<br>'))
        self.openSettingsButton.setVisible(settings)
        self.contentStack.setCurrentWidget(self.emptyPage)

    # --- scanning ---------------------------------------------------------------------------------------------------

    def rescan(self, only: Optional[List[str]] = None) -> bool:
        '''
        Lists the sources again on a worker thread and shows the result when it is done.
        :param only: rescan just these sources (by profile name).
        :return: False if nothing was started (a scan is running, or the setup is incomplete).
        '''
        self._flush_settings()   # a setting edited a moment ago is what this scan must use
        setup = self._setup
        if self._scanning or self.is_running or not setup.ready or setup.index_file is None:
            return False
        job = _ScanJob(setup.index_file, setup.profile, setup.settings, only, self._sources)
        job.signals.finished.connect(self._on_scan_finished)
        job.signals.errored.connect(self._on_scan_failed)
        self._active_job = job
        self._set_scanning(True)
        self.statusBar.showMessage('Scanning the library sources...')
        QThreadPool.globalInstance().start(job)
        return True

    def _set_scanning(self, scanning: bool) -> None:
        self._scanning = scanning
        self._progress.setVisible(scanning)
        self._refresh_view()

    def _on_scan_finished(self, result: ScanResult) -> None:
        self._active_job = None
        self._scanning = False
        self._stale = False   # the index describes the current settings again
        self._progress.setVisible(False)
        self.refresh_from_index()
        message = f'Scan finished: {result.titles:,} titles, {len(result.new):,} new'
        if result.errors:
            message += f'; {len(result.errors)} source(s) could not be listed'
        self.statusBar.showMessage(message)
        self.scan_finished.emit(result)

    def _on_scan_failed(self, message: str) -> None:
        self._active_job = None
        self._set_scanning(False)
        self.statusBar.showMessage(f'The scan failed: {message}')
        self.scan_failed.emit(message)

    def showEvent(self, event) -> None:
        if self._closed:   # shown again without the app's reload(): read the settings and the index again
            self.reload()
        super().showEvent(event)

    def closeEvent(self, event) -> None:
        '''
        A run in progress is not left going against a hidden window: the person is asked, and *Yes* cancels it (the title in
        hand finishes) and closes. Closing releases the index connection, and a scan or run that ends afterwards does not
        open it again (`reload()` does, when the window is shown again).
        '''
        if self.is_running:
            answer = QMessageBox.question(
                self, 'Run in progress', 'A run is in progress. Cancel it and close?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.cancel_run()
        self._flush_settings()
        self._preferences.set(WORKLIST_GEOMETRY, self.saveGeometry())
        self._closed = True
        self._close_index()  # a scan or run in flight has its own connection
        super().closeEvent(event)
