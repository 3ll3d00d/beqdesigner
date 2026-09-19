'''
The library work list -- design/library-sync/workflow-rework/design.md §12.10, chunk 26a (read-only).

A top-level window over the discovery index (`pipeline.library.index.LibraryIndex`): the pipeline strip with a count per
kind of work, a searchable, filterable table of every title in the library and what it needs next, and a Rescan that
lists the sources again. It is the intended replacement for `model.library_sync.LibrarySyncDialog`, which stays
untouched and working until chunk 27c; **this window does not run, accept, publish or commit anything yet** (chunk 26b).

What the window reads and where it does not think for itself:

* what each title needs, its detail, how long it has waited, whether it is new, its flags -- all straight from the
  index's rows (`TitleRow`), in the index's order (tier, then oldest first); see `model.worklist_model`;
* the profile -- from the `LIBRARY_PROFILE_PATH` file if there is one, else built from the Library Sync preferences
  (`model.worklist_profile`, the one function chunk 26c replaces);
* the strip's chips are `pipeline.library.selection.CHIPS`, and `current_selection()` is the `Selection` the filters
  amount to, which chunk 26b's action button hands to `plan_stages()`/`run_stages()`.

The cached index is shown at once (stale-while-revalidate). A rescan runs on a QRunnable in the global thread pool with its
own index connection, so the UI thread never waits for a slow source; the window rescans by itself only when the
index has never been scanned.
'''
import html
import logging
import time
from typing import Dict, List, Mapping, Optional

from qtpy.QtCore import QItemSelectionModel, QObject, QRunnable, Qt, QThreadPool, Signal
from qtpy.QtGui import QKeySequence, QShortcut
from qtpy.QtWidgets import QAbstractItemView, QButtonGroup, QHeaderView, QMainWindow, QProgressBar

from model.preferences import WORKLIST_GEOMETRY
from model.worklist_model import ALL_CHIPS, CHIP_ALL, CHIP_DONE, COL_DETAIL, COL_NEEDS, COL_SOURCE, COL_TITLE, \
    COL_WAITING, COL_YEAR, ID_ROLE, WorkListModel, WorkListProxy, warning_colour
from model.worklist_profile import ORIGIN_FILE, WorkListSetup, load_setup
from pipeline.library.index import LibraryIndex, ScanResult, SourceRow, TitleRow
from pipeline.library.profile import Profile
from pipeline.library.selection import CHIP_NEW, Selection, selection_from_chip
from pipeline.library.source import LibrarySource
from pipeline.library.state import NEEDS
from pipeline.library.status import ScanSettings
from ui.worklist import Ui_workListWindow

logger = logging.getLogger('worklist')

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
        self.__path, self.__profile, self.__settings = path, profile, settings
        self.__only, self.__sources = only, sources

    def run(self):
        try:
            with LibraryIndex(self.__path) as index:
                result = index.scan(self.__profile, self.__settings, only=self.__only, sources=self.__sources)
            self.signals.finished.emit(result)
        except Exception as error:
            logger.exception('Library scan failed')
            self.signals.errored.emit(f'{type(error).__name__}: {error}')


class WorkListWindow(QMainWindow, Ui_workListWindow):
    '''
    :param parent: the main window, or None.
    :param preferences: `model.preferences.Preferences`; read again by reload().
    :param auto_scan: rescan on opening if the index has never been scanned.
    :param sources: already-built sources by profile name, handed to the scan (tests); by default they are built from the
        profile's settings.
    :param clock: the time source, for "waiting" and "last scan" (tests).
    '''
    settings_requested = Signal()      # the empty state's button: the app opens Library Sync, where the setup lives for now
    scan_finished = Signal(object)     # a ScanResult, once the list shows it
    scan_failed = Signal(str)

    def __init__(self, parent, preferences, *, auto_scan: bool = True,
                 sources: Optional[Mapping[str, LibrarySource]] = None, clock=time.time):
        super().__init__(parent)
        self.setupUi(self)
        self.__preferences = preferences
        self.__auto_scan = auto_scan
        self.__sources = sources
        self.__clock = clock
        self.__setup: WorkListSetup = WorkListSetup(None, None, 'preferences')
        self.__index: Optional[LibraryIndex] = None
        self.__index_file: Optional[str] = None
        self.__sources_seen: List[SourceRow] = []
        self.__last_scan_at: Optional[float] = None
        self.__generation = 0
        self.__scanning = False
        self.__active_job: Optional[_ScanJob] = None
        self.__sort_column = -1
        self.__sort_order = Qt.SortOrder.AscendingOrder

        self.__model = WorkListModel(self, clock)
        self.__proxy = WorkListProxy(self)
        self.__proxy.setSourceModel(self.__model)
        self.__configure_table()
        self.__configure_chips()
        self.__progress = QProgressBar()
        self.__progress.setRange(0, 0)
        self.__progress.setMaximumWidth(140)
        self.__progress.setVisible(False)
        self.statusBar.addPermanentWidget(self.__progress)

        self.searchEdit.textChanged.connect(self.__on_search)
        self.sourceCombo.currentIndexChanged.connect(self.__on_source)
        self.rescanButton.clicked.connect(lambda: self.rescan())
        self.openSettingsButton.clicked.connect(self.settings_requested.emit)
        QShortcut(QKeySequence('Ctrl+F'), self, activated=self.searchEdit.setFocus)
        geometry = preferences.get(WORKLIST_GEOMETRY)
        if geometry is not None:
            self.restoreGeometry(geometry)
        self.reload()

    # --- construction ---------------------------------------------------------------------------------------------

    def __configure_table(self):
        table = self.workTable
        table.setModel(self.__proxy)
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
        header.sectionClicked.connect(self.__on_header_clicked)
        self.__proxy.modelReset.connect(self.__refresh_view)
        self.__proxy.layoutChanged.connect(self.__refresh_view)

    def __configure_chips(self):
        self.__chip_buttons = {
            CHIP_ALL: self.allChip, 'Attention': self.attentionChip, CHIP_NEW: self.newChip,
            'Extract': self.extractChip, 'Design': self.designChip, 'Review': self.reviewChip,
            'Publish': self.publishChip, 'Commit': self.commitChip, CHIP_DONE: self.doneChip}
        assert tuple(self.__chip_buttons) == ALL_CHIPS
        self.__chip_group = QButtonGroup(self)
        self.__chip_group.setExclusive(True)
        for chip, button in self.__chip_buttons.items():
            self.__chip_group.addButton(button)
            button.setToolTip(_CHIP_TIPS[chip])
            button.setStyleSheet(_CHIP_STYLE)
            sample = f'{chip} 0,000' + (' (hidden)' if chip == CHIP_DONE else '')
            button.setMinimumWidth(button.fontMetrics().horizontalAdvance(sample) + 30)
            button.toggled.connect(lambda checked, name=chip: checked and self.set_chip(name))

    # --- state ------------------------------------------------------------------------------------------------------

    @property
    def setup(self) -> WorkListSetup:
        return self.__setup

    @property
    def is_scanning(self) -> bool:
        return self.__scanning

    @property
    def chip(self) -> str:
        return self.__proxy.chip

    @property
    def proxy(self) -> WorkListProxy:
        return self.__proxy

    @property
    def model(self) -> WorkListModel:
        return self.__model

    def chip_counts(self) -> Dict[str, int]:
        ''' The number on each chip, under the current source and search. '''
        return self.__proxy.counts()

    def listed_ids(self) -> List[str]:
        ''' The titles listed now, in the order shown. '''
        return self.__proxy.ids()

    def selected_ids(self) -> List[str]:
        ''' The selected rows' catalogue ids, in the order shown (chunk 26b's actions work on these). '''
        model = self.workTable.selectionModel()
        rows = sorted(index.row() for index in model.selectedRows()) if model else []
        return [self.__proxy.id_at(row) for row in rows]

    def select_ids(self, ids) -> None:
        ''' Selects the listed rows with these ids (those not listed are ignored). '''
        wanted = set(ids)
        selection = self.workTable.selectionModel()
        selection.clearSelection()
        for row in range(self.__proxy.rowCount()):
            if self.__proxy.index(row, 0).data(ID_ROLE) in wanted:
                index = self.__proxy.index(row, 0)
                selection.select(index, QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)

    def current_selection(self) -> Selection:
        '''
        The `Selection` the chip, the source combo and the search box amount to -- the same titles the table lists.
        *All* (and *New*) leave Done out, as the table does; a chunk 26b action over "everything in the filter"
        passes this to `plan_stages()`.
        '''
        source, match = self.__proxy.source_filter, self.__proxy.text or None
        not_done = tuple(n for n in NEEDS if n != 'done')
        chip = self.__proxy.chip
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
        self.__setup = load_setup(self.__preferences)
        self.__open_index(self.__setup.index_file)
        self.__populate()
        if self.__auto_scan and self.__setup.ready and self.__index is not None and self.__generation == 0:
            self.rescan()

    def __open_index(self, path: Optional[str]) -> None:
        if path == self.__index_file and self.__index is not None:
            return
        self.__close_index()
        self.__index_file = path
        if path is None:
            return
        try:
            self.__index = LibraryIndex(path)
        except Exception:
            logger.exception('Could not open the index %s', path)
            self.__index = None

    def __close_index(self) -> None:
        if self.__index is not None:
            self.__index.close()
        self.__index, self.__index_file = None, None

    def __populate(self) -> None:
        ''' Replaces the rows with the index's, keeping the selection, the scroll position and the filters. '''
        self.__open_index(self.__setup.index_file)  # closed by closeEvent, and a scan can finish after that
        selected = self.selected_ids()
        scroll = self.workTable.verticalScrollBar().value()
        rows: List[TitleRow] = []
        sources: List[SourceRow] = []
        last_scan, generation = None, 0
        if self.__index is not None:
            try:
                summary = self.__index.summary()
                rows, sources = self.__index.titles(), summary.sources
                last_scan, generation = summary.last_scan_at, summary.generation
            except Exception:
                logger.exception('Could not read the index %s', self.__index_file)
        self.__sources_seen, self.__last_scan_at, self.__generation = sources, last_scan, generation
        self.__populate_sources(sources)
        self.__model.set_rows(rows)  # the proxy's modelReset refreshes the strip and the empty state
        if selected:
            self.select_ids(selected)
        self.workTable.verticalScrollBar().setValue(scroll)

    def __populate_sources(self, sources: List[SourceRow]) -> None:
        wanted = self.sourceCombo.currentData()
        self.sourceCombo.blockSignals(True)
        self.sourceCombo.clear()
        self.sourceCombo.addItem('All sources', None)
        for source in sources:
            self.sourceCombo.addItem(f'{source.name} (!)' if source.last_error else source.name, source.name)
        index = self.sourceCombo.findData(wanted)
        self.sourceCombo.setCurrentIndex(max(index, 0))
        self.sourceCombo.blockSignals(False)
        self.__proxy.set_source(self.sourceCombo.currentData())

    # --- filters ----------------------------------------------------------------------------------------------------

    def set_chip(self, chip: str) -> None:
        self.__chip_buttons[chip].setChecked(True)
        self.__proxy.set_chip(chip)
        self.__refresh_view()

    def __on_search(self, text: str) -> None:
        self.__proxy.set_text(text)
        self.__refresh_view()

    def __on_source(self, _index: int) -> None:
        self.__proxy.set_source(self.sourceCombo.currentData())
        self.__refresh_view()

    def __on_header_clicked(self, column: int) -> None:
        ''' Ascending, then descending, then back to the index's order (tier, oldest first). '''
        if self.__sort_column != column:
            self.__sort_column, self.__sort_order = column, Qt.SortOrder.AscendingOrder
        elif self.__sort_order == Qt.SortOrder.AscendingOrder:
            self.__sort_order = Qt.SortOrder.DescendingOrder
        else:
            self.__sort_column, self.__sort_order = -1, Qt.SortOrder.AscendingOrder
        self.workTable.horizontalHeader().setSortIndicator(self.__sort_column, self.__sort_order)
        self.__proxy.sort_by(self.__sort_column, self.__sort_order)

    @property
    def sort_column(self) -> int:
        ''' The column the table is sorted by, or -1 for the index's order. '''
        return self.__sort_column

    # --- what is shown ----------------------------------------------------------------------------------------------

    def __refresh_view(self, *_) -> None:
        ''' The strip's numbers, the scan text, the source lines, the Rescan button and the empty state. '''
        counts = self.__proxy.counts()
        for chip, button in self.__chip_buttons.items():
            text = f'{chip} {counts[chip]:,}'
            if chip == CHIP_DONE and not button.isChecked():
                text += ' (hidden)'
            button.setText(text)
            font = button.font()
            font.setBold(chip == 'Attention' and counts[chip] > 0)
            button.setFont(font)
        self.__refresh_scan_text()
        self.__refresh_source_status()
        self.__refresh_empty_state(counts)

    def __refresh_scan_text(self) -> None:
        if self.__scanning:
            self.lastScanLabel.setText('scanning...')
        elif self.__last_scan_at:
            self.lastScanLabel.setText(f'last scan {format_time(self.__last_scan_at, self.__clock())}')
        else:
            self.lastScanLabel.setText('never scanned')
        problems = self.__setup.problems
        self.rescanButton.setEnabled(self.__setup.ready and not self.__scanning)
        self.rescanButton.setText('Scanning...' if self.__scanning else 'Rescan')
        self.rescanButton.setToolTip(
            'Cannot scan yet:\n' + '\n'.join(problems) if problems else
            'The profile could not be read.' if self.__setup.error else
            'List every library source again and update what each title needs. '
            'Nothing is extracted, designed or published.')

    def __refresh_source_status(self) -> None:
        lines = describe_sources(self.__sources_seen, self.__clock())
        problems = list(self.__setup.problems) if self.__model.rowCount() else []  # else the empty state says it
        if not (any(bad for _, bad in lines) or len(lines) > 1 or problems):
            self.sourceStatusLabel.setVisible(False)
            return
        colour = warning_colour().name()
        parts = [f'<span style="color:{colour}">{html.escape(text)}</span>' if bad else html.escape(text)
                 for text, bad in lines]
        parts += [f'<span style="color:{colour}">{html.escape(text)}</span>' for text in problems]
        self.sourceStatusLabel.setText('<br>'.join(parts))
        self.sourceStatusLabel.setVisible(True)

    def __refresh_empty_state(self, counts: Dict[str, int]) -> None:
        setup, listed = self.__setup, self.__proxy.rowCount()
        title, detail, settings = '', '', False
        if listed:
            self.contentStack.setCurrentWidget(self.tablePage)
            return
        if setup.error:
            title = 'The library profile could not be read'
            detail = f'{setup.path}\n{setup.error}'
        elif setup.problems and not self.__model.rowCount():
            title = 'The library is not set up yet'
            detail = '\n'.join(setup.problems) + '\n\nThe work list uses the same settings as Library Sync ' \
                                                  '(Tools > Library Sync). Set them there, then reopen this window.'
            settings = setup.origin != ORIGIN_FILE
        elif self.__scanning and not self.__model.rowCount():
            title, detail = 'Scanning the library...', 'The titles appear when the scan finishes.'
        elif self.__generation == 0 and not self.__model.rowCount():
            title, detail = 'The library has not been scanned yet', 'Press Rescan to list the library sources.'
        elif not self.__model.rowCount():
            title = 'The last scan found no titles'
            detail = 'Check the library source in Library Sync, or the folders or JRiver browse node it points to.'
        elif self.__proxy.text or self.__proxy.source_filter:
            title, detail = 'Nothing matches', 'Clear the search box or choose All sources to see more.'
        elif self.__proxy.chip == 'Attention':
            title = 'Nothing needs attention'
        else:
            title = f'No titles in {self.__proxy.chip}'
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
        setup = self.__setup
        if self.__scanning or not setup.ready or setup.index_file is None:
            return False
        job = _ScanJob(setup.index_file, setup.profile, setup.settings, only, self.__sources)
        job.signals.finished.connect(self.__on_scan_finished)
        job.signals.errored.connect(self.__on_scan_failed)
        self.__active_job = job
        self.__set_scanning(True)
        self.statusBar.showMessage('Scanning the library sources...')
        QThreadPool.globalInstance().start(job)
        return True

    def __set_scanning(self, scanning: bool) -> None:
        self.__scanning = scanning
        self.__progress.setVisible(scanning)
        self.__refresh_view()

    def __on_scan_finished(self, result: ScanResult) -> None:
        self.__active_job = None
        self.__scanning = False
        self.__progress.setVisible(False)
        self.__populate()
        message = f'Scan finished: {result.titles:,} titles, {len(result.new):,} new'
        if result.errors:
            message += f'; {len(result.errors)} source(s) could not be listed'
        self.statusBar.showMessage(message)
        self.scan_finished.emit(result)

    def __on_scan_failed(self, message: str) -> None:
        self.__active_job = None
        self.__set_scanning(False)
        self.statusBar.showMessage(f'The scan failed: {message}')
        self.scan_failed.emit(message)

    def closeEvent(self, event) -> None:
        self.__preferences.set(WORKLIST_GEOMETRY, self.saveGeometry())
        self.__close_index()  # a scan in flight has its own connection; reload() reopens this one
        super().closeEvent(event)
