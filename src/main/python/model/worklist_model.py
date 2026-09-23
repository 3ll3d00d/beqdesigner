'''
The work list's table model and its sort/filter proxy -- design/library-sync/workflow-rework/design.md §12.10.

Both are read-only views over the discovery index: a row is a `pipeline.library.index.TitleRow`, and everything shown
(needs, detail, waiting, new-since-scan, the flags) is read from it, never derived again. The index hands the rows over
already in work-list order (tier, then oldest `state_since` first) and the model keeps that order, so the proxy's default
order is that order; a column sort is a temporary override.

The proxy answers exactly what `pipeline.library.selection.Selection` answers for the same chip, source and search
text (the window's `current_selection()` builds that Selection), so what is listed and what an action would work on
cannot disagree. The one addition is the default view's rule that Done is hidden.
'''
import time
from typing import Callable, Dict, List, Optional

from qtpy.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt
from qtpy.QtGui import QBrush, QColor, QGuiApplication, QPalette

from pipeline.library.index import TitleRow
from pipeline.library.selection import CHIP_NEW, CHIPS

CHIP_ALL = 'All'
CHIP_DONE = 'Done'
ALL_CHIPS = (CHIP_ALL,) + CHIPS   # the strip, left to right

COLUMNS = ('Title', 'Year', 'Source', 'Needs', 'Detail', 'Waiting', 'Run progress', 'Run details')
COL_TITLE, COL_YEAR, COL_SOURCE, COL_NEEDS, COL_DETAIL, COL_WAITING, COL_RUN_PROGRESS, COL_RUN_DETAILS = \
    range(len(COLUMNS))

ROW_ROLE = Qt.ItemDataRole.UserRole + 1      # the TitleRow
ID_ROLE = Qt.ItemDataRole.UserRole + 2       # the catalogue id
NEW_ROLE = Qt.ItemDataRole.UserRole + 3      # True for a title first seen by the latest scan
TIER_ROLE = Qt.ItemDataRole.UserRole + 4
SORT_ROLE = Qt.ItemDataRole.UserRole + 5     # what a column sorts by
RUNNING_ROLE = Qt.ItemDataRole.UserRole + 6  # the stage being run for this title now (extract...), else ''
RUN_STATE_ROLE = Qt.ItemDataRole.UserRole + 20  # transient run UI state; never part of the discovery index

_NEEDS_LABEL = {'attention': '! Attention', 'review': 'Review', 'extract': 'Extract', 'design': 'Design',
                'publish': 'Publish', 'commit': 'Commit', 'done': 'Done'}


_RUNNING_WORD = {'extract': 'Extracting', 'design': 'Designing', 'publish': 'Publishing', 'commit': 'Committing'}


def running_text(stage: str) -> str:
    ''' What the Needs column says of a title that is being worked on: "\u25b6 Designing...". '''
    return f'\u25b6 {_RUNNING_WORD.get(stage, stage.capitalize())}...'


def is_dark_palette() -> bool:
    return QGuiApplication.palette().color(QPalette.ColorRole.Window).lightness() < 128


def warning_colour() -> QColor:
    ''' Legible on the window background whether the platform theme is light or dark. '''
    return QColor('#ff7b72') if is_dark_palette() else QColor('#b3261e')


def format_waiting(since: float, now: float) -> str:
    ''' How long a title has been in its current state: 5m, 3h, 9d, 4mo, 2y. '''
    seconds = max(0, now - since)
    if seconds < 60:
        return 'now'
    if seconds < 3600:
        return f'{int(seconds // 60)}m'
    if seconds < 86400:
        return f'{int(seconds // 3600)}h'
    days = int(seconds // 86400)
    if days < 60:
        return f'{days}d'
    if days < 730:
        return f'{days // 30}mo'
    return f'{days // 365}y'


def title_text(row: TitleRow) -> str:
    text = row.title or row.display_name or row.id
    return f'{text} - season {row.season}' if row.unit == 'season' and row.season else text


def detail_text(row: TitleRow) -> str:
    ''' The index's one-line reason, then the flags that apply ("Already in catalogue"...). '''
    return ' · '.join(part for part in [row.detail, *row.flags] if part)


def _tooltip(row: TitleRow, column: int) -> Optional[str]:
    if column == COL_TITLE:
        lines = [title_text(row), row.path or row.id]
        if row.also_in:
            lines.append('Also in: ' + ', '.join(row.also_in))
        return '\n'.join(lines)
    if column == COL_DETAIL:
        text = detail_text(row)
        return text if not row.failure or row.failure in text else f'{text}\n{row.failure}'
    if column == COL_WAITING:
        return time.strftime('In this state since %Y-%m-%d %H:%M', time.localtime(row.state_since)) + \
            ('\nNew in the latest scan' if row.is_new else '')
    return None


class WorkListModel(QAbstractTableModel):
    ''' The index's titles, in the order the index gave them. Read-only. '''

    def __init__(self, parent=None, clock: Callable[[], float] = time.time):
        super().__init__(parent)
        self.__rows: List[TitleRow] = []
        self.__clock = clock
        self.__now = clock()
        self.__running: Dict[str, str] = {}   # title id -> the stage being run for it
        self.__run_state: Dict[str, dict] = {}
        self.__row_of: Dict[str, int] = {}

    def set_rows(self, rows: List[TitleRow]) -> None:
        self.beginResetModel()
        self.__rows = list(rows)
        self.__row_of = {row.id: i for i, row in enumerate(self.__rows)}
        self.__now = self.__clock()
        self.endResetModel()

    @property
    def rows(self) -> List[TitleRow]:
        return self.__rows

    def set_running(self, stages: Optional[Dict[str, str]] = None) -> None:
        '''
        Marks the titles being worked on right now ({title id: stage}), replacing the last marks: their Needs cell reads
        "\u25b6 Designing..." and the row is tinted, until a later call clears them. An id that is not listed is ignored.
        '''
        stages = dict(stages or {})
        changed = set(self.__running) | set(stages)
        self.__running = stages
        for title_id in changed:
            state = dict(self.__run_state.get(title_id, {}))
            if title_id in stages:
                state.update(active=True, queued=False, stage=stages[title_id], text=running_text(stages[title_id]))
            else:
                state.update(active=False, queued=False, text='')
            self.__run_state[title_id] = state
        for title_id in changed:
            row = self.__row_of.get(title_id)
            if row is not None:
                self.dataChanged.emit(self.index(row, 0), self.index(row, len(COLUMNS) - 1))

    @property
    def running(self) -> Dict[str, str]:
        return dict(self.__running)

    def set_run_state(self, title_id: str, **state) -> None:
        current = dict(self.__run_state.get(title_id, {}))
        current.update(state)
        self.__run_state[title_id] = current
        if current.get('active'):
            self.__running[title_id] = current.get('stage', '')
        else:
            self.__running.pop(title_id, None)
        row = self.__row_of.get(title_id)
        if row is not None:
            # Needs displays the stage marker from __running; include it with
            # both run columns so views and proxies repaint the whole state.
            self.dataChanged.emit(self.index(row, COL_NEEDS), self.index(row, COL_RUN_DETAILS))

    def clear_active_run_states(self) -> None:
        for title_id, state in list(self.__run_state.items()):
            if state.get('active') or state.get('queued'):
                state.update(active=False, queued=False, text='')
                self.set_run_state(title_id, **state)

    def run_state(self, title_id: str) -> dict:
        return dict(self.__run_state.get(title_id, {}))

    def row_at(self, row: int) -> TitleRow:
        return self.__rows[row]

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.__rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(COLUMNS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return COLUMNS[section]
        return None

    def __sort_key(self, row: TitleRow, column: int):
        if column == COL_TITLE:
            return title_text(row).casefold()
        if column == COL_YEAR:
            return row.year
        if column == COL_SOURCE:
            return row.source.casefold()
        if column == COL_NEEDS:
            return _NEEDS_LABEL.get(row.needs, row.needs)
        if column == COL_DETAIL:
            return row.detail.casefold()
        return row.state_since  # Waiting: oldest first is ascending

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row, column = self.__rows[index.row()], index.column()
        running = self.__running.get(row.id, '')
        if role == Qt.ItemDataRole.DisplayRole:
            if column == COL_NEEDS and running:
                return running_text(running)
            if column == COL_TITLE:
                return title_text(row)
            if column == COL_YEAR:
                return row.year
            if column == COL_SOURCE:
                return row.source
            if column == COL_NEEDS:
                return _NEEDS_LABEL.get(row.needs, row.needs)
            if column == COL_DETAIL:
                return detail_text(row)
            waiting = format_waiting(row.state_since, self.__now)
            return f'new \u00b7 {waiting}' if row.is_new else waiting
        if role == Qt.ItemDataRole.ToolTipRole:
            return _tooltip(row, column)
        if role == ROW_ROLE:
            return row
        if role == ID_ROLE:
            return row.id
        if role == NEW_ROLE:
            return row.is_new
        if role == TIER_ROLE:
            return row.tier
        if role == RUNNING_ROLE:
            return running
        if role == RUN_STATE_ROLE:
            return self.__run_state.get(row.id, {})
        if role == SORT_ROLE:
            return self.__sort_key(row, column)
        if role == Qt.ItemDataRole.BackgroundRole and (running or row.is_new):
            colour = QGuiApplication.palette().color(QPalette.ColorRole.Highlight)
            colour.setAlpha(110 if running else 55)   # a title being worked on is the stronger of the two
            return QBrush(colour)
        if role == Qt.ItemDataRole.ForegroundRole:
            if running:
                return QBrush(QGuiApplication.palette().color(QPalette.ColorRole.Text))
            if row.tier == 'attention' and column in (COL_NEEDS, COL_DETAIL):
                return QBrush(warning_colour())
            if row.tier == 'done':
                dimmed = QGuiApplication.palette().color(QPalette.ColorRole.Text)
                dimmed.setAlpha(150)  # the text colour, softened: legible on a light or a dark theme
                return QBrush(dimmed)
        if role == Qt.ItemDataRole.FontRole and column == COL_NEEDS and (running or row.tier == 'attention'):
            font = QGuiApplication.font()
            font.setBold(True)
            return font
        if role == Qt.ItemDataRole.TextAlignmentRole and column in (COL_YEAR, COL_WAITING):
            return int((Qt.AlignmentFlag.AlignHCenter if column == COL_YEAR else Qt.AlignmentFlag.AlignRight)
                       | Qt.AlignmentFlag.AlignVCenter)
        return None


def chip_accepts(chip: str, row: TitleRow) -> bool:
    '''
    Whether a strip chip lists a title. *All* is the default view: everything except Done. *New* is the titles first
    seen by the latest scan that are not Done; every other chip is one `needs` value.
    '''
    if chip == CHIP_ALL:
        return row.needs != 'done'
    if chip == CHIP_NEW:
        return row.is_new and row.needs != 'done'
    return row.needs == chip.lower()


def matches_text(row: TitleRow, text: str) -> bool:
    ''' The search box: `Selection.match`'s fields (title, display name, id, path), ignoring case. '''
    needle = text.casefold()
    return any(needle in field.casefold() for field in (row.title, row.display_name, row.id, row.path))


class WorkListProxy(QSortFilterProxyModel):
    '''
    Filters the model by chip, source, search text and independently matchable displayed columns, and sorts it. With no
    column chosen (`sort_by(-1)`) the rows are in the index's order: attention, human, machine, then done, each oldest
    first.
    '''

    def __init__(self, parent=None):
        super().__init__(parent)
        self.__chip = CHIP_ALL
        self.__source: Optional[str] = None
        self.__text = ''
        self.__column_text: Dict[int, str] = {}
        self.setSortRole(SORT_ROLE)
        self.setSortCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)

    @property
    def chip(self) -> str:
        return self.__chip

    @property
    def source_filter(self) -> Optional[str]:
        return self.__source

    @property
    def text(self) -> str:
        return self.__text

    @property
    def column_filters(self) -> Dict[int, str]:
        '''The non-empty per-column text filters, keyed by a `COL_*` constant.'''
        return dict(self.__column_text)

    def set_chip(self, chip: str) -> None:
        if chip not in ALL_CHIPS:
            raise ValueError(f'unknown chip {chip!r}; the chips are {", ".join(ALL_CHIPS)}')
        self.__chip = chip
        self.invalidateFilter()

    def set_source(self, source: Optional[str]) -> None:
        self.__source = source or None
        self.invalidateFilter()

    def set_text(self, text: str) -> None:
        self.__text = text.strip()
        self.invalidateFilter()

    def set_column_text(self, column: int, text: str) -> None:
        if column not in range(len(COLUMNS)):
            raise ValueError(f'unknown column {column}')
        text = text.strip()
        if text:
            self.__column_text[column] = text
        else:
            self.__column_text.pop(column, None)
        self.invalidateFilter()

    def sort_by(self, column: int, order=Qt.SortOrder.AscendingOrder) -> None:
        ''' Sorts by a column, or -1 for the index's own order. '''
        self.sort(column, order)

    def filterAcceptsRow(self, source_row, source_parent):
        rows = self.sourceModel().rows
        return self.__accepts(rows[source_row], self.__chip, source_row)

    def __accepts(self, row: TitleRow, chip: str, source_row: int) -> bool:
        return (chip_accepts(chip, row) and (self.__source is None or row.source == self.__source)
                and (not self.__text or matches_text(row, self.__text)) and self.__column_matches(source_row))

    def __column_matches(self, source_row: int) -> bool:
        source = self.sourceModel()
        return all(text.casefold() in str(source.index(source_row, column).data(Qt.ItemDataRole.DisplayRole) or '').casefold()
                   for column, text in self.__column_text.items())

    def lessThan(self, left, right):
        a, b = left.data(SORT_ROLE), right.data(SORT_ROLE)
        return a < b

    def counts(self) -> Dict[str, int]:
        '''
        What each chip would list under the current source and search: the strip's numbers. Unfiltered, they are
        the index's `summary().counts` (with Done and New as described at chip_accepts()).
        '''
        source = self.sourceModel()
        rows = source.rows if source is not None else []
        return {chip: sum(1 for number, row in enumerate(rows) if self.__accepts(row, chip, number))
                for chip in ALL_CHIPS}

    def id_at(self, row: int) -> str:
        return self.index(row, 0).data(ID_ROLE)

    def ids(self) -> List[str]:
        ''' The listed titles' ids, in the order shown. '''
        return [self.id_at(row) for row in range(self.rowCount())]
