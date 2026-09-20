'''
The sources half of the work list's settings drawer (chunk 26c): the profile's sources in priority order, and the dialog
that edits one of them.

A source is edited with the same `model.library_sources.SourcePage` kinds Library Sync uses (folders and globs; a JRiver
server, browse node, path mappings and id fields), not a copy of them: the dialog adds only what a profile source has and
a Library Sync preference does not -- a **name**, unique in the profile.
'''
from typing import Callable, Collection, Dict, List, Optional, Sequence

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QAbstractItemView, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, \
    QListWidget, QListWidgetItem, QPushButton, QStackedWidget, QVBoxLayout, QWidget, QComboBox

from model.library_sources import SourcePage, registered_source_kinds
from model.worklist_edit import move_item, unique_name
from model.worklist_model import warning_colour
from pipeline.library.profile import SourceSpec, build_source

PRIORITY_HELP = (
    '<b>The source at the top has the highest priority.</b> The same file in two sources is one title, and the first source owns it '
    '(the other shows it as "also in"). The same film in two different files stays two titles, each flagged as a possible '
    'duplicate. Reordering never renames a title: one that already exists keeps its id and its owner.')


def describe_spec(spec: SourceSpec) -> str:
    ''' One line saying what a source reads, for the list. '''
    settings = spec.settings
    if spec.kind == 'filesystem':
        globs = [str(g) for g in settings.get('globs') or []]
        shown = ', '.join(globs[:2]) + (f' and {len(globs) - 2} more' if len(globs) > 2 else '')
        return f'Filesystem: {shown or "nothing to search yet"}'
    if spec.kind == 'jriver':
        where = f"{settings.get('host', '?')}:{settings.get('port', '?')}"
        node = settings.get('browse_path') or (
            'the whole library' if settings.get('browse_node_id', -1) in (-1, None)
            else f"browse node {settings.get('browse_node_id')}")
        mappings = len(settings.get('path_mappings') or [])
        return f'JRiver {where}: {node}' + (f' ({mappings} path mapping{"" if mappings == 1 else "s"})' if mappings else '')
    return spec.kind


class SourceDialog(QDialog):
    '''
    Adds or edits one profile source: a name and the settings page of its kind.
    :param existing: the source being edited, None to add one (the kind is then chosen).
    :param taken_names: the other sources' names, which this one may not use.
    '''

    def __init__(self, parent, prefs, existing: Optional[SourceSpec], taken_names: Collection[str],
                 kinds: Optional[list] = None):
        super().__init__(parent)
        self.setWindowTitle('Edit source' if existing else 'Add source')
        self._prefs, self._existing, self._taken = prefs, existing, set(taken_names)
        self._spec: Optional[SourceSpec] = None
        self.nameEdit = QLineEdit()
        self.nameEdit.setPlaceholderText('A short unique name, e.g. films')
        self.kindCombo = QComboBox()
        self.pageStack = QStackedWidget()
        self.pages: Dict[str, SourcePage] = {}
        for kind in kinds or registered_source_kinds():
            page = kind.create_page()
            if existing is not None and existing.kind == kind.name:
                page.load_settings(existing.settings, prefs)
            else:
                page.load_defaults(prefs)
            self.pages[kind.name] = page
            self.kindCombo.addItem(kind.label, kind.name)
            self.pageStack.addWidget(page)
        if existing is not None:
            self.kindCombo.setCurrentIndex(max(self.kindCombo.findData(existing.kind), 0))
            self.kindCombo.setEnabled(False)   # a source's kind is fixed: add another source for another kind
            self.kindCombo.setToolTip('The kind of a source cannot change; add a new source instead')
        self.pageStack.setCurrentIndex(self.kindCombo.currentIndex())
        self.kindCombo.currentIndexChanged.connect(self.pageStack.setCurrentIndex)
        self._auto_name = '' if existing else unique_name(self.kindCombo.currentData() or 'source', self._taken)
        self.nameEdit.setText(existing.name if existing else self._auto_name)
        self.kindCombo.currentIndexChanged.connect(self.__kind_changed)
        self.nameErrorLabel = QLabel('')
        self.errorLabel = QLabel('')
        self.errorLabel.setVisible(False)
        for label in (self.nameErrorLabel, self.errorLabel):
            label.setWordWrap(True)
            label.setStyleSheet(f'color: {warning_colour().name()}')
        form = QFormLayout()
        form.addRow('Name', self.nameEdit)
        form.addRow('', self.nameErrorLabel)
        form.addRow('Kind', self.kindCombo)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.pageStack)
        layout.addWidget(self.errorLabel)
        layout.addWidget(self.buttons)
        self.nameEdit.textChanged.connect(self.__check_name)
        self.__check_name()
        self.setMinimumWidth(460)

    @property
    def page(self) -> SourcePage:
        ''' The settings page of the chosen kind -- the same class Library Sync shows. '''
        return self.pages[self.kindCombo.currentData()]

    @property
    def spec(self) -> Optional[SourceSpec]:
        ''' The source, once the dialog was accepted. '''
        return self._spec

    def __kind_changed(self, _index: int) -> None:
        ''' A new source is named after its kind, until the person types a name of their own. '''
        if self._existing is None and self.nameEdit.text() == self._auto_name:
            self._auto_name = unique_name(self.kindCombo.currentData() or 'source', self._taken)
            self.nameEdit.setText(self._auto_name)

    def name_problem(self) -> str:
        name = self.nameEdit.text().strip()
        if not name:
            return 'Give the source a name.'
        if name in self._taken:
            return f'There is already a source called "{name}": names must be unique.'
        return ''

    def __check_name(self, *_) -> None:
        problem = self.name_problem()
        self.nameErrorLabel.setText(problem)
        self.nameErrorLabel.setVisible(bool(problem))
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(not problem)

    def accept(self) -> None:
        problem = self.name_problem()
        if problem:
            self.nameErrorLabel.setText(problem)
            return
        kind = self.kindCombo.currentData()
        try:
            settings = self.page.settings()
            spec = SourceSpec(self.nameEdit.text().strip(), kind, settings)
            build_source(kind, settings)   # what a scan would build: the same check the profile file gets
        except ValueError as error:
            self.errorLabel.setText(str(error))
            self.errorLabel.setVisible(True)
            return
        self._spec = spec
        super().accept()


class SourcesTab(QWidget):
    '''
    The profile's sources, top = highest priority, drag to reorder (or Up/Down), add / edit / remove.
    `changed` carries the new tuple of `SourceSpec`s each time it changes, and `renamed(old, new)` says a source got a new
    name (ignore rules that name it follow it).
    '''
    changed = Signal(object)
    renamed = Signal(str, str)

    def __init__(self, prefs, parent=None, run_dialog: Optional[Callable[[QDialog], bool]] = None):
        super().__init__(parent)
        self._prefs = prefs
        self._sources: List[SourceSpec] = []
        self._run_dialog = run_dialog or (lambda dialog: dialog.exec() == QDialog.DialogCode.Accepted)
        self.helpLabel = QLabel(PRIORITY_HELP)
        self.helpLabel.setWordWrap(True)
        self.emptyLabel = QLabel('No library source yet. Add the folders to search, or a JRiver server and browse node.')
        self.emptyLabel.setWordWrap(True)
        self.sourceList = QListWidget()
        self.sourceList.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.sourceList.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.sourceList.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.sourceList.setToolTip('Drag a source up or down to change its priority')
        self.addButton = QPushButton('Add...')
        self.editButton = QPushButton('Edit...')
        self.removeButton = QPushButton('Remove')
        self.upButton = QPushButton('Up')
        self.downButton = QPushButton('Down')
        for button in (self.upButton, self.downButton):
            button.setToolTip('Change the source\'s priority (or drag it)')
        buttons = QHBoxLayout()
        for button in (self.addButton, self.editButton, self.removeButton, self.upButton, self.downButton):
            buttons.addWidget(button)
        buttons.addStretch()
        layout = QVBoxLayout(self)
        layout.addWidget(self.helpLabel)
        layout.addWidget(self.emptyLabel)
        layout.addWidget(self.sourceList, 1)
        layout.addLayout(buttons)
        self.sourceList.model().rowsMoved.connect(self.__rows_moved)
        self.sourceList.itemSelectionChanged.connect(self.__update_buttons)
        self.sourceList.itemDoubleClicked.connect(lambda _item: self.edit_selected())
        self.addButton.clicked.connect(lambda: self.add_source())
        self.editButton.clicked.connect(lambda: self.edit_selected())
        self.removeButton.clicked.connect(lambda: self.remove_selected())
        self.upButton.clicked.connect(lambda: self.move_selected(-1))
        self.downButton.clicked.connect(lambda: self.move_selected(1))
        self.__update_buttons()

    # --- the list ---------------------------------------------------------------------------------------------------

    @property
    def sources(self) -> tuple:
        return tuple(self._sources)

    def set_sources(self, sources: Sequence[SourceSpec]) -> None:
        ''' Shows these sources; nothing is emitted. '''
        selected = self.selected_name()
        self._sources = list(sources)
        self.__fill(selected)

    def __fill(self, select: Optional[str] = None) -> None:
        self.sourceList.blockSignals(True)
        self.sourceList.model().blockSignals(True)
        self.sourceList.clear()
        for number, spec in enumerate(self._sources, 1):
            item = QListWidgetItem(f'{number}.  {spec.name}\n      {describe_spec(spec)}')
            item.setData(Qt.ItemDataRole.UserRole, spec.name)
            item.setToolTip(describe_spec(spec) + ('\nHighest priority' if number == 1 else ''))
            self.sourceList.addItem(item)
            if spec.name == select:
                self.sourceList.setCurrentItem(item)
        self.sourceList.model().blockSignals(False)
        self.sourceList.blockSignals(False)
        self.emptyLabel.setVisible(not self._sources)
        self.__update_buttons()

    def selected_name(self) -> Optional[str]:
        item = self.sourceList.currentItem() if self.sourceList.selectedItems() else None
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def select(self, name: str) -> None:
        for row in range(self.sourceList.count()):
            if self.sourceList.item(row).data(Qt.ItemDataRole.UserRole) == name:
                self.sourceList.setCurrentRow(row)

    def __selected_index(self) -> int:
        name = self.selected_name()
        return next((i for i, s in enumerate(self._sources) if s.name == name), -1)

    def __update_buttons(self) -> None:
        row = self.__selected_index()
        self.editButton.setEnabled(row >= 0)
        self.removeButton.setEnabled(row >= 0)
        self.upButton.setEnabled(row > 0)
        self.downButton.setEnabled(0 <= row < len(self._sources) - 1)

    def __emit(self) -> None:
        self.changed.emit(self.sources)

    # --- priority ---------------------------------------------------------------------------------------------------

    def move_source(self, source: int, target: int) -> None:
        ''' Moves the source at `source` so it is at `target` (0 is the highest priority) -- what a drag does. '''
        if source == target or not (0 <= source < len(self._sources)) or not (0 <= target < len(self._sources)):
            return
        selected = self._sources[source].name
        self._sources = move_item(self._sources, source, target)
        self.__fill(selected)
        self.__emit()

    def move_selected(self, step: int) -> None:
        row = self.__selected_index()
        if row >= 0:
            self.move_source(row, min(max(row + step, 0), len(self._sources) - 1))

    def __rows_moved(self, *_args) -> None:
        ''' A drag (or the model's move API): the list is now in the new order, so the sources follow it. '''
        order = [self.sourceList.item(r).data(Qt.ItemDataRole.UserRole) for r in range(self.sourceList.count())]
        by_name = {s.name: s for s in self._sources}
        if sorted(order) != sorted(by_name):
            return
        selected = self.selected_name()
        self._sources = [by_name[name] for name in order]
        self.__fill(selected)   # renumbers
        self.__emit()

    # --- add, edit, remove ------------------------------------------------------------------------------------------

    def add_source(self) -> bool:
        dialog = SourceDialog(self, self._prefs, None, [s.name for s in self._sources])
        if not self._run_dialog(dialog) or dialog.spec is None:
            return False
        self._sources.append(dialog.spec)   # a new source has the lowest priority: it must not outrank what is there
        self.__fill(dialog.spec.name)
        self.__emit()
        return True

    def edit_selected(self) -> bool:
        row = self.__selected_index()
        if row < 0:
            return False
        old = self._sources[row]
        dialog = SourceDialog(self, self._prefs, old, [s.name for i, s in enumerate(self._sources) if i != row])
        if not self._run_dialog(dialog) or dialog.spec is None:
            return False
        self._sources[row] = dialog.spec
        self.__fill(dialog.spec.name)
        if dialog.spec.name != old.name:
            self.renamed.emit(old.name, dialog.spec.name)
        self.__emit()
        return True

    def remove_selected(self) -> bool:
        row = self.__selected_index()
        if row < 0:
            return False
        del self._sources[row]
        self.__fill()
        self.__emit()
        return True
