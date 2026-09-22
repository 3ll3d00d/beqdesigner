'''
The list of saved JRiver Media Center (MCWS) servers, shared by everything that talks to one: the DSP filter
manager's zone dialog, and the work list's JRiver sources.

Storage is the existing `JRIVER_MCWS_CONNECTIONS` preference, `{'host:port': (auth, secure)}` where `auth` is None
or `(username, password)` -- unchanged, so connections saved before this module existed are still there. A server's
alias (the FriendlyName its /Alive reports) lives in a separate `JRIVER_MCWS_ALIASES` preference, `{'host:port':
name}`, so that format stays compatible. Likewise a server's path mappings (its reported Windows folders -> folders on
this machine, see pipeline.library.pathmap) live in `JRIVER_MCWS_PATH_MAPPINGS`, `{'host:port': [[server, local], ...]}`.
`JRiverConnectionsWidget` is the one place they are added, tested and deleted (Preferences -> JRiver).
'''
import logging
import re
from dataclasses import dataclass, field, replace
from typing import Optional

import qtawesome as qta
from qtpy.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from qtpy.QtWidgets import QAbstractItemView, QCheckBox, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, \
    QFileDialog, QLineEdit, QListWidget, QListWidgetItem, QPlainTextEdit, QPushButton, QTableWidget, QTableWidgetItem, QToolButton, \
    QVBoxLayout, QWidget

from model.jriver.mcws import MCWSError, MediaServer
from model.jriver.field_mappings import JRiverFieldMappingsWidget
from model.preferences import JRIVER_MCWS_ALIASES, JRIVER_MCWS_CONNECTIONS, JRIVER_MCWS_FIELD_MAPPINGS, \
    JRIVER_MCWS_PATH_MAPPINGS
from pipeline.library.pathmap import PathMapping

logger = logging.getLogger('jriver.connections')

_ENDPOINT = re.compile(r'^[^\s:/]+:\d+$')


@dataclass(frozen=True)
class SavedConnection:
    endpoint: str  # 'host:port', the preference's key
    username: Optional[str] = None
    password: Optional[str] = None
    secure: bool = False
    alias: Optional[str] = None  # the server's own FriendlyName, once known
    path_mappings: tuple[PathMapping, ...] = ()  # the server's folders as folders on this machine
    # which library fields hold the IMDb/TMDb ids, only where they differ from the defaults:
    # {'movie': {'imdb': ['My field']}} (pipeline.library.jriver.normalise_external_id_fields)
    field_mappings: dict = field(default_factory=dict)

    @property
    def host(self) -> str:
        return self.endpoint.rpartition(':')[0] or self.endpoint

    @property
    def port(self) -> Optional[int]:
        port = self.endpoint.rpartition(':')[2]
        return int(port) if port.isdigit() else None

    @property
    def display_name(self) -> str:
        ''' What identifies the server to a person: its own name, with the address, else just the address. '''
        return f"{self.alias} ({self.endpoint})" if self.alias else self.endpoint

    @property
    def label(self) -> str:
        return f"{self.display_name} [{self.username}]" if self.username else f"{self.display_name} [Unauthenticated]"

    def to_media_server(self) -> MediaServer:
        return MediaServer(self.endpoint, (self.username, self.password) if self.username else None, self.secure)

    def to_preference(self) -> dict:
        return {self.endpoint: ((self.username, self.password) if self.username else None, self.secure)}


def parse_connections(raw: Optional[dict]) -> list[SavedConnection]:
    '''
    :param raw: the JRIVER_MCWS_CONNECTIONS preference. Values are (auth, secure); an older form was a flat
        (username, password, secure), and QSettings may hand back lists for tuples -- all are accepted.
    '''
    connections = []
    for endpoint, value in (raw or {}).items():
        value = list(value) if isinstance(value, (list, tuple)) else []
        if len(value) == 3:
            auth, secure = (value[0], value[1]), value[2]
        elif len(value) == 2:
            auth, secure = value
        else:
            auth, secure = None, False
        auth = tuple(auth) if auth else None
        connections.append(SavedConnection(endpoint, auth[0] if auth else None, auth[1] if auth else None,
                                           bool(secure)))
    return connections


def _parse_mappings(raw) -> tuple[PathMapping, ...]:
    pairs = (list(pair) for pair in (raw or []))
    return tuple(PathMapping(str(pair[0]), str(pair[1])) for pair in pairs if len(pair) == 2 and pair[0])


def load_connections(prefs) -> list[SavedConnection]:
    aliases = prefs.get(JRIVER_MCWS_ALIASES) or {}
    mappings = prefs.get(JRIVER_MCWS_PATH_MAPPINGS) or {}
    fields = prefs.get(JRIVER_MCWS_FIELD_MAPPINGS) or {}
    return [replace(c, alias=aliases.get(c.endpoint) or None, path_mappings=_parse_mappings(mappings.get(c.endpoint)),
                    field_mappings=dict(fields.get(c.endpoint) or {}))
            for c in parse_connections(prefs.get(JRIVER_MCWS_CONNECTIONS))]


def save_connections(prefs, connections: list[SavedConnection]) -> None:
    merged = {}
    for connection in connections:
        merged.update(connection.to_preference())
    prefs.set(JRIVER_MCWS_CONNECTIONS, merged)
    # only for servers still saved, so deleting one drops its alias too
    prefs.set(JRIVER_MCWS_ALIASES, {c.endpoint: c.alias for c in connections if c.alias})
    prefs.set(JRIVER_MCWS_PATH_MAPPINGS,
              {c.endpoint: [[m.source, m.target] for m in c.path_mappings] for c in connections if c.path_mappings})
    prefs.set(JRIVER_MCWS_FIELD_MAPPINGS, {c.endpoint: c.field_mappings for c in connections if c.field_mappings})


def set_field_mappings(prefs, endpoint: str, overrides: dict) -> None:
    ''' Replaces the id-field overrides of an already-saved server; empty means back to the defaults. '''
    stored = dict(prefs.get(JRIVER_MCWS_FIELD_MAPPINGS) or {})
    if overrides:
        stored[endpoint] = overrides
    else:
        stored.pop(endpoint, None)
    prefs.set(JRIVER_MCWS_FIELD_MAPPINGS, stored)


def set_path_mappings(prefs, endpoint: str, mappings) -> None:
    ''' Replaces the mappings of an already-saved server (they are edited without re-testing the connection). '''
    stored = dict(prefs.get(JRIVER_MCWS_PATH_MAPPINGS) or {})
    if mappings:
        stored[endpoint] = [[m.source, m.target] for m in mappings]
    else:
        stored.pop(endpoint, None)
    prefs.set(JRIVER_MCWS_PATH_MAPPINGS, stored)


def remember_alias(prefs, endpoint: str, alias: Optional[str]) -> bool:
    '''
    Records the FriendlyName learned for an already-saved server.
    :return: True if that changed what is stored.
    '''
    if not alias:
        return False
    aliases = dict(prefs.get(JRIVER_MCWS_ALIASES) or {})
    if aliases.get(endpoint) == alias or endpoint not in (prefs.get(JRIVER_MCWS_CONNECTIONS) or {}):
        return False
    aliases[endpoint] = alias
    prefs.set(JRIVER_MCWS_ALIASES, aliases)
    return True


class _TestSignals(QObject):
    finished = Signal(object, object)  # (None on success else the text to show, the server's FriendlyName or None)


class _TestJob(QRunnable):
    ''' Authenticates against a server off the UI thread -- a dead host takes several seconds to time out. '''

    def __init__(self, connection: SavedConnection):
        super().__init__()
        self.signals = _TestSignals()
        self.__connection = connection

    def run(self):
        try:
            server = self.__connection.to_media_server()
            server.authenticate()
            self.signals.finished.emit(None, server.friendly_name)
        except MCWSError as e:
            self.signals.finished.emit(f"{e.url} - {e.status_code}\n\n{e.msg}\n\n{e.resp}", None)
        except Exception as e:
            logger.exception('Unexpected failure testing %s', self.__connection.endpoint)
            self.signals.finished.emit(f'{type(e).__name__}: {e}', None)


class _AliasSignals(QObject):
    found = Signal(str, str)  # endpoint, FriendlyName


class _AliasJob(QRunnable):
    ''' Asks each server for its FriendlyName, quietly: an unreachable one just keeps showing its address. '''

    def __init__(self, connections: list[SavedConnection]):
        super().__init__()
        self.signals = _AliasSignals()
        self.__connections = connections

    def run(self):
        for connection in self.__connections:
            try:
                server = connection.to_media_server()
                server.authenticate()
                if server.friendly_name:
                    self.signals.found.emit(connection.endpoint, server.friendly_name)
            except Exception as e:
                logger.info('No FriendlyName from %s: %s', connection.endpoint, e)


class JRiverConnectionsWidget(QWidget):
    '''
    Add, edit, test and delete saved JRiver servers. Selecting a saved server loads it into the form; change it,
    Test, then Update. With nothing selected the form adds a new server (New clears the selection). Changes are
    written to preferences immediately (as the filter manager's dialog always did) and announced through
    `changed`. Testing runs in the background with a spinner, so the dialog stays responsive while a slow or dead
    host times out.
    '''
    changed = Signal()
    CONNECTION_ROLE = Qt.ItemDataRole.UserRole + 1

    def __init__(self, prefs, parent=None):
        super().__init__(parent)
        self.__prefs = prefs
        self.__tested: Optional[SavedConnection] = None
        self.__job: Optional[_TestJob] = None  # held so its signals outlive run()
        self.__alias_job: Optional[_AliasJob] = None

        self.savedConnections = QListWidget()
        self.deleteButton = QToolButton()
        self.deleteButton.setIcon(qta.icon('fa5s.trash-alt'))
        self.deleteButton.setToolTip('Delete selected connection')
        self.formLabel = QLabel('New server')
        self.endpointEdit = QLineEdit()
        self.endpointEdit.setPlaceholderText('host:port, e.g. 192.168.1.10:52199')
        self.httpsCheck = QCheckBox('Use HTTPS')
        self.authCheck = QCheckBox('Authenticate')
        self.usernameEdit = QLineEdit()
        self.passwordEdit = QLineEdit()
        self.passwordEdit.setEchoMode(QLineEdit.EchoMode.Password)
        self.newButton = QPushButton(qta.icon('fa5s.file'), 'New')
        self.newButton.setToolTip('Clear the selection and enter a new server')
        self.testButton = QPushButton(qta.icon('fa5s.sync'), 'Test')
        self.testButton.setToolTip('Check the connection to Media Center')
        self.addButton = QPushButton(qta.icon('fa5s.plus'), 'Add')
        self.addButton.setToolTip('Save this connection (enabled once a test has passed)')
        self.statusLabel = QLabel('')
        self.resultText = QPlainTextEdit()
        self.resultText.setReadOnly(True)
        self.resultText.setMaximumHeight(80)
        self.mappingsGroup = QGroupBox('Path mappings for the selected server')
        self.mappingsTable = QTableWidget(0, 2)
        self.mappingsTable.setHorizontalHeaderLabels(['Path as JRiver reports it', 'Same folder on this machine'])
        self.mappingsTable.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.mappingsTable.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.mappingsTable.setMaximumHeight(110)
        self.addMappingButton = QPushButton(qta.icon('fa5s.plus'), 'Add')
        self.chooseMappingFolderButton = QPushButton(qta.icon('fa5s.folder-open'), 'Choose local folder')
        self.removeMappingButton = QPushButton(qta.icon('fa5s.minus'), 'Remove')
        mappings_help = QLabel('JRiver reports paths as its own machine sees them, e.g. W:\\Films\\x.mkv. Map each such '
                               'folder to where it is on this machine, e.g. /mnt/films. The longest match wins; a '
                               'path no rule covers is used as reported.')
        mappings_help.setWordWrap(True)
        mapping_buttons = QHBoxLayout()
        mapping_buttons.addStretch()
        mapping_buttons.addWidget(self.addMappingButton)
        mapping_buttons.addWidget(self.chooseMappingFolderButton)
        mapping_buttons.addWidget(self.removeMappingButton)
        mappings_layout = QVBoxLayout(self.mappingsGroup)
        mappings_layout.addWidget(mappings_help)
        mappings_layout.addWidget(self.mappingsTable)
        mappings_layout.addLayout(mapping_buttons)
        self.fieldsGroup = QGroupBox('Metadata fields for the selected server')
        self.fieldMappings = JRiverFieldMappingsWidget()
        QVBoxLayout(self.fieldsGroup).addWidget(self.fieldMappings)

        saved = QHBoxLayout()
        saved.addWidget(self.savedConnections)
        saved.addWidget(self.deleteButton, alignment=Qt.AlignmentFlag.AlignTop)
        form = QFormLayout()
        form.addRow('Server', self.endpointEdit)
        form.addRow('', self.httpsCheck)
        form.addRow('', self.authCheck)
        form.addRow('Username', self.usernameEdit)
        form.addRow('Password', self.passwordEdit)
        buttons = QHBoxLayout()
        buttons.addWidget(self.newButton)
        buttons.addStretch()
        buttons.addWidget(self.statusLabel)
        buttons.addWidget(self.testButton)
        buttons.addWidget(self.addButton)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('Saved servers'))
        layout.addLayout(saved)
        layout.addWidget(self.formLabel)
        layout.addLayout(form)
        layout.addLayout(buttons)
        layout.addWidget(self.resultText)
        layout.addWidget(self.mappingsGroup)
        layout.addWidget(self.fieldsGroup)

        self.__form_fields = (self.endpointEdit, self.httpsCheck, self.authCheck, self.usernameEdit, self.passwordEdit)
        self.endpointEdit.textChanged.connect(self.__inputs_changed)
        self.usernameEdit.textChanged.connect(self.__inputs_changed)
        self.passwordEdit.textChanged.connect(self.__inputs_changed)
        self.httpsCheck.toggled.connect(self.__inputs_changed)
        self.authCheck.toggled.connect(self.__inputs_changed)
        self.savedConnections.itemSelectionChanged.connect(self.__selection_changed)
        self.newButton.clicked.connect(self.__new)
        self.testButton.clicked.connect(self.__test)
        self.addButton.clicked.connect(self.__save)
        self.deleteButton.clicked.connect(self.__delete_selected)
        self.addMappingButton.clicked.connect(self.__add_mapping)
        self.chooseMappingFolderButton.clicked.connect(self.__choose_mapping_folder)
        self.removeMappingButton.clicked.connect(self.__remove_mapping)
        self.mappingsTable.itemChanged.connect(self.__mappings_edited)
        self.mappingsTable.itemSelectionChanged.connect(self.__update_buttons)
        self.fieldMappings.edited.connect(self.__field_mappings_edited)

        for connection in load_connections(prefs):
            self.__add_row(connection)
        self.__inputs_changed()

    def connections(self) -> list[SavedConnection]:
        return [self.savedConnections.item(i).data(self.CONNECTION_ROLE) for i in range(self.savedConnections.count())]

    @property
    def testing(self) -> bool:
        return self.__job is not None

    def __selected(self) -> Optional[SavedConnection]:
        items = self.savedConnections.selectedItems()
        return items[0].data(self.CONNECTION_ROLE) if items else None

    def __add_row(self, connection: SavedConnection):
        item = QListWidgetItem(connection.label)
        item.setData(self.CONNECTION_ROLE, connection)
        self.savedConnections.addItem(item)

    def __entered(self) -> SavedConnection:
        authenticated = self.authCheck.isChecked()
        return SavedConnection(self.endpointEdit.text().strip(),
                               self.usernameEdit.text() if authenticated else None,
                               self.passwordEdit.text() if authenticated else None,
                               self.httpsCheck.isChecked())

    def __fill_form(self, connection: Optional[SavedConnection]):
        ''' Sets every field without each edit invalidating the test, then does that once. '''
        for control in self.__form_fields:
            control.blockSignals(True)
        connection = connection or SavedConnection('')
        self.endpointEdit.setText(connection.endpoint)
        self.httpsCheck.setChecked(connection.secure)
        self.authCheck.setChecked(connection.username is not None)
        self.usernameEdit.setText(connection.username or '')
        self.passwordEdit.setText(connection.password or '')
        for control in self.__form_fields:
            control.blockSignals(False)
        self.__inputs_changed()

    def __selection_changed(self):
        selected = self.__selected()
        self.__fill_form(selected)
        self.__load_mappings(selected)
        self.fieldMappings.set_connection(selected)

    def __load_mappings(self, connection: Optional[SavedConnection]):
        self.mappingsTable.blockSignals(True)
        self.mappingsTable.setRowCount(0)
        for mapping in (connection.path_mappings if connection else ()):
            self.__append_mapping_row(mapping.source, mapping.target)
        self.mappingsTable.blockSignals(False)
        self.__update_buttons()

    def __append_mapping_row(self, source: str = '', target: str = '') -> int:
        row = self.mappingsTable.rowCount()
        self.mappingsTable.insertRow(row)
        self.mappingsTable.setItem(row, 0, QTableWidgetItem(source))
        self.mappingsTable.setItem(row, 1, QTableWidgetItem(target))
        return row

    def __add_mapping(self):
        self.mappingsTable.blockSignals(True)
        row = self.__append_mapping_row()
        self.mappingsTable.blockSignals(False)
        self.mappingsTable.setCurrentCell(row, 0)
        self.mappingsTable.editItem(self.mappingsTable.item(row, 0))

    def __remove_mapping(self):
        rows = sorted({index.row() for index in self.mappingsTable.selectedIndexes()}, reverse=True)
        for row in rows:
            self.mappingsTable.removeRow(row)
        if rows:
            self.__mappings_edited()

    def __choose_mapping_folder(self):
        '''Fill the selected mapping's local side without requiring that mount to exist while it is edited.'''
        rows = sorted({index.row() for index in self.mappingsTable.selectedIndexes()})
        if not rows:
            return
        row = rows[0]
        current = self.mappingsTable.item(row, 1)
        folder = QFileDialog.getExistingDirectory(self, 'Choose local folder', current.text() if current else '')
        if folder:
            if current is None:
                current = QTableWidgetItem()
                self.mappingsTable.setItem(row, 1, current)
            current.setText(folder)

    def __table_mappings(self) -> tuple[PathMapping, ...]:
        ''' Rows with both cells filled; a half-typed row is kept on screen but not saved until it is complete. '''
        mappings = []
        for row in range(self.mappingsTable.rowCount()):
            source = (self.mappingsTable.item(row, 0).text() if self.mappingsTable.item(row, 0) else '').strip()
            target = (self.mappingsTable.item(row, 1).text() if self.mappingsTable.item(row, 1) else '').strip()
            if source and target:
                mappings.append(PathMapping(source, target))
        return tuple(mappings)

    def __field_mappings_edited(self, endpoint: str, overrides: dict):
        for row in range(self.savedConnections.count()):
            item = self.savedConnections.item(row)
            connection = item.data(self.CONNECTION_ROLE)
            if connection.endpoint == endpoint:
                set_field_mappings(self.__prefs, endpoint, overrides)
                item.setData(self.CONNECTION_ROLE, replace(connection, field_mappings=overrides))
                self.changed.emit()
                return

    def __mappings_edited(self, *_):
        items = self.savedConnections.selectedItems()
        if not items:
            return
        connection = items[0].data(self.CONNECTION_ROLE)
        mappings = self.__table_mappings()
        if mappings == connection.path_mappings:
            return
        set_path_mappings(self.__prefs, connection.endpoint, mappings)
        items[0].setData(self.CONNECTION_ROLE, replace(connection, path_mappings=mappings))
        self.changed.emit()

    def __new(self):
        self.savedConnections.clearSelection()  # its signal empties the form
        self.__fill_form(None)

    def __inputs_changed(self, *_):
        ''' Any edit invalidates the last test. '''
        self.__tested = None
        self.testButton.setIcon(qta.icon('fa5s.sync'))
        self.__update_buttons()

    def __update_buttons(self):
        editing = self.__selected() is not None
        busy = self.testing
        authenticated = self.authCheck.isChecked()
        entered = self.__entered()
        credentials_ok = not authenticated or (bool(entered.username) and bool(entered.password))
        self.formLabel.setText('Edit server' if editing else 'New server')
        self.addButton.setText('Update' if editing else 'Add')
        self.addButton.setIcon(qta.icon('fa5s.save' if editing else 'fa5s.plus'))
        self.usernameEdit.setEnabled(authenticated and not busy)
        self.passwordEdit.setEnabled(authenticated and not busy)
        for control in (self.endpointEdit, self.httpsCheck, self.authCheck):
            control.setEnabled(not busy)
        self.testButton.setEnabled(not busy and bool(_ENDPOINT.match(entered.endpoint)) and credentials_ok)
        self.addButton.setEnabled(not busy and self.__tested is not None)
        self.deleteButton.setEnabled(not busy and editing)
        self.newButton.setEnabled(not busy)
        self.savedConnections.setEnabled(not busy)
        self.mappingsGroup.setEnabled(editing and not busy)
        self.fieldsGroup.setEnabled(editing and not busy)
        self.removeMappingButton.setEnabled(bool(self.mappingsTable.selectedIndexes()))
        self.chooseMappingFolderButton.setEnabled(editing and not busy and bool(self.mappingsTable.selectedIndexes()))

    def __test(self):
        if self.testing:
            return
        entered = self.__entered()
        self.resultText.clear()
        self.statusLabel.setText('Testing...')
        self.testButton.setIcon(qta.icon('fa5s.spinner', animation=qta.Spin(self.testButton)))
        job = _TestJob(entered)
        job.signals.finished.connect(lambda error, name, entered=entered: self.__test_finished(entered, error, name))
        self.__job = job
        self.__update_buttons()
        QThreadPool.globalInstance().start(job)

    def __test_finished(self, entered: SavedConnection, error: Optional[str], friendly_name: Optional[str] = None):
        self.__job = None
        self.statusLabel.setText('')
        if error is None:
            known = next((c.alias for c in self.connections() if c.endpoint == entered.endpoint), None)
            self.__tested = replace(entered, alias=friendly_name or known)
            if friendly_name:
                self.statusLabel.setText(f'Connected to {friendly_name}')
            self.resultText.clear()
            self.testButton.setIcon(qta.icon('fa5s.check', color='green'))
        else:
            self.__tested = None
            self.resultText.setPlainText(error)
            self.testButton.setIcon(qta.icon('fa5s.times', color='red'))
        self.__update_buttons()

    def __save(self):
        tested = self.__tested
        if tested is None:
            return
        original = self.__selected()
        replaced = {tested.endpoint} | ({original.endpoint} if original else set())  # an edit may rename the server
        base = original or next((c for c in self.connections() if c.endpoint == tested.endpoint), None)
        if base is not None:
            # the form edits neither of these
            tested = replace(tested, path_mappings=base.path_mappings, field_mappings=base.field_mappings)
        connections = [c for c in self.connections() if c.endpoint not in replaced] + [tested]
        self.__replace_all(connections, select=tested.endpoint if original else None)

    def __delete_selected(self):
        doomed = {item.data(self.CONNECTION_ROLE).endpoint for item in self.savedConnections.selectedItems()}
        if doomed:
            self.__replace_all([c for c in self.connections() if c.endpoint not in doomed])

    def __replace_all(self, connections: list[SavedConnection], select: Optional[str] = None):
        '''
        Persist and redisplay. `select` re-selects an edited server (keeping the form on it); otherwise the
        selection and form are cleared, ready for the next new server.
        '''
        save_connections(self.__prefs, connections)
        self.savedConnections.clear()
        for connection in connections:
            self.__add_row(connection)
        for row in range(self.savedConnections.count()):
            if select is not None and self.savedConnections.item(row).data(self.CONNECTION_ROLE).endpoint == select:
                self.savedConnections.setCurrentRow(row)
                break
        else:
            self.__fill_form(None)
        self.changed.emit()

    def refresh_aliases(self) -> None:
        '''
        Fills in the FriendlyName of every saved server that has none yet, in the background -- for servers
        saved before names were recorded. Servers that don't answer are left showing their address.
        '''
        unnamed = [c for c in self.connections() if not c.alias]
        if not unnamed or self.__alias_job is not None:
            return
        job = _AliasJob(unnamed)
        job.signals.found.connect(self.__alias_found)
        self.__alias_job = job
        QThreadPool.globalInstance().start(job)

    def __alias_found(self, endpoint: str, alias: str):
        if not remember_alias(self.__prefs, endpoint, alias):
            return
        for row in range(self.savedConnections.count()):
            item = self.savedConnections.item(row)
            connection = item.data(self.CONNECTION_ROLE)
            if connection.endpoint == endpoint:
                named = replace(connection, alias=alias)
                item.setData(self.CONNECTION_ROLE, named)
                item.setText(named.label)
        self.changed.emit()
