'''
The list of saved JRiver Media Center (MCWS) servers, shared by everything that talks to one: the DSP filter
manager's zone dialog, and Library Sync's JRiver source.

Storage is the existing `JRIVER_MCWS_CONNECTIONS` preference, `{'host:port': (auth, secure)}` where `auth` is None
or `(username, password)` -- unchanged, so connections saved before this module existed are still there. A server's
alias (the FriendlyName its /Alive reports) lives in a separate `JRIVER_MCWS_ALIASES` preference, `{'host:port':
name}`, so that format stays compatible.
`JRiverConnectionsWidget` is the one place they are added, tested and deleted (Preferences -> JRiver).
'''
import logging
import re
from dataclasses import dataclass, replace
from typing import Optional

import qtawesome as qta
from qtpy.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from qtpy.QtWidgets import QCheckBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, \
    QPlainTextEdit, QPushButton, QToolButton, QVBoxLayout, QWidget

from model.jriver.mcws import MCWSError, MediaServer
from model.preferences import JRIVER_MCWS_ALIASES, JRIVER_MCWS_CONNECTIONS

logger = logging.getLogger('jriver.connections')

_ENDPOINT = re.compile(r'^[^\s:/]+:\d+$')


@dataclass(frozen=True)
class SavedConnection:
    endpoint: str  # 'host:port', the preference's key
    username: Optional[str] = None
    password: Optional[str] = None
    secure: bool = False
    alias: Optional[str] = None  # the server's own FriendlyName, once known

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


def load_connections(prefs) -> list[SavedConnection]:
    aliases = prefs.get(JRIVER_MCWS_ALIASES) or {}
    return [replace(c, alias=aliases.get(c.endpoint) or None)
            for c in parse_connections(prefs.get(JRIVER_MCWS_CONNECTIONS))]


def save_connections(prefs, connections: list[SavedConnection]) -> None:
    merged = {}
    for connection in connections:
        merged.update(connection.to_preference())
    prefs.set(JRIVER_MCWS_CONNECTIONS, merged)
    # only for servers still saved, so deleting one drops its alias too
    prefs.set(JRIVER_MCWS_ALIASES, {c.endpoint: c.alias for c in connections if c.alias})


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
        for field in self.__form_fields:
            field.blockSignals(True)
        connection = connection or SavedConnection('')
        self.endpointEdit.setText(connection.endpoint)
        self.httpsCheck.setChecked(connection.secure)
        self.authCheck.setChecked(connection.username is not None)
        self.usernameEdit.setText(connection.username or '')
        self.passwordEdit.setText(connection.password or '')
        for field in self.__form_fields:
            field.blockSignals(False)
        self.__inputs_changed()

    def __selection_changed(self):
        self.__fill_form(self.__selected())

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
        for field in (self.endpointEdit, self.httpsCheck, self.authCheck):
            field.setEnabled(not busy)
        self.testButton.setEnabled(not busy and bool(_ENDPOINT.match(entered.endpoint)) and credentials_ok)
        self.addButton.setEnabled(not busy and self.__tested is not None)
        self.deleteButton.setEnabled(not busy and editing)
        self.newButton.setEnabled(not busy)
        self.savedConnections.setEnabled(not busy)

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
