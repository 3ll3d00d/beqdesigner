'''
The list of saved JRiver Media Center (MCWS) servers, shared by everything that talks to one: the DSP filter
manager's zone dialog, and Library Sync's JRiver source.

Storage is the existing `JRIVER_MCWS_CONNECTIONS` preference, `{'host:port': (auth, secure)}` where `auth` is None
or `(username, password)` -- unchanged, so connections saved before this module existed are still there.
`JRiverConnectionsWidget` is the one place they are added, tested and deleted (Preferences -> JRiver).
'''
import logging
import re
from dataclasses import dataclass
from typing import Optional

import qtawesome as qta
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QCheckBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, \
    QPlainTextEdit, QPushButton, QToolButton, QVBoxLayout, QWidget

from model.jriver.mcws import MCWSError, MediaServer
from model.preferences import JRIVER_MCWS_CONNECTIONS

logger = logging.getLogger('jriver.connections')

_ENDPOINT = re.compile(r'^[^\s:/]+:\d+$')


@dataclass(frozen=True)
class SavedConnection:
    endpoint: str  # 'host:port', the preference's key
    username: Optional[str] = None
    password: Optional[str] = None
    secure: bool = False

    @property
    def host(self) -> str:
        return self.endpoint.rpartition(':')[0] or self.endpoint

    @property
    def port(self) -> Optional[int]:
        port = self.endpoint.rpartition(':')[2]
        return int(port) if port.isdigit() else None

    @property
    def label(self) -> str:
        return f"{self.endpoint} [{self.username}]" if self.username else f"{self.endpoint} [Unauthenticated]"

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
    return parse_connections(prefs.get(JRIVER_MCWS_CONNECTIONS))


def save_connections(prefs, connections: list[SavedConnection]) -> None:
    merged = {}
    for connection in connections:
        merged.update(connection.to_preference())
    prefs.set(JRIVER_MCWS_CONNECTIONS, merged)


class JRiverConnectionsWidget(QWidget):
    '''
    Add, test and delete saved JRiver servers. Changes are written to preferences immediately (as the filter
    manager's dialog always did), and announced through `changed`.
    '''
    changed = Signal()
    CONNECTION_ROLE = Qt.ItemDataRole.UserRole + 1

    def __init__(self, prefs, parent=None):
        super().__init__(parent)
        self.__prefs = prefs
        self.__tested: Optional[SavedConnection] = None

        self.savedConnections = QListWidget()
        self.deleteButton = QToolButton()
        self.deleteButton.setIcon(qta.icon('fa5s.trash-alt'))
        self.deleteButton.setToolTip('Delete selected connection')
        self.endpointEdit = QLineEdit()
        self.endpointEdit.setPlaceholderText('host:port, e.g. 192.168.1.10:52199')
        self.httpsCheck = QCheckBox('Use HTTPS')
        self.authCheck = QCheckBox('Authenticate')
        self.usernameEdit = QLineEdit()
        self.passwordEdit = QLineEdit()
        self.passwordEdit.setEchoMode(QLineEdit.EchoMode.Password)
        self.testButton = QPushButton(qta.icon('fa5s.sync'), 'Test')
        self.testButton.setToolTip('Check the connection to Media Center')
        self.addButton = QPushButton(qta.icon('fa5s.plus'), 'Add')
        self.addButton.setToolTip('Save this connection (enabled once a test has passed)')
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
        buttons.addStretch()
        buttons.addWidget(self.testButton)
        buttons.addWidget(self.addButton)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('Saved servers'))
        layout.addLayout(saved)
        layout.addWidget(QLabel('New server'))
        layout.addLayout(form)
        layout.addLayout(buttons)
        layout.addWidget(self.resultText)

        self.endpointEdit.textChanged.connect(self.__inputs_changed)
        self.usernameEdit.textChanged.connect(self.__inputs_changed)
        self.passwordEdit.textChanged.connect(self.__inputs_changed)
        self.httpsCheck.toggled.connect(self.__inputs_changed)
        self.authCheck.toggled.connect(self.__inputs_changed)
        self.savedConnections.itemSelectionChanged.connect(self.__update_buttons)
        self.testButton.clicked.connect(self.__test)
        self.addButton.clicked.connect(self.__add)
        self.deleteButton.clicked.connect(self.__delete_selected)

        for connection in load_connections(prefs):
            self.__add_row(connection)
        self.__inputs_changed()

    def connections(self) -> list[SavedConnection]:
        return [self.savedConnections.item(i).data(self.CONNECTION_ROLE) for i in range(self.savedConnections.count())]

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

    def __inputs_changed(self, *_):
        ''' Any edit invalidates the last test. '''
        self.__tested = None
        self.testButton.setIcon(qta.icon('fa5s.sync'))
        authenticated = self.authCheck.isChecked()
        self.usernameEdit.setEnabled(authenticated)
        self.passwordEdit.setEnabled(authenticated)
        self.__update_buttons()

    def __update_buttons(self):
        entered = self.__entered()
        credentials_ok = not self.authCheck.isChecked() or (bool(entered.username) and bool(entered.password))
        self.testButton.setEnabled(bool(_ENDPOINT.match(entered.endpoint)) and credentials_ok)
        self.addButton.setEnabled(self.__tested is not None)
        self.deleteButton.setEnabled(len(self.savedConnections.selectedItems()) > 0)

    def __test(self):
        entered = self.__entered()
        try:
            entered.to_media_server().authenticate()
        except MCWSError as e:
            self.__tested = None
            self.resultText.setPlainText(f"{e.url} - {e.status_code}\n\n{e.msg}\n\n{e.resp}")
            self.testButton.setIcon(qta.icon('fa5s.times', color='red'))
        else:
            self.__tested = entered
            self.resultText.clear()
            self.testButton.setIcon(qta.icon('fa5s.check', color='green'))
        self.__update_buttons()

    def __add(self):
        if self.__tested is None:
            return
        connections = [c for c in self.connections() if c.endpoint != self.__tested.endpoint]  # re-adding replaces
        connections.append(self.__tested)
        self.__replace_all(connections)
        self.endpointEdit.clear()
        self.usernameEdit.clear()
        self.passwordEdit.clear()

    def __delete_selected(self):
        doomed = {item.data(self.CONNECTION_ROLE).endpoint for item in self.savedConnections.selectedItems()}
        if doomed:
            self.__replace_all([c for c in self.connections() if c.endpoint not in doomed])

    def __replace_all(self, connections: list[SavedConnection]):
        save_connections(self.__prefs, connections)
        self.savedConnections.clear()
        for connection in connections:
            self.__add_row(connection)
        self.__inputs_changed()
        self.changed.emit()
