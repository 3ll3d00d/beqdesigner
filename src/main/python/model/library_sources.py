'''
The library sources Library Sync can run against, as the GUI sees them -- design/library-sync-pipeline-plan.md §11.3.

A *kind* is a name, a label and a settings page. The page loads/saves its own preferences and builds a headless
`LibrarySource` (pipeline.library) from what the user entered. Library Sync holds a registry of kinds and shows the
chosen kind's page, so adding Plex or Kodi later means registering one more kind here, not editing the dialog.
This is separate from pipeline.library.registry, which is the headless name -> source-instance registry: a kind also
carries widgets.
'''
from dataclasses import dataclass
from typing import Callable, Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, \
    QSpinBox, QVBoxLayout, QWidget

from model.jriver.connections import SavedConnection, load_connections
from model.preferences import LIBRARY_FILESYSTEM_GLOBS, LIBRARY_JRIVER_BROWSE_NODE, LIBRARY_JRIVER_CONNECTION
from pipeline.library.filesystem import FilesystemLibrarySource
from pipeline.library.jriver import JRiverLibrarySource
from pipeline.library.source import LibrarySource


class SourcePage(QWidget):
    ''' Settings for one kind of source. '''

    def load(self, prefs) -> None:
        ''' Fill the widgets from preferences. '''
        raise NotImplementedError

    def save(self, prefs) -> None:
        ''' Persist the widgets to preferences. '''
        raise NotImplementedError

    def build_source(self) -> LibrarySource:
        '''
        :raises ValueError: with a message fit to show the user, if the settings can't make a source yet.
        '''
        raise NotImplementedError


@dataclass(frozen=True)
class LibrarySourceKind:
    name: str  # what LIBRARY_SOURCE_DEFAULT stores
    label: str
    create_page: Callable[[], SourcePage]


_kinds: dict[str, LibrarySourceKind] = {}


def register_source_kind(kind: LibrarySourceKind) -> None:
    _kinds[kind.name] = kind


def unregister_source_kind(name: str) -> None:
    _kinds.pop(name, None)


def registered_source_kinds() -> list[LibrarySourceKind]:
    ''' In registration order, which is the order the picker shows them. '''
    return list(_kinds.values())


class FilesystemSourcePage(SourcePage):
    ''' Batch Extract's "raw filesystem" search: globs and/or folders, one per line. '''

    def __init__(self, parent=None):
        super().__init__(parent)
        self.globsEdit = QPlainTextEdit()
        self.globsEdit.setPlaceholderText('One per line: a folder, or a glob such as /films/**/*.mkv')
        self.globsEdit.setMaximumHeight(90)
        self.addFolderButton = QPushButton('Add folder...')
        self.addFolderButton.clicked.connect(self.__add_folder)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel('Folders and globs to search'))
        layout.addWidget(self.globsEdit)
        row = QHBoxLayout()
        row.addWidget(self.addFolderButton)
        row.addStretch()
        layout.addLayout(row)

    def globs(self) -> list[str]:
        return [line.strip() for line in self.globsEdit.toPlainText().splitlines() if line.strip()]

    def load(self, prefs) -> None:
        self.globsEdit.setPlainText('\n'.join(prefs.get(LIBRARY_FILESYSTEM_GLOBS) or []))

    def save(self, prefs) -> None:
        prefs.set(LIBRARY_FILESYSTEM_GLOBS, self.globs())

    def build_source(self) -> LibrarySource:
        globs = self.globs()
        if not globs:
            raise ValueError('Add at least one folder or glob to search')
        return FilesystemLibrarySource(globs)

    def __add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Choose a folder to search')
        if folder:
            self.globsEdit.setPlainText('\n'.join(self.globs() + [folder]))


class JRiverSourcePage(SourcePage):
    ''' A server from Preferences -> JRiver, and the browse node below which its titles live. '''

    def __init__(self, parent=None):
        super().__init__(parent)
        self.serverCombo = QComboBox()
        self.browseNodeSpin = QSpinBox()
        self.browseNodeSpin.setRange(-1, 2147483647)
        self.browseNodeSpin.setToolTip('-1 is the root of the browse tree')
        self.helpLabel = QLabel('Servers are managed in Preferences > JRiver.')
        self.helpLabel.setWordWrap(True)
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow('Server', self.serverCombo)
        form.addRow('Browse node ID', self.browseNodeSpin)
        form.addRow(self.helpLabel)

    def selected_connection(self) -> Optional[SavedConnection]:
        return self.serverCombo.currentData(Qt.ItemDataRole.UserRole)

    def load(self, prefs) -> None:
        self.serverCombo.clear()
        for connection in load_connections(prefs):
            self.serverCombo.addItem(connection.label, connection)
        wanted = prefs.get(LIBRARY_JRIVER_CONNECTION)
        for i in range(self.serverCombo.count()):
            if self.serverCombo.itemData(i).endpoint == wanted:
                self.serverCombo.setCurrentIndex(i)
        self.browseNodeSpin.setValue(prefs.get(LIBRARY_JRIVER_BROWSE_NODE))

    def save(self, prefs) -> None:
        connection = self.selected_connection()
        if connection is not None:
            prefs.set(LIBRARY_JRIVER_CONNECTION, connection.endpoint)
        prefs.set(LIBRARY_JRIVER_BROWSE_NODE, self.browseNodeSpin.value())

    def build_source(self) -> LibrarySource:
        connection = self.selected_connection()
        if connection is None:
            raise ValueError('Add a JRiver server in Preferences > JRiver first')
        if connection.port is None:
            raise ValueError(f'{connection.endpoint} is not a host:port address')
        return JRiverLibrarySource(connection.host, connection.port, self.browseNodeSpin.value(),
                                   username=connection.username, password=connection.password,
                                   ssl=connection.secure)


register_source_kind(LibrarySourceKind('filesystem', 'Filesystem', FilesystemSourcePage))
register_source_kind(LibrarySourceKind('jriver', 'JRiver Media Center', JRiverSourcePage))
