'''
The library sources the work list can run against, as the GUI sees them -- design/library-sync-pipeline-plan.md §11.3.

A *kind* is a name, a label and a settings page. The page loads/saves its own preferences and builds a headless
`LibrarySource` (pipeline.library) from what the user entered. The work list's sources tab holds a registry of kinds and shows the
chosen kind's page, so adding Plex or Kodi later means registering one more kind here, not editing the dialog.
This is separate from pipeline.library.registry, which is the headless name -> source-instance registry: a kind also
carries widgets.
'''
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, \
    QSpinBox, QVBoxLayout, QWidget

from model.jriver.connections import SavedConnection, load_connections
from model.browse_node_picker import JRiverBrowseNodePicker
from model.preferences import LIBRARY_FILESYSTEM_GLOBS, LIBRARY_JRIVER_BROWSE_NODE, LIBRARY_JRIVER_BROWSE_PATH, \
    LIBRARY_JRIVER_CONNECTION
from pipeline.library.filesystem import FilesystemLibrarySource
from pipeline.library.jriver import JRiverLibrarySource, list_browse_children
from pipeline.library.pathmap import mappings_from_config
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

    # A page also edits one source of a catalogue *profile* (the work list's settings drawer): the settings of a
    # profile source (pipeline.library.profile.SourceSpec.settings), not the library preferences.

    def load_settings(self, settings: Mapping[str, Any], prefs) -> None:
        ''' Fill the widgets from a profile source's settings. '''
        raise NotImplementedError

    def settings(self) -> Dict[str, Any]:
        '''
        The settings of a profile source these widgets describe.
        :raises ValueError: with a message fit to show the user, if they cannot make a source yet.
        '''
        raise NotImplementedError

    def load_defaults(self, prefs) -> None:
        ''' Fill the widgets for a new profile source: what was last used, where that is useful. '''
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

    def load_settings(self, settings: Mapping[str, Any], prefs) -> None:
        self._loaded = dict(settings)
        self.globsEdit.setPlainText('\n'.join(str(g) for g in settings.get('globs') or []))

    def load_defaults(self, prefs) -> None:
        self._loaded = {}
        self.globsEdit.setPlainText('')

    def settings(self) -> Dict[str, Any]:
        globs = self.globs()
        if not globs:
            raise ValueError('Add at least one folder or glob to search')
        return {**getattr(self, '_loaded', {}), 'globs': globs}   # anything else the file holds for it is kept

    def __add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Choose a folder to search')
        if folder:
            self.globsEdit.setPlainText('\n'.join(self.globs() + [folder]))


def jriver_source_settings(connection: SavedConnection, browse_node_id: int, browse_path: str = '') -> Dict[str, Any]:
    '''
    A saved server and a browse node, as the settings of a JRiver source in a catalogue profile (what
    `pipeline.library.profile.build_source` reads): its login, ssl, path mappings and id-field overrides are copied in,
    so the profile file is complete without Preferences. `browse_path` is only a label, for showing the node.
    '''
    settings: Dict[str, Any] = {'host': connection.host, 'port': connection.port, 'browse_node_id': int(browse_node_id),
                                'ssl': connection.secure}
    if browse_path:
        settings['browse_path'] = browse_path
    if connection.username:
        settings.update(username=connection.username, password=connection.password)
    if connection.path_mappings:
        settings['path_mappings'] = [{'from': m.source, 'to': m.target} for m in connection.path_mappings]
    if connection.field_mappings:
        settings['external_id_fields'] = connection.field_mappings
    return settings


def connection_from_settings(settings: Mapping[str, Any]) -> SavedConnection:
    ''' The server a profile's JRiver source names, as a connection (for a server that is not in Preferences). '''
    try:
        mappings = tuple(mappings_from_config(settings.get('path_mappings')))
    except ValueError:
        mappings = ()
    return SavedConnection(f"{settings.get('host', '')}:{settings.get('port', '')}", settings.get('username') or None,
                           settings.get('password') or None, bool(settings.get('ssl', False)),
                           path_mappings=mappings, field_mappings=dict(settings.get('external_id_fields') or {}))


class JRiverSourcePage(SourcePage):
    ''' A server from Preferences -> JRiver, and the browse node below which its titles live. '''

    def __init__(self, parent=None):
        super().__init__(parent)
        self.serverCombo = QComboBox()
        self.browseNodeSpin = QSpinBox()
        self.browseNodeSpin.setRange(-1, 2147483647)
        self.browseNodeSpin.setToolTip('-1 is the root of the browse tree')
        self.pickNodeButton = QPushButton('Choose...')
        self.pickNodeButton.setToolTip('Pick the node from the server\'s browse tree')
        self.rootNodeButton = QPushButton('Use library root')
        self.rootNodeButton.setToolTip('Use the root of this server\'s browse tree')
        self.nodePathLabel = QLabel('')
        self.nodePathLabel.setWordWrap(True)
        self.helpLabel = QLabel('Servers, path mappings and metadata fields are managed in Preferences > JRiver.')
        self.helpLabel.setWordWrap(True)
        self._loaded: Dict[str, Any] = {}   # the profile source being edited, so what this page does not show is kept
        self.mappingsNote = QLabel('')
        self.mappingsNote.setWordWrap(True)
        self.mappingsNote.setVisible(False)
        self.useSavedButton = QPushButton('Replace profile copy with current Preferences')
        self.useSavedButton.setToolTip('Copies the current login, path mappings and metadata fields from Preferences into this profile source.')
        self.useSavedButton.clicked.connect(self.__use_saved)
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow('Server', self.serverCombo)
        node_row = QHBoxLayout()
        node_row.addWidget(self.pickNodeButton)
        node_row.addWidget(self.rootNodeButton)
        form.addRow('Browse location', node_row)
        form.addRow('', self.nodePathLabel)
        form.addRow(self.helpLabel)
        form.addRow(self.mappingsNote)
        form.addRow(self.useSavedButton)
        self.pickNodeButton.clicked.connect(self.__pick_node)
        self.rootNodeButton.clicked.connect(self.__use_root_node)
        self.serverCombo.currentIndexChanged.connect(self.__update_pick_enabled)
        self.serverCombo.currentIndexChanged.connect(self.__update_mappings_note)
        # typing an id by hand makes any remembered path stale
        self.browseNodeSpin.valueChanged.connect(lambda _: self.nodePathLabel.setText(''))
        self.__update_pick_enabled()

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
        self.nodePathLabel.setText(prefs.get(LIBRARY_JRIVER_BROWSE_PATH))  # after the spin, which clears it
        self.__update_pick_enabled()

    def save(self, prefs) -> None:
        connection = self.selected_connection()
        if connection is not None:
            prefs.set(LIBRARY_JRIVER_CONNECTION, connection.endpoint)
        prefs.set(LIBRARY_JRIVER_BROWSE_NODE, self.browseNodeSpin.value())
        prefs.set(LIBRARY_JRIVER_BROWSE_PATH, self.nodePathLabel.text())

    def __update_pick_enabled(self, *_):
        self.pickNodeButton.setEnabled(self.selected_connection() is not None)

    def __pick_node(self):
        connection = self.selected_connection()
        if connection is None or connection.port is None:
            return

        def fetch(node_id: int):
            return list_browse_children(connection.host, connection.port, node_id, username=connection.username,
                                        password=connection.password, ssl=connection.secure)

        picker = JRiverBrowseNodePicker(self, fetch, self.browseNodeSpin.value())
        if picker.exec():
            self.browseNodeSpin.setValue(picker.selected_node_id)  # clears the label...
            self.nodePathLabel.setText(picker.selected_path)  # ...so set the path afterwards

    def __use_root_node(self):
        self.browseNodeSpin.setValue(-1)
        self.nodePathLabel.setText('Library root')

    def load_defaults(self, prefs) -> None:
        self._loaded = {}
        self.load(prefs)
        self.nodePathLabel.setText('')
        self.browseNodeSpin.setValue(-1)
        self.__update_mappings_note()

    def load_settings(self, settings: Mapping[str, Any], prefs) -> None:
        '''
        The server is the saved one at the same host:port; a server that is not saved (a hand-written profile) is
        listed too, as it is written, so the source can still be looked at and its node changed.
        '''
        self._loaded = dict(settings)
        self.serverCombo.blockSignals(True)
        self.serverCombo.clear()
        for connection in load_connections(prefs):
            self.serverCombo.addItem(connection.label, connection)
        named = connection_from_settings(settings)
        index = next((i for i in range(self.serverCombo.count())
                      if self.serverCombo.itemData(i).endpoint == named.endpoint), -1)
        if index < 0 and settings.get('host'):
            self.serverCombo.addItem(f'{named.endpoint} (not in Preferences)', named)
            index = self.serverCombo.count() - 1
        self.serverCombo.setCurrentIndex(max(index, 0))
        self.serverCombo.blockSignals(False)
        self.browseNodeSpin.setValue(int(settings.get('browse_node_id', -1) or -1))
        self.nodePathLabel.setText(str(settings.get('browse_path') or ''))   # after the spin, which clears it
        self.__update_pick_enabled()
        self.__update_mappings_note()

    def settings(self) -> Dict[str, Any]:
        '''
        The chosen server and node. A profile deliberately stores a copy of its connection settings so command-line and
        scheduled runs do not depend on desktop preferences; use “Replace profile copy...” to refresh that copy.
        '''
        connection = self.selected_connection()
        if connection is None:
            raise ValueError('Add a JRiver server in Preferences > JRiver first')
        if connection.port is None:
            raise ValueError(f'{connection.endpoint} is not a host:port address')
        fresh = jriver_source_settings(connection, self.browseNodeSpin.value(), self.nodePathLabel.text())
        if self._loaded and (self._loaded.get('host'), self._loaded.get('port')) == (connection.host, connection.port):
            kept = {k: v for k, v in self._loaded.items() if k not in ('browse_node_id', 'browse_path')}
            kept['browse_node_id'] = fresh['browse_node_id']
            if 'browse_path' in fresh:
                kept['browse_path'] = fresh['browse_path']
            return kept
        return fresh

    def __differs_from_saved(self) -> bool:
        ''' True if the profile's copy of this server's login and mappings is not what Preferences holds. '''
        connection = self.selected_connection()
        if not self._loaded or connection is None or self._loaded.get('host') != connection.host:
            return False
        saved = jriver_source_settings(connection, 0)
        keys = ('username', 'password', 'ssl', 'path_mappings', 'external_id_fields')
        return any(saved.get(k) != self._loaded.get(k) and (saved.get(k) or self._loaded.get(k)) for k in keys)

    def __update_mappings_note(self, *_):
        differs = self.__differs_from_saved()
        self.mappingsNote.setVisible(differs)
        self.useSavedButton.setVisible(self.selected_connection() is not None)
        if differs:
            self.mappingsNote.setText('This profile has its own copied JRiver login, path mappings and metadata fields; '
                                      'they differ from Preferences > JRiver. Replace the profile copy to update them.')

    def __use_saved(self):
        connection = self.selected_connection()
        if connection is None:
            return
        fresh = jriver_source_settings(connection, self.browseNodeSpin.value(), self.nodePathLabel.text())
        for key in ('username', 'password', 'ssl', 'path_mappings', 'external_id_fields'):
            self._loaded.pop(key, None)
        self._loaded.update({k: v for k, v in fresh.items() if k not in ('browse_node_id', 'browse_path')})
        self.__update_mappings_note()

    def build_source(self) -> LibrarySource:
        connection = self.selected_connection()
        if connection is None:
            raise ValueError('Add a JRiver server in Preferences > JRiver first')
        if connection.port is None:
            raise ValueError(f'{connection.endpoint} is not a host:port address')
        return JRiverLibrarySource(connection.host, connection.port, self.browseNodeSpin.value(),
                                   username=connection.username, password=connection.password,
                                   ssl=connection.secure, path_mappings=connection.path_mappings,
                                   external_id_fields=connection.field_mappings or None)


register_source_kind(LibrarySourceKind('filesystem', 'Filesystem', FilesystemSourcePage))
register_source_kind(LibrarySourceKind('jriver', 'JRiver Media Center', JRiverSourcePage))
