'''
Which of a JRiver library's fields hold the IMDb / TMDb ids -- design/library-sync-pipeline-plan.md §11.7.

Users keep these in different fields, so the choice is theirs. The box for each is editable text (one or more field
names, comma separated, first with a value wins), with the server's own fields offered from Library/Fields, which is
fetched in the background on request. What is stored is only what differs from the defaults, so a default that later
improves reaches everyone who never customised it.
'''
import logging
from dataclasses import replace
from typing import Optional

import qtawesome as qta
from qtpy.QtCore import QObject, QRunnable, QThreadPool, Signal
from qtpy.QtWidgets import QComboBox, QGridLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from pipeline.library.jriver import DEFAULT_EXTERNAL_ID_FIELDS, IDENTIFIERS, list_library_fields, \
    normalise_external_id_fields

logger = logging.getLogger('jriver.field_mappings')

KIND_LABELS = {'movie': 'Films', 'tv': 'TV shows'}
IDENTIFIER_LABELS = {'imdb': 'IMDb id field(s)', 'tmdb': 'TMDb id field(s)'}
# what an id can plausibly live in; dates, paths, images and the like are not offered
ID_DATA_TYPES = {'String', 'Integer'}


def overrides_from(effective: dict) -> dict:
    ''' The part of `effective` (by kind then identifier) that differs from the defaults, as plain lists. '''
    overrides: dict = {}
    for kind, by_identifier in effective.items():
        for identifier, names in by_identifier.items():
            if tuple(names) != DEFAULT_EXTERNAL_ID_FIELDS[kind][identifier]:
                overrides.setdefault(kind, {})[identifier] = list(names)
    return overrides


def split_names(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(',') if part.strip())


class _FieldsSignals(QObject):
    fetched = Signal(object)
    failed = Signal(str)


class _FieldsJob(QRunnable):
    ''' Library/Fields off the UI thread. `fetch` is injected so tests need no server. '''

    def __init__(self, fetch):
        super().__init__()
        self.signals = _FieldsSignals()
        self.__fetch = fetch

    def run(self):
        try:
            self.signals.fetched.emit(self.__fetch())
        except Exception as e:
            logger.exception('Unable to list library fields')
            self.signals.failed.emit(f'{type(e).__name__}: {e}')


class JRiverFieldMappingsWidget(QWidget):
    '''
    Edits the id fields of one server at a time (set_connection). Edits are announced through `edited` as
    (endpoint, overrides) for the owner to persist; nothing here writes preferences.
    '''
    edited = Signal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.__connection = None
        self.__known: set[str] = set()  # every field the server defines
        self.__offered = 0  # how many of them are offered as choices
        self.__job: Optional[_FieldsJob] = None
        self.__loading_form = False
        self.combos: dict[tuple[str, str], QComboBox] = {}

        grid = QGridLayout()
        for column, identifier in enumerate(IDENTIFIERS, start=1):
            grid.addWidget(QLabel(IDENTIFIER_LABELS[identifier]), 0, column)
        for row, kind in enumerate(DEFAULT_EXTERNAL_ID_FIELDS, start=1):
            grid.addWidget(QLabel(KIND_LABELS[kind]), row, 0)
            for column, identifier in enumerate(IDENTIFIERS, start=1):
                combo = QComboBox()
                combo.setEditable(True)
                combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
                combo.setToolTip('One or more field names, comma separated; the first with a value wins. '
                                 'Leave empty to ignore this id.')
                combo.lineEdit().editingFinished.connect(self.__committed)
                combo.activated.connect(self.__committed)
                self.combos[(kind, identifier)] = combo
                grid.addWidget(combo, row, column)
        self.loadButton = QPushButton(qta.icon('fa5s.download'), 'Load fields from server')
        self.loadButton.setToolTip("List the fields this server's library defines, to choose from")
        self.resetButton = QPushButton(qta.icon('fa5s.undo'), 'Defaults')
        self.resetButton.setToolTip('Go back to the default fields')
        self.statusLabel = QLabel('')
        self.statusLabel.setWordWrap(True)
        help_label = QLabel("Which library fields hold each title's IMDb and TMDb id. TV shows need the series id, "
                            "not the episode's.")
        help_label.setWordWrap(True)
        buttons = QGridLayout()
        buttons.addWidget(self.loadButton, 0, 0)
        buttons.addWidget(self.resetButton, 0, 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(help_label)
        layout.addLayout(grid)
        layout.addLayout(buttons)
        layout.addWidget(self.statusLabel)
        self.loadButton.clicked.connect(self.load_fields)
        self.resetButton.clicked.connect(self.__reset)
        self.set_connection(None)

    @property
    def loading(self) -> bool:
        return self.__job is not None

    def set_connection(self, connection) -> None:
        ''' Shows the effective fields of `connection` (a SavedConnection), or nothing to edit if None. '''
        self.__connection = connection
        self.__known = set()
        self.__offered = 0
        self.statusLabel.setText('')
        effective = normalise_external_id_fields(connection.field_mappings if connection else None)
        self.__loading_form = True
        for (kind, identifier), combo in self.combos.items():
            combo.clear()
            combo.setEditText(', '.join(effective[kind][identifier]))
        self.__loading_form = False
        self.__update_enabled()

    def __update_enabled(self):
        editable = self.__connection is not None and not self.loading
        for combo in self.combos.values():
            combo.setEnabled(editable)
        self.loadButton.setEnabled(editable)
        self.resetButton.setEnabled(editable)

    def effective(self) -> dict:
        return {kind: {identifier: split_names(self.combos[(kind, identifier)].currentText())
                       for identifier in IDENTIFIERS} for kind in DEFAULT_EXTERNAL_ID_FIELDS}

    def __committed(self, *_):
        if self.__loading_form or self.__connection is None:
            return
        overrides = overrides_from(self.effective())
        self.__warn_about_unknown()
        if overrides != self.__connection.field_mappings:
            self.__connection = replace(self.__connection, field_mappings=overrides)
            self.edited.emit(self.__connection.endpoint, overrides)

    def __reset(self):
        if self.__connection is None:
            return
        self.set_connection(replace(self.__connection, field_mappings={}))
        self.edited.emit(self.__connection.endpoint, {})

    def __warn_about_unknown(self):
        if not self.__known:
            return
        unknown = sorted({name for by_identifier in self.effective().values() for names in by_identifier.values()
                          for name in names if name not in self.__known})
        self.statusLabel.setText(f"Not a field on this server: {', '.join(unknown)}" if unknown
                                 else f'{self.__offered} candidate fields loaded')

    def load_fields(self, fetch=None):
        '''
        Fetches the server's fields in the background and offers them in every box.
        :param fetch: replaces the real Library/Fields call; for tests.
        '''
        connection = self.__connection
        if connection is None or self.loading:
            return
        if not callable(fetch):
            def fetch():
                return list_library_fields(connection.host, connection.port, username=connection.username,
                                           password=connection.password, ssl=connection.secure)
        job = _FieldsJob(fetch)
        job.signals.fetched.connect(self.__fields_loaded)
        job.signals.failed.connect(self.__fields_failed)
        self.__job = job
        self.statusLabel.setText('Loading fields...')
        self.loadButton.setIcon(qta.icon('fa5s.spinner', animation=qta.Spin(self.loadButton)))
        self.__update_enabled()
        QThreadPool.globalInstance().start(job)

    def __finish_loading(self):
        self.__job = None
        self.loadButton.setIcon(qta.icon('fa5s.download'))
        self.__update_enabled()

    def __fields_loaded(self, fields):
        self.__finish_loading()
        names = sorted({f.name for f in fields if f.data_type in ID_DATA_TYPES}, key=str.lower)
        self.__known = {f.name for f in fields}
        self.__offered = len(names)
        self.__loading_form = True
        for combo in self.combos.values():
            text = combo.currentText()
            combo.clear()
            combo.addItems(names)
            combo.setEditText(text)
        self.__loading_form = False
        self.__warn_about_unknown()

    def __fields_failed(self, message: str):
        self.__finish_loading()
        self.statusLabel.setText(f'Could not load fields: {message}')
