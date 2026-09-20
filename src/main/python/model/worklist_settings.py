'''
The work list's settings drawer -- design/library-sync/workflow-rework/design.md §12.4 and §12.10, chunk 26c.

It edits the **profile file** (the `LIBRARY_PROFILE_PATH` preference names it) and nothing else that matters: the file is the
durable state, the index a disposable cache. Everything the drawer does not manage -- the `designers:` section, unknown
sections, the rest of `run:` and `sync:` -- is kept, because every edit is a new `Profile` written with `Profile.to_config()`.

* **Persistence.** An edit is validated (a folder that can be used, a repository that is a git repository, a rule that
  `rule_from_config` accepts, sources with unique names), then the *whole* profile is checked to read back
  (`profile_from_config(profile.to_config())`), and a short moment later written atomically (`save_profile`: a temporary
  file, then `os.replace`). A profile that would not read back is not written: the reason is shown instead. The first save,
  when no profile file exists yet (the work list was running from the Library Sync preferences), asks where to put one
  (the app's configuration folder is offered), and stores the path in `LIBRARY_PROFILE_PATH`.
* **What the window does with it.** `saved(path)` says the file was written: the window reads the setup again, and says the
  list is out of date (Rescan) if what a scan describes changed. The drawer disables itself while a run or scan is going.

The three tabs are *Locations and options* (this module), *Sources* (`worklist_sources`) and *Ignore* (`worklist_ignore`).
The TMDB API key is not here (a secret does not belong in a file that gets shared): it stays in Preferences.
'''
import logging
import os
import time
from dataclasses import replace
from typing import Callable, Dict, List, Optional, Sequence

from qtpy.QtCore import QStandardPaths, QTimer, Qt, Signal
from qtpy.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, \
    QLineEdit, QPushButton, QScrollArea, QTabWidget, QVBoxLayout, QWidget

from model.preferences import LIBRARY_PROFILE_PATH, TMDB_API_KEY, WORKLIST_ACCEPT_THRESHOLD
from model.worklist_edit import LEVEL_ERROR, LEVEL_INFO, LEVEL_OK, PathCheck, check_directory, check_relative_dir, \
    check_repository, config_value, remote_owner_and_name, with_config
from model.worklist_ignore import IgnoreTab
from model.worklist_model import warning_colour
from model.worklist_profile import WorkListSetup
from model.worklist_sources import SourcesTab
from pipeline.library.index import TitleRow
from pipeline.library.profile import Profile, profile_from_config, save_profile
from pipeline.library.season import TV_MODES

logger = logging.getLogger('worklist.settings')

PROFILE_FILE_NAME = 'library-profile.yaml'
DEBOUNCE_MS = 400
TV_MODE_LABELS = {'episode': 'One filter per episode', 'season': 'Whole season as a single track'}
_PROFILE_FILTER = 'Library profile (*.yaml *.yml *.json);;All files (*)'


def default_profile_path() -> str:
    '''
    Where a new profile file is offered: **the app's configuration folder**, not the work directory. The profile is the one
    durable record of what the catalogue is made from, and the work directory is where extracted audio and the disposable
    index live, which a person may clear to reclaim space.
    '''
    root = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppConfigLocation) or os.path.expanduser('~')
    return os.path.join(root, PROFILE_FILE_NAME)


def choose_profile_file(parent, default: str, overwrite_ok: bool) -> str:
    ''' The file dialog for the profile file. :return: the chosen path, or '' if cancelled. '''
    if default and not os.path.isdir(os.path.dirname(default)):
        os.makedirs(os.path.dirname(default), exist_ok=True)   # so the dialog opens in the configuration folder
    options = QFileDialog.Option(0) if overwrite_ok else QFileDialog.Option.DontConfirmOverwrite
    path, _ = QFileDialog.getSaveFileName(parent, 'Library profile file', default, _PROFILE_FILTER, options=options)
    return path


class _PathRow(QWidget):
    '''
    A text field, an optional Browse button and a line saying whether the value can be used. `committed(text)` is emitted when
    editing finishes (Enter, leaving the field, or a folder chosen), not on every keystroke.
    '''
    committed = Signal(str)

    def __init__(self, browse: Optional[Callable[[str], str]] = None, placeholder: str = '', parent=None):
        super().__init__(parent)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        self.status = QLabel('')
        self.status.setWordWrap(True)
        self.status.setVisible(False)
        self.browseButton = QPushButton('Browse...') if browse else None
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.edit, 1)
        if self.browseButton:
            row.addWidget(self.browseButton)
            self.browseButton.clicked.connect(lambda: self.__browse(browse))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        layout.addLayout(row)
        layout.addWidget(self.status)
        self.edit.editingFinished.connect(lambda: self.committed.emit(self.edit.text().strip()))

    def __browse(self, browse) -> None:
        chosen = browse(self.edit.text().strip())
        if chosen:
            self.edit.setText(chosen)
            self.committed.emit(chosen)

    def set_text(self, text: str) -> None:
        self.edit.blockSignals(True)
        self.edit.setText(text)
        self.edit.blockSignals(False)

    def show_check(self, check: Optional[PathCheck]) -> None:
        self.status.setVisible(check is not None)
        if check is None:
            self.status.setText('')
            return
        colour = {LEVEL_ERROR: warning_colour().name(), LEVEL_OK: '#3a8f3a', LEVEL_INFO: ''}.get(check.level, '')
        self.status.setStyleSheet(f'color: {colour}' if colour else 'color: palette(mid)')
        self.status.setText(check.message)


class SettingsDrawer(QWidget):
    '''
    The settings editor, meant to sit in a dock beside the work list.
    :param rows_provider: the index's rows as the window shows them, for the live "would ignore N titles" count.
    :param choose_path: `(default, overwrite_ok) -> path` asks where the profile file goes ('' = cancelled); a dialog by
        default (tests hand in a function).
    :param run_dialog: how the source and rule dialogs are run (tests fill them in and accept them).
    :param debounce_ms: how long after the last edit the file is written.
    '''
    saved = Signal(str)                   # the profile file was written (its path)
    profile_file_changed = Signal(str)    # another profile file was chosen: the window reads it
    preferences_requested = Signal()      # the TMDB key's link
    edited = Signal()                     # an edit was made (it may not be written yet)

    def __init__(self, parent, preferences, *, rows_provider: Callable[[], Sequence[TitleRow]] = lambda: (),
                 choose_path: Optional[Callable[[str, bool], str]] = None, run_dialog=None,
                 debounce_ms: int = DEBOUNCE_MS, default_path: Callable[[], str] = default_profile_path):
        super().__init__(parent)
        self._prefs = preferences
        self._rows_provider = rows_provider
        self._choose_path = choose_path or (lambda default, overwrite_ok: choose_profile_file(self, default, overwrite_ok))
        self._default_path = default_path
        self._setup: Optional[WorkListSetup] = None
        self._profile: Optional[Profile] = None
        self._path = ''
        self._pending = False
        self._loading = False
        self._error = ''
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(debounce_ms)
        self._timer.timeout.connect(self.flush)
        self.__build(run_dialog)

    # --- construction -----------------------------------------------------------------------------------------------

    def __build(self, run_dialog) -> None:
        self.pathLabel = QLabel('')
        self.pathLabel.setWordWrap(True)
        self.pathLabel.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.changeFileButton = QPushButton('Change...')
        self.changeFileButton.setToolTip('Use another profile file (an existing one is read; a new name starts a new file '
                                         'from these settings)')
        self.changeFileButton.clicked.connect(lambda: self.change_profile_file())
        self.statusLabel = QLabel('')
        self.statusLabel.setWordWrap(True)
        self.statusLabel.setTextFormat(Qt.TextFormat.PlainText)
        self.busyLabel = QLabel('A run or scan is in progress: settings can be changed when it ends.')
        self.busyLabel.setWordWrap(True)
        self.busyLabel.setVisible(False)
        header = QHBoxLayout()
        header.addWidget(self.pathLabel, 1)
        header.addWidget(self.changeFileButton)

        self.workDir = _PathRow(lambda start: QFileDialog.getExistingDirectory(self, 'Work directory', start),
                                'Where extracted audio and projects go')
        self.queueDir = _PathRow(lambda start: QFileDialog.getExistingDirectory(self, 'Review queue directory', start),
                                 'Where designed titles wait for review')
        self.xmlRepo = _PathRow(lambda start: QFileDialog.getExistingDirectory(self, 'XML repository', start),
                                'A clone of the BEQ filter (XML) repository')
        self.xmlDir = _PathRow(None, "A folder inside the repository (empty: its top folder)")
        self.imagesRepo = _PathRow(lambda start: QFileDialog.getExistingDirectory(self, 'Images repository', start),
                                   'A clone of the image repository (optional)')
        self.imageDir = _PathRow(None, "A folder inside the repository (empty: its top folder)")
        self.imageOwner = QLineEdit()
        self.imageOwner.setPlaceholderText('Optional: the GitHub owner of the images repository')
        self.imageRepoName = QLineEdit()
        self.imageRepoName.setPlaceholderText('Optional: the images repository\'s GitHub name')
        self.imageNote = QLabel('')
        self.imageNote.setWordWrap(True)
        self.imageNote.setStyleSheet('color: palette(mid)')
        self.designerCombo = QComboBox()
        self.tvModeCombo = QComboBox()
        for mode in TV_MODES:
            self.tvModeCombo.addItem(TV_MODE_LABELS.get(mode, mode), mode)
        self.keepMultichannel = QCheckBox('Keep the multichannel extraction (and write a multichannel project)')
        self.acceptThreshold = QDoubleSpinBox()
        self.acceptThreshold.setRange(0.0, 1.0)
        self.acceptThreshold.setSingleStep(0.05)
        self.acceptThreshold.setDecimals(2)
        self.acceptThreshold.setToolTip('Bulk accept takes each title\'s top pick only at or above this confidence')
        self.tmdbLabel = QLabel('')
        self.tmdbLabel.setWordWrap(True)
        self.tmdbButton = QPushButton('Preferences...')
        self.tmdbButton.clicked.connect(self.preferences_requested.emit)

        where = QGroupBox('Where things go')
        form = QFormLayout(where)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)   # labels above the fields: a narrow drawer
        form.addRow('Work directory', self.workDir)
        form.addRow('Review queue', self.queueDir)
        repos = QGroupBox('Catalogue repositories (publish and commit)')
        form = QFormLayout(repos)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)   # labels above the fields: a narrow drawer
        form.addRow('XML repository', self.xmlRepo)
        form.addRow('XML folder', self.xmlDir)
        form.addRow('Images repository', self.imagesRepo)
        form.addRow('Images folder', self.imageDir)
        form.addRow('Image owner', self.imageOwner)
        form.addRow('Image repository', self.imageRepoName)
        form.addRow('', self.imageNote)
        options = QGroupBox('Designer and options')
        form = QFormLayout(options)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)   # labels above the fields: a narrow drawer
        form.addRow('Designer', self.designerCombo)
        form.addRow('TV shows', self.tvModeCombo)
        form.addRow('', self.keepMultichannel)
        form.addRow('Accept threshold', self.acceptThreshold)
        tmdb = QHBoxLayout()
        tmdb.addWidget(self.tmdbLabel, 1)
        tmdb.addWidget(self.tmdbButton)
        form.addRow('TMDB key', tmdb)
        inner = QWidget()
        column = QVBoxLayout(inner)
        column.addWidget(where)
        column.addWidget(repos)
        column.addWidget(options)
        column.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(inner)

        self.sourcesTab = SourcesTab(self._prefs, run_dialog=run_dialog)
        self.ignoreTab = IgnoreTab(self._prefs, run_dialog=run_dialog)
        self.tabs = QTabWidget()
        self.tabs.addTab(scroll, 'Locations')
        self.tabs.addTab(self.sourcesTab, 'Sources')
        self.tabs.addTab(self.ignoreTab, 'Ignore')
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('<b>Profile file</b>'))
        layout.addLayout(header)
        layout.addWidget(self.statusLabel)
        layout.addWidget(self.busyLabel)
        layout.addWidget(self.tabs, 1)

        self.workDir.committed.connect(lambda text: self.__set_directory('work_dir', self.workDir, text))
        self.queueDir.committed.connect(lambda text: self.__set_directory('queue_dir', self.queueDir, text))
        self.xmlRepo.committed.connect(lambda text: self.__set_repo('xml_repo', self.xmlRepo, text))
        self.imagesRepo.committed.connect(lambda text: self.__set_repo('images_repo', self.imagesRepo, text))
        self.xmlDir.committed.connect(lambda text: self.__set_folder('xml_dir', self.xmlDir, text))
        self.imageDir.committed.connect(lambda text: self.__set_folder('image_dir', self.imageDir, text))
        self.imageOwner.editingFinished.connect(
            lambda: self.__set_sync('image_owner', self.imageOwner.text().strip()))
        self.imageRepoName.editingFinished.connect(
            lambda: self.__set_sync('image_repo_name', self.imageRepoName.text().strip()))
        self.designerCombo.activated.connect(lambda _i: self.__set_run('designer', self.__chosen_designer()))
        self.tvModeCombo.activated.connect(lambda _i: self.__set_run('tv_mode', self.tvModeCombo.currentData()))
        self.keepMultichannel.clicked.connect(lambda checked: self.__set_run('keep_multichannel', bool(checked)))
        self.acceptThreshold.valueChanged.connect(self.__threshold_changed)
        self.sourcesTab.changed.connect(lambda sources: self.__edit(replace(self._profile, sources=tuple(sources))))
        self.sourcesTab.renamed.connect(self.ignoreTab.rename_source)
        self.ignoreTab.changed.connect(
            lambda rules, ignored: self.__edit(replace(self._profile, ignore=tuple(rules), ignored_titles=dict(ignored))))

    # --- loading ----------------------------------------------------------------------------------------------------

    @property
    def profile(self) -> Optional[Profile]:
        ''' The profile as edited (it may not be written yet: `flush()`). '''
        return self._profile

    @property
    def path(self) -> str:
        ''' The profile file being written, '' if none exists yet. '''
        return self._path

    @property
    def has_pending_edit(self) -> bool:
        return self._pending

    @property
    def error(self) -> str:
        ''' Why the last edit could not be written, '' if it could. '''
        return self._error

    def load(self, setup: WorkListSetup) -> None:
        '''
        Shows this setup's profile. An edit still waiting to be written is written first, and one that could not be is
        kept (the drawer is not overwritten under it).
        '''
        if self._pending and not self.flush():
            return
        self._setup = setup
        self._path = setup.path if setup.origin == 'file' else ''
        if setup.profile is None:
            self._profile = None
            self._show_unreadable(setup)
            return
        profile = setup.profile
        if not config_value(profile, 'run', 'designer') and setup.settings and setup.settings.designer:
            profile = with_config(profile, 'run', 'designer', setup.settings.designer)   # what a scan already uses
        self._profile = profile
        self._loading = True
        try:
            self.__fill(profile)
        finally:
            self._loading = False
        self.__show_path()
        self._error = ''
        self.statusLabel.setText('' if self._path else
                                 'These settings come from the Library Sync preferences. The first change you make here '
                                 'creates a profile file for them.')

    def _show_unreadable(self, setup: WorkListSetup) -> None:
        self.pathLabel.setText(setup.path)
        self.statusLabel.setText(f'This profile file could not be read: {setup.error}\nChoose another file, or fix this '
                                 f'one and reopen the window.')
        self.statusLabel.setStyleSheet(f'color: {warning_colour().name()}')
        self.tabs.setEnabled(False)

    def __fill(self, profile: Profile) -> None:
        self.tabs.setEnabled(True)
        self.statusLabel.setStyleSheet('')
        self.workDir.set_text(profile.work_dir)
        self.queueDir.set_text(profile.queue_dir)
        self.xmlRepo.set_text(profile.xml_repo)
        self.xmlDir.set_text(profile.xml_dir)
        self.imagesRepo.set_text(profile.images_repo)
        self.imageDir.set_text(profile.image_dir)
        self.workDir.show_check(check_directory(profile.work_dir) if profile.work_dir else None)
        self.queueDir.show_check(check_directory(profile.queue_dir) if profile.queue_dir else None)
        self.xmlRepo.show_check(check_repository(profile.xml_repo) if profile.xml_repo else None)
        self.imagesRepo.show_check(check_repository(profile.images_repo) if profile.images_repo else None)
        self.xmlDir.show_check(None)
        self.imageDir.show_check(None)
        self.imageOwner.setText(str(config_value(profile, 'sync', 'image_owner')))
        self.imageRepoName.setText(str(config_value(profile, 'sync', 'image_repo_name')))
        self.__refresh_image_note()
        self.__fill_designers(str(config_value(profile, 'run', 'designer')))
        self.tvModeCombo.setCurrentIndex(max(self.tvModeCombo.findData(config_value(profile, 'run', 'tv_mode', 'episode')), 0))
        self.keepMultichannel.setChecked(bool(config_value(profile, 'run', 'keep_multichannel', False)))
        self.acceptThreshold.setValue(float(self._prefs.get(WORKLIST_ACCEPT_THRESHOLD)))
        self.__refresh_tmdb()
        rows = list(self._rows_provider())
        self.sourcesTab.set_sources(profile.sources)
        self.ignoreTab.set_state(profile.ignore, profile.ignored_titles, rows, [s.name for s in profile.sources])

    def __fill_designers(self, wanted: str) -> None:
        from pipeline.designer.registry import registered_designers
        self.designerCombo.clear()
        names = registered_designers()
        self.designerCombo.addItems(names)
        if wanted and wanted not in names:
            self.designerCombo.addItem(f'{wanted} (not available)', wanted)   # the profile names one nobody registered
        index = self.designerCombo.findText(wanted) if wanted in names else self.designerCombo.findData(wanted)
        self.designerCombo.setCurrentIndex(max(index, 0))

    def refresh_designers(self) -> None:
        ''' The designer list again (the profile's own `designers:` are registered when it is read). '''
        if self._profile is not None:
            self.__fill_designers(str(config_value(self._profile, 'run', 'designer')))

    def __show_path(self) -> None:
        self.pathLabel.setText(self._path or 'None yet: created when you first change something')
        self.pathLabel.setToolTip('Comments in a hand-written profile file are not kept when this saves it; '
                                  'everything else in it is.')

    def __refresh_tmdb(self) -> None:
        key = (self._prefs.get(TMDB_API_KEY) or '').strip()
        self.tmdbLabel.setText('Set (kept in Preferences, never in the profile file).' if key else
                               'Not set: titles are designed without TMDB metadata. Set it in Preferences.')

    def __refresh_image_note(self) -> None:
        text = ('Only needed when the images repository is not on github.com under a plain git@github.com: or '
                'https://github.com/ URL (an SSH host alias, a mirror). Leave both empty otherwise: they are read from the '
                'repository\'s remote.')
        repo = self.imagesRepo.edit.text().strip()
        if repo and os.path.isdir(repo):
            parsed = remote_owner_and_name(repo)
            text += (f'\nThis repository\'s remote is github.com/{parsed[0]}/{parsed[1]}: nothing to set.' if parsed else
                     '\nThis repository\'s remote is not a plain github.com URL: set both, or images cannot be published.')
        self.imageNote.setText(text)

    def set_rows(self, rows: Sequence[TitleRow]) -> None:
        ''' The index's rows changed (a scan or a run ended): the live ignore count follows. '''
        if self._profile is not None:
            self.ignoreTab.set_rows(rows, [s.name for s in self._profile.sources])

    def set_busy(self, busy: bool) -> None:
        ''' A run or scan is going: nothing is edited until it ends. '''
        self.busyLabel.setVisible(busy)
        self.tabs.setEnabled(not busy and self._profile is not None)
        self.changeFileButton.setEnabled(not busy)

    def select_tab(self, name: str) -> None:
        self.tabs.setCurrentIndex({'locations': 0, 'sources': 1, 'ignore': 2}[name])

    # --- edits ------------------------------------------------------------------------------------------------------

    def __set_directory(self, name: str, row: _PathRow, text: str) -> None:
        if self._loading or self._profile is None or text == getattr(self._profile, name):
            return
        check = check_directory(text) if text else None
        row.show_check(check)
        if check is not None and not check.usable:
            return   # refused: the message says why, and the profile keeps its value
        if check is not None and check.level == LEVEL_INFO:
            try:
                os.makedirs(text, exist_ok=True)
            except OSError as error:
                row.show_check(PathCheck(LEVEL_ERROR, f'Could not create this folder: {error}'))
                return
            row.show_check(PathCheck(LEVEL_OK, 'Folder created'))
        self.__edit(replace(self._profile, **{name: text}))

    def __set_repo(self, name: str, row: _PathRow, text: str) -> None:
        if self._loading or self._profile is None or text == getattr(self._profile, name):
            return
        check = check_repository(text) if text else None
        row.show_check(check)
        if check is not None and not check.usable:
            return
        self.__edit(replace(self._profile, **{name: text}))
        if name == 'images_repo':
            self.__refresh_image_note()

    def __set_folder(self, name: str, row: _PathRow, text: str) -> None:
        if self._loading or self._profile is None or text == getattr(self._profile, name):
            return
        check = check_relative_dir(text)
        row.show_check(check if check.level == LEVEL_ERROR else None)
        if check.usable:
            self.__edit(replace(self._profile, **{name: text.strip().replace('\\', '/').strip('/')}))

    def __set_sync(self, key: str, value: str) -> None:
        if self._loading or self._profile is None or value == str(config_value(self._profile, 'sync', key)):
            return
        self.__edit(with_config(self._profile, 'sync', key, value))

    def __set_run(self, key: str, value) -> None:
        if self._loading or self._profile is None:
            return
        self.__edit(with_config(self._profile, 'run', key, value))

    def __threshold_changed(self, value: float) -> None:
        if not self._loading:
            self._prefs.set(WORKLIST_ACCEPT_THRESHOLD, round(float(value), 2))   # a preference: not part of the profile

    def __edit(self, profile: Profile) -> None:
        ''' An edit was made: validate the whole profile, and write it shortly (or say why it cannot be). '''
        if self._loading or self._profile is None:
            return
        error = self._problem(profile)
        if error:
            # refused: the edit is not applied (so later edits do not build on a profile that cannot be written) and the
            # lists are put back to what the profile holds; an earlier edit still waiting is still written
            self._error = error
            self.__show_error(error)
            self.__restore_lists()
            return
        self._profile = profile
        self._pending = True
        self._error = ''
        self.statusLabel.setStyleSheet('')
        self.statusLabel.setText('Saving...')
        self._timer.start()
        self.edited.emit()

    def __restore_lists(self) -> None:
        ''' Shows the sources and ignore rules of the profile as it stands (an edit to them was refused). '''
        profile = self._profile
        self._loading = True
        try:
            self.sourcesTab.set_sources(profile.sources)
            self.ignoreTab.set_state(profile.ignore, profile.ignored_titles, list(self._rows_provider()),
                                     [s.name for s in profile.sources])
        finally:
            self._loading = False

    def __chosen_designer(self) -> str:
        ''' The designer picked in the combo: its data if it has some (an unregistered name is listed as "name (not available)"). '''
        return str(self.designerCombo.currentData() or self.designerCombo.currentText())

    @staticmethod
    def _problem(profile: Profile) -> str:
        ''' '' if the profile would read back from what would be written, else why not. '''
        try:
            profile_from_config(profile.to_config())
        except ValueError as error:
            return str(error)
        return ''

    def __show_error(self, message: str) -> None:
        self.statusLabel.setStyleSheet(f'color: {warning_colour().name()}')
        self.statusLabel.setText(f'Not saved: {message}')

    # --- writing ----------------------------------------------------------------------------------------------------

    def flush(self) -> bool:
        '''
        Writes an edit that is waiting, now. :return: True if nothing is waiting or it was written; False if it could not be
        (the reason is shown, and the file is as it was).
        '''
        self._timer.stop()
        if not self._pending or self._profile is None:
            return True
        error = self._problem(self._profile)
        if error:
            self._error = error
            self.__show_error(error)
            return False
        path = self._path
        if not path:
            path = self._choose_path(self._default_path(), True)
            if not path:
                self._pending = False
                message = 'No profile file was chosen, so the change was not saved.'
                if self._setup is not None:
                    self._reset_to_setup()   # back to what the window has: the edit is dropped
                self._error = message
                self.__show_error(message)
                return False
        try:
            save_profile(self._profile, path)
        except (ValueError, OSError) as failure:
            self._error = f'{type(failure).__name__}: {failure}' if not isinstance(failure, ValueError) else str(failure)
            logger.warning('Could not write the profile %s: %s', path, self._error)
            self.__show_error(self._error)
            return False
        self._error, self._pending, self._path = '', False, path
        self._prefs.set(LIBRARY_PROFILE_PATH, path)
        self.__show_path()
        self.statusLabel.setStyleSheet('')
        self.statusLabel.setText(f'Saved {time.strftime("%H:%M:%S")}')
        self.saved.emit(path)
        return True

    def _reset_to_setup(self) -> None:
        self.load(self._setup)   # nothing is pending now, so this shows what the window has

    def change_profile_file(self) -> bool:
        '''
        Uses another profile file: an existing one is read (this drawer's unwritten edit is dropped), a new name gets the
        current settings. :return: False if cancelled or it could not be done.
        '''
        if self._pending and not self.flush():
            return False
        default = self._path or self._default_path()
        path = self._choose_path(default, False)
        if not path:
            return False
        if os.path.exists(path):
            self._prefs.set(LIBRARY_PROFILE_PATH, path)
            self.profile_file_changed.emit(path)
            return True
        if self._profile is None:
            return False
        try:
            save_profile(self._profile, path)
        except (ValueError, OSError) as failure:
            self._error = str(failure)
            self.__show_error(self._error)
            return False
        self._path = path
        self._prefs.set(LIBRARY_PROFILE_PATH, path)
        self.__show_path()
        self.statusLabel.setText(f'Saved {time.strftime("%H:%M:%S")}')
        self.profile_file_changed.emit(path)
        return True

    # --- what the window asks of it ---------------------------------------------------------------------------------

    def ignore_like(self, row: TitleRow) -> bool:
        ''' "Ignore titles like this...", from a work-list row. '''
        if self._profile is None:
            return False
        self.select_tab('ignore')
        return self.ignoreTab.ignore_like(row)

    def ignore_titles(self, ids: List[str], reason: str = '') -> None:
        ''' Ignores these titles one by one (the profile's `ignore_titles`). '''
        if self._profile is not None:
            self.ignoreTab.ignore_ids(ids, reason)
