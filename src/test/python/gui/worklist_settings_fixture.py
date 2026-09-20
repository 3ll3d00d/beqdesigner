'''
Shared setup for the settings drawer tests (chunk 26c): real temp-file Preferences, a real profile file, real git
repositories, real temp directories -- nothing a real Profile round trip can do is faked. Dialogs are not shown: the
window and drawer take a `run_dialog` callable, which these tests fill in and accept like a person would.
'''
import ui.beq  # noqa: F401 (must come first)

import subprocess
import time
from typing import Callable, Optional

import pytest
from qtpy.QtCore import QSettings
from qtpy.QtWidgets import QDialog

from model.preferences import DESIGNER_DEFAULT, DESIGNER_QUEUE_DIR, LIBRARY_FILESYSTEM_GLOBS, LIBRARY_PROFILE_PATH, \
    LIBRARY_WORK_DIR, Preferences
from model.worklist import WorkListWindow
from pipeline.designer.registry import register_designer, unregister_designer
from pipeline.library.profile import write_config_file
from worklist_fixture import make_index

NOW = 1_800_000_000.0
DESIGNER = 'test.settings'


@pytest.fixture(autouse=True)
def _designer():
    register_designer(DESIGNER, lambda request: None)
    yield
    unregister_designer(DESIGNER)
    unregister_designer('remote.one')   # registered from a profile's `designers:`


def make_prefs(tmp_path, profile: Optional[str] = None, configured: bool = False) -> Preferences:
    prefs = Preferences(QSettings(str(tmp_path / 'settings.ini'), QSettings.Format.IniFormat))
    if profile:
        prefs.set(LIBRARY_PROFILE_PATH, profile)
    if configured:   # the Library Sync preferences the work list falls back on until a profile file exists
        (tmp_path / 'work').mkdir(exist_ok=True)
        prefs.set(LIBRARY_WORK_DIR, str(tmp_path / 'work'))
        prefs.set(DESIGNER_QUEUE_DIR, str(tmp_path / 'queue'))
        prefs.set(LIBRARY_FILESYSTEM_GLOBS, ['/films/**/*.mkv'])
        prefs.set(DESIGNER_DEFAULT, DESIGNER)
    return prefs


def git_repo(path) -> str:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'init', '-q', str(path)], check=True)
    return str(path)


def profile_config(tmp_path, **extra) -> dict:
    ''' A complete profile: two sources, the directories, a repository, and sections the drawer does not manage. '''
    (tmp_path / 'work').mkdir(exist_ok=True)
    config = {
        'sources': [{'name': 'films', 'kind': 'filesystem', 'globs': ['/films/**/*.mkv']},
                    {'name': 'disk', 'kind': 'filesystem', 'globs': ['/mnt/extra']}],
        'run': {'work_dir': str(tmp_path / 'work'), 'queue_dir': str(tmp_path / 'queue'), 'designer': DESIGNER,
                'audio_types': ['DTS-HD MA'], 'coverage': 'complete_programme'},
        'sync': {'xml_repo': git_repo(tmp_path / 'xml'), 'commit_message': 'Add BEQ'},
        'designers': {'remote.one': {'url': 'http://localhost:1/design', 'timeout': 30}},
        'custom_section': {'keep': ['me', 1, True]},
    }
    for key, value in extra.items():
        config[key] = value
    return config


def write_profile(tmp_path, name: str = 'profile.yaml', **extra) -> str:
    path = str(tmp_path / name)
    write_config_file(path, profile_config(tmp_path, **extra))
    return path


def accepting(fill: Callable[[QDialog], None]) -> Callable[[QDialog], bool]:
    ''' A `run_dialog` that fills the dialog in and presses OK, as a person would; True if it was accepted. '''
    def run(dialog: QDialog) -> bool:
        fill(dialog)
        dialog.accept()
        return dialog.result() == QDialog.DialogCode.Accepted
    return run


class Dialogs:
    ''' The `run_dialog` a test hands to the window: set `.next` to what should happen to the next dialog. '''

    def __init__(self):
        self.next: Callable[[QDialog], bool] = lambda dialog: False
        self.seen = []

    def __call__(self, dialog: QDialog) -> bool:
        self.seen.append(dialog)
        return self.next(dialog)


def open_window(qtbot, tmp_path, prefs, dialogs: Optional[Dialogs] = None, rows=None, index_sources=(), generation=2,
                choose_path=None, debounce_ms: int = 5, **kwargs) -> WorkListWindow:
    if rows is not None:
        make_index(tmp_path / 'work', rows, index_sources, generation=generation, last_scan_at=NOW - 900)
    window = WorkListWindow(None, prefs, auto_scan=False, clock=lambda: NOW, run_dialog=dialogs,
                            choose_profile_path=choose_path, settings_debounce_ms=debounce_ms, **kwargs)
    qtbot.addWidget(window)
    window.show()
    return window


def commit(row, text: str) -> None:
    ''' Types into a path row and finishes editing (Enter, or leaving the field). '''
    row.edit.setText(text)
    row.edit.editingFinished.emit()


def commit_line(edit, text: str) -> None:
    edit.setText(text)
    edit.editingFinished.emit()


def wait_saved(qtbot, drawer) -> None:
    qtbot.waitUntil(lambda: not drawer.has_pending_edit, timeout=3000)
