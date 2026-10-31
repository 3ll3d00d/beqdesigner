'''
model/worklist.py: the work list reads the index again when another program (the pipeline service, the web app, the command
line) writes it -- on a timer while the window is shown and when the window becomes active -- rather than only after its own
scans, runs and decisions. `import ui.beq` first: see AGENTS.md gotcha 3.
'''
import ui.beq  # noqa: F401 (must come first)

import os
import sqlite3

import pytest
from qtpy.QtCore import QEvent, QSettings

import model.worklist
from model.preferences import DESIGNER_DEFAULT, DESIGNER_QUEUE_DIR, LIBRARY_FILESYSTEM_GLOBS, LIBRARY_WORK_DIR, \
    Preferences
from model.worklist import WorkListWindow
from pipeline.designer.registry import register_designer, unregister_designer
from worklist_fixture import make_index, title_row

NOW = 1_800_000_000.0
DESIGNER = 'test.worklist.poll'

@pytest.fixture(autouse=True)
def _designer():
    register_designer(DESIGNER, lambda request: None)
    yield
    unregister_designer(DESIGNER)

def _window(qtbot, tmp_path):
    prefs = Preferences(QSettings(str(tmp_path / 'settings.ini'), QSettings.Format.IniFormat))
    prefs.set(LIBRARY_WORK_DIR, str(tmp_path / 'work'))
    prefs.set(DESIGNER_QUEUE_DIR, str(tmp_path / 'queue'))
    prefs.set(LIBRARY_FILESYSTEM_GLOBS, ['/films/**/*.mkv'])
    prefs.set(DESIGNER_DEFAULT, DESIGNER)
    (tmp_path / 'work').mkdir()
    path = make_index(tmp_path / 'work', [title_row('t-heat', 'Heat', 'extract', state_since=NOW - 60, source='disk'),
                                          title_row('t-tenet', 'Tenet', 'design', state_since=NOW - 60, source='disk')],
                      [('disk', 'filesystem', NOW - 900, NOW - 900, '', 2)], last_scan_at=NOW - 900)
    window = WorkListWindow(None, prefs, auto_scan=False, clock=lambda: NOW)
    qtbot.addWidget(window)
    window.show()
    return window, path

def _another_program_writes(path, sql, *args):
    ''' Writes the index as another process would: its own connection, and the file's modification time moves on. '''
    before = os.stat(path).st_mtime_ns
    db = sqlite3.connect(path)
    with db:
        db.execute(sql, args)
    db.close()
    if os.stat(path).st_mtime_ns == before:   # a coarse filesystem clock: make the write visible as a real one would be
        os.utime(path, ns=(before + 1_000_000_000, before + 1_000_000_000))

def test_another_programs_write_is_shown_when_the_window_looks_again(qtbot, tmp_path):
    window, path = _window(qtbot, tmp_path)
    assert window.chip_counts()['Extract'] == 1

    _another_program_writes(path, "UPDATE titles SET needs = 'design' WHERE id = 't-heat'")

    assert window.refresh_if_index_changed() is True
    assert window.chip_counts()['Extract'] == 0 and window.chip_counts()['Design'] == 2

def test_an_unchanged_index_is_not_read_again(qtbot, tmp_path, monkeypatch):
    window, _ = _window(qtbot, tmp_path)
    reads = []
    monkeypatch.setattr(window, 'refresh_from_index', lambda: reads.append(1))

    assert window.refresh_if_index_changed() is False
    assert reads == []

def test_the_window_becoming_active_reads_a_changed_index(qtbot, tmp_path, monkeypatch):
    window, path = _window(qtbot, tmp_path)
    _another_program_writes(path, "UPDATE titles SET needs = 'review' WHERE id = 't-tenet'")
    monkeypatch.setattr(window, 'isActiveWindow', lambda: True)

    window.changeEvent(QEvent(QEvent.Type.ActivationChange))

    assert window.chip_counts()['Review'] == 1 and window.chip_counts()['Design'] == 0

def test_the_timer_looks_while_the_window_is_shown_and_stops_when_it_is_closed(qtbot, tmp_path):
    window, path = _window(qtbot, tmp_path)
    timer = window._index_poll_timer
    assert timer.isActive() and timer.interval() == model.worklist.INDEX_POLL_MS

    _another_program_writes(path, "UPDATE titles SET needs = 'review' WHERE id = 't-tenet'")
    timer.timeout.emit()
    assert window.chip_counts()['Review'] == 1

    window.close()
    assert not timer.isActive()
    _another_program_writes(path, "UPDATE titles SET needs = 'design' WHERE id = 't-tenet'")
    assert window.refresh_if_index_changed() is False   # a closed window does not open the index again
    assert not window.has_open_index

@pytest.mark.parametrize('state', ['_scanning', '_syncing', '_title_open', '_index_dirty'])
def test_it_does_not_read_while_something_else_will(qtbot, tmp_path, monkeypatch, state):
    ''' A scan, an index read, an open title page or unread decisions each read the index themselves when they end. '''
    window, path = _window(qtbot, tmp_path)
    _another_program_writes(path, "UPDATE titles SET needs = 'review' WHERE id = 't-tenet'")
    monkeypatch.setattr(window, state, True)

    assert window.refresh_if_index_changed() is False
    monkeypatch.setattr(window, state, False)
    assert window.refresh_if_index_changed() is True
