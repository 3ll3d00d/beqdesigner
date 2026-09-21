'''
Safety net for model/preferences.py's PreferencesDialog "Designers" page and DESIGNER_QUEUE_DIR default --
folded in from the old standalone model/designers.py's DesignersDialog (Tools > Designers) and the previously
unpersisted queue-directory fields on model/batch.py's BatchExtractDialog and the review dialog
(model/review.py's ReviewQueueDialog, deleted at chunk 27c; the queue directory is now remembered by model/worklist_review.py's window). pipeline.designer.http_binding's own wire format is already covered by
test_pipeline_designer_http_binding.py; this only exercises the Qt wiring + preference persistence +
registration on top.

Unlike the "X Axis Invalid" style checks elsewhere in accept(), a QMessageBox.exec() (a blocking modal, e.g. the
"Theme Change" restart-required alert) would hang a headless test -- monkeypatched to a no-op everywhere here.
'''
from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import QSettings
from qtpy.QtWidgets import QMessageBox

from model.preferences import (
    DESIGNER_DEFAULT, DESIGNER_HTTP_ENDPOINTS, DESIGNER_QUEUE_DIR, Preferences, PreferencesDialog,
    register_configured_designers, _REGISTERED_DESIGNER_PREFIX,
)
from pipeline.designer.registry import register_designer, registered_designers, unregister_designer


def _make_preferences(tmp_path):
    settings = QSettings(str(tmp_path / 'settings.ini'), QSettings.Format.IniFormat)
    return Preferences(settings)


@pytest.fixture(autouse=True)
def _no_blocking_dialogs(monkeypatch):
    ''' accept() can pop a restart-required alert (QMessageBox.exec) unrelated to what's under test here. '''
    monkeypatch.setattr(QMessageBox, 'exec', lambda self: None)


@pytest.fixture(autouse=True)
def _clean_registry():
    ''' Designer registration is a module-level global -- keep tests from leaking into each other. '''
    yield
    for name in list(registered_designers()):
        if name.startswith(_REGISTERED_DESIGNER_PREFIX):
            unregister_designer(name)


@pytest.fixture
def dialog(qtbot, tmp_path):
    prefs = _make_preferences(tmp_path)
    d = PreferencesDialog(prefs, str(tmp_path), MagicMock(), parent=None)
    qtbot.addWidget(d)
    return d, prefs


def test_starts_empty_with_no_configured_designers(dialog):
    d, _ = dialog
    assert d.designersTable.rowCount() == 0


def test_loads_existing_configured_designers(qtbot, tmp_path):
    prefs = _make_preferences(tmp_path)
    prefs.set(DESIGNER_HTTP_ENDPOINTS, [{'name': 'remote', 'url': 'http://example.invalid/design', 'headers': {}}])

    d = PreferencesDialog(prefs, str(tmp_path), MagicMock(), parent=None)
    qtbot.addWidget(d)

    assert d.designersTable.rowCount() == 1
    assert d.designersTable.item(0, 0).text() == 'remote'
    assert d.designersTable.item(0, 1).text() == 'http://example.invalid/design'


def test_add_and_accept_persists_and_registers(dialog):
    d, prefs = dialog
    d.add_designer_row()
    d.designersTable.item(0, 0).setText('remote')
    d.designersTable.item(0, 1).setText('http://example.invalid/design')

    d.accept()

    saved = prefs.get(DESIGNER_HTTP_ENDPOINTS)
    assert saved == [{'name': 'remote', 'url': 'http://example.invalid/design', 'headers': {}}]
    assert f'{_REGISTERED_DESIGNER_PREFIX}remote' in registered_designers()


def test_accept_with_valid_headers_json(dialog):
    d, prefs = dialog
    d.add_designer_row()
    d.designersTable.item(0, 0).setText('remote')
    d.designersTable.item(0, 1).setText('http://example.invalid/design')
    d.designersTable.item(0, 2).setText('{"Authorization": "Bearer xyz"}')

    d.accept()

    assert prefs.get(DESIGNER_HTTP_ENDPOINTS)[0]['headers'] == {'Authorization': 'Bearer xyz'}


def test_invalid_headers_json_leaves_designers_unsaved_but_still_closes(dialog, monkeypatch):
    d, prefs = dialog
    calls = []
    monkeypatch.setattr(QMessageBox, 'critical', lambda *a, **k: calls.append(a))
    d.add_designer_row()
    d.designersTable.item(0, 0).setText('remote')
    d.designersTable.item(0, 1).setText('http://example.invalid/design')
    d.designersTable.item(0, 2).setText('not json')

    d.accept()

    assert len(calls) == 1
    assert prefs.get(DESIGNER_HTTP_ENDPOINTS) == []
    assert d.result() == PreferencesDialog.DialogCode.Accepted


def test_duplicate_names_leaves_designers_unsaved(dialog, monkeypatch):
    d, prefs = dialog
    monkeypatch.setattr(QMessageBox, 'critical', lambda *a, **k: None)
    for i in range(2):
        d.add_designer_row()
        d.designersTable.item(i, 0).setText('remote')
        d.designersTable.item(i, 1).setText(f'http://example.invalid/{i}')

    d.accept()

    assert prefs.get(DESIGNER_HTTP_ENDPOINTS) == []


def test_remove_selected_row(dialog):
    d, _ = dialog
    d.add_designer_row()
    d.designersTable.item(0, 0).setText('remote')
    d.designersTable.item(0, 1).setText('http://example.invalid/design')
    d.designersTable.selectRow(0)

    d.remove_selected_designer_row()

    assert d.designersTable.rowCount() == 0


def test_register_configured_designers_replaces_stale_ones(qtbot, tmp_path):
    prefs = _make_preferences(tmp_path)
    prefs.set(DESIGNER_HTTP_ENDPOINTS, [{'name': 'old', 'url': 'http://example.invalid/1', 'headers': {}}])
    register_configured_designers(prefs)
    assert f'{_REGISTERED_DESIGNER_PREFIX}old' in registered_designers()

    prefs.set(DESIGNER_HTTP_ENDPOINTS, [{'name': 'new', 'url': 'http://example.invalid/2', 'headers': {}}])
    d = PreferencesDialog(prefs, str(tmp_path), MagicMock(), parent=None)
    qtbot.addWidget(d)
    d.accept()

    assert f'{_REGISTERED_DESIGNER_PREFIX}old' not in registered_designers()
    assert f'{_REGISTERED_DESIGNER_PREFIX}new' in registered_designers()


def test_design_queue_dir_field_starts_disabled_and_persists_on_accept(dialog, tmp_path):
    d, prefs = dialog
    assert d.designQueueDir.isEnabled() is False
    queue_dir = str(tmp_path)  # tmp_path itself always exists
    d.designQueueDir.setText(queue_dir)

    d.accept()

    assert prefs.get(DESIGNER_QUEUE_DIR) == queue_dir


def test_design_queue_dir_defaults_from_preference_on_open(qtbot, tmp_path):
    prefs = _make_preferences(tmp_path)
    prefs.set(DESIGNER_QUEUE_DIR, str(tmp_path))

    d = PreferencesDialog(prefs, str(tmp_path), MagicMock(), parent=None)
    qtbot.addWidget(d)

    assert d.designQueueDir.text() == str(tmp_path)


def test_design_queue_dir_does_not_default_from_a_deleted_directory(qtbot, tmp_path):
    prefs = _make_preferences(tmp_path)
    missing = str(tmp_path / 'does-not-exist')
    prefs.set(DESIGNER_QUEUE_DIR, missing)

    d = PreferencesDialog(prefs, str(tmp_path), MagicMock(), parent=None)
    qtbot.addWidget(d)

    assert d.designQueueDir.text() == ''


def test_default_designer_combo_lists_registered_designers_and_persists_the_choice(qtbot, tmp_path):
    register_designer('test.default_designer', lambda request: None)
    try:
        prefs = _make_preferences(tmp_path)
        d = PreferencesDialog(prefs, str(tmp_path), MagicMock(), parent=None)
        qtbot.addWidget(d)

        names = [d.defaultDesignerCombo.itemText(i) for i in range(d.defaultDesignerCombo.count())]
        assert 'test.default_designer' in names
        d.defaultDesignerCombo.setCurrentText('test.default_designer')

        d.accept()

        assert prefs.get(DESIGNER_DEFAULT) == 'test.default_designer'
    finally:
        unregister_designer('test.default_designer')


def test_default_designer_defaults_from_preference_on_open(qtbot, tmp_path):
    register_designer('test.default_designer2', lambda request: None)
    try:
        prefs = _make_preferences(tmp_path)
        prefs.set(DESIGNER_DEFAULT, 'test.default_designer2')

        d = PreferencesDialog(prefs, str(tmp_path), MagicMock(), parent=None)
        qtbot.addWidget(d)

        assert d.defaultDesignerCombo.currentText() == 'test.default_designer2'
    finally:
        unregister_designer('test.default_designer2')
