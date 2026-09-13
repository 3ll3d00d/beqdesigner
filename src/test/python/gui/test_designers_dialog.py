'''
Safety net for model/designers.py's DesignersDialog (design/http-designer-
binding-plan.md phase 3) -- constructs the real dialog and drives its
actual widgets. pipeline.designer.http_binding's own wire format is
already covered by test_pipeline_designer_http_binding.py; this only
exercises the Qt wiring + preference persistence + registration on top.
'''
import pytest
from qtpy.QtCore import QSettings

from model.designers import DesignersDialog, _REGISTERED_PREFIX, register_configured_designers
from model.preferences import DESIGNER_HTTP_ENDPOINTS, Preferences
from pipeline.designer.registry import registered_designers, unregister_designer


def _make_preferences(tmp_path):
    settings = QSettings(str(tmp_path / 'settings.ini'), QSettings.Format.IniFormat)
    return Preferences(settings)


@pytest.fixture(autouse=True)
def _clean_registry():
    ''' Designer registration is a module-level global -- keep tests from leaking into each other. '''
    yield
    for name in list(registered_designers()):
        if name.startswith(_REGISTERED_PREFIX):
            unregister_designer(name)


@pytest.fixture
def dialog(qtbot, tmp_path):
    prefs = _make_preferences(tmp_path)
    d = DesignersDialog(None, prefs)
    qtbot.addWidget(d)
    return d, prefs


def test_starts_empty_with_no_configured_designers(dialog):
    d, _ = dialog
    assert d.designersTable.rowCount() == 0


def test_loads_existing_configured_designers(qtbot, tmp_path):
    prefs = _make_preferences(tmp_path)
    prefs.set(DESIGNER_HTTP_ENDPOINTS, [{'name': 'remote', 'url': 'http://example.invalid/design', 'headers': {}}])

    d = DesignersDialog(None, prefs)
    qtbot.addWidget(d)

    assert d.designersTable.rowCount() == 1
    assert d.designersTable.item(0, 0).text() == 'remote'
    assert d.designersTable.item(0, 1).text() == 'http://example.invalid/design'


def test_add_and_save_persists_and_registers(dialog):
    d, prefs = dialog
    d._DesignersDialog__add_row()
    d.designersTable.item(0, 0).setText('remote')
    d.designersTable.item(0, 1).setText('http://example.invalid/design')

    d._DesignersDialog__save()

    saved = prefs.get(DESIGNER_HTTP_ENDPOINTS)
    assert saved == [{'name': 'remote', 'url': 'http://example.invalid/design', 'headers': {}}]
    assert f'{_REGISTERED_PREFIX}remote' in registered_designers()


def test_save_with_valid_headers_json(dialog):
    d, prefs = dialog
    d._DesignersDialog__add_row()
    d.designersTable.item(0, 0).setText('remote')
    d.designersTable.item(0, 1).setText('http://example.invalid/design')
    d.designersTable.item(0, 2).setText('{"Authorization": "Bearer xyz"}')

    d._DesignersDialog__save()

    assert prefs.get(DESIGNER_HTTP_ENDPOINTS)[0]['headers'] == {'Authorization': 'Bearer xyz'}


def test_invalid_headers_json_blocks_save(dialog, monkeypatch):
    d, prefs = dialog
    monkeypatch.setattr('model.designers.QMessageBox.critical', lambda *a, **k: None)
    d._DesignersDialog__add_row()
    d.designersTable.item(0, 0).setText('remote')
    d.designersTable.item(0, 1).setText('http://example.invalid/design')
    d.designersTable.item(0, 2).setText('not json')

    d._DesignersDialog__save()

    assert prefs.get(DESIGNER_HTTP_ENDPOINTS) == []
    assert f'{_REGISTERED_PREFIX}remote' not in registered_designers()


def test_duplicate_names_blocks_save(dialog, monkeypatch):
    d, prefs = dialog
    monkeypatch.setattr('model.designers.QMessageBox.critical', lambda *a, **k: None)
    for i in range(2):
        d._DesignersDialog__add_row()
        d.designersTable.item(i, 0).setText('remote')
        d.designersTable.item(i, 1).setText(f'http://example.invalid/{i}')

    d._DesignersDialog__save()

    assert prefs.get(DESIGNER_HTTP_ENDPOINTS) == []


def test_remove_selected_row(dialog):
    d, _ = dialog
    d._DesignersDialog__add_row()
    d.designersTable.item(0, 0).setText('remote')
    d.designersTable.item(0, 1).setText('http://example.invalid/design')
    d.designersTable.selectRow(0)

    d._DesignersDialog__remove_selected_row()

    assert d.designersTable.rowCount() == 0


def test_register_configured_designers_replaces_stale_ones(qtbot, tmp_path):
    prefs = _make_preferences(tmp_path)
    prefs.set(DESIGNER_HTTP_ENDPOINTS, [{'name': 'old', 'url': 'http://example.invalid/1', 'headers': {}}])
    register_configured_designers(prefs)
    assert f'{_REGISTERED_PREFIX}old' in registered_designers()

    prefs.set(DESIGNER_HTTP_ENDPOINTS, [{'name': 'new', 'url': 'http://example.invalid/2', 'headers': {}}])
    d = DesignersDialog(None, prefs)
    qtbot.addWidget(d)
    d._DesignersDialog__save()

    assert f'{_REGISTERED_PREFIX}old' not in registered_designers()
    assert f'{_REGISTERED_PREFIX}new' in registered_designers()
