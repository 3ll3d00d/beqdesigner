'''Safety-net coverage for the Library Sync dialog's real Qt wiring.'''
from qtpy.QtCore import QSettings
from qtpy.QtWidgets import QMessageBox

from model.library_sync import LibrarySyncDialog
from model.preferences import DESIGNER_DEFAULT, DESIGNER_QUEUE_DIR, JRIVER_MCWS_CONNECTIONS, LIBRARY_JRIVER_BROWSE_NODE, \
    LIBRARY_WORK_DIR, Preferences
from pipeline.designer.registry import register_designer, unregister_designer


def _preferences(tmp_path):
    return Preferences(QSettings(str(tmp_path / 'settings.ini'), QSettings.Format.IniFormat))


def test_library_sync_dialog_reuses_jriver_preferences_and_builds_a_run_config(qtbot, tmp_path):
    prefs = _preferences(tmp_path)
    prefs.set(JRIVER_MCWS_CONNECTIONS, {'media.local:52199': (('user', 'pass'), True)})
    prefs.set(LIBRARY_WORK_DIR, '/work')
    prefs.set(DESIGNER_QUEUE_DIR, '/queue')
    prefs.set(LIBRARY_JRIVER_BROWSE_NODE, 42)
    prefs.set(DESIGNER_DEFAULT, 'test.library')
    register_designer('test.library', lambda request: None)
    try:
        dialog = LibrarySyncDialog(None, prefs)
        qtbot.addWidget(dialog)

        source, config = dialog._LibrarySyncDialog__source_and_config()

        assert source.host == 'media.local'
        assert source.port == 52199
        assert source.browse_node_id == 42
        assert source.username == 'user'
        assert source.password == 'pass'
        assert source.ssl is True
        assert config.work_dir == '/work'
        assert config.queue_dir == '/queue'
        assert config.designer == 'test.library'
    finally:
        unregister_designer('test.library')


def test_sync_finished_does_not_count_a_refused_entry_as_published(qtbot, tmp_path, monkeypatch):
    warnings = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda parent, title, text: warnings.append(text))
    dialog = LibrarySyncDialog(None, _preferences(tmp_path))
    qtbot.addWidget(dialog)

    dialog._LibrarySyncDialog__sync_finished([{'id': 'ok'}, {'id': 'clash', 'error': 'project_conflict'}])

    assert dialog.statusLabel.text() == 'Published 1 accepted entries, 1 need attention'
    assert len(warnings) == 1
    assert 'clash: the mono and multichannel' in warnings[0]
    assert dialog.runButton.isEnabled()


def test_sync_finished_is_quiet_when_everything_published(qtbot, tmp_path, monkeypatch):
    warnings = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda parent, title, text: warnings.append(text))
    dialog = LibrarySyncDialog(None, _preferences(tmp_path))
    qtbot.addWidget(dialog)

    dialog._LibrarySyncDialog__sync_finished([{'id': 'ok'}])

    assert dialog.statusLabel.text() == 'Published 1 accepted entries'
    assert warnings == []


def test_sync_finished_says_how_many_entries_used_a_project_edit(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: None)
    dialog = LibrarySyncDialog(None, _preferences(tmp_path))
    qtbot.addWidget(dialog)

    dialog._LibrarySyncDialog__sync_finished([{'id': 'a', 'edited_project': 'mono'}, {'id': 'b'}])

    assert dialog.statusLabel.text() == 'Published 2 accepted entries (1 from your project edits)'
