'''Safety-net coverage for the Library Sync dialog's real Qt wiring.'''
import pytest
from qtpy.QtCore import QSettings
from qtpy.QtWidgets import QMessageBox

from model.library_sources import FilesystemSourcePage, JRiverSourcePage, LibrarySourceKind, SourcePage, \
    register_source_kind, unregister_source_kind
from model.library_sync import LibrarySyncDialog
from model.preferences import DESIGNER_DEFAULT, DESIGNER_QUEUE_DIR, JRIVER_MCWS_CONNECTIONS, LIBRARY_FILESYSTEM_GLOBS, \
    LIBRARY_JRIVER_BROWSE_NODE, LIBRARY_JRIVER_CONNECTION, LIBRARY_SOURCE_DEFAULT, LIBRARY_WORK_DIR, Preferences
from pipeline.designer.registry import register_designer, unregister_designer
from pipeline.library.filesystem import FilesystemLibrarySource
from pipeline.library.jriver import JRiverLibrarySource


def _preferences(tmp_path):
    return Preferences(QSettings(str(tmp_path / 'settings.ini'), QSettings.Format.IniFormat))


def test_library_sync_dialog_builds_a_jriver_source_from_the_shared_server_list(qtbot, tmp_path):
    prefs = _preferences(tmp_path)
    prefs.set(JRIVER_MCWS_CONNECTIONS, {'other.local:1': (None, False), 'media.local:52199': (('user', 'pass'), True)})
    prefs.set(LIBRARY_SOURCE_DEFAULT, 'jriver')
    prefs.set(LIBRARY_JRIVER_CONNECTION, 'media.local:52199')
    prefs.set(LIBRARY_WORK_DIR, '/work')
    prefs.set(DESIGNER_QUEUE_DIR, '/queue')
    prefs.set(LIBRARY_JRIVER_BROWSE_NODE, 42)
    prefs.set(DESIGNER_DEFAULT, 'test.library')
    register_designer('test.library', lambda request: None)
    try:
        dialog = LibrarySyncDialog(None, prefs)
        qtbot.addWidget(dialog)

        source, config = dialog._LibrarySyncDialog__source_and_config()

        assert dialog.sourceCombo.currentData() == 'jriver'
        assert isinstance(source, JRiverLibrarySource)
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


def _dialog(qtbot, tmp_path, **prefs_to_set):
    prefs = _preferences(tmp_path)
    prefs.set(LIBRARY_WORK_DIR, '/work')
    prefs.set(DESIGNER_QUEUE_DIR, '/queue')
    prefs.set(DESIGNER_DEFAULT, 'test.library')
    for key, value in prefs_to_set.items():
        prefs.set(key, value)
    register_designer('test.library', lambda request: None)
    dialog = LibrarySyncDialog(None, prefs)
    qtbot.addWidget(dialog)
    return dialog, prefs


@pytest.fixture(autouse=True)
def _designer_cleanup():
    yield
    unregister_designer('test.library')


def test_the_picker_offers_every_registered_kind_and_starts_on_the_filesystem(qtbot, tmp_path):
    dialog, _ = _dialog(qtbot, tmp_path)

    assert [dialog.sourceCombo.itemData(i) for i in range(dialog.sourceCombo.count())] == ['filesystem', 'jriver']
    assert dialog.sourceCombo.currentData() == 'filesystem'
    assert isinstance(dialog.sourceStack.currentWidget(), FilesystemSourcePage)


def test_choosing_a_kind_shows_its_page(qtbot, tmp_path):
    dialog, _ = _dialog(qtbot, tmp_path)

    dialog.sourceCombo.setCurrentIndex(dialog.sourceCombo.findData('jriver'))

    assert isinstance(dialog.sourceStack.currentWidget(), JRiverSourcePage)


def test_a_filesystem_source_is_built_from_the_globs_entered(qtbot, tmp_path):
    dialog, _ = _dialog(qtbot, tmp_path)
    page = dialog.sourceStack.currentWidget()
    page.globsEdit.setPlainText(f'{tmp_path}\n\n  {tmp_path}/**/*.mkv  \n')

    source, _ = dialog._LibrarySyncDialog__source_and_config()

    assert isinstance(source, FilesystemLibrarySource)
    assert source.globs == [str(tmp_path), f'{tmp_path}/**/*.mkv']


def test_an_unusable_source_is_reported_not_run(qtbot, tmp_path):
    dialog, _ = _dialog(qtbot, tmp_path)
    with pytest.raises(ValueError, match='at least one folder or glob'):
        dialog._LibrarySyncDialog__source_and_config()

    dialog.sourceCombo.setCurrentIndex(dialog.sourceCombo.findData('jriver'))  # no servers saved
    with pytest.raises(ValueError, match='Preferences > JRiver'):
        dialog._LibrarySyncDialog__source_and_config()


def test_the_chosen_kind_and_its_settings_persist(qtbot, tmp_path):
    dialog, prefs = _dialog(qtbot, tmp_path, **{LIBRARY_JRIVER_BROWSE_NODE: -1})
    dialog.sourceStack.widget(0).globsEdit.setPlainText('/films')
    dialog.sourceCombo.setCurrentIndex(dialog.sourceCombo.findData('jriver'))
    dialog.sourceStack.widget(1).browseNodeSpin.setValue(7)

    dialog._LibrarySyncDialog__persist_preferences()

    assert prefs.get(LIBRARY_SOURCE_DEFAULT) == 'jriver'
    assert prefs.get(LIBRARY_FILESYSTEM_GLOBS) == ['/films']
    assert prefs.get(LIBRARY_JRIVER_BROWSE_NODE) == 7
    reopened = LibrarySyncDialog(None, prefs)
    qtbot.addWidget(reopened)
    assert reopened.sourceCombo.currentData() == 'jriver'
    assert reopened.sourceStack.widget(0).globs() == ['/films']
    assert reopened.sourceStack.widget(1).browseNodeSpin.value() == 7


def test_a_newly_registered_kind_appears_without_editing_the_dialog(qtbot, tmp_path):
    class _PlexPage(SourcePage):
        def load(self, prefs): pass
        def save(self, prefs): pass
        def build_source(self): return 'a plex source'

    register_source_kind(LibrarySourceKind('plex', 'Plex', _PlexPage))
    try:
        dialog, _ = _dialog(qtbot, tmp_path)
        dialog.sourceCombo.setCurrentIndex(dialog.sourceCombo.findData('plex'))

        source, _ = dialog._LibrarySyncDialog__source_and_config()

        assert dialog.sourceCombo.itemText(dialog.sourceCombo.count() - 1) == 'Plex'
        assert source == 'a plex source'
    finally:
        unregister_source_kind('plex')


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


def test_the_tv_mode_defaults_to_per_episode_persists_and_reaches_the_run_config(qtbot, tmp_path):
    from model.preferences import LIBRARY_TV_MODE
    dialog, prefs = _dialog(qtbot, tmp_path, **{LIBRARY_SOURCE_DEFAULT: 'filesystem'})
    dialog.sourceStack.widget(0).globsEdit.setPlainText(str(tmp_path))
    assert dialog.tvModeCombo.currentData() == 'episode'
    assert dialog._LibrarySyncDialog__source_and_config()[1].tv_mode == 'episode'

    dialog.tvModeCombo.setCurrentIndex(dialog.tvModeCombo.findData('season'))
    assert dialog._LibrarySyncDialog__source_and_config()[1].tv_mode == 'season'
    dialog._LibrarySyncDialog__persist_preferences()

    assert prefs.get(LIBRARY_TV_MODE) == 'season'
    reopened = LibrarySyncDialog(None, prefs)
    qtbot.addWidget(reopened)
    assert reopened.tvModeCombo.currentData() == 'season'


def test_the_run_summary_counts_seasons_that_were_joined(qtbot, tmp_path):
    from pipeline.library.run import LibraryRunReport
    dialog, _ = _dialog(qtbot, tmp_path)

    dialog._LibrarySyncDialog__run_finished(LibraryRunReport(designed=['a'], seasons={'show-s01': ['e1', 'e2']}))

    assert dialog.statusLabel.text() == 'Designed 1, cached 0, failed 0, 1 season(s) joined'
