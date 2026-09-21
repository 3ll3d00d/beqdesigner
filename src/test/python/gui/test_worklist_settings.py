'''
model/worklist_settings.py and the window around it (chunk 26c): the settings drawer edits a REAL profile file. Real
Preferences over a temp file, real profile files, real git repositories and directories; dialogs are driven through the
`run_dialog` hook. `import ui.beq` first: see AGENTS.md gotcha 3.
'''
import ui.beq  # noqa: F401 (must come first)

import os
import subprocess

import pytest
from qtpy.QtCore import QModelIndex, Qt

from model.jriver.connections import SavedConnection, save_connections
from model.library_sources import FilesystemSourcePage, JRiverSourcePage
from model.preferences import LIBRARY_PROFILE_PATH, TMDB_API_KEY, WORKLIST_ACCEPT_THRESHOLD, DEFAULT_PREFS
from pipeline.designer.registry import registered_designers, unregister_designer
from pipeline.library.bulk import DEFAULT_ACCEPT_THRESHOLD
from pipeline.library.pathmap import PathMapping
from pipeline.library.profile import load_profile, read_config_file
from pipeline.library.source import LibraryItem
from worklist_settings_fixture import DESIGNER, NOW, Dialogs, _designer, accepting, commit, commit_line, git_repo, \
    make_prefs, open_window, profile_config, wait_saved, write_profile  # noqa: F401 (the autouse fixture)


def _bytes(path) -> bytes:
    with open(path, 'rb') as f:
        return f.read()


# --- persistence: each setting, unknown sections, refusal, atomic write ------------------------------------------------------

def test_editing_each_setting_persists_to_the_profile_file_and_a_new_window_shows_the_same(qtbot, tmp_path):
    path = write_profile(tmp_path)
    prefs = make_prefs(tmp_path, path)
    window = open_window(qtbot, tmp_path, prefs)
    drawer = window.open_settings()
    xml2, images = git_repo(tmp_path / 'xml2'), git_repo(tmp_path / 'images')

    commit(drawer.workDir, str(tmp_path / 'work2'))         # does not exist: created
    commit(drawer.queueDir, str(tmp_path / 'queue2'))
    commit(drawer.xmlRepo, xml2)
    commit(drawer.xmlDir, 'beq/xml')
    commit(drawer.imagesRepo, images)
    commit(drawer.imageDir, 'img')
    commit_line(drawer.imageOwner, 'me')
    commit_line(drawer.imageRepoName, 'beq-images')
    assert 'remote.one' in registered_designers()           # the profile's own `designers:` are in the combo
    index = drawer.designerCombo.findText('remote.one')
    assert index >= 0
    drawer.designerCombo.setCurrentIndex(index)
    drawer.designerCombo.activated.emit(index)
    drawer.tvModeCombo.setCurrentIndex(drawer.tvModeCombo.findData('season'))
    drawer.tvModeCombo.activated.emit(drawer.tvModeCombo.currentIndex())
    drawer.keepMultichannel.click()
    assert drawer.flush()

    saved = load_profile(path)
    assert (saved.work_dir, saved.queue_dir) == (str(tmp_path / 'work2'), str(tmp_path / 'queue2'))
    assert (saved.xml_repo, saved.xml_dir, saved.images_repo, saved.image_dir) == (xml2, 'beq/xml', images, 'img')
    assert os.path.isdir(tmp_path / 'work2')                # a folder that can be created is created
    assert saved.config['sync']['image_owner'] == 'me' and saved.config['sync']['image_repo_name'] == 'beq-images'
    assert saved.config['run']['designer'] == 'remote.one'
    assert saved.config['run']['tv_mode'] == 'season' and saved.config['run']['keep_multichannel'] is True

    again = open_window(qtbot, tmp_path, prefs)              # nothing but the file: every widget reads back the same
    other = again.open_settings()
    assert [w.edit.text() for w in (other.workDir, other.queueDir, other.xmlRepo, other.xmlDir, other.imagesRepo,
                                    other.imageDir)] == \
           [str(tmp_path / 'work2'), str(tmp_path / 'queue2'), xml2, 'beq/xml', images, 'img']
    assert (other.imageOwner.text(), other.imageRepoName.text()) == ('me', 'beq-images')
    assert other.designerCombo.currentText() == 'remote.one' and other.tvModeCombo.currentData() == 'season'
    assert other.keepMultichannel.isChecked()
    assert [s.name for s in other.sourcesTab.sources] == ['films', 'disk']


def test_what_the_drawer_does_not_manage_survives_every_edit(qtbot, tmp_path):
    path = write_profile(tmp_path)
    before = read_config_file(path)
    window = open_window(qtbot, tmp_path, make_prefs(tmp_path, path))
    drawer = window.open_settings()

    commit(drawer.queueDir, str(tmp_path / 'other-queue'))
    drawer.keepMultichannel.click()                      # a `run:` key, written into the file's own config
    drawer.sourcesTab.move_source(1, 0)
    drawer.ignoreTab.ignore_ids(['x-1'], 'broken')
    assert drawer.flush()

    after = read_config_file(path)
    for kept in ('designers', 'custom_section'):
        assert after[kept] == before[kept]
    assert after['sync']['commit_message'] == 'Add BEQ'
    assert after['run']['audio_types'] == ['DTS-HD MA'] and after['run']['coverage'] == 'complete_programme'
    assert after['ignore_titles'] == {'x-1': 'broken'}
    assert [s['name'] for s in after['sources']] == ['disk', 'films']


def test_a_change_is_written_by_itself_shortly_after_the_edit(qtbot, tmp_path):
    path = write_profile(tmp_path)
    window = open_window(qtbot, tmp_path, make_prefs(tmp_path, path))
    drawer = window.open_settings()
    before = _bytes(path)

    drawer.keepMultichannel.click()
    assert drawer.has_pending_edit and _bytes(path) == before      # not yet
    wait_saved(qtbot, drawer)

    assert _bytes(path) != before and load_profile(path).config['run']['keep_multichannel'] is True
    assert drawer.statusLabel.text().startswith('Saved')


def test_input_that_cannot_be_used_is_refused_with_the_reason_and_nothing_is_written(qtbot, tmp_path):
    path = write_profile(tmp_path)
    window = open_window(qtbot, tmp_path, make_prefs(tmp_path, path))
    drawer = window.open_settings()
    before, profile = _bytes(path), drawer.profile
    a_file = tmp_path / 'a-file'
    a_file.write_text('x')
    plain_dir = tmp_path / 'not-a-repo'
    plain_dir.mkdir()

    commit(drawer.workDir, str(a_file))
    assert 'file, not a folder' in drawer.workDir.status.text()
    commit(drawer.queueDir, str(a_file / 'sub'))
    assert 'cannot be created' in drawer.queueDir.status.text()
    commit(drawer.xmlRepo, str(plain_dir))
    assert 'not a git repository' in drawer.xmlRepo.status.text()
    commit(drawer.imagesRepo, str(tmp_path / 'nowhere'))
    assert 'does not exist' in drawer.imagesRepo.status.text()
    commit(drawer.xmlDir, '../outside')
    assert "no '..'" in drawer.xmlDir.status.text()
    commit(drawer.imageDir, '/abs/olute')
    assert 'relative' in drawer.imageDir.status.text()

    assert not drawer.has_pending_edit and drawer.profile == profile
    assert drawer.flush() and _bytes(path) == before      # nothing was pending, nothing was written


def test_a_write_that_fails_leaves_the_old_file_and_no_temporary_file_and_works_when_it_can(qtbot, tmp_path, monkeypatch):
    path = write_profile(tmp_path)
    window = open_window(qtbot, tmp_path, make_prefs(tmp_path, path), debounce_ms=10 ** 6)
    drawer = window.open_settings()
    before = _bytes(path)
    real_replace = os.replace

    def broken(*args, **kwargs):
        raise OSError('disk on fire')

    monkeypatch.setattr(os, 'replace', broken)
    drawer.keepMultichannel.click()
    assert drawer.flush() is False

    assert 'disk on fire' in drawer.error and drawer.statusLabel.text().startswith('Not saved')
    assert _bytes(path) == before
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith('.tmp')] == []

    monkeypatch.setattr(os, 'replace', real_replace)
    assert drawer.flush() is True
    assert load_profile(path).config['run']['keep_multichannel'] is True


def test_a_profile_that_would_not_read_back_is_not_written(qtbot, tmp_path):
    ''' A file in the older `sources: {name: ...}` shape cannot hold a second source without losing the unused one. '''
    config = profile_config(tmp_path)
    config['sources'] = {'jriver': {'host': 'h', 'port': 1, 'browse_node_id': 5}, 'filesystem': {'globs': ['/a']}}
    config['run']['source'] = 'filesystem'
    path = str(tmp_path / 'old.yaml')
    from pipeline.library.profile import write_config_file
    write_config_file(path, config)
    dialogs = Dialogs()
    window = open_window(qtbot, tmp_path, make_prefs(tmp_path, path), dialogs)
    drawer = window.open_settings()
    before = _bytes(path)

    dialogs.next = accepting(lambda d: (d.nameEdit.setText('extra'), d.page.globsEdit.setPlainText('/b')))
    assert drawer.sourcesTab.add_source()

    assert 'jriver' in drawer.error and drawer.statusLabel.text().startswith('Not saved: ')   # said as soon as it is edited
    assert drawer.flush()                                                     # nothing is waiting: the edit was not applied
    assert _bytes(path) == before
    assert 'extra' not in [s.name for s in drawer.profile.sources]
    assert [s.name for s in drawer.sourcesTab.sources] == [s.name for s in drawer.profile.sources]   # the list took it back

    drawer.keepMultichannel.setChecked(True)                                  # so a later edit does not fail the same way
    drawer.keepMultichannel.clicked.emit(True)
    assert drawer.error == '' and drawer.flush()
    assert load_profile(path).config['run']['keep_multichannel'] is True


# --- sources: priority and the reused pages ------------------------------------------------------------------------------------

def test_dragging_a_source_up_changes_its_priority_in_the_file(qtbot, tmp_path):
    path = write_profile(tmp_path)
    window = open_window(qtbot, tmp_path, make_prefs(tmp_path, path))
    drawer = window.open_settings()
    tab = drawer.sourcesTab

    assert tab.sourceList.model().moveRow(QModelIndex(), 1, QModelIndex(), 0)   # what a drag does to the list's model
    assert drawer.flush()

    assert [s.name for s in load_profile(path).sources] == ['disk', 'films']
    assert [tab.sourceList.item(i).text().split()[1] for i in range(2)] == ['disk', 'films']   # renumbered too
    assert 'highest priority' in tab.helpLabel.text() and 'first source owns' in tab.helpLabel.text()


def test_up_and_down_buttons_reorder_too(qtbot, tmp_path):
    path = write_profile(tmp_path)
    window = open_window(qtbot, tmp_path, make_prefs(tmp_path, path))
    drawer = window.open_settings()
    tab = drawer.sourcesTab
    tab.select('disk')
    assert tab.upButton.isEnabled() and not tab.downButton.isEnabled()

    qtbot.mouseClick(tab.upButton, Qt.MouseButton.LeftButton)
    assert drawer.flush()
    assert [s.name for s in load_profile(path).sources] == ['disk', 'films']

    tab.select('disk')
    qtbot.mouseClick(tab.downButton, Qt.MouseButton.LeftButton)
    assert drawer.flush()
    assert [s.name for s in load_profile(path).sources] == ['films', 'disk']


def test_a_filesystem_source_is_added_edited_and_removed_with_the_library_sync_page(qtbot, tmp_path):
    path = write_profile(tmp_path)
    dialogs = Dialogs()
    window = open_window(qtbot, tmp_path, make_prefs(tmp_path, path), dialogs)
    drawer = window.open_settings()
    tab = drawer.sourcesTab

    def fill(dialog):
        assert isinstance(dialog.page, FilesystemSourcePage)          # the page Library Sync uses, not a copy
        assert dialog.nameEdit.text() == 'filesystem'                 # named after its kind until typed over
        dialog.nameEdit.setText('tv')
        dialog.page.globsEdit.setPlainText('/tv/**/*.mkv\n/more')

    dialogs.next = accepting(fill)
    assert tab.add_source()
    assert drawer.flush()
    assert [(s.name, s.settings) for s in load_profile(path).sources][-1] == ('tv', {'globs': ['/tv/**/*.mkv', '/more']})
    assert [s.name for s in load_profile(path).sources] == ['films', 'disk', 'tv']       # a new one ranks last

    tab.select('tv')
    dialogs.next = accepting(lambda d: d.page.globsEdit.setPlainText('/tv'))
    assert tab.edit_selected()
    assert drawer.flush()
    assert load_profile(path).source('tv').settings == {'globs': ['/tv']}

    tab.select('films')
    assert tab.remove_selected()
    assert drawer.flush()
    assert [s.name for s in load_profile(path).sources] == ['disk', 'tv']


def test_a_jriver_source_is_added_from_a_saved_server_and_keeps_what_the_profile_holds_when_edited(qtbot, tmp_path):
    path = write_profile(tmp_path)
    prefs = make_prefs(tmp_path, path)
    save_connections(prefs, [SavedConnection('media.local:52199', 'user', 'pass', False,
                                             path_mappings=(PathMapping('W:\\Films', '/mnt/films'),))])
    dialogs = Dialogs()
    window = open_window(qtbot, tmp_path, prefs, dialogs)
    drawer = window.open_settings()
    tab = drawer.sourcesTab

    def fill(dialog):
        dialog.kindCombo.setCurrentIndex(dialog.kindCombo.findData('jriver'))
        assert isinstance(dialog.page, JRiverSourcePage)
        assert dialog.nameEdit.text() == 'jriver'
        dialog.page.browseNodeSpin.setValue(1007)

    dialogs.next = accepting(fill)
    assert tab.add_source()
    assert drawer.flush()
    settings = load_profile(path).source('jriver').settings
    assert (settings['host'], settings['port'], settings['browse_node_id']) == ('media.local', 52199, 1007)
    assert (settings['username'], settings['password']) == ('user', 'pass')
    assert settings['path_mappings'] == [{'from': 'W:\\Films', 'to': '/mnt/films'}]

    # a hand-written extra (`timeout`) and the profile's own copy of the mappings survive an edit of the node
    from dataclasses import replace
    profile = load_profile(path)
    hand = replace(profile.source('jriver'), settings={**settings, 'timeout': 30, 'path_mappings': [{'from': 'X:\\', 'to': '/x'}]})
    from pipeline.library.profile import save_profile
    save_profile(replace(profile, sources=tuple(hand if s.name == 'jriver' else s for s in profile.sources)), path)
    window.reload()
    drawer = window.open_settings()
    drawer.sourcesTab.select('jriver')
    seen = {}

    def edit(dialog):
        seen['note_visible'] = dialog.page.useSavedButton.isVisibleTo(dialog)   # it says the copy differs from Preferences
        dialog.page.browseNodeSpin.setValue(2000)

    dialogs.next = accepting(edit)
    assert drawer.sourcesTab.edit_selected()
    assert drawer.flush()
    edited = load_profile(path).source('jriver').settings
    assert edited['browse_node_id'] == 2000 and edited['timeout'] == 30
    assert edited['path_mappings'] == [{'from': 'X:\\', 'to': '/x'}]
    assert seen['note_visible'] is True


def test_a_source_whose_server_is_not_in_preferences_can_still_be_edited(qtbot, tmp_path):
    config = profile_config(tmp_path)
    config['sources'].insert(0, {'name': 'nas', 'kind': 'jriver', 'host': 'nas.lan', 'port': 52199,
                                 'browse_node_id': 12, 'browse_path': 'Films > Movies'})
    path = str(tmp_path / 'p.yaml')
    from pipeline.library.profile import write_config_file
    write_config_file(path, config)
    dialogs = Dialogs()
    window = open_window(qtbot, tmp_path, make_prefs(tmp_path, path), dialogs)
    drawer = window.open_settings()
    drawer.sourcesTab.select('nas')

    def edit(dialog):
        assert 'not in Preferences' in dialog.page.serverCombo.currentText()
        assert dialog.page.browseNodeSpin.value() == 12 and dialog.page.nodePathLabel.text() == 'Films > Movies'
        dialog.page.browseNodeSpin.setValue(13)

    dialogs.next = accepting(edit)
    assert drawer.sourcesTab.edit_selected() and drawer.flush()
    settings = load_profile(path).source('nas').settings
    assert (settings['host'], settings['port'], settings['browse_node_id']) == ('nas.lan', 52199, 13)


def test_source_names_must_be_unique_and_present(qtbot, tmp_path):
    path = write_profile(tmp_path)
    dialogs = Dialogs()
    window = open_window(qtbot, tmp_path, make_prefs(tmp_path, path), dialogs)
    drawer = window.open_settings()
    before = _bytes(path)
    seen = {}

    def fill(dialog):
        dialog.page.globsEdit.setPlainText('/x')
        dialog.nameEdit.setText('films')                                   # taken
        seen['duplicate'] = (dialog.name_problem(), dialog.buttons.button(dialog.buttons.StandardButton.Ok).isEnabled())
        dialog.accept()                                                    # refused
        seen['spec'] = dialog.spec
        dialog.nameEdit.setText('  ')
        seen['empty'] = dialog.name_problem()
        dialog.reject()

    dialogs.next = lambda d: fill(d) or False
    assert not drawer.sourcesTab.add_source()

    assert 'already a source called "films"' in seen['duplicate'][0] and seen['duplicate'][1] is False
    assert seen['spec'] is None and 'Give the source a name' in seen['empty']
    assert [s.name for s in drawer.sourcesTab.sources] == ['films', 'disk']
    assert drawer.flush() and _bytes(path) == before

    drawer.sourcesTab.select('disk')
    dialogs.next = accepting(lambda d: d.nameEdit.setText('films'))       # editing may not take another source's name
    assert not drawer.sourcesTab.edit_selected()
    assert [s.name for s in drawer.sourcesTab.sources] == ['films', 'disk']


def test_an_invalid_source_setting_is_shown_in_the_dialog_and_refused(qtbot, tmp_path):
    path = write_profile(tmp_path)
    dialogs = Dialogs()
    window = open_window(qtbot, tmp_path, make_prefs(tmp_path, path), dialogs)
    drawer = window.open_settings()
    seen = {}

    def fill(dialog):
        dialog.nameEdit.setText('empty')
        dialog.page.globsEdit.setPlainText('')
        dialog.accept()
        seen['error'] = dialog.errorLabel.text()
        dialog.reject()

    dialogs.next = lambda d: fill(d) or False
    assert not drawer.sourcesTab.add_source()
    assert 'at least one folder' in seen['error']


def test_renaming_a_source_updates_the_ignore_rules_that_name_it(qtbot, tmp_path):
    path = write_profile(tmp_path, ignore=[{'source': 'disk', 'kind': 'tv'}, {'path': '/x'}])
    dialogs = Dialogs()
    window = open_window(qtbot, tmp_path, make_prefs(tmp_path, path), dialogs)
    drawer = window.open_settings()
    drawer.sourcesTab.select('disk')

    dialogs.next = accepting(lambda d: d.nameEdit.setText('extra'))
    assert drawer.sourcesTab.edit_selected() and drawer.flush()

    saved = load_profile(path)
    assert [s.name for s in saved.sources] == ['films', 'extra']
    assert [r.source for r in saved.ignore] == ['extra', None]


# --- the first save, and the fallback until then -------------------------------------------------------------------------------

def test_the_first_change_with_no_profile_file_asks_where_creates_it_and_remembers_the_path(qtbot, tmp_path):
    prefs = make_prefs(tmp_path, configured=True)                 # the bootstrap: no profile file yet
    asked = []
    target = tmp_path / 'cfg' / 'library-profile.yaml'

    def choose(default, overwrite_ok):
        asked.append((default, overwrite_ok))
        target.parent.mkdir(exist_ok=True)
        return str(target)

    window = open_window(qtbot, tmp_path, prefs, choose_path=choose)
    drawer = window.open_settings()
    assert window.setup.origin == 'preferences' and drawer.path == ''
    assert 'None yet' in drawer.pathLabel.text() and 'saved library preferences' in drawer.statusLabel.text()

    drawer.keepMultichannel.click()
    assert drawer.flush()

    assert len(asked) == 1 and os.path.basename(asked[0][0]) == 'library-profile.yaml' and asked[0][1] is True
    assert target.is_file() and prefs.get(LIBRARY_PROFILE_PATH) == str(target)
    saved = load_profile(str(target))
    assert [(s.name, s.settings) for s in saved.sources] == [('filesystem', {'globs': ['/films/**/*.mkv']})]
    assert saved.work_dir == str(tmp_path / 'work') and saved.queue_dir == str(tmp_path / 'queue')
    assert saved.config['run']['designer'] == DESIGNER and saved.config['run']['keep_multichannel'] is True
    assert window.setup.origin == 'file' and window.setup.path == str(target)     # the file is authoritative from now on

    prefs.set(LIBRARY_PROFILE_PATH, str(target))
    drawer.keepMultichannel.click()
    assert drawer.flush() and len(asked) == 1                                     # not asked again


def test_the_default_place_for_a_new_profile_is_the_apps_configuration_folder_not_the_work_directory(tmp_path):
    from model.worklist_settings import PROFILE_FILE_NAME, default_profile_path
    default = default_profile_path()
    assert os.path.basename(default) == PROFILE_FILE_NAME and str(tmp_path / 'work') not in default


def test_cancelling_the_where_dialog_saves_nothing_and_takes_the_edit_back(qtbot, tmp_path):
    prefs = make_prefs(tmp_path, configured=True)
    window = open_window(qtbot, tmp_path, prefs, choose_path=lambda default, overwrite_ok: '')
    drawer = window.open_settings()

    drawer.keepMultichannel.click()
    assert drawer.flush() is False

    assert prefs.get(LIBRARY_PROFILE_PATH) == '' and 'No profile file was chosen' in drawer.statusLabel.text()
    assert not drawer.keepMultichannel.isChecked()                                # shows what the window has again
    assert not any(p.suffix in ('.yaml', '.json') for p in tmp_path.iterdir())


def test_the_bootstrap_stays_the_fallback_and_a_file_is_authoritative_once_it_exists(qtbot, tmp_path):
    prefs = make_prefs(tmp_path, configured=True)
    window = open_window(qtbot, tmp_path, prefs)
    assert window.setup.origin == 'preferences' and [s.name for s in window.setup.profile.sources] == ['filesystem']

    other = write_profile(tmp_path, name='mine.yaml')
    prefs.set(LIBRARY_PROFILE_PATH, other)
    window.reload()

    assert window.setup.origin == 'file'
    assert [s.name for s in window.setup.profile.sources] == ['films', 'disk']     # not the preferences' single source


# --- the banner, designers, staleness --------------------------------------------------------------------------------------

def test_the_incomplete_setup_banner_lists_what_is_missing_opens_the_drawer_and_goes_when_it_is_complete(qtbot, tmp_path):
    prefs = make_prefs(tmp_path)                                    # nothing set up
    target = tmp_path / 'profile.yaml'
    dialogs = Dialogs()
    window = open_window(qtbot, tmp_path, prefs, dialogs, choose_path=lambda default, overwrite_ok: str(target))

    assert window.setupBanner.isVisibleTo(window)
    text = window.setupBannerLabel.text()
    assert 'incomplete' in text and 'No library source' in text and 'No work directory' in text
    assert not window.settings_dock.isVisible()

    qtbot.mouseClick(window.setupBannerButton, Qt.MouseButton.LeftButton)
    assert window.settings_dock.isVisible()

    drawer = window.drawer
    dialogs.next = accepting(lambda d: d.page.globsEdit.setPlainText('/films'))
    assert drawer.sourcesTab.add_source()
    (tmp_path / 'work').mkdir()
    commit(drawer.workDir, str(tmp_path / 'work'))
    commit(drawer.queueDir, str(tmp_path / 'queue'))
    assert window.setupBanner.isVisibleTo(window)                   # not written yet: still incomplete
    drawer.designerCombo.setCurrentIndex(drawer.designerCombo.findText(DESIGNER))
    drawer.designerCombo.activated.emit(drawer.designerCombo.currentIndex())
    assert drawer.flush()

    assert not window.setupBanner.isVisibleTo(window) and window.setup.ready
    assert target.is_file()


def test_the_settings_button_and_the_empty_state_button_open_the_drawer(qtbot, tmp_path):
    window = open_window(qtbot, tmp_path, make_prefs(tmp_path))
    with qtbot.waitSignal(window.settings_requested, timeout=1000):
        qtbot.mouseClick(window.settingsButton, Qt.MouseButton.LeftButton)
    assert window.settings_dock.isVisible()
    window.settings_dock.hide()

    qtbot.mouseClick(window.openSettingsButton, Qt.MouseButton.LeftButton)
    assert window.settings_dock.isVisible()


def test_the_drawer_docks_beside_the_list_when_the_screen_has_room_and_floats_when_it_has_not(qtbot, tmp_path):
    from model.worklist import DRAWER_WIDTH
    window = open_window(qtbot, tmp_path, make_prefs(tmp_path))
    window.resize(1000, 700)
    window._available_width = lambda: 1920
    window.open_settings()
    assert not window.settings_dock.isFloating() and window.width() >= 1000 + DRAWER_WIDTH - 5

    small = open_window(qtbot, tmp_path, make_prefs(tmp_path))
    small.resize(1000, 700)
    small._available_width = lambda: 1024
    small.open_settings()
    assert small.settings_dock.isFloating() and small.width() == 1000     # the list keeps its size


def test_a_profiles_own_designers_are_registered_when_it_is_read(qtbot, tmp_path):
    config = profile_config(tmp_path)
    config['run']['designer'] = 'remote.one'
    path = str(tmp_path / 'p.yaml')
    from pipeline.library.profile import write_config_file
    write_config_file(path, config)
    unregister_designer('remote.one')

    window = open_window(qtbot, tmp_path, make_prefs(tmp_path, path))

    assert 'remote.one' in registered_designers()
    assert window.setup.ready and window.setup.problems == ()          # the designer it names is no longer "not available"
    assert window.drawer.designerCombo.currentText() == 'remote.one'


def test_a_designer_the_profile_names_but_nobody_registered_is_a_setup_problem(qtbot, tmp_path):
    config = profile_config(tmp_path)
    config['run']['designer'] = 'nobody.registered'
    path = str(tmp_path / 'p.yaml')
    from pipeline.library.profile import write_config_file
    write_config_file(path, config)

    window = open_window(qtbot, tmp_path, make_prefs(tmp_path, path))

    assert any("'nobody.registered' is not available" in p for p in window.setup.problems)
    assert window.setupBanner.isVisibleTo(window)
    assert 'nobody.registered (not available)' in [window.drawer.designerCombo.itemText(i)
                                                   for i in range(window.drawer.designerCombo.count())]


def test_a_designers_section_that_cannot_be_used_is_reported_not_raised(qtbot, tmp_path):
    config = profile_config(tmp_path)
    config['designers'] = {'broken': {'timeout': 3}}
    path = str(tmp_path / 'p.yaml')
    from pipeline.library.profile import write_config_file
    write_config_file(path, config)

    window = open_window(qtbot, tmp_path, make_prefs(tmp_path, path))

    assert any('designers' in p and 'needs a url' in p for p in window.setup.problems)


def test_a_change_that_alters_what_a_scan_says_offers_a_rescan_and_a_scan_clears_it(qtbot, tmp_path):
    path = write_profile(tmp_path)
    items = [LibraryItem(id='fs-1', source_path='/films/Kids/a.mkv', display_name='a', title='Alpha', year='2001'),
             LibraryItem(id='fs-2', source_path='/films/Crime/b.mkv', display_name='b', title='Beta', year='1999')]

    class Source:
        def list_items(self, **_):
            return list(items)

    (tmp_path / 'queue').mkdir(exist_ok=True)
    window = open_window(qtbot, tmp_path, make_prefs(tmp_path, path), sources={'films': Source(), 'disk': Source()})
    with qtbot.waitSignal(window.scan_finished, timeout=15000):
        window.rescan()
    assert not window.settings_stale and not window.staleBanner.isVisibleTo(window)
    drawer = window.open_settings()

    drawer.acceptThreshold.setValue(0.8)                       # a preference, not a discovery setting
    assert drawer.flush() and not window.settings_stale
    drawer.sourcesTab.move_source(1, 0)                        # the priority of the sources decides which owns a title
    assert drawer.flush()
    assert window.settings_stale and window.staleBanner.isVisibleTo(window)
    assert 'out of date' in window.staleBannerLabel.text()

    with qtbot.waitSignal(window.scan_finished, timeout=15000):
        qtbot.mouseClick(window.staleBannerButton, Qt.MouseButton.LeftButton)
    assert not window.settings_stale and not window.staleBanner.isVisibleTo(window)


def test_the_accept_threshold_is_a_preference_with_the_pipelines_default_and_is_not_in_the_profile_file(qtbot, tmp_path):
    assert DEFAULT_PREFS[WORKLIST_ACCEPT_THRESHOLD] == DEFAULT_ACCEPT_THRESHOLD == 0.90
    path = write_profile(tmp_path)
    prefs = make_prefs(tmp_path, path)
    window = open_window(qtbot, tmp_path, prefs)
    drawer = window.open_settings()
    before = _bytes(path)
    assert drawer.acceptThreshold.value() == 0.90

    drawer.acceptThreshold.setValue(0.75)

    assert prefs.get(WORKLIST_ACCEPT_THRESHOLD) == 0.75
    assert drawer.flush() and _bytes(path) == before and not drawer.has_pending_edit


def test_the_tmdb_key_stays_in_preferences_and_the_drawer_says_whether_it_is_set(qtbot, tmp_path):
    path = write_profile(tmp_path)
    prefs = make_prefs(tmp_path, path)
    prefs.set(TMDB_API_KEY, '')
    window = open_window(qtbot, tmp_path, prefs)
    drawer = window.open_settings()
    assert 'Not set' in drawer.tmdbLabel.text()
    with qtbot.waitSignal(window.preferences_requested, timeout=1000):
        qtbot.mouseClick(drawer.tmdbButton, Qt.MouseButton.LeftButton)

    prefs.set(TMDB_API_KEY, 'secret-key-123')
    window.reload()
    assert drawer.tmdbLabel.text().startswith('Set')
    drawer.keepMultichannel.click()
    assert drawer.flush()
    text = _bytes(path).decode()
    assert 'secret-key-123' not in text and 'tmdb_api_key' not in text and 'api_key' not in text


def test_the_image_owner_note_says_whether_the_remote_needs_it(qtbot, tmp_path):
    images = git_repo(tmp_path / 'images')
    subprocess.run(['git', '-C', images, 'remote', 'add', 'origin', 'ssh://gitea.lan/me/images.git'], check=True)
    path = write_profile(tmp_path, sync={'xml_repo': git_repo(tmp_path / 'xml'), 'images_repo': images})
    window = open_window(qtbot, tmp_path, make_prefs(tmp_path, path))
    drawer = window.open_settings()
    assert 'not a plain github.com URL' in drawer.imageNote.text()

    subprocess.run(['git', '-C', images, 'remote', 'set-url', 'origin', 'git@github.com:me/images.git'], check=True)
    drawer.imagesRepo.edit.setText(images + '/')
    commit(drawer.imagesRepo, images)
    window.reload()
    assert 'github.com/me/images' in drawer.imageNote.text()


def test_the_drawer_waits_while_a_run_or_scan_is_in_progress(qtbot, tmp_path):
    path = write_profile(tmp_path)
    window = open_window(qtbot, tmp_path, make_prefs(tmp_path, path))
    drawer = window.open_settings()
    assert drawer.tabs.isEnabled() and not drawer.busyLabel.isVisibleTo(drawer)

    window._set_scanning(True)
    assert not drawer.tabs.isEnabled() and drawer.busyLabel.isVisibleTo(drawer)
    assert not drawer.changeFileButton.isEnabled()

    window._set_scanning(False)
    assert drawer.tabs.isEnabled() and not drawer.busyLabel.isVisibleTo(drawer)


def test_an_edit_still_waiting_is_written_before_a_scan_reads_the_settings(qtbot, tmp_path):
    path = write_profile(tmp_path)
    (tmp_path / 'queue').mkdir(exist_ok=True)
    window = open_window(qtbot, tmp_path, make_prefs(tmp_path, path), debounce_ms=10 ** 6, sources={})
    drawer = window.open_settings()

    drawer.keepMultichannel.click()
    assert drawer.has_pending_edit and 'keep_multichannel' not in load_profile(path).config.get('run', {})

    window.rescan()

    assert not drawer.has_pending_edit and load_profile(path).config['run']['keep_multichannel'] is True
    assert window.setup.settings.keep_multichannel is True         # and the scan used it


# --- a profile file that cannot be read, and changing file ------------------------------------------------------------------

def test_a_profile_file_that_cannot_be_read_disables_the_editor_and_another_file_can_be_chosen(qtbot, tmp_path):
    bad = tmp_path / 'bad.yaml'
    bad.write_text('sources: [oops')
    good = write_profile(tmp_path, name='good.yaml')
    prefs = make_prefs(tmp_path, str(bad))
    window = open_window(qtbot, tmp_path, prefs, choose_path=lambda default, overwrite_ok: good)
    drawer = window.open_settings()

    assert not drawer.tabs.isEnabled() and 'could not be read' in drawer.statusLabel.text()
    assert window.setupBanner.isVisibleTo(window) and 'could not be read' in window.setupBannerLabel.text()
    assert drawer.profile is None

    assert drawer.change_profile_file()

    assert prefs.get(LIBRARY_PROFILE_PATH) == good and window.setup.profile is not None
    assert drawer.tabs.isEnabled() and drawer.path == good
    assert not window.setupBanner.isVisibleTo(window)


def test_a_new_profile_file_name_starts_a_file_from_the_current_settings(qtbot, tmp_path):
    path = write_profile(tmp_path)
    prefs = make_prefs(tmp_path, path)
    fresh = str(tmp_path / 'fresh.yaml')
    window = open_window(qtbot, tmp_path, prefs, choose_path=lambda default, overwrite_ok: fresh)
    drawer = window.open_settings()

    assert drawer.change_profile_file()

    assert prefs.get(LIBRARY_PROFILE_PATH) == fresh and os.path.isfile(fresh)
    assert load_profile(fresh).sources == load_profile(path).sources
    assert read_config_file(fresh)['custom_section'] == read_config_file(path)['custom_section']


def test_a_profile_sources_own_mappings_can_be_replaced_by_those_in_preferences(qtbot, tmp_path):
    config = profile_config(tmp_path)
    config['sources'].insert(0, {'name': 'nas', 'kind': 'jriver', 'host': 'media.local', 'port': 52199,
                                 'browse_node_id': 5, 'path_mappings': [{'from': 'X:\\', 'to': '/x'}]})
    path = str(tmp_path / 'p.yaml')
    from pipeline.library.profile import write_config_file
    write_config_file(path, config)
    prefs = make_prefs(tmp_path, path)
    save_connections(prefs, [SavedConnection('media.local:52199', path_mappings=(PathMapping('W:\\Films', '/mnt/films'),))])
    dialogs = Dialogs()
    window = open_window(qtbot, tmp_path, prefs, dialogs)
    drawer = window.open_settings()
    drawer.sourcesTab.select('nas')
    seen = {}

    def edit(dialog):
        seen['note'] = dialog.page.mappingsNote.isVisibleTo(dialog)
        dialog.page.useSavedButton.click()
        seen['gone'] = dialog.page.useSavedButton.isVisibleTo(dialog)

    dialogs.next = accepting(edit)
    assert drawer.sourcesTab.edit_selected() and drawer.flush()

    assert seen == {'note': True, 'gone': False}
    assert load_profile(path).source('nas').settings['path_mappings'] == [{'from': 'W:\\Films', 'to': '/mnt/films'}]


def test_choosing_the_not_available_entry_keeps_the_name_not_the_label(qtbot, tmp_path):
    config = profile_config(tmp_path)
    config['run']['designer'] = 'nobody.registered'
    path = str(tmp_path / 'p.yaml')
    from pipeline.library.profile import write_config_file
    write_config_file(path, config)
    window = open_window(qtbot, tmp_path, make_prefs(tmp_path, path))
    drawer = window.open_settings()

    index = drawer.designerCombo.findData('nobody.registered')
    assert index >= 0
    drawer.designerCombo.setCurrentIndex(index)
    drawer.designerCombo.activated.emit(index)
    drawer.tvModeCombo.setCurrentIndex(drawer.tvModeCombo.findData('season'))   # so the profile differs and is written
    drawer.tvModeCombo.activated.emit(drawer.tvModeCombo.currentIndex())
    assert drawer.flush()

    assert load_profile(path).config['run']['designer'] == 'nobody.registered'
