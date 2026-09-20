'''
The ignore half of the settings drawer (chunk 26c): rules with a live "would ignore N titles" count over the index's rows,
"Ignore titles like this..." from a work-list row, and ignoring one title. Real Preferences, a real profile file and a real
`LibraryIndex` (fixture rows, or a real scan). `import ui.beq` first: see AGENTS.md gotcha 3.
'''
import ui.beq  # noqa: F401 (must come first)

import json
import re

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QDialogButtonBox

from pipeline.library.profile import load_profile, read_config_file
from pipeline.library.source import LibraryItem
from worklist_fixture import title_row
from worklist_settings_fixture import NOW, Dialogs, _designer, accepting, make_prefs, open_window, profile_config, \
    write_profile  # noqa: F401 (the autouse fixture)

SOURCES = [('films', 'filesystem', NOW - 900, NOW - 900, '', 6), ('disk', 'filesystem', NOW - 900, NOW - 900, '', 3)]


def _rows():
    def row(title_id, title, path, kind='movie', year='2001', source='films', **more):
        return title_row(title_id, title, 'review', path=path, kind=kind, year=year, source=source, **more)

    return [
        row('k1', 'Bluey', '/films/Kids/Bluey/s01e01.mkv', kind='tv', year='2018'),
        row('k2', 'Frozen', '/films/Kids/Frozen.mkv', year='2013'),
        row('c1', 'Heat', '/films/Crime/Heat.mkv', year='1995', source='disk',
            external_ids=json.dumps({'imdb': 'tt0113277'})),
        row('o1', 'Metropolis', '/films/Old/Metropolis.mkv', year='1927'),
        row('n1', 'Mystery', '', year=''),                                       # no path, no year
        row('sh', 'Heat (copy)', '/films/Crime/Heat2.mkv', shadowed_by='c1'),   # not a title of its own
        row('gn', 'Gone', '/films/Kids/Gone.mkv', gone=1),                      # not there any more
        row('se', 'Show', '/tv/Show/Season 1', kind='tv', year='2020', unit='season')]


def _open(qtbot, tmp_path, dialogs=None, **config):
    path = write_profile(tmp_path, **config)
    window = open_window(qtbot, tmp_path, make_prefs(tmp_path, path), dialogs, rows=_rows(), index_sources=SOURCES)
    return window, window.open_settings(), path


def _ok(dialog) -> QDialogButtonBox.StandardButton:
    return dialog.buttons.button(QDialogButtonBox.StandardButton.Ok)


# --- the live count ------------------------------------------------------------------------------------------------------------

def test_the_preview_counts_the_titles_the_rules_would_ignore_from_the_index_rows(qtbot, tmp_path):
    window, drawer, path = _open(qtbot, tmp_path, ignore=[{'path': '/films/Kids/**'}, {'kind': 'tv'}, {'year': '<1960'}],
                                 ignore_titles={'c1': 'rip is broken', 'k1': 'also matched by a rule'})

    tab = drawer.ignoreTab
    preview = tab.preview()

    assert preview.total == 6                      # the shadowed and the gone are not titles
    assert preview.per_rule == (2, 2, 1)           # Bluey+Frozen ('Gone' is gone), Bluey+Show, Metropolis
    assert (preview.by_rules, preview.by_id, preview.matched) == (4, 1, 5)     # Bluey, Frozen, Show, Metropolis; Heat by id (Bluey is ignored twice: counted once)
    assert tab.previewLabel.text() == 'Would ignore 5 of 6 titles (4 by rule, 1 by id) (from the last scan)'


def test_the_count_says_what_it_cannot_judge_from_the_index_rows(qtbot, tmp_path):
    window, drawer, path = _open(qtbot, tmp_path, ignore=[
        {'year': '<1960'}, {'path': '/tv/**'}, {'external_ids': {'imdb': 'tt0113277'}}, {'source': 'elsewhere', 'kind': 'tv'}])

    notes = '\n'.join(drawer.ignoreTab.preview().notes)

    assert 'Rule 1 uses year: 1 title has no a four-digit year in the index, so it cannot match it.' in notes
    assert 'Rule 2 uses path: 1 title has no path in the index' in notes
    assert 'Rule 2: 1 season row is checked by the season\'s own path' in notes
    assert 'Rule 3 uses external ids: 5 titles have no external ids in the index' in notes
    assert 'Rule 4 names the source "elsewhere", which the profile does not have' in notes
    assert drawer.ignoreTab.notesLabel.isVisibleTo(drawer.ignoreTab)


def test_with_no_titles_yet_the_preview_says_to_rescan(qtbot, tmp_path):
    path = write_profile(tmp_path, ignore=[{'kind': 'tv'}])
    window = open_window(qtbot, tmp_path, make_prefs(tmp_path, path))          # no index at all
    drawer = window.open_settings()

    assert drawer.ignoreTab.preview().scanned is False
    assert 'Rescan' in drawer.ignoreTab.previewLabel.text()


def test_the_count_follows_the_rows_when_they_change(qtbot, tmp_path):
    window, drawer, path = _open(qtbot, tmp_path, ignore=[{'kind': 'tv'}])
    assert drawer.ignoreTab.preview().matched == 2

    drawer.set_rows(window.model.rows[:2])          # what refresh_from_index() hands it after a scan or a run

    assert drawer.ignoreTab.preview().total == 2 and drawer.ignoreTab.preview().matched == 1
    assert 'of 2 titles' in drawer.ignoreTab.previewLabel.text()


# --- the rule editor -------------------------------------------------------------------------------------------------------------

def test_a_rule_is_added_edited_and_removed_and_each_change_is_in_the_file(qtbot, tmp_path):
    dialogs = Dialogs()
    window, drawer, path = _open(qtbot, tmp_path, dialogs)
    tab = drawer.ignoreTab
    seen = {}

    def fill(dialog):
        dialog.set_value('path', '/films/Kids', True)
        dialog.set_value('kind', 'tv', True)
        dialog.reasonEdit.setText('not for the catalogue')
        seen['preview'] = dialog.previewLabel.text()

    dialogs.next = accepting(fill)
    assert tab.add_rule() and drawer.flush()

    assert read_config_file(path)['ignore'] == [{'path': '/films/Kids', 'kind': 'tv', 'reason': 'not for the catalogue'}]
    assert seen['preview'].startswith('This rule matches 1 title.') and 'Would ignore 1 of 6 titles' in seen['preview']
    assert tab.ruleList.item(0).text() == '1.  path /films/Kids and kind tv (not for the catalogue)'

    tab.ruleList.setCurrentRow(0)
    tab.ruleList.item(0).setSelected(True)

    def edit(dialog):
        assert dialog.checks['path'].isChecked() and dialog.pathEdit.text() == '/films/Kids'    # shows the rule
        assert dialog.checks['kind'].isChecked() and not dialog.checks['year'].isChecked()
        dialog.set_value('kind', 'tv', False)
        dialog.set_value('year', '<2015', True)

    dialogs.next = accepting(edit)
    assert tab.edit_selected() and drawer.flush()
    assert read_config_file(path)['ignore'] == [{'path': '/films/Kids', 'year': '<2015', 'reason': 'not for the catalogue'}]

    assert tab.remove_selected() and drawer.flush()
    assert 'ignore' not in read_config_file(path)
    assert tab.previewLabel.text() == 'No ignore rules: nothing is ignored.'


def test_every_kind_of_invalid_rule_is_explained_in_the_dialog_and_nothing_is_written(qtbot, tmp_path):
    dialogs = Dialogs()
    window, drawer, path = _open(qtbot, tmp_path, dialogs)
    errors, enabled, rules = {}, {}, []

    def probe(dialog):
        errors['none ticked'], enabled['none ticked'] = dialog.errorLabel.text(), _ok(dialog).isEnabled()
        for name, value in (('title', '('), ('year', '19x0'), ('path', ''), ('external_ids', 'imdb')):
            for other in dialog.checks:
                dialog.checks[other].setChecked(False)
            dialog.set_value(name, value, True)
            errors[name], enabled[name] = dialog.errorLabel.text(), _ok(dialog).isEnabled()
        dialog.accept()                                   # refused: still no rule
        rules.append(dialog.rule)
        return False

    dialogs.next = probe
    before = open(path, 'rb').read()
    assert not drawer.ignoreTab.add_rule()

    assert 'at least one of' in errors['none ticked']
    assert 'not a valid regular expression' in errors['title'] and 'unterminated' in errors['title']
    assert 'not a year' in errors['year']
    assert 'path is ticked but empty' in errors['path']
    assert 'name=value' in errors['external_ids']
    assert not any(enabled.values()) and len(enabled) == 5     # OK is off for every one of them
    assert rules == [None]
    assert drawer.flush() and open(path, 'rb').read() == before


# --- "Ignore titles like this..." and ignoring one title --------------------------------------------------------------------

def test_ignore_titles_like_this_opens_the_rule_editor_filled_from_the_row(qtbot, tmp_path):
    dialogs = Dialogs()
    window, drawer, path = _open(qtbot, tmp_path, dialogs)
    window.select_ids(['k1'])
    seen = {}

    def fill(dialog):
        seen['fields'] = (dialog.pathEdit.text(), dialog.kindCombo.currentText(), dialog.yearEdit.text(),
                          dialog.titleEdit.text(), dialog.sourceCombo.currentText())
        seen['ticked'] = [name for name, check in dialog.checks.items() if check.isChecked()]
        seen['just_this'] = dialog.thisTitleButton.isVisibleTo(dialog)
        seen['error'] = dialog.errorLabel.text()

    dialogs.next = accepting(fill)
    assert window.ignore_like_selected()
    assert drawer.flush()

    assert seen['fields'] == ('/films/Kids/Bluey', 'tv', '2018', 'Bluey', 'films')
    assert seen['ticked'] == ['path'] and seen['just_this'] is True and seen['error'] == ''   # the folder alone is the rule
    assert window.settings_dock.isVisible() and drawer.tabs.currentWidget() is drawer.ignoreTab
    assert read_config_file(path)['ignore'] == [{'path': '/films/Kids/Bluey'}]
    assert window.settings_stale                               # ignoring is applied by a scan: the banner says so


def test_the_prefilled_title_is_escaped_so_it_matches_the_title_literally(qtbot, tmp_path):
    from model.worklist_edit import prefill_from_row
    from pipeline.library.ignore import rule_from_config
    from pipeline.library.index import LibraryIndex, index_path
    from worklist_fixture import make_index
    title = 'Mission: Impossible (1996) [4K]'
    make_index(tmp_path, [title_row('x', title, 'review', path=r'D:\Films\Mission\m.mkv', year='1996')])
    with LibraryIndex(index_path(str(tmp_path))) as index:
        (row,) = index.titles()

    values = prefill_from_row(row)

    assert values['path'] == 'D:\\Films\\Mission' and values['year'] == '1996'
    assert re.search(values['title'], title) and not re.search(values['title'], 'Mission: Impossible')
    assert rule_from_config({'title': values['title']}).matches(LibraryItem('x', 'p', 'n', title=title))


def test_ignore_just_this_title_is_stored_with_its_reason_and_survives_a_reload(qtbot, tmp_path):
    dialogs = Dialogs()
    window, drawer, path = _open(qtbot, tmp_path, dialogs)
    window.select_ids(['k2'])

    def fill(dialog):
        dialog.reasonEdit.setText('bad rip')
        dialog.thisTitleButton.click()
        return True

    dialogs.next = fill
    assert window.ignore_like_selected() and drawer.flush()

    assert read_config_file(path)['ignore_titles'] == {'k2': 'bad rip'} and 'ignore' not in read_config_file(path)
    assert load_profile(path).ignored_titles == {'k2': 'bad rip'}
    assert window.settings_stale                                # a scan is what moves it to Done
    assert drawer.ignoreTab.titleList.item(0).text() == 'Frozen - bad rip'
    window.reload()
    assert drawer.ignoreTab.ignored_titles == {'k2': 'bad rip'}

    drawer.ignoreTab.titleList.item(0).setSelected(True)
    assert drawer.ignoreTab.unignore_selected() and drawer.flush()
    assert 'ignore_titles' not in read_config_file(path)


def test_selected_titles_can_be_ignored_one_by_one_from_the_button_and_the_menu(qtbot, tmp_path):
    window, drawer, path = _open(qtbot, tmp_path)
    assert not window.ignoreButton.isEnabled()                    # nothing selected

    window.select_ids(['k1', 'k2'])
    assert window.ignoreButton.isEnabled()
    like, single = window.ignoreButton.menu().actions()
    assert not like.isEnabled() and single.isEnabled()            # "like this" needs exactly one title

    assert window.ignore_selected_titles('not wanted') and drawer.flush()
    assert load_profile(path).ignored_titles == {'k1': 'not wanted', 'k2': 'not wanted'}

    window.select_ids(['o1'])
    assert window.ignoreButton.menu().actions()[0].isEnabled()


def test_the_context_menu_of_a_row_offers_ignore(qtbot, tmp_path):
    window, drawer, path = _open(qtbot, tmp_path)
    assert window.workTable.contextMenuPolicy() == Qt.ContextMenuPolicy.CustomContextMenu
    assert window.ignoreButton.menu() is not None
    assert [a.text() for a in window.ignoreButton.menu().actions()] == ['Ignore titles like this...', 'Ignore this title...']


def test_an_ignored_title_moves_to_done_at_the_rescan_the_banner_offers(qtbot, tmp_path):
    items = [LibraryItem(id='fs-1', source_path='/films/Kids/a.mkv', display_name='a', title='Alpha', year='2001'),
             LibraryItem(id='fs-2', source_path='/films/Crime/b.mkv', display_name='b', title='Beta', year='1999')]

    class Source:
        def list_items(self, **_):
            return list(items)

    (tmp_path / 'queue').mkdir(exist_ok=True)
    dialogs = Dialogs()
    path = write_profile(tmp_path, sources=[{'name': 'films', 'kind': 'filesystem', 'globs': ['/films']}])
    window = open_window(qtbot, tmp_path, make_prefs(tmp_path, path), dialogs, sources={'films': Source()})
    with qtbot.waitSignal(window.scan_finished, timeout=15000):
        window.rescan()
    assert {r.id: r.needs for r in window.model.rows} == {'fs-1': 'extract', 'fs-2': 'extract'}
    drawer = window.open_settings()

    dialogs.next = accepting(lambda d: (d.set_value('path', '/films/Kids', True), d.reasonEdit.setText('kids')))
    assert drawer.ignoreTab.add_rule() and drawer.flush()
    assert {r.id: r.needs for r in window.model.rows}['fs-1'] == 'extract'      # a rule takes effect at the scan
    assert window.settings_stale

    with qtbot.waitSignal(window.scan_finished, timeout=15000):
        qtbot.mouseClick(window.staleBannerButton, Qt.MouseButton.LeftButton)

    rows = {r.id: r for r in window.model.rows}
    assert rows['fs-1'].needs == 'done' and 'ignored by rule: path /films/Kids (kids)' in rows['fs-1'].ignored
    assert rows['fs-2'].needs == 'extract'
