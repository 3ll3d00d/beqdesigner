'''
model/worklist.py: the library work list window (chunk 26a, read-only), driven like a user would against a fixture
discovery index (worklist_fixture.py inserts the rows) under pytest-qt. `import ui.beq` first: see AGENTS.md gotcha 3.
'''
import ui.beq  # noqa: F401 (must come first)

import json
import os
import logging
import threading
import time

import pytest
from qtpy.QtCore import QSettings, Qt
from qtpy.QtWidgets import QApplication

from model.preferences import DESIGNER_DEFAULT, DESIGNER_QUEUE_DIR, LIBRARY_FILESYSTEM_GLOBS, LIBRARY_PROFILE_PATH, \
    LIBRARY_WORK_DIR, Preferences, SYSTEM_CHECK_FOR_UPDATES
from model.worklist import WorkListWindow, describe_sources, format_time
from model.worklist_model import COL_DETAIL, COL_NEEDS, COL_TITLE, COL_WAITING, NEW_ROLE, ROW_ROLE, format_waiting
from pipeline.designer.registry import register_designer, unregister_designer
from pipeline.library.index import LibraryIndex, index_path
from pipeline.library.selection import CHIPS, Selection
from pipeline.library.source import LibraryItem
from worklist_fixture import make_index, title_row

NOW = 1_800_000_000.0
DAY = 86400.0
DESIGNER = 'test.worklist'


@pytest.fixture(autouse=True)
def _designer():
    register_designer(DESIGNER, lambda request: None)
    yield
    unregister_designer(DESIGNER)


def _prefs(tmp_path, configured=True, **more):
    prefs = Preferences(QSettings(str(tmp_path / 'settings.ini'), QSettings.Format.IniFormat))
    if configured:
        prefs.set(LIBRARY_WORK_DIR, str(tmp_path / 'work'))
        prefs.set(DESIGNER_QUEUE_DIR, str(tmp_path / 'queue'))
        prefs.set(LIBRARY_FILESYSTEM_GLOBS, ['/films/**/*.mkv'])
        prefs.set(DESIGNER_DEFAULT, DESIGNER)
        (tmp_path / 'work').mkdir(exist_ok=True)
    for key, value in more.items():
        prefs.set(key, value)
    return prefs


def _rows():
    ''' One or more titles of every kind. Generation is 2, so `first_seen_generation=2` is new since the last scan. '''
    def row(title_id, title, needs, days, **fields):
        return title_row(title_id, title, needs, state_since=NOW - days * DAY, **fields)

    return [
        row('t-dune', 'Dune', 'attention', 3, detail='extract failed: file not found (path mapping?)', source='jriver'),
        row('t-heat', 'Heat', 'attention', 1, detail='source changed since accepted', source='disk',
            first_seen_generation=2),
        row('t-alien', 'Alien', 'review', 9, detail='conf 0.62 - 3 candidates', source='jriver', confidence=0.62),
        row('t-sicario', 'Sicario', 'review', 2, detail='conf 0.91 - 2 candidates', source='disk',
            first_seen_generation=2, in_catalogue=1),
        row('t-arrival', 'Arrival', 'review', 5, detail='conf 0.75 - 1 candidate', source='jriver'),
        row('t-gravity', 'Gravity', 'extract', 0.5, detail='new', source='disk', first_seen_generation=2),
        row('t-tenet', 'Tenet', 'design', 4, detail='new', source='jriver'),
        row('t-fury', 'Fury', 'publish', 6, detail='accepted, not written to the repository', source='jriver'),
        row('t-speed', 'Speed', 'commit', 7, detail='written, not committed', source='disk'),
        row('t-old', 'Old Film', 'done', 100, detail='pushed', source='jriver'),
        row('t-shadow', 'Shadow Film', 'done', 1, detail='shadowed: the same file is title t-dune', source='disk',
            first_seen_generation=2, shadowed_by='t-dune'),
    ]


SOURCES = [('jriver', 'jriver', NOW - 900, NOW - 900, '', 6), ('disk', 'filesystem', NOW - 900, NOW - 900, '', 5)]


def _window(qtbot, tmp_path, rows=None, sources=SOURCES, prefs=None, generation=2, auto_scan=False, **kwargs):
    prefs = prefs or _prefs(tmp_path)
    if rows is not None:
        make_index(tmp_path / 'work', rows, sources, generation=generation, last_scan_at=NOW - 900)
    window = WorkListWindow(None, prefs, auto_scan=auto_scan, clock=lambda: NOW, **kwargs)
    qtbot.addWidget(window)
    window.show()
    return window


def _click(qtbot, button):
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)


# --- the strip and the filters ------------------------------------------------------------------------------------------

def test_the_strip_counts_each_kind_of_work_and_hides_done(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, _rows())

    assert window.chip_counts() == {'All': 9, 'Attention': 2, 'New': 3, 'Extract': 1, 'Design': 1, 'Review': 3,
                                    'Publish': 1, 'Commit': 1, 'Done': 2}
    assert window.attentionChip.text() == 'Attention 2'
    assert window.reviewChip.text() == 'Review 3'
    assert window.allChip.text() == 'All 9'
    assert window.doneChip.text() == 'Done 2 (hidden)'
    assert window.chip == 'All' and window.allChip.isChecked()
    assert 't-old' not in window.listed_ids() and 't-shadow' not in window.listed_ids()  # Done is hidden by default


def test_the_strip_counts_agree_with_the_index_summary(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, _rows())

    with LibraryIndex(index_path(str(tmp_path / 'work'))) as index:
        summary = index.summary()
    counts = window.chip_counts()

    assert {n.capitalize(): c for n, c in summary.counts.items() if n != 'done'} == \
           {chip: counts[chip] for chip in ('Attention', 'Review', 'Extract', 'Design', 'Publish', 'Commit')}
    assert counts['Done'] == summary.counts['done']
    assert counts['New'] == 3  # the summary's 4 new titles less the Done one, which the strip does not count


@pytest.mark.parametrize('chip,button,expected', [
    ('Attention', 'attentionChip', ['t-dune', 't-heat']),
    ('Review', 'reviewChip', ['t-alien', 't-arrival', 't-sicario']),
    ('Extract', 'extractChip', ['t-gravity']),
    ('Design', 'designChip', ['t-tenet']),
    ('Publish', 'publishChip', ['t-fury']),
    ('Commit', 'commitChip', ['t-speed']),
    ('New', 'newChip', ['t-heat', 't-sicario', 't-gravity']),
    ('Done', 'doneChip', ['t-old', 't-shadow'])])
def test_clicking_a_chip_lists_only_that_kind_of_work(qtbot, tmp_path, chip, button, expected):
    window = _window(qtbot, tmp_path, _rows())

    _click(qtbot, getattr(window, button))

    assert window.chip == chip
    assert window.listed_ids() == expected
    assert window.contentStack.currentWidget() is window.tablePage


def test_done_is_shown_by_its_chip_and_all_brings_the_default_view_back(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, _rows())
    assert window.doneChip.text().endswith('(hidden)')

    _click(qtbot, window.doneChip)
    assert window.listed_ids() == ['t-old', 't-shadow']
    assert window.doneChip.text() == 'Done 2'  # nothing is hidden now

    _click(qtbot, window.allChip)
    assert 't-old' not in window.listed_ids() and len(window.listed_ids()) == 9
    assert window.doneChip.text() == 'Done 2 (hidden)'


def test_rows_are_ordered_by_tier_then_oldest_first(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, _rows())

    # attention (oldest first: Dune 3d, Heat 1d), then human (Alien 9d, Arrival 5d, Sicario 2d), then machine:
    # Fury 6d... in the index's order -- machine rows are ordered by how long each has waited, whatever it needs
    assert window.listed_ids() == ['t-dune', 't-heat', 't-alien', 't-arrival', 't-sicario',
                                   't-speed', 't-fury', 't-tenet', 't-gravity']
    assert window.sort_column == -1
    waits = [window.proxy.index(r, COL_WAITING).data() for r in range(window.proxy.rowCount())]
    assert waits[0] == '3d' and waits[1] == 'new · 1d' and waits[2] == '9d'


def test_the_order_is_the_index_order_not_something_the_window_works_out(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, _rows())

    with LibraryIndex(index_path(str(tmp_path / 'work'))) as index:
        assert window.listed_ids() == [r.id for r in index.titles(include_done=False)]


def test_a_header_click_sorts_by_that_column_and_a_third_click_restores_the_default_order(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, _rows())
    default = window.listed_ids()
    header = window.workTable.horizontalHeader()

    header.sectionClicked.emit(COL_TITLE)
    assert window.sort_column == COL_TITLE
    titles = [window.proxy.index(r, COL_TITLE).data() for r in range(window.proxy.rowCount())]
    assert titles == sorted(titles, key=str.casefold)

    header.sectionClicked.emit(COL_TITLE)
    titles = [window.proxy.index(r, COL_TITLE).data() for r in range(window.proxy.rowCount())]
    assert titles == sorted(titles, key=str.casefold, reverse=True)

    header.sectionClicked.emit(COL_TITLE)
    assert window.sort_column == -1
    assert window.listed_ids() == default


def test_search_matches_the_title_the_id_and_the_path_ignoring_case_and_the_strip_follows(qtbot, tmp_path):
    rows = _rows() + [title_row('t-path', 'Unrelated', 'review', path='/films/Kids/Zebra.mkv', source='disk')]
    window = _window(qtbot, tmp_path, rows)

    window.searchEdit.setText('dUnE')
    assert window.listed_ids() == ['t-dune']
    assert window.chip_counts()['Attention'] == 1 and window.chip_counts()['Review'] == 0

    window.searchEdit.setText('t-sic')  # an id
    assert window.listed_ids() == ['t-sicario']
    window.searchEdit.setText('kids/zebra')  # a path
    assert window.listed_ids() == ['t-path']

    window.searchEdit.setText('')
    assert len(window.listed_ids()) == 10


def test_the_source_combo_shows_one_sources_titles(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, _rows())
    assert [window.sourceCombo.itemText(i) for i in range(window.sourceCombo.count())] == \
           ['All sources', 'jriver', 'disk']

    window.sourceCombo.setCurrentIndex(window.sourceCombo.findData('disk'))

    assert window.listed_ids() == ['t-heat', 't-sicario', 't-speed', 't-gravity']
    assert window.chip_counts()['All'] == 4
    _click(qtbot, window.reviewChip)
    assert window.listed_ids() == ['t-sicario']  # the chip and the source both narrow it


def test_column_filters_match_only_their_own_column_and_combine(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, _rows())

    window.columnFilters[COL_NEEDS].setText('review')
    assert window.listed_ids() == ['t-alien', 't-arrival', 't-sicario']
    assert window.chip_counts()['Review'] == 3 and window.chip_counts()['All'] == 3

    window.columnFilters[COL_DETAIL].setText('0.75')
    assert window.listed_ids() == ['t-arrival']
    with LibraryIndex(index_path(str(tmp_path / 'work'))) as index:
        assert [row.id for row in window.current_selection().rows(index)] == ['t-arrival']

    window.clearColumnFiltersButton.click()
    assert not window.proxy.column_filters and len(window.listed_ids()) == 9


@pytest.mark.parametrize('chip', ['All', *CHIPS])
@pytest.mark.parametrize('source,text', [(None, ''), ('jriver', ''), (None, 'a'), ('disk', 'e')])
def test_what_is_listed_is_what_the_selection_it_amounts_to_selects(qtbot, tmp_path, chip, source, text):
    ''' 26b's action works on `current_selection()`; the table must list exactly those titles, in the same order. '''
    window = _window(qtbot, tmp_path, _rows())
    window.sourceCombo.setCurrentIndex(max(window.sourceCombo.findData(source), 0))
    window.searchEdit.setText(text)
    window.set_chip(chip)

    selection = window.current_selection()

    assert isinstance(selection, Selection)
    with LibraryIndex(index_path(str(tmp_path / 'work'))) as index:
        assert window.listed_ids() == [r.id for r in selection.rows(index)]


# --- new since the last scan ---------------------------------------------------------------------------------------------

def test_titles_first_seen_by_the_latest_scan_are_highlighted(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, _rows())
    model = window.proxy

    new, old = {}, {}
    for row in range(model.rowCount()):
        index = model.index(row, COL_TITLE)
        (new if index.data(NEW_ROLE) else old)[index.data(ROW_ROLE).id] = index
    assert set(new) == {'t-heat', 't-sicario', 't-gravity'}  # generation 2 == first_seen_generation
    assert all(i.data(Qt.ItemDataRole.BackgroundRole) is not None for i in new.values())
    assert all(i.data(Qt.ItemDataRole.BackgroundRole) is None for i in old.values())
    assert model.index(0, COL_WAITING).data().startswith('3d')
    heat = new['t-heat']
    assert model.index(heat.row(), COL_WAITING).data() == 'new · 1d'  # and says so in words too


def test_nothing_is_new_before_the_first_scan(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, _rows(), generation=0)

    assert window.chip_counts()['New'] == 0
    assert not any(window.proxy.index(r, 0).data(NEW_ROLE) for r in range(window.proxy.rowCount()))


# --- the empty states -----------------------------------------------------------------------------------------------------

def test_an_unconfigured_library_shows_an_empty_state_that_points_to_settings(qtbot, tmp_path):
    prefs = _prefs(tmp_path, configured=False)
    window = _window(qtbot, tmp_path, prefs=prefs)

    assert window.contentStack.currentWidget() is window.emptyPage
    assert 'not set up' in window.emptyTitleLabel.text()
    assert 'No work directory' in window.emptyDetailLabel.text()
    assert 'No library source' in window.emptyDetailLabel.text()
    assert window.openSettingsButton.isVisibleTo(window)
    assert not window.rescanButton.isEnabled()
    assert 'Cannot scan' in window.rescanButton.toolTip()
    with qtbot.waitSignal(window.settings_requested, timeout=1000):
        _click(qtbot, window.openSettingsButton)


def test_a_work_directory_that_does_not_exist_is_reported_and_not_created(qtbot, tmp_path):
    prefs = _prefs(tmp_path, **{LIBRARY_WORK_DIR: str(tmp_path / 'nowhere')})

    window = _window(qtbot, tmp_path, prefs=prefs)

    assert 'does not exist' in window.emptyDetailLabel.text()
    assert not (tmp_path / 'nowhere').exists()
    assert not window.rescanButton.isEnabled()


def test_a_configured_library_that_was_never_scanned_says_so_and_can_be_scanned(qtbot, tmp_path):
    window = _window(qtbot, tmp_path)  # an empty index: the work directory exists, generation 0

    assert window.contentStack.currentWidget() is window.emptyPage
    assert 'has not been scanned' in window.emptyTitleLabel.text()
    assert window.lastScanLabel.text() == 'never scanned'
    assert window.rescanButton.isEnabled()
    assert not window.openSettingsButton.isVisibleTo(window)


def test_a_filter_that_matches_nothing_says_so_rather_than_showing_an_empty_table(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, _rows())

    window.searchEdit.setText('zzzz')
    assert window.contentStack.currentWidget() is window.emptyPage
    assert 'Nothing matches' in window.emptyTitleLabel.text()

    window.searchEdit.setText('')
    assert window.contentStack.currentWidget() is window.tablePage


def test_a_profile_file_that_cannot_be_read_is_an_empty_state_not_an_exception(qtbot, tmp_path):
    bad = tmp_path / 'profile.yaml'
    bad.write_text('sources: [oops')
    window = _window(qtbot, tmp_path, prefs=_prefs(tmp_path, **{LIBRARY_PROFILE_PATH: str(bad)}))

    assert 'could not be read' in window.emptyTitleLabel.text()
    assert str(bad) in window.emptyDetailLabel.text()
    assert not window.rescanButton.isEnabled()
    assert not window.openSettingsButton.isVisibleTo(window)  # Library Sync's settings are not what is wrong


# --- sources that could not be listed -----------------------------------------------------------------------------------

def test_a_source_that_failed_to_list_is_flagged_with_why_and_which_listing_is_shown(qtbot, tmp_path):
    sources = [('jriver', 'jriver', NOW - 900, NOW - 900, '', 6),
               ('disk', 'filesystem', NOW - 60, NOW - 2 * DAY, 'ConnectionRefusedError: no route to host', 5)]
    window = _window(qtbot, tmp_path, _rows(), sources=sources)

    text = window.sourceStatusLabel.text()
    assert window.sourceStatusLabel.isVisibleTo(window)
    assert 'disk: could not be listed (ConnectionRefusedError: no route to host)' in text
    assert 'Showing its listing from' in text
    assert 'jriver: scanned' in text
    assert window.sourceCombo.findText('disk (!)') >= 0
    assert window.sourceCombo.findData('disk') >= 0
    assert window.listed_ids()  # its titles are still listed: the last listing is kept


def test_a_single_healthy_source_needs_no_status_line(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, _rows(), sources=SOURCES[:1])

    assert not window.sourceStatusLabel.isVisibleTo(window)
    assert window.lastScanLabel.text() == f'last scan {format_time(NOW - 900, NOW)}'


def test_describe_sources_covers_ok_never_scanned_failed_and_never_listed():
    lines = describe_sources([
        __row('ok', NOW - 60, NOW - 60, '', 1), __row('fresh', None, None, '', 0),
        __row('bad', NOW - 60, NOW - DAY, 'boom', 3), __row('never', NOW - 60, None, 'boom', 0)], NOW)

    assert lines[0] == (f'ok: scanned {format_time(NOW - 60, NOW)}, 1 title', False)
    assert lines[1] == ('fresh: not scanned yet', False)
    assert lines[2][1] and 'could not be listed (boom)' in lines[2][0] and 'Showing its listing from' in lines[2][0]
    assert lines[3][1] and 'never been listed' in lines[3][0]


def __row(name, scanned, ok, error, count):
    from pipeline.library.index import SourceRow
    return SourceRow(name, 0, 'filesystem', scanned, ok, error, count)


def test_times_show_the_date_only_when_it_is_not_today():
    today = format_time(NOW - 60, NOW)
    assert len(today) == 5 and today[2] == ':'
    assert len(format_time(NOW - 3 * DAY, NOW)) == len('2026-01-01 09:14')


def test_waiting_reads_as_minutes_hours_days_months():
    assert [format_waiting(NOW - s, NOW) for s in (5, 300, 7200, 3 * DAY, 90 * DAY, 800 * DAY)] == \
           ['now', '5m', '2h', '3d', '3mo', '2y']


# --- rescanning ----------------------------------------------------------------------------------------------------------

class _Source:
    ''' A library source that says which thread listed it, and can be held mid-listing. '''

    def __init__(self, titles=(), error=None, hold=False):
        self.titles, self.error = list(titles), error
        self.release = threading.Event()
        self.hold = hold
        self.entered = threading.Event()
        self.thread = None
        self.calls = 0

    def list_items(self, **query):
        self.calls += 1
        self.thread = threading.current_thread()
        self.entered.set()
        if self.hold:
            assert self.release.wait(20), 'the test never released the source'
        if self.error:
            raise self.error
        return [LibraryItem(id=f'fs-{t}', source_path=f'/films/{t}.mkv', display_name=t, title=t, year='2001',
                            fingerprint=f'fp-{t}') for t in self.titles]


def _scan_window(qtbot, tmp_path, source, generation=None, rows=(), prefs=None, **kwargs):
    ''' A window whose one profile source (the bootstrapped `filesystem`) is `source`. '''
    prefs = prefs or _prefs(tmp_path)
    if generation is not None:
        make_index(tmp_path / 'work', rows, [], generation=generation, last_scan_at=NOW)
    window = WorkListWindow(None, prefs, sources={'filesystem': source}, clock=time.time, **kwargs)
    qtbot.addWidget(window)
    window.show()
    return window


def test_rescan_lists_the_sources_on_a_worker_thread_and_never_blocks_the_ui(qtbot, tmp_path):
    source = _Source(['Alpha', 'Beta'], hold=True)
    window = _scan_window(qtbot, tmp_path, source, auto_scan=False)
    assert window.rescanButton.isEnabled() and window.rescanButton.text() == 'Rescan'

    _click(qtbot, window.rescanButton)

    # the click returned while the source is still listing: the UI thread was not the one waiting
    assert window.is_scanning
    assert window.rescanButton.text() == 'Scanning...' and not window.rescanButton.isEnabled()
    assert window.lastScanLabel.text() == 'scanning...'
    qtbot.waitUntil(source.entered.is_set, timeout=5000)
    assert source.thread is not threading.current_thread()
    QApplication.processEvents()  # the event loop still turns
    assert window.is_scanning and window.listed_ids() == []
    assert window.rescan() is False  # one scan at a time

    with qtbot.waitSignal(window.scan_finished, timeout=10000) as scan:
        source.release.set()

    assert not window.is_scanning and window.rescanButton.isEnabled()
    assert scan.args[0].titles == 2 and len(scan.args[0].new) == 2
    assert sorted(window.listed_ids()) == ['fs-Alpha', 'fs-Beta']
    assert 'Scan finished: 2 titles, 2 new' in window.statusBar.currentMessage()
    assert window.lastScanLabel.text().startswith('last scan ')
    assert window.chip_counts()['New'] == 2 and window.chip_counts()['Extract'] == 2


def test_the_window_scans_by_itself_on_opening_only_if_the_index_was_never_scanned(qtbot, tmp_path):
    fresh = _Source(['Alpha'])
    window = _scan_window(qtbot, tmp_path, fresh)
    qtbot.waitUntil(lambda: fresh.calls == 1 and not window.is_scanning, timeout=10000)
    assert window.listed_ids() == ['fs-Alpha']

    scanned = _Source(['Beta'])
    other = tmp_path / 'other'
    other.mkdir()
    prefs = _prefs(other)
    window = _scan_window(qtbot, other, scanned, generation=3, rows=[title_row('t-x', 'X', 'review')], prefs=prefs)
    QApplication.processEvents()
    assert not window.is_scanning and scanned.calls == 0  # cached index shown, no scan unasked
    assert window.listed_ids() == ['t-x']


def test_a_source_that_fails_during_a_rescan_is_flagged_and_keeps_its_last_titles(qtbot, tmp_path):
    good = _Source(['Alpha', 'Beta'])
    window = _scan_window(qtbot, tmp_path, good, auto_scan=False)
    with qtbot.waitSignal(window.scan_finished, timeout=10000):
        window.rescan()
    assert sorted(window.listed_ids()) == ['fs-Alpha', 'fs-Beta']

    good.error = ConnectionError('server is down')
    with qtbot.waitSignal(window.scan_finished, timeout=10000) as scan:
        window.rescan()

    assert scan.args[0].errors == {'filesystem': 'ConnectionError: server is down'}
    assert sorted(window.listed_ids()) == ['fs-Alpha', 'fs-Beta']  # kept
    assert window.sourceStatusLabel.isVisibleTo(window)
    assert 'server is down' in window.sourceStatusLabel.text()
    assert window.sourceCombo.findText('filesystem (!)') >= 0
    assert 'source(s) could not be listed' in window.statusBar.currentMessage()


def test_selection_survives_a_rescan(qtbot, tmp_path):
    source = _Source(['Alpha', 'Beta', 'Gamma'])
    window = _scan_window(qtbot, tmp_path, source, auto_scan=False)
    with qtbot.waitSignal(window.scan_finished, timeout=10000):
        window.rescan()
    window.select_ids(['fs-Beta', 'fs-Gamma'])
    assert sorted(window.selected_ids()) == ['fs-Beta', 'fs-Gamma']

    with qtbot.waitSignal(window.scan_finished, timeout=10000):
        window.rescan()

    assert sorted(window.selected_ids()) == ['fs-Beta', 'fs-Gamma']


def test_rescan_is_refused_when_the_setup_is_incomplete(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, prefs=_prefs(tmp_path, configured=False))

    assert window.rescan() is False
    assert not window.is_scanning


def test_the_window_reads_the_settings_again_when_reloaded(qtbot, tmp_path):
    prefs = _prefs(tmp_path, configured=False)
    window = _window(qtbot, tmp_path, prefs=prefs)
    assert window.contentStack.currentWidget() is window.emptyPage
    (tmp_path / 'work').mkdir()
    make_index(tmp_path / 'work', _rows(), SOURCES, last_scan_at=NOW)

    prefs.set(LIBRARY_WORK_DIR, str(tmp_path / 'work'))
    prefs.set(DESIGNER_QUEUE_DIR, str(tmp_path / 'queue'))
    prefs.set(LIBRARY_FILESYSTEM_GLOBS, ['/films/*.mkv'])
    prefs.set(DESIGNER_DEFAULT, DESIGNER)
    window.reload()

    assert window.contentStack.currentWidget() is window.tablePage
    assert len(window.listed_ids()) == 9


# --- the menu ------------------------------------------------------------------------------------------------------------

def test_the_tools_menu_has_the_work_list_as_its_only_library_entry_and_it_opens_the_window(qtbot, tmp_path):
    import app as app_module
    root = logging.getLogger()
    handlers = list(root.handlers)
    prefs = _prefs(tmp_path, **{SYSTEM_CHECK_FOR_UPDATES: False})
    main = app_module.BeqDesigner(QApplication.instance(), prefs)
    qtbot.addWidget(main)
    try:
        actions = main.menu_Tools.actions()
        assert main.action_Work_List in actions and not hasattr(main, 'action_Library_Sync')   # the classic dialog is gone (27c)
        assert main.action_Work_List.text() == 'Library &Work List'
        assert main.action_Work_List.shortcut().toString() == 'Ctrl+Shift+W'

        main.action_Work_List.trigger()

        window = main._BeqDesigner__work_list
        assert isinstance(window, WorkListWindow) and window.isVisible()
        assert window.parent() is main
        main.action_Work_List.trigger()  # shown again, not a second window
        assert main._BeqDesigner__work_list is window
    finally:
        for handler in list(root.handlers):
            if handler not in handlers:  # the main window's log viewer must not outlive it
                root.removeHandler(handler)


# --- the index file: never created just to look, failures shown, released on close ---------------------------------------------

def test_opening_on_a_work_directory_with_no_index_creates_nothing_and_says_it_was_never_scanned(qtbot, tmp_path):
    window = _window(qtbot, tmp_path)     # a work directory that exists, and no index in it

    assert not os.path.exists(index_path(str(tmp_path / 'work')))
    assert os.listdir(tmp_path / 'work') == []
    assert not window.has_open_index
    assert 'has not been scanned' in window.emptyTitleLabel.text()
    assert window.rescanButton.isEnabled()


def test_a_scan_creates_the_index_and_the_window_then_reads_it_and_a_never_scanned_window_still_scans_by_itself(
        qtbot, tmp_path):
    source = _Source(['Alpha'])
    window = _scan_window(qtbot, tmp_path, source)      # auto-scan: there is no index file, which is "never scanned"

    qtbot.waitUntil(lambda: source.calls == 1 and not window.is_scanning, timeout=10000)

    assert os.path.isfile(index_path(str(tmp_path / 'work')))
    assert window.listed_ids() == ['fs-Alpha'] and window.has_open_index


def test_an_index_that_exists_but_cannot_be_opened_is_shown_as_that_not_as_never_scanned(qtbot, tmp_path, monkeypatch):
    make_index(tmp_path / 'work', [], SOURCES, generation=2, last_scan_at=NOW)
    import model.worklist as worklist_module

    def refuse(path):
        raise PermissionError(13, 'Permission denied', path)

    monkeypatch.setattr(worklist_module, 'LibraryIndex', refuse)

    window = _window(qtbot, tmp_path, prefs=_prefs(tmp_path))

    title = window.emptyTitleLabel.text()
    assert 'could not be read' in title and 'not been scanned' not in title
    assert 'PermissionError' in window.emptyDetailLabel.text() and 'library-index' in window.emptyDetailLabel.text()
    assert not window.has_open_index


def test_closing_releases_the_index_and_a_scan_that_finishes_after_it_does_not_open_it_again(qtbot, tmp_path):
    source = _Source(['Alpha'], hold=True)
    window = _scan_window(qtbot, tmp_path, source, auto_scan=False)
    window.rescan()
    qtbot.waitUntil(source.entered.is_set, timeout=5000)

    window.close()
    assert not window.has_open_index

    with qtbot.waitSignal(window.scan_finished, timeout=10000):
        source.release.set()

    assert not window.has_open_index          # the finished scan did not reopen a connection in a closed window
    assert os.path.isfile(index_path(str(tmp_path / 'work')))    # the scan itself did its job
    window.show()                             # shown again: it reads the index it now has
    assert window.has_open_index and window.listed_ids() == ['fs-Alpha']
