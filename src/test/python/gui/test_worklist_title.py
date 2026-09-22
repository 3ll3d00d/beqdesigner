'''
model/worklist_title.py and model/worklist_titles.py, chunk 27a: the title page -- drilling in from the work list, the
candidates, commentary and chart, Accept & next / Skip / Reject, Previous / Next, and back to the table with the
selection, scroll and filters as they were. Driven like a user would against a fixture discovery index whose titles have
real queue entries, plus one test through a real scan (`fs-a`/`fs-b`: a decision moves a title from Review to Publish once
the page is left). `import ui.beq` first: see AGENTS.md gotcha 3.
'''
import ui.beq  # noqa: F401 (must come first)

import os
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest
from qtpy.QtCore import QSettings, Qt
from qtpy.QtGui import QShortcut
from qtpy.QtWidgets import QApplication

from model.preferences import DESIGNER_DEFAULT, DESIGNER_QUEUE_DIR, LIBRARY_FILESYSTEM_GLOBS, LIBRARY_WORK_DIR, \
    Preferences
from model.worklist import WorkListWindow
from model.worklist_model import ID_ROLE
from model.worklist_title import chart_data, next_waiting_id, notice_text, position_text, state_text
from pipeline.designer.registry import register_designer, unregister_designer
from pipeline.library.index import LibraryIndex
from pipeline.review import read_entry
from test_pipeline_library_index import DESIGNER, FakeSource, _entry as _real_entry, _extracted, _item
from worklist_fixture import make_index, title_row
from worklist_title_fixture import write_entry

NOW = 1_800_000_000.0
DAY = 86400.0
SOURCES = [('films', 'filesystem', NOW - 900, NOW - 900, '', 6)]


@pytest.fixture(autouse=True)
def _designer():
    register_designer(DESIGNER, lambda request: None)
    yield
    unregister_designer(DESIGNER)


def _prefs(tmp_path):
    prefs = Preferences(QSettings(str(tmp_path / 'settings.ini'), QSettings.Format.IniFormat))
    prefs.set(LIBRARY_WORK_DIR, str(tmp_path / 'work'))
    prefs.set(DESIGNER_QUEUE_DIR, str(tmp_path / 'queue'))
    prefs.set(LIBRARY_FILESYSTEM_GLOBS, [str(tmp_path / 'films' / '*.mkv')])
    prefs.set(DESIGNER_DEFAULT, DESIGNER)
    (tmp_path / 'work').mkdir(exist_ok=True)
    return prefs


def _row(title_id, title, needs, days, **fields):
    return title_row(title_id, title, needs, state_since=NOW - days * DAY, source='films', **fields)


def _index_row(needs, detail, review_state='none', title='T'):
    ''' What the pure functions read of a `TitleRow`. '''
    return SimpleNamespace(needs=needs, detail=detail, review_state=review_state, title=title, year='2001', flags=[],
                           source='films', kind='movie', path='')


def _rows():
    ''' Listed in this order: one that failed, three to review (oldest first), one to publish, one to extract. '''
    return [
        _row('a-dune', 'Dune', 'attention', 3, detail='extract failed: file not found', failure='file not found'),
        _row('r-alien', 'Alien', 'review', 9, detail='conf 0.90 - 2 candidates', review_state='pending', confidence=0.9),
        _row('r-arrival', 'Arrival', 'review', 5, detail='conf 0.90 - 2 candidates', review_state='pending'),
        _row('r-sicario', 'Sicario', 'review', 2, detail='conf 0.90 - 2 candidates', review_state='pending'),
        _row('p-fury', 'Fury', 'publish', 6, detail='accepted', review_state='accepted', publish_state='not_written'),
        _row('x-gravity', 'Gravity', 'extract', 1, detail='new'),
    ]


def _window(qtbot, tmp_path, entries=(), rows=None, **kwargs):
    ''' A shown window over the fixture rows; `entries` are `write_entry` calls (id, then keyword arguments). '''
    prefs = _prefs(tmp_path)
    make_index(tmp_path / 'work', rows if rows is not None else _rows(), SOURCES, generation=2, last_scan_at=NOW - 900)
    for entry_id, options in entries:
        write_entry(str(tmp_path / 'queue'), entry_id, **options)
    window = WorkListWindow(None, prefs, auto_scan=False, clock=lambda: NOW, **kwargs)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    window.activateWindow()   # offscreen: key events only reach a widget once its window is active
    qtbot.waitActive(window)
    return window


REVIEWABLE = [('r-alien', {}), ('r-arrival', {}), ('r-sicario', {}), ('p-fury', {'status': 'accepted', 'chosen': 1})]


def _queue(tmp_path):
    return str(tmp_path / 'queue')


def _status(tmp_path, entry_id):
    return read_entry(_queue(tmp_path), entry_id).status


def _focus(qtbot, widget):
    widget.setFocus()
    qtbot.waitUntil(widget.hasFocus)


def _click(qtbot, button):
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)


def _open(qtbot, window, title_id):
    ''' Opens the page and waits for the candidate list to have the keyboard, which is where the page puts it. '''
    assert window.open_title(title_id)
    qtbot.waitUntil(window.title_page.candidateList.hasFocus)
    return window.title_page


# --- what is said, with no widgets ----------------------------------------------------------------------------------------

def test_position_text():
    assert position_text(0, 37) == '1 of 37'
    assert position_text(1199, 1200) == '1,200 of 1,200'
    assert position_text(-1, 3) == '' and position_text(3, 3) == '' and position_text(0, 0) == ''


def test_the_next_waiting_title_is_after_this_one_and_wraps_round():
    ids = ['a', 'b', 'c', 'd']
    assert next_waiting_id(ids, 'a', lambda i: i in 'cd') == 'c'
    assert next_waiting_id(ids, 'c', lambda i: i in 'ad') == 'd'
    assert next_waiting_id(ids, 'd', lambda i: i in 'ab') == 'a'      # wraps to the top
    assert next_waiting_id(ids, 'b', lambda i: i == 'b') is None       # never the title itself
    assert next_waiting_id(ids, 'b', lambda i: False) is None
    assert next_waiting_id(ids, 'gone', lambda i: i == 'c') == 'c'     # not in the list: from the top


def test_chart_data_is_the_signal_and_the_filtered_signal(tmp_path):
    write_entry(str(tmp_path), 'a')
    entry = read_entry(str(tmp_path), 'a')

    curves = chart_data(entry, 1)

    assert [c.colour for c in curves] == ['grey', 'red']
    assert [c.name for c in curves] == ['Audio track (all channels mixed)',
                                        'Filtered audio track (all channels mixed)']
    numbered = chart_data(replace(entry, audio_stream=2), 1)
    assert [c.name for c in numbered] == ['Audio track 3 (all channels mixed)',
                                          'Filtered audio track 3 (all channels mixed)']
    assert (curves[0].y == 0).all() and curves[1].y.any()    # a flat signal, and what the filter makes of it
    assert len(chart_data(entry, 7)) == 1             # a pick out of range still shows the signal
    assert chart_data(None, 0) == []
    write_entry(str(tmp_path), 'declined', decline=True)
    assert chart_data(read_entry(str(tmp_path), 'declined'), 0) == []


def test_the_state_and_notice_say_what_the_title_is_waiting_for(tmp_path):
    write_entry(str(tmp_path), 'a')
    write_entry(str(tmp_path), 'declined', decline=True)
    entry, declined = read_entry(str(tmp_path), 'a'), read_entry(str(tmp_path), 'declined')
    review = _index_row('review', 'conf 0.90 - 2 candidates', 'pending')
    extract = _index_row('extract', 'new')
    failed = _index_row('attention', 'extract failed: file not found')

    assert state_text(entry, review) == 'Waiting for a decision. conf 0.90 - 2 candidates'
    # the row says pending, the entry has moved on: the row's detail is stale and is left out
    accepted = read_entry(str(tmp_path), 'a')
    accepted.status = 'accepted'
    assert state_text(accepted, review) == 'Accepted'
    assert state_text(None, extract) == 'Extract: new' and state_text(None, None) == ''
    assert notice_text(entry, review, str(tmp_path), '') == ''
    assert notice_text(declined, review, str(tmp_path), '').startswith('Declined: no_rolloff_detected -- nothing found')
    assert 'still has to be extracted' in notice_text(None, extract, str(tmp_path), '')
    assert notice_text(None, failed, str(tmp_path), '') == 'Nothing to review: extract failed: file not found'
    assert 'No review queue directory' in notice_text(None, extract, '', '')
    assert 'could not be read: boom' in notice_text(None, extract, str(tmp_path), 'boom')


# --- opening and closing the page -----------------------------------------------------------------------------------------

def test_double_clicking_a_row_opens_the_page_in_place_of_the_table(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    assert not window.title_page_open and window.title_page is None   # built on first use

    row = window.listed_ids().index('r-arrival')
    index = window.proxy.index(row, 0)
    window.workTable.doubleClicked.emit(index)

    page = window.title_page
    assert window.title_page_open and page.current_id == 'r-arrival'
    assert window.contentStack.currentWidget() is page
    assert not window.listHeader.isVisibleTo(window) and not window.listFooter.isVisibleTo(window)
    assert page.titleLabel.text() == 'Arrival (2001)'
    assert page.subtitleLabel.text() == 'films \u00b7 movie'
    assert page.stateLabel.text() == 'Waiting for a decision. conf 0.90 - 2 candidates'
    assert page.positionLabel.text() == f'{row + 1} of {len(window.listed_ids())}'


def test_enter_on_the_table_opens_the_current_title_but_enter_in_the_search_box_does_not(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    _focus(qtbot, window.searchEdit)
    qtbot.keyClick(window.searchEdit, Qt.Key.Key_Return)
    assert not window.title_page_open

    window.workTable.selectRow(window.listed_ids().index('r-sicario'))
    _focus(qtbot, window.workTable)
    qtbot.keyClick(window.workTable, Qt.Key.Key_Return)

    assert window.title_page_open and window.title_page.current_id == 'r-sicario'


def test_the_open_button_opens_the_selected_title_or_the_first_listed(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    assert window.openButton.isEnabled()

    _click(qtbot, window.openButton)
    assert window.title_page.current_id == window.listed_ids()[0]
    window.close_title()

    window.select_ids(['p-fury'])
    _click(qtbot, window.openButton)
    assert window.title_page.current_id == 'p-fury'


def test_nothing_opens_when_nothing_is_listed(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, rows=[])

    assert not window.openButton.isEnabled()
    assert window.open_title() is False and window.open_title('nope') is False


def test_esc_and_the_breadcrumb_go_back_to_the_table_as_it_was(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    window.searchEdit.setText('a')            # Alien, Arrival, Sicario?, Gravity, ... whatever contains an "a"
    listed = window.listed_ids()
    window.select_ids(listed[:2])
    scroll = window.workTable.verticalScrollBar().value()
    _open(qtbot, window, listed[1])
    window.title_page.next()                  # look at another one before going back
    qtbot.waitUntil(window.title_page.candidateList.hasFocus)

    qtbot.keyClick(window.title_page.candidateList, Qt.Key.Key_Escape)

    assert not window.title_page_open
    assert window.contentStack.currentWidget() is window.tablePage
    assert window.listHeader.isVisibleTo(window) and window.listFooter.isVisibleTo(window)
    assert window.listed_ids() == listed and window.searchEdit.text() == 'a'      # the filters
    assert window.selected_ids() == listed[:2]                                     # the selection is not moved
    assert window.workTable.verticalScrollBar().value() == scroll
    viewed = window.title_page.current_id
    assert window.workTable.currentIndex().data(ID_ROLE) == viewed                 # Enter would open it again

    window.open_title(listed[0])
    _click(qtbot, window.title_page.backButton)
    assert not window.title_page_open


def test_a_refresh_while_the_page_is_open_does_not_put_the_table_back(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    window.open_title('r-alien')

    window.refresh_from_index()

    assert window.title_page_open and window.contentStack.currentWidget() is window.title_page
    window.close_title()
    assert window.contentStack.currentWidget() is window.tablePage


# --- candidates, commentary and the chart ---------------------------------------------------------------------------------

def test_the_page_lists_the_candidates_and_the_commentary_of_the_highlighted_one(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _open(qtbot, window, 'r-alien')

    assert page.candidateList.count() == 2 and page.candidateList.currentRow() == 0    # the top pick
    assert page.candidateList.item(0).text().startswith('1: confidence=0.90 method=fitted')
    assert [(page.commentaryTable.item(r, 0).text(), page.commentaryTable.item(r, 1).text())
            for r in range(page.commentaryTable.rowCount())] == [('note', 'top pick')]

    qtbot.keyClick(page.candidateList, Qt.Key.Key_2)   # digits pick a candidate (1-based, as listed)

    assert page.picked == 1 and page.candidateList.currentRow() == 1
    assert page.commentaryTable.rowCount() == 2 and page.commentaryTable.item(0, 1).text() == 'alternative'
    assert page.pick_candidate(5) is False and page.picked == 1


def test_a_title_that_has_not_been_designed_says_why_and_offers_no_decision(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _open(qtbot, window, 'x-gravity')

    assert page.candidateList.count() == 0
    assert 'still has to be extracted' in page.noticeLabel.text() and page.noticeLabel.isVisibleTo(page)
    assert page.stateLabel.text() == 'Extract: new'
    assert not page.acceptButton.isEnabled() and not page.skipButton.isEnabled() and not page.rejectButton.isEnabled()
    assert page.accept() is False and page.skip() is False and page.reject() is False

    window.title_page.show_title('a-dune')
    assert page.noticeLabel.text() == 'Nothing to review: extract failed: file not found'


def test_a_declined_title_shows_the_reason_and_can_only_be_skipped_or_rejected(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, [('r-alien', {'decline': True}), ('r-arrival', {})])
    page = _open(qtbot, window, 'r-alien')

    assert page.noticeLabel.text().startswith('Declined: no_rolloff_detected -- nothing found')
    assert page.candidateList.count() == 0
    assert not page.acceptButton.isEnabled() and page.skipButton.isEnabled() and page.rejectButton.isEnabled()


def test_an_unreadable_entry_is_reported_and_the_rest_of_the_list_still_works(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, [('r-arrival', {})])
    (tmp_path / 'queue' / 'r-alien.json').write_text('{ not json')

    page = _open(qtbot, window, 'r-alien')

    assert 'could not be read' in page.noticeLabel.text() and not page.acceptButton.isEnabled()
    assert page.next() and page.current_id == 'r-arrival' and page.acceptButton.isEnabled()


def test_an_accepted_title_shows_the_candidate_that_was_chosen_and_offers_no_decision(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _open(qtbot, window, 'p-fury')

    assert page.candidateList.currentRow() == 1 and page.picked == 1
    assert page.decisionLabel.text() == 'Accepted candidate 2.'
    assert page.stateLabel.text() == 'Accepted'    # the row's detail says only that, so it is not said twice
    assert not page.acceptButton.isEnabled() and not page.skipButton.isEnabled() and not page.rejectButton.isEnabled()


# --- deciding ------------------------------------------------------------------------------------------------------------

def test_accept_and_next_accepts_the_highlighted_candidate_and_goes_to_the_next_title_waiting(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _open(qtbot, window, 'r-alien')
    decisions = []
    page.decided.connect(lambda title_id, status: decisions.append((title_id, status)))

    qtbot.keyClick(page.candidateList, Qt.Key.Key_2)
    _click(qtbot, page.acceptButton)

    entry = read_entry(_queue(tmp_path), 'r-alien')
    assert (entry.status, entry.chosen_candidate_index) == ('accepted', 1)
    assert decisions == [('r-alien', 'accepted')]
    assert page.current_id == 'r-arrival' and page.picked == 0        # the next one, from its top pick
    assert page.decisionLabel.text() == '2 in this list waiting for a decision.'   # arrival, sicario


def test_enter_on_the_candidate_list_accepts_and_a_letter_does_too(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _open(qtbot, window, 'r-alien')

    _focus(qtbot, page.candidateList)
    qtbot.keyClick(page.candidateList, Qt.Key.Key_Return)
    assert _status(tmp_path, 'r-alien') == 'accepted' and page.current_id == 'r-arrival'

    qtbot.keyClick(page.candidateList, Qt.Key.Key_A)
    assert _status(tmp_path, 'r-arrival') == 'accepted' and page.current_id == 'r-sicario'


def test_enter_elsewhere_on_the_page_decides_nothing(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _open(qtbot, window, 'r-alien')

    _focus(qtbot, page.commentaryTable)
    qtbot.keyClick(page.commentaryTable, Qt.Key.Key_Return)
    qtbot.keyClick(page.commentaryTable, Qt.Key.Key_Enter)

    assert _status(tmp_path, 'r-alien') == 'pending' and page.current_id == 'r-alien'


def test_skip_and_reject_decide_and_advance_like_accept(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _open(qtbot, window, 'r-alien')

    _click(qtbot, page.skipButton)
    assert _status(tmp_path, 'r-alien') == 'skipped' and page.current_id == 'r-arrival'
    qtbot.keyClick(page.candidateList, Qt.Key.Key_R)
    assert _status(tmp_path, 'r-arrival') == 'rejected' and page.current_id == 'r-sicario'


def test_a_skipped_title_can_still_be_accepted_or_rejected_but_not_skipped_again(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, [('r-alien', {'status': 'skipped'}), ('r-arrival', {})])
    page = _open(qtbot, window, 'r-alien')

    assert page.acceptButton.isEnabled() and page.rejectButton.isEnabled() and not page.skipButton.isEnabled()
    assert page.accept() and _status(tmp_path, 'r-alien') == 'accepted'


def test_the_next_title_waiting_skips_what_is_decided_and_wraps_round(qtbot, tmp_path):
    ''' From the last waiting title the page goes back up the list, to the one above it that is still waiting. '''
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _open(qtbot, window, 'r-sicario')

    assert page.accept()

    assert page.current_id == 'r-alien'       # nothing waiting below r-sicario: wrapped to the top
    assert page.accept() and page.current_id == 'r-arrival'


def test_with_nothing_else_waiting_the_page_stays_and_says_so(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, [('r-alien', {}), ('p-fury', {'status': 'accepted'})])
    page = _open(qtbot, window, 'r-alien')

    assert page.accept()

    assert page.current_id == 'r-alien'
    assert page.decisionLabel.text() == 'No other title in this list is waiting for a decision.'
    assert page.stateLabel.text() == 'Accepted' and not page.acceptButton.isEnabled()
    assert page.accept() is False                                  # nothing to decide twice


def test_a_title_decided_elsewhere_is_not_decided_again_and_the_page_shows_what_is_there_now(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _open(qtbot, window, 'r-alien')
    write_entry(_queue(tmp_path), 'r-alien', status='accepted', chosen=0)   # a publish or another window got there first

    assert page.reject() is False

    assert _status(tmp_path, 'r-alien') == 'accepted'                          # untouched
    assert 'Not changed: this title is accepted now.' == page.decisionLabel.text()
    assert page.stateLabel.text().startswith('Accepted') and not page.rejectButton.isEnabled()


def test_a_design_that_changed_under_the_page_is_not_accepted_blind(qtbot, tmp_path):
    ''' The person chose candidate 2 of a design that has since been redone with one candidate: nothing is written. '''
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _open(qtbot, window, 'r-alien')
    page.pick_candidate(1)
    write_entry(_queue(tmp_path), 'r-alien', count=1)

    assert page.accept() is False

    assert _status(tmp_path, 'r-alien') == 'pending'
    assert page.decisionLabel.text().startswith('Not accepted: the design of this title changed')
    assert page.candidateList.count() == 1                                    # reloaded: what is there now


def test_a_redesign_with_the_same_number_of_candidates_is_not_accepted_blind_either(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _open(qtbot, window, 'r-alien')
    page.pick_candidate(0)
    write_entry(_queue(tmp_path), 'r-alien', reverse=True)    # two candidates still, but candidate 1 is a different filter

    assert page.accept() is False

    assert _status(tmp_path, 'r-alien') == 'pending'
    assert page.decisionLabel.text().startswith('Not accepted: the design of this title changed')
    assert page.candidateList.item(0).text().startswith('1: confidence=0.40')     # what is there now


def test_a_write_that_fails_is_reported_and_leaves_the_page_where_it_is(qtbot, tmp_path, monkeypatch):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _open(qtbot, window, 'r-alien')
    decisions = []
    page.decided.connect(lambda *args: decisions.append(args))
    monkeypatch.setattr('model.worklist_title_decide.update_entry', lambda *a, **k: (_ for _ in ()).throw(OSError('disk full')))

    assert page.accept() is False

    assert page.current_id == 'r-alien' and decisions == []
    assert page.decisionLabel.text() == 'Not saved: OSError: disk full'
    assert _status(tmp_path, 'r-alien') == 'pending'


def _set_detail(tmp_path, title_id, detail):
    import sqlite3
    from pipeline.library.index import index_path
    db = sqlite3.connect(index_path(str(tmp_path / 'work')))
    with db:
        db.execute('UPDATE titles SET detail = ? WHERE id = ?', (detail, title_id))
    db.close()


def test_a_refresh_while_the_page_is_open_updates_its_header_and_keeps_the_highlighted_candidate(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _open(qtbot, window, 'r-alien')
    page.pick_candidate(1)
    _set_detail(tmp_path, 'r-alien', 'conf 0.90 - metadata incomplete: year')

    window.refresh_from_index()

    assert page.stateLabel.text() == 'Waiting for a decision. conf 0.90 - metadata incomplete: year'
    assert page.picked == 1 and page.candidateList.currentRow() == 1


def test_a_refresh_that_finds_a_redesign_goes_back_to_the_top_pick(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _open(qtbot, window, 'r-alien')
    page.pick_candidate(1)
    write_entry(_queue(tmp_path), 'r-alien', reverse=True)

    window.refresh_from_index()

    assert page.picked == 0 and page.candidateList.item(0).text().startswith('1: confidence=0.40')


# --- review findings: what is offered, and what a held key or a run may do -------------------------------------------------------

def _stale_rows():
    ''' A pending entry whose design is out of date (needs design) between two that are to review. '''
    rows = _rows()
    rows.insert(2, _row('s-stale', 'Stale', 'design', 4, detail='source or settings changed', review_state='pending'))
    return rows


STALE = [('r-alien', {}), ('s-stale', {}), ('r-arrival', {})]


def test_accept_and_next_skips_a_pending_title_whose_design_is_out_of_date(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, STALE, rows=_stale_rows())
    page = _open(qtbot, window, 'r-alien')

    assert page.decisionLabel.text() == '3 in this list waiting for a decision.'    # alien, arrival, sicario: not the stale one
    assert page.accept()

    assert page.current_id == 'r-arrival'


def test_a_pending_title_whose_design_is_out_of_date_cannot_be_accepted_but_can_be_rejected(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, STALE, rows=_stale_rows())
    page = _open(qtbot, window, 's-stale')

    assert not page.acceptButton.isEnabled() and page.rejectButton.isEnabled() and page.skipButton.isEnabled()
    assert 'needs design first' in page.decisionLabel.text()
    assert page.accept() is False and _status(tmp_path, 's-stale') == 'pending'
    assert page.reject() and _status(tmp_path, 's-stale') == 'rejected'


def test_no_decision_shortcut_repeats_when_a_key_is_held(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _open(qtbot, window, 'r-alien')
    keys = {}
    for shortcut in page.findChildren(QShortcut) + page.candidateList.findChildren(QShortcut):
        keys[shortcut.key().toString()] = shortcut.autoRepeat()

    assert {k: keys[k] for k in ('A', 'S', 'R', 'Return', 'Enter')} == dict.fromkeys(('A', 'S', 'R', 'Return', 'Enter'), False)


def test_the_failures_and_last_run_tabs_stay_hidden_while_the_page_is_open(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    _open(qtbot, window, 'r-alien')
    window._failed = [SimpleNamespace(id='x')]     # what makes `_refresh_details` want the panel

    window._refresh_details()
    assert not window.detailsTabs.isVisibleTo(window)

    window.close_title()
    assert window.detailsTabs.isVisibleTo(window)


def test_nothing_is_decided_on_a_title_a_run_is_working_on(qtbot, tmp_path):
    ''' A design in flight writes a new pending entry over whatever is there: an accept now would be lost. '''
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _open(qtbot, window, 'r-alien')

    window.model.set_running({'r-alien': 'design'})

    assert not page.acceptButton.isEnabled() and not page.skipButton.isEnabled() and not page.rejectButton.isEnabled()
    assert page.decisionLabel.text() == 'A run is working on this title now: wait for it to finish.'
    assert page.accept() is False and _status(tmp_path, 'r-alien') == 'pending'

    window.model.set_running({})
    assert page.acceptButton.isEnabled() and page.accept()


def test_skip_and_reject_are_not_applied_to_a_design_that_changed_either(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _open(qtbot, window, 'r-alien')
    write_entry(_queue(tmp_path), 'r-alien', reverse=True)

    assert page.reject() is False
    assert page.decisionLabel.text().startswith('Not rejected: the design of this title changed')
    write_entry(_queue(tmp_path), 'r-alien')
    page.reload()
    write_entry(_queue(tmp_path), 'r-alien', reverse=True)
    assert page.skip() is False and _status(tmp_path, 'r-alien') == 'pending'
    assert page.decisionLabel.text().startswith('Not skipped: the design of this title changed')


def test_a_decision_made_while_the_index_is_read_and_the_window_closed_is_still_read_afterwards(qtbot, tmp_path, monkeypatch):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    calls, gate = [], threading.Event()

    def refresh(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            gate.wait(10)

    monkeypatch.setattr(LibraryIndex, 'refresh', refresh)
    page = _open(qtbot, window, 'r-alien')
    assert page.accept()
    window.close_title()
    qtbot.waitUntil(lambda: len(calls) == 1)
    assert window._syncing

    window._on_title_decided('r-arrival', 'accepted')    # one more, while the first read is going
    window.close()                                       # ... and the window goes
    assert window._index_dirty and len(calls) == 1
    gate.set()

    qtbot.waitUntil(lambda: len(calls) == 2 and not window._syncing, timeout=10000)
    assert not window._index_dirty


def test_closing_during_a_scan_leaves_the_read_to_when_it_ends_and_a_failed_read_stays_dirty(qtbot, tmp_path, monkeypatch):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    calls = []
    monkeypatch.setattr(LibraryIndex, 'refresh', lambda self, *a, **k: calls.append(1))
    page = _open(qtbot, window, 'r-alien')
    page.accept()
    window._scanning = True

    window.close()

    assert calls == [] and window._index_dirty        # not a second writer on the UI thread

    window._scanning = False
    window.show()                                     # showing again reads the setup and opens the index
    monkeypatch.setattr(LibraryIndex, 'refresh', lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError('locked')))
    window._sync_index_now()
    assert window._index_dirty                        # the failed read is tried again later


def test_no_index_file_is_created_by_reading_after_a_decision(qtbot, tmp_path):
    from pipeline.library.index import index_path
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _open(qtbot, window, 'r-alien')
    assert page.accept()
    path = index_path(str(tmp_path / 'work'))
    os.remove(path)                                   # as if the work directory now held no index

    window.close_title()
    qtbot.wait(200)

    assert not os.path.exists(path) and not window._syncing and not window._index_dirty


def test_a_rescan_does_not_start_while_the_index_is_being_read(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    window._syncing = True

    assert window.rescan() is False


def test_decisions_are_remembered_until_the_index_has_read_them(qtbot, tmp_path, monkeypatch):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    monkeypatch.setattr(LibraryIndex, 'refresh', lambda self, *a, **k: None)
    page = _open(qtbot, window, 'r-alien')
    assert page.accept()

    window.refresh_from_index()                       # rows re-read, but the index has not read the decision yet
    assert page._decided_here == {'r-alien'}

    with qtbot.waitSignal(window.index_synced, timeout=10000):
        window.close_title()
    assert page._decided_here == set()


# --- moving through the list ------------------------------------------------------------------------------------------------

def test_previous_and_next_walk_the_list_the_page_was_opened_over(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    ids = window.listed_ids()
    page = _open(qtbot, window, ids[0])

    assert page.ids == ids
    assert not page.previousButton.isEnabled() and page.nextButton.isEnabled()
    assert page.positionLabel.text() == f'1 of {len(ids)}'

    _click(qtbot, page.nextButton)
    assert page.current_id == ids[1] and page.positionLabel.text() == f'2 of {len(ids)}'
    qtbot.keyClick(page.candidateList, Qt.Key.Key_Right, Qt.KeyboardModifier.AltModifier)
    assert page.current_id == ids[2]
    qtbot.keyClick(page.candidateList, Qt.Key.Key_Left, Qt.KeyboardModifier.AltModifier)
    _click(qtbot, page.previousButton)
    assert page.current_id == ids[0] and page.previous() is False

    page.show_title(ids[-1])
    assert not page.nextButton.isEnabled() and page.next() is False


def test_the_list_is_the_one_the_filters_showed_when_the_page_was_opened(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    window.set_chip('Review')
    window.open_title('r-arrival')

    assert window.title_page.ids == ['r-alien', 'r-arrival', 'r-sicario']
    assert window.title_page.positionLabel.text() == '2 of 3'


def test_accept_and_next_goes_only_to_titles_in_the_list(qtbot, tmp_path):
    ''' The search box leaves only Alien listed; Arrival and Sicario are waiting too, but are not in the list. '''
    window = _window(qtbot, tmp_path, REVIEWABLE)
    window.searchEdit.setText('Alien')
    window.open_title('r-alien')
    assert window.title_page.ids == ['r-alien']

    assert window.title_page.accept()

    assert window.title_page.current_id == 'r-alien' and _status(tmp_path, 'r-arrival') == 'pending'


# --- the index, after decisions -------------------------------------------------------------------------------------------

def test_the_index_is_read_again_once_when_the_page_is_left_not_after_each_decision(qtbot, tmp_path, monkeypatch):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    calls = []
    monkeypatch.setattr(LibraryIndex, 'refresh', lambda self, *a, **k: calls.append(a))
    page = _open(qtbot, window, 'r-alien')

    assert page.accept() and page.accept()
    assert calls == [] and not window._syncing

    with qtbot.waitSignal(window.index_synced, timeout=10000):
        window.close_title()

    assert len(calls) == 1 and not window._syncing
    assert calls[0][0] is window.setup.profile and calls[0][1] is window.setup.settings


def test_leaving_the_page_without_deciding_reads_nothing(qtbot, tmp_path, monkeypatch):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    calls = []
    monkeypatch.setattr(LibraryIndex, 'refresh', lambda self, *a, **k: calls.append(a))
    window.open_title('r-alien')
    window.title_page.next()

    window.close_title()
    qtbot.wait(100)

    assert calls == [] and not window._syncing


def test_the_read_waits_for_a_scan_and_is_done_when_it_ends(qtbot, tmp_path, monkeypatch):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    calls = []
    monkeypatch.setattr(LibraryIndex, 'refresh', lambda self, *a, **k: calls.append(a))
    window.open_title('r-alien')
    window.title_page.accept()
    window._scanning = True            # a scan is going: it reads the outputs itself, so nothing else writes the index

    window.close_title()
    qtbot.wait(100)
    assert calls == [] and window._index_dirty

    window._scanning = False
    with qtbot.waitSignal(window.index_synced, timeout=10000):
        assert window._sync_index_if_dirty()

    assert len(calls) == 1 and not window._index_dirty


def test_buttons_wait_while_the_index_is_being_read_and_a_failed_read_is_reported(qtbot, tmp_path, monkeypatch):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    monkeypatch.setattr(LibraryIndex, 'refresh', lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError('locked')))
    window.open_title('r-alien')
    window.title_page.accept()

    window.close_title()

    assert window._syncing and not window.rescanButton.isEnabled()      # not while it is going
    qtbot.waitUntil(lambda: not window._syncing, timeout=10000)
    assert window._index_dirty                                          # the decision is still not in the index
    assert 'may be out of date' in window.statusBar.currentMessage() and 'locked' in window.statusBar.currentMessage()
    assert window.rescanButton.isEnabled()


def test_closing_the_window_on_the_page_reads_the_index_now(qtbot, tmp_path, monkeypatch):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    calls = []
    monkeypatch.setattr(LibraryIndex, 'refresh', lambda self, *a, **k: calls.append(a))
    window.open_title('r-alien')
    window.title_page.accept()

    window.close()

    assert len(calls) == 1 and not window.title_page_open


# --- through a real scan -----------------------------------------------------------------------------------------------------

def _world(tmp_path, *names):
    world = SimpleNamespace(work=str(tmp_path / 'work'), queue=str(tmp_path / 'queue'))
    world.items = [_item(name) for name in names]
    for item in world.items:
        _extracted(world, item)
        _real_entry(world, item, status='pending', candidates=1)
    return world


def test_accepting_on_the_page_moves_the_titles_from_review_to_publish_once_the_page_is_left(qtbot, tmp_path):
    world = _world(tmp_path, 'a', 'b')
    window = WorkListWindow(None, _prefs(tmp_path), sources={'filesystem': FakeSource(world.items)}, auto_scan=False,
                            clock=time.time)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    window.activateWindow()
    qtbot.waitActive(window)
    with qtbot.waitSignal(window.scan_finished, timeout=30000):
        window.rescan()
    needs = lambda: {row.id: row.needs for row in window.model.rows}   # noqa: E731
    assert needs() == {'fs-a': 'review', 'fs-b': 'review'}

    page = _open(qtbot, window, 'fs-a')
    assert page.candidateList.count() == 1 and page.titleLabel.text() == 'Film a (2018)'
    assert page.accept() and page.current_id == 'fs-b'
    assert needs() == {'fs-a': 'review', 'fs-b': 'review'}      # the index has not read it yet
    assert page.accept() and page.current_id == 'fs-b'          # the last: nothing else waiting

    with qtbot.waitSignal(window.index_synced, timeout=30000):
        _click(qtbot, page.backButton)

    assert needs() == {'fs-a': 'publish', 'fs-b': 'publish'}
    assert window.publishButton.text() == 'Publish 2'
    assert QApplication.activeModalWidget() is None
