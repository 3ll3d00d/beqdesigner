'''
model/worklist_bulk.py, chunk 27c: the bulk-accept UI (threshold from the preference, a confirmation that names the count and lists
what is left out and why, nothing accepted without it), and the "settings changed since N titles were designed" banner over
`pipeline.library.drift`. Fixture-index tests for the words and the interlocks, and REAL SCANS for what matters: bulk accept moves
titles from Review to Publish, and the banner counts the accepted titles designed under another designer and offers Redesign.
`import ui.beq` first: see AGENTS.md gotcha 3.
'''
import ui.beq  # noqa: F401 (must come first)

import time
from types import SimpleNamespace

import pytest
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QApplication

from model.preferences import DESIGNER_DEFAULT, WORKLIST_ACCEPT_THRESHOLD
from model.worklist import WorkListWindow
from model.worklist_confirm import ConfirmDialog, accept_text, drift_text
from pipeline.designer.registry import register_designer, unregister_designer
from pipeline.library.bulk import AcceptPlan, Exclusion
from pipeline.review import read_entry, read_queue, update_entry, write_queue_entry
from test_pipeline_library_index import DESIGNER, FakeSource, _entry as _real_entry, _extracted, _item
from test_worklist_revise import _Ask
from test_worklist_title import NOW, SOURCES, _designer, _prefs, _queue, _row  # noqa: F401
from worklist_fixture import make_index
from worklist_project_fixture import edit_project, write_projects
from worklist_title_fixture import write_entry


_TIMERS = []


def _answer(accept: bool, seen: list):
    ''' Answers the confirmation when it appears: the plan is worked out on a worker first, so it is polled for. '''
    timer = QTimer()
    timer.setInterval(20)
    _TIMERS.append(timer)

    def respond():
        dialog = QApplication.activeModalWidget()
        if not isinstance(dialog, ConfirmDialog):
            return
        timer.stop()
        seen.append(dialog)
        (dialog.ok_button if accept else dialog.cancel_button).click()

    timer.timeout.connect(respond)
    timer.start()


def _set_confidence(tmp_path, entry_id, confidence):
    entry = read_entry(_queue(tmp_path), entry_id)
    entry.candidates[0].confidence = confidence
    write_queue_entry(_queue(tmp_path), entry)


def _fixture_window(qtbot, tmp_path, threshold=None):
    ''' Six titles waiting for review: two confident, one not, a decline, incomplete metadata and an edited project. '''
    prefs = _prefs(tmp_path)
    if threshold is not None:
        prefs.set(WORKLIST_ACCEPT_THRESHOLD, threshold)
    rows = [_row(i, t, 'review', d, review_state='pending', confidence=c, detail='review')
            for i, t, d, c in (('r-high1', 'High One', 9, 0.95), ('r-high2', 'High Two', 8, 0.92), ('r-low', 'Low', 7, 0.6),
                               ('r-decline', 'Declined', 6, None), ('r-nometa', 'No Metadata', 5, 0.95),
                               ('r-edited', 'Edited', 4, 0.95))]
    rows.append(_row('p-done', 'Done', 'publish', 3, review_state='accepted', publish_state='not_written'))
    make_index(tmp_path / 'work', rows, SOURCES, generation=2, last_scan_at=NOW - 900)
    queue = str(tmp_path / 'queue')
    for entry_id in ('r-high1', 'r-high2', 'r-edited'):
        write_entry(queue, entry_id)
    write_entry(queue, 'r-low')
    _set_confidence(tmp_path, 'r-low', 0.6)
    write_entry(queue, 'r-decline', decline=True)
    write_entry(queue, 'r-nometa', meta={'title': 'No Metadata'})
    write_entry(queue, 'p-done', status='accepted')
    edit_project(write_projects(tmp_path / 'work', 'r-edited')[0])
    window = WorkListWindow(None, prefs, auto_scan=False, clock=lambda: NOW)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    return window


# --- the words --------------------------------------------------------------------------------------------------------------------

def test_the_accept_confirmation_says_how_many_at_what_confidence_and_lists_what_is_left_out():
    plan = AcceptPlan(0.9, ['a', 'b'], [Exclusion('c', 'Cee', 'designer declined: nothing found')], below_threshold=3, not_for_review=2)
    heading, body, details = accept_text(plan)
    assert heading == 'Accept the top pick for 2 titles?'
    assert '0.90 or more' in body and 'bulk accepted, confidence &gt;= 0.90' in body
    assert 'Nothing is published' in body and 'left out' in body
    assert '3 titles waiting for review are below the threshold' in body and '2 titles in the selection are not waiting' in body
    assert details == 'Cee: designer declined: nothing found'
    assert accept_text(AcceptPlan(0.9, ['a']))[0] == 'Accept the top pick for 1 title?' and accept_text(AcceptPlan(0.9, ['a']))[2] == ''


def test_the_banner_text():
    assert drift_text(1).startswith('The settings changed since 1 accepted or published title was designed.')
    assert 'They keep the old design until you revise them' in drift_text(5)


# --- the button and the confirmation ----------------------------------------------------------------------------------------------

def test_the_button_counts_the_titles_at_or_above_the_threshold_in_the_selection_or_the_view(qtbot, tmp_path):
    window = _fixture_window(qtbot, tmp_path)
    assert window.acceptButton.text() == 'Accept top pick (4)'          # high1, high2, nometa, edited: 0.95, 0.92, 0.95, 0.95
    window.select_ids(['r-high1', 'r-low'])
    assert window.acceptButton.text() == 'Accept top pick (1)' and window.acceptButton.isEnabled()
    window.select_ids(['r-low'])
    assert window.acceptButton.text() == 'Accept top pick (0)' and not window.acceptButton.isEnabled()


def test_the_threshold_is_the_preference(qtbot, tmp_path):
    window = _fixture_window(qtbot, tmp_path, threshold=0.94)
    assert window.accept_threshold() == 0.94
    assert window.acceptButton.text() == 'Accept top pick (3)'
    assert '0.94' in window.acceptButton.toolTip()


def test_the_confirmation_names_the_count_the_threshold_and_every_title_left_out_with_its_reason(qtbot, tmp_path):
    window = _fixture_window(qtbot, tmp_path)
    seen = []
    _answer(False, seen)
    assert window.accept_selected() is True
    qtbot.waitUntil(lambda: bool(seen), timeout=30000)
    qtbot.waitUntil(lambda: not window.bulk_running and QApplication.activeModalWidget() is None, timeout=30000)
    text = seen[0].text
    assert seen[0].windowTitle() == 'Accept the top pick for 2 titles?'
    assert '0.90 or more' in text
    assert 'Declined: designer declined: nothing found' in text
    assert 'No Metadata: metadata incomplete' in text
    assert 'Edited: the mono project was edited since it was designed' in text
    assert '1 title waiting for review is below the threshold' in text


def test_declining_the_confirmation_accepts_nothing(qtbot, tmp_path):
    window = _fixture_window(qtbot, tmp_path)
    seen = []
    _answer(False, seen)
    window.accept_selected()
    qtbot.waitUntil(lambda: bool(seen) and not window.bulk_running, timeout=30000)
    assert all(e.status == 'pending' for e in read_queue(_queue(tmp_path)) if e.id.startswith('r-'))
    assert 'Cancelled: nothing was accepted' in window.runStatusLabel.text()
    assert all(e.reviewer_note is None for e in read_queue(_queue(tmp_path)))


def test_confirming_accepts_the_top_pick_of_the_eligible_titles_only_with_the_note_and_lists_the_rest(qtbot, tmp_path):
    window = _fixture_window(qtbot, tmp_path)
    seen = []
    _answer(True, seen)
    window.accept_selected()
    qtbot.waitUntil(lambda: bool(seen), timeout=30000)
    qtbot.waitUntil(lambda: not window.bulk_running and window.runStatusLabel.text().startswith('Accepted'), timeout=30000)

    entries = {e.id: e for e in read_queue(_queue(tmp_path))}
    for accepted in ('r-high1', 'r-high2'):
        assert entries[accepted].status == 'accepted' and entries[accepted].chosen_candidate_index == 0
        assert entries[accepted].reviewer_note == 'bulk accepted, confidence >= 0.90'
    for left in ('r-low', 'r-decline', 'r-nometa', 'r-edited'):
        assert entries[left].status == 'pending' and entries[left].reviewer_note is None
    assert window.runStatusLabel.text().startswith('Accepted the top pick for 2 titles (confidence 0.90 or more).')
    assert '3 left out' in window.runStatusLabel.text() and 'They now need Publish.' in window.runStatusLabel.text()
    outcomes = {l.title: l.outcome for l in window.results}
    assert outcomes['High One'] == outcomes['High Two'] == 'Accepted'
    assert outcomes['Edited'] == outcomes['No Metadata'] == outcomes['Declined'] == 'Not accepted'
    assert 'Low' not in outcomes


def test_only_the_selected_rows_are_considered(qtbot, tmp_path):
    window = _fixture_window(qtbot, tmp_path)
    window.select_ids(['r-high1'])
    seen = []
    _answer(True, seen)
    window.accept_selected()
    qtbot.waitUntil(lambda: bool(seen), timeout=30000)
    qtbot.waitUntil(lambda: not window.bulk_running, timeout=30000)
    assert seen[0].windowTitle() == 'Accept the top pick for 1 title?'
    statuses = {e.id: e.status for e in read_queue(_queue(tmp_path))}
    assert statuses['r-high1'] == 'accepted' and statuses['r-high2'] == 'pending'


def test_with_nothing_eligible_no_confirmation_is_offered_and_the_reasons_are_listed(qtbot, tmp_path):
    window = _fixture_window(qtbot, tmp_path)
    assert window.accept_selected(['r-decline', 'r-nometa', 'r-low'])
    qtbot.waitUntil(lambda: not window.bulk_running, timeout=30000)
    assert QApplication.activeModalWidget() is None
    assert window.runStatusLabel.text().startswith('Nothing to accept')
    assert '2 at or above the threshold need a person to look' in window.runStatusLabel.text()
    assert {l.title for l in window.results} == {'Declined', 'No Metadata'}
    assert all(e.status == 'pending' for e in read_queue(_queue(tmp_path)) if e.id.startswith('r-'))


def test_nothing_is_started_while_something_else_is_going_or_with_no_titles(qtbot, tmp_path):
    window = _fixture_window(qtbot, tmp_path)
    window._scanning = True
    try:
        assert window.accept_selected() is False
    finally:
        window._scanning = False
    assert window.accept_selected([]) is False           # (an empty selection would mean every title: refused)
    assert all(e.status == 'pending' for e in read_queue(_queue(tmp_path)) if e.id.startswith('r-'))


# --- through a real scan ----------------------------------------------------------------------------------------------------------

def _scanned(qtbot, tmp_path, confidences, status='pending', prefs=None, **options):
    world = SimpleNamespace(work=str(tmp_path / 'work'), queue=str(tmp_path / 'queue'))
    world.items = [_item(name) for name in confidences]
    for item, confidence in zip(world.items, confidences.values()):
        _extracted(world, item)
        _real_entry(world, item, status=status, confidence=confidence)
    window = WorkListWindow(None, prefs or _prefs(tmp_path), sources={'filesystem': FakeSource(world.items)}, auto_scan=False,
                            clock=time.time, **options)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    with qtbot.waitSignal(window.scan_finished, timeout=30000):
        window.rescan()
    return window


def _needs(window):
    return {row.id: row.needs for row in window.model.rows}


def test_bulk_accept_moves_the_confident_titles_from_review_to_publish_in_a_real_scan(qtbot, tmp_path):
    window = _scanned(qtbot, tmp_path, {'a': 0.95, 'b': 0.93, 'c': 0.5})
    assert _needs(window) == {'fs-a': 'review', 'fs-b': 'review', 'fs-c': 'review'}
    edit = write_projects(tmp_path / 'work', 'fs-b')[0]
    edit_project(edit)

    seen = []
    _answer(True, seen)
    window.accept_selected()
    qtbot.waitUntil(lambda: bool(seen), timeout=30000)
    qtbot.waitUntil(lambda: not window.bulk_running and window.runStatusLabel.text().startswith('Accepted'), timeout=30000)

    assert seen[0].windowTitle() == 'Accept the top pick for 1 title?'
    assert 'Film b: the mono project was edited' in seen[0].text
    assert _needs(window) == {'fs-a': 'publish', 'fs-b': 'review', 'fs-c': 'review'}      # the index was read again
    assert window.publishButton.text() == 'Publish 1'
    assert read_entry(_queue(tmp_path), 'fs-a').reviewer_note == 'bulk accepted, confidence >= 0.90'
    assert read_entry(_queue(tmp_path), 'fs-b').status == 'pending'


def test_a_lower_threshold_in_the_preferences_accepts_more(qtbot, tmp_path):
    prefs = _prefs(tmp_path)
    prefs.set(WORKLIST_ACCEPT_THRESHOLD, 0.4)
    window = _scanned(qtbot, tmp_path, {'a': 0.95, 'b': 0.5}, prefs=prefs)
    seen = []
    _answer(True, seen)
    window.accept_selected()
    qtbot.waitUntil(lambda: bool(seen), timeout=30000)
    qtbot.waitUntil(lambda: not window.bulk_running, timeout=30000)
    assert _needs(window) == {'fs-a': 'publish', 'fs-b': 'publish'}
    assert read_entry(_queue(tmp_path), 'fs-b').reviewer_note == 'bulk accepted, confidence >= 0.40'


# --- the "settings changed" banner --------------------------------------------------------------------------------------------------

def _other_designer(tmp_path, window):
    ''' The person picks another designer in the settings: the window reads them again. '''
    register_designer('another.designer', lambda request: None)
    window._preferences.set(DESIGNER_DEFAULT, 'another.designer')
    window.reload()


@pytest.fixture
def other_designer():
    yield
    unregister_designer('another.designer')


def test_the_banner_counts_the_accepted_titles_designed_under_another_designer_and_offers_redesign(
        qtbot, tmp_path, other_designer):
    ask = _Ask('design', 'new designer')
    window = _scanned(qtbot, tmp_path, {'a': 0.9, 'b': 0.9}, status='accepted', ask_revise=ask)
    qtbot.wait(200)
    assert not window.driftBanner.isVisibleTo(window) and window.drift_ids == []      # nothing changed yet

    _other_designer(tmp_path, window)
    qtbot.waitUntil(lambda: window.driftBanner.isVisibleTo(window), timeout=30000)

    assert window.drift_ids == ['fs-a', 'fs-b']
    assert window.driftBannerLabel.text().startswith('The settings changed since 2 accepted or published titles were designed.')
    assert _needs(window) == {'fs-a': 'publish', 'fs-b': 'publish'}                 # they are NOT put back in the list by it

    assert window.driftBannerButton.isEnabled()
    window.revise_drifted()
    qtbot.waitUntil(lambda: not window.bulk_running, timeout=30000)

    summary, _, default = ask.asked[0]
    assert default == 'design' and summary.count == 2 and summary.accepted == 2
    assert _needs(window) == {'fs-a': 'design', 'fs-b': 'design'}
    assert read_entry(_queue(tmp_path), 'fs-a').reviewer_note == 'Sent back for redesign: new designer'
    qtbot.waitUntil(lambda: not window.driftBanner.isVisibleTo(window), timeout=30000)      # nothing accepted is out of step now


def test_dismissing_hides_the_banner_until_the_set_of_titles_changes(qtbot, tmp_path, other_designer):
    window = _scanned(qtbot, tmp_path, {'a': 0.9, 'b': 0.9}, status='accepted')
    _other_designer(tmp_path, window)
    qtbot.waitUntil(lambda: window.driftBanner.isVisibleTo(window), timeout=30000)
    _dismiss = window.driftDismissButton
    _dismiss.click()
    assert not window.driftBanner.isVisibleTo(window)
    window.refresh_from_index()                                   # asked again, same answer: stays hidden
    qtbot.wait(300)
    assert not window.driftBanner.isVisibleTo(window) and window.drift_ids == ['fs-a', 'fs-b']
    window._set_drift(['fs-a'])                                   # a different set: shown again
    assert window.driftBanner.isVisibleTo(window)


def test_an_answer_to_an_older_question_is_dropped(qtbot, tmp_path):
    window = _scanned(qtbot, tmp_path, {'a': 0.9}, status='accepted')
    window._set_drift([])
    window._drift_token += 5
    window._on_drift((window._drift_token - 1, ['fs-a']))
    assert window.drift_ids == []
    window._on_drift((window._drift_token, ['fs-a']))
    assert window.drift_ids == ['fs-a']


def test_a_title_that_only_needs_review_is_never_in_the_banner(qtbot, tmp_path, other_designer):
    window = _scanned(qtbot, tmp_path, {'a': 0.9})                     # pending: a run redesigns it anyway
    _other_designer(tmp_path, window)
    qtbot.wait(300)
    assert window.drift_ids == [] and not window.driftBanner.isVisibleTo(window)


# --- the independent review of chunk 27c (2026-09-21) ---------------------------------------------------------------------------------------

def _answer_after(before, seen):
    ''' As `_answer(True, ...)`, but `before()` happens first: the world changes while the person is deciding. '''
    timer = QTimer()
    timer.setInterval(20)
    _TIMERS.append(timer)

    def respond():
        dialog = QApplication.activeModalWidget()
        if not isinstance(dialog, ConfirmDialog):
            return
        timer.stop()
        seen.append(dialog)
        before()
        dialog.ok_button.click()

    timer.timeout.connect(respond)
    timer.start()


def _index_update(tmp_path, sql, *args):
    import sqlite3
    from pipeline.library.index import index_path
    connection = sqlite3.connect(index_path(str(tmp_path / 'work')))
    try:
        connection.execute(sql, args)
        connection.commit()
    finally:
        connection.close()


def _run_accept(qtbot, window, before):
    seen = []
    _answer_after(before, seen)
    window.accept_selected()
    qtbot.waitUntil(lambda: bool(seen), timeout=30000)
    qtbot.waitUntil(lambda: not window.bulk_running, timeout=30000)     # (the accept job starts before the slot that asked returns)
    return seen


def test_a_title_the_second_plan_drops_as_no_longer_waiting_is_reported_not_silently_left_out(qtbot, tmp_path):
    ''' The second job plans again, so it accepts a subset of what was confirmed; the rest is accounted for. '''
    window = _fixture_window(qtbot, tmp_path)
    _run_accept(qtbot, window, lambda: _index_update(tmp_path, "UPDATE titles SET needs = 'publish' WHERE id = 'r-high2'"))

    statuses = {e.id: e.status for e in read_queue(_queue(tmp_path))}
    assert statuses['r-high1'] == 'accepted' and statuses['r-high2'] == 'pending'
    text = window.runStatusLabel.text()
    assert text.startswith('Accepted the top pick for 1 title (confidence 0.90 or more).')
    assert '1 of the 2 you confirmed was not accepted (1 no longer waiting for review)' in text
    lines = {l.id: (l.outcome, l.detail) for l in window.results}
    assert lines['r-high2'][0] == 'Not accepted' and 'no longer waiting for review' in lines['r-high2'][1]
    assert lines['r-high1'][0] == 'Accepted'


def test_a_title_that_changed_after_the_confirmation_is_counted_with_its_reason(qtbot, tmp_path):
    window = _fixture_window(qtbot, tmp_path)
    _run_accept(qtbot, window, lambda: update_entry(_queue(tmp_path), 'r-high1', status='skipped'))

    text = window.runStatusLabel.text()
    assert '1 of the 2 you confirmed was not accepted (1 changed after you confirmed)' in text
    lines = {l.id: (l.outcome, l.detail) for l in window.results}
    assert lines['r-high1'] == ('Not accepted', 'already skipped') and lines['r-high2'][0] == 'Accepted'


def test_every_confirmed_title_accepted_says_nothing_of_the_kind(qtbot, tmp_path):
    window = _fixture_window(qtbot, tmp_path)
    _run_accept(qtbot, window, lambda: None)
    assert 'you confirmed' not in window.runStatusLabel.text() and '3 left out' in window.runStatusLabel.text()


def test_something_else_starting_while_the_person_decides_stops_the_accept_and_says_so(qtbot, tmp_path):
    window = _fixture_window(qtbot, tmp_path)
    seen = []
    _answer_after(lambda: setattr(window, '_scanning', True), seen)
    assert window.accept_selected() is True
    qtbot.waitUntil(lambda: bool(seen), timeout=30000)
    qtbot.waitUntil(lambda: not window.bulk_running, timeout=30000)
    window._scanning = False
    assert 'Not accepted: something else started while you were deciding' in window.runStatusLabel.text()
    assert all(e.status == 'pending' for e in read_queue(_queue(tmp_path)) if e.id.startswith('r-'))


def test_no_confirmation_is_shown_for_a_window_that_was_closed_while_the_plan_was_worked_out(qtbot, tmp_path, monkeypatch):
    import threading
    import model.worklist_bulk as bulk_module
    window = _fixture_window(qtbot, tmp_path)
    release, entered = threading.Event(), threading.Event()
    real = bulk_module.plan_accept

    def gated(*args, **kwargs):
        entered.set()
        assert release.wait(30)
        return real(*args, **kwargs)

    asked = []
    monkeypatch.setattr(bulk_module, 'plan_accept', gated)
    monkeypatch.setattr(bulk_module.ConfirmDialog, 'exec', lambda self: asked.append(self.windowTitle()) or 0)
    assert window.accept_selected() is True
    qtbot.waitUntil(entered.is_set)
    window.close()
    release.set()
    qtbot.waitUntil(lambda: not window.bulk_running, timeout=30000)
    qtbot.wait(100)
    assert asked == [] and 'Cancelled' not in window.runStatusLabel.text() and 'something else' not in window.runStatusLabel.text()
    assert all(e.status == 'pending' for e in read_queue(_queue(tmp_path)) if e.id.startswith('r-'))


def test_decisions_made_on_the_title_page_while_a_bulk_job_ran_are_read_when_it_ends(qtbot, tmp_path):
    window = _fixture_window(qtbot, tmp_path)
    assert window.revise_ids(['p-done'], to='review') is True
    window._index_dirty = True                       # what a decision on the page does (the read is deferred while a job runs)
    with qtbot.waitSignal(window.index_synced, timeout=30000):
        pass
    assert not window._index_dirty and read_entry(_queue(tmp_path), 'p-done').status == 'pending'


def test_a_bulk_job_that_raises_leaves_the_window_usable_and_the_dirty_index_is_still_read(qtbot, tmp_path, monkeypatch):
    import model.worklist_bulk as bulk_module

    def boom(*args, **kwargs):
        raise RuntimeError('disk on fire')

    monkeypatch.setattr(bulk_module, 'revise_titles', boom)
    window = _fixture_window(qtbot, tmp_path)
    window.select_ids(['p-done'])
    assert window.revise_ids(['p-done'], to='review') is True and window.bulk_running
    window._index_dirty = True
    with qtbot.waitSignal(window.index_synced, timeout=30000):
        pass
    assert not window.bulk_running
    assert 'Revising failed' in window.runStatusLabel.text() and 'disk on fire' in window.runStatusLabel.text()
    assert window.reviseButton.isEnabled() and window._can_start()                  # not held busy by the job that died
