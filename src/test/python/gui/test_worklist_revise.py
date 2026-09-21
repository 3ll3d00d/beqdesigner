'''
model/worklist_revise.py, model/worklist_title_actions.py's revise half and model/worklist_bulk.py's Revise..., chunk 27c: Reopen /
Revise on the title page and on the selected rows -- the question says what will happen, the state changes (nothing is redesigned
or published), a published title needs the repositories (real temp git repos), a reason reaches the reviewer note, results and
errors are shown and never raised, the interlocks hold, and Accept / Skip / Reject apply again to a reopened title.
Two tests go through a REAL scan: a title's `needs` moves once the page is left (single) and after Revise... (bulk).
`import ui.beq` first: see AGENTS.md gotcha 3.
'''
import ui.beq  # noqa: F401 (must come first)

import os
import time
from types import SimpleNamespace

import pytest
from qtpy.QtCore import QSettings, Qt, QTimer
from qtpy.QtWidgets import QApplication

from model.preferences import LIBRARY_IMAGES_REPO, LIBRARY_XML_REPO, Preferences
from model.worklist import WorkListWindow
from model.worklist_revise import CHOICES, ReviseContext, ReviseDialog, ReviseOutcome, ReviseSummary, describe_outcome, \
    revise_context, revise_problem, revise_text, revise_titles, summarise_outcome
from pipeline.library.sync import commit_library, publish_library
from pipeline.publish.git import RepoTarget
from pipeline.review import read_entry, update_entry
from test_pipeline_library_commit import IMAGES_NAME, OWNER, _queue_entry, _track, repos  # noqa: F401 (a fixture)
from test_pipeline_library_index import FakeSource, _entry as _real_entry, _extracted, _item
from test_worklist_title import NOW, REVIEWABLE, SOURCES, _click, _designer, _open, _prefs, _queue, _row, _rows, _status  # noqa: F401
from worklist_fixture import make_index
from worklist_title_fixture import write_entry


def _window(qtbot, tmp_path, entries=REVIEWABLE, rows=None, repos_set=None, **kwargs):
    prefs = _prefs(tmp_path)
    if repos_set is not None:
        prefs.set(LIBRARY_XML_REPO, repos_set[0].local_path)
        prefs.set(LIBRARY_IMAGES_REPO, repos_set[2].local_path)
    make_index(tmp_path / 'work', rows if rows is not None else _rows(), SOURCES, generation=2, last_scan_at=NOW - 900)
    for entry_id, options in entries:
        write_entry(str(tmp_path / 'queue'), entry_id, **options)
    window = WorkListWindow(None, prefs, auto_scan=False, clock=lambda: NOW, **kwargs)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    window.activateWindow()
    qtbot.waitActive(window)
    return window


class _Ask:
    ''' The answer to the Revise question, and what it was asked. '''

    def __init__(self, to='review', reason='', cancel=False):
        self.to, self.reason, self.cancel, self.asked = to, reason, cancel, []

    def __call__(self, summary, context, default):
        self.asked.append((summary, context, default))
        return None if self.cancel else (self.to, self.reason)


def _published(tmp_path, repos, entry_id='p-pub', commit=False):
    ''' A published title whose files are really in the repositories (and committed, if asked). '''
    xml, _, images, _ = repos
    queue_dir = str(tmp_path / 'queue')
    _queue_entry(queue_dir, entry_id, 'Pub')
    publish_library(queue_dir, xml, images_repo=images, image_owner=OWNER, image_repo_name=IMAGES_NAME)
    if commit:
        commit_library(queue_dir, xml, images_repo=images, push=False)
        _track(xml)
        _track(images)
    return queue_dir


# --- the words and the rules, with no widgets -----------------------------------------------------------------------------------

def test_a_summary_counts_titles_by_what_reopening_will_do_to_them():
    summary = ReviseSummary.of([('published', 'pushed'), ('published', 'uncommitted'), ('accepted', ''), ('pending', ''),
                                ('skipped', ''), ('rejected', ''), ('none', '')], 'Heat')
    assert (summary.count, summary.published, summary.in_catalogue, summary.accepted, summary.pending, summary.decided,
            summary.no_entry) == (7, 2, 1, 1, 1, 2, 1)
    assert summary.unchanged_by_reopen == 2


def test_the_question_says_what_will_happen_before_anything_does():
    summary = ReviseSummary.of([('published', 'pushed'), ('published', 'uncommitted'), ('accepted', ''), ('pending', '')])
    heading, body = revise_text('review', summary)
    assert heading == 'Reopen for review: 4 titles?'
    assert 'Nothing is redesigned, extracted or published now' in body
    assert '1 title is accepted: the accept is undone' in body
    assert '1 title is already committed to the catalogue' in body and 'a revision' in body
    assert '1 title is published but not committed' in body and 'taken out of the repositories' in body
    assert '1 title is already waiting for review' in body
    assert 'reviewer note' in body
    _, redesign = revise_text('design', summary)
    assert 'designs it again' in redesign and 'edit you made to a project' in redesign
    _, extract = revise_text('extract', summary)
    assert 'extracted audio is forgotten' in extract
    assert revise_text('review', ReviseSummary.of([('accepted', '')], 'Heat'))[0] == 'Reopen for review: Heat?'


def test_what_cannot_be_done_is_said_before_anything_changes():
    context = ReviseContext('/q', '/w')
    published = ReviseSummary.of([('published', 'pushed')])
    assert 'XML repository' in revise_problem('review', published, context)
    assert revise_problem('review', published, ReviseContext('/q', '/w', RepoTarget('/x'))) == ''
    assert 'queue directory' in revise_problem('review', published, None)
    assert revise_problem('review', ReviseSummary.of([]), context) == 'No title is selected.'
    assert 'nothing to send back' in revise_problem('design', ReviseSummary.of([('none', '')]), context)
    assert 'already waiting for review' in revise_problem('review', ReviseSummary.of([('pending', '')]), context)
    assert revise_problem('design', ReviseSummary.of([('pending', '')]), context) == ''     # a pending title can be redesigned
    assert 'work directory' in revise_problem('extract', ReviseSummary.of([('accepted', '')]), ReviseContext('/q'))


def test_the_context_is_the_setups_queue_work_and_repositories(tmp_path):
    settings = SimpleNamespace(queue_dir='/q', work_dir='/w', xml_repo='/x', images_repo='', xml_dir='xml', image_dir='img')
    context = revise_context(SimpleNamespace(settings=settings))
    assert context == ReviseContext('/q', '/w', RepoTarget('/x'), None, 'xml', 'img')
    assert revise_context(SimpleNamespace(settings=None)) is None
    assert revise_context(SimpleNamespace(settings=SimpleNamespace(queue_dir='', work_dir='', xml_repo='', images_repo='',
                                                                   xml_dir='', image_dir=''))) is None


def test_revising_reopens_records_the_reason_and_reports_a_title_it_cannot_send_back_without_raising(tmp_path):
    queue = str(tmp_path / 'queue')
    write_entry(queue, 'a', status='accepted')
    write_entry(queue, 'b')                          # pending: cannot be reopened
    outcome = revise_titles(ReviseContext(queue), ['a', 'b', 'gone', 'a'], 'review', ' wrong poster ')
    assert outcome.revised == ['a'] and [i for i, _ in outcome.failed] == ['b', 'gone']
    assert 'already pending' in outcome.failed[0][1] and 'no queue entry' in outcome.failed[1][1]
    entry = read_entry(queue, 'a')
    assert entry.status == 'pending' and entry.reviewer_note == 'Reopened for review: wrong poster'
    assert read_entry(queue, 'b').status == 'pending'


def test_a_published_title_without_its_repository_is_reported_and_left_as_it_was(tmp_path, repos):
    queue = _published(tmp_path, repos)
    outcome = revise_titles(ReviseContext(queue), ['p-pub'], 'review')
    assert outcome.revised == [] and 'xml_repo' in outcome.failed[0][1]
    assert read_entry(queue, 'p-pub').status == 'published'


def test_the_outcome_is_worded_as_one_line_and_one_line_per_title():
    outcome = ReviseOutcome('design', revised=['a'], failed=[('b', 'because')], reverted={'a': ['x.xml']}, revisions={'a': 2})
    text, level = summarise_outcome(outcome)
    assert text == '1 title sent back for redesign; 1 could not be changed (see the Last run tab).'
    assert level == 'warn'
    lines = describe_outcome(outcome, {'a': 'Heat'})
    assert [(l.title, l.outcome) for l in lines] == [('b', 'Not changed'), ('Heat', 'Sent back for redesign')]
    assert 'x.xml' in lines[1].detail and 'revision 2' in lines[1].detail
    assert summarise_outcome(ReviseOutcome('review'))[0] == 'Nothing to do.'
    only_failed = summarise_outcome(ReviseOutcome('review', failed=[('b', 'nope')]))
    assert only_failed[1] == 'error'


def test_the_dialog_starts_where_asked_disables_ok_with_the_reason_and_reads_the_choice_and_reason(qtbot):
    summary = ReviseSummary.of([('published', 'pushed')], 'Heat')
    dialog = ReviseDialog(None, summary, ReviseContext('/q'), 'design')
    qtbot.addWidget(dialog)
    assert dialog.choice == 'design' and not dialog.ok_button.isEnabled() and 'XML repository' in dialog.problemLabel.text()
    assert dialog.cancel_button.isDefault() and not dialog.ok_button.autoDefault()
    assert list(dialog.radios) == list(CHOICES)
    ok = ReviseDialog(None, summary, ReviseContext('/q', '/w', RepoTarget('/x')), 'extract')
    qtbot.addWidget(ok)
    ok.reasonEdit.setText('  another stream  ')
    assert ok.choice == 'extract' and ok.reason == 'another stream' and ok.ok_button.isEnabled()
    ok.set_choice('review')
    assert ok.choice == 'review' and ok.ok_button.text() == 'Reopen for review' and 'Reopen for review: Heat?' in ok.text


# --- the title page ---------------------------------------------------------------------------------------------------------------

def test_reopening_an_accepted_title_from_the_page_makes_it_pending_and_it_can_be_decided_again(qtbot, tmp_path):
    ask = _Ask('review', 'wrong candidate')
    window = _window(qtbot, tmp_path, ask_revise=None)
    page = _open(qtbot, window, 'p-fury')
    page._hooks.ask_revise = ask
    assert page.acceptButton.isEnabled() is False and page.actions_bar.reviseButton.isEnabled()
    heard = []
    page.revised.connect(lambda i, to: heard.append((i, to)))

    assert page.revise() is True

    entry = read_entry(_queue(tmp_path), 'p-fury')
    assert entry.status == 'pending' and entry.chosen_candidate_index is None
    assert entry.reviewer_note == 'Reopened for review: wrong candidate'
    assert heard == [('p-fury', 'review')]
    assert ask.asked[0][0].accepted == 1 and ask.asked[0][2] == 'review'
    assert page.stateLabel.text().startswith('Waiting for a decision')
    assert page.acceptButton.isEnabled() and page.skipButton.isEnabled() and page.rejectButton.isEnabled()
    assert 'reopened for review' in page.decisionLabel.text().lower()

    assert page.accept() is True                    # Accept applies again
    assert _status(tmp_path, 'p-fury') == 'accepted'


@pytest.mark.parametrize('decision, status', [('skip', 'skipped'), ('reject', 'rejected')])
def test_skip_and_reject_apply_again_to_a_reopened_title(qtbot, tmp_path, decision, status):
    window = _window(qtbot, tmp_path)
    page = _open(qtbot, window, 'p-fury')
    assert page.revise('review')
    assert getattr(page, decision)() is True
    assert _status(tmp_path, 'p-fury') == status


@pytest.mark.parametrize('status', ['skipped', 'rejected'])
def test_a_skipped_or_rejected_title_can_be_reopened(qtbot, tmp_path, status):
    window = _window(qtbot, tmp_path, [('r-alien', {'status': status})])
    page = _open(qtbot, window, 'r-alien')
    assert page.revise('review', 'second thoughts') and _status(tmp_path, 'r-alien') == 'pending'


def test_a_pending_title_offers_redesign_first_and_reopening_it_is_refused_with_the_reason(qtbot, tmp_path):
    ask = _Ask('review')
    window = _window(qtbot, tmp_path)
    page = _open(qtbot, window, 'r-alien')
    page._hooks.ask_revise = ask
    assert page.revise() is False
    assert ask.asked[0][2] == 'design'
    assert 'already waiting for review' in page.decisionLabel.text() and page.decisionLabel.text().startswith('Not changed')
    assert read_entry(_queue(tmp_path), 'r-alien').reviewer_note is None


def test_redesigning_marks_the_design_stale_and_holds_accept_back_until_it_has_been_designed_again(qtbot, tmp_path):
    window = _window(qtbot, tmp_path)
    page = _open(qtbot, window, 'r-alien')
    assert page.acceptButton.isEnabled()

    assert page.revise('design', 'new designer') is True

    entry = read_entry(_queue(tmp_path), 'r-alien')
    assert entry.status == 'pending' and entry.design_fingerprint is None
    assert 'Sent back for redesign: new designer' in entry.reviewer_note
    assert not page.acceptButton.isEnabled() and page.skipButton.isEnabled()
    assert 'sent back for redesign' in page.decisionLabel.text()
    assert 'run Extract & design' in page.stateLabel.text()
    assert page.revised_here == {'r-alien': 'design'}
    assert page.accept() is False and _status(tmp_path, 'r-alien') == 'pending'
    assert page.next() and page.current_id == 'r-arrival'      # and Next does not walk back into it as "waiting"
    assert page.previous()
    assert 'r-alien' not in [i for i in page.ids if page._probably_waiting(i, window._rows_by_id())]


def test_re_extracting_needs_a_work_directory_and_forgets_the_recorded_extraction(qtbot, tmp_path):
    import json
    directory = tmp_path / 'work' / 'r-alien'
    directory.mkdir(parents=True)
    (directory / 'manifest.json').write_text(json.dumps({'mono_source_fingerprint': 'x', 'mono_params_hash': 'y'}))
    window = _window(qtbot, tmp_path)
    page = _open(qtbot, window, 'r-alien')
    assert page.revise('extract') is True
    assert json.loads((directory / 'manifest.json').read_text()) == {}
    assert read_entry(_queue(tmp_path), 'r-alien').design_fingerprint is None


def test_cancelling_the_question_changes_nothing(qtbot, tmp_path):
    window = _window(qtbot, tmp_path)
    page = _open(qtbot, window, 'p-fury')
    page._hooks.ask_revise = _Ask(cancel=True)
    before = read_entry(_queue(tmp_path), 'p-fury')
    assert page.revise() is False
    assert read_entry(_queue(tmp_path), 'p-fury') == before and page.revised_here == {}


def test_the_real_dialog_is_shown_says_what_will_happen_and_its_answer_is_used(qtbot, tmp_path):
    window = _window(qtbot, tmp_path)
    page = _open(qtbot, window, 'p-fury')
    seen = []

    def answer():
        dialog = QApplication.activeModalWidget()
        assert isinstance(dialog, ReviseDialog)
        seen.append(dialog.text)
        dialog.set_choice('design')
        dialog.reasonEdit.setText('better designer')
        dialog.ok_button.click()

    QTimer.singleShot(0, answer)
    _click(qtbot, page.actions_bar.reviseButton)

    assert 'Reopen for review: p-fury' in seen[0] and 'the accept is undone' in seen[0]
    entry = read_entry(_queue(tmp_path), 'p-fury')
    assert entry.status == 'pending' and entry.reviewer_note == 'Sent back for redesign: better designer'
    assert QApplication.activeModalWidget() is None


def test_nothing_can_be_sent_back_for_a_title_with_no_entry_or_one_a_run_is_working_on(qtbot, tmp_path):
    window = _window(qtbot, tmp_path)
    page = _open(qtbot, window, 'x-gravity')                    # no queue entry
    assert not page.actions_bar.reviseButton.isEnabled() and 'Nothing has been designed' in page.actions_bar.reviseButton.toolTip()
    assert page.revise('review') is False and 'Nothing has been designed' in page.decisionLabel.text()

    page = _open(qtbot, window, 'r-alien') if window.close_title() else None
    window.model.set_running({'r-alien': 'design'})
    assert not page.actions_bar.reviseButton.isEnabled() and 'A run is working on this title' in page.actions_bar.reviseButton.toolTip()
    assert page.revise('review') is False and _status(tmp_path, 'r-alien') == 'pending'
    window.model.set_running({})
    assert page.actions_bar.reviseButton.isEnabled()


def test_a_publish_or_commit_run_holds_back_the_revise_of_an_accepted_title_but_not_a_pending_one(qtbot, tmp_path):
    window = _window(qtbot, tmp_path)
    page = _open(qtbot, window, 'p-fury')
    window._job = object()
    window._run_context = SimpleNamespace(request=SimpleNamespace(through='commit'))
    try:
        assert window._revise_blocked('p-fury', 'accepted') and window._revise_blocked('p-fury', 'published')
        assert window._revise_blocked('r-alien', 'pending') == ''
        page.refresh_decisions()
        assert not page.actions_bar.reviseButton.isEnabled()
        window._run_context = SimpleNamespace(request=SimpleNamespace(through='design'))
        assert window._revise_blocked('p-fury', 'accepted') == ''
    finally:
        window._job = None
        window._run_context = None


def test_a_published_title_is_reopened_from_the_page_with_the_repositories_of_the_setup(qtbot, tmp_path, repos):
    queue = _published(tmp_path, repos)
    xml, _, images, _ = repos
    rows = _rows() + [_row('p-pub', 'Pub', 'commit', 4, review_state='accepted', publish_state='written',
                           commit_state='uncommitted')]
    window = _window(qtbot, tmp_path, entries=[], rows=rows, repos_set=repos)
    _open(qtbot, window, 'p-pub')
    page = window.title_page
    assert os.path.isfile(os.path.join(xml.local_path, 'p-pub.xml'))
    ask = _Ask('review', 'typo')
    page._hooks.ask_revise = ask

    assert page.revise() is True

    assert ask.asked[0][0].published == 1 and ask.asked[0][1].xml_repo == RepoTarget(xml.local_path)
    assert read_entry(queue, 'p-pub').status == 'pending'
    assert not os.path.exists(os.path.join(xml.local_path, 'p-pub.xml'))       # written, never committed: taken out again
    assert not os.path.exists(os.path.join(images.local_path, 'p-pub.png'))


def test_a_published_title_with_no_xml_repository_set_is_not_reopened_and_says_why(qtbot, tmp_path, repos):
    queue = _published(tmp_path, repos)
    rows = _rows() + [_row('p-pub', 'Pub', 'commit', 4, review_state='accepted', publish_state='written')]
    window = _window(qtbot, tmp_path, entries=[], rows=rows)          # no repositories in the preferences
    page = _open(qtbot, window, 'p-pub')
    assert page.revise('review') is False
    assert 'XML repository' in page.decisionLabel.text()
    assert read_entry(queue, 'p-pub').status == 'published'


def test_the_index_reads_the_revise_once_the_page_is_left_and_the_status_line_says_so(qtbot, tmp_path):
    window = _window(qtbot, tmp_path)
    page = _open(qtbot, window, 'p-fury')
    assert page.revise('review')
    assert window._index_dirty and 'p-fury: reopened for review' in window.statusBar.currentMessage()
    with qtbot.waitSignal(window.index_synced, timeout=30000):
        _click(qtbot, page.backButton)
    assert not window._index_dirty


# --- through a real scan ---------------------------------------------------------------------------------------------------------

def _scanned(qtbot, tmp_path, *names, status='accepted', **options):
    work = SimpleNamespace(work=str(tmp_path / 'work'), queue=str(tmp_path / 'queue'))
    work.items = [_item(name) for name in names]
    for item in work.items:
        _extracted(work, item)
        _real_entry(work, item, status=status)
    window = WorkListWindow(None, _prefs(tmp_path), sources={'filesystem': FakeSource(work.items)}, auto_scan=False,
                            clock=time.time, **options)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    window.activateWindow()
    qtbot.waitActive(window)
    with qtbot.waitSignal(window.scan_finished, timeout=30000):
        window.rescan()
    return window


def _needs(window):
    return {row.id: row.needs for row in window.model.rows}


def test_reopening_on_the_page_moves_the_title_from_publish_to_review_once_the_page_is_left(qtbot, tmp_path):
    window = _scanned(qtbot, tmp_path, 'a', 'b')
    assert _needs(window) == {'fs-a': 'publish', 'fs-b': 'publish'}
    page = _open(qtbot, window, 'fs-a')
    assert page.revise('review', 'look again')
    assert _needs(window) == {'fs-a': 'publish', 'fs-b': 'publish'}         # the index has not read it yet
    with qtbot.waitSignal(window.index_synced, timeout=30000):
        _click(qtbot, page.backButton)
    assert _needs(window) == {'fs-a': 'review', 'fs-b': 'publish'}
    assert window.publishButton.text() == 'Publish 1'


def test_redesigning_on_the_page_moves_the_title_to_design_in_a_real_scan(qtbot, tmp_path):
    window = _scanned(qtbot, tmp_path, 'a')
    page = _open(qtbot, window, 'fs-a')
    assert page.revise('design')
    with qtbot.waitSignal(window.index_synced, timeout=30000):
        _click(qtbot, page.backButton)
    assert _needs(window) == {'fs-a': 'design'}


# --- Revise... on the selected rows -----------------------------------------------------------------------------------------------

def test_the_revise_button_needs_a_selection_and_says_how_many(qtbot, tmp_path):
    window = _scanned(qtbot, tmp_path, 'a', 'b')
    assert not window.reviseButton.isEnabled() and window.reviseButton.text() == 'Revise...'
    window.select_ids(['fs-a', 'fs-b'])
    assert window.reviseButton.isEnabled() and window.reviseButton.text() == 'Revise (2)...'
    assert window.revise_ids([]) is False


def test_revising_the_selected_rows_asks_then_sends_them_back_and_the_list_follows_after_a_real_scan(qtbot, tmp_path):
    ask = _Ask('review', 'settings changed')
    window = _scanned(qtbot, tmp_path, 'a', 'b', ask_revise=ask)
    window.select_ids(['fs-a'])
    assert window.revise_selected() is True
    assert window.bulk_running and not window.reviseButton.isEnabled()      # interlock: one job at a time
    qtbot.waitUntil(lambda: not window.bulk_running, timeout=30000)

    assert ask.asked[0][0].accepted == 1 and ask.asked[0][2] == 'review'
    assert _needs(window) == {'fs-a': 'review', 'fs-b': 'publish'}
    assert read_entry(_queue(tmp_path), 'fs-a').reviewer_note == 'Reopened for review: settings changed'
    assert window.runStatusLabel.text() == '1 title reopened for review.'
    assert [(l.title, l.outcome) for l in window.results] == [('Film a', 'Reopened')]
    assert window.selected_ids() == ['fs-a']


def test_a_title_that_cannot_be_sent_back_is_a_line_in_the_last_run_tab_not_an_exception(qtbot, tmp_path):
    window = _scanned(qtbot, tmp_path, 'a', 'b', status='pending')
    assert window.revise_ids(['fs-b'], to='review') is False           # refused up front, with the reason, nothing raised
    assert 'already waiting for review' in window.runStatusLabel.text() and window.runStatusLabel.text().startswith('Not changed')
    assert window.revise_ids(['fs-a'], to='design') is True             # a pending title can be redesigned
    qtbot.waitUntil(lambda: not window.bulk_running, timeout=30000)
    assert _needs(window)['fs-a'] == 'design'


def test_a_bulk_revise_reports_the_titles_it_could_not_change_next_to_those_it_did(qtbot, tmp_path):
    window = _scanned(qtbot, tmp_path, 'a', 'b')
    update_entry(_queue(tmp_path), 'fs-b', status='pending', chosen_candidate_index=None)
    assert window.revise_ids(['fs-a', 'fs-b'], to='review')
    qtbot.waitUntil(lambda: not window.bulk_running, timeout=30000)
    by_title = {l.title: (l.outcome, l.detail) for l in window.results}
    assert by_title['Film a'][0] == 'Reopened' and by_title['Film b'][0] == 'Not changed'
    assert 'already pending' in by_title['Film b'][1]
    assert window.runStatusLabel.text() == '1 title reopened for review; 1 could not be changed (see the Last run tab).'


def test_a_revise_is_refused_while_a_scan_a_run_or_a_read_is_going(qtbot, tmp_path):
    window = _scanned(qtbot, tmp_path, 'a')
    for flag in ('_scanning', '_syncing'):
        setattr(window, flag, True)
        assert window.revise_ids(['fs-a'], to='review') is False and window.accept_selected(['fs-a']) is False
        setattr(window, flag, False)
    window._job = object()
    try:
        assert window.revise_ids(['fs-a'], to='review') is False
    finally:
        window._job = None
    assert _status(tmp_path, 'fs-a') == 'accepted'


def test_the_real_dialog_asks_on_the_selected_rows_and_cancel_changes_nothing(qtbot, tmp_path):
    window = _scanned(qtbot, tmp_path, 'a')
    window.select_ids(['fs-a'])
    seen = []

    def cancel():
        dialog = QApplication.activeModalWidget()
        assert isinstance(dialog, ReviseDialog)
        seen.append(dialog.text)
        dialog.cancel_button.click()

    QTimer.singleShot(0, cancel)
    assert window.revise_selected() is False
    assert 'Reopen for review: Film a?' in seen[0]
    assert _status(tmp_path, 'fs-a') == 'accepted' and 'Cancelled' in window.runStatusLabel.text()


def test_a_published_title_in_a_selection_needs_the_repository_and_is_not_touched_without_it(qtbot, tmp_path, repos):
    queue = _published(tmp_path, repos)
    rows = _rows() + [_row('p-pub', 'Pub', 'commit', 4, review_state='accepted', publish_state='written')]
    window = _window(qtbot, tmp_path, entries=[], rows=rows)
    assert window.revise_ids(['p-pub'], to='review') is False
    assert 'XML repository' in window.runStatusLabel.text() and read_entry(queue, 'p-pub').status == 'published'
