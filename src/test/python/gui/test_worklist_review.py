'''
model/worklist_review.py, chunk 27c: the Review folder window -- the title page over a queue directory (it replaces the old
`ReviewQueueDialog`, whose behaviour tests are ported here), and Publish / Commit through the work list's own code
(`publish_library` / `commit_library`) with the library profile's repositories, against real temp git repos.
`import ui.beq` first: see AGENTS.md gotcha 3.
'''
import ui.beq  # noqa: F401 (must come first)

import os

import pytest
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QApplication

import model.worklist_review as review_module
from model.preferences import DESIGNER_QUEUE_DIR, LIBRARY_PROFILE_PATH, WORKLIST_PUSH
from model.worklist_confirm import ConfirmDialog
from model.worklist_review import ReviewFolderWindow, describe_commit, describe_publish, entry_row, open_review_folder
from pipeline.review import QueueEntry, read_entry
from test_pipeline_library_commit import _queue_entry, _track, repos  # noqa: F401 (a fixture)
from test_worklist_real_pipeline import _profile_file
from test_worklist_revise import _Ask
from test_worklist_title import _click, _designer, _focus, _prefs  # noqa: F401
from worklist_title_fixture import candidates, complete_meta, write_entry


def _folder(qtbot, tmp_path, entries=(('a-alien', {}), ('b-arrival', {}), ('c-sicario', {})), prefs=None, **kwargs):
    prefs = prefs or _prefs(tmp_path)
    for entry_id, options in entries:
        write_entry(str(tmp_path / 'queue'), entry_id, **options)
    window = ReviewFolderWindow(None, prefs, **kwargs)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    window.activateWindow()
    qtbot.waitActive(window)
    return window


def _queue(tmp_path):
    return str(tmp_path / 'queue')


def _statuses(window):
    return {window.entryTable.item(r, 0).data(Qt.ItemDataRole.UserRole): window.entryTable.item(r, 1).text()
            for r in range(window.entryTable.rowCount())}


# --- what the page reads of an entry -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize('status, needs, review', [('pending', 'review', 'pending'), ('accepted', 'publish', 'accepted'),
                                                   ('published', 'done', 'accepted'), ('skipped', 'done', 'skipped'),
                                                   ('rejected', 'done', 'rejected')])
def test_an_entry_is_shown_with_the_row_its_status_amounts_to(status, needs, review):
    entry = QueueEntry(id='x', fs=1000, meta={'title': 'Heat', 'year': '1995'}, curve={}, candidates=candidates(1),
                       status=status, chosen_candidate_index=0 if status in ('accepted', 'published') else None)
    row = entry_row(entry)
    assert (row.needs, row.review_state, row.title, row.year, row.kind) == (needs, review, 'Heat', '1995', 'movie')
    assert row.confidence == 0.9 and row.candidate_count == 1


def test_a_series_entry_is_a_tv_row_and_a_title_less_one_falls_back_to_its_id():
    entry = QueueEntry(id='show-s01', fs=1000, meta={'season': '1'}, curve={}, candidates=[])
    row = entry_row(entry)
    assert row.kind == 'tv' and row.title == 'show-s01' and row.confidence is None


# --- opening a folder --------------------------------------------------------------------------------------------------------------

def test_the_remembered_folder_is_listed_on_opening_and_the_first_waiting_entry_is_shown(qtbot, tmp_path):
    window = _folder(qtbot, tmp_path, entries=(('a-done', {'status': 'skipped'}), ('b-wait', {}), ('c-wait', {})))
    assert window.queue_dir == _queue(tmp_path) and window.folderEdit.text() == _queue(tmp_path)
    assert window.entry_ids == ['b-wait', 'c-wait', 'a-done']            # waiting first, as read_queue sorts them
    assert window.page.current_id == 'b-wait' and window.entryTable.currentRow() == 0
    assert window.page.candidateList.count() == 2 and window.page.acceptButton.isEnabled()
    assert _statuses(window) == {'b-wait': 'pending', 'c-wait': 'pending', 'a-done': 'skipped'}


def test_the_table_shows_the_title_the_status_and_the_top_confidence(qtbot, tmp_path):
    window = _folder(qtbot, tmp_path, entries=(('a', {}), ('b', {'decline': True})))
    cells = {window.entryTable.item(r, 0).data(Qt.ItemDataRole.UserRole): [window.entryTable.item(r, c).text() for c in range(3)]
             for r in range(2)}
    assert cells == {'a': ['a', 'pending', '0.90'], 'b': ['b', 'pending', '']}


def test_with_no_remembered_folder_it_opens_empty_and_choosing_one_lists_and_remembers_it(qtbot, tmp_path):
    prefs = _prefs(tmp_path)
    prefs.set(DESIGNER_QUEUE_DIR, '')
    other = tmp_path / 'other'
    write_entry(str(other), 'z')
    window = _folder(qtbot, tmp_path, entries=(), prefs=prefs, choose_dir=lambda start: str(other))
    assert window.entry_ids == [] and window.queue_dir == ''
    assert not window.publishButton.isEnabled() and not window.refreshButton.isEnabled()
    assert window.browse() is True
    assert window.entry_ids == ['z'] and prefs.get(DESIGNER_QUEUE_DIR) == str(other)
    assert window.refreshButton.isEnabled()


def test_a_folder_with_no_entries_says_so_and_a_missing_folder_is_refused(qtbot, tmp_path):
    empty = tmp_path / 'empty'
    empty.mkdir()
    window = _folder(qtbot, tmp_path, entries=())
    assert window.load_queue_dir(str(empty)) is True
    assert window.stack.currentWidget() is window.emptyLabel and 'no queue entries' in window.emptyLabel.text()
    assert window.load_queue_dir(str(tmp_path / 'nope')) is False and 'is not a folder' in window.statusLabel.text()
    assert window.queue_dir == str(empty)


def test_a_damaged_entry_is_left_out_and_the_rest_can_still_be_reviewed(qtbot, tmp_path):
    write_entry(_queue(tmp_path), 'good')
    with open(os_path(tmp_path, 'bad.json'), 'w') as f:
        f.write('{not json')
    window = ReviewFolderWindow(None, _prefs(tmp_path))
    qtbot.addWidget(window)
    assert window.entry_ids == ['good'] and 'could not be read' in window.statusLabel.text()


def os_path(tmp_path, name):
    os.makedirs(_queue(tmp_path), exist_ok=True)
    return os.path.join(_queue(tmp_path), name)


def test_refresh_reads_new_entries_and_stays_on_the_same_title(qtbot, tmp_path):
    window = _folder(qtbot, tmp_path)
    window.page.show_title('b-arrival')
    write_entry(_queue(tmp_path), 'd-new')
    assert window.refresh() is True
    assert 'd-new' in window.entry_ids and window.page.current_id == 'b-arrival'
    assert window.entryTable.currentRow() == window.entry_ids.index('b-arrival')


# --- reviewing (ported from the old dialog's tests) ------------------------------------------------------------------------------------

def test_selecting_a_row_shows_that_entry_with_its_candidates_and_commentary(qtbot, tmp_path):
    window = _folder(qtbot, tmp_path)
    window.entryTable.selectRow(2)
    assert window.page.current_id == 'c-sicario' and window.page.candidateList.count() == 2
    window.page.pick_candidate(1)
    assert window.page.commentaryTable.item(1, 1).text() == 'more'


def test_a_declined_entry_shows_the_reason_and_cannot_be_accepted(qtbot, tmp_path):
    window = _folder(qtbot, tmp_path, entries=(('a', {'decline': True}),))
    page = window.page
    assert page.candidateList.count() == 0 and 'no_rolloff_detected' in page.noticeLabel.text()
    assert not page.acceptButton.isEnabled() and page.skipButton.isEnabled() and page.rejectButton.isEnabled()


def test_a_digit_key_changes_the_pick_without_accepting_and_accept_writes_the_picked_candidate_and_advances(qtbot, tmp_path):
    window = _folder(qtbot, tmp_path)
    page = window.page
    _focus(qtbot, page.candidateList)
    qtbot.keyClick(page.candidateList, Qt.Key.Key_2)
    assert page.picked == 1 and read_entry(_queue(tmp_path), 'a-alien').status == 'pending'
    assert page.accept() is True
    entry = read_entry(_queue(tmp_path), 'a-alien')
    assert (entry.status, entry.chosen_candidate_index) == ('accepted', 1)
    assert page.current_id == 'b-arrival' and window.entryTable.currentRow() == window.entry_ids.index('b-arrival')
    assert _statuses(window)['a-alien'] == 'accepted'


def test_skip_and_reject_are_distinct_and_advance_without_going_back_to_the_top(qtbot, tmp_path):
    window = _folder(qtbot, tmp_path)
    page = window.page
    page.show_title('b-arrival')
    assert page.skip() and read_entry(_queue(tmp_path), 'b-arrival').status == 'skipped'
    assert page.current_id == 'c-sicario'                        # not a-alien
    assert page.reject() and read_entry(_queue(tmp_path), 'c-sicario').status == 'rejected'
    assert _statuses(window) == {'a-alien': 'pending', 'b-arrival': 'skipped', 'c-sicario': 'rejected'}


def test_enter_on_the_list_moves_to_the_candidates_and_decides_nothing_and_enter_there_accepts(qtbot, tmp_path):
    window = _folder(qtbot, tmp_path)
    window.entryTable.setFocus()
    qtbot.waitUntil(window.entryTable.hasFocus)
    qtbot.keyClick(window.entryTable, Qt.Key.Key_Return)
    qtbot.waitUntil(window.page.candidateList.hasFocus)
    assert read_entry(_queue(tmp_path), 'a-alien').status == 'pending'
    qtbot.keyClick(window.page.candidateList, Qt.Key.Key_Return)
    assert read_entry(_queue(tmp_path), 'a-alien').status == 'accepted'


def test_escape_in_the_page_goes_to_the_list_and_does_not_blank_the_window(qtbot, tmp_path):
    window = _folder(qtbot, tmp_path)
    window.page._escape()
    qtbot.waitUntil(window.entryTable.hasFocus)
    assert window.stack.currentWidget() is window.page and window.page.current_id == 'a-alien'
    assert not window.page.backButton.isVisible()


def test_an_edit_that_cannot_be_saved_keeps_the_page_and_the_list_follows_it(qtbot, tmp_path):
    window = _folder(qtbot, tmp_path)
    page = window.page
    page.confirm_discard = lambda reason: False
    page.rightTabs.setCurrentIndex(1)
    box = page.metadata.episodesField
    _focus(qtbot, box)
    box.selectAll()
    qtbot.keyClicks(box, 'abc')                         # episodes that are not numbers cannot be saved
    window.entryTable.selectRow(1)
    assert page.current_id == 'a-alien' and window.entryTable.currentRow() == 0
    page.confirm_discard = lambda reason: True      # leave nothing behind for the window's teardown to ask about
    page.metadata.discard()


def test_metadata_edits_reach_the_entry_and_the_list_title(qtbot, tmp_path):
    window = _folder(qtbot, tmp_path)
    page = window.page
    page.rightTabs.setCurrentIndex(1)
    title = page.metadata.titleField
    _focus(qtbot, title)
    title.selectAll()
    qtbot.keyClicks(title, 'Alien 3')
    assert page.flush()
    assert read_entry(_queue(tmp_path), 'a-alien').meta['title'] == 'Alien 3'
    assert window.entryTable.item(0, 0).text() == 'Alien 3'


def test_reopen_and_revise_work_here_too_over_the_folder(qtbot, tmp_path):
    ask = _Ask('review', 'again')
    window = _folder(qtbot, tmp_path, entries=(('a', {'status': 'accepted'}),), ask_revise=ask)
    assert window.page.revise() is True
    assert read_entry(_queue(tmp_path), 'a').status == 'pending' and _statuses(window) == {'a': 'pending'}
    assert ask.asked[0][1].queue_dir == _queue(tmp_path)


def test_open_project_needs_the_main_window_and_is_offered_through_the_callable(qtbot, tmp_path):
    opened = []
    window = _folder(qtbot, tmp_path, open_project=lambda path: opened.append(path) or True)
    from worklist_project_fixture import write_projects
    write_projects(tmp_path / 'work', 'a-alien')
    window.page.refresh_projects()
    assert window.page.open_project('mono') is True and opened[0].endswith('a-alien.mono.beq')
    other = _folder(qtbot, tmp_path)
    assert other.page.open_project('mono') is False


# --- publish and commit: the work list's own code ---------------------------------------------------------------------------------

class _Answer:
    def __init__(self, accept=True, tick=None):
        self.seen = []
        timer = QTimer()
        timer.setInterval(20)
        self._timer = timer

        def respond():
            dialog = QApplication.activeModalWidget()
            if not isinstance(dialog, ConfirmDialog):
                return
            timer.stop()
            self.seen.append(dialog.text)
            if tick is not None:
                dialog.checkbox.setChecked(tick)
            (dialog.ok_button if accept else dialog.cancel_button).click()

        timer.timeout.connect(respond)
        timer.start()


def _repo_window(qtbot, tmp_path, repos, entries=None):
    prefs = _prefs(tmp_path)
    # a profile file: it is the one place the images repository's owner and name (which a GitHub URL would give) can be set
    prefs.set(LIBRARY_PROFILE_PATH, _profile_file(tmp_path, repos[0].local_path, repos[2].local_path))
    queue = str(tmp_path / 'queue')
    for entry_id, title in (entries or [('one', 'Heat'), ('two', 'Alien')]):
        _queue_entry(queue, entry_id, title)
    window = ReviewFolderWindow(None, prefs)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    return window


def test_publish_and_commit_use_the_profiles_repositories_and_the_work_lists_functions(qtbot, tmp_path, repos, monkeypatch):
    calls = {}
    real_publish, real_commit = review_module.publish_library, review_module.commit_library
    monkeypatch.setattr(review_module, 'publish_library', lambda *a, **k: calls.setdefault('publish', (a, k)) and real_publish(*a, **k))
    monkeypatch.setattr(review_module, 'commit_library', lambda *a, **k: calls.setdefault('commit', (a, k)) and real_commit(*a, **k))
    xml, xml_bare, images, images_bare = repos
    window = _repo_window(qtbot, tmp_path, repos)
    assert window.publishButton.text() == 'Publish accepted (2)' and window.publishButton.isEnabled()
    assert window.commitButton.text() == 'Commit published (0)' and not window.commitButton.isEnabled()

    answer = _Answer(True)
    with qtbot.waitSignal(window.published, timeout=60000):
        window.publish_accepted()
    assert answer.seen[0].startswith('Publish 2 titles?') and xml.local_path in answer.seen[0] and images.local_path in answer.seen[0]
    args, kwargs = calls['publish']
    assert args[0] == _queue(tmp_path) and args[1].local_path == xml.local_path
    assert kwargs['images_repo'].local_path == images.local_path
    assert kwargs['work_dir'] is None and sorted(kwargs['ids']) == ['one', 'two']       # no project directories: from the candidates
    assert window.resultsBox.toPlainText() == ''
    assert os.path.isfile(os.path.join(xml.local_path, 'xml', 'one.json')) and os.path.isfile(os.path.join(images.local_path, 'img', 'two.png'))
    assert _statuses(window) == {'one': 'published', 'two': 'published'}
    assert window.statusLabel.text().startswith('Published 2 titles: written into the repositories, not committed yet.')
    assert window.commitButton.text() == 'Commit published (2)' and window.publishButton.text() == 'Publish accepted (0)'

    answer = _Answer(True, tick=False)
    with qtbot.waitSignal(window.committed, timeout=60000):
        window.commit_published()
    assert answer.seen[0].startswith('Commit 2 titles?')
    assert calls['commit'][1]['push'] is False and calls['commit'][1]['images_repo'].local_path == images.local_path
    assert window.statusLabel.text().startswith('Committed. images: commit ') and 'not pushed' in window.statusLabel.text()
    log = os.popen(f'git -C {xml.local_path} log --format=%s').read()
    assert log.startswith('Publish 2 BEQ filters')
    assert window._preferences.get(WORKLIST_PUSH) is False


def test_an_entry_with_projects_in_the_work_directory_is_published_from_them_and_the_others_from_their_candidates(
        qtbot, tmp_path, repos, monkeypatch):
    from pipeline.library.sync import publish_library as real_publish
    from worklist_project_fixture import edit_project, write_projects
    seen = []
    monkeypatch.setattr(review_module, 'publish_library', lambda *a, **k: seen.append((k['work_dir'], k['ids'])) or real_publish(*a, **k))
    window = _repo_window(qtbot, tmp_path, repos)
    mono, _ = write_projects(tmp_path / 'work', 'one')
    edit_project(mono)
    _Answer(True)
    answer = _Answer(True)
    with qtbot.waitSignal(window.published, timeout=60000):
        window.publish_accepted()
    assert seen == [(str(tmp_path / 'work'), ['one']), (None, ['two'])]
    assert _statuses(window) == {'one': 'published', 'two': 'published'} and answer.seen
    assert 'LibsndfileError' not in window.statusLabel.text() + window.resultsBox.toPlainText()


def test_a_declined_publish_writes_nothing(qtbot, tmp_path, repos):
    xml, _, images, _ = repos
    window = _repo_window(qtbot, tmp_path, repos)
    answer = _Answer(False)
    window.publish_accepted()
    assert answer.seen and 'Publish cancelled' in window.statusLabel.text()
    assert not os.path.exists(os.path.join(xml.local_path, 'xml', 'one.json'))
    assert read_entry(_queue(tmp_path), 'one').status == 'accepted'


def test_an_entry_with_incomplete_metadata_is_refused_on_its_own_and_listed(qtbot, tmp_path, repos):
    ''' The old XML-only button raised mid-batch; the shared publish reports it per title (T9). '''
    window = _repo_window(qtbot, tmp_path, repos)
    from pipeline.review import update_entry
    update_entry(_queue(tmp_path), 'two', meta={'title': 'Alien'})           # no year, no audio types
    window.refresh()
    answer = _Answer(True)
    with qtbot.waitSignal(window.published, timeout=60000):
        window.publish_accepted()
    assert _statuses(window) == {'one': 'published', 'two': 'accepted'}
    assert '1 could not be published' in window.statusLabel.text()
    assert window.resultsBox.isVisibleTo(window) and window.resultsBox.toPlainText().startswith('two:')
    assert answer.seen


def test_without_an_xml_repository_publish_and_commit_say_why_and_do_nothing(qtbot, tmp_path):
    window = _folder(qtbot, tmp_path, entries=(('a', {'status': 'accepted'}),))
    assert not window.publishButton.isEnabled() and 'XML repository' in window.publishButton.toolTip()
    assert window.publish_accepted() is False and 'XML repository' in window.statusLabel.text()
    assert window.commit_published() is False


def test_a_failing_publish_is_shown_not_raised_and_the_window_carries_on(qtbot, tmp_path, repos, monkeypatch):
    window = _repo_window(qtbot, tmp_path, repos)

    def boom(*args, **kwargs):
        raise RuntimeError('disk on fire')

    monkeypatch.setattr(review_module, 'publish_library', boom)
    answer = _Answer(True)
    with qtbot.waitSignal(window.failed, timeout=60000):
        window.publish_accepted()
    assert answer.seen and 'disk on fire' in window.statusLabel.text() and not window.is_busy and window.publishButton.isEnabled()


def test_the_words_for_a_publish_and_a_commit():
    assert describe_publish([{'id': 'a'}, {'id': 'b', 'error': 'invalid_metadata'}]).endswith('1 could not be published (see below).')
    from pipeline.library.commit import CatalogueCommit, RepoCommit
    text = describe_commit(CatalogueCommit(RepoCommit('/x', ['a.json'], 'abcdef123456', True), RepoCommit('/i', [], None, False)))
    assert text == 'Committed. images: already committed, not pushed; XML: commit abcdef12, pushed.'


# --- opening it from the dialogs ---------------------------------------------------------------------------------------------------

def test_open_review_folder_uses_the_main_windows_one_window_when_there_is_one_else_makes_its_own(qtbot, tmp_path):
    from qtpy.QtWidgets import QWidget

    class _Main(QWidget):
        shown = []

        def showReviewFolderWindow(self, queue_dir):
            self.shown.append(queue_dir)
            return 'the main window\'s'

    main = _Main()
    qtbot.addWidget(main)
    owner = QWidget(main)
    assert open_review_folder(owner, _prefs(tmp_path), '/q') == "the main window's" and _Main.shown == ['/q']

    write_entry(_queue(tmp_path), 'a')
    lone = QWidget()
    qtbot.addWidget(lone)
    alone = open_review_folder(lone, _prefs(tmp_path), _queue(tmp_path))     # a child of `lone`: closed with it
    assert isinstance(alone, ReviewFolderWindow) and alone.isVisible() and alone.entry_ids == ['a']


def test_the_old_dialog_and_its_module_are_gone():
    import importlib
    for name in ('model.review', 'model.library_sync', 'ui.review', 'ui.library_sync'):
        with pytest.raises(ImportError):
            importlib.import_module(name)


def test_the_meta_defaults_and_work_dir_come_from_the_profile(qtbot, tmp_path):
    window = _folder(qtbot, tmp_path)
    assert window._work_dir() == str(tmp_path / 'work')
    assert window._revise_context().queue_dir == _queue(tmp_path)
    assert complete_meta('x')['year'] == '2001'
