'''
model/worklist_review.py, the independent review of chunk 27c (2026-09-21): the interlocks the folder window lacked while its own Publish
or Commit runs, the hold on a title sent back for redesign (released when it is designed again, and worded for the folder), the
revise question telling the truth about what it does to a published title's files (real git repos), the library profile read again
where a decision is made from it, and Publish being the work list's code path (its `config`, and no silent fall-back from an edited
project whose audio is gone).
`import ui.beq` first: see AGENTS.md gotcha 3.
'''
import ui.beq  # noqa: F401 (must come first)

import json
import os
import threading

import pytest
from qtpy.QtCore import QEvent
from qtpy.QtWidgets import QApplication

import model.worklist_review as review_module
from model.preferences import LIBRARY_PROFILE_PATH
from model.worklist_folder_state import commit_states, entry_row, split_for_publish
from model.worklist_review import ReviewFolderWindow
from model.worklist_revise import ReviseSummary, revise_text
from pipeline.library.sync import commit_library, publish_library
from pipeline.review import read_entry, update_entry
from test_pipeline_library_commit import IMAGES_NAME, OWNER, _queue_entry, repos  # noqa: F401 (a fixture)
from test_worklist_review import _Answer, _folder, _queue, _repo_window, _statuses
from test_worklist_revise import _Ask
from test_worklist_title import _prefs
from worklist_project_fixture import edit_project, write_projects
from worklist_title_fixture import write_entry


def _profile(tmp_path, xml='', images='', name='profile.json', **run):
    ''' A profile file naming the repositories given (none if empty), and `run` options. '''
    sync = {}
    if xml:
        sync.update(xml_repo=xml, xml_dir='xml')
    if images:
        sync.update(images_repo=images, image_dir='img', image_owner=OWNER, image_repo_name=IMAGES_NAME)
    path = tmp_path / name
    path.write_text(json.dumps({
        'sources': [{'name': 'films', 'kind': 'filesystem', 'globs': [str(tmp_path / 'films' / '*.mkv')]}],
        'run': {'work_dir': str(tmp_path / 'work'), 'queue_dir': str(tmp_path / 'queue'), 'designer': 'test.designer', **run},
        'sync': sync}))
    return str(path)


# --- 1. the folder window's own Publish / Commit is an interlock -----------------------------------------------------------------------

class _Gate:
    ''' A `publish_library` that waits to be let go: the job is "in flight" for as long as the test wants. '''

    def __init__(self, monkeypatch):
        self.release, self.entered = threading.Event(), threading.Event()
        real = review_module.publish_library

        def gated(*args, **kwargs):
            self.entered.set()
            assert self.release.wait(30)
            return real(*args, **kwargs)

        monkeypatch.setattr(review_module, 'publish_library', gated)


def test_while_a_publish_runs_its_titles_cannot_be_revised_decided_or_edited_and_the_buttons_say_why(
        qtbot, tmp_path, repos, monkeypatch):
    gate = _Gate(monkeypatch)
    window = _repo_window(qtbot, tmp_path, repos)
    page = window.page
    answer = _Answer(True)  # (kept: its timer answers the confirmation)
    assert window.publish_accepted() is True and window.is_busy
    qtbot.waitUntil(gate.entered.is_set)
    try:
        assert window._working_on == {'one': 'publish', 'two': 'publish'}
        assert page.current_id == 'one'
        assert not page.actions_bar.reviseButton.isEnabled()
        assert 'A run is working on this title now' in page.actions_bar.reviseButton.toolTip()
        assert page.revise(to='review') is False           # what reproduced: it went pending under the worker's feet
        assert 'A run is working on this title now' in page.decisionLabel.text()
        assert read_entry(_queue(tmp_path), 'one').status == 'accepted'
        assert page.metadata.titleField.isReadOnly() and not page.metadata.saveMetadataButton.isEnabled()
        assert not window.publishButton.isEnabled() and not window.refreshButton.isEnabled()
        # a title the job is not working on, that is in the repositories, waits too: revising it uses the same working trees
        assert 'catalogue repositories' in window._revise_blocked('other', 'published')
        assert window._revise_blocked('other', 'pending') == '' and window._revise_blocked('other', 'skipped') == ''
    finally:
        gate.release.set()
    qtbot.waitUntil(lambda: not window.is_busy, timeout=60000)
    assert window._working_on == {} and _statuses(window) == {'one': 'published', 'two': 'published'}
    assert page.actions_bar.reviseButton.isEnabled() and not page.metadata.titleField.isReadOnly()


def test_the_folder_cannot_be_changed_under_a_running_publish(qtbot, tmp_path, repos, monkeypatch):
    gate = _Gate(monkeypatch)
    window = _repo_window(qtbot, tmp_path, repos)
    other = tmp_path / 'other'
    write_entry(str(other), 'z')
    answer = _Answer(True)  # (kept: its timer answers the confirmation)
    window.publish_accepted()
    qtbot.waitUntil(gate.entered.is_set)
    try:
        assert window.load_queue_dir(str(other)) is False and 'A publish or commit is going' in window.statusLabel.text()
        assert window.queue_dir == _queue(tmp_path)
    finally:
        gate.release.set()
    qtbot.waitUntil(lambda: not window.is_busy, timeout=60000)


# --- 2. a title sent back for redesign is held back until it has been designed again -----------------------------------------------------

def test_a_title_sent_back_for_redesign_is_held_back_until_batch_extract_designs_it_again_and_refresh_shows_it(qtbot, tmp_path):
    ask = _Ask('design', 'again')
    window = _folder(qtbot, tmp_path, entries=(('a', {'status': 'accepted'}), ('b', {})), ask_revise=ask)
    page = window.page
    page.show_title('a')
    assert page.revise() is True

    assert read_entry(_queue(tmp_path), 'a').status == 'pending'
    assert not page.acceptButton.isEnabled()
    assert window.refresh() is True and not page.acceptButton.isEnabled()      # nothing was designed: still held back
    # the folder has no work list to run it from: the words say what to do here
    assert 'Run Batch Extract & Design on it again, then press Refresh first.' in page.decisionLabel.text()
    assert 'Extract & design from the work list' not in page.decisionLabel.text() + page.stateLabel.text()
    assert 'run Batch Extract & Design on it again, then press Refresh' in page.stateLabel.text()

    write_entry(_queue(tmp_path), 'a')                       # what Batch Extract & Design writes: a new pending entry
    assert window.refresh() is True

    assert page.current_id == 'a' and page.revised_here == {}
    assert page.acceptButton.isEnabled() and page.accept() is True
    assert read_entry(_queue(tmp_path), 'a').status == 'accepted'


def test_a_title_sent_back_for_redesign_is_not_the_next_one_to_decide_and_a_reopened_one_is(qtbot, tmp_path):
    window = _folder(qtbot, tmp_path, entries=(('a', {}), ('b', {'status': 'accepted'}), ('c', {'status': 'accepted'})),
                     ask_revise=_Ask('design'))
    page = window.page
    page.show_title('b')
    assert page.revise() is True
    page.show_title('c')
    page._hooks.ask_revise = _Ask('review')
    assert page.revise() is True
    page.show_title('a')
    assert page.skip() is True                                # a leaves; the next waiting one is c (reopened), not b (redesign)
    assert page.current_id == 'c'


def test_the_hold_is_kept_across_an_edit_of_the_metadata_the_page_made_itself(qtbot, tmp_path):
    window = _folder(qtbot, tmp_path, entries=(('a', {'status': 'accepted'}),), ask_revise=_Ask('design'))
    page = window.page
    assert page.revise() is True
    update_entry(_queue(tmp_path), 'a', meta={'title': 'Renamed', 'year': '2001', 'audio_types': ['DD 5.1']})   # (a metadata edit)
    window.refresh()
    assert page.revised_here == {'a': 'design'} and not page.acceptButton.isEnabled()


def test_in_the_work_list_the_hold_also_goes_when_the_entry_is_designed_again_and_stays_while_it_is_not(qtbot, tmp_path):
    from test_worklist_title import _open, _window
    window = _window(qtbot, tmp_path, [('r-alien', {'status': 'accepted'}), ('r-arrival', {})])
    page = _open(qtbot, window, 'r-alien')
    page._hooks.ask_revise = _Ask('design')
    assert page.revise() is True
    page.reload()
    assert page.revised_here == {'r-alien': 'design'} and not page.acceptButton.isEnabled()
    assert 'Run Extract & design from the work list first.' in page.decisionLabel.text()      # the work list's own words, unchanged

    write_entry(_queue(tmp_path), 'r-alien')                 # a run designed it again
    page.reload()
    assert page.revised_here == {}


# --- 3. the revise question tells the truth about a published title's files (real git) -------------------------------------------------------

def _published_and_committed(tmp_path, repos):
    ''' `one` committed to the catalogue, `two` written into the working trees and not committed. '''
    xml, _, images, _ = repos
    queue = _queue(tmp_path)
    for entry_id, title in (('one', 'Heat'), ('two', 'Alien')):
        _queue_entry(queue, entry_id, title)
    kwargs = dict(xml_dir='xml', image_dir='img', image_owner=OWNER, image_repo_name=IMAGES_NAME, images_repo=images)
    publish_library(queue, xml, ids=['one'], **kwargs)
    commit_library(queue, xml, images_repo=images, xml_dir='xml', image_dir='img', push=False, ids=['one'])
    publish_library(queue, xml, ids=['two'], **kwargs)
    return xml, images


def test_the_row_of_a_published_title_says_where_its_files_are_in_git(tmp_path, repos):
    xml, _ = _published_and_committed(tmp_path, repos)
    assert commit_states(['one', 'two'], xml.local_path, 'xml', 'img') == {'one': 'committed', 'two': 'uncommitted'}
    assert commit_states([], xml.local_path) == {}
    assert commit_states(['one'], '') == {'one': 'none'}                           # no repository: as the index says
    not_git = tmp_path / 'not-a-repo'
    not_git.mkdir()
    assert commit_states(['one'], str(not_git), 'xml', 'img') == {'one': 'unknown'}        # degrades, never raises
    assert commit_states(['one'], str(tmp_path / 'missing'), 'xml', 'img') == {'one': 'unknown'}
    assert entry_row(read_entry(_queue(tmp_path), 'one'), 'committed').commit_state == 'committed'
    update_entry(_queue(tmp_path), 'one', status='accepted', chosen_candidate_index=0)
    assert entry_row(read_entry(_queue(tmp_path), 'one'), 'committed').commit_state == 'none'   # (only a published entry has one)


def test_the_revise_dialog_says_what_will_happen_to_the_files_and_that_is_what_happens(qtbot, tmp_path, repos):
    xml, images = _published_and_committed(tmp_path, repos)
    prefs = _prefs(tmp_path)
    prefs.set(LIBRARY_PROFILE_PATH, _profile(tmp_path, xml.local_path, images.local_path))
    ask = _Ask('review')
    window = ReviewFolderWindow(None, prefs, ask_revise=ask)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    # (the entries were published before the window read the folder)
    for entry_id in ('one', 'two'):
        update_entry(_queue(tmp_path), entry_id, status='published')
    window.refresh()
    assert window._rows['one'].commit_state == 'committed' and window._rows['two'].commit_state == 'uncommitted'

    page = window.page
    page.show_title('one')
    assert page.revise() is True
    summary = ask.asked[0][0]
    assert (summary.published, summary.in_catalogue) == (1, 1)
    body = revise_text('review', summary)[1]
    assert 'already committed to the catalogue' in body and 'files stay' in body and 'taken out' not in body
    assert os.path.isfile(os.path.join(xml.local_path, 'xml', 'one.xml'))               # they stayed
    assert read_entry(_queue(tmp_path), 'one').revision == 1                           # ... and a revision began

    page.show_title('two')
    assert page.revise() is True
    summary = ask.asked[1][0]
    assert (summary.published, summary.in_catalogue) == (1, 0)
    body = revise_text('review', summary)[1]
    assert 'published but not committed' in body and 'taken out of the repositories' in body
    assert not os.path.exists(os.path.join(xml.local_path, 'xml', 'two.xml'))           # ... and they were
    assert read_entry(_queue(tmp_path), 'two').revision == 0


def test_the_question_reads_git_when_it_is_asked_not_when_the_folder_was_read(qtbot, tmp_path, repos):
    ''' Somebody committed in another terminal since the window read the folder. '''
    xml, images = _published_and_committed(tmp_path, repos)
    prefs = _prefs(tmp_path)
    prefs.set(LIBRARY_PROFILE_PATH, _profile(tmp_path, xml.local_path, images.local_path))
    update_entry(_queue(tmp_path), 'two', status='published')
    ask = _Ask('review')
    window = ReviewFolderWindow(None, prefs, ask_revise=ask)
    qtbot.addWidget(window)
    assert window._rows['two'].commit_state == 'uncommitted'
    commit_library(_queue(tmp_path), xml, images_repo=images, xml_dir='xml', image_dir='img', push=False, ids=['two'])
    window.page.show_title('two')
    assert window.page.revise() is True
    assert (ask.asked[0][0].published, ask.asked[0][0].in_catalogue) == (1, 1)


# --- 4. the library profile is read again where it is used --------------------------------------------------------------------------------

def test_a_repository_set_in_the_work_list_after_the_window_opened_is_seen_and_one_removed_is_not_used(
        qtbot, tmp_path, repos, monkeypatch):
    xml, images = repos[0], repos[2]
    window = _folder(qtbot, tmp_path, entries=(('a', {'status': 'accepted'}),))
    monkeypatch.setattr(window, 'isActiveWindow', lambda: True)       # (the offscreen platform does not always keep the focus)
    assert window._revise_context().xml_repo is None and not window.publishButton.isEnabled()

    window._preferences.set(LIBRARY_PROFILE_PATH, _profile(tmp_path, xml.local_path, images.local_path))
    assert window._revise_context().xml_repo.local_path == xml.local_path          # the revise question, at once
    assert window._revise_context().images_repo.local_path == images.local_path
    assert window._work_dir() == str(tmp_path / 'work')
    window._preferences.set(LIBRARY_PROFILE_PATH, _profile(tmp_path, xml.local_path, images.local_path, name='moved.json',
                                                           work_dir=str(tmp_path / 'elsewhere')))
    assert window._work_dir() == str(tmp_path / 'elsewhere')                       # (projects are looked for where the profile says now)
    window._preferences.set(LIBRARY_PROFILE_PATH, _profile(tmp_path, xml.local_path, images.local_path))
    QApplication.sendEvent(window, QEvent(QEvent.Type.ActivationChange))            # the person comes back to this window
    assert window.publishButton.isEnabled()

    window._preferences.set(LIBRARY_PROFILE_PATH, _profile(tmp_path, name='no-repos.json'))
    assert window._revise_context().xml_repo is None and window._revise_context().images_repo is None
    QApplication.sendEvent(window, QEvent(QEvent.Type.ActivationChange))
    assert not window.publishButton.isEnabled()


def test_the_metadata_defaults_are_read_again_after_a_moment_and_the_badge_follows(qtbot, tmp_path):
    now = [100.0]
    write_entry(_queue(tmp_path), 'a', meta={'title': 'Alien', 'year': '2001'})            # no audio types: incomplete
    window = _folder(qtbot, tmp_path, entries=(), clock=lambda: now[0])
    assert window.page.badgeLabel.text().startswith('Not ready to publish') or 'audio type' in window.page.badgeLabel.text()
    window._preferences.set(LIBRARY_PROFILE_PATH, _profile(tmp_path))
    with open(window._preferences.get(LIBRARY_PROFILE_PATH), 'r+') as f:
        config = json.load(f)
        config['run']['meta_defaults'] = {'audio_types': ['DD 5.1']}
        f.seek(0), f.truncate(), json.dump(config, f)
    assert window._meta_defaults() is None                    # read a moment ago: a keystroke does not read the file again
    now[0] += 5
    assert window._meta_defaults() == {'audio_types': ['DD 5.1']}


def test_the_project_badge_is_read_again_when_the_folder_window_becomes_active(qtbot, tmp_path, monkeypatch):
    ''' The page tells the user "This page notices when you come back": here too. '''
    mono, _ = write_projects(tmp_path / 'work', 'a-alien')
    window = _folder(qtbot, tmp_path, open_project=lambda path: True)
    monkeypatch.setattr(window, 'isActiveWindow', lambda: True)
    page = window.page
    assert page.actions_bar.projectBadge.text() == 'Projects as designed'
    edit_project(mono)
    assert page.actions_bar.projectBadge.text() == 'Projects as designed'              # nothing has told the page yet
    QApplication.sendEvent(window, QEvent(QEvent.Type.ActivationChange))
    assert page.actions_bar.projectBadge.text() == 'Modified since design: mono project'


# --- 5. Publish is the work list's code path -----------------------------------------------------------------------------------------------

def test_publish_is_given_the_profiles_analysis_config_as_a_run_is(qtbot, tmp_path, repos, monkeypatch):
    xml, images = repos[0], repos[2]
    prefs = _prefs(tmp_path)
    prefs.set(LIBRARY_PROFILE_PATH, _profile(tmp_path, xml.local_path, images.local_path, analysis={'target_fs': 500}))
    seen = []
    real = review_module.publish_library
    monkeypatch.setattr(review_module, 'publish_library', lambda *a, **k: seen.append(k['config']) or real(*a, **k))
    _queue_entry(_queue(tmp_path), 'one', 'Heat')
    window = ReviewFolderWindow(None, prefs)
    qtbot.addWidget(window)
    answer = _Answer(True)  # (kept: its timer answers the confirmation)
    with qtbot.waitSignal(window.published, timeout=60000):
        window.publish_accepted()
    assert seen[0].target_fs == 500 and seen[0] == window._setup.settings.config


def test_an_edited_project_whose_audio_is_gone_is_refused_not_published_from_the_candidate_without_the_edit(
        qtbot, tmp_path, repos):
    xml, _, images, _ = repos
    window = _repo_window(qtbot, tmp_path, repos)
    mono, _ = write_projects(tmp_path / 'work', 'one')
    edit_project(mono)                                       # a person's edit ...
    os.remove(os.path.join(tmp_path, 'work', 'one', 'mono.wav'))      # ... whose audio has since gone (a cleaned work directory)
    answer = _Answer(True)  # (kept: its timer answers the confirmation)
    with qtbot.waitSignal(window.published, timeout=60000):
        window.publish_accepted()

    assert _statuses(window) == {'one': 'accepted', 'two': 'published'}               # only `two`
    assert not os.path.exists(os.path.join(xml.local_path, 'xml', 'one.xml'))
    assert '1 could not be published' in window.statusLabel.text()
    shown = window.resultsBox.toPlainText()
    assert window.resultsBox.isVisibleTo(window) and shown.startswith('one:')
    assert 'one.mono.beq is there but the audio it was made from' in shown and 'change made in the project' in shown


def test_how_each_accepted_entry_is_split_for_publish(tmp_path):
    work = tmp_path / 'work'
    write_projects(work, 'with-audio')                                        # the mono project and its audio
    write_projects(work, 'gone')
    os.remove(work / 'gone' / 'mono.wav')                                     # the project without its audio
    (work / 'plain').mkdir()                                                  # a work directory of the title, with no project
    with_projects, without, refused = split_for_publish(str(work), ['with-audio', 'gone', 'plain', 'never-extracted'])
    assert (with_projects, without) == (['with-audio'], ['plain', 'never-extracted'])
    assert [r['id'] for r in refused] == ['gone'] and refused[0]['error'] == 'publish_failed'
    assert split_for_publish(None, ['gone', 'x']) == ([], ['gone', 'x'], [])       # no work directory: nothing to read
    assert split_for_publish('', ['x']) == ([], ['x'], [])


def test_the_summary_of_a_published_title_in_the_folder_counts_as_in_the_catalogue_only_when_committed():
    assert ReviseSummary.of([('published', 'committed')]).in_catalogue == 1
    assert ReviseSummary.of([('published', 'uncommitted')]).in_catalogue == 0
    assert ReviseSummary.of([('published', 'unknown')]).in_catalogue == 0
