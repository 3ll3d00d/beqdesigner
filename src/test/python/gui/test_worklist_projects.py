'''
model/worklist_projects.py and model/worklist_title_actions.py's project half, chunk 27c: *Open project* (mono and multichannel) on
the title page, opened through the callable the window is given by its parent (a stub here; the real `BeqDesigner` in the last
tests), and the "modified since design" badge read from `read_project_filter()`. Real `.beq` project files (the pipeline's own
writer) and real hand edits; nothing about a project is faked. `import ui.beq` first: see AGENTS.md gotcha 3.
'''
import ui.beq  # noqa: F401 (must come first)

import logging

import pytest
from qtpy.QtCore import QEvent
from qtpy.QtWidgets import QApplication, QFileDialog, QMessageBox

from model.preferences import SYSTEM_CHECK_FOR_UPDATES
from model.worklist_projects import ProjectState, badge, project_states, read_state
from test_worklist_title import REVIEWABLE, _designer, _open, _prefs, _rows, _window, SOURCES, NOW  # noqa: F401
from worklist_fixture import make_index
from worklist_project_fixture import corrupt_project, edit_project, resave_unchanged, write_projects
from worklist_title_fixture import write_entry


def _work(tmp_path):
    return tmp_path / 'work'


# --- what is on disk, with no widgets -------------------------------------------------------------------------------------------

def test_a_missing_project_is_not_there_and_not_an_error(tmp_path):
    state = read_state('mono', str(tmp_path / 'nope.beq'))
    assert (state.exists, state.edited, state.error, state.readable) == (False, False, '', False)


def test_a_project_the_pipeline_wrote_is_as_designed_and_one_a_person_saved_is_modified(tmp_path):
    mono, _ = write_projects(_work(tmp_path), 'a')
    assert read_state('mono', mono) == ProjectState('mono', mono, True, False, '')
    edit_project(mono)
    assert read_state('mono', mono).edited is True and read_state('mono', mono).readable


def test_a_project_saved_again_without_a_change_counts_as_modified_because_the_app_writes_no_stamp(tmp_path):
    mono, _ = write_projects(_work(tmp_path), 'a')
    resave_unchanged(mono)
    assert read_state('mono', mono).edited is True


def test_a_project_that_cannot_be_read_is_said_not_raised(tmp_path):
    mono, _ = write_projects(_work(tmp_path), 'a')
    corrupt_project(mono)
    state = read_state('mono', mono)
    assert state.exists and not state.readable and state.error and not state.edited


def test_a_valid_gzip_that_is_not_a_project_is_also_said_not_raised(tmp_path):
    import gzip
    path = tmp_path / 'not-a-project.beq'
    with gzip.open(path, 'wb') as f:
        f.write(b'[{}]')                                    # readable, but no filter presets: a KeyError, not an OSError
    state = read_state('mono', str(path))
    assert state.exists and not state.readable and 'KeyError' in state.error


def test_the_multichannel_project_belongs_to_a_title_with_a_multichannel_extraction_or_a_project_of_its_own(tmp_path):
    work = _work(tmp_path)
    write_projects(work, 'film')
    assert [s.kind for s in project_states(str(work), 'film')] == ['mono']
    write_projects(work, 'surround', multichannel=True)
    assert [(s.kind, s.exists) for s in project_states(str(work), 'surround')] == [('mono', True), ('multichannel', True)]
    # the multichannel.wav is there and the project not written yet: the title is multichannel, the project does not exist
    import os
    os.remove(project_states(str(work), 'surround')[1].path)
    assert [(s.kind, s.exists) for s in project_states(str(work), 'surround')] == [('mono', True), ('multichannel', False)]
    # the project is there and the wav is gone (re-extracted): it is still shown
    write_projects(work, 'orphan', multichannel=True)
    os.remove(os.path.join(str(work), 'orphan', 'multichannel.wav'))
    assert [s.kind for s in project_states(str(work), 'orphan')] == ['mono', 'multichannel']
    assert project_states('', 'film') == [] and project_states(str(work), '') == []


def test_the_badge_words(tmp_path):
    work = _work(tmp_path)
    mono, mc = write_projects(work, 'a', multichannel=True)
    assert badge([]) == ('', 'neutral')
    assert badge(project_states(str(work), 'a')) == ('Projects as designed', 'ok')
    edit_project(mc)
    assert badge(project_states(str(work), 'a')) == ('Modified since design: multichannel project', 'warn')
    edit_project(mono)
    assert badge(project_states(str(work), 'a'))[0] == 'Modified since design: mono and multichannel project'
    corrupt_project(mono)
    text, level = badge(project_states(str(work), 'a'))
    assert text.startswith('The mono project could not be read') and level == 'warn'
    import os
    os.remove(mono)
    os.remove(mc)
    assert badge(project_states(str(work), 'a')) == ('No project yet: it is written when the title is designed.', 'neutral')


# --- the page --------------------------------------------------------------------------------------------------------------------

class _Opener:
    ''' The stand-in for `BeqDesigner.open_project_file`. '''

    def __init__(self, result=True, error=None):
        self.opened, self.result, self.error = [], result, error

    def __call__(self, path):
        self.opened.append(path)
        if self.error:
            raise self.error
        return self.result


def _page(qtbot, tmp_path, opener, title_id='r-alien', **projects):
    window = _window(qtbot, tmp_path, REVIEWABLE, open_project=opener)
    return window, _open(qtbot, window, title_id)


def test_a_title_with_no_project_has_the_buttons_disabled_and_says_why(qtbot, tmp_path):
    opener = _Opener()
    _, page = _page(qtbot, tmp_path, opener)
    bar = page.actions_bar
    assert not bar.monoButton.isEnabled() and not bar.multichannelButton.isVisible()
    assert 'no mono project yet' in bar.monoButton.toolTip().lower()
    assert bar.projectBadge.text().startswith('No project yet')
    assert page.open_project('mono') is False and opener.opened == []
    assert 'no mono project' in bar.messageLabel.text().lower()


def test_the_mono_project_opens_through_the_callable_and_the_page_says_how_to_keep_an_edit(qtbot, tmp_path):
    opener = _Opener()
    mono, _ = write_projects(_work(tmp_path), 'r-alien')
    _, page = _page(qtbot, tmp_path, opener)
    bar = page.actions_bar
    assert bar.monoButton.isEnabled() and not bar.multichannelButton.isVisible()
    assert bar.projectBadge.text() == 'Projects as designed'

    qtbot.mouseClick(bar.monoButton, __import__('qtpy.QtCore', fromlist=['Qt']).Qt.MouseButton.LeftButton)

    assert opener.opened == [mono]
    assert 'Opened the mono project' in bar.messageLabel.text() and mono in bar.messageLabel.text()
    assert 'Save Project' in bar.messageLabel.text()


def test_the_multichannel_project_is_offered_for_a_multichannel_title_and_opens_that_file(qtbot, tmp_path):
    opener = _Opener()
    mono, mc = write_projects(_work(tmp_path), 'r-alien', multichannel=True)
    _, page = _page(qtbot, tmp_path, opener)
    bar = page.actions_bar
    assert bar.monoButton.isEnabled() and bar.multichannelButton.isVisible() and bar.multichannelButton.isEnabled()
    assert page.open_project('multichannel') is True
    assert opener.opened == [mc]
    assert page.open_project('mono') is True and opener.opened == [mc, mono]


def test_the_multichannel_button_is_disabled_until_its_project_is_written(qtbot, tmp_path):
    import os
    opener = _Opener()
    mono, mc = write_projects(_work(tmp_path), 'r-alien', multichannel=True)
    os.remove(mc)
    _, page = _page(qtbot, tmp_path, opener)
    bar = page.actions_bar
    assert bar.multichannelButton.isVisible() and not bar.multichannelButton.isEnabled()
    assert page.open_project('multichannel') is False and opener.opened == []
    assert bar.monoButton.isEnabled()


@pytest.mark.parametrize('edit, expected', [
    ('mono', 'Modified since design: mono project'),
    ('multichannel', 'Modified since design: multichannel project'),
    ('both', 'Modified since design: mono and multichannel project')])
def test_the_badge_says_which_project_was_modified_and_what_it_means(qtbot, tmp_path, edit, expected):
    mono, mc = write_projects(_work(tmp_path), 'r-alien', multichannel=True)
    for kind, path in (('mono', mono), ('multichannel', mc)):
        if edit in (kind, 'both'):
            edit_project(path)
    _, page = _page(qtbot, tmp_path, _Opener())
    badge_label = page.actions_bar.projectBadge
    assert badge_label.text() == expected
    tip = badge_label.toolTip()
    assert 'published' in tip and 'redesign keeps' in tip and 'bulk accept leaves' in tip and 'needs attention' in tip
    assert 'saved again without a change' in tip


def test_an_unreadable_project_is_said_and_cannot_be_opened(qtbot, tmp_path):
    opener = _Opener()
    mono, _ = write_projects(_work(tmp_path), 'r-alien')
    corrupt_project(mono)
    _, page = _page(qtbot, tmp_path, opener)
    bar = page.actions_bar
    assert bar.projectBadge.text().startswith('The mono project could not be read')
    assert not bar.monoButton.isEnabled() and 'could not be read' in bar.monoButton.toolTip()
    assert page.open_project('mono') is False and opener.opened == []
    assert 'could not be read' in bar.messageLabel.text()


def test_without_a_route_to_the_main_window_the_buttons_are_disabled_and_say_so(qtbot, tmp_path):
    write_projects(_work(tmp_path), 'r-alien')
    window = _window(qtbot, tmp_path, REVIEWABLE)       # no open_project given
    page = _open(qtbot, window, 'r-alien')
    assert not page.actions_bar.monoButton.isEnabled() and 'main window' in page.actions_bar.monoButton.toolTip()
    assert page.open_project('mono') is False and 'main window' in page.actions_bar.messageLabel.text()


def test_a_route_that_raises_or_declines_is_said_and_never_raised(qtbot, tmp_path):
    mono, _ = write_projects(_work(tmp_path), 'r-alien')
    _, page = _page(qtbot, tmp_path, _Opener(error=ValueError('not a project')))
    assert page.open_project('mono') is False
    assert 'Could not open the mono project: ValueError: not a project' in page.actions_bar.messageLabel.text()
    page.actions_bar.show_message('')

    declining = _Opener(result=False)
    page._hooks.open_project = declining
    assert page.open_project('mono') is False and declining.opened == [mono]
    assert page.actions_bar.messageLabel.text() == 'The mono project was not opened.'


def test_the_badge_is_read_again_when_the_window_becomes_active_after_editing_in_the_main_window(qtbot, tmp_path):
    mono, _ = write_projects(_work(tmp_path), 'r-alien')
    window, page = _page(qtbot, tmp_path, _Opener())
    assert page.actions_bar.projectBadge.text() == 'Projects as designed'

    edit_project(mono)      # in the main window, and saved over the same file
    assert page.actions_bar.projectBadge.text() == 'Projects as designed'     # nothing has told the page yet

    QApplication.sendEvent(window, QEvent(QEvent.Type.ActivationChange))    # the person comes back to this window

    assert page.actions_bar.projectBadge.text() == 'Modified since design: mono project'


def test_the_badge_is_read_for_each_title_shown(qtbot, tmp_path):
    other, _ = write_projects(_work(tmp_path), 'r-arrival')
    edit_project(other)
    write_projects(_work(tmp_path), 'r-alien')
    _, page = _page(qtbot, tmp_path, _Opener())
    assert page.actions_bar.projectBadge.text() == 'Projects as designed'
    assert page.next() and page.current_id == 'r-arrival'
    assert page.actions_bar.projectBadge.text() == 'Modified since design: mono project'
    assert page.next() and page.current_id == 'r-sicario'
    assert page.actions_bar.projectBadge.text().startswith('No project yet')


def test_the_project_buttons_never_take_enter_from_a_field(qtbot, tmp_path):
    ''' Every button is non-default, as on the rest of the page (design.md §12.1): Enter in a box presses none. '''
    _, page = _page(qtbot, tmp_path, _Opener())
    bar = page.actions_bar
    assert not any(b.autoDefault() for b in (bar.monoButton, bar.multichannelButton, bar.reviseButton))


# --- through the real BeqDesigner ------------------------------------------------------------------------------------------------

def _main_window(qtbot, tmp_path, monkeypatch, entries):
    import app as app_module
    make_index(tmp_path / 'work', _rows(), SOURCES, generation=2, last_scan_at=NOW - 900)
    for entry_id, options in entries:
        write_entry(str(tmp_path / 'queue'), entry_id, **options)
    prefs = _prefs(tmp_path)
    prefs.set(SYSTEM_CHECK_FOR_UPDATES, False)
    main = app_module.BeqDesigner(QApplication.instance(), prefs)
    qtbot.addWidget(main)
    monkeypatch.setattr(main, '_BeqDesigner__check_ffmpeg_available', lambda: True)
    main.showWorkListWindow()
    window = main._BeqDesigner__work_list
    qtbot.addWidget(window)
    return main, window


@pytest.fixture
def root_logging():
    root = logging.getLogger()
    handlers = list(root.handlers)
    yield
    for handler in list(root.handlers):
        if handler not in handlers:
            root.removeHandler(handler)


def test_the_real_main_window_loads_the_project_the_page_opens_and_save_project_offers_that_file(
        qtbot, tmp_path, monkeypatch, root_logging):
    mono, _ = write_projects(_work(tmp_path), 'r-alien')
    main, window = _main_window(qtbot, tmp_path, monkeypatch, REVIEWABLE)
    page = _open(qtbot, window, 'r-alien')
    assert len(main._BeqDesigner__signal_model) == 0

    assert page.open_project('mono') is True

    assert len(main._BeqDesigner__signal_model) == 1              # the project is in the main window now
    assert 'Loaded' in main.statusbar.currentMessage() and mono in main.statusbar.currentMessage()
    assert page.actions_bar.messageLabel.text().startswith('Opened the mono project')
    offered = []
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args, **kwargs: offered.append(args[args.index('Export Project') + 1]) or ('', ''))
    main.exportProject()
    assert offered == [mono]                                      # saved over the same file with one click

    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args, **kwargs: offered.append(args[args.index('Export Project') + 1]) or ('', ''))
    main._BeqDesigner__project_path = None
    main.exportProject()
    assert offered[-1] == 'project.beq'                           # a project that did not come from the work list: as before


def _offered(main, monkeypatch):
    ''' What Save Project would suggest as the file name (the dialog is answered "cancel"). '''
    offered = []
    monkeypatch.setattr(QFileDialog, 'getSaveFileName',
                        lambda *args, **kwargs: offered.append(args[args.index('Export Project') + 1]) or ('', ''))
    main.exportProject()
    return offered[-1]


def test_save_project_stops_offering_the_titles_file_once_the_signals_are_no_longer_the_ones_it_loaded(
        qtbot, tmp_path, monkeypatch, root_logging):
    ''' One careless click on Save Project must not overwrite a title's project with something else (what is saved is published). '''
    from model.iir import CompleteFilter, PeakingEQ
    mono, _ = write_projects(_work(tmp_path), 'r-alien')
    main, window = _main_window(qtbot, tmp_path, monkeypatch, REVIEWABLE)
    model = main._BeqDesigner__signal_model
    assert main.open_project_file(mono) is True
    assert _offered(main, monkeypatch) == mono

    # editing the filter is what the project was opened for: the file is still offered
    model[0].filter = CompleteFilter(fs=model[0].fs, filters=[PeakingEQ(model[0].fs, 40, 1, -2.0)])
    assert _offered(main, monkeypatch) == mono

    model.delete([0])                                  # the signal is gone: whatever is loaded now is not that project
    assert len(model) == 0 and _offered(main, monkeypatch) == 'project.beq'
    assert main.open_project_file(mono) is True        # opening it again offers it again
    assert _offered(main, monkeypatch) == mono


def test_the_real_main_window_asks_before_replacing_what_is_loaded_and_declining_changes_nothing(
        qtbot, tmp_path, monkeypatch, root_logging):
    mono, _ = write_projects(_work(tmp_path), 'r-alien')
    main, window = _main_window(qtbot, tmp_path, monkeypatch, REVIEWABLE)
    page = _open(qtbot, window, 'r-alien')
    assert main.open_project_file(mono) is True and len(main._BeqDesigner__signal_model) == 1
    asked = []
    monkeypatch.setattr(QMessageBox, 'question', lambda *args, **kwargs: asked.append(args[2]) or QMessageBox.StandardButton.No)

    assert page.open_project('mono') is False

    assert len(asked) == 1 and 'replaces the signals and filters loaded here' in asked[0]
    assert page.actions_bar.messageLabel.text() == 'The mono project was not opened.'
    assert len(main._BeqDesigner__signal_model) == 1

    monkeypatch.setattr(QMessageBox, 'question', lambda *args, **kwargs: QMessageBox.StandardButton.Yes)
    assert page.open_project('mono') is True


def test_the_real_main_window_refuses_a_file_that_is_not_a_project_and_the_page_says_so(
        qtbot, tmp_path, monkeypatch, root_logging):
    mono, _ = write_projects(_work(tmp_path), 'r-alien')
    main, window = _main_window(qtbot, tmp_path, monkeypatch, REVIEWABLE)
    page = _open(qtbot, window, 'r-alien')
    with open(mono, 'wb') as f:
        f.write(b'garbage')
    # (the badge would already say so; the route is asked anyway, as the bar allows it while the file is unreadable only if
    # the page did not see it: force the case where the file broke after the page read it)
    page.refresh_projects()
    page._hooks.open_project = main.open_project_file
    from model.worklist_projects import ProjectState
    page.project_states = lambda: [ProjectState('mono', mono, True)]
    assert page.open_project('mono') is False
    assert 'Could not open the mono project' in page.actions_bar.messageLabel.text()
    assert len(main._BeqDesigner__signal_model) == 0
