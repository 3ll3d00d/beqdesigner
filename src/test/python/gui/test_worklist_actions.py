'''
model/worklist.py, chunk 26b: the work list's actions -- selection, the action button, the run on a worker thread with
progress, the running-row marker and Cancel, the failures panel and Retry failed, and Publish and Commit behind
confirmations that name the repositories. Driven like a user would against a fixture discovery index with a FAKE pipeline
handed to the window as `run_stages_fn` (nothing here runs ffmpeg, a designer or git), plus one test that runs the real
`run_stages` against a scanned file that does not exist. `import ui.beq` first: see AGENTS.md gotcha 3.
'''
import ui.beq  # noqa: F401 (must come first)

import logging
import os
import sqlite3
import threading
import time
from typing import Dict, List, Optional

import pytest
from qtpy.QtCore import QSettings, Qt, QThreadPool, QTimer
from qtpy.QtWidgets import QApplication, QMessageBox

from model.preferences import DESIGNER_DEFAULT, DESIGNER_QUEUE_DIR, LIBRARY_FILESYSTEM_GLOBS, LIBRARY_IMAGES_REPO, \
    LIBRARY_WORK_DIR, LIBRARY_XML_REPO, Preferences, SYSTEM_CHECK_FOR_UPDATES, WORKLIST_PUSH
from model.worklist import WorkListWindow
from model.worklist_confirm import ConfirmDialog
from model.worklist_model import COL_NEEDS, RUNNING_ROLE
from pipeline.designer.registry import register_designer, unregister_designer
from pipeline.library.commit import CatalogueCommit, RepoCommit
from pipeline.library.index import LibraryIndex, index_path
from pipeline.library.run import LibraryRunReport
from pipeline.library.source import LibraryItem
from pipeline.library.stages import Progress, StagesReport
from worklist_fixture import make_index, title_row

NOW = 1_800_000_000.0
DAY = 86400.0
DESIGNER = 'test.worklist.actions'
SOURCES = [('films', 'filesystem', NOW - 900, NOW - 900, '', 20)]


@pytest.fixture(autouse=True)
def _designer():
    register_designer(DESIGNER, lambda request: None)
    yield
    unregister_designer(DESIGNER)


def _prefs(tmp_path, **more):
    prefs = Preferences(QSettings(str(tmp_path / 'settings.ini'), QSettings.Format.IniFormat))
    prefs.set(LIBRARY_WORK_DIR, str(tmp_path / 'work'))
    prefs.set(DESIGNER_QUEUE_DIR, str(tmp_path / 'queue'))
    prefs.set(LIBRARY_FILESYSTEM_GLOBS, [str(tmp_path / 'films' / '*.mkv')])
    prefs.set(DESIGNER_DEFAULT, DESIGNER)
    prefs.set(LIBRARY_XML_REPO, str(tmp_path / 'catalogue-xml'))
    prefs.set(LIBRARY_IMAGES_REPO, str(tmp_path / 'catalogue-images'))
    for key, value in more.items():
        prefs.set(key, value)
    (tmp_path / 'work').mkdir(exist_ok=True)
    return prefs


def _row(title_id, title, needs, days=1.0, **fields):
    return title_row(title_id, title, needs, state_since=NOW - days * DAY, **fields)


def _rows():
    ''' 3 extract, 2 design, 2 review, 3 publish (one out of date), 2 commit, 1 done, 3 attention (2 failed, 1 changed). '''
    return [
        _row('a-changed', 'Heat', 'attention', 9, detail='source changed since accepted'),
        _row('a-failed-x', 'Dune', 'attention', 8, detail='extract failed: file not found', extract_state='failed',
             failure='extract failed: file not found'),
        _row('a-failed-d', 'Ronin', 'attention', 7, detail='design failed: timed out', design_state='failed',
             failure='design failed: timed out'),
        _row('r-alien', 'Alien', 'review', 6),
        _row('r-arrival', 'Arrival', 'review', 5),
        _row('x-gravity', 'Gravity', 'extract', 4, detail='new'),
        _row('x-tenet', 'Tenet', 'extract', 3, detail='new'),
        _row('x-fury', 'Fury', 'extract', 2, detail='new'),
        _row('d-speed', 'Speed', 'design', 4, detail='extracted'),
        _row('d-twister', 'Twister', 'design', 3, detail='extracted'),
        _row('p-one', 'Sicario', 'publish', 3, detail='accepted', publish_state='not_written'),
        _row('p-two', 'Collateral', 'publish', 2, detail='accepted', publish_state='not_written'),
        _row('p-old', 'Baby Driver', 'publish', 1, detail='changed since published', publish_state='out_of_date'),
        _row('c-one', 'Jaws', 'commit', 2, detail='written, not committed', publish_state='written'),
        _row('c-two', 'Ronin 2', 'commit', 1, detail='written, not committed', publish_state='written'),
        _row('z-done', 'Old Film', 'done', 90, detail='pushed'),
    ]


class FakePipeline:
    '''
    Stands in for `run_stages`: records what it was asked, says which thread it ran on, reports progress the way the
    pipeline does, honours `should_cancel` between titles, and can be held mid-title or made to fail.
    '''

    def __init__(self, index_file: str, hold_at: Optional[int] = None, raises: Optional[Exception] = None,
                 fail: Optional[List[str]] = None, after=None):
        self.index_file = index_file
        self.hold_at, self.raises, self.fail, self.after = hold_at, raises, list(fail or ()), after
        self.calls: List[dict] = []
        self.thread = None
        self.entered = threading.Event()
        self.release = threading.Event()
        self.cancel_seen: List[bool] = []

    def __call__(self, profile, selection, through, *, run_config, index, publish=None, settings=None,
                 retry_failed=False, should_cancel=None, on_progress=None, **rest):
        ids = list(selection.ids)
        self.calls.append(dict(ids=ids, through=through, retry_failed=retry_failed, publish=publish,
                               run_config=run_config, settings=settings))
        self.thread = threading.current_thread()
        if self.raises:
            raise self.raises
        titles = {t.id: t.title for t in index.titles(ids=ids)}
        report = StagesReport(through, len(ids), run=LibraryRunReport())
        stage = 'publish' if through in ('publish', 'commit') else 'design'
        for n, title_id in enumerate(ids):
            if should_cancel():
                report.cancelled = True
                self.cancel_seen.append(True)
                break
            on_progress(Progress(n, len(ids), titles.get(title_id, title_id), stage, title_id))
            self.entered.set()
            if self.hold_at == n:
                assert self.release.wait(20), 'the test never released the pipeline'
            (report.run.failed if title_id in self.fail else report.run.designed).append(
                (title_id, 'RuntimeError: designer returned 503') if title_id in self.fail else title_id)
            report.attempted.append(title_id)
        if report.cancelled:
            report.not_run = [i for i in ids if i not in report.attempted]
        else:
            on_progress(Progress(len(ids), len(ids), '', '', ''))
        if self.after is not None:
            self.after(self.index_file, report)
        return report


def _update(index_file: str, ids, **columns) -> None:
    ''' What a real run does to the index: the pipeline's own refresh, done by hand on the fixture rows. '''
    db = sqlite3.connect(index_file)
    with db:
        for title_id in ids:
            db.execute(f'UPDATE titles SET {", ".join(f"{name} = ?" for name in columns)} WHERE id = ?',
                       [*columns.values(), title_id])
    db.close()


def _window(qtbot, tmp_path, pipeline=None, rows=None, prefs=None, **kwargs):
    prefs = prefs or _prefs(tmp_path)
    index_file = index_path(str(tmp_path / 'work'))
    if rows is not None or not os.path.exists(index_file):   # a test may have made the index already
        make_index(tmp_path / 'work', rows if rows is not None else _rows(), SOURCES, generation=2,
                   last_scan_at=NOW - 900)
    pipeline = pipeline if pipeline is not None else FakePipeline(index_file)
    window = WorkListWindow(None, prefs, auto_scan=False, clock=lambda: NOW, run_stages_fn=pipeline, **kwargs)
    qtbot.addWidget(window)
    window.show()
    return window, pipeline


def _click(qtbot, button):
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)


def _answer(accept: bool, seen: List[str], tick: Optional[bool] = None) -> None:
    '''
    Answers the modal ConfirmDialog the next action opens, as a person would: reads it, optionally (un)ticks its checkbox,
    then clicks its OK or Cancel button. Scheduled now, it runs once the dialog's event loop is going.
    '''
    def respond():
        dialog = QApplication.activeModalWidget()
        if not isinstance(dialog, ConfirmDialog):
            if dialog is not None:
                dialog.reject()
            seen.append(f'unexpected: {dialog!r}')
            return
        seen.append(dialog.text)
        seen.append(('checkbox', dialog.checkbox.isChecked() if dialog.checkbox is not None else None))
        if tick is not None:
            dialog.checkbox.setChecked(tick)
        (dialog.ok_button if accept else dialog.cancel_button).click()

    QTimer.singleShot(0, respond)


def _run_button_text(window) -> str:
    return window.runButton.text().replace('&&', '&')


def _cell(window, title_id: str, column: int, role=Qt.ItemDataRole.DisplayRole):
    return window.proxy.index(window.listed_ids().index(title_id), column).data(role)


# --- selecting ------------------------------------------------------------------------------------------------------------

def test_select_all_selects_everything_the_current_view_lists(qtbot, tmp_path):
    window, _ = _window(qtbot, tmp_path)
    _click(qtbot, window.publishChip)
    assert window.selected_ids() == []

    _click(qtbot, window.selectAllButton)

    assert window.selected_ids() == window.listed_ids() == ['p-one', 'p-two', 'p-old']
    assert window.selectionLabel.text() == '3 selected of 3 listed'
    _click(qtbot, window.clearSelectionButton)
    assert window.selected_ids() == []


def test_the_action_button_is_labelled_with_what_it_will_do_to_how_many_titles_and_says_what_it_skipped(qtbot, tmp_path):
    window, _ = _window(qtbot, tmp_path)
    window.select_ids(['x-gravity', 'x-tenet', 'd-speed', 'r-alien', 'p-one', 'a-changed'])

    # the machine tier runs (2 extract + 1 design), the rest is skipped and the button says why
    assert _run_button_text(window) == 'Extract & design 3 (3 of 6 skipped)'
    assert window.runButton.isEnabled()
    text = window.skippedLabel.text()
    assert text.startswith('3 of 6 skipped: ')
    for reason in ('1 waiting for review', '1 already accepted', '1 needs attention'):
        assert reason in text
    assert window.selectionLabel.text() == '6 selected of 15 listed'
    # the label is what plan_for says, which is what a run would do
    plan = window.plan_for('design')
    assert sorted(p.row.id for p in plan.planned) == ['d-speed', 'x-gravity', 'x-tenet']

    window.select_ids(['x-gravity', 'x-tenet'])
    assert _run_button_text(window) == 'Extract & design 2'
    assert not window.skippedLabel.isVisibleTo(window)


def test_with_nothing_selected_the_buttons_work_on_everything_the_view_lists_and_the_window_says_so(qtbot, tmp_path):
    window, _ = _window(qtbot, tmp_path)
    _click(qtbot, window.extractChip)

    assert window.selected_ids() == []
    assert window.selectionLabel.text() == 'None selected: the buttons work on all 3 listed'
    assert _run_button_text(window) == 'Extract & design 3'
    _click(qtbot, window.allChip)
    assert _run_button_text(window).startswith('Extract & design 5 (')  # the filter changed: so did the label


def test_publish_and_commit_buttons_count_only_the_titles_waiting_for_that_step(qtbot, tmp_path):
    window, _ = _window(qtbot, tmp_path)
    window.select_ids(['p-one', 'p-old', 'c-one', 'x-gravity'])

    assert window.publishButton.text() == 'Publish 2'
    assert window.commitButton.text() == 'Commit 1'
    assert window.publishButton.isEnabled() and window.commitButton.isEnabled()
    window.select_ids(['x-gravity'])
    assert window.publishButton.text() == 'Publish 0' and not window.publishButton.isEnabled()
    assert window.commitButton.text() == 'Commit 0' and not window.commitButton.isEnabled()


def test_publish_and_commit_are_disabled_with_a_reason_when_no_xml_repository_is_set(qtbot, tmp_path):
    prefs = _prefs(tmp_path, **{LIBRARY_XML_REPO: ''})
    window, _ = _window(qtbot, tmp_path, prefs=prefs)
    window.select_ids(['p-one', 'c-one'])

    assert not window.publishButton.isEnabled() and not window.commitButton.isEnabled()
    assert 'No XML repository is set' in window.publishButton.toolTip()


def test_the_selection_survives_a_reload_from_the_index(qtbot, tmp_path):
    window, pipeline = _window(qtbot, tmp_path)
    window.select_ids(['x-gravity', 'r-alien', 'p-two'])
    _update(pipeline.index_file, ['x-tenet'], needs='review', tier='human')

    window.refresh_from_index()

    assert sorted(window.selected_ids()) == ['p-two', 'r-alien', 'x-gravity']
    assert _cell(window, 'x-tenet', COL_NEEDS) == 'Review'   # and the list did change


# --- running ---------------------------------------------------------------------------------------------------------------

def test_a_run_happens_off_the_ui_thread_with_determinate_progress_and_a_running_row_marker(qtbot, tmp_path):
    index_file = make_index(tmp_path / 'work', _rows(), SOURCES, generation=2, last_scan_at=NOW - 900)
    pipeline = FakePipeline(index_file, hold_at=0,
                            after=lambda path, report: _update(path, report.run.designed, needs='review', tier='human'))
    window, _ = _window(qtbot, tmp_path, pipeline=pipeline, prefs=_prefs(tmp_path))
    window.select_ids(['x-gravity', 'x-tenet', 'x-fury'])

    _click(qtbot, window.runButton)

    qtbot.waitUntil(pipeline.entered.is_set, timeout=5000)
    qtbot.waitUntil(lambda: window.runProgress.value() == 1, timeout=5000)   # title 1 of 3 is in hand
    # the click returned while the pipeline is mid-title: it is not on the UI thread, and the UI still turns
    assert window.is_running
    assert pipeline.thread is not threading.current_thread()
    QApplication.processEvents()
    # determinate progress, and the status names the title and the stage
    assert window.runProgress.maximum() == 3 and window.runProgress.value() == 1
    assert window.runStatusLabel.text() == 'Designing Gravity  (1 of 3)'
    # the row shows its stage while it is worked on, and only that row
    assert window.model.running == {'x-gravity': 'design'}
    assert _cell(window, 'x-gravity', COL_NEEDS) == '▶ Designing...'
    assert _cell(window, 'x-gravity', COL_NEEDS, RUNNING_ROLE) == 'design'
    assert _cell(window, 'x-tenet', COL_NEEDS) == 'Extract'
    # what would conflict is disabled, and Cancel is offered
    for button in (window.runButton, window.publishButton, window.commitButton, window.retryButton,
                   window.rescanButton):
        assert not button.isEnabled(), button.objectName()
    assert window.cancelButton.isVisibleTo(window) and window.cancelButton.isEnabled()
    assert window.rescan() is False and window.run_selected() is False
    assert sorted(window.selected_ids()) == ['x-fury', 'x-gravity', 'x-tenet']    # never lost, however long it runs

    with qtbot.waitSignal(window.run_finished, timeout=10000) as run:
        pipeline.release.set()

    assert not window.is_running
    assert window.model.running == {}
    assert not window.cancelButton.isVisibleTo(window) and not window.runProgress.isVisibleTo(window)
    assert window.runStatusLabel.text() == 'Extract & design finished: 3 designed'
    assert run.args[0].run.designed == ['x-gravity', 'x-tenet', 'x-fury']
    # the window read the index again: the three moved from Extract to Review, without a rescan
    assert window.chip_counts()['Extract'] == 0 and window.chip_counts()['Review'] == 5
    assert _cell(window, 'x-tenet', COL_NEEDS) == 'Review'
    assert window.runButton.isEnabled() is False   # nothing selected is still listed as extractable
    assert pipeline.calls[0]['ids'] == ['x-gravity', 'x-tenet', 'x-fury'] and pipeline.calls[0]['through'] == 'design'
    assert pipeline.calls[0]['retry_failed'] is False


def test_the_run_is_handed_the_settings_a_scan_is_given_and_only_the_titles_it_will_work_on(qtbot, tmp_path):
    window, pipeline = _window(qtbot, tmp_path)
    window.select_ids(['x-gravity', 'r-alien', 'a-changed'])

    with qtbot.waitSignal(window.run_finished, timeout=10000):
        window.run_selected()

    call = pipeline.calls[0]
    assert call['ids'] == ['x-gravity']   # review and done are skipped and are not passed on
    assert call['run_config'].work_dir == str(tmp_path / 'work') and call['run_config'].designer == DESIGNER
    assert call['run_config'].queue_dir == str(tmp_path / 'queue')
    assert call['settings'] == window.setup.settings and call['publish'] is None
    # what was skipped is stated after the run
    assert '2 of 3 skipped: 1 needs attention, 1 waiting for review' in window.runStatusLabel.text()


def test_cancel_stops_after_the_current_title_and_reports_what_completed(qtbot, tmp_path):
    index_file = make_index(tmp_path / 'work', _rows(), SOURCES, generation=2, last_scan_at=NOW - 900)
    pipeline = FakePipeline(index_file, hold_at=1,
                            after=lambda path, report: _update(path, report.attempted, needs='review', tier='human'))
    window, _ = _window(qtbot, tmp_path, pipeline=pipeline, prefs=_prefs(tmp_path))
    window.select_ids(['x-gravity', 'x-tenet', 'x-fury', 'd-speed'])
    window.run_selected()
    qtbot.waitUntil(lambda: window.runProgress.value() == 2, timeout=5000)

    _click(qtbot, window.cancelButton)

    assert not window.cancelButton.isEnabled()
    assert 'Cancelling' in window.runStatusLabel.text()
    with qtbot.waitSignal(window.run_finished, timeout=10000) as run:
        pipeline.release.set()   # the title in hand (2) finishes; 3 and 4 are never started

    report = run.args[0]
    assert report.cancelled and len(report.attempted) == 2
    assert report.attempted == pipeline.calls[0]['ids'][:2]      # the first two, in order; the title in hand finished
    assert pipeline.cancel_seen == [True]
    assert len(report.not_run) == 2
    assert window.runStatusLabel.text() == 'Stopped after 2 of 4 titles (2 not run): 2 designed'
    lines = {line.id: line for line in window.results}
    assert [line.outcome for line in lines.values() if line.id in report.not_run] == ['Not run', 'Not run']
    assert not window.is_running and window.model.running == {}
    # the two that finished are reviewed now; the two that did not are still waiting
    assert window.chip_counts()['Review'] == 4 and window.chip_counts()['Extract'] + window.chip_counts()['Design'] == 3


def test_a_run_that_raises_shows_the_error_reloads_and_leaves_the_window_usable(qtbot, tmp_path):
    index_file = make_index(tmp_path / 'work', _rows(), SOURCES, generation=2, last_scan_at=NOW - 900)
    pipeline = FakePipeline(index_file, raises=RuntimeError('the designer is unreachable'))
    window, _ = _window(qtbot, tmp_path, pipeline=pipeline, prefs=_prefs(tmp_path))
    window.select_ids(['x-gravity', 'x-tenet'])

    with qtbot.waitSignal(window.run_failed, timeout=10000) as failed:
        window.run_selected()

    assert failed.args[0] == 'RuntimeError: the designer is unreachable'
    assert 'The run failed: RuntimeError: the designer is unreachable' in window.runStatusLabel.text()
    assert 'Help > Logs' in window.runStatusLabel.text()
    assert not window.is_running and window.model.running == {}
    assert window.runButton.isEnabled() and window.rescanButton.isEnabled()
    assert sorted(window.selected_ids()) == ['x-gravity', 'x-tenet']    # never lose the selection
    assert not window.cancelButton.isVisibleTo(window)


def test_titles_that_fail_are_listed_in_the_failures_panel_with_the_reason_and_the_stage(qtbot, tmp_path):
    index_file = make_index(tmp_path / 'work', _rows(), SOURCES, generation=2, last_scan_at=NOW - 900)

    def fail_them(path, report):
        _update(path, ['x-fury'], needs='attention', tier='attention', extract_state='failed',
                detail='extract failed: no audio stream', failure='extract failed: no audio stream')
        with LibraryIndex(path) as index:
            index.record_failure('x-fury', 'extract', 'RuntimeError: no audio stream', 'fp', 'key')

    pipeline = FakePipeline(index_file, fail=['x-fury'], after=fail_them)
    window, _ = _window(qtbot, tmp_path, pipeline=pipeline, prefs=_prefs(tmp_path))
    assert [f.id for f in window.failed] == ['a-failed-x', 'a-failed-d']   # what the index already remembered

    window.select_ids(['x-gravity', 'x-fury'])
    with qtbot.waitSignal(window.run_finished, timeout=10000):
        window.run_selected()

    failed = {f.id: f for f in window.failed}
    assert set(failed) == {'a-failed-x', 'a-failed-d', 'x-fury'}
    assert failed['x-fury'].reason == 'RuntimeError: no audio stream' and failed['x-fury'].stage == 'extract'
    assert failed['x-fury'].title == 'Fury'
    table = window.failuresTable
    assert table.rowCount() == 3
    row = [table.item(r, 0).text() for r in range(3)].index('Fury')
    assert [table.item(row, c).text() for c in range(4)] == ['Fury', 'films', 'extract', 'RuntimeError: no audio stream']
    assert window.detailsTabs.tabText(0) == 'Failures (3)' and window.detailsTabs.isVisibleTo(window)
    # ... and what the run said about each title
    outcomes = {line.id: (line.outcome, line.detail, line.level) for line in window.results}
    assert outcomes['x-fury'] == ('Failed', 'RuntimeError: designer returned 503', 'error')
    assert outcomes['x-gravity'] == ('Designed', '', 'ok')
    assert window.results[0].id == 'x-fury'     # problems first
    assert window.runStatusLabel.text() == 'Extract & design finished: 1 designed, 1 failed'


def test_the_failures_panel_is_hidden_when_nothing_failed(qtbot, tmp_path):
    rows = [r for r in _rows() if not r['id'].startswith('a-failed')]
    window, _ = _window(qtbot, tmp_path, rows=rows)

    assert window.failed == [] and not window.detailsTabs.isVisibleTo(window)


def test_retry_failed_runs_the_failed_titles_again_and_nothing_else(qtbot, tmp_path):
    window, pipeline = _window(qtbot, tmp_path)
    assert window.retryButton.text() == 'Retry 2 failed' and window.retryButton.isEnabled()
    window.select_ids(['x-gravity'])

    # an ordinary run does not retry a failure (it is "failed before"); Retry failed does, by ids, with the flag
    _answer(True, [])   # retrying every failure in the library is as big as a whole view: it asks first
    with qtbot.waitSignal(window.run_finished, timeout=10000):
        _click(qtbot, window.retryButton)

    call = pipeline.calls[0]
    assert call['ids'] == ['a-failed-x', 'a-failed-d'] and call['retry_failed'] is True and call['through'] == 'design'
    assert window.selected_ids() == ['x-gravity']   # the selection in the table was not disturbed


def test_retry_failed_can_be_limited_to_the_failures_selected_in_the_panel(qtbot, tmp_path):
    window, pipeline = _window(qtbot, tmp_path)
    window.failuresTable.selectRow(1)
    assert window.retryButton.text() == 'Retry 1 selected'

    with qtbot.waitSignal(window.run_finished, timeout=10000):
        _click(qtbot, window.retryButton)

    assert pipeline.calls[0]['ids'] == ['a-failed-d'] and pipeline.calls[0]['retry_failed'] is True


def test_an_ordinary_run_never_includes_a_title_that_failed_before(qtbot, tmp_path):
    window, pipeline = _window(qtbot, tmp_path)
    window.select_ids(['a-failed-x', 'a-failed-d', 'x-gravity'])

    plan = window.plan_for('design')
    assert [p.row.id for p in plan.planned] == ['x-gravity']
    assert {s.id for s in plan.skipped} == {'a-failed-x', 'a-failed-d'}
    assert '2 failed before' in window.skippedLabel.text()
    with qtbot.waitSignal(window.run_finished, timeout=10000):
        window.run_selected()
    assert pipeline.calls[0]['ids'] == ['x-gravity']


def test_a_run_over_a_whole_view_asks_first_and_declining_runs_nothing(qtbot, tmp_path):
    window, pipeline = _window(qtbot, tmp_path)
    _click(qtbot, window.extractChip)
    seen: List = []

    _answer(False, seen)
    assert window.run_selected() is False

    assert 'Extract and design 3 titles?' in seen[0] and 'every title that needs it in the current view' in seen[0]
    assert pipeline.calls == [] and not window.is_running
    _answer(True, seen := [])
    with qtbot.waitSignal(window.run_finished, timeout=10000):
        assert window.run_selected() is True
    assert pipeline.calls[0]['ids'] == ['x-gravity', 'x-tenet', 'x-fury']


def test_a_run_over_an_explicit_selection_needs_no_confirmation(qtbot, tmp_path, monkeypatch):
    window, pipeline = _window(qtbot, tmp_path)
    window.select_ids(['x-gravity', 'x-tenet'])
    asked = []
    monkeypatch.setattr(ConfirmDialog, 'exec', lambda self: asked.append(self.text) or 0)   # declines, if it is asked

    with qtbot.waitSignal(window.run_finished, timeout=10000):
        assert window.run_selected() is True

    assert asked == [] and len(pipeline.calls) == 1


def test_an_extract_run_is_refused_when_the_precheck_says_ffmpeg_is_missing(qtbot, tmp_path):
    asked = []
    window, pipeline = _window(qtbot, tmp_path, precheck=lambda through: asked.append(through) or False)
    window.select_ids(['x-gravity'])

    assert window.run_selected() is False

    assert asked == ['design'] and pipeline.calls == [] and not window.is_running
    # publish does not need ffmpeg
    window.select_ids(['p-one'])
    _answer(True, [])
    with qtbot.waitSignal(window.run_finished, timeout=10000):
        assert window.publish_selected() is True
    assert asked == ['design']


# --- publish and commit -----------------------------------------------------------------------------------------------------

def test_publish_names_the_repositories_and_the_count_and_runs_nothing_if_declined(qtbot, tmp_path):
    window, pipeline = _window(qtbot, tmp_path)
    window.select_ids(['p-one', 'p-two', 'p-old', 'x-gravity'])
    seen: List = []
    _answer(False, seen)

    assert window.publish_selected() is False

    text = seen[0]
    assert text.startswith('Publish 3 titles?')
    assert str(tmp_path / 'catalogue-xml') in text and str(tmp_path / 'catalogue-images') in text
    assert 'XML repository' in text and 'Images repository' in text
    assert '1 title of these is already published and out of date' in text      # the republish is named
    assert 'Nothing is committed or pushed' in text
    assert pipeline.calls == [] and not window.is_running
    assert 'nothing was written' in window.runStatusLabel.text()


def test_publish_when_agreed_runs_through_publish_for_the_titles_that_need_it_only(qtbot, tmp_path):
    window, pipeline = _window(qtbot, tmp_path)
    window.select_ids(['p-one', 'p-old', 'x-gravity', 'c-one'])
    _answer(True, [])

    with qtbot.waitSignal(window.run_finished, timeout=10000):
        assert window.publish_selected() is True

    call = pipeline.calls[0]
    assert call['through'] == 'publish' and call['ids'] == ['p-one', 'p-old']   # not the extract, not the commit one
    assert call['publish'].xml_repo.local_path == str(tmp_path / 'catalogue-xml')
    assert call['publish'].images_repo.local_path == str(tmp_path / 'catalogue-images')


def test_commit_confirmation_names_the_repositories_images_first_and_offers_a_push_checkbox_that_defaults_to_yes(qtbot,
                                                                                                                 tmp_path):
    window, pipeline = _window(qtbot, tmp_path)
    window.select_ids(['c-one', 'c-two'])
    seen: List = []
    _answer(False, seen)

    assert window.commit_selected() is False

    text = seen[0]
    assert text.startswith('Commit 2 titles?')
    assert text.index('Images repository') < text.index('XML repository')      # images first
    assert str(tmp_path / 'catalogue-images') in text and str(tmp_path / 'catalogue-xml') in text
    assert 'one commit per repository' in text
    assert seen[1] == ('checkbox', True)     # push is on unless the person unticks it
    assert pipeline.calls == []


def test_commit_pushes_by_default_and_unticking_commits_locally_only_and_is_remembered(qtbot, tmp_path):
    prefs = _prefs(tmp_path)
    window, pipeline = _window(qtbot, tmp_path, prefs=prefs)
    window.select_ids(['c-one'])
    _answer(True, [])
    with qtbot.waitSignal(window.run_finished, timeout=10000):
        window.commit_selected()
    assert pipeline.calls[0]['through'] == 'commit' and pipeline.calls[0]['ids'] == ['c-one']
    assert pipeline.calls[0]['publish'].push is True

    window.select_ids(['c-two'])
    _answer(True, [], tick=False)
    with qtbot.waitSignal(window.run_finished, timeout=10000):
        window.commit_selected()
    assert pipeline.calls[1]['publish'].push is False
    assert prefs.get(WORKLIST_PUSH) is False

    seen: List = []
    window.select_ids(['c-one'])
    _answer(False, seen)
    window.commit_selected()
    assert seen[1] == ('checkbox', False)     # the person's last choice is the default next time


def test_the_results_list_shows_each_title_published_or_refused_and_why(qtbot, tmp_path):
    def report(profile, selection, through, **kw):
        return StagesReport(
            'publish', 3, published=[{'id': 'p-one'}, {'id': 'p-old', 'republished': True, 'edited_project': 'mono'}],
            publish_errors=[{'id': 'p-two', 'error': 'project_conflict'},
                            {'id': 'c-one', 'error': 'invalid_metadata', 'problems': ['year is required']}],
            attempted=['p-one', 'p-old', 'p-two'])

    window, _ = _window(qtbot, tmp_path, pipeline=report)
    window.select_ids(['p-one', 'p-two', 'p-old'])
    _answer(True, [])

    with qtbot.waitSignal(window.run_finished, timeout=10000):
        window.publish_selected()

    lines = {line.id: line for line in window.results if line.id}
    assert lines['p-two'].outcome == 'Refused' and lines['p-two'].level == 'error'
    assert 'the mono and multichannel projects were edited independently' in lines['p-two'].detail
    assert lines['c-one'].outcome == 'Refused' and lines['c-one'].detail.endswith('year is required')
    assert lines['p-one'].outcome == 'Published' and lines['p-one'].level == 'ok'
    assert lines['p-old'].outcome == 'Republished' and lines['p-old'].detail == 'from your project edits'
    assert {line.id for line in window.results[:2]} == {'p-two', 'c-one'}     # the refusals come first
    assert window.detailsTabs.currentWidget() is window.resultsTab and window.detailsTabs.tabText(1) == 'Last run (4)'
    table = window.resultsTable
    texts = [[table.item(r, c).text() for c in range(3)] for r in range(table.rowCount())]
    assert ['Sicario', 'Published', ''] in texts
    assert ['Collateral', 'Refused'] == next(row[:2] for row in texts if row[0] == 'Collateral')
    assert window.runStatusLabel.text() == 'Publish finished: 2 published, 2 refused'


def test_commit_results_are_listed_per_title_and_per_repository(qtbot, tmp_path):
    index_file = make_index(tmp_path / 'work', _rows(), SOURCES, generation=2, last_scan_at=NOW - 900)
    xml = str(tmp_path / 'catalogue-xml')
    images = str(tmp_path / 'catalogue-images')

    def report(profile, selection, through, **kw):
        return StagesReport('commit', 2, committed=CatalogueCommit(
            xml=RepoCommit(xml, ['c-one.xml'], 'abcdef1234567', True),
            images=RepoCommit(images, ['c-one.png'], '1234567abcdef', True)), attempted=['c-one', 'c-two'])

    window, _ = _window(qtbot, tmp_path, pipeline=report)
    window.select_ids(['c-one', 'c-two'])
    _answer(True, [])

    with qtbot.waitSignal(window.run_finished, timeout=10000):
        window.commit_selected()

    lines = {line.id or line.title: line for line in window.results}
    assert lines['c-one'].outcome == 'Committed, pushed'
    assert lines['c-two'].outcome == 'Nothing to commit' and lines['c-two'].level == 'warn'
    assert lines['Images repository'].detail.startswith('commit 1234567a, pushed (1 file)')
    assert lines['XML repository'].detail.startswith('commit abcdef12, pushed (1 file)') and xml in \
           lines['XML repository'].detail
    assert window.results.index(lines['Images repository']) < window.results.index(lines['XML repository'])
    assert window.runStatusLabel.text() == 'Commit finished: 1 committed, 1 pushed'


def test_a_git_failure_while_committing_is_shown_as_an_error_against_the_titles(qtbot, tmp_path):
    def report(profile, selection, through, **kw):
        return StagesReport('commit', 1, commit_error='git failed: ! [rejected] main -> main', attempted=[])

    window, _ = _window(qtbot, tmp_path, pipeline=report)
    window.select_ids(['c-one'])
    _answer(True, [])

    with qtbot.waitSignal(window.run_finished, timeout=10000):
        window.commit_selected()

    lines = {line.id or line.title: line for line in window.results}
    assert lines['c-one'].outcome == 'Not committed' and lines['c-one'].level == 'error'
    assert 'rejected' in lines['Commit'].detail
    assert 'the commit failed' in window.runStatusLabel.text()


# --- review fixes: closing, ordering, prechecks, wording ---------------------------------------------------------------------

def _answer_box(yes: bool, seen: List[str]) -> None:
    ''' Answers the QMessageBox a close during a run opens, as a person would (reads it, clicks Yes or No). '''
    def respond():
        box = QApplication.activeModalWidget()
        seen.append(box.text() if isinstance(box, QMessageBox) else f'unexpected: {box!r}')
        if isinstance(box, QMessageBox):
            box.button(QMessageBox.StandardButton.Yes if yes else QMessageBox.StandardButton.No).click()

    QTimer.singleShot(0, respond)


def _held_run(qtbot, tmp_path, ids=('x-gravity', 'x-tenet')):
    index_file = make_index(tmp_path / 'work', _rows(), SOURCES, generation=2, last_scan_at=NOW - 900)
    pipeline = FakePipeline(index_file, hold_at=0)
    window, _ = _window(qtbot, tmp_path, pipeline=pipeline, prefs=_prefs(tmp_path))
    window.select_ids(list(ids))
    window.run_selected()
    qtbot.waitUntil(pipeline.entered.is_set, timeout=5000)
    return window, pipeline


def test_closing_during_a_run_asks_and_no_leaves_the_window_and_the_run_alone(qtbot, tmp_path):
    window, pipeline = _held_run(qtbot, tmp_path)
    seen: List[str] = []
    _answer_box(False, seen)

    window.close()

    assert seen == ['A run is in progress. Cancel it and close?']
    assert window.isVisible() and window.is_running and window.has_open_index
    assert window.cancelButton.isEnabled()      # the run was not cancelled
    with qtbot.waitSignal(window.run_finished, timeout=10000) as run:
        pipeline.release.set()
    assert not run.args[0].cancelled and pipeline.cancel_seen == []


def test_closing_during_a_run_and_saying_yes_cancels_it_hides_the_window_and_releases_the_index(qtbot, tmp_path):
    window, pipeline = _held_run(qtbot, tmp_path)
    _answer_box(True, [])

    window.close()

    assert not window.isVisible() and not window.has_open_index
    with qtbot.waitSignal(window.run_finished, timeout=10000) as run:
        pipeline.release.set()          # the title in hand finishes; the other is never started
    assert run.args[0].cancelled and pipeline.cancel_seen == [True]
    assert not window.is_running
    assert not window.has_open_index    # the run's end did not reopen a connection in a closed window
    window.show()                       # and showing it again reads the index it now has
    assert window.has_open_index and not window.is_running


def test_closing_with_no_run_going_does_not_ask(qtbot, tmp_path, monkeypatch):
    window, _ = _window(qtbot, tmp_path)
    monkeypatch.setattr(QMessageBox, 'question', lambda *a, **k: pytest.fail('asked with nothing running'))

    window.close()

    assert not window.isVisible() and not window.has_open_index


def test_the_ffmpeg_precheck_is_only_for_runs_that_have_something_to_extract(qtbot, tmp_path):
    asked = []
    window, pipeline = _window(qtbot, tmp_path, precheck=lambda through: asked.append(through) or False)
    window.select_ids(['d-speed', 'd-twister'])       # extracted already: they only need designing

    with qtbot.waitSignal(window.run_finished, timeout=10000):
        assert window.run_selected() is True

    assert asked == [] and pipeline.calls[0]['ids'] == ['d-speed', 'd-twister']
    window.select_ids(['d-speed', 'x-gravity'])        # one still needs extracting: ffmpeg is needed
    assert window.run_selected() is False
    assert asked == ['design']


def test_the_run_is_started_before_run_started_is_announced(qtbot, tmp_path, monkeypatch):
    order = []
    original = QThreadPool.start

    def start(pool, job, *args):
        order.append('started')
        return original(pool, job, *args)

    monkeypatch.setattr(QThreadPool, 'start', start)
    window, _ = _window(qtbot, tmp_path)
    window.run_started.connect(lambda request: order.append('announced'))
    window.select_ids(['x-gravity'])

    with qtbot.waitSignal(window.run_finished, timeout=10000):
        window.run_selected()

    assert order == ['started', 'announced']


def test_a_run_that_cannot_be_started_does_not_leave_the_window_running_for_ever(qtbot, tmp_path, monkeypatch):
    window, pipeline = _window(qtbot, tmp_path)
    window.select_ids(['x-gravity'])

    def refuse(pool, job, *args):
        raise RuntimeError('no threads')

    monkeypatch.setattr(QThreadPool, 'start', refuse)

    assert window.run_selected() is False

    assert not window.is_running and window.model.running == {}
    assert 'Cannot start: RuntimeError: no threads' in window.runStatusLabel.text()
    assert window.runButton.isEnabled() and not window.cancelButton.isVisibleTo(window)
    assert pipeline.calls == []


def test_retrying_every_failure_asks_first_naming_the_whole_library_and_retrying_the_selected_ones_does_not(qtbot,
                                                                                                          tmp_path):
    window, pipeline = _window(qtbot, tmp_path)
    assert 'every failed title in the library' in window.retryButton.toolTip()
    seen: List = []
    _answer(False, seen)

    assert window.retry_failed() is False

    assert seen[0].startswith('Retry 2 failed titles?') and 'wherever it is in the library' in seen[0]
    assert pipeline.calls == []
    window.failuresTable.selectRow(0)         # a choice made in the panel is explicit: no question
    with qtbot.waitSignal(window.run_finished, timeout=10000):
        assert window.retry_failed() is True
    assert pipeline.calls[0]['ids'] == ['a-failed-x']


class _HeldCommit:
    ''' A pipeline that is mid-commit (it says so, as run_stages does) until released. '''

    def __init__(self):
        self.entered, self.release = threading.Event(), threading.Event()

    def __call__(self, profile, selection, through, *, on_progress, should_cancel, **rest):
        ids = list(selection.ids)
        on_progress(Progress(0, len(ids), f'{len(ids)} titles', 'commit', ''))
        self.entered.set()
        assert self.release.wait(20)
        return StagesReport('commit', len(ids), attempted=ids, committed=CatalogueCommit(
            xml=RepoCommit('/x', [f'{i}.xml' for i in ids], 'abcdef1234', True)))


def test_cancelling_during_a_commit_says_a_commit_cannot_be_stopped_and_a_late_cancel_says_so(qtbot, tmp_path):
    pipeline = _HeldCommit()
    window, _ = _window(qtbot, tmp_path, pipeline=pipeline)
    window.select_ids(['c-one', 'c-two'])
    _answer(True, [])
    window.commit_selected()
    qtbot.waitUntil(pipeline.entered.is_set, timeout=5000)
    qtbot.waitUntil(lambda: 'Committing' in window.runStatusLabel.text(), timeout=5000)

    _click(qtbot, window.cancelButton)

    assert 'cannot be stopped' in window.runStatusLabel.text()
    assert 'Cancelling: the title being worked on' not in window.runStatusLabel.text()
    with qtbot.waitSignal(window.run_finished, timeout=10000):
        pipeline.release.set()
    text = window.runStatusLabel.text()      # it was not cancelled: the commit finished, and the window says so
    assert text.startswith('Cancel requested too late') and 'Commit finished' in text


def test_a_push_only_commit_says_push_in_its_button_its_confirmation_and_its_outcome(qtbot, tmp_path):
    rows = [r for r in _rows() if r['id'] not in ('c-one', 'c-two')] + [
        _row('c-one', 'Jaws', 'commit', 2, publish_state='written', commit_state='committed'),
        _row('c-two', 'Ronin 2', 'commit', 1, publish_state='written', commit_state='committed')]
    seen: List = []

    def report(profile, selection, through, **kw):
        return StagesReport('commit', 2, attempted=['c-one', 'c-two'], committed=CatalogueCommit(
            xml=RepoCommit(str(tmp_path / 'catalogue-xml'), ['c-one.xml', 'c-two.xml'], None, True)))

    window, _ = _window(qtbot, tmp_path, pipeline=report, rows=rows)
    window.select_ids(['c-one', 'c-two'])
    assert window.commitButton.text() == 'Push 2'
    _answer(True, seen)

    with qtbot.waitSignal(window.run_finished, timeout=10000):
        window.commit_selected()

    assert seen[0].startswith('Push 2 titles?') and 'Makes one commit' not in seen[0]
    assert 'Nothing new is committed' in seen[0]
    assert window.runStatusLabel.text() == 'Commit finished: 2 pushed'
    assert {line.id: line.outcome for line in window.results if line.id} == {'c-one': 'Pushed', 'c-two': 'Pushed'}


def test_a_multi_line_git_failure_is_one_line_in_the_row_whole_in_the_tooltip_and_once_in_the_details_area(qtbot,
                                                                                                         tmp_path):
    error = ("git failed: To /srv/beq-xml.git\n ! [rejected]        main -> main (fetch first)\n"
             "error: failed to push some refs to '/srv/beq-xml.git'")

    def report(profile, selection, through, **kw):
        return StagesReport('commit', 2, commit_error=error)

    window, _ = _window(qtbot, tmp_path, pipeline=report)
    window.select_ids(['c-one', 'c-two'])
    _answer(True, [])

    with qtbot.waitSignal(window.run_finished, timeout=10000):
        window.commit_selected()

    table = window.resultsTable
    texts = [[table.item(r, c).text() for c in range(3)] for r in range(table.rowCount())]
    assert all('\n' not in cell for row in texts for cell in row)
    assert ['Jaws', 'Not committed', 'git failed: ! [rejected] main -> main (fetch first) ...'] in texts
    commit_row = next(r for r in range(table.rowCount()) if table.item(r, 0).text() == 'Commit')
    assert 'failed to push some refs' in table.item(commit_row, 2).toolTip()
    # with nothing selected the Details area shows the failure once, whole
    assert window.resultDetails.isVisibleTo(window)
    assert window.resultDetails.toPlainText().count('failed to push some refs') == 1
    assert 'main -> main (fetch first)' in window.resultDetails.toPlainText()
    table.selectRow(0)                       # a title's own line has no more to say than its row: nothing is shown
    assert not window.resultDetails.isVisibleTo(window)
    table.selectRow(commit_row)
    assert 'failed to push some refs' in window.resultDetails.toPlainText()


def test_the_results_show_each_new_publish_and_commit_result_shape(qtbot, tmp_path):
    def report(profile, selection, through, **kw):
        return StagesReport(
            'commit', 3, published=[{'id': 'p-one'}],
            publish_errors=[{'id': 'p-two', 'error': 'git_failed', 'message': 'git add x failed (exit 128)'},
                            {'id': 'p-old', 'error': 'publish_failed', 'message': "ValueError: Unrecognised GitHub "
                                                                              "remote URL: '/srv/img.git'"}],
            committed=CatalogueCommit(xml=RepoCommit('/x', ['c-two.xml'], 'abcdef12', True), images=None,
                                      not_committed=['c-one.xml'],
                                      warnings=['c-two.xml names a report image (beq_spectrumURL) but no images '
                                                'repository was given']),
            attempted=['p-one', 'c-one', 'c-two'])

    window, _ = _window(qtbot, tmp_path, pipeline=report)
    window.select_ids(['p-one', 'p-two', 'p-old', 'c-one', 'c-two'])
    lines = {}
    for through in ('publish', 'commit'):
        _answer(True, [])
        with qtbot.waitSignal(window.run_finished, timeout=10000):
            window._begin(through)
        lines[through] = {line.id or line.title: line for line in window.results}

    both = lines['commit']
    assert both['p-two'].outcome == 'Git failed' and both['p-old'].outcome == 'Publish failed'
    assert both['p-old'].detail.startswith('Set image_owner and image_repo_name')
    assert both['c-one'].outcome == 'Not committed' and both['c-one'].level == 'error'
    assert both['Notice'].level == 'warn'
    assert window.runStatusLabel.text().startswith('Commit finished:')
    assert 'not committed (git ignores them)' in window.runStatusLabel.text()


# --- the real pipeline ------------------------------------------------------------------------------------------------------

class _Films:
    def list_items(self, **query):
        return [LibraryItem(id='fs-missing', source_path='/nonexistent/films/Missing.mkv', display_name='Missing',
                            title='Missing', year='2001', fingerprint='fp-1')]


def test_the_real_pipeline_runs_off_the_ui_thread_and_a_title_that_fails_is_remembered_and_listed(qtbot, tmp_path):
    prefs = _prefs(tmp_path)
    window = WorkListWindow(None, prefs, sources={'filesystem': _Films()}, auto_scan=False, clock=time.time)
    qtbot.addWidget(window)
    window.show()
    with qtbot.waitSignal(window.scan_finished, timeout=10000):
        window.rescan()
    assert window.listed_ids() == ['fs-missing'] and window.chip_counts()['Extract'] == 1
    window.select_ids(['fs-missing'])
    assert _run_button_text(window) == 'Extract & design 1'

    with qtbot.waitSignal(window.run_finished, timeout=60000) as run:
        window.run_selected()

    assert [i for i, _ in run.args[0].run.failed] == ['fs-missing']
    assert [f.id for f in window.failed] == ['fs-missing']       # the pipeline recorded it; the window read it back
    assert window.chip_counts()['Attention'] == 1 and window.chip_counts()['Extract'] == 0
    assert window.results[0].outcome == 'Failed'
    # it is not tried again unasked ...
    assert window.plan_for('design').planned == []
    assert '1 failed before' in window.skippedLabel.text()
    # ... but Retry failed does, and the failure is remembered against the same title again
    with qtbot.waitSignal(window.run_finished, timeout=60000) as again:
        window.retry_failed()
    assert [i for i, _ in again.args[0].run.failed] == ['fs-missing']
    assert not any('not tried' in line.outcome.lower() for line in window.results)
    assert [f.id for f in window.failed] == ['fs-missing']


# --- the menu ---------------------------------------------------------------------------------------------------------------

def test_the_tools_menu_opens_the_work_list_and_the_review_folder_and_the_classic_dialog_is_gone(
        qtbot, tmp_path, monkeypatch):
    import app as app_module
    from model.worklist_review import ReviewFolderWindow
    root = logging.getLogger()
    handlers = list(root.handlers)
    prefs = _prefs(tmp_path, **{SYSTEM_CHECK_FOR_UPDATES: False})
    main = app_module.BeqDesigner(QApplication.instance(), prefs)
    qtbot.addWidget(main)
    try:
        monkeypatch.setattr(main, '_BeqDesigner__check_ffmpeg_available', lambda: True)
        actions = main.menu_Tools.actions()
        assert main.action_Work_List in actions and main.action_Review_Folder in actions
        assert main.action_Work_List.text() == 'Library &Work List'
        assert main.action_Review_Folder.text() == 'Re&view Folder...'
        # the retired entries: the classic Library Sync dialog and "Review Batch Designs" are gone from the menu and the window
        assert not hasattr(main, 'action_Library_Sync') and not hasattr(main, 'action_Review_Batch_Designs')
        assert not hasattr(main, 'showLibrarySyncDialog') and not hasattr(main, 'showReviewQueueDialog')
        assert sorted(a.text() for a in actions if 'Sync' in a.text() and 'HTP' not in a.text()) == []

        main.action_Work_List.trigger()
        window = main._BeqDesigner__work_list
        assert isinstance(window, WorkListWindow) and window.isVisible()

        main.action_Review_Folder.trigger()
        folder = main._BeqDesigner__review_folder
        assert isinstance(folder, ReviewFolderWindow) and folder.isVisible()
        qtbot.addWidget(folder)

        # the work list's extract runs check for ffmpeg first, through the app
        checked = []
        monkeypatch.setattr(main, '_BeqDesigner__check_ffmpeg_available', lambda: checked.append(True) or False)
        qtbot.waitUntil(lambda: not window.is_scanning, timeout=10000)   # a never-scanned index is scanned on opening
        make_index(tmp_path / 'work', _rows(), SOURCES, generation=2, last_scan_at=NOW - 900)
        window.reload()
        window.select_ids(['x-gravity'])
        assert window.run_selected() is False and checked == [True]
    finally:
        for handler in list(root.handlers):
            if handler not in handlers:
                root.removeHandler(handler)
