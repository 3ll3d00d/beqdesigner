"""W1: full indexed failures and current retry events are distinct, copyable views."""
import ui.beq  # noqa: F401

import threading

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication

from model.execution_events import ExecutionEvent
from model.worklist_model import COL_DETAIL
from pipeline.library.index import LibraryIndex, index_path
from pipeline.library.run import LibraryRunReport
from pipeline.library.stages import Progress, StagesReport
from test_worklist_actions import NOW, _cell, _designer, _update, _window  # noqa: F401 (fixture)


@pytest.fixture(autouse=True)
def _finish_pending_draws(qtbot):
    yield
    # MagnitudeModel schedules a Qt redraw when the title page opens. Let it
    # finish before qtbot destroys the window and its matplotlib canvas.
    qtbot.wait(10)


@pytest.mark.parametrize('title_id,stage,word', [('a-failed-x', 'extract', 'extraction'),
                                              ('a-failed-d', 'design', 'design')])
def test_failure_tab_and_details_show_full_index_memory_and_copy_redacted_multiline_text(
        qtbot, tmp_path, title_id, stage, word):
    window, _ = _window(qtbot, tmp_path)
    message = 'Decoder failed\n' + ('diagnostic line\n' * 800) + 'password=private\nlast diagnostic'
    with LibraryIndex(index_path(str(tmp_path / 'work'))) as index:
        index.record_failure(title_id, stage, message, 'fp', 'key')
    window.refresh_from_index()
    window.select_ids([title_id])
    assert window.retryButton.text() == f'Retry 1 failed {word}'
    assert window.open_title(title_id)
    page = window.title_page
    assert page.actions_bar.retryButton.text() == f'Retry failed {word}'
    page.rightTabs.setCurrentWidget(page.failurePanel)
    text = page.failurePanel.output.toPlainText()
    assert text.startswith(f'Indexed failure ({stage}):\nDecoder failed\n')
    assert 'last diagnostic' in text and len(text) > 8192
    assert 'private' not in text and 'password=[REDACTED]' in text
    page.failurePanel.output.selectAll()
    page.failurePanel.output.copy()
    assert QApplication.clipboard().text() == text
    page.failurePanel.copyButton.click()
    assert QApplication.clipboard().text() == text
    page.actions_bar.detailsButton.click()
    dialog = window._detail_dialogs[title_id]
    assert text in dialog.output.toPlainText()
    assert 'No run events retained' in dialog.output.toPlainText()
    dialog.copyButton.click()
    assert QApplication.clipboard().text() == dialog.output.toPlainText()


@pytest.mark.parametrize('title_id,stage', [('a-failed-x', 'extract'), ('a-failed-d', 'design')])
@pytest.mark.parametrize('outcome', ['success', 'failed', 'cancelled'])
def test_retry_keeps_previous_failure_until_refreshed_result_and_separates_current_events(
        qtbot, tmp_path, title_id, stage, outcome):
    entered, release = threading.Event(), threading.Event()

    def pipeline(profile, selection, through, *, index, on_event, on_progress, should_cancel, **kwargs):
        on_event(ExecutionEvent('retry', title_id, stage, 'stage_started', NOW, 'Current retry'))
        on_progress(Progress(0, 1, 'Title', stage, title_id))
        on_event(ExecutionEvent('retry', title_id, stage, 'command_started', NOW,
                                command=('ffmpeg', '-i', 'https://user:password@host/media', '--api-key', 'secret-value')))
        entered.set()
        assert release.wait(15)
        if outcome == 'success':
            index.clear_failure(title_id)
            _update(index_path(str(tmp_path / 'work')), [title_id], needs='review', tier='human',
                    extract_state='current', design_state='current', detail='ready for review', failure='')
            return StagesReport(through, 1, run=LibraryRunReport(designed=[title_id]), attempted=[title_id])
        if outcome == 'failed':
            message = 'New retry failure\nsecond diagnostic'
            index.record_failure(title_id, stage, message, 'fp', 'key')
            _update(index_path(str(tmp_path / 'work')), [title_id], detail='New retry failure', failure=message)
            on_event(ExecutionEvent('retry', title_id, stage, 'failed', NOW + 1, message))
            return StagesReport(through, 1, run=LibraryRunReport(failed=[(title_id, message)]), attempted=[title_id])
        assert should_cancel()
        return StagesReport(through, 1, cancelled=True, not_run=[title_id])

    window, _ = _window(qtbot, tmp_path, pipeline=pipeline)
    assert window.open_title(title_id)
    page = window.title_page
    old = page.failurePanel.output.toPlainText().split(':\n', 1)[1]
    page.actions_bar.retryButton.click()
    qtbot.waitUntil(entered.is_set, timeout=5000)
    try:
        qtbot.waitUntil(lambda: 'Current attempt:' in page.failurePanel.attemptLabel.text(), timeout=5000)
        assert 'Previous indexed failure' in page.failurePanel.output.toPlainText()
        assert old in page.failurePanel.output.toPlainText()
        assert not page.actions_bar.retryButton.isEnabled()
        assert old in _cell(window, title_id, COL_DETAIL, Qt.ItemDataRole.ToolTipRole)
        page.actions_bar.detailsButton.click()
        qtbot.waitUntil(lambda: 'Command:' in window._detail_dialogs[title_id].output.toPlainText(), timeout=5000)
        text = window._detail_dialogs[title_id].output.toPlainText()
        assert old in text and 'Current run events:' in text
        assert 'secret-value' not in text and 'user:password' not in text
        if outcome == 'cancelled':
            assert window.cancel_run()
    finally:
        release.set()
    qtbot.waitUntil(lambda: not window.is_running, timeout=5000)
    assert not window.model.run_state(title_id).get('attempting')
    text = page.failurePanel.output.toPlainText()
    if outcome == 'success':
        assert text == '' and not page.rightTabs.isTabVisible(page._failure_tab)
        assert 'Previous indexed failure' not in window._detail_dialogs[title_id].output.toPlainText()
    else:
        assert text.startswith(f'Indexed failure ({stage}):')
        assert ('New retry failure\nsecond diagnostic' if outcome == 'failed' else old) in text
        assert page.actions_bar.retryButton.isEnabled()


def test_a_disjoint_run_expires_events_but_keeps_the_indexed_failure(qtbot, tmp_path):
    def pipeline(profile, selection, through, *, on_event, **kwargs):
        title_id = selection.ids[0]
        on_event(ExecutionEvent('run-' + title_id, title_id, 'design', 'failed', NOW, 'one-run history'))
        return StagesReport(through, 1, run=LibraryRunReport(failed=[(title_id, 'one-run history')]), attempted=[title_id])

    window, _ = _window(qtbot, tmp_path, pipeline=pipeline)
    with qtbot.waitSignal(window.run_finished, timeout=5000):
        window.retry_failed(['a-failed-d'])
    window._open_run_details('a-failed-d')
    assert 'one-run history' in window._detail_dialogs['a-failed-d'].output.toPlainText()
    window.select_ids(['x-gravity'])
    with qtbot.waitSignal(window.run_finished, timeout=5000):
        window.run_selected()
    assert 'a-failed-d' not in window._detail_dialogs
    assert window.open_title('a-failed-d')
    window.title_page.actions_bar.detailsButton.click()
    text = window._detail_dialogs['a-failed-d'].output.toPlainText()
    assert 'design failed: timed out' in text and 'one-run history' not in text
    assert 'No run events retained' in text


def test_cache_hits_have_explicit_details_without_process_commands(qtbot, tmp_path):
    def pipeline(profile, selection, through, **kwargs):
        title_id = selection.ids[0]
        return StagesReport(through, 1, run=LibraryRunReport(cached=[title_id], design_cached=[title_id]),
                            attempted=[title_id])

    window, _ = _window(qtbot, tmp_path, pipeline=pipeline)
    window.select_ids(['x-gravity'])
    with qtbot.waitSignal(window.run_finished, timeout=5000):
        window.run_selected()
    assert window.open_title('x-gravity')
    window.title_page.actions_bar.detailsButton.click()
    text = window._detail_dialogs['x-gravity'].output.toPlainText()
    assert 'Extract cache hit' in text and 'Design cache hit' in text
    assert 'no command ran' in text and 'Command:' not in text


def test_intermediate_index_refresh_keeps_old_failure_until_the_retry_finishes(qtbot, tmp_path):
    entered, release = threading.Event(), threading.Event()
    title_id = 'a-failed-x'

    def pipeline(profile, selection, through, *, index, on_event, **kwargs):
        index.clear_failure(title_id)
        _update(index_path(str(tmp_path / 'work')), [title_id], needs='design', tier='machine',
                extract_state='current', design_state='none', detail='extracted', failure='')
        # The UI refreshes the index when extraction hands off to design.
        on_event(ExecutionEvent('retry', title_id, 'design', 'stage_queued', NOW, 'Waiting for design slot'))
        entered.set()
        assert release.wait(15)
        _update(index_path(str(tmp_path / 'work')), [title_id], needs='review', tier='human',
                design_state='current', detail='ready for review')
        return StagesReport(through, 1, run=LibraryRunReport(designed=[title_id]), attempted=[title_id])

    window, _ = _window(qtbot, tmp_path, pipeline=pipeline)
    assert window.open_title(title_id)
    page = window.title_page
    page.actions_bar.retryButton.click()
    qtbot.waitUntil(entered.is_set, timeout=5000)
    try:
        qtbot.waitUntil(lambda: not window.failed or all(f.id != title_id for f in window.failed), timeout=5000)
        assert window.model.run_state(title_id)['attempting']
        assert 'Previous indexed failure' in page.failurePanel.output.toPlainText()
        assert 'file not found' in page.failurePanel.output.toPlainText()
        assert 'Queued for design' in page.failurePanel.attemptLabel.text()
        assert not page.actions_bar.retryButton.isEnabled()
    finally:
        release.set()
    qtbot.waitUntil(lambda: not window.is_running, timeout=5000)
    assert page.failurePanel.output.toPlainText() == ''
