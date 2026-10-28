'''
Deleting a title's old designs from the work list's title page (model/worklist_old_designs.py): its earlier design runs,
some or all; the current design is kept, and nothing is deleted while a run works on the title.

`import ui.beq` first: see AGENTS.md gotcha 3.
'''
import ui.beq  # noqa: F401 (must come first)

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication

from model.worklist_old_designs import OldDesignsDialog, describe_run
from pipeline.review import read_entry, update_entry
from modal import when_modal
from test_worklist_revise import _window
from test_worklist_title import _open, _queue


def _runs(n):
    return [{'variant_id': f'v{i}', 'designed_at': f'2026-10-0{i}T10:00:00Z', 'designer_build': 'beqforge 0.2.8',
             'candidates': [{'index': 0}, {'index': 1}] if i % 2 else []} for i in range(1, n + 1)]


def _with_runs(tmp_path, n=3, entry_id='p-fury'):
    update_entry(_queue(tmp_path), entry_id, variants=_runs(n), variant_id=f'v{n + 1}')


class _Ask:
    def __init__(self, answer):
        self.answer, self.asked = answer, []

    def __call__(self, title, runs):
        self.asked.append((title, [r['variant_id'] for r in runs]))
        return self.answer(runs) if callable(self.answer) else self.answer


def test_an_earlier_run_reads_as_one_line():
    assert describe_run({'variant_id': 'v2', 'designed_at': 'nope', 'candidates': [{}], 'designer_build': 'b'}) == \
        'v2 · nope · 1 design · b'
    assert describe_run({'variant_id': 'v1', 'declined': True}) == 'v1 · when not recorded · declined'


def test_some_old_designs_are_deleted_and_the_current_one_is_kept(qtbot, tmp_path):
    window = _window(qtbot, tmp_path)
    _with_runs(tmp_path)
    page = _open(qtbot, window, 'p-fury')
    action = page.actions_bar.oldDesignsAction
    assert action.isEnabled() and action.text() == 'Old designs (3)...'
    page._hooks.ask_old_designs = ask = _Ask(['v1', 'v3'])

    assert page.delete_old_designs() is True

    entry = read_entry(_queue(tmp_path), 'p-fury')
    assert [v['variant_id'] for v in entry.variants] == ['v2'] and entry.variant_id == 'v4'
    assert ask.asked[0][1] == ['v1', 'v2', 'v3']
    assert page.actions_bar.messageLabel.text() == 'Deleted 2 old designs; the current design is kept.'
    assert action.text() == 'Old designs (1)...'


def test_all_old_designs_are_deleted_through_the_real_dialog(qtbot, tmp_path):
    window = _window(qtbot, tmp_path)
    _with_runs(tmp_path)
    page = _open(qtbot, window, 'p-fury')
    seen = []

    def answer(dialog):
        seen.append([dialog.runs.item(i).text() for i in range(dialog.runs.count())])
        assert not dialog.deleteSelectedButton.isEnabled()          # nothing ticked yet
        dialog.runs.item(0).setCheckState(Qt.CheckState.Checked)
        assert dialog.deleteSelectedButton.isEnabled()
        dialog.deleteAllButton.click()

    when_modal(OldDesignsDialog, answer)
    page.actions_bar.oldDesignsAction.trigger()

    assert seen[0][0].startswith('v3 · ') and len(seen[0]) == 3      # newest first
    assert read_entry(_queue(tmp_path), 'p-fury').variants == []
    assert not page.actions_bar.oldDesignsAction.isEnabled()
    assert QApplication.activeModalWidget() is None


def test_cancelling_deletes_nothing_and_a_title_with_none_says_so(qtbot, tmp_path):
    window = _window(qtbot, tmp_path)
    _with_runs(tmp_path, 2)
    page = _open(qtbot, window, 'p-fury')
    page._hooks.ask_old_designs = _Ask(None)
    assert page.delete_old_designs() is False
    assert len(read_entry(_queue(tmp_path), 'p-fury').variants) == 2

    update_entry(_queue(tmp_path), 'p-fury', variants=[])
    page.reload()
    assert not page.actions_bar.oldDesignsAction.isEnabled()
    assert page.delete_old_designs() is False


def test_nothing_is_deleted_while_a_run_works_on_the_title(qtbot, tmp_path):
    window = _window(qtbot, tmp_path)
    _with_runs(tmp_path)
    page = _open(qtbot, window, 'p-fury')
    page._running = lambda: {'p-fury'}
    page._hooks.ask_old_designs = ask = _Ask(['v1'])
    assert page.delete_old_designs() is False
    assert len(read_entry(_queue(tmp_path), 'p-fury').variants) == 3 and ask.asked == []
    assert page.actions_bar.messageLabel.text().startswith('A run is working on this title')
