'''
Safety net for model/review.py's ReviewQueueDialog (design/candidate-review-
plan.md phase 3) -- constructs the real dialog and drives its actual
widgets/keyboard shortcuts against a hand-built queue directory.
pipeline.review's own persistence round-trip is already covered by
test_pipeline_review.py; this only exercises the Qt wiring on top of it.
'''
import numpy as np
import pytest
from qtpy.QtCore import QSettings

from model.codec import xydata_to_json
from model.iir import CompleteFilter, LowShelf, PeakingEQ
from model.preferences import Preferences
from model.review import ReviewQueueDialog
from model.xy import MagnitudeData
from pipeline.review import CandidateSummary, QueueEntry, read_entry, write_queue_entry


def _make_preferences(tmp_path):
    settings = QSettings(str(tmp_path / 'settings.ini'), QSettings.Format.IniFormat)
    return Preferences(settings)


def _curve_json():
    x = np.linspace(1.0, 500.0, 50)
    return xydata_to_json(MagnitudeData('avg', '', x, np.zeros_like(x)))


def _write_entry(queue_dir, entry_id, status='pending', chosen_candidate_index=None, decline=False):
    if decline:
        entry = QueueEntry(id=entry_id, fs=1000, meta={'title': entry_id}, curve=_curve_json(), candidates=[],
                          decline_reason='no_rolloff_detected', decline_message='nothing found')
    else:
        low_shelf = CompleteFilter(fs=1000, filters=[LowShelf(1000, 20, 0.7, 4.5)])
        peaking = CompleteFilter(fs=1000, filters=[PeakingEQ(1000, 100, 1, -3.0)])
        candidates = [
            CandidateSummary(filters=low_shelf.to_json(), confidence=0.9, method='fitted', mv_adjust_db=4.0,
                             gain_reduction_db=-1.0, commentary={'note': 'top pick'}),
            CandidateSummary(filters=peaking.to_json(), confidence=0.4, method='fitted', mv_adjust_db=1.0,
                             gain_reduction_db=0.0, commentary={'note': 'alternative'}),
        ]
        entry = QueueEntry(id=entry_id, fs=1000, meta={'title': entry_id}, curve=_curve_json(),
                          candidates=candidates, status=status, chosen_candidate_index=chosen_candidate_index)
    write_queue_entry(queue_dir, entry)
    return entry


@pytest.fixture
def dialog(qtbot, tmp_path):
    prefs = _make_preferences(tmp_path)
    d = ReviewQueueDialog(None, prefs)
    qtbot.addWidget(d)
    return d


def test_loading_a_queue_populates_the_table(tmp_path, dialog):
    queue_dir = str(tmp_path / 'queue')
    _write_entry(queue_dir, 'title-a')
    _write_entry(queue_dir, 'title-b')

    dialog.load_queue_dir(queue_dir)

    assert dialog.queueTable.model().rowCount() == 2


def test_selecting_a_row_populates_candidates_and_commentary(tmp_path, dialog):
    queue_dir = str(tmp_path / 'queue')
    _write_entry(queue_dir, 'title-a')
    dialog.load_queue_dir(queue_dir)

    dialog.queueTable.selectRow(0)

    assert dialog.candidateList.count() == 2
    assert dialog.candidateList.currentRow() == 0  # defaults to candidates[0], the top pick
    assert dialog.commentaryTable.rowCount() == 1
    assert dialog.commentaryTable.item(0, 0).text() == 'note'
    assert dialog.commentaryTable.item(0, 1).text() == 'top pick'


def test_declined_entry_shows_reason_and_disables_accept(tmp_path, dialog):
    queue_dir = str(tmp_path / 'queue')
    _write_entry(queue_dir, 'declined-title', decline=True)
    dialog.load_queue_dir(queue_dir)

    dialog.queueTable.selectRow(0)

    assert not dialog.declineReasonLabel.isHidden()  # isVisible() is always False for an unshown top-level dialog
    assert 'no_rolloff_detected' in dialog.declineReasonLabel.text()
    assert not dialog.acceptButton.isEnabled()
    assert dialog.skipButton.isEnabled()


def test_digit_key_changes_the_picked_candidate_without_accepting(tmp_path, dialog):
    queue_dir = str(tmp_path / 'queue')
    _write_entry(queue_dir, 'title-a')
    dialog.load_queue_dir(queue_dir)
    dialog.queueTable.selectRow(0)

    dialog._ReviewQueueDialog__pick_candidate(1)

    assert dialog.candidateList.currentRow() == 1
    assert read_entry(queue_dir, 'title-a').status == 'pending'  # picking alone doesn't accept


def test_accept_writes_the_chosen_candidate_and_advances_to_next_pending(tmp_path, dialog):
    queue_dir = str(tmp_path / 'queue')
    _write_entry(queue_dir, 'title-a')
    _write_entry(queue_dir, 'title-b')
    dialog.load_queue_dir(queue_dir)
    dialog.queueTable.selectRow(0)  # title-a, alphabetically first
    dialog._ReviewQueueDialog__pick_candidate(1)

    dialog._ReviewQueueDialog__accept_current()

    accepted = read_entry(queue_dir, 'title-a')
    assert accepted.status == 'accepted'
    assert accepted.chosen_candidate_index == 1
    current = dialog._ReviewQueueDialog__current_entry()
    assert current.id == 'title-b'
    assert current.status == 'pending'


def test_skip_leaves_entry_pending_and_advances(tmp_path, dialog):
    queue_dir = str(tmp_path / 'queue')
    _write_entry(queue_dir, 'title-a')
    _write_entry(queue_dir, 'title-b')
    dialog.load_queue_dir(queue_dir)
    dialog.queueTable.selectRow(0)

    dialog._ReviewQueueDialog__skip_current()

    assert read_entry(queue_dir, 'title-a').status == 'skipped'
    assert dialog._ReviewQueueDialog__current_entry().id == 'title-b'


def test_reject_is_permanent_and_distinct_from_skip(tmp_path, dialog):
    queue_dir = str(tmp_path / 'queue')
    _write_entry(queue_dir, 'title-a')
    dialog.load_queue_dir(queue_dir)
    dialog.queueTable.selectRow(0)

    dialog._ReviewQueueDialog__reject_current()

    assert read_entry(queue_dir, 'title-a').status == 'rejected'


def test_advance_does_not_jump_back_to_row_zero(tmp_path, dialog):
    '''
    Same principle as app.py's deleteSignal() fix earlier this session:
    accepting/skipping should move forward from wherever the reviewer was,
    not reset the selection to the top of the table.
    '''
    queue_dir = str(tmp_path / 'queue')
    _write_entry(queue_dir, 'title-a')
    _write_entry(queue_dir, 'title-b')
    _write_entry(queue_dir, 'title-c')
    dialog.load_queue_dir(queue_dir)
    dialog.queueTable.selectRow(1)  # title-b

    dialog._ReviewQueueDialog__skip_current()

    assert dialog._ReviewQueueDialog__current_entry().id == 'title-c'  # not title-a


def test_chart_shows_unfiltered_and_selected_candidate_curves(tmp_path, dialog):
    queue_dir = str(tmp_path / 'queue')
    _write_entry(queue_dir, 'title-a')
    dialog.load_queue_dir(queue_dir)
    dialog.queueTable.selectRow(0)

    curves = dialog._ReviewQueueDialog__get_chart_data()

    assert len(curves) == 2
    assert curves[0].colour == 'grey'
    assert curves[1].colour == 'red'
