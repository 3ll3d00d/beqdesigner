'''
Safety net for model/review.py's ReviewQueueDialog (design/candidate-review-
plan.md phase 3) -- constructs the real dialog and drives its actual
widgets/keyboard shortcuts against a hand-built queue directory.
pipeline.review's own persistence round-trip is already covered by
test_pipeline_review.py; this only exercises the Qt wiring on top of it.
'''
import numpy as np
import pytest
from qtpy.QtCore import QSettings, Qt
from qtpy.QtWidgets import QFileDialog, QMessageBox

from model.codec import xydata_to_json
from model.iir import CompleteFilter, LowShelf, PeakingEQ
from model.preferences import Preferences
from model.review import ReviewQueueDialog
from model.xy import MagnitudeData
from pipeline.metadata import BeqMetadata
from pipeline.review import CandidateSummary, QueueEntry, read_entry, write_queue_entry


def _make_preferences(tmp_path):
    settings = QSettings(str(tmp_path / 'settings.ini'), QSettings.Format.IniFormat)
    return Preferences(settings)


def _curve_json():
    x = np.linspace(1.0, 500.0, 50)
    return xydata_to_json(MagnitudeData('avg', '', x, np.zeros_like(x)))


def _write_entry(queue_dir, entry_id, status='pending', chosen_candidate_index=None, decline=False, meta=None,
                 art_path=None, art_overridden=False):
    if meta is None:
        meta = {'title': entry_id}
    if decline:
        entry = QueueEntry(id=entry_id, fs=1000, meta=meta, curve=_curve_json(), candidates=[],
                          decline_reason='no_rolloff_detected', decline_message='nothing found',
                          art_path=art_path, art_overridden=art_overridden)
    else:
        low_shelf = CompleteFilter(fs=1000, filters=[LowShelf(1000, 20, 0.7, 4.5)])
        peaking = CompleteFilter(fs=1000, filters=[PeakingEQ(1000, 100, 1, -3.0)])
        candidates = [
            CandidateSummary(filters=low_shelf.to_json(), confidence=0.9, method='fitted', mv_adjust_db=4.0,
                             gain_reduction_db=-1.0, commentary={'note': 'top pick'}),
            CandidateSummary(filters=peaking.to_json(), confidence=0.4, method='fitted', mv_adjust_db=1.0,
                             gain_reduction_db=0.0, commentary={'note': 'alternative'}),
        ]
        entry = QueueEntry(id=entry_id, fs=1000, meta=meta, curve=_curve_json(),
                          candidates=candidates, status=status, chosen_candidate_index=chosen_candidate_index,
                          art_path=art_path, art_overridden=art_overridden)
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


def test_load_queue_dir_remembers_it_as_the_designer_queue_dir_default(tmp_path, dialog):
    from model.preferences import DESIGNER_QUEUE_DIR
    queue_dir = str(tmp_path / 'queue')
    _write_entry(queue_dir, 'title-a')

    dialog.load_queue_dir(queue_dir)

    prefs = dialog._ReviewQueueDialog__preferences
    assert prefs.get(DESIGNER_QUEUE_DIR) == queue_dir


def test_dialog_defaults_to_the_remembered_queue_dir_on_open(qtbot, tmp_path):
    from model.preferences import DESIGNER_QUEUE_DIR
    prefs = _make_preferences(tmp_path)
    queue_dir = str(tmp_path / 'queue')
    _write_entry(queue_dir, 'title-a')
    prefs.set(DESIGNER_QUEUE_DIR, queue_dir)

    d = ReviewQueueDialog(None, prefs)
    qtbot.addWidget(d)

    assert d.queueDirEdit.text() == queue_dir
    assert d.queueTable.model().rowCount() == 1


def test_selecting_a_row_populates_the_metadata_form(tmp_path, dialog):
    queue_dir = str(tmp_path / 'queue')
    meta = {'title': 'Ready Player One', 'alt_title': 'RPO', 'sort_title': 'Ready Player One', 'year': '2018',
           'audio_types': ['Atmos', 'TrueHD 7.1'], 'edition': 'Extended', 'season': '', 'note': 'a note',
           'warning': 'flashing lights', 'language': 'French', 'source': 'Bluray', 'rating': 'PG-13',
           'author': 'someone', 'avs': 'https://example.com/post', 'runtime': '140', 'gain': '-2.0',
           'the_movie_db': '335984', 'genres': [{'id': 28, 'name': 'Action'}]}
    _write_entry(queue_dir, 'title-a', meta=meta)
    dialog.load_queue_dir(queue_dir)

    dialog.queueTable.selectRow(0)

    assert dialog.titleField.text() == 'Ready Player One'
    assert dialog.altTitleField.text() == 'RPO'
    assert dialog.sortTitleField.text() == 'Ready Player One'
    assert dialog.yearField.text() == '2018'
    assert dialog.audioTypesField.text() == 'Atmos, TrueHD 7.1'
    assert dialog.editionField.text() == 'Extended'
    assert dialog.warningField.text() == 'flashing lights'
    assert dialog.languageField.text() == 'French'
    assert dialog.sourceField.text() == 'Bluray'
    assert dialog.ratingField.text() == 'PG-13'
    assert dialog.authorField.text() == 'someone'
    assert dialog.avsField.text() == 'https://example.com/post'
    assert dialog.runtimeField.text() == '140'
    assert dialog.gainField.text() == '-2.0'
    assert dialog.movieDbIdField.text() == '335984'
    assert dialog.genresLabel.text() == 'Action'


def test_save_metadata_persists_edited_fields(tmp_path, dialog):
    queue_dir = str(tmp_path / 'queue')
    _write_entry(queue_dir, 'title-a')
    dialog.load_queue_dir(queue_dir)
    dialog.queueTable.selectRow(0)

    dialog.editionField.setText('Extended Cut')
    dialog._ReviewQueueDialog__save_metadata()

    assert read_entry(queue_dir, 'title-a').meta['edition'] == 'Extended Cut'


def test_save_metadata_does_not_overwrite_defaults_with_blank_fields(tmp_path, dialog):
    queue_dir = str(tmp_path / 'queue')
    _write_entry(queue_dir, 'title-a')
    dialog.load_queue_dir(queue_dir)
    dialog.queueTable.selectRow(0)

    assert dialog.languageField.text() == ''
    assert dialog.sourceField.text() == ''
    dialog._ReviewQueueDialog__save_metadata()

    saved_meta = read_entry(queue_dir, 'title-a').meta
    assert 'language' not in saved_meta
    assert 'source' not in saved_meta
    assert BeqMetadata(**saved_meta).language == 'English'
    assert BeqMetadata(**saved_meta).source == 'Disc'


def test_metadata_tab_is_read_only_once_accepted(tmp_path, dialog):
    queue_dir = str(tmp_path / 'queue')
    _write_entry(queue_dir, 'title-a', status='accepted', chosen_candidate_index=0)
    dialog.load_queue_dir(queue_dir)

    dialog.queueTable.selectRow(0)

    assert dialog.metadataTab.isEnabled() is False


def test_browse_art_sets_art_path_and_marks_overridden(tmp_path, dialog, monkeypatch):
    queue_dir = str(tmp_path / 'queue')
    _write_entry(queue_dir, 'title-a')
    dialog.load_queue_dir(queue_dir)
    dialog.queueTable.selectRow(0)

    chosen_path = str(tmp_path / 'poster.jpg')
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *a, **k: (chosen_path, 'Images (*.png *.jpg *.jpeg)'))

    dialog._ReviewQueueDialog__browse_art()

    updated = read_entry(queue_dir, 'title-a')
    assert updated.art_path == chosen_path
    assert updated.art_overridden is True


def test_clear_art_resets_override(tmp_path, dialog):
    from pipeline.review import update_entry
    queue_dir = str(tmp_path / 'queue')
    _write_entry(queue_dir, 'title-a')
    dialog.load_queue_dir(queue_dir)
    dialog.queueTable.selectRow(0)
    update_entry(queue_dir, 'title-a', art_path=str(tmp_path / 'poster.jpg'), art_overridden=True)
    dialog._ReviewQueueDialog__reload_queue()
    dialog.queueTable.selectRow(0)

    dialog._ReviewQueueDialog__clear_art()

    updated = read_entry(queue_dir, 'title-a')
    assert updated.art_path is None
    assert updated.art_overridden is False


def test_reload_tmdb_by_id_populates_form_without_saving(tmp_path, dialog, monkeypatch):
    queue_dir = str(tmp_path / 'queue')
    _write_entry(queue_dir, 'title-a')
    dialog.load_queue_dir(queue_dir)
    dialog.queueTable.selectRow(0)
    dialog.movieDbIdField.setText('335984')

    known_meta = BeqMetadata(title='Ready Player One', year='2018', alt_title='RPO', rating='PG-13',
                             runtime='140', the_movie_db='335984', genres=[{'id': 28, 'name': 'Action'}])
    monkeypatch.setattr('pipeline.metadata.tmdb_details_by_id', lambda *a, **k: known_meta)

    dialog._ReviewQueueDialog__reload_tmdb()

    assert dialog.titleField.text() == 'Ready Player One'
    assert dialog.altTitleField.text() == 'RPO'
    assert dialog.yearField.text() == '2018'
    assert dialog.ratingField.text() == 'PG-13'
    assert dialog.runtimeField.text() == '140'
    assert dialog.genresLabel.text() == 'Action'
    assert read_entry(queue_dir, 'title-a').meta == {'title': 'title-a'}  # unchanged -- reload doesn't auto-save


def test_escape_key_does_not_blank_the_dialog(tmp_path, dialog, qtbot):
    '''
    QDialog's default Escape behaviour calls reject(), which would hide() this dialog -- blank when embedded as
    a tab (model/batch.py), a way to lose review progress even standalone. reject() is overridden to a no-op.
    '''
    queue_dir = str(tmp_path / 'queue')
    _write_entry(queue_dir, 'title-a')
    dialog.load_queue_dir(queue_dir)
    dialog.show()

    dialog.reject()
    qtbot.keyClick(dialog, Qt.Key.Key_Escape)

    assert dialog.isVisible() is True
    assert dialog.queueTable.model().rowCount() == 1


def test_publish_finished_reports_refused_entries_instead_of_counting_them(dialog, tmp_path, monkeypatch):
    warnings = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda parent, title, text: warnings.append(text))
    _write_entry(str(tmp_path / 'queue'), 'one')
    dialog.load_queue_dir(str(tmp_path / 'queue'))

    dialog._ReviewQueueDialog__on_publish_finished([{'id': 'ok'}, {'id': 'clash', 'error': 'project_conflict'}])

    assert dialog.statusBar.currentMessage() == 'Published 1 title(s), 1 need attention'
    assert len(warnings) == 1 and 'clash' in warnings[0]


def test_the_episodes_in_scope_are_shown_saved_and_can_be_cleared(tmp_path, dialog):
    queue_dir = str(tmp_path / 'queue')
    _write_entry(queue_dir, 'show', meta={'title': 'Show', 'season': '1', 'episodes': [1, 2, 3, 5]})
    dialog.load_queue_dir(queue_dir)
    dialog.queueTable.selectRow(0)

    assert dialog.episodesField.text() == '1-3, 5'

    dialog.episodesField.setText('2-4, 7')
    dialog._ReviewQueueDialog__save_metadata()
    assert read_entry(queue_dir, 'show').meta['episodes'] == [2, 3, 4, 7]
    assert read_entry(queue_dir, 'show').meta['season'] == '1'  # untouched

    dialog.episodesField.setText('')
    dialog._ReviewQueueDialog__save_metadata()
    assert read_entry(queue_dir, 'show').meta['episodes'] == []


def test_an_unparseable_episodes_entry_is_reported_and_nothing_is_saved(tmp_path, dialog):
    queue_dir = str(tmp_path / 'queue')
    _write_entry(queue_dir, 'show', meta={'title': 'Show', 'episodes': [1, 2]})
    dialog.load_queue_dir(queue_dir)
    dialog.queueTable.selectRow(0)

    dialog.editionField.setText('Extended')
    dialog.episodesField.setText('one to three')
    dialog._ReviewQueueDialog__save_metadata()

    assert 'Episodes:' in dialog.metadataStatusLabel.text()
    stored = read_entry(queue_dir, 'show').meta
    assert stored['episodes'] == [1, 2] and 'edition' not in stored


def test_a_film_with_no_episodes_shows_a_blank_field(tmp_path, dialog):
    queue_dir = str(tmp_path / 'queue')
    _write_entry(queue_dir, 'film')
    dialog.load_queue_dir(queue_dir)
    dialog.queueTable.selectRow(0)

    assert dialog.episodesField.text() == ''


def _show_focused(qtbot, dialog, queue_dir, widget_name, on_metadata_tab=True):
    ''' Loads a two-entry queue, selects the first row and gives `widget_name` real keyboard focus. '''
    _write_entry(queue_dir, 'title-a')
    _write_entry(queue_dir, 'title-b')
    dialog.load_queue_dir(queue_dir)
    dialog.show()
    qtbot.waitExposed(dialog)
    dialog.activateWindow()  # offscreen: key events only reach a widget once its window is active
    qtbot.waitActive(dialog)
    dialog.queueTable.selectRow(0)
    if on_metadata_tab:
        dialog.detailTabs.setCurrentWidget(dialog.metadataTab)
    widget = getattr(dialog, widget_name)
    widget.setFocus()
    qtbot.waitUntil(widget.hasFocus)
    return widget


def test_enter_in_a_metadata_field_does_not_accept_the_entry(tmp_path, dialog, qtbot):
    queue_dir = str(tmp_path / 'queue')
    field = _show_focused(qtbot, dialog, queue_dir, 'editionField')

    qtbot.keyClick(field, Qt.Key.Key_Return)

    assert read_entry(queue_dir, 'title-a').status == 'pending'


def test_letter_and_digit_keys_typed_in_a_metadata_field_are_text_not_shortcuts(tmp_path, dialog, qtbot):
    ''' QLineEdit claims printable keys via ShortcutOverride, so the A/S/R/1-9 shortcuts never fire while typing. '''
    queue_dir = str(tmp_path / 'queue')
    field = _show_focused(qtbot, dialog, queue_dir, 'editionField')

    qtbot.keyClicks(field, 'asr12')

    assert field.text() == 'asr12'
    assert read_entry(queue_dir, 'title-a').status == 'pending'


def test_enter_on_the_queue_table_accepts_the_selected_entry(tmp_path, dialog, qtbot):
    ''' The behaviour chunk 20 must keep: Enter still accepts when focus is on the table (or candidate list). '''
    queue_dir = str(tmp_path / 'queue')
    table = _show_focused(qtbot, dialog, queue_dir, 'queueTable', on_metadata_tab=False)

    qtbot.keyClick(table, Qt.Key.Key_Return)

    assert read_entry(queue_dir, 'title-a').status == 'accepted'


def test_enter_on_the_candidate_list_accepts_the_picked_candidate(tmp_path, dialog, qtbot):
    queue_dir = str(tmp_path / 'queue')
    candidates = _show_focused(qtbot, dialog, queue_dir, 'candidateList', on_metadata_tab=False)
    dialog._ReviewQueueDialog__pick_candidate(1)

    qtbot.keyClick(candidates, Qt.Key.Key_Enter)

    accepted = read_entry(queue_dir, 'title-a')
    assert (accepted.status, accepted.chosen_candidate_index) == ('accepted', 1)


def test_accepting_an_entry_that_is_already_decided_is_a_no_op(tmp_path, dialog):
    queue_dir = str(tmp_path / 'queue')
    _write_entry(queue_dir, 'title-a', status='accepted', chosen_candidate_index=0)
    _write_entry(queue_dir, 'title-b', status='published', chosen_candidate_index=1)
    dialog.load_queue_dir(queue_dir)
    dialog.queueTable.selectRow(1)  # title-b

    dialog._ReviewQueueDialog__accept_current()

    assert read_entry(queue_dir, 'title-b').status == 'published'
    assert read_entry(queue_dir, 'title-b').chosen_candidate_index == 1


def _type_into(qtbot, dialog, queue_dir, text='Extended Cut'):
    _write_entry(queue_dir, 'title-a')
    _write_entry(queue_dir, 'title-b')
    dialog.load_queue_dir(queue_dir)
    dialog.queueTable.selectRow(0)
    dialog.editionField.clear()
    qtbot.keyClicks(dialog.editionField, text)


def _answer(monkeypatch, button):
    asked = []
    monkeypatch.setattr(QMessageBox, 'question', lambda *args: asked.append(args[1]) or button)
    return asked


def test_accepting_with_unsaved_metadata_saves_it_when_asked(tmp_path, dialog, qtbot, monkeypatch):
    queue_dir = str(tmp_path / 'queue')
    _type_into(qtbot, dialog, queue_dir)
    asked = _answer(monkeypatch, QMessageBox.StandardButton.Save)

    dialog._ReviewQueueDialog__accept_current()

    entry = read_entry(queue_dir, 'title-a')
    assert asked == ['Unsaved metadata']
    assert (entry.status, entry.meta['edition']) == ('accepted', 'Extended Cut')


def test_accepting_with_unsaved_metadata_can_discard_the_edits(tmp_path, dialog, qtbot, monkeypatch):
    queue_dir = str(tmp_path / 'queue')
    _type_into(qtbot, dialog, queue_dir)
    _answer(monkeypatch, QMessageBox.StandardButton.Discard)

    dialog._ReviewQueueDialog__accept_current()

    entry = read_entry(queue_dir, 'title-a')
    assert entry.status == 'accepted'
    assert 'edition' not in entry.meta


def test_cancelling_the_unsaved_metadata_prompt_leaves_the_entry_pending_and_the_edit_in_place(
        tmp_path, dialog, qtbot, monkeypatch):
    queue_dir = str(tmp_path / 'queue')
    _type_into(qtbot, dialog, queue_dir)
    _answer(monkeypatch, QMessageBox.StandardButton.Cancel)

    dialog._ReviewQueueDialog__accept_current()

    assert read_entry(queue_dir, 'title-a').status == 'pending'
    assert dialog.editionField.text() == 'Extended Cut'


def test_a_failed_save_does_not_go_on_to_accept(tmp_path, dialog, qtbot, monkeypatch):
    queue_dir = str(tmp_path / 'queue')
    _type_into(qtbot, dialog, queue_dir)
    dialog.episodesField.setText('not numbers')
    _answer(monkeypatch, QMessageBox.StandardButton.Save)

    dialog._ReviewQueueDialog__accept_current()

    assert read_entry(queue_dir, 'title-a').status == 'pending'
    assert 'Episodes' in dialog.metadataStatusLabel.text()


def test_accepting_without_edits_does_not_prompt(tmp_path, dialog, monkeypatch):
    queue_dir = str(tmp_path / 'queue')
    _write_entry(queue_dir, 'title-a')
    dialog.load_queue_dir(queue_dir)
    dialog.queueTable.selectRow(0)
    asked = _answer(monkeypatch, QMessageBox.StandardButton.Cancel)

    dialog._ReviewQueueDialog__accept_current()

    assert asked == []
    assert read_entry(queue_dir, 'title-a').status == 'accepted'


def test_saving_metadata_keeps_the_same_entry_selected_and_says_so(tmp_path, dialog, qtbot):
    queue_dir = str(tmp_path / 'queue')
    _write_entry(queue_dir, 'title-a')
    _write_entry(queue_dir, 'title-b')
    dialog.load_queue_dir(queue_dir)
    dialog.queueTable.selectRow(1)

    dialog.editionField.setText('Extended Cut')
    dialog._ReviewQueueDialog__save_metadata()

    assert dialog._ReviewQueueDialog__current_entry().id == 'title-b'
    assert dialog.metadataStatusLabel.text() == 'Saved'


def test_reopen_returns_an_accepted_entry_to_pending_on_the_same_row(tmp_path, dialog):
    queue_dir = str(tmp_path / 'queue')
    _write_entry(queue_dir, 'title-a')
    _write_entry(queue_dir, 'title-b', status='accepted', chosen_candidate_index=1)
    dialog.load_queue_dir(queue_dir)
    dialog.queueTable.selectRow(1)  # accepted entries sort after pending ones
    assert dialog._ReviewQueueDialog__current_entry().id == 'title-b'
    assert dialog.reopenButton.isEnabled()
    assert not dialog.metadataTab.isEnabled()

    dialog.reopenButton.click()

    reopened = read_entry(queue_dir, 'title-b')
    assert (reopened.status, reopened.chosen_candidate_index) == ('pending', None)
    assert dialog._ReviewQueueDialog__current_entry().id == 'title-b'
    assert dialog.metadataTab.isEnabled()
    assert not dialog.reopenButton.isEnabled()


def test_reopen_is_only_offered_for_an_accepted_entry(tmp_path, dialog):
    queue_dir = str(tmp_path / 'queue')
    for entry_id, status, chosen in (('a-pending', 'pending', None), ('b-published', 'published', 0),
                                     ('c-rejected', 'rejected', None)):
        _write_entry(queue_dir, entry_id, status=status, chosen_candidate_index=chosen)
    dialog.load_queue_dir(queue_dir)

    for row in range(dialog.queueTable.model().rowCount()):
        dialog.queueTable.selectRow(row)
        assert not dialog.reopenButton.isEnabled()


def test_accept_is_only_offered_while_an_entry_is_undecided(tmp_path, dialog):
    queue_dir = str(tmp_path / 'queue')
    for entry_id, status, chosen in (('a-pending', 'pending', None), ('b-skipped', 'skipped', None),
                                     ('c-accepted', 'accepted', 0), ('d-published', 'published', 0)):
        _write_entry(queue_dir, entry_id, status=status, chosen_candidate_index=chosen)
    dialog.load_queue_dir(queue_dir)

    offered = {}
    for row in range(dialog.queueTable.model().rowCount()):
        dialog.queueTable.selectRow(row)
        offered[dialog._ReviewQueueDialog__current_entry().status] = dialog.acceptButton.isEnabled()

    assert offered == {'pending': True, 'skipped': True, 'accepted': False, 'published': False}
