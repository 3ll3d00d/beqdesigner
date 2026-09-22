'''
The fixes from the independent review of chunk 27b (implementation-order.md, 27b "Review fixes"): a TMDB answer that arrives
after the page was left, the API key in error text, a refused Accept that must not leave the keyboard in a field, Skip and
Reject behind an unsaveable edit, a blank field unsetting its key, a TMDB id change dropping the season ids, a stale status
sentence, the layout at the window's minimum size, the close order, the badge of a skipped/rejected title, the wording for an
unreadable entry and a download that lands on a title a run is working on. Same style as `test_worklist_metadata.py`, whose
helpers it uses. `import ui.beq` first: see AGENTS.md gotcha 3.
'''
import ui.beq  # noqa: F401 (must come first)

import http.server
import logging
import os
import socket
import threading
from dataclasses import replace
from urllib.parse import quote, quote_plus

import pytest
import requests
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QMessageBox

from model.preferences import TMDB_API_KEY
from model.worklist import WorkListWindow
from model.worklist_metadata import REMOVE, _TmdbJob, apply_changes, badge_alarms, badge_text, redact
from pipeline.review import publication_meta, read_entry, update_entry
from test_worklist_metadata import INCOMPLETE, _clear, _found, _on_metadata, _png, _Response, _type, no_modal  # noqa: F401
from test_worklist_title import REVIEWABLE, _click, _open, _queue, _status, _window  # noqa: F401
from worklist_title_fixture import complete_meta

KEY = 'k3y/with space&sym+bols=1'     # every character a URL has to encode


# --- 1: a TMDB answer after the page was left ---------------------------------------------------------------------------------

def _slow_tmdb(monkeypatch, gate):
    def lookup(*args, **kwargs):
        gate.wait(10)
        return _found()
    monkeypatch.setattr('model.worklist_metadata.tmdb_lookup', lookup)
    monkeypatch.setattr('model.worklist_metadata.tmdb_details_by_id', lookup)


def test_a_tmdb_answer_after_back_is_dropped_and_never_saved_to_the_title_that_was_left(qtbot, tmp_path, monkeypatch):
    gate = threading.Event()
    _slow_tmdb(monkeypatch, gate)
    window = _window(qtbot, tmp_path, REVIEWABLE)
    window._preferences.set(TMDB_API_KEY, 'a-key')
    page = _on_metadata(qtbot, window, 'r-alien')
    before = read_entry(_queue(tmp_path), 'r-alien').meta
    try:
        assert page.metadata.reload_tmdb()
        assert window.close_title()
        with qtbot.waitSignal(page.metadata.tmdb_finished, timeout=10000) as done:
            gate.set()
    finally:
        gate.set()

    assert done.args[0] == '' and not page.metadata.dirty                    # nothing was applied to the hidden page
    assert window.open_title('r-arrival')                                     # opening another title flushes: nothing to flush
    assert read_entry(_queue(tmp_path), 'r-alien').meta == before
    assert read_entry(_queue(tmp_path), 'r-arrival').meta == complete_meta('r-arrival')


def test_a_tmdb_answer_that_comes_back_to_the_same_title_reopened_is_dropped_too(qtbot, tmp_path, monkeypatch):
    gate = threading.Event()
    _slow_tmdb(monkeypatch, gate)
    window = _window(qtbot, tmp_path, REVIEWABLE)
    window._preferences.set(TMDB_API_KEY, 'a-key')
    page = _on_metadata(qtbot, window, 'r-alien')
    try:
        assert page.metadata.reload_tmdb()
        assert window.close_title()
        _on_metadata(qtbot, window, 'r-alien')                                # back to the same title
        assert page.metadata.reloadTmdbButton.isEnabled()                     # the lookup that was in flight is not waited for
        with qtbot.waitSignal(page.metadata.tmdb_finished, timeout=10000) as done:
            gate.set()
    finally:
        gate.set()

    assert done.args[0] == '' and not page.metadata.dirty and page.metadata.titleField.text() == 'r-alien'


def test_a_download_in_flight_when_the_page_is_left_still_lands_on_its_title_and_does_not_block_the_next_one(
        qtbot, tmp_path, monkeypatch):
    ''' The deliberate behaviour stays (the title that asked gets it); leaving must not leave the Download button stuck. '''
    gate = threading.Event()
    monkeypatch.setattr(requests, 'get', lambda url, **kw: (gate.wait(10), _Response(_png()))[1])
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _on_metadata(qtbot, window, 'r-alien')
    _type(qtbot, page.metadata.artUrlField, 'https://example.com/p.png')
    try:
        assert page.metadata.download_art()
        assert window.close_title()
        _on_metadata(qtbot, window, 'r-alien')
        assert page.metadata.downloadArtButton.isEnabled()
        with qtbot.waitSignal(page.metadata.art_finished, timeout=10000):
            gate.set()
    finally:
        gate.set()

    assert read_entry(_queue(tmp_path), 'r-alien').art_path.endswith('r-alien.png')


# --- 2: the API key never reaches the screen or the log -----------------------------------------------------------------------

class _Unauthorised(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(401)
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture
def tmdb_server(monkeypatch):
    ''' A local TMDB that answers 401 to everything. '''
    server = http.server.HTTPServer(('127.0.0.1', 0), _Unauthorised)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr('pipeline.metadata.TMDB_BASE_URL', f'http://127.0.0.1:{server.server_port}/3')
    yield server
    server.shutdown()
    server.server_close()


def _no_key_in(text: str) -> None:
    for form in (KEY, quote(KEY, safe=''), quote(KEY), quote_plus(KEY)):
        assert form not in text, f'{form!r} is in {text!r}'
    assert 'k3y' not in text


def test_the_real_http_error_text_carries_the_key_and_redact_takes_it_out(tmdb_server):
    ''' Guards the premise: `requests` does put the key in the exception text. '''
    from pipeline.metadata import tmdb_lookup
    with pytest.raises(requests.HTTPError) as caught:
        tmdb_lookup('Heat', '1995', KEY)
    assert 'k3y' in str(caught.value) or quote_plus(KEY) in str(caught.value)

    cleaned = redact(str(caught.value), KEY)

    _no_key_in(cleaned)
    assert '401' in cleaned and 'api_key=***' in cleaned and 'query=Heat' in cleaned      # the rest of the sentence stays


def test_redact_masks_the_parameter_and_the_secret_however_it_is_written():
    assert redact('url: http://h/3/x?api_key=abcdef123456&append=x') == 'url: http://h/3/x?api_key=***&append=x'
    assert redact("(url='http://h/x?API_KEY=abc')") == "(url='http://h/x?API_KEY=***')"
    assert redact('the key is secret-key-1234 here', 'secret-key-1234') == 'the key is *** here'
    assert redact(f'encoded {quote(KEY, safe="")} and {quote_plus(KEY)}', KEY) == 'encoded *** and ***'
    assert redact('a short one: k', 'k') == 'a short one: k'                 # too short to be a key: it would mask words
    assert redact('nothing to hide') == 'nothing to hide'


def test_a_tmdb_http_error_on_the_status_line_has_no_key(qtbot, tmp_path, tmdb_server, no_modal):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    window._preferences.set(TMDB_API_KEY, KEY)
    page = _on_metadata(qtbot, window, 'r-alien')

    with qtbot.waitSignal(page.metadata.tmdb_finished, timeout=10000) as done:
        assert page.metadata.reload_tmdb()

    _no_key_in(done.args[0])
    _no_key_in(page.metadata.metadataStatusLabel.text())
    assert 'TMDB lookup failed' in done.args[0] and '401' in done.args[0]


def test_a_tmdb_connection_error_has_no_key_either(qtbot, tmp_path, monkeypatch):
    with socket.socket() as sock:                       # a port nothing listens on
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    monkeypatch.setattr('pipeline.metadata.TMDB_BASE_URL', f'http://127.0.0.1:{port}/3')
    job = _TmdbJob(KEY, '', 'Heat', '1995', 'movie')
    seen = []
    job.signals.failed.connect(seen.append)

    job.run()

    assert len(seen) == 1 and 'TMDB lookup failed' in seen[0]
    _no_key_in(seen[0])


def test_an_unexpected_error_is_redacted_in_the_message_and_in_the_log(qtbot, monkeypatch, caplog):
    def broken(*args, **kwargs):
        raise ValueError(f'cannot read http://h/3/movie/1?api_key={quote_plus(KEY)}&x=1 or {KEY}')
    monkeypatch.setattr('model.worklist_metadata.tmdb_lookup', broken)
    job = _TmdbJob(KEY, '', 'Heat', '1995', 'movie')
    seen = []
    job.signals.failed.connect(seen.append)

    with caplog.at_level(logging.DEBUG, logger='worklist'):
        job.run()

    _no_key_in(seen[0])
    assert 'ValueError' in seen[0] and 'TMDB lookup for Heat failed' in caplog.text
    _no_key_in(caplog.text)


# --- 3: a refused Accept leaves the keyboard where it was ---------------------------------------------------------------------

def test_a_refused_accept_key_does_not_move_the_keyboard_into_a_field_so_the_next_one_types_nothing(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, [('r-alien', {'meta': INCOMPLETE}), ('r-arrival', {})])
    page = _open(qtbot, window, 'r-alien')
    year = page.metadata.yearField

    for _ in range(3):                              # a retry, or the key auto-repeating
        qtbot.keyClick(page.candidateList, Qt.Key.Key_A)
    qtbot.keyClick(page.candidateList, Qt.Key.Key_Return)
    qtbot.keyClick(page.candidateList, Qt.Key.Key_Return)

    assert year.text() == '' and not page.metadata.dirty and _status(tmp_path, 'r-alien') == 'pending'
    assert page.candidateList.hasFocus()            # the keyboard stayed where it was
    assert 'year' not in read_entry(_queue(tmp_path), 'r-alien').meta
    assert page.rightTabs.currentIndex() == 1       # ... but the person is shown what is missing:
    assert year.styleSheet() and not page.metadata.titleField.styleSheet()
    assert page.decisionLabel.text().startswith('Fill in the missing metadata first')


def test_the_mark_on_the_missing_field_goes_when_it_is_typed_in_or_another_title_is_shown(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, [('r-alien', {'meta': INCOMPLETE}), ('r-arrival', {'meta': INCOMPLETE})])
    page = _open(qtbot, window, 'r-alien')
    qtbot.keyClick(page.candidateList, Qt.Key.Key_A)
    assert page.metadata.yearField.styleSheet()

    _type(qtbot, page.metadata.yearField, '1979')
    assert page.metadata.yearField.styleSheet() == ''

    _clear(qtbot, page.metadata.yearField)
    page.candidateList.setFocus()
    qtbot.keyClick(page.candidateList, Qt.Key.Key_A)
    assert page.metadata.yearField.styleSheet()
    page.confirm_discard = lambda reason: True
    page.metadata.discard()
    page.next()
    assert page.metadata.yearField.styleSheet() == ''


# --- 4: Skip and Reject are not trapped behind an edit that cannot be saved -------------------------------------------------

@pytest.mark.parametrize('decision, status', [('skip', 'skipped'), ('reject', 'rejected')])
def test_skip_and_reject_ask_discard_or_cancel_when_the_edit_cannot_be_saved(qtbot, tmp_path, decision, status):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _on_metadata(qtbot, window, 'r-alien')
    before = read_entry(_queue(tmp_path), 'r-alien').meta
    asked = []
    _type(qtbot, page.metadata.episodesField, 'abc')

    page.confirm_discard = lambda reason: asked.append(reason) or False           # Cancel
    assert getattr(page, decision)() is False
    assert _status(tmp_path, 'r-alien') == 'pending' and page.metadata.dirty and page.current_id == 'r-alien'
    assert len(asked) == 1 and asked[0].startswith('Not saved: Episodes:')

    page.confirm_discard = lambda reason: asked.append(reason) or True            # Discard
    assert getattr(page, decision)() is True

    entry = read_entry(_queue(tmp_path), 'r-alien')
    assert entry.status == status and entry.meta == before                       # decided; the bad edit was dropped, not written
    assert not page.metadata.dirty and page.current_id == 'r-arrival'


def test_accept_is_still_held_back_by_an_edit_that_cannot_be_saved_and_never_offers_to_discard_it(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _on_metadata(qtbot, window, 'r-alien')
    page.confirm_discard = lambda reason: pytest.fail('Accept must wait for the edit, not offer to drop it')
    _type(qtbot, page.metadata.episodesField, 'abc')

    assert page.accept() is False

    assert _status(tmp_path, 'r-alien') == 'pending' and page.metadata.dirty
    assert page.decisionLabel.text().startswith('Not moved: Not saved: Episodes:')
    page.confirm_discard = lambda reason: True      # for the window closing at the end of the test


def test_skip_with_a_good_edit_saves_it_and_asks_nothing(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _on_metadata(qtbot, window, 'r-alien')
    page.confirm_discard = lambda reason: pytest.fail('nothing to discard')
    _type(qtbot, page.metadata.noteField, 'kept')

    assert page.skip()

    entry = read_entry(_queue(tmp_path), 'r-alien')
    assert entry.status == 'skipped' and entry.meta['note'] == 'kept'


# --- 5: a blank field unsets its key so a default applies again ---------------------------------------------------------------

def test_a_blank_audio_types_title_or_year_is_removed_not_written_empty(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _on_metadata(qtbot, window, 'r-alien')

    _clear(qtbot, page.metadata.audioTypesField)
    _clear(qtbot, page.metadata.yearField)
    _clear(qtbot, page.metadata.titleField)
    assert page.flush()

    saved = read_entry(_queue(tmp_path), 'r-alien').meta
    assert 'audio_types' not in saved and 'year' not in saved and 'title' not in saved
    assert page.badgeLabel.text() == \
        'Not ready to publish: title is required; year is required; at least one audio type is required'


def test_clearing_audio_types_lets_the_profiles_default_apply_again(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    settings = window.setup.settings
    object.__setattr__(settings, 'meta_defaults', {'audio_types': ['Atmos']})
    try:
        page = _on_metadata(qtbot, window, 'r-alien')
        assert page.badgeLabel.text() == 'Metadata complete'
        _clear(qtbot, page.metadata.audioTypesField)
        assert page.flush()

        assert 'audio_types' not in read_entry(_queue(tmp_path), 'r-alien').meta
        assert page.badgeLabel.text() == 'Metadata complete' and page.acceptButton.isEnabled()   # the default, not "required"
        accepted = replace(read_entry(_queue(tmp_path), 'r-alien'), status='accepted', chosen_candidate_index=0)
        assert publication_meta(accepted, settings.meta_defaults).audio_types == ['Atmos']     # publish would use it too
    finally:
        object.__setattr__(settings, 'meta_defaults', None)


def test_everything_that_reads_the_meta_copes_with_a_title_year_and_audio_types_that_are_absent(qtbot, tmp_path):
    ''' What the removal relies on: no consumer indexes the key. '''
    from pipeline.library.status import metadata_problems
    from pipeline.metadata import validate
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _on_metadata(qtbot, window, 'r-alien')
    for box in (page.metadata.audioTypesField, page.metadata.yearField, page.metadata.titleField):
        _clear(qtbot, box)
    assert page.flush()

    entry = read_entry(_queue(tmp_path), 'r-alien')
    assert metadata_problems(entry.meta) == ('title is required', 'year is required', 'at least one audio type is required')
    accepted = replace(entry, status='accepted', chosen_candidate_index=0)
    assert validate(publication_meta(accepted, None)) == \
        ['title is required', 'year is required', 'at least one audio type is required']
    assert page.titleLabel.text().startswith('Alien')     # the header falls back to the index's row, not an error
    page.reload()
    assert page.metadata.titleField.text() == '' and page.metadata.audioTypesField.text() == ''
    assert page.next() and page.previous()


# --- 6: a new TMDB id drops the old series' season ids ----------------------------------------------------------------------

def test_a_change_of_the_tmdb_id_drops_the_season_id_and_episode_count_too():
    meta = {'the_movie_db': '1', 'season': '2', 'season_id': '77', 'season_episode_count': 8}

    assert apply_changes(meta, {'the_movie_db': '1'}) == meta                                   # the same show: still right
    assert apply_changes(meta, {'the_movie_db': '2'}) == {'the_movie_db': '2', 'season': '2'}
    assert apply_changes(meta, {'the_movie_db': REMOVE}) == {'season': '2'}
    assert apply_changes({'season_id': '77'}, {'the_movie_db': '5'}) == {'the_movie_db': '5'}
    assert apply_changes(meta, {'note': 'x'})['season_id'] == '77'                              # nothing to do with the show


def _series(**more):
    return {**complete_meta('r-alien'), 'the_movie_db': '1', 'season': '2', 'season_id': '77', 'season_episode_count': 8, **more}


def test_typing_another_tmdb_id_drops_the_old_seasons_ids(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, [('r-alien', {'meta': _series()}), ('r-arrival', {})])
    page = _on_metadata(qtbot, window, 'r-alien')

    _type(qtbot, page.metadata.movieDbIdField, '999')
    assert page.flush()

    saved = read_entry(_queue(tmp_path), 'r-alien').meta
    assert saved['the_movie_db'] == '999' and saved['season'] == '2'
    assert 'season_id' not in saved and 'season_episode_count' not in saved


def test_reloading_from_tmdb_with_another_id_drops_them_and_with_the_same_id_keeps_them(qtbot, tmp_path, monkeypatch):
    window = _window(qtbot, tmp_path, [('r-alien', {'meta': _series()}), ('r-arrival', {})])
    window._preferences.set(TMDB_API_KEY, 'a-key')
    page = _on_metadata(qtbot, window, 'r-alien')
    monkeypatch.setattr('model.worklist_metadata.tmdb_details_by_id', lambda *a, **k: _found(the_movie_db='1'))

    with qtbot.waitSignal(page.metadata.tmdb_finished, timeout=10000):
        page.metadata.reload_tmdb()
    assert page.flush()
    assert read_entry(_queue(tmp_path), 'r-alien').meta['season_id'] == '77'                    # the same series

    monkeypatch.setattr('model.worklist_metadata.tmdb_lookup', lambda *a, **k: _found(the_movie_db='555'))
    _clear(qtbot, page.metadata.movieDbIdField)
    assert page.flush()                                                                          # the id is gone: no by-id lookup
    with qtbot.waitSignal(page.metadata.tmdb_finished, timeout=10000):
        page.metadata.reload_tmdb()
    assert page.flush()

    saved = read_entry(_queue(tmp_path), 'r-alien').meta
    assert saved['the_movie_db'] == '555' and 'season_id' not in saved and 'season_episode_count' not in saved


# --- 7 and 11: the sentence about the panel goes when it stops being true ----------------------------------------------------

def test_the_sentence_about_a_run_goes_when_the_run_ends(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _on_metadata(qtbot, window, 'r-alien')

    window.model.set_running({'r-alien': 'design'})
    assert page.metadata.metadataStatusLabel.text().startswith('A run is working on this title now')
    window.model.set_running({})

    assert page.metadata.metadataStatusLabel.text() == ''
    assert not page.metadata.noteField.isReadOnly()


def test_the_sentence_about_a_run_goes_after_a_refresh_that_kept_the_edits_too(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _on_metadata(qtbot, window, 'r-alien')
    window.model.set_running({'r-alien': 'design'})
    assert 'A run is working' in page.metadata.metadataStatusLabel.text()

    page.metadata._running = lambda: {}          # the run ended without the model's signal reaching the panel
    page.reload()

    assert page.metadata.metadataStatusLabel.text() == ''


def test_a_message_that_is_not_about_the_run_is_not_wiped_when_the_run_ends(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _on_metadata(qtbot, window, 'r-alien')
    _type(qtbot, page.metadata.episodesField, 'abc')
    window.model.set_running({'r-alien': 'design'})
    assert page.flush() is False and 'Not saved: a run is working' in page.metadata.metadataStatusLabel.text()

    window.model.set_running({})

    assert 'Not saved: a run is working' in page.metadata.metadataStatusLabel.text()


def test_the_no_entry_sentence_goes_when_the_entry_appears(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, [('r-arrival', {})])
    page = _open(qtbot, window, 'r-alien')
    assert page.metadata.metadataStatusLabel.text().startswith('This title has no queue entry yet')

    from worklist_title_fixture import write_entry
    write_entry(_queue(tmp_path), 'r-alien')
    page.reload()

    assert page.metadata.metadataStatusLabel.text() == ''


def test_an_unreadable_entry_is_said_to_be_unreadable_not_undesigned(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    with open(os.path.join(_queue(tmp_path), 'r-alien.json'), 'w') as damaged:
        damaged.write('{ this is not json')

    page = _open(qtbot, window, 'r-alien')

    text = page.metadata.metadataStatusLabel.text()
    assert 'could not be read' in text and 'until it is designed' not in text and 'no queue entry yet' not in text
    assert page.noticeLabel.text().startswith('The queue entry could not be read')
    assert page.metadata.titleField.isReadOnly()


# --- 8: the form is usable at the window's minimum size -------------------------------------------------------------------------

@pytest.mark.parametrize('size', [(860, 520), (1100, 700)])
def test_the_form_is_usable_at_the_minimum_window_size(qtbot, tmp_path, size):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    window.resize(*size)
    page = _on_metadata(qtbot, window, 'r-alien', more=True)
    qtbot.wait(100)

    panel = page.metadata
    assert panel.titleField.width() >= 200 and panel.audioTypeChecks[0].width() >= 40
    assert not panel.metadataScroll.horizontalScrollBar().isVisible()                    # never a sideways scroll
    assert panel.artworkBox.isVisible() and panel.browseArtButton.isVisible()            # the artwork is in the same column
    form = panel.metadataFormHost
    assert form.width() <= panel.metadataScroll.viewport().width()
    page.rightTabs.setCurrentIndex(0)
    qtbot.wait(50)
    assert page.previewChart.isVisible() and page.previewChart.width() >= 250


# --- 9: closing: the run question comes before the edit question -----------------------------------------------------------------

def _closing(qtbot, tmp_path, monkeypatch, run_answer, discard_answer):
    ''' A window with a run in progress (faked) and an unsaveable edit; records the order the questions are asked in. '''
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _on_metadata(qtbot, window, 'r-alien')
    order = []
    monkeypatch.setattr(WorkListWindow, 'is_running', property(lambda self: True))
    monkeypatch.setattr(WorkListWindow, 'cancel_run', lambda self: order.append('cancel'))
    monkeypatch.setattr(QMessageBox, 'question', staticmethod(
        lambda *a, **k: order.append('run?') or (QMessageBox.StandardButton.Yes if run_answer else QMessageBox.StandardButton.No)))
    page.confirm_discard = lambda reason: order.append('edit?') or discard_answer
    _type(qtbot, page.metadata.episodesField, 'abc')
    return window, page, order


def test_answering_no_to_the_run_question_keeps_the_edit_and_never_asks_about_it(qtbot, tmp_path, monkeypatch):
    window, page, order = _closing(qtbot, tmp_path, monkeypatch, run_answer=False, discard_answer=True)

    assert window.close() is False

    assert order == ['run?'] and window.isVisible() and page.metadata.dirty
    assert page.metadata.episodesField.text() == 'abc'                          # before: discarded, and the window still open


def test_cancel_on_the_edit_question_keeps_everything_and_does_not_cancel_the_run(qtbot, tmp_path, monkeypatch):
    window, page, order = _closing(qtbot, tmp_path, monkeypatch, run_answer=True, discard_answer=False)

    assert window.close() is False

    assert order == ['run?', 'edit?'] and window.isVisible() and page.metadata.dirty


def test_yes_then_discard_cancels_the_run_and_closes(qtbot, tmp_path, monkeypatch):
    window, page, order = _closing(qtbot, tmp_path, monkeypatch, run_answer=True, discard_answer=True)

    window.close()

    assert order == ['run?', 'edit?', 'cancel'] and not window.isVisible()


# --- 10: the badge of a skipped or rejected title -------------------------------------------------------------------------------

def test_only_a_title_the_index_judges_is_alarmed_about():
    problems = ['year is required']
    assert [badge_alarms(problems, s) for s in ('pending', 'accepted', 'published', 'skipped', 'rejected')] == \
        [True, True, True, False, False]
    assert not badge_alarms([], 'pending')
    assert badge_text(problems, 'skipped', False) == 'Metadata incomplete: year is required'
    assert badge_text(problems, 'rejected', True) == 'Metadata incomplete: year is required (unsaved edits)'
    assert badge_text(problems, 'pending', False) == 'Not ready to publish: year is required'


@pytest.mark.parametrize('status, alarm', [('pending', True), ('accepted', True), ('published', True),
                                           ('skipped', False), ('rejected', False)])
def test_the_page_alarms_only_where_the_index_says_metadata_incomplete(qtbot, tmp_path, status, alarm):
    window = _window(qtbot, tmp_path, [('r-alien', {'meta': INCOMPLETE, 'status': status}), ('r-arrival', {})])
    page = _open(qtbot, window, 'r-alien')

    text = page.badgeLabel.text()
    assert text.startswith('Not ready to publish: ') if alarm else text.startswith('Metadata incomplete: ')
    assert page.rightTabs.tabText(1) == ('Metadata !' if alarm else 'Metadata')
    assert ('color' in page.badgeLabel.styleSheet()) is alarm            # the warning colour, or none


# --- 12: a download that lands on a title a run is now working on -----------------------------------------------------------------

def test_a_download_that_finishes_while_a_run_works_on_the_title_is_not_applied(qtbot, tmp_path, monkeypatch):
    gate = threading.Event()
    monkeypatch.setattr(requests, 'get', lambda url, **kw: (gate.wait(10), _Response(_png()))[1])
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _on_metadata(qtbot, window, 'r-alien')
    _type(qtbot, page.metadata.artUrlField, 'https://example.com/p.png')
    try:
        assert page.metadata.download_art()
        window.model.set_running({'r-alien': 'design'})
        with qtbot.waitSignal(page.metadata.art_finished, timeout=10000) as done:
            gate.set()
    finally:
        gate.set()

    entry = read_entry(_queue(tmp_path), 'r-alien')
    assert entry.art_path is None and entry.art_overridden is False                # the in-flight design would have replaced it
    assert 'a run is working on r-alien' in done.args[0]
    assert page.metadata.metadataStatusLabel.text() == done.args[0]
    assert page.metadata.artUrlField.text() != ''                                  # the URL is still there to try again
    window.model.set_running({})
    assert page.metadata.downloadArtButton.isEnabled()


def test_the_refusal_reaches_the_window_when_the_title_is_no_longer_shown(qtbot, tmp_path, monkeypatch):
    gate = threading.Event()
    monkeypatch.setattr(requests, 'get', lambda url, **kw: (gate.wait(10), _Response(_png()))[1])
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _on_metadata(qtbot, window, 'r-alien')
    _type(qtbot, page.metadata.artUrlField, 'https://example.com/p.png')
    try:
        assert page.metadata.download_art()
        window.model.set_running({'r-alien': 'design'})
        page.next()
        with qtbot.waitSignal(page.metadata.art_finished, timeout=10000):
            gate.set()
    finally:
        gate.set()

    assert read_entry(_queue(tmp_path), 'r-alien').art_path is None
    assert 'a run is working on r-alien' in window.statusBar.currentMessage()
