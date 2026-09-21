'''
model/worklist_metadata.py, model/worklist_artwork.py and the parts of model/worklist_title.py that use them, chunk 27b: the
title page's metadata form (Essentials / More), the validity badge, autosave on focus-out and the flush before the page
moves, Accept held back while the metadata is incomplete, TMDB Reload and the artwork -- driven like a person would through
the work list window, over real queue entries, with TMDB and HTTP replaced (never the network). Three tests go through a
REAL scan (and one, real git repositories): an edit to an accepted title's metadata makes it `review` again, and an edit to a
*published* title's metadata or artwork makes it Publish: out of date, once the page is left.
`import ui.beq` first: see AGENTS.md gotcha 3.
'''
import ui.beq  # noqa: F401 (must come first)

import os
import threading
import time
from types import SimpleNamespace

import pytest
import requests
from qtpy.QtCore import QBuffer, QIODevice, Qt
from qtpy.QtGui import QImage, QShortcut
from qtpy.QtWidgets import QApplication, QComboBox, QPlainTextEdit, QPushButton

from model.preferences import LIBRARY_PROFILE_PATH, TMDB_API_KEY
from model.worklist import WorkListWindow
from model.worklist_artwork import ArtworkError, cache_path, check_local_image, download_artwork, image_extension
from model.worklist_metadata import REMOVE, apply_changes, badge_text, field_texts, parse_audio_types, save_meta_changes
from pipeline.library.index import LibraryIndex
from pipeline.metadata import BeqMetadata
from pipeline.review import read_entry, update_entry
from test_pipeline_library_commit import repos  # noqa: F401 (a fixture)
from test_pipeline_library_index import FakeSource, _entry as _real_entry, _extracted, _item
from test_worklist_real_pipeline import _answer, _needs as _needs_of, _open as _open_scanned, _prefs as _real_prefs, \
    _profile_file, _world as _published_world
from test_worklist_title import REVIEWABLE, _click, _designer, _focus, _open, _prefs, _queue, _rows, _status, _window  # noqa: F401
from worklist_title_fixture import complete_meta, write_entry

INCOMPLETE = {'title': 'Alien'}    # what a library run leaves: no year, no audio types


# --- helpers -----------------------------------------------------------------------------------------------------------------

def _meta(tmp_path, entry_id='r-alien'):
    return read_entry(_queue(tmp_path), entry_id).meta


def _on_metadata(qtbot, window, title_id, more=False):
    ''' Opens the page on `title_id` with the Metadata tab showing (and the More section, if asked). '''
    page = _open(qtbot, window, title_id)
    page.confirm_discard = lambda reason: True      # a test that leaves an unsavable edit behind must not wait for a person
    page.rightTabs.setCurrentIndex(1)
    if more:
        page.metadata.moreButton.setChecked(True)
    return page


def _type(qtbot, box, text):
    ''' What a person does: click into the box, replace what is there, type. Nothing is saved yet. '''
    _focus(qtbot, box)
    box.selectAll()
    qtbot.keyClicks(box, text)


def _clear(qtbot, box):
    _focus(qtbot, box)
    box.selectAll()
    qtbot.keyClick(box, Qt.Key.Key_Delete)


def _png(width=4, height=6) -> bytes:
    return _image_bytes(width, height, 'PNG')


def _image_bytes(width, height, fmt) -> bytes:
    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(0xFF3366)
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    assert image.save(buffer, fmt)
    return bytes(buffer.data())


class _Response:
    def __init__(self, data=b'', status=200, chunks=None):
        self.data, self.status, self._chunks = data, status, chunks

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(f'{self.status} Client Error')

    def iter_content(self, chunk_size=1):
        yield from (self._chunks if self._chunks is not None else [self.data])


@pytest.fixture
def no_modal():
    ''' After a test: nothing may have been left waiting for an answer. '''
    yield
    assert QApplication.activeModalWidget() is None


def _found(**more) -> BeqMetadata:
    fields = dict(title='Heat', year='1995', alt_title='', overview='cops and robbers', rating='R', poster='/h.jpg',
                  runtime='170', the_movie_db='949', genres=[{'id': 80, 'name': 'Crime'}],
                  collection={'id': 1, 'name': 'A Collection'})
    fields.update(more)
    return BeqMetadata(**fields)


def _tmdb(monkeypatch, result=None, error=None, calls=None):
    ''' Replaces TMDB: `calls` collects (function, arguments). '''
    def lookup(*args, **kwargs):
        if calls is not None:
            calls.append(('lookup', args, kwargs))
        if error:
            raise error
        return result

    def details(*args, **kwargs):
        if calls is not None:
            calls.append(('details', args, kwargs))
        if error:
            raise error
        return result

    monkeypatch.setattr('model.worklist_metadata.tmdb_lookup', lookup)
    monkeypatch.setattr('model.worklist_metadata.tmdb_details_by_id', details)


# --- what is said and written, with no widgets --------------------------------------------------------------------------------

def test_the_boxes_show_the_meta_the_way_a_person_reads_it():
    texts = field_texts({'title': 'Heat', 'year': 1995, 'audio_types': ['Atmos', 'DTS-HD'], 'episodes': [1, 2, 3, 5],
                         'gain': None})

    assert texts['title'] == 'Heat' and texts['year'] == '1995' and texts['gain'] == ''
    assert texts['audio_types'] == 'Atmos, DTS-HD' and texts['episodes'] == '1-3, 5' and texts['note'] == ''
    assert field_texts({'audio_types': 'Atmos', 'episodes': 'x'})['audio_types'] == 'Atmos'   # odd data is shown, not lost
    assert parse_audio_types(' Atmos ,, DTS ') == ['Atmos', 'DTS']


def test_a_change_sets_or_removes_a_key_and_a_new_season_drops_the_old_seasons_tmdb_ids():
    meta = {'title': 'Show', 'note': 'old', 'season': '1', 'season_id': '77', 'season_episode_count': 8}

    assert apply_changes(meta, {'note': REMOVE, 'year': '2001'}) == \
        {'title': 'Show', 'season': '1', 'season_id': '77', 'season_episode_count': 8, 'year': '2001'}
    assert apply_changes(meta, {'season': '1'}) == meta                      # the same season: the ids are still right
    changed = apply_changes(meta, {'season': '2'})
    assert changed == {'title': 'Show', 'note': 'old', 'season': '2'}
    assert apply_changes(meta, {'season': REMOVE}) == {'title': 'Show', 'note': 'old'}
    assert meta['note'] == 'old'                                             # the input is not modified


def test_the_badge_says_what_is_missing_or_that_the_metadata_is_complete():
    assert badge_text(['year is required'], 'pending', False) == 'Not ready to publish: year is required'
    assert badge_text(['a', 'b'], 'accepted', True) == 'Not ready to publish: a; b (unsaved edits)'
    assert badge_text([], 'pending', False) == 'Metadata complete'
    assert badge_text([], 'accepted', False) == badge_text([], 'published', False) == 'Ready to publish'
    assert badge_text([], 'rejected', True) == 'Metadata complete (unsaved edits)'


def test_the_edit_is_merged_into_the_entry_as_it_is_on_disk_not_as_it_was_shown(tmp_path):
    write_entry(str(tmp_path), 'a', meta={**complete_meta('a'), 'note': 'mine'})
    update_entry(str(tmp_path), 'a', meta={**complete_meta('a'), 'note': 'mine', 'season_id': '9', 'rating': 'PG'})   # a run

    entry, wrote = save_meta_changes(str(tmp_path), 'a', {'year': '1999'})

    assert wrote and entry.meta['year'] == '1999'
    assert entry.meta['season_id'] == '9' and entry.meta['rating'] == 'PG' and entry.meta['note'] == 'mine'
    assert read_entry(str(tmp_path), 'a').meta == entry.meta
    assert save_meta_changes(str(tmp_path), 'a', {'year': '1999'})[1] is False    # nothing to write
    with pytest.raises(FileNotFoundError):
        save_meta_changes(str(tmp_path), 'nobody', {'year': '1'})


# --- artwork, with no widgets -------------------------------------------------------------------------------------------------

def test_only_complete_png_and_jpeg_images_are_images():
    assert image_extension(_png()) == '.png' and image_extension(_image_bytes(4, 4, 'JPG')) == '.jpg'
    for bad in (b'<html>nope</html>', b'', b'GIF89a', _png()[:40]):
        with pytest.raises(ArtworkError):
            image_extension(bad)


def test_a_local_file_must_be_an_image_that_exists(tmp_path):
    good, bad = tmp_path / 'p.png', tmp_path / 'p.txt'
    good.write_bytes(_png())
    bad.write_text('not an image')

    check_local_image(str(good))
    for path in (str(bad), str(tmp_path / 'missing.png'), str(tmp_path)):
        with pytest.raises(ArtworkError):
            check_local_image(path)


def test_a_download_is_kept_only_in_the_art_cache_under_a_safe_name(tmp_path):
    inside = os.path.join(str(tmp_path), '_art_cache')

    assert cache_path(str(tmp_path), 'fs-a', '.png') == os.path.join(inside, 'fs-a.png')
    hostile = cache_path(str(tmp_path), '../../etc/passwd', '.jpg')
    assert os.path.dirname(hostile) == inside and os.path.basename(hostile) == '_.._etc_passwd.jpg'   # one file name, in the cache
    assert os.path.dirname(cache_path(str(tmp_path), '/abs/path', '.png')) == inside


def test_a_download_writes_an_image_into_the_cache_and_nothing_else(tmp_path):
    path = download_artwork(' https://example.com/p.jpg ', str(tmp_path), 'fs-a',
                            get=lambda url, **kw: _Response(_png()))

    assert path == str(tmp_path / '_art_cache' / 'fs-a.png')            # named for what it is, not for the URL
    assert open(path, 'rb').read() == _png()
    assert os.listdir(tmp_path / '_art_cache') == ['fs-a.png']           # no temporary left behind


@pytest.mark.parametrize('url,response,message', [
    ('ftp://example.com/p.png', _Response(_png()), 'http'),
    ('file:///etc/passwd', _Response(_png()), 'http'),
    ('https://example.com/p.png', _Response(status=404), 'download failed'),
    ('https://example.com/p.png', _Response(b'<html>a page</html>'), 'not a png or jpeg'),
    ('https://example.com/p.png', _Response(chunks=[b'x' * (11 * 1024 * 1024)] * 2), 'bigger than'),
])
def test_a_download_that_is_not_an_image_is_refused_and_writes_nothing(tmp_path, url, response, message):
    with pytest.raises(ArtworkError, match=message):
        download_artwork(url, str(tmp_path), 'fs-a', get=lambda u, **kw: response)

    assert not os.path.exists(tmp_path / '_art_cache') or os.listdir(tmp_path / '_art_cache') == []


def test_a_network_error_is_an_artwork_error(tmp_path):
    def get(url, **kwargs):
        raise requests.ConnectionError('no route')

    with pytest.raises(ArtworkError, match='no route'):
        download_artwork('https://example.com/p.png', str(tmp_path), 'a', get=get)


# --- the form ---------------------------------------------------------------------------------------------------------------------

def test_the_metadata_tab_shows_the_entrys_meta_with_the_other_fields_under_more(qtbot, tmp_path):
    meta = {**complete_meta('r-alien'), 'title': 'Alien', 'alt_title': 'Alien 1979', 'season': '2', 'episodes': [1, 2, 3],
            'the_movie_db': '348', 'note': 'n', 'genres': [{'id': 1, 'name': 'Horror'}, {'id': 2, 'name': 'SciFi'}]}
    window = _window(qtbot, tmp_path, [('r-alien', {'meta': meta}), ('r-arrival', {})])
    page = _on_metadata(qtbot, window, 'r-alien')
    panel = page.metadata

    assert page.rightTabs.tabText(1) == 'Metadata' and page.rightTabs.currentWidget().isVisibleTo(page)
    assert (panel.titleField.text(), panel.yearField.text(), panel.audioTypesField.text()) == ('Alien', '2001', 'DD 5.1')
    assert (panel.seasonField.text(), panel.episodesField.text(), panel.movieDbIdField.text()) == ('2', '1-3', '348')
    assert panel.noteField.text() == 'n' and panel.altTitleField.text() == 'Alien 1979'
    assert not panel.moreBox.isVisibleTo(panel)                       # collapsed until asked for
    assert not panel.altTitleField.isVisibleTo(panel) and panel.titleField.isVisibleTo(panel)
    panel.moreButton.setChecked(True)
    assert panel.moreBox.isVisibleTo(panel) and panel.altTitleField.isVisibleTo(panel)
    assert panel.genresLabel.text() == 'Horror, SciFi'
    page.next()                                                      # another title: its own values, not Alien's
    assert panel.titleField.text() == 'r-arrival' and panel.altTitleField.text() == '' and panel.seasonField.text() == ''


def test_a_title_with_no_queue_entry_has_a_form_with_nothing_to_edit(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _on_metadata(qtbot, window, 'x-gravity')

    assert page.metadata.titleField.isReadOnly() and 'no queue entry' in page.metadata.metadataStatusLabel.text()
    assert not page.metadata.browseArtButton.isEnabled() and not page.metadata.reloadTmdbButton.isEnabled()
    assert not page.badgeLabel.isVisibleTo(page)
    assert page.metadata.flush() is True


# --- the badge and Accept -----------------------------------------------------------------------------------------------------

def test_the_badge_lists_what_is_missing_and_follows_what_is_typed_before_it_is_saved(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, [('r-alien', {'meta': INCOMPLETE}), ('r-arrival', {})])
    page = _on_metadata(qtbot, window, 'r-alien')

    assert page.badgeLabel.text() == 'Not ready to publish: year is required; at least one audio type is required'
    assert page.rightTabs.tabText(1) == 'Metadata !'
    _type(qtbot, page.metadata.yearField, '1979')
    assert page.badgeLabel.text() == 'Not ready to publish: at least one audio type is required (unsaved edits)'
    _type(qtbot, page.metadata.audioTypesField, 'Atmos')

    assert page.badgeLabel.text() == 'Metadata complete (unsaved edits)' and page.rightTabs.tabText(1) == 'Metadata'
    assert _meta(tmp_path) == {**INCOMPLETE, 'year': '1979'}              # the year was saved as its field was left ...
    assert 'audio_types' not in _meta(tmp_path)                           # ... typing alone in the last one has written nothing
    page.next()
    assert page.badgeLabel.text() == 'Metadata complete'                    # r-arrival's; and Alien's edit was saved
    assert _meta(tmp_path)['year'] == '1979'


def test_the_badge_uses_the_profiles_meta_defaults_as_the_index_does(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, [('r-alien', {'meta': {'title': 'Alien', 'year': '1979'}}), ('r-arrival', {})])
    page = _open(qtbot, window, 'r-alien')
    assert page.badgeLabel.text() == 'Not ready to publish: at least one audio type is required'

    settings = window.setup.settings
    object.__setattr__(settings, 'meta_defaults', {'audio_types': ['Atmos']})   # what the profile's sync: section gives
    try:
        page.reload()
        assert page.badgeLabel.text() == 'Metadata complete' and page.acceptButton.isEnabled()
    finally:
        object.__setattr__(settings, 'meta_defaults', None)


def test_accept_is_not_offered_while_the_metadata_is_incomplete_and_says_why(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, [('r-alien', {'meta': INCOMPLETE}), ('r-arrival', {})])
    page = _open(qtbot, window, 'r-alien')

    assert not page.acceptButton.isEnabled() and page.skipButton.isEnabled() and page.rejectButton.isEnabled()
    assert page.decisionLabel.text() == \
        'Fill in the missing metadata first (Metadata tab): year is required; at least one audio type is required.'
    assert page.accept() is False and _status(tmp_path, 'r-alien') == 'pending'
    assert page.rightTabs.currentIndex() == 1 and page.metadata.yearField.styleSheet()   # shown what is missing ...
    assert not page.metadata.yearField.hasFocus() and page.candidateList.hasFocus()      # ... without moving the keyboard


def test_typing_the_missing_metadata_offers_accept_and_accept_saves_it_first(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, [('r-alien', {'meta': INCOMPLETE}), ('r-arrival', {})])
    page = _on_metadata(qtbot, window, 'r-alien')

    _type(qtbot, page.metadata.yearField, '1979')
    _type(qtbot, page.metadata.audioTypesField, 'Atmos, DTS:X')
    assert page.acceptButton.isEnabled()                                    # live, before any save
    _click(qtbot, page.acceptButton)

    entry = read_entry(_queue(tmp_path), 'r-alien')
    assert entry.status == 'accepted' and entry.meta['audio_types'] == ['Atmos', 'DTS:X'] and entry.meta['year'] == '1979'
    assert page.current_id == 'r-arrival'


def test_skip_and_reject_are_not_held_up_by_incomplete_metadata(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, [('r-alien', {'meta': INCOMPLETE}), ('r-arrival', {'meta': INCOMPLETE})])
    page = _open(qtbot, window, 'r-alien')

    assert page.skip() and _status(tmp_path, 'r-alien') == 'skipped'
    assert page.reject() and _status(tmp_path, 'r-arrival') == 'rejected'


def test_accept_is_refused_if_the_metadata_was_emptied_by_someone_else_since_it_was_shown(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _open(qtbot, window, 'r-alien')
    assert page.acceptButton.isEnabled()
    update_entry(_queue(tmp_path), 'r-alien', meta={'title': 'Alien'})       # another window, or a script

    assert page.accept() is False

    assert _status(tmp_path, 'r-alien') == 'pending'
    assert page.decisionLabel.text().startswith('Not accepted: the metadata is not complete now: year is required')


def test_the_badge_of_an_accepted_title_says_ready_and_a_rejected_ones_only_says_complete(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, [('r-alien', {'status': 'accepted'}), ('r-arrival', {'status': 'rejected'})])
    page = _open(qtbot, window, 'r-alien')

    assert page.badgeLabel.text() == 'Ready to publish'
    page.next()
    assert page.badgeLabel.text() == 'Metadata complete'


# --- saving: on focus-out, and before the page moves ----------------------------------------------------------------------------

def test_an_edit_is_saved_when_the_field_is_left(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _on_metadata(qtbot, window, 'r-alien')
    saved = []
    page.changed.connect(saved.append)

    _type(qtbot, page.metadata.noteField, 'Extended cut')
    assert 'note' not in _meta(tmp_path) and page.metadata.dirty                     # still being typed
    assert page.metadata.saveMetadataButton.isEnabled()
    _focus(qtbot, page.metadata.warningField)                                       # focus-out of the note

    assert _meta(tmp_path)['note'] == 'Extended cut' and not page.metadata.dirty
    assert page.metadata.metadataStatusLabel.text() == 'Saved.' and saved == ['r-alien']
    assert not page.metadata.saveMetadataButton.isEnabled()


def test_enter_saves_the_field_and_decides_nothing_and_clicks_no_button(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _on_metadata(qtbot, window, 'r-alien', more=True)
    clicks = []
    for button in window.findChildren(QPushButton):
        assert not button.autoDefault()
        button.clicked.connect(lambda *_, b=button: clicks.append(b.text()))

    for box in page.metadata._boxes.values():
        _type(qtbot, box, 'x1' if box is not page.metadata.episodesField else '2')
        qtbot.keyClick(box, Qt.Key.Key_Return)
        qtbot.keyClick(box, Qt.Key.Key_Enter)

    assert clicks == [] and _status(tmp_path, 'r-alien') == 'pending' and page.current_id == 'r-alien'
    assert _meta(tmp_path)['note'] == 'x1' and _meta(tmp_path)['episodes'] == [2]   # each Enter saved


def test_the_decision_keys_and_digits_go_to_a_text_field_that_has_focus(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _on_metadata(qtbot, window, 'r-alien')
    _focus(qtbot, page.metadata.noteField)

    qtbot.keyClicks(page.metadata.noteField, 'aSrRs123456789')

    assert page.metadata.noteField.text() == 'aSrRs123456789'
    assert _status(tmp_path, 'r-alien') == 'pending' and page.current_id == 'r-alien' and page.picked == 0


def test_the_decision_keys_do_nothing_from_a_read_only_field_but_still_work_from_the_list(qtbot, tmp_path):
    ''' A read-only line edit does not claim printable keys, so its shortcut fires: it must not decide the title. '''
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _on_metadata(qtbot, window, 'r-alien')
    _focus(qtbot, page.metadata.artPathField)                                      # read only
    assert page.metadata.artPathField.isReadOnly()

    for key in (Qt.Key.Key_A, Qt.Key.Key_S, Qt.Key.Key_R, Qt.Key.Key_2):
        qtbot.keyClick(page.metadata.artPathField, key)

    assert _status(tmp_path, 'r-alien') == 'pending' and page.current_id == 'r-alien' and page.picked == 0
    _focus(qtbot, page.candidateList)
    qtbot.keyClick(page.candidateList, Qt.Key.Key_2)
    assert page.picked == 1                                                        # from the list, they work
    qtbot.keyClick(page.candidateList, Qt.Key.Key_A)
    assert _status(tmp_path, 'r-alien') == 'accepted'


def test_a_plain_text_box_and_an_editable_combo_keep_their_letters_too(qtbot, tmp_path):
    ''' What `WidgetWithChildrenShortcut` does with the other kinds of text input, should the page ever gain one. '''
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _open(qtbot, window, 'r-alien')
    plain, combo = QPlainTextEdit(page), QComboBox(page)
    combo.setEditable(True)
    for widget in (plain, combo):
        widget.show()
        _focus(qtbot, widget)
        qtbot.keyClicks(widget, 'asr12')

    assert plain.toPlainText() == 'asr12' and combo.currentText() == 'asr12'
    assert _status(tmp_path, 'r-alien') == 'pending' and page.picked == 0


def test_the_page_has_no_box_that_takes_typing_but_lets_the_decision_keys_through(qtbot, tmp_path):
    ''' A non-editable combo box does not claim letters; the page's own guard covers it, and this notices a new one. '''
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _open(qtbot, window, 'r-alien')
    combo = QComboBox(page)
    combo.addItems(['alpha', 'sierra', 'romeo'])
    combo.show()
    _focus(qtbot, combo)

    qtbot.keyClicks(combo, 'asr')

    assert _status(tmp_path, 'r-alien') == 'pending' and page.current_id == 'r-alien'


def test_esc_in_a_field_goes_back_to_the_candidate_list_saving_the_edit_and_a_second_esc_goes_back(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _on_metadata(qtbot, window, 'r-alien')
    _type(qtbot, page.metadata.noteField, 'kept')

    qtbot.keyClick(page.metadata.noteField, Qt.Key.Key_Escape)

    assert window.title_page_open and page.candidateList.hasFocus()
    assert _meta(tmp_path)['note'] == 'kept'
    qtbot.keyClick(page.candidateList, Qt.Key.Key_Escape)
    assert not window.title_page_open


@pytest.mark.parametrize('move', [
    lambda page, window: page.next(),
    lambda page, window: page.previous() or page.next(),
    lambda page, window: page.accept(),
    lambda page, window: page.skip(),
    lambda page, window: page.reject(),
    lambda page, window: window.close_title(),
    lambda page, window: window.close(),
], ids=['next', 'previous-next', 'accept', 'skip', 'reject', 'back', 'close-window'])
def test_an_edit_still_being_typed_is_saved_before_the_page_moves(qtbot, tmp_path, move):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _on_metadata(qtbot, window, 'r-arrival')
    _type(qtbot, page.metadata.noteField, 'never lost')                    # the focus is still in the box: no focus-out

    move(page, window)

    assert _meta(tmp_path, 'r-arrival')['note'] == 'never lost'


def test_saving_merges_only_what_was_touched_into_the_entry_as_it_is_now(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _on_metadata(qtbot, window, 'r-alien')
    _type(qtbot, page.metadata.noteField, 'mine')
    update_entry(_queue(tmp_path), 'r-alien', meta={**_meta(tmp_path), 'rating': 'PG-13', 'season_id': '5'})   # a run, meanwhile

    assert page.flush()

    meta = _meta(tmp_path)
    assert meta['note'] == 'mine' and meta['rating'] == 'PG-13' and meta['season_id'] == '5'
    assert page.metadata.ratingField.text() == 'PG-13'                       # and the form shows what is saved now


def test_a_blank_field_is_removed_so_a_default_applies_and_a_blank_title_is_flagged(qtbot, tmp_path):
    meta = {**complete_meta('r-alien'), 'note': 'old', 'language': 'French'}
    window = _window(qtbot, tmp_path, [('r-alien', {'meta': meta}), ('r-arrival', {})])
    page = _on_metadata(qtbot, window, 'r-alien', more=True)

    _clear(qtbot, page.metadata.noteField)
    _clear(qtbot, page.metadata.languageField)
    _clear(qtbot, page.metadata.titleField)
    assert page.flush()

    saved = _meta(tmp_path)
    assert 'note' not in saved and 'language' not in saved                   # unset, not ''
    assert 'title' not in saved and saved['year'] == '2001'                  # a blank title is unset too, and flagged
    assert page.badgeLabel.text() == 'Not ready to publish: title is required'


def test_audio_types_and_episodes_are_saved_as_lists(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _on_metadata(qtbot, window, 'r-alien')

    _type(qtbot, page.metadata.audioTypesField, 'Atmos ,  DTS:X,')
    _type(qtbot, page.metadata.seasonField, '3')
    _type(qtbot, page.metadata.episodesField, '1-3, 5, 5')
    assert page.flush()

    assert _meta(tmp_path)['audio_types'] == ['Atmos', 'DTS:X'] and _meta(tmp_path)['episodes'] == [1, 2, 3, 5]
    assert page.metadata.episodesField.text() == '1-3, 5' and page.metadata.audioTypesField.text() == 'Atmos, DTS:X'
    _clear(qtbot, page.metadata.episodesField)
    assert page.flush() and _meta(tmp_path)['episodes'] == []              # a blank box is a real answer here


def test_changing_the_season_drops_the_old_seasons_tmdb_ids(qtbot, tmp_path):
    meta = {**complete_meta('r-alien'), 'season': '1', 'season_id': '77', 'season_episode_count': 8}
    window = _window(qtbot, tmp_path, [('r-alien', {'meta': meta}), ('r-arrival', {})])
    page = _on_metadata(qtbot, window, 'r-alien')

    _type(qtbot, page.metadata.noteField, 'x')
    assert page.flush() and _meta(tmp_path)['season_id'] == '77'              # not touched: kept
    _type(qtbot, page.metadata.seasonField, '2')
    assert page.flush()

    saved = _meta(tmp_path)
    assert saved['season'] == '2' and 'season_id' not in saved and 'season_episode_count' not in saved


def test_an_unchanged_edit_writes_nothing(qtbot, tmp_path, monkeypatch):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _on_metadata(qtbot, window, 'r-alien')
    saved = []
    page.changed.connect(saved.append)
    _type(qtbot, page.metadata.titleField, 'r-alien')                       # the same text

    assert page.flush()

    assert saved == [] and not page.metadata.dirty


# --- invalid input and failures: shown, kept, never lost -----------------------------------------------------------------

def test_episodes_that_do_not_parse_are_not_written_and_are_shown(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _on_metadata(qtbot, window, 'r-alien')
    _type(qtbot, page.metadata.episodesField, '1-x')
    _type(qtbot, page.metadata.noteField, 'and this too')

    assert page.flush() is False

    assert 'episodes' not in _meta(tmp_path) and 'note' not in _meta(tmp_path)   # nothing of the edit was written
    assert page.metadata.metadataStatusLabel.text().startswith("Not saved: Episodes: '1-x' is not an episode number")
    assert page.metadata.dirty and page.metadata.noteField.text() == 'and this too'
    assert page.badgeLabel.text().startswith('Not ready to publish: Episodes:')


def test_an_edit_that_cannot_be_saved_stops_the_page_moving_or_deciding(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _on_metadata(qtbot, window, 'r-alien')
    page.confirm_discard = lambda reason: False          # Cancel: (what Skip and Reject ask; see test_worklist_metadata_fixes.py)
    _type(qtbot, page.metadata.episodesField, 'abc')

    assert page.next() is False and page.current_id == 'r-alien'
    assert page.accept() is False and page.skip() is False and page.reject() is False
    assert _status(tmp_path, 'r-alien') == 'pending'
    assert page.decisionLabel.text().startswith('Not moved: Not saved: Episodes:')
    assert page.rightTabs.currentIndex() == 1
    page.confirm_discard = lambda reason: True      # for the window closing at the end of the test


def test_leaving_with_an_edit_that_cannot_be_saved_asks_and_cancel_keeps_the_page_and_the_edit(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _on_metadata(qtbot, window, 'r-alien')
    reasons = []
    page.confirm_discard = lambda reason: reasons.append(reason) or False
    _type(qtbot, page.metadata.episodesField, 'abc')

    assert window.close_title() is False and window.title_page_open
    assert window.close() is False and window.isVisible()                   # closing the window is refused too
    assert len(reasons) == 2 and reasons[0].startswith('Not saved: Episodes:')
    assert page.metadata.episodesField.text() == 'abc' and page.metadata.dirty
    page.confirm_discard = lambda reason: True


def test_discarding_an_edit_that_cannot_be_saved_leaves_and_writes_nothing(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _on_metadata(qtbot, window, 'r-alien')
    page.confirm_discard = lambda reason: True
    before = _meta(tmp_path)
    _type(qtbot, page.metadata.episodesField, 'abc')

    assert window.close_title() is True and not window.title_page_open

    assert _meta(tmp_path) == before and not page.metadata.dirty


def test_the_default_question_offers_discard_and_cancel(qtbot, tmp_path, monkeypatch, real_ask_discard):
    from qtpy.QtWidgets import QMessageBox
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _on_metadata(qtbot, window, 'r-alien')
    page.confirm_discard = lambda reason: real_ask_discard(page, reason)      # the default (conftest.py answers it without a dialog)
    asked = []
    monkeypatch.setattr(QMessageBox, 'question', staticmethod(lambda *a: asked.append(a) or QMessageBox.StandardButton.Cancel))
    _type(qtbot, page.metadata.episodesField, 'abc')

    assert window.close_title() is False

    assert 'discard' in asked[0][2].lower() and asked[0][3] == (QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel)
    page.confirm_discard = lambda reason: True


def test_revert_throws_the_typed_edit_away_and_the_page_can_move_again(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _on_metadata(qtbot, window, 'r-alien')
    _type(qtbot, page.metadata.episodesField, 'abc')
    _type(qtbot, page.metadata.noteField, 'typed')

    _click(qtbot, page.metadata.revertButton)

    assert page.metadata.episodesField.text() == '' and page.metadata.noteField.text() == '' and not page.metadata.dirty
    assert page.next() and 'note' not in _meta(tmp_path)


def test_a_write_that_fails_is_shown_and_the_edit_is_kept(qtbot, tmp_path, monkeypatch):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _on_metadata(qtbot, window, 'r-alien')
    _type(qtbot, page.metadata.noteField, 'kept')
    monkeypatch.setattr('model.worklist_metadata.update_entry', lambda *a, **k: (_ for _ in ()).throw(OSError('disk full')))

    assert page.flush() is False

    assert page.metadata.metadataStatusLabel.text() == 'Not saved: OSError: disk full'
    assert page.metadata.dirty and page.metadata.noteField.text() == 'kept' and 'note' not in _meta(tmp_path)


def test_an_entry_that_vanished_is_reported_not_recreated(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _on_metadata(qtbot, window, 'r-alien')
    _type(qtbot, page.metadata.noteField, 'kept')
    os.remove(os.path.join(_queue(tmp_path), 'r-alien.json'))

    assert page.flush() is False

    assert 'no queue entry any more' in page.metadata.metadataStatusLabel.text()
    assert not os.path.exists(os.path.join(_queue(tmp_path), 'r-alien.json'))


def test_a_title_a_run_is_working_on_cannot_be_edited(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _on_metadata(qtbot, window, 'r-alien')
    _type(qtbot, page.metadata.noteField, 'typed just before')

    window.model.set_running({'r-alien': 'design'})

    assert page.metadata.noteField.isReadOnly() and not page.metadata.reloadTmdbButton.isEnabled()
    assert not page.metadata.browseArtButton.isEnabled()
    assert page.flush() is False and 'a run is working on this title' in page.metadata.metadataStatusLabel.text()
    assert 'note' not in _meta(tmp_path)
    window.model.set_running({})
    assert not page.metadata.noteField.isReadOnly() and page.flush() and _meta(tmp_path)['note'] == 'typed just before'


@pytest.mark.parametrize('status', ['pending', 'skipped', 'accepted', 'published', 'rejected'])
def test_the_metadata_of_a_title_can_be_edited_whatever_its_status(qtbot, tmp_path, status):
    window = _window(qtbot, tmp_path, [('r-alien', {'status': status}), ('r-arrival', {})])
    page = _on_metadata(qtbot, window, 'r-alien')
    assert not page.metadata.titleField.isReadOnly()

    _type(qtbot, page.metadata.noteField, 'edited')
    assert page.flush()

    entry = read_entry(_queue(tmp_path), 'r-alien')
    assert entry.meta['note'] == 'edited' and entry.status == status


def test_a_refresh_while_the_page_is_open_keeps_what_is_being_typed(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _on_metadata(qtbot, window, 'r-alien')
    _type(qtbot, page.metadata.noteField, 'half typed')
    update_entry(_queue(tmp_path), 'r-alien', meta={**_meta(tmp_path), 'rating': 'R'})

    window.refresh_from_index()

    assert page.metadata.noteField.text() == 'half typed' and page.metadata.dirty
    assert page.metadata.ratingField.text() == 'R'                          # what was not being edited is up to date


def test_the_header_and_state_line_follow_an_edit_though_the_rows_are_stale(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _on_metadata(qtbot, window, 'r-alien')
    assert page.titleLabel.text() == 'Alien (2001)'
    assert page.stateLabel.text() == 'Waiting for a decision. conf 0.90 - 2 candidates'

    _type(qtbot, page.metadata.titleField, 'Aliens')
    _type(qtbot, page.metadata.yearField, '1986')
    assert page.flush()

    assert page.titleLabel.text() == 'Aliens (1986)' and page.crumbLabel.text() == '› Aliens (1986)'
    assert page.stateLabel.text() == 'Waiting for a decision'                # the row's detail may be about old metadata


# --- the index learns of an edit ---------------------------------------------------------------------------------------------

def test_an_edit_makes_the_window_read_the_index_once_when_the_page_is_left(qtbot, tmp_path, monkeypatch):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    calls = []
    monkeypatch.setattr(LibraryIndex, 'refresh', lambda self, *a, **k: calls.append(a))
    page = _on_metadata(qtbot, window, 'r-alien')
    _type(qtbot, page.metadata.noteField, 'a')
    _focus(qtbot, page.metadata.warningField)
    _type(qtbot, page.metadata.warningField, 'b')
    assert page.flush()
    assert calls == [] and window._index_dirty

    with qtbot.waitSignal(window.index_synced, timeout=10000):
        window.close_title()

    assert len(calls) == 1 and not window._index_dirty


def test_leaving_after_looking_but_not_editing_reads_nothing(qtbot, tmp_path, monkeypatch):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    calls = []
    monkeypatch.setattr(LibraryIndex, 'refresh', lambda self, *a, **k: calls.append(a))
    page = _on_metadata(qtbot, window, 'r-alien')
    _type(qtbot, page.metadata.noteField, 'a')
    qtbot.keyClick(page.metadata.noteField, Qt.Key.Key_Backspace)          # ... and back to what it was
    assert page.flush()

    window.close_title()
    qtbot.wait(100)

    assert calls == [] and not window._index_dirty


# --- TMDB ---------------------------------------------------------------------------------------------------------------------

def test_reload_fills_what_tmdb_knows_and_the_result_is_saved_like_any_edit(qtbot, tmp_path, monkeypatch, no_modal):
    calls = []
    _tmdb(monkeypatch, _found(), calls=calls)
    window = _window(qtbot, tmp_path, [('r-alien', {'meta': {'title': 'Heat', 'year': '1995', 'audio_types': ['Atmos']}}),
                                       ('r-arrival', {})])
    page = _on_metadata(qtbot, window, 'r-alien', more=True)
    panel = page.metadata

    with qtbot.waitSignal(panel.tmdb_finished, timeout=10000) as done:
        assert panel.reload_tmdb()

    assert calls == [('lookup', ('Heat', '1995', window._preferences.get(TMDB_API_KEY)), {'kind': 'movie'})]
    assert done.args[0].startswith('Filled in from TMDB')
    assert (panel.movieDbIdField.text(), panel.runtimeField.text(), panel.ratingField.text()) == ('949', '170', 'R')
    assert panel.genresLabel.text() == 'Crime' and panel.audioTypesField.text() == 'Atmos'   # TMDB has none: not blanked
    assert panel.dirty and _meta(tmp_path)['title'] == 'Heat' and 'the_movie_db' not in _meta(tmp_path)   # not saved yet
    assert panel.reloadTmdbButton.isEnabled()

    assert page.flush()

    meta = _meta(tmp_path)
    assert meta['the_movie_db'] == '949' and meta['overview'] == 'cops and robbers' and meta['runtime'] == '170'
    assert meta['genres'] == [{'id': 80, 'name': 'Crime'}] and meta['collection'] == {'id': 1, 'name': 'A Collection'}
    assert meta['audio_types'] == ['Atmos'] and 'poster' not in meta


def test_reload_by_id_skips_the_search_and_a_series_is_looked_up_as_tv(qtbot, tmp_path, monkeypatch):
    calls = []
    _tmdb(monkeypatch, _found(title='Show'), calls=calls)
    window = _window(qtbot, tmp_path, [('r-alien', {'meta': {**complete_meta('r-alien'), 'season': '2'}}), ('r-arrival', {})])
    page = _on_metadata(qtbot, window, 'r-alien')

    _type(qtbot, page.metadata.movieDbIdField, '1399')
    with qtbot.waitSignal(page.metadata.tmdb_finished, timeout=10000):
        page.metadata.reload_tmdb()

    assert calls[0][0] == 'details' and calls[0][1][0] == '1399' and calls[0][2] == {'kind': 'tv'}


def test_a_tv_row_is_looked_up_as_tv_even_before_a_season_is_typed(qtbot, tmp_path, monkeypatch):
    calls = []
    _tmdb(monkeypatch, _found(), calls=calls)
    rows = _rows()
    rows[1]['kind'] = 'tv'                                                   # the index knows it is a series
    window = _window(qtbot, tmp_path, REVIEWABLE, rows=rows)
    page = _on_metadata(qtbot, window, 'r-alien')

    with qtbot.waitSignal(page.metadata.tmdb_finished, timeout=10000):
        page.metadata.reload_tmdb()

    assert calls[0][0] == 'lookup' and calls[0][2] == {'kind': 'tv'}


def test_reload_with_no_match_changes_nothing_and_says_so(qtbot, tmp_path, monkeypatch, no_modal):
    _tmdb(monkeypatch, BeqMetadata(title='Heat', year='1995'))                # what tmdb_lookup returns for no result
    window = _window(qtbot, tmp_path, [('r-alien', {'meta': {'title': 'Heat', 'year': '1995'}}), ('r-arrival', {})])
    page = _on_metadata(qtbot, window, 'r-alien')

    with qtbot.waitSignal(page.metadata.tmdb_finished, timeout=10000) as done:
        page.metadata.reload_tmdb()

    assert done.args[0] == 'TMDB has no match for "Heat" (1995). Nothing was changed.'
    assert not page.metadata.dirty and page.metadata.movieDbIdField.text() == ''


@pytest.mark.parametrize('error,fragment', [
    (requests.ConnectionError('no route to host'), 'TMDB lookup failed: no route to host'),
    (requests.HTTPError('401 Client Error: Unauthorized'), '401 Client Error'),
    (KeyError('title'), 'KeyError'),
])
def test_a_tmdb_failure_is_a_sentence_on_the_status_line_not_a_crash_or_a_modal(qtbot, tmp_path, monkeypatch, no_modal, error,
                                                                                  fragment):
    _tmdb(monkeypatch, error=error)
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _on_metadata(qtbot, window, 'r-alien')

    with qtbot.waitSignal(page.metadata.tmdb_finished, timeout=10000) as done:
        page.metadata.reload_tmdb()

    assert fragment in done.args[0] and fragment in page.metadata.metadataStatusLabel.text()
    assert not page.metadata.dirty and page.metadata.reloadTmdbButton.isEnabled()      # and it can be tried again


def test_reload_without_an_api_key_or_anything_to_look_up_says_so_and_asks_tmdb_nothing(qtbot, tmp_path, monkeypatch):
    calls = []
    _tmdb(monkeypatch, _found(), calls=calls)
    window = _window(qtbot, tmp_path, [('r-alien', {'meta': {'year': '1995'}}), ('r-arrival', {})])
    page = _on_metadata(qtbot, window, 'r-alien')

    assert page.metadata.reload_tmdb() is False                              # no title, no id
    assert page.metadata.metadataStatusLabel.text().startswith('Enter a title')
    _type(qtbot, page.metadata.titleField, 'Heat')
    window._preferences.set(TMDB_API_KEY, '')
    assert page.metadata.reload_tmdb() is False
    assert page.metadata.metadataStatusLabel.text() == 'No TMDB API key is set (Preferences).'
    assert calls == []


def test_a_lookup_that_comes_back_after_the_page_moved_on_is_dropped(qtbot, tmp_path, monkeypatch):
    gate = threading.Event()

    def slow(*args, **kwargs):
        gate.wait(10)
        return _found()

    monkeypatch.setattr('model.worklist_metadata.tmdb_lookup', slow)
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _on_metadata(qtbot, window, 'r-alien')
    try:
        assert page.metadata.reload_tmdb()
        assert not page.metadata.reloadTmdbButton.isEnabled()                # one at a time
        page.next()
        with qtbot.waitSignal(page.metadata.tmdb_finished, timeout=10000) as done:
            gate.set()
    finally:
        gate.set()

    assert done.args[0] == ''
    assert page.current_id == 'r-arrival' and page.metadata.titleField.text() == 'r-arrival' and not page.metadata.dirty
    assert page.metadata.reloadTmdbButton.isEnabled()


# --- artwork ------------------------------------------------------------------------------------------------------------------

def test_browsing_for_a_file_sets_the_artwork_and_marks_it_as_a_persons_choice(qtbot, tmp_path):
    poster = tmp_path / 'poster.png'
    poster.write_bytes(_png())
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _on_metadata(qtbot, window, 'r-alien')
    page.metadata.choose_file = lambda: str(poster)
    saved = []
    page.changed.connect(saved.append)

    _click(qtbot, page.metadata.browseArtButton)

    entry = read_entry(_queue(tmp_path), 'r-alien')
    assert entry.art_path == str(poster) and entry.art_overridden is True and saved == ['r-alien']
    assert page.metadata.artPathField.text() == str(poster) and page.metadata.artPreviewLabel.pixmap() is not None
    assert not page.metadata.artPreviewLabel.pixmap().isNull()
    assert page.metadata.metadataStatusLabel.text() == 'Artwork set.'


def test_a_file_that_is_not_an_image_is_refused_and_a_cancelled_dialog_does_nothing(qtbot, tmp_path):
    text = tmp_path / 'notes.txt'
    text.write_text('hello')
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _on_metadata(qtbot, window, 'r-alien')

    page.metadata.choose_file = lambda: str(text)
    assert page.metadata.browse_art() is False
    assert page.metadata.metadataStatusLabel.text() == 'Not used: that file is not a png or jpeg image.'
    page.metadata.choose_file = lambda: ''
    assert page.metadata.browse_art() is False

    entry = read_entry(_queue(tmp_path), 'r-alien')
    assert entry.art_path is None and entry.art_overridden is False


def test_clearing_the_artwork_forgets_the_file_and_marks_that_as_a_persons_choice_too(qtbot, tmp_path):
    poster = tmp_path / 'poster.png'
    poster.write_bytes(_png())
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _on_metadata(qtbot, window, 'r-alien')
    update_entry(_queue(tmp_path), 'r-alien', art_path=str(poster), art_overridden=False)
    page.reload()
    assert page.metadata.artPathField.text() == str(poster)

    _click(qtbot, page.metadata.clearArtButton)

    entry = read_entry(_queue(tmp_path), 'r-alien')
    assert entry.art_path is None and entry.art_overridden is False       # as the old dialog: cleared, and free to be found again
    assert page.metadata.artPathField.text() == '' and page.metadata.artPreviewLabel.text() == 'No artwork'
    assert page.metadata.metadataStatusLabel.text() == 'Artwork cleared.'


def test_a_chosen_file_that_has_gone_missing_is_said_so_in_the_preview(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    update_entry(_queue(tmp_path), 'r-alien', art_path=str(tmp_path / 'gone.png'), art_overridden=True)
    page = _on_metadata(qtbot, window, 'r-alien')

    assert page.metadata.artPreviewLabel.text() == 'The chosen file is missing'


def test_downloading_an_image_keeps_it_in_the_art_cache_and_uses_it(qtbot, tmp_path, monkeypatch, no_modal):
    monkeypatch.setattr(requests, 'get', lambda url, **kw: _Response(_image_bytes(8, 12, 'JPG')))
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _on_metadata(qtbot, window, 'r-alien')
    _type(qtbot, page.metadata.artUrlField, 'https://example.com/poster.jpg')

    with qtbot.waitSignal(page.metadata.art_finished, timeout=10000):
        _click(qtbot, page.metadata.downloadArtButton)

    entry = read_entry(_queue(tmp_path), 'r-alien')
    assert entry.art_path == os.path.join(_queue(tmp_path), '_art_cache', 'r-alien.jpg') and entry.art_overridden is True
    assert os.path.isfile(entry.art_path) and page.metadata.artUrlField.text() == ''
    assert not page.metadata.artPreviewLabel.pixmap().isNull() and page.metadata.downloadArtButton.isEnabled()


@pytest.mark.parametrize('response', [_Response(b'<html>a page</html>'), _Response(status=404)], ids=['not-image', 'http-error'])
def test_a_failed_download_is_a_sentence_and_changes_nothing(qtbot, tmp_path, monkeypatch, no_modal, response):
    monkeypatch.setattr(requests, 'get', lambda url, **kw: response)
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _on_metadata(qtbot, window, 'r-alien')
    _type(qtbot, page.metadata.artUrlField, 'https://example.com/poster.jpg')

    with qtbot.waitSignal(page.metadata.art_finished, timeout=10000) as done:
        page.metadata.download_art()

    assert done.args[0].startswith('Not used:') and page.metadata.artUrlField.text() != ''
    entry = read_entry(_queue(tmp_path), 'r-alien')
    assert entry.art_path is None and entry.art_overridden is False
    assert not os.path.exists(os.path.join(_queue(tmp_path), '_art_cache')) or not os.listdir(_queue(tmp_path) + '/_art_cache')


def test_a_download_that_finishes_after_the_page_moved_on_is_still_applied_to_its_own_title(qtbot, tmp_path, monkeypatch):
    gate = threading.Event()

    def slow(url, **kwargs):
        gate.wait(10)
        return _Response(_png())

    monkeypatch.setattr(requests, 'get', slow)
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _on_metadata(qtbot, window, 'r-alien')
    _type(qtbot, page.metadata.artUrlField, 'https://example.com/p.png')
    changed = []
    page.changed.connect(changed.append)
    try:
        assert page.metadata.download_art()
        page.next()
        with qtbot.waitSignal(page.metadata.art_finished, timeout=10000) as done:
            gate.set()
    finally:
        gate.set()

    assert done.args[0] == '' and changed == ['r-alien']
    assert read_entry(_queue(tmp_path), 'r-alien').art_path.endswith('r-alien.png')
    assert read_entry(_queue(tmp_path), 'r-arrival').art_path is None
    assert page.metadata.artPathField.text() == ''                          # this page is showing Arrival


def test_a_download_after_the_page_was_left_makes_the_window_read_the_index(qtbot, tmp_path, monkeypatch):
    gate = threading.Event()
    calls = []
    monkeypatch.setattr(LibraryIndex, 'refresh', lambda self, *a, **k: calls.append(a))
    monkeypatch.setattr(requests, 'get', lambda url, **kw: (gate.wait(10), _Response(_png()))[1])
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _on_metadata(qtbot, window, 'r-alien')
    _type(qtbot, page.metadata.artUrlField, 'https://example.com/p.png')
    try:
        assert page.metadata.download_art()
        assert window.close_title()
        assert calls == []
        with qtbot.waitSignal(window.index_synced, timeout=10000):
            gate.set()
    finally:
        gate.set()

    assert len(calls) == 1


# --- through a real scan ------------------------------------------------------------------------------------------------------

def _scanned(qtbot, tmp_path, *names, status='pending', meta=None):
    ''' A window over a real scan of titles whose queue entries are real (`test_pipeline_library_index`'s outputs). '''
    world = SimpleNamespace(work=str(tmp_path / 'work'), queue=str(tmp_path / 'queue'))
    world.items = [_item(name) for name in names]
    for item in world.items:
        _extracted(world, item)
        _real_entry(world, item, status=status, candidates=1, **({'meta': meta} if meta is not None else {}))
    window = WorkListWindow(None, _prefs(tmp_path), sources={'filesystem': FakeSource(world.items)}, auto_scan=False,
                            clock=time.time)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    window.activateWindow()
    qtbot.waitActive(window)
    with qtbot.waitSignal(window.scan_finished, timeout=30000):
        window.rescan()
    return window


def _row(window, title_id):
    return next(row for row in window.model.rows if row.id == title_id)


def test_blanking_the_year_of_an_accepted_title_makes_it_need_review_again_and_the_badge_agrees_with_the_index(qtbot, tmp_path):
    window = _scanned(qtbot, tmp_path, 'a', status='accepted')
    assert _row(window, 'fs-a').needs == 'publish'
    page = _on_metadata(qtbot, window, 'fs-a')
    assert page.badgeLabel.text() == 'Ready to publish'

    _clear(qtbot, page.metadata.yearField)
    assert page.flush()
    assert _row(window, 'fs-a').needs == 'publish'                            # the index has not read it yet
    with qtbot.waitSignal(window.index_synced, timeout=30000):
        _click(qtbot, page.backButton)

    row = _row(window, 'fs-a')
    assert row.needs == 'review' and row.detail == 'metadata incomplete: year is required'
    window.open_title('fs-a')
    assert page.badgeLabel.text() == 'Not ready to publish: year is required'    # the words the index used
    assert page.stateLabel.text() == 'Accepted. metadata incomplete: year is required'


def test_completing_the_metadata_of_an_accepted_title_sends_it_on_to_publish(qtbot, tmp_path):
    window = _scanned(qtbot, tmp_path, 'a', status='accepted', meta={'title': 'Film a', 'year': '2018'})
    row = _row(window, 'fs-a')
    assert row.needs == 'review' and row.detail == 'metadata incomplete: at least one audio type is required'
    page = _on_metadata(qtbot, window, 'fs-a')
    assert page.badgeLabel.text() == 'Not ready to publish: at least one audio type is required'

    _type(qtbot, page.metadata.audioTypesField, 'Atmos')
    with qtbot.waitSignal(window.index_synced, timeout=30000):
        window.close_title()

    assert _row(window, 'fs-a').needs == 'publish'


def test_a_pending_titles_incomplete_metadata_shows_in_the_row_detail_and_clears_after_an_edit(qtbot, tmp_path):
    window = _scanned(qtbot, tmp_path, 'a', 'b', meta={'title': 'Film'})
    assert 'metadata incomplete: year is required; at least one audio type is required' in _row(window, 'fs-a').detail
    page = _on_metadata(qtbot, window, 'fs-a')
    assert 'metadata incomplete' in page.stateLabel.text()
    _type(qtbot, page.metadata.yearField, '2018')
    _type(qtbot, page.metadata.audioTypesField, 'Atmos')
    assert page.flush()
    assert 'metadata incomplete' not in page.stateLabel.text()              # the row's detail is old: not repeated

    with qtbot.waitSignal(window.index_synced, timeout=30000):
        window.close_title()

    assert 'metadata incomplete' not in _row(window, 'fs-a').detail and _row(window, 'fs-b').needs == 'review'
    assert 'metadata incomplete' in _row(window, 'fs-b').detail


def _published(qtbot, tmp_path, repos):   # noqa: F811
    from test_pipeline_library_index import _ready
    xml, _, images, _ = repos
    _ready(repos)
    world = _published_world(tmp_path, 'a')
    prefs = _real_prefs(tmp_path, **{LIBRARY_PROFILE_PATH: _profile_file(tmp_path, xml.local_path, images.local_path)})
    window = _open_scanned(qtbot, prefs, world.items)
    window.activateWindow()                                                   # offscreen: keys reach a widget once its window is active
    qtbot.waitActive(window)
    assert _needs_of(window) == {'fs-a': 'publish'}
    _answer(True, [])
    with qtbot.waitSignal(window.run_finished, timeout=60000):
        _click(qtbot, window.publishButton)
    assert _row(window, 'fs-a').publish_state == 'written'
    window.activateWindow()                                                   # the confirmation box had the focus
    qtbot.waitActive(window)
    return window


def test_editing_the_metadata_of_a_published_title_makes_it_publish_out_of_date(qtbot, tmp_path, repos):
    window = _published(qtbot, tmp_path, repos)
    assert read_entry(_queue(tmp_path), 'fs-a').status == 'published'
    page = _on_metadata(qtbot, window, 'fs-a')
    assert page.badgeLabel.text() == 'Ready to publish'

    _type(qtbot, page.metadata.noteField, 'a typo fixed')
    with qtbot.waitSignal(window.index_synced, timeout=30000):
        _click(qtbot, page.backButton)

    row = _row(window, 'fs-a')
    assert (row.needs, row.publish_state, row.detail) == ('publish', 'out_of_date', 'changed since it was published')
    assert read_entry(_queue(tmp_path), 'fs-a').status == 'published'         # no second review needed


def test_changing_the_artwork_of_a_published_title_makes_it_publish_out_of_date_too(qtbot, tmp_path, repos):
    window = _published(qtbot, tmp_path, repos)
    poster = tmp_path / 'poster.png'
    poster.write_bytes(_png())
    page = _on_metadata(qtbot, window, 'fs-a')
    page.metadata.choose_file = lambda: str(poster)

    _click(qtbot, page.metadata.browseArtButton)
    with qtbot.waitSignal(window.index_synced, timeout=30000):
        window.close_title()

    assert (_row(window, 'fs-a').needs, _row(window, 'fs-a').publish_state) == ('publish', 'out_of_date')


def test_a_published_title_whose_metadata_is_emptied_needs_review_not_publish(qtbot, tmp_path, repos):
    window = _published(qtbot, tmp_path, repos)
    page = _on_metadata(qtbot, window, 'fs-a')

    _clear(qtbot, page.metadata.titleField)
    with qtbot.waitSignal(window.index_synced, timeout=30000):
        window.close_title()

    assert _row(window, 'fs-a').needs == 'review' and 'title is required' in _row(window, 'fs-a').detail


# --- the widgets ------------------------------------------------------------------------------------------------------------

def test_no_button_on_the_page_is_a_default_button_and_the_page_has_exactly_one_key_per_decision(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _open(qtbot, window, 'r-alien')

    assert all(not b.autoDefault() and not b.isDefault() for b in page.findChildren(QPushButton))
    keys = sorted(s.key().toString() for s in page.findChildren(QShortcut) if s.parent() is page)
    assert keys == sorted(['A', 'S', 'R', 'Esc', 'Alt+Left', 'Alt+Right', *'123456789'])


def test_the_metadata_panel_is_readable_in_a_dark_palette(qtbot, tmp_path):
    ''' No assertion about pixels: it builds and renders under a dark palette without an error (the look is checked by eye). '''
    from qtpy.QtGui import QColor, QPalette
    app = QApplication.instance()
    original = app.palette()
    dark = QPalette()
    dark.setColor(QPalette.ColorRole.Window, QColor('#2b2b2b'))
    dark.setColor(QPalette.ColorRole.WindowText, QColor('#dddddd'))
    app.setPalette(dark)
    try:
        window = _window(qtbot, tmp_path, [('r-alien', {'meta': INCOMPLETE}), ('r-arrival', {})])
        page = _on_metadata(qtbot, window, 'r-alien', more=True)
        assert not window.grab().isNull() and 'ff7b72' in page.badgeLabel.styleSheet()
    finally:
        app.setPalette(original)


def test_a_test_that_leaves_an_unsaveable_edit_behind_does_not_meet_a_modal_question_when_the_window_closes(qtbot, tmp_path):
    ''' conftest.py answers the default question (it once hung a teardown). A stray dialog is closed and recorded, never waited on. '''
    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import QMessageBox
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _open(qtbot, window, 'r-alien')                 # (the default `confirm_discard`: nothing set on it here)
    page.rightTabs.setCurrentIndex(1)
    _type(qtbot, page.metadata.episodesField, 'abc')       # cannot be saved, and is neither saved nor discarded
    stray = []

    def close_any_dialog():
        dialog = QApplication.activeModalWidget()
        if isinstance(dialog, QMessageBox):
            stray.append(dialog.text())
            dialog.reject()

    timer = QTimer()
    timer.setInterval(20)
    timer.timeout.connect(close_any_dialog)
    timer.start()
    try:
        window.close()
        qtbot.wait(100)
    finally:
        timer.stop()
    assert stray == [] and not window.isVisible()
