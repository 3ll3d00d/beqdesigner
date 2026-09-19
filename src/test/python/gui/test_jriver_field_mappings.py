'''
Choosing which library fields hold the IMDb/TMDb ids (plan §11.7): the editor, its storage, and how it reaches a
Library Sync source. No test talks to a server; Library/Fields is a fake.
'''
import ui.beq  # noqa: F401 -- see AGENTS.md's circular-import note

import threading

from qtpy.QtCore import QSettings

from model.jriver.connections import JRiverConnectionsWidget, SavedConnection, load_connections, save_connections, \
    set_field_mappings
from model.jriver.field_mappings import JRiverFieldMappingsWidget, overrides_from, split_names
from model.library_sources import JRiverSourcePage
from model.preferences import JRIVER_MCWS_FIELD_MAPPINGS, Preferences
from pipeline.library.jriver import DEFAULT_EXTERNAL_ID_FIELDS, LibraryField


def _prefs(tmp_path):
    return Preferences(QSettings(str(tmp_path / 'settings.ini'), QSettings.Format.IniFormat))


def _text(widget, kind, identifier):
    return widget.combos[(kind, identifier)].currentText()


def _commit(widget, kind, identifier, text):
    combo = widget.combos[(kind, identifier)]
    combo.setEditText(text)
    combo.lineEdit().editingFinished.emit()


FIELDS = [
    LibraryField('IMDb ID', 'IMDb ID', 'String'),
    LibraryField('TMDb ID', 'TMDb ID', 'String'),
    LibraryField('TheMovieDB Movie ID', 'TheMovieDB Movie ID', 'Integer'),
    LibraryField('IMDb Series ID', 'IMDb Series ID', 'String'),
    LibraryField('TheMovieDB Series ID', 'TheMovieDB Series ID', 'Integer'),
    LibraryField('My Film DB Id', 'My Film DB Id', 'Integer'),
    LibraryField('Date (year)', 'Year', 'Integer'),
    LibraryField('Filename', 'Filename', 'Path'),
    LibraryField('Date Imported', 'Date Imported', 'Date'),
    LibraryField('Notes', 'Notes', 'String'),
]


def _loaded(qtbot, widget, fields=FIELDS):
    widget.load_fields(lambda: fields)
    qtbot.waitUntil(lambda: not widget.loading)


# --- pure helpers ---------------------------------------------------------------------------------------------

def test_only_what_differs_from_the_defaults_is_an_override():
    effective = {kind: {i: tuple(names) for i, names in by.items()} for kind, by in DEFAULT_EXTERNAL_ID_FIELDS.items()}
    assert overrides_from(effective) == {}

    effective['movie']['imdb'] = ('My IMDb',)
    effective['tv']['tmdb'] = ()
    assert overrides_from(effective) == {'movie': {'imdb': ['My IMDb']}, 'tv': {'tmdb': []}}


def test_names_are_split_on_commas_and_trimmed():
    assert split_names(' A ,B,, C ') == ('A', 'B', 'C')
    assert split_names('') == ()
    assert split_names('Only One') == ('Only One',)


# --- the editor -----------------------------------------------------------------------------------------------

def test_nothing_is_editable_without_a_server(qtbot):
    widget = JRiverFieldMappingsWidget()
    qtbot.addWidget(widget)

    assert not any(c.isEnabled() for c in widget.combos.values())
    assert not widget.loadButton.isEnabled() and not widget.resetButton.isEnabled()


def test_it_shows_the_defaults_for_a_server_with_no_overrides(qtbot):
    widget = JRiverFieldMappingsWidget()
    qtbot.addWidget(widget)

    widget.set_connection(SavedConnection('a.local:1'))

    assert _text(widget, 'movie', 'imdb') == 'IMDb ID'
    assert _text(widget, 'movie', 'tmdb') == 'TheMovieDB Movie ID, TMDb ID'
    assert _text(widget, 'tv', 'imdb') == 'IMDb Series ID'
    assert _text(widget, 'tv', 'tmdb') == 'TheMovieDB Series ID'
    assert all(c.isEnabled() for c in widget.combos.values())


def test_it_shows_overrides_over_the_defaults_and_an_empty_list_as_off(qtbot):
    widget = JRiverFieldMappingsWidget()
    qtbot.addWidget(widget)

    widget.set_connection(SavedConnection('a.local:1', field_mappings={'movie': {'imdb': ['A', 'B'], 'tmdb': []}}))

    assert _text(widget, 'movie', 'imdb') == 'A, B'
    assert _text(widget, 'movie', 'tmdb') == ''
    assert _text(widget, 'tv', 'imdb') == 'IMDb Series ID'  # untouched


def test_committing_an_edit_announces_only_the_difference_from_the_defaults(qtbot):
    widget = JRiverFieldMappingsWidget()
    qtbot.addWidget(widget)
    widget.set_connection(SavedConnection('a.local:1'))

    with qtbot.waitSignal(widget.edited) as blocker:
        _commit(widget, 'movie', 'imdb', ' My IMDb , IMDb ID ')

    assert blocker.args == ['a.local:1', {'movie': {'imdb': ['My IMDb', 'IMDb ID']}}]


def test_clearing_a_box_switches_that_id_off_and_typing_the_default_back_removes_the_override(qtbot):
    widget = JRiverFieldMappingsWidget()
    qtbot.addWidget(widget)
    widget.set_connection(SavedConnection('a.local:1'))

    with qtbot.waitSignal(widget.edited) as off:
        _commit(widget, 'tv', 'tmdb', '')
    assert off.args[1] == {'tv': {'tmdb': []}}

    with qtbot.waitSignal(widget.edited) as back:
        _commit(widget, 'tv', 'tmdb', 'TheMovieDB Series ID')
    assert back.args[1] == {}


def test_committing_without_a_change_announces_nothing(qtbot):
    widget = JRiverFieldMappingsWidget()
    qtbot.addWidget(widget)
    widget.set_connection(SavedConnection('a.local:1'))
    announced = []
    widget.edited.connect(lambda *args: announced.append(args))

    _commit(widget, 'movie', 'imdb', 'IMDb ID')

    assert announced == []


def test_defaults_button_restores_them_and_announces_the_clear(qtbot):
    widget = JRiverFieldMappingsWidget()
    qtbot.addWidget(widget)
    widget.set_connection(SavedConnection('a.local:1', field_mappings={'movie': {'imdb': ['Custom']}}))

    with qtbot.waitSignal(widget.edited) as blocker:
        widget.resetButton.click()

    assert blocker.args == ['a.local:1', {}]
    assert _text(widget, 'movie', 'imdb') == 'IMDb ID'


def test_loaded_fields_are_offered_but_only_ones_an_id_could_live_in(qtbot):
    widget = JRiverFieldMappingsWidget()
    qtbot.addWidget(widget)
    widget.set_connection(SavedConnection('a.local:1'))

    _loaded(qtbot, widget)

    combo = widget.combos[('movie', 'imdb')]
    offered = [combo.itemText(i) for i in range(combo.count())]
    assert offered == ['Date (year)', 'IMDb ID', 'IMDb Series ID', 'My Film DB Id', 'Notes', 'TheMovieDB Movie ID',
                       'TheMovieDB Series ID', 'TMDb ID']  # no Path/Date types, sorted case-insensitively
    assert 'Filename' not in offered and 'Date Imported' not in offered
    assert combo.currentText() == 'IMDb ID'  # what was typed is kept
    assert widget.statusLabel.text() == '8 candidate fields loaded'


def test_a_configured_name_the_server_does_not_have_is_flagged(qtbot):
    widget = JRiverFieldMappingsWidget()
    qtbot.addWidget(widget)
    widget.set_connection(SavedConnection('a.local:1'))
    _loaded(qtbot, widget)

    _commit(widget, 'movie', 'imdb', 'IMDB, IMDb ID')

    assert widget.statusLabel.text() == 'Not a field on this server: IMDB'
    _commit(widget, 'movie', 'imdb', 'IMDb ID')
    assert 'Not a field' not in widget.statusLabel.text()


def test_loading_fields_is_in_the_background_with_progress_and_disables_editing(qtbot):
    widget = JRiverFieldMappingsWidget()
    qtbot.addWidget(widget)
    widget.set_connection(SavedConnection('a.local:1'))
    release = threading.Event()

    def slow():
        assert release.wait(5)
        return FIELDS

    widget.load_fields(slow)

    assert widget.loading
    assert widget.statusLabel.text() == 'Loading fields...'
    assert not widget.loadButton.isEnabled() and not any(c.isEnabled() for c in widget.combos.values())
    widget.load_fields(slow)  # ignored while busy
    release.set()
    qtbot.waitUntil(lambda: not widget.loading)
    assert widget.loadButton.isEnabled() and all(c.isEnabled() for c in widget.combos.values())


def test_a_failed_load_is_reported_and_leaves_the_boxes_usable(qtbot):
    widget = JRiverFieldMappingsWidget()
    qtbot.addWidget(widget)
    widget.set_connection(SavedConnection('a.local:1'))

    def refuse():
        raise ConnectionError('refused')

    widget.load_fields(refuse)
    qtbot.waitUntil(lambda: not widget.loading)

    assert widget.statusLabel.text() == 'Could not load fields: ConnectionError: refused'
    assert all(c.isEnabled() for c in widget.combos.values())


def test_switching_server_drops_the_previous_servers_field_list(qtbot):
    widget = JRiverFieldMappingsWidget()
    qtbot.addWidget(widget)
    widget.set_connection(SavedConnection('a.local:1'))
    _loaded(qtbot, widget)

    widget.set_connection(SavedConnection('b.local:2'))

    assert widget.combos[('movie', 'imdb')].count() == 0
    _commit(widget, 'movie', 'imdb', 'Anything')
    assert widget.statusLabel.text() == ''  # nothing known about this server, so nothing to flag


# --- storage and the Preferences page ----------------------------------------------------------------------

def test_overrides_are_stored_per_server_and_dropped_with_it(tmp_path):
    prefs = _prefs(tmp_path)
    save_connections(prefs, [SavedConnection('a.local:1', field_mappings={'movie': {'imdb': ['X']}}),
                             SavedConnection('b.local:2')])

    assert {c.endpoint: c.field_mappings for c in load_connections(prefs)} == {
        'a.local:1': {'movie': {'imdb': ['X']}}, 'b.local:2': {}}
    set_field_mappings(prefs, 'b.local:2', {'tv': {'tmdb': []}})
    assert prefs.get(JRIVER_MCWS_FIELD_MAPPINGS)['b.local:2'] == {'tv': {'tmdb': []}}
    set_field_mappings(prefs, 'a.local:1', {})
    save_connections(prefs, [c for c in load_connections(prefs) if c.endpoint == 'b.local:2'])
    assert list(prefs.get(JRIVER_MCWS_FIELD_MAPPINGS)) == ['b.local:2']


def _widget(qtbot, tmp_path, connections):
    prefs = _prefs(tmp_path)
    save_connections(prefs, connections)
    widget = JRiverConnectionsWidget(prefs)
    qtbot.addWidget(widget)
    return widget, prefs


def _select(widget, endpoint):
    for row in range(widget.savedConnections.count()):
        if widget.savedConnections.item(row).data(widget.CONNECTION_ROLE).endpoint == endpoint:
            widget.savedConnections.setCurrentRow(row)
            return


def test_the_fields_editor_follows_the_selected_server(qtbot, tmp_path):
    widget, _ = _widget(qtbot, tmp_path, [SavedConnection('a.local:1', field_mappings={'movie': {'imdb': ['Mine']}}),
                                          SavedConnection('b.local:2')])
    assert not widget.fieldsGroup.isEnabled()

    _select(widget, 'a.local:1')
    assert widget.fieldsGroup.isEnabled()
    assert _text(widget.fieldMappings, 'movie', 'imdb') == 'Mine'

    _select(widget, 'b.local:2')
    assert _text(widget.fieldMappings, 'movie', 'imdb') == 'IMDb ID'

    widget.newButton.click()
    assert not widget.fieldsGroup.isEnabled()


def test_an_edit_is_saved_at_once_and_survives_reselecting(qtbot, tmp_path):
    widget, prefs = _widget(qtbot, tmp_path, [SavedConnection('a.local:1'), SavedConnection('b.local:2')])
    _select(widget, 'a.local:1')

    with qtbot.waitSignal(widget.changed):
        _commit(widget.fieldMappings, 'tv', 'imdb', 'My Series IMDb')

    assert {c.endpoint: c.field_mappings for c in load_connections(prefs)}['a.local:1'] == \
        {'tv': {'imdb': ['My Series IMDb']}}
    _select(widget, 'b.local:2')
    _select(widget, 'a.local:1')
    assert _text(widget.fieldMappings, 'tv', 'imdb') == 'My Series IMDb'


def test_updating_a_server_keeps_its_field_overrides(qtbot, tmp_path, monkeypatch):
    from unittest.mock import MagicMock
    widget, prefs = _widget(qtbot, tmp_path, [SavedConnection('a.local:1', field_mappings={'movie': {'imdb': ['X']}})])
    monkeypatch.setattr('model.jriver.mcws.MediaServer.authenticate', MagicMock(return_value=True))
    _select(widget, 'a.local:1')

    widget.httpsCheck.setChecked(True)
    widget.testButton.click()
    qtbot.waitUntil(lambda: not widget.testing)
    widget.addButton.click()

    saved = load_connections(prefs)
    assert [(c.secure, c.field_mappings) for c in saved] == [(True, {'movie': {'imdb': ['X']}})]


# --- Library Sync -------------------------------------------------------------------------------------------

def test_the_source_is_built_with_the_servers_field_overrides(qtbot, tmp_path):
    prefs = _prefs(tmp_path)
    save_connections(prefs, [SavedConnection('media.local:52199',
                                             field_mappings={'movie': {'imdb': ['My IMDb']}})])
    page = JRiverSourcePage()
    qtbot.addWidget(page)
    page.load(prefs)

    source = page.build_source()

    assert source.external_id_fields['movie']['imdb'] == ('My IMDb',)
    assert 'My IMDb' in source.requested_fields
    assert source.external_id_fields['tv'] == {'imdb': ('IMDb Series ID',), 'tmdb': ('TheMovieDB Series ID',)}


def test_a_server_with_no_overrides_gets_the_defaults(qtbot, tmp_path):
    prefs = _prefs(tmp_path)
    save_connections(prefs, [SavedConnection('media.local:52199')])
    page = JRiverSourcePage()
    qtbot.addWidget(page)
    page.load(prefs)

    assert page.build_source().external_id_fields['movie']['imdb'] == ('IMDb ID',)
