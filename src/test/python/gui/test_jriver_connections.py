'''
The shared JRiver server list (model/jriver/connections.py): parsing of what QSettings hands back, the
Preferences -> JRiver management widget, and the filter manager's zone dialog picking from the same list.
'''
import ui.beq  # noqa: F401 -- must precede model.jriver.ui, see AGENTS.md's circular-import note

import threading
from unittest.mock import MagicMock

from qtpy.QtCore import QSettings

from model.jriver.connections import JRiverConnectionsWidget, SavedConnection, load_connections, parse_connections, \
    remember_alias, save_connections, set_path_mappings
from model.jriver.mcws import MCWSError
from model.preferences import JRIVER_MCWS_ALIASES, JRIVER_MCWS_CONNECTIONS, JRIVER_MCWS_PATH_MAPPINGS, Preferences
from pipeline.library.pathmap import PathMapping


def _prefs(tmp_path):
    return Preferences(QSettings(str(tmp_path / 'settings.ini'), QSettings.Format.IniFormat))


def test_parse_accepts_every_stored_shape():
    connections = parse_connections({
        '10.0.0.1:52199': (None, False),
        'media.local:52199': (('user', 'pw'), True),
        'legacy:52199': ('old', 'pw2', True),
        'listy:52199': [['a', 'b'], False],  # QSettings can round-trip a tuple as a list
    })

    assert connections == [
        SavedConnection('10.0.0.1:52199'),
        SavedConnection('media.local:52199', 'user', 'pw', True),
        SavedConnection('legacy:52199', 'old', 'pw2', True),
        SavedConnection('listy:52199', 'a', 'b', False),
    ]
    assert parse_connections(None) == []


def test_connection_properties_and_round_trip():
    connection = SavedConnection('media.local:52199', 'user', 'pw', True)

    assert (connection.host, connection.port) == ('media.local', 52199)
    assert connection.label == 'media.local:52199 [user]'
    assert SavedConnection('10.0.0.1:1').label == '10.0.0.1:1 [Unauthenticated]'
    assert parse_connections(connection.to_preference()) == [connection]


def test_widget_lists_saved_servers_and_only_adds_after_a_passing_test(qtbot, tmp_path, monkeypatch):
    prefs = _prefs(tmp_path)
    prefs.set(JRIVER_MCWS_CONNECTIONS, {'old.local:52199': (None, False)})
    widget = JRiverConnectionsWidget(prefs)
    qtbot.addWidget(widget)
    assert [c.endpoint for c in widget.connections()] == ['old.local:52199']
    assert not widget.testButton.isEnabled()

    widget.endpointEdit.setText('media.local:52199')
    widget.authCheck.setChecked(True)
    assert not widget.testButton.isEnabled()  # credentials missing
    widget.usernameEdit.setText('user')
    widget.passwordEdit.setText('pw')
    assert widget.testButton.isEnabled()
    assert not widget.addButton.isEnabled()

    authenticate = MagicMock(return_value=True)
    monkeypatch.setattr('model.jriver.mcws.MediaServer.authenticate', authenticate)
    widget.testButton.click()
    qtbot.waitUntil(lambda: not widget.testing)
    authenticate.assert_called_once()
    assert widget.addButton.isEnabled()

    with qtbot.waitSignal(widget.changed):
        widget.addButton.click()

    saved = {c.endpoint: c for c in load_connections(prefs)}  # QSettings does not preserve dict order
    assert set(saved) == {'old.local:52199', 'media.local:52199'}
    assert saved['media.local:52199'] == SavedConnection('media.local:52199', 'user', 'pw', False)
    assert widget.endpointEdit.text() == ''


def test_widget_reports_a_failed_test_and_does_not_allow_adding(qtbot, tmp_path, monkeypatch):
    widget = JRiverConnectionsWidget(_prefs(tmp_path))
    qtbot.addWidget(widget)
    widget.endpointEdit.setText('media.local:52199')

    def refuse(self):
        raise MCWSError('Connection failure', 'http://media.local:52199/MCWS/v1/Authenticate', 0, 'refused')

    monkeypatch.setattr('model.jriver.mcws.MediaServer.authenticate', refuse)
    widget.testButton.click()
    qtbot.waitUntil(lambda: not widget.testing)

    assert 'Connection failure' in widget.resultText.toPlainText()
    assert not widget.addButton.isEnabled()

    widget.endpointEdit.setText('media.local:52198')  # any edit also invalidates a previous pass
    assert not widget.addButton.isEnabled()


def test_widget_accepts_a_hostname_endpoint_and_rejects_a_bare_host(qtbot, tmp_path):
    widget = JRiverConnectionsWidget(_prefs(tmp_path))
    qtbot.addWidget(widget)

    widget.endpointEdit.setText('media.local')
    assert not widget.testButton.isEnabled()
    widget.endpointEdit.setText('media.local:52199')
    assert widget.testButton.isEnabled()


def test_widget_deletes_the_selected_server(qtbot, tmp_path):
    prefs = _prefs(tmp_path)
    prefs.set(JRIVER_MCWS_CONNECTIONS, {'a.local:1': (None, False), 'b.local:2': (('u', 'p'), False)})
    widget = JRiverConnectionsWidget(prefs)
    qtbot.addWidget(widget)

    widget.savedConnections.setCurrentRow(0)
    with qtbot.waitSignal(widget.changed):
        widget.deleteButton.click()

    assert [c.endpoint for c in load_connections(prefs)] == ['b.local:2']
    assert [c.endpoint for c in widget.connections()] == ['b.local:2']


def _widget_with(qtbot, tmp_path, connections, monkeypatch=None):
    prefs = _prefs(tmp_path)
    prefs.set(JRIVER_MCWS_CONNECTIONS, connections)
    widget = JRiverConnectionsWidget(prefs)
    qtbot.addWidget(widget)
    if monkeypatch is not None:
        monkeypatch.setattr('model.jriver.mcws.MediaServer.authenticate', MagicMock(return_value=True))
    return widget, prefs


def _select(widget, endpoint):
    for row in range(widget.savedConnections.count()):
        if widget.savedConnections.item(row).data(widget.CONNECTION_ROLE).endpoint == endpoint:
            widget.savedConnections.setCurrentRow(row)
            return
    raise AssertionError(endpoint)


def _test_and_wait(qtbot, widget):
    widget.testButton.click()
    qtbot.waitUntil(lambda: not widget.testing)


def test_selecting_a_saved_server_loads_it_into_the_form(qtbot, tmp_path):
    widget, _ = _widget_with(qtbot, tmp_path, {'a.local:1': (None, False), 'b.local:2': (('u', 'p'), True)})

    _select(widget, 'b.local:2')

    assert widget.endpointEdit.text() == 'b.local:2'
    assert widget.httpsCheck.isChecked()
    assert widget.authCheck.isChecked()
    assert (widget.usernameEdit.text(), widget.passwordEdit.text()) == ('u', 'p')
    assert widget.formLabel.text() == 'Edit server'
    assert widget.addButton.text() == 'Update'
    assert widget.usernameEdit.isEnabled()

    _select(widget, 'a.local:1')

    assert not widget.httpsCheck.isChecked()
    assert not widget.authCheck.isChecked()
    assert widget.usernameEdit.text() == '' and not widget.usernameEdit.isEnabled()


def test_a_selected_server_must_be_retested_after_a_change_before_it_can_be_updated(qtbot, tmp_path, monkeypatch):
    widget, prefs = _widget_with(qtbot, tmp_path, {'a.local:1': (('u', 'p'), False)}, monkeypatch)
    _select(widget, 'a.local:1')
    assert not widget.addButton.isEnabled()  # loading is not an edit, and nothing has been tested yet
    assert widget.testButton.isEnabled()

    widget.passwordEdit.setText('new-password')
    assert not widget.addButton.isEnabled()
    _test_and_wait(qtbot, widget)
    assert widget.addButton.isEnabled()
    with qtbot.waitSignal(widget.changed):
        widget.addButton.click()

    saved = load_connections(prefs)
    assert saved == [SavedConnection('a.local:1', 'u', 'new-password', False)]
    assert widget.savedConnections.count() == 1
    assert widget.endpointEdit.text() == 'a.local:1'  # still on the server just edited
    assert widget.passwordEdit.text() == 'new-password'
    assert widget.addButton.text() == 'Update'


def test_editing_the_endpoint_of_a_selected_server_renames_it_rather_than_adding_a_second(qtbot, tmp_path,
                                                                                            monkeypatch):
    widget, prefs = _widget_with(qtbot, tmp_path, {'a.local:1': (None, False), 'keep.local:9': (None, False)},
                                 monkeypatch)
    _select(widget, 'a.local:1')

    widget.endpointEdit.setText('renamed.local:2')
    _test_and_wait(qtbot, widget)
    widget.addButton.click()

    assert {c.endpoint for c in load_connections(prefs)} == {'renamed.local:2', 'keep.local:9'}


def test_a_failed_test_of_an_edit_leaves_the_saved_server_alone(qtbot, tmp_path, monkeypatch):
    widget, prefs = _widget_with(qtbot, tmp_path, {'a.local:1': (None, False)})
    monkeypatch.setattr('model.jriver.mcws.MediaServer.authenticate',
                        MagicMock(side_effect=MCWSError('Connection failure', 'http://a', 0, 'refused')))
    _select(widget, 'a.local:1')
    widget.endpointEdit.setText('a.local:2')

    _test_and_wait(qtbot, widget)

    assert not widget.addButton.isEnabled()
    assert 'Connection failure' in widget.resultText.toPlainText()
    assert load_connections(prefs) == [SavedConnection('a.local:1')]


def test_new_clears_the_selection_and_the_form_for_adding_another(qtbot, tmp_path):
    widget, _ = _widget_with(qtbot, tmp_path, {'a.local:1': (('u', 'p'), True)})
    _select(widget, 'a.local:1')

    widget.newButton.click()

    assert widget.savedConnections.selectedItems() == []
    assert widget.endpointEdit.text() == ''
    assert not widget.authCheck.isChecked() and not widget.httpsCheck.isChecked()
    assert widget.formLabel.text() == 'New server'
    assert widget.addButton.text() == 'Add'
    assert not widget.deleteButton.isEnabled()


def test_deleting_the_selected_server_also_clears_the_form(qtbot, tmp_path):
    widget, prefs = _widget_with(qtbot, tmp_path, {'a.local:1': (None, False)})
    _select(widget, 'a.local:1')

    widget.deleteButton.click()

    assert load_connections(prefs) == []
    assert widget.endpointEdit.text() == ''
    assert widget.formLabel.text() == 'New server'


def test_testing_runs_in_the_background_and_shows_progress_until_it_finishes(qtbot, tmp_path, monkeypatch):
    widget, _ = _widget_with(qtbot, tmp_path, {'a.local:1': (None, False), 'b.local:2': (None, False)})
    release = threading.Event()
    started = threading.Event()

    def slow(self):
        started.set()
        assert release.wait(5)
        return True

    monkeypatch.setattr('model.jriver.mcws.MediaServer.authenticate', slow)
    _select(widget, 'a.local:1')

    widget.testButton.click()

    # click() has returned while authenticate() is still blocked: the UI thread is free
    assert started.wait(5)
    assert widget.testing
    assert widget.statusLabel.text() == 'Testing...'
    for control in (widget.testButton, widget.addButton, widget.deleteButton, widget.newButton,
                    widget.savedConnections, widget.endpointEdit, widget.authCheck, widget.httpsCheck):
        assert not control.isEnabled()
    widget.testButton.click()  # a second click while busy is ignored
    release.set()
    qtbot.waitUntil(lambda: not widget.testing)

    assert widget.statusLabel.text() == ''
    assert widget.endpointEdit.isEnabled() and widget.savedConnections.isEnabled() and widget.newButton.isEnabled()
    assert widget.addButton.isEnabled()


def test_an_unexpected_error_while_testing_is_shown_not_swallowed(qtbot, tmp_path, monkeypatch):
    widget, _ = _widget_with(qtbot, tmp_path, {'a.local:1': (None, False)})
    monkeypatch.setattr('model.jriver.mcws.MediaServer.authenticate', MagicMock(side_effect=ValueError('boom')))
    _select(widget, 'a.local:1')

    _test_and_wait(qtbot, widget)

    assert 'ValueError: boom' in widget.resultText.toPlainText()
    assert not widget.addButton.isEnabled()
    assert not widget.testing


def test_filter_manager_dialog_picks_from_the_shared_list(qtbot, tmp_path, monkeypatch):
    from model.jriver.ui import MCWSDialog
    prefs = _prefs(tmp_path)
    prefs.set(JRIVER_MCWS_CONNECTIONS, {'a.local:1': (None, False), 'b.local:2': (('u', 'p'), True)})
    monkeypatch.setattr('model.jriver.mcws.MediaServer.get_zones', lambda self: {'Main': 0})

    dialog = MCWSDialog(None, prefs)
    qtbot.addWidget(dialog)

    assert [dialog.savedConnections.item(i).text() for i in range(dialog.savedConnections.count())] \
        == ['a.local:1 [Unauthenticated]', 'b.local:2 [u]']
    assert not dialog.upload.isEnabled()
    dialog.savedConnections.setCurrentRow(1)
    qtbot.waitUntil(lambda: dialog.zones.count() == 1)
    assert [dialog.zones.item(i).text() for i in range(dialog.zones.count())] == ['Main']
    assert dialog.upload.isEnabled()
    assert not hasattr(dialog, 'addNewButton')  # management lives in Preferences now


def test_preferences_dialog_has_a_jriver_page_managing_the_same_list(qtbot, tmp_path, monkeypatch):
    from model.preferences import PreferencesDialog
    prefs = _prefs(tmp_path)
    prefs.set(JRIVER_MCWS_CONNECTIONS, {'a.local:1': (None, False)})
    monkeypatch.setattr('model.jriver.mcws.MediaServer.authenticate', MagicMock(return_value=True))  # alias refresh

    dialog = PreferencesDialog(prefs, str(tmp_path), MagicMock())
    qtbot.addWidget(dialog)

    widget = dialog.jriverPage.findChild(JRiverConnectionsWidget)
    assert widget is not None
    assert [c.endpoint for c in widget.connections()] == ['a.local:1']


# --- aliases (the FriendlyName /Alive reports) ---------------------------------------------------------------

def _friendly(monkeypatch, names):
    ''' Makes authenticate() succeed and friendly_name answer per server (a missing endpoint means unreachable). '''
    current = {}

    def authenticate(self):
        endpoint = self._MediaServer__ip
        if endpoint not in names:
            raise MCWSError('Connection failure', f'http://{endpoint}', 0, 'refused')
        current[id(self)] = names[endpoint]
        return True

    monkeypatch.setattr('model.jriver.mcws.MediaServer.authenticate', authenticate)
    monkeypatch.setattr('model.jriver.mcws.MediaServer.friendly_name', property(lambda self: current.get(id(self))))


def test_a_named_server_is_shown_by_its_name_with_the_address():
    named = SavedConnection('10.0.0.1:52199', 'user', 'pw', False, alias='Living Room')

    assert named.display_name == 'Living Room (10.0.0.1:52199)'
    assert named.label == 'Living Room (10.0.0.1:52199) [user]'
    assert SavedConnection('10.0.0.1:52199', alias='Den').label == 'Den (10.0.0.1:52199) [Unauthenticated]'
    assert SavedConnection('10.0.0.1:52199').display_name == '10.0.0.1:52199'  # unnamed: unchanged


def test_aliases_are_stored_apart_from_the_connections_and_dropped_with_them(tmp_path):
    prefs = _prefs(tmp_path)
    keep = SavedConnection('a.local:1', 'u', 'p', True, alias='Keep')
    drop = SavedConnection('b.local:2', alias='Drop')

    save_connections(prefs, [keep, drop])

    assert prefs.get(JRIVER_MCWS_CONNECTIONS) == {'a.local:1': (('u', 'p'), True), 'b.local:2': (None, False)}
    assert prefs.get(JRIVER_MCWS_ALIASES) == {'a.local:1': 'Keep', 'b.local:2': 'Drop'}
    assert {c.endpoint: c.alias for c in load_connections(prefs)} == {'a.local:1': 'Keep', 'b.local:2': 'Drop'}

    save_connections(prefs, [keep])
    assert prefs.get(JRIVER_MCWS_ALIASES) == {'a.local:1': 'Keep'}


def test_connections_saved_before_aliases_existed_load_without_one(tmp_path):
    prefs = _prefs(tmp_path)
    prefs.set(JRIVER_MCWS_CONNECTIONS, {'a.local:1': (None, False)})

    assert load_connections(prefs) == [SavedConnection('a.local:1')]


def test_remember_alias_only_records_a_change_for_a_saved_server(tmp_path):
    prefs = _prefs(tmp_path)
    prefs.set(JRIVER_MCWS_CONNECTIONS, {'a.local:1': (None, False)})

    assert remember_alias(prefs, 'a.local:1', 'Den') is True
    assert remember_alias(prefs, 'a.local:1', 'Den') is False  # unchanged
    assert remember_alias(prefs, 'a.local:1', 'Renamed') is True
    assert remember_alias(prefs, 'a.local:1', None) is False
    assert remember_alias(prefs, 'a.local:1', '') is False
    assert remember_alias(prefs, 'gone.local:9', 'Ghost') is False  # not saved, so nothing to name
    assert prefs.get(JRIVER_MCWS_ALIASES) == {'a.local:1': 'Renamed'}


def test_a_passing_test_reports_and_saves_the_servers_own_name(qtbot, tmp_path, monkeypatch):
    widget, prefs = _widget_with(qtbot, tmp_path, {})
    _friendly(monkeypatch, {'media.local:52199': 'Cinema PC'})
    widget.endpointEdit.setText('media.local:52199')

    _test_and_wait(qtbot, widget)

    assert widget.statusLabel.text() == 'Connected to Cinema PC'
    widget.addButton.click()
    assert load_connections(prefs) == [SavedConnection('media.local:52199', alias='Cinema PC')]
    assert widget.savedConnections.item(0).text() == 'Cinema PC (media.local:52199) [Unauthenticated]'


def test_retesting_a_server_that_reports_no_name_keeps_the_name_it_had(qtbot, tmp_path, monkeypatch):
    prefs = _prefs(tmp_path)
    prefs.set(JRIVER_MCWS_CONNECTIONS, {'a.local:1': (None, False)})
    prefs.set(JRIVER_MCWS_ALIASES, {'a.local:1': 'Den'})
    widget = JRiverConnectionsWidget(prefs)
    qtbot.addWidget(widget)
    _friendly(monkeypatch, {'a.local:1': None})  # reachable, but reports no FriendlyName
    _select(widget, 'a.local:1')
    widget.httpsCheck.setChecked(True)

    _test_and_wait(qtbot, widget)
    widget.addButton.click()

    assert load_connections(prefs) == [SavedConnection('a.local:1', secure=True, alias='Den')]


def test_refresh_fills_in_names_for_unnamed_servers_only_and_skips_unreachable_ones(qtbot, tmp_path, monkeypatch):
    prefs = _prefs(tmp_path)
    prefs.set(JRIVER_MCWS_CONNECTIONS, {'old.local:1': (None, False), 'named.local:2': (None, False),
                                        'dead.local:3': (None, False)})
    prefs.set(JRIVER_MCWS_ALIASES, {'named.local:2': 'Already'})
    widget = JRiverConnectionsWidget(prefs)
    qtbot.addWidget(widget)
    _friendly(monkeypatch, {'old.local:1': 'Basement', 'named.local:2': 'Should not be asked'})

    with qtbot.waitSignal(widget.changed, timeout=5000):
        widget.refresh_aliases()

    names = {c.endpoint: c.alias for c in load_connections(prefs)}
    assert names == {'old.local:1': 'Basement', 'named.local:2': 'Already', 'dead.local:3': None}
    labels = {widget.savedConnections.item(i).data(widget.CONNECTION_ROLE).endpoint:
              widget.savedConnections.item(i).text() for i in range(widget.savedConnections.count())}
    assert labels['old.local:1'] == 'Basement (old.local:1) [Unauthenticated]'
    assert labels['named.local:2'] == 'Already (named.local:2) [Unauthenticated]'
    assert labels['dead.local:3'] == 'dead.local:3 [Unauthenticated]'


def test_refresh_does_nothing_when_every_server_is_already_named(qtbot, tmp_path, monkeypatch):
    prefs = _prefs(tmp_path)
    prefs.set(JRIVER_MCWS_CONNECTIONS, {'a.local:1': (None, False)})
    prefs.set(JRIVER_MCWS_ALIASES, {'a.local:1': 'Den'})
    widget = JRiverConnectionsWidget(prefs)
    qtbot.addWidget(widget)
    authenticate = MagicMock(return_value=True)
    monkeypatch.setattr('model.jriver.mcws.MediaServer.authenticate', authenticate)

    widget.refresh_aliases()
    qtbot.wait(100)

    authenticate.assert_not_called()


def test_the_filter_manager_dialog_shows_and_remembers_the_name_once_it_has_connected(qtbot, tmp_path, monkeypatch):
    from model.jriver.ui import MCWSDialog
    prefs = _prefs(tmp_path)
    prefs.set(JRIVER_MCWS_CONNECTIONS, {'a.local:1': (None, False)})
    _friendly(monkeypatch, {'a.local:1': 'Cinema PC'})
    monkeypatch.setattr('model.jriver.mcws.MediaServer.get_zones',
                        lambda self: (self.authenticate(), {'Main': 0})[1])
    dialog = MCWSDialog(None, prefs)
    qtbot.addWidget(dialog)
    assert dialog.savedConnections.item(0).text() == 'a.local:1 [Unauthenticated]'

    dialog.savedConnections.setCurrentRow(0)
    qtbot.waitUntil(lambda: dialog.zones.count() == 1)

    assert dialog.savedConnections.item(0).text() == 'Cinema PC (a.local:1) [Unauthenticated]'
    assert load_connections(prefs)[0].alias == 'Cinema PC'


def test_filter_manager_zone_loading_does_not_block_and_ignores_a_late_result_after_close(qtbot, tmp_path, monkeypatch):
    from model.jriver.ui import MCWSDialog
    prefs = _prefs(tmp_path)
    prefs.set(JRIVER_MCWS_CONNECTIONS, {'a.local:1': (None, False)})
    started, release = threading.Event(), threading.Event()

    def held(_server):
        started.set()
        release.wait(5)
        return {'Late zone': 1}

    monkeypatch.setattr('model.jriver.mcws.MediaServer.get_zones', held)
    dialog = MCWSDialog(None, prefs)
    qtbot.addWidget(dialog)
    dialog.savedConnections.setCurrentRow(0)

    assert started.wait(5)
    assert not dialog.upload.isEnabled()
    assert dialog.resultText.toPlainText() == 'Loading zones...'
    dialog.reject()
    release.set()
    qtbot.wait(100)

    assert dialog.zones.count() == 0


def test_filter_manager_zone_loading_shows_a_failure_inline(qtbot, tmp_path, monkeypatch):
    from model.jriver.ui import MCWSDialog
    prefs = _prefs(tmp_path)
    prefs.set(JRIVER_MCWS_CONNECTIONS, {'a.local:1': (None, False)})
    monkeypatch.setattr('model.jriver.mcws.MediaServer.get_zones', lambda _server: (_ for _ in ()).throw(ValueError('down')))
    dialog = MCWSDialog(None, prefs)
    qtbot.addWidget(dialog)
    dialog.savedConnections.setCurrentRow(0)

    qtbot.waitUntil(lambda: 'ValueError: down' in dialog.resultText.toPlainText())

    assert dialog.zones.isEnabled() and dialog.savedConnections.isEnabled()


# --- path mappings ------------------------------------------------------------------------------------------

FILMS = PathMapping('W:\\Films', '/mnt/films')
TV = PathMapping('W:\\TV', '/mnt/tv')


def _table(widget):
    t = widget.mappingsTable
    return [(t.item(r, 0).text(), t.item(r, 1).text()) for r in range(t.rowCount())]


def _type_row(widget, source, target):
    widget.addMappingButton.click()
    row = widget.mappingsTable.rowCount() - 1
    widget.mappingsTable.item(row, 0).setText(source)
    widget.mappingsTable.item(row, 1).setText(target)


def test_mappings_are_stored_per_server_and_survive_a_reload(tmp_path):
    prefs = _prefs(tmp_path)
    save_connections(prefs, [SavedConnection('a.local:1', path_mappings=(FILMS, TV)), SavedConnection('b.local:2')])

    loaded = {c.endpoint: c.path_mappings for c in load_connections(prefs)}

    assert loaded == {'a.local:1': (FILMS, TV), 'b.local:2': ()}
    assert prefs.get(JRIVER_MCWS_PATH_MAPPINGS) == {'a.local:1': [['W:\\Films', '/mnt/films'], ['W:\\TV', '/mnt/tv']]}
    save_connections(prefs, [SavedConnection('b.local:2')])  # deleting a server drops its mappings
    assert prefs.get(JRIVER_MCWS_PATH_MAPPINGS) == {}


def test_set_path_mappings_replaces_or_clears_one_servers_rules(tmp_path):
    prefs = _prefs(tmp_path)
    save_connections(prefs, [SavedConnection('a.local:1', path_mappings=(FILMS,)),
                             SavedConnection('b.local:2', path_mappings=(TV,))])

    set_path_mappings(prefs, 'a.local:1', [TV])
    assert {c.endpoint: c.path_mappings for c in load_connections(prefs)} == {'a.local:1': (TV,), 'b.local:2': (TV,)}
    set_path_mappings(prefs, 'a.local:1', [])
    assert {c.endpoint: c.path_mappings for c in load_connections(prefs)} == {'a.local:1': (), 'b.local:2': (TV,)}


def test_the_mapping_table_needs_a_selected_server_and_shows_its_rules(qtbot, tmp_path):
    prefs = _prefs(tmp_path)
    save_connections(prefs, [SavedConnection('a.local:1', path_mappings=(FILMS, TV)), SavedConnection('b.local:2')])
    widget = JRiverConnectionsWidget(prefs)
    qtbot.addWidget(widget)
    assert not widget.mappingsGroup.isEnabled()

    _select(widget, 'a.local:1')
    assert widget.mappingsGroup.isEnabled()
    assert _table(widget) == [('W:\\Films', '/mnt/films'), ('W:\\TV', '/mnt/tv')]

    _select(widget, 'b.local:2')
    assert _table(widget) == []

    widget.newButton.click()
    assert not widget.mappingsGroup.isEnabled()
    assert _table(widget) == []


def test_editing_the_table_saves_at_once_without_a_connection_test(qtbot, tmp_path):
    widget, prefs = _widget_with(qtbot, tmp_path, {'a.local:1': (None, False)})
    _select(widget, 'a.local:1')

    with qtbot.waitSignal(widget.changed):
        _type_row(widget, 'W:\\Films', '/mnt/films')

    assert load_connections(prefs)[0].path_mappings == (FILMS,)
    assert not widget.addButton.isEnabled()  # nothing about the connection itself changed


def test_local_mapping_folder_chooser_fills_the_selected_row(qtbot, tmp_path, monkeypatch):
    widget, prefs = _widget_with(qtbot, tmp_path, {'a.local:1': (None, False)})
    _select(widget, 'a.local:1')
    _type_row(widget, 'W:\\Films', '')
    monkeypatch.setattr('model.jriver.connections.QFileDialog.getExistingDirectory', lambda *args: '/mnt/films')

    widget.chooseMappingFolderButton.click()

    assert load_connections(prefs)[0].path_mappings == (FILMS,)


def test_a_half_typed_row_stays_on_screen_but_is_not_saved(qtbot, tmp_path):
    widget, prefs = _widget_with(qtbot, tmp_path, {'a.local:1': (None, False)})
    _select(widget, 'a.local:1')

    _type_row(widget, 'W:\\Films', '   ')

    assert _table(widget) == [('W:\\Films', '   ')]
    assert load_connections(prefs)[0].path_mappings == ()
    widget.mappingsTable.item(0, 1).setText('/mnt/films')
    assert load_connections(prefs)[0].path_mappings == (FILMS,)


def test_removing_a_row_removes_that_rule(qtbot, tmp_path):
    prefs = _prefs(tmp_path)
    save_connections(prefs, [SavedConnection('a.local:1', path_mappings=(FILMS, TV))])
    widget = JRiverConnectionsWidget(prefs)
    qtbot.addWidget(widget)
    _select(widget, 'a.local:1')

    widget.mappingsTable.selectRow(0)
    widget.removeMappingButton.click()

    assert load_connections(prefs)[0].path_mappings == (TV,)
    assert _table(widget) == [('W:\\TV', '/mnt/tv')]


def test_reselecting_a_server_shows_its_saved_rules_not_stale_ones(qtbot, tmp_path):
    widget, prefs = _widget_with(qtbot, tmp_path, {'a.local:1': (None, False), 'b.local:2': (None, False)})
    _select(widget, 'a.local:1')
    _type_row(widget, 'W:\\Films', '/mnt/films')
    _select(widget, 'b.local:2')

    _select(widget, 'a.local:1')

    assert _table(widget) == [('W:\\Films', '/mnt/films')]


def test_updating_a_server_keeps_its_mappings_including_when_it_is_renamed(qtbot, tmp_path, monkeypatch):
    prefs = _prefs(tmp_path)
    save_connections(prefs, [SavedConnection('a.local:1', path_mappings=(FILMS,))])
    widget = JRiverConnectionsWidget(prefs)
    qtbot.addWidget(widget)
    monkeypatch.setattr('model.jriver.mcws.MediaServer.authenticate', MagicMock(return_value=True))
    _select(widget, 'a.local:1')

    widget.httpsCheck.setChecked(True)
    _test_and_wait(qtbot, widget)
    widget.addButton.click()
    assert load_connections(prefs) == [SavedConnection('a.local:1', secure=True, path_mappings=(FILMS,))]

    widget.endpointEdit.setText('renamed.local:2')
    _test_and_wait(qtbot, widget)
    widget.addButton.click()

    assert load_connections(prefs) == [SavedConnection('renamed.local:2', secure=True, path_mappings=(FILMS,))]
    assert list(prefs.get(JRIVER_MCWS_PATH_MAPPINGS)) == ['renamed.local:2']
