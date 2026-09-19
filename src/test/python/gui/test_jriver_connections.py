'''
The shared JRiver server list (model/jriver/connections.py): parsing of what QSettings hands back, the
Preferences -> JRiver management widget, and the filter manager's zone dialog picking from the same list.
'''
import ui.beq  # noqa: F401 -- must precede model.jriver.ui, see AGENTS.md's circular-import note

import threading
from unittest.mock import MagicMock

from qtpy.QtCore import QSettings

from model.jriver.connections import JRiverConnectionsWidget, SavedConnection, load_connections, parse_connections
from model.jriver.mcws import MCWSError
from model.preferences import JRIVER_MCWS_CONNECTIONS, Preferences


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
    assert [dialog.zones.item(i).text() for i in range(dialog.zones.count())] == ['Main']
    assert dialog.upload.isEnabled()
    assert not hasattr(dialog, 'addNewButton')  # management lives in Preferences now


def test_preferences_dialog_has_a_jriver_page_managing_the_same_list(qtbot, tmp_path):
    from model.preferences import PreferencesDialog
    prefs = _prefs(tmp_path)
    prefs.set(JRIVER_MCWS_CONNECTIONS, {'a.local:1': (None, False)})

    dialog = PreferencesDialog(prefs, str(tmp_path), MagicMock())
    qtbot.addWidget(dialog)

    widget = dialog.jriverPage.findChild(JRiverConnectionsWidget)
    assert widget is not None
    assert [c.endpoint for c in widget.connections()] == ['a.local:1']
