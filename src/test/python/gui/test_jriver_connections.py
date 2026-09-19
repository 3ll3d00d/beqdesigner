'''
The shared JRiver server list (model/jriver/connections.py): parsing of what QSettings hands back, the
Preferences -> JRiver management widget, and the filter manager's zone dialog picking from the same list.
'''
import ui.beq  # noqa: F401 -- must precede model.jriver.ui, see AGENTS.md's circular-import note

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
