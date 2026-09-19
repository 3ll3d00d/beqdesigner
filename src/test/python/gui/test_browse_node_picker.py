'''The JRiver browse-node picker (plan §11.4) and its wiring into the JRiver source page.'''
from qtpy.QtCore import QSettings, Qt
from qtpy.QtWidgets import QDialogButtonBox

from model.browse_node_picker import ROOT_LABEL, JRiverBrowseNodePicker
from model.library_sources import JRiverSourcePage
from model.preferences import JRIVER_MCWS_CONNECTIONS, LIBRARY_JRIVER_BROWSE_NODE, LIBRARY_JRIVER_BROWSE_PATH, \
    Preferences
from pipeline.library.jriver import BrowseNode

TREE = {
    -1: [BrowseNode(1, 'Audio'), BrowseNode(2, 'Video')],
    1: [],
    2: [BrowseNode(21, 'Movies'), BrowseNode(22, 'Needs BEQ')],
    22: [BrowseNode(221, 'Atmos')],
}


def _fetch(calls):
    def fetch(node_id):
        calls.append(node_id)
        if node_id not in TREE:
            raise ConnectionError('server went away')
        return TREE[node_id]
    return fetch


def _children(item):
    return [item.child(i) for i in range(item.childCount())]


def _root(picker):
    return picker.tree.topLevelItem(0)


def _find(item, name):
    return next(c for c in _children(item) if c.text(0) == name)


def _preferences(tmp_path):
    return Preferences(QSettings(str(tmp_path / 'settings.ini'), QSettings.Format.IniFormat))


def test_the_root_is_listed_on_open_and_deeper_levels_only_when_expanded(qtbot):
    calls = []
    picker = JRiverBrowseNodePicker(None, _fetch(calls))
    qtbot.addWidget(picker)

    qtbot.waitUntil(lambda: _root(picker).childCount() == 2)
    assert _root(picker).text(0) == ROOT_LABEL
    assert [c.text(0) for c in _children(_root(picker))] == ['Audio', 'Video']
    assert calls == [-1]  # nothing below the root was fetched

    _find(_root(picker), 'Video').setExpanded(True)
    qtbot.waitUntil(lambda: _find(_root(picker), 'Video').childCount() == 2)
    assert calls == [-1, 2]


def test_a_level_is_fetched_once_however_often_it_is_toggled(qtbot):
    calls = []
    picker = JRiverBrowseNodePicker(None, _fetch(calls))
    qtbot.addWidget(picker)
    qtbot.waitUntil(lambda: _root(picker).childCount() == 2)
    video = _find(_root(picker), 'Video')

    video.setExpanded(True)
    qtbot.waitUntil(lambda: video.childCount() == 2)
    video.setExpanded(False)
    video.setExpanded(True)

    assert calls.count(2) == 1


def test_choosing_a_node_reports_its_id_and_path(qtbot):
    picker = JRiverBrowseNodePicker(None, _fetch([]))
    qtbot.addWidget(picker)
    qtbot.waitUntil(lambda: _root(picker).childCount() == 2)
    video = _find(_root(picker), 'Video')
    video.setExpanded(True)
    qtbot.waitUntil(lambda: video.childCount() == 2)
    needs_beq = _find(video, 'Needs BEQ')
    needs_beq.setExpanded(True)
    qtbot.waitUntil(lambda: needs_beq.childCount() == 1)

    picker.tree.setCurrentItem(_find(needs_beq, 'Atmos'))
    picker.buttonBox.button(QDialogButtonBox.StandardButton.Ok).click()

    assert picker.selected_node_id == 221
    assert picker.selected_path == 'Video > Needs BEQ > Atmos'


def test_the_root_can_be_chosen_and_starts_selected(qtbot):
    picker = JRiverBrowseNodePicker(None, _fetch([]))
    qtbot.addWidget(picker)

    assert picker.buttonBox.button(QDialogButtonBox.StandardButton.Ok).isEnabled()
    picker.buttonBox.button(QDialogButtonBox.StandardButton.Ok).click()

    assert (picker.selected_node_id, picker.selected_path) == (-1, ROOT_LABEL)


def test_cancelling_selects_nothing(qtbot):
    picker = JRiverBrowseNodePicker(None, _fetch([]))
    qtbot.addWidget(picker)

    picker.buttonBox.button(QDialogButtonBox.StandardButton.Cancel).click()

    assert picker.selected_node_id is None and picker.selected_path is None


def test_a_failed_fetch_is_reported_inline_and_retried_on_the_next_expand(qtbot):
    fetch_calls = []

    def flaky(node_id):
        fetch_calls.append(node_id)
        if node_id == 2 and fetch_calls.count(2) == 1:
            raise ConnectionError('server went away')
        return TREE[node_id]

    picker = JRiverBrowseNodePicker(None, flaky)
    qtbot.addWidget(picker)
    qtbot.waitUntil(lambda: _root(picker).childCount() == 2)
    video = _find(_root(picker), 'Video')

    video.setExpanded(True)
    qtbot.waitUntil(lambda: 'server went away' in picker.statusLabel.text())
    assert 'enter the id by hand' in picker.statusLabel.text()
    assert not video.isExpanded() and video.childCount() == 0

    video.setExpanded(True)
    qtbot.waitUntil(lambda: video.childCount() == 2)


def test_a_node_found_to_have_no_children_loses_its_expander(qtbot):
    picker = JRiverBrowseNodePicker(None, _fetch([]))
    qtbot.addWidget(picker)
    qtbot.waitUntil(lambda: _root(picker).childCount() == 2)
    audio = _find(_root(picker), 'Audio')

    audio.setExpanded(True)
    qtbot.waitUntil(lambda: audio.childIndicatorPolicy().name == 'DontShowIndicator')


# --- the JRiver source page -----------------------------------------------------------------------------------

def _page(qtbot, tmp_path, **prefs_to_set):
    prefs = _preferences(tmp_path)
    prefs.set(JRIVER_MCWS_CONNECTIONS, {'media.local:52199': (('u', 'p'), True)})
    for key, value in prefs_to_set.items():
        prefs.set(key, value)
    page = JRiverSourcePage()
    qtbot.addWidget(page)
    page.load(prefs)
    return page, prefs


class _FakePicker:
    result = (True, 22, 'Video > Needs BEQ')
    seen = {}

    def __init__(self, parent, fetch, current_id):
        _FakePicker.seen = {'fetch': fetch, 'current_id': current_id}
        self.selected_node_id, self.selected_path = _FakePicker.result[1:]

    def exec(self):
        return _FakePicker.result[0]


def test_picking_a_node_fills_the_id_and_path_and_persists_both(qtbot, tmp_path, monkeypatch):
    page, prefs = _page(qtbot, tmp_path, **{LIBRARY_JRIVER_BROWSE_NODE: 5})
    monkeypatch.setattr('model.library_sources.JRiverBrowseNodePicker', _FakePicker)

    page.pickNodeButton.click()

    assert _FakePicker.seen['current_id'] == 5
    assert page.browseNodeSpin.value() == 22
    assert page.nodePathLabel.text() == 'Video > Needs BEQ'
    page.save(prefs)
    assert prefs.get(LIBRARY_JRIVER_BROWSE_NODE) == 22
    assert prefs.get(LIBRARY_JRIVER_BROWSE_PATH) == 'Video > Needs BEQ'
    reopened, _ = _page(qtbot, tmp_path)
    assert reopened.nodePathLabel.text() == 'Video > Needs BEQ'
    assert reopened.browseNodeSpin.value() == 22


def test_the_picker_browses_the_selected_server(qtbot, tmp_path, monkeypatch):
    page, _ = _page(qtbot, tmp_path)
    monkeypatch.setattr('model.library_sources.JRiverBrowseNodePicker', _FakePicker)
    listed = []
    monkeypatch.setattr('model.library_sources.list_browse_children',
                        lambda *args, **kwargs: listed.append((args, kwargs)) or [])

    page.pickNodeButton.click()
    _FakePicker.seen['fetch'](7)

    assert listed == [(('media.local', 52199, 7), {'username': 'u', 'password': 'p', 'ssl': True})]


def test_cancelling_the_picker_changes_nothing(qtbot, tmp_path, monkeypatch):
    page, _ = _page(qtbot, tmp_path, **{LIBRARY_JRIVER_BROWSE_NODE: 5, LIBRARY_JRIVER_BROWSE_PATH: 'Kept'})
    monkeypatch.setattr(_FakePicker, 'result', (False, None, None))
    monkeypatch.setattr('model.library_sources.JRiverBrowseNodePicker', _FakePicker)

    page.pickNodeButton.click()

    assert page.browseNodeSpin.value() == 5
    assert page.nodePathLabel.text() == 'Kept'


def test_typing_an_id_by_hand_clears_the_remembered_path(qtbot, tmp_path):
    page, _ = _page(qtbot, tmp_path, **{LIBRARY_JRIVER_BROWSE_NODE: 5, LIBRARY_JRIVER_BROWSE_PATH: 'Old path'})
    assert page.nodePathLabel.text() == 'Old path'

    page.browseNodeSpin.setValue(9)

    assert page.nodePathLabel.text() == ''


def test_choosing_needs_a_server(qtbot, tmp_path):
    page = JRiverSourcePage()
    qtbot.addWidget(page)
    page.load(_preferences(tmp_path))  # no servers saved

    assert not page.pickNodeButton.isEnabled()
