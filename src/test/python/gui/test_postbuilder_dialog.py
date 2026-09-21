'''
Trial of pytest-qt as the standard way to test a real Qt widget: constructs
the actual CreateAVSPostDialog (model/postbuilder.py) under qtbot, offscreen
(no DISPLAY needed -- run with QT_QPA_PLATFORM=offscreen), drives it the way
a user would (type a title, click the TMDB button), and asserts the widget
fields populate correctly with requests.get mocked.

This is the safety net for CreateAVSPostDialog's TMDB lookup, which now
calls pipeline.metadata.tmdb_lookup()/tmdb_details_by_id() instead of its
own inline requests.get() calls -- this test passed against both the old
and new implementation (the _FakeResponse shape needed raise_for_status()
added for the new one, since tmdb_lookup() checks that instead of a bare
status_code).

Uses a real Preferences object backed by a temp-file QSettings (IniFormat)
rather than the user's actual QSettings("3ll3d00d", "beqdesigner") store, so
this never touches real application settings.
'''
from qtpy.QtCore import QSettings, Qt

from model.postbuilder import CreateAVSPostDialog
from model.preferences import Preferences, TMDB_API_KEY


class _FakeResponse:
    def __init__(self, json_body, status_code=200):
        self._json = json_body
        self.status_code = status_code

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"{self.status_code}")


def _make_preferences(tmp_path):
    settings = QSettings(str(tmp_path / 'settings.ini'), QSettings.Format.IniFormat)
    prefs = Preferences(settings)
    prefs.set(TMDB_API_KEY, 'dummy-key')
    return prefs


def test_load_tmdb_info_populates_fields_from_a_search_result(qtbot, tmp_path, monkeypatch):
    prefs = _make_preferences(tmp_path)
    dialog = CreateAVSPostDialog(None, prefs, filter_model=None, selected_signal=None)
    qtbot.addWidget(dialog)

    search_response = _FakeResponse({'results': [{'id': 335984}]})
    details_response = _FakeResponse({
        'title': 'Ready Player One',
        'original_title': 'Ready Player One',
        'poster_path': '/abc.jpg',
        'overview': 'A VR adventure.',
        'genres': [{'id': 28, 'name': 'Action'}],
        'belongs_to_collection': None,
        'runtime': 140,
        'release_date': '2018-03-29',
        'release_dates': {'results': [
            {'iso_3166_1': 'US', 'release_dates': [{'type': 3, 'certification': 'PG-13'}]},
        ]},
    })
    calls = []

    def fake_get(url, params, timeout=None):
        calls.append((url, params))
        return search_response if 'search' in url else details_response

    monkeypatch.setattr('requests.get', fake_get)

    dialog.titleField.setText('Ready Player One')
    dialog.yearField.setText('2018')
    qtbot.mouseClick(dialog.tmdbButton, Qt.MouseButton.LeftButton)

    assert dialog.titleField.text() == 'Ready Player One'
    assert dialog.movidDBIDField.text() == '335984'
    assert dialog.ratingField.text() == 'PG-13'
    assert dialog.runtimeField.text() == '140'
    assert dialog.posterURL == '/abc.jpg'
    assert dialog.overview == 'A VR adventure.'
    assert dialog.genres == [{'id': 28, 'name': 'Action'}]

    assert calls[0][1]['api_key'] == 'dummy-key'
    assert 'search/movie' in calls[0][0]
    assert 'movie/335984' in calls[1][0]


def test_load_tmdb_info_by_known_id_skips_the_search_call(qtbot, tmp_path, monkeypatch):
    ''' the movidDBIDField-populated path (load_tmdb_info's tmdb_id != '' branch). '''
    prefs = _make_preferences(tmp_path)
    dialog = CreateAVSPostDialog(None, prefs, filter_model=None, selected_signal=None)
    qtbot.addWidget(dialog)

    details_response = _FakeResponse({
        'title': 'Ready Player One',
        'original_title': 'Ready Player One',
        'poster_path': '/abc.jpg',
        'overview': 'A VR adventure.',
        'genres': [{'id': 28, 'name': 'Action'}],
        'belongs_to_collection': None,
        'runtime': 140,
        'release_date': '2018-03-29',
        'release_dates': {'results': []},
    })
    calls = []

    def fake_get(url, params, timeout=None):
        calls.append((url, params))
        return details_response

    monkeypatch.setattr('requests.get', fake_get)

    dialog.movidDBIDField.setText('335984')
    qtbot.mouseClick(dialog.tmdbButton, Qt.MouseButton.LeftButton)

    assert dialog.titleField.text() == 'Ready Player One'
    assert len(calls) == 1  # no search call -- went straight to details
    assert 'movie/335984' in calls[0][0]


def test_load_tmdb_info_swallows_an_http_error_and_stops_the_spinner(qtbot, tmp_path, monkeypatch):
    '''
    Mirrors the original behaviour: a non-2xx TMDB response results in no
    field changes (not a crash, not a stuck spinner) -- the original code
    checked status_code == 200 and silently did nothing otherwise.
    '''
    prefs = _make_preferences(tmp_path)
    dialog = CreateAVSPostDialog(None, prefs, filter_model=None, selected_signal=None)
    qtbot.addWidget(dialog)

    monkeypatch.setattr('requests.get', lambda url, params, timeout=None: _FakeResponse({}, status_code=404))

    dialog.titleField.setText('Some Unknown Title')
    dialog.movidDBIDField.setText('999999999')
    qtbot.mouseClick(dialog.tmdbButton, Qt.MouseButton.LeftButton)

    assert dialog.titleField.text() == 'Some Unknown Title'  # unchanged
    assert dialog.tmdbButton.text() == 'Load TMDB Info'  # spinner was stopped, not left spinning
    assert dialog.tmdbButton.isEnabled()


def test_build_metadata_reflects_form_state(qtbot, tmp_path):
    prefs = _make_preferences(tmp_path)
    dialog = CreateAVSPostDialog(None, prefs, filter_model=None, selected_signal=None)
    qtbot.addWidget(dialog)

    dialog.titleField.setText('Ready Player One')
    dialog.yearField.setText('2018')
    dialog.atmosCheckBox.setChecked(True)
    dialog.autofillSortTitle()

    metadata = dialog._CreateAVSPostDialog__build_metadata()

    assert metadata['beq_title'] == 'Ready Player One'
    assert metadata['beq_year'] == '2018'
    assert metadata['beq_sortTitle'] == 'ready player one'
    assert metadata['beq_audioTypes'] == ['Atmos']
