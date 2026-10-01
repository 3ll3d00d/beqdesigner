"""Published BEQ matching and a read-only, asynchronous catalogue pane."""
import ui.beq  # noqa: F401

import hashlib
import io
import json
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest
from PIL import Image
from qtpy.QtCore import QThreadPool, Qt

import model.worklist_published as published
from model.catalogue import CatalogueEntry
from model.preferences import BEQ_DOWNLOAD_DIR
from model.worklist_artwork import cache_path
from model.worklist_catalogue import CatalogueTarget, catalogue_target, matching_entries
from test_worklist_title import _designer, _open, _window  # noqa: F401


def record(**changes):
    return dict(title='Alien', year=1979, theMovieDB='348', audioTypes=['Atmos'], author='Alice',
                edition='Theatrical', language='English', source='Disc') | changes


def entry(**changes):
    return CatalogueEntry('0', record(**changes))


TARGET = CatalogueTarget('Alien', '1979', '348', audio_types=('Atmos',), edition='Theatrical',
                         language='English', source='Disc')


@pytest.fixture(autouse=True)
def finish_jobs(qtbot):
    yield
    qtbot.wait(10)
    assert QThreadPool.globalInstance().waitForDone(15000)
    qtbot.wait(10)


def png(colour='red'):
    output = io.BytesIO()
    Image.new('RGB', (120, 80), colour).save(output, format='PNG')
    return output.getvalue()


def panel(qtbot, tmp_path, records, monkeypatch, image_loader=None):
    window = _window(qtbot, tmp_path, entries=[('r-alien', {'meta': {
        'title': 'Alien', 'year': '1979', 'the_movie_db': '348', 'audio_types': ['Atmos'],
        'edition': 'Theatrical', 'language': 'English', 'source': 'Disc'}}),
        ('r-arrival', {'meta': {'title': 'Arrival', 'year': '2016', 'the_movie_db': '329865', 'audio_types': ['Atmos']}})])
    cache = tmp_path / 'catalogue'
    cache.mkdir()
    (cache / 'database.json').write_text(json.dumps(records))
    window._preferences.set(BEQ_DOWNLOAD_DIR, str(cache))
    monkeypatch.setattr(published, 'published_image', image_loader or (lambda *_: png()))
    page = _open(qtbot, window, 'r-alien')
    assert page.publishedPanel._catalogue_job is None  # no network until the tab is opened
    page.rightTabs.setCurrentWidget(page.publishedPanel)
    qtbot.waitUntil(lambda: page.publishedPanel._entries is not None)
    return window, page, page.publishedPanel


def test_identity_track_constraints_and_author_order():
    values = [entry(author='Zoe'), entry(author='alice'), entry(theMovieDB='999'), entry(year=1980, theMovieDB=''),
              entry(audioTypes=['DTS-HD MA']), entry(edition='Extended'), entry(language='French'), entry(source='Streaming'),
              entry(content_type='TV')]
    assert [e.author for e in matching_entries(TARGET, values)] == ['alice', 'Zoe']
    assert len(matching_entries(TARGET, values, True)) == 6
    assert matching_entries(None, values) == []
    assert matching_entries(CatalogueTarget(title='Alien'), values) == []
    assert matching_entries(replace(TARGET, tmdb=''), [entry(theMovieDB='', title='Other', altTitle='ALIÉN')])
    assert matching_entries(replace(TARGET, audio_types=('Dolby Atmos',)), [entry()])
    assert not matching_entries(replace(TARGET, audio_types=('DD+',)), [entry(audioTypes=['DD'])])


def test_tv_matches_season_and_overlapping_episodes():
    target = replace(TARGET, is_tv=True, season='2', episodes='3-5')
    records = [entry(content_type='TV', season='2', episode='1-3'),
               entry(content_type='TV', season='1', episode='3'),
               entry(content_type='TV', season='2', episode='6'), entry()]
    assert matching_entries(target, records) == records[:1]
    assert set(matching_entries(target, records, True)) == set(records[:3])


def test_target_uses_saved_metadata_defaults_and_index_fallback():
    row = SimpleNamespace(title='Alien', year=1979, external_ids={'tmdb': 'https://www.themoviedb.org/movie/348-alien'}, kind='film')
    target = catalogue_target(SimpleNamespace(meta={'audio_types': 'Atmos'}), row, {'source': 'Disc'})
    assert target == replace(TARGET, edition='', language='')
    assert catalogue_target(None, None).usable is False
    assert catalogue_target(SimpleNamespace(meta={'season': '2', 'episodes': [3, 4], 'the_movie_db': '001'}), row).is_tv


def test_single_pane_selects_authors_images_and_other_tracks_without_queue_writes(qtbot, tmp_path, monkeypatch):
    records = [record(author='Zoe', images=['https://images/zoe.png']),
               record(author='Alice', images=['https://images/chart.png', 'https://images/heatmap.png'],
                      catalogue_url='https://catalogue/alice', note='Check clipping', warning=['Warning'], mv=-3),
               record(author='Bob', audioTypes=['DTS-HD MA'], images=[])]
    calls = []
    def load(cache, url):
        calls.append((url, threading.current_thread()))
        return png('blue' if 'heatmap' in url else 'red')
    window, page, pane = panel(qtbot, tmp_path, records, monkeypatch, load)
    before = {p: p.read_bytes() for p in (tmp_path / 'queue').rglob('*.json')}
    candidate = page._picked
    qtbot.waitUntil(lambda: bool(pane.png))
    assert pane.entryChoice.count() == 2
    assert pane.entryChoice.currentData().author == 'Alice'
    assert '2 author(s)' in pane.statusLabel.text()
    assert 'Warning' in pane.details.toPlainText() and '-3' in pane.details.toPlainText()
    pane.imageChoice.setCurrentIndex(1)
    qtbot.waitUntil(lambda: pane.png == png('blue'))
    assert pane.imageChoice.currentText().endswith('heatmap')
    assert all(thread is not threading.main_thread() for _, thread in calls)
    opened = []
    monkeypatch.setattr(published.QDesktopServices, 'openUrl', lambda url: opened.append(url.toString()))
    qtbot.mouseClick(pane.catalogueButton, Qt.MouseButton.LeftButton)
    assert opened == ['https://catalogue/alice']
    pane.otherTracks.setChecked(True)
    assert pane.entryChoice.count() == 3
    pane.entryChoice.setCurrentIndex(1)
    assert pane.entryChoice.currentData().author == 'Bob'
    assert 'No published images' in pane.image.text()
    pane.image.resize(250, 150)
    qtbot.wait(10)
    assert 'No published images' in pane.image.text()
    assert page._picked == candidate
    assert before == {p: p.read_bytes() for p in (tmp_path / 'queue').rglob('*.json')}


def test_stale_image_is_dropped_and_next_title_uses_latest_request(qtbot, tmp_path, monkeypatch):
    started, release = threading.Event(), threading.Event()
    calls = []
    def load(cache, url):
        calls.append(url)
        if 'alien' in url:
            started.set()
            assert release.wait(5)
        return png('red' if 'alien' in url else 'blue')
    try:
        window, page, pane = panel(qtbot, tmp_path, [record(images=['https://images/alien.png']),
            record(title='Arrival', year=2016, theMovieDB='329865', images=['https://images/arrival.png'])], monkeypatch, load)
        qtbot.waitUntil(started.is_set)
        _open(qtbot, window, 'r-arrival')
        assert pane.png == b''
        release.set()
        qtbot.waitUntil(lambda: pane.png == png('blue'))
        assert calls == ['https://images/alien.png', 'https://images/arrival.png']
        assert pane.entryChoice.currentData().title == 'Arrival'
    finally:
        release.set()


def test_image_failure_can_retry_same_selection_and_survives_resize(qtbot, tmp_path, monkeypatch):
    calls = []
    def load(*_):
        calls.append(1)
        if len(calls) == 1:
            raise OSError('offline')
        return png()
    _, _, pane = panel(qtbot, tmp_path, [record(images=['https://images/a.png'])], monkeypatch, load)
    qtbot.waitUntil(lambda: 'offline' in pane.image.text())
    pane.image.resize(250, 150)
    qtbot.wait(10)
    assert 'offline' in pane.image.text()
    pane.imageChoice.activated.emit(0)
    qtbot.waitUntil(lambda: bool(pane.png))
    assert len(calls) == 2


def test_no_matches_and_missing_identity_have_clear_status(qtbot, tmp_path, monkeypatch):
    _, _, pane = panel(qtbot, tmp_path, [record(audioTypes=['DD'])], monkeypatch)
    assert pane.entryChoice.count() == 0 and 'Try Include other tracks' in pane.statusLabel.text()
    pane.set_target(CatalogueTarget())
    assert 'Set a TMDB ID' in pane.statusLabel.text()
    assert not pane.catalogueButton.isEnabled()


class Response:
    def __init__(self, data):
        self.data = data
        self.closed = False
    def raise_for_status(self):
        pass
    def iter_content(self, *_, **__):
        yield self.data
    def close(self):
        self.closed = True


def test_catalogue_download_is_atomic_and_offline_refresh_uses_shared_cache(tmp_path, monkeypatch):
    response = Response(json.dumps([record()]).encode())
    calls = []
    monkeypatch.setattr(published.requests, 'get', lambda url, **kw: (calls.append(kw), response)[1])
    entries, note = published.published_catalogue(str(tmp_path))
    assert entries[0].author == 'Alice' and response.closed and 'updated' in note
    assert calls == [{'timeout': 20, 'stream': True}]
    cached = (tmp_path / 'database.json').read_bytes()
    published.published_catalogue(str(tmp_path))
    assert len(calls) == 1
    monkeypatch.setattr(published.requests, 'get', lambda *_ , **__: Response(b'{}'))
    entries, note = published.published_catalogue(str(tmp_path), True)
    assert 'showing cached' in note and entries[0].author == 'Alice'
    assert (tmp_path / 'database.json').read_bytes() == cached
    assert not list(tmp_path.glob('.catalogue-*'))


def test_catalogue_failure_without_cache_is_reported_and_size_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(published.requests, 'get', lambda *_ , **__: Response(b'garbage'))
    with pytest.raises(ValueError):
        published.published_catalogue(str(tmp_path))
    monkeypatch.setattr(published, 'MAX_DATABASE_BYTES', 1)
    monkeypatch.setattr(published.requests, 'get', lambda *_ , **__: Response(b'[]'))
    with pytest.raises(ValueError, match='too large'):
        published.published_catalogue(str(tmp_path))
    assert not (tmp_path / 'database.json').exists()


def test_image_cache_uses_full_url_and_recovers_corrupt_cache(tmp_path, monkeypatch):
    urls = ['https://alice/image.png', 'https://bob/image.png']
    calls = []
    monkeypatch.setattr(published.requests, 'get', lambda url, **_: (calls.append(url), Response(png()))[1])
    for url in urls:
        assert published.published_image(str(tmp_path), url) == png()
    assert calls == urls
    assert published.published_image(str(tmp_path), urls[0]) == png() and calls == urls
    key = hashlib.sha256(urls[0].encode()).hexdigest()
    path = cache_path(str(tmp_path / '.published'), key, '.png')
    from pathlib import Path
    Path(path).write_bytes(b'broken')
    assert published.published_image(str(tmp_path), urls[0]) == png()
    assert calls == urls + urls[:1]


def test_leaving_title_discards_pending_image(qtbot, tmp_path, monkeypatch):
    started, release = threading.Event(), threading.Event()
    def load(*_):
        started.set()
        assert release.wait(5)
        return png()
    try:
        _, page, pane = panel(qtbot, tmp_path, [record(images=['https://images/a.png'])], monkeypatch, load)
        qtbot.waitUntil(started.is_set)
        assert page.leave()
        release.set()
        qtbot.waitUntil(lambda: pane._image_job is None)
        assert pane.png == b'' and pane.entryChoice.count() == 0
    finally:
        release.set()


def test_metadata_reload_updates_track_match(qtbot, tmp_path, monkeypatch):
    _, page, pane = panel(qtbot, tmp_path, [record(), record(author='Bob', audioTypes=['DD'])], monkeypatch)
    assert pane.entryChoice.currentData().author == 'Alice'
    from pipeline.review import read_entry, write_queue_entry
    item = read_entry(str(tmp_path / 'queue'), 'r-alien')
    item.meta['audio_types'] = ['DD']
    write_queue_entry(str(tmp_path / 'queue'), item)
    page.reload()
    assert pane.entryChoice.count() == 1 and pane.entryChoice.currentData().author == 'Bob'


def test_catalogue_failure_and_refresh_recovery_are_visible(qtbot, tmp_path, monkeypatch):
    window = _window(qtbot, tmp_path, entries=[('r-alien', {'meta': {'title': 'Alien', 'year': '1979'}})])
    window._preferences.set(BEQ_DOWNLOAD_DIR, str(tmp_path / 'missing-cache'))
    calls = []
    def load(*_, **__):
        calls.append(threading.current_thread())
        if len(calls) == 1:
            raise OSError('offline')
        return [entry()], 'Catalogue updated.'
    monkeypatch.setattr(published, 'published_catalogue', load)
    page = _open(qtbot, window, 'r-alien')
    pane = page.publishedPanel
    page.rightTabs.setCurrentWidget(pane)
    qtbot.waitUntil(lambda: 'offline' in pane.statusLabel.text())
    assert pane.refreshButton.isEnabled()
    qtbot.mouseClick(pane.refreshButton, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: pane.entryChoice.count() == 1)
    assert 'Catalogue updated' in pane.statusLabel.text()
    assert all(thread is not threading.main_thread() for thread in calls)


def test_controls_and_track_details_share_a_column_beside_the_image(qtbot, tmp_path):
    from test_worklist_title import _prefs
    view = published.PublishedPanel(_prefs(tmp_path))
    qtbot.addWidget(view)
    view.resize(1000, 600)
    view.show()
    qtbot.waitUntil(lambda: view.image.width() > 1)
    assert view.splitter.orientation() == Qt.Orientation.Horizontal
    assert view.splitter.count() == 3
    assert view.ourPane.isHidden()
    assert view.splitter.widget(0) is view.controlsColumn
    assert view.splitter.widget(1) is view.image
    for widget in (view.otherTracks, view.refreshButton, view.statusLabel, view.entryChoice,
                   view.imageChoice, view.catalogueButton, view.compareOurFilter, view.details):
        assert view.controlsColumn.isAncestorOf(widget)
    assert view.image.geometry().left() >= view.controlsColumn.geometry().right()
    assert view.image.height() == view.controlsColumn.height()
    assert view.details.height() > 130


def test_our_filter_can_be_shown_beside_the_published_image_and_follows_the_candidate(qtbot, tmp_path, monkeypatch):
    from test_worklist_title import REVIEWABLE
    monkeypatch.setattr(published, 'published_catalogue', lambda *_: ([entry(title='r-alien', year=2001, theMovieDB='', images=['https://images/chart.png'])], 'Ready'))
    monkeypatch.setattr(published, 'published_image', lambda *_: png())
    window = _window(qtbot, tmp_path, entries=REVIEWABLE)
    page = _open(qtbot, window, 'r-alien')
    page.rightTabs.setCurrentWidget(page.publishedPanel)
    pane = page.publishedPanel
    pane.otherTracks.setChecked(True)
    qtbot.waitUntil(lambda: bool(pane.png))
    before = pane.png
    assert pane.ourPane.isHidden()
    qtbot.mouseClick(pane.compareOurFilter, Qt.MouseButton.LeftButton)
    assert pane.ourPane.isVisible() and pane.image.isVisible()
    assert pane.ourChart.canvas.figure.axes[0].get_legend() is None
    assert pane.splitter.widget(1) is pane.image and pane.splitter.widget(2) is pane.ourPane
    first = pane._our_curves[-1].y.copy()
    assert page.pick_candidate(1)
    assert pane.ourStatus.text() == 'Selected design 2'
    import numpy as np
    assert not np.array_equal(first, pane._our_curves[-1].y)
    qtbot.waitUntil(lambda: bool(pane.png))
    assert pane.png == before
    qtbot.mouseClick(pane.compareOurFilter, Qt.MouseButton.LeftButton)
    assert pane.ourPane.isHidden() and pane.image.isVisible()
    assert window.close_title()
    assert pane._our_curves == [] and 'No design' in pane.ourStatus.text()
