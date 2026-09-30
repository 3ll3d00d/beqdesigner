"""Title-page spectrum comparison shares publication settings, renderer and edited-project resolution."""
import ui.beq  # noqa: F401

import io
import threading
from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf
from PIL import Image
from qtpy.QtCore import QThreadPool

import model.worklist_spectrum as spectrum
from model.codec import filter_from_json
from model.preferences import AUDIO_ANALYSIS_MARKER_TYPE, AUDIO_ANALYSIS_COLOUR_MIN, SPECTROGRAM_FLAT
from pipeline.config import AnalysisConfig
from pipeline.orchestrate import Session
from pipeline.publish.heatmap import heatmap_for, spec_from_preferences
from pipeline.publish.project import preview_published_projects
from pipeline.review import read_entry
from test_worklist_title import REVIEWABLE, _designer, _open, _window  # noqa: F401 (fixture)
from worklist_project_fixture import corrupt_project, edit_project, write_projects


@pytest.fixture(autouse=True)
def _finish_draws(qtbot):
    yield
    qtbot.wait(10)
    # A test must not leave a rendering job referencing its temporary files.
    assert QThreadPool.globalInstance().waitForDone(15000)
    qtbot.wait(10)


def _audio(tmp_path, title_id):
    directory = tmp_path / 'work' / title_id
    directory.mkdir(parents=True, exist_ok=True)
    fs = 1000
    t = np.arange(fs * 8) / fs
    sf.write(directory / 'mono.wav', 0.15 * np.sin(2 * np.pi * 25 * t), fs, subtype='FLOAT')
    return directory / 'mono.wav'


def _show(page):
    page.rightTabs.setCurrentWidget(page.spectrumPanel)


def test_title_open_precomputes_the_exact_publication_image_before_the_tab_is_shown(qtbot, tmp_path):
    _audio(tmp_path, 'r-alien')
    window = _window(qtbot, tmp_path, entries=REVIEWABLE)
    window._preferences.set(AUDIO_ANALYSIS_MARKER_TYPE, SPECTROGRAM_FLAT)
    window._preferences.set(AUDIO_ANALYSIS_COLOUR_MIN, -55)
    page = _open(qtbot, window, 'r-alien')
    assert page.rightTabs.tabText(page.rightTabs.indexOf(page.spectrumPanel)) == 'Spectrum comparison'
    assert page.rightTabs.currentWidget() is not page.spectrumPanel
    qtbot.waitUntil(lambda: bool(page.spectrumPanel.png), timeout=15000)
    cached_png = page.spectrumPanel.png
    _show(page)
    assert page.spectrumPanel.png == cached_png and page.spectrumPanel._job is None
    entry = read_entry(str(tmp_path / 'queue'), 'r-alien')
    signal = Session(window._setup.settings.config).load(str(tmp_path / 'work' / 'r-alien' / 'mono.wav'))
    expected = heatmap_for(signal, filter_from_json(entry.offered[0].filters), spec_from_preferences(window._preferences),
                           title=entry.meta['title'])
    assert np.array_equal(np.asarray(Image.open(io.BytesIO(page.spectrumPanel.png))),
                          np.asarray(Image.open(io.BytesIO(expected))))
    assert Image.open(io.BytesIO(page.spectrumPanel.png)).size == (1600, 900)
    assert 'Filtered on the left' in page.spectrumPanel.statusLabel.text()
    assert page.spectrumPanel._cached.spec.max_filtered_freq == page.spectrumPanel._cached.spec.max_unfiltered_freq == 40
    assert page.spectrumPanel._cached.spec.colour_min == -55


def test_rendering_runs_on_a_worker_and_follows_candidates_without_writing_projects(qtbot, tmp_path, monkeypatch):
    _audio(tmp_path, 'r-alien')
    window = _window(qtbot, tmp_path, entries=REVIEWABLE)
    calls = []
    real = spectrum.render_comparison

    def record(request):
        calls.append((request, threading.current_thread()))
        return real(request)

    monkeypatch.setattr(spectrum, 'render_comparison', record)
    page = _open(qtbot, window, 'r-alien')
    _show(page)
    qtbot.waitUntil(lambda: bool(page.spectrumPanel.png), timeout=15000)
    first = page.spectrumPanel.png
    assert calls[0][1] is not threading.main_thread()
    assert page.pick_candidate(1)
    qtbot.waitUntil(lambda: bool(page.spectrumPanel.png) and page.spectrumPanel.png != first, timeout=15000)
    assert len(calls) == 2 and calls[0][0].filters != calls[1][0].filters
    assert not (tmp_path / 'work' / 'r-alien' / 'r-alien.mono.beq').exists()
    assert page.pick_candidate(1)
    qtbot.wait(20)
    assert len(calls) == 2  # same candidate/spec/files reuse the current image
    assert page.pick_candidate(0)
    assert page.spectrumPanel.png == first and len(calls) == 2
    assert page.show_title('r-arrival')
    assert page.show_title('r-alien')
    assert page.spectrumPanel.png == first and len(calls) == 2
    page.spectrumPanel.refreshButton.click()
    qtbot.waitUntil(lambda: page.spectrumPanel._job is None, timeout=15000)
    assert len(calls) == 3  # explicit refresh bypasses the cache


@pytest.mark.parametrize('edited_side', ['mono', 'multichannel'])
def test_comparison_uses_saved_project_edits_as_publication_would(tmp_path, edited_side):
    mono, mc = write_projects(tmp_path / 'work', 'r-alien', multichannel=True)
    edit_project(mono if edited_side == 'mono' else mc)
    from test_worklist_title import _prefs
    from worklist_title_fixture import write_entry
    write_entry(str(tmp_path / 'queue'), 'r-alien')
    entry = read_entry(str(tmp_path / 'queue'), 'r-alien')
    request = spectrum.comparison_request(str(tmp_path / 'work'), entry, 1, 'Alien', AnalysisConfig(), _prefs(tmp_path))
    before = [open(path, 'rb').read() for path in (mono, mc)]
    data, note = spectrum.render_comparison(request)
    published = preview_published_projects(mono, mc, filter_from_json(entry.offered[1].filters))
    expected = heatmap_for(Session(request.config).load(request.mono_wav), published.filter, request.spec, title='Alien')
    assert data == expected and f'Saved {edited_side} project edits' in note
    assert before == [open(path, 'rb').read() for path in (mono, mc)]


def test_missing_audio_and_not_designed_titles_explain_why_the_tab_is_empty(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, entries=REVIEWABLE)
    page = _open(qtbot, window, 'r-alien')
    _show(page)
    assert 'No extracted mono audio' in page.spectrumPanel.statusLabel.text()
    assert page.spectrumPanel._job is None
    assert page.show_title('x-gravity')
    assert 'No design to compare' in page.spectrumPanel.statusLabel.text()
    assert not page.spectrumPanel.png


def test_a_render_error_is_shown_and_refresh_can_recover(qtbot, tmp_path):
    mono, _ = write_projects(tmp_path / 'work', 'r-alien')
    corrupt_project(mono)
    window = _window(qtbot, tmp_path, entries=REVIEWABLE)
    page = _open(qtbot, window, 'r-alien')
    _show(page)
    qtbot.waitUntil(lambda: 'Could not generate' in page.spectrumPanel.statusLabel.text(), timeout=15000)
    assert not page.spectrumPanel.png and page.spectrumPanel.refreshButton.isEnabled()
    write_projects(tmp_path / 'work', 'r-alien')
    page.spectrumPanel.refreshButton.click()
    qtbot.waitUntil(lambda: bool(page.spectrumPanel.png), timeout=15000)


def test_changing_title_drops_late_results_and_coalesces_the_next_comparison(qtbot, tmp_path, monkeypatch):
    for title in ('r-alien', 'r-arrival', 'r-sicario'):
        _audio(tmp_path, title)
    entered, release = threading.Event(), threading.Event()
    real = spectrum.render_comparison
    requests = []

    def gated(request):
        requests.append(request.title_id)
        if request.title_id == 'r-alien':
            entered.set()
            assert release.wait(15)
        return real(request)

    monkeypatch.setattr(spectrum, 'render_comparison', gated)
    window = _window(qtbot, tmp_path, entries=REVIEWABLE)
    page = _open(qtbot, window, 'r-alien')
    _show(page)
    qtbot.waitUntil(entered.is_set, timeout=5000)
    seen = []
    page.spectrumPanel.ready.connect(seen.append)
    try:
        assert page.show_title('r-arrival')
        assert page.show_title('r-sicario')
        assert not page.spectrumPanel.png and requests == ['r-alien']
    finally:
        release.set()
    qtbot.waitUntil(lambda: bool(page.spectrumPanel.png), timeout=15000)
    assert requests == ['r-alien', 'r-sicario'] and len(seen) == 1
    assert page.spectrumPanel._cached.title_id == 'r-sicario'


def test_leaving_the_page_discards_a_late_comparison(qtbot, tmp_path, monkeypatch):
    _audio(tmp_path, 'r-alien')
    entered, release = threading.Event(), threading.Event()
    real = spectrum.render_comparison

    def gated(request):
        entered.set()
        assert release.wait(15)
        return real(request)

    monkeypatch.setattr(spectrum, 'render_comparison', gated)
    window = _window(qtbot, tmp_path, entries=REVIEWABLE)
    page = _open(qtbot, window, 'r-alien')
    _show(page)
    qtbot.waitUntil(entered.is_set, timeout=5000)
    try:
        assert window.close_title()
    finally:
        release.set()
    qtbot.waitUntil(lambda: page.spectrumPanel._job is None, timeout=15000)
    assert page.spectrumPanel.png == b'' and page.spectrumPanel._cached is None


def test_preferences_and_saved_project_changes_invalidate_the_preview(qtbot, tmp_path):
    mono, _ = write_projects(tmp_path / 'work', 'r-alien')
    window = _window(qtbot, tmp_path, entries=REVIEWABLE)
    page = _open(qtbot, window, 'r-alien')
    _show(page)
    qtbot.waitUntil(lambda: bool(page.spectrumPanel.png), timeout=15000)
    old = page.spectrumPanel._cached
    edit_project(mono)
    page.refresh_projects()
    qtbot.waitUntil(lambda: page.spectrumPanel._cached is not None and page.spectrumPanel._cached != old, timeout=15000)
    assert 'Saved mono project edits' in page.spectrumPanel.statusLabel.text()
    window._preferences.set(AUDIO_ANALYSIS_COLOUR_MIN, -45)
    page.spectrumPanel.refreshButton.click()
    qtbot.waitUntil(lambda: page.spectrumPanel._cached is not None and page.spectrumPanel._cached.spec.colour_min == -45,
                    timeout=15000)


def test_conflicting_project_edits_refuse_the_comparison_without_changing_either_project(tmp_path):
    from test_pipeline_publish_project import _OTHER_HUMAN_FILTER, _hand_edit_filter
    from test_worklist_title import _prefs
    from worklist_title_fixture import write_entry
    mono, mc = write_projects(tmp_path / 'work', 'r-alien', multichannel=True)
    edit_project(mono)
    _hand_edit_filter(mc, _OTHER_HUMAN_FILTER)
    write_entry(str(tmp_path / 'queue'), 'r-alien')
    entry = read_entry(str(tmp_path / 'queue'), 'r-alien')
    request = spectrum.comparison_request(str(tmp_path / 'work'), entry, 0, 'Alien', AnalysisConfig(), _prefs(tmp_path))
    with pytest.raises(ValueError, match='edited mono and multichannel projects disagree'):
        spectrum.render_comparison(request)


def test_the_profile_analysis_config_and_missing_setup_are_supported(qtbot, tmp_path):
    _audio(tmp_path, 'r-alien')
    window = _window(qtbot, tmp_path, entries=REVIEWABLE)
    config = AnalysisConfig(target_fs=500, resolution=2)
    window._setup = replace(window._setup, settings=replace(window._setup.settings, config=config))
    page = _open(qtbot, window, 'r-alien')
    assert page.spectrumPanel._request.config == config
    window._setup = replace(window._setup, settings=None)
    page._refresh_spectrum()
    assert 'No work directory' in page.spectrumPanel.statusLabel.text()
    assert page.spectrumPanel._request is None


def test_preview_cache_is_bounded_and_evicts_the_least_recently_used_image(qtbot, tmp_path, monkeypatch):
    from test_worklist_title import _prefs
    from worklist_title_fixture import write_entry
    _audio(tmp_path, 'r-alien')
    write_entry(str(tmp_path / 'queue'), 'r-alien')
    request = spectrum.comparison_request(str(tmp_path / 'work'), read_entry(str(tmp_path / 'queue'), 'r-alien'),
                                          0, 'Alien', AnalysisConfig(), _prefs(tmp_path))
    panel = spectrum.SpectrumPanel()
    qtbot.addWidget(panel)
    monkeypatch.setattr(panel, '_start', lambda: None)
    buffer = io.BytesIO()
    Image.new('RGB', (2, 2)).save(buffer, format='PNG')
    png = buffer.getvalue()
    requests = [replace(request, title=f'Title {index}') for index in range(5)]
    for item in requests[:4]:
        panel.set_request(item)
        panel._finished(item, png, 'Ready')
    panel.set_request(requests[0])  # this revisit makes the first image the newest
    assert panel.png == png
    panel.set_request(requests[4])
    panel._finished(requests[4], png, 'Ready')
    assert [item[0] for item in panel._images] == [requests[2], requests[3], requests[0], requests[4]]
    panel.set_request(requests[1])
    assert panel.png == b''  # evicted image requires fresh background generation
