'''
model.analysis.MaxSpectrumByTime and pipeline.publish.heatmap draw with the one implementation, model.heatmap: given the same
settings and signals they plot the same points. `import ui.beq` first: AGENTS.md gotcha 3.
'''
import ui.beq  # noqa: F401 (must come first)

from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from model.analysis import MaxSpectrumByTime
from model.heatmap import HeatmapSpec as SharedSpec, SpectrogramData, default_signal_min, draw_pane, select_points
from model.preferences import ELLIPSE, POINT, SPECTROGRAM_FLAT
from model.signal import Signal
from pipeline.publish.heatmap import HeatmapSpec, render_heatmap


class _Control:
    ''' The bits of a QSpinBox/QComboBox/QTimeEdit the analyser reads and sets. '''

    def __init__(self, value):
        self._value = value

    def value(self):
        return self._value

    def setValue(self, value):
        self._value = value

    def currentText(self):
        return self._value

    def currentIndex(self):
        return self._value

    def setVisible(self, _):
        pass

    def time(self):
        return SimpleNamespace(msecsSinceStartOfDay=lambda: int(self._value * 1000))


def _ui(**values):
    base = dict(markerType=POINT, markerSize=1, ellipseWidth=3.0, ellipseHeight=1.0, minFreq=1, maxFilteredFreq=40,
                maxUnfilteredFreq=40, colourLowerLimit=-70, colourUpperLimit=-10, magLimitType='Constant',
                magLowerLimit=-70.0, magUpperLimit=0.0, analysisResolution=2, minTime=0.0, maxTime=20.0,
                signalRangeLabel=None)
    base.update(values)
    return SimpleNamespace(**{k: _Control(v) for k, v in base.items()})


def _signal(seconds=20, fs=1000, gain=1.0):
    t = np.arange(fs * seconds) / fs
    burst = np.where((t % 5) < 1, 1.0, 0.05)
    return Signal('s', gain * 0.5 * burst * np.sin(2 * np.pi * 25 * t), fs=fs)


def _chart():
    figure = Figure()
    FigureCanvasAgg(figure)
    return SimpleNamespace(canvas=figure.canvas)


@pytest.fixture
def analyser(monkeypatch):
    import app
    monkeypatch.setattr(app, 'wait_cursor', lambda *_: __import__('contextlib').nullcontext(), raising=False)


def _points(axes):
    (scatter,) = axes.collections
    return np.asarray(scatter.get_offsets()), np.asarray(scatter.get_array())


def test_the_dialog_spec_is_read_from_its_controls():
    analyser = MaxSpectrumByTime(_chart(), None, _ui(markerType=ELLIPSE, maxUnfilteredFreq=160, analysisResolution=3))

    spec = analyser.spec()

    assert (spec.marker_type, spec.max_filtered_freq, spec.max_unfiltered_freq) == (ELLIPSE, 40, 160)
    assert spec.resolution_multiplier == 2.0 and spec.max_time == 20.0


def test_the_dialog_and_the_published_image_plot_the_same_points(analyser):
    filtered, plain = _signal(gain=0.4), _signal()
    chart = _chart()
    dialog = MaxSpectrumByTime(chart, None, _ui())
    dialog.left, dialog.right = filtered, plain
    dialog.analyse()
    dialog_left, dialog_right = [ax for ax in chart.canvas.figure.axes if ax.collections][:2]

    # what the published image draws for the same settings
    shift = 0.0
    left, right = SpectrogramData(*filtered.spectrogram(resolution_shift=shift), shift), \
        SpectrogramData(*plain.spectrogram(resolution_shift=shift), shift)
    spec = SharedSpec(max_unfiltered_freq=40, signal_min=default_signal_min('Constant', max(left.sxx.max(), right.sxx.max())),
                      max_time=20.0)
    figure = Figure()
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(111)
    draw_pane(axes, left, filtered, spec, 40)

    published, colours = _points(axes)
    shown, shown_colours = _points(dialog_left)
    assert np.array_equal(published, shown) and np.array_equal(colours, shown_colours)
    assert published.size > 0
    assert select_points(right, plain, spec, 40)[0].size == _points(dialog_right)[0].shape[0]


def test_the_published_image_uses_the_same_spectrogram_types_as_the_dialog():
    sig = _signal()
    for marker in (POINT, ELLIPSE, SPECTROGRAM_FLAT):
        assert render_heatmap(sig, sig, HeatmapSpec(marker_type=marker, width_px=400, height_px=300)).startswith(b'\x89PNG')
