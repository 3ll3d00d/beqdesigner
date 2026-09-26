'''
The published heatmap (pipeline.publish.heatmap): Analyse Signal's compare view of the mono track, filtered on the left and
unfiltered on the right, drawn without a window.
'''
import io

import numpy as np
import pytest
from PIL import Image

from model.iir import CompleteFilter, PeakingEQ
from model.preferences import ELLIPSE, POINT, SPECTROGRAM_CONTOURED, SPECTROGRAM_FLAT
from model.signal import Signal
from pipeline.publish.heatmap import HeatmapSpec, heatmap_for, render_heatmap, spec_from_preferences


def _signal(fs=1000, seconds=20):
    t = np.arange(fs * seconds) / fs
    burst = np.where((t % 5) < 1, 1.0, 0.05)
    samples = 0.5 * burst * np.sin(2 * np.pi * 25 * t) + 0.01 * np.random.default_rng(1).standard_normal(t.size)
    return Signal('s', samples, fs=fs)


def _png(data: bytes):
    image = Image.open(io.BytesIO(data))
    assert image.format == 'PNG'
    return image


@pytest.mark.parametrize('marker', [POINT, ELLIPSE, SPECTROGRAM_FLAT, SPECTROGRAM_CONTOURED])
def test_every_marker_type_renders_a_png_of_the_specified_size(marker):
    sig = _signal()
    data = render_heatmap(sig, sig, HeatmapSpec(marker_type=marker, width_px=800, height_px=450))
    assert _png(data).size == (800, 450)


def test_a_filter_changes_the_left_pane_only():
    sig = _signal()
    boost = CompleteFilter(fs=1000, filters=[PeakingEQ(1000, 25, 2, 12)])
    filtered = heatmap_for(_wrapped(sig), boost, HeatmapSpec(marker_type=SPECTROGRAM_FLAT, width_px=600, height_px=400))
    plain = heatmap_for(_wrapped(sig), CompleteFilter(fs=1000, filters=[]),
                        HeatmapSpec(marker_type=SPECTROGRAM_FLAT, width_px=600, height_px=400))
    left, right = slice(0, 240), slice(330, 560)   # the two panes' pixels
    a, b = np.asarray(_png(filtered).convert('L')), np.asarray(_png(plain).convert('L'))
    assert not np.array_equal(a[:, left], b[:, left])
    assert np.array_equal(a[:, right], b[:, right])


class _wrapped:
    ''' What heatmap_for() reads of a SingleChannelSignalData: `.signal` and `.fs`. '''
    def __init__(self, signal):
        self.signal, self.fs = signal, signal.fs


def test_the_title_is_drawn():
    sig = _signal()
    with_title = render_heatmap(sig, sig, HeatmapSpec(width_px=600, height_px=400), title='A Very Long Title Indeed')
    without = render_heatmap(sig, sig, HeatmapSpec(width_px=600, height_px=400))
    assert with_title != without


def test_the_spec_follows_the_persons_analysis_preferences():
    class Prefs:
        values = {'audio/max_filtered_freq': 60, 'audio/max_unfiltered_freq': 200, 'audio/marker_type': ELLIPSE, 'audio/marker_size': 3, 'audio/ellipse_width': 2.0,
                  'audio/ellipse_height': 0.5, 'audio/min_freq': 2, 'audio/colour_max': -5, 'audio/colour_min': -65}

        def get(self, key):
            return self.values[key]

    spec = spec_from_preferences(Prefs())
    assert (spec.marker_type, spec.marker_size, spec.ellipse_width, spec.ellipse_height) == (ELLIPSE, 3, 2.0, 0.5)
    assert (spec.min_freq, spec.max_filtered_freq, spec.max_unfiltered_freq, spec.colour_min, spec.colour_max) == (2, 40, 40, -65, -5)
