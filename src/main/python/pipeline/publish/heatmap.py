'''
The published heatmap image: what Analyse Signal's *compare* mode draws (model.analysis.MaxSpectrumByTime) for a title's
mono track -- the filtered signal in the left pane and the unfiltered signal in the right, frequency across and time up,
coloured by level -- rendered without a window or a Qt widget, to PNG bytes.

The marker type, colour range and so on are a `HeatmapSpec`; the app builds one from the person's own Analyse Signal
preferences (`spec_from_preferences()`), so the published image looks like the one they see. The frequency range is the
spec's, 1-40 Hz by default, on both panes. It mirrors the dialog's rendering (threshold, sorting, colour limits,
axis formatting) rather than sharing code with it, because that code reads its controls off the dialog's widgets.
'''
import datetime
import io
import math
from dataclasses import dataclass
from typing import Optional

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter, MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from model.preferences import ELLIPSE, POINT, SPECTROGRAM_CONTOURED, SPECTROGRAM_FLAT

MAG_CONSTANT, MAG_PEAK, MAG_AVERAGE = 'Constant', 'Peak', 'Average'


@dataclass(frozen=True)
class HeatmapSpec:
    ''' How the heatmap is drawn. Defaults are the Analyse Signal dialog's own defaults, with both panes limited to 40 Hz. '''
    width_px: int = 1600
    height_px: int = 900
    dpi: int = 100
    marker_type: str = POINT              # point, ellipse, spectrogram (flat) or spectrogram (contoured)
    marker_size: int = 1
    ellipse_width: float = 3.0
    ellipse_height: float = 1.0
    min_freq: float = 1
    max_freq: float = 40                  # both panes; the dialog has one limit for each
    colour_min: float = -70
    colour_max: float = -10
    mag_limit_type: str = MAG_CONSTANT    # what the points/ellipses are thresholded against: Constant, Peak or Average
    signal_range_db: float = 60.0         # Constant: this far below the loudest hit; Peak: below each frequency's peak
    resolution_multiplier: float = 1.0    # the dialog's resolution selector, 1.0 being the signal's own default


def spec_from_preferences(preferences, max_freq: float = HeatmapSpec.max_freq) -> HeatmapSpec:
    ''' The person's Analyse Signal settings (`model.preferences` `audio/...`), limited to `max_freq` on both panes. '''
    from model.preferences import AUDIO_ANALYIS_MIN_FREQ, AUDIO_ANALYSIS_COLOUR_MAX, AUDIO_ANALYSIS_COLOUR_MIN, \
        AUDIO_ANALYSIS_ELLIPSE_HEIGHT, AUDIO_ANALYSIS_ELLIPSE_WIDTH, AUDIO_ANALYSIS_MARKER_SIZE, \
        AUDIO_ANALYSIS_MARKER_TYPE
    return HeatmapSpec(
        marker_type=preferences.get(AUDIO_ANALYSIS_MARKER_TYPE), marker_size=preferences.get(AUDIO_ANALYSIS_MARKER_SIZE),
        ellipse_width=preferences.get(AUDIO_ANALYSIS_ELLIPSE_WIDTH),
        ellipse_height=preferences.get(AUDIO_ANALYSIS_ELLIPSE_HEIGHT),
        min_freq=preferences.get(AUDIO_ANALYIS_MIN_FREQ), max_freq=max_freq,
        colour_min=preferences.get(AUDIO_ANALYSIS_COLOUR_MIN), colour_max=preferences.get(AUDIO_ANALYSIS_COLOUR_MAX))


def _hhmmss(seconds, _position) -> str:
    return str(datetime.timedelta(seconds=seconds))


class _Pane:
    ''' One signal's spectrogram, at the spec's resolution. '''

    def __init__(self, signal, spec: HeatmapSpec):
        self.signal = signal
        self.shift = math.log(spec.resolution_multiplier, 2)
        self.f, self.t, self.sxx = signal.spectrogram(resolution_shift=self.shift)


def _threshold(pane: _Pane, spec: HeatmapSpec, loudest: float) -> np.ndarray:
    ''' The level a point must reach, per frequency. '''
    if spec.mag_limit_type == MAG_PEAK:
        return pane.sxx.max(axis=-1) - spec.signal_range_db
    if spec.mag_limit_type == MAG_AVERAGE:
        return pane.signal.avg_spectrum(resolution_shift=pane.shift)[1]
    return np.full(pane.f.size, loudest - spec.signal_range_db)


def _draw(axes, pane: _Pane, spec: HeatmapSpec, loudest: float, duration: float):
    f, t, sxx = pane.f, pane.t, pane.sxx
    in_range = (f >= spec.min_freq) & (f <= spec.max_freq)
    if spec.marker_type == SPECTROGRAM_FLAT:
        drawn = axes.pcolormesh(f[in_range], t, sxx[in_range].transpose(), vmin=spec.colour_min, vmax=spec.colour_max,
                                shading='auto')
    else:
        # frequency-major, as the dialog lays it out; a point is drawn only if it is loud enough for its frequency
        x = f.repeat(t.size)
        y = np.tile(t, f.size)
        z = sxx.flatten()
        keep = np.repeat(in_range, t.size) & (z >= np.repeat(_threshold(pane, spec, loudest), t.size))
        x, y, z = x[keep], y[keep], z[keep]
        order = z.argsort()   # the quietest first, so the loudest overlay them
        x, y, z = x[order], y[order], z[order]
        if spec.marker_type == SPECTROGRAM_CONTOURED and z.size > 2:
            drawn = axes.tricontourf(x, y, z, np.sort(np.arange(spec.colour_max, spec.colour_min, -0.5)),
                                     vmin=spec.colour_min, vmax=spec.colour_max)
        else:
            marker = '.'
            if spec.marker_type == ELLIPSE:
                area = spec.ellipse_width * spec.ellipse_height * np.pi
                theta = np.arange(0, 2 * np.pi + 0.01, 0.1)
                marker = np.column_stack([spec.ellipse_width / area * np.cos(theta),
                                          spec.ellipse_height / area * np.sin(theta)])
            drawn = axes.scatter(x, y, c=z, s=matplotlib.rcParams['lines.markersize'] ** 2.0 * (spec.marker_size ** 2),
                                 vmin=spec.colour_min, vmax=spec.colour_max, marker=marker)
    axes.set_xlim(left=spec.min_freq, right=spec.max_freq)
    axes.set_ylim(bottom=0, top=duration)
    axes.grid(linestyle='-', which='major', linewidth=1, alpha=0.3)
    axes.yaxis.set_major_formatter(FuncFormatter(_hhmmss))
    axes.yaxis.set_major_locator(MaxNLocator(nbins=24, min_n_ticks=8, steps=[1, 3, 6]))
    axes.xaxis.set_major_locator(MaxNLocator(nbins=12, steps=[1, 5, 10], min_n_ticks=8))
    return drawn


def render_heatmap(filtered, unfiltered, spec: HeatmapSpec = HeatmapSpec(), title: str = '') -> bytes:
    '''
    :param filtered: the signal with the filter applied, shown on the left (a `model.signal.Signal`).
    :param unfiltered: the signal as it is, shown on the right.
    :return: PNG bytes.
    '''
    left, right = _Pane(filtered, spec), _Pane(unfiltered, spec)
    loudest = max(float(np.max(left.sxx)), float(np.max(right.sxx)))
    duration = max(float(left.t[-1]), float(right.t[-1])) if left.t.size and right.t.size else 1.0
    figure = Figure(figsize=(spec.width_px / spec.dpi, spec.height_px / spec.dpi), dpi=spec.dpi)
    FigureCanvasAgg(figure)
    grid = GridSpec(1, 2, figure=figure, width_ratios=[1, 1], wspace=0.0)
    left_axes = figure.add_subplot(grid[0, 0])
    right_axes = figure.add_subplot(grid[0, 1])
    _draw(left_axes, left, spec, loudest, duration)
    drawn = _draw(right_axes, right, spec, loudest, duration)
    right_axes.set_yticklabels([])
    right_axes.get_yaxis().set_tick_params(length=0)
    left_axes.set_xlabel('Filtered (Hz)')
    right_axes.set_xlabel('Unfiltered (Hz)')
    cax = make_axes_locatable(right_axes).append_axes('right', size='2%', pad=0.05)
    figure.colorbar(drawn, cax=cax, label='dB')
    if title:
        figure.suptitle(title, fontsize=18)
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format='png', dpi=spec.dpi)
    return buffer.getvalue()


def heatmap_for(signal, complete_filter, spec: HeatmapSpec = HeatmapSpec(), title: str = '',
                fs: Optional[int] = None) -> bytes:
    '''
    The heatmap of a track and what `complete_filter` does to it: the filter is applied as Analyse Signal does
    (`SingleChannelSignalData.filter_signal`): resampled to the signal's rate and run through `sosfilt`.
    :param signal: the mono track, a `SingleChannelSignalData` (its `.signal` and `.fs` are used).
    '''
    raw = signal.signal
    sos = complete_filter.resample(fs or signal.fs, copy_listener=False).get_sos()
    filtered = raw.sosfilter(sos) if len(sos) > 0 else raw
    return render_heatmap(filtered, raw, spec, title)
