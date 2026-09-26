'''
The heatmap Analyse Signal draws (max spectrum by time: where the heavy hits are, in frequency across and time up), as
functions of a `HeatmapSpec` and matplotlib axes -- no widget is read here.

Two callers share it: `model.analysis.MaxSpectrumByTime` (the dialog, which builds a spec from its controls) and
`pipeline.publish.heatmap` (the published image, which builds one from the person's saved Analyse Signal preferences), so
the published image is drawn by the code that draws what the person sees. The dialog keeps what is about being a window:
its caches, updating a scatter in place and clearing the figure when the layout changes.
'''
import datetime
import math
from dataclasses import dataclass, replace
from typing import Optional, Tuple

import matplotlib
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter, MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from model.preferences import AUDIO_ANALYIS_MAX_FILTERED_FREQ, AUDIO_ANALYIS_MAX_UNFILTERED_FREQ, \
    AUDIO_ANALYIS_MIN_FREQ, AUDIO_ANALYSIS_COLOUR_MAX, AUDIO_ANALYSIS_COLOUR_MIN, AUDIO_ANALYSIS_ELLIPSE_HEIGHT, \
    AUDIO_ANALYSIS_ELLIPSE_WIDTH, AUDIO_ANALYSIS_MARKER_SIZE, AUDIO_ANALYSIS_MARKER_TYPE, ELLIPSE, POINT, \
    SPECTROGRAM_CONTOURED, SPECTROGRAM_FLAT

MAG_CONSTANT, MAG_PEAK, MAG_AVERAGE = 'Constant', 'Peak', 'Average'
SIGNAL_RANGE_DB = 60.0   # how far below the loudest hit (Constant) or each frequency's peak (Peak) a point is still drawn
# the resolution selector's steps: 1.0 is the signal's own default segment length
MULTIPLIERS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]


@dataclass(frozen=True)
class HeatmapSpec:
    ''' Everything the heatmap is drawn from that is not the signal. Defaults are Analyse Signal's own. '''
    marker_type: str = POINT              # point, ellipse, spectrogram (flat) or spectrogram (contoured)
    marker_size: int = 1
    ellipse_width: float = 3.0
    ellipse_height: float = 1.0
    min_freq: float = 1
    max_filtered_freq: float = 40         # the left pane
    max_unfiltered_freq: float = 160      # the right pane
    colour_min: float = -70
    colour_max: float = -10
    mag_limit_type: str = MAG_CONSTANT    # what points and ellipses are thresholded against
    signal_min: Optional[float] = None    # Constant: the level (dB); Peak: added to each peak. None: default_signal_min()
    resolution_multiplier: float = 1.0
    min_time: float = 0.0                 # seconds
    max_time: Optional[float] = None      # seconds; None is the end of the signal


def spec_from_preferences(preferences, cls=HeatmapSpec, **overrides) -> HeatmapSpec:
    ''' The person's saved Analyse Signal settings (`audio/...`); `overrides` are fields of `cls` (a HeatmapSpec). '''
    values = dict(
        marker_type=preferences.get(AUDIO_ANALYSIS_MARKER_TYPE), marker_size=preferences.get(AUDIO_ANALYSIS_MARKER_SIZE),
        ellipse_width=preferences.get(AUDIO_ANALYSIS_ELLIPSE_WIDTH),
        ellipse_height=preferences.get(AUDIO_ANALYSIS_ELLIPSE_HEIGHT), min_freq=preferences.get(AUDIO_ANALYIS_MIN_FREQ),
        max_filtered_freq=preferences.get(AUDIO_ANALYIS_MAX_FILTERED_FREQ),
        max_unfiltered_freq=preferences.get(AUDIO_ANALYIS_MAX_UNFILTERED_FREQ),
        colour_min=preferences.get(AUDIO_ANALYSIS_COLOUR_MIN), colour_max=preferences.get(AUDIO_ANALYSIS_COLOUR_MAX))
    values.update(overrides)
    return cls(**values)


def default_signal_min(mag_limit_type: str, loudest: float) -> Optional[float]:
    ''' What the signal range control is set to on analysing: 60 dB under the loudest hit, or under each peak. '''
    if mag_limit_type == MAG_CONSTANT:
        return loudest - SIGNAL_RANGE_DB
    if mag_limit_type == MAG_PEAK:
        return -SIGNAL_RANGE_DB
    return None


@dataclass(frozen=True)
class SpectrogramData:
    ''' A signal's spectrogram at one resolution: `sxx` is (frequency, time), x/y/z the same flattened frequency-major. '''
    f: np.ndarray
    t: np.ndarray
    sxx: np.ndarray
    shift: float

    @property
    def x(self):
        return self.f.repeat(self.t.size)

    @property
    def y(self):
        return np.tile(self.t, self.f.size)

    @property
    def z(self):
        return self.sxx.flatten()


def resolution_shift(multiplier: float) -> float:
    return math.log(multiplier, 2)


def spectrogram_data(signal, shift: float) -> SpectrogramData:
    ''' :param signal: a `model.signal.Signal`. '''
    f, t, sxx = signal.spectrogram(resolution_shift=shift)
    return SpectrogramData(f, t, sxx, shift)


def loudest(*panes: SpectrogramData) -> float:
    return max(float(np.max(p.sxx)) for p in panes)


def _threshold(data: SpectrogramData, signal, spec: HeatmapSpec) -> np.ndarray:
    ''' The level a point must reach, per frequency. '''
    if spec.mag_limit_type == MAG_CONSTANT:
        return np.full(data.f.size, spec.signal_min)
    if spec.mag_limit_type == MAG_PEAK:
        return data.sxx.max(axis=-1) + spec.signal_min
    return signal.avg_spectrum(resolution_shift=data.shift)[1]


def select_points(data: SpectrogramData, signal, spec: HeatmapSpec, max_freq: float
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    :return: (x, y, z) of the points loud enough for their frequency and inside the frequency and time limits,
        quietest first so that later points overlay them.
    '''
    x, y, z = data.x, data.y, data.z
    keep = (z >= np.repeat(_threshold(data, signal, spec), data.t.size)) & (x >= spec.min_freq) & (x <= max_freq) \
        & (y >= spec.min_time)
    if spec.max_time is not None:
        keep &= y <= spec.max_time
    x, y, z = x[keep], y[keep], z[keep]
    order = z.argsort()
    return x[order], y[order], z[order]


def _slices(data: SpectrogramData, spec: HeatmapSpec, max_freq: float):
    f, t = data.f, data.t
    f_min = np.argmax(f >= spec.min_freq)
    f_max = np.argmin(f <= max_freq)
    t_min = np.argmax(t >= spec.min_time) if spec.min_time > 0 else 0
    t_max = -1 if spec.max_time is None or spec.max_time >= t[-1] else np.argmin(t <= spec.max_time)
    return slice(f_min, f_max), slice(t_min, t_max)


def _marker(spec: HeatmapSpec):
    if spec.marker_type == ELLIPSE:
        area = spec.ellipse_width * spec.ellipse_height * np.pi
        theta = np.arange(0, 2 * np.pi + 0.01, 0.1)
        return np.column_stack([spec.ellipse_width / area * np.cos(theta), spec.ellipse_height / area * np.sin(theta)])
    return '.'


def _marker_area(spec: HeatmapSpec) -> float:
    return matplotlib.rcParams['lines.markersize'] ** 2.0 * (spec.marker_size ** 2)


def draw_pane(axes, data: SpectrogramData, signal, spec: HeatmapSpec, max_freq: float, scatter=None):
    '''
    Draws one pane. With a `scatter` from an earlier call (a point or ellipse plot) its points are replaced in place and
    nothing else is touched; the spectrogram types are drawn once. Otherwise the pane is drawn and its axes formatted.
    :return: the artist to hang a colour bar from.
    '''
    x, y, z = select_points(data, signal, spec, max_freq)
    area = _marker_area(spec)
    if scatter is not None:
        if spec.marker_type in (POINT, ELLIPSE):
            new_data = np.c_[x, y]
            scatter.set_offsets(new_data)
            scatter.set_clim(vmin=spec.colour_min, vmax=spec.colour_max)
            scatter.set_array(z)
            scatter.set_sizes(np.full(new_data.size, area))
        return scatter
    if spec.marker_type == SPECTROGRAM_FLAT:
        f_slice, t_slice = _slices(data, spec, max_freq)
        scatter = axes.pcolormesh(data.f[f_slice], data.t[t_slice], data.sxx[f_slice, t_slice].transpose(),
                                  vmin=spec.colour_min, vmax=spec.colour_max, shading='auto')
    elif spec.marker_type == SPECTROGRAM_CONTOURED and z.size > 2:
        scatter = axes.tricontourf(x, y, z, np.sort(np.arange(spec.colour_max, spec.colour_min, -0.5)),
                                   vmin=spec.colour_min, vmax=spec.colour_max)
    else:
        scatter = axes.scatter(x, y, c=z, s=area, vmin=spec.colour_min, vmax=spec.colour_max, marker=_marker(spec))
    axes.yaxis.set_major_formatter(FuncFormatter(seconds_to_hhmmss))
    axes.yaxis.set_major_locator(MaxNLocator(nbins=24, min_n_ticks=8, steps=[1, 3, 6]))
    axes.xaxis.set_major_locator(MaxNLocator(nbins=12, steps=[1, 5, 10], min_n_ticks=8))
    return scatter


def set_limits(axes, spec: HeatmapSpec, max_freq: float, duration: float) -> None:
    axes.set_xlim(left=spec.min_freq, right=max_freq)
    axes.set_ylim(bottom=spec.min_time, top=duration if spec.max_time is None else spec.max_time)


def add_grid(axes) -> None:
    axes.grid(linestyle='-', which='major', linewidth=1, alpha=0.3)


def width_ratios(spec: HeatmapSpec, adjusted: bool = False):
    a, b = spec.max_filtered_freq, spec.max_unfiltered_freq
    if adjusted:
        a1, b1 = a * 0.95, b * 1.05
        return [a1 / (a + b1), b / (a + b1)]
    return [a, b]


def two_pane_axes(figure, spec: HeatmapSpec):
    ''' :return: (left, right) axes, side by side and touching, the widths in proportion to the frequency ranges. '''
    grid = GridSpec(1, 2, width_ratios=width_ratios(spec, adjusted=True), wspace=0.00)
    grid.tight_layout(figure)
    left = figure.add_subplot(grid.new_subplotspec((0, 0)))
    add_grid(left)
    right = figure.add_subplot(grid.new_subplotspec((0, 1)))
    add_grid(right)
    return left, right


def hide_y_axis(axes) -> None:
    axes.set_yticklabels([])
    axes.get_yaxis().set_tick_params(length=0)


def add_colour_bar(axes, artist):
    cax = make_axes_locatable(axes).append_axes('right', size='5%', pad=0.05)
    return axes.figure.colorbar(artist, cax=cax)


def seconds_to_hhmmss(x, pos=None) -> str:
    ''' formats a seconds value to hhmmss '''
    return str(datetime.timedelta(seconds=x))


def with_default_range(spec: HeatmapSpec, *panes: SpectrogramData) -> HeatmapSpec:
    ''' `spec` with the signal range filled in from the loudest hit, if it has none. '''
    if spec.signal_min is not None:
        return spec
    return replace(spec, signal_min=default_signal_min(spec.mag_limit_type, loudest(*panes)))
