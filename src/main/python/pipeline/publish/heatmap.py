'''
The published heatmap image: what Analyse Signal's *compare* mode shows for a title's mono track -- the filtered signal on
the left, the unfiltered on the right -- rendered without a window, to PNG bytes.

The drawing is `model.heatmap`'s, the code the dialog draws with; this module is the figure around it (size, title, colour
bar) and the defaults of a published image: the person's own Analyse Signal preferences (`spec_from_preferences()`), limited
to 1-40 Hz on both panes.
'''
import io
from dataclasses import dataclass
from typing import Optional

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from model import heatmap as shared
from model.heatmap import add_colour_bar, draw_pane, hide_y_axis, resolution_shift, set_limits, spectrogram_data, \
    two_pane_axes, with_default_range


@dataclass(frozen=True)
class HeatmapSpec(shared.HeatmapSpec):
    ''' The dialog's settings plus the image's size; both panes reach 40 Hz unless told otherwise. '''
    width_px: int = 1600
    height_px: int = 900
    dpi: int = 100
    max_filtered_freq: float = 40
    max_unfiltered_freq: float = 40


def spec_from_preferences(preferences, max_freq: float = 40) -> HeatmapSpec:
    ''' The person's Analyse Signal settings, limited to `max_freq` on both panes. '''
    return shared.spec_from_preferences(preferences, cls=HeatmapSpec, max_filtered_freq=max_freq,
                                        max_unfiltered_freq=max_freq)


def render_heatmap(filtered, unfiltered, spec: HeatmapSpec = HeatmapSpec(), title: str = '') -> bytes:
    '''
    :param filtered: the signal with the filter applied, shown on the left (a `model.signal.Signal`).
    :param unfiltered: the signal as it is, shown on the right.
    :return: PNG bytes.
    '''
    shift = resolution_shift(spec.resolution_multiplier)
    left, right = spectrogram_data(filtered, shift), spectrogram_data(unfiltered, shift)
    spec = with_default_range(spec, left, right)
    duration = max(float(left.t[-1]), float(right.t[-1])) if left.t.size and right.t.size else 1.0
    figure = Figure(figsize=(spec.width_px / spec.dpi, spec.height_px / spec.dpi), dpi=spec.dpi)
    FigureCanvasAgg(figure)
    left_axes, right_axes = two_pane_axes(figure, spec)
    draw_pane(left_axes, left, filtered, spec, spec.max_filtered_freq)
    set_limits(left_axes, spec, spec.max_filtered_freq, duration)
    drawn = draw_pane(right_axes, right, unfiltered, spec, spec.max_unfiltered_freq)
    set_limits(right_axes, spec, spec.max_unfiltered_freq, duration)
    hide_y_axis(right_axes)
    left_axes.set_xlabel('Filtered (Hz)')
    right_axes.set_xlabel('Unfiltered (Hz)')
    add_colour_bar(right_axes, drawn).set_label('dB')
    if title:
        figure.suptitle(title, fontsize=18)
    buffer = io.BytesIO()
    figure.savefig(buffer, format='png', dpi=spec.dpi)
    return buffer.getvalue()


def heatmap_for(signal, complete_filter, spec: HeatmapSpec = HeatmapSpec(), title: str = '',
                fs: Optional[int] = None) -> bytes:
    '''
    The heatmap of a track and what `complete_filter` does to it: applied as Analyse Signal does
    (`SingleChannelSignalData.filter_signal`), resampled to the signal's rate and run through `sosfilt`.
    :param signal: the mono track, a `SingleChannelSignalData` (its `.signal` and `.fs` are used).
    '''
    raw = signal.signal
    sos = complete_filter.resample(fs or signal.fs, copy_listener=False).get_sos()
    filtered = raw.sosfilter(sos) if len(sos) > 0 else raw
    return render_heatmap(filtered, raw, spec, title)
