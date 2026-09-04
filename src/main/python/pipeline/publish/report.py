'''
pipeline/publish/report.py: headless report renderer -- design/pipeline-
implementation-plan.md phase 3 (item 9).

A spec-driven render of the "Pixel Perfect Image | Chart" layout -- the only
layout docs/ui/report.md says matters for publishing ("All reports shared on
avsforum use the Pixel Perfect Image | Chart layout"). This is not a port of
SaveReportDialog (model/report.py); it is built against that layout's output
shape, per design/api-headless-pipeline.md §6's "spec, not a port" framing.

D4's follow-on (§14) -- fixed output width vs poster-driven, and whether the
filter table sits inside the chart axes or below it -- is answered by that
existing layout, not invented here:
  - the chart renders at a fixed pixel size (ReportSpec.width_px/height_px/
    dpi); the poster is resized to *match the chart's pixel width*
    afterward, not the other way round (mirrors
    SaveReportDialog.__concat_images, model/report.py:599).
  - the filter table is a matplotlib Table overlaid on the chart's own axes
    via a relative bbox (mirrors __make_table's `table_axes = ...axes_1`
    branch, model/report.py:184-190), not a separate subplot -- because in
    "Pixel Perfect Image | Chart" no filter_spec subplot is created
    (model/report.py:330-334), so replace_table falls into that branch.

Reuses MagnitudeModel (model/magnitude.py) for curve rendering via a
minimal canvas shim -- MagnitudeModel only touches chart.canvas.figure/
mpl_connect/mpl_disconnect/draw_idle, all present on a plain
FigureCanvasAgg. MagnitudeModel transitively imports qtpy (via
model/limits.py, for its interactive dialogs, which this module never
calls) -- this is a deliberate, plan-endorsed reuse (see the implementation
plan's "canvas shim" note), verified headless: importing this module
constructs no QApplication and needs no DISPLAY.

Scope trim: output is PNG only. SaveReportDialog's JPEG path exists to let
a user pick their own poster's format and prompts to convert it to RGB;
that prompt is interactive by nature and out of scope for a headless
renderer -- PNG has no such conversion question.
'''
import io
import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import matplotlib
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.table import Table

from model.iir import CompoundPassFilter, ComplexHighPass
from model.magnitude import MagnitudeModel, SINGLE_SUBPLOT_SPEC
from model.xy import MagnitudeData

TABLE_COLUMNS = ('Freq', 'Gain', 'Q', 'Type', 'Total')


@dataclass(frozen=True)
class ReportSpec:
    '''
    The fixed layout parameters SaveReportDialog measured off screen/window
    state (§6); here they are plain inputs. Defaults mirror that dialog's
    own preference defaults (model/preferences.py DEFAULT_PREFS) except
    width_px/height_px/dpi, which have no headless equivalent -- the dialog
    measured those from the window -- so a fixed 1600x900 @ 100dpi is
    chosen as a sane forum-post size.
    '''
    width_px: int = 1600
    height_px: int = 900
    dpi: int = 100
    title_font_size: int = 36
    show_legend: bool = False
    grid_alpha: float = 0.5
    x_min: float = 1
    x_max: float = 160
    x_scale: str = 'linear'
    show_table_header: bool = True
    table_font_size: int = 10
    table_row_height_multiplier: float = 1.2
    table_alpha: float = 1.0
    # (x0, y0, x1, y1), axes-relative -- matches REPORT_FILTER_X0/X1/Y0/Y1 defaults
    table_bbox: Tuple[float, float, float, float] = (0.748, 0.75, 1.0, 1.0)


class _SpecPreferences:
    '''
    A minimal stand-in for model.preferences.Preferences -- MagnitudeModel
    only needs a `.get(key)` -- backed by a plain dict instead of QSettings
    so this module never has to construct one. Any key it doesn't recognise
    (e.g. the graph/expand_y lookup MagnitudeModel makes unconditionally)
    resolves to its production default, False.
    '''
    def __init__(self, spec: ReportSpec):
        self.__values = {'x_min': spec.x_min, 'x_max': spec.x_max, 'x_scale': spec.x_scale}

    def get(self, key, default_if_unset=True):
        return self.__values.get(key, False)


class _AggChart:
    ''' the `chart` MagnitudeModel expects: an object with a `.canvas`. '''
    def __init__(self, figure: Figure):
        self.canvas = FigureCanvasAgg(figure)


def _table_print(filt, show_header: bool):
    ''' Qt-free port of SaveReportDialog.__table_print (model/report.py:424). '''
    vals = [str(filt.freq) if show_header else f"{filt.freq} Hz"] if hasattr(filt, 'freq') else ['']
    if isinstance(filt, CompoundPassFilter):
        vals.append('N/A')
        vals.append(f"{filt.type.value}{filt.order}")
        vals.append('HPF' if isinstance(filt, ComplexHighPass) else 'LPF')
        vals.append('')
    elif filt.filter_type.startswith('LPF') or filt.filter_type.startswith('HPF'):
        vals.append('N/A')
        vals.append(f"{round(filt.q, 4)}")
        vals.append(filt.filter_type)
        vals.append('')
    else:
        gain = filt.gain if hasattr(filt, 'gain') else 0
        g_suffix = ' dB' if gain != 0 and not show_header else ''
        vals.append(f"{gain:+g}{g_suffix}" if gain != 0 else 'N/A')
        vals.append(f"{round(filt.q, 4)}" if hasattr(filt, 'q') else 'N/A')
        filter_type = filt.filter_type
        if len(filt) > 1:
            filter_type += f" x{len(filt)}"
        vals.append(filter_type)
        if gain != 0 and len(filt) > 1:
            vals.append(f"{len(filt) * gain:+g}{g_suffix}")
        else:
            vals.append('')
    return vals


def _make_filter_table(axes, figure: Figure, filters: Sequence, spec: ReportSpec,
                       mv_offset: Optional[float] = None) -> Optional[Table]:
    '''
    Qt-free port of SaveReportDialog.__make_table/__add_filters_to_table
    (model/report.py:175-230), fixed to the bbox-on-chart-axes branch --
    the only one "Pixel Perfect Image | Chart" ever takes.
    '''
    filters = list(filters)
    if len(filters) == 0:
        return None
    bbox = spec.table_bbox
    table_loc = {'bbox': (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])}
    row_height = (spec.table_font_size / 72.0 * figure.dpi / axes.bbox.height * spec.table_row_height_multiplier)
    fc = axes.get_facecolor()
    cell_kwargs = {'facecolor': fc}

    table = Table(axes, **table_loc)
    table.set_zorder(1000)
    cols = TABLE_COLUMNS
    col_width = (1 / len(cols)) * table_loc['bbox'][2]
    cells = [_table_print(f, spec.show_table_header) for f in filters]
    if mv_offset is not None and not math.isclose(mv_offset, 0.0):
        cells.append(['', f"{mv_offset:+g}", '', 'MV', ''])
    ec = matplotlib.rcParams['axes.edgecolor']
    if spec.show_table_header:
        for idx, label in enumerate(cols):
            cell = table.add_cell(0, idx, width=col_width, height=row_height, text=label, loc='center',
                                  edgecolor=ec, **cell_kwargs)
            cell.set_alpha(spec.table_alpha)
    for idx, row in enumerate(cells):
        for col_idx, cell_text in enumerate(row):
            cell = table.add_cell(idx + (1 if spec.show_table_header else 0), col_idx, width=col_width,
                                  height=row_height, text=cell_text, loc='center', edgecolor=ec, **cell_kwargs)
            cell.PAD = 0.02
            cell.set_alpha(spec.table_alpha)
    axes.add_table(table)
    return table


def render_chart_png(curves: Sequence[MagnitudeData], filters: Sequence, spec: ReportSpec = ReportSpec(),
                     title: str = '', mv_offset: Optional[float] = None) -> bytes:
    '''
    Renders the magnitude chart + overlaid filter table -- the "chart" half
    of a pixel-perfect report -- to PNG bytes, at spec.width_px x
    spec.height_px pixels.
    '''
    figure = Figure(figsize=(spec.width_px / spec.dpi, spec.height_px / spec.dpi), dpi=spec.dpi)
    chart = _AggChart(figure)
    preferences = _SpecPreferences(spec)
    magnitude_model = MagnitudeModel('report', chart, preferences, lambda reference=None: list(curves), 'Signals',
                                     show_legend=lambda: spec.show_legend, subplot_spec=SINGLE_SUBPLOT_SPEC,
                                     grid_alpha=spec.grid_alpha, x_min_pref_key='x_min', x_max_pref_key='x_max',
                                     x_scale_pref_key='x_scale')
    _make_filter_table(magnitude_model.limits.axes_1, figure, filters, spec, mv_offset=mv_offset)
    if title:
        magnitude_model.limits.axes_1.set_title(title, fontsize=spec.title_font_size)
    chart.canvas.draw_idle()
    buf = io.BytesIO()
    figure.savefig(buf, format='png', dpi=spec.dpi)
    return buf.getvalue()


def compose_with_poster(chart_png: bytes, poster_path: Optional[str]) -> bytes:
    '''
    Stacks a poster image above the chart, resizing the poster to the
    chart's pixel width -- mirrors SaveReportDialog.__concat_images
    (model/report.py:578-609), which resizes to the *chart's* width and
    pastes the poster first (top), chart second (bottom).
    Returns chart_png unchanged if poster_path is None.
    '''
    if poster_path is None:
        return chart_png
    chart_image = Image.open(io.BytesIO(chart_png))
    poster_image = Image.open(poster_path)
    if poster_image.size[0] != chart_image.size[0]:
        new_height = round(poster_image.size[1] * (chart_image.size[0] / poster_image.size[0]))
        poster_image = poster_image.resize((chart_image.size[0], new_height), Image.Resampling.LANCZOS)
    if poster_image.mode != chart_image.mode:
        poster_image = poster_image.convert(chart_image.mode)
    final_image = Image.new(chart_image.mode, (chart_image.size[0], chart_image.size[1] + poster_image.size[1]))
    final_image.paste(poster_image, (0, 0))
    final_image.paste(chart_image, (0, poster_image.size[1]))
    out = io.BytesIO()
    final_image.save(out, format='PNG')
    return out.getvalue()


def render_report(curves: Sequence[MagnitudeData], filters: Sequence, poster_path: Optional[str] = None,
                  spec: ReportSpec = ReportSpec(), title: str = '', mv_offset: Optional[float] = None) -> bytes:
    ''' Renders the full report (poster + chart, or chart alone) to PNG bytes. '''
    chart_png = render_chart_png(curves, filters, spec=spec, title=title, mv_offset=mv_offset)
    return compose_with_poster(chart_png, poster_path)
