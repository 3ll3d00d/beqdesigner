'''
Phase 3 (item 9) of design/pipeline-implementation-plan.md: the headless
report renderer. Verifies the filter-table formatting logic against known
values (same shape as SaveReportDialog.__table_print), that a full render
produces valid, correctly-sized PNG bytes with no display/QApplication
involved, and that poster compositing matches the width-matching,
poster-on-top behaviour of SaveReportDialog.__concat_images.
'''
import io

import numpy as np
import pytest
from PIL import Image

from model.iir import CompleteFilter, LowShelf, PeakingEQ
from model.xy import MagnitudeData
from pipeline.publish.report import ReportSpec, _table_print, compose_with_poster, render_chart_png, render_report


def _rp1_filters(fs=96000):
    return CompleteFilter(fs=fs, filters=[LowShelf(fs, 18.0, 0.7, 4.5, count=5), PeakingEQ(fs, 40.0, 2.0, -3.0)])


def _flat_curve(name='Signal'):
    x = np.geomspace(1, 160, 50)
    return MagnitudeData(name, None, x, np.zeros_like(x))


def test_table_print_matches_expected_rp1_row_shape():
    filters = list(_rp1_filters())
    shelf, peak = filters
    assert _table_print(shelf, show_header=True) == ['18.0', '+4.5', '0.7', 'LS x5', '+22.5']
    assert _table_print(peak, show_header=True) == ['40.0', '-3', '2.0', 'PEQ', '']


def test_render_chart_png_is_the_requested_pixel_size():
    spec = ReportSpec(width_px=640, height_px=360, dpi=80)
    png = render_chart_png([_flat_curve()], list(_rp1_filters()), spec=spec, title='Ready Player One')
    image = Image.open(io.BytesIO(png))
    assert image.size == (640, 360)
    assert image.format == 'PNG'


def test_render_report_without_poster_matches_chart_only():
    spec = ReportSpec(width_px=320, height_px=180, dpi=80)
    chart_png = render_chart_png([_flat_curve()], list(_rp1_filters()), spec=spec)
    full_png = render_report([_flat_curve()], list(_rp1_filters()), poster_path=None, spec=spec)
    assert chart_png == full_png


def test_compose_with_poster_resizes_to_chart_width_and_stacks_poster_on_top(tmp_path):
    spec = ReportSpec(width_px=400, height_px=225, dpi=80)
    chart_png = render_chart_png([_flat_curve()], list(_rp1_filters()), spec=spec)
    chart_image = Image.open(io.BytesIO(chart_png))

    poster_path = str(tmp_path / 'poster.jpg')
    Image.new('RGB', (300, 450), color=(10, 20, 30)).save(poster_path, format='JPEG')

    composed = compose_with_poster(chart_png, poster_path)
    composed_image = Image.open(io.BytesIO(composed))

    assert composed_image.size[0] == chart_image.size[0]
    expected_poster_height = round(450 * (chart_image.size[0] / 300))
    assert composed_image.size[1] == chart_image.size[1] + expected_poster_height
    # poster pasted first (top): the top-left pixel should be the poster's fill colour, not white chart background
    top_left = composed_image.convert('RGB').getpixel((0, 0))
    assert top_left != (255, 255, 255)


def test_compose_with_poster_returns_chart_unchanged_when_no_poster():
    spec = ReportSpec(width_px=200, height_px=120, dpi=80)
    chart_png = render_chart_png([_flat_curve()], list(_rp1_filters()), spec=spec)
    assert compose_with_poster(chart_png, None) == chart_png


def test_render_report_headless_constructs_no_qapplication():
    '''
    Mirrors test_pipeline_qt_boundary.py's guarantee: MagnitudeModel is
    reused (and transitively imports qtpy via model.limits) but must never
    construct a QApplication.

    Deliberately a subprocess test (see test_pipeline_qt_boundary.py's
    docstring): a real Qt widget test elsewhere in the same pytest session
    (e.g. src/test/python/gui/) legitimately constructs a QApplication via
    pytest-qt's qapp fixture, which then persists for the rest of that
    process -- an in-process assert here would fail depending on test
    order/selection, not on anything this module actually did.
    '''
    import subprocess
    import sys

    script = (
        "from pipeline.publish.report import ReportSpec, render_report\n"
        "import numpy as np\n"
        "from model.xy import MagnitudeData\n"
        "from model.iir import CompleteFilter, LowShelf, PeakingEQ\n"
        "fs = 96000\n"
        "filters = list(CompleteFilter(fs=fs, filters=[LowShelf(fs, 18.0, 0.7, 4.5, count=5), "
        "PeakingEQ(fs, 40.0, 2.0, -3.0)]))\n"
        "x = np.geomspace(1, 160, 50)\n"
        "curve = MagnitudeData('Signal', None, x, np.zeros_like(x))\n"
        "render_report([curve], filters, spec=ReportSpec(width_px=200, height_px=120, dpi=80))\n"
        "from qtpy.QtWidgets import QApplication\n"
        "assert QApplication.instance() is None, 'render_report constructed a QApplication'\n"
        "print('OK')\n"
    )
    env = _env_without_display()
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=30, env=env)
    assert result.returncode == 0, f'stdout={result.stdout!r} stderr={result.stderr!r}'
    assert 'OK' in result.stdout


def _env_without_display():
    import os
    import pathlib
    env = dict(os.environ)
    for key in ('DISPLAY', 'WAYLAND_DISPLAY', 'QT_QPA_PLATFORM'):
        env.pop(key, None)
    src_main = str((pathlib.Path(__file__).parent / '..' / '..' / 'main' / 'python').resolve())
    env['PYTHONPATH'] = src_main
    return env


def test_pipeline_publish_report_module_has_no_qtpy_import():
    import ast
    import pathlib
    source = (pathlib.Path(__file__).parent / '..' / '..' / 'main' / 'python' / 'pipeline' / 'publish' / 'report.py').resolve()
    tree = ast.parse(source.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(n.name.startswith('qtpy') for n in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith('qtpy')
