'''
Safety net for model/filter.py's FilterDialog.create_shaping_filter()/
create_pass_filter() -- written BEFORE rewiring the shaping-filter branches
onto pipeline.filters.FilterSpec/create_filter().

Constructs the real dialog (small=True to skip the live preview chart,
which needs nothing this test cares about) and drives its actual widgets.
'''
import math

import pytest
from qtpy.QtCore import QSettings

import ui.beq  # noqa: F401 -- must be imported before model.filter (-> ui.filter -> app): ui/beq.py's
                # mid-file `from app import PlotWidgetWithDateAxis` only resolves if ui.beq's own
                # Ui_MainWindow (defined earlier in the same file) is already cached in sys.modules
                # by the time app.py's top-level `from ui.beq import Ui_MainWindow` runs to satisfy it --
                # true whichever of ui.beq/app starts the cycle first, but not if neither has run yet.
from model.filter import FilterDialog, FilterModel
from model.iir import AllPass, CompleteFilter, Gain, HighShelf, LinkwitzTransform, LowShelf, PeakingEQ, \
    SecondOrder_HighPass, SecondOrder_LowPass
from model.preferences import Preferences
from model.signal import Signal


def _make_preferences(tmp_path):
    settings = QSettings(str(tmp_path / 'settings.ini'), QSettings.Format.IniFormat)
    return Preferences(settings)


def _make_dialog(qtbot, tmp_path, valid_filter_types=None):
    prefs = _make_preferences(tmp_path)
    signal = Signal('test', __import__('numpy').zeros(1000), fs=1000)
    filter_model = FilterModel(None, prefs)
    filter_model.filter = CompleteFilter(fs=1000)
    dialog = FilterDialog(prefs, signal, filter_model, lambda: None, small=True,
                          valid_filter_types=valid_filter_types)
    qtbot.addWidget(dialog)
    return dialog


def _select(dialog, filter_type):
    idx = dialog.filterType.findText(filter_type)
    assert idx >= 0, f"{filter_type} not found in filterType combo"
    dialog.filterType.setCurrentIndex(idx)


def test_create_shaping_filter_low_shelf(qtbot, tmp_path):
    dialog = _make_dialog(qtbot, tmp_path)
    _select(dialog, 'Low Shelf')
    dialog.freq.setValue(18.0)
    dialog.filterGain.setValue(4.5)
    dialog.filterQ.setValue(0.7)
    dialog.filterCount.setValue(5)

    filt = dialog.create_shaping_filter()

    assert isinstance(filt, LowShelf)
    assert filt.freq == 18.0
    assert filt.gain == 4.5
    assert math.isclose(filt.q, 0.7)
    assert filt.count == 5


def test_create_shaping_filter_high_shelf(qtbot, tmp_path):
    dialog = _make_dialog(qtbot, tmp_path)
    _select(dialog, 'High Shelf')
    dialog.freq.setValue(80.0)
    dialog.filterGain.setValue(-2.0)
    dialog.filterQ.setValue(1.1)
    dialog.filterCount.setValue(2)

    filt = dialog.create_shaping_filter()

    assert isinstance(filt, HighShelf)
    assert filt.freq == 80.0 and filt.gain == -2.0 and filt.count == 2


def test_create_shaping_filter_peq(qtbot, tmp_path):
    dialog = _make_dialog(qtbot, tmp_path)
    _select(dialog, 'PEQ')
    dialog.freq.setValue(40.0)
    dialog.filterGain.setValue(-3.0)
    dialog.filterQ.setValue(2.0)

    filt = dialog.create_shaping_filter()

    assert isinstance(filt, PeakingEQ)
    assert filt.freq == 40.0 and filt.gain == -3.0 and math.isclose(filt.q, 2.0)


def test_create_shaping_filter_gain(qtbot, tmp_path):
    dialog = _make_dialog(qtbot, tmp_path)
    _select(dialog, 'Gain')
    dialog.filterGain.setValue(6.0)

    filt = dialog.create_shaping_filter()

    assert isinstance(filt, Gain)
    assert filt.gain == 6.0


def test_create_shaping_filter_variable_q_lpf(qtbot, tmp_path):
    dialog = _make_dialog(qtbot, tmp_path)
    _select(dialog, 'Variable Q LPF')
    dialog.freq.setValue(120.0)
    dialog.filterQ.setValue(0.9)

    filt = dialog.create_shaping_filter()

    assert isinstance(filt, SecondOrder_LowPass)
    assert filt.freq == 120.0 and math.isclose(filt.q, 0.9)


def test_create_shaping_filter_variable_q_hpf(qtbot, tmp_path):
    dialog = _make_dialog(qtbot, tmp_path)
    _select(dialog, 'Variable Q HPF')
    dialog.freq.setValue(30.0)
    dialog.filterQ.setValue(0.6)

    filt = dialog.create_shaping_filter()

    assert isinstance(filt, SecondOrder_HighPass)
    assert filt.freq == 30.0 and math.isclose(filt.q, 0.6)


def test_create_shaping_filter_all_pass(qtbot, tmp_path):
    dialog = _make_dialog(qtbot, tmp_path)
    _select(dialog, 'All Pass')
    dialog.freq.setValue(50.0)
    dialog.filterQ.setValue(1.5)

    filt = dialog.create_shaping_filter()

    assert isinstance(filt, AllPass)
    assert filt.freq == 50.0 and math.isclose(filt.q, 1.5)


def test_create_shaping_filter_linkwitz_transform_unaffected(qtbot, tmp_path):
    ''' out of scope for the pipeline.filters rewire -- different parameter shape (f0/q0/fp/qp). '''
    dialog = _make_dialog(qtbot, tmp_path)
    _select(dialog, 'Linkwitz Transform')
    dialog.f0.setValue(20.0)
    dialog.q0.setValue(0.5)
    dialog.fp.setValue(15.0)
    dialog.qp.setValue(0.7)

    filt = dialog.create_shaping_filter()

    assert isinstance(filt, LinkwitzTransform)


def test_create_shaping_filter_snaps_sqrt2_q_value(qtbot, tmp_path):
    '''
    filterQ is a spinbox with limited decimal precision; a value close to
    1/sqrt(2) at that precision gets snapped to the precise constant rather
    than the rounded display value -- preserved exactly by the rewire.
    '''
    dialog = _make_dialog(qtbot, tmp_path)
    _select(dialog, 'PEQ')
    dialog.freq.setValue(100.0)
    dialog.filterGain.setValue(1.0)
    sqrt2 = 1.0 / (2.0 ** 0.5)
    dialog.filterQ.setValue(round(sqrt2, dialog.filterQ.decimals()))

    filt = dialog.create_shaping_filter()

    assert filt.q == sqrt2
