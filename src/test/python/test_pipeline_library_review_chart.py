'''
pipeline/library/review_chart.py's workbench curves (design/web-parity.md WP1): several designs at once, their own responses,
and the main window's smoothing.
'''
from dataclasses import replace

import numpy as np
import pytest

from model.codec import xydata_to_json
from model.xy import MagnitudeData
from pipeline.library.review_chart import SMOOTHING, workbench_curves
from pipeline.review import read_entry
from review_entry_fixture import write_entry


@pytest.fixture
def entry(tmp_path):
    write_entry(str(tmp_path), 'a', rejected=1)
    x = np.linspace(1.0, 500.0, 2000)
    dense = xydata_to_json(MagnitudeData('avg', '', x, np.sin(x)))
    return replace(read_entry(str(tmp_path), 'a'), curve=dense, peak_curve=None)


@pytest.mark.parametrize('smoothing', SMOOTHING)
def test_every_smoothing_the_main_window_offers_draws_a_dense_curve(entry, smoothing):
    curves = workbench_curves(entry, [0], smoothing, responses=True)
    assert [(c.kind, c.filtered, c.candidate) for c in curves] == \
        [('average', False, None), ('average', True, 0), ('filter', True, 0)]
    if smoothing is not None:   # in place, on the same frequencies, and calmer (model.xy.smooth_octave, smooth_savgol)
        assert curves[0].data.x.size == 2000
        assert np.std(curves[0].data.y) < np.std(np.sin(np.linspace(1.0, 500.0, 2000)))
        assert abs(np.mean(curves[0].data.y)) < 3     # and not lifted: the old band integration added tens of dB


def test_designs_out_of_range_are_left_out_and_an_unknown_smoothing_is_refused(entry):
    assert [c.candidate for c in workbench_curves(entry, [7, 2])] == [None, 2]
    with pytest.raises(ValueError, match='not a smoothing'):
        workbench_curves(entry, [], smoothing=5)


def test_the_response_is_the_same_whatever_the_smoothing(entry):
    plain = workbench_curves(entry, [0], None, responses=True)[-1].data
    smoothed = workbench_curves(entry, [0], 3, responses=True)[-1].data
    assert np.array_equal(plain.x, smoothed.x) and np.array_equal(plain.y, smoothed.y)


def test_a_published_beq_is_applied_at_the_tracks_own_frequencies(tmp_path):
    ''' A 1 kHz extraction's curve and a 96 kHz catalogue filter have as many points (32769) over different spans: each
    published BEQ must lift the track by its own response at each frequency, not by its response at another one. '''
    from pipeline.library.review_chart import published_curves
    write_entry(str(tmp_path), 'b')
    x = np.linspace(0.0, 500.0, 32769)
    entry = replace(read_entry(str(tmp_path), 'b'), curve=xydata_to_json(MagnitudeData('avg', '', x, np.full(x.size, -60.0))),
                    peak_curve=None)
    under = {}
    for gain in (10.0, 2.0):
        response, average = published_curves(entry, [{'type': 'LowShelf', 'freq': 30, 'q': 0.707, 'gain': gain}], 'a')
        assert response.data.x.size == average.data.x.size == 32769 and response.data.x[-1] > 47000
        at = int(np.argmin(np.abs(average.data.x - 20.0)))
        assert average.data.x[-1] == 500.0
        lift = np.interp(20.0, response.data.x, response.data.y)
        assert average.data.y[at] == pytest.approx(-60.0 + lift, abs=0.05) and lift > gain * 0.8
        under[gain] = average.data.y[at]
    assert under[10.0] - under[2.0] > 6   # two authors' BEQs no longer draw the same curve
