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
    if smoothing == 'SG':      # in place, on the same frequencies
        assert curves[0].data.x.size == 2000 and np.std(curves[0].data.y) < np.std(np.sin(np.linspace(1.0, 500.0, 2000)))
    elif smoothing is not None:   # onto 1/N-octave band centres, as the main window draws them
        assert curves[0].data.x.size < 2000 and np.array_equal(curves[0].data.x, curves[1].data.x)


def test_designs_out_of_range_are_left_out_and_an_unknown_smoothing_is_refused(entry):
    assert [c.candidate for c in workbench_curves(entry, [7, 2])] == [None, 2]
    with pytest.raises(ValueError, match='not a smoothing'):
        workbench_curves(entry, [], smoothing=5)


def test_the_response_is_the_same_whatever_the_smoothing(entry):
    plain = workbench_curves(entry, [0], None, responses=True)[-1].data
    smoothed = workbench_curves(entry, [0], 3, responses=True)[-1].data
    assert np.array_equal(plain.x, smoothed.x) and np.array_equal(plain.y, smoothed.y)
