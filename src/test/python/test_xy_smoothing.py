'''
Fractional-octave smoothing (model.xy.smooth_octave), as the main window, the title page and the service's chart all
use it: a mean of power over the 1/N octave around each point, on the curve's own frequencies. It used to integrate
bands -- summing amplitudes onto band centres from 5 Hz -- which raised the level by tens of dB and dropped the bottom.
'''
import numpy as np
import pytest

from model.xy import MagnitudeData, smooth, smooth_octave

X = np.linspace(0, 500, 32769)   # a Welch grid at 1 kHz, as a signal's curves have


@pytest.mark.parametrize('fraction', [1, 2, 3, 6, 12, 24, 48])
def test_a_flat_spectrum_keeps_its_level_and_its_frequencies(fraction):
    x, y = smooth_octave(X, np.full_like(X, -20.0), fraction)
    assert np.array_equal(x, X)
    assert np.allclose(y, -20.0)


def test_a_single_loud_point_spreads_out_and_comes_down():
    y = np.full_like(X, -60.0)
    peak = np.searchsorted(X, 30.0)
    y[peak] = 0.0
    _, third = smooth_octave(X, y, 3)
    _, twelfth = smooth_octave(X, y, 12)
    assert third[peak] < -10 and third.max() <= 0.0 and third[peak - 10] > -60
    assert twelfth[peak] > third[peak]                       # a finer fraction averages over less


def test_the_mean_is_of_power_not_of_decibels():
    y = np.where(np.arange(X.size) % 2 == 0, 0.0, -100.0)   # alternating 0 dB and almost nothing
    _, smoothed = smooth_octave(X, y, 1)
    middle = np.searchsorted(X, 100.0)
    assert smoothed[middle] == pytest.approx(10 * np.log10(0.5), abs=0.1)   # -3 dB, not -50


def test_the_bottom_of_the_curve_is_kept_and_a_point_alone_in_its_window_is_unchanged():
    y = 3.0 * np.sin(X)
    x, smoothed = smooth_octave(X, y, 48)
    assert x[1] < 5 and smoothed[0] == y[0] and smoothed[1] == y[1]   # 0 Hz, and a point with no neighbour that close


def test_an_uneven_grid_is_averaged_by_the_span_each_point_stands_for():
    x = np.array([10.0, 10.1, 10.2, 10.3, 12.0])             # crowded at 10 Hz, one far point
    y = np.array([0.0, 0.0, 0.0, 0.0, -30.0])
    _, smoothed = smooth_octave(x, y, 1)
    unweighted = 10 * np.log10((4 + 10 ** -3) / 5)
    assert smoothed[2] < unweighted                           # the far point counts for the span it covers


def test_magnitude_data_and_the_main_windows_choices_use_it():
    data = MagnitudeData('avg', '', X, np.full_like(X, -20.0), smooth_type=6)
    assert np.array_equal(data.x, X) and np.allclose(data.y, -20.0)
    assert smooth(X, np.full_like(X, -20.0))[1][0] == -20.0   # no smoothing
    with pytest.raises(ValueError):
        smooth_octave(X, X, 0)
