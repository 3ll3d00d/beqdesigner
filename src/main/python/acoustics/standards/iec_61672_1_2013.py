import numpy as np

NOMINAL_FREQ = [10.0, 12.5, 16.0, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
                1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000]

NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES = np.array(NOMINAL_FREQ, dtype=np.float64)
"""Nominal 1/3-octave frequencies. See table 3.
"""

NOMINAL_OCTAVE_CENTER_FREQUENCIES = NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES[2::3]
"""Nominal 1/1-octave frequencies. Based on table 3.
"""
