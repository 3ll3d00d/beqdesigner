'''
Peak/RMS/crest/headroom, extracted from three near-identical inline copies
(model/waveform.py's __recalc_stats, model/analysis.py's signal setter,
model/extract.py's __calc_headroom) into a single Qt-free implementation --
design/pipeline-implementation-plan.md phase 1 (B5).

Faithfully replicates the original arithmetic, including its behaviour on a
silent (all-zero) signal: peak=0 makes 1.0/peak -> inf and 0/0 -> nan, both
of which flow through math.log unchanged (no ZeroDivisionError, no
exception) rather than being special-cased here -- the goal is identical
numbers to what the GUI already shows, not different behaviour for the same
input.
'''
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Stats:
    peak: float
    rms: float
    crest: float
    headroom: float
    fs: int


def signal_stats(samples: np.ndarray, fs: int) -> Stats:
    '''
    :param samples: the raw sample values (typically in -1.0..1.0 full-scale units).
    :param fs: the sample rate `samples` was measured at -- carried on the
        result purely as provenance (D1 in design/api-headless-pipeline.md),
        not used in the calculation itself, so a later consumer (e.g. a
        published beq_gain) is never ambiguous about what it was measured
        against.
    :return: peak, rms, crest factor and headroom, all in dB except peak
        (linear, full-scale = 1.0).
    '''
    peak_value = np.nanmax(np.abs(samples))
    rms_level_raw = np.sqrt(np.mean(np.square(np.abs(samples))))
    crest = 20 * math.log(peak_value / rms_level_raw, 10)
    rms = 20 * math.log(rms_level_raw, 10)
    headroom = 20 * math.log(1.0 / peak_value, 10)
    return Stats(peak=float(peak_value), rms=float(rms), crest=float(crest), headroom=float(headroom), fs=fs)
