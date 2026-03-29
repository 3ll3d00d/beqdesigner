from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.signal import unit_impulse

from model.signal import Signal

JRIVER_FS: int = 48000


def flatten(items: Iterable) -> Iterable:
    if isinstance(items, Iterable):
        for x in items:
            if isinstance(x, Iterable):
                yield from flatten(x)
            else:
                yield x
    else:
        yield items


class RoutingError(ValueError):
    pass


class ImpossibleRoutingError(RoutingError):

    def __init__(self, msg: str):
        super().__init__(msg)


class UnsupportedRoutingError(RoutingError):

    def __init__(self, msg: str):
        super().__init__(msg)


def s2f(val: any) -> float:
    return float(val.replace(",", ".")) if isinstance(val, str) else float(val)


def make_dirac_pulse(channel: str, analysis_resolution=1.0):
    fs = JRIVER_FS
    return Signal(channel, unit_impulse(int(fs / 2), 'mid'), fs=fs, analysis_resolution=analysis_resolution,
                  rescale_x=False)


def make_silence(channel: str):
    fs = JRIVER_FS
    return Signal(channel, np.zeros(int(fs / 2)), fs=fs)
