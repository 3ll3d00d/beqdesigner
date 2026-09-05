'''
FilterSpec + create_filter(): a plain, Qt-free replacement for the GUI's
widget-bound filter creation (model/filter.py's create_shaping_filter /
create_pass_filter, which read Qt widgets and dispatch on display strings
like 'Low Shelf') -- design/pipeline-implementation-plan.md phase 1 (B3).
model/filter.py's create_shaping_filter() was rewired onto this module
directly (its GUI-rewiring follow-up); the FilterSpecType set below is
exactly the set that dialog needs for its shaping-filter branches.

Scope, deliberately still trimmed: Linkwitz Transform and the compound
Low/High Pass filters (model/filter.py's create_pass_filter(), built from
ComplexLowPass/ComplexHighPass -- multiple cascaded biquads synthesising a
Butterworth/Linkwitz-Riley response, not a single Biquad) don't fit this
module's one-spec-one-Biquad shape at all: Linkwitz Transform takes four
independent frequency/Q pairs (f0/q0/fp/qp), nothing like freq/gain/q/s/
count, and a compound pass filter is a filter *set*, not a filter. Both
stay as direct model/iir.py construction in the dialog.
'''
from dataclasses import dataclass
from typing import Literal, Optional

FilterSpecType = Literal['peaking_eq', 'low_shelf', 'high_shelf', 'gain', 'variable_q_lpf', 'variable_q_hpf',
                         'all_pass']


@dataclass(frozen=True)
class FilterSpec:
    '''
    :var type: one of FilterSpecType.
    :var freq: centre/corner frequency in Hz. Required for every type except gain.
    :var gain: gain in dB. Meaningful for peaking_eq/low_shelf/high_shelf/gain only
        (0.0 is a valid, if useless, value); ignored by variable_q_lpf/variable_q_hpf/all_pass,
        which have no gain concept.
    :var q: required for peaking_eq/variable_q_lpf/variable_q_hpf/all_pass; for
        low_shelf/high_shelf, either this or `s` must be given (not both) --
        docs/workflow/beq.md documents BEQ shelves in S (slope/stacked count),
        while the hardware format wants Q, so forcing one convention on every
        caller would fight whichever method they're using. Use
        resolved_q()/resolved_s() to get both regardless of which was supplied.
    :var s: shelf slope, alternative to `q`. low_shelf/high_shelf only.
    :var count: stacked shelf repeat count. low_shelf/high_shelf only, default 1.
    '''
    type: FilterSpecType
    freq: Optional[float] = None
    gain: float = 0.0
    q: Optional[float] = None
    s: Optional[float] = None
    count: int = 1

    def resolved_q(self) -> float:
        ''' :return: the Q, converting from `s` if that's what was supplied. '''
        if self.q is not None:
            return self.q
        if self.s is not None:
            from model.iir import s_to_q
            return s_to_q(self.s, self.gain)
        raise ValueError(f"{self.type} needs either q or s")

    def resolved_s(self) -> float:
        ''' :return: the shelf slope S, converting from `q` if that's what was supplied. '''
        if self.s is not None:
            return self.s
        if self.q is not None:
            from model.iir import q_to_s
            return q_to_s(self.q, self.gain)
        raise ValueError(f"{self.type} needs either q or s")


def create_filter(spec: FilterSpec, fs: int):
    '''
    :param spec: the filter spec.
    :param fs: the sample rate to construct the filter at.
    :return: the constructed Biquad (PeakingEQ, LowShelf, HighShelf, Gain,
        SecondOrder_LowPass, SecondOrder_HighPass or AllPass).
    :raises ValueError: on an invalid or incomplete spec.
    '''
    from model.iir import PeakingEQ, LowShelf, HighShelf, Gain, SecondOrder_LowPass, SecondOrder_HighPass, AllPass

    if spec.type == 'gain':
        return Gain(fs, spec.gain)

    if spec.q is not None and spec.s is not None:
        raise ValueError(f"{spec.type} spec gives both q and s -- supply exactly one")
    if spec.freq is None:
        raise ValueError(f"{spec.type} needs freq")

    if spec.type in ('peaking_eq', 'variable_q_lpf', 'variable_q_hpf', 'all_pass'):
        if spec.s is not None:
            raise ValueError(f"{spec.type} has no shelf slope -- use q")
        if spec.q is None:
            raise ValueError(f"{spec.type} needs q")

    if spec.type == 'peaking_eq':
        return PeakingEQ(fs, spec.freq, spec.q, spec.gain)
    elif spec.type == 'low_shelf':
        return LowShelf(fs, spec.freq, spec.resolved_q(), spec.gain, spec.count)
    elif spec.type == 'high_shelf':
        return HighShelf(fs, spec.freq, spec.resolved_q(), spec.gain, spec.count)
    elif spec.type == 'variable_q_lpf':
        return SecondOrder_LowPass(fs, spec.freq, q=spec.q)
    elif spec.type == 'variable_q_hpf':
        return SecondOrder_HighPass(fs, spec.freq, q=spec.q)
    elif spec.type == 'all_pass':
        return AllPass(fs, spec.freq, spec.q)
    else:
        raise ValueError(f"Unknown filter type {spec.type}")
