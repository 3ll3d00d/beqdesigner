'''
FilterSpec + create_filter(): a plain, Qt-free replacement for the GUI's
widget-bound filter creation (model/filter.py's create_shaping_filter /
create_pass_filter, which read Qt widgets and dispatch on display strings
like 'Low Shelf') -- design/pipeline-implementation-plan.md phase 1 (B3).

Scope, deliberately trimmed for this pass: PeakingEQ, LowShelf, HighShelf and
Gain -- the shaping filters the BEQ workflow (docs/workflow/beq.md) and the
publish contract (designer-interface.md, D2) actually need. The GUI's pass
filters / All Pass / Linkwitz Transform are out of scope here; nothing about
this shape prevents adding them as further FilterSpec variants later, but
today's pipeline callers (Phase 2's designer contract, the manual FilterSpec
path in the API sketch) don't need them yet.
'''
from dataclasses import dataclass
from typing import Literal, Optional

FilterSpecType = Literal['peaking_eq', 'low_shelf', 'high_shelf', 'gain']


@dataclass(frozen=True)
class FilterSpec:
    '''
    :var type: one of FilterSpecType.
    :var freq: centre/corner frequency in Hz. Required for peaking_eq/low_shelf/high_shelf.
    :var gain: gain in dB. Required for all types (0.0 is a valid, if useless, value).
    :var q: required for peaking_eq; for low_shelf/high_shelf, either this or `s`
        must be given (not both) -- docs/workflow/beq.md documents BEQ shelves
        in S (slope/stacked count), while the hardware format wants Q, so
        forcing one convention on every caller would fight whichever method
        they're using. Use resolved_q()/resolved_s() to get both regardless
        of which was supplied.
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
    :return: the constructed Biquad (PeakingEQ, LowShelf, HighShelf or Gain).
    :raises ValueError: on an invalid or incomplete spec.
    '''
    from model.iir import PeakingEQ, LowShelf, HighShelf, Gain

    if spec.type == 'gain':
        return Gain(fs, spec.gain)

    if spec.q is not None and spec.s is not None:
        raise ValueError(f"{spec.type} spec gives both q and s -- supply exactly one")
    if spec.freq is None:
        raise ValueError(f"{spec.type} needs freq")

    if spec.type == 'peaking_eq':
        if spec.s is not None:
            raise ValueError("peaking_eq has no shelf slope -- use q")
        if spec.q is None:
            raise ValueError("peaking_eq needs q")
        return PeakingEQ(fs, spec.freq, spec.q, spec.gain)
    elif spec.type == 'low_shelf':
        return LowShelf(fs, spec.freq, spec.resolved_q(), spec.gain, spec.count)
    elif spec.type == 'high_shelf':
        return HighShelf(fs, spec.freq, spec.resolved_q(), spec.gain, spec.count)
    else:
        raise ValueError(f"Unknown filter type {spec.type}")
