'''
DesignRequest / DesignResponse / BiquadSpec, transcribed field-for-field from
design/designer-interface.md v1.0 §2/§3 -- the formal, standalone contract
for a pluggable filter designer. Any deviation from that document here is a
bug in this file, not a design decision made here; design/pipeline-
implementation-plan.md phase 2 (item 12).

design/designer-interface.md is the source of truth for what these fields
mean and why -- this module only defines their shapes.
'''
from dataclasses import dataclass
from typing import Literal, Optional

from numpy import ndarray

CONTRACT_VERSION = '1.0'

BiquadType = Literal['peaking_eq', 'low_shelf', 'high_shelf']
DesignMethod = Literal['exact', 'fitted', 'non_parametric']
ChannelScope = Literal['all_channels', 'lfe_only', 'mixed']
Coverage = Literal['complete_programme', 'excerpt']


@dataclass(frozen=True)
class DesignRequest:
    contract_version: str
    fs: int
    mono_mix: ndarray
    coverage: Coverage
    channels: Optional[dict] = None  # dict[str, ndarray], kept loose to avoid importing numpy typing extras


@dataclass(frozen=True)
class BiquadSpec:
    type: BiquadType
    freq_hz: float
    gain_db: float
    q: float


@dataclass(frozen=True)
class DesignCandidate:
    filters: list  # list[BiquadSpec]
    confidence: float
    mv_adjust_db: float
    method: DesignMethod

    # fit-quality -- error against an identified model's exact target for
    # 'exact'/'fitted', or against a constructed target curve for
    # 'non_parametric'; None only when there's genuinely no target to
    # measure against
    residual_db: Optional[float] = None
    residual_band_hz: Optional[tuple] = None  # (low_hz, high_hz)

    # structured, human-facing notes about this candidate -- dict[str, str],
    # rendered as-is by a report/GUI; not machine-parsed by the caller
    commentary: Optional[dict] = None

    # optional either way -- diagnostic only, never published
    fc_hz: Optional[float] = None
    slope: Optional[float] = None
    fc_uncertainty_hz: Optional[float] = None
    slope_uncertainty: Optional[float] = None
    channel_scope: Optional[ChannelScope] = None


@dataclass(frozen=True)
class DesignResponse:
    contract_version: str

    # success: a non-empty list[DesignCandidate], best (most preferred) first --
    # candidates[0] is the only one the caller acts on automatically; the rest
    # are carried through for a human reviewing the report. decline fields left None
    candidates: Optional[list] = None  # list[DesignCandidate]

    # decline: these two populated, candidates left None
    decline_reason: Optional[str] = None
    decline_message: Optional[str] = None


def build_request(mono_mix: ndarray, fs: int, coverage: Coverage = 'complete_programme',
                  channels: Optional[dict] = None) -> DesignRequest:
    ''' Convenience constructor that fills in contract_version. '''
    return DesignRequest(contract_version=CONTRACT_VERSION, fs=fs, mono_mix=mono_mix, coverage=coverage,
                         channels=channels)
