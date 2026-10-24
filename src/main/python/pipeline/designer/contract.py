'''
DesignRequest / DesignResponse / BiquadSpec, transcribed field-for-field from
design/designer-interface.md v1.5 §2/§3 -- the formal, standalone contract
for a pluggable filter designer. Any deviation from that document here is a
bug in this file, not a design decision made here; design/pipeline-
implementation-plan.md phase 2 (item 12).

design/designer-interface.md is the source of truth for what these fields
mean and why -- this module only defines their shapes.
'''
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

from numpy import ndarray

CONTRACT_VERSION = '1.1'
ADJUSTMENTS_VERSION = '1.5'   # what a request carrying adjustments says it is (§7)

# 1.5: the adjustment kinds §2 defines, and the outcomes a designer answers each with (§3)
ADJUSTMENT_KINDS = ('exclude_band', 'floor', 'max_boost', 'goal_tilt', 'note')
OUTCOMES = ('honoured', 'adapted', 'declined', 'not_understood')

BiquadType = Literal['peaking_eq', 'low_shelf', 'high_shelf']
DesignMethod = Literal['exact', 'fitted', 'non_parametric']
ChannelScope = Literal['all_channels', 'lfe_only', 'mixed']
ResidualDirection = Literal['over', 'under']

# 1.3: the fields that describe how well a candidate's filters match their target, beside residual_db/residual_band_hz
FIT_FIELDS = ('residual_at_hz', 'residual_direction', 'residual_rms_db', 'residual_within_tolerance',
              'residual_tolerance_db')
Coverage = Literal['complete_programme', 'excerpt']


@dataclass(frozen=True)
class DesignRequest:
    contract_version: str
    fs: int
    mono_mix: ndarray
    coverage: Coverage
    channels: Optional[dict] = None  # dict[str, ndarray], kept loose to avoid importing numpy typing extras

    # the caller's bass-management configuration, if it has one to report --
    # None when there's no bass management (or it hasn't been decided) --
    # {'lpf_fs': float, 'lpf_position': 'Before'|'After'|'Off',
    #  'headroom_type': 'WCS' | numeric-string-dB, 'clip_before': bool, 'clip_after': bool}
    bass_management: Optional[dict] = None
    adjustments: Optional[tuple] = None   # 1.5: tuple[DesignAdjustment], None or empty for none


@dataclass(frozen=True)
class DesignAdjustment:
    ''' 1.5 (§2, "Adjustments"): a change the caller asks for; the designer answers it in an AdjustmentOutcome. '''
    id: str
    kind: str
    params: dict
    text: Optional[str] = None


@dataclass(frozen=True)
class AdjustmentOutcome:
    ''' 1.5 (§3, "Adjustment outcomes"): what the designer did with one adjustment. '''
    id: str
    outcome: str        # one of OUTCOMES
    applied: Optional[dict] = None
    reason: Optional[str] = None


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
    mv_adjust_db: float  # NOT a clipping-cost estimate -- see gain_reduction_db below
    method: DesignMethod

    # the actual publication-blocking cost: min(20*log10(1/peak), 0) of the filtered
    # sub feed, i.e. <= 0 always, 0 meaning no reduction needed. None if not computed.
    gain_reduction_db: Optional[float] = None

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

    # 1.1: why this design was judged unfit to publish -- non-empty on every entry of DesignResponse.rejected, None on
    # every entry of candidates. Human-facing, never machine-parsed
    rejection_reasons: Optional[list] = None  # list[str]

    # 1.3: how well `filters` match the target they were fitted to (never whether the target is right), each None
    # whenever residual_db is. residual_at_hz/residual_direction: where |error| = residual_db, and whether the filters
    # boost more ('over') or less ('under') than the target there. residual_rms_db: RMS error over residual_band_hz
    # weighted by the boost the target asks for (max(target_db, 0)); None when it asks for none. residual_within_
    # tolerance: the share of that requested boost within residual_tolerance_db of the target, in [0, 1]
    residual_at_hz: Optional[float] = None
    residual_direction: Optional[ResidualDirection] = None
    residual_rms_db: Optional[float] = None
    residual_within_tolerance: Optional[float] = None
    residual_tolerance_db: Optional[float] = None


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

    # 1.1, with either shape: designs built and judged unfit to publish, for a human to review only -- never applied or
    # published by the caller on its own. None when there is nothing to show; never an empty list
    rejected: Optional[list] = None  # list[DesignCandidate]

    # 1.4, with either shape: the designer's identity as far as the answer is concerned (design/design-version.md)
    design_version: Optional[int] = None   # bumped by any change that can alter an answer to the same request
    design_settings: Optional[str] = None  # digest of the designer's own settings that can alter an answer
    designer_build: Optional[str] = None   # the build, for people
    adjustment_outcomes: Optional[list] = None   # 1.5: list[AdjustmentOutcome], one per request adjustment
    # 1.6, with either shape: the designer's record of this run, relative to its shared root (§3, "The run record"); the
    # caller keeps it and sends it back to /impact unchanged. None when the designer has no shared root or said none
    run_record: Optional[str] = None


IDENTITY_FIELDS = ('design_version', 'design_settings', 'designer_build')


def build_request(mono_mix: ndarray, fs: int, coverage: Coverage = 'complete_programme',
                  channels: Optional[dict] = None, bass_management: Optional[dict] = None,
                  adjustments: Optional[Sequence[DesignAdjustment]] = None) -> DesignRequest:
    '''
    Convenience constructor that enforces the sample-array part of the designer contract, and that adjustments (1.5) have
    an id and a kind each, ids unique. With adjustments, the request says 1.5.
    '''
    adjustments = tuple(adjustments or ())
    ids = [a.id for a in adjustments]
    if any(not isinstance(a.id, str) or not a.id or not isinstance(a.kind, str) or not a.kind
           or not isinstance(a.params, dict) for a in adjustments):
        raise ValueError('each adjustment needs an id, a kind and params (a dict)')
    if len(set(ids)) != len(ids):
        raise ValueError(f'adjustment ids must be unique: {ids}')
    if getattr(mono_mix, 'ndim', None) != 1:
        raise ValueError('mono_mix must be a one-dimensional sample array')
    if channels is not None:
        expected = len(mono_mix)
        invalid = []
        for name, samples in channels.items():
            dimensions = getattr(samples, 'ndim', None)
            if dimensions != 1:
                invalid.append(f'{name}: expected a one-dimensional array, got {dimensions}-D')
            elif len(samples) != expected:
                invalid.append(f'{name}: {len(samples):,} samples (mono_mix has {expected:,})')
        if invalid:
            raise ValueError('channels must be one-dimensional and match mono_mix length: ' + '; '.join(invalid))
    return DesignRequest(contract_version=ADJUSTMENTS_VERSION if adjustments else CONTRACT_VERSION, fs=fs,
                         mono_mix=mono_mix, coverage=coverage, channels=channels, bass_management=bass_management,
                         adjustments=adjustments or None)
