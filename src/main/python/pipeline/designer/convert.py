'''
Validates a DesignResponse against design/designer-interface.md §3-§5 and
converts a validated one into this repo's own CompleteFilter -- design/
pipeline-implementation-plan.md phase 2 (item 12). This is the contract
boundary: nothing downstream of to_complete_filter() should need to know
whether the CompleteFilter it received came from a registered designer or
was built by hand via pipeline.filters.

Deliberately independent of any particular designer implementation -- see
test_pipeline_designer_contract.py, which exercises every rejection path
below using a fake in-process designer, not a real one.
'''
import math
from typing import List

from pipeline.designer.contract import BiquadSpec, DesignResponse
from pipeline.filters import FilterSpec, create_filter

ALLOWED_BIQUAD_TYPES = {'peaking_eq', 'low_shelf', 'high_shelf'}
MAX_BIQUAD_SECTIONS = 10


class ContractViolation(ValueError):
    ''' Raised when a DesignResponse violates design/designer-interface.md §3-§5. '''


def validate_response(response: DesignResponse) -> None:
    '''
    :raises ContractViolation: if response violates the contract.
    '''
    success_fields = {
        'filters': response.filters,
        'confidence': response.confidence,
        'mv_adjust_db': response.mv_adjust_db,
        'method': response.method,
        'residual_db': response.residual_db,
        'residual_band_hz': response.residual_band_hz,
    }
    is_decline = response.decline_reason is not None
    is_success = response.filters is not None

    if is_success and is_decline:
        raise ContractViolation("DesignResponse populates both filters and decline_reason -- exactly one is allowed")
    if not is_success and not is_decline:
        raise ContractViolation("DesignResponse populates neither filters nor decline_reason")

    if is_decline:
        if not isinstance(response.decline_reason, str) or len(response.decline_reason) == 0:
            raise ContractViolation("decline_reason must be a non-empty string")
        populated = [name for name, value in success_fields.items() if name != 'filters' and value is not None]
        if populated:
            raise ContractViolation(
                f"decline response also populates success-path field(s): {', '.join(populated)}")
        return

    # success path
    if response.confidence is None:
        raise ContractViolation("success response has no confidence")
    if not (0.0 <= response.confidence <= 1.0):
        raise ContractViolation(f"confidence must be in [0.0, 1.0], got {response.confidence}")
    if response.mv_adjust_db is None or not math.isfinite(response.mv_adjust_db):
        raise ContractViolation(f"mv_adjust_db must be a finite number, got {response.mv_adjust_db}")

    if len(response.filters) == 0:
        raise ContractViolation("success response has an empty filters list -- decline instead of an empty list")
    if len(response.filters) > MAX_BIQUAD_SECTIONS:
        raise ContractViolation(
            f"{len(response.filters)} biquad sections exceeds the budget of {MAX_BIQUAD_SECTIONS}")

    for i, spec in enumerate(response.filters):
        _validate_biquad_spec(spec, i)

    if response.method == 'non_parametric' and (response.residual_db is not None
                                                 or response.residual_band_hz is not None):
        raise ContractViolation("method='non_parametric' has no exact target -- residual_db/residual_band_hz "
                                "must be None")


def _validate_biquad_spec(spec: BiquadSpec, index: int) -> None:
    if spec.type not in ALLOWED_BIQUAD_TYPES:
        raise ContractViolation(
            f"filters[{index}].type '{spec.type}' is not publishable -- must be one of {sorted(ALLOWED_BIQUAD_TYPES)}")
    if not math.isfinite(spec.freq_hz) or spec.freq_hz <= 0:
        raise ContractViolation(f"filters[{index}].freq_hz must be > 0, got {spec.freq_hz}")
    if not math.isfinite(spec.gain_db):
        raise ContractViolation(f"filters[{index}].gain_db must be finite, got {spec.gain_db}")
    if not math.isfinite(spec.q) or spec.q <= 0:
        raise ContractViolation(f"filters[{index}].q must be > 0, got {spec.q}")


_TYPE_MAP = {'peaking_eq': 'peaking_eq', 'low_shelf': 'low_shelf', 'high_shelf': 'high_shelf'}


def to_complete_filter(response: DesignResponse, fs: int):
    '''
    Validates response, then converts its filters into a CompleteFilter at
    the given (publish-target) fs. §5: a BiquadSpec carries no fs and no
    count -- repeated identical entries are collapsed into a single stacked
    shelf here, since that representation is this repo's concern, not the
    designer's.
    :raises ContractViolation: if response is invalid.
    '''
    from model.iir import CompleteFilter

    validate_response(response)
    if response.decline_reason is not None:
        raise ContractViolation("cannot convert a declined response -- check decline_reason first")

    biquads = [create_filter(spec, fs) for spec in _collapse(response.filters)]
    return CompleteFilter(fs=fs, filters=biquads, description='designer')


def _collapse(specs: List[BiquadSpec]) -> List[FilterSpec]:
    '''
    Groups identical (type, freq_hz, gain_db, q) BiquadSpecs -- order
    preserved by first occurrence -- into a single FilterSpec with `count`
    set (shelves only; peaking_eq has no stacking concept in this codebase,
    so repeats stay as separate FilterSpecs).
    '''
    counts = {}
    order = []
    for spec in specs:
        key = (spec.type, spec.freq_hz, spec.gain_db, spec.q)
        if key not in counts:
            order.append(key)
            counts[key] = 0
        counts[key] += 1

    result = []
    for key in order:
        filt_type, freq_hz, gain_db, q = key
        count = counts[key]
        if filt_type == 'peaking_eq':
            result.extend(FilterSpec(type='peaking_eq', freq=freq_hz, gain=gain_db, q=q) for _ in range(count))
        else:
            result.append(FilterSpec(type=filt_type, freq=freq_hz, gain=gain_db, q=q, count=count))
    return result
