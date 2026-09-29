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

from pipeline.designer.contract import BiquadSpec, DesignCandidate, DesignResponse
from pipeline.filters import FilterSpec, create_filter

ALLOWED_BIQUAD_TYPES = {'peaking_eq', 'low_shelf', 'high_shelf'}
MAX_BIQUAD_SECTIONS = 10


class ContractViolation(ValueError):
    ''' Raised when a DesignResponse violates design/designer-interface.md §3-§5. '''


def validate_response(response: DesignResponse) -> None:
    '''
    :raises ContractViolation: if response violates the contract -- in `rejected` (1.1) as much as in `candidates`.
    '''
    is_decline = response.decline_reason is not None
    is_success = response.candidates is not None

    if is_success and is_decline:
        raise ContractViolation("DesignResponse populates both candidates and decline_reason -- exactly one is allowed")
    if not is_success and not is_decline:
        raise ContractViolation("DesignResponse populates neither candidates nor decline_reason")

    # 1.1: either shape may carry the designs that were judged unfit to publish; every one a full, loadable candidate
    # with reasons, in no particular order (confidence measures evidence, not whether a design passed)
    if response.rejected is not None:
        if not isinstance(response.rejected, list) or len(response.rejected) == 0:
            raise ContractViolation("rejected must be a non-empty list, or None when there is nothing to show")
        for i, candidate in enumerate(response.rejected):
            _validate_candidate(candidate, i, 'rejected')
            reasons = candidate.rejection_reasons
            if not isinstance(reasons, list) or not reasons or \
                    not all(isinstance(reason, str) and reason.strip() for reason in reasons):
                raise ContractViolation(f"rejected[{i}].rejection_reasons must be a non-empty list of non-empty strings")

    if is_decline:
        if not isinstance(response.decline_reason, str) or len(response.decline_reason) == 0:
            raise ContractViolation("decline_reason must be a non-empty string")
        return

    # success path
    if len(response.candidates) == 0:
        raise ContractViolation("success response has an empty candidates list -- decline instead of an empty list")

    previous_confidence = None
    for i, candidate in enumerate(response.candidates):
        _validate_candidate(candidate, i)
        if candidate.rejection_reasons is not None:
            raise ContractViolation(
                f"candidates[{i}] has rejection_reasons -- a design with reasons against it belongs in rejected")
        if previous_confidence is not None and candidate.confidence > previous_confidence:
            raise ContractViolation(
                f"candidates[{i}].confidence ({candidate.confidence}) exceeds candidates[{i - 1}]'s "
                f"({previous_confidence}) -- candidates must be ordered best (most preferred) first")
        previous_confidence = candidate.confidence


def _validate_candidate(candidate: DesignCandidate, index: int, where: str = 'candidates') -> None:
    name = f"{where}[{index}]"
    if candidate.confidence is None:
        raise ContractViolation(f"{name} has no confidence")
    if not (0.0 <= candidate.confidence <= 1.0):
        raise ContractViolation(f"{name}.confidence must be in [0.0, 1.0], got {candidate.confidence}")
    if candidate.mv_adjust_db is None or not math.isfinite(candidate.mv_adjust_db):
        raise ContractViolation(f"{name}.mv_adjust_db must be a finite number, got {candidate.mv_adjust_db}")

    if candidate.gain_reduction_db is not None:
        if not math.isfinite(candidate.gain_reduction_db) or candidate.gain_reduction_db > 0:
            raise ContractViolation(
                f"{name}.gain_reduction_db must be a finite number <= 0 (0 = no reduction needed), "
                f"got {candidate.gain_reduction_db}")

    if candidate.filters is None or len(candidate.filters) == 0:
        raise ContractViolation(f"{name} has an empty filters list -- omit the candidate instead")
    if len(candidate.filters) > MAX_BIQUAD_SECTIONS:
        raise ContractViolation(
            f"{name}: {len(candidate.filters)} biquad sections exceeds the budget of {MAX_BIQUAD_SECTIONS}")

    for i, spec in enumerate(candidate.filters):
        _validate_biquad_spec(spec, f"{name}.filters[{i}]")

    if candidate.commentary is not None:
        if not isinstance(candidate.commentary, dict):
            raise ContractViolation(f"{name}.commentary must be a dict[str, str]")
        for key, value in candidate.commentary.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ContractViolation(
                    f"{name}.commentary must be a dict[str, str], got key={key!r} value={value!r}")


def _validate_biquad_spec(spec: BiquadSpec, prefix: str) -> None:
    if spec.type not in ALLOWED_BIQUAD_TYPES:
        raise ContractViolation(
            f"{prefix}.type '{spec.type}' is not publishable -- must be one of {sorted(ALLOWED_BIQUAD_TYPES)}")
    if not math.isfinite(spec.freq_hz) or spec.freq_hz <= 0:
        raise ContractViolation(f"{prefix}.freq_hz must be > 0, got {spec.freq_hz}")
    if not math.isfinite(spec.gain_db):
        raise ContractViolation(f"{prefix}.gain_db must be finite, got {spec.gain_db}")
    if not math.isfinite(spec.q) or spec.q <= 0:
        raise ContractViolation(f"{prefix}.q must be > 0, got {spec.q}")


_TYPE_MAP = {'peaking_eq': 'peaking_eq', 'low_shelf': 'low_shelf', 'high_shelf': 'high_shelf'}


def to_complete_filter(response: DesignResponse, fs: int):
    '''
    Validates response, then converts its top-ranked candidate
    (candidates[0]) into a CompleteFilter at the given (publish-target) fs --
    the only candidate the caller acts on automatically; see
    alternative_filters() for the rest. §5: a BiquadSpec carries no fs and no
    count -- repeated identical entries are collapsed into a single stacked
    shelf here, since that representation is this repo's concern, not the
    designer's.
    :raises ContractViolation: if response is invalid.
    '''
    validate_response(response)
    if response.decline_reason is not None:
        raise ContractViolation("cannot convert a declined response -- check decline_reason first")

    return _candidate_to_complete_filter(response.candidates[0], fs)


def alternative_filters(response: DesignResponse, fs: int) -> list:
    '''
    Validates response, then converts every candidate after the top-ranked
    one (candidates[1:]) into a CompleteFilter, in the same best-first order
    -- for a human reviewing the report to compare against; never simulated
    or published automatically.
    :raises ContractViolation: if response is invalid.
    '''
    validate_response(response)
    if response.decline_reason is not None:
        raise ContractViolation("cannot convert a declined response -- check decline_reason first")

    return [_candidate_to_complete_filter(candidate, fs) for candidate in response.candidates[1:]]


def rejected_filters(response: DesignResponse, fs: int) -> list:
    '''
    Validates response, then converts every `rejected` design (1.1) into a CompleteFilter, in the designer's order -- for a
    human to look at and, if they choose, to publish as an override of the designer's verdict; never acted on automatically.
    Either shape may carry them; [] when there are none.
    :raises ContractViolation: if response is invalid.
    '''
    validate_response(response)
    return [_candidate_to_complete_filter(candidate, fs) for candidate in response.rejected or []]


def _candidate_to_complete_filter(candidate: DesignCandidate, fs: int):
    from model.iir import CompleteFilter

    biquads = [create_filter(spec, fs) for spec in _collapse(candidate.filters)]
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
