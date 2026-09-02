'''
Phase 2 (item 12) of design/pipeline-implementation-plan.md: the designer
contract (design/designer-interface.md v1.0) exercised via a fake in-process
designer -- no real designer implementation is needed to validate this
boundary, and none should be waited on to test it.
'''
import math

import pytest

from pipeline.designer.contract import DesignRequest, DesignResponse, BiquadSpec, build_request


def _success(**overrides):
    defaults = dict(
        contract_version='1.0',
        filters=[BiquadSpec(type='low_shelf', freq_hz=15.810, gain_db=15.918, q=0.7071)],
        confidence=0.94,
        mv_adjust_db=15.918,
        method='exact',
        residual_db=0.0008,
        residual_band_hz=(5.0, 200.0),
    )
    defaults.update(overrides)
    return DesignResponse(**defaults)


def _decline(**overrides):
    defaults = dict(contract_version='1.0', decline_reason='no_rolloff_detected')
    defaults.update(overrides)
    return DesignResponse(**defaults)


# --- registry -----------------------------------------------------------

def test_registry_round_trip():
    from pipeline.designer.registry import register_designer, get_designer, unregister_designer, registered_designers

    def fake(request: DesignRequest) -> DesignResponse:
        return _decline()

    register_designer('test.fake', fake)
    try:
        assert 'test.fake' in registered_designers()
        assert get_designer('test.fake') is fake
        request = build_request(mono_mix=None, fs=1000)
        assert get_designer('test.fake')(request).decline_reason == 'no_rolloff_detected'
    finally:
        unregister_designer('test.fake')
    assert 'test.fake' not in registered_designers()


def test_get_unregistered_designer_raises():
    from pipeline.designer.registry import get_designer
    with pytest.raises(KeyError):
        get_designer('does.not.exist')


# --- validate_response: happy paths --------------------------------------

def test_valid_success_response_passes_validation():
    from pipeline.designer.convert import validate_response
    validate_response(_success())


def test_valid_decline_response_passes_validation():
    from pipeline.designer.convert import validate_response
    validate_response(_decline())


def test_non_parametric_without_residual_is_valid():
    from pipeline.designer.convert import validate_response
    validate_response(_success(method='non_parametric', residual_db=None, residual_band_hz=None))


# --- validate_response: rejection paths ----------------------------------

def test_both_success_and_decline_populated_rejected():
    from pipeline.designer.convert import validate_response, ContractViolation
    with pytest.raises(ContractViolation, match='both filters and decline_reason'):
        validate_response(_success(decline_reason='no_rolloff_detected'))


def test_neither_success_nor_decline_populated_rejected():
    from pipeline.designer.convert import validate_response, ContractViolation
    with pytest.raises(ContractViolation, match='neither filters nor decline_reason'):
        validate_response(DesignResponse(contract_version='1.0'))


def test_decline_with_success_field_populated_rejected():
    from pipeline.designer.convert import validate_response, ContractViolation
    with pytest.raises(ContractViolation, match='decline response also populates'):
        validate_response(_decline(confidence=0.5))


def test_wrong_filter_type_rejected():
    from pipeline.designer.convert import validate_response, ContractViolation
    bad = _success(filters=[BiquadSpec(type='all_pass', freq_hz=20.0, gain_db=0.0, q=0.7)])
    with pytest.raises(ContractViolation, match='not publishable'):
        validate_response(bad)


def test_budget_exceeded_rejected():
    from pipeline.designer.convert import validate_response, ContractViolation
    too_many = [BiquadSpec(type='peaking_eq', freq_hz=40.0 + i, gain_db=1.0, q=1.0) for i in range(11)]
    with pytest.raises(ContractViolation, match='exceeds the budget'):
        validate_response(_success(filters=too_many))


def test_empty_filters_list_rejected():
    from pipeline.designer.convert import validate_response, ContractViolation
    with pytest.raises(ContractViolation, match='empty filters list'):
        validate_response(_success(filters=[]))


@pytest.mark.parametrize('bad_spec,match', [
    (BiquadSpec(type='low_shelf', freq_hz=0.0, gain_db=1.0, q=0.7), 'freq_hz'),
    (BiquadSpec(type='low_shelf', freq_hz=-10.0, gain_db=1.0, q=0.7), 'freq_hz'),
    (BiquadSpec(type='low_shelf', freq_hz=float('inf'), gain_db=1.0, q=0.7), 'freq_hz'),
    (BiquadSpec(type='low_shelf', freq_hz=18.0, gain_db=float('nan'), q=0.7), 'gain_db'),
    (BiquadSpec(type='low_shelf', freq_hz=18.0, gain_db=1.0, q=0.0), 'q must be'),
    (BiquadSpec(type='low_shelf', freq_hz=18.0, gain_db=1.0, q=float('inf')), 'q must be'),
])
def test_non_finite_or_invalid_biquad_values_rejected(bad_spec, match):
    from pipeline.designer.convert import validate_response, ContractViolation
    with pytest.raises(ContractViolation, match=match):
        validate_response(_success(filters=[bad_spec]))


def test_non_parametric_with_residual_rejected():
    from pipeline.designer.convert import validate_response, ContractViolation
    with pytest.raises(ContractViolation, match='non_parametric'):
        validate_response(_success(method='non_parametric', residual_db=0.5, residual_band_hz=(5.0, 100.0)))


def test_confidence_out_of_range_rejected():
    from pipeline.designer.convert import validate_response, ContractViolation
    with pytest.raises(ContractViolation, match='confidence'):
        validate_response(_success(confidence=1.5))


def test_missing_confidence_rejected():
    from pipeline.designer.convert import validate_response, ContractViolation
    with pytest.raises(ContractViolation, match='no confidence'):
        validate_response(_success(confidence=None))


def test_non_finite_mv_adjust_rejected():
    from pipeline.designer.convert import validate_response, ContractViolation
    with pytest.raises(ContractViolation, match='mv_adjust_db'):
        validate_response(_success(mv_adjust_db=float('nan')))


def test_missing_decline_reason_string_rejected():
    from pipeline.designer.convert import validate_response, ContractViolation
    with pytest.raises(ContractViolation, match='decline_reason'):
        validate_response(DesignResponse(contract_version='1.0', decline_reason=''))


def test_declined_response_cannot_be_converted():
    from pipeline.designer.convert import to_complete_filter, ContractViolation
    with pytest.raises(ContractViolation, match='declined'):
        to_complete_filter(_decline(), fs=48000)


# --- to_complete_filter: conversion + collapsing -------------------------

def test_to_complete_filter_matches_rp1_worked_example():
    '''
    The exact example from designer-interface.md §6 -- 2x the same low_shelf,
    which must collapse into a single stacked shelf (count=2) rather than
    two separate LowShelf filters, matching this repo's own beq_gain/count
    fidelity concern (design/api-headless-pipeline.md §11.1).
    '''
    from model.iir import LowShelf
    from pipeline.designer.convert import to_complete_filter

    response = _success(filters=[
        BiquadSpec(type='low_shelf', freq_hz=15.810, gain_db=15.918, q=0.7071),
        BiquadSpec(type='low_shelf', freq_hz=15.810, gain_db=15.918, q=0.7071),
    ])
    complete = to_complete_filter(response, fs=96000)

    assert len(complete) == 1
    shelf = complete.filters[0]
    assert isinstance(shelf, LowShelf)
    assert shelf.count == 2
    assert math.isclose(shelf.freq, 15.810)
    assert math.isclose(shelf.gain, 15.918)
    assert math.isclose(shelf.q, 0.7071)


def test_to_complete_filter_does_not_collapse_peaking_eq():
    from model.iir import PeakingEQ
    from pipeline.designer.convert import to_complete_filter

    response = _success(filters=[
        BiquadSpec(type='peaking_eq', freq_hz=40.0, gain_db=-3.0, q=2.0),
        BiquadSpec(type='peaking_eq', freq_hz=40.0, gain_db=-3.0, q=2.0),
    ])
    complete = to_complete_filter(response, fs=96000)

    assert len(complete) == 2
    assert all(isinstance(f, PeakingEQ) for f in complete.filters)


def test_to_complete_filter_gain_convention_matches_shelf_identity():
    '''
    designer-interface.md §5's worked identity: zero at fz, pole at fp,
    shared Q -> low_shelf(freq=sqrt(fz*fp), gain=40*log10(fz/fp), q=Q).
    Confirms the RBJ A=10**(gain_db/40) convention lands on the same
    coefficients model/iir.py's LowShelf already uses -- no separate
    conversion needed at this boundary.
    '''
    from pipeline.designer.convert import to_complete_filter
    from model.iir import LowShelf

    fz, fp, q = 25.0, 10.0, 0.7071
    freq_hz = math.sqrt(fz * fp)
    gain_db = 40 * math.log10(fz / fp)

    response = _success(filters=[BiquadSpec(type='low_shelf', freq_hz=freq_hz, gain_db=gain_db, q=q)])
    complete = to_complete_filter(response, fs=96000)
    via_contract = complete.filters[0]

    direct = LowShelf(96000, freq_hz, q, gain_db)
    assert via_contract.get_sos() == direct.get_sos()


def test_pipeline_designer_modules_have_no_qtpy_import():
    import ast
    import pathlib
    pkg_dir = (pathlib.Path(__file__).parent / '..' / '..' / 'main' / 'python' / 'pipeline' / 'designer').resolve()
    for source in pkg_dir.glob('*.py'):
        tree = ast.parse(source.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any(n.name.startswith('qtpy') for n in node.names), source
            elif isinstance(node, ast.ImportFrom):
                assert node.module is None or not node.module.startswith('qtpy'), source
