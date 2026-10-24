'''
docs/schema/http_designer_request.schema.json and http_designer_health.schema.json -- designer-interface.md §7.1's
machine-checkable shapes, including 1.2's arrays by reference: an array is inline or by reference, never both, a
reference carries its digest, and its path is relative with no way out of the root.
'''
import json
import os

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from pipeline.designer.contract import build_request
from pipeline.designer.http_binding import _request_to_json

_SCHEMA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'docs', 'schema')
_DIGEST = 'ab' * 32


def _validator(name):
    with open(os.path.join(_SCHEMA_DIR, name)) as f:
        schema = json.load(f)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


@pytest.fixture(scope='module')
def request_schema():
    return _validator('http_designer_request.schema.json')


@pytest.fixture(scope='module')
def health_schema():
    return _validator('http_designer_health.schema.json')


def _by_reference(path='t_1/mono.wav', channel=0, **overrides):
    array = {'dtype': 'float64', 'shape': [4], 'file': {'path': path, 'channel': channel}, 'sha256': _DIGEST}
    array.update(overrides)
    return array


def _body(mono_mix, channels=None, version='1.2'):
    return {'contract_version': version, 'fs': 1000, 'coverage': 'complete_programme', 'mono_mix': mono_mix,
            'channels': channels}


def _inline_body():
    return _request_to_json(build_request(np.arange(4.0), 1000, channels={'L': np.arange(4.0)}))


def test_what_the_binding_sends_inline_is_valid(request_schema):
    request_schema.validate(_inline_body())


def test_an_array_by_reference_is_valid(request_schema):
    request_schema.validate(_body(_by_reference()))


def test_a_request_may_mix_inline_and_by_reference_arrays(request_schema):
    inline = _inline_body()
    request_schema.validate(_body(inline['mono_mix'], {'L': _by_reference('t_1/multichannel.wav', 0)}))
    request_schema.validate(_body(_by_reference(), inline['channels']))


def test_an_array_with_both_forms_is_refused(request_schema):
    both = _by_reference(data_base64=_inline_body()['mono_mix']['data_base64'])
    assert not request_schema.is_valid(_body(both))


def test_a_reference_without_its_digest_is_refused(request_schema):
    array = _by_reference()
    del array['sha256']
    assert not request_schema.is_valid(_body(array))


@pytest.mark.parametrize('digest', ['AB' * 32, 'ab' * 31, 'zz' * 32])
def test_a_digest_must_be_64_lower_case_hex_digits(request_schema, digest):
    assert not request_schema.is_valid(_body(_by_reference(sha256=digest)))


@pytest.mark.parametrize('path', ['/srv/work/t_1/mono.wav', '../t_1/mono.wav', 't_1/../../etc/passwd', 't_1/..',
                                  '..', 't_1\\mono.wav', 'C:/work/mono.wav', ''])
def test_a_path_that_is_absolute_or_could_leave_the_root_is_refused(request_schema, path):
    assert not request_schema.is_valid(_body(_by_reference(path=path)))


@pytest.mark.parametrize('path', ['mono.wav', 't_1/mono.wav', 'a..b/multichannel.wav', 'Season 1/..x.wav'])
def test_a_relative_path_is_accepted(request_schema, path):
    request_schema.validate(_body(_by_reference(path=path)))


def test_a_negative_channel_is_refused(request_schema):
    assert not request_schema.is_valid(_body(_by_reference(channel=-1)))


def test_the_health_answer_needs_the_version_and_whether_there_is_a_shared_root(health_schema):
    health_schema.validate({'contract_version': '1.2', 'shared_root': True, 'status': 'ok'})
    assert not health_schema.is_valid({'status': 'ok'})
    assert not health_schema.is_valid({'contract_version': '1.2', 'shared_root': 'yes'})


@pytest.fixture(scope='module')
def response_schema():
    return _validator('http_designer_response.schema.json')


def _response(**fit):
    candidate = {'filters': [{'type': 'low_shelf', 'freq_hz': 20.0, 'gain_db': 6.0, 'q': 0.7}], 'confidence': 0.9,
                 'mv_adjust_db': 6.0, 'method': 'fitted', 'residual_db': 1.1, 'residual_band_hz': [5.0, 212.0], **fit}
    return {'contract_version': '1.3', 'candidates': [candidate]}


_FIT = {'residual_at_hz': 9.0, 'residual_direction': 'under', 'residual_rms_db': 0.3,
        'residual_within_tolerance': 0.96, 'residual_tolerance_db': 0.5}


def test_a_1_3_response_reporting_its_fit_is_valid(response_schema):
    assert not list(response_schema.iter_errors(_response(**_FIT)))
    assert not list(response_schema.iter_errors(_response()))   # without them: a 1.2 response
    assert not list(response_schema.iter_errors(_response(**{k: None for k in _FIT})))


@pytest.mark.parametrize('field, value', [('residual_direction', 'sideways'), ('residual_rms_db', -0.1),
                                          ('residual_within_tolerance', 1.5), ('residual_tolerance_db', 0)])
def test_a_fit_field_out_of_range_is_refused_by_the_schema(response_schema, field, value):
    assert list(response_schema.iter_errors(_response(**{**_FIT, field: value})))


_IDENTITY = {'design_version': 4, 'design_settings': 'd0', 'designer_build': '0.2.6 (frozen build)'}


def test_a_1_4_response_and_health_saying_the_designers_identity_are_valid(response_schema, health_schema):
    assert not list(response_schema.iter_errors({**_response(), **_IDENTITY}))
    assert not list(response_schema.iter_errors({**_response(), **{k: None for k in _IDENTITY}}))
    health_schema.validate({'contract_version': '1.2', 'shared_root': True, **_IDENTITY,
                            'design_settings_detail': {'strategy': ['flatten']}})


@pytest.mark.parametrize('field, value', [('design_version', -1), ('design_version', 1.5), ('design_settings', ''),
                                          ('designer_build', 3)])
def test_an_identity_out_of_range_is_refused_by_the_schemas(response_schema, health_schema, field, value):
    assert list(response_schema.iter_errors({**_response(), **_IDENTITY, field: value}))
    assert list(health_schema.iter_errors({'contract_version': '1.2', 'shared_root': True, **_IDENTITY, field: value}))


_ADJUSTMENTS = [{'id': 'a1', 'kind': 'exclude_band', 'params': {'low_hz': 28.0, 'high_hz': 34.0}},
                {'id': 'a2', 'kind': 'goal_tilt', 'params': {'db_per_octave': 1.0}, 'text': 'a little more weight'},
                {'id': 'a3', 'kind': 'note', 'params': {}, 'text': 'the opening logo is not the film'},
                {'id': 'a4', 'kind': 'x-beqforge-strategy', 'params': {'names': ['parametric']}, 'text': None}]


def test_a_1_5_request_with_adjustments_is_valid(request_schema):
    request_schema.validate({**_inline_body(), 'contract_version': '1.5', 'adjustments': _ADJUSTMENTS})
    request_schema.validate({**_inline_body(), 'adjustments': None})


@pytest.mark.parametrize('adjustment', [{'kind': 'floor', 'params': {'hz': 8}}, {'id': 'a', 'params': {}},
                                        {'id': 'a', 'kind': 'floor'}, {'id': '', 'kind': 'floor', 'params': {}},
                                        {'id': 'a', 'kind': 'floor', 'params': [8]},
                                        {'id': 'a', 'kind': 'floor', 'params': {}, 'why': 'extra'}])
def test_an_adjustment_not_of_the_shape_is_refused(request_schema, adjustment):
    assert not request_schema.is_valid({**_inline_body(), 'contract_version': '1.5', 'adjustments': [adjustment]})


def test_an_unknown_kind_or_bad_params_pass_the_schema_to_be_answered_in_the_outcome(request_schema):
    request_schema.validate({**_inline_body(), 'contract_version': '1.5',
                             'adjustments': [{'id': 'a', 'kind': 'floor', 'params': {'hz': -3}},
                                             {'id': 'b', 'kind': 'make_it_better', 'params': {}}]})


_OUTCOMES = [{'id': 'a1', 'outcome': 'honoured'},
             {'id': 'a2', 'outcome': 'adapted', 'applied': {'db_per_octave': 0.5}, 'reason': 'capped by the evidence'},
             {'id': 'a3', 'outcome': 'not_understood', 'applied': None, 'reason': 'no time ranges in this designer'},
             {'id': 'a4', 'outcome': 'declined', 'reason': 'unknown strategy'}]


def test_a_1_5_response_answering_its_adjustments_is_valid(response_schema):
    response_schema.validate({**_response(), 'contract_version': '1.5', 'adjustment_outcomes': _OUTCOMES})
    decline = {'contract_version': '1.5', 'decline_reason': 'no_rolloff_detected', 'adjustment_outcomes': _OUTCOMES}
    response_schema.validate(decline)


@pytest.mark.parametrize('outcome', [{'id': 'a', 'outcome': 'ignored', 'reason': 'x'},
                                     {'id': 'a', 'outcome': 'adapted', 'reason': 'x'},
                                     {'id': 'a', 'outcome': 'adapted', 'applied': {'db': 3}},
                                     {'id': 'a', 'outcome': 'declined'},
                                     {'id': 'a', 'outcome': 'declined', 'reason': ''},
                                     {'id': 'a', 'outcome': 'declined', 'applied': {'db': 3}, 'reason': 'x'},
                                     {'outcome': 'honoured'}])
def test_an_outcome_without_what_it_needs_is_refused(response_schema, outcome):
    assert not response_schema.is_valid({**_response(), 'contract_version': '1.5', 'adjustment_outcomes': [outcome]})


def test_health_may_list_the_adjustment_kinds_it_knows(health_schema):
    kinds = {'floor': {'description': 'restore nothing below', 'params': {
        'hz': {'type': 'number', 'unit': 'Hz', 'minimum': 1, 'maximum': 40, 'default': 5}}},
        'note': {'description': 'free text'}}
    health_schema.validate({'contract_version': '1.5', 'shared_root': True, 'adjustment_kinds': kinds})
    assert not health_schema.is_valid({'contract_version': '1.5', 'shared_root': True,
                                       'adjustment_kinds': {'floor': {'params': {}}}})
    assert not health_schema.is_valid({'contract_version': '1.5', 'shared_root': True,
                                       'adjustment_kinds': {'floor': {'description': 'x',
                                                                      'params': {'hz': {'type': 'float'}}}}})


_RECORD = 'records/designer-0123456789abcdef-89abcdef-v6.run.json.gz'


def test_a_1_6_response_may_name_its_run_record(response_schema):
    assert not list(response_schema.iter_errors({**_response(), 'run_record': _RECORD}))
    assert not list(response_schema.iter_errors({**_response(), 'run_record': None}))
    decline = {'contract_version': '1.5', 'decline_reason': 'no_usable_plateau', 'decline_message': 'x',
               'run_record': _RECORD}
    assert not list(response_schema.iter_errors(decline))


@pytest.mark.parametrize('path', ['/work/records/x.run.json.gz', '\\\\ZALAGA-UNRAID\\beq\\work\\records\\x.run.json.gz',
                                  'records\\x.run.json.gz', '../records/x.run.json.gz', 'C:/records/x.gz', ''])
def test_a_run_record_that_is_absolute_or_could_leave_the_root_is_refused(response_schema, path):
    assert list(response_schema.iter_errors({**_response(), 'run_record': path}))


def test_health_may_say_it_answers_impact(health_schema):
    health_schema.validate({'contract_version': '1.6', 'shared_root': True, 'impact': True, **_IDENTITY})
    assert not health_schema.is_valid({'contract_version': '1.6', 'shared_root': True, 'impact': 'yes'})


@pytest.fixture(scope='module')
def impact_request_schema():
    return _validator('http_designer_impact_request.schema.json')


@pytest.fixture(scope='module')
def impact_response_schema():
    return _validator('http_designer_impact_response.schema.json')


def _impact_request(records):
    return {'contract_version': '1.6', 'records': records}


def test_an_impact_request_names_records_by_id_and_relative_path(impact_request_schema):
    impact_request_schema.validate(_impact_request([{'id': 'jriver-1', 'path': _RECORD}]))
    impact_request_schema.validate(_impact_request([{'id': f't{i}', 'path': _RECORD} for i in range(500)]))


@pytest.mark.parametrize('body', [
    _impact_request([]),
    _impact_request([{'id': f't{i}', 'path': _RECORD} for i in range(501)]),
    _impact_request([{'id': 'a'}]),
    _impact_request([{'id': '', 'path': _RECORD}]),
    _impact_request([{'id': 'a', 'path': '/work/records/x.gz'}]),
    _impact_request([{'id': 'a', 'path': 'records/../../etc/passwd'}]),
    _impact_request([{'id': 'a', 'path': _RECORD, 'extra': 1}]),
    {'contract_version': '1.5', 'records': [{'id': 'a', 'path': _RECORD}]},
])
def test_an_impact_request_out_of_shape_is_refused(impact_request_schema, body):
    assert not impact_request_schema.is_valid(body)


def _impact_response(*results, changes=()):
    return {'contract_version': '1.6', 'design_version': 7, 'design_settings': '9cf04c30b0f0', 'designer_build': 'x',
            'changes': list(changes), 'results': list(results)}


_CHANGE = {'design_version': 7, 'id': 'fit-choice', 'summary': 'the fitter chooses among near-equal fits'}
_REASON = {'change': 'fit-choice', 'feature': 'fit.choice', 'detail': 'near-equal alternatives'}


def test_an_impact_response_with_every_tier_is_valid(impact_response_schema):
    impact_response_schema.validate(_impact_response(
        {'id': 'a', 'tier': 'will_move', 'score': 1.0, 'from_design_version': 6, 'reasons': [_REASON]},
        {'id': 'b', 'tier': 'likely', 'score': 0.72, 'from_design_version': 6, 'reasons': [_REASON]},
        {'id': 'c', 'tier': 'possible', 'score': 0.1, 'from_design_version': 6, 'reasons': [_REASON]},
        {'id': 'd', 'tier': 'unaffected', 'score': 0.0, 'from_design_version': 6, 'reasons': []},
        {'id': 'e', 'tier': 'unknown', 'score': None, 'from_design_version': None,
         'reasons': [{'change': None, 'feature': None, 'detail': 'record not found'}]},
        changes=[_CHANGE]))


@pytest.mark.parametrize('result', [
    {'id': 'a', 'tier': 'unknown', 'score': 0.5, 'from_design_version': None, 'reasons': [_REASON]},
    {'id': 'a', 'tier': 'likely', 'score': None, 'from_design_version': 6, 'reasons': [_REASON]},
    {'id': 'a', 'tier': 'likely', 'score': 0.5, 'from_design_version': 6, 'reasons': []},
    {'id': 'a', 'tier': 'likely', 'score': 1.5, 'from_design_version': 6, 'reasons': [_REASON]},
    {'id': 'a', 'tier': 'maybe', 'score': 0.5, 'from_design_version': 6, 'reasons': [_REASON]},
    {'id': 'a', 'tier': 'likely', 'score': 0.5, 'from_design_version': 6, 'reasons': [{'detail': 'x'}]},
    {'id': 'a', 'tier': 'likely', 'score': 0.5, 'reasons': [_REASON]},
])
def test_an_impact_result_out_of_shape_is_refused(impact_response_schema, result):
    assert not impact_response_schema.is_valid(_impact_response(result))
