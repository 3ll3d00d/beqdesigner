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
