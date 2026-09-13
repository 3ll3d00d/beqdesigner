'''
The HTTP designer binding -- design/http-designer-binding-plan.md phase 1,
design/designer-interface.md §7 ("nothing here stops a subprocess or HTTP
binding later that serialises the same fields as JSON... only how they
cross a process boundary does [change]").

http_designer(url) returns a plain DesignerCallable (the same shape an
in-process designer already has -- Callable[[DesignRequest], DesignResponse])
that POSTs a JSON-encoded DesignRequest to `url` and decodes a JSON-encoded
DesignResponse back. Register it exactly like any other designer:

    register_designer('my.remote.designer', http_designer('http://host:port/design'))

Chosen over a subprocess binding for deployment flexibility (§ intro of the
plan doc): the designer can be local, on another machine, written in any
language, or shared across a team, without beqd spawning or managing it.
'''
import base64
from typing import Optional

import numpy as np
import requests

from pipeline.designer.contract import BiquadSpec, CONTRACT_VERSION, DesignCandidate, DesignRequest, DesignResponse

_ARRAY_DTYPE = 'float64'

_CANDIDATE_FIELDS = (
    'filters', 'confidence', 'mv_adjust_db', 'method', 'gain_reduction_db', 'residual_db', 'residual_band_hz',
    'commentary', 'fc_hz', 'slope', 'fc_uncertainty_hz', 'slope_uncertainty', 'channel_scope',
)


class HttpDesignerError(RuntimeError):
    '''
    Raised for anything that isn't a well-formed DesignResponse coming back
    from an HTTP designer -- a connection failure, a non-2xx status, a body
    that isn't valid JSON, or JSON that doesn't match DesignResponse's shape.
    Per designer-interface.md §1, the caller treats this as an
    implementation failure, not a signal -- nothing downstream runs on it,
    and it is never coerced into a decline. A *well-formed* DesignResponse
    that fails §3-§5 validation is unaffected by this binding -- that still
    goes through pipeline.designer.convert.validate_response as normal.
    '''


def _ndarray_to_json(arr: np.ndarray) -> dict:
    as_f64 = np.ascontiguousarray(arr, dtype='<f8')
    return {'dtype': _ARRAY_DTYPE, 'shape': list(as_f64.shape), 'data_base64': base64.b64encode(as_f64.tobytes()).decode('ascii')}


def _ndarray_from_json(d: dict) -> np.ndarray:
    if d.get('dtype') != _ARRAY_DTYPE:
        raise HttpDesignerError(f"unsupported array dtype {d.get('dtype')!r}, expected {_ARRAY_DTYPE!r}")
    data = np.frombuffer(base64.b64decode(d['data_base64']), dtype='<f8')
    return data.reshape(d['shape'])


def _request_to_json(request: DesignRequest) -> dict:
    return {
        'contract_version': request.contract_version,
        'fs': request.fs,
        'coverage': request.coverage,
        'mono_mix': _ndarray_to_json(request.mono_mix),
        'channels': {k: _ndarray_to_json(v) for k, v in request.channels.items()} if request.channels else None,
        'bass_management': request.bass_management,
    }


def _response_from_json(body: dict) -> DesignResponse:
    try:
        if body.get('candidates') is not None:
            candidates = [_candidate_from_json(c) for c in body['candidates']]
            return DesignResponse(contract_version=body.get('contract_version', CONTRACT_VERSION),
                                  candidates=candidates)
        return DesignResponse(contract_version=body.get('contract_version', CONTRACT_VERSION),
                              decline_reason=body.get('decline_reason'), decline_message=body.get('decline_message'))
    except (KeyError, TypeError, ValueError) as e:
        raise HttpDesignerError(f"response body does not match DesignResponse's shape: {e}") from e


def _candidate_from_json(c: dict) -> DesignCandidate:
    kwargs = {k: c[k] for k in _CANDIDATE_FIELDS if k in c}
    kwargs['filters'] = [BiquadSpec(**spec) for spec in kwargs['filters']]
    if kwargs.get('residual_band_hz') is not None:
        kwargs['residual_band_hz'] = tuple(kwargs['residual_band_hz'])
    return DesignCandidate(**kwargs)


def http_designer(url: str, timeout: float = 300.0, headers: Optional[dict] = None):
    '''
    :param url: the designer's HTTP endpoint -- one POST per design() call.
    :param timeout: seconds to wait for a response (default 300s -- a real
        design computation over a full-length programme may be slow).
    :param headers: passed through on every request, e.g. a bearer token
        for a non-localhost/shared designer. No larger auth framework.
    :return: a DesignerCallable -- register it with
        pipeline.designer.registry.register_designer() like any other.
    '''
    def _call(request: DesignRequest) -> DesignResponse:
        payload = _request_to_json(request)
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as e:
            raise HttpDesignerError(f"POST {url} failed: {e}") from e
        try:
            body = response.json()
        except ValueError as e:
            raise HttpDesignerError(f"{url} did not return a valid JSON body: {e}") from e
        return _response_from_json(body)

    return _call
