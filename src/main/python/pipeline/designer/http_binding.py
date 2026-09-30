'''
The HTTP designer binding -- design/archive/http-designer-binding-plan.md phase 1,
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
import hashlib
import logging
import os
from typing import Any, Mapping, Optional
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import requests

from pipeline.designer.registry import register_designer
from pipeline.designer.contract import BiquadSpec, CONTRACT_VERSION, DesignCandidate, DesignRequest, DesignResponse
from pipeline.designer.sources import ArraySource, AudioSources

logger = logging.getLogger('pipeline.designer.http_binding')

BY_REFERENCE_VERSION = '1.2'
_ERROR_DETAIL_LIMIT = 500

_ARRAY_DTYPE = 'float64'

_CANDIDATE_FIELDS = (
    'filters', 'confidence', 'mv_adjust_db', 'method', 'gain_reduction_db', 'residual_db', 'residual_band_hz',
    'commentary', 'fc_hz', 'slope', 'fc_uncertainty_hz', 'slope_uncertainty', 'channel_scope', 'rejection_reasons',
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


def sha256_of(arr: np.ndarray) -> str:
    ''' designer-interface.md §7.1's digest: SHA-256 of the array as C-contiguous little-endian float64 bytes. '''
    return hashlib.sha256(memoryview(np.ascontiguousarray(arr, dtype='<f8'))).hexdigest()


def _relative_to(path: str, root: str) -> Optional[str]:
    ''' `path` relative to `root` in POSIX form, both with symlinks resolved; None if it is not under the root. '''
    real_root = os.path.realpath(root)
    real_path = os.path.realpath(path)
    try:
        if os.path.commonpath([real_root, real_path]) != real_root or real_path == real_root:
            return None
    except ValueError:  # different drives
        return None
    return os.path.relpath(real_path, real_root).replace(os.sep, '/')


def _ndarray_by_reference(arr: np.ndarray, source: Optional[ArraySource], shared_root: str, fs: int) -> Optional[dict]:
    '''
    The array as a `file` reference (§7.1, 1.2), or None when it must go inline: no source, a file outside the shared
    root, or a WAV whose header says the array cannot be its column unchanged (another rate, length or channel count --
    it was resampled, trimmed, or is not what the caller thinks). The digest is of the array held here, so a file that
    differs anyway is the designer's 422, never a silently different design.
    '''
    if source is None:
        return None
    relative = _relative_to(source.path, shared_root)
    if relative is None:
        logger.info(f"{source.path} is not under the shared root {shared_root}, sending it inline")
        return None
    import soundfile as sf
    try:
        info = sf.info(source.path)
    except (RuntimeError, OSError) as e:
        logger.info(f"{source.path} cannot be read ({e}), sending it inline")
        return None
    as_f64 = np.ascontiguousarray(arr, dtype='<f8')
    if info.format.upper() != 'WAV' or info.samplerate != fs or info.frames != as_f64.shape[0] \
            or not 0 <= source.channel < info.channels:
        logger.info(f"{source.path} column {source.channel} is not the array as held ({info.format} "
                    f"{info.samplerate} Hz x {info.frames} x {info.channels} against {fs} Hz x {as_f64.shape[0]}), "
                    f"sending it inline")
        return None
    return {'dtype': _ARRAY_DTYPE, 'shape': list(as_f64.shape),
            'file': {'path': relative, 'channel': source.channel}, 'sha256': sha256_of(as_f64)}


def _request_to_json(request: DesignRequest, sources: Optional[AudioSources] = None,
                     shared_root: Optional[str] = None) -> dict:
    '''
    The §7.1 body. With `sources` and a `shared_root`, each array whose source can be referenced goes as a `file`
    (1.2), and the body then says "1.2"; the rest go inline, and a body with nothing by reference is exactly 1.1's.
    '''
    def encode(arr, source):
        if sources is not None and shared_root:
            by_reference = _ndarray_by_reference(arr, source, shared_root, request.fs)
            if by_reference is not None:
                return by_reference
        return _ndarray_to_json(arr)

    payload = {
        'contract_version': request.contract_version,
        'fs': request.fs,
        'coverage': request.coverage,
        'mono_mix': encode(request.mono_mix, sources.mono if sources else None),
        'channels': {k: encode(v, sources.channels.get(k) if sources else None) for k, v in request.channels.items()}
        if request.channels else None,
        'bass_management': request.bass_management,
    }
    if has_reference(payload):
        payload['contract_version'] = BY_REFERENCE_VERSION
    return payload


def has_reference(payload: dict) -> bool:
    ''' Whether a §7.1 body holds an array by reference. '''
    return 'file' in payload['mono_mix'] or any('file' in v for v in (payload.get('channels') or {}).values())


def _response_from_json(body: dict) -> DesignResponse:
    try:
        # 1.1: either shape may add `rejected`; an empty list is passed on as one, for validate_response() to refuse
        rejected = [_candidate_from_json(c) for c in body['rejected']] if body.get('rejected') is not None else None
        if body.get('candidates') is not None:
            candidates = [_candidate_from_json(c) for c in body['candidates']]
            return DesignResponse(contract_version=body.get('contract_version', CONTRACT_VERSION),
                                  candidates=candidates, rejected=rejected)
        return DesignResponse(contract_version=body.get('contract_version', CONTRACT_VERSION),
                              decline_reason=body.get('decline_reason'), decline_message=body.get('decline_message'),
                              rejected=rejected)
    except (KeyError, TypeError, ValueError) as e:
        raise HttpDesignerError(f"response body does not match DesignResponse's shape: {e}") from e


def _candidate_from_json(c: dict) -> DesignCandidate:
    kwargs = {k: c[k] for k in _CANDIDATE_FIELDS if k in c}
    kwargs['filters'] = [BiquadSpec(**spec) for spec in kwargs['filters']]
    if kwargs.get('residual_band_hz') is not None:
        kwargs['residual_band_hz'] = tuple(kwargs['residual_band_hz'])
    return DesignCandidate(**kwargs)


def health_url(url: str) -> str:
    ''' `GET /health` on the design endpoint's origin (§7.1). '''
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, '/health', '', ''))


def _error_detail(response) -> str:
    ''' What a refusing designer said: its JSON `error`, else the start of its body. '''
    try:
        body = response.json()
        if isinstance(body, dict) and body.get('error'):
            return str(body['error'])[:_ERROR_DETAIL_LIMIT]
    except ValueError:
        pass
    return (getattr(response, 'text', '') or '')[:_ERROR_DETAIL_LIMIT]


def check_takes_references(url: str, timeout: float = 10.0, headers: Optional[dict] = None) -> None:
    '''
    Asks the designer's `/health` whether it can take arrays by reference (§7.1): contract 1.2 or later and a shared
    root configured.
    :raises HttpDesignerError: naming what it answered, if it cannot or did not answer.
    '''
    target = health_url(url)
    try:
        response = requests.get(target, headers=headers, timeout=timeout)
        response.raise_for_status()
        body = response.json()
    except (requests.RequestException, ValueError) as e:
        raise HttpDesignerError(f"GET {target} failed, so nothing is sent to it by reference: {e}") from e
    version = str(body.get('contract_version', '')) if isinstance(body, dict) else ''
    try:
        recent = tuple(int(p) for p in version.split('.')[:2]) >= (1, 2)
    except ValueError:
        recent = False
    if not recent or body.get('shared_root') is not True:
        raise HttpDesignerError(f"{target} answered {body!r}: it cannot take arrays by reference (it needs contract "
                                f"1.2 or later and a shared root), so it is sent none. Configure its shared root, or "
                                f"take by_reference off this designer")


def http_designer(url: str, timeout: float = 300.0, headers: Optional[dict] = None, shared_root: Optional[str] = None):
    '''
    :param url: the designer's HTTP endpoint -- one POST per design() call.
    :param timeout: seconds to wait for a response (default 300s -- a real
        design computation over a full-length programme may be slow).
    :param headers: passed through on every request, e.g. a bearer token
        for a non-localhost/shared designer. No larger auth framework.
    :param shared_root: the caller's end of the root it shares with the designer (§7.1, 1.2) -- its work directory. When
        given, the callable also takes `sources=` (register it with takes_sources=True), and each array whose source WAV
        is under this root is sent by reference; the designer's `/health` is checked once, before the first such
        request. None (the default): every array goes inline, as in 1.1.
    :return: a DesignerCallable -- register it with
        pipeline.designer.registry.register_designer() like any other.
    '''
    checked = []

    def _post(payload: dict) -> DesignResponse:
        if has_reference(payload) and not checked:
            check_takes_references(url, headers=headers)
            checked.append(True)
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            raise HttpDesignerError(f"POST {url} failed: {e}") from e
        try:
            response.raise_for_status()
        except requests.RequestException as e:
            # a 422 (an array by reference it could not honour) or a 400 says why; neither is a decline, and neither
            # is retried inline (§7.1)
            detail = _error_detail(response)
            raise HttpDesignerError(f"POST {url} failed: {e}" + (f": {detail}" if detail else '')) from e
        try:
            body = response.json()
        except ValueError as e:
            raise HttpDesignerError(f"{url} did not return a valid JSON body: {e}") from e
        return _response_from_json(body)

    if shared_root is None:
        def _call(request: DesignRequest) -> DesignResponse:
            return _post(_request_to_json(request))
    else:
        def _call(request: DesignRequest, sources: Optional[AudioSources] = None) -> DesignResponse:
            return _post(_request_to_json(request, sources, shared_root))
    return _call


def register_declared_designers(declared: Optional[Mapping[str, Any]]) -> list:
    '''
    Registers the HTTP designers a configuration file declares under `designers:` -- name -> URL, or name -> {url, timeout,
    headers} -- so that a run can find them. The CLI does it for its config file, and the work list for its profile.
    :return: the names registered.
    :raises ValueError: if an entry has no URL (nothing after it is registered).
    '''
    names = []
    for name, spec in (declared or {}).items():
        if isinstance(spec, str):
            spec = {'url': spec}
        if not isinstance(spec, Mapping) or not spec.get('url'):
            raise ValueError(f"designer {name!r} needs a url")
        register_designer(name, http_designer(spec['url'], timeout=float(spec.get('timeout', 300.0)),
                                              headers=spec.get('headers') or None))
        names.append(name)
    return names
