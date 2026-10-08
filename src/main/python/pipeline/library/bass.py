'''
The playback chain a designer is told about (TODO R8): DesignRequest.bass_management (designer-interface.md §2), from a
profile's `run.bass_management` or, for Batch Design, the app's Preferences.

    run:
      bass_management: {lpf_fs: 80, lpf_position: Before, headroom_type: WCS, clip_before: false, clip_after: false}

Every key is optional and takes the default below; leaving the section out sends no bass management at all, and the
designer then assumes its own model (beqforge says which in its commentary). No Qt.
'''
from typing import Any, Mapping, Optional

from model.preferences import BM_LPF_OPTIONS

DEFAULTS = {'lpf_fs': 80.0, 'lpf_position': 'Before', 'headroom_type': 'WCS', 'clip_before': False,
            'clip_after': False}


def _headroom(value: Any) -> str:
    text = str(value).strip()
    if text.upper() == 'WCS':
        return 'WCS'
    try:
        float(text)
    except ValueError:
        raise ValueError(f'run.bass_management.headroom_type must be WCS or a number of dB, not {value!r}') from None
    return text


def bass_management(value: Optional[Mapping[str, Any]]) -> Optional[dict]:
    '''
    The contract's bass_management dict from a profile's `run.bass_management`, or None if it has none.
    :raises ValueError: naming the key that is wrong.
    '''
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError('run.bass_management must be a mapping of lpf_fs, lpf_position, headroom_type, clip_before '
                         'and clip_after')
    unknown = set(value) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'unknown run.bass_management key(s): {", ".join(sorted(map(str, unknown)))}')
    result = dict(DEFAULTS)
    if 'lpf_fs' in value:
        lpf = value['lpf_fs']
        if isinstance(lpf, bool) or not isinstance(lpf, (int, float)) or not 0 < lpf < 1000:
            raise ValueError(f'run.bass_management.lpf_fs must be a crossover in Hz, not {lpf!r}')
        result['lpf_fs'] = float(lpf)
    if 'lpf_position' in value:
        position = str(value['lpf_position']).strip().capitalize()
        if position not in BM_LPF_OPTIONS:
            raise ValueError(f'run.bass_management.lpf_position must be one of {", ".join(BM_LPF_OPTIONS)}')
        result['lpf_position'] = position
    if 'headroom_type' in value:
        result['headroom_type'] = _headroom(value['headroom_type'])
    for key in ('clip_before', 'clip_after'):
        if key in value:
            if not isinstance(value[key], bool):
                raise ValueError(f'run.bass_management.{key} must be true or false')
            result[key] = value[key]
    return result


def from_preferences(lpf_fs, lpf_position) -> dict:
    ''' Batch Design's: the crossover and its position from Preferences (Analysis), the rest the defaults. '''
    return bass_management({'lpf_fs': float(lpf_fs), 'lpf_position': lpf_position})


def describe(value: Optional[Mapping[str, Any]]) -> str:
    ''' For a reviewer: what the designer was told the playback chain is. '''
    if not value:
        return 'none sent: the designer assumed its own playback chain'
    return (f'{value["lpf_fs"]:g} Hz crossover, low-pass {value["lpf_position"].lower()}, '
            f'headroom {value["headroom_type"]}'
            + (', clipping before' if value.get('clip_before') else '')
            + (', clipping after' if value.get('clip_after') else ''))
