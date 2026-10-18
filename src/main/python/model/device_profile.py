'''
What a filter becomes on a device: the device's internal sample rate and how it stores coefficients (design/device-profiles.md).

A device profile is a rate (48 or 96 kHz) and a coefficient storage (float32, or signed fixed point with so many integer
and fractional bits). The filter is recalculated at the profile's rate in double precision, its coefficients stored as the
device stores them, and the response of what is stored compared with the response of what was designed. At low
frequencies and high Q the difference can be large: float32 at 96 kHz turns a 5 Hz Q 6 +12 dB peak into +14 dB at 4 Hz
and -2 dB at 5 Hz.

The built-in profiles are the devices beqdesigner exports to (model/dsp_type.py): the HD-class miniDSPs keep float32
coefficients at 96 kHz, the earlier ones Q5.23 fixed point at 48 kHz. The HTP-1 and JRiver compute in double precision,
so a filter is unchanged on them and they need no profile. A person can add their own. The storage models and the
magnitude calculation are beqforge's (`beqoptimiser`), so beqdesigner and its optimiser agree on what a device does.

Qt-free: the preferences it reads are given to it.
'''
import json
from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from model.preferences import DEVICE_PROFILES_CUSTOM, DISPLAY_DEVICE_PROFILES

RATES = (48000, 96000)
STORAGES = ('float32', 'fixed')
ERROR_BAND_HZ = (2.0, 200.0)   # where a BEQ acts, and where beqoptimiser judges its error


@dataclass(frozen=True)
class DeviceProfile:
    name: str
    rate: int
    storage: str              # 'float32' or 'fixed'
    integer_bits: int = 5     # fixed point only, including the sign bit
    fractional_bits: int = 23
    builtin: bool = False

    def __post_init__(self):
        if not self.name.strip():
            raise ValueError('a device profile needs a name')
        if self.rate not in RATES:
            raise ValueError(f'a device profile rate must be one of {", ".join(str(r) for r in RATES)}')
        if self.storage not in STORAGES:
            raise ValueError(f'a device profile storage must be one of {", ".join(STORAGES)}')
        if self.storage == 'fixed' and (not 1 <= self.integer_bits <= 16 or not 0 <= self.fractional_bits <= 48):
            raise ValueError('fixed point needs 1-16 integer bits and 0-48 fractional bits')

    @property
    def storage_name(self) -> str:
        return 'float32' if self.storage == 'float32' else f'Q{self.integer_bits}.{self.fractional_bits}'

    def describe(self) -> str:
        ''' "float32 at 96 kHz" '''
        return f'{self.storage_name} at {self.rate // 1000} kHz'

    def precision(self):
        from beqoptimiser import FixedPoint, Float32
        return Float32() if self.storage == 'float32' else FixedPoint(self.integer_bits, self.fractional_bits)

    def to_json(self) -> dict:
        data = asdict(self)
        data.pop('builtin')
        return data

    @classmethod
    def from_json(cls, data: dict) -> 'DeviceProfile':
        return cls(name=str(data['name']), rate=int(data['rate']), storage=str(data['storage']),
                   integer_bits=int(data.get('integer_bits', 5)), fractional_bits=int(data.get('fractional_bits', 23)))


def _builtin_profiles() -> Tuple[DeviceProfile, ...]:
    from model.dsp_type import DspType
    hd = [t.display_name for t in DspType if t.is_minidsp and t.hd_compatible]
    fixed = [t.display_name for t in DspType if t.is_minidsp and not t.hd_compatible]
    return (DeviceProfile(f'miniDSP {"/".join(hd)}', 96000, 'float32', builtin=True),
            DeviceProfile(f'miniDSP {"/".join(fixed)}', 48000, 'fixed', 5, 23, builtin=True))


BUILTIN_PROFILES = _builtin_profiles()


def custom_profiles(preferences) -> List[DeviceProfile]:
    ''' The user's own profiles; one that no longer reads is left out rather than failing the window. '''
    try:
        stored = json.loads(preferences.get(DEVICE_PROFILES_CUSTOM) or '[]')
    except ValueError:
        return []
    found = []
    for data in stored if isinstance(stored, list) else []:
        try:
            found.append(DeviceProfile.from_json(data))
        except (KeyError, TypeError, ValueError):
            continue
    return found


def all_profiles(preferences) -> List[DeviceProfile]:
    return [*BUILTIN_PROFILES, *custom_profiles(preferences)]


def save_custom_profiles(preferences, profiles: Sequence[DeviceProfile]) -> None:
    '''
    :raises ValueError: if two profiles (a custom one and a built-in included) share a name.
    '''
    names = [p.name for p in (*BUILTIN_PROFILES, *profiles)]
    duplicate = next((n for n in names if names.count(n) > 1), None)
    if duplicate is not None:
        raise ValueError(f'there is already a device profile called {duplicate!r}')
    preferences.set(DEVICE_PROFILES_CUSTOM, json.dumps([p.to_json() for p in profiles if not p.builtin]))
    # a profile no longer there is no longer shown
    shown = set(shown_profile_names(preferences)) & set(names)
    preferences.set(DISPLAY_DEVICE_PROFILES, json.dumps(sorted(shown)))


def shown_profile_names(preferences) -> List[str]:
    try:
        names = json.loads(preferences.get(DISPLAY_DEVICE_PROFILES) or '[]')
    except ValueError:
        return []
    return [str(n) for n in names] if isinstance(names, list) else []


def shown_profiles(preferences) -> List[DeviceProfile]:
    ''' The profiles the main chart shows the filter on, in the order they are listed. '''
    names = set(shown_profile_names(preferences))
    return [p for p in all_profiles(preferences) if p.name in names]


def set_shown(preferences, name: str, shown: bool) -> None:
    names = set(shown_profile_names(preferences))
    if shown:
        names.add(name)
    else:
        names.discard(name)
    preferences.set(DISPLAY_DEVICE_PROFILES, json.dumps(sorted(names)))


@dataclass(frozen=True)
class Realised:
    ''' A filter as a profile realises it, beside the filter as designed at the same rate. '''
    profile: DeviceProfile
    freqs: np.ndarray
    designed_db: np.ndarray        # double precision coefficients at the profile's rate
    realised_db: Optional[np.ndarray]   # the stored coefficients; None if they cannot be stored at all
    stable: bool
    problem: str = ''              # why there is no realised response, or why it is not to be trusted

    @property
    def max_error_db(self) -> Optional[float]:
        ''' The largest difference between realised and designed over ERROR_BAND_HZ; None if it cannot be realised. '''
        if self.realised_db is None:
            return None
        band = (self.freqs >= ERROR_BAND_HZ[0]) & (self.freqs <= ERROR_BAND_HZ[1])
        if not band.any():
            return None
        return float(np.max(np.abs(self.realised_db[band] - self.designed_db[band])))

    def describe(self) -> str:
        ''' "miniDSP 2x4 HD/... (float32 at 96 kHz): max error 2.37 dB (2-200 Hz)" '''
        head = f'{self.profile.name} ({self.profile.describe()})'
        if self.realised_db is None:
            return f'{head}: {self.problem}'
        error = self.max_error_db
        said = f'max error {error:.2f} dB ({ERROR_BAND_HZ[0]:g}-{ERROR_BAND_HZ[1]:g} Hz)' if error is not None else 'no error band'
        return f'{head}: {said}' + ('' if self.stable else ' - UNSTABLE as stored')


def realise(filt, profile: DeviceProfile, freqs: np.ndarray) -> Realised:
    '''
    :param filt: anything with resample(fs) whose result has get_sos() (a model.iir filter, or a CompleteFilter).
    :param freqs: where to calculate the responses, in Hz (below the profile's Nyquist).
    '''
    from beqoptimiser import magnitude, stable
    freqs = np.asarray(freqs, dtype=float)
    freqs = freqs[(freqs >= 0) & (freqs < profile.rate / 2)]
    at_rate = filt.resample(profile.rate) if getattr(filt, 'fs', None) != profile.rate else filt
    sos = np.asarray(at_rate.get_sos() or [], dtype=float)
    if sos.size == 0:
        flat = np.zeros_like(freqs)
        return Realised(profile, freqs, flat, flat, True)
    designed = magnitude(sos, freqs, profile.rate)
    try:
        stored = profile.precision().quantise(sos)
    except ValueError as error:
        return Realised(profile, freqs, designed, None, False, f'cannot be stored ({error})')
    stored[:, 3] = 1.0   # a0 is implicit on the device: it is never stored
    return Realised(profile, freqs, designed, magnitude(stored, freqs, profile.rate), stable(stored))
