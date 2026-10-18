'''
What a filter becomes on a device (model/device_profile.py): built-in profiles for the devices beqdesigner exports to, a
person's own, and the filter's response as a profile stores its coefficients, against the filter as designed.
'''
import json

import numpy as np
import pytest

from model.device_profile import BUILTIN_PROFILES, DeviceProfile, all_profiles, custom_profiles, realise, \
    save_custom_profiles, set_shown, shown_profile_names, shown_profiles
from model.iir import CompleteFilter, LowShelf, PeakingEQ
from model.preferences import DEFAULT_PREFS, DEVICE_PROFILES_CUSTOM, DISPLAY_DEVICE_PROFILES

FREQS = np.linspace(1, 250, 1000)


class Prefs:
    ''' Preferences' get/set, over a dict: model/device_profile.py is Qt-free and needs no QSettings. '''

    def __init__(self, **values):
        self.values = dict(values)

    def get(self, key):
        return self.values.get(key, DEFAULT_PREFS.get(key))

    def set(self, key, value):
        self.values[key] = value


def _filter(*filters, fs=1000):
    return CompleteFilter(fs=fs, filters=list(filters))


HD, FIXED = BUILTIN_PROFILES


def test_the_built_in_profiles_are_the_minidsp_families_by_how_they_store_coefficients():
    assert (HD.rate, HD.storage, HD.builtin) == (96000, 'float32', True)
    assert (FIXED.rate, FIXED.storage, FIXED.integer_bits, FIXED.fractional_bits) == (48000, 'fixed', 5, 23)
    assert '2x4 HD' in HD.name and 'SHD' in HD.name and '10x10' in FIXED.name and '2x4 HD' not in FIXED.name
    assert HD.describe() == 'float32 at 96 kHz' and FIXED.describe() == 'Q5.23 at 48 kHz'


def test_a_low_high_q_peak_is_badly_damaged_by_float32_at_96khz():
    ''' The F2 pilot's case: float32 coefficients put many dB of error into a 5 Hz Q 6 peak at 96 kHz. '''
    realised = realise(_filter(PeakingEQ(1000, 5, 6, 12)), HD, FREQS)

    assert realised.stable and realised.max_error_db > 10
    at_5hz = np.argmin(np.abs(realised.freqs - 5))
    assert realised.designed_db[at_5hz] == pytest.approx(12, abs=0.1)   # the design is the filter as asked for
    assert realised.realised_db[at_5hz] < 0                             # the device's is not


def test_an_ordinary_shelf_is_barely_changed():
    realised = realise(_filter(LowShelf(1000, 40, 0.7, 6)), FIXED, FREQS)
    assert realised.max_error_db < 0.05 and realised.stable


def test_more_fractional_bits_keep_a_peak_closer_to_its_design():
    coarse = DeviceProfile('coarse', 48000, 'fixed', 5, 16)
    fine = DeviceProfile('fine', 48000, 'fixed', 5, 30)
    peak = _filter(PeakingEQ(1000, 10, 4, 10))
    assert realise(peak, fine, FREQS).max_error_db < realise(peak, coarse, FREQS).max_error_db


def test_coefficients_outside_fixed_point_range_cannot_be_stored():
    tiny = DeviceProfile('tiny', 48000, 'fixed', 1, 23)   # [-0.5, 0.5): a biquad's a1 of about -2 does not fit
    realised = realise(_filter(PeakingEQ(1000, 30, 1, 6)), tiny, FREQS)

    assert realised.realised_db is None and realised.max_error_db is None
    assert 'cannot be stored' in realised.describe()


def test_no_filter_is_flat_on_every_device():
    realised = realise(_filter(), HD, FREQS)
    assert realised.max_error_db == 0


def test_the_summary_line_names_the_profile_its_storage_and_its_error():
    said = realise(_filter(PeakingEQ(1000, 5, 6, 12)), HD, FREQS).describe()
    assert said.startswith(f'{HD.name} (float32 at 96 kHz): max error ') and said.endswith(' dB (2-200 Hz)')


@pytest.mark.parametrize('fields, message', [
    (dict(name=' ', rate=48000, storage='float32'), 'needs a name'),
    (dict(name='x', rate=44100, storage='float32'), 'rate must be one of 48000, 96000'),
    (dict(name='x', rate=48000, storage='float16'), 'storage must be one of'),
    (dict(name='x', rate=48000, storage='fixed', integer_bits=0), 'integer bits'),
])
def test_a_profile_that_makes_no_sense_is_refused(fields, message):
    with pytest.raises(ValueError, match=message):
        DeviceProfile(**fields)


def test_custom_profiles_are_kept_in_the_preferences_and_listed_after_the_built_ins():
    prefs = Prefs()
    mine = DeviceProfile('Behringer DCX', 96000, 'fixed', 8, 24)
    save_custom_profiles(prefs, [mine])

    assert json.loads(prefs.values[DEVICE_PROFILES_CUSTOM]) == [
        {'name': 'Behringer DCX', 'rate': 96000, 'storage': 'fixed', 'integer_bits': 8, 'fractional_bits': 24}]
    assert custom_profiles(prefs) == [mine]
    assert all_profiles(prefs) == [*BUILTIN_PROFILES, mine]


def test_two_profiles_cannot_share_a_name():
    with pytest.raises(ValueError, match='already a device profile called'):
        save_custom_profiles(Prefs(), [DeviceProfile(HD.name, 48000, 'float32')])


def test_an_unreadable_stored_profile_is_left_out_not_fatal():
    prefs = Prefs(**{DEVICE_PROFILES_CUSTOM: json.dumps([{'name': 'ok', 'rate': 48000, 'storage': 'float32'},
                                                         {'name': 'bad', 'rate': 1}, 'nonsense'])})
    assert [p.name for p in custom_profiles(prefs)] == ['ok']
    assert custom_profiles(Prefs(**{DEVICE_PROFILES_CUSTOM: '{not json'})) == []


def test_profiles_are_shown_and_hidden_by_name_and_a_deleted_one_stops_being_shown():
    prefs = Prefs()
    mine = DeviceProfile('mine', 48000, 'float32')
    save_custom_profiles(prefs, [mine])
    set_shown(prefs, HD.name, True)
    set_shown(prefs, 'mine', True)
    assert shown_profiles(prefs) == [HD, mine]

    set_shown(prefs, HD.name, False)
    assert shown_profile_names(prefs) == ['mine']
    save_custom_profiles(prefs, [])
    assert shown_profile_names(prefs) == [] and json.loads(prefs.values[DISPLAY_DEVICE_PROFILES]) == []
