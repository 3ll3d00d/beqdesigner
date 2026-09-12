import pytest

from model.iir import CompleteFilter, PeakingEQ, LowShelf
from model.signal import SingleChannelSignalData


def make_signal(fs=1000):
    filt = CompleteFilter(fs=fs)
    filt.save(PeakingEQ(fs, 100, 1.0, 3.0))
    avg = LowShelf(fs, 30, 1, 10).get_transfer_function().get_magnitude()
    peak = LowShelf(fs, 30, 1, 10, count=2).get_transfer_function().get_magnitude()
    return SingleChannelSignalData('test', fs, xy_data=[avg, peak], filter=filt)


def test_default_signal_has_single_default_preset():
    signal = make_signal()
    assert list(signal.filter_presets.keys()) == ['Default']
    assert signal.active_filter_preset == 'Default'
    assert signal.filter is signal.filter_presets['Default']


def test_add_filter_preset_does_not_activate_it():
    signal = make_signal()
    signal.add_filter_preset('Movie')
    assert set(signal.filter_presets.keys()) == {'Default', 'Movie'}
    assert signal.active_filter_preset == 'Default'
    assert len(signal.filter_presets['Movie']) == 0


def test_add_filter_preset_duplicate_name_raises():
    signal = make_signal()
    signal.add_filter_preset('Movie')
    with pytest.raises(ValueError):
        signal.add_filter_preset('Movie')


def test_activate_filter_preset_updates_active_filter():
    signal = make_signal()
    signal.add_filter_preset('Movie')
    signal.activate_filter_preset('Movie')
    assert signal.active_filter_preset == 'Movie'
    assert signal.filter is signal.filter_presets['Movie']
    assert len(signal.filter) == 0


def test_activate_unknown_filter_preset_raises():
    signal = make_signal()
    with pytest.raises(ValueError):
        signal.activate_filter_preset('nope')


def test_editing_active_filter_updates_the_stored_preset():
    signal = make_signal()
    signal.add_filter_preset('Movie')
    signal.activate_filter_preset('Movie')
    new_filter = CompleteFilter(fs=signal.fs)
    new_filter.save(PeakingEQ(signal.fs, 50, 1.0, -2.0))
    signal.filter = new_filter
    assert signal.filter_presets['Movie'] is new_filter
    # switching away and back to Default should leave it untouched
    signal.activate_filter_preset('Default')
    assert len(signal.filter) == 1


def test_duplicate_filter_preset_copies_content_without_activating():
    signal = make_signal()
    signal.duplicate_filter_preset('Default', 'Music')
    assert signal.active_filter_preset == 'Default'
    assert 'Music' in signal.filter_presets
    assert signal.filter_presets['Music'] is not signal.filter_presets['Default']
    assert len(signal.filter_presets['Music']) == len(signal.filter_presets['Default'])


def test_duplicate_filter_preset_missing_source_raises():
    signal = make_signal()
    with pytest.raises(ValueError):
        signal.duplicate_filter_preset('nope', 'Music')


def test_duplicate_filter_preset_existing_target_raises():
    signal = make_signal()
    with pytest.raises(ValueError):
        signal.duplicate_filter_preset('Default', 'Default')


def test_rename_filter_preset_preserves_order_and_active_state():
    signal = make_signal()
    signal.add_filter_preset('Movie')
    signal.add_filter_preset('Music')
    signal.activate_filter_preset('Movie')
    signal.rename_filter_preset('Movie', 'Cinema')
    assert list(signal.filter_presets.keys()) == ['Default', 'Cinema', 'Music']
    assert signal.active_filter_preset == 'Cinema'


def test_rename_filter_preset_existing_target_raises():
    signal = make_signal()
    signal.add_filter_preset('Movie')
    with pytest.raises(ValueError):
        signal.rename_filter_preset('Movie', 'Default')


def test_remove_filter_preset_reactivates_another_when_active_removed():
    signal = make_signal()
    signal.add_filter_preset('Movie')
    signal.activate_filter_preset('Movie')
    signal.remove_filter_preset('Movie')
    assert 'Movie' not in signal.filter_presets
    assert signal.active_filter_preset == 'Default'
    assert signal.filter is signal.filter_presets['Default']


def test_remove_last_filter_preset_raises():
    signal = make_signal()
    with pytest.raises(ValueError):
        signal.remove_filter_preset('Default')


def test_remove_unknown_filter_preset_raises():
    signal = make_signal()
    signal.add_filter_preset('Movie')
    with pytest.raises(ValueError):
        signal.remove_filter_preset('nope')
