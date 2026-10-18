'''
Optimising a filter for device profiles (model/device_optimise.py): the filter as beqforge's sections and back, a real
search on a filter float32 at 96 kHz damages, and when a result may be applied.
'''
import pytest

from model.device_optimise import Optimisation, ProfileOutcome, from_sections, optimise_for, to_sections
from model.device_profile import BUILTIN_PROFILES
from model.iir import CompleteFilter, Gain, LowShelf, PeakingEQ, SecondOrder_HighPass

HD, FIXED = BUILTIN_PROFILES


def test_peaks_and_shelves_become_sections_and_anything_else_is_held():
    from beqoptimiser import Fixed, Section
    peq, shelf, hpf = PeakingEQ(1000, 5, 6, 12), LowShelf(1000, 25, 0.8, 6, count=3), SecondOrder_HighPass(1000, 10)

    sections = to_sections([peq, shelf, hpf])

    assert sections[0] == Section('PeakingEQ', 5.0, 6.0, 12.0)
    assert sections[1:4] == [Section('LowShelf', 25.0, 0.8, 6.0)] * 3   # a shelf x3 is three identical sections
    assert isinstance(sections[4], Fixed) and len(sections) == 5
    assert list(sections[4].sos(96000)) == pytest.approx(hpf.resample(96000).get_sos()[0])


def test_the_new_parameters_replace_their_filters_by_id_keeping_a_shelfs_count():
    from beqoptimiser import Section
    peq, shelf, gain = PeakingEQ(1000, 5, 6, 12), LowShelf(1000, 25, 0.8, 6, count=2), Gain(1000, -3)
    new = [Section('PeakingEQ', 5.01, 6.12, 12.01), Section('LowShelf', 25.0, 0.8, 6.0),
           Section('LowShelf', 25.0, 0.8, 6.0), to_sections([gain])[0]]

    changed = from_sections([peq, shelf, gain], new)

    assert len(changed) == 1   # the shelf did not move, and a gain is never adjusted
    old, replacement = changed[0]
    assert old is peq and isinstance(replacement, PeakingEQ) and replacement.id == peq.id and replacement.fs == 1000
    assert (replacement.freq, replacement.q, replacement.gain) == (5.01, 6.12, 12.01)

    moved = from_sections([shelf], [Section('LowShelf', 25.1, 0.79, 6.02)] * 2)
    assert moved[0][1].count == 2 and moved[0][1].id == shelf.id and isinstance(moved[0][1], LowShelf)


def test_a_low_high_q_peak_is_brought_much_closer_on_float32_at_96khz():
    peak = PeakingEQ(1000, 5, 6, 12)

    result = optimise_for([peak], [HD])

    assert result.outcome in ('replacement', 'improvement') and result.applicable
    [p] = result.profiles
    assert p.before_db > 15 and p.after_db < p.before_db / 2
    [(old, new)] = result.changed
    assert abs(new.freq - 5) <= 0.05 and abs(new.q - 6) <= 0.12 + 1e-9 and abs(new.gain - 12) <= 0.1 + 1e-9
    assert new.freq == round(new.freq, 2) and new.q == round(new.q, 4) and new.gain == round(new.gain, 2)


def test_both_profiles_are_optimised_together_and_a_stacked_shelf_stays_stacked():
    filters = [PeakingEQ(1000, 5, 6, 12), LowShelf(1000, 25, 0.8, 6, count=2)]

    result = optimise_for(filters, [HD, FIXED])

    assert [p.profile for p in result.profiles] == [HD, FIXED]
    assert all(p.after_db is not None and p.after_db < p.before_db for p in result.profiles)
    shelves = [new for old, new in result.changed if isinstance(new, LowShelf)]
    assert all(s.count == 2 for s in shelves)


def test_a_filter_unstable_as_stored_is_said_so_not_called_unimprovable():
    hpf_at_3hz = [PeakingEQ(1000, 5, 6, 12), SecondOrder_HighPass(1000, 3)]   # float32 at 96 kHz makes it unstable

    result = optimise_for(hpf_at_3hz, [HD])

    assert not result.applicable and result.changed == ()
    assert 'unstable or cannot be realised' in result.describe() and HD.name in result.describe()


def test_a_cascade_with_a_high_pass_raises_no_numpy_warnings():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        result = optimise_for([PeakingEQ(1000, 5, 6, 12), SecondOrder_HighPass(1000, 10)], [FIXED])
    assert result.profiles[0].before_db != float('inf')


def test_a_result_that_makes_any_profile_worse_is_not_applicable():
    old, new = PeakingEQ(1000, 5, 6, 12), PeakingEQ(1000, 5.01, 6, 12)
    better, worse = ProfileOutcome(HD, 10, 2), ProfileOutcome(FIXED, 1, 1.5)

    assert Optimisation('improvement', 0.5, (better,), ((old, new),)).applicable
    assert not Optimisation('improvement', 0.5, (better, worse), ((old, new),)).applicable
    assert not Optimisation('no_replacement', 0.5, (better,), ((old, new),)).applicable
    assert not Optimisation('improvement', 0.5, (better,), ()).applicable


def test_it_needs_a_profile():
    with pytest.raises(ValueError, match='at least one device profile'):
        optimise_for(CompleteFilter(fs=1000, filters=[PeakingEQ(1000, 5, 6, 12)]), [])
