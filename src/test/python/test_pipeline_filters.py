'''
Phase 1 (B3) of design/pipeline-implementation-plan.md: FilterSpec + create_filter().
'''
import math

import pytest


def test_peaking_eq_round_trip():
    from pipeline.filters import FilterSpec, create_filter
    from model.iir import PeakingEQ

    spec = FilterSpec(type='peaking_eq', freq=40.0, gain=-3.0, q=2.0)
    filt = create_filter(spec, fs=1000)

    assert isinstance(filt, PeakingEQ)
    assert filt.freq == 40.0
    assert filt.gain == -3.0
    assert filt.q == 2.0


def test_low_shelf_with_q():
    from pipeline.filters import FilterSpec, create_filter
    from model.iir import LowShelf

    spec = FilterSpec(type='low_shelf', freq=18.0, gain=4.5, q=0.7, count=5)
    filt = create_filter(spec, fs=1000)

    assert isinstance(filt, LowShelf)
    assert filt.freq == 18.0
    assert filt.gain == 4.5
    assert filt.q == 0.7
    assert filt.count == 5


def test_low_shelf_with_s_matches_q_construction():
    '''
    docs/workflow/beq.md documents shelves in S -- constructing via s should
    produce the exact same biquad as constructing the equivalent q directly.
    '''
    from pipeline.filters import FilterSpec, create_filter
    from model.iir import s_to_q

    gain = 4.5
    s = 2.2
    q = s_to_q(s, gain)

    via_s = create_filter(FilterSpec(type='low_shelf', freq=18.0, gain=gain, s=s), fs=1000)
    via_q = create_filter(FilterSpec(type='low_shelf', freq=18.0, gain=gain, q=q), fs=1000)

    assert math.isclose(via_s.q, via_q.q, rel_tol=1e-12)
    assert via_s.get_sos() == via_q.get_sos()


def test_high_shelf_with_s():
    from pipeline.filters import FilterSpec, create_filter
    from model.iir import HighShelf

    spec = FilterSpec(type='high_shelf', freq=8000.0, gain=-2.0, s=1.0)
    filt = create_filter(spec, fs=48000)
    assert isinstance(filt, HighShelf)


def test_gain_filter():
    from pipeline.filters import FilterSpec, create_filter
    from model.iir import Gain

    filt = create_filter(FilterSpec(type='gain', gain=-4.0), fs=1000)
    assert isinstance(filt, Gain)
    assert filt.gain == -4.0


def test_variable_q_lpf():
    from pipeline.filters import FilterSpec, create_filter
    from model.iir import SecondOrder_LowPass

    filt = create_filter(FilterSpec(type='variable_q_lpf', freq=120.0, q=0.9), fs=1000)
    assert isinstance(filt, SecondOrder_LowPass)
    assert filt.freq == 120.0
    assert filt.q == 0.9


def test_variable_q_hpf():
    from pipeline.filters import FilterSpec, create_filter
    from model.iir import SecondOrder_HighPass

    filt = create_filter(FilterSpec(type='variable_q_hpf', freq=30.0, q=0.6), fs=1000)
    assert isinstance(filt, SecondOrder_HighPass)
    assert filt.freq == 30.0
    assert filt.q == 0.6


def test_all_pass():
    from pipeline.filters import FilterSpec, create_filter
    from model.iir import AllPass

    filt = create_filter(FilterSpec(type='all_pass', freq=50.0, q=1.5), fs=1000)
    assert isinstance(filt, AllPass)
    assert filt.freq == 50.0
    assert filt.q == 1.5


def test_resolved_q_and_s_report_both_regardless_of_which_was_supplied():
    from pipeline.filters import FilterSpec
    from model.iir import q_to_s, s_to_q

    by_q = FilterSpec(type='low_shelf', freq=18.0, gain=4.5, q=0.7)
    assert by_q.resolved_q() == 0.7
    assert math.isclose(by_q.resolved_s(), q_to_s(0.7, 4.5))

    by_s = FilterSpec(type='low_shelf', freq=18.0, gain=4.5, s=2.2)
    assert by_s.resolved_s() == 2.2
    assert math.isclose(by_s.resolved_q(), s_to_q(2.2, 4.5))


@pytest.mark.parametrize('spec_kwargs,match', [
    ({'type': 'low_shelf', 'freq': 18.0, 'gain': 4.5, 'q': 0.7, 's': 2.2}, 'both q and s'),
    ({'type': 'peaking_eq', 'freq': 40.0, 'gain': -3.0}, 'needs q'),
    ({'type': 'peaking_eq', 'freq': 40.0, 'gain': -3.0, 's': 2.2}, 'no shelf slope'),
    ({'type': 'low_shelf', 'gain': 4.5, 'q': 0.7}, 'needs freq'),
    ({'type': 'variable_q_lpf', 'freq': 120.0}, 'needs q'),
    ({'type': 'variable_q_hpf', 'freq': 30.0, 's': 2.2}, 'no shelf slope'),
    ({'type': 'all_pass', 'freq': 50.0}, 'needs q'),
])
def test_invalid_specs_are_rejected(spec_kwargs, match):
    from pipeline.filters import FilterSpec, create_filter
    with pytest.raises(ValueError, match=match):
        create_filter(FilterSpec(**spec_kwargs), fs=1000)


def test_pipeline_filters_module_has_no_qtpy_import():
    import ast
    import pathlib
    source = (pathlib.Path(__file__).parent / '..' / '..' / 'main' / 'python' / 'pipeline' / 'filters.py').resolve()
    tree = ast.parse(source.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(n.name.startswith('qtpy') for n in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith('qtpy')
