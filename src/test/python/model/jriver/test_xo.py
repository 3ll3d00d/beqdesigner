from typing import Dict

import pytest

from model.iir import ComplexLowPass, FilterType, ComplexHighPass
from model.jriver import ImpossibleRoutingError, UnsupportedRoutingError, JRIVER_FS
from model.jriver.common import get_channel_idx
from model.jriver.filter import Mix, MixType, Gain, MultiwayFilter, MultiwayCrossover, StandardXO
from model.jriver.routing import Matrix, calculate_compound_routing_filter


def lpf(freq: float) -> ComplexLowPass:
    return ComplexLowPass(FilterType.BUTTERWORTH, 2, JRIVER_FS, freq)


def hpf(freq: float) -> ComplexHighPass:
    return ComplexHighPass(FilterType.BUTTERWORTH, 2, JRIVER_FS, freq)


@pytest.fixture
def stereo() -> Matrix:
    return Matrix({'L': 1, 'R': 1}, ['L', 'R'])


def test_stereo_passthrough(stereo):
    stereo.enable('L', 0, 'L')
    stereo.enable('R', 0, 'R')
    routing_filters = calculate_compound_routing_filter(stereo).filters
    assert not routing_filters


def test_stereo_swap(stereo):
    stereo.enable('L', 0, 'R')
    stereo.enable('R', 0, 'L')

    with pytest.raises(ImpossibleRoutingError):
        calculate_compound_routing_filter(stereo)


@pytest.fixture
def two_in_four() -> Matrix:
    return Matrix({'L': 2, 'R': 2}, ['L', 'R', 'C', 'SW'])


def test_two_in_four(two_in_four):
    two_in_four.enable('L', 0, 'L')
    two_in_four.enable('L', 1, 'C')
    two_in_four.enable('R', 0, 'R')
    two_in_four.enable('R', 1, 'SW')
    mw_filters = [
        MultiwayFilter('L', ['L', 'C'], MultiwayCrossover('L', [StandardXO(['L'], ['C'], low_pass=lpf(2000), high_pass=hpf(2000))], []).graph.filters, {}),
        MultiwayFilter('R', ['R', 'SW'], MultiwayCrossover('R', [StandardXO(['R'], ['SW'], low_pass=lpf(2000), high_pass=hpf(2000))], []).graph.filters, {})
    ]
    mc_filters = calculate_compound_routing_filter(two_in_four, xo_filters=mw_filters).filters
    assert mc_filters
    assert all(isinstance(x, MultiwayFilter) for x in mc_filters)

    f = mc_filters.pop(0)
    assert f.filters.pop(0).get_all_vals() == [mix('L', MixType.COPY, 'U2')]
    assert str(f.filters.pop(0)) == 'HP BW2 2kHz  [U2]'
    assert f.filters.pop(0).get_all_vals() == [mix('U2', MixType.COPY, 'C')]
    assert str(f.filters.pop(0)) == 'LP BW2 2kHz  [L]'
    assert not f.filters

    f = mc_filters.pop(0)
    assert f.filters.pop(0).get_all_vals() == [mix('R', MixType.COPY, 'U2')]
    assert str(f.filters.pop(0)) == 'HP BW2 2kHz  [U2]'
    assert f.filters.pop(0).get_all_vals() == [mix('U2', MixType.COPY, 'SW')]
    assert str(f.filters.pop(0)) == 'LP BW2 2kHz  [R]'
    assert not f.filters
    assert not mc_filters


def test_two_in_four_in_order(two_in_four):
    two_in_four.enable('L', 0, 'L')
    two_in_four.enable('L', 1, 'R')
    two_in_four.enable('R', 0, 'C')
    two_in_four.enable('R', 1, 'SW')
    with pytest.raises(ImpossibleRoutingError):
        calculate_compound_routing_filter(two_in_four)


def test_two_in_four_circular(two_in_four):
    two_in_four.enable('L', 0, 'R')
    two_in_four.enable('L', 1, 'C')
    two_in_four.enable('R', 0, 'L')
    two_in_four.enable('R', 1, 'SW')
    with pytest.raises(ImpossibleRoutingError):
        calculate_compound_routing_filter(two_in_four)


@pytest.fixture
def two_in_eight() -> Matrix:
    return Matrix({'L': 4, 'R': 4}, ['L', 'R', 'C', 'SW', 'SL', 'SR', 'RL', 'RR'])


def test_two_in_eight_passthrough(two_in_eight):
    two_in_eight.enable('L', 0, 'L')
    two_in_eight.enable('L', 1, 'C')
    two_in_eight.enable('L', 2, 'SW')
    two_in_eight.enable('L', 3, 'SL')
    two_in_eight.enable('R', 0, 'R')
    two_in_eight.enable('R', 1, 'SR')
    two_in_eight.enable('R', 2, 'RL')
    two_in_eight.enable('R', 3, 'RR')
    mw_filters = [
        MultiwayFilter('L',
                       ['L', 'C', 'SW', 'SL'],
                       MultiwayCrossover('L', [StandardXO(['L'], ['C']), StandardXO(['C'], ['SW']), StandardXO(['SW'], ['SL'])], []).graph.filters, {}),
        MultiwayFilter('R',
                       ['R', 'SR', 'RL', 'RR'],
                       MultiwayCrossover('R', [StandardXO(['R'], ['SR']), StandardXO(['SR'], ['RL']), StandardXO(['RL'], ['RR'])], []).graph.filters, {}),
    ]
    mc_filters = calculate_compound_routing_filter(two_in_eight, xo_filters=mw_filters).filters
    assert mc_filters
    assert all(isinstance(x, MultiwayFilter) for x in mc_filters)
    f = mc_filters.pop(0)
    assert f.filters.pop(0).get_all_vals() == [mix('L', MixType.COPY, 'C')]
    assert f.filters.pop(0).get_all_vals() == [mix('C', MixType.COPY, 'SW')]
    assert f.filters.pop(0).get_all_vals() == [mix('SW', MixType.COPY, 'SL')]
    assert not f.filters
    f = mc_filters.pop(0)
    assert f.filters.pop(0).get_all_vals() == [mix('R', MixType.COPY, 'SR')]
    assert f.filters.pop(0).get_all_vals() == [mix('SR', MixType.COPY, 'RL')]
    assert f.filters.pop(0).get_all_vals() == [mix('RL', MixType.COPY, 'RR')]
    assert not f.filters
    assert not mc_filters


def test_two_in_eight_in_order(two_in_eight):
    two_in_eight.enable('L', 0, 'L')
    two_in_eight.enable('L', 1, 'R')
    two_in_eight.enable('L', 2, 'C')
    two_in_eight.enable('L', 3, 'SW')
    two_in_eight.enable('R', 0, 'SL')
    two_in_eight.enable('R', 1, 'SR')
    two_in_eight.enable('R', 2, 'RL')
    two_in_eight.enable('R', 3, 'RR')
    with pytest.raises(ImpossibleRoutingError):
        calculate_compound_routing_filter(two_in_eight)


def test_two_in_eight_shared_sub(two_in_eight):
    two_in_eight.enable('L', 0, 'SW')
    two_in_eight.enable('L', 1, 'L')
    two_in_eight.enable('L', 2, 'C')
    two_in_eight.enable('L', 3, 'SL')
    two_in_eight.enable('R', 0, 'SW')
    two_in_eight.enable('R', 1, 'R')
    two_in_eight.enable('R', 2, 'SR')
    two_in_eight.enable('R', 3, 'RL')
    mw_filters = [
        MultiwayFilter('L',
                       ['SW', 'L', 'C', 'SL'],
                       MultiwayCrossover('L', [StandardXO(['SW'], ['L']), StandardXO(['L'], ['C']), StandardXO(['C'], ['SL'])], []).graph.filters, {}),
        MultiwayFilter('R',
                       ['SW', 'R', 'SR', 'RL'],
                       MultiwayCrossover('R', [StandardXO(['SW'], ['R']), StandardXO(['R'], ['SR']), StandardXO(['SR'], ['RL'])], []).graph.filters, {}),
    ]
    mc_filters = calculate_compound_routing_filter(two_in_eight, xo_filters=mw_filters).filters
    assert mc_filters
    assert all(isinstance(x, MultiwayFilter) for x in mc_filters)
    f = mc_filters.pop(0)
    assert f.filters.pop(0).get_all_vals() == [mix('L', MixType.ADD, 'SW')]
    assert f.filters.pop(0).get_all_vals() == [mix('L', MixType.COPY, 'C')]
    assert f.filters.pop(0).get_all_vals() == [mix('C', MixType.COPY, 'SL')]
    assert not f.filters
    f = mc_filters.pop(0)
    assert f.filters.pop(0).get_all_vals() == [mix('R', MixType.ADD, 'SW')]
    assert f.filters.pop(0).get_all_vals() == [mix('R', MixType.COPY, 'SR')]
    assert f.filters.pop(0).get_all_vals() == [mix('SR', MixType.COPY, 'RL')]
    assert not f.filters
    assert not mc_filters


@pytest.fixture
def five_one() -> Matrix:
    return Matrix({'L': 2, 'R': 2, 'C': 2, 'SW': 1, 'SL': 2, 'SR': 2}, ['L', 'R', 'C', 'SW', 'SL', 'SR'])


def test_five_one_passthrough_bass_manage(five_one):
    for c in ['L', 'R', 'C', 'SW', 'SL', 'SR']:
        five_one.enable(c, 0, 'SW')
        if c != 'SW':
            five_one.enable(c, 1, c)

    main_channels = ['L', 'R', 'C', 'SL', 'SR']
    mw_filters = __make_bm_multiway(main_channels)
    mc_filters = calculate_compound_routing_filter(five_one, xo_filters=mw_filters, main_adjust=-15, lfe_adjust=-5, lfe_channel_idx=5).filters
    assert mc_filters
    assert all(isinstance(x, (MultiwayFilter, Gain)) for x in mc_filters)
    assert mc_filters.pop(0).get_all_vals() == [gain('SW', -5)]
    for c in main_channels:
        assert mc_filters.pop(0).get_all_vals() == [gain(c, -15)]
    for c in main_channels:
        f = mc_filters.pop(0)
        assert f.filters.pop(0).get_all_vals() == [mix(c, MixType.ADD, 'SW')]
        assert not f.filters
    assert not mc_filters


def __make_bm_multiway(main_channels):
    return [MultiwayFilter(c, [c, 'SW'], MultiwayCrossover(c, [StandardXO(['SW'], [c])], []).graph.filters, {}) for c in main_channels]


def test_five_one_swap_main(five_one):
    for c in ['L', 'R', 'C', 'SW', 'SL', 'SR']:
        five_one.enable(c, 0, 'SW')
        if c not in ['SW', 'SL', 'SR']:
            five_one.enable(c, 1, c)
    five_one.enable('SL', 1, 'SR')
    five_one.enable('SR', 1, 'SL')
    with pytest.raises(ImpossibleRoutingError):
        calculate_compound_routing_filter(five_one, lfe_channel_idx=5)


def test_five_one_swap_sub_to_c(five_one):
    for c in ['L', 'R', 'C', 'SW', 'SL', 'SR']:
        five_one.enable(c, 0, 'C')
        if c not in ['C', 'SW']:
            five_one.enable(c, 1, c)
    five_one.enable('C', 1, 'SW')
    with pytest.raises(ImpossibleRoutingError):
        calculate_compound_routing_filter(five_one, lfe_channel_idx=5)


@pytest.fixture
def multi_bm() -> Matrix:
    return Matrix({'L': 4, 'R': 4, 'C': 4, 'SW': 1, 'SL': 2, 'SR': 2, 'RL': 2, 'RR': 2},
                  ['L', 'R', 'C', 'SW', 'SL', 'SR', 'RL', 'RR', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16'])


def test_multi_way_mains_simple_bm(multi_bm):
    multiway_channels = [('L', 'C9', 'C10'), ('R', 'C11', 'C12'), ('C', 'C13', 'C14')]
    other_channels = ['SL', 'SR', 'RL', 'RR']

    for vals in multiway_channels:
        multi_bm.enable(vals[0], 0, 'SW')
        for i, v in enumerate(vals):
            multi_bm.enable(vals[0], i + 1, v)
    for c in other_channels:
        multi_bm.enable(c, 0, 'SW')
        multi_bm.enable(c, 1, c)
    multi_bm.enable('SW', 0, 'SW')

    mw_filters = []
    for vals in multiway_channels:
        mw_filters.append(
            MultiwayFilter(vals[0],
                           ['SW'] + list(vals),
                           MultiwayCrossover(vals[0], [StandardXO(['SW'], [vals[0]]), StandardXO([vals[0]], [vals[1]]), StandardXO([vals[1]], [vals[2]])], []).graph.filters, {}),
        )

    mw_filters = mw_filters + __make_bm_multiway(other_channels)
    mc_filters = calculate_compound_routing_filter(multi_bm, xo_filters=mw_filters, lfe_channel_idx=5).filters

    assert all(isinstance(x, MultiwayFilter) for x in mc_filters)

    for vals in multiway_channels:
        f = mc_filters.pop(0)
        assert f.filters.pop(0).get_all_vals() == [mix(vals[0], MixType.ADD, 'SW')]
        assert f.filters.pop(0).get_all_vals() == [mix(vals[0], MixType.COPY, vals[1])]
        assert f.filters.pop(0).get_all_vals() == [mix(vals[1], MixType.COPY, vals[2])]
        assert not f.filters

    for c in other_channels:
        f = mc_filters.pop(0)
        assert f.filters.pop(0).get_all_vals() == [mix(c, MixType.ADD, 'SW')]
        assert not f.filters

    assert not mc_filters


def test_multi_way_mains_simple_bm_with_shared_sub(multi_bm):
    multiway_channels = [('L', 'C9', 'C10'), ('R', 'C11', 'C12'), ('C', 'C13', 'C14')]
    other_channels = ['SL', 'SR', 'RL', 'RR']

    for vals in multiway_channels:
        multi_bm.enable(vals[0], 0, 'SW')
        if vals[0] != 'C':
            multi_bm.enable(vals[0], 0, 'C15')
        for i, v in enumerate(vals):
            multi_bm.enable(vals[0], i + 1, v)
    for c in other_channels:
        multi_bm.enable(c, 0, 'SW')
        multi_bm.enable(c, 1, c)
    multi_bm.enable('SW', 0, 'SW')

    mw_filters = []
    for vals in multiway_channels:
        if vals[0] != 'C':
            mw_filters.append(
                MultiwayFilter(vals[0],
                               ['SW', 'C15'] + list(vals),
                               MultiwayCrossover(vals[0], [StandardXO(['SW', 'C15'], [vals[0]]), StandardXO([vals[0]], [vals[1]]), StandardXO([vals[1]], [vals[2]])], []).graph.filters, {}),
            )
        else:
            mw_filters.append(
                MultiwayFilter(vals[0],
                               ['SW'] + list(vals),
                               MultiwayCrossover(vals[0], [StandardXO(['SW'], [vals[0]]), StandardXO([vals[0]], [vals[1]]), StandardXO([vals[1]], [vals[2]])], []).graph.filters, {}),
            )

    mw_filters = mw_filters + __make_bm_multiway(other_channels)
    mc_filters = calculate_compound_routing_filter(multi_bm, xo_filters=mw_filters, lfe_channel_idx=5).filters

    assert all(isinstance(x, MultiwayFilter) for x in mc_filters)

    for vals in multiway_channels:
        f = mc_filters.pop(0)
        assert f.filters.pop(0).get_all_vals() == [mix(vals[0], MixType.ADD, 'SW')]
        if vals[0] != 'C':
            assert f.filters.pop(0).get_all_vals() == [mix(vals[0], MixType.ADD, 'C15')]
        assert f.filters.pop(0).get_all_vals() == [mix(vals[0], MixType.COPY, vals[1])]
        assert f.filters.pop(0).get_all_vals() == [mix(vals[1], MixType.COPY, vals[2])]
        assert not f.filters

    for c in other_channels:
        f = mc_filters.pop(0)
        assert f.filters.pop(0).get_all_vals() == [mix(c, MixType.ADD, 'SW')]
        assert not f.filters

    assert not mc_filters


@pytest.fixture
def seven_one() -> Matrix:
    return Matrix({'L': 2, 'R': 2, 'C': 2, 'SW': 1, 'SL': 2, 'SR': 2},
                  ['L', 'R', 'C', 'SW', 'SL', 'SR', 'RL', 'RR'])


def test_stereo_subs_overwrite(seven_one):
    __map_stereo_subs_bm(seven_one, 'SW')

    mw_filters = \
        [MultiwayFilter(c, [c, 'SW'], MultiwayCrossover(c, [StandardXO(['SW'], [c])], []).graph.filters, {}) for c in ['L', 'SL']] + \
        [MultiwayFilter(c, [c, 'RR'], MultiwayCrossover(c, [StandardXO(['RR'], [c])], []).graph.filters, {}) for c in ['R', 'SR']] + \
        [MultiwayFilter('C', ['C', 'SW', 'RR'], MultiwayCrossover('C', [StandardXO(['SW', 'RR'], ['C'])], []).graph.filters, {})]
    with pytest.raises(UnsupportedRoutingError):
        calculate_compound_routing_filter(seven_one, xo_filters=mw_filters, lfe_channel_idx=5)


def __map_stereo_subs_bm(stereo_subs: Matrix, sub1: str) -> None:
    for c in ['L', 'C', 'SW', 'SL']:
        stereo_subs.enable(c, 0, sub1)
    for c in ['R', 'C', 'SW', 'SR']:
        stereo_subs.enable(c, 0, 'RR')
    for c in ['L', 'R', 'C', 'SL', 'SR']:
        stereo_subs.enable(c, 1, c)


def test_stereo_subs(seven_one):
    __map_stereo_subs_bm(seven_one, 'RL')

    mw_filters = \
        [MultiwayFilter(c, [c, 'RL'], MultiwayCrossover(c, [StandardXO(['RL'], [c])], []).graph.filters, {}) for c in ['L', 'SL']] + \
        [MultiwayFilter(c, [c, 'RR'], MultiwayCrossover(c, [StandardXO(['RR'], [c])], []).graph.filters, {}) for c in ['R', 'SR']] + \
        [MultiwayFilter('C', ['C', 'RL', 'RR'], MultiwayCrossover('C', [StandardXO(['RL', 'RR'], ['C'])], []).graph.filters, {})]
    mc_filters = calculate_compound_routing_filter(seven_one, xo_filters=mw_filters, lfe_channel_idx=5).filters
    assert mc_filters
    assert all(isinstance(x, MultiwayFilter) for x in mc_filters)

    f = mc_filters.pop(0)
    assert f.filters.pop(0).get_all_vals() == [mix('L', MixType.ADD, 'RL')]
    assert not f.filters
    f = mc_filters.pop(0)
    assert f.filters.pop(0).get_all_vals() == [mix('SL', MixType.ADD, 'RL')]
    assert not f.filters
    f = mc_filters.pop(0)
    assert f.filters.pop(0).get_all_vals() == [mix('R', MixType.ADD, 'RR')]
    assert not f.filters
    f = mc_filters.pop(0)
    assert f.filters.pop(0).get_all_vals() == [mix('SR', MixType.ADD, 'RR')]
    assert not f.filters
    f = mc_filters.pop(0)
    assert f.filters.pop(0).get_all_vals() == [mix('C', MixType.ADD, 'RL')]
    assert f.filters.pop(0).get_all_vals() == [mix('C', MixType.ADD, 'RR')]
    assert not f.filters

    assert not mc_filters


def test_collapse_surround():
    five_one = Matrix({'L': 2, 'R': 2, 'C': 2, 'SW': 1, 'SL': 1, 'SR': 1}, ['L', 'R', 'C', 'SW', 'SL', 'SR', 'RL', 'RR'])
    for c in ['L', 'R', 'C', 'SW']:
        five_one.enable(c, 0, 'SW')
        if c != 'SW':
            five_one.enable(c, 1, c)
    five_one.enable('SL', 0, 'RL')
    five_one.enable('SR', 0, 'RL')

    mw_filters = __make_bm_multiway(['L', 'R', 'C'])
    mc_filters = calculate_compound_routing_filter(five_one, xo_filters=mw_filters, lfe_channel_idx=5).filters
    assert mc_filters
    assert all(isinstance(x, MultiwayFilter) for x in mc_filters)


def mix(src: str, mt: MixType, dst: str, gain: float = 0.0) -> Dict[str, str]:
    return {
        **Mix.default_values(),
        'Source': str(get_channel_idx(src)),
        'Gain': f"{gain:.7g}",
        'Destination': str(get_channel_idx(dst)),
        'Mode': str(mt.value)
    }


def gain(channel: str, gain: float) -> Dict[str, str]:
    return {
        **Gain.default_values(),
        'Gain': f"{gain:.7g}",
        'Channels': str(get_channel_idx(channel))
    }
