from typing import List

import pytest

from model.iir import ComplexLowPass, FilterType, ComplexHighPass
from model.jriver import ImpossibleRoutingError, JRIVER_FS, flatten
from model.jriver.filter import MultiwayFilter, MultiChannelSystem, CompositeXODescriptor, XODescriptor, \
    WayDescriptor, WayValues, MDSXO, MDSPoint
from model.jriver.routing import Matrix


def lpf(freq: float) -> ComplexLowPass:
    return ComplexLowPass(FilterType.BUTTERWORTH, 2, JRIVER_FS, freq)


def hpf(freq: float) -> ComplexHighPass:
    return ComplexHighPass(FilterType.BUTTERWORTH, 2, JRIVER_FS, freq)


def p(freq: float, order: int = 4, ft: FilterType = FilterType.LINKWITZ_RILEY) -> list:
    return [ft.display_name, order, freq]


def n_way(in_ch: str, freq: List[float], channels: List[str], way_args: List[dict] = None, ss: float = 0.0):
    ways = []
    if way_args is None:
        way_args = [{}] * len(channels)
    assert (len(freq) + 1) == len(channels) == len(way_args)
    for i in range(len(freq) + 1):
        args = {} if way_args[i] is None else way_args[i]
        if i == 0:
            ways.append(WayDescriptor(WayValues(i, lp=p(freq[i]), hp=p(ss) if ss else None, **args), [channels[i]]))
        elif i == len(freq):
            ways.append(WayDescriptor(WayValues(i, hp=p(freq[i - 1]), **args), [channels[i]]))
        else:
            ways.append(WayDescriptor(WayValues(i, lp=p(freq[i]), hp=p(freq[i - 1]), **args), [channels[i]]))
    return XODescriptor(in_ch, ways)


def n_way_mds(in_ch: str, *channels: str):
    return XODescriptor(in_ch, [WayDescriptor(WayValues(i), [c]) for i, c in enumerate(channels)])


@pytest.fixture
def two_in_four() -> Matrix:
    return Matrix({'L': 2, 'R': 2}, ['L', 'R', 'C', 'SW'])


def test_stereo_two_way_lr4_symmetric(two_in_four):
    xo_desc = [
        CompositeXODescriptor({
            'L': n_way('L', [2000.0], ['L', 'C']),
            'R': n_way('R', [2000.0], ['R', 'SW']),
        }, [None] * 2, [])
    ]
    for d in xo_desc:
        copy_to_matrix(two_in_four, d)

    mcs = MultiChannelSystem(xo_desc)

    mc_filter = mcs.calculate_filters(two_in_four)

    assert mc_filter
    mc_filters = mc_filter.filters
    assert mc_filters
    assert all(isinstance(x, MultiwayFilter) for x in mc_filters)

    filter_list = '\n' + '\n'.join([str(f) for mc in mc_filters for f in flatten(mc)]) + '\n'
    assert filter_list == """
Copy L to U2 +0 dB
HP LR4 2kHz  [U2]
Copy U2 to C +0 dB
LP LR4 2kHz  [L]
Copy R to U2 +0 dB
HP LR4 2kHz  [U2]
Copy U2 to SW +0 dB
LP LR4 2kHz  [R]
"""


def test_stereo_two_way_lr4_symmetric_tweaked(two_in_four):
    xo_desc = [
        CompositeXODescriptor({
            'L': n_way('L', [2000.0], ['L', 'C'], way_args=[{'delay_millis': 1.5, 'gain': -1.0, 'inverted': True}, {}]),
            'R': n_way('R', [2000.0], ['R', 'SW'], way_args=[{}, {'delay_millis': -2.5, 'gain': 2.1}])
        }, [None] * 2, [])
    ]

    for d in xo_desc:
        copy_to_matrix(two_in_four, d)

    mcs = MultiChannelSystem(xo_desc)

    mc_filter = mcs.calculate_filters(two_in_four)

    assert mc_filter
    mc_filters = mc_filter.filters
    assert mc_filters
    assert all(isinstance(x, MultiwayFilter) for x in mc_filters)

    filter_list = '\n' + '\n'.join([str(f) for mc in mc_filters for f in flatten(mc)]) + '\n'
    assert filter_list == """
Copy L to U2 +0 dB
HP LR4 2kHz  [U2]
Copy U2 to C +0 dB
LP LR4 2kHz  [L]
Polarity [L]
Delay +1.5 ms [L]
Gain -1 dB [L]
Copy R to U2 +0 dB
HP LR4 2kHz  [U2]
Delay -2.5 ms [U2]
Copy U2 to SW +2.1 dB
LP LR4 2kHz  [R]
"""


def test_stereo_two_way_with_sub(two_in_four):
    xo_desc = [
        CompositeXODescriptor({
            'L': n_way('L', [100.0], ['SW', 'L']),
            'R': n_way('R', [100.0], ['SW', 'R'])
        }, [None] * 2, [])
    ]

    for d in xo_desc:
        copy_to_matrix(two_in_four, d)

    mcs = MultiChannelSystem(xo_desc)

    mc_filter = mcs.calculate_filters(two_in_four)

    assert mc_filter
    mc_filters = mc_filter.filters
    assert mc_filters
    assert all(isinstance(x, MultiwayFilter) for x in mc_filters)

    filter_list = '\n' + '\n'.join([str(f) for mc in mc_filters for f in flatten(mc)]) + '\n'
    assert filter_list == """
Copy L to U2 +0 dB
Copy L to U1 +0 dB
HP LR4 100Hz  [U2]
Copy U2 to L +0 dB
LP LR4 100Hz  [U1]
Add U1 to SW +0 dB
Copy R to U2 +0 dB
Copy R to U1 +0 dB
HP LR4 100Hz  [U2]
Copy U2 to R +0 dB
LP LR4 100Hz  [U1]
Add U1 to SW +0 dB
"""


def test_stereo_two_way_mds(two_in_four):
    xo_desc = [
        CompositeXODescriptor({
            'L': n_way_mds('L', 'L', 'C'),
            'R': n_way_mds('R', 'R', 'SW')
        }, [MDSXO(4, 100.0, lp_channel=['L'], hp_channel=['C']), MDSXO(4, 100.0, lp_channel=['R'], hp_channel=['SW'])], [MDSPoint(0, 4, 100.0)])
    ]

    for d in xo_desc:
        copy_to_matrix(two_in_four, d)

    mcs = MultiChannelSystem(xo_desc)

    mc_filter = mcs.calculate_filters(two_in_four)

    assert mc_filter
    mc_filters = mc_filter.filters
    assert mc_filters
    assert all(isinstance(x, MultiwayFilter) for x in mc_filters)

    filter_list = '\n' + '\n'.join([str(f) for mc in mc_filters for f in flatten(mc)]) + '\n'
    assert filter_list == """
Copy L to U1 +0 dB
Copy L to U2 +0 dB
LP BESM64 77.07Hz  [U1]
Delay +5.950662 ms [U2]
Subtract U1 from U2 +0 dB
Copy U2 to U1 +0 dB
LP BESM64 77.07Hz  [U1]
Delay +5.950662 ms [U2]
Subtract U1 from U2 +0 dB
Copy U2 to U1 +0 dB
Delay +5.950662 ms [L]
Delay +5.950662 ms [L]
Subtract U1 from L +0 dB
Move U1 to C +0 dB
Copy R to U1 +0 dB
Copy R to U2 +0 dB
LP BESM64 77.07Hz  [U1]
Delay +5.950662 ms [U2]
Subtract U1 from U2 +0 dB
Copy U2 to U1 +0 dB
LP BESM64 77.07Hz  [U1]
Delay +5.950662 ms [U2]
Subtract U1 from U2 +0 dB
Copy U2 to U1 +0 dB
Delay +5.950662 ms [R]
Delay +5.950662 ms [R]
Subtract U1 from R +0 dB
Move U1 to SW +0 dB
"""


def test_stereo_two_way_mds_asym(two_in_four):
    xo_desc = [
        CompositeXODescriptor({'L': n_way_mds('L', 'L', 'C'), }, [MDSXO(4, 100.0, lp_channel=['L'], hp_channel=['C'])], [MDSPoint(0, 4, 100.0)]),
        CompositeXODescriptor({'R': n_way_mds('R', 'R', 'SW'), }, [MDSXO(4, 250.0, lp_channel=['R'], hp_channel=['SW'])], [MDSPoint(0, 4, 250.0)])
    ]
    for d in xo_desc:
        copy_to_matrix(two_in_four, d)

    mcs = MultiChannelSystem(xo_desc)

    mc_filter = mcs.calculate_filters(two_in_four)

    assert mc_filter
    mc_filters = mc_filter.filters
    assert mc_filters

    filter_list = '\n' + '\n'.join([str(f) for mc in mc_filters for f in flatten(mc)]) + '\n'
    assert filter_list == """
Copy L to U1 +0 dB
Copy L to U2 +0 dB
LP BESM64 77.07Hz  [U1]
Delay +5.950662 ms [U2]
Subtract U1 from U2 +0 dB
Copy U2 to U1 +0 dB
LP BESM64 77.07Hz  [U1]
Delay +5.950662 ms [U2]
Subtract U1 from U2 +0 dB
Copy U2 to U1 +0 dB
Delay +5.950662 ms [L]
Delay +5.950662 ms [L]
Subtract U1 from L +0 dB
Move U1 to C +0 dB
Copy R to U1 +0 dB
Copy R to U2 +0 dB
LP BESM64 192.66Hz  [U1]
Delay +2.380322 ms [U2]
Subtract U1 from U2 +0 dB
Copy U2 to U1 +0 dB
LP BESM64 192.66Hz  [U1]
Delay +2.380322 ms [U2]
Subtract U1 from U2 +0 dB
Copy U2 to U1 +0 dB
Delay +2.380322 ms [R]
Delay +2.380322 ms [R]
Subtract U1 from R +0 dB
Move U1 to SW +0 dB
Delay +7.14068 ms [R, SW]
"""


def test_stereo_two_way_mds_sub(two_in_four):
    xo_desc = [
        CompositeXODescriptor({
            'L': n_way_mds('L', 'SW', 'L'),
            'R': n_way_mds('R', 'SW', 'R')
        }, [MDSXO(4, 100.0, lp_channel=['L'], hp_channel=['C'])], [MDSPoint(0, 4, 100.0)])
    ]
    for d in xo_desc:
        copy_to_matrix(two_in_four, d)

    mcs = MultiChannelSystem(xo_desc)

    mc_filter = mcs.calculate_filters(two_in_four)

    assert mc_filter
    mc_filters = mc_filter.filters
    assert mc_filters

    filter_list = '\n' + '\n'.join([str(f) for mc in mc_filters for f in flatten(mc)]) + '\n'
    assert filter_list == """
Copy L to U1 +0 dB
Copy L to U2 +0 dB
LP BESM64 77.07Hz  [U1]
Delay +5.950662 ms [U2]
Subtract U1 from U2 +0 dB
Copy U2 to U1 +0 dB
LP BESM64 77.07Hz  [U1]
Delay +5.950662 ms [U2]
Subtract U1 from U2 +0 dB
Copy U2 to U1 +0 dB
Copy L to C +0 dB
Delay +5.950662 ms [C]
Delay +5.950662 ms [C]
Subtract U1 from C +0 dB
Move U1 to L +0 dB
Copy R to U1 +0 dB
Copy R to U2 +0 dB
LP BESM64 77.07Hz  [U1]
Delay +5.950662 ms [U2]
Subtract U1 from U2 +0 dB
Copy U2 to U1 +0 dB
LP BESM64 77.07Hz  [U1]
Delay +5.950662 ms [U2]
Subtract U1 from U2 +0 dB
Copy U2 to U1 +0 dB
Copy R to SW +0 dB
Delay +5.950662 ms [SW]
Delay +5.950662 ms [SW]
Subtract U1 from SW +0 dB
Move U1 to R +0 dB
Add C to SW +0 dB
Mute [C]
"""


def test_two_in_four_overwrite_input(two_in_four):
    xo_desc = [
        CompositeXODescriptor({
            'L': n_way('L', [2000], ['L', 'R']),
            'R': n_way('R', [2000], ['C', 'SW'])
        }, [None] * 2, [])
    ]
    for d in xo_desc:
        copy_to_matrix(two_in_four, d)

    with pytest.raises(ImpossibleRoutingError):
        MultiChannelSystem(xo_desc).calculate_filters(two_in_four)


@pytest.fixture
def two_in_eight() -> Matrix:
    return Matrix({'L': 4, 'R': 4}, ['L', 'R', 'C', 'SW', 'SL', 'SR', 'RL', 'RR'])


def test_stereo_4way_symmetric_lr4_with_ss_gain(two_in_eight):
    xo_desc = [
        CompositeXODescriptor({
            'L': n_way('L', [100.0, 500.0, 3000.0], ['L', 'C', 'SW', 'SL'], ss=10.0, way_args=[{'gain': 4.0}, {}, {}, {}]),
            'R': n_way('R', [100.0, 500.0, 3000.0], ['R', 'SR', 'RL', 'RR']),
        }, [None] * 4, [])
    ]

    for d in xo_desc:
        copy_to_matrix(two_in_eight, d)

    mcs = MultiChannelSystem(xo_desc)

    mc_filter = mcs.calculate_filters(two_in_eight)

    assert mc_filter
    mc_filters = mc_filter.filters
    assert mc_filters
    assert all(isinstance(x, MultiwayFilter) for x in mc_filters)

    filter_list = '\n' + '\n'.join([str(f) for mc in mc_filters for f in flatten(mc)]) + '\n'
    assert filter_list == """
Copy L to U2 +0 dB
HP LR4 100Hz  [U2]
Copy U2 to C +0 dB
HP LR4 10Hz  [L]
LP LR4 100Hz  [L]
Gain +4 dB [L]
Copy C to U2 +0 dB
HP LR4 500Hz  [U2]
Copy U2 to SW +0 dB
LP LR4 500Hz  [C]
Copy SW to U2 +0 dB
HP LR4 3kHz  [U2]
Copy U2 to SL +0 dB
LP LR4 3kHz  [SW]
Copy R to U2 +0 dB
HP LR4 100Hz  [U2]
Copy U2 to SR +0 dB
LP LR4 100Hz  [R]
Copy SR to U2 +0 dB
HP LR4 500Hz  [U2]
Copy U2 to RL +0 dB
LP LR4 500Hz  [SR]
Copy RL to U2 +0 dB
HP LR4 3kHz  [U2]
Copy U2 to RR +0 dB
LP LR4 3kHz  [RL]
"""


def copy_to_matrix(matrix, xo_desc):
    for desc in xo_desc.xo_descriptors.values():
        for i, way in enumerate(desc.ways):
            for o in way.out_channels:
                matrix.enable(desc.in_channel, i, o)


def test_two_in_eight_overwrite_input(two_in_eight):
    xo_desc = [
        CompositeXODescriptor({
            'L': n_way('L', [100.0, 500.0, 2000], ['L', 'R', 'C', 'SW']),
            'R': n_way('R', [100.0, 500.0, 2000], ['SL', 'SR', 'RL', 'RR'])
        }, [None] * 4, [])
    ]
    for d in xo_desc:
        copy_to_matrix(two_in_eight, d)

    with pytest.raises(ImpossibleRoutingError):
        MultiChannelSystem(xo_desc).calculate_filters(two_in_eight)


def test_stereo_4way_with_sub(two_in_eight):
    xo_desc = [
        CompositeXODescriptor({
            'L': n_way('L', [100.0, 500.0, 3000.0], ['SW', 'L', 'C', 'SL']),
            'R': n_way('R', [100.0, 500.0, 3000.0], ['SW', 'R', 'SR', 'RL']),
        }, [None] * 4, [])
    ]

    for d in xo_desc:
        copy_to_matrix(two_in_eight, d)

    mcs = MultiChannelSystem(xo_desc)

    mc_filter = mcs.calculate_filters(two_in_eight)

    assert mc_filter
    mc_filters = mc_filter.filters
    assert mc_filters
    assert all(isinstance(x, MultiwayFilter) for x in mc_filters)

    filter_list = '\n' + '\n'.join([str(f) for mc in mc_filters for f in flatten(mc)]) + '\n'
    assert filter_list == """
Copy L to U2 +0 dB
Copy L to U1 +0 dB
HP LR4 100Hz  [U2]
Copy U2 to L +0 dB
LP LR4 100Hz  [U1]
Add U1 to SW +0 dB
Copy L to U2 +0 dB
HP LR4 500Hz  [U2]
Copy U2 to C +0 dB
LP LR4 500Hz  [L]
Copy C to U2 +0 dB
HP LR4 3kHz  [U2]
Copy U2 to SL +0 dB
LP LR4 3kHz  [C]
Copy R to U2 +0 dB
Copy R to U1 +0 dB
HP LR4 100Hz  [U2]
Copy U2 to R +0 dB
LP LR4 100Hz  [U1]
Add U1 to SW +0 dB
Copy R to U2 +0 dB
HP LR4 500Hz  [U2]
Copy U2 to SR +0 dB
LP LR4 500Hz  [R]
Copy SR to U2 +0 dB
HP LR4 3kHz  [U2]
Copy U2 to RL +0 dB
LP LR4 3kHz  [SR]
"""


@pytest.fixture
def five_one() -> Matrix:
    return Matrix({'L': 2, 'R': 2, 'C': 2, 'SW': 1, 'SL': 2, 'SR': 2}, ['L', 'R', 'C', 'SW', 'SL', 'SR'])


def test_bass_managed_five_one_with_lfe_lpf(five_one):
    xo_desc = [
        CompositeXODescriptor({c: n_way(c, [100.0], ['SW', c]) for c in ['L', 'R', 'C', 'SL', 'SR']}, [None] * 2, [])
    ]

    for d in xo_desc:
        copy_to_matrix(five_one, d)

    mcs = MultiChannelSystem(xo_desc)

    mc_filter = mcs.calculate_filters(five_one)

    assert mc_filter
    mc_filters = mc_filter.filters
    assert mc_filters
    assert all(isinstance(x, MultiwayFilter) for x in mc_filters)

    filter_list = '\n' + '\n'.join([str(f) for mc in mc_filters for f in flatten(mc)]) + '\n'
    assert filter_list == """
Gain -5 dB [SW]
Gain -15 dB [L]
Gain -15 dB [R]
Gain -15 dB [C]
Gain -15 dB [SL]
Gain -15 dB [SR]
LP LR4 120Hz  [SW]
Copy L to U2 +0 dB
Copy L to U1 +0 dB
HP LR4 100Hz  [U2]
Copy U2 to L +0 dB
LP LR4 100Hz  [U1]
Add U1 to SW +0 dB
Copy R to U2 +0 dB
Copy R to U1 +0 dB
HP LR4 100Hz  [U2]
Copy U2 to R +0 dB
LP LR4 100Hz  [U1]
Add U1 to SW +0 dB
Copy C to U2 +0 dB
Copy C to U1 +0 dB
HP LR4 100Hz  [U2]
Copy U2 to C +0 dB
LP LR4 100Hz  [U1]
Add U1 to SW +0 dB
Copy SL to U2 +0 dB
Copy SL to U1 +0 dB
HP LR4 100Hz  [U2]
Copy U2 to SL +0 dB
LP LR4 100Hz  [U1]
Add U1 to SW +0 dB
Copy SR to U2 +0 dB
Copy SR to U1 +0 dB
HP LR4 100Hz  [U2]
Copy U2 to SR +0 dB
LP LR4 100Hz  [U1]
Add U1 to SW +0 dB
"""


def test_bass_managed_five_one_mds_no_spare_channels(five_one):
    mains = ['L', 'R', 'C', 'SL', 'SR']
    xo_desc = [
        CompositeXODescriptor({c: n_way_mds('SW', c) for c in mains},
                              [MDSXO(4, 100.0, lp_channel=['SW'], hp_channel=[c]) for c in mains],
                              [MDSPoint(0, 4, 100.0)])
    ]

    for d in xo_desc:
        copy_to_matrix(five_one, d)

    mcs = MultiChannelSystem(xo_desc)

    with pytest.raises(ImpossibleRoutingError):
        mcs.calculate_filters(five_one)


@pytest.fixture
def five_one_plus_six() -> Matrix:
    return Matrix({'L': 2, 'R': 2, 'C': 2, 'SW': 1, 'SL': 2, 'SR': 2}, ['L', 'R', 'C', 'SW', 'SL', 'SR', 'RL', 'RR', 'C9', 'C10', 'C11', 'C12'])


def test_bass_managed_five_one_mds_with_spare_channels(five_one_plus_six):
    mains = ['L', 'R', 'C', 'SL', 'SR']
    xo_desc = [
        CompositeXODescriptor({c: n_way_mds('SW', c) for c in mains},
                              [MDSXO(4, 100.0, lp_channel=['SW'], hp_channel=[c]) for c in mains],
                              [MDSPoint(0, 4, 100.0)])
    ]

    for d in xo_desc:
        copy_to_matrix(five_one_plus_six, d)

    mcs = MultiChannelSystem(xo_desc)

    mcs.calculate_filters(five_one_plus_six)


def test_five_one_swap_main(five_one):
    for c in ['L', 'R', 'C', 'SW', 'SL', 'SR']:
        five_one.enable(c, 0, 'SW')
        if c not in ['SW', 'SL', 'SR']:
            five_one.enable(c, 1, c)
    five_one.enable('SL', 1, 'SR')
    five_one.enable('SR', 1, 'SL')
    # with pytest.raises(ImpossibleRoutingError):
    #     calculate_compound_routing_filter(five_one, lfe_channel_idx=5)


def test_five_one_swap_sub_to_c(five_one):
    for c in ['L', 'R', 'C', 'SW', 'SL', 'SR']:
        five_one.enable(c, 0, 'C')
        if c not in ['C', 'SW']:
            five_one.enable(c, 1, c)
    five_one.enable('C', 1, 'SW')
    # with pytest.raises(ImpossibleRoutingError):
    #     calculate_compound_routing_filter(five_one, lfe_channel_idx=5)


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
    # for vals in multiway_channels:
    #     mw_filters.append(
    #         MultiwayFilter(vals[0],
    #                        ['SW'] + list(vals),
    #                        MultiwayCrossover(vals[0], [StandardXO(['SW'], [vals[0]]), StandardXO([vals[0]], [vals[1]]), StandardXO([vals[1]], [vals[2]])],
    #                                          []).graph.filters, {}),
    #     )
    #
    # mw_filters = mw_filters + __make_bm_multiway(other_channels)
    # mc_filters = calculate_compound_routing_filter(multi_bm, xo_filters=mw_filters, lfe_channel_idx=5).filters
    #
    # assert all(isinstance(x, MultiwayFilter) for x in mc_filters)
    #
    # for vals in multiway_channels:
    #     f = mc_filters.pop(0)
    #     assert f.filters.pop(0).get_all_vals() == [mix(vals[0], MixType.ADD, 'SW')]
    #     assert f.filters.pop(0).get_all_vals() == [mix(vals[0], MixType.COPY, vals[1])]
    #     assert f.filters.pop(0).get_all_vals() == [mix(vals[1], MixType.COPY, vals[2])]
    #     assert not f.filters
    #
    # for c in other_channels:
    #     f = mc_filters.pop(0)
    #     assert f.filters.pop(0).get_all_vals() == [mix(c, MixType.ADD, 'SW')]
    #     assert not f.filters
    #
    # assert not mc_filters


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

    # mw_filters = []
    # for vals in multiway_channels:
    #     if vals[0] != 'C':
    #         mw_filters.append(
    #             MultiwayFilter(vals[0],
    #                            ['SW', 'C15'] + list(vals),
    #                            MultiwayCrossover(vals[0],
    #                                              [StandardXO(['SW', 'C15'], [vals[0]]), StandardXO([vals[0]], [vals[1]]), StandardXO([vals[1]], [vals[2]])],
    #                                              []).graph.filters, {}),
    #         )
    #     else:
    #         mw_filters.append(
    #             MultiwayFilter(vals[0],
    #                            ['SW'] + list(vals),
    #                            MultiwayCrossover(vals[0], [StandardXO(['SW'], [vals[0]]), StandardXO([vals[0]], [vals[1]]), StandardXO([vals[1]], [vals[2]])],
    #                                              []).graph.filters, {}),
    #         )
    #
    # mw_filters = mw_filters + __make_bm_multiway(other_channels)
    # mc_filters = calculate_compound_routing_filter(multi_bm, xo_filters=mw_filters, lfe_channel_idx=5).filters
    #
    # assert all(isinstance(x, MultiwayFilter) for x in mc_filters)
    #
    # for vals in multiway_channels:
    #     f = mc_filters.pop(0)
    #     assert f.filters.pop(0).get_all_vals() == [mix(vals[0], MixType.ADD, 'SW')]
    #     if vals[0] != 'C':
    #         assert f.filters.pop(0).get_all_vals() == [mix(vals[0], MixType.ADD, 'C15')]
    #     assert f.filters.pop(0).get_all_vals() == [mix(vals[0], MixType.COPY, vals[1])]
    #     assert f.filters.pop(0).get_all_vals() == [mix(vals[1], MixType.COPY, vals[2])]
    #     assert not f.filters
    #
    # for c in other_channels:
    #     f = mc_filters.pop(0)
    #     assert f.filters.pop(0).get_all_vals() == [mix(c, MixType.ADD, 'SW')]
    #     assert not f.filters
    #
    # assert not mc_filters


@pytest.fixture
def seven_one() -> Matrix:
    return Matrix({'L': 2, 'R': 2, 'C': 2, 'SW': 1, 'SL': 2, 'SR': 2},
                  ['L', 'R', 'C', 'SW', 'SL', 'SR', 'RL', 'RR'])


def test_stereo_subs_overwrite(seven_one):
    __map_stereo_subs_bm(seven_one, 'SW')

    # mw_filters = \
    #     [MultiwayFilter(c, [c, 'SW'], MultiwayCrossover(c, [StandardXO(['SW'], [c])], []).graph.filters, {}) for c in ['L', 'SL']] + \
    #     [MultiwayFilter(c, [c, 'RR'], MultiwayCrossover(c, [StandardXO(['RR'], [c])], []).graph.filters, {}) for c in ['R', 'SR']] + \
    #     [MultiwayFilter('C', ['C', 'SW', 'RR'], MultiwayCrossover('C', [StandardXO(['SW', 'RR'], ['C'])], []).graph.filters, {})]
    # with pytest.raises(UnsupportedRoutingError):
    #     calculate_compound_routing_filter(seven_one, xo_filters=mw_filters, lfe_channel_idx=5)


def __map_stereo_subs_bm(stereo_subs: Matrix, sub1: str) -> None:
    for c in ['L', 'C', 'SW', 'SL']:
        stereo_subs.enable(c, 0, sub1)
    for c in ['R', 'C', 'SW', 'SR']:
        stereo_subs.enable(c, 0, 'RR')
    for c in ['L', 'R', 'C', 'SL', 'SR']:
        stereo_subs.enable(c, 1, c)


def test_stereo_subs(seven_one):
    __map_stereo_subs_bm(seven_one, 'RL')

    # mw_filters = \
    #     [MultiwayFilter(c, [c, 'RL'], MultiwayCrossover(c, [StandardXO(['RL'], [c])], []).graph.filters, {}) for c in ['L', 'SL']] + \
    #     [MultiwayFilter(c, [c, 'RR'], MultiwayCrossover(c, [StandardXO(['RR'], [c])], []).graph.filters, {}) for c in ['R', 'SR']] + \
    #     [MultiwayFilter('C', ['C', 'RL', 'RR'], MultiwayCrossover('C', [StandardXO(['RL', 'RR'], ['C'])], []).graph.filters, {})]
    # mc_filters = calculate_compound_routing_filter(seven_one, xo_filters=mw_filters, lfe_channel_idx=5).filters
    # assert mc_filters
    # assert all(isinstance(x, MultiwayFilter) for x in mc_filters)
    #
    # f = mc_filters.pop(0)
    # assert f.filters.pop(0).get_all_vals() == [mix('L', MixType.ADD, 'RL')]
    # assert not f.filters
    # f = mc_filters.pop(0)
    # assert f.filters.pop(0).get_all_vals() == [mix('SL', MixType.ADD, 'RL')]
    # assert not f.filters
    # f = mc_filters.pop(0)
    # assert f.filters.pop(0).get_all_vals() == [mix('R', MixType.ADD, 'RR')]
    # assert not f.filters
    # f = mc_filters.pop(0)
    # assert f.filters.pop(0).get_all_vals() == [mix('SR', MixType.ADD, 'RR')]
    # assert not f.filters
    # f = mc_filters.pop(0)
    # assert f.filters.pop(0).get_all_vals() == [mix('C', MixType.ADD, 'RL')]
    # assert f.filters.pop(0).get_all_vals() == [mix('C', MixType.ADD, 'RR')]
    # assert not f.filters
    #
    # assert not mc_filters


def test_collapse_surround():
    five_one = Matrix({'L': 2, 'R': 2, 'C': 2, 'SW': 1, 'SL': 1, 'SR': 1}, ['L', 'R', 'C', 'SW', 'SL', 'SR', 'RL', 'RR'])
    for c in ['L', 'R', 'C', 'SW']:
        five_one.enable(c, 0, 'SW')
        if c != 'SW':
            five_one.enable(c, 1, c)
    five_one.enable('SL', 0, 'RL')
    five_one.enable('SR', 0, 'RL')

    # mw_filters = __make_bm_multiway(['L', 'R', 'C'])
    # mc_filters = calculate_compound_routing_filter(five_one, xo_filters=mw_filters, lfe_channel_idx=5).filters
    # assert mc_filters
    # assert all(isinstance(x, MultiwayFilter) for x in mc_filters)
