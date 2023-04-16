from typing import List, Union

import pytest

from model.iir import ComplexLowPass, FilterType, ComplexHighPass
from model.jriver import ImpossibleRoutingError, JRIVER_FS, flatten, UnsupportedRoutingError
from model.jriver.filter import MultiwayFilter, MultiChannelSystem, CompositeXODescriptor, XODescriptor, \
    WayDescriptor, WayValues, MDSXO, MDSPoint, Gain, Delay, Mix, Mute
from model.jriver.routing import Matrix


def lpf(freq: float) -> ComplexLowPass:
    return ComplexLowPass(FilterType.BUTTERWORTH, 2, JRIVER_FS, freq)


def hpf(freq: float) -> ComplexHighPass:
    return ComplexHighPass(FilterType.BUTTERWORTH, 2, JRIVER_FS, freq)


def p(freq: float, order: int = 4, ft: FilterType = FilterType.LINKWITZ_RILEY) -> list:
    return [ft.display_name, order, freq]


def n_way(in_ch: str, freq: List[float], channels: List[Union[str, List[str]]], way_args: List[dict] = None, ss: float = 0.0):
    ways = []
    if way_args is None:
        way_args = [{}] * (len(freq) + 1)
    assert (len(freq) + 1) == len(way_args)
    for i in range(len(freq) + 1):
        args = {} if way_args[i] is None else way_args[i]
        out_chs = channels[i] if isinstance(channels[i], list) else [channels[i]]
        if i == 0:
            ways.append(WayDescriptor(WayValues(i, lp=p(freq[i]), hp=p(ss) if ss else None, **args), out_chs))
        elif i == len(freq):
            ways.append(WayDescriptor(WayValues(i, hp=p(freq[i - 1]), **args), out_chs))
        else:
            ways.append(WayDescriptor(WayValues(i, lp=p(freq[i]), hp=p(freq[i - 1]), **args), out_chs))
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
        },
        [MDSXO(4, 100.0, lp_channel=['SW'], hp_channel=['L']), MDSXO(4, 100.0, lp_channel=['SW'], hp_channel=['R'])],
        [MDSPoint(0, 4, 100.0)])
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
Add R to SW +0 dB
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
    return Matrix({'SW': 1, 'L': 2, 'R': 2, 'C': 2, 'SL': 2, 'SR': 2}, ['L', 'R', 'C', 'SW', 'SL', 'SR'])


def test_bass_managed_five_one_with_lfe_lpf(five_one):
    xo_desc = [
        CompositeXODescriptor({'SW': XODescriptor('SW', [WayDescriptor(WayValues(0, lp=p(120)), ['SW'])])},
                              [None], []),
        CompositeXODescriptor({c: n_way(c, [100.0], ['SW', c]) for c in ['L', 'R', 'C', 'SL', 'SR']},
                              [None] * 2, [])
    ]

    for d in xo_desc:
        copy_to_matrix(five_one, d)

    mcs = MultiChannelSystem(xo_desc)

    from model.jriver.common import get_channel_idx
    mc_filter = mcs.calculate_filters(five_one, main_adjust=-15, lfe_adjust=-5, lfe_channel_idx=get_channel_idx('SW'))

    assert mc_filter
    mc_filters = mc_filter.filters
    assert mc_filters
    assert all(isinstance(x, (MultiwayFilter, Gain)) for x in mc_filters)

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
        CompositeXODescriptor({'SW': XODescriptor('SW', [WayDescriptor(WayValues(0, lp=p(120)), ['SW'])])},
                              [None],
                              []),
        CompositeXODescriptor({c: n_way_mds(c, 'SW', c) for c in mains},
                              [MDSXO(4, 100.0, lp_channel=['SW'], hp_channel=[c]) for c in mains],
                              [MDSPoint(0, 4, 100.0)])
    ]

    for d in xo_desc:
        copy_to_matrix(five_one, d)

    mcs = MultiChannelSystem(xo_desc)

    with pytest.raises(UnsupportedRoutingError):
        mcs.calculate_filters(five_one)


@pytest.fixture
def five_one_plus_six() -> Matrix:
    return Matrix({'SW': 1, 'L': 2, 'R': 2, 'C': 2, 'RL': 2, 'RR': 2}, ['L', 'R', 'C', 'SW', 'SL', 'SR', 'RL', 'RR', 'C9', 'C10', 'C11', 'C12'])


def test_bass_managed_five_one_mds_with_spare_channels(five_one_plus_six):
    mains = ['L', 'R', 'C', 'RL', 'RR']
    xo_desc = [
        CompositeXODescriptor({'SW': XODescriptor('SW', [WayDescriptor(WayValues(0, lp=p(120)), ['SW'])])},
                              [None],
                              []),
        CompositeXODescriptor({c: n_way_mds(c, 'SW', c) for c in mains},
                              [MDSXO(4, 100.0, lp_channel=['SW'], hp_channel=[c]) for c in mains],
                              [MDSPoint(0, 4, 100.0)])
    ]

    for d in xo_desc:
        copy_to_matrix(five_one_plus_six, d)

    mcs = MultiChannelSystem(xo_desc)

    from model.jriver.common import get_channel_idx
    mc_filter = mcs.calculate_filters(five_one_plus_six, main_adjust=-15, lfe_adjust=-5, lfe_channel_idx=get_channel_idx('SW'))
    assert mc_filter
    mc_filters = mc_filter.filters
    assert mc_filters
    assert all(isinstance(x, (MultiwayFilter, Gain, Delay, Mix, Mute)) for x in mc_filters)

    filter_list = '\n' + '\n'.join([str(f) for mc in mc_filters for f in flatten(mc)]) + '\n'
    assert filter_list == """
Gain -5 dB [SW]
Gain -15 dB [L]
Gain -15 dB [R]
Gain -15 dB [C]
Gain -15 dB [RL]
Gain -15 dB [RR]
LP LR4 120Hz  [SW]
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
Copy L to SL +0 dB
Delay +5.950662 ms [SL]
Delay +5.950662 ms [SL]
Subtract U1 from SL +0 dB
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
Copy R to SR +0 dB
Delay +5.950662 ms [SR]
Delay +5.950662 ms [SR]
Subtract U1 from SR +0 dB
Move U1 to R +0 dB
Copy C to U1 +0 dB
Copy C to U2 +0 dB
LP BESM64 77.07Hz  [U1]
Delay +5.950662 ms [U2]
Subtract U1 from U2 +0 dB
Copy U2 to U1 +0 dB
LP BESM64 77.07Hz  [U1]
Delay +5.950662 ms [U2]
Subtract U1 from U2 +0 dB
Copy U2 to U1 +0 dB
Copy C to C9 +0 dB
Delay +5.950662 ms [C9]
Delay +5.950662 ms [C9]
Subtract U1 from C9 +0 dB
Move U1 to C +0 dB
Copy RL to U1 +0 dB
Copy RL to U2 +0 dB
LP BESM64 77.07Hz  [U1]
Delay +5.950662 ms [U2]
Subtract U1 from U2 +0 dB
Copy U2 to U1 +0 dB
LP BESM64 77.07Hz  [U1]
Delay +5.950662 ms [U2]
Subtract U1 from U2 +0 dB
Copy U2 to U1 +0 dB
Copy RL to C10 +0 dB
Delay +5.950662 ms [C10]
Delay +5.950662 ms [C10]
Subtract U1 from C10 +0 dB
Move U1 to RL +0 dB
Copy RR to U1 +0 dB
Copy RR to U2 +0 dB
LP BESM64 77.07Hz  [U1]
Delay +5.950662 ms [U2]
Subtract U1 from U2 +0 dB
Copy U2 to U1 +0 dB
LP BESM64 77.07Hz  [U1]
Delay +5.950662 ms [U2]
Subtract U1 from U2 +0 dB
Copy U2 to U1 +0 dB
Copy RR to C11 +0 dB
Delay +5.950662 ms [C11]
Delay +5.950662 ms [C11]
Subtract U1 from C11 +0 dB
Move U1 to RR +0 dB
Delay +6.367991 ms [SW]
Add SL to SW +0 dB
Mute [SL]
Add SR to SW +0 dB
Mute [SR]
Add C9 to SW +0 dB
Mute [C9]
Add C10 to SW +0 dB
Mute [C10]
Add C11 to SW +0 dB
Mute [C11]
"""


@pytest.fixture
def multi_bm() -> Matrix:
    return Matrix({'SW': 1, 'L': 4, 'R': 4, 'C': 4, 'RL': 2, 'RR': 2},
                  ['L', 'R', 'C', 'SW', 'SL', 'SR', 'RL', 'RR', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14'])


def test_multi_way_mains_simple_bm(multi_bm):
    multiway_channels = [('L', 'SL', 'SR'), ('R', 'C9', 'C10'), ('C', 'C11', 'C12')]
    singleway_channels = ['RL', 'RR']

    xo_desc = [
        CompositeXODescriptor({'SW': XODescriptor('SW', [WayDescriptor(WayValues(0, lp=p(120)), ['SW'])])},
                              [None],
                              []),
        CompositeXODescriptor({c[0]: n_way(c[0], [100.0, 350.0, 1250.0], ['SW'] + list(c)) for c in multiway_channels},
                              [None] * len(multiway_channels[0]),
                              []),
        CompositeXODescriptor({c: n_way(c, [100.0], ['SW', c]) for c in singleway_channels},
                              [None],
                              [])
    ]

    for d in xo_desc:
        copy_to_matrix(multi_bm, d)

    mcs = MultiChannelSystem(xo_desc)

    from model.jriver.common import get_channel_idx
    mc_filter = mcs.calculate_filters(multi_bm, main_adjust=-15, lfe_adjust=-5, lfe_channel_idx=get_channel_idx('SW'))
    assert mc_filter
    mc_filters = mc_filter.filters
    assert mc_filters
    assert all(isinstance(x, (MultiwayFilter, Gain, Delay, Mix, Mute)) for x in mc_filters)

    filter_list = '\n' + '\n'.join([str(f) for mc in mc_filters for f in flatten(mc)]) + '\n'
    assert filter_list == """
Gain -5 dB [SW]
Gain -15 dB [L]
Gain -15 dB [R]
Gain -15 dB [C]
Gain -15 dB [RL]
Gain -15 dB [RR]
LP LR4 120Hz  [SW]
Copy L to U2 +0 dB
Copy L to U1 +0 dB
HP LR4 100Hz  [U2]
Copy U2 to L +0 dB
LP LR4 100Hz  [U1]
Add U1 to SW +0 dB
Copy L to U2 +0 dB
HP LR4 350Hz  [U2]
Copy U2 to SL +0 dB
LP LR4 350Hz  [L]
Copy SL to U2 +0 dB
HP LR4 1.25kHz  [U2]
Copy U2 to SR +0 dB
LP LR4 1.25kHz  [SL]
Copy R to U2 +0 dB
Copy R to U1 +0 dB
HP LR4 100Hz  [U2]
Copy U2 to R +0 dB
LP LR4 100Hz  [U1]
Add U1 to SW +0 dB
Copy R to U2 +0 dB
HP LR4 350Hz  [U2]
Copy U2 to C9 +0 dB
LP LR4 350Hz  [R]
Copy C9 to U2 +0 dB
HP LR4 1.25kHz  [U2]
Copy U2 to C10 +0 dB
LP LR4 1.25kHz  [C9]
Copy C to U2 +0 dB
Copy C to U1 +0 dB
HP LR4 100Hz  [U2]
Copy U2 to C +0 dB
LP LR4 100Hz  [U1]
Add U1 to SW +0 dB
Copy C to U2 +0 dB
HP LR4 350Hz  [U2]
Copy U2 to C11 +0 dB
LP LR4 350Hz  [C]
Copy C11 to U2 +0 dB
HP LR4 1.25kHz  [U2]
Copy U2 to C12 +0 dB
LP LR4 1.25kHz  [C11]
Copy RL to U2 +0 dB
Copy RL to U1 +0 dB
HP LR4 100Hz  [U2]
Copy U2 to RL +0 dB
LP LR4 100Hz  [U1]
Add U1 to SW +0 dB
Copy RR to U2 +0 dB
Copy RR to U1 +0 dB
HP LR4 100Hz  [U2]
Copy U2 to RR +0 dB
LP LR4 100Hz  [U1]
Add U1 to SW +0 dB
"""


def test_multi_way_mains_simple_bm_with_shared_sub(multi_bm):
    multiway_channels = [('L', 'SL', 'SR'), ('R', 'C9', 'C10'), ('C', 'C11', 'C12')]
    singleway_channels = ['RL', 'RR']

    xo_desc = [
        CompositeXODescriptor({'SW': XODescriptor('SW', [WayDescriptor(WayValues(0, lp=p(120)), ['SW', 'C13'])])},
                              [None],
                              []),
        CompositeXODescriptor({c[0]: n_way(c[0], [100.0, 350.0, 1250.0], [['SW', 'C13']] + list(c)) for c in multiway_channels},
                              [None] * len(multiway_channels[0]),
                              []),
        CompositeXODescriptor({c: n_way(c, [100.0], [['SW', 'C13'], c]) for c in singleway_channels},
                              [None],
                              [])
    ]

    for d in xo_desc:
        copy_to_matrix(multi_bm, d)

    mcs = MultiChannelSystem(xo_desc)

    from model.jriver.common import get_channel_idx
    mc_filter = mcs.calculate_filters(multi_bm, main_adjust=-15, lfe_adjust=-5, lfe_channel_idx=get_channel_idx('SW'))
    assert mc_filter
    mc_filters = mc_filter.filters
    assert mc_filters
    assert all(isinstance(x, (MultiwayFilter, Gain, Delay, Mix, Mute)) for x in mc_filters)

    filter_list = '\n' + '\n'.join([str(f) for mc in mc_filters for f in flatten(mc)]) + '\n'
    assert filter_list == """
Gain -5 dB [SW]
Gain -15 dB [L]
Gain -15 dB [R]
Gain -15 dB [C]
Gain -15 dB [RL]
Gain -15 dB [RR]
LP LR4 120Hz  [SW]
Add SW to C13 +0 dB
Copy L to U2 +0 dB
Copy L to U1 +0 dB
HP LR4 100Hz  [U2]
Copy U2 to L +0 dB
LP LR4 100Hz  [U1]
Add U1 to SW +0 dB
Add U1 to C13 +0 dB
Copy L to U2 +0 dB
HP LR4 350Hz  [U2]
Copy U2 to SL +0 dB
LP LR4 350Hz  [L]
Copy SL to U2 +0 dB
HP LR4 1.25kHz  [U2]
Copy U2 to SR +0 dB
LP LR4 1.25kHz  [SL]
Copy R to U2 +0 dB
Copy R to U1 +0 dB
HP LR4 100Hz  [U2]
Copy U2 to R +0 dB
LP LR4 100Hz  [U1]
Add U1 to SW +0 dB
Add U1 to C13 +0 dB
Copy R to U2 +0 dB
HP LR4 350Hz  [U2]
Copy U2 to C9 +0 dB
LP LR4 350Hz  [R]
Copy C9 to U2 +0 dB
HP LR4 1.25kHz  [U2]
Copy U2 to C10 +0 dB
LP LR4 1.25kHz  [C9]
Copy C to U2 +0 dB
Copy C to U1 +0 dB
HP LR4 100Hz  [U2]
Copy U2 to C +0 dB
LP LR4 100Hz  [U1]
Add U1 to SW +0 dB
Add U1 to C13 +0 dB
Copy C to U2 +0 dB
HP LR4 350Hz  [U2]
Copy U2 to C11 +0 dB
LP LR4 350Hz  [C]
Copy C11 to U2 +0 dB
HP LR4 1.25kHz  [U2]
Copy U2 to C12 +0 dB
LP LR4 1.25kHz  [C11]
Copy RL to U2 +0 dB
Copy RL to U1 +0 dB
HP LR4 100Hz  [U2]
Copy U2 to RL +0 dB
LP LR4 100Hz  [U1]
Add U1 to SW +0 dB
Add U1 to C13 +0 dB
Copy RR to U2 +0 dB
Copy RR to U1 +0 dB
HP LR4 100Hz  [U2]
Copy U2 to RR +0 dB
LP LR4 100Hz  [U1]
Add U1 to SW +0 dB
Add U1 to C13 +0 dB
"""


@pytest.fixture
def seven_one() -> Matrix:
    return Matrix({'SW': 1, 'L': 2, 'R': 2, 'C': 2, 'RL': 2, 'RR': 2},
                  ['L', 'R', 'C', 'SW', 'SL', 'SR', 'RL', 'RR'])


def test_stereo_subs(seven_one):
    subs = {'L': 'SW', 'R': 'SR', 'C': ['SW', 'SR'], 'RL': 'SW', 'RR': 'SR'}
    xo_desc = [
        CompositeXODescriptor({'SW': XODescriptor('SW', [WayDescriptor(WayValues(0, lp=p(120)), ['SW'])])},
                              [None],
                              []),
        CompositeXODescriptor({c: n_way(c, [100.0], [sw, c]) for c, sw in subs.items()},
                              [None, None],
                              [])
    ]

    for d in xo_desc:
        copy_to_matrix(seven_one, d)

    mcs = MultiChannelSystem(xo_desc)

    mc_filter = mcs.calculate_filters(seven_one)
    assert mc_filter
    mc_filters = mc_filter.filters
    assert mc_filters
    assert all(isinstance(x, (MultiwayFilter, Gain, Mix)) for x in mc_filters)

    filter_list = '\n' + '\n'.join([str(f) for mc in mc_filters for f in flatten(mc)]) + '\n'
    assert filter_list == """
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
Add U1 to SR +0 dB
Copy C to U2 +0 dB
Copy C to U1 +0 dB
HP LR4 100Hz  [U2]
Copy U2 to C +0 dB
LP LR4 100Hz  [U1]
Add U1 to SW +0 dB
Add U1 to SR +0 dB
Copy RL to U2 +0 dB
Copy RL to U1 +0 dB
HP LR4 100Hz  [U2]
Copy U2 to RL +0 dB
LP LR4 100Hz  [U1]
Add U1 to SW +0 dB
Copy RR to U2 +0 dB
Copy RR to U1 +0 dB
HP LR4 100Hz  [U2]
Copy U2 to RR +0 dB
LP LR4 100Hz  [U1]
Add U1 to SR +0 dB
"""


def test_collapse_surround():
    five_one = Matrix({'SW': 1, 'L': 2, 'R': 2, 'C': 2, 'SL': 1, 'SR': 1}, ['L', 'R', 'C', 'SW', 'SL', 'SR', 'RL', 'RR'])
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
