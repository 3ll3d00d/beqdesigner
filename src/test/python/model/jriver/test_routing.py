from typing import Dict

import pytest

from model.jriver.common import get_channel_idx
from model.jriver.filter import Mix, MixType, Gain
from model.jriver.routing import Matrix, calculate_compound_routing_filter, convert_to_routes


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

    routing_filters = calculate_compound_routing_filter(stereo).filters
    assert routing_filters
    assert routing_filters.pop(0).get_all_vals() == [mix('R', MixType.COPY, 'U1')]
    assert routing_filters.pop(0).get_all_vals() == [mix('L', MixType.COPY, 'R')]
    assert routing_filters.pop(0).get_all_vals() == [mix('U1', MixType.COPY, 'L')]
    assert not routing_filters


@pytest.fixture
def two_in_four() -> Matrix:
    return Matrix({'L': 2, 'R': 2}, ['L', 'R', 'C', 'SW'])


def test_two_in_four_passthrough(two_in_four):
    two_in_four.enable('L', 0, 'L')
    two_in_four.enable('L', 1, 'C')
    two_in_four.enable('R', 0, 'R')
    two_in_four.enable('R', 1, 'SW')
    routing_filters = calculate_compound_routing_filter(two_in_four).filters
    assert routing_filters
    assert routing_filters.pop(0).get_all_vals() == [mix('L', MixType.COPY, 'C')]
    assert routing_filters.pop(0).get_all_vals() == [mix('R', MixType.COPY, 'SW')]
    assert not routing_filters


def test_two_in_four_in_order(two_in_four):
    two_in_four.enable('L', 0, 'L')
    two_in_four.enable('L', 1, 'R')
    two_in_four.enable('R', 0, 'C')
    two_in_four.enable('R', 1, 'SW')
    routing_filters = calculate_compound_routing_filter(two_in_four).filters
    assert routing_filters
    assert routing_filters.pop(0).get_all_vals() == [mix('R', MixType.COPY, 'C')]
    assert routing_filters.pop(0).get_all_vals() == [mix('R', MixType.COPY, 'SW')]
    assert routing_filters.pop(0).get_all_vals() == [mix('L', MixType.COPY, 'R')]
    assert not routing_filters


def test_two_in_four_circular(two_in_four):
    two_in_four.enable('L', 0, 'R')
    two_in_four.enable('L', 1, 'C')
    two_in_four.enable('R', 0, 'L')
    two_in_four.enable('R', 1, 'SW')
    routing_filters = calculate_compound_routing_filter(two_in_four).filters
    assert routing_filters
    assert routing_filters.pop(0).get_all_vals() == [mix('R', MixType.COPY, 'SW')]
    assert routing_filters.pop(0).get_all_vals() == [mix('R', MixType.COPY, 'U1')]
    assert routing_filters.pop(0).get_all_vals() == [mix('L', MixType.COPY, 'R')]
    assert routing_filters.pop(0).get_all_vals() == [mix('L', MixType.COPY, 'C')]
    assert routing_filters.pop(0).get_all_vals() == [mix('U1', MixType.COPY, 'L')]
    assert not routing_filters


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
    routing_filters = calculate_compound_routing_filter(two_in_eight).filters
    assert routing_filters
    assert routing_filters.pop(0).get_all_vals() == [mix('L', MixType.COPY, 'C')]
    assert routing_filters.pop(0).get_all_vals() == [mix('L', MixType.COPY, 'SW')]
    assert routing_filters.pop(0).get_all_vals() == [mix('L', MixType.COPY, 'SL')]
    assert routing_filters.pop(0).get_all_vals() == [mix('R', MixType.COPY, 'SR')]
    assert routing_filters.pop(0).get_all_vals() == [mix('R', MixType.COPY, 'RL')]
    assert routing_filters.pop(0).get_all_vals() == [mix('R', MixType.COPY, 'RR')]
    assert not routing_filters


def test_two_in_eight_in_order(two_in_eight):
    two_in_eight.enable('L', 0, 'L')
    two_in_eight.enable('L', 1, 'R')
    two_in_eight.enable('L', 2, 'C')
    two_in_eight.enable('L', 3, 'SW')
    two_in_eight.enable('R', 0, 'SL')
    two_in_eight.enable('R', 1, 'SR')
    two_in_eight.enable('R', 2, 'RL')
    two_in_eight.enable('R', 3, 'RR')
    routing_filters = calculate_compound_routing_filter(two_in_eight).filters
    assert routing_filters
    assert routing_filters.pop(0).get_all_vals() == [mix('R', MixType.COPY, 'SL')]
    assert routing_filters.pop(0).get_all_vals() == [mix('R', MixType.COPY, 'SR')]
    assert routing_filters.pop(0).get_all_vals() == [mix('R', MixType.COPY, 'RL')]
    assert routing_filters.pop(0).get_all_vals() == [mix('R', MixType.COPY, 'RR')]
    assert routing_filters.pop(0).get_all_vals() == [mix('L', MixType.COPY, 'R')]
    assert routing_filters.pop(0).get_all_vals() == [mix('L', MixType.COPY, 'C')]
    assert routing_filters.pop(0).get_all_vals() == [mix('L', MixType.COPY, 'SW')]
    assert not routing_filters


def test_two_in_eight_shared_sub(two_in_eight):
    two_in_eight.enable('L', 0, 'L')
    two_in_eight.enable('L', 1, 'R')
    two_in_eight.enable('L', 2, 'C')
    two_in_eight.enable('L', 3, 'SW')
    two_in_eight.enable('R', 0, 'L')
    two_in_eight.enable('R', 1, 'SL')
    two_in_eight.enable('R', 2, 'SR')
    two_in_eight.enable('R', 3, 'RL')
    routing_filters = calculate_compound_routing_filter(two_in_eight).filters
    assert routing_filters
    assert routing_filters.pop(0).get_all_vals() == [mix('R', MixType.COPY, 'SL')]
    assert routing_filters.pop(0).get_all_vals() == [mix('R', MixType.COPY, 'SR')]
    assert routing_filters.pop(0).get_all_vals() == [mix('R', MixType.COPY, 'RL')]
    assert routing_filters.pop(0).get_all_vals() == [mix('R', MixType.COPY, 'U1')]
    assert routing_filters.pop(0).get_all_vals() == [mix('L', MixType.COPY, 'R')]
    assert routing_filters.pop(0).get_all_vals() == [mix('L', MixType.COPY, 'C')]
    assert routing_filters.pop(0).get_all_vals() == [mix('L', MixType.COPY, 'SW')]
    assert routing_filters.pop(0).get_all_vals() == [mix('U1', MixType.ADD, 'L')]
    assert not routing_filters


@pytest.fixture
def five_one() -> Matrix:
    return Matrix({'L': 2, 'R': 2, 'C': 2, 'SW': 1, 'SL': 2, 'SR': 2}, ['L', 'R', 'C', 'SW', 'SL', 'SR'])


def test_five_one_passthrough_no_gain(five_one):
    for c in ['L', 'R', 'C', 'SW', 'SL', 'SR']:
        five_one.enable(c, 0, 'SW')
        if c != 'SW':
            five_one.enable(c, 1, c)
    routing_filters = calculate_compound_routing_filter(five_one, lfe_channel_idx=5).filters
    assert routing_filters
    assert routing_filters.pop(0).get_all_vals() == [mix('L', MixType.ADD, 'SW')]
    assert routing_filters.pop(0).get_all_vals() == [mix('R', MixType.ADD, 'SW')]
    assert routing_filters.pop(0).get_all_vals() == [mix('C', MixType.ADD, 'SW')]
    assert routing_filters.pop(0).get_all_vals() == [mix('SL', MixType.ADD, 'SW')]
    assert routing_filters.pop(0).get_all_vals() == [mix('SR', MixType.ADD, 'SW')]
    assert not routing_filters


def test_five_one_passthrough_bass_manage(five_one):
    for c in ['L', 'R', 'C', 'SW', 'SL', 'SR']:
        five_one.enable(c, 0, 'SW')
        if c != 'SW':
            five_one.enable(c, 1, c)
    routing_filters = calculate_compound_routing_filter(five_one, main_adjust=-15, lfe_adjust=-5,
                                                        lfe_channel_idx=5).filters
    assert routing_filters
    assert routing_filters.pop(0).get_all_vals() == [gain('SW', -5)]
    assert routing_filters.pop(0).get_all_vals() == [mix('L', MixType.ADD, 'SW', -15)]
    assert routing_filters.pop(0).get_all_vals() == [mix('R', MixType.ADD, 'SW', -15)]
    assert routing_filters.pop(0).get_all_vals() == [mix('C', MixType.ADD, 'SW', -15)]
    assert routing_filters.pop(0).get_all_vals() == [mix('SL', MixType.ADD, 'SW', -15)]
    assert routing_filters.pop(0).get_all_vals() == [mix('SR', MixType.ADD, 'SW', -15)]
    assert not routing_filters


def test_five_one_swap_main(five_one):
    for c in ['L', 'R', 'C', 'SW', 'SL', 'SR']:
        five_one.enable(c, 0, 'SW')
        if c not in ['SW', 'SL', 'SR']:
            five_one.enable(c, 1, c)
    five_one.enable('SL', 1, 'SR')
    five_one.enable('SR', 1, 'SL')
    routing_filters = calculate_compound_routing_filter(five_one, lfe_channel_idx=5).filters
    assert routing_filters
    assert routing_filters.pop(0).get_all_vals() == [mix('L', MixType.ADD, 'SW')]
    assert routing_filters.pop(0).get_all_vals() == [mix('R', MixType.ADD, 'SW')]
    assert routing_filters.pop(0).get_all_vals() == [mix('C', MixType.ADD, 'SW')]
    assert routing_filters.pop(0).get_all_vals() == [mix('SL', MixType.ADD, 'SW')]
    assert routing_filters.pop(0).get_all_vals() == [mix('SR', MixType.ADD, 'SW')]
    assert routing_filters.pop(0).get_all_vals() == [mix('SR', MixType.COPY, 'U1')]
    assert routing_filters.pop(0).get_all_vals() == [mix('SL', MixType.COPY, 'SR')]
    assert routing_filters.pop(0).get_all_vals() == [mix('U1', MixType.COPY, 'SL')]
    assert not routing_filters


def test_five_one_swap_sub_to_c(five_one):
    for c in ['L', 'R', 'C', 'SW', 'SL', 'SR']:
        five_one.enable(c, 0, 'C')
        if c not in ['C', 'SW']:
            five_one.enable(c, 1, c)
    five_one.enable('C', 1, 'SW')
    routing_filters = calculate_compound_routing_filter(five_one, main_adjust=-15, lfe_adjust=-5,
                                                        lfe_channel_idx=5).filters
    assert routing_filters
    assert routing_filters.pop(0).get_all_vals() == [gain('SW', -5)]
    assert routing_filters.pop(0).get_all_vals() == [mix('L', MixType.ADD, 'SW', -15)]
    assert routing_filters.pop(0).get_all_vals() == [mix('R', MixType.ADD, 'SW', -15)]
    assert routing_filters.pop(0).get_all_vals() == [mix('C', MixType.ADD, 'SW', -15)]
    assert routing_filters.pop(0).get_all_vals() == [mix('SL', MixType.ADD, 'SW', -15)]
    assert routing_filters.pop(0).get_all_vals() == [mix('SR', MixType.ADD, 'SW', -15)]
    assert routing_filters.pop(0).get_all_vals() == [mix('SW', MixType.COPY, 'U1')]
    assert routing_filters.pop(0).get_all_vals() == [mix('C', MixType.COPY, 'SW')]
    assert routing_filters.pop(0).get_all_vals() == [mix('U1', MixType.COPY, 'C')]
    assert not routing_filters


@pytest.fixture
def multi_bm() -> Matrix:
    return Matrix({'L': 4, 'R': 4, 'C': 4, 'SW': 1, 'SL': 2, 'SR': 2, 'RL': 2, 'RR': 2},
                  ['L', 'R', 'C', 'SW', 'SL', 'SR', 'RL', 'RR', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16'])


def test_multi_way_mains_simple_bm(multi_bm):
    for c in ['L', 'R', 'C', 'SW', 'SL', 'SR', 'RL', 'RR']:
        multi_bm.enable(c, 0, 'SW')
        if c != 'SW':
            multi_bm.enable(c, 1, c)
    multi_bm.enable('L', 2, 'C9')
    multi_bm.enable('L', 3, 'C10')
    multi_bm.enable('R', 2, 'C11')
    multi_bm.enable('R', 3, 'C12')
    multi_bm.enable('C', 2, 'C13')
    multi_bm.enable('C', 3, 'C14')
    routing_filters = calculate_compound_routing_filter(multi_bm, lfe_channel_idx=5).filters
    assert routing_filters
    assert routing_filters.pop(0).get_all_vals() == [mix('L', MixType.ADD, 'SW')]
    assert routing_filters.pop(0).get_all_vals() == [mix('R', MixType.ADD, 'SW')]
    assert routing_filters.pop(0).get_all_vals() == [mix('C', MixType.ADD, 'SW')]
    assert routing_filters.pop(0).get_all_vals() == [mix('SL', MixType.ADD, 'SW')]
    assert routing_filters.pop(0).get_all_vals() == [mix('SR', MixType.ADD, 'SW')]
    assert routing_filters.pop(0).get_all_vals() == [mix('RL', MixType.ADD, 'SW')]
    assert routing_filters.pop(0).get_all_vals() == [mix('RR', MixType.ADD, 'SW')]
    assert routing_filters.pop(0).get_all_vals() == [mix('L', MixType.COPY, 'C9')]
    assert routing_filters.pop(0).get_all_vals() == [mix('L', MixType.COPY, 'C10')]
    assert routing_filters.pop(0).get_all_vals() == [mix('R', MixType.COPY, 'C11')]
    assert routing_filters.pop(0).get_all_vals() == [mix('R', MixType.COPY, 'C12')]
    assert routing_filters.pop(0).get_all_vals() == [mix('C', MixType.COPY, 'C13')]
    assert routing_filters.pop(0).get_all_vals() == [mix('C', MixType.COPY, 'C14')]
    assert not routing_filters


def test_multi_way_mains_simple_bm_with_shared_sub(multi_bm):
    for c in ['L', 'R']:
        multi_bm.enable(c, 0, 'C15')
        multi_bm.enable(c, 1, c)
    for c in ['C', 'SW', 'SL', 'SR', 'RL', 'RR']:
        multi_bm.enable(c, 0, 'SW')
        if c != 'SW':
            multi_bm.enable(c, 1, c)
    multi_bm.enable('L', 2, 'C9')
    multi_bm.enable('L', 3, 'C10')
    multi_bm.enable('R', 2, 'C11')
    multi_bm.enable('R', 3, 'C12')
    multi_bm.enable('C', 2, 'C13')
    multi_bm.enable('C', 3, 'C14')
    routing_filters = calculate_compound_routing_filter(multi_bm, lfe_channel_idx=5).filters
    assert routing_filters
    assert routing_filters.pop(0).get_all_vals() == [mix('C', MixType.ADD, 'SW')]
    assert routing_filters.pop(0).get_all_vals() == [mix('SL', MixType.ADD, 'SW')]
    assert routing_filters.pop(0).get_all_vals() == [mix('SR', MixType.ADD, 'SW')]
    assert routing_filters.pop(0).get_all_vals() == [mix('RL', MixType.ADD, 'SW')]
    assert routing_filters.pop(0).get_all_vals() == [mix('RR', MixType.ADD, 'SW')]
    assert routing_filters.pop(0).get_all_vals() == [mix('L', MixType.COPY, 'C9')]
    assert routing_filters.pop(0).get_all_vals() == [mix('L', MixType.COPY, 'C10')]
    assert routing_filters.pop(0).get_all_vals() == [mix('R', MixType.COPY, 'C11')]
    assert routing_filters.pop(0).get_all_vals() == [mix('R', MixType.COPY, 'C12')]
    assert routing_filters.pop(0).get_all_vals() == [mix('C', MixType.COPY, 'C13')]
    assert routing_filters.pop(0).get_all_vals() == [mix('C', MixType.COPY, 'C14')]
    assert routing_filters.pop(0).get_all_vals() == [mix('L', MixType.ADD, 'C15')]
    assert routing_filters.pop(0).get_all_vals() == [mix('R', MixType.ADD, 'C15')]
    assert not routing_filters


@pytest.fixture
def stereo_subs() -> Matrix:
    return Matrix({'L': 2, 'R': 2, 'C': 2, 'SW': 1, 'SL': 2, 'SR': 2},
                  ['L', 'R', 'C', 'SW', 'SL', 'SR', 'RL', 'RR'])


def test_stereo_subs(stereo_subs):
    for c in ['L', 'C', 'SW', 'SL']:
        stereo_subs.enable(c, 0, 'SW')
    for c in ['R', 'C', 'SW', 'SR']:
        stereo_subs.enable(c, 0, 'RR')
    for c in ['L', 'R', 'C', 'SL', 'SR']:
        stereo_subs.enable(c, 1, c)
    routing_filters = calculate_compound_routing_filter(stereo_subs, lfe_channel_idx=5).filters
    assert routing_filters
    assert routing_filters.pop(0).get_all_vals() == [mix('L', MixType.ADD, 'U1')]
    assert routing_filters.pop(0).get_all_vals() == [mix('C', MixType.ADD, 'U1')]
    assert routing_filters.pop(0).get_all_vals() == [mix('SW', MixType.ADD, 'U1')]
    assert routing_filters.pop(0).get_all_vals() == [mix('SL', MixType.ADD, 'U1')]
    assert routing_filters.pop(0).get_all_vals() == [mix('R', MixType.ADD, 'RR')]
    assert routing_filters.pop(0).get_all_vals() == [mix('C', MixType.ADD, 'RR')]
    assert routing_filters.pop(0).get_all_vals() == [mix('SW', MixType.ADD, 'RR')]
    assert routing_filters.pop(0).get_all_vals() == [mix('SR', MixType.ADD, 'RR')]
    assert routing_filters.pop(0).get_all_vals() == [mix('U1', MixType.COPY, 'SW')]
    assert not routing_filters


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


@pytest.fixture
def five_one_in_seven() -> Matrix:
    return Matrix({'L': 2, 'R': 2, 'C': 2, 'SW': 1, 'SL': 2, 'SR': 2}, ['L', 'R', 'C', 'SW', 'SL', 'SR', 'RL', 'RR'])


def test_five_one_multi_sub(five_one_in_seven):
    for c in ['L', 'R', 'C', 'SW', 'SL', 'SR']:
        five_one_in_seven.enable(c, 0, 'SW')
        five_one_in_seven.enable(c, 0, 'RL')
        if c != 'SW':
            five_one_in_seven.enable(c, 1, c)
    simple_routes, summed_routes = convert_to_routes(five_one_in_seven)
    assert not simple_routes
    assert summed_routes
