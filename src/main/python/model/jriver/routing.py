import functools
import json
import logging
import math
from collections import defaultdict
from itertools import groupby
from typing import Optional, Dict, List, Tuple

from model.jriver import ImpossibleRoutingError, UnsupportedRoutingError
from model.jriver.common import get_channel_idx, get_channel_name
from model.jriver.filter import MixType, Mix, CompoundRoutingFilter, Filter, Gain, MultiwayFilter, Delay

logger = logging.getLogger('jriver.routing')

LFE_ADJUST_KEY = 'l'
ROUTING_KEY = 'r'
EDITORS_KEY = 'e'
EDITOR_NAME_KEY = 'n'
UNDERLYING_KEY = 'u'
WAYS_KEY = 'w'
SYM_KEY = 's'
LFE_IN_KEY = 'x'


class NoMixChannelError(Exception):
    pass


@functools.total_ordering
class Route:
    def __init__(self, i: int, w: int, o: int, mt: Optional[MixType] = None):
        self.i = i
        self.w = w
        self.o = o
        self.mt = mt

    def __repr__(self):
        return f"{self.i}.{self.w} -> {self.o} {self.mt.name if self.mt else ''}"

    def pp(self) -> str:
        return f"{get_channel_name(self.i)}.{self.w} -> {get_channel_name(self.o)}"

    def __lt__(self, other):
        if isinstance(other, Route):
            delta = self.i - other.i
            if delta == 0:
                return self.w - other.w
            return delta < 0
        return False


class Matrix:

    def __init__(self, inputs: Dict[str, int], outputs: List[str]):
        self.__inputs = inputs
        self.__outputs = outputs
        self.__row_keys = self.__make_row_keys()
        # input channel -> way -> output channel -> enabled
        self.__ways: Dict[str, Dict[int, Dict[str, bool]]] = self.__make_default_ways()

    @property
    def empty_outputs(self) -> List[int]:
        return [o for o in range(self.columns) if not any(self.is_routed(i, o) for i in range(self.rows))]

    @property
    def active_routes(self) -> List[Route]:
        '''
        :return: The active links from inputs to outputs defined by this matrix.
        '''
        return [Route(get_channel_idx(k1), k2, get_channel_idx(k3))
                for k1, v1 in self.__ways.items() for k2, v2 in v1.items() for k3, v3 in v2.items() if v3]

    def __make_default_ways(self) -> Dict[str, Dict[int, Dict[str, bool]]]:
        return {i: {w: {c: False for c in self.__outputs} for w in range(ways)} for i, ways in self.__inputs.items()}

    def __make_row_keys(self) -> List[Tuple[str, int]]:
        return [(c, w) for c, ways in self.__inputs.items() for w in range(ways)]

    @property
    def rows(self) -> int:
        return len(self.__row_keys)

    def row_name(self, idx: int) -> str:
        c, w = self.__row_keys[idx]
        suffix = '' if self.__inputs[c] < 2 else f" - {w + 1}"
        return f"{c}{suffix}"

    @property
    def columns(self) -> int:
        return len(self.__outputs)

    def column_name(self, idx: int) -> str:
        return self.__outputs[idx]

    def toggle(self, row: int, column: int) -> str:
        c, w = self.__row_keys[row]
        output_channel = self.__outputs[column]
        now_enabled = not self.__ways[c][w][output_channel]
        self.__ways[c][w][output_channel] = now_enabled
        error_msg = None
        if now_enabled:
            # TODO verify
            # try:
            #     self.get_routes()
            # except ValueError as e:
            #     logger.exception(f"Unable to activate route from {c}{w} to {output_channel}: circular dependency")
            #     error_msg = 'Unable to route, circular dependency'
            #     self.__ways[c][w][output_channel] = False
            pass
        return error_msg

    def enable(self, channel: str, way: int, output: str):
        self.__ways[channel][way][output] = True

    def is_routed(self, row: int, column: int) -> bool:
        c, w = self.__row_keys[row]
        return self.__ways[c][w][self.__outputs[column]]

    def __repr__(self):
        return f"{self.__ways}"

    def clone(self):
        clone = Matrix(self.__inputs, self.__outputs)
        clone.__copy_matrix_values(self.__ways)
        return clone

    def __copy_matrix_values(self, source: Dict[str, Dict[int, Dict[str, bool]]]):
        for k1, v1 in source.items():
            for k2, v2 in v1.items():
                for k3, v3 in v2.items():
                    self.__ways[k1][k2][k3] = v3

    def resize(self, channel: str, ways: int):
        old_len = self.__inputs[channel]
        if ways < old_len:
            self.__inputs[channel] = ways
            self.__row_keys = self.__make_row_keys()
            for i in range(ways, old_len):
                del self.__ways[channel][i]
        elif ways > old_len:
            self.__inputs[channel] = ways
            self.__row_keys = self.__make_row_keys()
            old_ways = self.__ways
            self.__ways = self.__make_default_ways()
            self.__copy_matrix_values(old_ways)

    def get_mapping(self) -> Dict[str, Dict[int, str]]:
        '''
        :return: channel mapping as input channel -> way -> output channel
        '''
        mapping = defaultdict(dict)
        for input_channel, v1 in self.__ways.items():
            for way, v2 in v1.items():
                for output_channel, routed in v2.items():
                    if routed:
                        prefix = f"{mapping[input_channel][way]};" if way in mapping[input_channel] else ''
                        mapping[input_channel][way] = f"{prefix}{get_channel_idx(output_channel)}"
        return mapping

    def encode(self) -> List[str]:
        '''
        :return: currently stored routings in encoded form.
        '''
        routings = []
        for input_channel, v1 in self.__ways.items():
            for way, v2 in v1.items():
                for output_channel, routed in v2.items():
                    if routed:
                        routings.append(f"{input_channel}/{way}/{output_channel}")
        return routings

    def decode(self, routings: List[str]) -> None:
        '''
        Reloads the routing generated by encode.
        :param routings: the routings.
        '''
        for input_channel, v1 in self.__ways.items():
            for way, v2 in v1.items():
                for output_channel in v2.keys():
                    v2[output_channel] = False
        for ch, ways in groupby([r.split('/') for r in routings], lambda r: r[0]):
            ways = list(ways)
            self.resize(ch, len({w[1] for w in ways}))
            for w in ways:
                self.__ways[ch][int(w[1])][w[2]] = True

    def is_input(self, channel: str):
        return channel in self.__inputs.keys()

    def get_free_output_channels(self) -> List[int]:
        '''
        :return: the output channels which have no assigned inputs.
        '''
        used_outputs = set([k3 for k1, v1 in self.__ways.items() for k2, v2 in v1.items() for k3, v3 in v2.items()
                            if v3])
        return [get_channel_idx(o) for o in self.__outputs if o not in used_outputs]

    @property
    def input_channel_indexes(self) -> List[int]:
        return [get_channel_idx(i) for i in self.__inputs]

    @property
    def input_channel_names(self) -> List[str]:
        return [i for i in self.__inputs]

    def get_output_channels_for(self, input_channels: List[str]) -> Dict[str, List[Tuple[int, List[str]]]]:
        '''
        :param input_channels: the input channel names.
        :return: the output channel for each way by input channel.
        '''
        return {get_channel_name(c): sorted([(w, [c[1] for c in cs]) for w, cs in groupby([(r.w, get_channel_name(r.o)) for r in rs], key=lambda x: x[0])],
                                            key=lambda w: w[0])
                for c, rs in groupby([r for r in sorted(self.active_routes) if get_channel_name(r.i) in input_channels], lambda r: r.i)}


def collate_many_to_one_routes(summed_routes_by_output: Dict[int, List[Route]]) -> List[Tuple[List[int], List[Route]]]:
    '''
    Collates output channels that are fed by identical sets of inputs.
    :param summed_routes_by_output: the summed routes.
    :return: the collated routes.
    '''
    summed_routes: List[Tuple[List[int], List[Route]]] = []
    for output_channel, summed_route in summed_routes_by_output.items():
        route_inputs = sorted([f"{r.i}_{r.w}" for r in summed_route])
        matched = False
        for alt_c, alt_r in summed_routes_by_output.items():
            if alt_c != output_channel and not matched:
                alt_inputs = sorted([f"{r.i}_{r.w}" for r in alt_r])
                if alt_inputs == route_inputs:
                    matched = True
                    found = False
                    if summed_routes:
                        for s_r in summed_routes:
                            if alt_c in s_r[0] and not found:
                                found = True
                                s_r[0].append(output_channel)
                    if not found:
                        summed_routes.append(([output_channel], summed_route))
        if not matched:
            summed_routes.append(([output_channel], summed_route))
    return summed_routes


def group_routes_by_output_channel(active_routes: List[Route]) -> Tuple[List[Route], List[Tuple[List[int], List[Route]]]]:
    '''
    :param active_routes: the active routes.
    :return: the direct routes (1 input to 1 output), the summed routes grouped by output channel (multiple inputs
    going to a single output) collated so that inputs that feed the same output are collected together.
    '''
    summed_routes: Dict[int, List[Route]] = defaultdict(list)
    for r in active_routes:
        summed_routes[r.o].append(r)
    direct_routes = [Route(v1.i, v1.w, v1.o, MixType.COPY)
                     for v in summed_routes.values() if len(v) == 1 for v1 in v if v1.i != v1.o]
    summed_routes = {k: v for k, v in summed_routes.items() if len(v) > 1}
    return direct_routes, collate_many_to_one_routes(summed_routes)


def normalise_delays(channels_by_delay: Dict[float, List[str]]) -> Dict[float, List[str]]:
    if channels_by_delay:
        max_delay = max(channels_by_delay.keys())
        result: Dict[float, List[str]] = defaultdict(list)
        for k, v in channels_by_delay.items():
            normalised_delay = round(max_delay - k, 6)
            if not math.isclose(normalised_delay, 0.0):
                result[normalised_delay].extend(v)
        return result
    else:
        return {}


def calculate_compound_routing_filter(matrix: Matrix,
                                      editor_meta: Optional[List[dict]] = None,
                                      xo_induced_delay: Dict[float, List[str]] = None,
                                      xo_filters: List[MultiwayFilter] = None,
                                      main_adjust: int = 0,
                                      lfe_adjust: int = 0,
                                      lfe_channel_idx: Optional[int] = None) -> CompoundRoutingFilter:
    '''
    Calculates the filters required to route and bass manage, if necessary, the input channels.
    :param matrix: the routing matrix.
    :param editor_meta: extra editor metadata.
    :param xo_induced_delay: additional delay induced by the xo filter.
    :param xo_filters: the XO filters.
    :param main_adjust: the gain adjustment for a main channel when bass management is required.
    :param lfe_adjust: the gain adjustment for the LFE channel when bass management is required.
    :param lfe_channel_idx: the lfe channel index.
    :return: the filters.
    '''
    active_routes = matrix.active_routes
    input_channel_indexes = matrix.input_channel_indexes
    one_to_one_routes_by_output_channel, many_to_one_routes_by_output_channels = group_routes_by_output_channel(active_routes)

    illegal_routes = [r for r in one_to_one_routes_by_output_channel if r.o in input_channel_indexes and r.i != r.o]
    if illegal_routes:
        raise ImpossibleRoutingError(f"Overwriting an input channel is not supported - {illegal_routes}")

    # detect if a summed route is writing to a channel which is an input for another summed route
    if len(many_to_one_routes_by_output_channels) > 1:
        empty_outputs = matrix.empty_outputs
        for i1, vals1 in enumerate(many_to_one_routes_by_output_channels):
            output_channels1, summed_routes1 = vals1
            for i2, vals2 in enumerate(many_to_one_routes_by_output_channels):
                if i1 != i2:
                    output_channels2, summed_routes2 = vals2
                    inputs2 = {r.i for r in summed_routes2}
                    overlap = inputs2.intersection({o for o in output_channels1})
                    if overlap:
                        overlapping_channel_names = [get_channel_name(o) for o in overlap]
                        summed_routes_pp = [r.pp() for r in summed_routes2]
                        msg = f"Unable to write summed output to {overlapping_channel_names}, is input channel for {summed_routes_pp}"
                        if empty_outputs:
                            msg = f"{msg}... use an empty channel instead: {[get_channel_name(o) for o in empty_outputs]}"
                        else:
                            msg = f"{msg}... Additional output channels required to provide room for mixing"
                        if len(overlap) <= len(empty_outputs):
                            # TODO implement
                            raise UnsupportedRoutingError(msg)
                        else:
                            raise ImpossibleRoutingError(msg)

    filters: List[Filter] = []
    if lfe_channel_idx:
        if lfe_adjust != 0:
            filters.append(Gain({
                'Enabled': '1',
                'Type': Gain.TYPE,
                'Gain': f"{lfe_adjust:.7g}",
                'Channels': str(lfe_channel_idx)
            }))
    if main_adjust:
        for i in input_channel_indexes:
            if i != lfe_channel_idx:
                filters.append(Gain({
                    'Enabled': '1',
                    'Type': Gain.TYPE,
                    'Gain': f"{main_adjust:.7g}",
                    'Channels': str(i)
                }))

    normalised_delays = normalise_delays(xo_induced_delay)
    delay_filters = []

    summed_channels = [get_channel_name(i) for r in many_to_one_routes_by_output_channels for i in r[0]]
    summed_channel_delays: Dict[str, float] = {}
    for delay, channels in normalised_delays.items():
        single_route_channels = []
        for c in channels:
            if c in summed_channels:
                summed_channel_delays[c] = delay
            else:
                single_route_channels.append(str(get_channel_idx(c)))
        delay_filters.append(Delay({
            **Delay.default_values(),
            'Channels': ';'.join(single_route_channels),
            'Delay': f"{delay:.7g}",
        }))
    if summed_channel_delays:
        # TODO now what?
        pass

    summed_inputs = [y for x in many_to_one_routes_by_output_channels for y in x[0]]
    if xo_filters:
        for xo in xo_filters:
            for i, f in enumerate(xo.filters):
                if isinstance(f, Mix):
                    # only change to add if it's not the subtract used by an MDS filter
                    if f.dst_idx in summed_inputs and f.mix_type != MixType.SUBTRACT:
                        xo.filters[i] = Mix({**f.get_all_vals()[0], **{'Mode': MixType.ADD.value}})

    meta = __create_routing_metadata(matrix, editor_meta, lfe_channel_idx, lfe_adjust)
    return CompoundRoutingFilter(json.dumps(meta), filters, delay_filters, xo_filters)


def __count_lfe_routes(lfe_channel_idx, direct_routes, summed_routes_by_output_channels) -> int:
    lfe_route_count = 0
    if lfe_channel_idx:
        direct_lfe_routes = len([r.i for r in direct_routes if r.i == lfe_channel_idx])
        grouped_lfe_routes = len([1 for _, rs in summed_routes_by_output_channels
                                  if any(r.i == lfe_channel_idx for r in rs)])
        lfe_route_count += direct_lfe_routes + grouped_lfe_routes
        if lfe_route_count > 1:
            logger.debug(f"LFE is included in {lfe_route_count} routes")
    return lfe_route_count


def __create_routing_metadata(matrix: Matrix, editor_meta: Optional[List[dict]], lfe_channel: Optional[int],
                              lfe_adjust: Optional[int]) -> dict:
    meta = {
        EDITORS_KEY: editor_meta if editor_meta else [],
        ROUTING_KEY: matrix.encode(),
    }
    if lfe_channel:
        meta[LFE_IN_KEY] = lfe_channel
    if lfe_adjust:
        meta[LFE_ADJUST_KEY] = lfe_adjust
    return meta
