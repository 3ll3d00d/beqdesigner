import functools
import logging
from collections import defaultdict
from itertools import groupby
from typing import Dict, List, Tuple

from model.jriver.common import get_channel_idx, get_channel_name

logger = logging.getLogger('jriver.routing')


@functools.total_ordering
class Route:
    def __init__(self, i: int, w: int, o: int):
        self.i = i
        self.w = w
        self.o = o

    def __repr__(self):
        return f"{self.i}.{self.w} -> {self.o}"

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
        error_msg = ''
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

    def disable(self, channel: str, way: int, output: str):
        self.__ways[channel][way][output] = False

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

    @property
    def channel_mapping(self) -> Dict[str, Dict[int, List[str]]]:
        '''
        :return: channel mapping as input channel -> way -> output channels
        '''
        mapping: Dict[str, Dict[int, List[str]]] = {}
        for input_channel, v1 in self.__ways.items():
            for way, v2 in v1.items():
                for output_channel, routed in v2.items():
                    if routed:
                        if input_channel not in mapping:
                            mapping[input_channel] = defaultdict(list)
                        mapping[input_channel][way].append(output_channel)
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

    @property
    def free_output_channels(self) -> List[int]:
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

    def group_active_routes_by_output(self) -> Tuple[List[Route], List[Tuple[List[int], List[Route]]]]:
        '''
        :return: the direct routes (1 input to 1 output), the summed routes grouped by output channel (multiple inputs
        going to a single output) collated so that inputs that feed the same output are collected together.
        '''
        summed_routes: Dict[int, List[Route]] = defaultdict(list)
        for r in self.active_routes:
            summed_routes[r.o].append(r)
        direct_routes = [Route(v1.i, v1.w, v1.o)
                         for v in summed_routes.values() if len(v) == 1 for v1 in v if v1.i != v1.o]
        summed_routes = {k: v for k, v in summed_routes.items() if len(v) > 1}
        return direct_routes, self.__collate_many_to_one_routes(summed_routes)

    @staticmethod
    def __collate_many_to_one_routes(summed_routes_by_output: Dict[int, List[Route]]) -> List[Tuple[List[int], List[Route]]]:
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
