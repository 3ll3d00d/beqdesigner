import logging
from collections import defaultdict
from typing import Optional, Dict, List, Tuple

from model.jriver.common import get_channel_idx, user_channel_indexes, get_channel_name
from model.jriver.filter import MixType, Mix

logger = logging.getLogger('jriver.routing')


class Route:
    def __init__(self, i: int, w: int, o: int, mt: Optional[MixType]):
        self.i = i
        self.w = w
        self.o = o
        self.mt = mt

    def __repr__(self):
        return f"{self.i}.{self.w} -> {self.o} {self.mt.name if self.mt else ''}"


class Matrix:

    def __init__(self, inputs: Dict[str, int], outputs: List[str]):
        self.__inputs = inputs
        self.__outputs = outputs
        self.__row_keys = self.__make_row_keys()
        # input channel -> way -> output channel -> enabled
        self.__ways: Dict[str, Dict[int, Dict[str, bool]]] = self.__make_default_ways()

    def get_routes(self, shift_lfe_route: Optional[Route] = None) -> List[Route]:
        '''
        :return: the routes defined by this matrix.
        '''
        routes: List[Route] = []
        if shift_lfe_route:
            routes.append(shift_lfe_route)
        for input_channel, v1 in self.__ways.items():
            for way, v2 in v1.items():
                for output_channel, routed in v2.items():
                    if routed:
                        mt = None if input_channel == output_channel else MixType.MOVE
                        routes.append(Route(get_channel_idx(input_channel), way, get_channel_idx(output_channel), mt))
        return self.__reorder_routes(self.__sort_routes(self.__fixup_mix_types(routes)))

    @staticmethod
    def __fixup_mix_types(routes: List[Route]) -> List[Route]:
        '''
        :param routes: the raw routes (all using MOVE).
        :return: the routes with mixtype corrected to use add or copy as necessary.
        '''
        inputs_received = defaultdict(int)
        outputs_routed = defaultdict(int)
        for r in routes:
            inputs_received[r.o] = inputs_received[r.o] + 1
            outputs_routed[(r.i, r.w)] = outputs_routed[(r.i, r.w)] + 1
        routes_with_mix = []
        for r in routes:
            actual_mt = r.mt
            if inputs_received[r.o] > 1 and r.i != r.o:
                actual_mt = MixType.ADD
            elif outputs_routed[(r.i, r.w)] > 0 and r.mt:
                actual_mt = MixType.COPY
            if actual_mt:
                routes_with_mix.append(Route(r.i, r.w, r.o, actual_mt))
        return routes_with_mix

    @staticmethod
    def __reorder_routes(routes: List[Route]) -> List[Route]:
        '''
        Reorders routing to ensure inputs are not overridden with outputs. Attempts to break circular dependencies
        using user channels if possible.
        :param routes: the routes.
        :return: the reordered routes.
        '''
        ordered_routes: List[Tuple[Route, int]] = []
        u1_channel_idx = user_channel_indexes()[0]
        for r in routes:
            def repack() -> Tuple[Route, int]:
                return r, -1
            # just add the first route as there is nothing to reorder
            if not ordered_routes or not r.mt:
                ordered_routes.append(repack())
            else:
                insert_at = -1
                for idx, o_r in enumerate(ordered_routes):
                    # if a route wants to write to this input, make sure this route comes first
                    if o_r[0].o == r.i:
                        insert_at = idx
                        break
                if insert_at == -1:
                    # the normal case, nothing to reorder so just add it
                    ordered_routes.append(repack())
                else:
                    # search for circular dependencies, i.e. if this route wants to write to the input of a later route
                    # (and hence overwriting that input channel)
                    broke_circular: Optional[Route] = None
                    for o_r in ordered_routes[insert_at:]:
                        if o_r[0].i == r.o:
                            inserted_route = ordered_routes[insert_at]
                            # make sure we only copy to the user channel once
                            if inserted_route[0] != r.i or inserted_route[2] != u1_channel_idx:
                                ordered_routes.insert(insert_at, (Route(r.i, r.w, u1_channel_idx, MixType.COPY), r.o))
                            # cache the copy from the user channel to the actual output
                            broke_circular = Route(u1_channel_idx, r.w, r.o, MixType.COPY)
                            break
                    if broke_circular:
                        # append the route after the last use of the route output as an input
                        candidate_idx = -1
                        for idx, o_r in enumerate(ordered_routes):
                            if o_r[0].i == broke_circular.o:
                                candidate_idx = idx + 1
                        if candidate_idx == -1:
                            raise ValueError(f"Logical error, circular dependency detected but now missing")
                        elif candidate_idx == len(ordered_routes):
                            ordered_routes.append((broke_circular, -1))
                        else:
                            ordered_routes.insert(candidate_idx, (broke_circular, -1))
                    else:
                        # no circular dependency but make sure we insert before any copies to the user channel
                        if insert_at > 0:
                            inserted = ordered_routes[insert_at - 1]
                            if inserted[1] > -1 and inserted[0].i == r.i:
                                insert_at -= 1
                        ordered_routes.insert(insert_at, repack())
        # validate that the proposed routes make sense
        u1_in_use_for = -1
        failed = False
        output: List[Route] = []
        for r, target in ordered_routes:
            if target > -1:
                if u1_in_use_for == -1:
                    u1_in_use_for = target
                else:
                    if target != u1_in_use_for:
                        failed = True
            if r.i == u1_channel_idx:
                if r.o != u1_in_use_for:
                    failed = True
                else:
                    u1_in_use_for = -1
            output.append(r)
        if failed:
            raise ValueError(f'Unresolvable circular dependencies found in {ordered_routes}')
        return output

    def __sort_routes(self, routes: List[Route]) -> List[Route]:
        '''
        applies a topological sort to the input-output routes, primarily resolves issues with bass management additions.
        :param routes: the channel routes.
        :return: the sorted routes.
        '''
        edges = defaultdict(set)
        for r in routes:
            edges[f"O{r.o}"].add(f"I{r.i}")
        for i in self.__inputs.keys():
            if i in self.__outputs:
                edges[f"O{get_channel_idx(i)}"].add(f"I{get_channel_idx(i)}")
        sorted_channels = []
        for d in Matrix.do_sort(edges):
            sorted_channels.extend(sorted(d))
        sorted_channel_mapping = {int(v[1:]): i for i, v in enumerate(sorted_channels)}
        sorted_routes = sorted(routes, key=lambda x: (sorted_channel_mapping[r.i], r.w))
        return sorted_routes

    @staticmethod
    def do_sort(to_sort):
        edges = to_sort.copy()
        from functools import reduce
        extra_items_in_deps = reduce(set.union, edges.values()) - set(edges.keys())
        edges.update({item: set() for item in extra_items_in_deps})
        while True:
            ordered = set(item for item, dep in edges.items() if len(dep) == 0)
            if not ordered:
                break
            yield ordered
            edges = {item: (dep - ordered) for item, dep in edges.items() if item not in ordered}
        if len(edges) != 0:
            formatted = ', '.join([f'{k}:{v}' for k, v in sorted(edges.items())])
            raise ValueError(f'Circular dependencies found in {formatted}')

    def __make_default_ways(self) -> Dict[str, Dict[int, Dict[str, bool]]]:
        return {i: {w: {c: False for c in self.__outputs} for w in range(ways)} for i, ways in self.__inputs.items()}

    def __make_row_keys(self) -> List[Tuple[str, int]]:
        return [(c, w) for c, ways in self.__inputs.items() for w in range(ways)]

    @property
    def rows(self):
        return len(self.__row_keys)

    def row_name(self, idx: int):
        c, w = self.__row_keys[idx]
        suffix = '' if self.__inputs[c] < 2 else f" - {w+1}"
        return f"{c}{suffix}"

    @property
    def columns(self):
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
            try:
                self.get_routes()
            except ValueError as e:
                logger.exception(f"Unable to activate route from {c}{w} to {output_channel}: circular dependency")
                error_msg = 'Unable to route, circular dependency'
                self.__ways[c][w][output_channel] = False
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
        for r in routings:
            i, w, o = r.split('/')
            self.__ways[i][int(w)][o] = True

    def is_input(self, channel: str):
        return channel in self.__inputs.keys()


def calculate_routing_filters(main_adjust: int, lfe_adjust: int, lfe_channel_idx: int, sw_channel_idx: int,
                              matrix: Matrix) -> List[Mix]:
    '''
    Calculates a set of Mix filters required to handle the routing from inputs to outputs.
    :param main_adjust: the gain adjustment for a main channel when bass management is required.
    :param lfe_adjust: the gain adjustment for the LFE channel when bass management is required.
    :param lfe_channel_idx: the lfe channel index.
    :param sw_channel_idx: the primary sw output channel index.
    :return: the filters.
    '''
    shift_lfe_route: Optional[Route] = None
    # only add the extra route if we're not copying to an existing input channel
    if lfe_channel_idx != sw_channel_idx and not matrix.is_input(get_channel_name(sw_channel_idx)):
        shift_lfe_route = Route(lfe_channel_idx, 0, sw_channel_idx, MixType.COPY)
    routes = matrix.get_routes(shift_lfe_route)
    return [convert_to_filter(r.i, r.mt, r.o, main_adjust, lfe_adjust, sw_channel_idx) for r in routes if r.mt]


def convert_to_filter(i: int, mt: MixType, o: int, main_adjust: float = 0.0, lfe_adjust: float = 0.0,
                        sw_channel: int = 0) -> Mix:
    vals = Mix.default_values()
    vals['Source'] = str(i)
    vals['Destination'] = str(o)
    vals['Mode'] = str(mt.value)
    if mt == MixType.ADD:
        adjust = main_adjust if o == sw_channel else lfe_adjust
        vals['Gain'] = f"{adjust:.7g}"
    return Mix(vals)

