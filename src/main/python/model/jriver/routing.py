import json
import logging
from collections import defaultdict
from typing import Optional, Dict, List, Tuple

from model.jriver.common import get_channel_idx, user_channel_indexes
from model.jriver.filter import MixType, Mix, CompoundRoutingFilter, Filter, Gain, XOFilter

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


class Route:
    def __init__(self, i: int, w: int, o: int, mt: Optional[MixType] = None):
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

    def get_active_routes(self) -> List[Route]:
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
        for r in routings:
            i, w, o = r.split('/')
            self.__ways[i][int(w)][o] = True

    def is_input(self, channel: str):
        return channel in self.__inputs.keys()

    def get_free_output_channels(self) -> List[int]:
        '''
        :return: the output channels which have no assigned inputs.
        '''
        used_outputs = set([k3 for k1, v1 in self.__ways.items() for k2, v2 in v1.items() for k3, v3 in v2.items()
                            if v3])
        return [get_channel_idx(o) for o in self.__outputs if o not in used_outputs]

    def get_input_channels(self) -> List[int]:
        return [get_channel_idx(i) for i in self.__inputs]


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
                        broke_circular = Route(u1_channel_idx, r.w, r.o, r.mt if r.mt else MixType.COPY)
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
        # TODO this does not make sense with how bass management is implemented for stereo subs
        logger.warning(f'Unresolvable circular dependencies found in {ordered_routes}')
    return output


def collate_routes(summed_routes_by_output: Dict[int, List[Route]]) -> List[Tuple[List[int], List[Route]]]:
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


def group_routes_by_output(matrix: Matrix):
    '''
    :param matrix: the matrix.
    :return: the direct routes (1 input to 1 output), the summed routes grouped by output channel (multiple inputs going to a single output).
    '''
    summed_routes: Dict[int, List[Route]] = defaultdict(list)
    for r in matrix.get_active_routes():
        summed_routes[r.o].append(r)
    direct_routes = [Route(v1.i, v1.w, v1.o, MixType.COPY)
                     for v in summed_routes.values() if len(v) == 1 for v1 in v if v1.i != v1.o]
    summed_routes = {k: v for k, v in summed_routes.items() if len(v) > 1}
    return direct_routes, summed_routes


def convert_to_routes(matrix: Matrix):
    '''
    :param matrix: the matrix.
    :return: the routes.
    '''
    simple_routes, summed_routes_by_output = group_routes_by_output(matrix)
    collated_summed_routes: List[Tuple[List[int], List[Route]]] = collate_routes(summed_routes_by_output)
    return simple_routes, collated_summed_routes


def __create_summed_output_channel_for_shared_lfe(output_channels: List[int],
                                                  routes: List[Route],
                                                  main_adjust: int,
                                                  lfe_channel_idx: int,
                                                  empty_channels: List[int],
                                                  input_channels: List[int]) -> Tuple[List[Filter], List[Route]]:
    '''
    converts routing of an lfe based mix to a set of filters where the LFE channel has been included in more than 1
    distinct combination of main channels.
    :param output_channels: the output channels.
    :param routes: the routes.
    :param main_adjust: main level adjustment.
    :param lfe_channel_idx: the lfe channel idx.
    :param empty_channels: channels which can be used to stage outputs.
    :return: the filters, additional direct routes (to copy the summed output to other channels).
    '''
    # calculate a target channel as any output channel which is not an input channel
    target_channel: Optional[int] = next((c for c in output_channels if c not in input_channels), None)
    if not target_channel:
        # if no such channel can be found then take the next empty channel (unlikely situation in practice)
        if empty_channels:
            target_channel = empty_channels.pop(0)
        else:
            # if no free channels left then blow up
            raise NoMixChannelError()

    filters: List[Filter] = []
    direct_routes: List[Route] = []
    # add all routes to the output
    for r in routes:
        vals = {
            **Mix.default_values(),
            'Source': str(r.i),
            'Destination': str(target_channel),
            'Mode': str(MixType.ADD.value)
        }
        if r.i != lfe_channel_idx and main_adjust != 0:
            vals['Gain'] = f"{main_adjust:.7g}"
        filters.append(Mix(vals))
    # copy from the mix target channel to the actual outputs
    for c in output_channels:
        if c != target_channel:
            direct_routes.append(Route(target_channel, 0, c, MixType.COPY))

    return filters, direct_routes


def __create_summed_output_channel_for_dedicated_lfe(output_channels: List[int], routes: List[Route], main_adjust: int,
                                                     lfe_channel_idx: int) -> Tuple[List[Filter], List[Route]]:
    '''
    converts routing of an lfe based mix to a set of filters when the LFE channel is not shared across multiple outputs.
    :param output_channels: the output channels.
    :param routes: the routes.
    :param main_adjust: main level adjustment.
    :param lfe_channel_idx: the lfe channel idx.
    :return: the filters, additional direct routes (to copy the summed output to other channels).
    '''
    filters: List[Filter] = []
    direct_routes: List[Route] = []
    # accumulate main channels into the LFE channel with an appropriate adjustment
    for r in routes:
        if r.i != lfe_channel_idx:
            vals = {
                **Mix.default_values(),
                'Source': str(r.i),
                'Destination': str(lfe_channel_idx),
                'Mode': str(MixType.ADD.value)
            }
            if r.i != lfe_channel_idx and main_adjust != 0:
                vals['Gain'] = f"{main_adjust:.7g}"
            filters.append(Mix(vals))
    # copy the LFE channel to any required output channel
    for c in output_channels:
        if c != lfe_channel_idx:
            direct_routes.append(Route(lfe_channel_idx, 0, c, MixType.COPY))
    return filters, direct_routes


def calculate_compound_routing_filter(matrix: Matrix, editor_meta: Optional[List[dict]] = None,
                                      xo_filters: List[XOFilter] = None, main_adjust: int = 0, lfe_adjust: int = 0,
                                      lfe_channel_idx: Optional[int] = None) -> CompoundRoutingFilter:
    '''
    Calculates the filters required to route and bass manage, if necessary, the input channels.
    :param matrix: the routing matrix.
    :param editor_meta: extra editor metadata.
    :param xo_filters: the XO filters.
    :param main_adjust: the gain adjustment for a main channel when bass management is required.
    :param lfe_adjust: the gain adjustment for the LFE channel when bass management is required.
    :param lfe_channel_idx: the lfe channel index.
    :return: the filters.
    '''
    direct_routes, summed_routes = group_routes_by_output(matrix)
    empty_channels = user_channel_indexes() + matrix.get_free_output_channels()
    summed_routes_by_output_channels: List[Tuple[List[int], List[Route]]] = collate_routes(summed_routes)
    lfe_route_count = __count_lfe_routes(lfe_channel_idx, direct_routes, summed_routes_by_output_channels)

    # Scenarios handled
    # 1) Standard bass management
    # = LFE channel is routed to the SW output with 1 or more other channels added to it
    #    - do nothing to the LFE channel (gain already reduced by lfe_adjust)
    #    - add other channels to the LFE channel
    # 2) Standard bass management with LFE routed to some other output channel
    #    - as 1 with additional direct route to move the SW output to the additional channel
    # 3) Standard bass management with multiple SW outputs
    #    - as 1 with additional direct route to copy the SW output to the additional channel
    # 4) Stereo bass
    # = LFE channel routed to >1 channel and combined differing sets of main channels
    #    - find a free channel to mix into
    #    - add all inputs to the channel
    #    - copy from here to the output channels
    # 5) Non LFE based subwoofers
    # = >1 main channels to be summed into some output channel and copied to some other channels
    #    - do direct routes first to copy the input channels to their target channels
    #    - mix into each summed channel
    filters: List[Filter] = []
    if lfe_adjust != 0 and lfe_channel_idx:
        filters.append(Gain({
            'Enabled': '1',
            'Type': Gain.TYPE,
            'Gain': f"{lfe_adjust:.7g}",
            'Channels': str(lfe_channel_idx)
        }))

    for output_channels, summed_route in summed_routes_by_output_channels:
        includes_lfe = lfe_channel_idx and any((r.i == lfe_channel_idx for r in summed_route))
        if includes_lfe:
            if lfe_route_count > 1:
                route_filters, extra_routes = \
                    __create_summed_output_channel_for_shared_lfe(output_channels, summed_route, main_adjust,
                                                                  lfe_channel_idx, empty_channels,
                                                                  matrix.get_input_channels())
            else:
                route_filters, extra_routes = \
                    __create_summed_output_channel_for_dedicated_lfe(output_channels, summed_route, main_adjust,
                                                                     lfe_channel_idx)
            if extra_routes:
                direct_routes.extend(extra_routes)
            if route_filters:
                filters.extend(route_filters)
        else:
            for c in output_channels:
                for r in summed_route:
                    if r.i != r.o:
                        direct_routes.append(Route(r.i, r.w, c, MixType.ADD))

    ordered_routes = __reorder_routes(direct_routes)
    for o_r in ordered_routes:
        filters.append(Mix({
            **Mix.default_values(),
            'Source': str(o_r.i),
            'Destination': str(o_r.o),
            'Mode': str(o_r.mt.value)
        }))
    meta = __create_routing_metadata(matrix, editor_meta, lfe_channel_idx, lfe_adjust)
    return CompoundRoutingFilter(json.dumps(meta), filters, xo_filters, [])


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

