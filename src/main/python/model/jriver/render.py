from __future__ import annotations

import logging
import re
from typing import Dict, Optional, List, Tuple, Iterable, Sequence

from model.jriver.common import SHORT_USER_CHANNELS, get_channel_name
from model.jriver.filter import FilterGraph, Filter, ChannelFilter, ComplexChannelFilter, Mix, \
    MixType, CompoundRoutingFilter, XOFilter

logger = logging.getLogger('jriver.render')


class GraphRenderer:

    def __init__(self, graph: FilterGraph, colours: Tuple[str, str] = None):
        self.__graph = graph
        self.__colours = colours

    def generate(self, vertical: bool = False, selected_nodes: Optional[Iterable[str]] = None) -> str:
        '''
        Generates a graphviz rendering of the graph.
        :param vertical: if true, use top to bottom alignment.
        :param selected_nodes: any nodes to highlight.
        :return: the rendered dot notation.
        '''
        gz = self.__init_gz(vertical)
        nodes: Dict[str, str] = {}
        edges: List[str] = []
        ranks: List[List[str]] = []
        last_node_by_channel: Dict[str, Optional[str]] = {
            c: f'IN:{c}' if c in self.__graph.input_channels and c not in SHORT_USER_CHANNELS else None
            for c in self.__graph.output_channels
        }
        for f in self.__graph.filters:
            f.reset()
            if isinstance(f, Sequence):
                if isinstance(f, CompoundRoutingFilter):
                    ranks.append([])
                for f1 in f:
                    f1.reset()
                    added_nodes = self.__process_filter(f1, last_node_by_channel, nodes, edges, selected_nodes)
                    if isinstance(f1, XOFilter):
                        ranks[-1].extend(added_nodes)
            else:
                self.__process_filter(f, last_node_by_channel, nodes, edges, selected_nodes)

        gz += '\n'.join(nodes.values())
        gz += '\n'
        gz += '\n'.join(edges)
        gz += '\n'

        for c, n in last_node_by_channel.items():
            if n and c not in SHORT_USER_CHANNELS:
                gz += f"  {n} -> OUT:{c};"
                gz += '\n'

        for rank in ranks:
            if rank:
                gz += f"  {{rank = same; {'; '.join(rank)};}}"
                gz += '\n'
        gz += "}"
        return gz

    def __process_filter(self, f: Filter, last_node_by_channel: Dict[str, Optional[str]], nodes: Dict[str, str],
                         edges: List[str], selected_nodes: Optional[Iterable[str]]) -> List[str]:
        added_nodes = []
        if isinstance(f, Mix):
            dst_channel = get_channel_name(f.dst_idx)
            src_channel = get_channel_name(f.src_idx)
            if f.mix_type == MixType.ADD or f.mix_type == MixType.SUBTRACT:
                last_dst = last_node_by_channel[dst_channel]
                target_name: str = "SUM" if f.mix_type == MixType.ADD else "SUBTRACT"
                if not last_dst or not re.search(fr'.*\[label=".*{target_name}"].*', nodes[last_dst],
                                                 flags=re.DOTALL):
                    added_nodes.append(self.__create_node(dst_channel, f, nodes, selected_nodes, suffix=target_name))
                    last_node_by_channel[dst_channel] = added_nodes[-1]
                    if last_dst:
                        self.__add_edge(edges, last_dst, added_nodes[-1])
                    last_dst = added_nodes[-1]
                self.__add_edge(edges, last_node_by_channel[src_channel], last_dst)
            elif f.mix_type == MixType.MOVE or f.mix_type == MixType.COPY:
                last_src_node = last_node_by_channel[src_channel]
                added_nodes.append(self.__create_node(dst_channel, f, nodes, selected_nodes))
                last_node_by_channel[dst_channel] = added_nodes[-1]
                if last_src_node:
                    self.__add_edge(edges, last_src_node, added_nodes[-1])
                    if f.mix_type == MixType.MOVE:
                        last_node_by_channel[src_channel] = None
            elif f.mix_type == MixType.SWAP:
                src_node_name = self.__create_node(src_channel, f, nodes, selected_nodes)
                dst_node_name = self.__create_node(dst_channel, f, nodes, selected_nodes)
                previous_src_node = last_node_by_channel[src_channel]
                previous_dst_node = last_node_by_channel[dst_channel]
                if previous_src_node:
                    self.__add_edge(edges, previous_src_node, src_node_name)
                if previous_dst_node:
                    self.__add_edge(edges, previous_dst_node, dst_node_name)
                last_node_by_channel[src_channel] = dst_node_name
                last_node_by_channel[dst_channel] = src_node_name
                added_nodes.extend([src_node_name, dst_node_name])
        else:
            channel_names = f.channel_names \
                if isinstance(f, ChannelFilter) or isinstance(f, ComplexChannelFilter) \
                else self.__graph.output_channels
            for c in channel_names:
                added_nodes.append(self.__create_node(c, f, nodes, selected_nodes))
                last_node = last_node_by_channel[c]
                self.__add_edge(edges, last_node, added_nodes[-1])
                last_node_by_channel[c] = added_nodes[-1]
        return added_nodes

    @staticmethod
    def __add_edge(edges: List[str], src: str, dst: str, label: str = None):
        label = f' [label="{label}"]' if label else ''
        edges.append(f"  {src} -> {dst}{label};")

    @staticmethod
    def __get_node_name(channel: str, f: Filter):
        return f"{channel}_{f.id}"

    def __create_node(self, channel: str, f: Filter, node_cache: Dict[str, str],
                      selected_nodes: Optional[Iterable[str]], suffix: str = None) -> str:
        label_prefix = f"[{channel}]\n" if channel in SHORT_USER_CHANNELS else ''
        node_name = self.__get_node_name(channel, f)
        txt = f"  {node_name} [label=\"{label_prefix}{suffix if suffix else f.short_desc()}\""
        if selected_nodes and next((n for n in selected_nodes if n == node_name), None) is not None:
            fill_colour = f"\"{self.__colours[1]}\"" if self.__colours else 'lightgrey'
            txt += f" style=filled fillcolor={fill_colour}"
        txt += "]"
        node_cache[node_name] = txt
        f.nodes.append(node_name)
        return node_name

    def __init_gz(self, vertical) -> str:
        gz = "digraph G {\n"
        gz += f"  rankdir={'TB' if vertical else 'LR'};\n"
        gz += "  node [\n"
        gz += "    shape=\"box\"\n"
        if self.__colours:
            gz += f"    color=\"{self.__colours[0]}\"\n"
            gz += f"    fontcolor=\"{self.__colours[0]}\"\n"
        gz += "  ];\n"
        if self.__colours:
            gz += "  edge [\n"
            gz += f"    color=\"{self.__colours[0]}\"\n"
            gz += "  ];\n"
            gz += "  graph [\n"
            gz += f"    color=\"{self.__colours[0]}\"\n"
            gz += f"    fontcolor=\"{self.__colours[0]}\"\n"
            gz += "  ];"
            gz += "\n"
        gz += "\n"
        gz += self.__create_io_record('IN', self.__create_record(self.__graph.input_channels))
        gz += "\n"
        gz += self.__create_io_record('OUT', self.__create_record(self.__graph.output_channels))
        gz += "\n"
        gz += "\n"
        return gz

    @staticmethod
    def __create_record(channels):
        return '|'.join([f"<{c}> {c}" for c in channels if c not in SHORT_USER_CHANNELS])

    @staticmethod
    def __create_io_record(name, definition):
        return f"  {name} [shape=record label=\"{definition}\"];"


def render_dot(txt):
    from graphviz import Source
    return Source(txt, format='svg').pipe()
