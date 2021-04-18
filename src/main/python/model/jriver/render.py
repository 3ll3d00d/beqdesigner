from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, Optional, List, Tuple, Iterable

from model.jriver.common import SHORT_USER_CHANNELS
from model.jriver.filter import FilterGraph, Node, collect_nodes

logger = logging.getLogger('jriver.render')


class GraphRenderer:

    def __init__(self, graph: FilterGraph, colours: Tuple[str, str] = None):
        self.__graph = graph
        self.__colours = colours

    def generate(self, vertical: bool, selected_nodes: Optional[Iterable[str]] = None) -> str:
        '''
        Generates a graphviz rendering of the graph.
        :param vertical: if true, use top to bottom alignment.
        :param selected_nodes: any nodes to highlight.
        :return: the rendered dot notation.
        '''
        gz = self.__init_gz(vertical)
        edges = self.__generate_edges()
        if edges:
            gz += self.__append_edge_definitions(edges, self.__generate_node_definitions(selected_nodes=selected_nodes))
        gz += "}"
        return gz

    @staticmethod
    def __append_edge_definitions(edges: List[List[str]], node_defs: Dict[str, str]) -> str:
        output = defaultdict(str)
        nodes_added = []
        for channel_edge in edges:
            if channel_edge[0] in SHORT_USER_CHANNELS and channel_edge[-1] == f"END:{channel_edge[0]}":
                pass
            else:
                GraphRenderer.__append_node(channel_edge[0], channel_edge[-2], node_defs, nodes_added, output)
                GraphRenderer.__append_node(channel_edge[0], channel_edge[-1], node_defs, nodes_added, output)
                output[channel_edge[0]] += f"{channel_edge[1]}{channel_edge[2]} -> {channel_edge[3]};\n"
        return '\n'.join(output.values())

    @staticmethod
    def __append_node(channel: str, node: str, node_defs: Dict[str, str], nodes_added: List[str], output: Dict[str, str]):
        node_def = node_defs.pop(node, None)
        if node_def:
            output[channel] += f"{node_def}\n"
            nodes_added.append(node)
        else:
            if ':' not in node and node not in nodes_added:
                logger.warning(f"No def found for {node}")

    def __generate_edges(self) -> List[List[str]]:
        edges_by_txt: Dict[str, List[str]] = {}
        for channel, node in self.__graph.nodes_by_channel.items():
            self.__locate_edges(channel, node, edges_by_txt)
        return self.__coalesce_edges(list(edges_by_txt.values()))

    def __coalesce_edges(self, edges: List[List[str]]) -> List[List[str]]:
        '''
        Searches for chains of add/subtract/copy operations in a single channel, collapses them and relinks
        associated nodes to the root chain.
        '''
        chains: Dict[str, List[str]] = self.__find_chains(edges)
        if chains:
            coalesced_edges: List[List[str]] = []
            for e in edges:
                found = False
                for root, coalesced in chains.items():
                    combined = [root] + coalesced
                    # ignore the chain itself
                    if e[-2] in combined and e[-1] in combined:
                        found = True
                        break
                    else:
                        match = next((c for c in coalesced if e[-1] == c), None)
                        if match:
                            coalesced_edges.append([e[0], e[1], e[-2], root])
                            found = True
                            break
                        else:
                            match = next((c for c in coalesced if e[-2] == c), None)
                            if match:
                                coalesced_edges.append([e[0], e[1], root, e[-1]])
                                found = True
                                break
                if not found:
                    coalesced_edges.append(e)
            return coalesced_edges
        else:
            return [e for e in edges]

    @staticmethod
    def __find_chains(edges: List[List[str]]) -> Dict[str, List[str]]:
        '''
        Searches the supplied edges for contiguous mix operations on the same channel.
        :param edges: the edges.
        :return: the root node vs linked nodes.
        '''
        chains: List[List[str]] = []
        for e in edges:
            if not e[-2].startswith('IN:'):
                if e[-1].endswith('Copy') \
                        or (e[-1].endswith(('Add', 'Subtract')) and e[-2].split('_')[0] == e[-1].split('_')[0]):
                    linked = False
                    for c in chains:
                        if e[-2] in c:
                            c.append(e[-1])
                            linked = True
                            break
                    if not linked:
                        chains.append(e[-2:])
        return {c[0]: c[1:] for c in chains if len(c) > 2}

    @staticmethod
    def __create_record(channels):
        return '|'.join([f"<{c}> {c}" for c in channels if c not in SHORT_USER_CHANNELS])

    @staticmethod
    def __locate_edges(channel: str, start_node: Node, visited_edges: Dict[str, List[str]]):
        for end_node in start_node.downstream:
            edge_txt = f"{start_node.name} -> {end_node.name}"
            if edge_txt not in visited_edges:
                indent = '    ' if end_node.channel in SHORT_USER_CHANNELS else '  '
                if end_node.name.startswith('OUT'):
                    target_channel = end_node.channel
                elif end_node.channel in SHORT_USER_CHANNELS:
                    target_channel = end_node.channel
                elif start_node.channel in SHORT_USER_CHANNELS:
                    target_channel = start_node.channel
                else:
                    target_channel = start_node.channel
                visited_edges[edge_txt] = [target_channel, indent, start_node.name, end_node.name]
            GraphRenderer.__locate_edges(channel, end_node, visited_edges)

    @staticmethod
    def __create_io_record(name, definition):
        return f"  {name} [shape=record label=\"{definition}\"];"

    def __create_channel_nodes(self, channel, nodes, selected_nodes: Optional[Iterable[str]] = None) -> Dict[str, str]:
        node_defs = {}
        label_prefix = f"[{channel}]\n" if channel in SHORT_USER_CHANNELS else ''
        for node in nodes:
            if node.filt and node.name != f"END:{channel}":
                txt = f"  {node.name} [label=\"{label_prefix}{node.filt.short_desc()}\""
                if selected_nodes and next((n for n in selected_nodes if n == node.name), None) is not None:
                    fill_colour = f"\"{self.__colours[1]}\"" if self.__colours else 'lightgrey'
                    txt += f" style=filled fillcolor={fill_colour}"
                txt += "]"
                node_defs[node.name] = txt
        return node_defs

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

    def __generate_node_definitions(self, selected_nodes: Optional[Iterable[str]] = None) -> Dict[str, str]:
        '''
        :param selected_nodes: the nodes to render.
        :return: node.name -> node definition (as a 2 entry tuple for main channels and user channels).
        '''
        node_defs: Dict[str, str] = {}
        for c, node in self.__graph.nodes_by_channel.items():
            node_defs |= self.__create_channel_nodes(c, collect_nodes(node, []), selected_nodes=selected_nodes)
        return node_defs


def render_dot(txt):
    from graphviz import Source
    return Source(txt, format='svg').pipe()

