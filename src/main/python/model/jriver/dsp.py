from __future__ import annotations

import logging
import time
from builtins import isinstance
from typing import Dict, List, Tuple, Type, Callable

from model.jriver.codec import get_peq_block_order, get_output_format, NoFiltersError, get_peq_key_name, \
    extract_filters, filts_to_xml, include_filters_in_dsp, item_to_dicts
from model.jriver.formats import user_channel_indexes, get_channel_name, get_channel_idx, OutputFormat
from model.jriver.filter import FilterGraph, create_single_filter, Filter, Divider, complex_filter_classes_by_type, \
    set_filter_ids
from model.log import to_millis
from model.signal import Signal

logger = logging.getLogger('jriver.dsp')


class JRiverDSP:

    def __init__(self, name: str, txt_provider: Callable[[], str], colours: Tuple[str, str] = (None,),
                 on_delta: Callable[[bool, bool], None] = None, convert_q: bool = False, allow_padding: bool = False,
                 use_atmos_channels: bool = False):
        self.__active_idx = 0
        self.__on_delta = on_delta
        self.__filename = name
        self.__colours = colours
        self.__use_atmos_channels = use_atmos_channels
        start = time.time()
        self.__input_config_txt = txt_provider()
        peq_block_order = get_peq_block_order(self.__input_config_txt)
        self.__output_format: OutputFormat = get_output_format(self.__input_config_txt, allow_padding)
        self.__graphs: List[FilterGraph] = []
        self.__signals: Dict[str, Signal] = {}
        for block in peq_block_order:
            out_names = self.channel_names(output=True)
            in_names = out_names if self.__graphs else self.channel_names(output=False)
            try:
                mc_filters = self.__parse_peq(self.__input_config_txt, block, convert_q)
            except NoFiltersError:
                mc_filters = []
            self.__graphs.append(FilterGraph(block, in_names, out_names, mc_filters, on_delta=on_delta,
                                             convert_q=convert_q))
        end = time.time()
        logger.info(f"Parsed {name} in {to_millis(start, end)}ms")

    @property
    def output_format(self) -> OutputFormat:
        return self.__output_format

    @property
    def filename(self):
        return self.__filename

    @property
    def signals(self) -> List[Signal]:
        return list(self.__signals.values()) if self.__signals else []

    @property
    def graph_count(self) -> int:
        return len(self.__graphs)

    def graph(self, idx) -> FilterGraph:
        return self.__graphs[idx]

    def as_dot(self, idx, vertical=True, selected_nodes=None) -> str:
        return self.graph(idx).render(colours=self.__colours, vertical=vertical, selected_nodes=selected_nodes)

    def channel_names(self, short=True, output=False, exclude_user=False):
        idxs = self.output_format.get_output_channel_indexes(self.__use_atmos_channels) if output \
            else self.output_format.get_input_channel_indexes(self.__use_atmos_channels)
        return [get_channel_name(i, short=short) for i in idxs if not exclude_user or i not in user_channel_indexes()]

    @staticmethod
    def channel_name(i):
        return get_channel_name(i)

    def __parse_peq(self, xml, block, convert_q):
        peq_block = get_peq_key_name(block)
        _, filt_element = extract_filters(xml, peq_block)
        filt_fragments = [v + ')' for v in filt_element.text.split(')') if v]
        if len(filt_fragments) < 2:
            raise ValueError('Invalid input file - Unexpected <Value> format')
        individual_filters = [create_single_filter(self.__migrate_channels(d), convert_q)
                              for d in [item_to_dicts(f) for f in filt_fragments[2:]] if d]
        return self.__extract_custom_filters(individual_filters)

    def __migrate_channels(self, vals: Dict[str, str]) -> Dict[str, str]:
        '''
        Migrates a parsed filter's raw channel indexes onto the currently active channel scheme (see
        OutputFormat.migrate_channel_index) - so a filter recorded against the legacy numbered pool by
        an older config/client keeps controlling the same real channel once use_atmos_channels flips
        the declared output/input set over to the 35.0.39+ Atmos+Extra scheme. Most filter types carry
        their channel(s) in a semicolon-joined 'Channels' value; Mix is the one type that instead uses
        single-channel 'Source'/'Destination' values.
        '''
        ch = vals.get('Channels', None)
        if ch:
            migrated = ';'.join(str(self.__output_format.migrate_channel_index(int(c), self.__use_atmos_channels))
                                for c in ch.split(';'))
            if migrated != ch:
                vals = {**vals, 'Channels': migrated}
        for key in ('Source', 'Destination'):
            val = vals.get(key, None)
            if val is not None:
                migrated_val = str(self.__output_format.migrate_channel_index(int(val), self.__use_atmos_channels))
                if migrated_val != val:
                    vals = {**vals, key: migrated_val}
        return vals

    def __migrate_channel_name(self, name: str) -> str:
        '''
        The name-based counterpart to __migrate_channels - some complex filter types (Multiway/XO/
        XOBM) cache a channel name directly in their own Divider metadata rather than deriving it
        purely from their constituent filters' raw indexes (see
        ComplexFilter.migrate_channel_metadata). Falls back to returning name unchanged for anything
        that isn't a recognised channel short name.
        '''
        try:
            idx = get_channel_idx(name)
        except ValueError:
            return name
        return get_channel_name(self.__output_format.migrate_channel_index(idx, self.__use_atmos_channels))

    def __extract_custom_filters(self, individual_filters: List[Filter]) -> List[Filter]:
        '''
        Combines individual filters into ComplexFilter instances based on divider text.
        :param individual_filters: the raw filters.
        :return: the coalesced filters.
        '''
        output_filters: List[Filter] = []
        buffer_stack: List[Tuple[Type, str, List[Filter]]] = []
        for f in individual_filters:
            if isinstance(f, Divider):
                self.__handle_divider(buffer_stack, output_filters, f)
            else:
                store_in = buffer_stack[-1][2] if buffer_stack else output_filters
                store_in.append(f)
        return set_filter_ids(output_filters)

    def __handle_divider(self, buffer: List[Tuple[Type, str, List[Filter]]], output_filters: List[Filter],
                         f: Divider):
        match = next((c.get_complex_filter_data(f.text) for c in complex_filter_classes_by_type.values()
                      if c.get_complex_filter_data(f.text)), None)
        if match is None:
            if buffer:
                buffer[-1][2].append(f)
            else:
                logger.debug(f"Passing through divider outside complex filter parsing - {f.text}")
                output_filters.append(f)
        else:
            filt_cls, data = match
            is_end = filt_cls.is_end_of_complex_filter_data(f.text)
            if is_end:
                if buffer:
                    if filt_cls == buffer[-1][0]:
                        _, meta, accumulated = buffer.pop()
                        migrated_meta = filt_cls.migrate_channel_metadata(meta, self.__migrate_channel_name)
                        complex_filt = filt_cls.create(migrated_meta, accumulated)
                        store_in = buffer[-1][2] if buffer else output_filters
                        store_in.append(complex_filt)
                    else:
                        raise ValueError(f"Mismatched start/end complex filter detected {buffer[0]} vs {filt_cls}")
                else:
                    raise ValueError(f"Empty complex filter {buffer}")
            else:
                buffer.append((filt_cls, data, []))
        return buffer

    def __repr__(self):
        return f"{self.__filename}"

    def activate(self, active_idx: int) -> None:
        '''
        Activates the selected graph & generates signals accordingly.
        :param active_idx: the active graph index.
        '''
        self.__active_idx = active_idx
        self.active_graph.activate()
        self.simulate()

    def simulate(self):
        self.__signals = self.active_graph.simulate()

    @property
    def active_graph(self) -> FilterGraph:
        return self.graph(self.__active_idx)

    def write_to_file(self, file=None) -> None:
        '''
        Writes the dsp config to the default file or the file provided.
        :param file: the file, if any.
        '''
        output_file = self.filename if file is None else file
        logger.info(f"Writing to {output_file}")
        with open(output_file, mode='w', newline='\r\n') as f:
            f.write(self.config_txt())
        logger.info(f"Written new config to {output_file}")

    def config_txt(self, convert_q: bool = False) -> str:
        new_txt = self.__input_config_txt
        for graph in self.__graphs:
            xml_filts = [filts_to_xml(f.get_all_vals(convert_q=convert_q)) for f in graph.filters]
            new_txt = include_filters_in_dsp(get_peq_key_name(graph.stage), new_txt, xml_filts)
        return new_txt
