import logging
from abc import abstractmethod, ABC
from typing import List, Optional, Tuple, Callable
from uuid import uuid4

from qtpy.QtCore import QObject, Signal, QRunnable
from qtpy.QtWidgets import QFileDialog

from model.iir import Passthrough, PeakingEQ, Shelf, LowShelf, HighShelf, Biquad
from model.preferences import BEQ_DOWNLOAD_DIR

logger = logging.getLogger('minidsp')


class XmlParser(ABC):
    def __init__(self, minidsp_type, optimise_filters, pad=True):
        self.__minidsp_type = minidsp_type
        self.__optimise_filters = optimise_filters
        self.__should_pad = pad

    @property
    def minidsp_type(self):
        return self.__minidsp_type

    def __preprocess(self, filt):
        was_optimised = False
        fs = self.minidsp_type.target_fs
        filters_required = self.minidsp_type.filters_required
        filters = flatten_filters(filt)
        if self.__should_pad is True:
            padding, filters = pad_with_passthrough(filters, fs=fs, required=filters_required)
            if padding < 0:
                if self.__optimise_filters is True:
                    from model.filter import optimise_filters
                    padding, filters = pad_with_passthrough(optimise_filters(filt, fs, -padding), fs, filters_required)
                    was_optimised = True
                else:
                    raise TooManyFilters(f"BEQ has too many filters for device (remove {abs(padding)} biquads)")
        return filters, was_optimised

    def convert(self, dst, filt, metadata=None):
        filters, was_optimised = self.__preprocess(filt)
        output_config = self._overwrite(self.__ensure_fs(filters), dst, metadata)
        return output_config, was_optimised

    def __ensure_fs(self, filters):
        fs = self.minidsp_type.target_fs
        return [f.resample(fs) if f.fs != fs else f for f in filters]

    @staticmethod
    def file_extension():
        return '.xml'

    @staticmethod
    def newline():
        return None

    @abstractmethod
    def _overwrite(self, filters, target, metadata=None):
        pass


class TwoByFourXmlParser(XmlParser):
    '''
    Handles the 2x4 model
    '''
    def __init__(self, minidsp_type, optimise_filters):
        super().__init__(minidsp_type, optimise_filters)

    def _overwrite(self, filters, target, metadata=None):
        import xml.etree.ElementTree as ET
        import re
        logger.info(f"Copying {len(filters)} to {target}")
        et_tree = ET.parse(target)
        root = et_tree.getroot()
        filter_matcher = re.compile('^EQ_ch([1-2])_1_([1-6])$')
        bq_matcher = re.compile('^EQ_ch([1-2])_1_([1-6])_([A-B][0-2])$')
        for child in root:
            if child.tag == 'filter':
                filter_name = child.attrib['name']
                matches = filter_matcher.match(filter_name)
                if matches is not None and len(matches.groups()) == 2:
                    filt_slot = matches.group(2)
                    if int(filt_slot) > len(filters):
                        root.remove(child)
                    else:
                        filt = filters[int(filt_slot) - 1]
                        if isinstance(filt, Passthrough):
                            child.find('freq').text = '1000'
                            child.find('q').text = '1'
                            child.find('gain').text = '0'
                            child.find('boost').text = '0'
                            child.find('type').text = 'PK'
                            child.find('bypass').text = '1'
                            child.find('basic').text = 'true'
                        else:
                            child.find('freq').text = str(filt.freq)
                            child.find('q').text = str(round(filt.q, 4))
                            child.find('boost').text = str(filt.gain)
                            child.find('type').text = get_minidsp_filter_code(filt)
                            child.find('bypass').text = '0'
                            child.find('basic').text = 'true'
            elif child.tag == 'item':
                filter_name = child.attrib['name']
                matches = bq_matcher.match(filter_name)
                if matches is not None and len(matches.groups()) == 3:
                    filt_slot = matches.group(2)
                    biquad_coeff = matches.group(3)
                    if int(filt_slot) > len(filters):
                        root.remove(child)
                    else:
                        filt = filters[int(filt_slot) - 1]
                        if isinstance(filt, Passthrough):
                            child.find('dec').text = '0'
                            child.find('hex').text = '00800000' if biquad_coeff == 'B0' else '00800000'
                        else:
                            child.find('dec').text = '0'
                            hex_txt = filt.format_biquads(True, separator=',', show_index=True, to_hex=True,
                                                          fixed_point=True)[0]
                            hex_val = dict(item.split("=") for item in hex_txt.split(','))[biquad_coeff.lower()]
                            child.find('hex').text = hex_val

        return ET.tostring(root, encoding='unicode')


class HDXmlParser(XmlParser):
    '''
    Handles HD models (2x4HD and 10x10HD)
    '''
    def __init__(self, minidsp_type, optimise_filters, selected_channels=None, in_out_split=None):
        super().__init__(minidsp_type, optimise_filters, pad=in_out_split is None)
        self.__in_out_split = in_out_split
        if selected_channels:
            self.__selected_channels = [self.__extract_channel(i, minidsp_type) for i in selected_channels]
        else:
            self.__selected_channels = minidsp_type.filter_channels

    @staticmethod
    def __extract_channel(txt, minidsp_type):
        if len(txt) == 1:
            return txt[0]
        elif txt[0:5] == 'Input':
            return txt[-1]
        elif txt[0:6] == 'Output':
            return str(int(txt[-1]) + minidsp_type.input_channel_count)
        else:
            raise ValueError(f"Unsupported channel {txt}")

    def __should_overwrite(self, filt_channel, filt_slot):
        '''
        :param filt_channel: the filter channel.
        :param filt_slot: the filter slot.
        :param filt_idx: the filter index.
        :return: True if this filter should be overwritten.
        '''
        if self.__in_out_split is None:
            return filt_channel in self.__selected_channels
        else:
            if filt_channel == '1' or filt_channel == '2':
                return int(filt_slot) <= int(self.__in_out_split[0])
            else:
                return int(filt_slot) <= int(self.__in_out_split[1])

    def _overwrite(self, filters, target, metadata=None):
        '''
        Overwrites the PEQ_1_x and PEQ_2_x filters (or the 1-4 filters for the SHD).
        :param filters: the filters.
        :param target: the target file.
        :param metadata: the minidsp metadata.
        :return: the xml to output to the file.
        '''
        import xml.etree.ElementTree as ET
        logger.info(f"Copying {len(filters)} to {target}")
        et_tree = ET.parse(target)
        root = et_tree.getroot()
        for child in root:
            if child.tag == 'filter':
                if 'name' in child.attrib:
                    filter_tokens = child.attrib['name'].split('_')
                    if len(filter_tokens) == 3:
                        (filt_type, filt_channel, filt_slot) = filter_tokens
                        if filt_type == 'PEQ':
                            if self.__should_overwrite(filt_channel, filt_slot):
                                if int(filt_slot) > len(filters) and self.__in_out_split is None:
                                    root.remove(child)
                                else:
                                    idx = int(filt_slot)-1
                                    if idx < len(filters):
                                        filt = filters[idx]
                                        if isinstance(filt, Passthrough):
                                            child.find('freq').text = '1000'
                                            child.find('q').text = '0.7'
                                            child.find('boost').text = '0'
                                            child.find('type').text = 'PK'
                                            child.find('bypass').text = '1'
                                        else:
                                            child.find('freq').text = str(filt.freq)
                                            child.find('q').text = str(round(filt.q, 4))
                                            child.find('boost').text = str(filt.gain)
                                            child.find('type').text = get_minidsp_filter_code(filt)
                                            child.find('bypass').text = '0'
                                        dec_txt = filt.format_biquads(True, separator=',',
                                                                      show_index=False, to_hex=False)[0]
                                        child.find('dec').text = f"{dec_txt},"
                                        hex_txt = filt.format_biquads(True, separator=',',
                                                                      show_index=False, to_hex=True,
                                                                      fixed_point=self.minidsp_type.is_fixed_point_hardware())[0]
                                        child.find('hex').text = f"{hex_txt},"
                                    else:
                                        if self.__in_out_split is None:
                                            raise ValueError(f"Missing filter at idx {idx}")
        if metadata is not None:
            metadata_tag = ET.Element('beq_metadata')
            for key, value in metadata.items():
                tag = ET.Element(key)

                if isinstance(value,list):
                    subKey = key[key.startswith("beq_") and len("beq_"):]
                    subKey = subKey[:-1]
                    for item in value:
                        sub_tag = ET.Element(subKey)
                        if isinstance(item,dict):
                            sub_tag.text = item.get("name")
                            sub_tag.set('id', str(item.get("id")))
                        else:
                            sub_tag.text = item
                        tag.append(sub_tag)
                elif isinstance(value,dict):
                    tag.text = value.get("name")
                    tag.set('id', str(value.get("id")))
                else:
                    tag.text = value
                metadata_tag.append(tag)

            root.append(metadata_tag)

        return ET.tostring(root, encoding='unicode')


def get_minidsp_filter_code(filt):
    '''
    :param filt: the filter.
    :return: the string filter type for a minidsp xml.
    '''
    if isinstance(filt, PeakingEQ):
        return 'PK'
    elif isinstance(filt, LowShelf):
        return 'SL'
    elif isinstance(filt, HighShelf):
        return 'SH'
    else:
        raise ValueError(f"Unknown minidsp filter type {type(filt)}")


def xml_to_filt(file, fs=1000, unroll=False) -> List[Biquad]:
    ''' Extracts a set of filters from the provided minidsp file '''
    from model.iir import PeakingEQ, LowShelf, HighShelf

    filts = __extract_filters(file)
    output = []
    for filt_tup, count in filts.items():
        filt_dict = dict(filt_tup)
        if filt_dict['type'] == 'SL':
            for i in range(0, count if unroll is True else 1):
                filt = LowShelf(fs, float(filt_dict['freq']), float(filt_dict['q']), float(filt_dict['boost']),
                                count=1 if unroll is True else count)
                output.append(filt)
        elif filt_dict['type'] == 'SH':
            for i in range(0, count if unroll is True else 1):
                filt = HighShelf(fs, float(filt_dict['freq']), float(filt_dict['q']), float(filt_dict['boost']),
                                 count=1 if unroll is True else count)
                output.append(filt)
        elif filt_dict['type'] == 'PK':
            for i in range(0, count):
                filt = PeakingEQ(fs, float(filt_dict['freq']), float(filt_dict['q']), float(filt_dict['boost']))
                output.append(filt)
        else:
            logger.info(f"Ignoring unknown filter type {filt_dict}")
    return output


def __extract_filters(file):
    import xml.etree.ElementTree as ET
    from collections import Counter

    ignore_vals = ['hex', 'dec']
    tree = ET.parse(file)
    root = tree.getroot()
    filts = {}
    for child in root:
        if child.tag == 'filter':
            if 'name' in child.attrib:
                current_filt = None
                filter_tokens = child.attrib['name'].split('_')
                (filt_type, filt_channel, filt_slot) = filter_tokens
                if len(filter_tokens) == 3:
                    if filt_type == 'PEQ':
                        if filt_channel not in filts:
                            filts[filt_channel] = {}
                        filt = filts[filt_channel]
                        if filt_slot not in filt:
                            filt[filt_slot] = {}
                        current_filt = filt[filt_slot]
                        for val in child:
                            if val.tag not in ignore_vals:
                                current_filt[val.tag] = val.text
                if current_filt is not None:
                    if 'bypass' in current_filt and current_filt['bypass'] == '1':
                        del filts[filt_channel][filt_slot]
                    elif 'boost' in current_filt and current_filt['boost'] == '0':
                        del filts[filt_channel][filt_slot]
    final_filt = None
    # if 1 and 2 are identical then throw one away
    if '1' in filts and '2' in filts:
        filt_1 = filts['1']
        filt_2 = filts['2']
        if filt_1 == filt_2:
            final_filt = list(filt_1.values())
        else:
            raise ValueError(f"Different input filters found in {file} - Input 1: {filt_1} - Input 2: {filt_2}")
    elif '1' in filts:
        final_filt = list(filts['1'].values())
    elif '2' in filts:
        final_filt = list(filts['2'].values())
    else:
        if len(filts.keys()) == 1:
            for k in filts.keys():
                final_filt = filts[k]
        else:
            raise ValueError(f"Multiple active filters found in {file} - {filts}")
    if final_filt is None:
        raise ValueError(f"No filters found in {file}")
    return Counter([tuple(f.items()) for f in final_filt])


def pad_with_passthrough(filters, fs, required):
    '''
    Pads to the required number of biquads.
    :param filters: the filters.
    :param fs: sample rate.
    :param required: no of required biquads.
    :return: the raw biquad filters.
    '''
    padding = required - len(filters)
    if padding > 0:
        pad_filters = [Passthrough(fs=fs, f_id=uuid4())] * padding
        filters.extend(pad_filters)
    return padding, filters


def flatten_filters(filter):
    '''
    Flattens the provided filter, i.e. unrolls shelf filters.
    :param filter: the filter.
    :return: the flattened filters as a list.
    '''
    flattened_filters = []
    for filt in filter:
        if isinstance(filt, PeakingEQ):
            flattened_filters.append(filt)
        elif isinstance(filt, Shelf):
            flattened_filters.extend(filt.flatten())
    return flattened_filters


class TooManyFilters(Exception):
    pass


class OptimisedFilters(Exception):
    def __init__(self, flattened_filters):
        self.flattened_filters = flattened_filters


def load_as_filter(parent, preferences, fs, unroll=False) -> Tuple[Optional[List[Biquad]], Optional[str]]:
    '''
    allows user to select a minidsp xml file and load it as a filter.
    '''
    selected = QFileDialog.getOpenFileName(parent=parent, directory=preferences.get(BEQ_DOWNLOAD_DIR),
                                           caption='Load Minidsp XML Filter', filter='Filter (*.xml)')
    filt_file = selected[0] if selected is not None else None
    if filt_file is not None and len(filt_file) > 0:
        return load_filter_file(filt_file, fs, unroll=unroll), filt_file
    return None, None


def load_filter_file(filt_file, fs, unroll=False) -> Optional[List[Biquad]]:
    '''
    loads a given minidsp xml file as a filter.
    :param filt_file: the file.
    :param fs: the fs.
    :param unroll: whether to unroll the filter.
    :return: filter
    '''
    filt = xml_to_filt(filt_file, fs, unroll=unroll)
    if filt is not None and len(filt) > 0:
        for f in filt:
            f.id = uuid4()
        return filt
    return None


class FilterPublisherSignals(QObject):
    ON_START: int = 1
    ON_COMPLETE: int = 0
    ON_ERROR: int = 2

    on_status = Signal(int, name='on_status')


class FilterPublisher(QRunnable):

    def __init__(self, filt: List[Biquad], slot: Optional[int], minidsp_rs_binary: str, minidsp_rs_options: str,
                 status_handler: Callable[[int], None]):
        super().__init__()
        self.__signals = FilterPublisherSignals()
        self.__slot = slot
        self.__filt = filt
        self.__signals.on_status.connect(status_handler)
        self.__signals.on_status.emit(FilterPublisherSignals.ON_START)
        from plumbum import local
        cmd = local[minidsp_rs_binary]
        if minidsp_rs_options:
            self.__runner = cmd[minidsp_rs_options.split(' ')]
        else:
            self.__runner = cmd

    def run(self):
        try:
            if self.__slot:
                self.__send_config()
            for c in range(2):
                idx = 0
                for f in self.__filt:
                    for bq in f.format_biquads(True, separator='|', show_index=False):
                        coeffs = bq.split('|')
                        if len(coeffs) != 5:
                            raise ValueError(f"Invalid coeff count {len(coeffs)} at idx {idx}")
                        else:
                            self.__send_biquad(str(c), str(idx), coeffs)
                            idx += 1
                for i in range(idx, 10):
                    self.__send_bypass(str(c), str(i), True)
            self.__signals.on_status.emit(FilterPublisherSignals.ON_COMPLETE)
        except Exception as e:
            logger.exception(f"Unexpected failure during filter publication")
            self.__signals.on_status.emit(FilterPublisherSignals.ON_ERROR)

    def __send_config(self):
        # minidsp config <slot>
        cmd = self.__runner['config', str(self.__slot)]
        logger.info(f"Executing {cmd}")
        cmd.run(timeout=5)

    def __send_biquad(self, channel: str, idx: str, coeffs: List[str]):
        # minidsp input <channel> peq <index> set -- <b0> <b1> <b2> <a1> <a2>
        cmd = self.__runner['input', channel, 'peq', idx, 'set', '--', coeffs]
        logger.info(f"Executing {cmd}")
        cmd.run(timeout=5)
        self.__send_bypass(channel, idx, False)

    def __send_bypass(self, channel: str, idx: str, bypass: bool):
        # minidsp input <channel> bypass on
        cmd = self.__runner['input', channel, 'peq', idx, 'bypass', 'on' if bypass else 'off']
        logger.info(f"Executing {cmd}")
        cmd.run(timeout=5)
