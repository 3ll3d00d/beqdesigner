import logging
from pathlib import Path
from typing import List

from model.iir import ComplexFilter, SOS
from model.jriver.codec import get_peq_key_name, filts_to_xml, include_filters_in_dsp
from model.jriver.common import JRIVER_CHANNELS
from model.jriver.filter import convert_filter_to_mc_dsp

logger = logging.getLogger('jriver.parser')


class JRiverParser:

    def __init__(self, block=0, channels=('Subwoofer',)):
        self.__block = get_peq_key_name(block)
        self.__target_channels = ';'.join([str(JRIVER_CHANNELS.index(c)) for c in channels])

    def convert(self, dst, filt: ComplexFilter, **kwargs):
        from model.minidsp import flatten_filters
        flat_filts: List[SOS] = flatten_filters(filt)
        config_txt = Path(dst).read_text()
        if len(flat_filts) > 0:
            logger.info(f"Copying {len(flat_filts)} to {dst}")
            # generate the xml formatted filters
            xml_filts = [filts_to_xml(convert_filter_to_mc_dsp(f, self.__target_channels).get_all_vals())
                         for f in flat_filts]
            config_txt = include_filters_in_dsp(self.__block, config_txt, xml_filts, replace=False)
        else:
            logger.warning(f"Nop for empty filter file {dst}")
        return config_txt, False

    @staticmethod
    def file_extension():
        return '.dsp'

    @staticmethod
    def newline():
        return '\r\n'
