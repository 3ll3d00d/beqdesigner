'''
The DSP devices a BEQ can be exported to, and what each needs (sample rate, biquad slots, channels, fixed point). Qt-free:
the headless pipeline's XML export uses it.
'''
from enum import Enum


class DspType(Enum):
    MINIDSP_TWO_BY_FOUR_HD = ('2x4 HD', True, True, False, (('1', '2'), ('3', '4', '5', '6')), 'xml')
    MINIDSP_TWO_BY_FOUR = ('2x4', False, True, False, None, 'xml')
    MINIDSP_TEN_BY_TEN = ('10x10', False, True, False, None, 'xml')
    MINIDSP_SHD = ('SHD', True, True, False, None, 'xml')
    MINIDSP_EIGHTY_EIGHT_BM = ('88BM', True, True, False, None, 'xml')
    MINIDSP_HTX = ('HTx', True, True, False, None, 'xml')
    MONOPRICE_HTP1 = ('HTP-1', False, False, False, None, 'json')
    JRIVER_PEQ1 = ('JRiver PEQ1', False, False, False, None, 'dsp')
    JRIVER_PEQ2 = ('JRiver PEQ2', False, False, False, None, 'dsp')

    def __init__(self, display_name, hd_compatible, is_minidsp, is_experimental, split_channels, ext):
        self.display_name = display_name
        self.hd_compatible = hd_compatible
        self.is_minidsp = is_minidsp
        self.is_experimental = is_experimental
        self.split_channels = split_channels
        self.extension = ext

    @property
    def can_split(self):
        return self.split_channels is not None

    def is_fixed_point_hardware(self):
        return not self.hd_compatible

    @property
    def filters_required(self):
        '''
        :return: the no of filter slots expected.
        '''
        return 10 if self.hd_compatible else 6

    @property
    def target_fs(self):
        '''
        :return: the fs for the selected minidsp.
        '''
        return 96000 if self.hd_compatible else 48000

    @classmethod
    def parse(cls, dsp_type):
        return next((t for t in cls if t.display_name == dsp_type), None)

    @property
    def filter_channels(self):
        '''
        :return: list of valid channels.
        '''
        if self == DspType.MINIDSP_TEN_BY_TEN:
            return [str(x) for x in range(11, 19)]
        elif self == DspType.MINIDSP_SHD:
            return ['1', '2', '3', '4']
        elif self == DspType.MINIDSP_EIGHTY_EIGHT_BM:
            return ['3']
        elif self == DspType.MINIDSP_HTX:
            return [str(x) for x in range(1, 9)]
        else:
            return ['1', '2']

    @property
    def input_channel_count(self):
        return 0 if self == DspType.MINIDSP_HTX else 2
