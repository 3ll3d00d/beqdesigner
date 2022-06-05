from enum import Enum, auto

from typing import List, Tuple, Dict

import numpy as np
from scipy.signal import unit_impulse

from model.signal import Signal

USER_CHANNELS = ['User 1', 'User 2']
SHORT_USER_CHANNELS = ['U1', 'U2']
JRIVER_NAMED_CHANNELS = [None, None, 'Left', 'Right', 'Centre', 'Subwoofer', 'Surround Left', 'Surround Right',
                         'Rear Left', 'Rear Right', None] + USER_CHANNELS
JRIVER_SHORT_NAMED_CHANNELS = [None, None, 'L', 'R', 'C', 'SW', 'SL', 'SR', 'RL', 'RR', None] + SHORT_USER_CHANNELS
JRIVER_CHANNELS = JRIVER_NAMED_CHANNELS + [f"Channel {i + 9}" for i in range(24)]
JRIVER_SHORT_CHANNELS = JRIVER_SHORT_NAMED_CHANNELS + [f"C{i + 9}" for i in range(24)]


def get_all_channel_names(short: bool = True) -> List[str]:
    contents = JRIVER_SHORT_CHANNELS if short else JRIVER_CHANNELS
    user_c = SHORT_USER_CHANNELS if short else USER_CHANNELS
    return [c for c in contents if c and c not in user_c]


def get_channel_indexes(names: List[str], short: bool = True) -> List[int]:
    contents = JRIVER_SHORT_CHANNELS if short else JRIVER_CHANNELS
    return [contents.index(n) for n in names]


def user_channel_indexes() -> List[int]:
    '''
    :return: the channel indexes of the user channels.
    '''
    return [JRIVER_SHORT_NAMED_CHANNELS.index(c) for c in SHORT_USER_CHANNELS]


def short_to_long(short: str) -> str:
    return JRIVER_CHANNELS[JRIVER_SHORT_CHANNELS.index(short)]


def get_channel_name(idx: int, short: bool = True) -> str:
    '''
    Converts a channel index to a named channel.
    :param idx: the index.
    :param short: get the short name if true.
    :return: the name.
    '''
    channels = JRIVER_SHORT_CHANNELS if short else JRIVER_CHANNELS
    return channels[idx]


def get_channel_idx(name: str, short: bool = True) -> int:
    '''
    Converts a channel name to an index.
    :param name the name.
    :param short: search via short name if true.
    :return: the index.
    '''
    channels = JRIVER_SHORT_CHANNELS if short else JRIVER_CHANNELS
    return channels.index(name)


def pop_channels(vals: List[Dict[str, str]]):
    '''
    :param vals: a set of filter values.
    :return: the values without the Channel key.
    '''
    return [{k: v for k, v in d.items() if k != 'Channels'} for d in vals]


def make_dirac_pulse(channel: str):
    fs = 48000
    return Signal(channel, unit_impulse(fs * 4, 'mid') * 23453.66, fs=fs)


def make_silence(channel: str):
    fs = 48000
    return Signal(channel, np.zeros(fs * 4), fs=fs)


class OutputFormat:

    def __init__(self, display_name: str, input_channels: int, output_channels: int, lfe_channels: int,
                 xml_vals: Tuple[int, ...], max_padding: int):
        self.__lfe_channels = lfe_channels
        self.__output_channels = output_channels
        self.__display_name = display_name
        self.__input_channels = input_channels
        self.__xml_vals = xml_vals
        self.__paddings: List[int] = range(2, max_padding + 1, 2) if max_padding > 0 else []
        all_names = get_all_channel_names()
        # special case for 2.1
        if self.__lfe_channels > 0 and self.__input_channels < 4:
            self.__input_channel_indexes = [2, 3, 5]
        else:
            self.__input_channel_indexes = get_channel_indexes(all_names[:input_channels])
        if self.__lfe_channels > 0 and self.__output_channels < 4:
            self.__output_channel_indexes = [2, 3, 5] + user_channel_indexes()
        else:
            self.__output_channel_indexes = sorted(get_channel_indexes(all_names[:output_channels]) +
                                                   user_channel_indexes())

    def __str__(self):
        return self.__display_name

    @property
    def display_name(self):
        return self.__display_name

    @property
    def input_channels(self) -> int:
        return self.__input_channels

    @property
    def output_channels(self) -> int:
        return self.__output_channels

    @property
    def input_channel_indexes(self) -> List[int]:
        return self.__input_channel_indexes

    @property
    def output_channel_indexes(self) -> List[int]:
        return self.__output_channel_indexes

    @property
    def lfe_channels(self) -> int:
        return self.__lfe_channels

    @property
    def paddings(self) -> List[int]:
        return self.__paddings

    @property
    def xml_vals(self) -> Tuple[int, ...]:
        return self.__xml_vals

    def is_compatible(self, version: int) -> bool:
        if not self.paddings:
            return version == 28
        else:
            return True

    @classmethod
    def from_output_channels(cls, count: int):
        f: OutputFormat
        for f in OUTPUT_FORMATS.values():
            if f.output_channels == count:
                return f
        raise ValueError(f"Unsupported count {count}")

    def has_channels(self, channels: List[int]):
        return set(self.output_channel_indexes).issuperset(set(channels))


# order is important otherwise from_output_channels will yield bad results
OUTPUT_FORMATS: Dict[str, OutputFormat] = {
    'MONO': OutputFormat('Mono', 1, 1, 0, (1,), 16),
    'STEREO': OutputFormat('Stereo', 2, 2, 0, (2,), 16),
    'FOUR': OutputFormat('4 channel', 4, 4, 0, (2, 2), 16),
    'THREE_ONE': OutputFormat('3.1', 4, 4, 1, (4, 0, 15), 16),
    'FIVE_ONE': OutputFormat('5.1', 6, 6, 1, (6,), 16),
    'SEVEN_ONE': OutputFormat('7.1', 8, 8, 1, (8,), 16),
    'TWO_ONE': OutputFormat('2.1', 3, 6, 1, (3,), 16),
    'TEN': OutputFormat('10 channels', 8, 10, 1, (10,), 0),
    'TWELVE': OutputFormat('12 channels', 8, 12, 1, (12,), 0),
    'FOURTEEN': OutputFormat('14 channels', 8, 14, 1, (14,), 0),
    'SIXTEEN': OutputFormat('16 channels', 8, 16, 1, (16,), 0),
    'EIGHTEEN': OutputFormat('18 channels', 8, 18, 1, (18,), 0),
    'TWENTY': OutputFormat('20 channels', 8, 20, 1, (20,), 0),
    'TWENTY_TWO': OutputFormat('22 channels', 8, 22, 1, (22,), 0),
    'TWENTY_FOUR': OutputFormat('24 channels', 8, 24, 1, (24,), 0),
    'THIRTY_TWO': OutputFormat('32 channels', 8, 32, 1, (32,), 0),
    'SOURCE': OutputFormat('Source', 8, 8, 1, (0,), 16),
    'STEREO_IN_FOUR': OutputFormat('Stereo in a 4 channel container', 2, 4, 0, (2, 2), 0),
    'STEREO_IN_FIVE': OutputFormat('Stereo in a 5.1 channel container', 2, 6, 0, (2, 4), 0),
    'STEREO_IN_SEVEN': OutputFormat('Stereo in a 7.1 channel container', 2, 8, 0, (2, 6), 0),
    'FIVE_ONE_IN_SEVEN': OutputFormat('5.1 in a 7.1 container', 6, 8, 1, (6, 2), 0),
}

