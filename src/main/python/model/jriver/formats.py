import functools
from typing import List, Dict, Tuple

# 2-9
SURROUND_CHANNELS = ['Left', 'Right', 'Centre', 'Subwoofer', 'Surround Left', 'Surround Right', 'Rear Left',
                     'Rear Right']
SURROUND_SHORT_CHANNELS = ['L', 'R', 'C', 'SW', 'SL', 'SR', 'RL', 'RR']
SURROUND_CHANNEL_INDEXES = list(range(2, 10))
# 11-12
USER_CHANNELS = ['User 1', 'User 2']
SHORT_USER_CHANNELS = ['U1', 'U2']
USER_CHANNEL_INDEXES = list(range(11, 13))
# 13-36
NUMBER_CHANNELS = [f"Channel {i + 9}" for i in range(24)]
SHORT_NUMBER_CHANNELS = [f"C{i + 9}" for i in range(24)]
NUMBER_CHANNEL_INDEXES = list(range(13, 37))
# 37-52
EXTRA_CHANNELS = [f'Extra {i}' for i in range(1, 17)]
EXTRA_SHORT_CHANNELS = [f'X{i}' for i in range(1, 17)]
EXTRA_CHANNEL_INDEXES = list(range(37, 53))
# 54-61
ATMOS_CHANNELS = ['Left Height Front', 'Right Height Front', 'Left Height Rear', 'Right Height Rear', 'Left Top Middle',
                  'Right Top Middle', 'Left Width', 'Right Width']
# NB: JRiver's own short codes don't match the long names above (its own inconsistency) - mirror them
# verbatim rather than deriving "logically correct" ones from the long name.
SHORT_ATMOS_CHANNELS = ['LTF', 'RTF', 'LTR', 'RTR', 'LTM', 'RTM', 'LW', 'RW']
ATMOS_CHANNEL_INDEXES = list(range(54, 62))

# call MCWS/v1/Library/Get?Settings=1
# get zip file
# extract User Settings.ini
# look for
# New Extra Channels System 2=i:"0" (extra channels is disabled)
# MC 35.0.39 completes the Atmos channels (54-61) and exposes the Extra channels (37-52) - see
# use_atmos_channels in JRiverDSP/OutputFormat. This is a point release within the 35.x line, not
# major version 36. Prior to 35.0.39, only 54-57 worked; MC28-33 had neither Atmos nor Extra
# channels at all (see AGENTS.md for the full pre-/post-35.0.39 comparison).

JRIVER_NAMED_CHANNELS = [None, None] + SURROUND_CHANNELS + [None] + USER_CHANNELS
JRIVER_SHORT_NAMED_CHANNELS = [None, None] + SURROUND_SHORT_CHANNELS + [None] + SHORT_USER_CHANNELS

JRIVER_HIGHER_CHANNELS = NUMBER_CHANNELS
JRIVER_SHORT_HIGHER_CHANNELS = SHORT_NUMBER_CHANNELS

# index 53 is an unused gap between the Extra channels (37-52) and the Atmos channels (54-61)
JRIVER_CHANNELS = JRIVER_NAMED_CHANNELS + JRIVER_HIGHER_CHANNELS + EXTRA_CHANNELS + [None] + ATMOS_CHANNELS
JRIVER_SHORT_CHANNELS = (JRIVER_SHORT_NAMED_CHANNELS + JRIVER_SHORT_HIGHER_CHANNELS + EXTRA_SHORT_CHANNELS + [None] +
                         SHORT_ATMOS_CHANNELS)

JRIVER_REAL_NAMED_CHANNELS = JRIVER_NAMED_CHANNELS[2:-3] + JRIVER_HIGHER_CHANNELS
JRIVER_SHORT_REAL_NAMED_CHANNEL = JRIVER_SHORT_NAMED_CHANNELS[2:-3] + JRIVER_SHORT_HIGHER_CHANNELS


def get_all_channel_names(short: bool = True, use_atmos_channels: bool = False,
                          padding_only: bool = False, base_channel_count: int = 8) -> List[str]:
    '''
    :param short: get short names if true.
    :param use_atmos_channels: if true, order the channels beyond the base as the full 9.1.6 Atmos
    layout (MC 35.0.39+) followed by the Extra channels, instead of the legacy generically numbered
    channels (< 35.0.39). See AGENTS.md for why this can't just always be true.
    :param padding_only: if true (and use_atmos_channels is true), use the Extra channels alone rather
    than Atmos+Extra combined - this is the 35.0.39+ pool a dynamically padded "+N padding channels"
    instance draws from, as opposed to a deliberately-chosen static immersive layout (5.1.2/7.1.4/
    9.1.6/32ch) which draws Atmos channels first. Confirmed empirically against a real MC36 7.1+2
    capture: the 2 padding channels resolved to X1/X2 (Extra), not LTF/RTF (Atmos) or Channel 9/10
    (legacy numbered) - see AGENTS.md and mc36_seven_one_plus_two_padding.dsp.
    :param base_channel_count: how many of the 8 surround channels (L,R,C,SW,SL,SR,RL,RR, in that
    order) this format's bed actually uses - 8 for a 7.1-based bed, 6 for a 5.1-based bed (no rear
    surrounds, freeing 2 slots for e.g. 5.1.2's height channels). Confirmed empirically: a real
    5.1.2 capture uses L/R/C/SW/SL/SR/LTF/RTF (8 total), not the 7.1 bed plus 2 extra - see
    AGENTS.md and OUTPUT_FORMATS['FIVE_ONE_TWO'].
    :return: the channel names, in output-format assignment order.
    '''
    base = (SURROUND_SHORT_CHANNELS if short else SURROUND_CHANNELS)[:base_channel_count]
    if use_atmos_channels:
        if padding_only:
            extra = EXTRA_SHORT_CHANNELS if short else EXTRA_CHANNELS
        else:
            extra = (SHORT_ATMOS_CHANNELS + EXTRA_SHORT_CHANNELS) if short else (ATMOS_CHANNELS + EXTRA_CHANNELS)
    else:
        extra = SHORT_NUMBER_CHANNELS if short else NUMBER_CHANNELS
    return base + extra


def get_channel_indexes(names: List[str], short: bool = True) -> List[int]:
    contents = JRIVER_SHORT_CHANNELS if short else JRIVER_CHANNELS
    return [contents.index(n) for n in names]


def get_real_channel_name(idx: int, short: bool = True) -> str:
    contents = JRIVER_SHORT_REAL_NAMED_CHANNEL if short else JRIVER_REAL_NAMED_CHANNELS
    return contents[idx]


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


@functools.total_ordering
class OutputFormat:

    def __init__(self, display_name: str, input_channels: int, output_channels: int, lfe_channels: int,
                 xml_vals: Tuple[int, ...], max_padding: int, template: bool = True, base_channels: int = 8):
        self.__lfe_channels = lfe_channels
        self.__output_channels = output_channels
        self.__display_name = display_name
        self.__input_channels = input_channels
        self.__xml_vals = xml_vals
        # how many of the 8 surround channels this format's bed actually uses - 8 for a 7.1-based
        # bed, 6 for a 5.1-based bed (no rear surrounds). Only matters once channels beyond the base
        # are in play (use_atmos_channels/legacy numbered) - see get_all_channel_names's
        # base_channel_count. Confirmed empirically for FIVE_ONE_TWO: real 5.1.2 capture uses
        # L/R/C/SW/SL/SR/LTF/RTF (8 total, RL/RR dropped entirely), not the full 7.1 bed plus 2 more.
        self.__base_channels = base_channels
        # a template=False instance is a one-off "+N padding channels" bolt-on dynamically constructed
        # by codec.get_output_format's padded branch, as opposed to one of the static OUTPUT_FORMATS
        # entries. Padding has always meant a generic scratch/spare channel pool (that's what BEQD's
        # own crossover engine has always used it for), but which pool changed at the 35.0.39 point
        # release (not major version 36 - see mcws.ATMOS_CHANNELS_MIN_VERSION): legacy numbered
        # (< 35.0.39, confirmed via the real mc35_mixed_channel_eras.dsp capture) vs Extra channels
        # alone, never Atmos (35.0.39+, confirmed via the real mc36_seven_one_plus_two_padding.dsp
        # capture) - unlike a deliberately-chosen fully immersive static format (5.1.2/7.1.4/9.1.6/32
        # channels), which draws Atmos channels first regardless of version - see
        # get_output_channel_indexes/get_all_channel_names's padding_only.
        self.__is_padded_instance = not template
        if template:
            self.__paddings: List[int] = list(range(2, max_padding + 1, 2)) if max_padding > 0 else []
        else:
            self.__paddings: List[int] = [max_padding] if max_padding else []

    def get_output_channel_indexes(self, use_atmos_channels: bool = False) -> List[int]:
        '''
        :param use_atmos_channels: if true, assign channels beyond the base 8 using the 35.0.39+ scheme
        rather than the legacy generically numbered channels - Atmos+Extra ordering for a static,
        deliberately-immersive format, or Extra alone for a dynamically padded instance (see
        __init__ and get_all_channel_names's padding_only) - confirmed empirically against a real
        MC36 7.1+2 capture (mc36_seven_one_plus_two_padding.dsp), whose 2 padding channels resolved
        to X1/X2, not Channel 9/10.
        :return: the output channel indexes for this format.
        '''
        if self.__lfe_channels > 0 and self.__output_channels < 4:
            return [2, 3, 5] + user_channel_indexes()
        all_names = get_all_channel_names(use_atmos_channels=use_atmos_channels,
                                          padding_only=self.__is_padded_instance,
                                          base_channel_count=self.__base_channels)
        return sorted(get_channel_indexes(all_names[:self.__output_channels]) + user_channel_indexes())

    def get_input_channel_indexes(self, use_atmos_channels: bool = False) -> List[int]:
        '''
        :param use_atmos_channels: if true, assign channels beyond the base 8 using the 35.0.39+ scheme
        rather than the legacy generically numbered channels - see get_output_channel_indexes.
        :return: the input channel indexes for this format.
        '''
        all_names = get_all_channel_names(use_atmos_channels=use_atmos_channels,
                                          padding_only=self.__is_padded_instance,
                                          base_channel_count=self.__base_channels)
        # swap SL/SR and RL/RR for the 8 or more channel case - only applies when the bed actually
        # has RL/RR (a 7.1-based bed); a 5.1-based bed (base_channels=6, e.g. FIVE_ONE_TWO) has
        # nothing at those positions to swap with, so leave it untouched.
        if self.__base_channels == 8:
            all_names_8 = all_names[0:4] + all_names[6:8] + all_names[4:6] + all_names[8:]
        else:
            all_names_8 = all_names
        # special case for 2.1
        if self.__lfe_channels > 0 and self.__input_channels < 4:
            return [2, 3, 5]
        elif self.__lfe_channels > 0 and self.__input_channels == 6:
            if self.__output_channels == 6:
                return get_channel_indexes(all_names[:self.__output_channels])
            else:
                return get_channel_indexes(all_names_8[0:6])
        else:
            if self.__output_channels < 7:
                return get_channel_indexes(all_names[:self.__input_channels])
            else:
                return get_channel_indexes(all_names_8[:self.__input_channels])

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
        return self.get_input_channel_indexes()

    @property
    def output_channel_indexes(self) -> List[int]:
        return self.get_output_channel_indexes()

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

    def migrate_channel_index(self, idx: int, use_atmos_channels: bool) -> int:
        '''
        A filter's raw channel index is baked into the config XML - but if it was recorded against
        the legacy generically-numbered pool (13-36, pre-35.0.39) and this format is now being
        interpreted with use_atmos_channels=True, the "same" scratch/extra channel is really addressed
        by a different raw index under the 35.0.39+ scheme (Atmos+Extra for a static immersive format,
        Extra alone for a padded instance - see get_all_channel_names's padding_only). Without this, a
        filter parsed from an older capture (or a live pre-35.0.39-authored config now loaded from a
        35.0.39+ server) silently falls outside the newly-declared channel set and becomes an orphaned
        scratch filter instead of continuing to control the channel it actually applies to.
        :param idx: the raw channel index as parsed from the config.
        :param use_atmos_channels: whether this format is being interpreted under the 35.0.39+ scheme.
        :return: the index to actually use - migrated if idx is a legacy-pool index and
        use_atmos_channels is true, otherwise idx unchanged.
        '''
        if use_atmos_channels and NUMBER_CHANNEL_INDEXES[0] <= idx <= NUMBER_CHANNEL_INDEXES[-1]:
            position = idx - NUMBER_CHANNEL_INDEXES[0]
            new_pool = EXTRA_SHORT_CHANNELS if self.__is_padded_instance \
                else (SHORT_ATMOS_CHANNELS + EXTRA_SHORT_CHANNELS)
            if position < len(new_pool):
                return get_channel_idx(new_pool[position])
        return idx

    def __lt__(self, other):
        if isinstance(other, OutputFormat):
            delta = self.output_channels - other.output_channels
            if delta == 0:
                return self.input_channels - other.input_channels
            return delta < 0
        return -1


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
    # MC35
    'FIVE_ONE_TWO': OutputFormat('5.1.2', 8, 8, 1, (8,), 16, base_channels=6),
    'SEVEN_ONE_FOUR': OutputFormat('7.1.4 (12 channels)', 12, 12, 1, (12,), 16),
    'NINE_ONE_SIX': OutputFormat('9.1.6 (Dolby Atmos 16 channels)', 16, 16, 1, (16,), 16),
}
