from __future__ import annotations

import abc
import json
import logging
import math
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Optional, Callable, Union, Type, Sequence, overload, Tuple, Iterable

import numpy as np

from model import iir
from model.iir import SOS, s_to_q, q_to_s, FirstOrder_LowPass, FirstOrder_HighPass, PassFilter, CompoundPassFilter, \
    FilterType, SecondOrder_LowPass, ComplexLowPass, SecondOrder_HighPass, ComplexHighPass, CompleteFilter, \
    BiquadWithQGain, PeakingEQ, LowShelf as LS, Gain as G, LinkwitzTransform as LT, AllPass as AP, MDS_FREQ_DIVISOR
from model.jriver import JRIVER_FS, flatten
from model.jriver.codec import filts_to_xml
from model.jriver.common import get_channel_name, pop_channels, get_channel_idx, JRIVER_SHORT_CHANNELS, \
    make_dirac_pulse, make_silence, SHORT_USER_CHANNELS
from model.log import to_millis
from model.signal import Signal

logger = logging.getLogger('jriver.filter')


class Filter(ABC):

    def __init__(self, short_name: str):
        self.__short_name = short_name
        self.__f_id = -1
        self.__nodes: List[str] = []

    def reset(self) -> None:
        self.__nodes = []

    @property
    def id(self) -> int:
        return self.__f_id

    @id.setter
    def id(self, f_id: int):
        self.__f_id = f_id

    @property
    def nodes(self) -> List[str]:
        return self.__nodes

    @property
    def short_name(self) -> str:
        return self.__short_name

    def short_desc(self) -> str:
        return self.short_name

    @property
    def enabled(self) -> bool:
        return True

    def encode(self):
        return filts_to_xml(self.get_all_vals())

    @abstractmethod
    def get_all_vals(self, convert_q: bool = False) -> List[Dict[str, str]]:
        pass

    def get_editable_filter(self) -> Optional[SOS]:
        return None

    def get_filter_op(self) -> FilterOp:
        return NopFilterOp()

    def is_mine(self, idx: int) -> bool:
        return True

    def can_merge(self, o: Filter) -> bool:
        return False

    def can_split(self) -> bool:
        return False

    def split(self) -> List[Filter]:
        return [self]

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Filter):
            return self.get_all_vals() == o.get_all_vals()
        return False


class SingleFilter(Filter, ABC):

    def __init__(self, vals, short_name):
        super().__init__(short_name)
        self.__vals = vals
        self.__enabled = vals['Enabled'] == '1'
        self.__type_code = vals['Type']

    @property
    def enabled(self):
        return self.__enabled

    def get_all_vals(self, convert_q: bool = False) -> List[Dict[str, str]]:
        vals = {
            'Enabled': '1' if self.__enabled else '0',
            'Type': self.__type_code,
            **self.get_vals(convert_q=convert_q)
        }
        if self.key_order:
            return [{k: vals[k] for k in self.key_order}]
        else:
            return [vals]

    @property
    def key_order(self) -> List[str]:
        return []

    def get_vals(self, convert_q: bool = False) -> Dict[str, str]:
        return {}

    def print_disabled(self):
        return '' if self.enabled else f" *** DISABLED ***"

    @classmethod
    def default_values(cls) -> Dict[str, str]:
        return {
            'Enabled': '1',
            'Type': cls.TYPE
        }


class ChannelFilter(SingleFilter, ABC):

    def __init__(self, vals, short_name):
        super().__init__(vals, short_name)
        ch = vals.get('Channels', None)
        if ch:
            self.__channels = [int(c) for c in ch.split(';')]
        else:
            raise ValueError(f"No channels specified in filter {short_name} {vals}")
        self.__channel_names = [get_channel_name(i) for i in self.__channels]

    @property
    def channels(self) -> List[int]:
        return self.__channels

    @property
    def channel_names(self) -> List[str]:
        return self.__channel_names

    def get_vals(self, convert_q: bool = False) -> Dict[str, str]:
        return {
            'Channels': ';'.join([str(c) for c in self.channels]),
            **super().get_vals(convert_q=convert_q)
        }

    def __repr__(self):
        return f"{self.__class__.__name__} {self.print_channel_names()}{self.print_disabled()}"

    def print_channel_names(self):
        return f"[{', '.join(self.channel_names)}]"

    def can_merge(self, o: Filter):
        if isinstance(o, ChannelFilter):
            if self.__class__ == o.__class__:
                if pop_channels(self.get_all_vals()) == pop_channels(o.get_all_vals()):
                    repeating_channels = set(self.channels).intersection(set(o.channels))
                    return len(repeating_channels) == 0
        return False

    def can_split(self) -> bool:
        return len(self.channels) > 1

    def split(self) -> List[Filter]:
        return [create_single_filter({**self.get_all_vals()[0], 'Channels': str(c)}) for c in self.channels]

    def is_mine(self, idx):
        return idx in self.channels

    def pop_channel(self, channel_name: str) -> Optional[ChannelFilter]:
        '''
        Removes a channel from the filter.
        :param channel_name: the (short) channel name to remove.
        :return the filter without the channel or None if it was the last channel.
        '''
        if channel_name in self.__channel_names:
            if len(self.__channel_names) == 1:
                return None
            else:
                vals = self.get_all_vals()
                # guaranteed to be a single item in this list if we're in this class
                c = self.__class__(vals[0])
                c.__channel_names.remove(channel_name)
                c.__channels.remove(get_channel_idx(channel_name))
                return c
        else:
            raise ValueError(f"{channel_name} not found in {self}")

    @classmethod
    def default_values(cls) -> Dict[str, str]:
        return {
            **super(ChannelFilter, cls).default_values(),
            'Channels': ''
        }


class GainQFilter(ChannelFilter, ABC):

    def __init__(self, vals, create_iir, short_name, convert_q: bool = False):
        super().__init__(vals, short_name)
        self.__create_iir = create_iir
        self.__gain = float(vals['Gain'])
        self.__frequency = float(vals['Frequency'])
        self.__q = self.from_jriver_q(float(vals['Q']), self.__gain, convert_q)

    @property
    def key_order(self) -> List[str]:
        return ['Enabled', 'Slope', 'Q', 'Type', 'Gain', 'Frequency', 'Channels']

    @property
    def freq(self) -> float:
        return self.__frequency

    @property
    def q(self) -> float:
        return self.__q

    @property
    def gain(self) -> float:
        return self.__gain

    def get_vals(self, convert_q: bool = False) -> Dict[str, str]:
        return {
            'Slope': '12',
            'Q': f"{self.to_jriver_q(self.__q, self.__gain, convert_q=convert_q):.12g}",
            'Gain': f"{self.__gain:.7g}",
            'Frequency': f"{self.__frequency:.7g}",
            **super().get_vals(convert_q=convert_q)
        }

    @classmethod
    def from_jriver_q(cls, q: float, gain: float, convert_q: bool = False) -> float:
        return q

    @classmethod
    def to_jriver_q(cls, q: float, gain: float, convert_q: bool = False) -> float:
        return q

    def get_filter_op(self) -> FilterOp:
        if self.enabled:
            f = self.get_editable_filter()
            if f:
                sos = self.get_editable_filter().get_sos()
                if sos:
                    return SosFilterOp(sos)
        return NopFilterOp()

    def get_editable_filter(self) -> Optional[SOS]:
        return self.__create_iir(JRIVER_FS, self.__frequency, self.__q, self.__gain, f_id=self.id)

    @classmethod
    def default_values(cls) -> Dict[str, str]:
        return {
            **super(GainQFilter, cls).default_values(),
            'Slope': '12'
        }

    def __repr__(self):
        return f"{self.__class__.__name__} {self.__gain:+.7g} dB Q={self.__q:.4g} at {self.__frequency:.7g} Hz {self.print_channel_names()}{self.print_disabled()}"


class Peak(GainQFilter):
    TYPE = '3'

    def __init__(self, vals):
        super().__init__(vals, iir.PeakingEQ, 'Peak')


class LowShelf(GainQFilter):
    TYPE = '10'

    def __init__(self, vals, convert_q: bool = False):
        super().__init__(vals, iir.LowShelf, 'LS', convert_q=convert_q)

    @classmethod
    def from_jriver_q(cls, q: float, gain: float, convert_q: bool = False):
        return s_to_q(q, gain) if convert_q else q

    @classmethod
    def to_jriver_q(cls, q: float, gain: float, convert_q: bool = False):
        return q_to_s(q, gain) if convert_q else q


class HighShelf(GainQFilter):
    TYPE = '11'

    def __init__(self, vals, convert_q: bool = False):
        super().__init__(vals, iir.HighShelf, 'HS', convert_q=convert_q)

    @classmethod
    def from_jriver_q(cls, q: float, gain: float, convert_q: bool = False) -> float:
        return s_to_q(q, gain) if convert_q else q

    @classmethod
    def to_jriver_q(cls, q: float, gain: float, convert_q: bool = False) -> float:
        return q_to_s(q, gain) if convert_q else q


class AllPass(ChannelFilter):
    TYPE = '17'

    def __init__(self, vals):
        super().__init__(vals, 'APF')
        self.__frequency = float(vals['Frequency'])
        self.__q = float(vals['Q'])

    @property
    def key_order(self) -> List[str]:
        return ['Enabled', 'Slope', 'Q', 'Type', 'Version', 'Gain', 'Frequency', 'Channels']

    @property
    def freq(self) -> float:
        return self.__frequency

    @property
    def q(self) -> float:
        return self.__q

    def get_vals(self, convert_q: bool = False) -> Dict[str, str]:
        return {
            'Slope': '12',
            'Version': '1',
            'Gain': '0',
            'Q': f"{self.__q:.12g}",
            'Frequency': f"{self.__frequency:.7g}",
            **super().get_vals(convert_q=convert_q)
        }

    def get_filter_op(self) -> FilterOp:
        if self.enabled:
            f = self.get_editable_filter()
            if f:
                sos = self.get_editable_filter().get_sos()
                if sos:
                    return SosFilterOp(sos)
        return NopFilterOp()

    def get_editable_filter(self) -> Optional[SOS]:
        return iir.AllPass(JRIVER_FS, self.__frequency, self.__q, f_id=self.id)

    @classmethod
    def default_values(cls) -> Dict[str, str]:
        return {
            **super(AllPass, cls).default_values(),
            'Slope': '12',
            'Gain': '0',
            'Version': '1'
        }

    def __repr__(self):
        return f"{self.__class__.__name__} Q={self.__q:.4g} at {self.__frequency:.7g} Hz {self.print_channel_names()}{self.print_disabled()}"


class Pass(ChannelFilter, ABC):

    def __init__(self, vals: dict, short_name: str,
                 one_pole_ctor: Callable[..., Union[FirstOrder_LowPass, FirstOrder_HighPass]],
                 two_pole_ctor: Callable[..., PassFilter],
                 many_pole_ctor: Callable[..., CompoundPassFilter],
                 convert_q: bool = False):
        super().__init__(vals, short_name)
        self.__order = int(int(vals['Slope']) / 6)
        self.__frequency = float(vals['Frequency'])
        self.__jriver_q = float(vals['Q'])
        self.__q = self.from_jriver_q(self.jriver_q) if self.order == 2 and convert_q else self.jriver_q
        self.__ctors = (one_pole_ctor, two_pole_ctor, many_pole_ctor)

    @classmethod
    def from_jriver_q(cls, q: float) -> float:
        return q / 2 ** 0.5

    @classmethod
    def to_jriver_q(cls, q: float) -> float:
        return q * 2 ** 0.5

    @property
    def freq(self) -> float:
        return self.__frequency

    @property
    def jriver_q(self) -> float:
        return self.__jriver_q

    @property
    def q(self):
        return self.__q

    @property
    def order(self) -> int:
        return self.__order

    def get_vals(self, convert_q: bool = False) -> Dict[str, str]:
        q = self.to_jriver_q(self.jriver_q) if self.order == 2 and convert_q else self.jriver_q
        return {
            'Gain': '0',
            'Slope': f"{self.order * 6}",
            'Q': f"{q:.12g}",
            'Frequency': f"{self.__frequency:.7g}",
            **super().get_vals(convert_q=convert_q)
        }

    @property
    def key_order(self) -> List[str]:
        return ['Enabled', 'Slope', 'Q', 'Type', 'Gain', 'Frequency', 'Channels']

    def get_filter_op(self) -> FilterOp:
        if self.enabled:
            sos = self.get_editable_filter().get_sos()
            if sos:
                return SosFilterOp(sos)
        return NopFilterOp()

    def short_desc(self):
        q_suffix = ''
        if not math.isclose(self.jriver_q, 1.0):
            q_suffix = f" VarQ"
        if self.freq >= 1000:
            f = f"{self.freq / 1000.0:.3g}k"
        else:
            f = f"{self.freq:g}"
        return f"{self.short_name}{self.order} BW {f}{q_suffix}"

    def get_editable_filter(self) -> Optional[SOS]:
        '''
        Creates a set of biquads which translate the non standard jriver Q into a real Q value.
        :return: the filters.
        '''
        if self.order == 1:
            return self.__ctors[0](JRIVER_FS, self.freq, f_id=self.id)
        elif self.order == 2:
            return self.__ctors[1](JRIVER_FS, self.freq, q=self.q, f_id=self.id)
        else:
            return self.__ctors[2](FilterType.BUTTERWORTH, self.order, JRIVER_FS, self.freq, q=self.q, f_id=self.id)

    def __repr__(self):
        return f"{self.__class__.__name__} Order={self.order} Q={self.q:.4g} at {self.freq:.7g} Hz {self.print_channel_names()}{self.print_disabled()}"


class LowPass(Pass):
    TYPE = '1'

    def __init__(self, vals, convert_q: bool = False):
        super().__init__(vals, 'LP', FirstOrder_LowPass, SecondOrder_LowPass, ComplexLowPass, convert_q=convert_q)


class HighPass(Pass):
    TYPE = '2'

    def __init__(self, vals, convert_q: bool = False):
        super().__init__(vals, 'HP', FirstOrder_HighPass, SecondOrder_HighPass, ComplexHighPass, convert_q=convert_q)


class Gain(ChannelFilter):
    TYPE = '4'

    def __init__(self, vals):
        super().__init__(vals, 'GAIN')
        self.__gain = float(vals['Gain'])

    @property
    def gain(self):
        return self.__gain

    def get_vals(self, convert_q: bool = False) -> Dict[str, str]:
        return {
            'Gain': f"{self.__gain:.7g}",
            **super().get_vals()
        }

    @property
    def key_order(self) -> List[str]:
        return ['Enabled', 'Type', 'Gain', 'Channels']

    def __repr__(self):
        return f"Gain {self.__gain:+.7g} dB {self.print_channel_names()}{self.print_disabled()}"

    def get_filter_op(self) -> FilterOp:
        return GainFilterOp(self.__gain) if self.enabled else NopFilterOp()

    def short_desc(self):
        return f"{self.__gain:+.7g} dB"

    def get_editable_filter(self) -> Optional[SOS]:
        return iir.Gain(JRIVER_FS, self.gain, f_id=self.id)


class BitdepthSimulator(SingleFilter):
    TYPE = '13'

    def __init__(self, vals):
        super().__init__(vals, 'BITDEPTH')
        self.__bits = int(vals['Bits'])
        self.__dither = vals['Dither']

    def get_vals(self, convert_q: bool = False) -> Dict[str, str]:
        return {
            'Bits': str(self.__bits),
            'Dither': self.__dither
        }

    @property
    def key_order(self) -> List[str]:
        return ['Enabled', 'Bits', 'Type', 'Dither']

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.__bits} bits){self.print_disabled()}"


class Delay(ChannelFilter):
    TYPE = '7'

    def __init__(self, vals):
        super().__init__(vals, 'DELAY')
        self.__delay = float(vals['Delay'].replace(",", "."))

    @property
    def delay(self) -> float:
        return self.__delay

    def get_vals(self, convert_q: bool = False) -> Dict[str, str]:
        return {
            'Delay': f"{self.__delay:.7g}",
            **super().get_vals()
        }

    @property
    def key_order(self) -> List[str]:
        return ['Enabled', 'Delay', 'Type', 'Channels']

    def __repr__(self):
        return f"{self.__class__.__name__} {self.__delay:+.7g} ms {self.print_channel_names()}{self.print_disabled()}"

    def short_desc(self):
        return f"{self.__delay:+.7g}ms"

    def get_filter_op(self) -> FilterOp:
        return DelayFilterOp(self.__delay) if self.enabled else NopFilterOp()


class Divider(SingleFilter):
    TYPE = '20'

    def __init__(self, vals):
        super().__init__(vals, '---')
        self.__text = vals['Text'] if 'Text' in vals else ''

    @property
    def text(self):
        return self.__text

    def get_vals(self, convert_q: bool = False) -> Dict[str, str]:
        return {
            'Text': self.__text
        }

    def __repr__(self):
        return self.__text


class LimiterMode(Enum):
    BRICKWALL = 0
    ADAPTIVE = 1


class Limiter(ChannelFilter):
    TYPE = '9'

    def __init__(self, vals):
        super().__init__(vals, 'LIMITER')
        self.__hold = vals['Hold']
        self.__mode = vals['Mode']
        self.__level = vals['Level']
        self.__release = vals['Release']
        self.__attack = vals['Attack']

    def get_vals(self, convert_q: bool = False) -> Dict[str, str]:
        return {
            'Hold': self.__hold,
            'Mode': self.__mode,
            'Level': self.__level,
            'Release': self.__release,
            'Attack': self.__attack,
            **super().get_vals()
        }

    @property
    def key_order(self) -> List[str]:
        return ['Enabled', 'Hold', 'Type', 'Mode', 'Channels', 'Level', 'Release', 'Attack']

    def __repr__(self):
        return f"{LimiterMode(int(self.__mode)).name.capitalize()} {self.__class__.__name__} at {self.__level} dB {self.print_channel_names()}{self.print_disabled()}"


class LinkwitzTransform(ChannelFilter):
    TYPE = '8'

    def __init__(self, vals):
        super().__init__(vals, 'LT')
        self.__fp = float(vals['Fp'])
        self.__qp = float(vals['Qp'])
        self.__fz = float(vals['Fz'])
        self.__qz = float(vals['Qz'])
        self.__prevent_clipping = vals['PreventClipping']

    def get_vals(self, convert_q: bool = False) -> Dict[str, str]:
        return {
            'Fp': f"{self.__fp:.7g}",
            'Qp': f"{self.__qp:.4g}",
            'Fz': f"{self.__fz:.7g}",
            'Qz': f"{self.__qz:.4g}",
            'PreventClipping': self.__prevent_clipping,
            **super().get_vals()
        }

    @property
    def key_order(self) -> List[str]:
        return ['Enabled', 'Fp', 'Qp', 'Type', 'Fz', 'Channels', 'Qz', 'PreventClipping']

    def __repr__(self):
        return f"{self.__class__.__name__} {self.__fz:.7g} Hz / {self.__qz:.4g} -> {self.__fp:.7g} Hz / {self.__qp:.4g} {self.print_channel_names()}{self.print_disabled()}"

    def get_filter_op(self) -> FilterOp:
        if self.enabled:
            sos = self.get_editable_filter().get_sos()
            if sos:
                return SosFilterOp(sos)
        return NopFilterOp()

    def get_editable_filter(self) -> Optional[SOS]:
        return iir.LinkwitzTransform(JRIVER_FS, self.__fz, self.__qz, self.__fp, self.__qp)


class LinkwitzRiley(SingleFilter):
    TYPE = '16'

    def __init__(self, vals):
        super().__init__(vals, 'LR')
        self.__freq = float(vals['Frequency'])

    def get_vals(self, convert_q: bool = False) -> Dict[str, str]:
        return {
            'Frequency': f"{self.__freq:.7g}"
        }

    @property
    def key_order(self) -> List[str]:
        return ['Enabled', 'Type', 'Frequency']

    def __repr__(self):
        return f"{self.__class__.__name__} {self.__freq:.7g} Hz{self.print_disabled()}"


class MidSideDecoding(SingleFilter):
    TYPE = '19'

    def __init__(self, vals):
        super().__init__(vals, 'MS Decode')

    def get_vals(self, convert_q: bool = False) -> Dict[str, str]:
        return {}

    def __repr__(self):
        return f"{self.__class__.__name__}{self.print_disabled()}"


class MidSideEncoding(SingleFilter):
    TYPE = '18'

    def __init__(self, vals):
        super().__init__(vals, 'MS Encode')

    def get_vals(self, convert_q: bool = False) -> Dict[str, str]:
        return {}

    def __repr__(self):
        return f"{self.__class__.__name__}{self.print_disabled()}"


class MixType(Enum):
    ADD = 0
    COPY = 1
    MOVE = 2
    SWAP = 3, 'and'
    SUBTRACT = 4, 'from'

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        obj._value_ = args[0]
        return obj

    def __init__(self, _: str, modifier: str = 'to'):
        self.___modifier = modifier

    def __str__(self):
        return self.value

    @property
    def modifier(self):
        return self.___modifier


class Mix(SingleFilter):
    TYPE = '6'

    def __init__(self, vals):
        super().__init__(vals, f"{MixType(int(vals['Mode'])).name.capitalize()}")
        self.__source = vals['Source']
        self.__destination = vals['Destination']
        self.__gain = float(vals['Gain'])
        # mode: 3 = swap, 1 = copy, 2 = move, 0 = add, 4 = subtract
        self.__mode = int(vals['Mode'])

    @property
    def src_idx(self):
        return int(self.__source)

    @property
    def dst_idx(self):
        return int(self.__destination)

    @property
    def mix_type(self) -> MixType:
        return MixType(int(self.__mode))

    @property
    def gain(self) -> float:
        return self.__gain

    def get_vals(self, convert_q: bool = False) -> Dict[str, str]:
        return {
            'Source': self.__source,
            'Gain': f"{self.__gain:.7g}",
            'Destination': self.__destination,
            'Mode': f"{self.__mode}"
        }

    @property
    def key_order(self) -> List[str]:
        return ['Enabled', 'Type', 'Source', 'Gain', 'Destination', 'Mode']

    def __repr__(self):
        mix_type = MixType(self.__mode)
        src_name = get_channel_name(int(self.__source))
        dst_name = get_channel_name(int(self.__destination))
        return f"{mix_type.name.capitalize()} {src_name} {mix_type.modifier} {dst_name} {self.__gain:+.7g} dB{self.print_disabled()}"

    def is_mine(self, idx):
        return self.src_idx == idx

    def short_desc(self):
        mix_type = self.mix_type
        if mix_type == MixType.MOVE or mix_type == MixType.MOVE:
            return f"{mix_type.name}\nto {get_channel_name(int(self.__destination))}"
        elif mix_type == MixType.SWAP:
            return f"{mix_type.name}\n{get_channel_name(int(self.__source))}-{get_channel_name(int(self.__destination))}"
        else:
            return super().short_desc()

    def get_filter_op(self) -> FilterOp:
        if self.enabled:
            if self.mix_type == MixType.ADD:
                return AddFilterOp(GainFilterOp(self.__gain))
            elif self.mix_type == MixType.SUBTRACT:
                return SubtractFilterOp(GainFilterOp(self.__gain))
            elif not math.isclose(self.__gain, 0.0):
                return GainFilterOp(self.__gain)
        return NopFilterOp()

    @classmethod
    def default_values(cls) -> Dict[str, str]:
        return {
            **super(Mix, cls).default_values(),
            'Source': str(get_channel_idx('L')),
            'Gain': '0',
            'Destination': str(get_channel_idx('R'))
        }


class Order(SingleFilter):
    TYPE = '12'

    def __init__(self, vals):
        super().__init__(vals, 'ORDER')
        self.__order = vals['Order'].split(',')
        self.__named_order = [get_channel_name(int(i)) for i in self.__order]

    def get_vals(self, convert_q: bool = False) -> Dict[str, str]:
        return {
            'Order': ','.join(self.__order)
        }

    @property
    def key_order(self) -> List[str]:
        return ['Enabled', 'Order', 'Type']

    def __repr__(self):
        return f"{self.__class__.__name__} {self.__named_order}{self.print_disabled()}"


class Mute(ChannelFilter):
    TYPE = '5'

    def __init__(self, vals):
        super().__init__(vals, 'MUTE')
        self.__gain = float(vals['Gain'])

    def get_vals(self, convert_q: bool = False) -> Dict[str, str]:
        return {
            'Gain': f"{self.__gain:.7g}",
            **super().get_vals()
        }

    @property
    def key_order(self) -> List[str]:
        return ['Enabled', 'Type', 'Gain', 'Channels']


class Polarity(ChannelFilter):
    TYPE = '15'

    def __init__(self, vals):
        super().__init__(vals, 'INVERT')

    @property
    def key_order(self) -> List[str]:
        return ['Enabled', 'Type', 'Channels']

    def get_filter_op(self) -> FilterOp:
        return InvertPolarityFilterOp()


class SubwooferLimiter(ChannelFilter):
    TYPE = '14'

    def __init__(self, vals):
        super().__init__(vals, 'SW Limiter')
        self.__level = float(vals['Level'])

    def get_vals(self, convert_q: bool = False) -> Dict[str, str]:
        return {
            'Level': f"{self.__level:.7g}",
            **super().get_vals()
        }

    @property
    def key_order(self) -> List[str]:
        return ['Enabled', 'Type', 'Channels', 'Level']

    def __repr__(self):
        return f"{self.__class__.__name__} {self.__level:+.7g} dB {self.print_channel_names()}{self.print_disabled()}"


class ComplexFilter(Filter):

    def __init__(self, filters: List[Filter]):
        super().__init__(self.custom_type())
        self.__f_id = -1
        self.__prefix: Filter = self.__make_divider(True)
        self.__filters = filters
        self.__suffix: Filter = self.__make_divider(False)

    def short_desc(self):
        return f"{self.short_name}"

    @property
    def filters(self) -> List[Filter]:
        return self.__filters

    @Filter.id.getter
    def id(self) -> int:
        return self.__f_id

    @id.setter
    def id(self, f_id: int):
        self.__f_id = f_id
        for i, f in enumerate(self.filters):
            f.id = f"{f_id}_{i}"

    def get_all_vals(self, convert_q: bool = False) -> List[Dict[str, str]]:
        all_filters: List[Filter] = [self.__prefix] + self.__filters + [self.__suffix]
        return [v for f in all_filters for v in f.get_all_vals(convert_q=convert_q)]

    def get_filter_op(self) -> FilterOp:
        if self.enabled:
            fop = CompositeFilterOp()
            for f in self.filters:
                fop.append(f.get_filter_op())
            return fop
        else:
            return NopFilterOp()

    @classmethod
    def get_complex_filter_data(cls, text: str) -> Optional[Type, str]:
        '''
        :param text: the text to parse
        :return: none if this text does not identify a complex filter otherwise a tuple specifying whether the filter
        is a start or end of a filter (true if end) then the filter data.
        '''
        if text.startswith(f"***{cls.custom_type()}_START"):
            return cls, text.split('|')[1][:-3]
        elif text.startswith(f"***{cls.custom_type()}_END"):
            return cls, text.split('|')[1][:-3]
        else:
            return None

    @classmethod
    def is_end_of_complex_filter_data(cls, text: str) -> Optional[bool]:
        '''
        :param text: the text to parse
        :return: true if this denotes the closing marker for a complex filter.
        '''
        return text.startswith(f"***{cls.custom_type()}_END")

    @classmethod
    @abstractmethod
    def custom_type(cls) -> str:
        pass

    @classmethod
    @abstractmethod
    def create(cls, data: str, child_filters: List[Filter]):
        pass

    def metadata(self) -> str:
        return ''

    def __repr__(self):
        return self.short_desc()

    def __make_divider(self, start: bool):
        return Divider({
            'Enabled': '1',
            'Type': Divider.TYPE,
            'Text': f"***{self.custom_type()}_{'START' if start else 'END'}|{self.metadata()}***"
        })


class ComplexChannelFilter(ComplexFilter, ABC):

    def __init__(self, filters: List[Filter]):
        super().__init__(filters)
        first_filt = filters[0]
        if hasattr(first_filt, 'channels') and hasattr(first_filt, 'channel_names'):
            self.__channels = first_filt.channels
            self.__channel_names = first_filt.channel_names
        else:
            raise ValueError(f"Unsupported filter type {first_filt}")

    @property
    def channels(self) -> List[int]:
        return self.__channels

    @property
    def channel_names(self) -> List[str]:
        return self.__channel_names

    def __repr__(self):
        return f"{self.short_name} [{', '.join(self.channel_names)}]"

    def is_mine(self, idx):
        return self.filters[0].is_mine(idx)


class GEQFilter(ComplexChannelFilter):

    def __init__(self, filters: List[Filter]):
        super().__init__(filters)

    @classmethod
    def custom_type(cls) -> str:
        return 'GEQ'

    @classmethod
    def create(cls, data: str, child_filters: List[Filter]):
        return GEQFilter(child_filters)


class MSOFilter(ComplexFilter, Sequence[ChannelFilter]):

    def __init__(self, filters: List[ChannelFilter]):
        super().__init__(filters)
        self.__all_channel_names = [get_channel_name(i) for i in sorted(list({c for f in filters for c in f.channels}))]

    @classmethod
    def custom_type(cls) -> str:
        return 'MSO'

    @classmethod
    def create(cls, data: str, child_filters: List[Filter]):
        bad_filters = [c for c in child_filters if not isinstance(c, ChannelFilter)]
        good_filters = [c for c in child_filters if isinstance(c, ChannelFilter)]
        if bad_filters:
            raise ValueError(f"Unsupported filter types supplied {bad_filters}")
        return MSOFilter(good_filters)

    @overload
    @abstractmethod
    def __getitem__(self, i: int) -> Filter: ...

    @overload
    def __getitem__(self, s: slice) -> Sequence[Filter]: ...

    def __getitem__(self, i: int) -> Filter:
        return self.filters[i]

    def __len__(self) -> int:
        return len(self.filters)

    def __repr__(self):
        return f"{self.short_name} [{', '.join(self.__all_channel_names)}]"


class XOFilterType(Enum):
    LPF = 1
    HPF = 2
    BPF = 3
    PASS = 4


class MultiwayFilter(ComplexFilter, Sequence[Filter]):

    def __init__(self, input_channel: str, output_channels: List[str], filters: List[Filter], meta: dict):
        self.__input_channel = input_channel
        self.__output_channels = output_channels
        self.__meta = meta
        super().__init__(filters)

    def short_desc(self):
        return f"MultiwayXO {self.input_channel} {self.output_channels}"

    def metadata(self) -> str:
        return json.dumps({
            'i': self.input_channel,
            'o': self.output_channels,
            'm': self.__meta
        })

    @property
    def ui_meta(self) -> dict:
        return self.__meta

    @overload
    @abstractmethod
    def __getitem__(self, i: int) -> Filter:
        ...

    @overload
    def __getitem__(self, s: slice) -> Sequence[Filter]:
        ...

    def __getitem__(self, i: int) -> Filter:
        return self.filters[i]

    def __len__(self) -> int:
        return len(self.filters)

    @property
    def input_channel(self) -> str:
        return self.__input_channel

    @property
    def output_channels(self) -> List[str]:
        return self.__output_channels

    @property
    def channel_names(self) -> List[str]:
        return list({x for x in ([self.input_channel] + self.output_channels)})

    @classmethod
    def custom_type(cls) -> str:
        return 'Multiway'

    @classmethod
    def create(cls, data: str, child_filters: List[Filter]):
        meta = json.loads(data)
        return MultiwayFilter(meta['i'], meta['o'], child_filters, meta['m'])


class XOFilter(ComplexChannelFilter):

    def __init__(self, input_channel: str, way: int, filters: List[Filter]):
        self.__input_channel = input_channel
        self.__way = way
        self.__filter_type = self.__calculate_filter_type(filters)
        super().__init__(filters)

    def short_desc(self):
        return f"{self.__filter_type.name}"

    def metadata(self) -> str:
        return f"{self.__input_channel}/{self.__way}/{self.__filter_type.name}"

    @property
    def input_channel(self) -> str:
        return self.__input_channel

    @property
    def way(self) -> int:
        return self.__way

    @classmethod
    def custom_type(cls) -> str:
        return 'XO'

    @classmethod
    def create(cls, data: str, child_filters: List[Filter]):
        tokens = data.split('/')
        return XOFilter(tokens[0], int(tokens[1]), child_filters)

    @staticmethod
    def __calculate_filter_type(filters: List[Filter]) -> XOFilterType:
        ft: XOFilterType = XOFilterType.PASS
        for f in filters:
            if isinstance(f, LowPass):
                ft = XOFilterType.LPF if ft == XOFilterType.PASS else XOFilterType.BPF
            elif isinstance(f, HighPass):
                ft = XOFilterType.HPF if ft == XOFilterType.PASS else XOFilterType.BPF
            elif isinstance(f, CustomPassFilter):
                editable = f.get_editable_filter()
                if isinstance(editable, ComplexLowPass):
                    ft = XOFilterType.LPF if ft == XOFilterType.PASS else XOFilterType.BPF
                elif isinstance(editable, ComplexHighPass):
                    ft = XOFilterType.HPF if ft == XOFilterType.PASS else XOFilterType.BPF
        return ft


class CompoundRoutingFilter(ComplexFilter, Sequence[Filter]):

    def __init__(self, metadata: str, routing: List[Filter], xo: List[MultiwayFilter]):
        self.__metadata = metadata
        self.routing = routing if routing is not None else []
        self.xo = xo if xo is not None else []
        all_filters = self.routing + self.xo
        all_channels = set()
        for f in all_filters:
            if hasattr(f, 'channel_names'):
                for c in f.channel_names:
                    all_channels.add(c)
            elif isinstance(f, Mix):
                all_channels.add(get_channel_name(f.src_idx))
                all_channels.add(get_channel_name(f.dst_idx))
        self.__channel_names = [x for x in JRIVER_SHORT_CHANNELS if x in all_channels]
        super().__init__(all_filters)

    @overload
    @abstractmethod
    def __getitem__(self, i: int) -> Filter:
        ...

    @overload
    def __getitem__(self, s: slice) -> Sequence[Filter]:
        ...

    def __getitem__(self, i: int) -> Filter:
        return self.filters[i]

    def __len__(self) -> int:
        return len(self.filters)

    def __repr__(self):
        return f"XOBM [{', '.join(self.__channel_names)}]"

    def metadata(self) -> str:
        return self.__metadata

    @classmethod
    def custom_type(cls) -> str:
        return 'XOBM'

    @classmethod
    def create(cls, data: str, child_filters: List[Filter]):
        routing_filters = []
        xo_filters = []
        # filters arranged as routing then multiway
        for f in child_filters:
            if isinstance(f, (MultiwayFilter, XOFilter)):
                xo_filters.append(f)
            else:
                routing_filters.append(f)
        return CompoundRoutingFilter(data, routing_filters, xo_filters)


class CustomPassFilter(ComplexChannelFilter):

    def __init__(self, name, filters: List[Filter]):
        self.__name = name
        super().__init__(filters)

    def short_desc(self):
        tokens = self.__name.split('/')
        freq = float(tokens[3])
        if freq >= 1000:
            f = f"{freq / 1000.0:.3g}k"
        else:
            f = f"{freq:g}"
        return f"{tokens[0]} {tokens[1]}{tokens[2]} {f}Hz"

    def get_filter_op(self) -> FilterOp:
        if self.enabled:
            return SosFilterOp(self.get_editable_filter().get_sos())
        return NopFilterOp()

    def get_editable_filter(self) -> Optional[SOS]:
        return self.__decode_custom_filter()

    def __decode_custom_filter(self) -> SOS:
        '''
        Decodes a custom filter name into a filter.
        :param desc: the filter description.
        :return: the filter.
        '''
        tokens = self.__name.split('/')
        if len(tokens) == 5:
            ft = tokens[1]
            ft = 'BESM3' if ft == 'BESM' else ft
            f_type = FilterType(ft)
            order = int(tokens[2])
            freq = float(tokens[3])
            q = float(tokens[4])
            if tokens[0] == 'HP':
                return ComplexHighPass(f_type, order, JRIVER_FS, freq, q)
            elif tokens[0] == 'LP':
                return ComplexLowPass(f_type, order, JRIVER_FS, freq, q)
        raise ValueError(f"Unable to decode {self.__name}")

    def metadata(self) -> str:
        return self.__name

    @classmethod
    def custom_type(cls) -> str:
        return 'PASS'

    def __repr__(self):
        return f"{self.short_desc()}  [{', '.join(self.channel_names)}]"

    @classmethod
    def create(cls, data: str, child_filters: List[Filter]):
        return CustomPassFilter(data, child_filters)


def all_subclasses(cls):
    return set(cls.__subclasses__()).union([s for c in cls.__subclasses__() for s in all_subclasses(c)])


complex_filter_classes_by_type: Dict[str, Type[ComplexFilter]] = {c.custom_type(): c for c in
                                                                  all_subclasses(ComplexFilter)}
filter_classes_by_type: Dict[str, Type[Filter]] = {c.TYPE: c for c in all_subclasses(Filter) if hasattr(c, 'TYPE')}

convert_q_types = [LowShelf.TYPE, HighShelf.TYPE, LowPass.TYPE, HighPass.TYPE]


def create_single_filter(vals: Dict[str, str], convert_q: bool = False) -> Filter:
    '''
    :param vals: the vals from the encoded xml format.
    :param convert_q: whether Q should be converted from the plain value.
    :return: a filter type.
    '''
    type_: Type[Filter] = filter_classes_by_type[vals['Type']]
    # noinspection PyTypeChecker
    return type_(vals, convert_q) if vals['Type'] in convert_q_types else type_(vals)


def convert_filter_to_mc_dsp(filt: SOS, target_channels: str, convert_q: bool = False) -> Filter:
    '''
    :param filt: a filter.
    :param target_channels: the channels to output to.
    :param convert_q: true if backwards compatibility with MC28 is required.
    :return: a filter
    '''
    if isinstance(filt, BiquadWithQGain):
        if isinstance(filt, PeakingEQ):
            f_type = Peak
            q = filt.q
        else:
            q = q_to_s(filt.q, filt.gain) if convert_q else filt.q
            f_type = LowShelf if isinstance(filt, LS) else HighShelf
        return f_type({
            'Enabled': '1',
            'Slope': '12',
            'Q': f"{q:.7g}",
            'Type': f_type.TYPE,
            'Gain': f"{filt.gain:.7g}",
            'Frequency': f"{filt.freq:.7g}",
            'Channels': target_channels
        })
    elif isinstance(filt, G):
        return Gain({
            'Enabled': '1',
            'Type': Gain.TYPE,
            'Gain': f"{filt.gain:.7g}",
            'Channels': target_channels
        })
    elif isinstance(filt, LT):
        return LinkwitzTransform({
            'Enabled': '1',
            'Type': LinkwitzTransform.TYPE,
            'Fz': filt.f0,
            'Qz': filt.q0,
            'Fp': filt.fp,
            'Qp': filt.qp,
            'PreventClipping': 0,
            'Channels': target_channels
        })
    elif isinstance(filt, CompoundPassFilter):
        if filt.type == FilterType.BUTTERWORTH and filt.order in [4, 6, 8]:
            pass_type = HighPass if isinstance(filt, ComplexHighPass) else LowPass
            vals = __make_high_order_mc_pass_filter(filt, pass_type.TYPE, target_channels)
            return pass_type(vals)
        else:
            return __make_mc_custom_pass_filter(filt, target_channels)
    elif isinstance(filt, PassFilter):
        pass_type = HighPass if isinstance(filt, SecondOrder_HighPass) else LowPass
        return pass_type(__make_mc_pass_filter(filt, pass_type.TYPE, target_channels))
    elif isinstance(filt, FirstOrder_LowPass) or isinstance(filt, FirstOrder_HighPass):
        pass_type = HighPass if isinstance(filt, FirstOrder_HighPass) else LowPass
        return pass_type(__make_mc_pass_filter(filt, pass_type.TYPE, target_channels))
    elif isinstance(filt, AP):
        return AllPass({
            'Enabled': '1',
            'Slope': '12',
            'Q': f"{filt.q:.7g}",
            'Type': AllPass.TYPE,
            'Version': '1',
            'Gain': '0',
            'Frequency': f"{filt.freq:.7g}",
            'Channels': target_channels
        })
    else:
        raise ValueError(f"Unsupported filter type {filt}")


def __make_mc_custom_pass_filter(p_filter: CompoundPassFilter, target_channels: str) -> CustomPassFilter:
    pass_type = HighPass if isinstance(p_filter, ComplexHighPass) else LowPass
    mc_filts = [pass_type(__make_mc_pass_filter(f, pass_type.TYPE, target_channels)) for f in p_filter.filters]
    type_code = 'HP' if pass_type == HighPass else 'LP'
    encoded = f"{type_code}/{p_filter.type.value}/{p_filter.order}/{p_filter.freq:.7g}/{p_filter.q:.4g}"
    return CustomPassFilter(encoded, mc_filts)


def __make_high_order_mc_pass_filter(f: CompoundPassFilter, filt_type: str, target_channels: str) -> Dict[str, str]:
    return {
        'Enabled': '1',
        'Slope': f"{f.order * 6}",
        'Type': filt_type,
        'Q': f"{f.q:.12g}",
        'Frequency': f"{f.freq:.7g}",
        'Gain': '0',
        'Channels': target_channels
    }


def __make_mc_pass_filter(f: Union[FirstOrder_LowPass, FirstOrder_HighPass, PassFilter], filt_type: str,
                          target_channels: str) -> Dict[str, str]:
    return {
        'Enabled': '1',
        'Slope': f"{f.order * 6}",
        'Type': filt_type,
        'Q': f"{f.q:.12g}" if hasattr(f, 'q') else '1',
        'Frequency': f"{f.freq:.7g}",
        'Gain': '0',
        'Channels': target_channels
    }


class FilterOp(ABC):

    @abstractmethod
    def apply(self, input_signal: Signal) -> Signal:
        pass

    @property
    def ready(self):
        return True

    def __repr__(self):
        class_name = self.__class__.__name__
        return f"{class_name[:-8] if class_name.endswith('FilterOp') else class_name}"


class NopFilterOp(FilterOp):

    def apply(self, input_signal: Signal) -> Signal:
        return input_signal


class CompositeFilterOp(FilterOp):

    def __init__(self):
        self.__children: List[FilterOp] = []

    def append(self, f: FilterOp):
        if not isinstance(f, NopFilterOp):
            if self.__children and isinstance(self.__children[-1], SosFilterOp) and isinstance(f, SosFilterOp):
                self.__children[-1].extend(f)
            else:
                self.__children.append(f)

    def apply(self, input_signal: Signal) -> Signal:
        o = input_signal
        for c in self.__children:
            o = c.apply(o)
        return o


class SosFilterOp(FilterOp):

    def __init__(self, sos: List[List[float]]):
        self.__sos = sos

    def apply(self, input_signal: Signal) -> Signal:
        return input_signal.sosfilter(self.__sos)

    def extend(self, more_sos: SosFilterOp):
        self.__sos += more_sos.__sos


class DelayFilterOp(FilterOp):

    def __init__(self, delay_millis: float):
        self.__delay_ms = delay_millis

    def apply(self, input_signal: Signal) -> Signal:
        return input_signal.shift(self.__delay_ms)


class InvertPolarityFilterOp(FilterOp):

    def apply(self, input_signal: Signal) -> Signal:
        return input_signal.invert()


class CombineFilterOp(FilterOp, ABC):

    def __init__(self, gain: GainFilterOp = None):
        self.__gain = gain if gain else GainFilterOp()
        self.__inbound_signal: Optional[Signal] = None

    def accept(self, signal: Signal):
        if self.ready:
            raise ValueError(f"Attempting to reuse AddFilterOp")
        self.__inbound_signal = signal
        return self

    @property
    def gain(self) -> GainFilterOp:
        return self.__gain

    @property
    def ready(self):
        return self.__inbound_signal is not None

    @property
    def src_signal(self) -> Optional[Signal]:
        return self.__inbound_signal


class AddFilterOp(CombineFilterOp):

    def __init__(self, gain: GainFilterOp = None):
        super().__init__(gain)

    def apply(self, input_signal: Signal) -> Signal:
        if self.src_signal:
            return input_signal.add(self.gain.apply(self.src_signal).samples)
        else:
            raise SimulationFailed()


class SubtractFilterOp(CombineFilterOp):

    def __init__(self, gain: GainFilterOp = None):
        super().__init__(gain)

    def apply(self, input_signal: Signal) -> Signal:
        if self.src_signal:
            return input_signal.subtract(self.gain.apply(self.src_signal).samples)
        else:
            raise SimulationFailed()


class GainFilterOp(FilterOp):

    def __init__(self, gain_db: float = 0.0):
        self.__gain_db = gain_db

    def apply(self, input_signal: Signal) -> Signal:
        return input_signal.offset(self.__gain_db)


class FilterGraph:

    def __init__(self, stage: int, input_channels: List[str], output_channels: List[str], filts: List[Filter],
                 on_delta: Callable[[bool, bool], None] = None, convert_q: bool = False, regen: bool = True):
        self.__on_delta = on_delta
        self.__editing: Optional[Tuple[str, List[str]], int] = None
        self.__stage = stage
        self.__active_idx = -1
        self.__filt_cache = [[]]
        if filts:
            for filt in filts:
                self.__insert(filt, len(self.filters) + 1, regen=False)
        self.__output_channels = output_channels
        self.__input_channels = input_channels
        self.__convert_q = convert_q
        self.__sim: Dict[str, Signal] = {}
        self.__render: str = ''
        if regen:
            self.__regen()

    def undo(self) -> bool:
        changed = False
        if len(self.__filt_cache) > 1:
            if self.__active_idx == -1:
                self.__active_idx = len(self.__filt_cache) - 2
                changed = True
            elif self.__active_idx != 0:
                self.__active_idx = self.__active_idx - 1
                changed = True
        return changed

    def redo(self) -> bool:
        changed = False
        if self.__active_idx != -1:
            changed = True
            self.__active_idx += 1
            if self.__active_idx == len(self.__filt_cache) - 1:
                self.__active_idx = -1
        return changed

    def activate(self):
        self.__update_history()

    def __regen(self):
        '''
        Regenerates the graph.
        '''
        from model.jriver.render import GraphRenderer
        self.__render = GraphRenderer(self).generate()

    def render(self, colours=None, vertical=True, selected_nodes=None) -> str:
        '''
        Renders the graph to dot notation.
        :param colours: the colours to use in the rendering.
        :param vertical: whether to orient top to bottom or left to right.
        :param selected_nodes: the nodes which to be highlighted.
        :return: the dot rendering.
        '''
        from model.jriver.render import GraphRenderer
        return GraphRenderer(self, colours=colours).generate(vertical=vertical, selected_nodes=selected_nodes)

    def __repr__(self):
        return f"Stage {self.__stage} - {len(self.filters)} Filters - {len(self.__input_channels)} in {len(self.__output_channels)} out"

    @property
    def stage(self):
        return self.__stage

    def get_filter_at_node(self, node_name: str) -> Optional[Filter]:
        '''
        Locates the editable filter for the given node.
        :param node_name: the node to search for.
        :return: the filter, if any.
        '''
        for f in self.filters:
            if node_name in f.nodes:
                return f
            elif isinstance(f, Sequence):
                for f1 in flatten(f):
                    if node_name in f1.nodes:
                        return f
        return None

    def get_editable_node_chain(self, node_name: str) -> Tuple[int, List[str]]:
        '''
        Looks up and down the filter list to compile a list of nodes that can be edited as a single unit.
        :param node_name: the node name.
        :return: the index of this node in the chain, the chain of nodes that can be edited as one.
        '''
        node_idx = -1
        node_chain = []
        channel = node_name.split('_')[0]
        found = False
        node_filter = self.get_filter_at_node(node_name)
        if node_filter and node_filter.get_editable_filter():
            for f in self.get_filters_by_channel(channel):
                if f == node_filter:
                    found = True
                    node_idx = len(node_chain)
                if f.get_editable_filter():
                    for n in f.nodes:
                        if n.startswith(f"{channel}_"):
                            node_chain.append(n)
                else:
                    if found:
                        break
                    else:
                        node_idx = -1
                        node_chain = []
        return node_idx, node_chain

    def get_filters_by_channel(self, channel: str) -> List[Filter]:
        '''
        :param channel: the channel name.
        :return: all filters applicable to that channel.
        '''
        channel_idx = get_channel_idx(channel)
        return [f for f in self.filters if f.is_mine(channel_idx)]

    def get_filter_by_id(self, f_id: int) -> Optional[Filter]:
        '''
        Locates the filter with the given id.
        :param f_id: the filter id.
        :return: the filter, if any.
        '''
        return next((f for f in self.filters if f.id == f_id), None)

    def reorder(self, start: int, end: int, to: int) -> None:
        '''
        Moves the filters at indexes start to end to a new position & regenerates the graph.
        :param start: the starting index.
        :param end: the ending index.
        :param to: the position to move to.
        '''
        logger.info(f"Moved rows {start}:{end + 1} to idx {to}")
        new_filters = self.filters[0: start] + self.filters[end + 1:]
        to_insert = self.filters[start: end + 1]
        for i, f in enumerate(to_insert):
            new_filters.insert(to + i - (1 if start < to else 0), f)
        logger.debug(f"Order: {[f for f in new_filters]}")
        self.__append(new_filters)
        self.__regen()

    @property
    def filters(self) -> List[Filter]:
        return self.__filt_cache[self.__active_idx]

    @property
    def all_filters(self) -> Iterable[Filter]:
        for f in flatten(self.filters):
            yield f

    @property
    def input_channels(self):
        return self.__input_channels

    @property
    def output_channels(self):
        return self.__output_channels

    def __append(self, filters: List[Filter] = None):
        filters = filters if filters is not None else [f for f in self.filters]
        if self.__active_idx != -1:
            self.__filt_cache = self.__filt_cache[:self.__active_idx + 1]
        self.__filt_cache.append(filters)
        self.__active_idx = -1
        self.__update_history()

    def __update_history(self):
        logger.info(f"FiltCache: {len(self.__filt_cache)}, idx: {self.__active_idx}")
        if self.__on_delta:
            self.__on_delta(len(self.__filt_cache) > 1, self.__active_idx != -1)

    def start_edit(self, channel: str, node_chain: List[str], insert_at: int):
        self.__editing = (channel, node_chain, insert_at)

    def end_edit(self, new_filters: CompleteFilter):
        '''
        Replaces the nodes identified by the node_chain with the provided set of new filters.
        :param new_filters: the filters.
        '''
        if self.__editing:
            self.__append()
            node_chain: List[str]
            channel_name, node_chain, insert_at = self.__editing
            channel_idx = str(get_channel_idx(channel_name))
            old_filters: List[Tuple[Filter, str]] = [(self.get_filter_at_node(n), n.split('_')[0]) for n in node_chain]
            new_chain_filter: Optional[Filter] = None
            last_match = -1
            must_regen = False
            offset = 0
            # for each new filter
            # look for a matching old filter
            #   if match
            #     if skipped past old filters, delete them
            #   insert filter
            for i, f in enumerate(new_filters):
                new_filter = convert_filter_to_mc_dsp(f, channel_idx, convert_q=self.__convert_q)
                if last_match < len(old_filters):
                    new_filt_vals = new_filter.get_all_vals()
                    handled = False
                    for j in range(last_match + 1, len(old_filters)):
                        old_filter, filter_channel = old_filters[j]
                        old_filt_vals = old_filter.get_all_vals()
                        if pop_channels(new_filt_vals) == pop_channels(old_filt_vals):
                            if (j - last_match) > 1:
                                offset -= self.__delete_filters(old_filters[last_match + 1:j])
                                must_regen = True
                            insert_at = self.__get_filter_idx(old_filter) + 1
                            offset = 0
                            last_match = j
                            new_chain_filter = old_filter
                            handled = True
                            break
                    if not handled:
                        must_regen = True
                        self.__insert(new_filter, insert_at + offset, regen=False)
                        new_chain_filter = new_filter
                        offset += 1
                else:
                    must_regen = True
                    self.__insert(new_filter, insert_at + offset, regen=False)
                    new_chain_filter = new_filter
                    offset += 1
            if last_match + 1 < len(old_filters):
                if self.__delete_filters(old_filters[last_match + 1:]):
                    must_regen = True
                    if not new_chain_filter:
                        new_chain_filter = old_filters[:last_match + 1][0][0]
            if must_regen:
                self.__regen()
                new_chain_node: str = next(n for n in new_chain_filter.nodes if n.split('_')[0] == channel_name)
                editable_chain: List[str] = self.get_editable_node_chain(new_chain_node)[1]
                first_in_chain = self.get_filter_at_node(editable_chain[0])
                self.__editing = (channel_name, editable_chain, self.__get_filter_idx(first_in_chain) + 1)
            return must_regen

    def __get_filter_idx(self, to_match: Filter) -> int:
        return next((i for i, f in enumerate(self.filters) if f.id == to_match.id))

    def __delete_filters(self, to_delete: List[Tuple[Filter, str]]) -> int:
        '''
        Removes the provided channel specific filters.
        :param to_delete: the channel-filter combinations to eliminate.
        :return: the number of deleted filters.
        '''
        deleted = 0
        for filt_to_delete, filt_channel_to_delete in to_delete:
            logger.debug(f"Deleting {filt_channel_to_delete} from {filt_to_delete}")
            if isinstance(filt_to_delete, ChannelFilter):
                replacement = filt_to_delete.pop_channel(filt_channel_to_delete)
                if replacement is None:
                    self.__delete(filt_to_delete)
                    deleted += 1
                else:
                    self.__replace(replacement, filt_to_delete)
            else:
                self.__delete(filt_to_delete)
                deleted += 1
        return deleted

    def __delete(self, filt_to_delete):
        self.filters.pop(self.__get_filter_idx(filt_to_delete))

    def clear_filters(self) -> None:
        '''
        Removes all filters.
        '''
        self.__append([])
        self.__regen()

    def delete_channel(self, channel_filter: ChannelFilter, channel: str) -> bool:
        '''
        Removes the channel from the filter, deleting the filter itself if it has no channels left.
        :param channel_filter: the filter.
        :param channel: the channel.
        :returns true if the filter is deleted.
        '''
        logger.debug(f"Deleting {channel} from {channel_filter}")
        replacement = channel_filter.pop_channel(channel)
        if replacement is None:
            self.__delete(channel_filter)
        else:
            self.replace(channel_filter, replacement)
        self.__regen()
        return replacement is None

    def delete(self, filters: List[Filter]) -> None:
        '''
        Removes the provided list of filters.
        :param filters: the filters to remove.
        '''
        ids_to_delete: List[int] = [f.id for f in filters]
        self.__append([f for f in self.filters if f.id not in ids_to_delete])
        self.__regen()

    def insert(self, to_insert: Filter, at: int, regen=True) -> None:
        '''
        Inserts a filter a specified position.
        :param to_insert: the filter to insert.
        :param at: the position to insert at.
        :param regen: react to the insertion by regenerating the graph.
        '''
        self.__append()
        self.__insert(to_insert, at, regen)

    def append(self, to_append: Filter, regen=True) -> None:
        self.insert(to_append, 2 ** 24, regen=regen)

    def __insert(self, to_insert: Filter, at: int, regen=True) -> None:
        if at < len(self.filters):
            if len(self.filters) == 1:
                filt_id = self.filters[0].id / 2
            else:
                filt_id = ((self.filters[at].id - self.filters[at - 1].id) / 2) + self.filters[at - 1].id
                if filt_id >= self.filters[at].id:
                    raise ValueError(
                        f"Unable to insert filter at {at}, attempting to insert {filt_id} before {self.filters[at].id}")
        else:
            if len(self.filters) == 0:
                filt_id = 2 ** 24
            else:
                filt_id = self.filters[-1].id + 2 ** 24
        to_insert.id = filt_id
        self.filters.insert(at, to_insert)
        if regen:
            self.__regen()

    def replace(self, old_filter: Filter, new_filter: Filter) -> bool:
        '''
        Replaces the specified filter.
        :param old_filter: the filter to replace.
        :param new_filter:  the replacement filter.
        :return: true if it was replaced.
        '''
        self.__append()
        return self.__replace(new_filter, old_filter)

    def __replace(self, new_filter, old_filter):
        try:
            new_filter.id = old_filter.id
            self.filters[self.__get_filter_idx(old_filter)] = new_filter
            self.__regen()
            return True
        except:
            return False

    def simulate(self, analysis_resolution=1.0, recalc=True) -> Dict[str, Signal]:
        """
        Applies each filter to unit impulse per channel.
        """
        if recalc or self.__sim is None:
            start = time.time()
            signals: Dict[str, Tuple[Signal, Optional[SosFilterOp]]] = {
                c: (
                    make_dirac_pulse(c,
                                     analysis_resolution=analysis_resolution) if c in self.input_channels else make_silence(
                        c), None)
                for c in self.output_channels
            }
            for f in self.all_filters:
                self.__simulate_filter(f, signals)
            end = time.time()
            final_output = {k: v[1].apply(v[0]) if v[1] else v[0] for k, v in signals.items()}
            logger.info(f"Generated {len(signals)} signals in {to_millis(start, end)} ms")
            self.__sim = final_output
        return self.__sim

    @staticmethod
    def __simulate_filter(f: Filter, signals: Dict[str, Tuple[Signal, Optional[SosFilterOp]]]):
        if not f.enabled:
            return
        if isinstance(f, ChannelFilter) or isinstance(f, ComplexChannelFilter):
            for c in f.channel_names:
                signal, pending_sos = signals[c]
                filter_op = f.get_filter_op()
                if isinstance(filter_op, SosFilterOp):
                    if pending_sos:
                        pending_sos.extend(filter_op)
                    else:
                        signals[c] = (signal, filter_op)
                else:
                    if pending_sos:
                        signal = pending_sos.apply(signal)
                    signal = filter_op.apply(signal)
                    signals[c] = (signal, None)
        elif isinstance(f, Mix):
            dst_channel = get_channel_name(f.dst_idx)
            dst_signal, pending_sos = signals[dst_channel]
            if pending_sos:
                dst_signal = pending_sos.apply(dst_signal)

            src_channel = get_channel_name(f.src_idx)
            src_signal, pending_sos = signals[src_channel]
            if pending_sos:
                src_signal = pending_sos.apply(src_signal)

            filter_op = f.get_filter_op()
            if f.mix_type == MixType.ADD or f.mix_type == MixType.SUBTRACT:
                dst_signal = filter_op.accept(src_signal).apply(dst_signal)
            elif f.mix_type == MixType.MOVE:
                dst_signal = filter_op.apply(src_signal).copy(new_name=dst_channel)
                src_signal = make_dirac_pulse(src_channel)
            elif f.mix_type == MixType.COPY:
                dst_signal = filter_op.apply(src_signal).copy(new_name=dst_channel)
            elif f.mix_type == MixType.SWAP:
                src_signal = filter_op.apply(dst_signal).copy(new_name=src_channel)
                dst_signal = filter_op.apply(src_signal).copy(new_name=dst_channel)

            signals[src_channel] = (src_signal, None)
            signals[dst_channel] = (dst_signal, None)
        elif not isinstance(f.get_filter_op(), NopFilterOp):
            for c in signals.keys():
                signal, pending_sos = signals[c]
                if pending_sos:
                    signal = pending_sos.apply(signal)
                signals[c] = (f.get_filter_op().apply(signal), None)


class SimulationFailed(Exception):
    pass


def set_filter_ids(filters: List[Filter]) -> List[Filter]:
    '''
    ensures filters have a monotonic filter id.
    :param filters: filters that may need IDs.
    :return: filters with IDs set.
    '''
    for i, f in enumerate(filters):
        f.id = (i + 1) * (2 ** 24)
        if isinstance(f, ComplexFilter):
            for i1, f1 in enumerate(f.filters):
                f1.id = f.id + 1 + i1
    return filters


class XO:

    def __init__(self, out_channel_lp: List[str], out_channel_hp: List[str], fs: int = JRIVER_FS):
        self.__in_channel = None
        self.out_channel_lp = out_channel_lp
        self.out_channel_hp = out_channel_hp
        self.extra_delay_millis: float = 0.0
        self.__fs = fs

    @property
    def fs(self) -> int:
        return self.__fs

    @property
    def in_channel(self) -> str:
        return self.__in_channel

    @in_channel.setter
    def in_channel(self, in_channel: str):
        self.__in_channel = in_channel

    @abc.abstractmethod
    def calc_filters(self) -> List[Filter]:
        raise NotImplementedError()

    @abc.abstractmethod
    def to_json(self) -> dict:
        raise NotImplementedError()

    def __repr__(self):
        return f"{self.out_channel_lp}/{self.out_channel_hp}"


class StandardXO(XO):

    def __init__(self, out_channel_lp: List[str], out_channel_hp: List[str],
                 low_pass: Optional[ComplexLowPass] = None, high_pass: Optional[ComplexHighPass] = None,
                 fs: int = JRIVER_FS):
        super().__init__(out_channel_lp, out_channel_hp, fs=fs)
        self.__low_pass = low_pass
        self.__high_pass = high_pass

    def to_json(self) -> dict:
        return {
            'l': self.__low_pass.to_json() if self.__low_pass else {},
            'h': self.__high_pass.to_json() if self.__high_pass else {},
            'c': [self.out_channel_lp, self.out_channel_hp]
        }

    @staticmethod
    def from_json(d: dict) -> StandardXO:
        from model.codec import filter_from_json
        return StandardXO(*d['c'],
                          low_pass=filter_from_json(d['l']) if d['l'] else None,
                          high_pass=filter_from_json(d['h']) if d['h'] else None)

    def calc_filters(self) -> List[Filter]:
        assert self.in_channel is not None

        lp_out = self.out_channel_lp[0]
        lp_final = self.in_channel

        hp_out = self.out_channel_hp[0]
        hp_in = self.in_channel

        filters = []
        if self.__high_pass and self.in_channel != hp_out:
            filters.append(create_single_filter({
                **Mix.default_values(),
                'Source': str(get_channel_idx(self.in_channel)),
                'Destination': str(get_channel_idx('U2')),
                'Mode': str(MixType.COPY.value if self.__low_pass else MixType.MOVE.value)
            }))

        extra_lp_mt = MixType.COPY
        if self.__low_pass:
            if self.in_channel == lp_out:
                filters.append(convert_filter_to_mc_dsp(self.__low_pass, str(get_channel_idx(lp_out))))
            else:
                filters.append(create_single_filter({
                    **Mix.default_values(),
                    'Source': str(get_channel_idx(self.in_channel)),
                    'Destination': str(get_channel_idx('U1')),
                    'Mode': str(MixType.COPY.value)
                }))
                filters.append(convert_filter_to_mc_dsp(self.__low_pass, str(get_channel_idx('U1'))))
                filters.append(create_single_filter({
                    **Mix.default_values(),
                    'Source': str(get_channel_idx('U1')),
                    'Destination': str(get_channel_idx(lp_out)),
                    'Mode': str(MixType.COPY.value if len(self.out_channel_lp) > 1 else MixType.MOVE.value)
                }))
                lp_final = 'U1'
                extra_lp_mt = MixType.MOVE
            if len(self.out_channel_lp) > 1:
                for i, extra_lp_out in enumerate(self.out_channel_lp[1:]):
                    mode = extra_lp_mt
                    if mode == MixType.MOVE and i + 1 != len(self.out_channel_lp):
                        mode = MixType.COPY
                    filters.append(create_single_filter({
                        **Mix.default_values(),
                        'Source': str(get_channel_idx(lp_final)),
                        'Destination': str(get_channel_idx(extra_lp_out)),
                        'Mode': str(mode.value)
                    }))
        elif self.in_channel != lp_out:
            filters.append(create_single_filter({
                **Mix.default_values(),
                'Source': str(get_channel_idx(self.in_channel)),
                'Destination': str(get_channel_idx(lp_out)),
                'Mode': str(MixType.COPY.value)
            }))

        extra_hp_mt = MixType.COPY
        if self.__high_pass:
            if self.in_channel == hp_out:
                filters.append(convert_filter_to_mc_dsp(self.__high_pass, str(get_channel_idx(hp_out))))
            else:
                filters.append(convert_filter_to_mc_dsp(self.__high_pass, str(get_channel_idx('U2'))))
                filters.append(create_single_filter({
                    **Mix.default_values(),
                    'Source': str(get_channel_idx('U2')),
                    'Destination': str(get_channel_idx(hp_out)),
                    'Mode': str(str(MixType.COPY.value if len(self.out_channel_hp) > 1 else MixType.MOVE.value))
                }))
                hp_in = 'U2'
                extra_hp_mt = MixType.MOVE
            if len(self.out_channel_hp) > 1:
                for i, extra_hp_out in enumerate(self.out_channel_hp[1:]):
                    mode = extra_hp_mt
                    if mode == MixType.MOVE and i + 1 != len(self.out_channel_hp):
                        mode = MixType.COPY
                    filters.append(create_single_filter({
                        **Mix.default_values(),
                        'Source': str(get_channel_idx(hp_in)),
                        'Destination': str(get_channel_idx(extra_hp_out)),
                        'Mode': str(mode.value)
                    }))
        elif self.in_channel != hp_out and hp_out != lp_out:
            filters.append(create_single_filter({
                **Mix.default_values(),
                'Source': str(get_channel_idx(self.in_channel)),
                'Destination': str(get_channel_idx(hp_out)),
                'Mode': str(MixType.COPY.value)
            }))

        if not self.__low_pass and not self.__high_pass:
            extra_channels = (self.out_channel_lp[1:] if len(self.out_channel_lp) > 1 else []) + (self.out_channel_hp[1:] if len(self.out_channel_hp) > 1 else [])
            if extra_channels:
                for c in {e for e in extra_channels}:
                    filters.append(create_single_filter({
                        **Mix.default_values(),
                        'Source': str(get_channel_idx(self.in_channel)),
                        'Destination': str(get_channel_idx(c)),
                        'Mode': str(MixType.COPY.value)
                    }))

        return filters

    @staticmethod
    def __make_mc_filter(value: float, filt_type: Type[SingleFilter], channel_indexes: str, key: str) -> Filter:
        return create_single_filter({
            **filt_type.default_values(),
            key: f"{value:.7g}",
            'Channels': channel_indexes
        })

    def __repr__(self):
        lp = f"LP: {self.__low_pass}" if self.__low_pass else ''
        hp = f"HP: {self.__high_pass}" if self.__high_pass else ''
        return f"{super().__repr__()} {hp} {lp}"


class MDSXO(XO):

    def __init__(self, order: int, target_fc: float, fc_divisor: float = 0.0, lp_channel: List[str] = None,
                 hp_channel: List[str] = None, fs: int = JRIVER_FS, calc: bool = False):
        super().__init__(lp_channel, hp_channel, fs=fs)
        self.__order = order
        self.__target_fc = target_fc
        self.__fc_divisor = fc_divisor if fc_divisor != 0.0 else MDS_FREQ_DIVISOR[order]
        self.__dc_gd_millis = self.__calc_dc_gd(self.__make_low_pass())
        self.__multi_way_delay_filter: Optional[Delay] = None
        self.__graph: Optional[FilterGraph] = None
        self.__output = None
        self.__crossing = None
        self.__lp_slope = None
        self.__hp_slope = None
        if calc is True:
            self.__recalc()

    def to_json(self) -> dict:
        return {'o': self.__order, 'f': self.__target_fc, 'd': self.__fc_divisor, 'l': self.out_channel_lp,
                'h': self.out_channel_hp}

    @staticmethod
    def from_json(self, d: dict) -> MDSXO:
        return MDSXO(d['o'], d['f'], fc_divisor=d['d'], lp_channel=d['l'], hp_channel=d['h'])

    @XO.in_channel.setter
    def in_channel(self, in_channel):
        XO.in_channel.fset(self, in_channel)
        self.__calc_graph()

    @property
    def filter_graph(self) -> FilterGraph:
        self.__recalc()
        return self.__graph

    def __recalc(self):
        if self.__graph is not None:
            self.__graph = None
            self.__calc_graph()
        if self.__output is not None:
            self.__output = None
            self.__calc_output()

    @property
    def delay(self) -> float:
        return self.__dc_gd_millis

    @property
    def delay_samples(self) -> float:
        return self.delay / (1 / (self.fs / 1000))

    @property
    def target_fc(self) -> float:
        return self.__target_fc

    @property
    def actual_fc(self) -> float:
        self.__metrics()
        return self.__crossing

    @property
    def lp_slope(self) -> float:
        self.__metrics()
        return self.__lp_slope

    @property
    def hp_slope(self) -> float:
        self.__metrics()
        return self.__hp_slope

    def __metrics(self):
        if self.__crossing is None:
            self.__crossing, self.__lp_slope, self.__hp_slope = self.__assess_performance()

    @property
    def fc_delta(self) -> float:
        return self.__target_fc - self.actual_fc

    @property
    def optimised_divisor(self) -> float:
        return self.actual_fc / self.__target_fc * self.__fc_divisor

    @property
    def fc_divisor(self) -> float:
        return self.__fc_divisor

    @property
    def lp_output(self) -> Signal:
        self.__calc_output()
        return self.__output[self.out_channel_lp]

    def __calc_output(self):
        self.__recalc()
        if self.__output is None:
            self.__output = self.__graph.simulate(analysis_resolution=0.1)

    @property
    def hp_output(self) -> Signal:
        self.__calc_output()
        return self.__output[self.out_channel_hp]

    @property
    def order(self) -> int:
        return self.__order

    def __assess_performance(self) -> Tuple[float, float, float]:
        lp_x, lp_y = self.lp_output.avg
        hp_x, hp_y = self.hp_output.avg
        fc_idx = np.argmax(hp_y >= lp_y).item()
        hp_slope_start_idx = np.argmax(hp_x > (self.target_fc * 0.25))
        hp_slope_end_idx = np.argmax(hp_x > (self.target_fc * 0.125))
        hp_slope = hp_y[hp_slope_start_idx.item()] - hp_y[hp_slope_end_idx.item()]
        lp_slope_start_idx = np.argmax(lp_x > self.target_fc * 4)
        lp_slope_end_idx = np.argmax(lp_x > self.target_fc * 8)
        lp_slope = lp_y[lp_slope_start_idx.item()] - lp_y[lp_slope_end_idx.item()]
        return lp_x[fc_idx], lp_slope, hp_slope

    def calc_filters(self) -> List[Filter]:
        self.__calc_graph()
        return [f for f in self.__graph.filters]

    def __calc_graph(self):
        '''
        Applies filters required to implement the "Matched-Delay Subtractive Crossover Pair With 4th-Order Highpass
        Slope" (copyright Gregory Berchin). This is implemented as:

                   -- Delay --(+)     -- Delay --(+)   U2
                  /             \    /             \
                 /               \  /          U1   \
        input    ---- LPF ----- (-) ---- LPF ------ (-) ----- highpass
                 \                                   \
                  -------- Delay ------- Delay ---- (+) ----- lowpass
        :return:
        '''
        assert self.in_channel is not None
        channels = list({c for c in ([self.in_channel] + self.out_channel_lp + self.out_channel_hp + SHORT_USER_CHANNELS)})
        out_lp_ch = self.out_channel_lp[0]
        out_hp_ch = self.out_channel_hp[0]
        graph = FilterGraph(0, channels, channels, [])
        # copy input to user channels
        graph.append(Mix({
            **Mix.default_values(),
            'Source': str(get_channel_idx(self.in_channel)),
            'Destination': str(get_channel_idx('U1')),
            'Mode': str(MixType.COPY.value)
        }), regen=False)
        graph.append(Mix({
            **Mix.default_values(),
            'Source': str(get_channel_idx(self.in_channel)),
            'Destination': str(get_channel_idx('U2')),
            'Mode': str(MixType.COPY.value)
        }), regen=False)
        # create the high pass output
        self.__add_delayed_low_pass(graph, self.__make_low_pass())
        self.__add_delayed_low_pass(graph, self.__make_low_pass())
        # prepare the low pass output
        if self.in_channel != out_lp_ch:
            # if the input is not the lp channel then copy it
            graph.append(Mix({
                **Mix.default_values(),
                'Source': str(get_channel_idx(self.in_channel)),
                'Destination': str(get_channel_idx(out_lp_ch)),
                'Mode': str(MixType.COPY.value)
            }), regen=False)
        # delay the lp by 2d (split into 2 to ensure the same delay is applied at every stage to )
        graph.append(Delay({
            **Delay.default_values(),
            'Channels': str(get_channel_idx(out_lp_ch)),
            'Delay': f"{self.__dc_gd_millis:.7g}",
        }), regen=False)
        graph.append(Delay({
            **Delay.default_values(),
            'Channels': str(get_channel_idx(out_lp_ch)),
            'Delay': f"{self.__dc_gd_millis:.7g}",
        }), regen=False)
        # subtract U1 to make the low pass output
        graph.append(Mix({
            **Mix.default_values(),
            'Source': str(get_channel_idx('U1')),
            'Destination': str(get_channel_idx(out_lp_ch)),
            'Mode': str(MixType.SUBTRACT.value)
        }), regen=False)
        # move high pass to output
        graph.append(Mix({
            **Mix.default_values(),
            'Source': str(get_channel_idx('U1')),
            'Destination': str(get_channel_idx(out_hp_ch)),
            'Mode': str(MixType.MOVE.value)
        }), regen=False)
        if not math.isclose(self.extra_delay_millis, 0.0):
            graph.append(Delay({
                **Delay.default_values(),
                'Channels': str(get_channel_idx(out_lp_ch)),
                'Delay': f"{self.extra_delay_millis:.7g}",
            }), regen=False)
        # copy to any remaining output channels
        for extra_out in self.out_channel_lp[1:]:
            graph.append(Mix({
                **Mix.default_values(),
                'Source': str(get_channel_idx(out_lp_ch)),
                'Destination': str(get_channel_idx(extra_out)),
                'Mode': str(MixType.COPY.value)
            }), regen=False)
        for extra_out in self.out_channel_hp[1:]:
            graph.append(Mix({
                **Mix.default_values(),
                'Source': str(get_channel_idx(out_hp_ch)),
                'Destination': str(get_channel_idx(extra_out)),
                'Mode': str(MixType.COPY.value)
            }), regen=False)

        self.__graph = graph

    def __make_low_pass(self) -> ComplexLowPass:
        return ComplexLowPass(FilterType.BESSEL_MAG6, self.__order, self.fs, self.lp_fc)

    @property
    def lp_fc(self) -> float:
        return self.__target_fc / self.__fc_divisor

    def __add_delayed_low_pass(self, graph, low_pass: ComplexLowPass):
        graph.append(convert_filter_to_mc_dsp(low_pass, str(get_channel_idx('U1'))), regen=False)
        graph.append(Delay({
            **Delay.default_values(),
            'Channels': str(get_channel_idx('U2')),
            'Delay': f"{self.__dc_gd_millis:.7g}",
        }), regen=False)
        graph.append(Mix({
            **Mix.default_values(),
            'Source': str(get_channel_idx('U1')),
            'Destination': str(get_channel_idx('U2')),
            'Mode': str(MixType.SUBTRACT.value)
        }), regen=False)
        graph.append(Mix({
            **Mix.default_values(),
            'Source': str(get_channel_idx('U2')),
            'Destination': str(get_channel_idx('U1')),
            'Mode': str(MixType.COPY.value)
        }), regen=False)

    def __calc_dc_gd(self, low_pass: ComplexLowPass) -> float:
        from scipy.signal import group_delay
        gd = None
        for i, f in enumerate(low_pass.filters):
            # w=1 calculates at DC only
            f_s, gd_c = group_delay((f.b, f.a), w=1)
            gd_c = gd_c / self.fs * 1000.0
            if i == 0:
                gd = gd_c
            else:
                gd = gd + gd_c
        return round(gd[0], 6)

    def __repr__(self):
        return f"MDS {super().__repr__()} {self.order} / {self.target_fc:.2f}"


class WayValues:
    def __init__(self, way: int, delay_millis: float = 0.0, gain: float = 0.0, inverted: bool = False,
                 lp: list = None, hp: list = None):
        self.way = way
        self.delay_millis = delay_millis
        self.gain = gain
        self.inverted = inverted
        self.lp = lp
        self.hp = hp

    def to_json(self) -> str:
        return json.dumps({
            'w': self.way,
            'd': self.delay_millis,
            'g': self.gain,
            'i': 'Y' if self.inverted else 'N',
            'l': self.lp if self.lp else [],
            'h': self.hp if self.hp else []
        })

    @staticmethod
    def from_json(v: dict) -> WayValues:
        lp = v['l']
        hp = v['h']
        return WayValues(v['w'], v['d'], v['g'], True if v['i'] == 'Y' else False,
                         lp=lp if lp else None, hp=hp if hp else None)

    def __repr__(self) -> str:
        return f"W{self.way} Gain: {self.gain:.2f} Delay: {self.delay_millis:.3f} Inverted: {self.inverted}"


class MultiwayCrossover:
    '''
    Example of how multiway output is generated
    input -> LP -> L
          -> HP -> way 2 -> LP -> R
                         -> HP -> way 3 -> LP -> C
                                        -> HP -> SL
    '''

    def __init__(self, in_channel: str, crossovers: List[XO], way_values: List[WayValues],
                 subsonic_filter: Optional[ComplexHighPass] = None, fs: int = JRIVER_FS):
        self.__fs = fs
        assert in_channel is not None
        self.__in_channel = in_channel
        self.__xos = crossovers
        self.__channels: List[str] = []
        delay = 0
        for i, xo in enumerate(reversed(self.__xos)):
            if i != 0 and delay != 0:
                xo.extra_delay_millis = delay
            if isinstance(xo, MDSXO):
                delay += (xo.delay * 2)

        filters = []
        channels_by_way = []
        last_x: Optional[XO] = None
        for i, x in enumerate(self.__xos):
            x.in_channel = self.__in_channel if i == 0 else last_x.out_channel_hp[0]
            if i == 0 and subsonic_filter:
                for lp_out in x.out_channel_lp:
                    filters.append(convert_filter_to_mc_dsp(subsonic_filter, str(get_channel_idx(lp_out))))
            x_filters = x.calc_filters()
            filters.extend(x_filters)
            last_x = x
            channels_by_way.extend(x.out_channel_lp)
            for lp_out in x.out_channel_lp:
                if lp_out not in self.__channels:
                    self.__channels.append(lp_out)
        if last_x:
            for hp_out in x.out_channel_hp:
                if hp_out not in self.__channels:
                    self.__channels.append(hp_out)
            channels_by_way.extend(last_x.out_channel_hp)

        for i, v in enumerate(way_values):
            if v is not None:
                try:
                    output_index = self.__channels[i]
                    if v.inverted:
                        filters.append(Polarity({
                            'Enabled': '1',
                            'Type': Polarity.TYPE,
                            'Channels': str(get_channel_idx(output_index))
                        }))
                    if not math.isclose(v.gain, 0.0):
                        filters.append(create_single_filter({
                            **Gain.default_values(),
                            'Gain': f"{v.gain:.7g}",
                            'Channels': str(get_channel_idx(output_index))
                        }))
                    if not math.isclose(v.delay_millis, 0.0):
                        filters.append(create_single_filter({
                            **Delay.default_values(),
                            'Delay': f"{v.delay_millis:.7g}",
                            'Channels': str(get_channel_idx(output_index))
                        }))
                except IndexError as e:
                    raise e

        channels_with_user = list(self.__channels) + SHORT_USER_CHANNELS
        self.__graph = FilterGraph(0, channels_with_user, channels_with_user, filters, regen=False)
        self.__output = None

    @property
    def output(self) -> List[Signal]:
        if self.__output is None:
            self.__output = self.__graph.simulate(analysis_resolution=0.1)
        return [self.__output[c] for c in self.__channels]

    @property
    def sum(self) -> Signal:
        from functools import reduce
        return reduce(lambda s1, s2: s1.add(s2), self.output)

    @property
    def graph(self) -> FilterGraph:
        return self.__graph

    def __repr__(self):
        return ' -- '.join([str(w) for w in self.__xos])


def optimise_mds(order: int, fc: float) -> MDSXO:
    attempts = []
    factor = MDS_FREQ_DIVISOR[order]
    while True:
        mds = MDSXO(order, fc, fc_divisor=factor)
        attempts.append(mds)
        if mds.actual_fc == 0.0:
            logger.warning(f"No cross found for MDS {order}:{fc:.2f} - {mds}")
            break
        else:
            factor = mds.optimised_divisor
            if len(attempts) > 6:
                delta_fc_1 = abs(attempts[-2].fc_delta)
                delta_fc_2 = abs(attempts[-1].fc_delta)
                logger.warning(
                    f"Unable to reach target after {len(attempts)} for {order}/{fc:.2f}, discarding last [{delta_fc_1:.2f} vs {delta_fc_2:.2f}]")
                break
            elif len(attempts) > 1:
                delta_fc_1 = attempts[-2].fc_delta
                delta_fc_2 = attempts[-1].fc_delta
                if math.isclose(round(delta_fc_1, 1), round(delta_fc_2, 1)):
                    attempts.pop()
                    logger.info(f"Optimal solution reached after {len(attempts)} attempts for {order}/{fc:.2f}")
                    break
                elif abs(delta_fc_2) >= abs(delta_fc_1):
                    logger.warning(
                        f"solutions diverging after {len(attempts)} for {order}/{fc:.2f}, discarding last [{delta_fc_1:.2f} vs {delta_fc_2:.2f}]")
                    attempts.pop()
                    break
    return attempts[-1]
