from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from typing import List, Dict, Optional, Callable, Union, Type, Sequence, overload, Tuple

import math

from model import iir
from model.iir import SOS, s_to_q, q_to_s, FirstOrder_LowPass, FirstOrder_HighPass, PassFilter, CompoundPassFilter, \
    FilterType, SecondOrder_LowPass, ComplexLowPass, SecondOrder_HighPass, ComplexHighPass, CompleteFilter, \
    BiquadWithQGain, PeakingEQ, LowShelf as LS, Gain as G, LinkwitzTransform as LT
from model.jriver.codec import filts_to_xml
from model.jriver.common import get_channel_name, pop_channels, get_channel_idx, JRIVER_SHORT_CHANNELS, \
    SHORT_USER_CHANNELS
from model.signal import Signal

logger = logging.getLogger('jriver.filter')


class Filter(ABC):

    def __init__(self, short_name: str):
        self.__short_name = short_name
        self.__f_id = -1
        self.__nodes: List[Node] = []

    def reset(self) -> None:
        self.__nodes = []

    @property
    def id(self) -> int:
        return self.__f_id

    @id.setter
    def id(self, f_id: int):
        self.__f_id = f_id

    @property
    def nodes(self) -> List[Node]:
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
    def get_all_vals(self) -> List[Dict[str, str]]:
        pass

    def get_editable_filter(self) -> Optional[SOS]:
        return None

    def get_filter(self) -> FilterOp:
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


class SingleFilter(Filter):

    def __init__(self, vals, short_name):
        super().__init__(short_name)
        self.__vals = vals
        self.__enabled = vals['Enabled'] == '1'
        self.__type_code = vals['Type']

    @property
    def enabled(self):
        return self.__enabled

    def get_all_vals(self) -> List[Dict[str, str]]:
        vals = {
            'Enabled': '1' if self.__enabled else '0',
            'Type': self.__type_code,
            **self.get_vals()
        }
        if self.key_order:
            return [{k: vals[k] for k in self.key_order}]
        else:
            return [vals]

    @property
    def key_order(self) -> List[str]:
        return []

    def get_vals(self) -> Dict[str, str]:
        return {}

    def print_disabled(self):
        return '' if self.enabled else f" *** DISABLED ***"

    @classmethod
    def default_values(cls) -> Dict[str, str]:
        return {
            'Enabled': '1',
            'Type': cls.TYPE
        }


class ChannelFilter(SingleFilter):

    def __init__(self, vals, short_name):
        super().__init__(vals, short_name)
        self.__channels = [int(c) for c in vals['Channels'].split(';')]
        self.__channel_names = [get_channel_name(i) for i in self.__channels]

    @property
    def channels(self) -> List[int]:
        return self.__channels

    @property
    def channel_names(self) -> List[str]:
        return self.__channel_names

    def get_vals(self) -> Dict[str, str]:
        return {
            'Channels': ';'.join([str(c) for c in self.channels]),
            **super().get_vals()
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
        return [create_peq({**self.get_all_vals()[0], 'Channels': str(c)}) for c in self.channels]

    def is_mine(self, idx):
        return idx in self.channels

    def pop_channel(self, channel_name: str):
        '''
        Removes a channel from the filter.
        :param channel_name: the (short) channel name to remove.
        '''
        if channel_name in self.__channel_names:
            self.__channel_names.remove(channel_name)
            self.__channels.remove(get_channel_idx(channel_name))
        else:
            raise ValueError(f"{channel_name} not found in {self}")

    @classmethod
    def default_values(cls) -> Dict[str, str]:
        return {
            **super(ChannelFilter, cls).default_values(),
            'Channels': ''
        }


class GainQFilter(ChannelFilter):

    def __init__(self, vals, create_iir, short_name):
        super().__init__(vals, short_name)
        self.__create_iir = create_iir
        self.__gain = float(vals['Gain'])
        self.__frequency = float(vals['Frequency'])
        self.__q = self.from_jriver_q(float(vals['Q']), self.__gain)

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

    def get_vals(self) -> Dict[str, str]:
        return {
            'Slope': '12',
            'Q': f"{self.to_jriver_q(self.__q, self.__gain):.4g}",
            'Gain': f"{self.__gain:.7g}",
            'Frequency': f"{self.__frequency:.7g}",
            **super().get_vals()
        }

    @classmethod
    def from_jriver_q(cls,  q: float, gain: float) -> float:
        return q

    @classmethod
    def to_jriver_q(cls,  q: float, gain: float) -> float:
        return q

    def get_filter(self) -> FilterOp:
        sos = self.get_editable_filter().get_sos()
        if sos:
            return SosFilterOp(sos)
        else:
            return NopFilterOp()

    def get_editable_filter(self) -> Optional[SOS]:
        return self.__create_iir(48000, self.__frequency, self.__q, self.__gain, f_id=self.id)

    def __repr__(self):
        return f"{self.__class__.__name__} {self.__gain:+.7g} dB Q={self.__q:.4g} at {self.__frequency:.7g} Hz {self.print_channel_names()}{self.print_disabled()}"


class Peak(GainQFilter):
    TYPE = '3'

    def __init__(self, vals):
        super().__init__(vals, iir.PeakingEQ, 'Peak')


class LowShelf(GainQFilter):
    TYPE = '10'

    def __init__(self, vals):
        super().__init__(vals, iir.LowShelf, 'LS')

    @classmethod
    def from_jriver_q(cls, q: float, gain: float):
        return s_to_q(q, gain)

    @classmethod
    def to_jriver_q(cls, q: float, gain: float):
        return q_to_s(q, gain)


class HighShelf(GainQFilter):
    TYPE = '11'

    def __init__(self, vals):
        super().__init__(vals, iir.HighShelf, 'HS')

    @classmethod
    def from_jriver_q(cls, q: float, gain: float) -> float:
        return s_to_q(q, gain)

    @classmethod
    def to_jriver_q(cls, q: float, gain: float) -> float:
        return q_to_s(q, gain)


class Pass(ChannelFilter):

    def __init__(self, vals: dict, short_name: str,
                 one_pole_ctor: Callable[..., Union[FirstOrder_LowPass, FirstOrder_HighPass]],
                 two_pole_ctor: Callable[..., PassFilter],
                 many_pole_ctor: Callable[..., CompoundPassFilter]):
        super().__init__(vals, short_name)
        self.__order = int(int(vals['Slope']) / 6)
        self.__frequency = float(vals['Frequency'])
        self.__jriver_q = float(vals['Q'])
        self.__ctors = (one_pole_ctor, two_pole_ctor, many_pole_ctor)

    @classmethod
    def from_jriver_q(cls, q: float) -> float:
        return q / 2**0.5

    @classmethod
    def to_jriver_q(cls, q: float) -> float:
        return q * 2**0.5

    @property
    def freq(self) -> float:
        return self.__frequency

    @property
    def jriver_q(self) -> float:
        return self.__jriver_q

    @property
    def q(self):
        if self.order == 2:
            return self.from_jriver_q(self.jriver_q)
        else:
            return self.jriver_q

    @property
    def order(self) -> int:
        return self.__order

    def get_vals(self) -> Dict[str, str]:
        return {
            'Gain': '0',
            'Slope': f"{self.order * 6}",
            'Q': f"{self.jriver_q:.4g}",
            'Frequency': f"{self.__frequency:.7g}",
            **super().get_vals()
        }

    @property
    def key_order(self) -> List[str]:
        return ['Enabled', 'Slope', 'Q', 'Type', 'Gain', 'Frequency', 'Channels']

    def get_filter(self) -> FilterOp:
        sos = self.get_editable_filter().get_sos()
        if sos:
            return SosFilterOp(sos)
        else:
            return NopFilterOp()

    def short_desc(self):
        q_suffix = ''
        if not math.isclose(self.jriver_q, 1.0):
            q_suffix = f" VarQ"
        if self.freq >= 1000:
            f = f"{self.freq/1000.0:.3g}k"
        else:
            f = f"{self.freq:g}"
        return f"{self.short_name}{self.order} BW {f}{q_suffix}"

    def get_editable_filter(self) -> Optional[SOS]:
        '''
        Creates a set of biquads which translate the non standard jriver Q into a real Q value.
        :return: the filters.
        '''
        if self.order == 1:
            return self.__ctors[0](48000, self.freq, f_id=self.id)
        elif self.order == 2:
            return self.__ctors[1](48000, self.freq, q=self.q, f_id=self.id)
        else:
            return self.__ctors[2](FilterType.BUTTERWORTH, self.order, 48000, self.freq, q_scale=self.q, f_id=self.id)

    def __repr__(self):
        return f"{self.__class__.__name__} Order={self.order} Q={self.q:.4g} at {self.freq:.7g} Hz {self.print_channel_names()}{self.print_disabled()}"


class LowPass(Pass):
    TYPE = '1'

    def __init__(self, vals):
        super().__init__(vals, 'LP', FirstOrder_LowPass, SecondOrder_LowPass, ComplexLowPass)


class HighPass(Pass):
    TYPE = '2'

    def __init__(self, vals):
        super().__init__(vals, 'HP', FirstOrder_HighPass, SecondOrder_HighPass, ComplexHighPass)


class Gain(ChannelFilter):
    TYPE = '4'

    def __init__(self, vals):
        super().__init__(vals, 'GAIN')
        self.__gain = float(vals['Gain'])

    @property
    def gain(self):
        return self.__gain

    def get_vals(self) -> Dict[str, str]:
        return {
            'Gain': f"{self.__gain:.7g}",
            **super().get_vals()
        }

    @property
    def key_order(self) -> List[str]:
        return ['Enabled', 'Type', 'Gain', 'Channels']

    def __repr__(self):
        return f"Gain {self.__gain:+.7g} dB {self.print_channel_names()}{self.print_disabled()}"

    def get_filter(self) -> FilterOp:
        return GainFilterOp(self.__gain)

    def short_desc(self):
        return f"{self.__gain:+.7g} dB"

    def get_editable_filter(self) -> Optional[SOS]:
        return iir.Gain(48000, self.gain, f_id=self.id)


class BitdepthSimulator(SingleFilter):
    TYPE = '13'

    def __init__(self, vals):
        super().__init__(vals, 'BITDEPTH')
        self.__bits = int(vals['Bits'])
        self.__dither = vals['Dither']

    def get_vals(self) -> Dict[str, str]:
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
        self.__delay = float(vals['Delay'])

    @property
    def delay(self) -> float:
        return self.__delay

    def get_vals(self) -> Dict[str, str]:
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

    def get_filter(self) -> FilterOp:
        return DelayFilterOp(self.__delay)


class Divider(SingleFilter):
    TYPE = '20'

    def __init__(self, vals):
        super().__init__(vals, '---')
        self.__text = vals['Text'] if 'Text' in vals else ''

    @property
    def text(self):
        return self.__text

    def get_vals(self) -> Dict[str, str]:
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

    def get_vals(self) -> Dict[str, str]:
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

    def get_vals(self) -> Dict[str, str]:
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

    def get_filter(self) -> FilterOp:
        sos = iir.LinkwitzTransform(48000, self.__fz, self.__qz, self.__fp, self.__qp).get_sos()
        if sos:
            return SosFilterOp(sos)
        else:
            return NopFilterOp()


class LinkwitzRiley(SingleFilter):
    TYPE = '16'

    def __init__(self, vals):
        super().__init__(vals, 'LR')
        self.__freq = float(vals['Frequency'])

    def get_vals(self) -> Dict[str, str]:
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

    def get_vals(self) -> Dict[str, str]:
        return {}

    def __repr__(self):
        return f"{self.__class__.__name__}{self.print_disabled()}"


class MidSideEncoding(SingleFilter):
    TYPE = '18'

    def __init__(self, vals):
        super().__init__(vals, 'MS Encode')

    def get_vals(self) -> Dict[str, str]:
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

    def get_vals(self) -> Dict[str, str]:
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
        mix_type = MixType(self.__mode)
        if mix_type == MixType.MOVE or mix_type == MixType.MOVE:
            return f"{mix_type.name}\nto {get_channel_name(int(self.__destination))}"
        elif mix_type == MixType.SWAP:
            return f"{mix_type.name}\n{get_channel_name(int(self.__source))}-{get_channel_name(int(self.__destination))}"
        else:
            return super().short_desc()

    def get_filter(self) -> FilterOp:
        if self.mix_type == MixType.ADD:
            return AddFilterOp(GainFilterOp(self.__gain))
        elif self.mix_type == MixType.SUBTRACT:
            return SubtractFilterOp(GainFilterOp(self.__gain))
        else:
            # TODO only apply gain if the filter is provided to dst_idx
            return GainFilterOp(self.__gain)

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

    def get_vals(self) -> Dict[str, str]:
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

    def get_vals(self) -> Dict[str, str]:
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

    def get_filter(self) -> FilterOp:
        return InvertPolarityFilterOp()


class SubwooferLimiter(ChannelFilter):
    TYPE = '14'

    def __init__(self, vals):
        super().__init__(vals, 'SW Limiter')
        self.__level = float(vals['Level'])

    def get_vals(self) -> Dict[str, str]:
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
        self.__prefix: Filter = self.__make_divider(True)
        self.__filters = filters
        self.__suffix: Filter = self.__make_divider(False)

    def short_desc(self):
        return f"{self.short_name}"

    @property
    def filters(self) -> List[Filter]:
        return self.__filters

    def get_all_vals(self) -> List[Dict[str, str]]:
        all_filters: List[Filter] = [self.__prefix] + self.__filters + [self.__suffix]
        return [v for f in all_filters for v in f.get_all_vals()]

    def get_editable_filter(self) -> Optional[SOS]:
        editable_filters = [f.get_editable_filter() for f in self.__filters if f.get_editable_filter()]
        return CompleteFilter(fs=48000, filters=editable_filters, description=self.short_name, sort_by_id=True)

    def get_filter(self) -> FilterOp:
        return SosFilterOp(self.get_editable_filter().get_sos())

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


class XOFilterType(Enum):
    LPF = 1
    HPF = 2
    BPF = 3
    PASS = 4


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

    def __init__(self, metadata: str, routing: List[Filter], xo: List[XOFilter], sw_routing: List[Filter]):
        self.__metadata = metadata
        all_filters = routing if routing else [] + xo if xo else [] + sw_routing if sw_routing else []
        all_channels = set()
        for f in all_filters:
            if hasattr(f, 'channel_names'):
                for c in f.channel_names:
                    all_channels.add(c)
            elif isinstance(f, Mix):
                all_channels.add(get_channel_name(f.src_idx))
                all_channels.add(get_channel_name(f.dst_idx))
        self.__channel_names = [x for _, x in sorted(zip(all_channels, JRIVER_SHORT_CHANNELS)) if x]
        super().__init__(all_filters)

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
        return f"XOBM [{', '.join(self.__channel_names)}]"

    def metadata(self) -> str:
        return self.__metadata

    @classmethod
    def custom_type(cls) -> str:
        return 'XOBM'

    @classmethod
    def create(cls, data: str, child_filters: List[Filter]):
        phase = 0
        filters = [[], [], []]
        # filters arranged as mix then XO then more mix
        for f in child_filters:
            is_mix = isinstance(f, Mix)
            if is_mix and phase > 0:
                phase = 2
            elif not is_mix and phase == 0:
                phase = 1
            filters[phase].append(f)
        return CompoundRoutingFilter(data, *filters)


class CustomPassFilter(ComplexChannelFilter):

    def __init__(self, name, filters: List[Filter]):
        self.__name = name
        super().__init__(filters)

    def short_desc(self):
        tokens = self.__name.split('/')
        freq = float(tokens[3])
        if freq >= 1000:
            f = f"{freq/1000.0:.3g}k"
        else:
            f = f"{freq:g}"
        return f"{tokens[0]}{tokens[2]} {tokens[1]} {f}"

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
            f_type = FilterType(tokens[1])
            order = int(tokens[2])
            freq = float(tokens[3])
            q_scale = float(tokens[4])
            if tokens[0] == 'HP':
                return ComplexHighPass(f_type, order, 48000, freq, q_scale)
            elif tokens[0] == 'LP':
                return ComplexLowPass(f_type, order, 48000, freq, q_scale)
        raise ValueError(f"Unable to decode {self.__name}")

    def metadata(self) -> str:
        return self.__name

    @classmethod
    def custom_type(cls) -> str:
        return 'PASS'

    @classmethod
    def create(cls, data: str, child_filters: List[Filter]):
        return CustomPassFilter(data, child_filters)


def all_subclasses(cls):
    return set(cls.__subclasses__()).union([s for c in cls.__subclasses__() for s in all_subclasses(c)])


complex_filter_classes_by_type: Dict[str, Type[ComplexFilter]] = {c.custom_type(): c for c in all_subclasses(ComplexFilter)}
filter_classes_by_type: Dict[str, Type[Filter]]  = {c.TYPE: c for c in all_subclasses(Filter) if hasattr(c, 'TYPE')}


def create_peq(vals: Dict[str, str]) -> Filter:
    '''
    :param vals: the vals from the encoded xml format.
    :param channels: the available channel names.
    :return: a filter type.
    '''
    type_: Type[Filter] = filter_classes_by_type[vals['Type']]
    # noinspection PyTypeChecker
    return type_(vals)


def convert_filter_to_mc_dsp(filt: SOS, target_channels: str) -> Filter:
    '''
    :param filt: a filter.
    :param target_channels: the channels to output to.
    :return: a filter
    '''
    if isinstance(filt, BiquadWithQGain):
        if isinstance(filt, PeakingEQ):
            f_type = Peak
            q = filt.q
        else:
            q = q_to_s(filt.q, filt.gain)
            f_type = LowShelf if isinstance(filt, LS) else HighShelf
        return f_type({
            'Enabled': '1',
            'Slope': '12',
            'Q': f"{q:.4g}",
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
            vals = __make_high_order_mc_pass_filter(filt, pass_type.TYPE, pass_type.to_jriver_q, target_channels)
            return pass_type(vals)
        else:
            return __make_mc_custom_pass_filter(filt, target_channels)
    elif isinstance(filt, PassFilter):
        pass_type = HighPass if isinstance(filt, SecondOrder_HighPass) else LowPass
        return pass_type(__make_mc_pass_filter(filt, pass_type.TYPE, pass_type.to_jriver_q, target_channels))
    elif isinstance(filt, FirstOrder_LowPass) or isinstance(filt, FirstOrder_HighPass):
        pass_type = HighPass if isinstance(filt, FirstOrder_HighPass) else LowPass
        return pass_type(__make_mc_pass_filter(filt, pass_type.TYPE, pass_type.to_jriver_q, target_channels))
    else:
        raise ValueError(f"Unsupported filter type {filt}")


def __make_mc_custom_pass_filter(p_filter: CompoundPassFilter, target_channels: str) -> CustomPassFilter:
    pass_type = HighPass if isinstance(p_filter, ComplexHighPass) else LowPass
    mc_filts = [pass_type(__make_mc_pass_filter(f, pass_type.TYPE, pass_type.to_jriver_q, target_channels))
                for f in p_filter.filters]
    type_code = 'HP' if pass_type == HighPass else 'LP'
    encoded = f"{type_code}/{p_filter.type.value}/{p_filter.order}/{p_filter.freq:.7g}/{p_filter.q_scale:.4g}"
    return CustomPassFilter(encoded, mc_filts)


def __make_high_order_mc_pass_filter(f: CompoundPassFilter, filt_type: str, convert_q: Callable[[float], float],
                                     target_channels: str) -> Dict[str, str]:
    return {
        'Enabled': '1',
        'Slope': f"{f.order * 6}",
        'Type': filt_type,
        'Q': f"{convert_q(f.q_scale):.4g}",
        'Frequency': f"{f.freq:.7g}",
        'Gain': '0',
        'Channels': target_channels
    }


def __make_mc_pass_filter(f: Union[FirstOrder_LowPass, FirstOrder_HighPass, PassFilter],
                          filt_type: str, convert_q: Callable[[float], float], target_channels: str) -> Dict[str, str]:
    return {
        'Enabled': '1',
        'Slope': f"{f.order * 6}",
        'Type': filt_type,
        'Q': f"{convert_q(f.q):.4g}" if hasattr(f, 'q') else '1',
        'Frequency': f"{f.freq:.7g}",
        'Gain': '0',
        'Channels': target_channels
    }


class FilterOp(ABC):

    def __init__(self):
        self.node_id = None

    @abstractmethod
    def apply(self, input_signal: Signal) -> Signal:
        pass

    @property
    def ready(self):
        return True

    def __repr__(self):
        class_name = self.__class__.__name__
        return f"{class_name[:-8] if class_name.endswith('FilterOp') else class_name} [{self.node_id}]"


class FilterPipe:

    def __init__(self, op: FilterOp, parent: Optional[FilterPipe]):
        self.op = op
        self.next: Optional[FilterPipe] = None
        self.parent = parent

    @property
    def ready(self):
        return self.op.ready

    def __repr__(self):
        if self.next:
            return f"{self.op} -> {self.next}"
        else:
            return f"{self.op}"


def coalesce_ops(ops: List[FilterOp]) -> Optional[FilterPipe]:
    root: Optional[FilterPipe] = None
    tmp: Optional[FilterPipe] = None
    last_sos: Optional[SosFilterOp] = None
    for op in ops:
        if isinstance(op, SosFilterOp):
            if last_sos:
                last_sos.extend(op)
            else:
                last_sos = op
        else:
            if last_sos:
                if root is None:
                    tmp = FilterPipe(last_sos, None)
                    root = tmp
                else:
                    tmp.next = FilterPipe(last_sos, parent=tmp)
                    tmp = tmp.next
                last_sos = None
            if root is None:
                tmp = FilterPipe(op, None)
                root = tmp
            else:
                tmp.next = FilterPipe(op, tmp)
                tmp = tmp.next
    if last_sos:
        if root is None:
            root = FilterPipe(last_sos, None)
        else:
            tmp.next = FilterPipe(last_sos, tmp)
    return root


class NopFilterOp(FilterOp):

    def __init__(self):
        super().__init__()

    def apply(self, input_signal: Signal) -> Signal:
        return input_signal


class BranchFilterOp(FilterOp):

    def __init__(self, branch: FilterOp, node_id: str):
        super().__init__()
        self.node_id = node_id
        self.__branch = branch
        self.__source_signal: Optional[Signal] = None

    def apply(self, input_signal: Signal) -> Signal:
        self.__source_signal = input_signal
        return input_signal

    @property
    def source_signal(self) -> Optional[Signal]:
        return self.__source_signal

    def is_source_for(self, target: FilterOp) -> bool:
        return self.__branch == target


class SosFilterOp(FilterOp):

    def __init__(self, sos: List[List[float]]):
        super().__init__()
        self.__sos = sos

    def apply(self, input_signal: Signal) -> Signal:
        return input_signal.sosfilter(self.__sos)

    def extend(self, more_sos: SosFilterOp):
        self.__sos += more_sos.__sos


class DelayFilterOp(FilterOp):

    def __init__(self, delay_millis: float, fs: int = 48000):
        super().__init__()
        self.__shift_samples = int((delay_millis / 1000) / (1.0 / fs))

    def apply(self, input_signal: Signal) -> Signal:
        return input_signal.shift(self.__shift_samples)


class InvertPolarityFilterOp(FilterOp):

    def __init__(self):
        super().__init__()

    def apply(self, input_signal: Signal) -> Signal:
        return input_signal.invert()


class AddFilterOp(FilterOp):

    def __init__(self, gain: GainFilterOp = None):
        super().__init__()
        self.__gain = gain if gain else GainFilterOp()
        self.__inbound_signal: Optional[Signal] = None

    def accept(self, signal: Signal):
        if self.ready:
            raise ValueError(f"Attempting to reuse AddFilterOp")
        self.__inbound_signal = signal

    @property
    def ready(self):
        return self.__inbound_signal is not None

    def apply(self, input_signal: Signal) -> Signal:
        return input_signal.add(self.__gain.apply(self.__inbound_signal).samples)


class SubtractFilterOp(FilterOp):

    def __init__(self, gain: GainFilterOp = None):
        super().__init__()
        self.__gain = gain if gain else GainFilterOp()
        self.__inbound_signal: Optional[Signal] = None

    def accept(self, signal: Signal):
        if self.ready:
            raise ValueError(f"Attempting to reuse AddFilterOp")
        self.__inbound_signal = signal

    @property
    def ready(self):
        return self.__inbound_signal is not None

    def apply(self, input_signal: Signal) -> Signal:
        return input_signal.add(self.__gain.apply(self.__inbound_signal).samples)


class GainFilterOp(FilterOp):

    def __init__(self, gain_db: float = 0.0):
        super().__init__()
        self.__gain_db = gain_db

    def apply(self, input_signal: Signal) -> Signal:
        return input_signal.offset(self.__gain_db)


class Node:
    '''
    A single node in a filter chain, linked in both directions to parent(s) and children, if any.
    '''

    def __init__(self, rank: int, name: str, filt: Optional[Filter], channel: str):
        self.name = name
        self.rank = rank
        self.parent: Optional[Filter] = None
        self.__filt = filt
        self.__filter_op = filt.get_filter() if filt else NopFilterOp()
        if isinstance(filt, Mix) and isinstance(self.__filter_op, GainFilterOp):
            if filt.is_mine(get_channel_idx(channel)):
                # special case mix operations as gain offset should only be applied on the destination
                self.__filter_op = NopFilterOp()
        self.__filter_op.node_id = name
        self.channel = channel
        self.visited = False
        if self.filt is None and self.channel is None:
            raise ValueError('Must have either filter or channel')
        self.__down_edges: List[Node] = []
        self.__up_edges: List[Node] = []

    def add(self, node: Node, link_branch: bool = False):
        if node not in self.downstream:
            self.downstream.append(node)
            node.upstream.append(self)
            if link_branch is True:
                self.__filter_op = BranchFilterOp(node.filter_op, self.name)

    def has_editable_filter(self):
        return self.__filt and self.__filt.get_editable_filter() is not None

    @property
    def editable_filter(self) -> Optional[SOS]:
        '''
        :return: a beqd filter, if one can be provided by this filter.
        '''
        if self.__filt:
            return self.__filt.get_editable_filter()
        return None

    @property
    def editable_node_chain(self) -> Tuple[int, List[Node]]:
        '''
        A contiguous chain of filters which can be edited in the filter dialog as one unit.
        :return: idx of this node in the chain, the list of nodes.
        '''
        chain = []
        pos = 0
        if self.has_editable_filter():
            t = self
            while len(t.upstream) == 1 and t.upstream[0].has_editable_filter():
                chain.insert(0, t.upstream[0])
                t = t.upstream[0]
            pos = len(chain)
            chain.append(self)
            t = self
            while len(t.downstream) == 1 and t.downstream[0].has_editable_filter():
                chain.append(t.downstream[0])
                t = t.downstream[0]
        return pos, chain

    @property
    def filt(self) -> Optional[Filter]:
        return self.__filt

    @property
    def downstream(self) -> List[Node]:
        return self.__down_edges

    @property
    def upstream(self) -> List[Node]:
        return self.__up_edges

    def detach(self):
        '''
        Detaches this node from the upstream nodes.
        '''
        for u in self.upstream:
            u.downstream.remove(self)
        self.__up_edges = []

    @property
    def filter_op(self) -> FilterOp:
        return self.__filter_op

    def __repr__(self):
        return f"{self.name}{'' if self.__up_edges else ' - ROOT'} - {self.filt} -> {self.__down_edges}"

    @classmethod
    def swap(cls, n1: Node, n2: Node):
        '''
        Routes n1 to n2 downstream and vice versa.
        :param n1: first node.
        :param n2: second node.
        '''
        n2_target_upstream = [t for t in n1.upstream]
        n1_target_upstream = [t for t in n2.upstream]
        n1.detach()
        n2.detach()
        for to_attach in n1_target_upstream:
            to_attach.add(n1)
        for to_attach in n2_target_upstream:
            to_attach.add(n2)

    @classmethod
    def replace(cls, src: Node, dst: Node) -> List[Node]:
        '''
        Replaces the contents of dest with src.
        :param src: src.
        :param dst: dest.
        :returns detached downstream.
        '''
        # detach dst from its upstream node(s)
        dst.detach()
        # copy the downstream nodes from src and then remove them from src
        tmp_downstream = [d for d in src.downstream]
        src.downstream.clear()
        # attach the downstream nodes from dst to src
        for d in dst.downstream:
            d.detach()
            src.add(d)
        # remove those downstream nodes from dst
        dst.downstream.clear()
        # return any downstream nodes orphaned from the old src
        return tmp_downstream

    @classmethod
    def copy(cls, src: Node, dst: Node):
        '''
        Copies the contents of src to dst, same as replace but leaves the upstream intact on the src.
        :param src: src.
        :param dst: dest.
        '''
        # detach dst from its upstream node(s)
        dst.detach()
        # add dst a new downstream to src
        src.add(dst)


class FilterGraph:

    def __init__(self, stage: int, input_channels: List[str], output_channels: List[str], filts: List[Filter]):
        self.__editing: Optional[Tuple[str, List[Node]], int] = None
        self.__stage = stage
        self.__filts = filts
        self.__output_channels = output_channels
        self.__input_channels = input_channels
        self.__nodes_by_name: Dict[str, Node] = {}
        self.__nodes_by_channel: Dict[str, Node] = {}
        self.__filter_pipes_by_channel: Dict[str, FilterPipe] = {}
        self.__regen()

    @property
    def stage(self):
        return self.__stage

    def __regen(self):
        '''
        Regenerates the graph.
        '''
        self.__nodes_by_name = {}
        for f in self.__filts:
            f.reset()
        self.__nodes_by_channel = self.__generate_nodes()
        self.__filter_pipes_by_channel = self.__generate_filter_paths()

    def __generate_nodes(self) -> Dict[str, Node]:
        '''
        Parses the supplied filters into a linked set of nodes per channel.
        :return: the linked node per channel.
        '''
        return self.__prune(self.__link(self.__create_nodes()))

    def __prune(self, by_channel: Dict[str, Node]) -> Dict[str, Node]:
        '''
        Prunes non input channels so they don't start with an input.
        :param by_channel: the nodes by channel
        :return: the pruned nodes by channel.
        '''
        pruned = {}
        for c, n in by_channel.items():
            if c in self.__input_channels:
                pruned[c] = n
            else:
                if len(n.downstream) > 1:
                    logger.error(f"Unexpected multiple downstream for non input channel {n}")
                    pruned[c] = n
                elif n.downstream:
                    pruned[c] = n.downstream[0]
        return pruned

    @property
    def filter_pipes_by_channel(self) -> Dict[str, FilterPipe]:
        return self.__filter_pipes_by_channel

    def get_filter_at_node(self, node_name: str) -> Optional[Filter]:
        '''
        Locates the filter for the given node.
        :param node_name: the node to search for.
        :return: the filter, if any.
        '''
        node = self.get_node(node_name)
        if node and node.filt:
            return node.filt
        return None

    def get_filter_by_id(self, f_id: int) -> Optional[Filter]:
        '''
        Locates the filter with the given id.
        :param f_id: the filter id.
        :return: the filter, if any.
        '''
        return next((f for f in self.__filts if f.id == f_id), None)

    def get_node(self, node_name: str) -> Optional[Node]:
        '''
        Locates the named node.
        :param node_name: the node to search for.
        :return: the node, if any.
        '''
        return self.__nodes_by_name.get(node_name, None)

    def reorder(self, start: int, end: int, to: int) -> None:
        '''
        Moves the filters at indexes start to end to a new position & regenerates the graph.
        :param start: the starting index.
        :param end: the ending index.
        :param to: the position to move to.
        '''
        logger.info(f"Moved rows {start}:{end+1} to idx {to}")
        new_filters = self.filters[0: start] + self.filters[end+1:]
        to_insert = self.filters[start: end+1]
        for i, f in enumerate(to_insert):
            new_filters.insert(to + i - (1 if start < to else 0), f)
        logger.debug(f"Order: {[f for f in new_filters]}")
        self.__filts = new_filters
        self.__regen()

    @property
    def filters(self) -> List[Filter]:
        return self.__filts

    @property
    def nodes_by_channel(self) -> Dict[str, Node]:
        return self.__nodes_by_channel

    @property
    def input_channels(self):
        return self.__input_channels

    @property
    def output_channels(self):
        return self.__output_channels

    def __create_nodes(self) -> Dict[str, List[Node]]:
        '''
        transforms each filter into a node, one per channel the filter is applicable to.
        :return: nodes by channel name.
        '''
        # create a channel/filter grid
        by_channel: Dict[str, List[Node]] = {c: [Node(0, f"IN:{c}", None, c)] if c not in SHORT_USER_CHANNELS else []
                                             for c in self.__output_channels}
        i = 1
        for idx, f in enumerate(self.__filts):
            if isinstance(f, Sequence):
                for f1 in f:
                    if self.__process_filter(f1, by_channel, i):
                        for n in f1.nodes:
                            f.nodes.append(n)
                            n.parent = f
                        i += 1
            else:
                if self.__process_filter(f, by_channel, i):
                    i += 1
        # add output nodes
        for c, nodes in by_channel.items():
            if c not in SHORT_USER_CHANNELS:
                nodes.append(Node(i * 100, f"OUT:{c}", None, c))
        return by_channel

    def __process_filter(self, f: Filter, by_channel: Dict[str, List[Node]], i: int) -> bool:
        '''
        Converts the filter into a node on the target channel.
        :param f: the filter.
        :param by_channel: the store of nodes.
        :param i: the filter index.
        :return: true if a node was added.
        '''
        if not isinstance(f, Divider) and f.enabled:
            for channel_name, nodes in by_channel.items():
                channel_idx = get_channel_idx(channel_name)
                if f.is_mine(channel_idx):
                    # mix is added as a node to both channels
                    if isinstance(f, Mix):
                        dst_channel_name = get_channel_name(f.dst_idx)
                        by_channel[dst_channel_name].append(self.__make_node(i, dst_channel_name, f))
                        nodes.append(self.__make_node(i, channel_name, f))
                    else:
                        nodes.append(self.__make_node(i, channel_name, f))
            return True
        return False

    def __make_node(self, phase: int, channel_name: str, filt: Filter):
        node = Node(phase * 100, f"{channel_name}_{phase}00_{filt.short_name}", filt, channel_name)
        if node.name in self.__nodes_by_name:
            logger.warning(f"Duplicate node name detected in {channel_name}!!! {node.name}")
        self.__nodes_by_name[node.name] = node
        filt.nodes.append(node)
        return node

    def __link(self, by_channel: Dict[str, List[Node]]) -> Dict[str, Node]:
        '''
        Assembles edges between nodes. For all filters except Mix, this is just a link to the preceding. node.
        Copy, move and swap mixes change this relationship.
        :param by_channel: the nodes by channel
        :return: the linked nodes by channel.
        '''
        return self.__remix(self.__initialise_graph(by_channel))

    @staticmethod
    def __initialise_graph(by_channel):
        input_by_channel: Dict[str, Node] = {}
        for c, nodes in by_channel.items():
            upstream: Optional[Node] = None
            for node in nodes:
                if c not in input_by_channel:
                    input_by_channel[c] = node
                if upstream:
                    upstream.add(node)
                upstream = node
        return input_by_channel

    def __remix(self, by_channel: Dict[str, Node], orphaned_nodes: Dict[str, List[Node]] = None) -> Dict[str, Node]:
        '''
        Applies mix operations to the nodes.
        :param by_channel: the simply linked nodes.
        :param orphaned_nodes: any orphaned nodes.
        :return: the remixed nodes.
        '''
        if orphaned_nodes is None:
            orphaned_nodes = defaultdict(list)
        for c, node in by_channel.items():
            self.__remix_node(node, by_channel, orphaned_nodes)
        self.__remix_orphans(by_channel, orphaned_nodes)
        bad_nodes = [n for n in self.__collect_all_nodes(by_channel) if not self.__is_linked(n)]
        if bad_nodes:
            logger.warning(f"Found {len(bad_nodes)} badly linked nodes")
        return by_channel

    @staticmethod
    def __is_linked(node: Node):
        for u in node.upstream:
            if node not in u.downstream:
                logger.debug(f"Node not cross linked with upstream {u.name} - {node.name}")
                return False
        return True

    def __remix_orphans(self, by_channel, orphaned_nodes) -> None:
        '''
        Allows orphaned nodes to be considered in the remix.
        :param by_channel: the nodes by channel.
        :param orphaned_nodes: orphaned nodes.
        '''
        while True:
            if orphaned_nodes:
                c_to_remove = []
                for c, orphans in orphaned_nodes.items():
                    new_root = orphans.pop(0)
                    by_channel[f"{c}:{new_root.rank}"] = new_root
                    if not orphans:
                        c_to_remove.append(c)
                orphaned_nodes = {k: v for k, v in orphaned_nodes.items() if k not in c_to_remove}
                self.__remix(by_channel, orphaned_nodes=orphaned_nodes)
            else:
                break

    def __remix_node(self, node: Node, by_channel: Dict[str, Node], orphaned_nodes: Dict[str, List[Node]]) -> None:
        '''
        Applies a mix to a particular node.
        :param node: the node involved in the mix.
        :param by_channel: the entire graph so far.
        :param orphaned_nodes: unlinked node.
        '''
        downstream = [d for d in node.downstream]
        if not node.visited:
            node.visited = True
            channel_name = self.__extract_channel_name(node)
            f = node.filt
            if isinstance(f, Mix) and f.enabled:
                if f.mix_type == MixType.SWAP:
                    downstream = self.__swap_node(by_channel, channel_name, downstream, f, node, orphaned_nodes)
                elif f.mix_type == MixType.MOVE or f.mix_type == MixType.COPY:
                    downstream = self.__copy_or_replace_node(by_channel, channel_name, downstream, f, node,
                                                             orphaned_nodes)
                elif f.mix_type == MixType.ADD or f.mix_type == MixType.SUBTRACT:
                    self.__add_or_subtract_node(by_channel, f, node, orphaned_nodes)
        for d in downstream:
            self.__remix_node(d, by_channel, orphaned_nodes)

    @staticmethod
    def __extract_channel_name(node: Node) -> str:
        if '_' in node.name:
            sep = '_'
        elif ':' in node.name:
            sep = ':'
        else:
            raise ValueError(f"Unable to extract channel name from {node}")
        return node.name[0:node.name.index(sep)]

    def __add_or_subtract_node(self, by_channel: Dict[str, Node], f: Mix, node: Node,
                               orphaned_nodes: Dict[str, List[Node]]) -> None:
        '''
        Applies an add or subtract mix operation if the filter owns the supplied node.
        :param by_channel: the filter graph.
        :param f: the mix filter.
        :param node: the node to remix.
        :param orphaned_nodes: unlinked nodes.
        '''
        if f.is_mine(get_channel_idx(node.channel)):
            dst_channel_name = get_channel_name(f.dst_idx)
            try:
                dst_node = self.__find_owning_node_in_channel(by_channel[dst_channel_name], f, dst_channel_name)
                node.add(dst_node, link_branch=True)
            except ValueError:
                dst_node = self.__find_owning_node_in_orphans(dst_channel_name, f, orphaned_nodes)
                if dst_node:
                    dst_node.visited = True
                    node.add(dst_node, link_branch=True)
                else:
                    logger.debug(f"No match for {f} in {dst_channel_name}, presumed orphaned")
                    node.visited = False

    def __find_owning_node_in_orphans(self, dst_channel_name, f, orphaned_nodes):
        return next((n for n in orphaned_nodes.get(dst_channel_name, [])
                     if self.__owns_filter(n, f, dst_channel_name)), None)

    def __copy_or_replace_node(self, by_channel: Dict[str, Node], channel_name: str, downstream: List[Node], f: Mix,
                               node: Node, orphaned_nodes: Dict[str, List[Node]]) -> List[Node]:
        '''
        Applies a copy or replace operation to the supplied node.
        :param by_channel: the filter graph.
        :param channel_name: the channel being mixed.
        :param downstream: the downstream nodes.
        :param f: the mix filter.
        :param node: the node that is the subject of the mix.
        :param orphaned_nodes: unlinked nodes.
        :return: the nodes that are now downstream of this node.
        '''
        src_channel_name = get_channel_name(f.src_idx)
        if src_channel_name == channel_name and node.channel == channel_name:
            dst_channel_name = get_channel_name(f.dst_idx)
            try:
                dst_node = self.__find_owning_node_in_channel(by_channel[dst_channel_name], f, dst_channel_name)
                if f.mix_type == MixType.COPY:
                    Node.copy(node, dst_node)
                else:
                    new_downstream = Node.replace(node, dst_node)
                    if new_downstream:
                        if len(new_downstream) > 1:
                            txt = f"{channel_name} - {new_downstream}"
                            raise ValueError(f"Unexpected multiple downstream nodes on replace in channel {txt}")
                        if not new_downstream[0].name.startswith('OUT:'):
                            orphaned_nodes[channel_name].append(new_downstream[0])
                downstream = node.downstream
            except ValueError:
                logger.debug(f"No match for {f} in {dst_channel_name}, presumed orphaned")
                node.visited = False
        else:
            node.visited = False
        return downstream

    def __swap_node(self, by_channel: Dict[str, Node], channel_name: str, downstream: List[Node], f: Mix,
                    node: Node, orphaned_nodes: Dict[str, List[Node]]) -> List[Node]:
        '''
        Applies a swap operation to the supplied node.
        :param by_channel: the filter graph.
        :param channel_name: the channel being mixed.
        :param downstream: the downstream nodes.
        :param f: the mix filter.
        :param node: the node that is the subject of the mix.
        :param orphaned_nodes: unlinked nodes.
        :return: the nodes that are now downstream of this node.
        '''
        src_channel_name = get_channel_name(f.src_idx)
        if src_channel_name == channel_name:
            dst_channel_name = get_channel_name(f.dst_idx)
            swap_channel_name = src_channel_name if channel_name == dst_channel_name else dst_channel_name
            try:
                swap_node = self.__find_owning_node_in_channel(by_channel[swap_channel_name], f, swap_channel_name)
            except ValueError:
                swap_node = self.__find_owning_node_in_orphans(swap_channel_name, f, orphaned_nodes)
            if swap_node:
                Node.swap(node, swap_node)
                downstream = node.downstream
            else:
                logger.debug(f"No match for {f} in {swap_channel_name}, presumed orphaned")
                node.visited = False
        else:
            node.visited = False
        return downstream

    @staticmethod
    def __find_owning_node_in_channel(node: Node, match: Filter, owning_channel_name: str) -> Node:
        if FilterGraph.__owns_filter(node, match, owning_channel_name):
            return node
        for d in node.downstream:
            return FilterGraph.__find_owning_node_in_channel(d, match, owning_channel_name)
        raise ValueError(f"No match for '{match}' in '{node}'")

    @staticmethod
    def __owns_filter(node: Node, match: Filter, owning_channel_name: str) -> bool:
        # TODO compare by id?
        return node.filt and node.filt == match and node.channel == owning_channel_name

    def __generate_filter_paths(self) -> Dict[str, FilterPipe]:
        '''
        Converts the filter graph into a filter pipeline per channel.
        :return: a FilterPipe by output channel.
        '''
        output_nodes = self.__get_output_nodes()
        filter_pipes: Dict[str, FilterPipe] = {}
        for channel_name, output_node in output_nodes.items():
            parent = self.__get_parent(output_node)
            filters: List[FilterOp] = []
            while parent is not None:
                f = parent.filter_op
                if f:
                    filters.append(f)
                parent = self.__get_parent(parent)
            filter_pipes[channel_name] = coalesce_ops(filters[::-1])
        return filter_pipes

    @staticmethod
    def __get_parent(node: Node) -> Optional[Node]:
        '''
        Provides the actual parent node for this node. If the node has multiple upstreams then it must be accepting
        an inbound mix operation, in this case that upstream must be ignored.
        :param node: the node.
        :return: the parent node, if any.
        '''
        if node.upstream:
            if len(node.upstream) == 1:
                return node.upstream[0]
            else:
                if len(node.upstream) == 2:
                    parent = next((u for u in node.upstream
                                   if u.channel == node.channel and not FilterGraph.__is_inbound_mix(node, u)), None)
                    if parent is None:
                        raise ValueError(f"Unable to locate parent for {node.name} in -> {[n.name for n in node.upstream]}")
                    return parent
                else:
                    raise ValueError(f">2 upstream found! {node.name} -> {[n.name for n in node.upstream]}")
        else:
            return None

    @staticmethod
    def __is_inbound_mix(node: Node, upstream: Node):
        '''
        :param node: the node.
        :param upstream: the upstream node.
        :return: true if the upstream is a add or subtract mix operation landing in this node from a different channel.
        '''
        f = upstream.filt
        return isinstance(f, Mix) \
               and (f.mix_type == MixType.ADD or f.mix_type == MixType.SUBTRACT) \
               and upstream.channel != node.channel

    def __get_output_nodes(self) -> Dict[str, Node]:
        output = {}
        for node in self.__collect_all_nodes(self.__nodes_by_channel):
            if node.name.startswith('OUT:'):
                output[node.channel] = node
        return output

    @staticmethod
    def __collect_all_nodes(nodes_by_channel: Dict[str, Node]) -> List[Node]:
        nodes = []
        for root in nodes_by_channel.values():
            collect_nodes(root, nodes)
        return nodes

    def start_edit(self, channel: str, node_chain: List[Node], insert_at: int):
        self.__editing = (channel, node_chain, insert_at)

    def end_edit(self, new_filters: CompleteFilter):
        '''
        Replaces the nodes identified by the node_chain with the provided set of new filters.
        :param new_filters: the filters.
        '''
        if self.__editing:
            node_chain: List[Node]
            channel_name, node_chain, insert_at = self.__editing
            channel_idx = str(get_channel_idx(channel_name))
            old_filters: List[Tuple[Filter, str]] = [(n.filt, n.channel) for n in node_chain]
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
                new_filter = convert_filter_to_mc_dsp(f, channel_idx)
                if last_match < len(old_filters):
                    handled = False
                    for j in range(last_match + 1, len(old_filters)):
                        old_filter, filter_channel = old_filters[j]
                        old_filt_vals = old_filter.get_all_vals()
                        if pop_channels(new_filter.get_all_vals()) == pop_channels(old_filt_vals):
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
                        self.insert(new_filter, insert_at + offset, regen=False)
                        new_chain_filter = new_filter
                        offset += 1
                else:
                    must_regen = True
                    self.insert(new_filter, insert_at + offset, regen=False)
                    new_chain_filter = new_filter
                    offset += 1
            if last_match + 1 < len(old_filters):
                if self.__delete_filters(old_filters[last_match + 1:]):
                    must_regen = True
                    if not new_chain_filter:
                        new_chain_filter = old_filters[:last_match + 1][0][0]
            if must_regen:
                self.__regen()
                new_chain_node: Node = next((n for n in self.__collect_all_nodes(self.__nodes_by_channel)
                                             if n.channel == channel_name and n.filt and n.filt.id == new_chain_filter.id))
                editable_chain: List[Node] = new_chain_node.editable_node_chain[1]
                self.__editing = (channel_name, editable_chain, self.__get_filter_idx(editable_chain[0].filt) + 1)
            return must_regen

    def __get_filter_idx(self, to_match: Filter) -> int:
        return next((i for i, f in enumerate(self.__filts) if f.id == to_match.id))

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
                filt_to_delete.pop_channel(filt_channel_to_delete)
                if not filt_to_delete.channels:
                    self.__delete(filt_to_delete)
                    deleted += 1
            else:
                self.__delete(filt_to_delete)
                deleted += 1
        return deleted

    def __delete(self, filt_to_delete):
        self.__filts.pop(self.__get_filter_idx(filt_to_delete))

    def clear_filters(self) -> None:
        '''
        Removes all filters.
        '''
        self.__filts = []
        self.__regen()

    def delete_channel(self, channel_filter: ChannelFilter, channel: str) -> bool:
        '''
        Removes the channel from the filter, deleting the filter itself if it has no channels left.
        :param channel_filter: the filter.
        :param channel: the channel.
        :returns true if the filter is deleted.
        '''
        logger.debug(f"Deleting {channel} from {channel_filter}")
        channel_filter.pop_channel(channel)
        deleted_channel = False
        if not channel_filter.channels:
            self.__delete(channel_filter)
            deleted_channel = True
        self.__regen()
        return deleted_channel

    def delete(self, filters: List[Filter]) -> None:
        '''
        Removes the provided list of filters.
        :param filters: the filters to remove.
        '''
        ids_to_delete: List[int] = [f.id for f in filters]
        self.__filts = [f for f in self.__filts if f.id not in ids_to_delete]
        self.__regen()

    def insert(self, to_insert: Filter, at: int, regen=True) -> None:
        '''
        Inserts a filter a specified position.
        :param to_insert: the filter to insert.
        :param at: the position to insert at.
        :param regen: react to the insertion by regenerating the graph.
        '''
        if at < len(self.__filts):
            if len(self.__filts) == 1:
                filt_id = self.__filts[0].id / 2
            else:
                filt_id = ((self.__filts[at].id - self.__filts[at-1].id) / 2) + self.__filts[at-1].id
                if filt_id >= self.__filts[at].id:
                    raise ValueError(f"Unable to insert filter at {at}, attempting to insert {filt_id} before {self.__filts[at].id}")
        else:
            if len(self.__filts) == 0:
                filt_id = 2**24
            else:
                filt_id = self.__filts[-1].id + 2**24
        to_insert.id = filt_id
        self.__filts.insert(at, to_insert)
        if regen:
            self.__regen()

    def replace(self, old_filter: Filter, new_filter: Filter) -> bool:
        '''
        Replaces the specified filter.
        :param old_filter: the filter to replace.
        :param new_filter:  the replacement filter.
        :return: true if it was replaced.
        '''
        try:
            new_filter.id = old_filter.id
            self.__filts[self.__get_filter_idx(old_filter)] = new_filter
            self.__regen()
            return True
        except:
            return False

    def get_nodes_for_filter(self, filter: Filter) -> List[str]:
        '''
        Locates the node(s) occupied by this filter.
        :param filter: the filter.
        :return: the node names.
        '''
        return [n.name for n in self.__collect_all_nodes(self.nodes_by_channel) if n.filt and n.filt.id == filter.id]


def collect_nodes(node: Node, arr: List[Node]) -> List[Node]:
    if node not in arr:
        arr.append(node)
    for d in node.downstream:
        collect_nodes(d, arr)
    return arr
