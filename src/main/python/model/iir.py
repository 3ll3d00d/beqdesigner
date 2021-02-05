# from http://www.musicdsp.org/files/Audio-EQ-Cookbook.txt
import logging
import struct
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum
from functools import reduce
from typing import Optional, List, Callable

import math
import numpy as np
from scipy import signal

from model.xy import ComplexData

DEFAULT_Q = 1 / np.math.sqrt(2.0)

logger = logging.getLogger('iir')

import decimal

ctx = decimal.Context()
ctx.prec = 17

COMBINED = 'Combined'

# from https://gist.github.com/endolith/4982787
BESSEL_Q_FACTOR = {
    1: [-1],
    2: [0.57735026919],
    3: [-1, 0.691046625825],
    4: [0.805538281842, 0.521934581669],
    5: [-1, 0.916477373948, 0.563535620851],
    6: [1.02331395383, 0.611194546878, 0.510317824749],
    7: [-1, 1.12625754198, 0.660821389297, 0.5323556979],
    8: [1.22566942541, 0.710852074442, 0.559609164796, 0.505991069397],
    9: [-1, 1.32191158474, 0.76061100441, 0.589406099688, 0.519708624045],
    10: [1.41530886916, 0.809790964842, 0.620470155556, 0.537552151325, 0.503912727276],
    11: [-1, 1.50614319627, 0.858254347397, 0.652129790265, 0.557757625275, 0.513291150482],
    12: [1.59465693507, 0.905947107025, 0.684008068137, 0.579367238641, 0.525936202016, 0.502755558204],
    13: [-1, 1.68105842736, 0.952858075613, 0.715884117238, 0.60182181594, 0.540638359678, 0.509578259933],
    14: [1.76552743493, 0.998998442993, 0.747625068271, 0.624777082395, 0.556680772868, 0.519027293158, 0.502045428643],
    15: [-1, 1.84821988785, 1.04439091113, 0.779150095987, 0.648012471324, 0.573614183126, 0.530242036742,
         0.507234085654],
    16: [1.9292718407, 1.08906376917, 0.810410302962, 0.671382379377, 0.591144659703, 0.542678365981, 0.514570953471,
         0.501578400482],
    17: [-1, 2.0088027125, 1.13304758938, 0.841376937749, 0.694788531655, 0.609073284112, 0.555976702005,
         0.523423635236, 0.505658277957],
    18: [2.08691792612, 1.17637337045, 0.872034231424, 0.718163551101, 0.627261751983, 0.569890924765, 0.533371782078,
         0.511523796759, 0.50125489338],
    19: [-1, 2.16371105964, 1.21907150269, 0.902374908665, 0.741460774758, 0.645611852961, 0.584247604949,
         0.544125898196, 0.518697311353, 0.504547600962],
    20: [2.23926560629, 1.26117120993, 0.932397288146, 0.764647810579, 0.664052481472, 0.598921924986, 0.555480327396,
         0.526848630061, 0.509345928377, 0.501021580965],
    21: [-1, 2.31365642136, 1.30270026567, 0.96210341835, 0.787702113969, 0.682531805651, 0.613821586135,
         0.567286339654, 0.535741766356, 0.515281087097, 0.503735024056],
    22: [2.38695091667, 1.34368488961, 0.991497755204, 0.81060830488, 0.701011199665, 0.628878390935, 0.57943181849,
         0.545207253735, 0.52208637596, 0.507736060535, 0.500847111042],
    23: [-1, 2.45921005855, 1.38414971551, 1.02058630948, 0.833356335852, 0.71946106813, 0.64404152916, 0.591833716479,
         0.555111517796, 0.529578662133, 0.512723802741, 0.503124630056],
    24: [2.53048919562, 1.42411783481, 1.04937620183, 0.85593899901, 0.737862159044, 0.659265671705, 0.604435823473,
         0.565352679646, 0.537608804383, 0.51849505465, 0.506508536474, 0.500715908905],
    25: [-1, 2.60083876344, 1.46361085888, 1.07787504693, 0.878352946895, 0.756194508791, 0.674533119177,
         0.617161247256, 0.575889371424, 0.54604850857, 0.524878372745, 0.510789585775, 0.502642143876]
}

BESSEL_PHASE_MATCHED_F_MULTIPLIER = {
    1: [1.0],
    2: [1.0],
    3: [0.941600026533, 1.03054454544],
    4: [1.05881751607, 0.944449808226],
    5: [0.926442077388, 1.08249898167, 0.959761595159],
    6: [1.10221694805, 0.977488555538, 0.928156550439],
    7: [0.919487155649, 1.11880560415, 0.994847495138, 0.936949152329],
    8: [1.13294518316, 1.01102810214, 0.948341760923, 0.920583104484],
    9: [0.915495779751, 1.14514968183, 1.02585472504, 0.960498668791, 0.926247266902],
    10: [1.1558037036, 1.03936894925, 0.972611094341, 0.934100034374, 0.916249124617],
    11: [0.912906724455, 1.16519741334, 1.05168282959, 0.984316740934, 0.942949951193, 0.920193132602],
    12: [1.17355271619, 1.06292406317, 0.99546178686, 0.952166527388, 0.92591429605, 0.913454385093],
    13: [0.91109146642, 1.18104182776, 1.07321545484, 1.00599396165, 0.961405619782, 0.932611355794, 0.916355696498],
    14: [1.18780032771, 1.08266791426, 1.0159112847, 0.970477661528, 0.939810654318, 0.920703208467, 0.911506866227],
    15: [0.909748233727, 1.19393639282, 1.09137890831, 1.02523633131, 0.97927996807, 0.947224181054, 0.925936706555,
         0.91372962678],
    16: [1.19953740587, 1.09943305993, 1.03400291299, 0.987760087301, 0.954673832805, 0.93169889496, 0.917142770586,
         0.910073839264],
    17: [0.908714103725, 1.20467475213, 1.10690349924, 1.04224907075, 0.995895132988, 0.962048803251, 0.937756408953,
         0.921340810203, 0.911830715155],
    18: [1.20940734995, 1.11385337718, 1.05001339566, 1.00367972938, 0.969280631132, 0.943954185964, 0.926050333934,
         0.91458037566, 0.908976081436],
    19: [0.907893623592, 1.21378428545, 1.12033730486, 1.05733313013, 1.01111885678, 0.97632792811, 0.950188250195,
         0.931083047917, 0.918020722296, 0.910399240009],
    20: [1.21784680466, 1.1264026302, 1.0642432684, 1.0182234301, 0.983167015494, 0.956388509221, 0.936307259119,
         0.921939345182, 0.912660567121, 0.90810906098],
    21: [0.907227918651, 1.22162983803, 1.13209053887, 1.0707761723, 1.02500765809, 0.989785773052, 0.962507744041,
         0.941630539095, 0.926182679425, 0.915530785895, 0.909284134383],
    22: [1.22516318078, 1.13743698469, 1.07696149672, 1.03148734658, 0.996179868375, 0.968513835739, 0.946988833876,
         0.930637530698, 0.918842345489, 0.911176680853, 0.90740679425],
    23: [0.906504834665, 1.22847241656, 1.14247346478, 1.08282633643, 1.03767860165, 1.00235036472, 0.974386503777,
         0.952333544924, 0.93521970212, 0.922499715897, 0.913522360451, 0.908537315966],
    24: [1.23157964663, 1.14722769588, 1.0883951254, 1.04359863738, 1.00829752657, 0.980122054682, 0.957614328211,
         0.939902288094, 0.926326979303, 0.916382125274, 0.91007413458, 0.906792423356],
    25: [0.90735557785, 1.23450407249, 1.15172412685, 1.09369031049, 1.04926215576, 1.01403218959, 0.985701250074,
         0.962823028773, 0.944627742698, 0.930311087183, 0.919369633361, 0.912650977333, 0.906702031339],
}

BESSEL_3DB_F_MULTIPLIER = {
    1: [1.0],
    2: [1.27201964951],
    3: [1.32267579991, 1.44761713315],
    4: [1.60335751622, 1.43017155999],
    5: [1.50231627145, 1.75537777664, 1.5563471223],
    6: [1.9047076123, 1.68916826762, 1.60391912877],
    7: [1.68436817927, 2.04949090027, 1.82241747886, 1.71635604487],
    8: [2.18872623053, 1.95319575902, 1.8320926012, 1.77846591177],
    9: [1.85660050123, 2.32233235836, 2.08040543586, 1.94786513423, 1.87840422428],
    10: [2.45062684305, 2.20375262593, 2.06220731793, 1.98055310881, 1.94270419166],
    11: [2.01670147346, 2.57403662106, 2.32327165002, 2.17445328051, 2.08306994025, 2.03279787154],
    12: [2.69298925084, 2.43912611431, 2.28431825401, 2.18496722634, 2.12472538477, 2.09613322542],
    13: [2.16608270526, 2.80787865058, 2.55152585818, 2.39170950692, 2.28570254744, 2.21724536226, 2.178598197],
    14: [2.91905714471, 2.66069088948, 2.49663434571, 2.38497976939, 2.30961462222, 2.26265746534, 2.24005716132],
    15: [2.30637004989, 3.02683647605, 2.76683540993, 2.5991524698, 2.48264509354, 2.4013780964, 2.34741064497,
         2.31646357396],
    16: [3.13149167404, 2.87016099416, 2.69935018044, 2.57862945683, 2.49225505119, 2.43227707449, 2.39427710712,
         2.37582307687],
    17: [2.43892718901, 3.23326555056, 2.97085412104, 2.7973260082, 2.67291517463, 2.58207391498, 2.51687477182,
         2.47281641513, 2.44729196328],
    18: [3.33237300564, 3.06908580184, 2.89318259511, 2.76551588399, 2.67073340527, 2.60094950474, 2.55161764546,
         2.52001358804, 2.50457164552],
    19: [2.56484755551, 3.42900487079, 3.16501220302, 2.98702207363, 2.85646430456, 2.75817808906, 2.68433211497,
         2.630358907, 2.59345714553, 2.57192605454],
    20: [3.52333123464, 3.25877569704, 3.07894353744, 2.94580435024, 2.84438325189, 2.76691082498, 2.70881411245,
         2.66724655259, 2.64040228249, 2.62723439989],
    21: [2.68500843719, 3.61550427934, 3.35050607023, 3.16904164639, 3.03358632774, 2.92934454178, 2.84861318802,
         2.78682554869, 2.74110646014, 2.70958138974, 2.69109396043],
    22: [3.70566068548, 3.44032173223, 3.2574059854, 3.11986367838, 3.01307175388, 2.92939234605, 2.86428726094,
         2.81483068055, 2.77915465405, 2.75596888377, 2.74456638588],
    23: [2.79958271812, 3.79392366765, 3.52833084348, 3.34412104851, 3.204690112, 3.09558498892, 3.00922346183,
         2.94111672911, 2.88826359835, 2.84898013042, 2.82125512753, 2.80585968356],
    24: [3.88040469682, 3.61463243697, 3.4292654707, 3.28812274966, 3.17689762788, 3.08812364257, 3.01720732972,
         2.96140104561, 2.91862858495, 2.88729479473, 2.8674198668, 2.8570800015],
    25: [2.91440986162, 3.96520496584, 3.69931726336, 3.51291368484, 3.37021124774, 3.25705322752, 3.16605475731,
         3.09257032034, 3.03412738742, 2.98814254637, 2.9529987927, 2.9314185899, 2.91231068194],
}


def float_to_str(f, to_hex=False, minidsp_style=False, fixed_point=False):
    """
    Convert the given float to a string without scientific notation or in the correct hex format.
    """
    if to_hex is True:
        return float_to_hex(f, minidsp_style, fixed_point)
    else:
        d1 = ctx.create_decimal(repr(f))
        return format(d1, 'f')


def float_to_hex(f, minidsp_style, fixed_point):
    '''
    Converts a floating number to its 32bit IEEE 754 hex representation if fixed_point is None else converts to
    Q5.23 format.
    :param f: the float.
    :param minidsp_style: if true, don't print 0x prefix.
    :return: the hex value.
    '''
    if fixed_point is True:
        # use 2s complement for negative values
        # the 10x10HD plays no sound if 0x prefix and causes feedback if not 0 padded to 8 chars.
        val = int(-f * (2 ** 23)) ^ 0xFFFFFFF if f < 0 else int(f * (2 ** 23))
        return f"{val:08x}"
    else:
        value = struct.unpack('<I', struct.pack('<f', f))[0]
        return f"{value:{'#' if minidsp_style is True else ''}010x}"


class SOS(ABC):

    def __init__(self, f_id=-1):
        self.__id = f_id

    @property
    def id(self):
        return self.__id

    @id.setter
    def id(self, id):
        self.__id = id

    @abstractmethod
    def get_sos(self) -> Optional[List[List[float]]]:
        pass


class Biquad(SOS):
    def __init__(self, fs, f_id=-1):
        super().__init__(f_id=f_id)
        self.fs = fs
        self.a, self.b = self._compute_coeffs()
        self.__transfer_function = None

    def __eq__(self, o: object) -> bool:
        equal = self.__class__.__name__ == o.__class__.__name__
        equal &= self.fs == o.fs
        return equal

    def __repr__(self):
        return self.description

    @property
    def description(self):
        description = ''
        if hasattr(self, 'display_name'):
            description += self.display_name
        return description

    @property
    def filter_type(self):
        return self.__class__.__name__

    def __len__(self):
        return 1

    @abstractmethod
    def _compute_coeffs(self):
        pass

    @abstractmethod
    def sort_key(self):
        pass

    def get_transfer_function(self) -> ComplexData:
        '''
        Computes the transfer function of the filter.
        :return: the transfer function.
        '''
        if self.__transfer_function is None:
            from model.preferences import X_RESOLUTION
            import time
            start = time.time()
            w, h = signal.freqz(b=self.b, a=self.a, worN=X_RESOLUTION)
            end = time.time()
            logger.debug(f"freqz in {round((end - start) * 1000, 3)}ms")
            f = w * self.fs / (2 * np.pi)
            self.__transfer_function = ComplexData(f"{self}", f, h)
        return self.__transfer_function

    def get_impulse_response(self, dt=None, n=None):
        '''
        Converts the 2nd order section to a transfer function (b, a) and then computes the IR of the discrete time
        system.
        :param dt: the time delta.
        :param n: the no of samples to output, defaults to 1s.
        :return: t, y
        '''
        t, y = signal.dimpulse(signal.dlti(*signal.sos2tf(np.array(self.get_sos())),
                                           dt=1 / self.fs if dt is None else dt),
                               n=self.fs if n is None else n)
        return t, y

    def format_biquads(self, minidsp_style, separator=',\n', show_index=True, to_hex=False, fixed_point=False):
        ''' Creates a biquad report '''
        kwargs = {'to_hex': to_hex, 'minidsp_style': minidsp_style, 'fixed_point': fixed_point}
        a = separator.join(
            [f"{self.__format_index('a', idx, show_index)}{float_to_str(-x if minidsp_style else x, **kwargs)}"
             for idx, x in enumerate(self.a) if idx != 0 or minidsp_style is False])
        b = separator.join([f"{self.__format_index('b', idx, show_index)}{float_to_str(x, **kwargs)}"
                            for idx, x in enumerate(self.b)])
        return [f"{b}{separator}{a}"]

    @staticmethod
    def __format_index(prefix, idx, show_index):
        if show_index:
            return f"{prefix}{idx}="
        else:
            return ''

    def get_sos(self) -> Optional[List[List[float]]]:
        return [np.concatenate((self.b, self.a)).tolist()]


class LinkwitzTransform(Biquad):

    def __init__(self, fs, f0, q0, fp, qp, f_id=-1):
        '''
        B11 = (f0 + fp) / 2.0
        B26 = (2.0 * pi * f0) * (2.0 * pi * f0)
        B27 = (2.0 * pi * f0) / q0
        B30 = (2.0 * pi * fp) * (2.0 * pi * fp)
        B31 = (2.0 * pi * fp) / qp
        B34 = (2.0 * pi * B11) / tan(pi * B11 / fs)
        B35 = B30 + (B34 * B31) + (B34 * B34)

        a0 = 1
        a1 = (2.0 * (B30 - (B34 * B34)) / B35)
        a2 = ((B30 - B34 * B31 + (B34 * B34)) / B35)
        b0 = (B26 + B34 * B27 + (B34 * B34)) / B35
        b1 = 2.0 * (B26 - (B34 * B34)) / B35
        b2 = (B26 - B34 * B27 + (B34 * B34)) / B35
        '''
        self.f0 = f0
        self.fp = fp
        self.q0 = q0
        self.qp = qp
        super().__init__(fs, f_id=f_id)

    @property
    def filter_type(self):
        return 'Linkwitz Transform'

    @property
    def display_name(self):
        return 'LT'

    def _compute_coeffs(self):
        b11 = (self.f0 + self.fp) / 2.0
        two_pi = 2.0 * math.pi
        b26 = (two_pi * self.f0) ** 2.0
        b27 = (two_pi * self.f0) / self.q0
        b30 = (two_pi * self.fp) ** 2.0
        b31 = (two_pi * self.fp) / self.qp
        b34 = (two_pi * b11) / math.tan(math.pi * b11 / self.fs)
        b35 = b30 + (b34 * b31) + (b34 * b34)
        a0 = 1.0
        a1 = 2.0 * (b30 - (b34 * b34)) / b35
        a2 = (b30 - b34 * b31 + (b34 * b34)) / b35
        b0 = (b26 + b34 * b27 + (b34 * b34)) / b35
        b1 = 2.0 * (b26 - (b34 * b34)) / b35
        b2 = (b26 - b34 * b27 + (b34 * b34)) / b35
        return np.array([a0, a1, a2]), np.array([b0, b1, b2])

    def resample(self, new_fs):
        '''
        Creates a filter at the specified fs.
        :param new_fs: the new fs.
        :return: the new filter.
        '''
        return LinkwitzTransform(new_fs, self.f0, self.fp, self.q0, self.qp, f_id=self.id)

    def __eq__(self, o: object) -> bool:
        return super().__eq__(o) and self.f0 == o.f0 and self.fp == o.fp and self.q0 == o.q0 and self.qp == o.qp

    @property
    def description(self):
        return super().description + f" {self.f0}/{self.q0} -> {self.fp}/{self.qp}"

    def sort_key(self):
        return f"{self.f0:05}{self.fp:05}{self.filter_type}"

    def to_json(self):
        return {
            '_type': self.__class__.__name__,
            'fs': self.fs,
            'f0': self.f0,
            'fp': self.fp,
            'q0': self.q0,
            'qp': self.qp
        }


class Gain(Biquad):
    def __init__(self, fs, gain, f_id=-1):
        self.gain = gain
        super().__init__(fs, f_id=f_id)

    @property
    def filter_type(self):
        return 'Gain'

    @property
    def display_name(self):
        return 'Gain'

    def _compute_coeffs(self):
        return np.array([1.0, 0.0, 0.0]), np.array([10.0 ** (self.gain / 20.0), 0.0, 0.0])

    def resample(self, new_fs):
        '''
        Creates a filter at the specified fs.
        :param new_fs: the new fs.
        :return: the new filter.
        '''
        return Gain(new_fs, self.gain, f_id=self.id)

    def sort_key(self):
        return f"00000{self.gain:05}{self.filter_type}"

    def to_json(self):
        return {
            '_type': self.__class__.__name__,
            'fs': self.fs,
            'gain': self.gain
        }


class BiquadWithQ(Biquad):
    def __init__(self, fs, freq, q, f_id=-1):
        self.freq = round(float(freq), 2)
        self.q = round(float(q), 4)
        self.w0 = 2.0 * math.pi * freq / fs
        self.cos_w0 = math.cos(self.w0)
        self.sin_w0 = math.sin(self.w0)
        self.alpha = self.sin_w0 / (2.0 * self.q)
        super().__init__(fs, f_id=f_id)

    def __eq__(self, o: object) -> bool:
        return super().__eq__(o) and self.freq == o.freq

    @property
    def description(self):
        return super().description + f" {self.freq}/{self.q}"

    def sort_key(self):
        return f"{self.freq:05}00000{self.filter_type}"


class Passthrough(Gain):
    def __init__(self, fs=1000, f_id=-1):
        super().__init__(fs, 0, f_id=f_id)

    @property
    def display_name(self):
        return 'Passthrough'

    @property
    def description(self):
        return 'Passthrough'

    def sort_key(self):
        return "ZZZZZZZZZZZZZZ"

    def resample(self, new_fs):
        return Passthrough(fs=new_fs, f_id=self.id)

    def to_json(self):
        return {
            '_type': self.__class__.__name__,
            'fs': self.fs
        }


class BiquadWithQGain(BiquadWithQ):
    def __init__(self, fs, freq, q, gain, f_id=-1):
        self.gain = round(float(gain), 3)
        super().__init__(fs, freq, q, f_id=f_id)

    def __eq__(self, o: object) -> bool:
        return super().__eq__(o) and self.gain == o.gain

    @property
    def description(self):
        return super().description + f"/{self.gain}dB"

    def sort_key(self):
        return f"{self.freq:05}{self.gain:05}{self.filter_type}"


class PeakingEQ(BiquadWithQGain):
    '''
    H(s) = (s^2 + s*(A/Q) + 1) / (s^2 + s/(A*Q) + 1)

            b0 =   1 + alpha*A
            b1 =  -2*cos(w0)
            b2 =   1 - alpha*A
            a0 =   1 + alpha/A
            a1 =  -2*cos(w0)
            a2 =   1 - alpha/A
    '''

    def __init__(self, fs, freq, q, gain, f_id=-1):
        super().__init__(fs, freq, q, gain, f_id=f_id)

    @property
    def filter_type(self):
        return 'PEQ'

    @property
    def display_name(self):
        return 'PEQ'

    def _compute_coeffs(self):
        A = 10.0 ** (self.gain / 40.0)
        a = np.array([1.0 + self.alpha / A, -2.0 * self.cos_w0, 1.0 - self.alpha / A], dtype=np.float64)
        b = np.array([1.0 + self.alpha * A, -2.0 * self.cos_w0, 1.0 - self.alpha * A], dtype=np.float64)
        return a / a[0], b / a[0]

    def resample(self, new_fs):
        '''
        Creates a filter at the specified fs.
        :param new_fs: the new fs.
        :return: the new filter.
        '''
        return PeakingEQ(new_fs, self.freq, self.q, self.gain, f_id=self.id)

    def to_json(self):
        return {
            '_type': self.__class__.__name__,
            'fs': self.fs,
            'fc': self.freq,
            'q': self.q,
            'gain': self.gain
        }


def q_to_s(q, gain):
    '''
    translates Q to S for a shelf filter.
    :param q: the Q.
    :param gain: the gain.
    :return: the S.
    '''
    return 1.0 / ((((1.0 / q) ** 2.0 - 2.0) / (
            (10.0 ** (gain / 40.0)) + 1.0 / (10.0 ** (gain / 40.0)))) + 1.0)


def s_to_q(s, gain):
    '''
    translates S to Q for a shelf filter.
    :param s: the S.
    :param gain: the gain.
    :return: the Q.
    '''
    A = 10.0 ** (gain / 40.0)
    return 1.0 / math.sqrt(((A + 1.0 / A) * (1.0 / s - 1.0)) + 2.0)


def max_permitted_s(gain):
    '''
    Calculates the max S for the specified gain where max S = the S that results in a Q of 20.
    :param gain: the gain.
    :return: the max S.
    '''
    A = 10.0 ** (gain / 40.0)
    X = A + (1.0 / A)
    # -1.9975 = (1/Q*1/Q) + 2 (i.e. comes from rearranging the s to q equation to solve for S
    max_s = 1 / (-1.9975 / X + 1)
    return max_s


class Shelf(BiquadWithQGain):
    def __init__(self, fs, freq, q, gain, count, f_id=-1):
        self.A = 10.0 ** (gain / 40.0)
        super().__init__(fs, freq, q, gain, f_id=f_id)
        self.count = count
        self.__cached_cascade = None

    def q_to_s(self):
        '''
        :return: the filter Q as S
        '''
        return q_to_s(self.q, self.gain)

    def __len__(self):
        return self.count

    def flatten(self):
        '''
        :return: an iterable of length count of this shelf where each shelf has count=1
        '''
        if self.count == 1:
            return [self]
        else:
            return [self.__class__(self.fs, self.freq, self.q, self.gain, 1)] * self.count

    def get_transfer_function(self):
        single = super().get_transfer_function()
        if self.count == 1:
            return single
        elif self.count > 1:
            if self.__cached_cascade is None:
                self.__cached_cascade = get_cascade_transfer_function(self.__repr__(), [single] * self.count)
            return self.__cached_cascade
        else:
            raise ValueError('Shelf must have non zero count')

    def format_biquads(self, minidsp_style, separator=',\n', show_index=True, to_hex=False, fixed_point=False):
        single = super().format_biquads(minidsp_style, separator=separator, show_index=show_index,
                                        to_hex=to_hex, fixed_point=fixed_point)
        if self.count == 1:
            return single
        elif self.count > 1:
            return single * self.count
        else:
            raise ValueError('Shelf must have non zero count')

    def get_sos(self):
        return super().get_sos() * self.count

    def to_json(self):
        return {
            '_type': self.__class__.__name__,
            'fs': self.fs,
            'fc': self.freq,
            'q': self.q,
            'gain': self.gain,
            'count': self.count
        }

    @property
    def description(self):
        if self.count > 1:
            return super().description + f" x{self.count}"
        else:
            return super().description


class LowShelf(Shelf):
    '''
    lowShelf: H(s) = A * (s^2 + (sqrt(A)/Q)*s + A)/(A*s^2 + (sqrt(A)/Q)*s + 1)

            b0 =    A*( (A+1) - (A-1)*cos(w0) + 2*sqrt(A)*alpha )
            b1 =  2*A*( (A-1) - (A+1)*cos(w0)                   )
            b2 =    A*( (A+1) - (A-1)*cos(w0) - 2*sqrt(A)*alpha )
            a0 =        (A+1) + (A-1)*cos(w0) + 2*sqrt(A)*alpha
            a1 =   -2*( (A-1) + (A+1)*cos(w0)                   )
            a2 =        (A+1) + (A-1)*cos(w0) - 2*sqrt(A)*alpha
    '''

    def __init__(self, fs, freq, q, gain, count=1, f_id=-1):
        super().__init__(fs, freq, q, gain, count, f_id=f_id)

    @property
    def filter_type(self):
        return 'LS'

    @property
    def display_name(self):
        return 'Low Shelf'

    def _compute_coeffs(self):
        A = 10.0 ** (self.gain / 40.0)
        a = np.array([
            (A + 1) + ((A - 1) * self.cos_w0) + (2.0 * math.sqrt(A) * self.alpha),
            -2.0 * ((A - 1) + ((A + 1) * self.cos_w0)),
            (A + 1) + ((A - 1) * self.cos_w0) - (2.0 * math.sqrt(A) * self.alpha)
        ], dtype=np.float64)
        b = np.array([
            A * ((A + 1) - ((A - 1) * self.cos_w0) + (2.0 * math.sqrt(A) * self.alpha)),
            2.0 * A * ((A - 1) - ((A + 1) * self.cos_w0)),
            A * ((A + 1) - ((A - 1) * self.cos_w0) - (2 * math.sqrt(A) * self.alpha))
        ], dtype=np.float64)
        return a / a[0], b / a[0]

    def resample(self, new_fs):
        '''
        Creates a filter at the specified fs.
        :param new_fs: the new fs.
        :return: the new filter.
        '''
        return LowShelf(new_fs, self.freq, self.q, self.gain, self.count, f_id=self.id)


class HighShelf(Shelf):
    '''
    highShelf: H(s) = A * (A*s^2 + (sqrt(A)/Q)*s + 1)/(s^2 + (sqrt(A)/Q)*s + A)

                b0 =    A*( (A+1) + (A-1)*cos(w0) + 2*sqrt(A)*alpha )
                b1 = -2*A*( (A-1) + (A+1)*cos(w0)                   )
                b2 =    A*( (A+1) + (A-1)*cos(w0) - 2*sqrt(A)*alpha )
                a0 =        (A+1) - (A-1)*cos(w0) + 2*sqrt(A)*alpha
                a1 =    2*( (A-1) - (A+1)*cos(w0)                   )
                a2 =        (A+1) - (A-1)*cos(w0) - 2*sqrt(A)*alpha

    '''

    def __init__(self, fs, freq, q, gain, count=1, f_id=-1):
        super().__init__(fs, freq, q, gain, count, f_id=f_id)

    def __eq__(self, o: object) -> bool:
        return super().__eq__(o) and self.count == o.count

    @property
    def filter_type(self):
        return 'HS'

    @property
    def display_name(self):
        return 'High Shelf'

    def _compute_coeffs(self):
        A = self.A
        cos_w0 = self.cos_w0
        alpha = self.alpha
        a = np.array([
            (A + 1) - ((A - 1) * cos_w0) + (2.0 * math.sqrt(A) * alpha),
            2.0 * ((A - 1) - ((A + 1) * cos_w0)),
            (A + 1) - ((A - 1) * cos_w0) - (2.0 * math.sqrt(A) * alpha)
        ], dtype=np.float64)
        b = np.array([
            A * ((A + 1) + ((A - 1) * cos_w0) + (2.0 * math.sqrt(A) * alpha)),
            -2.0 * A * ((A - 1) + ((A + 1) * cos_w0)),
            A * ((A + 1) + ((A - 1) * cos_w0) - (2.0 * math.sqrt(A) * alpha))
        ], dtype=np.float64)
        return a / a[0], b / a[0]

    def resample(self, new_fs):
        '''
        Creates a filter at the specified fs.
        :param new_fs: the new fs.
        :return: the new filter.
        '''
        return HighShelf(new_fs, self.freq, self.q, self.gain, self.count, f_id=self.id)


class FirstOrder_LowPass(Biquad):
    '''
    A one pole low pass filter.
    '''

    def __init__(self, fs, freq, f_id=-1):
        self.freq = round(float(freq), 2)
        self.order = 1
        super().__init__(fs, f_id=f_id)

    def __eq__(self, o: object) -> bool:
        return super().__eq__(o) and self.order == o.order and self.freq == o.freq

    @property
    def filter_type(self):
        return 'LPF1'

    @property
    def display_name(self):
        return 'LPF1'

    def _compute_coeffs(self):
        a1 = math.exp(-2.0 * math.pi * (self.freq / self.fs))
        b0 = 1.0 - a1
        a = np.array([1.0, -a1, 0.0], dtype=np.float64)
        b = np.array([b0, 0.0, 0.0])
        return a, b

    def resample(self, new_fs):
        '''
        Creates a filter at the specified fs.
        :param new_fs: the new fs.
        :return: the new filter.
        '''
        return FirstOrder_LowPass(new_fs, self.freq, f_id=self.id)

    def sort_key(self):
        return f"{self.freq:05}00000{self.filter_type}"

    def to_json(self):
        return {
            '_type': self.__class__.__name__,
            'fs': self.fs,
            'fc': self.freq
        }


class FirstOrder_HighPass(Biquad):
    '''
    A one pole high pass filter.
    '''

    def __init__(self, fs, freq, f_id=-1):
        self.freq = freq
        self.order = 1
        super().__init__(fs, f_id=f_id)

    def __eq__(self, o: object) -> bool:
        return super().__eq__(o) and self.freq == o.freq

    @property
    def filter_type(self):
        return 'HPF1'

    @property
    def display_name(self):
        return 'HPF1'

    def _compute_coeffs(self):
        # TODO work out how to implement this directly
        sos = signal.butter(1, self.freq / (0.5 * self.fs), btype='high', output='sos')
        # a1 = -math.exp(-2.0 * math.pi * (0.5 - (self.freq / self.fs)))
        # b0 = 1.0 + a1
        # a = np.array([1.0, -a1, 0.0], dtype=np.float64)
        # b = np.array([b0, 0.0, 0.0])
        return sos[0][3:6], sos[0][0:3]

    def resample(self, new_fs):
        '''
        Creates a filter at the specified fs.
        :param new_fs: the new fs.
        :return: the new filter.
        '''
        return FirstOrder_HighPass(new_fs, self.freq, f_id=self.id)

    def sort_key(self):
        return f"{self.freq:05}00000{self.filter_type}"

    def to_json(self):
        return {
            '_type': self.__class__.__name__,
            'fs': self.fs,
            'fc': self.freq
        }


def fs_freq_q_json(o):
    return {
        '_type': o.__class__.__name__,
        'fs': o.fs,
        'fc': o.freq,
        'q': o.q
    }


class PassFilter(BiquadWithQ):

    def __init__(self, fs, freq, order, q=DEFAULT_Q, f_id=-1):
        super().__init__(fs, freq, q, f_id=f_id)
        self.order = order

    def __eq__(self, o: object) -> bool:
        return super().__eq__(o) and self.order == o.order

    def to_json(self):
        return fs_freq_q_json(self)


class SecondOrder_LowPass(PassFilter):
    '''
    LPF:        H(s) = 1 / (s^2 + s/Q + 1)

            b0 =  (1 - cos(w0))/2
            b1 =   1 - cos(w0)
            b2 =  (1 - cos(w0))/2
            a0 =   1 + alpha
            a1 =  -2*cos(w0)
            a2 =   1 - alpha
    '''

    def __init__(self, fs, freq, q=DEFAULT_Q, f_id=-1):
        super().__init__(fs, freq, 2, q, f_id=f_id)

    @property
    def filter_type(self):
        return 'LPF2'

    @property
    def display_name(self):
        return 'Variable Q LPF'

    def _compute_coeffs(self):
        a = np.array([
            1.0 + self.alpha,
            -2.0 * self.cos_w0,
            1.0 - self.alpha
        ], dtype=np.float64)
        b = np.array([
            (1.0 - self.cos_w0) / 2.0,
            1.0 - self.cos_w0,
            (1.0 - self.cos_w0) / 2.0
        ], dtype=np.float64)
        return a / a[0], b / a[0]

    def resample(self, new_fs):
        '''
        Creates a filter at the specified fs.
        :param new_fs: the new fs.
        :return: the new filter.
        '''
        return SecondOrder_LowPass(new_fs, self.freq, q=self.q, f_id=self.id)


class SecondOrder_HighPass(PassFilter):
    '''
    HPF:        H(s) = s^2 / (s^2 + s/Q + 1)

            b0 =  (1 + cos(w0))/2
            b1 = -(1 + cos(w0))
            b2 =  (1 + cos(w0))/2
            a0 =   1 + alpha
            a1 =  -2*cos(w0)
            a2 =   1 - alpha

    '''

    def __init__(self, fs, freq, q=DEFAULT_Q, f_id=-1):
        super().__init__(fs, freq, 2, q, f_id=f_id)

    @property
    def filter_type(self):
        return 'HPF2'

    @property
    def display_name(self):
        return 'Variable Q HPF'

    def _compute_coeffs(self):
        a = np.array([
            1.0 + self.alpha,
            -2.0 * self.cos_w0,
            1.0 - self.alpha
        ], dtype=np.float64)
        b = np.array([
            (1.0 + self.cos_w0) / 2.0,
            -(1.0 + self.cos_w0),
            (1.0 + self.cos_w0) / 2.0
        ], dtype=np.float64)
        return a / a[0], b / a[0]

    def resample(self, new_fs):
        '''
        Creates a filter at the specified fs.
        :param new_fs: the new fs.
        :return: the new filter.
        '''
        return SecondOrder_HighPass(new_fs, self.freq, q=self.q, f_id=self.id)


class AllPass(BiquadWithQ):
    '''
    APF:        H(s) = (s^2 - s/Q + 1) / (s^2 + s/Q + 1)

            b0 =   1 - alpha
            b1 =  -2*cos(w0)
            b2 =   1 + alpha
            a0 =   1 + alpha
            a1 =  -2*cos(w0)
            a2 =   1 - alpha
    '''

    def __init__(self, fs, freq, q, f_id=-1):
        super().__init__(fs, freq, q, f_id=f_id)

    @property
    def filter_type(self):
        return 'APF'

    @property
    def display_name(self):
        return 'All Pass'

    def _compute_coeffs(self):
        a = np.array([
            1.0 + self.alpha,
            -2.0 * self.cos_w0,
            1.0 - self.alpha
        ], dtype=np.float64)
        b = np.array([
            1.0 - self.alpha,
            -2.0 * self.cos_w0,
            1.0 + self.alpha
        ], dtype=np.float64)
        return a / a[0], b / a[0]

    def resample(self, new_fs):
        '''
        Creates a filter at the specified fs.
        :param new_fs: the new fs.
        :return: the new filter.
        '''
        return AllPass(new_fs, self.freq, self.q, f_id=self.id)

    def to_json(self):
        return fs_freq_q_json(self)


class ComplexFilter(SOS, Sequence):
    '''
    A filter composed of many other filters.
    '''

    def __init__(self, fs=1000, filters=None, description='Complex', preset_idx=-1, listener=None, f_id=-1,
                 sort_by_id: bool = False):
        super().__init__(f_id=f_id)
        self.filters = [f for f in filters if f] if filters is not None else []
        self.__sort_by_id = sort_by_id
        self.description = description
        self.__fs = fs
        self.listener = listener
        self.__on_change()
        self.__cached_transfer = None
        self.preset_idx = preset_idx

    @property
    def fs(self):
        return self.__fs

    def __getitem__(self, i):
        return self.filters[i]

    def __len__(self):
        return len(self.filters)

    def __repr__(self):
        return self.description

    def __eq__(self, o: object) -> bool:
        equal = self.__class__.__name__ == o.__class__.__name__
        equal &= self.description == o.description
        equal &= self.id == o.id
        equal &= self.filters == o.filters
        return equal

    def child_names(self):
        return [x.__repr__() for x in self.filters]

    @property
    def sort_by_id(self) -> bool:
        return self.__sort_by_id

    @property
    def filter_type(self):
        return 'Complex'

    def save(self, filter):
        '''
        Saves the filter with the given id, removing an existing one if necessary.
        :param filter: the filter.
        '''
        self.save0(filter, self.filters)
        self.__on_change()

    def __on_change(self):
        '''
        Resets some cached values when the filter changes.
        '''
        self.filters.sort(key=lambda f: f.sort_key() if not self.__sort_by_id else f.id)
        self.__cached_transfer = None
        self.preset_idx = -1
        if self.listener is not None:
            if hasattr(self.listener, 'name'):
                logger.debug(f"Propagating filter change to listener {self.listener.name}")
            self.listener.on_filter_change(self)

    def save0(self, filter, filters):
        if filter is not None:
            match = next((idx for idx, f in enumerate(filters) if f.id == filter.id), None)
            if match is not None:
                filters[match] = filter
            else:
                filters.append(filter)
        return filters

    def removeByIndex(self, indices):
        '''
        Removes the filter with the given indexes.
        :param indices: the indices to remove.
        '''
        self.filters = [filter for idx, filter in enumerate(self.filters) if idx not in indices]
        self.__on_change()

    def get_transfer_function(self):
        '''
        Computes the transfer function of the filter.
        :return: the transfer function.
        '''
        if self.__cached_transfer is None:
            if len(self.filters) == 0:
                return Passthrough(fs=self.fs).get_transfer_function()
            else:
                self.__cached_transfer = get_cascade_transfer_function(self.__repr__(),
                                                                       [x.get_transfer_function() for x in
                                                                        self.filters])
        return self.__cached_transfer

    def format_biquads(self, invert_a, separator=',\n', show_index=True, to_hex=False, fixed_point=False):
        '''
        Formats the filter into a biquad report.
        :param fixed_point: if true, output biquads in fixed point format.
        :param to_hex: convert the biquad to a hex format (for minidsp).
        :param separator: separator biquads with the string.
        :param show_index: whether to include the biquad index.
        :param invert_a: whether to invert the a coeffs.
        :return: the report.
        '''
        import itertools
        return list(itertools.chain(*[f.format_biquads(invert_a,
                                                       separator=separator,
                                                       show_index=show_index,
                                                       to_hex=to_hex,
                                                       fixed_point=fixed_point)
                                      for f in self.filters]))

    def get_sos(self):
        ''' outputs the filter in cascaded second order sections ready for consumption by sosfiltfilt '''
        return [x for f in self.filters for x in f.get_sos()]

    def get_impulse_response(self, dt=None, n=None):
        '''
        Converts the 2nd order section to a transfer function (b, a) and then computes the IR of the discrete time
        system.
        :param dt: the time delta.
        :param n: the no of samples to output, defaults to 1s.
        :return: t, y
        '''
        t, y = signal.dimpulse(signal.dlti(*signal.sos2tf(np.array(self.get_sos())),
                                           dt=1 / self.fs if dt is None else dt),
                               n=self.fs if n is None else n)
        return t, y

    def to_json(self):
        return {
            '_type': self.__class__.__name__,
            'description': self.description,
            'fs': self.__fs,
            'filters': [x.to_json() for x in self.filters]
        }


class CompleteFilter(ComplexFilter):

    def __init__(self, fs=1000, filters=None, description=COMBINED, preset_idx=-1, listener=None, f_id=-1,
                 sort_by_id=False):
        super().__init__(fs=fs, filters=filters, description=description, preset_idx=preset_idx, listener=listener,
                         f_id=f_id, sort_by_id=sort_by_id)

    def preview(self, filter):
        '''
        Creates a new filter with the supplied filter saved into it.
        :param filter: the filter.
        :return: a copied filter.
        '''
        return CompleteFilter(fs=self.fs, filters=self.save0(filter, self.filters.copy()),
                              description=self.description, preset_idx=self.preset_idx, listener=None,
                              sort_by_id=self.sort_by_id)

    def resample(self, new_fs, copy_listener=True):
        '''
        Creates a new filter at the desired fs.
        :param new_fs: the fs.
        :param copy_listener: if true, carry the listener forward to the resampled filter.
        :return: the new filter.
        '''
        listener = self.listener if copy_listener else None
        if len(self) > 0:
            return CompleteFilter(filters=[f.resample(new_fs) for f in self.filters], description=self.description,
                                  preset_idx=self.preset_idx, listener=listener, fs=new_fs, f_id=self.id)
        else:
            return CompleteFilter(description=self.description, preset_idx=self.preset_idx, listener=listener,
                                  fs=new_fs, f_id=self.id)

    @property
    def biquads(self):
        count = 0
        for f in self:
            count += len(f)
        return count


class FilterType(Enum):
    BUTTERWORTH = ('BW', 'Butterworth')
    LINKWITZ_RILEY = ('LR', 'Linkwitz-Riley')
    BESSEL_PHASE = ('BESP', 'Bessel Phase Matched')
    BESSEL_MAG = ('BESM', 'Bessel')

    def __new__(cls, name, display_name):
        entry = object.__new__(cls)
        entry._value_ = name
        entry.display_name = display_name
        return entry

    @staticmethod
    def value_of(display_name: str):
        return next((f for f in FilterType if f.display_name == display_name), None)


class CompoundPassFilter(ComplexFilter):
    '''
    A high or low pass filter of different types and orders that are implemented using one or more biquads.
    '''

    def __init__(self, high_or_low, one_pole_ctor, two_pole_ctor,
                 bessel_freq_calculator: Callable[[float, float], float], filter_type: FilterType, order, fs, freq,
                 q_scale=1.0, f_id=-1):
        self.__bw1 = one_pole_ctor
        self.__bw2 = two_pole_ctor
        self.__bessel_freq_calculator = bessel_freq_calculator
        self.type = filter_type
        self.order = order
        self.freq = round(float(freq), 2)
        if self.type is FilterType.LINKWITZ_RILEY:
            if self.order % 2 != 0:
                raise ValueError("LR filters must be even order")
        if self.order == 0:
            raise ValueError("Filter cannot have order = 0")
        self.__filter_type = f"{high_or_low} {filter_type.value}{order}"
        self.__q_scale = q_scale
        super().__init__(fs=fs, filters=self.calculate_biquads(fs), description=f"{self.__filter_type}/{self.freq}Hz",
                         f_id=f_id)

    @property
    def q_scale(self):
        return self.__q_scale

    @property
    def filter_type(self):
        return self.__filter_type

    def sort_key(self):
        return f"{self.freq:05}{self.order:05}{self.filter_type}"

    def calculate_biquads(self, fs) -> List[Biquad]:
        if self.type is FilterType.BUTTERWORTH:
            if self.order == 1:
                return [self.__bw1(fs, self.freq)]
            elif self.order == 2:
                return [self.__bw2(fs, self.freq)]
            else:
                return self.__calculate_high_order_bw(fs, self.order)
        elif self.type is FilterType.LINKWITZ_RILEY:
            # LRx is 2 * BW(x/2)
            if self.order == 2:
                return [self.__bw1(fs, self.freq) for _ in range(0, 2)]
            elif self.order == 4:
                return [self.__bw2(fs, self.freq) for _ in range(0, 2)]
            else:
                bw_order = int(self.order / 2)
                return self.__calculate_high_order_bw(fs, bw_order) + self.__calculate_high_order_bw(fs, bw_order)
        elif self.type is FilterType.BESSEL_PHASE:
            q = BESSEL_Q_FACTOR[self.order]
            f_mult = BESSEL_PHASE_MATCHED_F_MULTIPLIER[self.order]
            if len(q) != len(f_mult):
                raise ValueError(f"Invalid Bessel phase matched parameters for {self.order} - {q} - {f_mult}")
            return [self.__calculate_bessel_filter(fs, q[i], f_mult[i]) for i in range(len(q))]
        elif self.type is FilterType.BESSEL_MAG:
            q = BESSEL_Q_FACTOR[self.order]
            f_mult = BESSEL_3DB_F_MULTIPLIER[self.order]
            if len(q) != len(f_mult):
                raise ValueError(f"Invalid Bessel phase matched parameters for {self.order} - {q} - {f_mult}")
            return [self.__calculate_bessel_filter(fs, q[i], f_mult[i]) for i in range(len(q))]
        else:
            raise ValueError("Unknown filter type " + str(self.type))

    def __calculate_bessel_filter(self, fs: int, q: float, f_mult: float) -> Biquad:
        if math.isclose(q, -1.0):
            return self.__bw1(fs, self.__bessel_freq_calculator(self.freq, f_mult))
        else:
            return self.__bw2(fs, self.__bessel_freq_calculator(self.freq, f_mult), q)

    def __calculate_high_order_bw(self, fs, order):
        # approach taken from http://www.earlevel.com/main/2016/09/29/cascading-filters/
        biquads = []
        pairs = order >> 1
        odd_poles = order & 1
        pole_inc = np.math.pi / order
        first_angle = pole_inc
        if not odd_poles:
            first_angle /= 2
        else:
            biquads.append(self.__bw1(fs, self.freq, 0.5))
        biquads += [
            self.__bw2(fs, self.freq, (1.0 / (2.0 * math.cos(first_angle + x * pole_inc)) * self.q_scale))
            for x in range(0, pairs)
        ]
        return biquads


class ComplexLowPass(CompoundPassFilter):
    '''
    A low pass filter of different types and orders that are implemented using one or more biquads.
    '''

    def __init__(self, filter_type, order, fs, freq, q_scale=1.0, f_id=-1):
        super().__init__('Low', FirstOrder_LowPass, SecondOrder_LowPass, lambda a, b: a*b, filter_type, order, fs, freq,
                         q_scale=q_scale, f_id=f_id)

    @property
    def display_name(self):
        return 'Low Pass'

    def resample(self, new_fs):
        '''
        Creates a new filter at the desired fs.
        :param new_fs: the fs.
        :return: the new filter.
        '''
        return ComplexLowPass(self.type, self.order, new_fs, self.freq, f_id=self.id)

    def to_json(self):
        return {
            '_type': self.__class__.__name__,
            'filter_type': self.type.value,
            'order': self.order,
            'fs': self.fs,
            'fc': self.freq
        }


class ComplexHighPass(CompoundPassFilter):
    '''
    A high pass filter of different types and orders that are implemented using one or more biquads.
    '''

    def __init__(self, filter_type, order, fs, freq, q_scale=1.0, f_id=-1):
        super().__init__('High', FirstOrder_HighPass, SecondOrder_HighPass, lambda a, b: a/b, filter_type, order, fs,
                         freq, q_scale=q_scale, f_id=f_id)

    @property
    def display_name(self):
        return 'High Pass'

    def resample(self, new_fs):
        '''
        Creates a new filter at the desired fs.
        :param new_fs: the fs.
        :return: the new filter.
        '''
        return ComplexHighPass(self.type, self.order, new_fs, self.freq, f_id=self.id)

    def to_json(self):
        return {
            '_type': self.__class__.__name__,
            'filter_type': self.type.value,
            'order': self.order,
            'fs': self.fs,
            'fc': self.freq
        }


def as_equalizer_apo(filt):
    '''
    formats a filter in Equalizer APO config format (https://sourceforge.net/p/equalizerapo/wiki/Configuration%20reference/)
    :param filt: the filter.
    :return: the text.
    '''
    if isinstance(filt, PeakingEQ):
        return f"ON PK Fc {filt.freq:g} Hz Gain {filt.gain:g} dB Q {filt.q:g}"
    elif isinstance(filt, Shelf):
        if filt.count == 1:
            return f"ON {filt.filter_type}C Fc {filt.freq:g} Hz Gain {filt.gain:g} dB Q {filt.q:g}"
        else:
            return [as_equalizer_apo(f) for f in filt.flatten()]
    elif isinstance(filt, AllPass):
        return f"ON AP Fc {filt.freq:g} Hz Q {filt.q:g}"
    else:
        return None


def get_cascade_transfer_function(name, responses) -> ComplexData:
    '''
    The transfer function for a cascade of filters.
    :param name: the name.
    :param responses: the individual filter responses.
    :return: the transfer function (ComplexData)
    '''
    return ComplexData(name, responses[0].x, reduce((lambda x, y: x * y), [r.y for r in responses]))

