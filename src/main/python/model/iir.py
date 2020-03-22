# from http://www.musicdsp.org/files/Audio-EQ-Cookbook.txt
import logging
import math
import struct
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum
from functools import reduce

import numpy as np
from scipy import signal

from model.xy import MagnitudeData

DEFAULT_Q = 1 / np.math.sqrt(2.0)

logger = logging.getLogger('iir')

import decimal

ctx = decimal.Context()
ctx.prec = 17

COMBINED = 'Combined'


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


class Biquad(ABC):
    def __init__(self, fs, f_id=-1):
        self.fs = fs
        self.a, self.b = self._compute_coeffs()
        self.id = f_id
        self.__transferFunction = None

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

    def getTransferFunction(self):
        '''
        Computes the transfer function of the filter.
        :return: the transfer function.
        '''
        if self.__transferFunction is None:
            from model.preferences import X_RESOLUTION
            import time
            start = time.time()
            w, h = signal.freqz(b=self.b, a=self.a, worN=X_RESOLUTION)
            end = time.time()
            logger.debug(f"freqz in {round((end - start) * 1000, 3)}ms")
            f = w * self.fs / (2 * np.pi)
            self.__transferFunction = ComplexData(self.__repr__(), f, h)
        return self.__transferFunction

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

    def get_sos(self):
        return [np.concatenate((self.b, self.a)).tolist()]


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

    def getTransferFunction(self):
        single = super().getTransferFunction()
        if self.count == 1:
            return single
        elif self.count > 1:
            if self.__cached_cascade is None:
                self.__cached_cascade = getCascadeTransferFunction(self.__repr__(), [single] * self.count)
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


class FirstOrder_LowPass(BiquadWithQ):
    '''
    A one pole low pass filter.
    '''

    def __init__(self, fs, freq, q=DEFAULT_Q, f_id=-1):
        super().__init__(fs, freq, q, f_id=f_id)
        self.order = 1

    def __eq__(self, o: object) -> bool:
        return super().__eq__(o) and self.order == o.order

    @property
    def filter_type(self):
        return 'LPF1'

    @property
    def display_name(self):
        return 'Variable Q LPF'

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
        return FirstOrder_LowPass(new_fs, self.freq, q=self.q, f_id=self.id)

    def to_json(self):
        return fs_freq_q_json(self)


class FirstOrder_HighPass(BiquadWithQ):
    '''
    A one pole high pass filter.
    '''

    def __init__(self, fs, freq, q=DEFAULT_Q, f_id=-1):
        super().__init__(fs, freq, q, f_id=f_id)
        self.order = 1

    def __eq__(self, o: object) -> bool:
        return super().__eq__(o) and self.order == o.order

    @property
    def filter_type(self):
        return 'HPF1'

    @property
    def display_name(self):
        return 'Variable Q HPF'

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
        return FirstOrder_HighPass(new_fs, self.freq, q=self.q, f_id=self.id)

    def to_json(self):
        return fs_freq_q_json(self)


def fs_freq_q_json(o):
    return {
        '_type': o.__class__.__name__,
        'fs': o.fs,
        'fc': o.freq,
        'q': o.q
    }


class SecondOrder_LowPass(BiquadWithQ):
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
        super().__init__(fs, freq, q, f_id=f_id)
        self.order = 2

    def __eq__(self, o: object) -> bool:
        return super().__eq__(o) and self.order == o.order

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

    def to_json(self):
        return fs_freq_q_json(self)


class SecondOrder_HighPass(BiquadWithQ):
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
        super().__init__(fs, freq, q, f_id=f_id)
        self.order = 2

    def __eq__(self, o: object) -> bool:
        return super().__eq__(o) and self.order == o.order

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

    def to_json(self):
        return fs_freq_q_json(self)


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


def getCascadeTransferFunction(name, responses):
    '''
    The transfer function for a cascade of filters.
    :param name: the name.
    :param responses: the individual filter responses.
    :return: the transfer function (ComplexData)
    '''
    return ComplexData(name, responses[0].x, reduce((lambda x, y: x * y), [r.y for r in responses]))


class ComplexFilter(Sequence):
    '''
    A filter composed of many other filters.
    '''

    def __init__(self, fs=1000, filters=None, description='Complex', preset_idx=-1, listener=None, f_id=-1,
                 sort_by_id=False):
        self.filters = filters if filters is not None else []
        self.__sort_by_id = sort_by_id
        self.description = description
        self.__fs = fs
        self.id = f_id
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

    def getTransferFunction(self):
        '''
        Computes the transfer function of the filter.
        :return: the transfer function.
        '''
        if self.__cached_transfer is None:
            if len(self.filters) == 0:
                return Passthrough(fs=self.fs).getTransferFunction()
            else:
                self.__cached_transfer = getCascadeTransferFunction(self.__repr__(),
                                                                    [x.getTransferFunction() for x in self.filters])
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
        return CompleteFilter(filters=self.save0(filter, self.filters.copy()),
                              description=self.description, listener=None, fs=self.fs)

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
    BUTTERWORTH = 'BW'
    LINKWITZ_RILEY = 'LR'


class CompoundPassFilter(ComplexFilter):
    '''
    A high or low pass filter of different types and orders that are implemented using one or more biquads.
    '''

    def __init__(self, high_or_low, one_pole_ctor, two_pole_ctor, filter_type, order, fs, freq, f_id=-1):
        self.__bw1 = one_pole_ctor
        self.__bw2 = two_pole_ctor
        self.type = filter_type
        self.order = order
        self.freq = round(float(freq), 2)
        if self.type is FilterType.LINKWITZ_RILEY:
            if self.order % 2 != 0:
                raise ValueError("LR filters must be even order")
        if self.order == 0:
            raise ValueError("Filter cannot have order = 0")
        self.__filter_type = f"{high_or_low} {filter_type.value}{order}"
        super().__init__(fs=fs, filters=self._calculate_biquads(fs), description=f"{self.__filter_type}/{self.freq}Hz",
                         f_id=f_id)

    @property
    def filter_type(self):
        return self.__filter_type

    def sort_key(self):
        return f"{self.freq:05}{self.order:05}{self.filter_type}"

    def _calculate_biquads(self, fs):
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
        else:
            raise ValueError("Unknown filter type " + str(self.type))

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
            self.__bw2(fs, self.freq, 1.0 / (2.0 * math.cos(first_angle + x * pole_inc))) for x in range(0, pairs)
        ]
        return biquads


class ComplexLowPass(CompoundPassFilter):
    '''
    A low pass filter of different types and orders that are implemented using one or more biquads.
    '''

    def __init__(self, filter_type, order, fs, freq, f_id=-1):
        super().__init__('Low', FirstOrder_LowPass, SecondOrder_LowPass, filter_type, order, fs, freq, f_id=f_id)

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

    def __init__(self, filter_type, order, fs, freq, f_id=-1):
        super().__init__('High', FirstOrder_HighPass, SecondOrder_HighPass, filter_type, order, fs, freq, f_id=f_id)

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


class ComplexData:
    '''
    Value object for storing complex data.
    '''

    def __init__(self, name, x, y, scaleFactor=1):
        self.name = name
        self.x = x
        self.y = y
        self.scaleFactor = scaleFactor
        self.__cached_mag_ref = None
        self.__cached_mag = None
        self.__cached_phase = None

    def getMagnitude(self, ref=1, colour=None, linestyle='-'):
        if self.__cached_mag_ref is not None and math.isclose(ref, self.__cached_mag_ref):
            self.__cached_mag.colour = colour
            self.__cached_mag.linestyle = linestyle
        else:
            self.__cached_mag_ref = ref
            y = np.abs(self.y) * self.scaleFactor / ref
            # avoid divide by zero issues when converting to decibels
            y[np.abs(y) < 0.0000001] = 0.0000001
            self.__cached_mag = MagnitudeData(self.name, None, self.x, 20 * np.log10(y), colour=colour,
                                              linestyle=linestyle)
        return self.__cached_mag

    def getPhase(self, colour=None):
        if self.__cached_phase is None:
            self.__cached_phase = MagnitudeData(self.name, None, self.x, np.angle(self.y), colour=colour)
        return self.__cached_phase
