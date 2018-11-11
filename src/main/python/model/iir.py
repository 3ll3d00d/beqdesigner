# from http://www.musicdsp.org/files/Audio-EQ-Cookbook.txt
import logging
import math
from abc import ABC, abstractmethod
from collections import Sequence
from enum import Enum
from functools import reduce

import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline

DEFAULT_Q = 1 / np.math.sqrt(2.0)

logger = logging.getLogger('iir')

import decimal

ctx = decimal.Context()
ctx.prec = 15

COMBINED = 'Combined'


def float_to_str(f):
    """
    Convert the given float to a string, without resorting to scientific notation
    """
    d1 = ctx.create_decimal(repr(f))
    return format(d1, 'f')


class Biquad(ABC):
    def __init__(self, fs):
        self.fs = fs
        self.a, self.b = self._compute_coeffs()
        self.id = -1
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

    def getTransferFunction(self):
        '''
        Computes the transfer function of the filter.
        :param filt: the filter.
        :return: the transfer function.
        '''
        if self.__transferFunction is None:
            w, h = signal.freqz(b=self.b, a=self.a, worN=max(1 << (self.fs - 1).bit_length(), 8192))
            f = w * self.fs / (2 * np.pi)
            self.__transferFunction = ComplexData(self.__repr__(), f, h)
        return self.__transferFunction

    def format_biquads(self, minidsp_style, separator=',\n'):
        ''' Creates a biquad report '''
        a = separator.join(
            [f"a{idx}={float_to_str(-x if minidsp_style else x)}" for idx, x in enumerate(self.a) if
             idx != 0 or minidsp_style is False])
        b = separator.join([f"b{idx}={float_to_str(x)}" for idx, x in enumerate(self.b)])
        return [f"{b}{separator}{a}"]

    def get_sos(self):
        return [np.concatenate((self.b, self.a)).tolist()]


class Gain(Biquad):
    def __init__(self, fs, gain):
        self.gain = gain
        super().__init__(fs)

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
        return Gain(new_fs, self.gain)

    def to_json(self):
        return {
            '_type': self.__class__.__name__,
            'fs': self.fs,
            'gain': self.gain
        }


class BiquadWithQ(Biquad):
    def __init__(self, fs, freq, q):
        self.freq = round(freq, 2)
        self.q = round(q, 4)
        self.w0 = 2.0 * math.pi * freq / fs
        self.cos_w0 = math.cos(self.w0)
        self.sin_w0 = math.sin(self.w0)
        self.alpha = self.sin_w0 / (2.0 * self.q)
        super().__init__(fs)

    def __eq__(self, o: object) -> bool:
        return super().__eq__(o) and self.freq == o.freq

    @property
    def description(self):
        return super().description + f"{self.freq}/{self.q}"


class Passthrough(Gain):
    def __init__(self, fs=1000):
        super().__init__(fs, 0)

    @property
    def description(self):
        return 'Passthrough'

    def to_json(self):
        return {
            '_type': self.__class__.__name__,
            'fs': self.fs
        }


class BiquadWithQGain(BiquadWithQ):
    def __init__(self, fs, freq, q, gain):
        self.gain = round(gain, 3)
        super().__init__(fs, freq, q)

    def __eq__(self, o: object) -> bool:
        return super().__eq__(o) and self.gain == o.gain

    @property
    def description(self):
        return super().description + f"/{self.gain}dB"


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

    def __init__(self, fs, freq, q, gain):
        super().__init__(fs, freq, q, gain)

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
        return PeakingEQ(new_fs, self.freq, self.q, self.gain)

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
    def __init__(self, fs, freq, q, gain, count):
        self.A = 10.0 ** (gain / 40.0)
        super().__init__(fs, freq, q, gain)
        self.count = count
        self.__cached_cascade = None

    def q_to_s(self):
        '''
        :return: the filter Q as S
        '''
        return q_to_s(self.q, self.gain)

    def __len__(self):
        return self.count

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

    def format_biquads(self, minidsp_style, separator=',\n'):
        single = super().format_biquads(minidsp_style, separator=separator)
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

    def __init__(self, fs, freq, q, gain, count=1):
        super().__init__(fs, freq, q, gain, count)

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
        return LowShelf(new_fs, self.freq, self.q, self.gain, self.count)


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

    def __init__(self, fs, freq, q, gain, count=1):
        super().__init__(fs, freq, q, gain, count)

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
        return HighShelf(new_fs, self.freq, self.q, self.gain, self.count)


class FirstOrder_LowPass(BiquadWithQ):
    '''
    A one pole low pass filter.
    '''

    def __init__(self, fs, freq, q=DEFAULT_Q):
        super().__init__(fs, freq, q)
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
        return FirstOrder_LowPass(new_fs, self.freq, q=self.q)

    def to_json(self):
        return fs_freq_q_json(self)


class FirstOrder_HighPass(BiquadWithQ):
    '''
    A one pole high pass filter.
    '''

    def __init__(self, fs, freq, q=DEFAULT_Q):
        super().__init__(fs, freq, q)
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
        return sos[0][3:5], sos[0][0:2]

    def resample(self, new_fs):
        '''
        Creates a filter at the specified fs.
        :param new_fs: the new fs.
        :return: the new filter.
        '''
        return FirstOrder_HighPass(new_fs, self.freq, q=self.q)

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

    def __init__(self, fs, freq, q=DEFAULT_Q):
        super().__init__(fs, freq, q)
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
        return SecondOrder_LowPass(new_fs, self.freq, q=self.q)

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

    def __init__(self, fs, freq, q=DEFAULT_Q):
        super().__init__(fs, freq, q)
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
        return SecondOrder_HighPass(new_fs, self.freq, q=self.q)

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

    def __init__(self, fs, freq, q):
        super().__init__(fs, freq, q)

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
        return AllPass(new_fs, self.freq, self.q)

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

    def __init__(self, filters=None, description='Complex', preset_idx=-1, listener=None):
        self.filters = filters if filters is not None else []
        self.description = description
        self.id = -1
        self.listener = listener
        self.__on_change()
        self.__cached_transfer = None
        self.preset_idx = preset_idx

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
        self.__cached_transfer = None
        self.preset_idx = -1
        if self.listener is not None:
            if hasattr(self.listener, 'name'):
                logger.debug(f"Propagating filter change to listener {self.listener.name}")
            self.listener.on_filter_change(self)

    def save0(self, filter, filters):
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
                return Passthrough().getTransferFunction()
            else:
                self.__cached_transfer = getCascadeTransferFunction(self.__repr__(),
                                                                    [x.getTransferFunction() for x in self.filters])
        return self.__cached_transfer

    def format_biquads(self, invert_a, separator=',\n'):
        '''
        Formats the filter into a biquad report.
        :param invert_a: whether to invert the a coeffs.
        :return: the report.
        '''
        return [f.format_biquads(invert_a, separator=separator) for f in self.filters]

    def get_sos(self):
        ''' outputs the filter in cascaded second order sections ready for consumption by sosfiltfilt '''
        return [x for f in self.filters for x in f.get_sos()]

    def to_json(self):
        return {
            '_type': self.__class__.__name__,
            'description': self.description,
            'filters': [x.to_json() for x in self.filters]
        }


class CompleteFilter(ComplexFilter):

    def __init__(self, filters=None, description=COMBINED, preset_idx=-1, listener=None):
        super().__init__(filters=filters, description=description, preset_idx=preset_idx, listener=listener)

    def preview(self, filter):
        '''
        Creates a new filter with the supplied filter saved into it.
        :param filter: the filter.
        :return: a copied filter.
        '''
        return CompleteFilter(self.save0(filter, self.filters.copy()), self.description, listener=None)

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
                                  preset_idx=self.preset_idx, listener=listener)
        else:
            return CompleteFilter(description=self.description, preset_idx=self.preset_idx, listener=listener)


class FilterType(Enum):
    BUTTERWORTH = 'BW'
    LINKWITZ_RILEY = 'LR'


class CompoundPassFilter(ComplexFilter):
    '''
    A high or low pass filter of different types and orders that are implemented using one or more biquads.
    '''

    def __init__(self, high_or_low, one_pole_ctor, two_pole_ctor, filter_type, order, fs, freq):
        self.__bw1 = one_pole_ctor
        self.__bw2 = two_pole_ctor
        self.type = filter_type
        self.order = order
        self.fs = fs
        self.freq = round(freq, 2)
        if self.type is FilterType.LINKWITZ_RILEY:
            if self.order % 2 != 0:
                raise ValueError("LR filters must be even order")
        if self.order == 0:
            raise ValueError("Filter cannot have order = 0")
        self.__filter_type = f"{high_or_low} {filter_type.value}{order}"
        super().__init__(filters=self._calculate_biquads(), description=f"{self.__filter_type}/{self.freq}Hz")

    @property
    def filter_type(self):
        return self.__filter_type

    def _calculate_biquads(self):
        if self.type is FilterType.BUTTERWORTH:
            if self.order == 1:
                return [self.__bw1(self.fs, self.freq)]
            elif self.order == 2:
                return [self.__bw2(self.fs, self.freq)]
            else:
                return self.__calculate_high_order_bw(self.order)
        elif self.type is FilterType.LINKWITZ_RILEY:
            # LRx is 2 * BW(x/2)
            if self.order == 2:
                return [self.__bw1(self.fs, self.freq) for _ in range(0, 2)]
            elif self.order == 4:
                return [self.__bw2(self.fs, self.freq) for _ in range(0, 2)]
            else:
                bw_order = int(self.order / 2)
                return self.__calculate_high_order_bw(bw_order) + self.__calculate_high_order_bw(bw_order)
        else:
            raise ValueError("Unknown filter type " + str(self.type))

    def __calculate_high_order_bw(self, order):
        # approach taken from http://www.earlevel.com/main/2016/09/29/cascading-filters/
        biquads = []
        pairs = order >> 1
        odd_poles = order & 1
        pole_inc = np.math.pi / order
        first_angle = pole_inc
        if not odd_poles:
            first_angle /= 2
        else:
            biquads.append(self.__bw1(self.fs, self.freq, 0.5))
        biquads += [self.__bw2(self.fs, self.freq, 1.0 / (2.0 * math.cos(first_angle + x * pole_inc))) for x in
                    range(0, pairs)]
        return biquads


class ComplexLowPass(CompoundPassFilter):
    '''
    A low pass filter of different types and orders that are implemented using one or more biquads.
    '''

    def __init__(self, filter_type, order, fs, freq):
        super().__init__('Low', FirstOrder_LowPass, SecondOrder_LowPass, filter_type, order, fs, freq)

    @property
    def display_name(self):
        return 'Low Pass'

    def resample(self, new_fs):
        '''
        Creates a new filter at the desired fs.
        :param new_fs: the fs.
        :return: the new filter.
        '''
        return ComplexLowPass(self.type, self.order, new_fs, self.freq)

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

    def __init__(self, filter_type, order, fs, freq):
        super().__init__('High', FirstOrder_HighPass, SecondOrder_HighPass, filter_type, order, fs, freq)

    @property
    def display_name(self):
        return 'High Pass'

    def resample(self, new_fs):
        '''
        Creates a new filter at the desired fs.
        :param new_fs: the fs.
        :return: the new filter.
        '''
        return ComplexHighPass(self.type, self.order, new_fs, self.freq)

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
            self.__cached_mag = XYData(self.name, None, self.x, 20 * np.log10(y), colour=colour,
                                       linestyle=linestyle)
        return self.__cached_mag

    def getPhase(self, colour=None):
        if self.__cached_phase is None:
            self.__cached_phase = XYData(self.name, None, self.x, np.angle(self.y), colour=colour)
        return self.__cached_phase


class XYData:
    '''
    Value object for showing data on a magnitude graph.
    '''

    def __init__(self, name, description, x, y, colour=None, linestyle='-'):
        self.__name = name
        self.__description = description
        self.x = x
        self.y = np.nan_to_num(y)
        # TODO consider a variable spacing so we don't go overboard for full range view
        required_points = math.ceil(self.x[-1]) * 4
        if self.y.size != required_points:
            new_x = np.linspace(self.x[0], self.x[-1], num=required_points, endpoint=True)
            cs = CubicSpline(self.x, self.y)
            new_y = cs(new_x)
            logger.debug(f"Interpolating {name} from {self.y.size} to {required_points}")
            self.x = new_x
            self.y = new_y
        self.__equal_energy_adjusted = None
        self.colour = colour
        self.linestyle = linestyle
        self.miny = np.ma.masked_invalid(y).min()
        self.maxy = np.ma.masked_invalid(y).max()
        self.__rendered = False
        self.__normalised_cache = {}

    @property
    def name(self):
        if self.__description is None:
            return self.__name
        else:
            return f"{self.__name}_{self.__description}"

    @property
    def internal_name(self):
        return self.__name

    @internal_name.setter
    def internal_name(self, name):
        self.__name = name

    @property
    def internal_description(self):
        return self.__description

    def __repr__(self):
        return f"XYData: {self.name} - {self.x.size} - {self.colour}"

    def __eq__(self, o: object) -> bool:
        equal = self.__class__.__name__ == o.__class__.__name__
        equal &= self.name == o.name
        # allow tolerance because of the way we serialise to save space
        equal &= np.allclose(self.x, o.x)
        equal &= np.allclose(self.y, o.y, rtol=1e-5, atol=1e-5)
        equal &= self.colour == o.colour
        equal &= self.linestyle == o.linestyle
        return equal

    @property
    def rendered(self):
        return self.__rendered

    @rendered.setter
    def rendered(self, value):
        self.__rendered = value

    def normalise(self, target):
        '''
        Normalises the y value against the target y.
        :param target: the target.
        :return: a normalised XYData.
        '''
        if target.name not in self.__normalised_cache:
            logger.debug(f"Normalising {self.name} against {target.name}")
            count = min(self.x.size, target.x.size) - 1
            self.__normalised_cache[target.name] = XYData(self.__name, self.__description, self.x[0:count],
                                                          self.y[0:count] - target.y[0:count], colour=self.colour,
                                                          linestyle=self.linestyle)
        return self.__normalised_cache[target.name]

    def filter(self, filt):
        '''
        Adds filt.y to the data.y as we're dealing in the frequency domain. Interpolates the smaller xy if required so
        we can just add them together.
        :param filt: the filter in XYData form.
        :return: the filtered response.
        '''
        if self.x.size != filt.x.size:
            logger.debug(f"Interpolating filt {filt.x.size} vs self {self.x.size}")
            if self.x.size > filt.x.size:
                interp_y = np.interp(self.x, filt.x, filt.y)
                return XYData(self.__name, f"{self.__description}-filtered", self.x, self.y + interp_y,
                              colour=self.colour, linestyle='-')
            else:
                interp_y = np.interp(filt.x, self.x, self.y)
                return XYData(self.__name, f"{self.__description}-filtered", filt.x, filt.y + interp_y,
                              colour=self.colour, linestyle='-')
        else:
            return XYData(self.__name, f"{self.__description}-filtered", self.x, self.y + filt.y, colour=self.colour,
                          linestyle='-')

    def with_equal_energy_adjustment(self):
        ''' returns the equal energy adjusted version of this data. '''
        if self.__equal_energy_adjusted is None:
            x = np.linspace(self.x[1], self.x[-1], num=self.x.size)
            adjustment = np.log10(x) * 10
            self.__equal_energy_adjusted = XYData(self.__name, self.__description, self.x, self.y + adjustment,
                                                  colour=self.colour, linestyle=self.linestyle)
        return self.__equal_energy_adjusted
