# from http://www.musicdsp.org/files/Audio-EQ-Cookbook.txt
import logging
import math
from abc import ABC, abstractmethod
from enum import Enum
from functools import reduce

import numpy as np
from scipy import signal

DEFAULT_Q = 1 / np.math.sqrt(2.0)

logger = logging.getLogger('iir')


class Biquad(ABC):
    def __init__(self, fs, freq, q):
        self.fs = fs
        self.freq = round(freq, 2)
        self.q = q
        self.w0 = 2.0 * math.pi * self.freq / self.fs
        self.cos_w0 = math.cos(self.w0)
        self.sin_w0 = math.sin(self.w0)
        self.alpha = self.sin_w0 / (2.0 * self.q)
        self.a, self.b = self._compute_coeffs()
        self.id = -1

    def __repr__(self):
        return self.description

    @property
    def description(self):
        description = ''
        if hasattr(self, 'display_name'):
            description += self.display_name
        description += f" {self.freq}/{self.q}"
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
        w, h = signal.freqz(b=self.b, a=self.a, worN=1 << (self.fs - 1).bit_length())
        f = w * self.fs / (2 * np.pi)
        return ComplexData(self.__repr__(), f, h)


class BiquadWithGain(Biquad):
    def __init__(self, fs, freq, q, gain):
        self.gain = round(gain, 3)
        super().__init__(fs, freq, q)

    @property
    def description(self):
        return super().description + f"/{self.gain}dB"


class PeakingEQ(BiquadWithGain):
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
    def display_name(self):
        return 'Peak'

    def _compute_coeffs(self):
        A = 10.0 ** (self.gain / 40.0)
        a = np.array([1.0 + self.alpha / A, -2.0 * self.cos_w0, 1.0 - self.alpha / A], dtype=np.float64)
        b = np.array([1.0 + self.alpha * A, -2.0 * self.cos_w0, 1.0 - self.alpha * A], dtype=np.float64)
        return a / a[0], b / a[0]


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
    return 1.0 / math.sqrt(((A + 1.0 / A) * ((1.0 / s) - 1.0)) + 2.0)


class Shelf(BiquadWithGain):
    def __init__(self, fs, freq, q, gain):
        self.A = 10.0 ** (gain / 40.0)
        super().__init__(fs, freq, q, gain)

    def q_to_s(self):
        '''
        :return: the filter Q as S
        '''
        return q_to_s(self.q, self.gain)


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

    def __init__(self, fs, freq, q, gain):
        super().__init__(fs, freq, q, gain)

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

    def __init__(self, fs, freq, q, gain):
        super().__init__(fs, freq, q, gain)

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


class FirstOrder_LowPass(Biquad):
    '''
    A one pole low pass filter.
    '''

    def __init__(self, fs, freq, q=DEFAULT_Q):
        super().__init__(fs, freq, q)
        self.order = 1

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


class FirstOrder_HighPass(Biquad):
    '''
    A one pole high pass filter.
    '''

    def __init__(self, fs, freq, q=DEFAULT_Q):
        super().__init__(fs, freq, q)
        self.order = 1

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


class SecondOrder_LowPass(Biquad):
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


class SecondOrder_HighPass(Biquad):
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


class AllPass(Biquad):
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


class ComplexFilter:
    '''
    A filter composed of many other filters.
    '''

    def __init__(self, filters=None, description='Complex'):
        self.__filters = filters if filters is not None else []
        self.description = description
        self.id = -1

    def __getitem__(self, i):
        return self.__filters[i]

    def __len__(self):
        return len(self.__filters)

    def __repr__(self):
        return self.description

    @property
    def filter_type(self):
        return 'Complex'

    def add(self, filter):
        '''
        Adds a new filter.
        :param filter: the filter.
        '''
        self.__filters.append(filter)

    def replace(self, filter):
        '''
        Replaces the filter with the given id.
        :param filter: the filter.
        '''
        match = next((f for f in self.__filters if f.id == filter.id), None)
        if match:
            self.__filters.remove(match)
        self.add(filter)

    def removeByIndex(self, indices):
        '''
        Removes the filter with the given indexes.
        :param indices: the indices to remove.
        '''
        self.__filters = [filter for idx, filter in enumerate(self.__filters) if idx not in indices]

    def getTransferFunction(self):
        '''
        Computes the transfer function of the filter.
        :return: the transfer function.
        '''
        responses = [x.getTransferFunction() for x in self.__filters]
        return ComplexData(self.__repr__(), responses[0].x, reduce((lambda x, y: x * y), [r.y for r in responses]))


class FilterType(Enum):
    BUTTERWORTH = 'BW'
    LINKWITZ_RILEY = 'LR'


class CompoundPassFilter(ComplexFilter):
    '''
    A high or low pass filter of different types and orders that are implemented using one or more biquads.
    '''

    def __init__(self, one_pole_ctor, two_pole_ctor, filter_type, order, fs, freq):
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
        self.__filter_type = f"{filter_type.value}{order}"
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
        super().__init__(FirstOrder_LowPass, SecondOrder_LowPass, filter_type, order, fs, freq)

    @property
    def display_name(self):
        return 'Low Pass'


class ComplexHighPass(CompoundPassFilter):
    '''
    A high pass filter of different types and orders that are implemented using one or more biquads.
    '''

    def __init__(self, filter_type, order, fs, freq):
        super().__init__(FirstOrder_HighPass, SecondOrder_HighPass, filter_type, order, fs, freq)

    @property
    def display_name(self):
        return 'High Pass'


class ComplexData:
    '''
    Value object for storing complex data.
    '''

    def __init__(self, name, x, y, scaleFactor=1):
        self.name = name
        self.x = x
        self.y = y
        self.scaleFactor = scaleFactor

    def getMagnitude(self, ref=1, colour=None):
        y = np.abs(self.y) * self.scaleFactor / ref
        return XYData(self.name, self.x, 20 * np.log10(y), colour=colour)

    def getPhase(self, colour=None):
        return XYData(self.name, self.x, np.angle(self.y), colour=colour)


class XYData:
    '''
    Value object for showing data on a magnitude graph.
    '''

    def __init__(self, name, x, y, colour=None, linestyle='-'):
        self.name = name
        self.x = x
        self.y = y
        self.colour = colour
        self.linestyle = linestyle

    def normalise(self, target):
        '''
        Normalises the y value against the target y.
        :param target: the target.
        :return: a normalised XYData.
        '''
        return XYData(self.name, self.x, self.y - target.y, colour=self.colour, linestyle=self.linestyle)

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
                return XYData(f"{self.name}-filtered", self.x, self.y + interp_y, colour=self.colour,
                              linestyle=self.linestyle)
            else:
                interp_y = np.interp(filt.x, self.x, self.y)
                return XYData(f"{self.name}-filtered", filt.x, filt.y + interp_y, colour=self.colour,
                              linestyle=self.linestyle)
        else:
            return XYData(f"{self.name}-filtered", self.x, self.y + filt.y, colour=self.colour,
                          linestyle=self.linestyle)
