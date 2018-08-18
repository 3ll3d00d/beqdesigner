# from http://www.musicdsp.org/files/Audio-EQ-Cookbook.txt

import math
from abc import ABC, abstractmethod
from enum import Enum
from functools import reduce

import numpy as np
from scipy import signal

DEFAULT_Q = 1 / np.math.sqrt(2.0)


class Biquad(ABC):
    def __init__(self, fs, freq, q):
        self.fs = fs
        self.freq = freq
        self.q = q
        self.w0 = 2.0 * math.pi * self.freq / self.fs
        self.cos_w0 = math.cos(self.w0)
        self.sin_w0 = math.sin(self.w0)
        self.alpha = self.sin_w0 / (2.0 * self.q)
        self.a, self.b = self._compute_coeffs()

    def __repr__(self):
        return 'woot'

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
        # TODO calculate worN based on fs
        w, h = signal.freqz(b=self.b, a=self.a, worN=65536)
        f = w * self.fs / (2 * np.pi)
        return ComplexData(self.__repr__(), f, h)


class PeakingEQ(Biquad):
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
        self.gain = gain
        super().__init__(fs, freq, q)

    def _compute_coeffs(self):
        A = 10.0 ** (self.gain / 40.0)
        a = np.array([1.0 + self.alpha / A, -2.0 * self.cos_w0, 1.0 - self.alpha / A], dtype=np.float64)
        b = np.array([1.0 + self.alpha * A, -2.0 * self.cos_w0, 1.0 - self.alpha * A], dtype=np.float64)
        return a / a[0], b / a[0]


class Shelf(Biquad):
    def __init__(self, fs, freq, q, gain):
        self.gain = gain
        self.A = 10.0 ** (self.gain / 40.0)
        super().__init__(fs, freq, q)

    def q_to_s(self):
        '''
        :return: the filter Q as S
        '''
        return 1 / ((((1 / self.q) ** 2 - 2) / ((10 ^ (self.gain / 40)) + 1 / (10 ^ (self.gain / 40)))) + 1)


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

    @property
    def filter_type(self):
        return 'LPF1'

    def _compute_coeffs(self):
        b1 = math.exp(-2.0 * math.pi * (self.freq / self.fs))
        a0 = 1.0 - b1
        a = np.array([a0, 0.0, 0.0], dtype=np.float64)
        b = np.array([1.0, -b1, 0.0])
        return a / a[0], b / a[0]


class FirstOrder_HighPass(Biquad):
    '''
    A one pole high pass filter.
    '''

    def __init__(self, fs, freq, q=DEFAULT_Q):
        super().__init__(fs, freq, q)

    @property
    def filter_type(self):
        return 'HPF1'

    def _compute_coeffs(self):
        b1 = math.exp(-2.0 * math.pi * (0.5 - self.freq / self.fs))
        a0 = 1.0 + b1
        a = np.array([a0, 0.0, 0.0], dtype=np.float64)
        b = np.array([1.0, -b1, 0.0])
        return a / a[0], b / a[0]


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

    @property
    def filter_type(self):
        return 'LPF2'

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

    @property
    def filter_type(self):
        return 'HPF2'

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

    def __init__(self, filters=None):
        self.__filters = filters if filters is not None else []

    def __getitem__(self, i):
        return self.__filters[i]

    def __len__(self):
        return len(self.__filters)

    def __repr__(self):
        return 'woot'

    @property
    def filter_type(self):
        return 'Complex'

    def add(self, filter):
        self.__filters.append(filter)

    def remove(self, indices):
        self.__filters = [filter for idx, filter in enumerate(self.__filters) if idx not in indices]

    def getTransferFunction(self):
        '''
        Computes the transfer function of the filter.
        :return: the transfer function.
        '''
        responses = [x.getTransferFunction() for x in self.__filters]
        return ComplexData(self.__repr__(), responses[0].x, reduce((lambda x, y: x * y), [r.y for r in responses]))


class FilterType(Enum):
    BUTTERWORTH = 'BW',
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
        self.freq = freq
        if self.type is FilterType.LINKWITZ_RILEY:
            if self.order % 2 != 0:
                raise ValueError("LR filters must be even order")
        if self.order == 0:
            raise ValueError("Filter cannot have order = 0")
        super().__init__(self._calculate_biquads())
        self.__filter_type = f"{filter_type.value[0]}{order}"

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
                # approach taken from http://www.earlevel.com/main/2016/09/29/cascading-filters/
                biquads = []
                pairs = self.order >> 1
                odd_poles = self.order & 1
                pole_inc = np.math.pi / self.order
                first_angle = pole_inc
                if not odd_poles:
                    first_angle /= 2
                else:
                    biquads.append(self.__bw1(self.fs, self.freq, 0.5))
                biquads += [self.__bw2(self.fs, self.freq, 1.0 / (2.0 * math.cos(first_angle + x * pole_inc))) for x in
                            range(0, pairs)]
                return biquads
        elif self.type is FilterType.LINKWITZ_RILEY:
            return [self.__bw2(self.fs, self.freq) for _ in range(0, int(self.order / 2))]
        else:
            raise ValueError("Unknown filter type " + str(self.type))


class ComplexLowPass(CompoundPassFilter):
    '''
    A low pass filter of different types and orders that are implemented using one or more biquads.
    '''

    def __init__(self, filter_type, order, fs, freq):
        super().__init__(FirstOrder_LowPass, SecondOrder_LowPass, filter_type, order, fs, freq)


class ComplexHighPass(CompoundPassFilter):
    '''
    A high pass filter of different types and orders that are implemented using one or more biquads.
    '''

    def __init__(self, filter_type, order, fs, freq):
        super().__init__(FirstOrder_HighPass, SecondOrder_HighPass, filter_type, order, fs, freq)


class ComplexData:
    '''
    Value object for storing complex data.
    '''

    def __init__(self, name, x, y, scaleFactor=1):
        self.name = name
        self.x = x
        self.y = y
        self.scaleFactor = scaleFactor

    def getMagnitude(self, ref):
        y = np.abs(self.y) * self.scaleFactor / ref
        return XYData(self.name, self.x, 20 * np.log10(y))

    def getPhase(self):
        return XYData(self.name, self.x, np.angle(self.y))


class XYData:
    '''
    Value object for showing data on a magnitude graph.
    '''

    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y

    def normalise(self, target):
        '''
        Normalises the y value against the target y.
        :param target: the target.
        :return: a normalised XYData.
        '''
        return XYData(self.name, self.x, self.y - target.y)
