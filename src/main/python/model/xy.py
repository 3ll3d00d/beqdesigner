import logging
import time

import numpy as np
from PyQt5.QtCore import QSettings
from scipy.interpolate import PchipInterpolator

from model.preferences import DISPLAY_SMOOTH_GRAPHS, Preferences

SAVGOL_WINDOW_LENGTH = 101
SAVGOL_POLYORDER = 7

logger = logging.getLogger('xy')
preferences = Preferences(QSettings("3ll3d00d", "beqdesigner"))


class MagnitudeData:
    '''
    Value object for showing data on a magnitude graph.
    '''

    def __init__(self, name, description, x, y, colour=None, linestyle='-', smooth_type=None):
        self.__name = name
        self.__description = description
        self.__x_normal_x = None
        self.__x_normal_y = None
        results = smooth(x, y, smooth_type)
        self.x = results[0]
        self.y = np.nan_to_num(results[1])
        self.__equal_energy_adjusted = None
        self.colour = colour
        self.linestyle = linestyle
        self.__normalised_cache = {}
        self.__smooth_type = smooth_type

    @property
    def name(self):
        if self.__description is None:
            return self.__name
        else:
            return f"{self.__name}_{self.__description}"

    @property
    def description(self):
        return self.__description

    @description.setter
    def description(self, description):
        self.__description = description

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
        return f"MagnitudeData: {self.name} - {self.x.size} - {self.colour}"

    def __eq__(self, o: object) -> bool:
        equal = self.__class__.__name__ == o.__class__.__name__
        equal &= self.name == o.name
        # allow tolerance because of the way we serialise to save space
        equal &= np.allclose(self.x, o.x, rtol=1e-5, atol=1e-5)
        equal &= np.allclose(self.y, o.y, rtol=1e-5, atol=1e-5)
        equal &= self.colour == o.colour
        equal &= self.linestyle == o.linestyle
        return equal

    def normalise(self, target):
        '''
        Normalises the y value against the target y.
        :param target: the target.
        :return: a normalised MagnitudeData.
        '''
        if target.name not in self.__normalised_cache:
            logger.debug(f"Normalising {self.name} against {target.name}")
            self_step = self.x[1] - self.x[0]
            target_step = target.x[1] - target.x[0]
            if self_step == target_step:
                count = min(self.x.size, target.x.size) - 1
                self.__normalised_cache[target.name] = MagnitudeData(self.__name,
                                                                     self.__description,
                                                                     self.x[0:count],
                                                                     self.y[0:count] - target.y[0:count],
                                                                     colour=self.colour,
                                                                     linestyle=self.linestyle,
                                                                     smooth_type=self.__smooth_type)
            else:
                if self.x[-1] == target.x[-1]:
                    # same max so upsample to the more precise one
                    if self_step < target_step:
                        new_x = self.x
                        new_y = self.y - interp(target.x, target.y, self.x)[1]
                    else:
                        new_x = target.x
                        new_y = self.y - interp(self.x, self.y, target.x)[1]
                elif self.x[-1] > target.x[-1]:
                    # restrict the self data range to the limits of the target
                    capped_x = self.x[self.x <= target.x[-1]]
                    capped_y = self.y[0:capped_x.size]
                    if self_step < target_step:
                        new_x = capped_x
                        new_y = capped_y - interp(target.x, target.y, capped_x)[1]
                    else:
                        new_x = target.x
                        new_y = interp(capped_x, capped_y, target.x)[1] - target.y
                else:
                    # restrict the target data range to the limits of the self
                    capped_x = target.x[target.x <= self.x[-1]]
                    capped_y = target.y[0:capped_x.size]
                    if self_step < target_step:
                        new_x = self.x
                        new_y = self.y - interp(capped_x, capped_y, self.x)[1]
                    else:
                        new_x = capped_x
                        new_y = interp(self.x, self.y, target.x)[1] - target.y
                self.__normalised_cache[target.name] = MagnitudeData(self.__name,
                                                                     self.__description,
                                                                     new_x,
                                                                     new_y,
                                                                     colour=self.colour,
                                                                     linestyle=self.linestyle,
                                                                     smooth_type=self.__smooth_type)

        return self.__normalised_cache[target.name]

    def filter(self, filt):
        '''
        Adds filt.y to the data.y as we're dealing in the frequency domain. Interpolates the smaller xy if required so
        we can just add them together.
        :param filt: the filter in MagnitudeData form.
        :return: the filtered response.
        '''
        if self.x.size != filt.x.size:
            logger.debug(f"Interpolating filt {filt.x.size} vs self {self.x.size}")
            if self.x.size > filt.x.size:
                _, interp_y = interp(filt.x, filt.y, self.x)
                return MagnitudeData(self.__name, f"{self.__description}-filtered", self.x, self.y + interp_y,
                                     colour=self.colour,
                                     linestyle='-',
                                     smooth_type=self.__smooth_type)
            else:
                _, interp_y = interp(self.x, self.y, filt.x)
                return MagnitudeData(self.__name, f"{self.__description}-filtered", filt.x, filt.y + interp_y,
                                     colour=self.colour,
                                     linestyle='-',
                                     smooth_type=self.__smooth_type)
        else:
            return MagnitudeData(self.__name, f"{self.__description}-filtered", self.x, self.y + filt.y,
                                 colour=self.colour,
                                 linestyle='-',
                                 smooth_type=self.__smooth_type)

    def smooth(self, smooth_type):
        '''
        Creates a new MagnitudeData with the specified smoothing applied.
        :param smooth_type: the smoothing type.
        :return: the smoothed data.
        '''
        if smooth_type is None:
            return self
        return MagnitudeData(self.__name, self.__description, self.x, self.y,
                             colour=self.colour,
                             linestyle=self.linestyle,
                             smooth_type=smooth_type)

    def with_equal_energy_adjustment(self):
        ''' returns the equal energy adjusted version of this data. '''
        if self.__equal_energy_adjusted is None:
            x = np.linspace(self.x[1], self.x[-1], num=self.x.size)
            adjustment = np.log10(x) * 10
            self.__equal_energy_adjusted = MagnitudeData(self.__name, self.__description, self.x, self.y + adjustment,
                                                         colour=self.colour,
                                                         linestyle=self.linestyle,
                                                         smooth_type=self.__smooth_type)
        return self.__equal_energy_adjusted


def smooth(x, y, smooth_type=None):
    '''
    Smooths the data.
    :param x: the frequencies.
    :param y: magnitude.
    :param smooth_type: type of smoothing.
    :return: smoothed data.
    '''
    if smooth_type is not None:
        try:
            int(smooth_type)
        except:
            return smooth_savgol(x, y, smooth_type)
        return smooth_octave(x, y, smooth_type)
    else:
        return x, y


def smooth_octave(x, y, smooth_type):
    '''
    Performs fractional octave smoothing.
    :param x: frequencies.
    :param y: magnitude.
    :param smooth_type: fractional octave.
    :return: the smoothed data.
    '''
    from acoustics.smooth import fractional_octaves
    from model.signal import db_to_amplitude, amplitude_to_db
    octave_x, smoothed_y = fractional_octaves(x, db_to_amplitude(y), fraction=smooth_type)
    return octave_x.center, amplitude_to_db(smoothed_y)


def smooth_savgol(x, y, smooth_type):
    '''
    Performs Savitzky-Golay smoothing.
    :param x: frequencies.
    :param y: magnitude.
    :param smooth_type: fractional octave.
    :return: the smoothed data.
    '''
    from scipy.signal import savgol_filter
    tokens = smooth_type.split('/')
    if len(tokens) == 1:
        wl = SAVGOL_WINDOW_LENGTH
        poly = SAVGOL_POLYORDER
    else:
        wl = int(tokens[1])
        poly = int(tokens[2])
    smoothed_y = savgol_filter(y, wl, poly)
    return x, smoothed_y


def must_interpolate(smooth_type):
    '''
    determines whether the smoothing type means we need to interpolate.
    :param smooth_type: the smoothing type.
    :return: true if we have to interpolate the xy data.
    '''
    if smooth_type is None:
        return False
    else:
        try:
            int(smooth_type)
            return True
        except:
            return False


def interp(x1, y1, x2):
    ''' Interpolates xy based on the preferred smoothing style. '''
    start = time.time()
    smooth = preferences.get(DISPLAY_SMOOTH_GRAPHS)
    if smooth:
        cs = PchipInterpolator(x1, y1)
        y2 = cs(x2)
    else:
        y2 = np.interp(x2, x1, y1)
    end = time.time()
    logger.debug(f"Interpolation from {len(x1)} to {len(x2)} in {round((end - start) * 1000, 3)}ms")
    return x2, y2

