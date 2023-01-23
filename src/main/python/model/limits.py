import logging
import math
from math import log10

import numpy as np
from qtpy.QtWidgets import QDialog
from matplotlib.ticker import EngFormatter, Formatter, NullFormatter, MaxNLocator, AutoMinorLocator
from qtpy import QtWidgets

from ui.limits import Ui_graphLayoutDialog
from ui.values import Ui_valuesDialog

logger = logging.getLogger('limits')


class PrintFirstHalfFormatter(Formatter):
    '''
    A custom formatter which uses a NullFormatter for some labels and delegates to another formatter for others.
    '''

    def __init__(self, other, max_val=5):
        self.__other = other
        self.__null = NullFormatter()
        self.__max = log10(max_val)

    def __call__(self, x, pos=None):
        func = self.__other if self.shouldShow(x) else self.__null
        return func(x, pos)

    def shouldShow(self, x):
        return log10(x) % 1 <= self.__max


class PhaseRangeCalculator:

    def calculate(self, y_range):
        return -180.0, 360.0


class ImpulseRangeCalculator:

    def calculate(self, y_range):
        return -1.0, 1.0


class DecibelRangeCalculator:
    '''
    A calculator for y axis ranges of dBFS signals.
    '''

    def __init__(self, default_range=60, expand=False):
        self.__default_range = default_range
        self.__expand_range = expand

    @property
    def expand_range(self):
        return self.__expand_range

    @expand_range.setter
    def expand_range(self, expand_range):
        self.__expand_range = expand_range

    def calculate(self, y_range):
        '''
        Calculates the min/max in the data.
        :param data: the data.
        :param max_range: the max range.
        :return: min, max
        '''
        vmax = y_range[1]
        # coerce max to a round value
        multiple = 5 if self.__default_range <= 40 else 10
        if not math.isclose(vmax % multiple, 0):
            vmax = (vmax - vmax % multiple) + multiple
        else:
            vmax += multiple
        return vmax - self.__range(y_range, self.__default_range), vmax

    def __range(self, data_range, target_range):
        if self.expand_range is True and data_range[1] - data_range[0] >= target_range - 10:
            return self.__range(data_range, target_range+10)
        else:
            return target_range


def configure_freq_axis(axes, x_scale):
    '''
    sets up the freq axis formatters (which you seem to have to call constantly otherwise matplotlib keeps
    reinstating the default log format)
    '''
    axes.set_xscale(x_scale)
    hzFormatter = EngFormatter(places=0)
    axes.get_xaxis().set_major_formatter(hzFormatter)
    axes.set_xlabel('Hz')
    if x_scale == 'log':
        axes.get_xaxis().set_minor_formatter(PrintFirstHalfFormatter(hzFormatter))
    else:
        axes.get_xaxis().set_major_locator(MaxNLocator(nbins=24, steps=[1, 2, 4, 5, 10], min_n_ticks=8))
        axes.get_xaxis().set_minor_locator(AutoMinorLocator(2))


class Limits:
    '''
    Value object to hold graph limits to decouple the dialog from the chart.
    '''

    def __init__(self, name, redraw_func, axes_1, x_lim, x_axis_configurer=configure_freq_axis,
                 y1_range_calculator=DecibelRangeCalculator(), x_scale='log', axes_2=None, y2_range_calculator=None):
        '''
        :param name: the name of the chart.
        :param redraw_func: redraws the owning canvas.
        :param axes_1: the primary y axes.
        :param default_y_range: y axis range (scalar).
        :param x_lim: x limits (min, max)
        :param x_scale: the x axis scale type (linear, log etc)
        :param axes_2: the secondary y axes (if any).
        '''
        self.name = name
        self.__redraw_func = redraw_func
        self.__configure_x_axis = x_axis_configurer
        self.__y1_range_calculator = y1_range_calculator
        self.__y2_range_calculator = y2_range_calculator if y2_range_calculator else y1_range_calculator
        self.axes_1 = axes_1
        self.x_scale = x_scale
        self.x_min = x_lim[0]
        self.x_max = x_lim[1]
        self.__y1_auto = True
        self.__y2_auto = True
        self.axes_1.yaxis.set_major_locator(MaxNLocator(nbins=24, steps=[1, 2, 5, 10], min_n_ticks=8))
        self.y1_min, self.y1_max = self.__y1_range_calculator.calculate((0, 0))
        if axes_2 is not None:
            self.y2_min, self.y2_max = self.__y2_range_calculator.calculate((0, 0))
            self.axes_2 = axes_2
            self.axes_2.yaxis.set_major_locator(MaxNLocator(nbins=24, steps=[1, 2, 5, 10], min_n_ticks=8))
        else:
            self.axes_2 = self.y2_min = self.y2_max = None
        self.propagate_to_axes()

    @property
    def y1_auto(self):
        return self.__y1_auto

    @y1_auto.setter
    def y1_auto(self, auto):
        self.__y1_auto = auto

    @property
    def y2_auto(self):
        return self.__y2_auto

    @y2_auto.setter
    def y2_auto(self, auto):
        self.__y2_auto = auto

    def configure_x_axis(self):
        '''
        Allows external code to update the x axis.
        '''
        self.__configure_x_axis(self.axes_1, self.x_scale)

    def propagate_to_axes(self, draw=False):
        '''
        Updates the chart with the current values.
        '''
        self.axes_1.set_xlim(left=self.x_min, right=self.x_max)
        self.axes_1.set_ylim(bottom=self.y1_min, top=self.y1_max)
        if self.axes_2 is not None:
            self.axes_2.set_ylim(bottom=self.y2_min, top=self.y2_max)
        self.__configure_x_axis(self.axes_1, self.x_scale)
        if draw:
            logger.debug(f"{self.name} Redrawing axes on limits change")
            self.__redraw_func()

    def set_expand_y(self, expand):
        '''
        sets the expand mode on the y range calculator.
        :param expand: true or false.
        '''
        if isinstance(self.__y1_range_calculator, DecibelRangeCalculator):
            self.__y1_range_calculator.expand_range = expand
        if isinstance(self.__y2_range_calculator, DecibelRangeCalculator):
            self.__y2_range_calculator.expand_range = expand

    def update(self, x_min=None, x_max=None, y1_min=None, y1_max=None, y2_min=None, y2_max=None, x_scale=None,
               draw=False):
        '''
        Accepts new values from the dialog and propagates that to the chart.
        :param x_min:
        :param x_max:
        :param y1_min:
        :param y1_max:
        :param y2_min:
        :param y2_max:
        '''
        if x_min is not None:
            self.x_min = x_min
        if x_max is not None:
            self.x_max = x_max
        if y1_min is not None:
            self.y1_min = y1_min
        if y1_max is not None:
            self.y1_max = y1_max
        if y2_min is not None:
            self.y2_min = y2_min
        if y2_min is not None:
            self.y2_max = y2_max
        if x_scale is not None and x_scale != self.x_scale:
            self.x_scale = x_scale
        self.propagate_to_axes(draw)

    def shift(self, x_min=None, x_max=None, y1_min=None, y1_max=None, y2_min=None, y2_max=None, draw=True):
        '''
        Shifts the specified axis by the given amount.
        :param x_min: lower x.
        :param x_max: upper x.
        :param y1_min: lower y1.
        :param y1_max: upper y1.
        :param y2_min: lower y2.
        :param y2_max: upper y2.
        '''
        if x_min is not None:
            self.x_min = self.x_min + x_min
        if x_max is not None:
            self.x_max = self.x_max + x_max
        if y1_min is not None:
            self.y1_min = self.y1_min + y1_min
        if y1_max is not None:
            self.y1_max = self.y1_max + y1_max
        if y2_min is not None:
            self.y2_min = self.y2_min + y2_min
        if y2_max is not None:
            self.y2_max = self.y2_max + y2_max
        self.propagate_to_axes(draw)

    def on_data_change(self, primary_range, secondary_range):
        '''
        Updates the y axes when the data changes.
        :param primary_range: the primary y range.
        :param secondary_range: the secondary y range.
        '''
        if self.y1_auto is True:
            new_min, new_max = self.__y1_range_calculator.calculate(primary_range)
            y1_changed = new_min != self.y1_min or new_max != self.y1_max
            if y1_changed:
                logger.debug(f"{self.name} y1 axis changed from {self.y1_min}/{self.y1_max} to {new_min}/{new_max}")
            self.y1_min = new_min
            self.y1_max = new_max
            if y1_changed:
                self.axes_1.set_ylim(bottom=self.y1_min, top=self.y1_max)
        if self.y2_auto is True and self.axes_2 is not None:
            new_min, new_max = self.__y2_range_calculator.calculate(secondary_range)
            y2_changed = new_min != self.y2_min or new_max != self.y2_max
            if y2_changed:
                logger.debug(f"{self.name} y2 axis changed from {self.y2_min}/{self.y2_max} to {new_min}/{new_max}")
            self.y2_min = new_min
            self.y2_max = new_max
            if y2_changed:
                self.axes_2.set_ylim(bottom=self.y2_min, top=self.y2_max)


class ValuesDialog(QDialog, Ui_valuesDialog):
    '''
    Provides a mechanism for looking at the values listed in the chart without taking up screen estate.
    '''

    def __init__(self, data):
        super(ValuesDialog, self).__init__()
        self.setupUi(self)
        self.__step = 1
        self.__min_x, self.__max_x, self.data = self.__interpolate_data(self.__step, data)
        self.valueFields = []
        for idx, xy in enumerate(data):
            label = QtWidgets.QLabel(self)
            label.setObjectName(f"label{idx+1}")
            label.setText(xy.get_label())
            self.formLayout.setWidget(idx + 1, QtWidgets.QFormLayout.LabelRole, label)
            lineEdit = QtWidgets.QLineEdit(self)
            lineEdit.setEnabled(False)
            lineEdit.setObjectName(f"value{idx+1}")
            self.valueFields.append(lineEdit)
            self.formLayout.setWidget(idx + 1, QtWidgets.QFormLayout.FieldRole, lineEdit)
        if len(data) > 0:
            xdata = data[0].get_xdata()
            self.freq.setMinimum(xdata[0])
            self.freq.setSingleStep(self.__step)
            self.freq.setMaximum(xdata[-1])
            self.freq.setValue(xdata[0])
            self.freq.setEnabled(True)
        else:
            self.freq.setEnabled(False)

    @staticmethod
    def __interpolate_data(step, data):
        '''
        Interpolates the data using a simple 1D interpolation so the values becomes a simple lookup.
        :param step: the step in x values.
        :param data: the input data.
        :return: min_x, max_x, the interpolated data.
        '''
        min_x = data[0].get_xdata()[0]
        max_x = max(d.get_xdata()[-1] for d in data)
        x2 = np.arange(min_x, max_x + step, step)
        return min_x, max_x, [(x2, np.interp(x2, d.get_xdata(), d.get_ydata())) for d in data]

    def updateValues(self, freq):
        '''
        propagates the freq value change.
        '''
        freq_idx = int(freq / self.__step)
        for idx, xy in enumerate(self.data):
            val = xy[1][freq_idx]
            self.valueFields[idx].setText(str(round(val, 3)))


class LimitsDialog(QDialog, Ui_graphLayoutDialog):
    '''
    Provides some basic chart controls.
    '''

    def __init__(self, limits, x_min=1, x_max=24000, y1_min=-200, y1_max=200, y2_min=-200, y2_max=200, parent=None):
        super(LimitsDialog, self).__init__(parent)
        self.setupUi(self)
        self.__limits = limits
        self.hzLog.setChecked(limits.x_scale == 'log')
        self.xMin.setMinimum(x_min)
        self.xMin.setMaximum(x_max - 1)
        self.xMin.setValue(self.__limits.x_min)
        self.xMax.setMinimum(x_min + 1)
        self.xMax.setMaximum(x_max)
        self.xMax.setValue(self.__limits.x_max)
        self.y1Min.setMinimum(y1_min)
        self.y1Min.setMaximum(y1_max - 1)
        self.y1Min.setValue(self.__limits.y1_min)
        self.y1Max.setMinimum(y1_min + 1)
        self.y1Max.setMaximum(y1_max)
        self.y1Max.setValue(self.__limits.y1_max)
        if limits.axes_2 is not None:
            self.y2Min.setMinimum(y2_min)
            self.y2Min.setMaximum(y2_max - 1)
            self.y2Min.setValue(self.__limits.y2_min)
            self.y2Max.setMinimum(y2_min + 1)
            self.y2Max.setMaximum(y2_max)
            self.y2Max.setValue(self.__limits.y2_max)
        else:
            self.y2Min.setEnabled(False)
            self.y2Max.setEnabled(False)

    def changeLimits(self):
        '''
        Updates the chart limits.
        '''
        self.__limits.update(x_min=self.xMin.value(), x_max=self.xMax.value(), y1_min=self.y1Min.value(),
                               y1_max=self.y1Max.value(), y2_min=self.y2Min.value(), y2_max=self.y2Max.value(),
                               x_scale='log' if self.hzLog.isChecked() else 'linear', draw=True)

    def fullRangeLimits(self):
        ''' changes the x limits to show a full range signal '''
        self.xMin.setValue(20)
        self.xMax.setValue(20000)
        self.hzLog.setChecked(True)

    def bassLimits(self):
        ''' changes the x limits to show a bass limited signal '''
        self.xMin.setValue(1)
        self.xMax.setValue(160)
        self.hzLog.setChecked(False)