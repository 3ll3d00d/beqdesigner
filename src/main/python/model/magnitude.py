import logging
import time
from math import log10

import numpy as np
from PyQt5.QtWidgets import QDialog
from matplotlib.ticker import EngFormatter, Formatter, NullFormatter, FixedLocator, LinearLocator

from ui.limits import Ui_graphLayoutDialog

logger = logging.getLogger('magnitude')


class PrintFirstHalfFormatter(Formatter):
    '''
    A custom formatter which uses a NullFormatter for some labels and delegates to another formatter for others.
    '''

    def __init__(self, other, maxVal=5):
        self.__other = other
        self.__null = NullFormatter()
        self.__max = log10(maxVal)

    def __call__(self, x, pos=None):
        func = self.__other if self.shouldShow(x) else self.__null
        return func(x, pos)

    def shouldShow(self, x):
        return log10(x) % 1 <= self.__max


class MagnitudeModel:
    '''
    Allows a set of filters to be displayed on a chart as magnitude responses.
    '''

    def __init__(self, name, chart, primaryDataProvider, primaryName, secondaryDataProvider=None, secondaryName=None):
        self.__name = name
        self.__chart = chart
        self.__primary = primaryDataProvider
        self.__primary_axes = self.__chart.canvas.figure.add_subplot(111)
        self.__primary_axes.set_ylabel(f"dBFS ({primaryName})")
        self.__primary_reference_curve = None
        self.__primary_curves = {}
        self.__secondary = secondaryDataProvider
        self.__secondary_curves = {}
        self.__secondary_reference_curve = None
        if self.__secondary is not None:
            self.__secondary_axes = self.__primary_axes.twinx()
            self.__secondary_axes.set_ylabel(f"dBFS ({secondaryName})")
        else:
            self.__secondary_axes = None
        # TODO default to half nyquist
        self.limits = Limits(self.__chart.canvas, self.__primary_axes, 60.0, x=(2, 250), axes_2=self.__secondary_axes)
        self.limits.propagate_to_axes(draw=True)
        self.__legend = None
        self.__legend_cid = None
        self.__primary_axes.grid(linestyle='-', which='major', linewidth=1, alpha=0.5)
        self.__primary_axes.grid(linestyle='--', which='minor', linewidth=1, alpha=0.5)

    def __repr__(self):
        return self.__name

    def show_limits(self):
        '''
        Shows the limits dialog.
        '''
        LimitsDialog(self.limits).exec()

    def get_curve_names(self, primary=True):
        '''
        :param primary: if true get the primary curves.
        :return: the names of all the curves in the chart.
        '''
        return list(self.__primary_curves.keys()) if primary else list(self.__secondary_curves.keys())

    def display(self):
        '''
        Updates the contents of the magnitude chart
        '''
        start = time.time()
        self.__display_curves(self.__primary, self.__primary_curves, self.__primary_axes,
                              self.__primary_reference_curve)
        if self.__secondary is not None:
            self.__display_curves(self.__secondary, self.__secondary_curves, self.__secondary_axes,
                                  self.__secondary_reference_curve)
        self.limits.configure_freq_axis()
        self.__make_legend()
        mid = time.time()
        self.__chart.canvas.draw()
        end = time.time()
        logger.debug(f"{self} Calc : {round(mid-start,3)}s Redraw: {round(end-mid,3)}s")

    def __display_curves(self, dataProvider, curves, axes, reference_curve):
        data = dataProvider.getMagnitudeData(reference=reference_curve)
        for x in data:
            self.__create_or_update_curve(axes, curves, x)
        curve_names = [x.name for x in data]
        to_delete = [curve for name, curve in curves.items() if name not in curve_names]
        for curve in to_delete:
            curve.remove()
            del curves[curve.get_label()]
        if len(data) > 0:
            self.limits.on_data_change(axes, np.concatenate([x.y for x in data]))

    def __create_or_update_curve(self, axes, curves, data):
        curve = curves.get(data.name, None)
        if curve:
            curve.set_data(data.x, data.y)
        else:
            curves[data.name] = axes.semilogx(data.x, data.y,
                                              linewidth=2,
                                              antialiased=True,
                                              linestyle='solid',
                                              color=data.colour,
                                              label=data.name)[0]

    def __make_legend(self):
        '''
        Add a legend that allows you to make a line visible or invisible by clicking on it.
        ripped from https://matplotlib.org/2.0.0/examples/event_handling/legend_picking.html
        and https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
        '''
        if self.__legend is not None:
            self.__legend.remove()
        if self.__legend_cid is not None:
            self.__chart.canvas.mpl_disconnect(self.__legend_cid)

        lines = list(self.__primary_curves.values()) + list(self.__secondary_curves.values())
        if len(lines) > 0:
            ncol = int(len(lines) / 3) if len(lines) % 3 == 0 else int(len(lines) / 3) + 1
            self.__legend = self.__primary_axes.legend(lines, [l.get_label() for l in lines], loc=3, ncol=ncol,
                                                       fancybox=True,
                                                       shadow=True)
            lined = dict()
            for legline, origline in zip(self.__legend.get_lines(), lines):
                legline.set_picker(5)  # 5 pts tolerance
                lined[legline] = origline

            def onpick(event):
                # on the pick event, find the orig line corresponding to the legend proxy line, and toggle the visibility
                legline = event.artist
                origline = lined[legline]
                vis = not origline.get_visible()
                origline.set_visible(vis)
                # Change the alpha on the line in the legend so we can see what lines have been toggled
                if vis:
                    legline.set_alpha(1.0)
                else:
                    legline.set_alpha(0.2)
                self.__chart.canvas.draw()

            self.__legend_cid = self.__chart.canvas.mpl_connect('pick_event', onpick)

    def normalise(self, primary=True, curve=None):
        '''
        Redraws with the normalised data.
        :param curve: the reference curve (if any).
        '''
        if primary == True:
            self.__primary_reference_curve = curve
        else:
            self.__secondary_reference_curve = curve
        self.display()


class Limits:
    '''
    Value object to hold graph limits to decouple the dialog from the chart.
    '''

    def __init__(self, canvas, axes_1, default_y_range, x, axes_2=None):
        self.__canvas = canvas
        self.__default_y_range = default_y_range
        self.axes_1 = axes_1
        self.x_scale = 'log'
        self.__x_scale_changed = False
        self.x_min = x[0]
        self.x_max = x[1]
        numticks = 13
        self.axes_1.yaxis.set_major_locator(LinearLocator(numticks))
        self.y1_min, self.y1_max = self.calculate_dBFS_scales(np.zeros(1))
        if axes_2 is not None:
            self.y2_min, self.y2_max = self.calculate_dBFS_scales(np.zeros(1))
            self.axes_2 = axes_2
            self.axes_2.yaxis.set_major_locator(LinearLocator(numticks))
        else:
            self.axes_2 = self.y2_min = self.y2_max = None
        self.propagate_to_axes()

    def propagate_to_axes(self, draw=False):
        '''
        Updates the chart with the current values.
        '''
        self.axes_1.set_xlim(left=self.x_min, right=self.x_max)
        self.axes_1.set_ylim(bottom=self.y1_min, top=self.y1_max)
        if self.axes_2 is not None:
            self.axes_2.set_ylim(bottom=self.y2_min, top=self.y2_max)
            # ensure the ticks are aligned with the primary axis ticks
            # from https://stackoverflow.com/questions/45037386/trouble-aligning-ticks-for-matplotlib-twinx-axes
            # f = lambda x: self.y2_min + (x - self.y1_min) / (self.y1_max - self.y1_min) * (self.y2_max - self.y2_min)
            # ticks = f(self.axes_1.get_yticks())
            # self.axes_2.yaxis.set_major_locator(FixedLocator(ticks))
        self.configure_freq_axis()
        if draw:
            self.__canvas.draw()

    def update(self, x_min=None, x_max=None, y1_min=None, y1_max=None, y2_min=None, y2_max=None, x_scale=None, draw=False):
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
            self.__x_scale_changed = True
        self.propagate_to_axes(draw)

    def calculate_dBFS_scales(self, data):
        '''
        Calculates the min/max in the data.
        :param data: the data.
        :param max_range: the max range.
        :return: min, max
        '''
        vmax = np.math.ceil(np.nanmax(data))
        # coerce max to a round value
        multiple = 5 if self.__default_y_range <= 40 else 10
        if vmax % multiple != 0:
            vmax = (vmax - vmax % multiple) + multiple
        else:
            vmax += multiple
        return vmax - self.__default_y_range, vmax

    def on_data_change(self, axes, data):
        if axes is self.axes_1:
            if self.is_auto_1():
                self.y1_min, self.y1_max = self.calculate_dBFS_scales(data)
                self.propagate_to_axes(draw=False)
        elif axes is self.axes_2:
            if self.is_auto_2():
                self.y2_min, self.y2_max = self.calculate_dBFS_scales(data)
                self.propagate_to_axes(draw=False)
        else:
            raise ValueError(f"Unknown axes provided to on_data_change {axes}")

    def is_auto_1(self):
        '''
        :return: True if y_1 is on auto update.
        '''
        return True

    def is_auto_2(self):
        '''
        :return: True if y_2 is on auto update.
        '''
        return True

    def configure_freq_axis(self):
        '''
        sets up the freq axis formatters.
        :return:
        '''
        if self.__x_scale_changed:
            logger.debug(f"Reconfiguring Freq Axis to {self.x_scale}")
            self.axes_1.set_xscale(self.x_scale)
            hzFormatter = EngFormatter(places=0)
            self.axes_1.get_xaxis().set_major_formatter(hzFormatter)
            self.axes_1.get_xaxis().set_minor_formatter(PrintFirstHalfFormatter(hzFormatter))
            self.axes_1.set_xlabel('Hz')
            self.__x_scale_changed = False


class LimitsDialog(QDialog, Ui_graphLayoutDialog):
    '''
    Provides some basic chart controls.
    '''

    def __init__(self, limits, parent=None):
        super(LimitsDialog, self).__init__(parent)
        self.setupUi(self)
        self.__limits = limits
        self.hzLog.setChecked(limits.x_scale == 'log')
        self.xMin.setValue(self.__limits.x_min)
        self.xMax.setValue(self.__limits.x_max)
        self.y1Min.setValue(self.__limits.y1_min)
        self.y1Max.setValue(self.__limits.y1_max)
        self.y2Min.setValue(self.__limits.y2_min)
        self.y2Max.setValue(self.__limits.y2_max)

    def changeLimits(self):
        '''
        Updates the chart limits.
        '''
        self.__limits.update(x_min=self.xMin.value(), x_max=self.xMax.value(), y1_min=self.y1Min.value(),
                             y1_max=self.y1Max.value(), y2_min=self.y2Min.value(), y2_max=self.y2Max.value(),
                             x_scale='log' if self.hzLog.isChecked() else 'linear', draw=True)
