import logging
import time
from math import log10

import numpy as np
from PyQt5.QtWidgets import QDialog
from matplotlib.ticker import EngFormatter, Formatter, NullFormatter

from ui.limits import Ui_graphLayoutDialog

logger = logging.getLogger('magnitude')


def calculate_dBFS_Scales(data, maxRange=60):
    '''
    Calculates the min/max in the data.
    :param data: the data.
    :param maxRange: the max range.
    :return: max, min
    '''
    vmax = np.math.ceil(np.nanmax(data))
    # coerce max to a round value
    multiple = 5 if maxRange <= 40 else 10
    if vmax % multiple != 0:
        vmax = (vmax - vmax % multiple) + multiple
    else:
        vmax += multiple
    return vmax, vmax - maxRange


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
        self.__dBRange = 40
        self.__primary = primaryDataProvider
        self.__primary_axes = self.__chart.canvas.figure.add_subplot(111)
        self.__primary_axes.set_ylabel(f"dBFS ({primaryName})")
        self.__primary_reference_curve = None
        self.__primary_curves = {}
        self.__configure_freq_axis()
        self.__update_y_lim(self.__primary_axes, np.zeros(1))
        self.__secondary = secondaryDataProvider
        self.__secondary_curves = {}
        self.__secondary_reference_curve = None
        if self.__secondary is not None:
            self.__secondary_axes = self.__primary_axes.twinx()
            self.__secondary_axes.set_ylabel(f"dBFS ({secondaryName})")
            self.__update_y_lim(self.__secondary_axes, np.zeros(1))
        # TODO set to half nyquist or make it a user selection
        self.__legend = None
        self.__legend_cid = None
        self.__primary_axes.grid(linestyle='-', which='major', linewidth=1, alpha=0.5)
        self.__primary_axes.grid(linestyle='--', which='minor', linewidth=1, alpha=0.5)

    def __repr__(self):
        return self.__name

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
        self.__configure_freq_axis()
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
            self.__update_y_lim(axes, np.concatenate([x.y for x in data]))

    def __update_y_lim(self, axes, data):
        ymax, ymin = calculate_dBFS_Scales(data, maxRange=self.__dBRange)
        axes.set_ylim(bottom=ymin, top=ymax)

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

    def __configure_freq_axis(self):
        hzFormatter = EngFormatter(places=0)
        self.__primary_axes.get_xaxis().set_major_formatter(hzFormatter)
        self.__primary_axes.get_xaxis().set_minor_formatter(PrintFirstHalfFormatter(hzFormatter))
        self.__primary_axes.set_xlim(left=2, right=250)
        self.__primary_axes.set_xlabel('Hz')

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
    def __init__(self, x_min, x_max, y_min, y_max, axes):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.__axes = axes

    def update(self, x_min, x_max, y_min, y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        # update the axes


class LimitsDialog(QDialog, Ui_graphLayoutDialog):
    '''
    Provides some basic chart controls.
    '''

    def __init__(self, mag, parent=None):
        super(LimitsDialog, self).__init__(parent)
        self.setupUi(self)
        self.__mag = mag
        bot, top = self.__mag.__axes.get_ylim()
        l, r = self.__mag.__axes.get_xlim()
        self.xMin.setValue(l)
        self.xMax.setValue(r)
        self.yMin.setValue(bot)
        self.yMax.setValue(top)

    def toggleLogScale(self):
        pass

    def changeLimits(self):
        '''
        Updates the chart limits.
        '''
        # TODO bridge this via the limits class
        # self.__limits.update(self.xMin.value(), self.xMax.value(), self.yMin.value(), self.yMax.value())
        self.__mag.__axes.set_ylim(bottom=self.yMin.value(), top=self.yMax.value())
        self.__mag.__axes.set_xlim(left=self.xMin.value(), right=self.xMax.value())
        self.__mag.__chart.canvas.draw()
