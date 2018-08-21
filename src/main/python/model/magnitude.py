import logging
import time
from math import log10

import numpy as np
from matplotlib.ticker import EngFormatter, Formatter, NullFormatter

from model.filter import COMBINED

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

    def __init__(self, chart, dataProvider):
        self.__chart = chart
        self.__axes = self.__chart.canvas.figure.add_subplot(111)
        self.__dataProvider = dataProvider
        self.__curves = {}
        self.__dBRange = 40
        self.__update_y_lim(np.zeros(1))
        self.__axes.set_xlim(left=2, right=250)
        self.__axes.set_ylabel('dBFS')
        self.__axes.set_xlabel('Hz')
        self.__legend = None
        self.__legend_cid = None
        self.__configureFreqAxisFormatting()
        self.__axes.grid(linestyle='-', which='major', linewidth=1, alpha=0.5)
        self.__axes.grid(linestyle='--', which='minor', linewidth=1, alpha=0.5)

    def __repr__(self):
        return 'magnitude'

    def display(self):
        '''
        Updates the contents of the magnitude chart
        '''
        start = time.time()
        data = self.__dataProvider.getMagnitudeData()
        for idx, x in enumerate(data):
            if x.name == COMBINED:
                colour = 'k'
            else:
                colour = self.__chart.getColour(idx, len(data))
            self.__create_or_update_curve(x, colour)
        curve_names = [x.name for x in data]
        to_delete = [curve for name, curve in self.__curves.items() if name not in curve_names]
        for curve in to_delete:
            curve.remove()
            del self.__curves[curve.get_label()]
        if len(data) > 0:
            self.__update_y_lim(np.concatenate([x.y for x in data]))
        self.__makeClickableLegend()
        mid = time.time()
        self.__chart.canvas.draw()
        end = time.time()
        logger.debug(f"Calc : {round(mid-start,3)}s Redraw: {round(end-mid,3)}s")

    def __update_y_lim(self, data):
        self.__configureFreqAxisFormatting()
        ymax, ymin = calculate_dBFS_Scales(data, maxRange=self.__dBRange)
        self.__axes.set_ylim(bottom=ymin, top=ymax)

    def __create_or_update_curve(self, data, colour):
        curve = self.__curves.get(data.name, None)
        if curve:
            curve.set_data(data.x, data.y)
        else:
            self.__curves[data.name] = self.__axes.semilogx(data.x, data.y,
                                                            linewidth=2,
                                                            antialiased=True,
                                                            linestyle='solid',
                                                            color=colour,
                                                            label=data.name)[0]

    def __configureFreqAxisFormatting(self):
        hzFormatter = EngFormatter(places=0)
        self.__axes.get_xaxis().set_major_formatter(hzFormatter)
        self.__axes.get_xaxis().set_minor_formatter(PrintFirstHalfFormatter(hzFormatter))

    def __makeClickableLegend(self):
        '''
        Add a legend that allows you to make a line visible or invisible by clicking on it.
        ripped from https://matplotlib.org/2.0.0/examples/event_handling/legend_picking.html
        and https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
        '''
        if self.__legend is not None:
            self.__legend.remove()
        if self.__legend_cid is not None:
            self.__chart.canvas.mpl_disconnect(self.__legend_cid)

        lines = self.__curves.values()
        if len(lines) > 0:
            ncol = int(len(lines) / 3) if len(lines) % 3 == 0 else int(len(lines) / 3) + 1
            self.__legend = self.__axes.legend(lines, [l.get_label() for l in lines], loc=3, ncol=ncol, fancybox=True,
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
