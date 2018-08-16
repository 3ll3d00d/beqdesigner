from math import log10

import numpy as np
from matplotlib.ticker import EngFormatter, Formatter, NullFormatter

from model.filter import COMBINED


def calculate_dBFS_Scales(data, maxRange=60):
    '''
    Calculates the min/max in the data.
    :param data: the data.
    :param maxRange: the max range.
    :return: max, min
    '''
    vmax = np.math.ceil(np.nanmax(data))
    # coerce max to a round value
    multiple = 5 if maxRange <= 30 else 10
    if vmax % multiple != 0:
        vmax = (vmax - vmax % multiple) + multiple
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

    def __init__(self, chart, filterModel):
        self.__chart = chart
        self.__axes = self.__chart.canvas.figure.add_subplot(111)
        self.__filterModel = filterModel
        self.__curves = {}
        self.__dBRange = 40
        self._update_y_lim(np.zeros(1))
        self.__axes.set_xlim(left=2, right=500)
        self.configureFreqAxisFormatting()
        self.__axes.grid(linestyle='-', which='major', linewidth=1, alpha=0.5)
        self.__axes.grid(linestyle='--', which='minor', linewidth=1, alpha=0.5)

    def __repr__(self):
        return 'filter'

    def display(self):
        '''
        Updates the contents of the magnitude chart
        '''
        data = self.__filterModel.getMagnitudeData()
        for idx, x in enumerate(data):
            if x.name == COMBINED:
                colour = 'k'
            else:
                colour = self.__chart.getColour(idx, len(data))
            self._create_or_update_curve(x, colour)
        self._update_y_lim(np.concatenate([x.y for x in data]))
        self.makeClickableLegend()
        self.__chart.canvas.draw()

    def _update_y_lim(self, data):
        self.configureFreqAxisFormatting()
        ymax, ymin = calculate_dBFS_Scales(data, maxRange=self.__dBRange)
        self.__axes.set_ylim(bottom=ymin, top=ymax)

    def _create_or_update_curve(self, data, colour):
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

    def configureFreqAxisFormatting(self):
        hzFormatter = EngFormatter(places=0)
        self.__axes.get_xaxis().set_major_formatter(hzFormatter)
        self.__axes.get_xaxis().set_minor_formatter(PrintFirstHalfFormatter(hzFormatter))

    def makeClickableLegend(self):
        '''
        Add a legend that allows you to make a line visible or invisible by clicking on it.
        ripped from https://matplotlib.org/2.0.0/examples/event_handling/legend_picking.html
        and https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
        '''
        lines = self.__curves.values()
        legend = self.__axes.legend(lines, [l.get_label() for l in lines], loc=8, fancybox=True, shadow=True)
        lined = dict()
        for legline, origline in zip(legend.get_lines(), lines):
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

        self.__chart.canvas.mpl_connect('pick_event', onpick)
