import logging
import math
from math import log10

from PyQt5.QtWidgets import QDialog
from matplotlib.ticker import EngFormatter, Formatter, NullFormatter, LinearLocator
from qtpy import QtWidgets

from ui.limits import Ui_graphLayoutDialog
from ui.values import Ui_valuesDialog

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


class AxesManager:
    def __init__(self, dataProvider, axes):
        self.__provider = dataProvider
        self.__axes = axes
        self.reference_curve = None
        self.__curves = {}
        self.__maxy = 0
        self.__miny = 0

    def artists(self):
        '''
        Gets the artists painting on this axes.
        :return: the artists.
        '''
        return list(self.__curves.values())

    def curve_names(self):
        '''
        The artist names.
        :return: the names.
        '''
        return list(self.__curves.keys())

    def display_curves(self):
        '''
        Displays the cached data on the specified axes, removing existing curves as required.
        :param data: the data.
        :param curves: the stored curves.
        :param axes: the axes on which to display.
        '''
        if self.__provider is not None:
            data = self.__provider.getMagnitudeData(reference=self.reference_curve)
            if len(data) > 0:
                curve_names = [self.__create_or_update_curve(x) for x in data if data is not None]
                self.__miny = math.floor(min([x.miny for x in data]))
                self.__maxy = math.ceil(max([x.maxy for x in data]))
            else:
                curve_names = []
            to_delete = [curve for name, curve in self.__curves.items() if name not in curve_names]
            for curve in to_delete:
                curve.remove()
                del self.__curves[curve.get_label()]

    def __create_or_update_curve(self, data):
        '''
        sets the data on the curve, creating a new artist if necessary.
        :param data: the data to set on the curve.
        :param the updated (or created) curve.
        :return the curve name.
        '''
        curve = self.__curves.get(data.name, None)
        if curve:
            if data.rendered is False:
                curve.set_data(data.x, data.y)
                data.rendered = True
            if data.linestyle != curve.get_linestyle():
                curve.set_linestyle(data.linestyle)
            if data.colour != curve.get_color():
                curve.set_color(data.colour)
        else:
            self.__curves[data.name] = self.__axes.semilogx(data.x, data.y,
                                                            linewidth=2,
                                                            antialiased=True,
                                                            linestyle=data.linestyle,
                                                            color=data.colour,
                                                            label=data.name)[0]
            data.rendered = True
        return data.name

    def get_ylimits(self):
        '''
        :return: min y, max y
        '''
        return self.__miny, self.__maxy

    def make_legend(self, lines, ncol):
        '''
        makes a legend for the given lines.
        :param lines: the lines to display
        :param ncol: the no of columns to put them in.
        :return the legend.
        '''
        return self.__axes.legend(lines, [l.get_label() for l in lines], loc=3, ncol=ncol, fancybox=True, shadow=True)


class MagnitudeModel:
    '''
    Allows a set of filters to be displayed on a chart as magnitude responses.
    '''

    def __init__(self, name, chart, primaryDataProvider, primaryName, secondaryDataProvider=None, secondaryName=None):
        self.__name = name
        self.__chart = chart
        primary_axes = self.__chart.canvas.figure.add_subplot(111)
        primary_axes.set_ylabel(f"dBFS ({primaryName})")
        primary_axes.grid(linestyle='-', which='major', linewidth=1, alpha=0.5)
        primary_axes.grid(linestyle='--', which='minor', linewidth=1, alpha=0.5)
        self.__dummy_artist = [primary_axes.semilogx([5], [0], visible=False)[0]]
        self.__primary = AxesManager(primaryDataProvider, primary_axes)
        if secondaryDataProvider is None:
            secondary_axes = None
        else:
            secondary_axes = primary_axes.twinx()
            secondary_axes.set_ylabel(f"dBFS ({secondaryName})")
        self.__secondary = AxesManager(secondaryDataProvider, secondary_axes)
        self.limits = Limits(self.__repr__(), self.__redraw_func, primary_axes, 60.0, x=(2, 250), axes_2=secondary_axes)
        self.limits.propagate_to_axes(draw=True)
        self.__legend = None
        self.__legend_cid = None
        self.redraw()

    def __redraw_func(self):
        self.__chart.canvas.draw_idle()

    def __repr__(self):
        return self.__name

    def redraw(self):
        '''
        Gets the current state of the graph
        '''
        self.__display_all_curves()
        self.__make_legend()
        self.__chart.canvas.draw_idle()

    def show_limits(self):
        '''
        Shows the limits dialog.
        '''
        LimitsDialog(self.limits).exec()

    def show_values(self):
        '''
        Shows the values dialog.
        '''
        ValuesDialog(self.__primary.artists() + self.__secondary.artists()).exec()

    def get_curve_names(self, primary=True):
        '''
        :param primary: if true get the primary curves.
        :return: the names of all the curves in the chart.
        '''
        return self.__primary.curve_names() if primary else self.__secondary.curve_names()

    def __display_all_curves(self):
        '''
        Updates all the curves with the currently cached data.
        '''
        self.__primary.display_curves()
        self.__secondary.display_curves()
        self.limits.configure_freq_axis()
        self.limits.on_data_change(self.__primary.get_ylimits(), self.__secondary.get_ylimits())

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

        lines = self.__primary.artists() + self.__secondary.artists()
        if len(lines) > 0:
            ncol = int(len(lines) / 3) if len(lines) % 3 == 0 else int(len(lines) / 3) + 1
            self.__legend = self.__primary.make_legend(lines, ncol)
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
        Sets the data series that will act as the reference.
        :param curve: the reference curve (if any).
        '''
        if primary is True:
            self.__primary.reference_curve = curve
        else:
            self.__secondary.reference_curve = curve
        self.redraw()


class Limits:
    '''
    Value object to hold graph limits to decouple the dialog from the chart.
    '''

    def __init__(self, name, redraw_func, axes_1, default_y_range, x, axes_2=None):
        self.name = name
        self.__redraw_func = redraw_func
        self.__default_y_range = default_y_range
        self.axes_1 = axes_1
        self.x_scale = 'log'
        self.x_min = x[0]
        self.x_max = x[1]
        numticks = 13
        self.axes_1.yaxis.set_major_locator(LinearLocator(numticks))
        self.y1_min, self.y1_max = self.calculate_dBFS_scales((0, 0))
        if axes_2 is not None:
            self.y2_min, self.y2_max = self.calculate_dBFS_scales((0, 0))
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
        self.configure_freq_axis()
        if draw:
            logger.debug(f"{self.name} Redrawing axes on limits change")
            self.__redraw_func()

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

    def calculate_dBFS_scales(self, y_range):
        '''
        Calculates the min/max in the data.
        :param data: the data.
        :param max_range: the max range.
        :return: min, max
        '''
        vmax = y_range[1]
        # coerce max to a round value
        multiple = 5 if self.__default_y_range <= 40 else 10
        if vmax % multiple != 0:
            vmax = (vmax - vmax % multiple) + multiple
        else:
            vmax += multiple
        return vmax - self.__default_y_range, vmax

    def on_data_change(self, primary_range, secondary_range):
        '''
        Updates the y axes when the data changes.
        :param primary_range: the primary y range.
        :param secondary_range: the secondary y range.
        '''
        if self.is_auto_1():
            new_min, new_max = self.calculate_dBFS_scales(primary_range)
            y1_changed = new_min != self.y1_min or new_max != self.y1_max
            if y1_changed:
                logger.debug(f"{self.name} y1 axis changed from {self.y1_min}/{self.y1_max} to {new_min}/{new_max}")
            self.y1_min = new_min
            self.y1_max = new_max
            if y1_changed:
                self.axes_1.set_ylim(bottom=self.y1_min, top=self.y1_max)
        if self.is_auto_2() and self.axes_2 is not None:
            new_min, new_max = self.calculate_dBFS_scales(secondary_range)
            y2_changed = new_min != self.y2_min or new_max != self.y2_max
            if y2_changed:
                logger.debug(f"{self.name} y2 axis changed from {self.y2_min}/{self.y2_max} to {new_min}/{new_max}")
            self.y2_min = new_min
            self.y2_max = new_max
            if y2_changed:
                self.axes_2.set_ylim(bottom=self.y2_min, top=self.y2_max)

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
        sets up the freq axis formatters (which you seem to have to call constantly otherwise matplotlib keeps
        reinstating the default log format)
        '''
        self.axes_1.set_xscale(self.x_scale)
        hzFormatter = EngFormatter(places=0)
        self.axes_1.get_xaxis().set_major_formatter(hzFormatter)
        self.axes_1.get_xaxis().set_minor_formatter(PrintFirstHalfFormatter(hzFormatter))
        self.axes_1.set_xlabel('Hz')


class ValuesDialog(QDialog, Ui_valuesDialog):
    '''
    Provides a mechanism for looking at the values listed in the chart without taking up screen estate.
    '''

    def __init__(self, data):
        super(ValuesDialog, self).__init__()
        self.setupUi(self)
        self.data = data
        self.__step = 0
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
            self.__step = xdata[1] - xdata[0]
            self.freq.setMinimum(xdata[0])
            self.freq.setSingleStep(self.__step)
            self.freq.setMaximum(xdata[-1])
            self.freq.setValue(xdata[0])
            self.freq.setEnabled(True)
        else:
            self.freq.setEnabled(False)

    def updateValues(self, freq):
        '''
        propagates the freq value change.
        :return:
        '''
        freq_idx = int(freq / self.__step)
        for idx, xy in enumerate(self.data):
            val = xy.get_ydata()[freq_idx]
            self.valueFields[idx].setText(str(round(val, 3)))

    def redraw(self, curves):
        '''
        Loads a new set of curves into the screen.
        :param curves: the curves.
        '''
        self.curves = curves


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
