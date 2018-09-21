import logging
import math

from matplotlib.gridspec import GridSpec

from model.limits import Limits, LimitsDialog, ValuesDialog, dBRangeCalculator
from model.preferences import GRAPH_X_AXIS_SCALE, GRAPH_X_MIN, GRAPH_X_MAX

logger = logging.getLogger('magnitude')

SINGLE_SUBPLOT_SPEC = GridSpec(1, 1).new_subplotspec((0, 0), 1, 1)


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
            curve.set_data(data.x, data.y)
            curve.set_linestyle(data.linestyle)
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

    def hide_axes_if_empty(self):
        '''
        Hides the axes if there is nothing here.
        '''
        if self.__axes is not None:
            self.__axes.get_yaxis().set_visible(len(self.curve_names()) > 0)


class MagnitudeModel:
    '''
    Allows a set of filters to be displayed on a chart as magnitude responses.
    '''

    def __init__(self, name, chart, preferences, primaryDataProvider, primaryName, secondaryDataProvider=None,
                 secondaryName=None, show_legend=lambda: True, db_range=60, subplot_spec=SINGLE_SUBPLOT_SPEC,
                 redraw_listener=None, grid_alpha=0.5):
        self.__name = name
        self.__chart = chart
        self.__redraw_listener = redraw_listener
        self.__show_legend = show_legend
        primary_axes = self.__chart.canvas.figure.add_subplot(subplot_spec)
        primary_axes.set_ylabel(f"dBFS ({primaryName})")
        primary_axes.grid(linestyle='-', which='major', linewidth=1, alpha=grid_alpha)
        primary_axes.grid(linestyle='--', which='minor', linewidth=1, alpha=grid_alpha)
        self.__primary = AxesManager(primaryDataProvider, primary_axes)
        if secondaryDataProvider is None:
            secondary_axes = None
        else:
            secondary_axes = primary_axes.twinx()
            secondary_axes.set_ylabel(f"dBFS ({secondaryName})")
        self.__secondary = AxesManager(secondaryDataProvider, secondary_axes)
        self.limits = Limits(self.__repr__(), self.__redraw_func, primary_axes,
                             x_lim=(preferences.get(GRAPH_X_MIN), preferences.get(GRAPH_X_MAX)),
                             y_range_calculator=dBRangeCalculator(db_range), axes_2=secondary_axes,
                             x_scale=preferences.get(GRAPH_X_AXIS_SCALE))
        self.limits.propagate_to_axes(draw=True)
        self.__legend = None
        self.__legend_cid = None
        self.redraw()

    def __redraw_func(self):
        self.__chart.canvas.draw_idle()
        if callable(self.__redraw_listener):
            self.__redraw_listener()

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
        self.limits.configure_x_axis()
        self.__secondary.hide_axes_if_empty()
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

        if self.__show_legend():
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
