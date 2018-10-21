import math
import numpy as np

from matplotlib.ticker import FuncFormatter


class WaveformModel:
    '''
    Displays and interacts with a waveform that is linked to the spectrum view.
    '''

    def __init__(self, chart, headroom):
        self.__chart = chart
        self.__axes = self.__chart.canvas.figure.add_subplot(111)
        self.__init_chart(draw=True)
        self.__line_one = DraggableLine(self.__axes, initial=0)
        self.__line_two = DraggableLine(self.__axes, initial=1)
        self.__headroom = headroom
        self.__signal = None
        self.__curve = None

    def get_time_range(self):
        ''' a 2 item iterable containing the current time range. '''
        t = [self.__line_one.get_x(), self.__line_two.get_x()]
        t.sort()
        return t

    @property
    def signal(self):
        return self.__signal

    @signal.setter
    def signal(self, signal):
        self.__signal = signal
        if signal is not None:
            headroom = 20 * math.log(1.0 / np.nanmax(np.abs(signal.samples)), 10)
        else:
            headroom = 0.0
        self.__headroom.setValue(headroom)
        if self.__curve is not None:
            self.__curve.set_data([], [])
        self.__init_chart()

    def __init_chart(self, draw=False):
        ''' initialise the axis limits and grid. '''
        self.__axes.set_ylim(bottom=-1, top=1)
        self.__axes.set_xlim(left=0, right=1)
        self.__axes.grid(linestyle='-', which='major', linewidth=1, alpha=0.5)
        self.__axes.grid(linestyle='--', which='minor', linewidth=1, alpha=0.5)
        self.__axes.set_xlabel('Time')
        from model.analysis import seconds_to_hhmmss
        self.__axes.xaxis.set_major_formatter(FuncFormatter(seconds_to_hhmmss))
        if draw is True:
            self.__redraw()

    def __redraw(self):
        self.__chart.canvas.draw_idle()

    def clear(self):
        '''
        Resets the analyser.
        '''
        self.signal = None

    def analyse(self):
        '''
        Calculates the spectrum view.
        '''
        from app import wait_cursor
        with wait_cursor(f"Analysing"):
            step = 1.0 / self.signal.fs
            x = np.arange(0, self.signal.durationSeconds, step)
            y = self.signal.samples
            # self.__limits.y1_min = -1.0
            # self.__limits.y1_max = 1.0
            if self.__curve is None:
                self.__curve = self.__axes.plot(x, y, linewidth=1, color='cyan')[0]
            else:
                self.__curve.set_data(x, y)
                # self.__limits.on_data_change((self.__limits.y1_min, self.__limits.y1_max), [])
            # self.__limits.propagate_to_axes(draw=True)

    def zoom_in(self):
        ''' zooms in on the line position (+/- a bit) '''
        time_range = self.get_time_range()
        self.__axes.set_xlim(left=time_range[0], right=time_range[1])
        self.__chart.canvas.draw_idle()

    def zoom_out(self):
        ''' zooms out on the line position (+/- a bit) '''
        self.__axes.set_xlim(left=0, right=self.__get_max_time())
        self.__chart.canvas.draw_idle()

    def __get_max_time(self):
        return self.signal.durationSeconds if self.signal else 1


class DraggableLine:
    ''' A line that can be dragged. '''

    def __init__(self, ax, initial=0):
        self.__axes = ax
        self.c = ax.get_figure().canvas
        self.__line = self.__axes.axvline(x=initial, linewidth=1, linestyle='--', picker=5)
        self.c.draw_idle()
        self.__pick_sid = self.c.mpl_connect('pick_event', self.__pick_line)
        self.__follow_handler = None
        self.__release_handler = None

    def get_x(self):
        ''' the x position of the line '''
        return self.__line.get_xdata()[0]

    def __pick_line(self, event):
        ''' connects events to track the mouse if this line was picked '''
        if event.artist == self.__line:
            self.__follow_handler = self.c.mpl_connect("motion_notify_event", self.__drag)
            self.__release_handler = self.c.mpl_connect("button_press_event", self.__release)

    def __drag(self, event):
        ''' updates the line position '''
        self.__line.set_xdata([event.xdata, event.xdata])
        self.c.draw_idle()

    def __release(self, event):
        ''' disconnects the event handlers so we stop dragging this line '''
        self.c.mpl_disconnect(self.__follow_handler)
        self.c.mpl_disconnect(self.__release_handler)
