import math

import numpy as np
import qtawesome as qta
from PyQt5.QtCore import QTime
from matplotlib.ticker import FuncFormatter


class WaveformController:
    def __init__(self, signal_model, chart, signal_selector, headroom, is_filtered, start_time, end_time,
                 apply_limits_btn, reset_limits_btn, zoom_in_btn, zoom_out_btn, y_min, y_max):
        self.__signal_model = signal_model
        self.__is_filtered = is_filtered
        self.__start_time = start_time
        self.__end_time = end_time
        self.__apply_limits_btn = apply_limits_btn
        self.__reset_limits_btn = reset_limits_btn
        self.__zoom_in_btn = zoom_in_btn
        self.__zoom_out_btn = zoom_out_btn
        self.__selector = signal_selector
        self.__chart_model = WaveformModel(chart, headroom, self.__to_seconds(start_time), self.__to_seconds(end_time),
                                           y_min, y_max)
        self.__apply_limits_btn.setIcon(qta.icon('fa5s.check'))
        self.__reset_limits_btn.setIcon(qta.icon('fa5s.times'))
        self.__zoom_in_btn.setIcon(qta.icon('fa5s.search-plus'))
        self.__zoom_out_btn.setIcon(qta.icon('fa5s.search-minus'))
        self.__reset_limits_btn.clicked.connect(self.reset_limits)
        self.__apply_limits_btn.clicked.connect(self.apply_limits)
        self.__is_filtered.stateChanged['int'].connect(self.toggle_filter)
        self.__selector.currentIndexChanged['QString'].connect(self.update_waveform)
        self.__start_time.editingFinished.connect(self.__chart_model.set_left_gate)
        self.__end_time.editingFinished.connect(self.__chart_model.set_right_gate)
        self.__zoom_in_btn.clicked.connect(self.__chart_model.zoom_in)
        self.__zoom_out_btn.clicked.connect(self.__chart_model.zoom_out)
        self.update_waveform(None)

    def __to_seconds(self, time_widget):
        ''' yields a time in seconds from a QTimeEdit '''
        def convert():
            return time_widget.time().msecsSinceStartOfDay() / 1000.0

        return convert

    def refresh_selector(self):
        ''' Updates the selector with the available signals. '''
        currently_selected = self.__selector.currentText()
        from model.report import block_signals
        with block_signals(self.__selector):
            self.__selector.clear()
            self.__selector.addItem('  ')
            for s in self.__signal_model:
                if s.signal is not None:
                    self.__selector.addItem(s.name)
            idx = self.__selector.findText(currently_selected)
            if idx > -1:
                self.__selector.setCurrentIndex(idx)
            else:
                self.__chart_model.clear()

    def update_waveform(self, signal_name):
        ''' displays the waveform for the selected signal '''
        signal_data = self.__get_signal_data(signal_name)
        if signal_data is None:
            self.__reset_time(self.__start_time)
            self.__reset_time(self.__end_time)
            self.__chart_model.clear()
        else:
            self.__start_time.setEnabled(True)
            self.__end_time.setEnabled(True)
            # TODO update maximum time
            # TODO if start > end, reset
            self.__chart_model.signal = signal_data.signal
            self.__chart_model.analyse()

    def __get_signal_data(self, signal_name):
        signal_data = next((s for s in self.__signal_model if s.name == signal_name), None)
        return signal_data

    def __reset_time(self, time_widget):
        ''' resets and disables the supplied time field. '''
        from model.report import block_signals
        with block_signals(time_widget):
            time_widget.clearMaximumDateTime()
            time_widget.setTime(QTime())
            time_widget.setEnabled(False)

    def toggle_filter(self, state):
        ''' Applies or removes the filter from the visible waveform '''
        signal_data = self.__get_signal_data(self.__selector.currentText())
        if signal_data is not None:
            if state:
                sos = signal_data.active_filter.resample(signal_data.fs).get_sos()
                if len(sos) > 0:
                    signal = signal_data.signal.sosfilter(sos)
                else:
                    signal = signal_data.signal
            else:
                signal = signal_data.signal
            self.__chart_model.signal = signal
            self.__chart_model.analyse()

    def apply_limits(self):
        ''' Updates the visible spectrum for the selected waveform limits '''
        pass

    def reset_limits(self):
        ''' Resets the visible spectrum for the selected waveform limits '''
        pass


class WaveformModel:
    '''
    Displays and interacts with a waveform that is linked to the spectrum view.
    '''

    def __init__(self, chart, headroom, x_min, x_max, y_min, y_max):
        self.__chart = chart
        self.__x_min = x_min
        self.__x_max = x_max
        self.__y_min = y_min
        self.__y_max = y_max
        self.__y_min.editingFinished.connect(self.__update_y_range)
        self.__y_max.editingFinished.connect(self.__update_y_range)
        self.__axes = self.__chart.canvas.figure.add_subplot(111)
        self.__init_chart(draw=True)
        self.left_gate = DraggableLine(self.__axes, initial=0)
        self.right_gate = DraggableLine(self.__axes, initial=1)
        self.__headroom = headroom
        self.__signal = None
        self.__curve = None

    def __update_y_range(self):
        ''' changes the y limits '''
        self.__axes.set_ylim(self.__y_min.value(), self.__y_max.value())
        self.redraw()

    def get_time_range(self):
        ''' a 2 item iterable containing the current time range. '''
        t = [self.left_gate.get_x(), self.right_gate.get_x()]
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
        self.__axes.set_ylim(bottom=self.__y_min.value(), top=self.__y_max.value())
        self.__axes.set_xlim(left=self.__x_min(), right=self.__x_max())
        self.__axes.grid(linestyle='-', which='major', linewidth=1, alpha=0.5)
        self.__axes.grid(linestyle='--', which='minor', linewidth=1, alpha=0.5)
        self.__axes.set_xlabel('Time')
        from model.analysis import seconds_to_hhmmss
        self.__axes.xaxis.set_major_formatter(FuncFormatter(seconds_to_hhmmss))
        if draw is True:
            self.redraw()

    def redraw(self):
        self.__chart.canvas.draw_idle()

    def clear(self):
        '''
        Resets the analyser.
        '''
        self.signal = None
        self.left_gate.reset()
        self.right_gate.reset()
        self.redraw()

    def analyse(self):
        '''
        Calculates the spectrum view.
        '''
        from app import wait_cursor
        with wait_cursor(f"Analysing"):
            step = 1.0 / self.signal.fs
            x = np.arange(0, self.signal.durationSeconds, step)
            y = self.signal.samples
            if self.__curve is None:
                self.__curve = self.__axes.plot(x, y, linewidth=1, color='cyan')[0]
            else:
                self.__curve.set_data(x, y)
            self.redraw()

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

    def set_left_gate(self):
        ''' updates the start time '''
        self.left_gate.move(self.__x_min())
        self.redraw()

    def set_right_gate(self):
        ''' updates the end time '''
        self.right_gate.move(self.__x_max())
        self.redraw()


class DraggableLine:
    ''' A line that can be dragged. '''

    def __init__(self, ax, initial=0):
        self.__axes = ax
        self.__initial = initial
        self.c = ax.get_figure().canvas
        self.__line = self.__axes.axvline(x=initial, linewidth=1, linestyle='--', picker=5)
        self.__colour = self.__line.get_color()
        self.c.draw_idle()
        self.__pick_sid = self.c.mpl_connect('pick_event', self.__pick_line)
        self.__release_handler = None

    def reset(self):
        ''' moves the line to the initial position. '''
        self.move(self.__initial)

    def move(self, position):
        ''' moves the line to the specified position '''
        self.__line.set_xdata([position, position])

    def get_x(self):
        ''' the x position of the line '''
        return self.__line.get_xdata()[0]

    def __pick_line(self, event):
        ''' connects events to track the mouse if this line was picked '''
        if event.artist == self.__line:
            self.__line.set_color('red')
            self.c.draw_idle()
            self.__release_handler = self.c.mpl_connect("button_release_event", self.__release)

    def __release(self, event):
        ''' disconnects the event handlers so we stop dragging this line '''
        self.move(event.xdata)
        self.__line.set_color(self.__colour)
        self.c.mpl_disconnect(self.__release_handler)
        self.c.draw_idle()
