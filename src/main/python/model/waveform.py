import math

import numpy as np
import qtawesome as qta
from matplotlib.font_manager import FontProperties
from pyqtgraph import mkPen
from qtpy.QtCore import QTime
from qtpy.QtGui import QFont


class WaveformController:
    def __init__(self, signal_model, chart, signal_selector, headroom, is_filtered, start_time, end_time,
                 apply_limits_btn, reset_limits_btn, zoom_in_btn, zoom_out_btn, y_min, y_max):
        self.__signal_model = signal_model
        self.__current_signal = None
        self.__is_filtered = is_filtered
        self.__start_time = start_time
        self.__end_time = end_time
        self.__apply_limits_btn = apply_limits_btn
        self.__reset_limits_btn = reset_limits_btn
        self.__zoom_in_btn = zoom_in_btn
        self.__zoom_out_btn = zoom_out_btn
        self.__selector = signal_selector
        self.__chart_model = WaveformModel(chart, headroom, start_time, end_time, y_min, y_max)
        self.__apply_limits_btn.setIcon(qta.icon('fa5s.check'))
        self.__reset_limits_btn.setIcon(qta.icon('fa5s.times'))
        self.__zoom_in_btn.setIcon(qta.icon('fa5s.search-plus'))
        self.__zoom_out_btn.setIcon(qta.icon('fa5s.search-minus'))
        self.__reset_limits_btn.clicked.connect(self.reset_limits)
        self.__apply_limits_btn.clicked.connect(self.apply_limits)
        self.__is_filtered.stateChanged['int'].connect(self.toggle_filter)
        self.__selector.currentIndexChanged['QString'].connect(self.update_waveform)
        self.__zoom_in_btn.clicked.connect(self.__chart_model.zoom_in)
        self.__zoom_out_btn.clicked.connect(self.__chart_model.zoom_out)
        self.update_waveform(None)

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
        if self.__current_signal is not None:
            self.__current_signal.unregister_listener(self.on_filter_update)
        self.__current_signal = self.__get_signal_data(signal_name)
        if self.__current_signal is None:
            self.__reset_time(self.__start_time)
            self.__reset_time(self.__end_time)
            self.__chart_model.clear()
        else:
            self.__current_signal.register_listener(self.on_filter_update)
            self.__start_time.setEnabled(True)
            duration = QTime(0, 0, 0).addMSecs(self.__current_signal.signal.durationSeconds * 1000.0)
            self.__start_time.setMaximumTime(duration)
            self.__end_time.setEnabled(True)
            self.__end_time.setMaximumTime(duration)
            self.__end_time.setTime(duration)
            self.toggle_filter(self.__is_filtered.isChecked())

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
        signal_name = self.__selector.currentText()
        signal_data = self.__get_signal_data(signal_name)
        if signal_data is not None:
            if state:
                sos = signal_data.active_filter.resample(signal_data.fs, copy_listener=False).get_sos()
                if len(sos) > 0:
                    signal = signal_data.signal.sosfilter(sos)
                else:
                    signal = signal_data.signal
            else:
                signal = signal_data.signal
            self.__chart_model.signal = signal
            self.__chart_model.analyse()

    def on_filter_update(self):
        ''' if the signal is filtered then updated the chart when the filter changes. '''
        if self.__is_filtered.isChecked():
            self.toggle_filter(True)

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
        self.__x_min.editingFinished.connect(self.__update_x_range)
        self.__x_max.editingFinished.connect(self.__update_x_range)
        self.__y_min = y_min
        self.__y_max = y_max
        self.__y_min.editingFinished.connect(self.__update_y_range)
        self.__y_max.editingFinished.connect(self.__update_y_range)
        self.__chart = chart
        label_font = QFont()
        fp = FontProperties()
        label_font.setPointSize(fp.get_size_in_points() * 0.7)
        label_font.setFamily(fp.get_name())
        for name in ['left', 'right', 'bottom', 'top']:
            self.__chart.getPlotItem().getAxis(name).setTickFont(label_font)
        self.__chart.getPlotItem().showGrid(x=True, y=True, alpha=0.5)
        self.__chart.getPlotItem().disableAutoRange()
        self.__chart.getPlotItem().setLimits(xMin=0.0, xMax=1.0, yMin=-1.0, yMax=1.0)
        self.__chart.getPlotItem().setXRange(0, 1, padding=0.0)
        self.__chart.getPlotItem().setYRange(-1, 1, padding=0.0)
        self.__chart.getPlotItem().setDownsampling(ds=True, auto=True, mode='peak')
        self.__chart.getPlotItem().layout.setContentsMargins(10, 20, 30, 20)
        self.__chart.getPlotItem().sigXRangeChanged.connect(self.__propagate_x_range)
        self.__chart.getPlotItem().sigYRangeChanged.connect(self.__propagate_y_range)
        self.__headroom = headroom
        self.__signal = None
        self.__curve = None

    def __propagate_x_range(self, _, range):
        ''' passes the updates range to the fields '''
        from model.report import block_signals
        with block_signals(self.__x_min):
            self.__x_min.setTime(QTime(0, 0, 0).addMSecs(range[0] * 1000.0))
        with block_signals(self.__x_max):
            self.__x_max.setTime(QTime(0, 0, 0).addMSecs(range[1] * 1000.0))

    def __propagate_y_range(self, _, range):
        ''' passes the updates range to the fields '''
        from model.report import block_signals
        with block_signals(self.__y_min):
            self.__y_min.setValue(range[0])
        with block_signals(self.__y_max):
            self.__y_max.setValue(range[1])

    def __update_y_range(self):
        ''' changes the y limits '''
        self.__chart.getPlotItem().setYRange(self.__y_min.value(), self.__y_max.value(), padding=0.0)

    def __update_x_range(self):
        ''' changes the y limits '''
        self.__chart.getPlotItem().setXRange(self.__to_seconds(self.__x_min), self.__to_seconds(self.__x_max),
                                             padding=0.0)

    def get_time_range(self):
        ''' a 2 item iterable containing the current time range. '''
        return [self.__x_min.time().msecsSinceStartOfDay() / 1000.0,
                self.__x_max.time().msecsSinceStartOfDay() / 1000.0]

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
        x_range = self.__chart.getPlotItem().getAxis('bottom').range
        if self.signal is None:
            x_max = 1.0
        else:
            x_max = self.signal.durationSeconds
        self.__chart.getPlotItem().setLimits(xMin=0.0, xMax=x_max)
        if x_range[1] > x_max:
            self.__chart.getPlotItem().setXRange(x_range[0], x_max, padding=0.0)
        if self.__curve is not None and signal is None:
            self.__chart.getPlotItem().removeItem(self.__curve)
            self.__curve = None

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
            if self.__curve is None:
                self.__curve = self.__chart.plot(x, y, pen=mkPen('c', width=1))
                self.zoom_out()
            else:
                self.__curve.setData(x, y)

    def zoom_in(self):
        ''' zooms in on the line position (+/- a bit) '''
        time_range = self.get_time_range()
        self.__chart.getPlotItem().setXRange(time_range[0], time_range[1], padding=0.0)
        if self.signal is not None:
            self.__chart.getPlotItem().setYRange(np.nanmin(self.signal.samples), np.nanmax(self.signal.samples),
                                                 padding=0.0)

    def zoom_out(self):
        ''' zooms out on the line position (+/- a bit) '''
        self.__chart.getPlotItem().setXRange(0, self.__get_max_time(), padding=0.0)
        self.__chart.getPlotItem().setYRange(-1, 1, padding=0.0)

    def __get_max_time(self):
        return self.signal.durationSeconds if self.signal else 1

    def __to_seconds(self, time_widget):
        ''' yields a time in seconds from a QTimeEdit '''
        return time_widget.time().msecsSinceStartOfDay() / 1000.0
