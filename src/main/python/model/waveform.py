import math

import numpy as np
import qtawesome as qta
from matplotlib.font_manager import FontProperties
from pyqtgraph import mkPen
from qtpy.QtCore import QTime
from qtpy.QtGui import QFont

from model.magnitude import MagnitudeModel
from model.preferences import get_filter_colour
from model.signal import SignalDialog


class WaveformController:
    def __init__(self, preferences, signal_model, waveform_chart, spectrum_chart, signal_selector, headroom,
                 is_filtered, start_time, end_time, show_spectrum_btn, hide_spectrum_btn, zoom_in_btn, zoom_out_btn,
                 source_file, load_signal_btn, show_limits_btn, y_min, y_max):
        self.__preferences = preferences
        self.__signal_model = signal_model
        self.__current_signal = None
        self.__active_signal = None
        self.__is_filtered = is_filtered
        self.__start_time = start_time
        self.__end_time = end_time
        self.__show_spectrum_btn = show_spectrum_btn
        self.__hide_spectrum_btn = hide_spectrum_btn
        self.__zoom_in_btn = zoom_in_btn
        self.__zoom_out_btn = zoom_out_btn
        self.__load_signal_btn = load_signal_btn
        self.__source_file = source_file
        self.__selector = signal_selector
        self.__waveform_chart_model = WaveformModel(waveform_chart, headroom, start_time, end_time, y_min, y_max,
                                                    self.__on_x_range_change)
        spectrum_chart.vbl.setContentsMargins(1, 1, 1, 1)
        spectrum_chart.setVisible(False)
        self.__magnitude_model = MagnitudeModel('spectrum', spectrum_chart, preferences, self, 'Spectrum',
                                                show_legend=lambda: False)
        self.__show_limits_btn = show_limits_btn
        # match the pyqtgraph layout
        self.__load_signal_btn.clicked.connect(self.__load_signal)
        self.__show_spectrum_btn.setIcon(qta.icon('fa5s.eye'))
        self.__hide_spectrum_btn.setIcon(qta.icon('fa5s.eye-slash'))
        self.__zoom_in_btn.setIcon(qta.icon('fa5s.search-plus'))
        self.__zoom_out_btn.setIcon(qta.icon('fa5s.search-minus'))
        self.__load_signal_btn.setIcon(qta.icon('fa5s.folder-open'))
        self.__show_limits_btn.setIcon(qta.icon('fa5s.arrows-alt'))
        self.__show_spectrum_btn.clicked.connect(self.show_spectrum)
        self.__hide_spectrum_btn.clicked.connect(self.hide_spectrum)
        self.__show_limits_btn.clicked.connect(self.__magnitude_model.show_limits)
        self.__is_filtered.stateChanged['int'].connect(self.toggle_filter)
        self.__selector.currentIndexChanged['QString'].connect(self.update_waveform)
        self.__zoom_in_btn.clicked.connect(self.__waveform_chart_model.zoom_in)
        self.__zoom_out_btn.clicked.connect(self.__zoom_out)
        self.update_waveform(None)

    def __zoom_out(self):
        ''' zooms out and hides the spectrum '''
        self.hide_spectrum()
        self.__waveform_chart_model.zoom_out()

    def __on_x_range_change(self, enable):
        ''' updates the view if the x range changes. '''
        self.__show_spectrum_btn.setEnabled(enable)
        self.__hide_spectrum_btn.setEnabled(enable)
        self.__show_limits_btn.setEnabled(enable)
        if enable:
            self.__magnitude_model.redraw()

    def getMagnitudeData(self, reference=None):
        '''
        :param reference: ignored as we don't expose a normalisation control in this chart.
        :return: the peak and avg spectrum for the currently filtered signal (if any).
        '''
        if self.__active_signal is not None and self.__magnitude_model.is_visible():
            sig = self.__active_signal.cut(to_seconds(self.__start_time), to_seconds(self.__end_time))
            sig.calculate_peak_average(self.__preferences)
            return sig.getXY(idx=self.__selector.currentIndex() - 1)
        return []

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
                else:
                    self.__selector.addItem(f"-- {s.name}")
            idx = self.__selector.findText(currently_selected)
            if idx > -1:
                self.__selector.setCurrentIndex(idx)
            else:
                self.__reset_controls()

    def __reset_controls(self):
        self.__waveform_chart_model.clear()
        self.__source_file.clear()
        self.__on_x_range_change(False)
        self.hide_spectrum()

    def update_waveform(self, signal_name):
        ''' displays the waveform for the selected signal '''
        if self.__current_signal is not None:
            self.__current_signal.unregister_listener(self.on_filter_update)
        self.__current_signal = self.__get_signal_data(signal_name)
        if self.__current_signal is None:
            self.__reset_time(self.__start_time)
            self.__reset_time(self.__end_time)
            self.__load_signal_btn.setEnabled(True)
            self.__reset_controls()
            self.__active_signal = None
        else:
            self.__load_signal_btn.setEnabled(False)
            metadata = self.__current_signal.signal.metadata
            if metadata is not None:
                self.__source_file.setText(f"{ metadata['src']} - C{ metadata['channel']}")
            self.__current_signal.register_listener(self.on_filter_update)
            self.__start_time.setEnabled(True)
            duration = QTime(0, 0, 0).addMSecs(self.__current_signal.signal.durationSeconds * 1000.0)
            self.__start_time.setMaximumTime(duration)
            self.__end_time.setEnabled(True)
            self.__end_time.setMaximumTime(duration)
            self.__end_time.setTime(duration)
            self.toggle_filter(self.__is_filtered.isChecked())

    def __load_signal(self):
        signal_name = self.__selector.currentText()
        if signal_name.startswith('-- '):
            signal_name = signal_name[3:]
        signal_data = self.__get_signal_data(signal_name)
        if signal_data is not None and signal_data.signal is None:
            AssociateSignalDialog(self.__preferences, signal_data).exec()
            if signal_data.signal is not None:
                self.__selector.setItemText(self.__selector.currentIndex(), signal_name)
                self.update_waveform(signal_name)

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
            self.__active_signal = signal
            self.__waveform_chart_model.signal = signal
            self.__waveform_chart_model.idx = self.__selector.currentIndex() - 1
            self.__waveform_chart_model.analyse()
            self.__magnitude_model.redraw()

    def on_filter_update(self):
        ''' if the signal is filtered then updated the chart when the filter changes. '''
        if self.__is_filtered.isChecked():
            self.toggle_filter(True)

    def show_spectrum(self):
        ''' Updates the visible spectrum for the selected waveform limits '''
        self.__magnitude_model.set_visible(True)
        self.__magnitude_model.redraw()

    def hide_spectrum(self):
        ''' Resets the visible spectrum for the selected waveform limits '''
        self.__magnitude_model.set_visible(False)


class WaveformModel:
    '''
    Displays and interacts with a waveform that is linked to the spectrum view.
    '''

    def __init__(self, chart, headroom, x_min, x_max, y_min, y_max, on_x_range_change):
        self.idx = 0
        self.__chart = chart
        self.__on_x_range_change = on_x_range_change
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
        self.__x_min.setTime(QTime(0, 0, 0).addMSecs(range[0] * 1000.0))
        self.__x_max.setTime(QTime(0, 0, 0).addMSecs(range[1] * 1000.0))
        self.__propagate_btn_state_on_xrange_change()

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
        max_secs, min_secs = self.__propagate_btn_state_on_xrange_change()
        self.__chart.getPlotItem().setXRange(min_secs, max_secs, padding=0.0)

    def __propagate_btn_state_on_xrange_change(self):
        min_secs, max_secs = self.get_time_range()
        enable_btns = (not math.isclose(min_secs, 0.0)) or (max_secs < self.__get_max_time())
        self.__on_x_range_change(enable_btns)
        return max_secs, min_secs

    def get_time_range(self):
        ''' a 2 item iterable containing the current time range. '''
        return [to_seconds(self.__x_min), to_seconds(self.__x_max)]

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


def to_seconds(time_widget):
    ''' yields a time in seconds from a QTimeEdit '''
    return time_widget.time().msecsSinceStartOfDay() / 1000.0


class AssociateSignalDialog(SignalDialog):
    ''' Customises the signal dialog to allow a single channel to be loaded from a wav only '''

    def __init__(self, preferences, signal_data):
        super().__init__(preferences, None, allow_multichannel=False)
        self.__signal_data = signal_data
        self.signalTypeTabs.setTabEnabled(1, False)

    def save(self, signals):
        ''' writes the underlying signal into the existing signaldata '''
        self.__signal_data.signal = signals[0].signal

    def selectFile(self):
        ''' autopopulates and disables the signal name field (as it is irrelevant here) '''
        super().selectFile()
        self.wavSignalName.setText(self.__signal_data.name)
        self.wavSignalName.setEnabled(False)