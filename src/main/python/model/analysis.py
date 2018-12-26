import datetime
import logging
import math
import time

import matplotlib
import numpy as np
import qtawesome as qta
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter, MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QDialog

from model.limits import Limits, LimitsDialog
from model.preferences import GRAPH_X_MIN, GRAPH_X_MAX, POINT, ELLIPSE, SPECTROGRAM_CONTOURED, SPECTROGRAM_FLAT, \
    AUDIO_ANALYSIS_MARKER_SIZE, AUDIO_ANALYSIS_MARKER_TYPE, AUDIO_ANALYSIS_ELLIPSE_WIDTH, AUDIO_ANALYSIS_ELLIPSE_HEIGHT, \
    AUDIO_ANALYIS_MIN_FREQ, AUDIO_ANALYIS_MAX_UNFILTERED_FREQ, AUDIO_ANALYIS_MAX_FILTERED_FREQ, \
    AUDIO_ANALYSIS_COLOUR_MAX, AUDIO_ANALYSIS_COLOUR_MIN, AUDIO_ANALYSIS_SIGNAL_MIN, AUDIO_ANALYSIS_GEOMETRY
from model.signal import select_file, readWav
from ui.analysis import Ui_analysisDialog

logger = logging.getLogger('analysis')

MULTIPLIERS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]


class AnalyseSignalDialog(QDialog, Ui_analysisDialog):
    def __init__(self, preferences, signal_model, allow_load=True):
        super(AnalyseSignalDialog, self).__init__()
        self.setupUi(self)
        self.setWindowFlags(self.windowFlags() | Qt.WindowSystemMenuHint | Qt.WindowMinMaxButtonsHint)
        self.__preferences = preferences
        self.filePicker.setIcon(qta.icon('fa5s.folder-open'))
        self.showLimitsButton.setIcon(qta.icon('fa5s.arrows-alt'))
        self.updateChart.setIcon(qta.icon('fa5s.sync'))
        self.saveChart.setIcon(qta.icon('fa5s.save'))
        self.saveLayout.setIcon(qta.icon('fa5s.save'))
        self.minFreq.setValue(preferences.get(GRAPH_X_MIN))
        self.maxUnfilteredFreq.setValue(preferences.get(GRAPH_X_MAX))
        self.__info = None
        self.__signal = None
        self.__filtered_signals = {}
        self.__spectrum_analyser = MaxSpectrumByTime(self.spectrumChart, self.__preferences, self)
        self.__waveform_analyser = Waveform(self.waveformChart, self)
        self.__signal_model = signal_model
        from model.report import block_signals
        with block_signals(self.markerType):
            self.markerType.addItem(POINT)
            self.markerType.addItem(ELLIPSE)
            self.markerType.addItem(SPECTROGRAM_CONTOURED)
            self.markerType.addItem(SPECTROGRAM_FLAT)
        self.markerType.setCurrentText(POINT)
        self.updateChart.setEnabled(False)
        self.__duration = 0
        self.loadButton.setEnabled(False)
        self.compareSignalsButton.setEnabled(False)
        self.__init_from_prefs()
        self.__clear()
        if allow_load:
            self.__init_for_load()
        else:
            self.__init_for_compare()

    def __init_for_compare(self):
        ''' initialises the fields relevant to a compare signal based analysis. '''
        self.analysisFrame.setVisible(False)
        self.analysisTabs.removeTab(1)
        self.__load_signal_selector(self.leftSignal)
        self.__load_signal_selector(self.rightSignal)

    def __load_signal_selector(self, selector):
        ''' loads the signals into a selector. '''
        for s in self.__signal_model:
            if s.signal is not None:
                selector.addItem(s.name)
        for bm in self.__signal_model.bass_managed_signals:
            selector.addItem(f"(BM) {bm.name}")

    def __init_for_load(self):
        ''' initialises the fields relevant to a load from file based analysis. '''
        self.signalFrame.setVisible(False)
        self.copyFilter.addItem('No Filter')
        for s in self.__signal_model:
            if s.master is None:
                self.copyFilter.addItem(s.name)
        if len(self.__signal_model) == 0:
            self.copyFilter.setEnabled(False)

    def __init_from_prefs(self):
        ''' initialises the various form fields from preferences '''
        self.markerSize.setValue(self.__preferences.get(AUDIO_ANALYSIS_MARKER_SIZE))
        self.markerType.setCurrentText(self.__preferences.get(AUDIO_ANALYSIS_MARKER_TYPE))
        self.ellipseWidth.setValue(self.__preferences.get(AUDIO_ANALYSIS_ELLIPSE_WIDTH))
        self.ellipseHeight.setValue(self.__preferences.get(AUDIO_ANALYSIS_ELLIPSE_HEIGHT))
        self.minFreq.setValue(self.__preferences.get(AUDIO_ANALYIS_MIN_FREQ))
        self.maxUnfilteredFreq.setValue(self.__preferences.get(AUDIO_ANALYIS_MAX_UNFILTERED_FREQ))
        self.maxFilteredFreq.setValue(self.__preferences.get(AUDIO_ANALYIS_MAX_FILTERED_FREQ))
        self.colourUpperLimit.setValue(self.__preferences.get(AUDIO_ANALYSIS_COLOUR_MAX))
        self.colourLowerLimit.setValue(self.__preferences.get(AUDIO_ANALYSIS_COLOUR_MIN))
        self.magLowerLimit.setValue(self.__preferences.get(AUDIO_ANALYSIS_SIGNAL_MIN))
        geometry = self.__preferences.get(AUDIO_ANALYSIS_GEOMETRY)
        if geometry is not None:
            self.restoreGeometry(geometry)

    def save_layout(self):
        ''' saves the layout and prefs '''
        self.__preferences.set(AUDIO_ANALYSIS_MARKER_SIZE, self.markerSize.value())
        self.__preferences.set(AUDIO_ANALYSIS_MARKER_TYPE, self.markerType.currentText())
        self.__preferences.set(AUDIO_ANALYSIS_ELLIPSE_WIDTH, self.ellipseWidth.value())
        self.__preferences.set(AUDIO_ANALYSIS_ELLIPSE_HEIGHT, self.ellipseHeight.value())
        self.__preferences.set(AUDIO_ANALYIS_MIN_FREQ, self.minFreq.value())
        self.__preferences.set(AUDIO_ANALYIS_MAX_UNFILTERED_FREQ, self.maxUnfilteredFreq.value())
        self.__preferences.set(AUDIO_ANALYIS_MAX_FILTERED_FREQ, self.maxFilteredFreq.value())
        self.__preferences.set(AUDIO_ANALYSIS_COLOUR_MAX, self.colourUpperLimit.value())
        self.__preferences.set(AUDIO_ANALYSIS_COLOUR_MIN, self.colourLowerLimit.value())
        self.__preferences.set(AUDIO_ANALYSIS_SIGNAL_MIN, self.magLowerLimit.value())
        self.__preferences.set(AUDIO_ANALYSIS_GEOMETRY, self.saveGeometry())

    def update_marker_type(self, marker):
        ''' hides form controls that are not relevant to the selected marker '''
        if marker == POINT:
            self.magLimitType.setVisible(True)
            self.magLimitTypeLabel.setVisible(True)
            self.ellipseHeight.setEnabled(False)
            self.ellipseHeightLabel.setEnabled(False)
            self.ellipseWidth.setEnabled(False)
            self.ellipseWidthLabel.setEnabled(False)
            self.magLimitTypeLabel.setVisible(True)
            self.magLimitType.setVisible(True)
            self.set_mag_range_type(self.magLimitType.currentText())
        elif marker == ELLIPSE:
            self.magLimitType.setVisible(True)
            self.magLimitTypeLabel.setVisible(True)
            self.ellipseHeight.setEnabled(True)
            self.ellipseHeightLabel.setEnabled(True)
            self.ellipseWidth.setEnabled(True)
            self.ellipseWidthLabel.setEnabled(True)
            self.magLimitTypeLabel.setVisible(True)
            self.magLimitType.setVisible(True)
            self.set_mag_range_type(self.magLimitType.currentText())
        else:
            self.magLimitType.setVisible(False)
            self.magLimitTypeLabel.setVisible(False)
            self.ellipseHeight.setEnabled(False)
            self.ellipseHeightLabel.setEnabled(False)
            self.ellipseWidth.setEnabled(False)
            self.ellipseWidthLabel.setEnabled(False)
            self.magLimitTypeLabel.setVisible(False)
            self.magLimitType.setVisible(False)
            self.set_mag_range_type('')

    def enable_compare(self):
        ''' enables the compare button if left and right have different signals selected '''
        enabled = self.leftSignal.currentText() != self.rightSignal.currentText() \
                  or self.filterLeft.isChecked() != self.filterRight.isChecked()
        self.compareSignalsButton.setEnabled(enabled)

    def compare_signals(self):
        ''' Loads the selected signals into the spectrum chart. '''
        left_signal = self.__get_signal_data(self.leftSignal.currentText())
        right_signal = self.__get_signal_data(self.rightSignal.currentText())
        max_duration = math.floor(max(left_signal.duration_seconds, right_signal.duration_seconds) * 1000)
        self.startTime.setTime(QtCore.QTime(0, 0, 0))
        self.startTime.setEnabled(True)
        max_time = QtCore.QTime(0, 0, 0).addMSecs(max_duration)
        self.endTime.setMaximumTime(max_time)
        self.endTime.setTime(max_time)
        self.endTime.setEnabled(True)
        self.maxTime.setMaximumTime(max_time)
        self.maxTime.setTime(max_time)
        self.__init_resolution_selector(left_signal.signal)
        self.__spectrum_analyser.left = left_signal.filter_signal(filt=self.filterLeft.isChecked())
        self.__spectrum_analyser.right = right_signal.filter_signal(filt=self.filterRight.isChecked())
        self.__spectrum_analyser.analyse()
        self.updateChart.setEnabled(True)

    def __get_signal_data(self, signal_name):
        if signal_name is not None and signal_name.startswith('(BM) '):
            return next((s for s in self.__signal_model.bass_managed_signals if s.name == signal_name[5:]), None)
        else:
            return next((s for s in self.__signal_model if s.name == signal_name), None)

    def save_chart(self):
        ''' opens the save chart dialog '''
        from app import SaveChartDialog
        if self.analysisTabs.currentIndex() == 0:
            SaveChartDialog(self, 'peak spectrum', self.spectrumChart.canvas.figure).exec()
        elif self.analysisTabs.currentIndex() == 1:
            SaveChartDialog(self, 'waveform', self.waveformChart.canvas.figure).exec()

    def select_wav_file(self):
        '''
        Allows the user to select a file and laods info about it
        '''
        file = select_file(self, ['wav', 'flac'])
        if file is not None:
            self.__clear()
            self.file.setText(file)
            import soundfile as sf
            self.__info = sf.info(file)
            self.channelSelector.clear()
            for i in range(0, self.__info.channels):
                self.channelSelector.addItem(f"{i + 1}")
            self.channelSelector.setEnabled(self.__info.channels > 1)
            self.startTime.setTime(QtCore.QTime(0, 0, 0))
            self.startTime.setEnabled(True)
            self.__duration = math.floor(self.__info.duration * 1000)
            max_time = QtCore.QTime(0, 0, 0).addMSecs(self.__duration)
            self.endTime.setMaximumTime(max_time)
            self.endTime.setTime(max_time)
            self.endTime.setEnabled(True)
            self.maxTime.setMaximumTime(max_time)
            self.maxTime.setTime(max_time)
            self.loadButton.setEnabled(True)
            self.updateChart.setEnabled(True)
        else:
            self.__signal = None

    def __clear(self):
        self.__spectrum_analyser.clear()
        self.startTime.setEnabled(False)
        self.endTime.setEnabled(False)
        self.loadButton.setEnabled(False)
        self.updateChart.setEnabled(False)
        self.analysisResolution.clear()
        self.__signal = None
        self.__info = None
        self.__duration = 0

    def load_file(self):
        '''
        Loads a signal from the given file.
        '''
        start = end = None
        start_millis = self.startTime.time().msecsSinceStartOfDay()
        if start_millis > 0:
            start = start_millis
        end_millis = self.endTime.time().msecsSinceStartOfDay()
        if end_millis < self.__duration or start is not None:
            end = end_millis
        channel = int(self.channelSelector.currentText())
        from model.preferences import ANALYSIS_TARGET_FS
        from app import wait_cursor
        with wait_cursor(f"Loading {self.__info.name}"):
            self.__signal = readWav('analysis', self.__info.name, channel=channel, start=start, end=end,
                                    target_fs=self.__preferences.get(ANALYSIS_TARGET_FS))
            self.__filtered_signals = {}
            self.__init_resolution_selector(self.__signal)
            self.show_chart()

    def __init_resolution_selector(self, signal):
        ''' sets up the resolution selector based on the size of the underlying file. '''
        from model.report import block_signals
        with block_signals(self.analysisResolution):
            self.analysisResolution.clear()
            default_length = signal.getSegmentLength()
            for m in MULTIPLIERS:
                freq_res = float(signal.fs) / (default_length * m)
                time_res = (m * default_length) / signal.fs
                self.analysisResolution.addItem(f"{freq_res:.3f} Hz / {time_res:.3f} s")
            self.analysisResolution.setCurrentIndex(2)

    def show_chart(self):
        '''
        Analyses the data for the currently selected chart.
        '''
        if self.__signal is not None:
            filtered_signal = self.__get_filtered_signal()
            idx = self.analysisTabs.currentIndex()
            if idx == 0:
                self.__spectrum_analyser.right = self.__signal
                self.__spectrum_analyser.left = filtered_signal
                self.__spectrum_analyser.analyse()
            elif idx == 1:
                self.__waveform_analyser.signal = filtered_signal if filtered_signal is not None else self.__signal
                self.__waveform_analyser.analyse()

    def update_chart(self):
        ''' updates the currently selected chart. '''
        if self.__signal is not None:
            idx = self.analysisTabs.currentIndex()
            if idx == 0:
                self.__spectrum_analyser.update_chart()
        elif self.analysisTabs.count() == 1:
            if self.__spectrum_analyser.left is not None or self.__spectrum_analyser.right is not None:
                self.__spectrum_analyser.update_chart()

    def update_filter(self, idx):
        ''' reacts to filter changes '''
        if self.__signal is not None:
            if self.analysisTabs.currentIndex() == 0:
                self.__spectrum_analyser.right = self.__signal
                self.__spectrum_analyser.left = self.__get_filtered_signal()
                self.__spectrum_analyser.analyse()
            elif self.analysisTabs.currentIndex() == 1:
                self.show_chart()

    def __get_filtered_signal(self):
        sig = next((s for s in self.__signal_model if s.name == self.copyFilter.currentText()), None)
        if sig is None:
            return None
        else:
            if sig.name not in self.__filtered_signals:
                start = time.time()
                filt = sig.filter
                self.__filtered_signals[sig.name] = self.__signal.sosfilter(filt.resample(self.__signal.fs).get_sos())
                end = time.time()
                logger.debug(f"Filtered in {round(end - start, 3)}s")
            return self.__filtered_signals[sig.name]

    def show_limits(self):
        idx = self.analysisTabs.currentIndex()
        if idx == 1:
            self.__waveform_analyser.show_limits()

    def set_mag_range_type(self, type):
        self.__spectrum_analyser.set_mag_range_type(type)

    def reject(self):
        ''' ensure signals are cleared from memory '''
        self.__clear()
        super().reject()


class WaveformRange:
    '''
    A range calculator that just returns -1/1 or the signal range
    '''

    def __init__(self, is_db=False):
        self.is_db = is_db

    def calculate(self, y_range):
        if self.is_db is True:
            return y_range[0], 0.0
        else:
            return -1.0, 1.0


class Waveform:
    '''
    An analyser that simply shows the waveform.
    '''

    def __init__(self, chart, ui):
        self.__chart = chart
        self.__ui = ui
        self.__axes = self.__chart.canvas.figure.add_subplot(111)
        self.__waveform_range = WaveformRange(is_db=self.__ui.magnitudeDecibels.isChecked())
        self.__limits = Limits('waveform', self.__redraw, self.__axes, x_axis_configurer=self.configure_time_axis,
                               y_range_calculator=self.__waveform_range, x_lim=(0, 1), x_scale='linear')
        self.__signal = None
        self.__curve = None

    def configure_time_axis(self, axes, x_scale):
        axes.set_xscale(x_scale)
        axes.set_xlabel('Time')
        axes.xaxis.set_major_formatter(FuncFormatter(seconds_to_hhmmss))

    @property
    def signal(self):
        return self.__signal

    @signal.setter
    def signal(self, signal):
        self.__signal = signal
        self.__limits.x_min = 0
        self.__limits.x_max = 1
        if signal is not None:
            self.__limits.x_max = signal.duration_seconds
            headroom = 20 * math.log(1.0 / np.nanmax(np.abs(signal.samples)), 10)
        else:
            headroom = 0.0
        self.__ui.headroom.setValue(headroom)
        if self.__curve is not None:
            self.__curve.set_data([], [])
        self.__init_chart()

    def __init_chart(self, draw=False):
        self.__limits.propagate_to_axes(draw=False)
        self.__axes.grid(linestyle='-', which='major', linewidth=1, alpha=0.5)
        self.__axes.grid(linestyle='--', which='minor', linewidth=1, alpha=0.5)
        if draw is True:
            self.__redraw()

    def __redraw(self):
        self.__chart.canvas.draw_idle()

    def clear(self):
        '''
        Resets the analyser.
        '''
        self.signal = None

    def show_limits(self):
        '''
        Shows the graph limits dialog.
        '''
        if self.signal is not None:
            LimitsDialog(self.__limits, x_min=0, x_max=self.signal.duration_seconds, y1_min=-1, y1_max=1).exec()

    def analyse(self):
        '''
        Calculates the spectrum view.
        '''
        from app import wait_cursor
        with wait_cursor(f"Analysing"):
            step = 1.0 / self.signal.fs
            x = np.arange(0, self.signal.duration_seconds, step)
            y = self.signal.samples
            if self.__ui.magnitudeDecibels.isChecked():
                y = np.copy(y)
                y[y == 0.0] = 0.000000001
                y = 20 * np.log10(np.abs(y))
                self.__limits.y1_max = 0.0
                self.__limits.y1_min = math.floor(np.min(y))
            else:
                self.__limits.y1_min = -1.0
                self.__limits.y1_max = 1.0
            self.__waveform_range.is_db = self.__ui.magnitudeDecibels.isChecked()
            if self.__curve is None:
                self.__curve = self.__axes.plot(x, y, linewidth=1, color='cyan')[0]
            else:
                self.__curve.set_data(x, y)
                self.__limits.on_data_change((self.__limits.y1_min, self.__limits.y1_max), [])
            self.__limits.propagate_to_axes(draw=True)


class OnePlusRange:
    def __init__(self):
        pass

    def calculate(self, y_range):
        return y_range[0], y_range[1] + 1


class MaxSpectrumByTime:
    '''
    An analyser that highlights where the heavy hits are in time by frequency
    '''

    def __init__(self, chart, preferences, ui):
        self.__chart = chart
        self.__ui = ui
        self.__preferences = preferences
        self.__layout_change = False

        self.__left_axes = None
        self.__left_signal = None
        self.__left_scatter = None
        self.__left_cache = {}

        self.__right_axes = None
        self.__right_signal = None
        self.__right_scatter = None
        self.__right_cache = {}

        self.__cb = None
        self.__current_marker = None
        self.__current_ellipse_width = None
        self.__current_ellipse_height = None

        self.__width_ratio = None
        self.__init_mag_range()

    @property
    def right(self):
        return self.__right_signal

    @right.setter
    def right(self, right):
        self.__right_signal = right
        self.__clear_scatter(self.__right_scatter)

    def __clear_scatter(self, scatter):
        if scatter is not None:
            scatter.set_offsets(np.c_[np.array([]), np.array([])])
            scatter.set_array(np.array([]))

    @property
    def left(self):
        return self.__left_signal

    @left.setter
    def left(self, left):
        self.__layout_change = (
                (left is None and self.__left_signal is not None)
                or
                (left is not None and self.__left_signal is None)
        )
        self.__left_signal = left
        self.__clear_scatter(self.__left_scatter)

    def set_mag_range_type(self, type):
        ''' updates the chart controls '''
        mag_max = None
        if 'sxx' in self.__right_cache:
            mag_max = np.max(self.__right_cache['sxx'])
        if 'sxx' in self.__left_cache:
            mag_max = max(mag_max, np.max(self.__left_cache['sxx']))
        if type == 'Constant':
            self.__ui.magUpperLimit.setVisible(True)
            self.__ui.magLowerLimit.setVisible(True)
            self.__ui.signalRangeLabel.setVisible(True)
            if mag_max:
                self.__ui.magUpperLimit.setValue(mag_max)
                self.__ui.magLowerLimit.setValue(mag_max - 60.0)
        elif type == 'Peak':
            self.__ui.magUpperLimit.setVisible(False)
            self.__ui.magLowerLimit.setVisible(True)
            self.__ui.magLowerLimit.setValue(-60.0)
            self.__ui.signalRangeLabel.setVisible(True)
        else:
            self.__ui.magUpperLimit.setVisible(False)
            self.__ui.magLowerLimit.setVisible(False)
            self.__ui.signalRangeLabel.setVisible(False)

    def __redraw(self):
        self.__chart.canvas.draw_idle()

    def clear(self):
        '''
        Resets the analyser.
        '''
        self.left = None
        self.right = None

    def update_chart(self):
        ''' Updates the chart for the cached data'''
        from app import wait_cursor
        with wait_cursor(f"Updating"):
            self.__clear_on_layout_change()
            if self.__left_signal is None:
                self.__render_one_only()
            else:
                self.__render_both()
            if self.__cb is None:
                divider = make_axes_locatable(self.__right_axes)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                self.__cb = self.__right_axes.figure.colorbar(self.__right_scatter, cax=cax)
            self.__redraw()

    def __render_both(self):
        ''' renders two plots, one with the filtered and one without. '''
        if self.__left_axes is None:
            self.__width_ratio = self.__make_width_ratio()
            gs = GridSpec(1, 2, width_ratios=self.__make_width_ratio(adjusted=True), wspace=0.00)
            gs.tight_layout(self.__chart.canvas.figure)
            self.__left_axes = self.__chart.canvas.figure.add_subplot(gs.new_subplotspec((0, 0)))
            self.__add_grid(self.__left_axes)
            self.__right_axes = self.__chart.canvas.figure.add_subplot(gs.new_subplotspec((0, 1)))
            self.__add_grid(self.__right_axes)

        self.__left_scatter = self.__render_scatter(self.__left_cache, self.__left_axes,
                                                     self.__left_scatter,
                                                     self.__ui.maxFilteredFreq.value(), self.left)
        self.__set_limits(self.__left_axes, self.__ui.minFreq, self.__ui.maxFilteredFreq,
                          self.__ui.minTime, self.__ui.maxTime)
        self.__right_scatter = self.__render_scatter(self.__right_cache, self.__right_axes, self.__right_scatter,
                                                    self.__ui.maxUnfilteredFreq.value(), self.right)
        self.__right_axes.set_yticklabels([])
        self.__right_axes.get_yaxis().set_tick_params(length=0)
        self.__set_limits(self.__right_axes, self.__ui.minFreq, self.__ui.maxUnfilteredFreq,
                          self.__ui.minTime, self.__ui.maxTime)

    def __make_width_ratio(self, adjusted=False):
        a = self.__ui.maxFilteredFreq.value()
        b = self.__ui.maxUnfilteredFreq.value()
        if adjusted:
            a1 = a * 0.95
            b1 = b * 1.05
            return [a1 / (a + b1), b / (a + b1)]
        else:
            return [a, b]

    def __render_one_only(self):
        ''' renders a single plot with the unfiltered only '''
        if self.__right_axes is None:
            self.__right_axes = self.__chart.canvas.figure.add_subplot(111)
            self.__add_grid(self.__right_axes)
        self.__right_scatter = self.__render_scatter(self.__right_cache, self.__right_axes, self.__right_scatter,
                                                    self.__ui.maxUnfilteredFreq.value(), self.right)
        self.__set_limits(self.__right_axes, self.__ui.minFreq, self.__ui.maxUnfilteredFreq,
                          self.__ui.minTime, self.__ui.maxTime)

    def __add_grid(self, axes):
        ''' adds a grid to the given axes '''
        axes.grid(linestyle='-', which='major', linewidth=1, alpha=0.3)

    def __clear_on_layout_change(self):
        ''' Clears the chart if the layout has fundamentally changed. '''
        if not self.__layout_change:
            if self.__width_ratio is not None:
                current_width_ratio = self.__make_width_ratio()
                if self.__width_ratio[0] != current_width_ratio[0] or self.__width_ratio[1] != current_width_ratio[1]:
                    self.__layout_change = True
        if not self.__layout_change:
            self.__layout_change = (
                    self.__current_marker is not None
                    and
                    self.__current_marker != self.__ui.markerType.currentText()
            )
        if not self.__layout_change:
            self.__layout_change = (
                    (self.__current_ellipse_width is not None and not math.isclose(self.__current_ellipse_width,
                                                                                   self.__ui.ellipseWidth.value()))
                    or
                    (self.__current_ellipse_height is not None and not math.isclose(self.__current_ellipse_height,
                                                                                    self.__ui.ellipseHeight.value()))
            )
        if self.__layout_change is True:
            self.__chart.canvas.figure.clear()
            self.__left_axes = None
            self.__left_scatter = None
            self.__right_axes = None
            self.__right_scatter = None
            self.__cb = None
            self.__current_marker = None
            self.__current_ellipse_width = None
            self.__current_ellipse_height = None
            self.__width_ratio = None
            self.__layout_change = False

    def __set_limits(self, axes, x_min, x_max, y_min, y_max):
        axes.set_xlim(left=x_min.value(), right=x_max.value())
        axes.set_ylim(bottom=y_min.time().msecsSinceStartOfDay() / 1000.0,
                      top=y_max.time().msecsSinceStartOfDay() / 1000.0)

    def analyse(self):
        '''
        Calculates the spectrum view.
        '''
        from app import wait_cursor
        with wait_cursor(f"Analysing"):
            self.__cache_xyz(self.__left_signal, self.__left_cache)
            self.__cache_xyz(self.__right_signal, self.__right_cache)
            self.__init_mag_range()
            self.update_chart()

    def __init_mag_range(self):
        self.set_mag_range_type(self.__ui.magLimitType.currentText())

    def __render_scatter(self, cache, axes, scatter, max_freq, signal):
        ''' renders a scatter plot showing the biggest hits '''
        Sxx, f, resolution_shift, t, x, y, z = self.__load_from_cache(cache, signal)
        # determine the threshold based on the mode we're in
        if self.__ui.magLimitType.currentText() == 'Constant':
            Pthreshold = np.array([self.__ui.magLowerLimit.value()]).repeat(f.size)
        elif self.__ui.magLimitType.currentText() == 'Peak':
            Pthreshold = Sxx.max(axis=-1) + self.__ui.magLowerLimit.value()
        else:
            _, Pthreshold = signal.spectrum(resolution_shift=resolution_shift)

        vmax = self.__ui.colourUpperLimit.value()
        vmin = self.__ui.colourLowerLimit.value()
        stack = np.column_stack((x, y, z))
        # filter by signal level
        above_threshold = stack[stack[:, 2] >= np.tile(Pthreshold, t.size)]
        # filter by graph limis
        above_threshold = above_threshold[above_threshold[:, 0] >= self.__ui.minFreq.value()]
        above_threshold = above_threshold[above_threshold[:, 0] <= max_freq]
        f_min = np.argmax(f >= self.__ui.minFreq.value())
        f_max = np.argmin(f <= max_freq)
        min_time = self.__ui.minTime.time().msecsSinceStartOfDay()
        max_time = self.__ui.maxTime.time().msecsSinceStartOfDay()
        t_min = 0
        if min_time > 0:
            above_threshold = above_threshold[above_threshold[:, 1] >= (min_time / 1000.0)]
            t_min = np.argmax(t >= (min_time / 1000.0))
        above_threshold = above_threshold[above_threshold[:, 1] <= (max_time / 1000.0)]
        if max_time / 1000.0 >= t[-1]:
            t_max = -1
        else:
            t_max = np.argmin(t <= (max_time / 1000.0))
        # sort so the lowest magnitudes are plotted first (as later plots overlay earlier ones)
        above_threshold = above_threshold[above_threshold[:, 2].argsort()]
        # then split into the constituent columns for plotting
        x = above_threshold[:, 0]
        y = above_threshold[:, 1]
        z = above_threshold[:, 2]
        # marker size
        s = matplotlib.rcParams['lines.markersize'] ** 2.0 * (self.__ui.markerSize.value() ** 2)
        # now plot or update
        if scatter is None:
            self.__current_marker = self.__ui.markerType.currentText()
            if self.__current_marker == SPECTROGRAM_FLAT:
                imshow_data = np.copy(Sxx)
                pixels = imshow_data[f_min:f_max, t_min:t_max]
                scatter = axes.pcolormesh(f[f_min:f_max], t[t_min:t_max], pixels.transpose(), vmin=vmin, vmax=vmax)
            elif self.__current_marker == SPECTROGRAM_CONTOURED:
                scatter = axes.tricontourf(x, y, z, np.sort(np.arange(vmax, vmin, -0.5)), vmin=vmin, vmax=vmax)
            else:
                marker = '.'
                if self.__current_marker == POINT:
                    pass
                elif self.__current_marker == ELLIPSE:
                    rx, ry = self.__ui.ellipseWidth.value(), self.__ui.ellipseHeight.value()
                    area = rx * ry * np.pi
                    theta = np.arange(0, 2 * np.pi + 0.01, 0.1)
                    marker = np.column_stack([rx / area * np.cos(theta), ry / area * np.sin(theta)])
                    self.__current_ellipse_width = rx
                    self.__current_ellipse_height = ry
                scatter = axes.scatter(x, y, c=z, s=s, vmin=vmin, vmax=vmax, marker=marker)
            axes.yaxis.set_major_formatter(FuncFormatter(seconds_to_hhmmss))
            axes.yaxis.set_major_locator(MaxNLocator(nbins=24, min_n_ticks=8, steps=[1, 3, 6]))
        else:
            if self.__current_marker == POINT or self.__current_marker == ELLIPSE:
                new_data = np.c_[x, y]
                scatter.set_offsets(new_data)
                scatter.set_clim(vmin=vmin, vmax=vmax)
                scatter.set_array(z)
                scatter.set_sizes(np.full(new_data.size, s))
            elif self.__current_marker == SPECTROGRAM_FLAT:
                pass
            elif self.__current_marker == SPECTROGRAM_CONTOURED:
                pass
        return scatter

    def __cache_xyz(self, signal, cache):
        ''' analyses the signal and caches the data '''
        if signal is not None:
            multiplier_idx = self.__ui.analysisResolution.currentIndex()
            resolution_shift = math.log(MULTIPLIERS[multiplier_idx], 2)
            f, t, Sxx = signal.spectrogram(resolution_shift=resolution_shift)
            x = f.repeat(t.size)
            y = np.tile(t, f.size)
            z = Sxx.flatten()
            cache['sxx'] = Sxx
            cache['f'] = f
            cache['res_shift'] = resolution_shift
            cache['t'] = t
            cache['x'] = x
            cache['y'] = y
            cache['z'] = z

    def __load_from_cache(self, cache, signal):
        ''' loads the signal from the cache, recalculating it if necessary. '''
        res_idx = self.__ui.analysisResolution.currentIndex()
        if 'res_shift' in cache and cache['res_shift'] != math.log(MULTIPLIERS[res_idx], 2):
            self.__cache_xyz(signal, cache)
        return cache['sxx'], cache['f'], cache['res_shift'], cache['t'], cache['x'], cache['y'], cache['z']


class SlaveRange:
    def __init__(self, vals):
        self.__vals = vals

    def calculate(self, y_range):
        return self.__vals


def seconds_to_hhmmss(x, pos):
    ''' formats a seconds value to hhmmss '''
    return str(datetime.timedelta(seconds=x))
