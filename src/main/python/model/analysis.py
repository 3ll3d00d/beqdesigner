import logging
import math

import numpy as np
import qtawesome as qta
from mpl_toolkits.axes_grid1 import make_axes_locatable
from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QDialog

from model.limits import Limits, LimitsDialog
from model.signal import select_file, readWav
from ui.analysis import Ui_analysisDialog

logger = logging.getLogger('analysis')


class AnalyseSignalDialog(QDialog, Ui_analysisDialog):
    def __init__(self, preferences):
        super(AnalyseSignalDialog, self).__init__()
        self.setupUi(self)
        self.setWindowFlags(self.windowFlags() | Qt.WindowSystemMenuHint | Qt.WindowMinMaxButtonsHint)
        self.__preferences = preferences
        self.filePicker.setIcon(qta.icon('fa.folder-open-o'))
        self.showLimitsButton.setIcon(qta.icon('ei.move'))
        self.__info = None
        self.__signal = None
        self.__spectrum_analyser = MaxSpectrumByTime(self.spectrumChart, self.__preferences, self)
        self.__waveform_analyser = Waveform(self.waveformChart, self)
        self.__duration = 0
        self.loadButton.setEnabled(False)
        self.__clear()

    def select_wav_file(self):
        '''
        Allows the user to select a file and laods info about it
        '''
        file = select_file(self, 'wav')
        if file is not None:
            self.__clear()
            self.file.setText(file)
            import soundfile as sf
            self.__info = sf.info(file)
            self.channelSelector.clear()
            for i in range(0, self.__info.channels):
                self.channelSelector.addItem(f"{i+1}")
            self.channelSelector.setEnabled(self.__info.channels > 1)
            self.startTime.setTime(QtCore.QTime(0, 0, 0))
            self.startTime.setEnabled(True)
            self.__duration = math.floor(self.__info.duration * 1000)
            self.endTime.setTime(QtCore.QTime(0, 0, 0).addMSecs(self.__duration))
            self.endTime.setEnabled(True)
            self.loadButton.setEnabled(True)
        else:
            self.__signal = None

    def __clear(self):
        self.__spectrum_analyser.clear()
        self.startTime.setEnabled(False)
        self.endTime.setEnabled(False)
        self.loadButton.setEnabled(False)
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
            self.__spectrum_analyser.signal = self.__signal
            self.__waveform_analyser.signal = self.__signal
            self.show_chart()

    def show_chart(self):
        '''
        Shows the currently selected chart.
        '''
        if self.__signal is not None:
            idx = self.analysisTabs.currentIndex()
            if idx == 0:
                self.__spectrum_analyser.analyse()
            elif idx == 1:
                self.__waveform_analyser.analyse()

    def show_limits(self):
        idx = self.analysisTabs.currentIndex()
        if idx == 0:
            self.__spectrum_analyser.show_limits()
        elif idx == 1:
            self.__waveform_analyser.show_limits()

    def allow_clip_choice(self):
        if self.clipAtAverage.isChecked():
            self.clipToAbsolute.setEnabled(False)
            self.dbRange.setEnabled(False)
        else:
            self.dbRange.setEnabled(True)
            self.clipToAbsolute.setEnabled(True)
        self.show_chart()

    def clip_to_abs(self):
        if self.clipToAbsolute.isChecked():
            self.clipAtAverage.setEnabled(False)
            self.dbRange.setEnabled(True)
        else:
            self.clipAtAverage.setEnabled(True)
        self.show_chart()


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

    @property
    def signal(self):
        return self.__signal

    @signal.setter
    def signal(self, signal):
        self.__signal = signal
        self.__limits.x_min = 0
        self.__limits.x_max = 1
        if signal is not None:
            self.__limits.x_max = signal.durationSeconds
        if self.__curve is not None:
            self.__curve.set_data([])
        self.__init_chart(draw=True)

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
            LimitsDialog(self.__limits, x_min=0, x_max=self.signal.durationSeconds, y1_min=-1, y1_max=1).exec()

    def analyse(self):
        '''
        Calculates the spectrum view.
        '''
        from app import wait_cursor
        with wait_cursor(f"Analysing"):
            step = 1.0 / self.signal.fs
            x = np.arange(0, self.signal.durationSeconds, step)
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
        return y_range[0] - 1, y_range[1] + 1


class MaxSpectrumByTime:
    '''
    An analyser that highlights where the heavy hits are in time by frequency
    '''

    def __init__(self, chart, preferences, ui):
        self.__chart = chart
        self.__ui = ui
        self.__preferences = preferences
        self.__axes = self.__chart.canvas.figure.add_subplot(111)
        self.__limits = Limits('spectrum', self.__redraw, self.__axes, y_range_calculator=OnePlusRange(),
                               x_lim=(2, 120))
        self.__signal = None
        self.__scatter = None
        self.__scatter_cb = None

    @property
    def signal(self):
        return self.__signal

    @signal.setter
    def signal(self, signal):
        self.__signal = signal
        self.__limits.y1_min = -1
        self.__limits.y1_max = 1
        if signal is not None:
            self.__limits.y1_max = signal.durationSeconds
        if self.__scatter is not None:
            self.__scatter.set_offsets(np.c_[np.array([]), np.array([])])
            self.__scatter.set_array(np.array([]))
        self.__init_chart(draw=True)

    def __init_chart(self, draw=False):
        self.__limits.propagate_to_axes(draw=False)
        self.__axes.set_ylabel('Time')
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
            LimitsDialog(self.__limits, y1_min=0, y1_max=self.signal.durationSeconds).exec()

    def analyse(self):
        '''
        Calculates the spectrum view.
        '''
        from app import wait_cursor
        with wait_cursor(f"Analysing"):
            from model.preferences import ANALYSIS_RESOLUTION
            multiplier = int(1 / float(self.__preferences.get(ANALYSIS_RESOLUTION)))
            f, t, Sxx = self.signal.spectrogram(segmentLengthMultiplier=multiplier)
            x = np.tile(f, t.size)
            y = t.repeat(f.size)
            z = Sxx.flatten()
            if self.__ui.clipAtAverage.isChecked():
                _, Pthreshold = self.signal.spectrum(segmentLengthMultiplier=multiplier)
            else:
                if self.__ui.clipToAbsolute.isChecked():
                    Pthreshold = np.array([np.max(Sxx) + self.__ui.dbRange.value()]).repeat(f.size)
                else:
                    # add the dbRange because it's shown as a negative value
                    Pthreshold = Sxx.max(axis=-1) + self.__ui.dbRange.value()
            Pthreshold = np.tile(Pthreshold, t.size)
            vmax = math.ceil(np.max(Sxx.max(axis=-1)))
            vmin = vmax - self.__ui.colourRange.value()
            stack = np.column_stack((x, y, z))
            # filter by signal level
            above_threshold = stack[stack[:, 2] > Pthreshold]
            # filter by graph limis
            above_threshold = above_threshold[above_threshold[:, 0] >= self.__limits.x_min]
            above_threshold = above_threshold[above_threshold[:, 0] <= self.__limits.x_max]
            above_threshold = above_threshold[above_threshold[:, 1] >= self.__limits.y1_min]
            above_threshold = above_threshold[above_threshold[:, 1] <= self.__limits.y1_max]
            x = above_threshold[:, 0]
            y = above_threshold[:, 1]
            z = above_threshold[:, 2]
            # now plot or update
            if self.__scatter is None:
                self.__scatter = self.__axes.scatter(x, y, c=z, vmin=vmin, vmax=vmax)
                divider = make_axes_locatable(self.__axes)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                self.__scatter_cb = self.__axes.figure.colorbar(self.__scatter, cax=cax)
            else:
                new_data = np.c_[x, y]
                self.__scatter.set_offsets(new_data)
                self.__scatter.set_clim(vmin=vmin, vmax=vmax)
                self.__scatter.set_array(z)
            self.__limits.propagate_to_axes(draw=True)
