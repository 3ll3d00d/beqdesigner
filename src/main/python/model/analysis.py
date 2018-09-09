import math

import numpy as np
import qtawesome as qta
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QDialog

from model.limits import Limits, LimitsDialog
from model.signal import select_file, readWav
from ui.analysis import Ui_analysisDialog


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
        self.__spectrum_analyser = MaxSpectrumByTime(self.spectrumChart, self.__preferences)
        self.__duration = 0
        self.__waveform_axes = None
        self.loadButton.setEnabled(False)
        self.__clear()
        self.__init_charts()

    def __init_charts(self):
        self.__waveform_axes = self.waveformChart.canvas.figure.add_subplot(111)
        self.__init_chart(self.__waveform_axes)

    def __init_chart(self, axes):
        axes.set_ylabel('Time')
        axes.grid(linestyle='-', which='major', linewidth=1, alpha=0.5)
        axes.grid(linestyle='--', which='minor', linewidth=1, alpha=0.5)

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
        if self.__waveform_axes is not None:
            self.__waveform_axes.clear()

    def load_file(self):
        '''
        Loads a signal from the given file.
        '''
        start = end = None
        start_millis = self.startTime.time().msecsSinceStartOfDay()
        if start_millis > 0:
            start = start_millis
        end_millis = self.endTime.time().msecsSinceStartOfDay()
        if end_millis < self.__duration:
            end = end_millis
        channel = int(self.channelSelector.currentText())
        from model.preferences import ANALYSIS_TARGET_FS
        from app import wait_cursor
        with wait_cursor(f"Loading {self.__info.name}"):
            self.__signal = readWav('analysis', self.__info.name, channel=channel, start=start, end=end,
                                    target_fs=self.__preferences.get(ANALYSIS_TARGET_FS))
            self.__spectrum_analyser.signal = self.__signal
            self.show_chart()

    def show_chart(self):
        '''
        Shows the currently selected chart.
        '''
        idx = self.analysisTabs.currentIndex()
        if idx == 0:
            self.__spectrum_analyser.analyse()
        elif idx == 1:
            pass

    def show_limits(self):
        idx = self.analysisTabs.currentIndex()
        if idx == 0:
            self.__spectrum_analyser.show_limits()
        elif idx == 1:
            # LimitsDialog(self.__spectrum_limits).exec()
            pass


class MaxSpectrumByTime:
    def __init__(self, chart, preferences):
        self.__chart = chart
        self.__preferences = preferences
        self.__axes = self.__chart.canvas.figure.add_subplot(111)
        self.__limits = Limits('spectrum', self.__redraw, self.__axes, x_lim=(2, 250))
        self.__signal = None
        self.__scatter = None
        self.__scatter_cb = None

    @property
    def signal(self):
        return self.__signal

    @signal.setter
    def signal(self, signal):
        self.__signal = signal

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
        self.__axes.clear()
        self.__init_chart(draw=True)

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
        from model.preferences import ANALYSIS_RESOLUTION
        multiplier = int(1 / float(self.__preferences.get(ANALYSIS_RESOLUTION)))
        f, t, Sxx = self.signal.spectrogram(segmentLengthMultiplier=multiplier)
        max_mags = Sxx.max(axis=-1)
        max_times = t[Sxx.argmax(axis=-1)]
        self.__scatter = self.__axes.scatter(f, max_times, c=max_mags, cmap=get_cmap('viridis'))
        divider = make_axes_locatable(self.__axes)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        self.__scatter_cb = self.__axes.figure.colorbar(self.__scatter, cax=cax)
        self.__limits.on_data_change((np.min(t), np.max(t)), [])
        self.__limits.propagate_to_axes(draw=True)
