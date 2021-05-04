import logging
import math

import numpy as np
import qtawesome as qta
from matplotlib.font_manager import FontProperties
from qtpy.QtCore import QTime
from qtpy.QtGui import QFont
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QDialog

from model.magnitude import MagnitudeModel
from model.preferences import BM_LPF_OPTIONS, BASS_MANAGEMENT_LPF_FS
from model.signal import SignalDialog, SIGNAL_SOURCE_FILE, SIGNAL_CHANNEL, SingleChannelSignalData
from ui.stats import Ui_signalStatsDialog

logger = logging.getLogger('waveform')


class WaveformController:
    def __init__(self, preferences, signal_model, waveform_chart, spectrum_chart, signal_selector, rms_level, headroom,
                 crest_factor, bm_headroom, bm_lpf_position, bm_clip_before, bm_clip_after, is_filtered,
                 apply_hard_clip, start_time, end_time, show_spectrum_btn, hide_spectrum_btn, zoom_in_btn, zoom_out_btn,
                 compare_spectrum_btn, source_file, load_signal_btn, show_limits_btn, show_stats_btn, y_min, y_max,
                 bm_hpf, waveform_chart_btn):
        self.__is_visible = False
        self.__selected_name = None
        self.__preferences = preferences
        self.__signal_model = signal_model
        self.__current_signal = None
        self.__active_signal = None
        self.__is_filtered = is_filtered
        self.__apply_hard_clip = apply_hard_clip
        self.__bm_headroom = bm_headroom
        self.__bm_hpf = bm_hpf
        self.__bm_lpf_position = bm_lpf_position
        for x in BM_LPF_OPTIONS:
            self.__bm_lpf_position.addItem(x)
        self.__bm_clip_before = bm_clip_before
        self.__bm_clip_after = bm_clip_after
        self.__start_time = start_time
        self.__end_time = end_time
        self.__show_spectrum_btn = show_spectrum_btn
        self.__hide_spectrum_btn = hide_spectrum_btn
        self.__compare_spectrum_btn = compare_spectrum_btn
        self.__zoom_in_btn = zoom_in_btn
        self.__zoom_out_btn = zoom_out_btn
        self.__load_signal_btn = load_signal_btn
        self.__show_stats_btn = show_stats_btn
        self.__source_file = source_file
        self.__selector = signal_selector
        self.__waveform_chart_model = WaveformModel(waveform_chart, crest_factor, rms_level, headroom, start_time,
                                                    end_time, y_min, y_max, self.__on_x_range_change)
        spectrum_chart.vbl.setContentsMargins(1, 1, 1, 1)
        spectrum_chart.setVisible(False)
        self.__magnitude_model = MagnitudeModel('spectrum', spectrum_chart, preferences, self.get_curve_data,
                                                'Spectrum', show_legend=lambda: False)
        self.__show_limits_btn = show_limits_btn
        # match the pyqtgraph layout
        self.__load_signal_btn.clicked.connect(self.__load_signal)
        self.__show_spectrum_btn.setIcon(qta.icon('fa5s.eye'))
        self.__hide_spectrum_btn.setIcon(qta.icon('fa5s.eye-slash'))
        self.__zoom_in_btn.setIcon(qta.icon('fa5s.search-plus'))
        self.__zoom_out_btn.setIcon(qta.icon('fa5s.search-minus'))
        self.__load_signal_btn.setIcon(qta.icon('fa5s.folder-open'))
        self.__show_limits_btn.setIcon(qta.icon('fa5s.arrows-alt'))
        self.__compare_spectrum_btn.setIcon(qta.icon('fa5s.chart-area'))
        self.__show_stats_btn.setIcon(qta.icon('fa5s.info-circle'))
        waveform_chart_btn.setIcon(qta.icon('fa5s.save'))
        waveform_chart_btn.setToolTip('Export to Chart')
        self.__show_spectrum_btn.clicked.connect(self.show_spectrum)
        self.__hide_spectrum_btn.clicked.connect(self.hide_spectrum)
        self.__compare_spectrum_btn.clicked.connect(self.compare_spectrum)
        self.__show_limits_btn.clicked.connect(self.__magnitude_model.show_limits)
        self.__show_stats_btn.clicked.connect(self.show_stats)
        self.__is_filtered.stateChanged['int'].connect(self.toggle_filter)
        self.__apply_hard_clip.stateChanged['int'].connect(self.toggle_hard_clip)
        self.__bm_headroom.currentIndexChanged['QString'].connect(self.change_bm_headroom)
        self.__bm_lpf_position.currentIndexChanged['QString'].connect(self.change_bm_lpf_position)
        self.__bm_hpf.stateChanged['int'].connect(self.toggle_hpf)
        self.__bm_clip_before.stateChanged['int'].connect(self.toggle_bm_clip_before)
        self.__bm_clip_after.stateChanged['int'].connect(self.toggle_bm_clip_after)
        self.__selector.currentIndexChanged['QString'].connect(self.update_waveform)
        self.__zoom_in_btn.clicked.connect(self.__waveform_chart_model.zoom_in)
        self.__zoom_out_btn.clicked.connect(self.__zoom_out)
        waveform_chart_btn.clicked.connect(self.save_charts)
        self.update_waveform(None)

    def save_charts(self):
        self.__waveform_chart_model.export_chart()
        if self.__magnitude_model.is_visible() is True:
            self.__magnitude_model.export_chart()

    @property
    def is_visible(self):
        return self.__is_visible

    @property
    def selected_name(self):
        return self.__selected_name

    @selected_name.setter
    def selected_name(self, selected_name):
        self.__selected_name = selected_name

    def __zoom_out(self):
        ''' zooms out and hides the spectrum '''
        self.hide_spectrum()
        self.__waveform_chart_model.zoom_out()

    def __on_x_range_change(self):
        ''' updates the view if the x range changes. '''
        if self.__magnitude_model.is_visible():
            self.__magnitude_model.redraw()

    def get_curve_data(self, reference=None):
        '''
        :param reference: ignored as we don't expose a normalisation control in this chart.
        :return: the peak and avg spectrum for the currently filtered signal (if any).
        '''
        if self.__active_signal is not None and self.__magnitude_model.is_visible():
            sig = SingleChannelSignalData(signal=self.__active_signal.cut(to_seconds(self.__start_time),
                                                                          to_seconds(self.__end_time)))
            sig.reindex(self.__selector.currentIndex() - 1)
            return sig.get_all_xy()
        return []

    def on_visibility_change(self, show=True):
        '''
        Reacts to a change in visibility by deselecting the visible chart or showing the previously selected one.
        :param show: True if we're showing the chart.
        '''
        self.__is_visible = show
        if show is True:
            if self.__selected_name is not None:
                idx = self.__selector.findText(self.__selected_name)
                if idx > -1:
                    self.__selector.setCurrentIndex(idx)
                else:
                    if self.__selector.count() == 2:
                        self.__selector.setCurrentIndex(1)
                    else:
                        self.__reset_controls()
        else:
            self.__selected_name = self.__selector.currentText()
            self.__selector.setCurrentIndex(0)

    def refresh_selector(self):
        ''' Updates the selector with the available signals. '''
        currently_selected = self.__selector.currentText()
        from model.report import block_signals
        with block_signals(self.__selector):
            self.__selector.clear()
            self.__selector.addItem('  ')
            for bm in self.__signal_model.bass_managed_signals:
                self.__selector.addItem(f"(BM) {bm.name}")
                for c in bm.channels:
                    if c.signal is not None:
                        self.__selector.addItem(c.name)
                    else:
                        self.__selector.addItem(f"-- {c.name}")
            for s in self.__signal_model.non_bm_signals:
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
        self.__on_x_range_change()
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
            self.__bm_headroom.setEnabled(False)
            self.__bm_lpf_position.setEnabled(False)
            self.__bm_hpf.setEnabled(False)
            self.__bm_hpf.setChecked(False)
            self.__bm_clip_before.setEnabled(False)
            self.__bm_clip_after.setEnabled(False)
            self.__show_stats_btn.setEnabled(False)
            self.__reset_controls()
            self.__active_signal = None
        else:
            self.__load_signal_btn.setEnabled(False)
            self.__show_stats_btn.setEnabled(True)
            metadata = self.__current_signal.metadata
            if metadata is not None:
                if SIGNAL_CHANNEL in metadata:
                    self.__source_file.setText(f"{metadata[SIGNAL_SOURCE_FILE]} - C{metadata[SIGNAL_CHANNEL]}")
                else:
                    self.__source_file.setText(metadata[SIGNAL_SOURCE_FILE])
            from model.report import block_signals
            if signal_name.startswith('(BM) '):
                with block_signals(self.__bm_hpf):
                    self.__bm_hpf.setEnabled(False)
                    self.__bm_hpf.setChecked(False)
                with block_signals(self.__bm_headroom):
                    self.__bm_headroom.setEnabled(True)
                    self.__bm_headroom.setCurrentText(self.__current_signal.bm_headroom_type)
                with block_signals(self.__bm_lpf_position):
                    self.__bm_lpf_position.setEnabled(True)
                    self.__bm_lpf_position.setCurrentText(self.__current_signal.bm_lpf_position)
                with block_signals(self.__bm_clip_before):
                    self.__bm_clip_before.setEnabled(True)
                    self.__bm_clip_before.setChecked(self.__current_signal.clip_before)
                with block_signals(self.__bm_clip_after):
                    self.__bm_clip_after.setEnabled(True)
                    self.__bm_clip_after.setChecked(self.__current_signal.clip_after)
            else:
                if signal_name.endswith('LFE'):
                    with block_signals(self.__bm_hpf):
                        self.__bm_hpf.setEnabled(False)
                        self.__bm_hpf.setChecked(False)
                else:
                    with block_signals(self.__bm_hpf):
                        self.__bm_hpf.setEnabled(True)
                        self.__bm_hpf.setChecked(self.__current_signal.high_pass)
                self.__bm_headroom.setEnabled(False)
                self.__bm_lpf_position.setEnabled(False)
                self.__bm_clip_before.setEnabled(False)
                self.__bm_clip_after.setEnabled(False)
            self.__current_signal.register_listener(self.on_filter_update)
            self.__start_time.setEnabled(True)
            duration = QTime(0, 0, 0).addMSecs(self.__current_signal.duration_seconds * 1000.0)
            self.__start_time.setMaximumTime(duration)
            self.__end_time.setEnabled(True)
            self.__end_time.setMaximumTime(duration)
            self.__end_time.setTime(duration)
            self.toggle_filter(Qt.Checked if self.__is_filtered.isChecked() else Qt.Unchecked)

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
        if signal_name is not None and signal_name.startswith('(BM) '):
            return next((s for s in self.__signal_model.bass_managed_signals if s.name == signal_name[5:]), None)
        else:
            return next((s for s in self.__signal_model if s.name == signal_name), None)

    @staticmethod
    def __reset_time(time_widget):
        ''' resets and disables the supplied time field. '''
        from model.report import block_signals
        with block_signals(time_widget):
            time_widget.clearMaximumDateTime()
            time_widget.setTime(QTime())
            time_widget.setEnabled(False)

    def __set_bm_signal_attr(self, attr_name, attr_value):
        signal_name = self.__selector.currentText()
        if signal_name.startswith('(BM) '):
            signal_data = self.__get_signal_data(signal_name)
            if signal_data is not None:
                setattr(signal_data, attr_name, attr_value)
                self.toggle_filter(Qt.Checked if self.__is_filtered.isChecked() else Qt.Unchecked)

    def change_bm_headroom(self, headroom):
        ''' Changes the headroom allowed for bass management '''
        self.__set_bm_signal_attr('bm_headroom_type', headroom)

    def change_bm_lpf_position(self, lpf_position):
        ''' Changes the LPF applied during bass management '''
        self.__set_bm_signal_attr('bm_lpf_position', lpf_position)

    def toggle_bm_clip_before(self, state):
        ''' Changes whether to clip the signal before summation '''
        self.__set_bm_signal_attr('clip_before', state == Qt.Checked)

    def toggle_bm_clip_after(self, state):
        ''' Changes whether to clip the signal after summation '''
        self.__set_bm_signal_attr('clip_after', state == Qt.Checked)

    def toggle_hpf(self, state):
        ''' applies or removes the hpf from the visible waveform '''
        signal_name = self.__selector.currentText()
        signal_data = self.__get_signal_data(signal_name)
        if signal_data is not None:
            signal_data.high_pass = state == Qt.Checked
            from app import wait_cursor
            with wait_cursor():
                signal = signal_data.filter_signal(filt=self.__is_filtered.isChecked(),
                                                   clip=self.__apply_hard_clip.isChecked(),
                                                   post_filt=self.__get_post_filt_hpf(apply=state == Qt.Checked))
                self.__active_signal = signal
                self.__waveform_chart_model.signal = signal
                self.__waveform_chart_model.idx = self.__selector.currentIndex() - 1
                self.__waveform_chart_model.analyse()
                if self.__magnitude_model.is_visible():
                    self.__magnitude_model.redraw()

    def __get_post_filt_hpf(self, apply=None):
        post_filt = None
        if apply is None:
            apply = self.__bm_hpf.isChecked() and self.__bm_hpf.isEnabled()
        if apply is True:
            from model.iir import FilterType, ComplexHighPass
            hpf_fs = self.__preferences.get(BASS_MANAGEMENT_LPF_FS)
            post_filt = ComplexHighPass(FilterType.LINKWITZ_RILEY, 4, 1000, hpf_fs)
        return post_filt

    def toggle_filter(self, state):
        ''' Applies or removes the filter from the visible waveform '''
        signal_name = self.__selector.currentText()
        signal_data = self.__get_signal_data(signal_name)
        if signal_data is not None:
            from app import wait_cursor
            with wait_cursor():
                signal = signal_data.filter_signal(filt=state == Qt.Checked,
                                                   clip=self.__apply_hard_clip.isChecked(),
                                                   post_filt=self.__get_post_filt_hpf())
                self.__active_signal = signal
                self.__waveform_chart_model.signal = signal
                self.__waveform_chart_model.idx = self.__selector.currentIndex() - 1
                self.__waveform_chart_model.analyse()
                if self.__magnitude_model.is_visible():
                    self.__magnitude_model.redraw()

    def toggle_hard_clip(self, state):
        ''' Applies or removes the hard clip option from the visible waveform '''
        signal_name = self.__selector.currentText()
        signal_data = self.__get_signal_data(signal_name)
        if signal_data is not None:
            from app import wait_cursor
            with wait_cursor():
                signal = signal_data.filter_signal(filt=self.__is_filtered.isChecked(),
                                                   clip=state == Qt.Checked,
                                                   post_filt=self.__get_post_filt_hpf())
                self.__active_signal = signal
                self.__waveform_chart_model.signal = signal
                self.__waveform_chart_model.idx = self.__selector.currentIndex() - 1
                self.__waveform_chart_model.analyse()
                if self.__magnitude_model.is_visible():
                    self.__magnitude_model.redraw()

    def on_filter_update(self):
        ''' if the signal is filtered then updated the chart when the filter changes. '''
        if self.__is_filtered.isChecked():
            self.toggle_filter(Qt.Checked)

    def show_spectrum(self):
        ''' Updates the visible spectrum for the selected waveform limits '''
        self.__magnitude_model.set_visible(True)
        self.__magnitude_model.redraw()

    def hide_spectrum(self):
        ''' Resets the visible spectrum for the selected waveform limits '''
        self.__magnitude_model.set_visible(False)

    def compare_spectrum(self):
        from model.analysis import AnalyseSignalDialog
        AnalyseSignalDialog(self.__preferences, self.__signal_model, allow_load=False).exec()

    def show_stats(self):
        ''' Shows the signal stats dialog. '''
        StatsDialog(self.__active_signal, self.__waveform_chart_model.rms).exec()


class WaveformModel:
    '''
    Displays and interacts with a waveform that is linked to the spectrum view.
    '''

    def __init__(self, chart, crest_factor, rms_level, headroom, x_min, x_max, y_min, y_max, on_x_range_change):
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
        self.__rms_level = rms_level
        self.__crest_factor = crest_factor
        self.__signal = None
        self.__curve = None

    def export_chart(self):
        '''
        exports the waveform chart.
        '''
        from app import SaveChartDialog, PyQtGraphExportProcessor
        SaveChartDialog(self.__chart, 'waveform', self.__chart, PyQtGraphExportProcessor()).exec()

    @property
    def rms(self):
        return self.__rms_level.value()

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
        ''' changes the x limits in response to user input. '''
        max_secs, min_secs = self.__propagate_btn_state_on_xrange_change()
        self.__chart.getPlotItem().setXRange(min_secs, max_secs, padding=0.0)

    def __propagate_btn_state_on_xrange_change(self):
        min_secs, max_secs = self.get_time_range()
        self.__recalc_stats(self.signal.cut(min_secs, max_secs) if self.signal else None)
        self.__on_x_range_change()
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
        self.__recalc_stats(signal)
        x_range = self.__chart.getPlotItem().getAxis('bottom').range
        if self.signal is None:
            x_max = 1.0
        else:
            x_max = self.signal.duration_seconds
        self.__chart.getPlotItem().setLimits(xMin=0.0, xMax=x_max)
        if x_range[1] > x_max:
            self.__chart.getPlotItem().setXRange(x_range[0], x_max, padding=0.0)
        if self.__curve is not None and signal is None:
            self.__chart.getPlotItem().removeItem(self.__curve)
            self.__curve = None

    def __recalc_stats(self, signal):
        if signal is not None:
            peak_value = np.nanmax(np.abs(signal.samples))
            rms_level_raw = np.sqrt(np.mean(np.square(np.abs(signal.samples))))
            crest_factor = 20 * math.log(peak_value / rms_level_raw, 10)
            rms_level = 20 * math.log(rms_level_raw, 10)
            headroom = 20 * math.log(1.0 / peak_value, 10)
        else:
            crest_factor = 0.0
            rms_level = 0.0
            headroom = 0.0
        self.__headroom.setValue(headroom)
        self.__rms_level.setValue(rms_level)
        self.__crest_factor.setValue(crest_factor)

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
            x = np.linspace(0, self.signal.duration_seconds, endpoint=False, num=len(self.signal.samples))
            y = self.signal.samples
            if self.__curve is None:
                from pyqtgraph import mkPen
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
        return self.signal.duration_seconds if self.signal else 1


def to_seconds(time_widget):
    ''' yields a time in seconds from a QTimeEdit '''
    return time_widget.time().msecsSinceStartOfDay() / 1000.0


# TODO initialise signal dialog with values from the signal
class AssociateSignalDialog(SignalDialog):
    ''' Customises the signal dialog to allow a single channel to be loaded from a wav only '''

    def __init__(self, preferences, signal_data):
        super().__init__(preferences, None, allow_multichannel=False)
        self.__signal_data = signal_data
        self.signalTypeTabs.setTabEnabled(1, False)

    def save(self, signal):
        ''' writes the underlying signal into the existing signaldata '''
        self.__signal_data.signal = signal.signal

    def selectFile(self):
        ''' autopopulates and disables the signal name field (as it is irrelevant here) '''
        super().selectFile()
        self.wavSignalName.setText(self.__signal_data.name)
        self.wavSignalName.setEnabled(False)


class StatsDialog(QDialog, Ui_signalStatsDialog):

    def __init__(self, signal, rms, level_adjust=0.0, parent=None):
        from app import wait_cursor
        with wait_cursor():
            super(StatsDialog, self).__init__(parent=parent) if parent is not None else super(StatsDialog, self).__init__()
            self.__level_adjust = level_adjust
            self.__signal = signal
            self.setupUi(self)
            self.rms.setValue(rms)
            self.__set_peaks()
            self.set_extension()

    def set_composite(self):
        '''
        recalculates the composite level.
        '''
        values = []
        if self.includeRMS.isChecked():
            values.append(self.rms.value())
            logger.info(f"Including RMS in composite {values[-1]:g} dB")
        if self.includeFast.isChecked():
            values.append(self.fastPeak.value())
            logger.info(f"Including fast in composite {values[-1]:g} dB")
        if self.includeSlow.isChecked():
            values.append(self.slowPeak.value())
            logger.info(f"Including slow in composite {values[-1]:g} dB")
        if self.includeInstant.isChecked():
            values.append(self.instantPeak.value())
            logger.info(f"Including instant in composite {values[-1]:g} dB")
        if self.includeCustom.isChecked():
            values.append(self.customPeak.value())
            logger.info(f"Including Custom in composite {values[-1]:g} dB")
        if len(values) > 0:
            self.compositeLevel.setValue(10.0 * np.log10((10.0**(np.array(values) / 10.0)).mean()))
        else:
            self.compositeLevel.setValue(self.compositeLevel.minimum())
        self.rate_dynamics()

    def set_custom_peak(self):
        ''' calculates the custom peak based on the specified integration time. '''
        self.customPeak.setValue(self.__get_peak_value(self.__signal, self.customIntegrationTime.value() / 1000.0))

    def set_extension(self):
        '''
        finds the frequency of the peak frequency bin in both peak and average curves then finds the point at which
        the signal is lower by the specified dB.
        '''
        x_peak, y_peak = self.__find_extension(*self.__signal.peak)
        x_avg, y_avg = self.__find_extension(*self.__signal.avg)
        self.averageExtensionY2.setValue(y_avg[1])
        self.averageExtensionX2.setValue(x_avg[1])
        self.averageExtensionY1.setValue(y_avg[0])
        self.averageExtensionX1.setValue(x_avg[0])
        self.peakExtensionY2.setValue(y_peak[1])
        self.peakExtensionX2.setValue(x_peak[1])
        self.peakExtensionY1.setValue(y_peak[0])
        self.peakExtensionX1.setValue(x_peak[0])
        self.rate_extension()

    def __find_extension(self, x, y):
        upper_y_idx = np.argmax(y)
        upper_x = x[upper_y_idx]
        lower_y = y[upper_y_idx] + self.extensionLimit.value()
        lower_x = x[np.argmax(y >= lower_y)]
        return (lower_x, upper_x), (lower_y, y[upper_y_idx])

    def __set_peaks(self):
        self.instantPeak.setValue(self.__get_peak_value(self.__signal, 0.035))
        self.fastPeak.setValue(self.__get_peak_value(self.__signal, 0.125))
        self.slowPeak.setValue(self.__get_peak_value(self.__signal, 1.0))
        self.set_custom_peak()
        self.set_composite()

    def rate_level(self, val):
        adjusted_level = val + self.__level_adjust
        if adjusted_level >= -15.5:
            rate = 5
        elif adjusted_level >= -18.0:
            rate = 4
        elif adjusted_level >= -20.5:
            rate = 3
        elif adjusted_level >= -23.0:
            rate = 2
        else:
            rate = 1
        self.levelRating.setValue(rate)

    def rate_dynamics(self):
        range = self.instantPeak.value() - self.rms.value()
        self.dynamics.setValue(range)
        logger.info(f"Dynamics {range:g} - {self.instantPeak.value():g} / {self.rms.value():g}")
        if range >= 27.5:
            rate = 5
        elif range >= 25.0:
            rate = 4
        elif range >= 22.5:
            rate = 3
        elif range >= 20.0:
            rate = 4
        else:
            rate = 1
        self.dynamicsRating.setValue(rate)

    def rate_extension(self):
        ''' Rates extension based on the lower frequency limit. '''
        if self.peakExtensionX2.value() > self.averageExtensionX2.value():
            lower_limit = self.peakExtensionX1.value()
            src = 'Peak'
        elif math.isclose(self.peakExtensionX2.value(), self.averageExtensionX2.value()):
            lower_limit = min(self.peakExtensionX1.value(), self.averageExtensionX1.value())
            src = 'Peak' if lower_limit == self.peakExtensionX1.value() else 'Average'
        else:
            lower_limit = self.averageExtensionX1.value()
            src = 'Average'
        self.extensionSource.setCurrentText(src)
        if lower_limit < 10:
            rate = 5
        elif lower_limit < 15:
            rate = 4
        elif lower_limit < 20:
            rate = 3
        elif lower_limit < 25:
            rate = 2
        else:
            rate = 1
        self.extensionRating.setValue(rate)

    # get RMS via an exponentially weighted moving average
    # produces the same result as just getting the rms of the signal but leavng here for a reminder in case it is needed
    # @staticmethod
    # def window_rms(data, fs, window_size):
    #     import pandas as pd
    #     length = fs * window_size
    #     alpha = math.exp(-1/length)
    #     mean = pd.DataFrame(data=data)[0].ewm(alpha=alpha).mean().to_numpy()
    #     return mean

    @staticmethod
    def __get_peak_value(signal, window_size):
        window = int(signal.fs * window_size)
        to_pad = signal.samples.shape[-1] % window

        def rms(row):
            val = np.sqrt(np.mean(np.square(np.abs(row))))
            return 20*math.log(val, 10) if val > 0 else -120.0

        chunks = np.pad(signal.samples, (0, window - to_pad)).reshape(-1, window)
        rms_per_window = np.apply_along_axis(rms, 1, chunks)
        return np.max(rms_per_window)
