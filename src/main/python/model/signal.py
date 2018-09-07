import datetime
import logging
import math
import re
import time
import typing
from collections import Sequence
from pathlib import Path

import numpy as np
import resampy
from qtpy import QtCore
from qtpy.QtCore import QAbstractTableModel, QModelIndex, QVariant, Qt
from qtpy.QtWidgets import QDialog, QFileDialog, QDialogButtonBox
from scipy import signal

from model.codec import signaldata_to_json
from model.iir import XYData, CompleteFilter
from model.magnitude import MagnitudeModel
from model.preferences import AVG_COLOURS, PEAK_COLOURS, get_avg_colour, get_peak_colour, SHOW_PEAK, \
    SHOW_AVERAGE, SHOW_FILTERED_ONLY, SHOW_UNFILTERED_ONLY, DISPLAY_SHOW_SIGNALS, \
    DISPLAY_SHOW_FILTERED_SIGNALS
from ui.signal import Ui_addSignalDialog

logger = logging.getLogger('signal')

""" speclab reports a peak of 0dB but, by default, we report a peak of -3dB """
SPECLAB_REFERENCE = 1 / (2 ** 0.5)


class SignalData:
    '''
    Provides a mechanism for caching the assorted xy data surrounding a signal.
    '''

    def __init__(self, name, fs, xy_data, filter, duration_hhmmss=None, start_hhmmss=None, end_hhmmss=None):
        self.__filter = None
        self.__name = name
        self.fs = fs
        self.duration_hhmmss = duration_hhmmss
        self.start_hhmmss = start_hhmmss
        self.end_hhmmss = end_hhmmss
        self.raw = xy_data
        self.__slaves = []
        self.master = None
        self.filtered = []
        self.reference_name = None
        self.reference = []
        self.filter = filter

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        self.__name = name
        self.raw[0].internal_name = name
        self.raw[1].internal_name = name
        if len(self.filtered) == 2:
            self.filtered[0].internal_name = name
            self.filtered[1].internal_name = name

    @property
    def filter(self):
        return self.__filter

    @filter.setter
    def filter(self, filt):
        self.__filter = filt
        self.__filter.listener = self
        self.on_filter_change(filt)
        for s in self.__slaves:
            s.on_filter_change(filt)

    def __repr__(self) -> str:
        return f"SignalData {self.name}-{self.fs}"

    def reindex(self, idx):
        self.raw[0].colour = get_avg_colour(idx)
        self.raw[1].colour = get_peak_colour(idx)
        if len(self.filtered) == 2:
            self.filtered[0].colour = get_avg_colour(idx)
            self.filtered[1].colour = get_peak_colour(idx)

    def enslave(self, signal):
        '''
        Allows a signal to be linked to this one so they share the same filter.
        :param signal the signal.
        '''
        logger.debug(f"Enslaving {signal} to {self}")
        self.__slaves.append(signal)
        signal.master = self
        signal.on_filter_change(self.__filter)

    def free(self, signal):
        '''
        unlinks a signal from this one.
        :param signal: the signal to unlink
        '''
        logger.debug(f"Freeing {signal} from {self}")
        self.__slaves.remove(signal)
        signal.master = None
        signal.on_filter_change(None)

    def on_reference_change(self):
        pass

    def get_all_xy(self):
        '''
        :return all the xy data
        '''
        return self.raw + self.filtered

    def on_filter_change(self, filt):
        '''
        Updates the cached filtered response when the filter changes.
        :param filt: the filter.
        '''
        if filt is None:
            self.filtered = []
            for r in self.raw:
                r.linestyle = '-'
        elif isinstance(filt, CompleteFilter):
            # TODO detect if the filter has changed and only recalc if it has
            filter_mag = filt.getTransferFunction().getMagnitude()
            self.filtered = [f.filter(filter_mag) for f in self.raw]
            for r in self.raw:
                r.linestyle = '--'
        else:
            raise ValueError(f"Unsupported filter type {filt}")


class SignalModel(Sequence):
    '''
    A model to hold onto the signals.
    '''

    def __init__(self, view, default_signal, preferences, on_update=lambda _: True):
        self.__signals = []
        self.default_signal = default_signal
        self.__view = view
        self.__on_update = on_update
        self.__preferences = preferences
        self.__table = None

    @property
    def table(self):
        return self.__table

    @table.setter
    def table(self, table):
        self.__table = table
        self.__table.resizeColumns(self.__view)

    def __getitem__(self, i):
        return self.__signals[i]

    def __len__(self):
        return len(self.__signals)

    def to_json(self):
        '''
        :return: a json compatible format of the data in the model.
        '''
        return [signaldata_to_json(x) for x in self.__signals]

    def __decorate_edit(self, func):
        def wrapper(*args, **kwargs):
            if self.__table is not None:
                self.__table.beginResetModel()
            func(*args, **kwargs)
            self.post_update()
            if self.__table is not None:
                self.__table.endResetModel()

        wrapper()

    def add(self, signal):
        '''
        Add the supplied signals ot the model.
        :param signals: the signal.
        '''

        def do_add():
            signal.reindex(len(self.__signals))
            self.__signals.append(signal)

        self.__decorate_edit(do_add)

    def post_update(self):
        from app import flatten
        show_signals = self.__preferences.get(DISPLAY_SHOW_SIGNALS)
        show_filtered_signals = self.__preferences.get(DISPLAY_SHOW_FILTERED_SIGNALS)
        pattern = self.__get_visible_signal_name_filter(show_filtered_signals, show_signals)
        visible_signal_names = [x.name for x in flatten([y for x in self.__signals for y in x.get_all_xy()])]
        if pattern is not None:
            visible_signal_names = [x for x in visible_signal_names if pattern.match(x) is not None]
        self.__on_update(visible_signal_names)
        self.__table.resizeColumns(self.__view)

    def __get_visible_signal_name_filter(self, show_filtered_signals, show_signals):
        '''
        Creates a regex that will filter the signal names according to the avg/peak patterns.
        :param show_filtered_signals: which filtered signals to show.
        :param show_signals: which signals to show.
        :return: the pattern (if any)
        '''
        pattern = None
        if show_signals == SHOW_AVERAGE:
            if show_filtered_signals == SHOW_FILTERED_ONLY:
                pattern = re.compile(".*_avg-filtered$")
            elif show_filtered_signals == SHOW_UNFILTERED_ONLY:
                pattern = re.compile(".*_avg$")
            else:
                pattern = re.compile(".*_avg(-filtered)?$")
        elif show_signals == SHOW_PEAK:
            if show_filtered_signals == SHOW_FILTERED_ONLY:
                pattern = re.compile(".*_peak-filtered$")
            elif show_filtered_signals == SHOW_UNFILTERED_ONLY:
                pattern = re.compile(".*_peak$")
            else:
                pattern = re.compile(".*_peak(-filtered)?$")
        else:
            if show_filtered_signals == SHOW_FILTERED_ONLY:
                pattern = re.compile(".*_(avg|peak)-filtered$")
            elif show_filtered_signals == SHOW_UNFILTERED_ONLY:
                pattern = re.compile(".*_(avg|peak)")
        return pattern

    def remove(self, signal):
        '''
        Remove the specified signal from the model.
        :param signal: the signal to remove.
        '''

        def do_remove():
            self.__signals.remove(signal)
            for idx, s in enumerate(self.__signals):
                s.reindex(idx)

        self.__decorate_edit(do_remove)

    def delete(self, indices):
        '''
        Delete the signals at the given indices.
        :param indices: the indices to remove.
        '''

        def do_delete():
            self.__signals = [s for idx, s in enumerate(self.__signals) if idx not in indices]

        self.__decorate_edit(do_delete)

    def getMagnitudeData(self, reference=None):
        '''
        :param reference: the curve against which to normalise.
        :return: the peak and avg spectrum for the signals (if any) + the filter signals.
        '''
        from app import flatten
        results = list(flatten([s.get_all_xy() for s in self.__signals]))
        show_signals = self.__preferences.get(DISPLAY_SHOW_SIGNALS)
        show_filtered_signals = self.__preferences.get(DISPLAY_SHOW_FILTERED_SIGNALS)
        pattern = self.__get_visible_signal_name_filter(show_filtered_signals, show_signals)
        if pattern is not None:
            results = [x for x in results if pattern.match(x.name) is not None]
        if reference is not None:
            ref_data = next((x for x in results if x.name == reference), None)
            if ref_data:
                results = [x.normalise(ref_data) for x in results]
        return results

    def replace(self, signals):
        '''
        Replaces the contents of the model with the supplied signals
        :param signals: the signals
        '''

        def do_replace():
            self.__signals = signals
            for idx, s in enumerate(self.__signals):
                s.reindex(idx)

        self.__decorate_edit(do_replace)


class Signal:
    """ a source models some input to the analysis system, it provides the following attributes:
        :var samples: an ndarray that represents the signal itself
        :var fs: the sample rate
    """

    def __init__(self, name, samples, fs=48000):
        self.name = name
        self.samples = samples
        self.fs = fs
        self.durationSeconds = len(self.samples) / self.fs
        self.startSeconds = 0
        self.end = self.durationSeconds
        # formatted for display purposes
        self.duration_hhmmss = str(datetime.timedelta(seconds=self.durationSeconds))
        self.start_hhmmss = str(datetime.timedelta(seconds=self.startSeconds))
        self.end_hhmmss = self.duration_hhmmss
        self.__avg = None
        self.__peak = None
        self.__cached = []

    def getSegmentLength(self):
        """
        Calculates a segment length such that the frequency resolution of the resulting analysis is in the region of 
        ~1Hz subject to a lower limit of the number of samples in the signal.
        For example, if we have a 10s signal with an fs is 500 then we convert fs-1 to the number of bits required to 
        hold this number in binary (i.e. 111110011 so 9 bits) and then do 1 << 9 which gives us 100000000 aka 512. Thus
        we have ~1Hz resolution.
        :return: the segment length.
        """
        return min(1 << (self.fs - 1).bit_length(), self.samples.shape[-1])

    def raw(self):
        """
        :return: the raw sample data
        """
        return self.samples

    def _cq(self, analysisFunc, segmentLengthMultipler=1):
        slices = []
        initialNperSeg = self.getSegmentLength() * segmentLengthMultipler
        nperseg = initialNperSeg
        # the no of slices is based on a requirement for approximately 1Hz resolution to 128Hz and then halving the
        # resolution per octave. We calculate this as the
        # bitlength(fs) - bitlength(128) + 2 (1 for the 1-128Hz range and 1 for 2**n-fs Hz range)
        bitLength128 = int(128).bit_length()
        for x in range(0, (self.fs - 1).bit_length() - bitLength128 + 2):
            f, p = analysisFunc(x, nperseg)
            n = round(2 ** (x + bitLength128 - 1) / (self.fs / nperseg))
            m = 0 if x == 0 else round(2 ** (x + bitLength128 - 2) / (self.fs / nperseg))
            slices.append((f[m:n], p[m:n]))
            nperseg /= 2
        f = np.concatenate([n[0] for n in slices])
        p = np.concatenate([n[1] for n in slices])
        return f, p

    def spectrum(self, ref=SPECLAB_REFERENCE, segmentLengthMultiplier=1, mode=None, window=None, **kwargs):
        """
        analyses the source to generate the linear spectrum.
        :param ref: the reference value for dB purposes.
        :param segmentLengthMultiplier: allow for increased resolution.
        :param mode: cq or none.
        :return:
            f : ndarray
            Array of sample frequencies.
            Pxx : ndarray
            linear spectrum.
        """

        def analysisFunc(x, nperseg, **kwargs):
            f, Pxx_spec = signal.welch(self.samples, self.fs, nperseg=nperseg, scaling='spectrum', detrend=False,
                                       window=window if window else 'hann', **kwargs)
            Pxx_spec = np.sqrt(Pxx_spec)
            # it seems a 3dB adjustment is required to account for the change in nperseg
            if x > 0:
                Pxx_spec = amplitude_to_db(Pxx_spec, ref * SPECLAB_REFERENCE)
            else:
                Pxx_spec = amplitude_to_db(Pxx_spec, ref)
            return f, Pxx_spec

        if mode == 'cq':
            return self._cq(analysisFunc, segmentLengthMultiplier)
        else:
            return analysisFunc(segmentLengthMultiplier, self.getSegmentLength() * segmentLengthMultiplier, **kwargs)

    def peakSpectrum(self, ref=SPECLAB_REFERENCE, segmentLengthMultiplier=1, mode=None, window=None):
        """
        analyses the source to generate the max values per bin per segment
        :param segmentLengthMultiplier: allow for increased resolution.
        :param mode: cq or none.
        :param window: window type.
        :return:
            f : ndarray
            Array of sample frequencies.
            Pxx : ndarray
            linear spectrum max values.
        """

        def analysisFunc(x, nperseg):
            freqs, _, Pxy = signal.spectrogram(self.samples,
                                               self.fs,
                                               window=window if window else ('tukey', 0.25),
                                               nperseg=int(nperseg),
                                               noverlap=int(nperseg // 2),
                                               detrend=False,
                                               scaling='spectrum')
            Pxy_max = np.sqrt(Pxy.max(axis=-1).real)
            if x > 0:
                Pxy_max = amplitude_to_db(Pxy_max, ref=ref * SPECLAB_REFERENCE)
            else:
                Pxy_max = amplitude_to_db(Pxy_max, ref=ref)
            return freqs, Pxy_max

        if mode == 'cq':
            return self._cq(analysisFunc, segmentLengthMultiplier)
        else:
            return analysisFunc(segmentLengthMultiplier, self.getSegmentLength() * segmentLengthMultiplier)

    def spectrogram(self, segmentLengthMultiplier=1, window='hann'):
        """
        analyses the source to generate a spectrogram
        :param segmentLengthMultiplier: allow for increased resolution.
        :return:
            t : ndarray
            Array of time slices.
            f : ndarray
            Array of sample frequencies.
            Pxx : ndarray
            linear spectrum values.
        """
        t, f, Sxx = signal.spectrogram(self.samples,
                                       self.fs,
                                       window=window,
                                       nperseg=self.getSegmentLength() * segmentLengthMultiplier,
                                       detrend=False,
                                       scaling='spectrum')
        Sxx = np.sqrt(Sxx)
        Sxx = amplitude_to_db(Sxx)
        return t, f, Sxx

    def filter(self, a, b):
        """
        Applies a digital filter via filtfilt.
        :param a: the a coeffs.
        :param b: the b coeffs.
        :return: the filtered signal.
        """
        return Signal(signal.filtfilt(b, a, self.samples), fs=self.fs)

    def resample(self, new_fs):
        '''
        Resamples to the new fs (if required).
        :param new_fs: the new fs.
        :return: the signal
        '''
        if new_fs != self.fs:
            start = time.time()
            resampled = Signal(self.name,
                               resampy.resample(self.samples, self.fs, new_fs, filter=self.load_resampy_filter), new_fs)
            end = time.time()
            logger.info(f"Resampled {self.name} from {self.fs} to {new_fs} in {round(end-start, 3)}s")
            return resampled
        else:
            return self

    def load_resampy_filter(self):
        '''
        A replacement for resampy.load_filter that is compatible with pyinstaller.
        :return: same values as resampy.load_filter
        '''
        import sys
        import os
        filter_name = 'kaiser_fast'
        if getattr(sys, 'frozen', False):
            data = np.load(os.path.join(sys._MEIPASS, '_resampy_filters', os.path.extsep.join([filter_name, 'npz'])))
        else:
            import pkg_resources
            fname = os.path.join('data', os.path.extsep.join([filter_name, 'npz']))
            data = np.load(pkg_resources.resource_filename(__name__, fname))

        return data['half_window'], data['precision'], data['rolloff']

    def getXY(self, idx=0):
        '''
        :return: renders itself as peak/avg spectrum xydata.
        '''
        if self.__avg is not None:
            self.__cached[0].colour = AVG_COLOURS[idx]
            self.__cached[1].colour = PEAK_COLOURS[idx]
            return self.__cached
        else:
            logger.error(f"getXY called on {self.name} before calculate, must be error!")
            return []

    def calculate(self, multiplier, avg_window, peak_window):
        '''
        caches the peak and avg spectrum.
        :param multiplier: the resolution.
        :param avg_window: the avg window.
        :param peak_window: the peak window.
        '''
        self.__avg = self.spectrum(segmentLengthMultiplier=multiplier, window=avg_window)
        self.__peak = self.peakSpectrum(segmentLengthMultiplier=multiplier, window=peak_window)
        self.__cached = [XYData(self.name, 'avg', self.__avg[0], self.__avg[1]),
                         XYData(self.name, 'peak', self.__peak[0], self.__peak[1])]


def amplitude_to_db(s, ref=1.0):
    '''
    Convert an amplitude spectrogram to dB-scaled spectrogram. Implementation taken from librosa to avoid adding a
    dependency on librosa for a single function.
    :param s: the amplitude spectrogram.
    :return: s_db : np.ndarray ``s`` measured in dB
    '''
    magnitude = np.abs(np.asarray(s))
    power = np.square(magnitude, out=magnitude)
    ref_power = np.abs(ref ** 2)
    amin = 1e-10
    top_db = 80.0
    log_spec = 10.0 * np.log10(np.maximum(amin, power))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_power))
    log_spec = np.maximum(log_spec, log_spec.max() - top_db)
    return log_spec


class SignalTableModel(QAbstractTableModel):
    '''
    A Qt table model to feed the signal view.
    '''

    def __init__(self, model, parent=None):
        super().__init__(parent=parent)
        self.__headers = ['Name', 'Fs', 'Duration', 'Start', 'End']
        self.__signalModel = model
        self.__signalModel.table = self

    def rowCount(self, parent: QModelIndex = ...):
        return len(self.__signalModel)

    def columnCount(self, parent: QModelIndex = ...):
        return len(self.__headers)

    def flags(self, idx):
        flags = super().flags(idx)
        if idx.column() == 0:
            flags |= Qt.ItemIsEditable
        return flags

    def setData(self, idx, value, role=None):
        if idx.column() == 0:
            self.__signalModel[idx.row()].name = value
            self.dataChanged.emit(idx, idx, [])
            return True
        return super().setData(idx, value, role=role)

    def data(self, index: QModelIndex, role: int = ...) -> typing.Any:
        if not index.isValid():
            return QVariant()
        elif role != Qt.DisplayRole:
            return QVariant()
        else:
            signal_at_row = self.__signalModel[index.row()]
            if index.column() == 0:
                return QVariant(signal_at_row.name)
            elif index.column() == 1:
                return QVariant(signal_at_row.fs)
            elif index.column() == 2:
                return QVariant(signal_at_row.duration_hhmmss)
            elif index.column() == 3:
                return QVariant(signal_at_row.start_hhmmss)
            elif index.column() == 4:
                return QVariant(signal_at_row.end_hhmmss)
            else:
                return QVariant()

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = ...) -> typing.Any:
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return QVariant(self.__headers[section])
        return QVariant()

    def resizeColumns(self, view):
        for x in range(0, len(self.__headers)):
            view.resizeColumnToContents(x)


def _select_file(owner, file_type):
    '''
    Presents a file picker for selecting a file that contains a signal.
    '''
    dialog = QFileDialog(parent=owner)
    dialog.setFileMode(QFileDialog.ExistingFile)
    dialog.setNameFilter(f"*.{file_type}")
    dialog.setWindowTitle(f"Select Signal File")
    if dialog.exec():
        selected = dialog.selectedFiles()
        if len(selected) > 0:
            return selected[0]
    return None


class WavLoader:
    '''
    Loads signals from wav files.
    '''

    def __init__(self, dialog, preferences):
        self.__preferences = preferences
        self.__dialog = dialog
        self.__signal = None
        self.__peak = None
        self.__avg = None
        self.__duration = 0

    def select_wav_file(self):
        file = _select_file(self.__dialog, 'wav')
        if file is not None:
            self.__dialog.wavFile.setText(file)
            self.load_signal()

    def clear_signal(self):
        self.__signal = None
        self.__peak = None
        self.__avg = None
        self.__duration = 0
        self.__dialog.wavStartTime.setEnabled(False)
        self.__dialog.wavEndTime.setEnabled(False)
        self.__dialog.buttonBox.button(QDialogButtonBox.Ok).setEnabled(False)

    def load_signal(self):
        '''
        Loads metadata about the signal from the file and propagates it to the form fields.
        :param file: the file.
        '''
        if self.__signal is not None:
            self.clear_signal()
        import soundfile as sf
        info = sf.info(self.__dialog.wavFile.text())
        self.__dialog.wavFs.setText(f"{info.samplerate} Hz")
        self.__dialog.wavChannelSelector.clear()
        for i in range(0, info.channels):
            self.__dialog.wavChannelSelector.addItem(f"{i+1}")
        self.__dialog.wavChannelSelector.setEnabled(info.channels > 1)
        self.__dialog.wavStartTime.setTime(QtCore.QTime(0, 0, 0))
        self.__dialog.wavStartTime.setEnabled(True)
        self.__duration = math.floor(info.duration * 1000)
        self.__dialog.wavEndTime.setTime(QtCore.QTime(0, 0, 0).addMSecs(self.__duration))
        self.__dialog.wavEndTime.setEnabled(True)
        self.__dialog.wavSignalName.setEnabled(True)

    def prepare_signal(self):
        '''
        Reads the actual file and calculates the relevant peak/avg spectrum.
        '''
        start = end = None
        start_millis = self.__dialog.wavStartTime.time().msecsSinceStartOfDay()
        if start_millis > 0:
            start = start_millis
        end_millis = self.__dialog.wavEndTime.time().msecsSinceStartOfDay()
        if end_millis < self.__duration:
            end = end_millis
        # defer to avoid circular imports
        from model.preferences import ANALYSIS_TARGET_FS, ANALYSIS_RESOLUTION, ANALYSIS_PEAK_WINDOW, \
            ANALYSIS_AVG_WINDOW
        self.__signal = readWav(self.__dialog.wavSignalName.text(), self.__dialog.wavFile.text(),
                                channel=int(self.__dialog.wavChannelSelector.currentText()), start=start, end=end,
                                target_fs=self.__preferences.get(ANALYSIS_TARGET_FS))
        multiplier = int(1 / float(self.__preferences.get(ANALYSIS_RESOLUTION)))
        peak_window = self.__get_window(ANALYSIS_PEAK_WINDOW)
        avg_window = self.__get_window(ANALYSIS_AVG_WINDOW)
        logger.debug(f"Analysing {self.__dialog.wavSignalName.text()} at {multiplier}x resolution "
                     f"using {peak_window} peak window and {avg_window} avg window")
        self.__signal.calculate(multiplier, avg_window, peak_window)
        self.__dialog.buttonBox.button(QDialogButtonBox.Ok).setEnabled(True)

    def __get_window(self, key):
        from model.preferences import ANALYSIS_WINDOW_DEFAULT
        window = self.__preferences.get(key)
        if window is None or window == ANALYSIS_WINDOW_DEFAULT:
            window = None
        else:
            if window == 'tukey':
                window = (window, 0.25)
        return window

    def get_magnitude_data(self):
        if self.__signal is not None:
            return self.__signal.getXY()
        else:
            return []

    def can_save(self):
        '''
        :return: true if we can save a new signal.
        '''
        return self.__signal is not None

    def enable_ok(self):
        enabled = len(self.__dialog.wavSignalName.text()) > 0 and self.can_save()
        self.__dialog.buttonBox.button(QDialogButtonBox.Ok).setEnabled(enabled)

    def get_signal(self):
        '''
        Converts the loaded signal into a SignalData.
        :return: the signal data.
        '''
        return SignalData(self.__signal.name, self.__signal.fs, self.__signal.getXY(), CompleteFilter(),
                          duration_hhmmss=self.__signal.duration_hhmmss, start_hhmmss=self.__signal.start_hhmmss,
                          end_hhmmss=self.__signal.end_hhmmss)


class FrdLoader:
    '''
    Loads signals from frd files.
    '''

    def __init__(self, dialog):
        self.__dialog = dialog
        self.__peak = None
        self.__avg = None

    def _read_from_file(self):
        file = _select_file(self.__dialog, 'frd')
        if file is not None:
            comment_char = None
            with open(file) as f:
                c = f.read(1)
                if not c.isalnum():
                    comment_char = c
            f, m = np.genfromtxt(file, comments=comment_char, unpack=True)
            return file, f, m
        return None, None, None

    def select_peak_file(self):
        '''
        Asks the user to pick a file containing the peak series magnitude response.
        '''
        name, f, m = self._read_from_file()
        if name is not None:
            signal_name = Path(name).resolve().stem
            if signal_name.endswith('_filter_peak'):
                signal_name = signal_name[:-12]
            elif signal_name.endswith('_peak'):
                signal_name = signal_name[:-5]
            self.__peak = XYData(signal_name, 'peak', f, m, colour=PEAK_COLOURS[0])
            self.__dialog.frdSignalName.setText(signal_name)
            self.__dialog.frdPeakFile.setText(name)
            self.__enable_fields()

    def select_avg_file(self):
        '''
        Asks the user to pick a file containing the avg series magnitude response.
        '''
        name, f, m = self._read_from_file()
        if name is not None:
            signal_name = Path(name).resolve().stem
            if signal_name.endswith('_filter_avg'):
                signal_name = signal_name[:-11]
            elif signal_name.endswith('_avg'):
                signal_name = signal_name[:-4]
            self.__avg = XYData(signal_name, 'avg', f, m, colour=AVG_COLOURS[0])
            self.__dialog.frdSignalName.setText(signal_name)
            self.__dialog.frdAvgFile.setText(name)
            self.__enable_fields()

    def __enable_fields(self):
        '''
        Enables the fs field if we have both measurements.
        '''
        self.__dialog.frdSignalName.setEnabled(self.__peak is not None or self.__avg is not None)
        if self.__peak is not None and self.__avg is not None:
            # TODO read the header?
            self.__dialog.frdFs.setValue(int(np.max(self.__peak.x) * 2))
            self.__dialog.frdFs.setEnabled(True)
        self.enable_ok()

    def enable_ok(self):
        enabled = len(self.__dialog.frdSignalName.text()) > 0 and self.can_save()
        self.__dialog.buttonBox.button(QDialogButtonBox.Ok).setEnabled(enabled)

    def clear_signal(self):
        self.__peak = None
        self.__avg = None
        self.__enable_fields()

    def get_magnitude_data(self):
        data = []
        if self.__avg is not None:
            data.append(self.__avg)
        if self.__peak is not None:
            data.append(self.__peak)
        return data

    def can_save(self):
        return self.__avg is not None and self.__peak is not None

    def get_signal(self):
        name = self.__dialog.frdSignalName.text()
        self.__avg.name = f"{self.__dialog.frdSignalName.text()}_avg"
        self.__peak.name = f"{self.__dialog.frdSignalName.text()}_peak"
        # TODO set fs on filter
        return SignalData(name, self.__dialog.frdFs.value(), self.get_magnitude_data(), CompleteFilter())


class SignalDialog(QDialog, Ui_addSignalDialog):
    '''
    Alows user to extract a signal from a wav or frd.
    '''

    def __init__(self, preferences, signalModel, parent=None):
        super(SignalDialog, self).__init__(parent=parent)
        self.setupUi(self)
        self.__loaders = [WavLoader(self, preferences), FrdLoader(self)]
        self.__loader_idx = self.signalTypeTabs.currentIndex()
        self.__signalModel = signalModel
        self.__magnitudeModel = MagnitudeModel('preview', self.previewChart, self, 'Signal')
        for s in self.__signalModel:
            self.filterSelect.addItem(s.name)
        if len(self.__signalModel) == 0:
            self.filterSelect.setEnabled(False)
            self.linkedSignal.setEnabled(False)
        self.clearSignal(draw=False)

    def changeLoader(self, idx):
        self.__loader_idx = idx
        self.__loaders[self.__loader_idx].enable_ok()
        self.__magnitudeModel.redraw()

    def selectFile(self):
        '''
        Presents a file picker for selecting a wav file that contains a signal.
        '''
        self.__loaders[self.__loader_idx].select_wav_file()

    def selectPeakFile(self):
        '''
        Presents a file picker for selecting a frd file that contains the peak signal.
        '''
        self.__loaders[self.__loader_idx].select_peak_file()
        self.__magnitudeModel.redraw()

    def selectAvgFile(self):
        '''
        Presents a file picker for selecting a frd file that contains the avg signal.
        '''
        self.__loaders[self.__loader_idx].select_avg_file()
        self.__magnitudeModel.redraw()

    def clearSignal(self, draw=True):
        ''' clears the current signal '''
        self.__loaders[self.__loader_idx].clear_signal()
        if draw:
            self.__magnitudeModel.redraw()

    def enablePreview(self, text):
        '''
        Ensures we can only preview once we have a name.
        '''
        self.previewButton.setEnabled(len(text) > 0)

    def prepareSignal(self):
        '''
        Loads the signal and displays the chart.
        '''
        from app import wait_cursor
        with wait_cursor('Preparing Signal'):
            self.__loaders[self.__loader_idx].prepare_signal()
            self.__magnitudeModel.redraw()

    def enableOk(self):
        '''
        Enables the ok button if we can save.
        '''
        self.__loaders[self.__loader_idx].enable_ok()

    def getMagnitudeData(self, reference=None):
        '''
        :param reference: ignored as we don't expose a normalisation control in this chart.
        :return: the peak and avg spectrum for the currently loaded signal (if any).
        '''
        return self.__loaders[self.__loader_idx].get_magnitude_data()

    def masterFilterChanged(self, idx):
        '''
        enables the linked signal checkbox if we have selected a filter.
        :param idx: the selected index.
        '''
        self.linkedSignal.setEnabled(idx > 0)

    def accept(self):
        '''
        Adds the signal to the model and exits if we have a signal (which we should because the button is disabled
        until we do).
        '''
        loader = self.__loaders[self.__loader_idx]
        if loader.can_save():
            signal = loader.get_signal()
            selected_filter_idx = self.filterSelect.currentIndex()
            if selected_filter_idx > 0:
                master = self.__signalModel[selected_filter_idx - 1]
                if self.linkedSignal.isChecked():
                    master.enslave(signal)
                else:
                    signal.filter = master.filter.resample(signal.fs)
            self.__signalModel.add(signal)
            QDialog.accept(self)


def readWav(name, input_file, channel=1, start=None, end=None, target_fs=1000) -> Signal:
    """ reads a wav file into a Signal.
    :param input_file: a path to the input signal file
    :param channel: the channel to read.
    :param start: the time to start reading from in ms
    :param end: the time to end reading from in ms.
    :param target_fs: the fs of the Signal to return (resampling if necessary)
    :returns: Signal.
    """
    import soundfile as sf
    if start is not None or end is not None:
        info = sf.info(input_file)
        startFrame = 0 if start is None else int(start * (info.samplerate / 1000))
        endFrame = None if end is None else int(start * (info.samplerate / 1000))
        ys, frameRate = sf.read(input_file, start=startFrame, stop=endFrame)
    else:
        ys, frameRate = sf.read(input_file)
    signal = Signal(name, ys[::channel], frameRate)
    if target_fs is None or target_fs == 0:
        target_fs = signal.fs
    return signal.resample(target_fs)
