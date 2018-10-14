import datetime
import logging
import math
import re
import time
import typing
from collections import Sequence
from pathlib import Path

import numpy as np
import qtawesome as qta
import resampy
from qtpy import QtCore
from qtpy.QtCore import QAbstractTableModel, QModelIndex, QVariant, Qt
from qtpy.QtWidgets import QDialog, QFileDialog, QDialogButtonBox
from scipy import signal

from model.codec import signaldata_to_json
from model.iir import XYData, CompleteFilter
from model.magnitude import MagnitudeModel
from model.preferences import get_avg_colour, get_peak_colour, SHOW_PEAK, \
    SHOW_AVERAGE, SHOW_FILTERED_ONLY, SHOW_UNFILTERED_ONLY, DISPLAY_SHOW_SIGNALS, \
    DISPLAY_SHOW_FILTERED_SIGNALS, ANALYSIS_TARGET_FS
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
        self.slaves = []
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

    @property
    def active_filter(self):
        return self.master.filter if self.master is not None else self.filter

    @filter.setter
    def filter(self, filt):
        self.__filter = filt
        self.__filter.listener = self
        self.on_filter_change(filt)

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
        logger.debug(f"Enslaving {signal.name} to {self.name}")
        self.slaves.append(signal)
        signal.master = self
        signal.on_filter_change(self.__filter)

    def free_all(self):
        '''
        Frees all slaves.
        '''
        for s in self.slaves:
            logger.debug(f"Freeing {s} from {self}")
            s.master = None
            s.filter = CompleteFilter()
        self.slaves = []

    def free(self):
        '''
        if this is a slave, frees itself from the master.
        '''
        if self.master is not None:
            logger.debug(f"Freeing {self.name} from {self.master.name}")
            self.master.slaves.remove(self)
            self.master = None
            self.filter = CompleteFilter()

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
        logger.debug(f"Applying filter change to {self.name}")
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
        for s in self.slaves:
            logger.debug(f"Propagating filter change to {s.name}")
            s.on_filter_change(filt)


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

    def __getitem__(self, i):
        return self.__signals[i]

    def __len__(self):
        return len(self.__signals)

    def to_json(self):
        '''
        :return: a json compatible format of the data in the model.
        '''
        return [signaldata_to_json(x) for x in self.__signals]

    def free_all(self):
        '''
        Frees all signals from their masters.
        '''
        if self.__table is not None:
            self.__table.beginResetModel()
        for signal in self:
            signal.free_all()
        if self.__table is not None:
            self.__table.endResetModel()

    def enslave(self, master_name, slave_names):
        '''
        Enslaves the named slaves to the named master.
        :param master_name: the master.
        :param slave_names: the slaves.
        '''
        logger.info(f"Enslaving {slave_names} to {master_name}")
        if self.__table is not None:
            self.__table.beginResetModel()
        master = self.find_by_name(master_name)
        if master is not None:
            for slave_name in slave_names:
                slave = self.find_by_name(slave_name)
                if slave is not None:
                    master.enslave(slave)
        if self.__table is not None:
            self.__table.endResetModel()

    def add(self, signal):
        '''
        Add the supplied signals ot the model.
        :param signals: the signal.
        '''
        before_size = len(self.__signals)
        if self.__table is not None:
            self.__table.beginInsertRows(QModelIndex(), before_size, before_size)
        signal.reindex(before_size)
        self.__signals.append(signal)
        self.post_update()
        if self.__table is not None:
            self.__table.endInsertRows()

    def add_all(self, signals):
        '''
        Add the supplied signals ot the model.
        :param signals: the signal.
        '''
        before_size = len(self.__signals)
        if self.__table is not None:
            self.__table.beginInsertRows(QModelIndex(), before_size, before_size + (len(signals) - 1))
        for s in signals:
            s.reindex(len(self.__signals))
            self.__signals.append(s)
        self.post_update()
        if self.__table is not None:
            self.__table.endInsertRows()

    def post_update(self):
        from app import flatten
        show_signals = self.__preferences.get(DISPLAY_SHOW_SIGNALS)
        show_filtered_signals = self.__preferences.get(DISPLAY_SHOW_FILTERED_SIGNALS)
        pattern = self.__get_visible_signal_name_filter(show_filtered_signals, show_signals)
        visible_signal_names = [x.name for x in flatten([y for x in self.__signals for y in x.get_all_xy()])]
        if pattern is not None:
            visible_signal_names = [x for x in visible_signal_names if pattern.match(x) is not None]
        self.__on_update(visible_signal_names)

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
                pattern = re.compile(".*_(avg|peak)$")
        return pattern

    def remove(self, signal):
        '''
        Remove the specified signal from the model.
        :param signal: the signal to remove.
        '''
        idx = self.__signals.index(signal)
        if self.__table is not None:
            self.__table.beginRemoveRows(QModelIndex(), idx, idx)
        del self.__signals[idx]
        for idx, s in enumerate(self.__signals):
            s.reindex(idx)
        self.__ensure_master_slave_integrity()
        self.post_update()
        if self.__table is not None:
            self.__table.endRemoveRows()

    def delete(self, indices):
        '''
        Delete the signals at the given indices.
        :param indices: the indices to remove.
        '''
        self.replace([s for idx, s in enumerate(self.__signals) if idx not in indices])

    def get_all_magnitude_data(self):
        '''
        :return: the raw xy data.
        '''
        from app import flatten
        results = list(flatten([s.get_all_xy() for s in self.__signals]))
        return results

    def getMagnitudeData(self, reference=None):
        '''
        :param reference: the curve against which to normalise.
        :return: the peak and avg spectrum for the signals (if any) + the filter signals.
        '''
        results = self.get_all_magnitude_data()
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
        if self.__table is not None:
            self.__table.beginResetModel()
        self.__signals = signals
        for idx, s in enumerate(self.__signals):
            s.reindex(idx)
        self.__ensure_master_slave_integrity()
        self.post_update()
        if self.__table is not None:
            self.__table.endResetModel()

    def __ensure_master_slave_integrity(self):
        '''
        Verifies that all master/slaves mentioned by signals actually exist in the model. Used when signals are deleted.
        '''
        for s in self.__signals:
            slave_count_before = len(s.slaves)
            if slave_count_before > 0:
                s.slaves = [slave for slave in s.slaves if slave in self.__signals]
                delta = slave_count_before - len(s.slaves)
                if delta > 0:
                    logger.info(f"Removed {delta} missing slaves from {s.name}")
            if s.master is not None:
                master = next((m for m in self.__signals if m.name == s.master.name), None)
                if master is None:
                    logger.info(f"Removing missing master {s.master.name} from {s.name}")
                    s.free()

    def find_by_name(self, name):
        '''
        :param name: the signal name.
        :return: the signal with that name (or None).
        '''
        return next((s for s in self if s.name == name), None)


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

    def cut(self, start, end):
        ''' slices a section out of the signal '''
        if start < self.durationSeconds and end <= self.durationSeconds:
            return Signal(self.name, self.samples[(start * self.fs): (end * self.fs)+1], fs=self.fs)
        else:
            return self

    def getSegmentLength(self, resolution_shift=0):
        """
        Calculates a segment length such that the frequency resolution of the resulting analysis is in the region of 
        ~1Hz subject to a lower limit of the number of samples in the signal.
        For example, if we have a 10s signal with an fs is 500 then we convert fs-1 to the number of bits required to 
        hold this number in binary (i.e. 111110011 so 9 bits) and then do 1 << 9 which gives us 100000000 aka 512. Thus
        we have ~1Hz resolution.
        :param resolution_shift: shifts the resolution up or down by the specified number of bits.
        :return: the segment length.
        """
        return min(1 << ((self.fs - 1).bit_length() - int(resolution_shift)), self.samples.shape[-1])

    def raw(self):
        """
        :return: the raw sample data
        """
        return self.samples

    def spectrum(self, ref=SPECLAB_REFERENCE, resolution_shift=0, window=None, **kwargs):
        """
        analyses the source to generate the linear spectrum.
        :param ref: the reference value for dB purposes.
        :param resolution_shift: allows resolution to go down (if positive) or up (if negative).
        :param mode: cq or none.
        :return:
            f : ndarray
            Array of sample frequencies.
            Pxx : ndarray
            linear spectrum.
        """
        nperseg = self.getSegmentLength(resolution_shift=resolution_shift)
        f, Pxx_spec = signal.welch(self.samples, self.fs, nperseg=nperseg, scaling='spectrum', detrend=False,
                                   window=window if window else 'hann', **kwargs)
        # a 3dB adjustment is required to account for the change in nperseg
        Pxx_spec = amplitude_to_db(np.sqrt(Pxx_spec), ref * SPECLAB_REFERENCE)
        return f, Pxx_spec

    def peakSpectrum(self, ref=SPECLAB_REFERENCE, resolution_shift=0, window=None):
        """
        analyses the source to generate the max values per bin per segment
        :param resolution_shift: allows resolution to go down (if positive) or up (if negative).
        :param mode: cq or none.
        :param window: window type.
        :return:
            f : ndarray
            Array of sample frequencies.
            Pxx : ndarray
            linear spectrum max values.
        """
        nperseg = self.getSegmentLength(resolution_shift=resolution_shift)
        freqs, _, Pxy = signal.spectrogram(self.samples,
                                           self.fs,
                                           window=window if window else ('tukey', 0.25),
                                           nperseg=int(nperseg),
                                           noverlap=int(nperseg // 2),
                                           detrend=False,
                                           scaling='spectrum')
        Pxy_max = np.sqrt(Pxy.max(axis=-1).real)
        Pxy_max = amplitude_to_db(Pxy_max, ref=ref * SPECLAB_REFERENCE)
        return freqs, Pxy_max

    def spectrogram(self, ref=SPECLAB_REFERENCE, resolution_shift=0, window=None):
        """
        analyses the source to generate a spectrogram
        :param resolution_shift: allows resolution to go down (if positive) or up (if negative).
        :return:
            f : ndarray
            Array of time slices.
            t : ndarray
            Array of sample frequencies.
            Pxx : ndarray
            linear spectrum values.
        """
        nperseg = self.getSegmentLength(resolution_shift=resolution_shift)
        f, t, Sxx = signal.spectrogram(self.samples,
                                       self.fs,
                                       window=window if window else ('tukey', 0.25),
                                       nperseg=nperseg,
                                       noverlap=int(nperseg // 2),
                                       detrend=False,
                                       scaling='spectrum')
        Sxx = amplitude_to_db(np.sqrt(Sxx), ref=ref * SPECLAB_REFERENCE)
        return f, t, Sxx

    def filter(self, a, b):
        """
        Applies a digital filter via filtfilt.
        :param a: the a coeffs.
        :param b: the b coeffs.
        :return: the filtered signal.
        """
        return Signal(self.name, signal.filtfilt(b, a, self.samples), fs=self.fs)

    def resample(self, new_fs):
        '''
        Resamples to the new fs (if required).
        :param new_fs: the new fs.
        :return: the signal
        '''
        if new_fs != self.fs:
            start = time.time()
            resampled = Signal(self.name,
                               resampy.resample(self.samples, self.fs, new_fs, filter=self.load_resampy_filter()),
                               new_fs)
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
        if getattr(sys, 'frozen', False):
            def __load_frozen():
                import os
                data = np.load(
                    os.path.join(sys._MEIPASS, '_resampy_filters', os.path.extsep.join(['kaiser_fast', 'npz'])))
                return data['half_window'], data['precision'], data['rolloff']

            return __load_frozen
        else:
            return 'kaiser_fast'

    def getXY(self, idx=0):
        '''
        :return: renders itself as peak/avg spectrum xydata.
        '''
        if self.__avg is not None:
            self.__cached[0].colour = get_avg_colour(idx)
            self.__cached[1].colour = get_peak_colour(idx)
            return self.__cached
        else:
            logger.error(f"getXY called on {self.name} before calculate, must be error!")
            return []

    def calculate(self, resolution_shift, avg_window, peak_window):
        '''
        caches the peak and avg spectrum.
        :param resolution_shift: the resolution shift.
        :param avg_window: the avg window.
        :param peak_window: the peak window.
        '''
        self.__avg = self.spectrum(resolution_shift=resolution_shift, window=avg_window)
        self.__peak = self.peakSpectrum(resolution_shift=resolution_shift, window=peak_window)
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
        self.__headers = ['Name', 'Linked', 'Fs', 'Duration']
        self.__signal_model = model
        self.__signal_model.table = self

    def rowCount(self, parent: QModelIndex = ...):
        return len(self.__signal_model)

    def columnCount(self, parent: QModelIndex = ...):
        return len(self.__headers)

    def flags(self, idx):
        flags = super().flags(idx)
        if idx.column() == 0:
            flags |= Qt.ItemIsEditable
        return flags

    def setData(self, idx, value, role=None):
        if idx.column() == 0:
            self.__signal_model[idx.row()].name = value
            self.dataChanged.emit(idx, idx, [])
            return True
        return super().setData(idx, value, role=role)

    def data(self, index: QModelIndex, role: int = ...) -> typing.Any:
        if not index.isValid():
            return QVariant()
        elif role != Qt.DisplayRole:
            return QVariant()
        else:
            signal_at_row = self.__signal_model[index.row()]
            if index.column() == 0:
                return QVariant(signal_at_row.name)
            if index.column() == 1:
                if signal_at_row.master is not None:
                    return QVariant(f"S - {signal_at_row.master.name}")
                elif len(signal_at_row.slaves) > 0:
                    return QVariant(f"M {len(signal_at_row.slaves)}")
                else:
                    return QVariant('')
            elif index.column() == 2:
                return QVariant(signal_at_row.fs)
            elif index.column() == 3:
                return QVariant(signal_at_row.duration_hhmmss)
            else:
                return QVariant()

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = ...) -> typing.Any:
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return QVariant(self.__headers[section])
        return QVariant()


def select_file(owner, file_types):
    '''
    Presents a file picker for selecting a file that contains a signal.
    '''
    dialog = QFileDialog(parent=owner)
    dialog.setFileMode(QFileDialog.ExistingFile)
    filt = ' '.join([f"*.{f}" for f in file_types])
    dialog.setNameFilter(f"Audio ({filt})")
    dialog.setWindowTitle(f"Select Signal File")
    if dialog.exec():
        selected = dialog.selectedFiles()
        if len(selected) > 0:
            return selected[0]
    return None


class AutoWavLoader:
    '''
    Loads signals from wav files without user interaction
    '''

    def __init__(self, preferences):
        self.__signal = None
        self.__preferences = preferences
        self.__start = None
        self.__end = None
        self.info = None

    def reset(self):
        '''
        Clears internal state.
        '''
        self.__signal = None
        self.info = None

    def auto_load(self, file, name_provider, decimate):
        '''
        Loads the file and automatically creates a signal for each channel found.
        :param file: the file to load.
        :param name_provider: a callable that yields a signal name for the given channel.
        :param decimate: if true, decimate
        :return: the signals.
        '''
        self.load(file)
        return [self.__auto_load(x + 1, name_provider(x, self.info.channels), decimate)
                for x in range(0, self.info.channels)]

    def load(self, file):
        '''
        Gets info about the file.
        :param file: the file.
        :return: if auto is True, a signal for each channel.
        '''
        self.reset()
        import soundfile as sf
        self.info = sf.info(file)

    def __auto_load(self, channel, name, decimate):
        self.prepare(name=name, channel=channel, decimate=decimate)
        return self.get_signal()

    def set_range(self, start=None, end=None):
        '''
        Sets the range to load from the file.
        :param start: the start position, if any.
        :param end: the end position, if any.
        '''
        self.__start = start
        self.__end = end

    def prepare(self, name=None, channel_count=1, channel=1, decimate=True):
        '''
        Loads and analyses the wav with the specified parameters.
        :param name: the signal name, if none use the file name + channel.
        :param channel: the channel
        :param channel_count: the channel count, only used for creating a default name.
        :param decimate: if true, decimate the wav.
        '''
        # defer to avoid circular imports
        from model.preferences import ANALYSIS_TARGET_FS, ANALYSIS_RESOLUTION, ANALYSIS_PEAK_WINDOW, \
            ANALYSIS_AVG_WINDOW
        if name is None:
            name = Path(self.info.name).resolve().stem
            if channel_count > 1:
                name += f"_c{channel}"
        target_fs = self.__preferences.get(ANALYSIS_TARGET_FS) if decimate is True else self.info.samplerate
        self.__signal = readWav(name, self.info.name, channel=channel, start=self.__start, end=self.__end,
                                target_fs=target_fs)
        resolution_shift = math.log(self.__preferences.get(ANALYSIS_RESOLUTION), 2)
        peak_wnd = self.__get_window(ANALYSIS_PEAK_WINDOW)
        avg_wnd = self.__get_window(ANALYSIS_AVG_WINDOW)
        logger.debug(
            f"Analysing {self.info.name} at {resolution_shift}x resolution using {peak_wnd}/{avg_wnd} peak/avg windows")
        self.__signal.calculate(resolution_shift, avg_wnd, peak_wnd)

    def __get_window(self, key):
        from model.preferences import ANALYSIS_WINDOW_DEFAULT
        window = self.__preferences.get(key)
        if window is None or window == ANALYSIS_WINDOW_DEFAULT:
            window = None
        else:
            if window == 'tukey':
                window = (window, 0.25)
        return window

    def get_signal(self):
        '''
        Converts the loaded signal into a SignalData.
        :return: the signal data.
        '''
        return SignalData(self.__signal.name, self.__signal.fs, self.__signal.getXY(), CompleteFilter(),
                          duration_hhmmss=self.__signal.duration_hhmmss, start_hhmmss=self.__signal.start_hhmmss,
                          end_hhmmss=self.__signal.end_hhmmss)

    def get_magnitude_data(self):
        '''
        :return: magnitude data for this signal, if we have one.
        '''
        return self.__signal.getXY() if self.has_signal() else []

    def has_signal(self):
        '''
        :return: True if we have a signal.
        '''
        return self.__signal is not None


class DialogWavLoaderBridge:
    '''
    Loads signals from wav files.
    '''

    def __init__(self, dialog, preferences):
        self.__preferences = preferences
        self.__dialog = dialog
        self.__auto_loader = AutoWavLoader(preferences)
        self.__duration = 0

    def select_wav_file(self):
        file = select_file(self.__dialog, ['wav', 'flac'])
        if file is not None:
            self.clear_signal()
            self.__dialog.wavFile.setText(file)
            self.__auto_loader.load(file)
            self.__load_info()

    def clear_signal(self):
        self.__auto_loader.reset()
        self.__duration = 0
        self.__dialog.wavStartTime.setEnabled(False)
        self.__dialog.wavEndTime.setEnabled(False)
        self.__dialog.buttonBox.button(QDialogButtonBox.Ok).setEnabled(False)

    def __load_info(self):
        '''
        Loads metadata about the signal from the file and propagates it to the form fields.
        '''
        info = self.__auto_loader.info
        self.__dialog.wavFs.setText(f"{info.samplerate} Hz")
        self.__dialog.decimate.setEnabled(info.samplerate != self.__preferences.get(ANALYSIS_TARGET_FS))
        self.__dialog.wavChannelSelector.clear()
        for i in range(0, info.channels):
            self.__dialog.wavChannelSelector.addItem(f"{i+1}")
        self.__dialog.wavChannelSelector.setEnabled(info.channels > 1)
        self.__dialog.loadAllChannels.setEnabled(info.channels > 1)
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
        if end_millis < self.__duration or start is not None:
            end = end_millis
        channel = int(self.__dialog.wavChannelSelector.currentText())
        self.__auto_loader.set_range(start=start, end=end)
        self.__auto_loader.prepare(name=self.__dialog.wavSignalName.text(), channel=channel,
                                   decimate=self.__dialog.decimate.isChecked())
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
        return self.__auto_loader.get_magnitude_data()

    def can_save(self):
        '''
        :return: true if we can save a new signal.
        '''
        return self.__auto_loader.has_signal()

    def enable_ok(self):
        enabled = len(self.__dialog.wavSignalName.text()) > 0 and self.can_save()
        self.__dialog.buttonBox.button(QDialogButtonBox.Ok).setEnabled(enabled)

    def get_signals(self):
        '''
        Converts the loaded signal into a SignalData.
        :return: the signal data.
        '''
        if self.__dialog.loadAllChannels.isChecked():
            from model.extract import get_channel_name
            name_provider = lambda channel, channel_count: get_channel_name(self.__dialog.wavSignalName.text(), channel,
                                                                            channel_count)
            return self.__auto_loader.auto_load(self.__dialog.wavFile.text(), name_provider,
                                                self.__dialog.decimate.isChecked())
        else:
            return [self.__auto_loader.get_signal()]


class FrdLoader:
    '''
    Loads signals from frd files.
    '''

    def __init__(self, dialog):
        self.__dialog = dialog
        self.__peak = None
        self.__avg = None

    def _read_from_file(self):
        file = select_file(self.__dialog, ['frd'])
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
            self.__peak = XYData(signal_name, 'peak', f, m, colour=get_peak_colour(0))
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
            self.__avg = XYData(signal_name, 'avg', f, m, colour=get_avg_colour(0))
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

    def get_signals(self):
        frd_name = self.__dialog.frdSignalName.text()
        self.__avg.internal_name = frd_name
        self.__peak.internal_name = frd_name
        return [SignalData(frd_name, self.__dialog.frdFs.value(), self.get_magnitude_data(), CompleteFilter())]


class SignalDialog(QDialog, Ui_addSignalDialog):
    '''
    Alows user to extract a signal from a wav or frd.
    '''

    def __init__(self, preferences, signal_model, parent=None):
        super(SignalDialog, self).__init__(parent=parent)
        self.setupUi(self)
        self.wavFilePicker.setIcon(qta.icon('fa.folder-open-o'))
        self.frdAvgFilePicker.setIcon(qta.icon('fa.folder-open-o'))
        self.frdPeakFilePicker.setIcon(qta.icon('fa.folder-open-o'))
        self.__loaders = [DialogWavLoaderBridge(self, preferences), FrdLoader(self)]
        self.__loader_idx = self.signalTypeTabs.currentIndex()
        self.__signal_model = signal_model
        self.__magnitudeModel = MagnitudeModel('preview', self.previewChart, preferences, self, 'Signal')
        if len(self.__signal_model) == 0:
            if len(self.__signal_model.default_signal.filter) > 0:
                self.filterSelect.addItem('Default')
            else:
                self.filterSelectLabel.setEnabled(False)
                self.filterSelect.setEnabled(False)
                self.linkedSignal.setEnabled(False)
        else:
            for s in self.__signal_model:
                if s.master is None:
                    self.filterSelect.addItem(s.name)
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
            from app import wait_cursor
            with wait_cursor(f"Saving signals"):
                signals = loader.get_signals()
                if len(signals) > 0:
                    selected_filter_idx = self.filterSelect.currentIndex()
                    if selected_filter_idx > 0:  # 0 because the dropdown has a None value first
                        if self.filterSelect.currentText() == 'Default':
                            self.__apply_default_filter(signals)
                        else:
                            master = self.__signal_model[selected_filter_idx - 1]
                            for s in signals:
                                if self.linkedSignal.isChecked():
                                    master.enslave(s)
                                else:
                                    s.filter = master.filter.resample(s.fs)
                    self.__signal_model.add_all(signals)
                    QDialog.accept(self)
                else:
                    logger.warning(f"No signals produced by loader")

    def __apply_default_filter(self, signals):
        '''
        Copies forward the default filter, using the 1st generated signal as the master if the user has chosen to link
        them.
        :param signals: the signals.
        '''
        if self.linkedSignal.isChecked():
            master = None
            for idx, s in enumerate(signals):
                if idx == 0:
                    s.filter = self.__signal_model.default_signal.filter.resample(s.fs)
                    master = s
                else:
                    master.enslave(s)
        else:
            for s in signals:
                s.filter = self.__signal_model.default_signal.filter.resample(s.fs)


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
        endFrame = None if end is None else int(end * (info.samplerate / 1000))
        ys, frameRate = sf.read(input_file, start=startFrame, stop=endFrame, always_2d=True)
    else:
        ys, frameRate = sf.read(input_file, always_2d=True)
    signal = Signal(name, ys[:, channel - 1], frameRate)
    if target_fs is None or target_fs == 0:
        target_fs = signal.fs
    return signal.resample(target_fs)
