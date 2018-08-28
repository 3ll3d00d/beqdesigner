import datetime
import logging
import math
import time
import typing
from collections import Sequence

import numpy as np
import resampy
from qtpy import QtCore
from qtpy.QtWidgets import QDialog, QFileDialog, QDialogButtonBox
from qtpy.QtCore import QAbstractTableModel, QModelIndex, QVariant, Qt
from scipy import signal

from model.iir import XYData
from model.magnitude import MagnitudeModel
from ui.signal import Ui_addSignalDialog

logger = logging.getLogger('signal')

""" speclab reports a peak of 0dB but, by default, we report a peak of -3dB """
SPECLAB_REFERENCE = 1 / (2 ** 0.5)

WINDOWS = ['barthann', 'bartlett', 'blackman', 'blackmanharris', 'bohman', 'boxcar', 'cosine', 'flattop', 'hamming',
           'hann', 'nuttall', 'parzen', 'triang', 'tukey']

# keep peak green and avg red
AVG_COLOURS = [
    '#ff0000',
    '#990000',
    '#ff6666',
    '#ff9999',
    '#660000',
    '#4c0000',
]

PEAK_COLOURS = [
    '#00ff00',
    '#009900',
    '#99ff99',
    '#ccffcc',
    '#006600',
    '#004c00',
]


class SignalData:
    '''
    Provides a mechanism for caching the assorted xy data surrounding a signal.
    '''

    def __init__(self, idx, signal, filter):
        self.signal = signal
        self.__filter = filter
        self.raw = signal.getXY(idx=idx)
        self.filtered = []
        self.on_filter_change(filter)
        self.reference_name = None
        self.reference = []

    def reindex(self, idx):
        self.raw = self.signal.getXY(idx=idx)
        self.on_filter_change(self.__filter)

    def on_filter_change(self, filter):
        '''
        Updates the filtered response with the new filter.
        :param filter: the filter.
        '''
        if filter is None:
            self.filtered = []
            for r in self.raw:
                r.linestyle = '-'
        else:
            filter_mag = filter.getMagnitude()
            self.filtered = [f.filter(filter_mag) for f in self.raw]
            for r in self.raw:
                r.linestyle = '--'

    def on_reference_change(self):
        pass

    def get_all_xy(self):
        '''
        :return all the xy data
        '''
        return self.raw + self.filtered


class SignalModel(Sequence):
    '''
    A model to hold onto the signals.
    '''

    def __init__(self, view, filterModel, on_update=lambda _: True):
        self.__signals = []
        self.__view = view
        self.__on_update = on_update
        self.__filterModel = filterModel
        self.__filterModel.register(self)
        self.__table = None

    @property
    def table(self):
        return self.__table

    @table.setter
    def table(self, table):
        self.__table = table
        self.__table.resizeColumns(self.__view)

    def __getitem__(self, i):
        return self.__signals[i].signal

    def __len__(self):
        return len(self.__signals)

    def add(self, signal):
        '''
        Add the supplied signals ot the model.
        :param signals: the signal.
        '''
        if self.__table is not None:
            self.__table.beginResetModel()
        self.__signals.append(SignalData(len(self.__signals), signal, self.__filterModel.getTransferFunction()))
        self.post_update()
        if self.__table is not None:
            self.__table.endResetModel()

    def post_update(self):
        from app import flatten
        self.__on_update([x.name for x in flatten([y for x in self.__signals for y in x.get_all_xy()])])
        self.__table.resizeColumns(self.__view)

    def remove(self, signal):
        '''
        Remove the specified signal from the model.
        :param signal: the signal to remove.
        '''
        if self.__table is not None:
            self.__table.beginResetModel()
        self.__signals.remove(signal)
        for idx, s in self.__signals:
            s.reindex(idx)
        self.post_update()
        if self.__table is not None:
            self.__table.endResetModel()

    def delete(self, indices):
        '''
        Delete the signals at the given indices.
        :param indices: the indices to remove.
        '''
        if self.__table is not None:
            self.__table.beginResetModel()
        self.__signals = [s for idx, s in enumerate(self.__signals) if idx not in indices]
        self.post_update()
        if self.__table is not None:
            self.__table.endResetModel()

    def onFilterChange(self):
        '''
        Updates the cached data when the filter changes.
        '''
        for s in self.__signals:
            s.on_filter_change(self.__filterModel.getTransferFunction())

    def getMagnitudeData(self, reference=None):
        '''
        :param reference: the curve against which to normalise.
        :return: the peak and avg spectrum for the signals (if any) + the filter signals.
        '''
        from app import flatten
        results = list(flatten([s.get_all_xy() for s in self.__signals]))
        if reference is not None:
            ref_data = next((x for x in results if x.name == reference), None)
            if ref_data:
                results = [x.normalise(ref_data) for x in results]
        return results


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
        self.__cached = [XYData(f"{self.name}_avg", self.__avg[0], self.__avg[1]),
                         XYData(f"{self.name}_peak", self.__peak[0], self.__peak[1])]


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


class SignalDialog(QDialog, Ui_addSignalDialog):
    '''
    Alows user to extract a signal from a wav or frd.
    '''

    def __init__(self, settings, signalModel, parent=None):
        super(SignalDialog, self).__init__(parent=parent)
        self.setupUi(self)
        self.__settings = settings
        self.__signalModel = signalModel
        self.__magnitudeModel = MagnitudeModel('preview', self.previewChart, self, 'Signal', animate_interval=200)
        self.__duration = 0
        self.__signal = None
        self.__peak = None
        self.__avg = None
        self.clearSignal(draw=False)

    def selectFile(self):
        '''
        Presents a file picker for selecting a file that contains a signal.
        '''
        dialog = QFileDialog(parent=self)
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setNameFilter(f"*.{self.fileTypePicker.currentText()}")
        dialog.setWindowTitle(f"Select Signal File")
        if dialog.exec():
            selected = dialog.selectedFiles()
            if len(selected) > 0:
                self.file.setText(selected[0])
                self.loadSignal(selected[0])

    def clearSignal(self, draw=True):
        ''' clears the current signal '''
        self.__signal = None
        self.__peak = None
        self.__avg = None
        self.__duration = 0
        self.startTime.setEnabled(False)
        self.endTime.setEnabled(False)
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(False)

    def loadSignal(self, file):
        '''
        Loads the signal from the file.
        :param file: the file.
        '''
        if self.__signal is not None:
            self.clearSignal()
        import soundfile as sf
        info = sf.info(file)
        self.fs.setText(f"{info.samplerate} Hz")
        self.channelSelector.clear()
        for i in range(0, info.channels):
            self.channelSelector.addItem(f"{i+1}")
        self.channelSelector.setEnabled(info.channels > 1)
        self.startTime.setTime(QtCore.QTime(0, 0, 0))
        self.startTime.setEnabled(True)
        self.__duration = math.floor(info.duration * 1000)
        self.endTime.setTime(QtCore.QTime(0, 0, 0).addMSecs(self.__duration))
        self.endTime.setEnabled(True)
        self.signalName.setEnabled(True)

    def enablePreview(self):
        '''
        Ensures we can only preview once we have a name.
        '''
        self.previewButton.setEnabled(len(self.signalName.text()) > 0)

    def prepareSignal(self):
        '''
        Loads the signal and displays the chart.
        '''
        from app import wait_cursor
        with wait_cursor('Preparing Signal'):
            start = end = None
            startMillis = self.startTime.time().msecsSinceStartOfDay()
            if startMillis > 0:
                start = startMillis
            endMillis = self.endTime.time().msecsSinceStartOfDay()
            if endMillis < self.__duration:
                end = endMillis
            # defer to avoid circular imports
            from model.preferences import ANALYSIS_TARGET_FS, ANALYSIS_RESOLUTION, ANALYSIS_PEAK_WINDOW, \
                ANALYSIS_AVG_WINDOW
            self.__signal = readWav(self.signalName.text(), self.file.text(),
                                    channel=int(self.channelSelector.currentText()), start=start, end=end,
                                    target_fs=self.__settings.value(ANALYSIS_TARGET_FS))
            multiplier = int(1 / float(self.__settings.value(ANALYSIS_RESOLUTION)))
            peak_window = self.__get_window(ANALYSIS_PEAK_WINDOW)
            avg_window = self.__get_window(ANALYSIS_AVG_WINDOW)
            logger.debug(f"Analysing {self.signalName.text()} at {multiplier}x resolution "
                         f"using {peak_window} peak window and {avg_window} avg window")
            self.__signal.calculate(multiplier, avg_window, peak_window)
            self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(True)

    def __get_window(self, key):
        from model.preferences import ANALYSIS_WINDOW_DEFAULT
        window = self.__settings.value(key)
        if window is None or window == ANALYSIS_WINDOW_DEFAULT:
            window = None
        else:
            if window == 'tukey':
                window = (window, 0.25)
        return window

    def getMagnitudeData(self, reference=None):
        '''
        :param reference: ignored as we don't expose a normalisation control in this chart.
        :return: the peak and avg spectrum for the currently loaded signal (if any).
        '''
        if self.__signal is not None:
            return self.__signal.getXY()
        else:
            return []

    def accept(self):
        '''
        Adds the signal to the model and exits if we have a signal (which we should because the button is disabled
        until we do).
        '''
        if self.__signal is not None:
            self.__signalModel.add(self.__signal)
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
