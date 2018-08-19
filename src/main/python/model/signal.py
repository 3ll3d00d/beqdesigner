import datetime
import logging
import typing
from collections import Sequence

import numpy as np
from qtpy.QtCore import QAbstractTableModel, QModelIndex, QVariant, Qt
from scipy import signal

logger = logging.getLogger('signal')


class SignalModel(Sequence):
    '''
    A model to hold onto the signals.
    '''

    def __init__(self, view):
        self.__signals = []
        self.__view = view
        self.table = None

    def __getitem__(self, i):
        return self.__signals[i]

    def __len__(self):
        return len(self.__signals)

    def add(self, signals):
        '''
        Add the supplied signals ot the model.
        :param signals: the signals.
        '''
        if self.table is not None:
            self.table.beginResetModel()
        self.__signals.extend(signals)
        if self.table is not None:
            self.table.endResetModel()

    def remove(self, signal):
        '''
        Remove the specified signal from the model.
        :param signal: the signal to remove.
        '''
        if self.table is not None:
            self.table.beginResetModel()
        self.__signals.remove(signal)
        if self.table is not None:
            self.table.endResetModel()

    def delete(self, indices):
        '''
        Delete the signals at the given indices.
        :param indices: the indices to remove.
        '''
        if self.table is not None:
            self.table.beginResetModel()
        self.__signals = [signal for idx, signal in enumerate(self.__signals) if idx not in indices]
        self.table.resizeColumns(self.__view)
        if self.table is not None:
            self.table.endResetModel()


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

    def spectrum(self, segmentLengthMultiplier=1, mode=None, **kwargs):
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
                                       **kwargs)
            Pxx_spec = np.sqrt(Pxx_spec)
            # it seems a 3dB adjustment is required to account for the change in nperseg
            if x > 0:
                Pxx_spec = Pxx_spec / (10 ** ((3 * x) / 20))
            Pxx_spec = amplitude_to_db(Pxx_spec)
            return f, Pxx_spec

        if mode == 'cq':
            return self._cq(analysisFunc, segmentLengthMultiplier)
        else:
            return analysisFunc(0, self.getSegmentLength() * segmentLengthMultiplier, **kwargs)

    def peakSpectrum(self, segmentLengthMultiplier=1, mode=None, window='hann'):
        """
        analyses the source to generate the max values per bin per segment
        :param segmentLengthMultiplier: allow for increased resolution.
        :param mode: cq or none.
        :return:
            f : ndarray
            Array of sample frequencies.
            Pxx : ndarray
            linear spectrum max values.
        """

        def analysisFunc(x, nperseg):
            freqs, _, Pxy = signal.spectrogram(self.samples,
                                               self.fs,
                                               window=window,
                                               nperseg=int(nperseg),
                                               noverlap=int(nperseg // 2),
                                               detrend=False,
                                               scaling='spectrum')
            Pxy_max = np.sqrt(Pxy.max(axis=-1).real)
            if x > 0:
                Pxy_max = Pxy_max / (10 ** ((3 * x) / 20))
            Pxy_max = amplitude_to_db(Pxy_max)
            return freqs, Pxy_max

        if mode == 'cq':
            return self._cq(analysisFunc, segmentLengthMultiplier)
        else:
            return analysisFunc(0, self.getSegmentLength() * segmentLengthMultiplier)

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


def readWav(inputSignalFile, selectedChannel=1, start=None, end=None) -> Signal:
    """ reads a wav file into a Signal.
    :param inputSignalFile: a path to the input signal file
    :param selectedChannel: the channel to read.
    :param start: the time to start reading from in HH:mm:ss.SSS format.
    :param end: the time to end reading from in HH:mm:ss.SSS format.
    :returns: Signal.
    """

    def asFrames(time, fs):
        hours, minutes, seconds = (time.split(":"))[-3:]
        hours = int(hours)
        minutes = int(minutes)
        seconds = float(seconds)
        millis = int((3600000 * hours) + (60000 * minutes) + (1000 * seconds))
        return int(millis * (fs / 1000))

    import soundfile as sf
    if start is not None or end is not None:
        info = sf.info(inputSignalFile)
        startFrame = 0 if start is None else asFrames(start, info.samplerate)
        endFrame = None if end is None else asFrames(end, info.samplerate)
        ys, frameRate = sf.read(inputSignalFile, start=startFrame, stop=endFrame)
    else:
        ys, frameRate = sf.read(inputSignalFile)
    return Signal(ys[::selectedChannel], frameRate)


def amplitude_to_db(s):
    '''
    Convert an amplitude spectrogram to dB-scaled spectrogram. Implementation taken from librosa to avoid adding a
    dependency on librosa for a single function.
    :param s: the amplitude spectrogram.
    :return: s_db : np.ndarray ``s`` measured in dB
    '''
    magnitude = np.abs(np.asarray(s))
    power = np.square(magnitude, out=magnitude)
    ref_value = 1.0
    amin = 1e-10
    top_db = 80.0
    log_spec = 10.0 * np.log10(np.maximum(amin, power))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))
    log_spec = np.maximum(log_spec, log_spec.max() - top_db)
    return log_spec


class SignalTableModel(QAbstractTableModel):
    '''
    A Qt table model to feed the signal view.
    '''

    def __init__(self, model, parent=None):
        super().__init__(parent=parent)
        self._headers = ['Name', 'Duration', 'Fs', 'Start', 'End']
        self._signalModel = model
        self._signalModel.table = self

    def rowCount(self, parent: QModelIndex = ...):
        return len(self._signalModel)

    def columnCount(self, parent: QModelIndex = ...):
        return len(self._headers)

    def data(self, index: QModelIndex, role: int = ...) -> typing.Any:
        if not index.isValid():
            return QVariant()
        elif role != Qt.DisplayRole:
            return QVariant()
        else:
            signal_at_row = self._signalModel[index.row()]
            if index.column() == 0:
                return QVariant(signal_at_row.name)
            elif index.column() == 1:
                return QVariant(signal_at_row.duration_hhmmss)
            elif index.column() == 2:
                return QVariant(signal_at_row.fs)
            elif index.column() == 3:
                return QVariant(signal_at_row.start_hhmmss)
            elif index.column() == 4:
                return QVariant(signal_at_row.end_hhmmss)
            else:
                return QVariant()

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = ...) -> typing.Any:
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return QVariant(self._headers[section])
        return QVariant()

    def resizeColumns(self, view):
        for x in range(0, len(self._headers)):
            view.resizeColumnToContents(x)
