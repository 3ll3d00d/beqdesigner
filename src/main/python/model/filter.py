import typing
from collections import Sequence

import numpy as np
from qtpy.QtCore import QAbstractTableModel, QModelIndex, QVariant, Qt
from scipy import signal
from yodel.filter import Biquad

COMBINED = 'Combined'


class Filter:
    def __init__(self, idx, fs, type, freq, q, gain=0.0):
        self.idx = idx
        self.type = type
        self.fs = fs
        self.freq = freq
        self.q = q
        self.gain = gain
        self.biquad = Biquad()
        if type == 'Low Shelf':
            self.biquad.low_shelf(fs, freq, q, gain)
        elif type == 'High Shelf':
            self.biquad.high_shelf(fs, freq, q, gain)
        elif type == 'Peak':
            self.biquad.peak(fs, freq, q, gain)
        elif type == 'Low Pass':
            self.biquad.low_pass(fs, freq, q)
        elif type == 'High Pass':
            self.biquad.high_pass(fs, freq, q)
        else:
            raise ValueError("Unknown filter type " + type)

    def __repr__(self):
        repr = f"Filter {self.idx}: {self.type} - {self.freq}/{self.q}"
        if self.gain != 0.0:
            repr += f"/{self.gain}dB"
        return repr


class FilterModel(Sequence):
    '''
    A model to hold onto the filters.
    '''

    def __init__(self, view):
        self.__filters = []
        self.__view = view
        self.table = None

    def __getitem__(self, i):
        return self.__filters[i]

    def __len__(self):
        return len(self.__filters)

    def add(self, fs, type, freq, q, gain=0.0):
        '''
        Stores a new filter.
        :param fs: the sample rate.
        :param type: the filter type.
        :param freq: the filter centre frequency in Hz.
        :param q: the q of the filter.
        :param gain: the filter gain in dB (if any).
        '''
        if self.table is not None:
            self.table.beginResetModel()
        self.__filters.append(Filter(len(self.__filters), fs, type, freq, q, gain))
        self.table.resizeColumns(self.__view)
        if self.table is not None:
            self.table.endResetModel()

    def delete(self, indices):
        '''
        Deletes the filter at the specified index.
        :param indices the indexes to delete.
        '''
        if self.table is not None:
            self.table.beginResetModel()
        self.__filters = [filter for idx, filter in enumerate(self.__filters) if idx not in indices]
        self.table.resizeColumns(self.__view)
        if self.table is not None:
            self.table.endResetModel()

    def getMagnitudeData(self):
        '''
        :return: the magnitude response of each filter.
        '''
        responses = [self._getFreqResponse(x) for x in self.__filters]
        if len(responses) > 1:
            responses += [self._getFreqResponse()]
        return responses

    def _getFreqResponse(self, filt=None):
        input = signal.unit_impulse(48000, idx='mid')
        output = np.zeros(48000)
        if filt is None:
            for f in self.__filters:
                f.biquad.process(input, output)
                input = output
        else:
            filt.biquad.process(input, output)

        nperseg = min(1 << (48000 - 1).bit_length(), output.shape[-1])
        f, Pxx_spec = signal.welch(output, 48000, nperseg=nperseg, scaling='spectrum', detrend=False)
        Pxx_spec = 20.0 * np.log10(np.sqrt(Pxx_spec))
        return XYData(filt.__repr__() if filt is not None else COMBINED, f, Pxx_spec)

class XYData:
    '''
    Value object for showing data on a magnitude graph.
    '''

    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y

    def normalise(self, target):
        '''
        Normalises the y value against the target y.
        :param target: the target.
        :return: a normalised XYData.
        '''
        return XYData(self.name, self.x, self.y - target.y)


class FilterTableModel(QAbstractTableModel):
    '''
    A Qt table model to feed the filter view.
    '''

    def __init__(self, model, parent=None):
        super().__init__(parent=parent)
        self._headers = ['Type', 'Freq', 'Q', 'Gain']
        self._filterModel = model
        self._filterModel.table = self

    def rowCount(self, parent: QModelIndex = ...):
        return len(self._filterModel)

    def columnCount(self, parent: QModelIndex = ...):
        return len(self._headers)

    def data(self, index: QModelIndex, role: int = ...) -> typing.Any:
        if not index.isValid():
            return QVariant()
        elif role != Qt.DisplayRole:
            return QVariant()
        else:
            if index.column() == 0:
                return QVariant(self._filterModel[index.row()].type)
            elif index.column() == 1:
                return QVariant(self._filterModel[index.row()].freq)
            elif index.column() == 2:
                return QVariant(self._filterModel[index.row()].q)
            elif index.column() == 3:
                return QVariant(self._filterModel[index.row()].gain)
            else:
                return QVariant()

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = ...) -> typing.Any:
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return QVariant(self._headers[section])
        return QVariant()

    def resizeColumns(self, view):
        for x in range(0, len(self._headers)):
            view.resizeColumnToContents(x)
