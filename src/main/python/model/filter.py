import logging
import math
import time
import typing
from collections import Sequence
from uuid import uuid4

from qtpy.QtWidgets import QDialog, QDialogButtonBox
from qtpy.QtCore import QAbstractTableModel, QModelIndex, QVariant, Qt

from model.iir import ComplexFilter, FilterType, LowShelf, HighShelf, PeakingEQ, SecondOrder_LowPass, \
    SecondOrder_HighPass, ComplexLowPass, ComplexHighPass
from ui.filter import Ui_editFilterDialog

COMBINED = 'Combined'

logger = logging.getLogger('filter')


class FilterModel(Sequence):
    '''
    A model to hold onto the filters.
    '''

    def __init__(self, view, showIndividualFilters):
        self.__filter = ComplexFilter(description=COMBINED)
        self.__view = view
        self.__showIndividualFilters = showIndividualFilters
        self.table = None

    def __getitem__(self, i):
        return self.__filter[i]

    def __len__(self):
        return len(self.__filter)

    def add(self, filter):
        '''
        Stores a new filter.
        :param filter: the filter.
        '''
        if self.table is not None:
            self.table.beginResetModel()
        self.__filter.add(filter)
        self.table.resizeColumns(self.__view)
        if self.table is not None:
            self.table.endResetModel()

    def replace(self, filter):
        '''
        Replaces the given filter.
        :param filter: the filter.
        '''
        if self.table is not None:
            self.table.beginResetModel()
        self.__filter.replace(filter)
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
        self.__filter.removeByIndex(indices)
        self.table.resizeColumns(self.__view)
        if self.table is not None:
            self.table.endResetModel()

    def getMagnitudeData(self):
        '''
        :return: the magnitude response of each filter.
        '''
        includeIndividualFilters = self.__showIndividualFilters.isChecked()
        if len(self.__filter) > 0:
            start = time.time()
            children = [x.getTransferFunction() for x in self.__filter]
            results = [self.__filter.getTransferFunction()]
            if includeIndividualFilters and len(self) > 1:
                results += children
            mags = [r.getMagnitude() for r in results]
            end = time.time()
            logger.debug(f"Calculated {len(mags)} transfer functions in {end-start}ms")
            return mags
        else:
            return []


class FilterTableModel(QAbstractTableModel):
    '''
    A Qt table model to feed the filter view.
    '''

    def __init__(self, model, parent=None):
        super().__init__(parent=parent)
        self._headers = ['Type', 'Freq', 'Q', 'Gain', 'Biquads']
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
            filter_at_row = self._filterModel[index.row()]
            if index.column() == 0:
                return QVariant(filter_at_row.filter_type)
            elif index.column() == 1:
                return QVariant(filter_at_row.freq)
            elif index.column() == 2:
                if hasattr(filter_at_row, 'q'):
                    return QVariant(filter_at_row.q)
                else:
                    return QVariant('N/A')
            elif index.column() == 3:
                if hasattr(filter_at_row, 'gain'):
                    return QVariant(filter_at_row.gain)
                else:
                    return QVariant('N/A')
            elif index.column() == 4:
                return QVariant(len(filter_at_row))
            else:
                return QVariant()

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = ...) -> typing.Any:
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return QVariant(self._headers[section])
        return QVariant()

    def resizeColumns(self, view):
        for x in range(0, len(self._headers)):
            view.resizeColumnToContents(x)


class FilterDialog(QDialog, Ui_editFilterDialog):
    '''
    Add/Edit Filter dialog
    '''
    gain_required = ['Low Shelf', 'High Shelf', 'Peak']

    def __init__(self, filterModel, fs=48000, filter=None, parent=None):
        super(FilterDialog, self).__init__(parent)
        self.setupUi(self)
        self.filterModel = filterModel
        self.filter = filter
        self.fs = fs
        if self.filter is not None:
            self.setWindowTitle('Edit Filter')
            if hasattr(self.filter, 'gain'):
                self.filterGain.setValue(self.filter.gain)
            if hasattr(self.filter, 'q'):
                self.filterQ.setValue(self.filter.q)
            self.freq.setValue(self.filter.freq)
            if hasattr(self.filter, 'order'):
                self.filterOrder.setValue(self.filter.order)
            if hasattr(self.filter, 'type'):
                displayName = 'Butterworth' if filter.type is FilterType.BUTTERWORTH else 'Linkwitz-Riley'
                self.passFilterType.setCurrentIndex(self.passFilterType.findText(displayName))
            self.filterType.setCurrentIndex(self.filterType.findText(filter.display_name))
        else:
            self.buttonBox.button(QDialogButtonBox.Save).setText('Add')
        self.enableOkIfGainIsValid()

    def accept(self):
        if self.__is_pass_filter():
            filt = self.create_pass_filter()
        else:
            filt = self.create_shaping_filter()
        if self.filter is None:
            filt.id = uuid4()
            self.filterModel.add(filt)
        else:
            filt.id = self.filter.id
            self.filterModel.replace(filt)

    def create_shaping_filter(self):
        '''
        Creates a filter of the specified type.
        :param idx: the index.
        :param fs: the sampling frequency.
        :param type: the filter type.
        :param freq: the corner frequency.
        :param q: the filter Q.
        :param gain: the filter gain (if any).
        :return: the filter.
        '''
        if self.filterType.currentText() == 'Low Shelf':
            return LowShelf(self.fs, self.freq.value(), self.filterQ.value(), self.filterGain.value())
        elif self.filterType.currentText() == 'High Shelf':
            return HighShelf(self.fs, self.freq.value(), self.filterQ.value(), self.filterGain.value())
        elif self.filterType.currentText() == 'Peak':
            return PeakingEQ(self.fs, self.freq.value(), self.filterQ.value(), self.filterGain.value())
        elif self.filterType.currentText() == 'Variable Q LPF':
            return SecondOrder_LowPass(self.fs, self.freq.value(), self.filterQ.value())
        elif self.filterType.currentText() == 'Variable Q HPF':
            return SecondOrder_HighPass(self.fs, self.freq.value(), self.filterQ.value())
        else:
            raise ValueError(f"Unknown filter type {self.filterType.currentText()}")

    def create_pass_filter(self):
        '''
        Creates a predefined high or low pass filter.
        :return: the filter.
        '''
        if self.filterType.currentText() == 'Low Pass':
            return ComplexLowPass(FilterType[self.passFilterType.currentText().upper().replace('-', '_')],
                                  self.filterOrder.value(), self.fs, self.freq.value())
        else:
            return ComplexHighPass(FilterType[self.passFilterType.currentText().upper().replace('-', '_')],
                                   self.filterOrder.value(), self.fs, self.freq.value())

    def __is_pass_filter(self):
        '''
        :return: true if the current options indicate a predefined high or low pass filter.
        '''
        selectedFilter = self.filterType.currentText()
        return selectedFilter == 'Low Pass' or selectedFilter == 'High Pass'

    def enableFilterParams(self):
        if self.__is_pass_filter():
            self.passFilterType.setEnabled(True)
            self.filterOrder.setEnabled(True)
            self.filterQ.setEnabled(False)
            self.filterGain.setEnabled(False)
        else:
            self.passFilterType.setEnabled(False)
            self.filterOrder.setEnabled(False)
            self.filterQ.setEnabled(True)
            self.filterGain.setEnabled(self.__is_gain_required())
        self.enableOkIfGainIsValid()

    def changeOrderStep(self):
        '''
        Sets the order step based on the type of high/low pass filter to ensure that LR only allows even orders.
        '''
        if self.passFilterType.currentText() == 'Butterworth':
            self.filterOrder.setSingleStep(1)
        elif self.passFilterType.currentText() == 'Linkwitz-Riley':
            if self.filterOrder.value() % 2 != 0:
                self.filterOrder.setValue(2)
            self.filterOrder.setSingleStep(2)

    def enableOkIfGainIsValid(self):
        if self.__is_gain_required():
            self.buttonBox.button(QDialogButtonBox.Save).setEnabled(not math.isclose(self.filterGain.value(), 0.0))
        else:
            self.buttonBox.button(QDialogButtonBox.Save).setEnabled(True)

    def __is_gain_required(self):
        return self.filterType.currentText() in self.gain_required
