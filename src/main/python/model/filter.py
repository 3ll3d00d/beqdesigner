import logging
import time
import typing
from collections import Sequence
from functools import reduce

from qtpy.QtCore import QAbstractTableModel, QModelIndex, QVariant, Qt

from model.iir import ComplexData, ComplexFilter

COMBINED = 'Combined'

logger = logging.getLogger('filter')


class FilterModel(Sequence):
    '''
    A model to hold onto the filters.
    '''

    def __init__(self, view):
        self.__filter = ComplexFilter(description=COMBINED)
        self.__view = view
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

    def getMagnitudeData(self, includeIndividualFilters):
        '''
        :param includeIndividualFilters: if true, include the individual filters in the response
        :return: the magnitude response of each filter.
        '''
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
