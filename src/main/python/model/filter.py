import logging
import math
import typing
from collections import Sequence
from uuid import uuid4

import qtawesome as qta
from qtpy import QtCore
from qtpy.QtCore import QAbstractTableModel, QModelIndex, QVariant, Qt
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QDialog, QDialogButtonBox

from model.iir import FilterType, LowShelf, HighShelf, PeakingEQ, SecondOrder_LowPass, \
    SecondOrder_HighPass, ComplexLowPass, ComplexHighPass, q_to_s, s_to_q, max_permitted_s, CompleteFilter, COMBINED, \
    Passthrough
from model.magnitude import MagnitudeModel
from mpl import get_line_colour
from ui.filter import Ui_editFilterDialog

logger = logging.getLogger('filter')

SHOW_ALL_FILTERS = 'All'
SHOW_COMBINED_FILTER = 'Total'
SHOW_NO_FILTERS = 'None'
SHOW_FILTER_OPTIONS = [SHOW_ALL_FILTERS, SHOW_COMBINED_FILTER, SHOW_NO_FILTERS]


class FilterModel(Sequence):
    '''
    A model to hold onto the filters.
    '''

    def __init__(self, view, show_filters=lambda: SHOW_ALL_FILTERS, on_update=lambda _: True):
        self.filter = CompleteFilter()
        self.__view = view
        self.__show_filters = show_filters
        self.__table = None
        self.__listeners = []
        self.__on_update = on_update

    @property
    def table(self):
        return self.__table

    @table.setter
    def table(self, table):
        self.__table = table
        self.__table.resizeColumns(self.__view)

    def __getitem__(self, i):
        return self.filter[i]

    def __len__(self):
        return len(self.filter)

    def register(self, listener):
        '''
        Registers a listener for filter change events.
        :param listener: the listener.
        '''
        self.__listeners.append(listener)

    def deregister(self, listener):
        '''
        Deregisters a listener on filter change events.
        :param listener: the listener
        '''
        self.__listeners.remove(listener)

    def save(self, filter):
        '''
        Stores the filter.
        :param filter: the filter.
        '''
        if self.__table is not None:
            self.__table.beginResetModel()
        self.filter.save(filter)
        self.post_update()
        if self.__table is not None:
            self.__table.endResetModel()

    def preview(self, filter):
        '''
        Previews the effect of saving the supplied filter.
        :param filter: the filter.
        :return: a previewed filter.
        '''
        return self.filter.preview(filter)

    def delete(self, indices):
        '''
        Deletes the filter at the specified index.
        :param indices the indexes to delete.
        '''
        if self.__table is not None:
            self.__table.beginResetModel()
        self.filter.removeByIndex(indices)
        self.post_update()
        if self.__table is not None:
            self.__table.endResetModel()

    def post_update(self, filter_change=True):
        '''
        Reacts to a change in the model.
        :param filter_change: true if the change came from an actual filter change.
        '''
        if self.__table is not None:
            self.__table.resizeColumns(self.__view)
        visible_filter_names = []
        show_filters = self.__show_filters()
        if show_filters != SHOW_NO_FILTERS:
            visible_filter_names.append(self.filter.__repr__())
        if show_filters == SHOW_ALL_FILTERS:
            visible_filter_names += self.filter.child_names()
        if filter_change:
            for l in self.__listeners:
                l.onFilterChange()
        self.__on_update(visible_filter_names)

    def getMagnitudeData(self, reference=None):
        '''
        :param reference: the name of the reference data.
        :return: the magnitude response of each filter.
        '''
        show_filters = self.__show_filters()
        if show_filters == SHOW_NO_FILTERS:
            return []
        elif len(self.filter) == 0:
            return []
        else:
            children = [x.getTransferFunction() for x in self.filter]
            combined = self.filter.getTransferFunction()
            results = [combined]
            if show_filters == SHOW_ALL_FILTERS and len(self) > 1:
                results += children
            mags = [r.getMagnitude() for r in results]
            for idx, m in enumerate(mags):
                if m.name == COMBINED:
                    m.colour = 'c'
                else:
                    m.colour = get_line_colour(idx, len(mags) - 1)
            if reference is not None:
                ref_data = next((x for x in mags if x.name == reference), None)
                if ref_data:
                    mags = [x.normalise(ref_data) for x in mags]
            return mags

    def getTransferFunction(self):
        '''
        :return: the transfer function for this filter (in total) if we have any filters or None if we have none.
        '''
        if len(self.filter) > 0:
            return self.filter.getTransferFunction()
        return None


class FilterTableModel(QAbstractTableModel):
    '''
    A Qt table model to feed the filter view.
    '''

    def __init__(self, model, parent=None):
        super().__init__(parent=parent)
        self._headers = ['Type', 'Freq', 'Q', 'S', 'Gain', 'Biquads']
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
                if hasattr(filter_at_row, 'q_to_s'):
                    return QVariant(round(filter_at_row.q_to_s(), 3))
                else:
                    return QVariant('N/A')
            elif index.column() == 4:
                if hasattr(filter_at_row, 'gain'):
                    return QVariant(filter_at_row.gain)
                else:
                    return QVariant('N/A')
            elif index.column() == 5:
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
    is_shelf = ['Low Shelf', 'High Shelf']
    gain_required = is_shelf + ['Peak']
    q_steps = [0.0001, 0.001, 0.01, 0.1]
    gain_steps = [0.1, 1.0]
    freq_steps = [0.1, 1.0, 2.0, 5.0]
    passthrough = Passthrough()

    def __init__(self, filterModel, fs=48000, filter=None, parent=None):
        # prevent signals from recalculating the filter before we've populated the fields
        self.__starting = True
        super(FilterDialog, self).__init__(parent)
        # for shelf filter, allow input via Q or S not both
        self.__q_is_active = True
        # allow user to control the steps for different fields
        self.__q_step_idx = 0
        self.__s_step_idx = 0
        self.__gain_step_idx = 0
        self.__freq_step_idx = 0
        # init the UI itself
        self.setupUi(self)
        # underlying filter model
        self.filterModel = filterModel
        self.__filter = filter
        self.__combined_preview = filterModel.filter
        # populate the fields with values if we're editing an existing filter
        self.__original_id = self.__filter.id if filter is not None else None
        self.fs = fs if filter is None else filter.fs
        if self.__filter is not None:
            self.setWindowTitle('Edit Filter')
            if hasattr(self.__filter, 'gain'):
                self.filterGain.setValue(self.__filter.gain)
            if hasattr(self.__filter, 'q'):
                self.filterQ.setValue(self.__filter.q)
            self.freq.setValue(self.__filter.freq)
            if hasattr(self.__filter, 'order'):
                self.filterOrder.setValue(self.__filter.order)
            if hasattr(self.__filter, 'type'):
                displayName = 'Butterworth' if filter.type is FilterType.BUTTERWORTH else 'Linkwitz-Riley'
                self.passFilterType.setCurrentIndex(self.passFilterType.findText(displayName))
            if hasattr(self.__filter, 'count'):
                self.filterCount.setValue(self.__filter.count)
            self.filterType.setCurrentIndex(self.filterType.findText(filter.display_name))
        # configure visible/enabled fields for the current filter type
        self.enableFilterParams()
        self.enableOkIfGainIsValid()
        self.freq.setMaximum(self.fs / 2.0)
        self.__starting = False
        # init the chart
        self.__magnitudeModel = MagnitudeModel('preview', self.previewChart, self, 'Filter')
        # ensure the preview graph is shown if we have something to show
        self.previewFilter()

    def save(self):
        ''' Stores the filter in the model. '''
        if self.__filter is not None:
            if self.__original_id is None:
                self.__filter.id = uuid4()
            self.filterModel.save(self.__filter)
            self.previewFilter()

    def accept(self):
        ''' Saves and exits. '''
        self.save()
        QDialog.accept(self)

    def previewFilter(self):
        ''' creates a filter if the params are valid '''
        if not self.__starting:
            if self.__is_valid_filter():
                if self.__is_pass_filter():
                    self.__filter = self.create_pass_filter()
                else:
                    self.__filter = self.create_shaping_filter()
                if self.__original_id is not None:
                    self.__filter.id = self.__original_id
                self.__combined_preview = self.filterModel.preview(self.__filter)
                self.passthrough.rendered = False
            else:
                self.__combined_preview = self.filterModel.filter
                self.__filter = None
            self.__magnitudeModel.redraw()

    def getMagnitudeData(self, reference=None):
        ''' preview of the filter to display on the chart '''
        if self.__filter is not None:
            result = [self.__filter.getTransferFunction().getMagnitude(colour='m')]
            if len(self.filterModel) > 0 and self.showCombined.isChecked():
                result.append(self.__combined_preview.getTransferFunction().getMagnitude(colour='c'))
            return result
        else:
            if len(self.filterModel) > 0 and self.showCombined.isChecked():
                return [self.__combined_preview.getTransferFunction().getMagnitude(colour='c')]
            else:
                return [self.passthrough.getTransferFunction().getMagnitude(colour='m')]

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
            return LowShelf(self.fs, self.freq.value(), self.filterQ.value(), self.filterGain.value(),
                            self.filterCount.value())
        elif self.filterType.currentText() == 'High Shelf':
            return HighShelf(self.fs, self.freq.value(), self.filterQ.value(), self.filterGain.value(),
                             self.filterCount.value())
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
        '''
        Configures the various input fields for the currently selected filter type.
        '''
        if self.__is_pass_filter():
            self.passFilterType.setVisible(True)
            self.filterOrder.setVisible(True)
            self.orderLabel.setVisible(True)
            self.filterQ.setVisible(False)
            self.filterQLabel.setVisible(False)
            self.qStepButton.setVisible(False)
            self.filterGain.setVisible(False)
            self.gainStepButton.setVisible(False)
            self.gainLabel.setVisible(False)
        else:
            self.passFilterType.setVisible(False)
            self.filterOrder.setVisible(False)
            self.orderLabel.setVisible(False)
            self.qStepButton.setVisible(True)
            self.filterQ.setVisible(True)
            self.filterQLabel.setVisible(True)
            self.filterGain.setVisible(self.__is_gain_required())
            self.gainStepButton.setVisible(self.__is_gain_required())
            self.gainLabel.setVisible(self.__is_gain_required())
        is_shelf_filter = self.__is_shelf_filter()
        if is_shelf_filter:
            self.filterQ.setEnabled(self.__q_is_active)
            self.filterS.setEnabled(not self.__q_is_active)
            # set icons
            inactive_icon = qta.icon('fa.chevron-circle-left')
            if self.__q_is_active is True:
                self.qStepButton.setText(str(self.q_steps[self.__q_step_idx % len(self.q_steps)]))
                self.sStepButton.setIcon(inactive_icon)
            else:
                self.qStepButton.setIcon(inactive_icon)
                self.sStepButton.setText(str(self.q_steps[self.__s_step_idx % len(self.q_steps)]))
        self.gainStepButton.setText(str(self.gain_steps[self.__gain_step_idx % len(self.gain_steps)]))
        self.freqStepButton.setText(str(self.freq_steps[self.__freq_step_idx % len(self.freq_steps)]))
        self.filterCountLabel.setVisible(is_shelf_filter)
        self.filterCount.setVisible(is_shelf_filter)
        self.sLabel.setVisible(is_shelf_filter)
        self.filterS.setVisible(is_shelf_filter)
        self.sStepButton.setVisible(is_shelf_filter)
        self.addButton.setIcon(qta.icon('fa.plus'))
        self.addButton.setIconSize(QtCore.QSize(32, 32))
        self.addMoreButton.setIcon(qta.icon('fa.save'))
        self.addMoreButton.setIconSize(QtCore.QSize(32, 32))
        self.exitButton.setIcon(qta.icon('fa.sign-out'))
        self.exitButton.setIconSize(QtCore.QSize(32, 32))
        self.enableOkIfGainIsValid()

    def changeOrderStep(self):
        '''
        Sets the order step based on the type of high/low pass filter to ensure that LR only allows even orders.
        '''
        if self.passFilterType.currentText() == 'Butterworth':
            self.filterOrder.setSingleStep(1)
            self.filterOrder.setMinimum(1)
        elif self.passFilterType.currentText() == 'Linkwitz-Riley':
            if self.filterOrder.value() % 2 != 0:
                self.filterOrder.setValue(max(2, self.filterOrder.value() - 1))
            self.filterOrder.setSingleStep(2)
            self.filterOrder.setMinimum(2)

    def enableOkIfGainIsValid(self):
        ''' enables the save button if we have a valid filter. '''
        self.addMoreButton.setEnabled(self.__is_valid_filter())
        self.addButton.setEnabled(self.__is_valid_filter())

    def __is_valid_filter(self):
        '''
        :return: true if the current params are valid.
        '''
        if self.__is_gain_required():
            return not math.isclose(self.filterGain.value(), 0.0)
        else:
            return True

    def __is_gain_required(self):
        return self.filterType.currentText() in self.gain_required

    def __is_shelf_filter(self):
        return self.filterType.currentText() in self.is_shelf

    def recalcShelfFromQ(self, q):
        '''
        Updates S based on the selected value of Q.
        :param q: the q.
        '''
        gain = self.filterGain.value()
        if self.__q_is_active is True and not math.isclose(gain, 0.0):
            self.filterS.setValue(q_to_s(q, gain))

    def recalcShelfFromGain(self, gain):
        '''
        Updates S based on the selected gain.
        :param gain: the gain.
        '''
        if not math.isclose(gain, 0.0):
            max_s = round(max_permitted_s(gain), 4)
            self.filterS.setMaximum(max_s)
            if self.__q_is_active is True:
                q = self.filterQ.value()
                self.filterS.setValue(q_to_s(q, gain))
            else:
                if self.filterS.value() > max_s:
                    self.filterS.blockSignals(True)
                    self.filterS.setValue(max_s, 4)
                    self.filterS.blockSignals(False)
                self.filterQ.setValue(s_to_q(self.filterS.value(), gain))

    def recalcShelfFromS(self, s):
        '''
        Updates the shelf based on a change in S
        :param s: the new S
        '''
        gain = self.filterGain.value()
        if self.__q_is_active is False and not math.isclose(gain, 0.0):
            self.filterQ.setValue(s_to_q(s, gain))

    def handleSToolButton(self):
        '''
        Reacts to the S tool button click.
        '''
        if self.__q_is_active is True:
            self.__q_is_active = False
            self.filterS.setEnabled(True)
            self.sStepButton.setIcon(QIcon())
            self.filterQ.setEnabled(False)
            self.qStepButton.setIcon(qta.icon('fa.chevron-circle-left'))
        else:
            self.__s_step_idx += 1
        idx = self.__s_step_idx % len(self.q_steps)
        self.sStepButton.setText(str(self.q_steps[idx]))
        self.filterS.setSingleStep(self.q_steps[idx])

    def handleQToolButton(self):
        '''
        Reacts to the q tool button click.
        '''
        if self.__q_is_active is True:
            self.__q_step_idx += 1
        else:
            self.__q_is_active = True
            self.filterS.setEnabled(False)
            self.qStepButton.setIcon(QIcon())
            self.filterQ.setEnabled(True)
            self.sStepButton.setIcon(qta.icon('fa.chevron-circle-left'))
        idx = self.__q_step_idx % len(self.q_steps)
        self.qStepButton.setText(str(self.q_steps[idx]))
        self.filterQ.setSingleStep(self.q_steps[idx])

    def handleGainToolButton(self):
        '''
        Reacts to the gain tool button click.
        '''
        self.__gain_step_idx += 1
        idx = self.__gain_step_idx % len(self.gain_steps)
        self.gainStepButton.setText(str(self.gain_steps[idx]))
        self.filterGain.setSingleStep(self.gain_steps[idx])

    def handleFreqToolButton(self):
        '''
        Reacts to the frequency tool button click.
        '''
        self.__freq_step_idx += 1
        idx = self.__freq_step_idx % len(self.freq_steps)
        self.freqStepButton.setText(str(self.freq_steps[idx]))
        self.freq.setSingleStep(self.freq_steps[idx])
