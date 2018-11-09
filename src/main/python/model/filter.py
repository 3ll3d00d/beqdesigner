import logging
import math
import typing
from collections import Sequence
from uuid import uuid4

import qtawesome as qta
from qtpy import QtCore
from qtpy.QtCore import QAbstractTableModel, QModelIndex, QVariant, Qt
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QDialog

from model.iir import FilterType, LowShelf, HighShelf, PeakingEQ, SecondOrder_LowPass, \
    SecondOrder_HighPass, ComplexLowPass, ComplexHighPass, q_to_s, s_to_q, max_permitted_s, CompleteFilter, COMBINED, \
    Passthrough, Gain
from model.limits import dBRangeCalculator
from model.magnitude import MagnitudeModel
from model.preferences import SHOW_ALL_FILTERS, SHOW_NO_FILTERS, FILTER_COLOURS, DISPLAY_SHOW_FILTERS, DISPLAY_Q_STEP, \
    DISPLAY_GAIN_STEP, DISPLAY_S_STEP, DISPLAY_FREQ_STEP, get_filter_colour
from ui.filter import Ui_editFilterDialog

logger = logging.getLogger('filter')


class FilterModel(Sequence):
    '''
    A model to hold onto the filters and provide magnitude data to a chart about those filters.
    '''

    def __init__(self, view, label, preferences, on_update=lambda _: True):
        self.__filter = CompleteFilter()
        self.__view = view
        self.__preferences = preferences
        self.__table = None
        self.__label = label
        self.__on_update = on_update

    @property
    def filter(self):
        return self.__filter

    @filter.setter
    def filter(self, filt):
        if filt is None:
            filt = CompleteFilter()
        if isinstance(filt, CompleteFilter):
            if self.__table is not None:
                self.__table.beginResetModel()
            self.__filter = filt
            if self.__filter.listener is not None:
                self.__label.setText(f"Filter - {filt.listener.name}")
            else:
                self.__label.setText(f"Filter - Default")
            self.post_update()
            if self.__table is not None:
                self.__table.endResetModel()
        else:
            raise ValueError(f"FilterModel only accepts CompleteFilter, ignoring {filt}")

    @property
    def table(self):
        return self.__table

    @table.setter
    def table(self, table):
        self.__table = table

    def __getitem__(self, i):
        return self.filter[i]

    def __len__(self):
        return len(self.filter)

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

    def post_update(self):
        '''
        Reacts to a change in the model.
        '''
        visible_filter_names = []
        show_filters = self.__preferences.get(DISPLAY_SHOW_FILTERS)
        if show_filters != SHOW_NO_FILTERS:
            visible_filter_names.append(self.filter.__repr__())
        if show_filters == SHOW_ALL_FILTERS:
            visible_filter_names += self.filter.child_names()
        self.__on_update(visible_filter_names)

    def getMagnitudeData(self, reference=None):
        '''
        :param reference: the name of the reference data.
        :return: the magnitude response of each filter.
        '''
        show_filters = self.__preferences.get(DISPLAY_SHOW_FILTERS)
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
                    m.colour = FILTER_COLOURS[0]
                else:
                    m.colour = FILTER_COLOURS[(idx + 1) % len(FILTER_COLOURS)]
            if reference is not None:
                ref_data = next((x for x in mags if x.name == reference), None)
                if ref_data:
                    mags = [x.normalise(ref_data) for x in mags]
            return mags

    def resample(self, fs):
        '''
        :param fs: the requested fs.
        :return: the filter at that fs.
        '''
        return self.filter.resample(fs)

    def getTransferFunction(self, fs=None):
        '''
        :return: the transfer function for this filter (in total) if we have any filters or None if we have none.
        '''
        if len(self.filter) > 0:
            if fs is not None:
                return self.filter.resample(fs).getTransferFunction()
            else:
                return self.filter.getTransferFunction()
        return None


class FilterTableModel(QAbstractTableModel):
    '''
    A Qt table model to feed the filter view.
    '''

    def __init__(self, model, parent=None):
        super().__init__(parent=parent)
        self.__headers = ['Type', 'Freq', 'Q', 'S', 'Gain', 'Biquads']
        self.__filter_model = model
        self.__filter_model.table = self

    def rowCount(self, parent: QModelIndex = ...):
        return len(self.__filter_model)

    def columnCount(self, parent: QModelIndex = ...):
        return len(self.__headers)

    def data(self, index: QModelIndex, role: int = ...) -> typing.Any:
        if not index.isValid():
            return QVariant()
        elif role != Qt.DisplayRole:
            return QVariant()
        else:
            filter_at_row = self.__filter_model[index.row()]
            if index.column() == 0:
                return QVariant(filter_at_row.filter_type)
            elif index.column() == 1:
                if hasattr(filter_at_row, 'freq'):
                    return QVariant(filter_at_row.freq)
                else:
                    return QVariant('N/A')
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
            return QVariant(self.__headers[section])
        return QVariant()


class FilterDialog(QDialog, Ui_editFilterDialog):
    '''
    Add/Edit Filter dialog
    '''
    is_shelf = ['Low Shelf', 'High Shelf']
    gain_required = is_shelf + ['PEQ', 'Gain']
    q_steps = [0.0001, 0.001, 0.01, 0.1]
    gain_steps = [0.1, 1.0]
    freq_steps = [0.1, 1.0, 2.0, 5.0]
    passthrough = Passthrough()

    def __init__(self, preferences, signal, filter_model, filter=None, parent=None):
        self.__preferences = preferences
        # prevent signals from recalculating the filter before we've populated the fields
        self.__starting = True
        if parent is not None:
            super(FilterDialog, self).__init__(parent)
        else:
            super(FilterDialog, self).__init__()
        # for shelf filter, allow input via Q or S not both
        self.__q_is_active = True
        # allow user to control the steps for different fields, default to reasonably quick moving values
        self.__q_step_idx = self.__get_step(self.q_steps, self.__preferences.get(DISPLAY_Q_STEP), 3)
        self.__s_step_idx = self.__get_step(self.q_steps, self.__preferences.get(DISPLAY_S_STEP), 3)
        self.__gain_step_idx = self.__get_step(self.gain_steps, self.__preferences.get(DISPLAY_GAIN_STEP), 0)
        self.__freq_step_idx = self.__get_step(self.freq_steps, self.__preferences.get(DISPLAY_FREQ_STEP), 1)
        # init the UI itself
        self.setupUi(self)
        self.__set_q_step(self.q_steps[self.__q_step_idx])
        self.__set_s_step(self.q_steps[self.__s_step_idx])
        self.__set_gain_step(self.gain_steps[self.__gain_step_idx])
        self.__set_freq_step(self.freq_steps[self.__freq_step_idx])
        # underlying filter model
        self.__signal = signal
        self.__filter_model = filter_model
        if self.__filter_model.filter.listener is not None:
            logger.debug(f"Selected filter has listener {self.__filter_model.filter.listener.name}")
        self.__filter = None
        self.__original_id = filter.id if filter is not None else None
        self.__combined_preview = signal.filter
        # init the chart
        self.__magnitudeModel = MagnitudeModel('preview', self.previewChart, preferences, self, 'Filter', db_range_calc=dBRangeCalculator(30))
        # load the selector and populate the fields
        self.__refresh_selector()

    @property
    def filter(self):
        return self.__filter

    @filter.setter
    def filter(self, filt):
        self.__filter = filt

    def __refresh_selector(self):
        ''' loads the selector with the filters '''
        from model.report import block_signals
        with block_signals(self.filterSelector):
            current_idx = max(self.filterSelector.currentIndex(), 0)
            self.filterSelector.clear()
            self.filterSelector.addItem('Add New Filter')
            for idx, f in enumerate(self.__filter_model):
                self.filterSelector.addItem(f"{idx+1}: {f}")
                if self.__original_id is not None and self.__original_id == f.id:
                    current_idx = idx + 1
            self.filterSelector.setCurrentIndex(current_idx)
            self.select_filter(current_idx)

    def select_filter(self, idx):
        ''' Refreshes the params and display with the selected filter '''
        if idx == 0:
            self.filter = None
            self.__original_id = None
            self.setWindowTitle('Add Filter')
        else:
            selected_filter = self.__filter_model[idx - 1]
            self.__original_id = selected_filter.id
            # populate the fields with values if we're editing an existing filter
            self.setWindowTitle(f"Edit Filter {idx}")
            if hasattr(selected_filter, 'gain'):
                self.filterGain.setValue(selected_filter.gain)
            if hasattr(selected_filter, 'q'):
                self.filterQ.setValue(selected_filter.q)
            if hasattr(selected_filter, 'freq'):
                self.freq.setValue(selected_filter.freq)
            if hasattr(selected_filter, 'order'):
                self.filterOrder.setValue(selected_filter.order)
            if hasattr(selected_filter, 'type'):
                displayName = 'Butterworth' if selected_filter.type is FilterType.BUTTERWORTH else 'Linkwitz-Riley'
                self.passFilterType.setCurrentIndex(self.passFilterType.findText(displayName))
            if hasattr(selected_filter, 'count'):
                self.filterCount.setValue(selected_filter.count)
            self.filterType.setCurrentText(selected_filter.display_name)
        # configure visible/enabled fields for the current filter type
        self.enableFilterParams()
        self.enableOkIfGainIsValid()
        self.freq.setMaximum(self.__signal.fs / 2.0)
        self.__starting = False
        # ensure the preview graph is shown if we have something to show
        self.previewFilter()

    def __is_edit(self):
        return self.__original_id is not None

    def __get_step(self, steps, value, default_idx):
        for idx, val in enumerate(steps):
            if str(val) == value:
                return idx
        return default_idx

    def save_filter(self):
        ''' Stores the filter in the model. '''
        if self.filter is not None:
            if not self.__is_edit():
                self.filter.id = uuid4()
            self.__filter_model.save(self.filter)

    def accept(self):
        ''' Saves an existing filter. '''
        self.previewFilter()
        self.save_filter()
        self.__refresh_selector()

    def previewFilter(self):
        ''' creates a filter if the params are valid '''
        if not self.__starting:
            if self.__is_valid_filter():
                if self.__is_pass_filter():
                    self.filter = self.create_pass_filter(self.__original_id)
                else:
                    self.filter = self.create_shaping_filter(self.__original_id)
                self.__combined_preview = self.__filter_model.preview(self.filter)
                self.passthrough.rendered = False
            else:
                self.__combined_preview = self.__filter_model.preview(Passthrough(fs=self.__signal.fs))
                self.filter = None
            self.__magnitudeModel.redraw()

    def getMagnitudeData(self, reference=None):
        ''' preview of the filter to display on the chart '''
        if self.filter is not None:
            result = [self.filter.getTransferFunction().getMagnitude(colour=get_filter_colour(0), linestyle=':')]
            if len(self.__filter_model) > 0:
                self.__add_combined_filter(result)
                self.__add_individual_filters(result)
            return result
        else:
            if len(self.__filter_model) > 0:
                result = []
                self.__add_combined_filter(result)
                self.__add_individual_filters(result)
                return result
            else:
                return [self.passthrough.getTransferFunction().getMagnitude(colour=get_filter_colour(1), linestyle=':')]

    def __add_combined_filter(self, result):
        if self.showCombined.isChecked():
            result.append(self.__combined_preview.getTransferFunction()
                                                 .getMagnitude(colour=get_filter_colour(len(result))))

    def __add_individual_filters(self, result):
        if self.showIndividual.isChecked():
            for f in self.__filter_model:
                if f.id != self.filter.id:
                    result.append(f.getTransferFunction().getMagnitude(colour=get_filter_colour(len(result))))

    def create_shaping_filter(self, original_id):
        '''
        Creates a filter of the specified type.
        :param original_id: the original id.
        :return: the filter.
        '''
        filt = None
        if self.filterType.currentText() == 'Low Shelf':
            filt = LowShelf(self.__signal.fs, self.freq.value(), self.filterQ.value(), self.filterGain.value(),
                            self.filterCount.value())
        elif self.filterType.currentText() == 'High Shelf':
            filt = HighShelf(self.__signal.fs, self.freq.value(), self.filterQ.value(), self.filterGain.value(),
                             self.filterCount.value())
        elif self.filterType.currentText() == 'PEQ':
            filt = PeakingEQ(self.__signal.fs, self.freq.value(), self.filterQ.value(), self.filterGain.value())
        elif self.filterType.currentText() == 'Gain':
            filt = Gain(self.__signal.fs, self.filterGain.value())
        elif self.filterType.currentText() == 'Variable Q LPF':
            filt = SecondOrder_LowPass(self.__signal.fs, self.freq.value(), self.filterQ.value())
        elif self.filterType.currentText() == 'Variable Q HPF':
            filt = SecondOrder_HighPass(self.__signal.fs, self.freq.value(), self.filterQ.value())
        if filt is None:
            raise ValueError(f"Unknown filter type {self.filterType.currentText()}")
        else:
            filt.id = original_id
        return filt

    def create_pass_filter(self, original_id):
        '''
        Creates a predefined high or low pass filter.
        :param original_id: the id.
        :return: the filter.
        '''
        filt = None
        if self.filterType.currentText() == 'Low Pass':
            filt = ComplexLowPass(FilterType[self.passFilterType.currentText().upper().replace('-', '_')],
                                  self.filterOrder.value(), self.__signal.fs, self.freq.value())
        else:
            filt = ComplexHighPass(FilterType[self.passFilterType.currentText().upper().replace('-', '_')],
                                   self.filterOrder.value(), self.__signal.fs, self.freq.value())
        filt.id = original_id
        return filt

    def __is_pass_filter(self):
        '''
        :return: true if the current options indicate a predefined high or low pass filter.
        '''
        selectedFilter = self.filterType.currentText()
        return selectedFilter == 'Low Pass' or selectedFilter == 'High Pass'

    def __is_gain_filter(self):
        '''
        :return: true if the current options indicate a predefined high or low pass filter.
        '''
        selectedFilter = self.filterType.currentText()
        return selectedFilter == 'Gain'

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
            if self.__is_gain_filter():
                self.qStepButton.setVisible(False)
                self.filterQ.setVisible(False)
                self.filterQLabel.setVisible(False)
                self.freq.setVisible(False)
                self.freqStepButton.setVisible(False)
                self.freqLabel.setVisible(False)
            else:
                self.qStepButton.setVisible(True)
                self.filterQ.setVisible(True)
                self.filterQLabel.setVisible(True)
                self.freq.setVisible(True)
                self.freqStepButton.setVisible(True)
                self.freqLabel.setVisible(True)
            self.filterGain.setVisible(self.__is_gain_required())
            self.gainStepButton.setVisible(self.__is_gain_required())
            self.gainLabel.setVisible(self.__is_gain_required())
        is_shelf_filter = self.__is_shelf_filter()
        if is_shelf_filter:
            self.filterQ.setEnabled(self.__q_is_active)
            self.filterS.setEnabled(not self.__q_is_active)
            # set icons
            inactive_icon = qta.icon('fa5s.chevron-circle-left')
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
        self.saveButton.setIcon(qta.icon('fa5s.save'))
        self.saveButton.setIconSize(QtCore.QSize(32, 32))
        self.exitButton.setIcon(qta.icon('fa5s.sign-out-alt'))
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
        ''' enables the save buttons if we have a valid filter. '''
        self.saveButton.setEnabled(self.__is_valid_filter())

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
            self.qStepButton.setIcon(qta.icon('fa5s.chevron-circle-left'))
        else:
            self.__s_step_idx += 1
        self.__set_s_step(self.q_steps[self.__s_step_idx % len(self.q_steps)])

    def __set_s_step(self, step_val):
        self.__preferences.set(DISPLAY_S_STEP, str(step_val))
        self.sStepButton.setText(str(step_val))
        self.filterS.setSingleStep(step_val)

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
            self.sStepButton.setIcon(qta.icon('fa5s.chevron-circle-left'))
        self.__set_q_step(self.q_steps[self.__q_step_idx % len(self.q_steps)])

    def __set_q_step(self, step_val):
        self.__preferences.set(DISPLAY_Q_STEP, str(step_val))
        self.qStepButton.setText(str(step_val))
        self.filterQ.setSingleStep(step_val)

    def handleGainToolButton(self):
        '''
        Reacts to the gain tool button click.
        '''
        self.__gain_step_idx += 1
        self.__set_gain_step(self.gain_steps[self.__gain_step_idx % len(self.gain_steps)])

    def __set_gain_step(self, step_val):
        self.__preferences.set(DISPLAY_GAIN_STEP, str(step_val))
        self.gainStepButton.setText(str(step_val))
        self.filterGain.setSingleStep(step_val)

    def handleFreqToolButton(self):
        '''
        Reacts to the frequency tool button click.
        '''
        self.__freq_step_idx += 1
        self.__set_freq_step(self.freq_steps[self.__freq_step_idx % len(self.freq_steps)])

    def __set_freq_step(self, step_val):
        self.__preferences.set(DISPLAY_FREQ_STEP, str(step_val))
        self.freqStepButton.setText(str(step_val))
        self.freq.setSingleStep(step_val)
