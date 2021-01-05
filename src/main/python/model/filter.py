import json
import logging
import typing
from collections import defaultdict
from collections.abc import Sequence
from uuid import uuid4

import math
import qtawesome as qta
from qtpy import QtCore
from qtpy.QtCore import QAbstractTableModel, QModelIndex, QVariant, Qt
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QDialog, QFileDialog, QMessageBox, QHeaderView, QTableView

from model.iir import FilterType, LowShelf, HighShelf, PeakingEQ, SecondOrder_LowPass, \
    SecondOrder_HighPass, ComplexLowPass, ComplexHighPass, q_to_s, s_to_q, max_permitted_s, CompleteFilter, COMBINED, \
    Passthrough, Gain, Shelf, LinkwitzTransform
from model.limits import dBRangeCalculator, PhaseRangeCalculator
from model.magnitude import MagnitudeModel
from model.preferences import SHOW_ALL_FILTERS, SHOW_NO_FILTERS, FILTER_COLOURS, DISPLAY_SHOW_FILTERS, DISPLAY_Q_STEP, \
    DISPLAY_GAIN_STEP, DISPLAY_S_STEP, DISPLAY_FREQ_STEP, get_filter_colour, FILTERS_DEFAULT_Q, FILTERS_DEFAULT_FREQ, \
    FILTERS_GEOMETRY
from ui.filter import Ui_editFilterDialog

logger = logging.getLogger('filter')


class FilterModel(Sequence):
    '''
    A model to hold onto the filters and provide magnitude data to a chart about those filters.
    '''

    def __init__(self, view, preferences, label=None, on_update=lambda _: True):
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
            if self.__label is not None:
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

    def clone(self, detach=False):
        '''
        Clones the current filter.
        '''
        clone = self.filter.preview(None)
        if detach is True:
            for f in clone:
                f.id = uuid4()
        return clone

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

    def get_curve_data(self, reference=None):
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
            children = [x.get_transfer_function() for x in self.filter]
            combined = self.filter.get_transfer_function()
            results = [combined]
            if show_filters == SHOW_ALL_FILTERS and len(self) > 1:
                results += children
            mags = [r.get_magnitude() for r in results]
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

    def get_transfer_function(self, fs=None):
        '''
        :return: the transfer function for this filter (in total) if we have any filters or None if we have none.
        '''
        if len(self.filter) > 0:
            if fs is not None:
                return self.filter.resample(fs).get_transfer_function()
            else:
                return self.filter.get_transfer_function()
        return None


class FilterTableModel(QAbstractTableModel):
    '''
    A Qt table model to feed the filter view.
    '''

    def __init__(self, model, parent=None):
        super().__init__(parent=parent) if parent is not None else super().__init__()
        self.__headers = ['Type', 'Freq', 'Q', 'S', 'Gain', 'Biquads']
        self.__filter_model = model
        self.__filter_model.table = self

    def rowCount(self, parent: QModelIndex = ..., *args, **kwargs):
        return len(self.__filter_model)

    def columnCount(self, parent: QModelIndex = ..., *args, **kwargs):
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
    gain_steps = [0.01, 0.1, 1.0]
    freq_steps = [0.01, 0.1, 1.0, 2.0, 5.0]
    passthrough = Passthrough()

    def __init__(self, preferences, signal, filter_model, redraw_main, selected_filter=None, parent=None,
                 valid_filter_types=None, **kwargs):
        self.__preferences = preferences
        super(FilterDialog, self).__init__(parent) if parent is not None else super(FilterDialog, self).__init__()
        self.__redraw_main = redraw_main
        # for shelf filter, allow input via Q or S not both
        self.__q_is_active = True
        # allow user to control the steps for different fields, default to reasonably quick moving values
        self.__q_step_idx = self.__get_step(self.q_steps, self.__preferences.get(DISPLAY_Q_STEP), 3)
        self.__s_step_idx = self.__get_step(self.q_steps, self.__preferences.get(DISPLAY_S_STEP), 3)
        self.__gain_step_idx = self.__get_step(self.gain_steps, self.__preferences.get(DISPLAY_GAIN_STEP), 0)
        self.__freq_step_idx = self.__get_step(self.freq_steps, self.__preferences.get(DISPLAY_FREQ_STEP), 2)
        # init the UI itself
        self.setupUi(self)
        self.__snapshot = FilterModel(self.snapshotFilterView, self.__preferences, on_update=self.__on_snapshot_change)
        self.__working = FilterModel(self.workingFilterView, self.__preferences, on_update=self.__on_working_change)
        self.__selected_id = None
        self.__decorate_ui()
        self.__set_q_step(self.q_steps[self.__q_step_idx])
        self.__set_s_step(self.q_steps[self.__s_step_idx])
        self.__set_gain_step(self.gain_steps[self.__gain_step_idx])
        self.__set_freq_step(self.freq_steps[self.__freq_step_idx])
        # underlying filter model
        self.__signal = signal
        self.__filter_model = filter_model
        if self.__filter_model.filter.listener is not None:
            logger.debug(f"Selected filter has listener {self.__filter_model.filter.listener.name}")
        self.__magnitude_model = None
        # remove unsupported filter types
        if valid_filter_types:
            to_remove = []
            for i in range(self.filterType.count()):
                if self.filterType.itemText(i) not in valid_filter_types:
                    to_remove.append(i)
            for i1, i2 in enumerate(to_remove):
                self.filterType.removeItem(i2 - i1)
        # copy the filter into the working table
        self.__working.filter = self.__filter_model.clone()
        # and initialise the view
        for idx, f in enumerate(self.__working):
            selected = selected_filter is not None and f.id == selected_filter.id
            f.id = uuid4()
            if selected is True:
                self.__selected_id = f.id
                self.workingFilterView.selectRow(idx)
        if self.__selected_id is None:
            self.__add_working_filter()
        # init the chart
        self.__magnitude_model = MagnitudeModel('preview', self.previewChart, preferences,
                                                self.__get_data(), 'Filter', fill_primary=True,
                                                secondary_data_provider=self.__get_data('phase'),
                                                secondary_name='Phase', secondary_prefix='deg', fill_secondary=False,
                                                db_range_calc=dBRangeCalculator(30, expand=True),
                                                y2_range_calc=PhaseRangeCalculator(), show_y2_in_legend=False,
                                                **kwargs)
        self.__restore_geometry()
        self.filterType.setFocus()

    def __restore_geometry(self):
        ''' loads the saved window size '''
        geometry = self.__preferences.get(FILTERS_GEOMETRY)
        if geometry is not None:
            self.restoreGeometry(geometry)

    def closeEvent(self, QCloseEvent):
        ''' Stores the window size on close '''
        self.__preferences.set(FILTERS_GEOMETRY, self.saveGeometry())
        super().closeEvent(QCloseEvent)

    def __select_working_filter(self):
        ''' Loads the selected filter into the edit fields. '''
        selection = self.workingFilterView.selectionModel()
        if selection.hasSelection():
            idx = selection.selectedRows()[0].row()
            self.headerLabel.setText(f"Working Filter {idx+1}")
            self.__select_filter(self.__working[idx])

    def __select_snapshot_filter(self):
        ''' Loads the selected filter into the edit fields. '''
        selection = self.snapshotFilterView.selectionModel()
        if selection.hasSelection():
            idx = selection.selectedRows()[0].row()
            self.headerLabel.setText(f"Snapshot Filter {idx+1}")
            self.__select_filter(self.__snapshot[idx])

    def __on_snapshot_change(self, _):
        ''' makes the snapshot table visible when we have one. '''
        self.snapshotFilterView.setVisible(len(self.__snapshot) > 0)
        self.snapshotViewButtonWidget.setVisible(len(self.__snapshot) > 0)
        if self.__magnitude_model is not None:
            self.__magnitude_model.redraw()
        return True

    def __on_working_change(self, visible_names):
        ''' ensure the graph redraws when a filter changes. '''
        if self.__magnitude_model is not None:
            self.__magnitude_model.redraw()
        return True

    def __decorate_ui(self):
        ''' polishes the UI by setting tooltips, adding icons and connecting widgets to functions. '''
        self.__set_tooltips()
        self.__set_icons()
        self.__connect_working_buttons()
        self.__connect_snapshot_buttons()
        self.__link_table_views()

    def __link_table_views(self):
        ''' Links the table views into the dialog. '''
        self.snapshotFilterView.setVisible(False)
        self.snapshotViewButtonWidget.setVisible(False)
        self.snapshotFilterView.setModel(FilterTableModel(self.__snapshot))
        self.snapshotFilterView.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.snapshotFilterView.selectionModel().selectionChanged.connect(self.__select_snapshot_filter)
        self.workingFilterView.setModel(FilterTableModel(self.__working))
        self.workingFilterView.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.workingFilterView.selectionModel().selectionChanged.connect(self.__select_working_filter)

    def __set_icons(self):
        self.saveButton.setIcon(qta.icon('fa5s.save'))
        self.saveButton.setIconSize(QtCore.QSize(32, 32))
        self.exitButton.setIcon(qta.icon('fa5s.sign-out-alt'))
        self.exitButton.setIconSize(QtCore.QSize(32, 32))
        self.snapFilterButton.setIcon(qta.icon('fa5s.copy'))
        self.acceptSnapButton.setIcon(qta.icon('fa5s.check'))
        self.loadSnapButton.setIcon(qta.icon('fa5s.folder-open'))
        self.resetButton.setIcon(qta.icon('fa5s.undo'))
        self.optimiseButton.setIcon(qta.icon('fa5s.magic'))
        self.addWorkingRowButton.setIcon(qta.icon('fa5s.plus'))
        self.addSnapshotRowButton.setIcon(qta.icon('fa5s.plus'))
        self.removeWorkingRowButton.setIcon(qta.icon('fa5s.minus'))
        self.removeSnapshotRowButton.setIcon(qta.icon('fa5s.minus'))
        self.limitsButton.setIcon(qta.icon('fa5s.arrows-alt'))
        self.fullRangeButton.setIcon(qta.icon('fa5s.expand'))
        self.subOnlyButton.setIcon(qta.icon('fa5s.compress'))
        self.importWorkingButton.setIcon(qta.icon('fa5s.file-import'))
        self.importSnapshotButton.setIcon(qta.icon('fa5s.file-import'))

    def __set_tooltips(self):
        self.addSnapshotRowButton.setToolTip('Add new filter to snapshot')
        self.removeSnapshotRowButton.setToolTip('Remove selected filter from snapshot')
        self.addWorkingRowButton.setToolTip('Add new filter')
        self.removeWorkingRowButton.setToolTip('Remove selected filter')
        self.snapFilterButton.setToolTip('Freeze snapshot')
        self.loadSnapButton.setToolTip('Load snapshot')
        self.acceptSnapButton.setToolTip('Apply snapshot')
        self.resetButton.setToolTip('Reset snapshot')
        self.optimiseButton.setToolTip('Optimise filters')
        self.targetBiquadCount.setToolTip('Optimised filter target biquad count')
        self.saveButton.setToolTip('Save')
        self.exitButton.setToolTip('Exit')
        self.limitsButton.setToolTip('Set Graph Limits')
        self.importWorkingButton.setToolTip('Import Filters from REW')
        self.importSnapshotButton.setToolTip('Import Filters from REW')

    def __connect_working_buttons(self):
        ''' Connects the buttons associated with the working filter. '''
        self.addWorkingRowButton.clicked.connect(self.__add_working_filter)
        self.removeWorkingRowButton.clicked.connect(self.__remove_working_filter)
        self.importWorkingButton.clicked.connect(self.__import_working_filters)

    def __add_working_filter(self):
        ''' adds a new filter. '''
        self.__add_filter(self.__make_default_filter(), self.__working, self.workingFilterView)

    def __remove_working_filter(self):
        ''' removes the selected filter. '''
        self.__remove_filter(self.workingFilterView, self.__working)

    def __connect_snapshot_buttons(self):
        ''' Connects the buttons associated with the snapshot filter. '''
        self.snapFilterButton.clicked.connect(self.__snap_filter)
        self.resetButton.clicked.connect(self.__clear_snapshot)
        self.acceptSnapButton.clicked.connect(self.__apply_snapshot)
        self.loadSnapButton.clicked.connect(self.__load_filter_as_snapshot)
        self.optimiseButton.clicked.connect(self.__optimise_filter)
        self.addSnapshotRowButton.clicked.connect(self.__add_snapshot_filter)
        self.removeSnapshotRowButton.clicked.connect(self.__remove_snapshot_filter)
        self.importSnapshotButton.clicked.connect(self.__import_snapshot_filters)

    def __add_snapshot_filter(self):
        ''' adds a new filter. '''
        self.__add_filter(self.__make_default_filter(), self.__snapshot, self.snapshotFilterView)

    @staticmethod
    def __add_filter(new_filter, filter_model: FilterModel, filter_view: QTableView):
        filter_model.save(new_filter)
        for idx, f in enumerate(filter_model):
            if f.id == new_filter.id:
                filter_view.selectRow(idx)

    def __make_default_filter(self):
        ''' Creates a new filter using the default preferences or by copying the currently selected filter. '''
        active_model, _ = self.__get_active()
        if len(active_model) > 0:
            for f in active_model:
                if f.id == self.__selected_id:
                    new_f = f.resample(self.__signal.fs)
                    new_f.id = uuid4()
                    return new_f
        return LowShelf(self.__signal.fs,
                        self.__preferences.get(FILTERS_DEFAULT_FREQ),
                        self.__preferences.get(FILTERS_DEFAULT_Q),
                        0.0,
                        f_id=uuid4())

    def __remove_snapshot_filter(self):
        ''' removes the selected filter. '''
        self.__remove_filter(self.snapshotFilterView, self.__snapshot)

    @staticmethod
    def __remove_filter(filter_view: QTableView, filter_model: FilterModel):
        selection = filter_view.selectionModel()
        if selection.hasSelection():
            to_delete = [r.row() for r in selection.selectedRows()]
            filter_model.delete(to_delete)
            if len(filter_model) > 0:
                filter_view.selectRow(min(max(to_delete), len(filter_model) - 1))

    def __import_working_filters(self):
        self.__import_filters(self.__working, self.workingFilterView)

    def __import_snapshot_filters(self):
        self.__import_filters(self.__snapshot, self.snapshotFilterView)

    def __import_filters(self, filter_model: FilterModel, filter_view: QTableView):
        selected = QFileDialog.getOpenFileName(parent=self, caption='Import REW Filters', filter='Filter (*.txt)')
        if selected and selected[0]:
            filts = []
            discarded = defaultdict(list)
            with open(selected[0]) as f:
                for line in f:
                    tokens = line.split()
                    if len(tokens) > 4 and tokens[0] == 'Filter' and tokens[2] == 'ON':
                        filt = None
                        if tokens[3] == 'PK':
                            filt = PeakingEQ(self.__signal.fs, float(tokens[5]), float(tokens[11]), float(tokens[8]), f_id=uuid4())
                        elif tokens[3] == 'LSQ':
                            filt = LowShelf(self.__signal.fs, float(tokens[5]), float(tokens[11]), float(tokens[8]), f_id=uuid4())
                        elif tokens[3] == 'HSQ':
                            filt = HighShelf(self.__signal.fs, float(tokens[5]), float(tokens[11]), float(tokens[8]), f_id=uuid4())
                        else:
                            discarded[tokens[3]].append(tokens[1][:-1])
                        if filt:
                            filts.append(filt)
            if filts:
                for f in filts:
                    self.__add_filter(f, filter_model, filter_view)
            if discarded.keys():
                msg_box = QMessageBox()
                formatted = '\n'.join([f"{k} - Filter{'s' if len(v) > 1 else ''} {','.join(v)}" for k,v in discarded.items()])
                msg_box.setText(f"Ignored filters\n\n{formatted}")
                msg_box.setIcon(QMessageBox.Information)
                msg_box.setWindowTitle('Ignored Unsupported Filter Types')
                msg_box.exec()

    def __snap_filter(self):
        '''
        Captures the current filter as the snapshot.
        '''
        self.__snapshot.filter = self.__working.clone()

    def __clear_snapshot(self):
        ''' Removes the current snapshot. '''
        self.__snapshot.delete(range(len(self.__snapshot)))

    def __apply_snapshot(self):
        ''' Saves the snapshot as the filter. '''
        if len(self.__snapshot) > 0:
            snap = self.__snapshot.filter
            self.__clear_snapshot()
            snap.description = self.__filter_model.filter.description
            self.__filter_model.filter = snap

    def __load_filter_as_snapshot(self):
        ''' Allows a filter to be loaded from a supported file format and set as the snapshot. '''
        result = QMessageBox.question(self,
                                      'Load Filter or XML?',
                                      f"Do you want to load from a filter or a minidsp beq file?"
                                      f"\n\nClick Yes to load from a filter or No for a beq file",
                                      QMessageBox.Yes | QMessageBox.No,
                                      QMessageBox.No)
        load_xml = result == QMessageBox.No
        loaded_snapshot = None
        if load_xml is True:
            from model.minidsp import load_as_filter
            filters, _ = load_as_filter(self, self.__preferences, self.__signal.fs)
            if filters is not None:
                loaded_snapshot = CompleteFilter(fs=self.__signal.fs, filters=filters, description='Snapshot')
        else:
            loaded_snapshot = load_filter(self)
        if loaded_snapshot is not None:
            self.__snapshot.filter = loaded_snapshot

    def __optimise_filter(self):
        '''
        Optimises the current filter and stores it as a snapshot.
        '''
        current_filter = self.__working.clone()
        to_save = self.targetBiquadCount.value() - current_filter.biquads
        if to_save < 0:
            optimised_filter = CompleteFilter(fs=current_filter.fs,
                                              filters=optimise_filters(current_filter, current_filter.fs, -to_save),
                                              description='Optimised')
            self.__snapshot.filter = optimised_filter
        else:
            QMessageBox.information(self,
                                    'Optimise Filters',
                                    f"Current filter uses {current_filter.biquads} biquads so no optimisation required")

    def show_limits(self):
        ''' shows the limits dialog for the filter chart. '''
        self.__magnitude_model.show_limits()

    def show_full_range(self):
        ''' sets the limits to full range. '''
        self.__magnitude_model.show_full_range()

    def show_sub_only(self):
        ''' sets the limits to sub only. '''
        self.__magnitude_model.show_sub_only()

    def __select_filter(self, selected_filter):
        ''' Refreshes the params and display with the selected filter '''
        from model.report import block_signals
        self.__selected_id = selected_filter.id
        # populate the fields with values if we're editing an existing filter
        if hasattr(selected_filter, 'gain'):
            with block_signals(self.filterGain):
                self.filterGain.setValue(selected_filter.gain)
        if hasattr(selected_filter, 'q'):
            with block_signals(self.filterQ):
                self.filterQ.setValue(selected_filter.q)
        if hasattr(selected_filter, 'freq'):
            with block_signals(self.freq):
                self.freq.setValue(selected_filter.freq)
        if hasattr(selected_filter, 'order'):
            with block_signals(self.filterOrder):
                self.filterOrder.setValue(selected_filter.order)
        if hasattr(selected_filter, 'type'):
            displayName = 'Butterworth' if selected_filter.type is FilterType.BUTTERWORTH else 'Linkwitz-Riley'
            with block_signals(self.passFilterType):
                self.passFilterType.setCurrentIndex(self.passFilterType.findText(displayName))
        if hasattr(selected_filter, 'count') and issubclass(type(selected_filter), Shelf):
            with block_signals(self.filterCount):
                self.filterCount.setValue(selected_filter.count)
        with block_signals(self.filterType):
            self.filterType.setCurrentText(selected_filter.display_name)
        # configure visible/enabled fields for the current filter type
        self.enableFilterParams()
        with block_signals(self.freq):
            self.freq.setMaximum(self.__signal.fs / 2.0)

    @staticmethod
    def __get_step(steps, value, default_idx):
        for idx, val in enumerate(steps):
            if str(val) == value:
                return idx
        return default_idx

    def __write_to_filter_model(self):
        ''' Stores the filter in the model. '''
        self.__filter_model.filter = self.__working.filter
        self.__signal.filter = self.__filter_model.filter
        self.__redraw_main()

    def accept(self):
        ''' Saves the filter. '''
        self.previewFilter()
        self.__write_to_filter_model()

    def previewFilter(self):
        ''' creates a filter if the params are valid '''
        active_model, active_view = self.__get_active()
        active_model.save(self.__create_filter())
        self.__ensure_filter_is_selected(active_model, active_view)

    def __get_active(self) -> typing.Tuple[FilterModel, QTableView]:
        if self.headerLabel.text().startswith('Working') or len(self.headerLabel.text()) == 0:
            active_model = self.__working
            active_view = self.workingFilterView
        else:
            active_model = self.__snapshot
            active_view = self.snapshotFilterView
        return active_model, active_view

    def __ensure_filter_is_selected(self, active_model, active_view):
        '''
        Filter model resets the model on every change, this clears the selection so we have to restore that selection
        to ensure the row remains visibly selected while also blocking signals to avoid a pointless update of the
        fields.
        '''
        for idx, f in enumerate(active_model):
            if f.id == self.__selected_id:
                from model.report import block_signals
                with block_signals(active_view):
                    active_view.selectRow(idx)

    def __get_data(self, mode='mag'):
        return lambda *args, **kwargs: self.get_curve_data(mode, *args, **kwargs)

    def get_curve_data(self, mode, reference=None):
        ''' preview of the filter to display on the chart '''
        result = []
        if mode == 'mag' or self.showPhase.isChecked():
            extra = 0
            if len(self.__filter_model) > 0:
                result.append(self.__filter_model.get_transfer_function()
                                                 .get_data(mode=mode, colour=get_filter_colour(len(result))))
            else:
                extra += 1
            if len(self.__working) > 0:
                result.append(self.__working.get_transfer_function()
                                            .get_data(mode=mode, colour=get_filter_colour(len(result)), linestyle='-'))
            else:
                extra += 1
            if len(self.__snapshot) > 0:
                result.append(self.__snapshot.get_transfer_function()
                                             .get_data(mode=mode, colour=get_filter_colour(len(result) + extra),
                                                       linestyle='-.'))
            else:
                extra += 1
            active_model, _ = self.__get_active()
            for f in active_model:
                if self.showIndividual.isChecked() or f.id == self.__selected_id:
                    style = '--' if f.id == self.__selected_id else ':'
                    result.append(f.get_transfer_function()
                                   .get_data(mode=mode, colour=get_filter_colour(len(result) + extra), linestyle=style))
        return result

    def create_shaping_filter(self):
        '''
        Creates a filter of the specified type.
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
        elif self.filterType.currentText() == 'Linkwitz Transform':
            filt = LinkwitzTransform(self.__signal.fs, self.f0.value(), self.q0.value(), self.fp.value(), self.qp.value())
        if filt is None:
            raise ValueError(f"Unknown filter type {self.filterType.currentText()}")
        else:
            filt.id = self.__selected_id
        return filt

    def __create_filter(self):
        ''' creates a filter from the currently selected parameters. '''
        return self.create_pass_filter() if self.__is_pass_filter() else self.create_shaping_filter()

    def create_pass_filter(self):
        '''
        Creates a predefined high or low pass filter.
        :return: the filter.
        '''
        if self.filterType.currentText() == 'Low Pass':
            filt = ComplexLowPass(FilterType[self.passFilterType.currentText().upper().replace('-', '_')],
                                  self.filterOrder.value(), self.__signal.fs, self.freq.value())
        else:
            filt = ComplexHighPass(FilterType[self.passFilterType.currentText().upper().replace('-', '_')],
                                   self.filterOrder.value(), self.__signal.fs, self.freq.value())
        filt.id = self.__selected_id
        return filt

    def __is_pass_filter(self):
        '''
        :return: true if the current options indicate a predefined high or low pass filter.
        '''
        selected_filter = self.filterType.currentText()
        return selected_filter == 'Low Pass' or selected_filter == 'High Pass'

    def __is_gain_filter(self):
        '''
        :return: true if the current options indicate a gain filter.
        '''
        selected_filter = self.filterType.currentText()
        return selected_filter == 'Gain'

    def __is_linkwitz_transform(self):
        '''
        :return: true if the current options indicate an LT.
        '''
        selected_filter = self.filterType.currentText()
        return selected_filter == 'Linkwitz Transform'

    def enableFilterParams(self):
        '''
        Configures the various input fields for the currently selected filter type.
        '''
        self.f0.setVisible(False)
        self.q0.setVisible(False)
        self.fp.setVisible(False)
        self.qp.setVisible(False)
        self.ltInLabel.setVisible(False)
        self.ltOutLabel.setVisible(False)
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
            elif self.__is_linkwitz_transform():
                self.qStepButton.setVisible(False)
                self.filterQ.setVisible(False)
                self.filterQLabel.setVisible(False)
                self.freq.setVisible(False)
                self.freqStepButton.setVisible(False)
                self.freqLabel.setVisible(False)
                self.f0.setVisible(True)
                self.q0.setVisible(True)
                self.fp.setVisible(True)
                self.qp.setVisible(True)
                self.ltInLabel.setVisible(True)
                self.ltOutLabel.setVisible(True)
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

    def show_phase_response(self):
        self.__magnitude_model.redraw()


def optimise_filters(filters, fs, to_save):
    '''
    Attempts to optimise the no of filters required.
    :param filters: the filters to optimise.
    :param fs: the sigma fs.
    :param to_save: how many filters we want to save.
    :return: the optimised filters.
    '''
    unstackable = sorted([f for f in filters if isinstance(f, Shelf) and f.count > 1],
                         key=lambda f: f.count, reverse=True)
    unstackable = sorted(unstackable, key=lambda f: f.gain * f.count, reverse=True)
    weights = [abs(w.gain * w.count) for w in unstackable]
    total_weight = sum(weights)
    allocated = []
    for idx, weight in enumerate(weights):
        weight = float(weight)
        p = weight / total_weight
        max_amount = unstackable[idx].count - 1
        distributed_amount = min(round(p * (to_save - sum(allocated))), max_amount)
        total_weight -= weight
        allocated.append(distributed_amount)
    new_filts = [f for f in filters if not isinstance(f, Shelf) or f.count < 2]
    for idx, s in enumerate(unstackable):
        reduce_by = allocated[idx]
        if reduce_by > 0:
            total_gain = s.gain * s.count
            new_count = s.count - reduce_by
            if new_count > 0:
                new_gain = total_gain / new_count
                tmp_shelf = LowShelf(fs, s.freq, s.q, new_gain, count=new_count)
                average_s = (tmp_shelf.q_to_s() + s.q_to_s()) / 2
                new_shelf = LowShelf(fs, s.freq, s_to_q(average_s, new_gain), new_gain, count=new_count)
                logger.info(f"Replacing {s} with {new_shelf}")
                new_filts.append(new_shelf)
            else:
                raise ValueError(f"Attempted to reduce shelf count to 0 from {s.count}")
        else:
            new_filts.append(s)
    return new_filts


def load_filter(parent, status_bar=None):
    '''
    Presents a file dialog to the user so they can choose a filter to load.
    :return: the loaded filter, if any.
    '''
    dialog = QFileDialog(parent=parent)
    dialog.setFileMode(QFileDialog.ExistingFile)
    dialog.setNameFilter(f"*.filter")
    dialog.setWindowTitle(f"Load Filter")
    if dialog.exec():
        selected = dialog.selectedFiles()
        if len(selected) > 0:
            with open(selected[0], 'r') as infile:
                input = json.load(infile)
                if status_bar is not None:
                    status_bar.showMessage(f"Loaded filter from {infile.name}")
                return input
    return None
