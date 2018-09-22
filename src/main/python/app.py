import collections
import gzip
import json
import logging
import math
import os
import re
import sys
from contextlib import contextmanager

import matplotlib

matplotlib.use("Qt5Agg")
from model.report import SaveReportDialog
from model.batch import BatchExtractDialog
from model.analysis import AnalyseSignalDialog
from model.link import LinkSignalsDialog
from model.preferences import DISPLAY_SHOW_FILTERED_SIGNALS
from ui.delegates import RegexValidator

import qtawesome as qta
from matplotlib import style

from model.export import ExportSignalDialog, Mode
from model.iir import Passthrough, CompleteFilter
from ui.biquad import Ui_exportBiquadDialog
from ui.savechart import Ui_saveChartDialog

from qtpy import QtCore
from qtpy.QtCore import QSettings
from qtpy.QtGui import QIcon, QFont, QCursor
from qtpy.QtWidgets import QMainWindow, QApplication, QErrorMessage, QAbstractItemView, QDialog, QFileDialog, \
    QHeaderView, QMessageBox

from model.extract import ExtractAudioDialog
from model.filter import FilterTableModel, FilterModel, FilterDialog
from model.log import RollingLogger
from model.magnitude import MagnitudeModel
from model.preferences import PreferencesDialog, BINARIES_GROUP, ANALYSIS_TARGET_FS, STYLE_MATPLOTLIB_THEME, \
    Preferences, \
    SCREEN_GEOMETRY, SCREEN_WINDOW_STATE, FILTERS_PRESET_x, DISPLAY_SHOW_LEGEND, DISPLAY_SHOW_FILTERS, \
    SHOW_FILTER_OPTIONS, SHOW_SIGNAL_OPTIONS, DISPLAY_SHOW_SIGNALS, SHOW_FILTERED_SIGNAL_OPTIONS
from model.signal import SignalModel, SignalTableModel, SignalDialog, SignalData
from ui.beq import Ui_MainWindow

logger = logging.getLogger('beq')


@contextmanager
def wait_cursor(msg=None):
    '''
    Allows long running functions to show a busy cursor.
    :param msg: a message to put in the status bar.
    '''
    try:
        QApplication.setOverrideCursor(QCursor(QtCore.Qt.WaitCursor))
        yield
    finally:
        QApplication.restoreOverrideCursor()


class BeqDesigner(QMainWindow, Ui_MainWindow):
    '''
    The main UI.
    '''

    def __init__(self, app, parent=None):
        super(BeqDesigner, self).__init__(parent)
        self.logger = logging.getLogger('beqdesigner')
        self.app = app
        self.preferences = Preferences(QSettings("3ll3d00d", "beqdesigner"))
        if getattr(sys, 'frozen', False):
            self.__style_path_root = sys._MEIPASS
        else:
            self.__style_path_root = os.path.dirname(__file__)
        self.__version = 'UNKNOWN'
        with open(os.path.join(self.__style_path_root, 'VERSION')) as version_file:
            self.__version = version_file.read().strip()
        matplotlib_theme = self.preferences.get(STYLE_MATPLOTLIB_THEME)
        if matplotlib_theme is not None:
            if matplotlib_theme.startswith('beq'):
                style.use(os.path.join(self.__style_path_root, 'style', 'mpl', f"{matplotlib_theme}.mplstyle"))
            else:
                style.use(matplotlib_theme)
        self.setupUi(self)
        self.limitsButton.setIcon(qta.icon('ei.move'))
        self.showValuesButton.setIcon(qta.icon('ei.eye-open'))
        # logs
        self.logViewer = RollingLogger(self.preferences, parent=self)
        self.actionShow_Logs.triggered.connect(self.logViewer.show_logs)
        self.actionPreferences.triggered.connect(self.showPreferences)
        # init a default signal for when we want to edit a filter without a signal
        default_mag = Passthrough().getTransferFunction().getMagnitude()
        self.__default_signal = SignalData('default', self.preferences.get(ANALYSIS_TARGET_FS),
                                           [default_mag, default_mag], filter=CompleteFilter())
        # init the filter view selector
        self.showFilters.blockSignals(True)
        for x in SHOW_FILTER_OPTIONS:
            self.showFilters.addItem(x)
        selected = self.preferences.get(DISPLAY_SHOW_FILTERS)
        selected_idx = self.showFilters.findText(selected)
        if selected_idx != -1:
            self.showFilters.setCurrentIndex(selected_idx)
        else:
            logger.info(f"Ignoring unknown cached preference for {DISPLAY_SHOW_FILTERS} - {selected}")
        self.showFilters.blockSignals(False)
        # filter view/model
        self.filterView.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.__filter_model = FilterModel(self.filterView, self.filtersLabel, self.preferences,
                                          on_update=self.on_filter_change)
        self.__filter_table_model = FilterTableModel(self.__filter_model, parent=parent)
        self.filterView.setModel(self.__filter_table_model)
        self.filterView.selectionModel().selectionChanged.connect(self.on_filter_selected)
        self.filterView.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for i in range(1, 4):
            getattr(self, f"action_load_preset_{i}").triggered.connect(self.load_preset(i))
            getattr(self, f"action_clear_preset_{i}").triggered.connect(self.clear_preset(i))
            getattr(self, f"action_store_preset_{i}").triggered.connect(self.set_preset(i))
            self.enable_preset(i)
        # init the signal view selector
        self.showSignals.blockSignals(True)
        for x in SHOW_SIGNAL_OPTIONS:
            self.showSignals.addItem(x)
        selected = self.preferences.get(DISPLAY_SHOW_SIGNALS)
        selected_idx = self.showSignals.findText(selected)
        if selected_idx != -1:
            self.showSignals.setCurrentIndex(selected_idx)
        else:
            logger.info(f"Ignoring unknown cached preference for {DISPLAY_SHOW_SIGNALS} - {selected}")
        self.showSignals.blockSignals(False)
        # init the filtered signal view selecter
        self.showFilteredSignals.blockSignals(True)
        for x in SHOW_FILTERED_SIGNAL_OPTIONS:
            self.showFilteredSignals.addItem(x)
        selected = self.preferences.get(DISPLAY_SHOW_FILTERED_SIGNALS)
        selected_idx = self.showFilteredSignals.findText(selected)
        if selected_idx != -1:
            self.showFilteredSignals.setCurrentIndex(selected_idx)
        else:
            logger.info(f"Ignoring unknown cached preference for {DISPLAY_SHOW_FILTERED_SIGNALS} - {selected}")
        self.showFilteredSignals.blockSignals(False)
        # signal model
        self.__cached_selected_signal = None
        self.__configure_signal_model(parent)
        # magnitude
        self.showLegend.setChecked(bool(self.preferences.get(DISPLAY_SHOW_LEGEND)))
        self.__magnitude_model = MagnitudeModel('main', self.mainChart, self.preferences, self.__signal_model,
                                                'Signals', self.__filter_model, 'Filters',
                                                show_legend=lambda: self.showLegend.isChecked())
        self.__filter_model.filter = self.__default_signal.filter
        # processing
        self.ensurePathContainsExternalTools()
        # extraction
        self.actionExtract_Audio.triggered.connect(self.showExtractAudioDialog)
        self.action_Batch_Extract.triggered.connect(self.showBatchExtractDialog)
        # analysis
        self.actionAnalyse_Audio.triggered.connect(self.showAnalyseAudioDialog)
        # import
        self.actionLoad_Filter.triggered.connect(self.importFilter)
        self.actionLoad_Signal.triggered.connect(self.importSignal)
        self.action_Load_Project.triggered.connect(self.importProject)
        # export
        self.actionSave_Report.triggered.connect(self.exportReport)
        self.actionSave_Chart.triggered.connect(self.exportChart)
        self.actionExport_Biquad.triggered.connect(self.exportBiquads)
        self.actionSave_Filter.triggered.connect(self.exportFilter)
        self.actionExport_FRD.triggered.connect(self.showExportFRDDialog)
        self.actionSave_Signal.triggered.connect(self.showExportSignalDialog)
        self.action_Save_Project.triggered.connect(self.exportProject)
        self.actionAbout.triggered.connect(self.showAbout)

    def __configure_signal_model(self, parent):
        '''
        Wires up the signal model so that
        * only one row can be selected at a time
        * there is always a row selected
        * the filter model is always honouring the selected row correctly (where correctly means the filter model is
        operating on the filter of the selected signal if it is an unlinked signal or a master OR the filter of the
        master of the selected signal if the signal is a slave).
        This last point is crucial as otherwise the linked signals won't update at the same time.
        '''
        self.signalView.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.signalView.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.signalView.setSelectionMode(QAbstractItemView.SingleSelection)
        self.__signal_model = SignalModel(self.signalView, self.__default_signal, self.preferences,
                                          on_update=self.on_signal_change)
        self.__signal_table_model = SignalTableModel(self.__signal_model, parent=parent)
        self.signalView.setModel(self.__signal_table_model)
        self.signalView.selectionModel().selectionChanged.connect(self.on_signal_selected)
        self.signalView.model().dataChanged.connect(self.on_signal_data_change)
        self.signalView.model().modelAboutToBeReset.connect(self.on_signal_data_about_to_reset)
        self.signalView.model().modelReset.connect(self.on_signal_data_reset)
        self.signalView.model().rowsInserted.connect(self.__select_on_inserted_signals)
        self.signalView.model().rowsRemoved.connect(self.__select_on_removed_signals)
        self.signalView.setItemDelegateForColumn(0, RegexValidator('^.+$'))

    def on_signal_data_about_to_reset(self):
        '''
        Caches the selected signal so we can restore it post reset.
        '''
        signal_select = self.signalView.selectionModel()
        if signal_select.hasSelection() and len(signal_select.selectedRows()) == 1:
            self.__cached_selected_signal = self.__signal_model[signal_select.selectedRows()[0].row()]

    def on_signal_data_reset(self):
        '''
        Restores the cached selected signal if we have one.
        '''
        selected_row = -1
        if self.__cached_selected_signal is not None:
            for idx, s in enumerate(self.__signal_model):
                if s.name == self.__cached_selected_signal.name:
                    selected_row = idx
            self.__cached_selected_signal = None
        else:
            signal_count = len(self.__signal_model)
            if signal_count > 0:
                selected_row = signal_count - 1
        if selected_row > -1:
            logger.debug(f"Selected row {selected_row} after signal data reset")
            self.signalView.selectRow(selected_row)

    def on_signal_data_change(self):
        '''
        Delegates to on_signal_change which we can't call directly because of the argument mismatch (dataChanged emits
        the QModelIndexes specifying which data has changed which we don't care about).
        '''
        logger.debug(f"Handling signal data change")
        self.on_signal_selected()
        self.on_signal_change()

    def on_signal_change(self, names=None):
        '''
        Reacts to a change in the signal model by updating the reference and redrawing the chart.
        :param names: the signal names.
        '''
        if names is not None:
            self.update_reference_series(names, self.signalReference, True)
        self.linkSignalButton.setEnabled(len(self.__signal_model) > 1)
        self.__magnitude_model.redraw()

    def __get_selected_signal(self):
        '''
        :return: the selected signal (as selected in the table going back to its master if the selected signal is a
        slave) or the default signal if nothing is selected.
        '''
        signal_select = self.signalView.selectionModel()
        if signal_select.hasSelection() and len(signal_select.selectedRows()) == 1:
            signal = self.__signal_model[signal_select.selectedRows()[0].row()]
            if signal.master is not None:
                signal = signal.master
        else:
            signal = self.__signal_model.default_signal
        logger.debug(f"Selected signal is {signal.name}")
        return signal

    def __select_on_inserted_signals(self, idx, first, last):
        '''
        Selects the last signal added by default.
        :param idx: the idx.
        :param first: the first row inserted.
        :param last: the last row inserted.
        '''
        logger.debug(f"Selecting signal {last} on insert")
        self.signalView.selectRow(last)

    def __select_on_removed_signals(self, idx, first, last):
        '''
        Ensures the correct signal is selected when signals are removed. This defaults to the last signal in the table
        if there are any signals otherwise it reverts to the default signal.
        :param idx: the idx.
        :param first: first row removed.
        :param last: last row removed.
        '''
        signal_count = len(self.__signal_model)
        if signal_count > 0:
            logger.debug(f"Selecting signal {signal_count-1} on remove")
            self.signalView.selectRow(signal_count - 1)
        else:
            logger.debug(f"No signals in model, selecting default filter on remove")
            self.on_signal_selected()

    def addSignal(self):
        '''
        Adds signals via the signal dialog.
        '''
        SignalDialog(self.preferences, self.__signal_model, parent=self).show()

    def linkSignals(self):
        '''
        Lets the user link signals via a matrix mapping.
        '''
        LinkSignalsDialog(self.__signal_model, parent=self).exec()

    def deleteSignal(self):
        '''
        Deletes the currently selected signals.
        '''
        selection = self.signalView.selectionModel()
        if selection.hasSelection():
            self.__signal_model.delete([x.row() for x in selection.selectedRows()])
        if len(self.__signal_model) > 0:
            self.signalView.selectRow(0)
        else:
            self.signalView.clearSelection()
            # nothing in qt appears to emit selectionChanged when you clear the selection so have to call it ourselves
            self.on_signal_selected()

    def importSignal(self):
        '''
        Allows the user to load a signal from a saved file.
        '''

        def parser(file_name):
            with gzip.open(file_name, 'r') as infile:
                return json.loads(infile.read().decode('utf-8'))

        input = self.__load('*.signal', 'Load Signal', parser)
        if input is not None:
            from model.codec import signaldata_from_json
            self.__signal_model.add(signaldata_from_json(input))

    def on_signal_selected(self):
        '''
        Enables the edit & delete button if there are selected rows.
        '''
        selection = self.signalView.selectionModel()
        self.deleteSignalButton.setEnabled(selection.hasSelection())
        if len(selection.selectedRows()) == 1:
            self.__filter_model.filter = self.__get_selected_signal().filter
        else:
            if len(self.__filter_model.filter) > 0:
                self.__default_signal.filter = self.__filter_model.filter
            # we have to set the filter on the default signal and then set that back onto the filter model to ensure
            # the right links are established re listeners and label names
            self.__filter_model.filter = self.__default_signal.filter

    def on_filter_change(self, names):
        '''
        Reacts to a change in the filter model by updating the reference, redrawing the chart and other filter related
        things.
        :param names: the signal names.
        '''
        self.update_reference_series(names, self.filterReference, False)
        self.__magnitude_model.redraw()
        self.__enable_save_filter()
        self.__check_active_preset(self.__filter_model.filter.preset_idx)

    def importFilter(self):
        '''
        Allows the user to replace the current filter with one loaded from a file.
        '''
        filter_json = self.__load_filter()
        if filter_json is not None:
            self.__apply_filter(filter_json)
            self.__magnitude_model.redraw()

    def __apply_filter(self, filter_json):
        from model.codec import filter_from_json
        signal = self.__get_selected_signal()
        filt = filter_from_json(filter_json)
        signal.filter = filt
        self.__filter_model.filter = filt

    def __load_filter(self):
        '''
        Presents a file dialog to the user so they can choose a filter to load.
        :return: the loaded filter, if any.
        '''
        dialog = QFileDialog(parent=self)
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setNameFilter(f"*.filter")
        dialog.setWindowTitle(f"Load Filter")
        if dialog.exec():
            selected = dialog.selectedFiles()
            if len(selected) > 0:
                with open(selected[0], 'r') as infile:
                    input = json.load(infile)
                    self.statusbar.showMessage(f"Loaded filter from {infile.name}")
                    return input
        return None

    def addFilter(self):
        '''
        Adds a filter via the filter dialog.
        '''
        FilterDialog(self.preferences, self.__get_selected_signal(), self.__filter_model, parent=self).show()

    def editFilter(self):
        '''
        Edits the currently selected filter via the filter dialog.
        '''
        selection = self.filterView.selectionModel()
        if selection.hasSelection() and len(selection.selectedRows()) == 1:
            signal = self.__get_selected_signal()
            FilterDialog(self.preferences, signal, self.__filter_model,
                         filter=signal.filter[selection.selectedRows()[0].row()], parent=self).show()

    def deleteFilter(self):
        '''
        Deletes the selected filters.
        '''
        selection = self.filterView.selectionModel()
        if selection.hasSelection():
            self.__filter_model.delete([x.row() for x in selection.selectedRows()])

    def __enable_save_filter(self):
        '''
        Enables the save filter if we have filters to save.
        '''
        self.actionSave_Filter.setEnabled(len(self.__filter_model) > 0)

    def on_filter_selected(self):
        '''
        Enables the edit & delete button if there are selected rows.
        '''
        selection = self.filterView.selectionModel()
        self.deleteFilterButton.setEnabled(selection.hasSelection())
        self.editFilterButton.setEnabled(len(selection.selectedRows()) == 1)

    def exportFilter(self):
        '''
        Allows the user to save the current filter to a file.
        '''
        dialog = QFileDialog(parent=self)
        dialog.setFileMode(QFileDialog.AnyFile)
        dialog.setNameFilter(f"*.filter")
        dialog.setWindowTitle(f"Save Filter")
        dialog.setLabelText(QFileDialog.Accept, 'Save')
        if dialog.exec():
            selected = dialog.selectedFiles()
            if len(selected) > 0:
                if not selected[0].endswith('.filter'):
                    selected[0] += '.filter'
                with open(selected[0], 'w+') as outfile:
                    json.dump(self.__filter_model.filter.to_json(), outfile)
                    self.statusbar.showMessage(f"Saved filter to {outfile.name}")

    def exportChart(self):
        '''
        Saves the currently selected chart to a file.
        '''
        dialog = SaveChartDialog(self, 'beq', self.mainChart.canvas.figure, self.statusbar)
        dialog.exec()

    def exportReport(self):
        '''
        Saves the currently selected chart to a file.
        '''
        dialog = SaveReportDialog(self, self.preferences, self.__signal_model, self.__filter_model, self.statusbar)
        dialog.exec()

    def exportBiquads(self):
        '''
        Shows the biquads for the current filter set.
        '''
        dialog = ExportBiquadDialog(self.__filter_model.filter)
        dialog.exec()

    def __load(self, filter, title, parser):
        '''
        Presents a file dialog to the user so they can choose something to load.
        :return: the loaded thing, if any.
        '''
        input = None
        dialog = QFileDialog(parent=self)
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setNameFilter(filter)
        dialog.setWindowTitle(title)
        if dialog.exec():
            selected = dialog.selectedFiles()
            if len(selected) > 0:
                input = parser(selected[0])
                if input is not None:
                    self.statusbar.showMessage(f"Loaded {selected[0]}")
        return input

    def ensurePathContainsExternalTools(self):
        '''
        Ensures that all external tool paths are on the path.
        '''
        path = os.environ.get('PATH', [])
        paths = path.split(os.pathsep)
        locs = self.preferences.get_all(BINARIES_GROUP)
        if len(locs) > 0:
            logging.info(f"Adding {locs} to PATH")
            os.environ['PATH'] = os.pathsep.join([l for l in locs if l not in paths]) + os.pathsep + path
        else:
            logger.warning(f"No {BINARIES_GROUP} paths set")
            # TODO attempt to call each binary with a test command to test if they are really on the path

    def setupUi(self, main_window):
        super().setupUi(self)
        geometry = self.preferences.get(SCREEN_GEOMETRY)
        if geometry is not None:
            self.restoreGeometry(geometry)
        else:
            screen_geometry = self.app.desktop().availableGeometry()
            if screen_geometry.height() < 800:
                self.showMaximized()
        window_state = self.preferences.get(SCREEN_WINDOW_STATE)
        if window_state is not None:
            self.restoreState(window_state)

    def closeEvent(self, *args, **kwargs):
        '''
        Saves the window state on close.
        :param args:
        :param kwargs:
        '''
        self.preferences.set(SCREEN_GEOMETRY, self.saveGeometry())
        self.preferences.set(SCREEN_WINDOW_STATE, self.saveState())
        super().closeEvent(*args, **kwargs)
        self.app.closeAllWindows()

    def showPreferences(self):
        '''
        Shows the preferences dialog.
        '''
        PreferencesDialog(self.preferences, self.__style_path_root, parent=self).exec()

    def update_reference_series(self, names, combo, primary=True):
        '''
        Updates the reference series dropdown with the current curve names.
        '''
        current_reference = combo.currentText()
        try:
            combo.blockSignals(True)
            combo.clear()
            combo.addItem('None')
            for name in names:
                combo.addItem(name)
            idx = combo.findText(current_reference)
            if idx != -1:
                combo.setCurrentIndex(idx)
            else:
                self.__magnitude_model.normalise(primary=primary)
        finally:
            combo.blockSignals(False)

    def showAbout(self):
        msg_box = QMessageBox()
        msg_box.setText(
            f"<a href='https://github.com/3ll3d00d/beqdesigner'>BEQ Designer</a> v{self.__version} by 3ll3d00d")
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle('About')
        msg_box.exec()

    def showExtractAudioDialog(self):
        '''
        Show the extract audio dialog.
        '''
        ExtractAudioDialog(self, self.preferences, self.__signal_model).show()

    def showBatchExtractDialog(self):
        '''
        Show the batch extract dialog.
        '''
        BatchExtractDialog(self, self.preferences).show()

    def showAnalyseAudioDialog(self):
        '''
        Show the analyse audio dialog.
        '''
        AnalyseSignalDialog(self.preferences).exec()

    def showExportFRDDialog(self):
        '''
        Shows the export frd dialog.
        '''
        ExportSignalDialog(self.preferences, self.__signal_model, self, self.statusbar).exec()

    def showExportSignalDialog(self):
        '''
        Shows the export signal dialog.
        '''
        ExportSignalDialog(self.preferences, self.__signal_model, self, self.statusbar, mode=Mode.SIGNAL).exec()

    def exportProject(self):
        '''
        Exports the project to a file.
        '''
        file_name = QFileDialog(self).getSaveFileName(self, 'Export Project', f"project.beq",
                                                      "BEQ Project (*.beq)")
        file_name = str(file_name[0]).strip()
        if len(file_name) > 0:
            output = self.__signal_model.to_json()
            if not file_name.endswith('.beq'):
                file_name += '.beq'
            with gzip.open(file_name, 'wb+') as outfile:
                outfile.write(json.dumps(output).encode('utf-8'))
            self.statusbar.showMessage(f"Saved project to {file_name}")

    def importProject(self):
        '''
        Allows the user to load a fresh project.
        '''

        def parser(file_name):
            with gzip.open(file_name, 'r') as infile:
                return json.loads(infile.read().decode('utf-8'))

        input = self.__load('*.beq', 'Load Project', parser)
        if input is not None:
            from model.codec import signalmodel_from_json
            self.__signal_model.replace(signalmodel_from_json(input))
            self.__magnitude_model.redraw()

    def normaliseSignalMagnitude(self):
        '''
        Handles reference series change.
        '''
        if self.signalReference.currentText() == 'None':
            self.__magnitude_model.normalise(primary=True)
        else:
            self.__magnitude_model.normalise(primary=True, curve=self.signalReference.currentText())

    def normaliseFilterMagnitude(self):
        '''
        Handles reference series change.
        '''
        if self.filterReference.currentText() == 'None':
            self.__magnitude_model.normalise(primary=False)
        else:
            self.__magnitude_model.normalise(primary=False, curve=self.filterReference.currentText())

    def showLimits(self):
        '''
        Shows the limits dialog for the main chart.
        '''
        self.__magnitude_model.show_limits()

    def showValues(self):
        '''
        Shows the values dialog for the main chart.
        '''
        self.__magnitude_model.show_values()

    def changeFilterVisibility(self, selected_filters):
        '''
        Changes which filters are visible on screen.
        '''
        self.preferences.set(DISPLAY_SHOW_FILTERS, selected_filters)
        self.__filter_model.post_update()
        self.__magnitude_model.redraw()

    def changeSignalVisibility(self, selected_signals):
        '''
        Changes which signals are visible on screen.
        '''
        self.preferences.set(DISPLAY_SHOW_SIGNALS, selected_signals)
        self.__signal_model.post_update()
        self.__magnitude_model.redraw()

    def changeSignalFilterVisibility(self, selected):
        '''
        Changes which filtered signals are visible on screen.
        '''
        self.preferences.set(DISPLAY_SHOW_FILTERED_SIGNALS, selected)
        self.__signal_model.post_update()
        self.__magnitude_model.redraw()

    def changeLegendVisibility(self):
        '''
        Changes whether the legend is visible.
        '''
        self.preferences.set(DISPLAY_SHOW_LEGEND, self.showLegend.isChecked())
        self.__magnitude_model.redraw()

    def applyPreset1(self):
        '''
        Applies the preset to the model.
        '''
        self.__apply_preset(1)

    def applyPreset2(self):
        '''
        Applies the preset to the model.
        '''
        self.__apply_preset(2)

    def applyPreset3(self):
        '''
        Applies the preset to the model.
        '''
        self.__apply_preset(3)

    def __apply_preset(self, idx):
        preset_key = FILTERS_PRESET_x % idx
        preset = self.preferences.get(preset_key)
        if preset is not None:
            self.__apply_filter(preset)
            self.__filter_model.filter.preset_idx = idx
            self.__check_active_preset(idx)

    def __check_active_preset(self, preset_idx):
        pattern = re.compile("^preset[0-9]Button$")
        for attr in [at for at in dir(self) if pattern.match(at)]:
            getattr(self, attr).setIcon(QIcon())
        if preset_idx > 0:
            if hasattr(self, f"preset{preset_idx}Button"):
                getattr(self, f"preset{preset_idx}Button").setIcon(qta.icon('fa.check'))
            else:
                logger.warning(f"Ignoring attempt to activate an unknown preset {preset_idx}")

    def enable_preset(self, idx):
        preset_key = FILTERS_PRESET_x % idx
        getattr(self, f"preset{idx}Button").setEnabled(self.preferences.has(preset_key))

    def load_preset(self, idx):
        '''
        Allows the user to load a preset from a saved filter.
        :param idx: the index.
        '''

        def __load_preset():
            preset_key = FILTERS_PRESET_x % idx
            input = self.__load_filter()
            if input is not None:
                logger.info(f"Loaded filter for preset {idx}")
                self.preferences.set(preset_key, input)
            self.enable_preset(idx)

        return __load_preset

    def set_preset(self, idx):
        '''
        Saves the current filter to the numbered preset.
        :param idx: the index.
        '''

        def __set_preset():
            preset_key = FILTERS_PRESET_x % idx
            if len(self.__filter_model) > 0:
                input = self.__filter_model.filter.to_json()
                self.preferences.set(preset_key, input)
                self.enable_preset(idx)

        return __set_preset

    def clear_preset(self, idx):
        '''
        Yields a function which will clears the specified preset.
        :param idx: the preset index.
        '''
        prefs = self.preferences
        button = getattr(self, f"preset{idx}Button")

        def __clear_preset():
            prefs.set(FILTERS_PRESET_x % idx, None)
            button.setEnabled(False)

        return __clear_preset


class SaveChartDialog(QDialog, Ui_saveChartDialog):
    '''
    Save Chart dialog
    '''

    def __init__(self, parent, name, figure, statusbar):
        super(SaveChartDialog, self).__init__(parent)
        self.setupUi(self)
        self.name = name
        self.figure = figure
        self.__dpi = self.figure.dpi
        self.__x, self.__y = self.figure.get_size_inches() * self.figure.dpi
        self.__aspectRatio = self.__x / self.__y
        self.widthPixels.setValue(self.__x)
        self.heightPixels.setValue(self.__y)
        self.statusbar = statusbar
        self.__dialog = QFileDialog(parent=self)

    def accept(self):
        formats = "Portable Network Graphic (*.png)"
        fileName = self.__dialog.getSaveFileName(self, 'Export Chart', f"{self.name}.png", formats)
        if fileName:
            outputFile = str(fileName[0]).strip()
            if len(outputFile) == 0:
                return
            else:
                scaleFactor = self.widthPixels.value() / self.__x
                self.figure.savefig(outputFile, format='png', dpi=self.__dpi * scaleFactor)
                self.statusbar.showMessage(f"Saved {self.name} to {outputFile}", 5000)
        QDialog.accept(self)

    def set_height(self, newWidth):
        '''
        Updates the height as the width changes according to the aspect ratio.
        :param newWidth: the new width.
        '''
        self.heightPixels.setValue(int(math.floor(newWidth / self.__aspectRatio)))


class ExportBiquadDialog(QDialog, Ui_exportBiquadDialog):
    '''
    Export Biquads Dialog
    '''

    def __init__(self, filter):
        super(ExportBiquadDialog, self).__init__()
        self.setupUi(self)
        self.__filter = filter
        self.updateBiquads()

    def updateBiquads(self):
        if self.__filter is not None and len(self.__filter) > 0:
            self.__filter = self.__filter.resample(int(self.fs.currentText()))
            biquads = list(flatten([self.__filter.format_biquads(self.minidspFormat.isChecked())]))
            if len(biquads) < self.maxBiquads.value():
                passthrough = [Passthrough()] * (self.maxBiquads.value() - len(biquads))
                biquads += list(flatten([x.format_biquads(self.minidspFormat.isChecked()) for x in passthrough]))
            text = "\n".join([f"biquad{idx},\n{bq}" for idx, bq in enumerate(biquads)])
            self.biquads.setPlainText(text)


def flatten(l):
    '''
    flatten an irregularly shaped list of lists (of lists of lists...)
    solution from https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists
    '''
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def make_app():
    app = QApplication(sys.argv)
    if getattr(sys, 'frozen', False):
        icon_path = os.path.join(sys._MEIPASS, 'Icon.ico')
    else:
        icon_path = os.path.abspath(os.path.join(os.path.dirname('__file__'), '../icons/Icon.ico'))
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    return app


if __name__ == '__main__':
    app = make_app()
    form = BeqDesigner(app)
    # setup the error handler
    e_dialog = QErrorMessage(form)
    e_dialog.setWindowModality(QtCore.Qt.WindowModal)
    font = QFont()
    font.setFamily("Consolas")
    font.setPointSize(8)
    e_dialog.setFont(font)
    # add the exception handler so we can see the errors in a QErrorMessage
    sys._excepthook = sys.excepthook


    def dump_exception_to_log(exctype, value, tb):
        import traceback
        print(exctype, value, tb)
        global e_dialog
        if e_dialog is not None:
            formatted = traceback.format_exception(etype=exctype, value=value, tb=tb)
            e_dialog.setWindowTitle('Unexpected Error')
            url = 'https://github.com/3ll3d00d/beqdesigner/issues/new'
            msg = f"Unexpected Error detected, go to {url} to log the issue<p>{'<br>'.join(formatted)}"
            e_dialog.showMessage(msg)
            e_dialog.resize(1200, 400)


    sys.excepthook = dump_exception_to_log

    # show the form and exec the app
    form.show()
    app.exec_()
