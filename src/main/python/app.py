import logging
import os
import sys
import qtawesome as qta
from contextlib import contextmanager

import matplotlib

matplotlib.use("Qt5Agg")

from qtpy.QtCore import QSettings
from qtpy.QtGui import QIcon, QFont, QCursor
from qtpy.QtWidgets import QMainWindow, QApplication, QErrorMessage, QAbstractItemView

from model.extract import ExtractAudioDialog
from model.filter import FilterTableModel, FilterModel, FilterDialog
from model.log import RollingLogger
from model.magnitude import MagnitudeModel, LimitsDialog
from model.preferences import PreferencesDialog, BINARIES
from model.signal import SignalModel, SignalTableModel, SignalDialog
from ui.beq import Ui_MainWindow

from qtpy import QtCore

logger = logging.getLogger('beq')


class BeqDesigner(QMainWindow, Ui_MainWindow):
    '''
    The main UI.
    '''

    def __init__(self, app, parent=None):
        super(BeqDesigner, self).__init__(parent)
        self.logger = logging.getLogger('beqdesigner')
        self.app = app
        self.settings = QSettings("3ll3d00d", "beqdesigner")
        self.setupUi(self)
        self.limitsButton.setIcon(qta.icon('ei.move'))
        # logs
        self.logViewer = RollingLogger(parent=self)
        self.actionShow_Logs.triggered.connect(self.logViewer.show_logs)
        self.actionPreferences.triggered.connect(self.showPreferences)
        # init the filter view/model
        self.filterView.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.__filterModel = FilterModel(self.filterView, self.showIndividualFilters)
        self.__filterTableModel = FilterTableModel(self.__filterModel, parent=parent)
        self.filterView.setModel(self.__filterTableModel)
        self.filterView.selectionModel().selectionChanged.connect(self.changeFilterButtonState)
        # signal model
        self.signalView.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.signalView.setSelectionMode(QAbstractItemView.SingleSelection)
        self.__signalModel = SignalModel(self.signalView, self.__filterModel)
        self.__signalTableModel = SignalTableModel(self.__signalModel, parent=parent)
        self.signalView.setModel(self.__signalTableModel)
        self.signalView.selectionModel().selectionChanged.connect(self.changeSignalButtonState)
        # magnitude
        self.__magnitudeModel = MagnitudeModel(self.filterChart, self.__signalModel)
        # processing
        self.ensurePathContainsExternalTools()
        # extraction
        self.actionExtract_Audio.triggered.connect(self.showExtractAudioDialog)

    def ensurePathContainsExternalTools(self):
        '''
        Ensures that all external tool paths are on the path.
        '''
        path = os.environ.get('PATH', [])
        paths = path.split(os.pathsep)
        locs = set([self.settings.value(f"binaries/{x}") for x in BINARIES])
        logging.info(f"Adding {locs} to PATH")
        os.environ['PATH'] = os.pathsep.join([l for l in locs if l not in paths]) + os.pathsep + path

    def setupUi(self, mainWindow):
        super().setupUi(self)
        geometry = self.settings.value("geometry")
        if not geometry == None:
            self.restoreGeometry(geometry)
        else:
            screenGeometry = self.app.desktop().availableGeometry()
            if screenGeometry.height() < 800:
                self.showMaximized()
        windowState = self.settings.value("windowState")
        if not windowState == None:
            self.restoreState(windowState)

    def closeEvent(self, *args, **kwargs):
        '''
        Saves the window state on close.
        :param args:
        :param kwargs:
        '''
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        super().closeEvent(*args, **kwargs)
        self.app.closeAllWindows()

    def showPreferences(self):
        '''
        Shows the preferences dialog.
        '''
        PreferencesDialog(self.settings, parent=self).exec()

    def addFilter(self):
        '''
        Adds a filter via the filter dialog.
        '''
        FilterDialog(self.__filterModel).exec()

    def editFilter(self):
        '''
        Edits the currently selected filter via the filter dialog.
        '''
        selection = self.filterView.selectionModel()
        if selection.hasSelection() and len(selection.selectedRows()) == 1:
            FilterDialog(self.__filterModel, filter=self.__filterModel[selection.selectedRows()[0].row()]).exec()

    def deleteFilter(self):
        '''
        Deletes the selected filters.
        '''
        selection = self.filterView.selectionModel()
        if selection.hasSelection():
            self.__filterModel.delete([x.row() for x in selection.selectedRows()])

    def addSignal(self):
        '''
        Adds signals via the signal dialog.
        '''
        SignalDialog(self.settings, self.__signalModel, parent=self).exec()

    def editSignal(self):
        '''
        Edits the currently selected signal via the signal dialog.
        '''
        pass

    def deleteSignal(self):
        '''
        Deletes the currently selected signals.
        '''
        selection = self.signalView.selectionModel()
        if selection.hasSelection():
            self.__signalModel.delete([x.row() for x in selection.selectedRows()])

    def changeFilterButtonState(self):
        '''
        Enables the edit & delete button if there are selected rows.
        '''
        selection = self.filterView.selectionModel()
        self.deleteFilterButton.setEnabled(selection.hasSelection())
        self.editFilterButton.setEnabled(len(selection.selectedRows()) == 1)

    def changeSignalButtonState(self):
        '''
        Enables the edit & delete button if there are selected rows.
        '''
        selection = self.signalView.selectionModel()
        self.deleteSignalButton.setEnabled(selection.hasSelection())
        self.editSignalButton.setEnabled(len(selection.selectedRows()) == 1)

    def display(self):
        '''
        Updates the chart.
        '''
        self.__magnitudeModel.display()

    def changeVisibilityOfIndividualFilters(self):
        '''
        Adds or removes the individual filter transfer functions to/from the graph.
        '''
        self.__magnitudeModel.display()

    def showExtractAudioDialog(self):
        '''
        Show the extract audio dialog.
        '''
        ExtractAudioDialog(self.settings, parent=self).exec()

    def normaliseMagnitude(self):
        '''
        Handles reference series change.
        '''
        pass

    def showLimits(self):
        '''
        Shows the limits dialog for the main chart.
        '''
        LimitsDialog(self.__magnitudeModel).exec()


e_dialog = None


def main():
    app = QApplication(sys.argv)
    if getattr(sys, 'frozen', False):
        iconPath = os.path.join(sys._MEIPASS, 'Icon.ico')
    else:
        iconPath = os.path.abspath(os.path.join(os.path.dirname('__file__'), '../icons/Icon.ico'))
    if os.path.exists(iconPath):
        app.setWindowIcon(QIcon(iconPath))
    form = BeqDesigner(app)
    # setup the error handler
    global e_dialog
    e_dialog = QErrorMessage(form)
    e_dialog.setWindowModality(QtCore.Qt.WindowModal)
    font = QFont()
    font.setFamily("Consolas")
    font.setPointSize(8)
    e_dialog.setFont(font)
    form.show()
    app.exec_()


# display exceptions in a QErrorMessage so the user knows what just happened
sys._excepthook = sys.excepthook


def dump_exception_to_log(exctype, value, tb):
    import traceback
    if e_dialog is not None:
        formatted = traceback.format_exception(etype=exctype, value=value, tb=tb)
        msg = '<br>'.join(formatted)
        e_dialog.setWindowTitle('Unexpected Error')
        e_dialog.showMessage(msg)
        e_dialog.resize(1200, 400)
    else:
        print(exctype, value, tb)


sys.excepthook = dump_exception_to_log

if __name__ == '__main__':
    main()


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
