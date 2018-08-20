import logging
import math
import os
import sys
from contextlib import contextmanager
from uuid import uuid4

import matplotlib
from qtpy.QtCore import QSettings
from qtpy.QtGui import QIcon, QFont, QCursor
from qtpy.QtWidgets import QMainWindow, QApplication, QErrorMessage, QAbstractItemView, QDialog, QDialogButtonBox, \
    QFileDialog

from model.filter import FilterTableModel, FilterModel
from model.iir import LowShelf, HighShelf, PeakingEQ, ComplexLowPass, \
    FilterType, ComplexHighPass, SecondOrder_HighPass, SecondOrder_LowPass
from model.log import RollingLogger
from model.magnitude import MagnitudeModel
from model.signal import SignalModel, SignalTableModel, ExtractAudioDialog
from ui.beq import Ui_MainWindow
from ui.filter import Ui_editFilterDialog
from ui.preferences import Ui_preferencesDialog

matplotlib.use("Qt5Agg")

from qtpy import QtCore, QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
import colorcet as cc

# from http://colorcet.pyviz.org/index.html
inverse = {}
for k, v in cc.cm_n.items():
    if not k[-2:] == "_r":
        inverse[v] = inverse.get(v, [])
        inverse[v].insert(0, k)
all_cms = sorted({',  '.join(reversed(v)): k for (k, v) in inverse.items()}.items())
cms_by_name = dict(all_cms)

logger = logging.getLogger('beq')


# Matplotlib canvas class to create figure
class MplCanvas(Canvas):
    def __init__(self):
        self.figure = Figure(tight_layout=True)
        Canvas.__init__(self, self.figure)
        Canvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        Canvas.updateGeometry(self)


# Matplotlib widget
class MplWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.canvas = MplCanvas()
        self.vbl = QtWidgets.QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)
        self._cmap = self.getColourMap('rainbow')

    def getColourMap(self, name):
        return cms_by_name.get(name, cms_by_name.get('bgyw'))

    def getColour(self, idx, count):
        '''
        :param idx: the colour index.
        :return: the colour at that index.
        '''
        return self._cmap(idx / count)


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
        # logs
        self.logViewer = RollingLogger(parent=self)
        self.actionShow_Logs.triggered.connect(self.logViewer.show_logs)
        self.actionPreferences.triggered.connect(self.showPreferences)
        # init the filter view/model
        self.filterView.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.__filterModel = FilterModel(self.filterView)
        self.__filterTableModel = FilterTableModel(self.__filterModel, parent=parent)
        self.filterView.setModel(self.__filterTableModel)
        self.filterView.selectionModel().selectionChanged.connect(self.changeFilterButtonState)
        # magnitude
        self.__magnitudeModel = MagnitudeModel(self.filterChart, self.__filterModel)
        # signal model
        self.signalView.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.signalView.setSelectionMode(QAbstractItemView.SingleSelection)
        self.__signalModel = SignalModel(self.signalView)
        self.__signalTableModel = SignalTableModel(self.__signalModel, parent=parent)
        self.signalView.setModel(self.__signalTableModel)
        self.signalView.selectionModel().selectionChanged.connect(self.changeSignalButtonState)
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
        locs = set([self.settings.value(f"binaries/{x}") for x in ['ffmpeg', 'ffprobe', 'sox']])
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
        pass
        # ExtractAudioDialog(self.settings, parent=self).exec()

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
        self.__magnitudeModel.display(self.showIndividualFilters.isChecked())

    def changeVisibilityOfIndividualFilters(self):
        '''
        Adds or removes the individual filter transfer functions to/from the graph.
        '''
        if self.showIndividualFilters.isChecked():
            self.__magnitudeModel.display(True)
        else:
            self.__magnitudeModel.removeIndividualFilters()

    def showExtractAudioDialog(self):
        ExtractAudioDialog(self.settings, parent=self).exec()


class PreferencesDialog(QDialog, Ui_preferencesDialog):
    '''
    Allows user to set some basic preferences.
    '''

    def __init__(self, settings, parent=None):
        super(PreferencesDialog, self).__init__(parent)
        self.setupUi(self)
        self.settings = settings

        soxLoc = self.settings.value('binaries/sox')
        if soxLoc:
            if os.path.isdir(soxLoc):
                self.soxDirectory.setText(soxLoc)

        ffmpegLoc = self.settings.value('binaries/ffmpeg')
        if ffmpegLoc:
            if os.path.isdir(ffmpegLoc):
                self.ffmpegDirectory.setText(ffmpegLoc)

        ffprobeLoc = self.settings.value('binaries/ffprobe')
        if ffprobeLoc:
            if os.path.isdir(ffprobeLoc):
                self.ffprobeDirectory.setText(ffprobeLoc)

        targetFs = self.settings.value('analysis/target_fs')
        if targetFs:
            self.targetFs.setValue(targetFs)
        else:
            self.settings.setValue('analysis/target_fs', self.targetFs.value())

        outputDir = self.settings.value('extraction/output_dir')
        if outputDir:
            if os.path.isdir(outputDir):
                self.defaultOutputDirectory.setText(outputDir)

    def accept(self):
        '''
        Saves the locations if they exist.
        '''
        soxLoc = self.soxDirectory.text()
        if os.path.isdir(soxLoc):
            self.settings.setValue('binaries/sox', soxLoc)
        ffmpegLoc = self.ffmpegDirectory.text()
        if os.path.isdir(ffmpegLoc):
            self.settings.setValue('binaries/ffmpeg', ffmpegLoc)
        ffprobeLoc = self.ffprobeDirectory.text()
        if os.path.isdir(ffprobeLoc):
            self.settings.setValue('binaries/ffprobe', ffprobeLoc)
        outputDir = self.defaultOutputDirectory.text()
        if os.path.isdir(outputDir):
            self.settings.setValue('extraction/output_dir', outputDir)
        self.settings.setValue('analysis/target_fs', self.targetFs.value())
        QDialog.accept(self)

    def __get_directory(self, name):
        dialog = QFileDialog(parent=self)
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setNameFilter(f"{name}.exe")
        dialog.setWindowTitle(f"Select {name}.exe")
        if dialog.exec():
            selected = dialog.selectedFiles()
            if len(selected) > 0:
                return selected[0]
        return None

    def showFfmpegDirectoryPicker(self):
        loc = self.__get_directory('ffmpeg')
        if loc is not None:
            dirname = os.path.dirname(loc)
            self.ffmpegDirectory.setText(dirname)
            if os.path.exists(os.path.join(dirname, 'ffprobe.exe')):
                self.ffprobeDirectory.setText(dirname)

    def showFfprobeDirectoryPicker(self):
        loc = self.__get_directory('ffprobe')
        if loc is not None:
            dirname = os.path.dirname(loc)
            self.ffprobeDirectory.setText(dirname)
            if os.path.exists(os.path.join(dirname, 'ffmpeg.exe')):
                self.ffmpegDirectory.setText(dirname)

    def showSoxDirectoryPicker(self):
        loc = self.__get_directory('sox')
        if loc is not None:
            self.soxDirectory.setText(os.path.dirname(loc))

    def showDefaultOutputDirectoryPicker(self):
        dialog = QFileDialog(parent=self)
        dialog.setFileMode(QFileDialog.DirectoryOnly)
        dialog.setWindowTitle(f"Select Extract Audio Output Directory")
        if dialog.exec():
            selected = dialog.selectedFiles()
            if len(selected) > 0:
                self.defaultOutputDirectory.setText(selected[0])


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
