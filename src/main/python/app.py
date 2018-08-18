import logging
import math
import os
import sys
from contextlib import contextmanager

import matplotlib
from qtpy.QtCore import QSettings
from qtpy.QtGui import QIcon, QFont, QCursor
from qtpy.QtWidgets import QMainWindow, QApplication, QErrorMessage, QAbstractItemView, QDialog, QDialogButtonBox

from model.filter import FilterTableModel, FilterModel
from model.iir import LowShelf, HighShelf, PeakingEQ, ComplexLowPass, \
    FilterType, ComplexHighPass, SecondOrder_HighPass, SecondOrder_LowPass
from model.log import RollingLogger
from model.magnitude import MagnitudeModel
from ui.beq import Ui_MainWindow
from ui.filter import Ui_editFilterDialog

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
        self.filterView.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.logViewer = RollingLogger(parent=self)
        self.__filterModel = FilterModel(self.filterView)
        self.__filterTableModel = FilterTableModel(self.__filterModel, parent=parent)
        self.filterView.setModel(self.__filterTableModel)
        self.filterView.selectionModel().selectionChanged.connect(self.changeButtonState)
        self.__magnitudeModel = MagnitudeModel(self.filterChart, self.__filterModel)

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

    def addFilter(self):
        '''
        Adds a filter via the filter dialog.
        '''
        FilterDialog(self.__filterModel, parent=self).exec()

    def editFilter(self):
        '''
        Edits the currently selected filter via the filter dialog.
        '''
        FilterDialog(self.__filterModel, filter=None, parent=self).exec()

    def deleteFilter(self):
        '''
        Deletes the selected filters.
        '''
        selection = self.filterView.selectionModel()
        if selection.hasSelection():
            self.__filterModel.delete([x.row() for x in selection.selectedRows()])

    def changeButtonState(self):
        '''
        Enables the delete button if there are selected rows.
        '''
        selection = self.filterView.selectionModel()
        self.deleteSelected.setEnabled(selection.hasSelection())
        self.editFilterButton.setEnabled(len(selection.selectedRows()) == 1)

    def display(self):
        '''
        Updates the chart.
        '''
        self.__magnitudeModel.display(self.showIndividualFilters.isChecked())

    def changeVisibilityOfIndividualFilters(self):
        '''
        Adds
        :return:
        '''
        if self.showIndividualFilters.isChecked():
            self.__magnitudeModel.display(True)
        else:
            self.__magnitudeModel.removeIndividualFilters()


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
        else:
            self.buttonBox.button(QDialogButtonBox.Save).setText('Add')
        self.enableOkIfGainIsValid()

    def accept(self):
        if self.filter is None:
            if self.__is_pass_filter():
                filt = self.create_pass_filter()
            else:
                filt = self.create_shaping_filter()
            self.filterModel.add(filt)
        else:
            pass

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
        elif self.filterType.currentText() == 'Low Pass':
            return SecondOrder_LowPass(self.fs, self.freq.value(), self.filterQ.value())
        elif self.filterType.currentText() == 'High Pass':
            return SecondOrder_HighPass(self.fs, self.freq.value(), self.filterQ.value())
        else:
            raise ValueError(f"Unknown filter type {self.filterType.currentText()}")

    def create_pass_filter(self):
        '''
        Creates a predefined high or low pass filter.
        :return: the filter.
        '''
        if self.filterType.currentText() == 'Low Pass':
            return ComplexLowPass(FilterType[self.passFilterType.currentText().upper()], self.filterOrder.value(),
                                  self.fs, self.freq.value())
        else:
            return ComplexHighPass(FilterType[self.passFilterType.currentText().upper()], self.filterOrder.value(),
                                   self.fs, self.freq.value())

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
            self.filterGain.setEnabled(True)
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
