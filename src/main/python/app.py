import logging
import os
import sys
from contextlib import contextmanager

import matplotlib
from qtpy.QtCore import QSettings
from qtpy.QtGui import QIcon, QFont, QCursor
from qtpy.QtWidgets import QMainWindow, QApplication, QErrorMessage, QAbstractItemView

from model.filter import FilterTableModel, FilterModel
from model.log import RollingLogger
from model.magnitude import MagnitudeModel
from ui.beq import Ui_MainWindow

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
        self.filterView.selectionModel().selectionChanged.connect(self.changeDeleteButtonState)
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

    def filterTypeChanged(self, text):
        '''
        Displays the gain field if the filter type isn't a high or low pass
        '''
        self.filterGain.setEnabled(self.isGainRequired(text))

    def isGainRequired(self, text):
        '''
        :param text: the filter type.
        :return: true if the filter type requires gain.
        '''
        return text != 'Low Pass' and text != 'High Pass'

    def addFilter(self):
        '''
        Adds a filter to the model.
        '''
        self.__filterModel.add(48000, self.filterType.currentText(), self.freq.value(), self.filterQ.value(),
                               self.filterGain.value())

    def deleteFilter(self):
        '''
        Deletes the selected filters.
        '''
        selection = self.filterView.selectionModel()
        if selection.hasSelection():
            self.__filterModel.delete([x.row() for x in selection.selectedRows()])

    def changeDeleteButtonState(self):
        '''
        Enables the delete button if there are selected rows.
        '''
        selection = self.filterView.selectionModel()
        self.deleteSelected.setEnabled(selection.hasSelection())

    def enableAddIfValid(self):
        '''
        Enables the add button if the values are valid for this filter type.
        '''
        enabled = True
        if self.isGainRequired(self.filterType.currentText()):
            enabled = self.filterGain.value() != 0.0
        self.addFilterButton.setEnabled(enabled)

    def display(self):
        '''
        Updates the chart.
        '''
        self.__magnitudeModel.display()

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
