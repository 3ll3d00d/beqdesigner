'''
The chart limits and values dialogs, apart from `model.limits`, which the headless pipeline imports without Qt.
'''
import numpy as np
from qtpy import QtWidgets
from qtpy.QtWidgets import QDialog

from ui.limits import Ui_graphLayoutDialog
from ui.values import Ui_valuesDialog


class ValuesDialog(QDialog, Ui_valuesDialog):
    '''
    Provides a mechanism for looking at the values listed in the chart without taking up screen estate.
    '''

    def __init__(self, data):
        super(ValuesDialog, self).__init__()
        self.setupUi(self)
        self.__step = 1
        self.__min_x, self.__max_x, self.data = self.__interpolate_data(self.__step, data)
        self.valueFields = []
        for idx, xy in enumerate(data):
            label = QtWidgets.QLabel(self)
            label.setObjectName(f"label{idx+1}")
            label.setText(xy.get_label())
            self.formLayout.setWidget(idx + 1, QtWidgets.QFormLayout.ItemRole.LabelRole, label)
            lineEdit = QtWidgets.QLineEdit(self)
            lineEdit.setEnabled(False)
            lineEdit.setObjectName(f"value{idx+1}")
            self.valueFields.append(lineEdit)
            self.formLayout.setWidget(idx + 1, QtWidgets.QFormLayout.ItemRole.FieldRole, lineEdit)
        if len(data) > 0:
            xdata = data[0].get_xdata()
            self.freq.setMinimum(xdata[0])
            self.freq.setSingleStep(self.__step)
            self.freq.setMaximum(xdata[-1])
            self.freq.setValue(xdata[0])
            self.freq.setEnabled(True)
        else:
            self.freq.setEnabled(False)

    @staticmethod
    def __interpolate_data(step, data):
        '''
        Interpolates the data using a simple 1D interpolation so the values becomes a simple lookup.
        :param step: the step in x values.
        :param data: the input data.
        :return: min_x, max_x, the interpolated data.
        '''
        min_x = data[0].get_xdata()[0]
        max_x = max(d.get_xdata()[-1] for d in data)
        x2 = np.arange(min_x, max_x + step, step)
        return min_x, max_x, [(x2, np.interp(x2, d.get_xdata(), d.get_ydata())) for d in data]

    def updateValues(self, freq):
        '''
        propagates the freq value change.
        '''
        freq_idx = int(freq / self.__step)
        for idx, xy in enumerate(self.data):
            val = xy[1][freq_idx]
            self.valueFields[idx].setText(str(round(val, 3)))


class LimitsDialog(QDialog, Ui_graphLayoutDialog):
    '''
    Provides some basic chart controls.
    '''

    def __init__(self, limits, x_min=1, x_max=24000, y1_min=-200, y1_max=200, y2_min=-200, y2_max=200, parent=None):
        super(LimitsDialog, self).__init__(parent)
        self.setupUi(self)
        self.__limits = limits
        self.hzLog.setChecked(limits.x_scale == 'log')
        x_min = int(min(x_min, limits.x_min))
        self.xMin.setMinimum(x_min)
        x_max = int(max(x_max, limits.x_max))
        self.xMin.setMaximum(x_max - 1)
        self.xMin.setValue(round(self.__limits.x_min))
        self.xMax.setMinimum(x_min + 1)
        self.xMax.setMaximum(x_max)
        self.xMax.setValue(round(self.__limits.x_max))
        self.y1Min.setMinimum(y1_min)
        self.y1Min.setMaximum(y1_max - 1)
        self.y1Min.setValue(round(self.__limits.y1_min))
        self.y1Max.setMinimum(y1_min + 1)
        self.y1Max.setMaximum(y1_max)
        self.y1Max.setValue(round(self.__limits.y1_max))
        if limits.axes_2 is not None:
            self.y2Min.setMinimum(y2_min)
            self.y2Min.setMaximum(y2_max - 1)
            self.y2Min.setValue(round(self.__limits.y2_min))
            self.y2Max.setMinimum(y2_min + 1)
            self.y2Max.setMaximum(y2_max)
            self.y2Max.setValue(round(self.__limits.y2_max))
        else:
            self.y2Min.setEnabled(False)
            self.y2Max.setEnabled(False)

    def changeLimits(self):
        '''
        Updates the chart limits.
        '''
        self.__limits.update(x_min=self.xMin.value(), x_max=self.xMax.value(), y1_min=self.y1Min.value(),
                               y1_max=self.y1Max.value(), y2_min=self.y2Min.value(), y2_max=self.y2Max.value(),
                               x_scale='log' if self.hzLog.isChecked() else 'linear', draw=True)

    def fullRangeLimits(self):
        ''' changes the x limits to show a full range signal '''
        self.xMin.setValue(20)
        self.xMax.setValue(20000)
        self.hzLog.setChecked(True)

    def bassLimits(self):
        ''' changes the x limits to show a bass limited signal '''
        self.xMin.setValue(1)
        self.xMax.setValue(160)
        self.hzLog.setChecked(False)