# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'values.ui'
#
# Created by: PyQt5 UI code generator 5.13.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_valuesDialog(object):
    def setupUi(self, valuesDialog):
        valuesDialog.setObjectName("valuesDialog")
        valuesDialog.resize(400, 393)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(valuesDialog.sizePolicy().hasHeightForWidth())
        valuesDialog.setSizePolicy(sizePolicy)
        valuesDialog.setSizeGripEnabled(True)
        self.formLayout = QtWidgets.QFormLayout(valuesDialog)
        self.formLayout.setObjectName("formLayout")
        self.label = QtWidgets.QLabel(valuesDialog)
        self.label.setObjectName("label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.freq = QtWidgets.QDoubleSpinBox(valuesDialog)
        self.freq.setDecimals(1)
        self.freq.setMinimum(1.0)
        self.freq.setMaximum(1000.0)
        self.freq.setObjectName("freq")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.freq)

        self.retranslateUi(valuesDialog)
        self.freq.valueChanged['double'].connect(valuesDialog.updateValues)
        QtCore.QMetaObject.connectSlotsByName(valuesDialog)

    def retranslateUi(self, valuesDialog):
        _translate = QtCore.QCoreApplication.translate
        valuesDialog.setWindowTitle(_translate("valuesDialog", "Values"))
        self.label.setText(_translate("valuesDialog", "Freq (Hz)"))
