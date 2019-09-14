# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'savechart.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_saveChartDialog(object):
    def setupUi(self, saveChartDialog):
        saveChartDialog.setObjectName("saveChartDialog")
        saveChartDialog.setWindowModality(QtCore.Qt.ApplicationModal)
        saveChartDialog.resize(259, 155)
        saveChartDialog.setModal(True)
        self.gridLayout = QtWidgets.QGridLayout(saveChartDialog)
        self.gridLayout.setObjectName("gridLayout")
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.widthPixels = QtWidgets.QSpinBox(saveChartDialog)
        self.widthPixels.setMinimum(1)
        self.widthPixels.setMaximum(8192)
        self.widthPixels.setObjectName("widthPixels")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.widthPixels)
        self.heightPixels = QtWidgets.QSpinBox(saveChartDialog)
        self.heightPixels.setEnabled(False)
        self.heightPixels.setMinimum(1)
        self.heightPixels.setMaximum(8192)
        self.heightPixels.setObjectName("heightPixels")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.heightPixels)
        self.label = QtWidgets.QLabel(saveChartDialog)
        self.label.setObjectName("label")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label)
        self.label_2 = QtWidgets.QLabel(saveChartDialog)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.gridLayout.addLayout(self.formLayout, 0, 0, 1, 1)
        self.buttonBox = QtWidgets.QDialogButtonBox(saveChartDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Save)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 1, 0, 1, 1)

        self.retranslateUi(saveChartDialog)
        self.buttonBox.accepted.connect(saveChartDialog.accept)
        self.buttonBox.rejected.connect(saveChartDialog.reject)
        self.widthPixels.valueChanged['int'].connect(saveChartDialog.set_height)
        QtCore.QMetaObject.connectSlotsByName(saveChartDialog)

    def retranslateUi(self, saveChartDialog):
        _translate = QtCore.QCoreApplication.translate
        saveChartDialog.setWindowTitle(_translate("saveChartDialog", "Save Chart"))
        self.label.setText(_translate("saveChartDialog", "Width"))
        self.label_2.setText(_translate("saveChartDialog", "Height"))
