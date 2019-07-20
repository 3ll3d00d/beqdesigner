# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'export.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_exportSignalDialog(object):
    def setupUi(self, exportSignalDialog):
        exportSignalDialog.setObjectName("exportSignalDialog")
        exportSignalDialog.resize(408, 96)
        self.gridLayout = QtWidgets.QGridLayout(exportSignalDialog)
        self.gridLayout.setObjectName("gridLayout")
        self.buttonBox = QtWidgets.QDialogButtonBox(exportSignalDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Save)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 1, 0, 1, 1)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.signal = QtWidgets.QComboBox(exportSignalDialog)
        self.signal.setObjectName("signal")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.signal)
        self.signalLabel = QtWidgets.QLabel(exportSignalDialog)
        self.signalLabel.setObjectName("signalLabel")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.signalLabel)
        self.gridLayout.addLayout(self.formLayout, 0, 0, 1, 1)

        self.retranslateUi(exportSignalDialog)
        self.buttonBox.accepted.connect(exportSignalDialog.accept)
        self.buttonBox.rejected.connect(exportSignalDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(exportSignalDialog)

    def retranslateUi(self, exportSignalDialog):
        _translate = QtCore.QCoreApplication.translate
        exportSignalDialog.setWindowTitle(_translate("exportSignalDialog", "Export Signal"))
        self.signalLabel.setText(_translate("exportSignalDialog", "Signal"))
