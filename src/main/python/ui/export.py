# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'export.ui',
# licensing of 'export.ui' applies.
#
# Created: Sun Jun 30 22:06:40 2019
#      by: pyside2-uic  running on PySide2 5.13.0
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

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
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("accepted()"), exportSignalDialog.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("rejected()"), exportSignalDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(exportSignalDialog)

    def retranslateUi(self, exportSignalDialog):
        exportSignalDialog.setWindowTitle(QtWidgets.QApplication.translate("exportSignalDialog", "Export Signal", None, -1))
        self.signalLabel.setText(QtWidgets.QApplication.translate("exportSignalDialog", "Signal", None, -1))

