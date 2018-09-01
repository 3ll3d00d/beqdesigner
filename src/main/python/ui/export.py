# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'export.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_exportFRDDialog(object):
    def setupUi(self, exportFRDDialog):
        exportFRDDialog.setObjectName("exportFRDDialog")
        exportFRDDialog.resize(408, 96)
        self.gridLayout = QtWidgets.QGridLayout(exportFRDDialog)
        self.gridLayout.setObjectName("gridLayout")
        self.buttonBox = QtWidgets.QDialogButtonBox(exportFRDDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Save)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 1, 0, 1, 1)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.series = QtWidgets.QComboBox(exportFRDDialog)
        self.series.setObjectName("series")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.series)
        self.seriesLabel = QtWidgets.QLabel(exportFRDDialog)
        self.seriesLabel.setObjectName("seriesLabel")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.seriesLabel)
        self.gridLayout.addLayout(self.formLayout, 0, 0, 1, 1)

        self.retranslateUi(exportFRDDialog)
        self.buttonBox.accepted.connect(exportFRDDialog.accept)
        self.buttonBox.rejected.connect(exportFRDDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(exportFRDDialog)

    def retranslateUi(self, exportFRDDialog):
        _translate = QtCore.QCoreApplication.translate
        exportFRDDialog.setWindowTitle(_translate("exportFRDDialog", "Export FRD"))
        self.seriesLabel.setText(_translate("exportFRDDialog", "Series"))

