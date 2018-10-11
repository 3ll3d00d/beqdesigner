# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'edit_mapping.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_editMappingDialog(object):
    def setupUi(self, editMappingDialog):
        editMappingDialog.setObjectName("editMappingDialog")
        editMappingDialog.resize(415, 95)
        self.formLayout = QtWidgets.QFormLayout(editMappingDialog)
        self.formLayout.setObjectName("formLayout")
        self.channelIdxLabel = QtWidgets.QLabel(editMappingDialog)
        self.channelIdxLabel.setObjectName("channelIdxLabel")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.channelIdxLabel)
        self.signalLabel = QtWidgets.QLabel(editMappingDialog)
        self.signalLabel.setObjectName("signalLabel")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.signalLabel)
        self.channelIdx = QtWidgets.QLineEdit(editMappingDialog)
        self.channelIdx.setReadOnly(True)
        self.channelIdx.setObjectName("channelIdx")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.channelIdx)
        self.signal = QtWidgets.QComboBox(editMappingDialog)
        self.signal.setObjectName("signal")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.signal)

        self.retranslateUi(editMappingDialog)
        QtCore.QMetaObject.connectSlotsByName(editMappingDialog)

    def retranslateUi(self, editMappingDialog):
        _translate = QtCore.QCoreApplication.translate
        editMappingDialog.setWindowTitle(_translate("editMappingDialog", "Edit Mapping"))
        self.channelIdxLabel.setText(_translate("editMappingDialog", "Channel"))
        self.signalLabel.setText(_translate("editMappingDialog", "Signal"))

