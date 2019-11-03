# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'edit_mapping.ui'
#
# Created by: PyQt5 UI code generator 5.13.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_editMappingDialog(object):
    def setupUi(self, editMappingDialog):
        editMappingDialog.setObjectName("editMappingDialog")
        editMappingDialog.resize(415, 160)
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
        self.applyToAll = QtWidgets.QCheckBox(editMappingDialog)
        self.applyToAll.setObjectName("applyToAll")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.applyToAll)
        self.buttonBox = QtWidgets.QDialogButtonBox(editMappingDialog)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Save)
        self.buttonBox.setObjectName("buttonBox")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.buttonBox)

        self.retranslateUi(editMappingDialog)
        self.buttonBox.accepted.connect(editMappingDialog.accept)
        self.buttonBox.rejected.connect(editMappingDialog.reject)
        # QtCore.QMetaObject.connectSlotsByName(editMappingDialog)

    def retranslateUi(self, editMappingDialog):
        _translate = QtCore.QCoreApplication.translate
        editMappingDialog.setWindowTitle(_translate("editMappingDialog", "Edit Mapping"))
        self.channelIdxLabel.setText(_translate("editMappingDialog", "Channel"))
        self.signalLabel.setText(_translate("editMappingDialog", "Signal"))
        self.applyToAll.setText(_translate("editMappingDialog", "Apply to all channels?"))
