# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mso.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_msoDialog(object):
    def setupUi(self, msoDialog):
        msoDialog.setObjectName("msoDialog")
        msoDialog.resize(628, 556)
        self.gridLayout = QtWidgets.QGridLayout(msoDialog)
        self.gridLayout.setObjectName("gridLayout")
        self.file = QtWidgets.QLineEdit(msoDialog)
        self.file.setObjectName("file")
        self.gridLayout.addWidget(self.file, 0, 1, 1, 1)
        self.fileSelect = QtWidgets.QToolButton(msoDialog)
        self.fileSelect.setObjectName("fileSelect")
        self.gridLayout.addWidget(self.fileSelect, 0, 2, 1, 1)
        self.status = QtWidgets.QLineEdit(msoDialog)
        self.status.setObjectName("status")
        self.gridLayout.addWidget(self.status, 1, 1, 1, 1)
        self.filterStatus = QtWidgets.QToolButton(msoDialog)
        self.filterStatus.setObjectName("filterStatus")
        self.gridLayout.addWidget(self.filterStatus, 1, 2, 1, 1)
        self.filterList = QtWidgets.QListWidget(msoDialog)
        self.filterList.setObjectName("filterList")
        self.gridLayout.addWidget(self.filterList, 2, 1, 1, 3)
        self.buttonBox = QtWidgets.QDialogButtonBox(msoDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 3, 0, 1, 4)

        self.retranslateUi(msoDialog)
        self.buttonBox.accepted.connect(msoDialog.accept) # type: ignore
        self.buttonBox.rejected.connect(msoDialog.reject) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(msoDialog)

    def retranslateUi(self, msoDialog):
        _translate = QtCore.QCoreApplication.translate
        msoDialog.setWindowTitle(_translate("msoDialog", "MSO Filter"))
        self.fileSelect.setText(_translate("msoDialog", "..."))
        self.filterStatus.setText(_translate("msoDialog", "..."))
