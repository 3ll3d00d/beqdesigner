# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'newversion.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_newVersionDialog(object):
    def setupUi(self, newVersionDialog):
        newVersionDialog.setObjectName("newVersionDialog")
        newVersionDialog.resize(586, 544)
        self.verticalLayout = QtWidgets.QVBoxLayout(newVersionDialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.headerLayout = QtWidgets.QHBoxLayout()
        self.headerLayout.setObjectName("headerLayout")
        self.message = QtWidgets.QLabel(newVersionDialog)
        self.message.setObjectName("message")
        self.headerLayout.addWidget(self.message)
        self.headerLayout.setStretch(0, 1)
        self.verticalLayout.addLayout(self.headerLayout)
        self.versionTable = QtWidgets.QTableView(newVersionDialog)
        self.versionTable.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.versionTable.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.versionTable.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.versionTable.setObjectName("versionTable")
        self.verticalLayout.addWidget(self.versionTable)
        self.releaseNotes = QtWidgets.QTextBrowser(newVersionDialog)
        self.releaseNotes.setOpenExternalLinks(True)
        self.releaseNotes.setObjectName("releaseNotes")
        self.verticalLayout.addWidget(self.releaseNotes)
        self.buttonBox = QtWidgets.QDialogButtonBox(newVersionDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)
        self.verticalLayout.setStretch(1, 1)
        self.verticalLayout.setStretch(2, 1)

        self.retranslateUi(newVersionDialog)
        self.buttonBox.accepted.connect(newVersionDialog.accept)
        self.buttonBox.rejected.connect(newVersionDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(newVersionDialog)

    def retranslateUi(self, newVersionDialog):
        _translate = QtCore.QCoreApplication.translate
        newVersionDialog.setWindowTitle(_translate("newVersionDialog", "New Version Available!"))
        self.message.setText(_translate("newVersionDialog", "Message"))
