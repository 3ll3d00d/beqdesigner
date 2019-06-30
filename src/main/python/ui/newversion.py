# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'newversion.ui',
# licensing of 'newversion.ui' applies.
#
# Created: Sun Jun 30 22:06:42 2019
#      by: pyside2-uic  running on PySide2 5.13.0
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

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
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("accepted()"), newVersionDialog.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("rejected()"), newVersionDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(newVersionDialog)

    def retranslateUi(self, newVersionDialog):
        newVersionDialog.setWindowTitle(QtWidgets.QApplication.translate("newVersionDialog", "New Version Available!", None, -1))
        self.message.setText(QtWidgets.QApplication.translate("newVersionDialog", "Message", None, -1))

