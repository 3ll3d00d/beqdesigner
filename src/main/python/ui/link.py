# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'link.ui',
# licensing of 'link.ui' applies.
#
# Created: Sat Jun 29 23:16:15 2019
#      by: pyside2-uic  running on PySide2 5.13.0
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_linkSignalDialog(object):
    def setupUi(self, linkSignalDialog):
        linkSignalDialog.setObjectName("linkSignalDialog")
        linkSignalDialog.resize(833, 325)
        self.gridLayout = QtWidgets.QGridLayout(linkSignalDialog)
        self.gridLayout.setObjectName("gridLayout")
        self.addToMaster = QtWidgets.QToolButton(linkSignalDialog)
        self.addToMaster.setObjectName("addToMaster")
        self.gridLayout.addWidget(self.addToMaster, 0, 2, 1, 1)
        self.masterCandidates = QtWidgets.QComboBox(linkSignalDialog)
        self.masterCandidates.setObjectName("masterCandidates")
        self.gridLayout.addWidget(self.masterCandidates, 0, 1, 1, 1)
        self.masterCandidatesLabel = QtWidgets.QLabel(linkSignalDialog)
        self.masterCandidatesLabel.setObjectName("masterCandidatesLabel")
        self.gridLayout.addWidget(self.masterCandidatesLabel, 0, 0, 1, 1)
        self.linkSignals = QtWidgets.QTableView(linkSignalDialog)
        self.linkSignals.setObjectName("linkSignals")
        self.gridLayout.addWidget(self.linkSignals, 1, 0, 1, 3)
        self.buttonBox = QtWidgets.QDialogButtonBox(linkSignalDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Save)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 2, 0, 1, 3)
        self.gridLayout.setColumnStretch(1, 1)

        self.retranslateUi(linkSignalDialog)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("accepted()"), linkSignalDialog.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("rejected()"), linkSignalDialog.reject)
        QtCore.QObject.connect(self.addToMaster, QtCore.SIGNAL("clicked()"), linkSignalDialog.addMaster)
        QtCore.QMetaObject.connectSlotsByName(linkSignalDialog)

    def retranslateUi(self, linkSignalDialog):
        linkSignalDialog.setWindowTitle(QtWidgets.QApplication.translate("linkSignalDialog", "Link Signals", None, -1))
        self.addToMaster.setText(QtWidgets.QApplication.translate("linkSignalDialog", "...", None, -1))
        self.masterCandidatesLabel.setText(QtWidgets.QApplication.translate("linkSignalDialog", "Make Master", None, -1))

