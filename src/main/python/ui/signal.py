# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'signal.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_addSignalDialog(object):
    def setupUi(self, addSignalDialog):
        addSignalDialog.setObjectName("addSignalDialog")
        addSignalDialog.resize(740, 289)
        self.gridLayout = QtWidgets.QGridLayout(addSignalDialog)
        self.gridLayout.setObjectName("gridLayout")
        self.audioStreams = QtWidgets.QComboBox(addSignalDialog)
        self.audioStreams.setObjectName("audioStreams")
        self.gridLayout.addWidget(self.audioStreams, 1, 1, 1, 1)
        self.filterSpec = QtWidgets.QLineEdit(addSignalDialog)
        self.filterSpec.setObjectName("filterSpec")
        self.gridLayout.addWidget(self.filterSpec, 2, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(addSignalDialog)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)
        self.inputFile = QtWidgets.QLineEdit(addSignalDialog)
        self.inputFile.setEnabled(False)
        self.inputFile.setObjectName("inputFile")
        self.gridLayout.addWidget(self.inputFile, 0, 1, 1, 1)
        self.label = QtWidgets.QLabel(addSignalDialog)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.inputFilePicker = QtWidgets.QToolButton(addSignalDialog)
        self.inputFilePicker.setObjectName("inputFilePicker")
        self.gridLayout.addWidget(self.inputFilePicker, 0, 2, 1, 1)
        self.label_2 = QtWidgets.QLabel(addSignalDialog)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.buttonBox = QtWidgets.QDialogButtonBox(addSignalDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Apply|QtWidgets.QDialogButtonBox.Cancel)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 3, 1, 1, 1)
        self.conversionProgress = QtWidgets.QProgressBar(addSignalDialog)
        self.conversionProgress.setEnabled(True)
        self.conversionProgress.setProperty("value", 0)
        self.conversionProgress.setTextVisible(True)
        self.conversionProgress.setOrientation(QtCore.Qt.Horizontal)
        self.conversionProgress.setInvertedAppearance(False)
        self.conversionProgress.setTextDirection(QtWidgets.QProgressBar.TopToBottom)
        self.conversionProgress.setObjectName("conversionProgress")
        self.gridLayout.addWidget(self.conversionProgress, 4, 1, 1, 2)
        self.gridLayout.setColumnStretch(0, 1)
        self.gridLayout.setColumnStretch(1, 6)

        self.retranslateUi(addSignalDialog)
        self.buttonBox.accepted.connect(addSignalDialog.accept)
        self.buttonBox.rejected.connect(addSignalDialog.reject)
        self.inputFilePicker.clicked.connect(addSignalDialog.selectFile)
        self.audioStreams.currentIndexChanged['int'].connect(addSignalDialog.setFilterSpec)
        QtCore.QMetaObject.connectSlotsByName(addSignalDialog)

    def retranslateUi(self, addSignalDialog):
        _translate = QtCore.QCoreApplication.translate
        addSignalDialog.setWindowTitle(_translate("addSignalDialog", "Add Signal"))
        self.label_3.setText(_translate("addSignalDialog", "Filter Spec"))
        self.label.setText(_translate("addSignalDialog", "File"))
        self.inputFilePicker.setText(_translate("addSignalDialog", "..."))
        self.label_2.setText(_translate("addSignalDialog", "Streams"))

