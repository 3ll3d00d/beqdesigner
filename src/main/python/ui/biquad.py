# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'biquad.ui',
# licensing of 'biquad.ui' applies.
#
# Created: Sun Jun 30 22:06:39 2019
#      by: pyside2-uic  running on PySide2 5.13.0
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_exportBiquadDialog(object):
    def setupUi(self, exportBiquadDialog):
        exportBiquadDialog.setObjectName("exportBiquadDialog")
        exportBiquadDialog.resize(691, 810)
        self.gridLayout = QtWidgets.QGridLayout(exportBiquadDialog)
        self.gridLayout.setObjectName("gridLayout")
        self.fs = QtWidgets.QComboBox(exportBiquadDialog)
        self.fs.setObjectName("fs")
        self.gridLayout.addWidget(self.fs, 2, 1, 1, 1)
        self.maxBiquadsLabel = QtWidgets.QLabel(exportBiquadDialog)
        self.maxBiquadsLabel.setObjectName("maxBiquadsLabel")
        self.gridLayout.addWidget(self.maxBiquadsLabel, 3, 0, 1, 1)
        self.fsLabel = QtWidgets.QLabel(exportBiquadDialog)
        self.fsLabel.setObjectName("fsLabel")
        self.gridLayout.addWidget(self.fsLabel, 2, 0, 1, 1)
        self.maxBiquads = QtWidgets.QSpinBox(exportBiquadDialog)
        self.maxBiquads.setMinimum(1)
        self.maxBiquads.setMaximum(100)
        self.maxBiquads.setProperty("value", 10)
        self.maxBiquads.setObjectName("maxBiquads")
        self.gridLayout.addWidget(self.maxBiquads, 3, 1, 1, 1)
        self.biquads = QtWidgets.QPlainTextEdit(exportBiquadDialog)
        font = QtGui.QFont()
        font.setFamily("Consolas")
        self.biquads.setFont(font)
        self.biquads.setReadOnly(True)
        self.biquads.setTextInteractionFlags(QtCore.Qt.TextSelectableByKeyboard|QtCore.Qt.TextSelectableByMouse)
        self.biquads.setObjectName("biquads")
        self.gridLayout.addWidget(self.biquads, 4, 0, 1, 3)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.setDefaults = QtWidgets.QToolButton(exportBiquadDialog)
        self.setDefaults.setObjectName("setDefaults")
        self.horizontalLayout.addWidget(self.setDefaults)
        self.outputFormat = QtWidgets.QComboBox(exportBiquadDialog)
        self.outputFormat.setObjectName("outputFormat")
        self.horizontalLayout.addWidget(self.outputFormat)
        self.showHex = QtWidgets.QCheckBox(exportBiquadDialog)
        self.showHex.setChecked(False)
        self.showHex.setObjectName("showHex")
        self.horizontalLayout.addWidget(self.showHex)
        self.saveToFile = QtWidgets.QToolButton(exportBiquadDialog)
        self.saveToFile.setObjectName("saveToFile")
        self.horizontalLayout.addWidget(self.saveToFile)
        self.copyToClipboardBtn = QtWidgets.QToolButton(exportBiquadDialog)
        self.copyToClipboardBtn.setObjectName("copyToClipboardBtn")
        self.horizontalLayout.addWidget(self.copyToClipboardBtn)
        self.gridLayout.addLayout(self.horizontalLayout, 1, 1, 1, 1)
        self.gridLayout.setColumnStretch(0, 1)
        self.gridLayout.setColumnStretch(1, 4)

        self.retranslateUi(exportBiquadDialog)
        QtCore.QObject.connect(self.showHex, QtCore.SIGNAL("clicked()"), exportBiquadDialog.updateBiquads)
        QtCore.QObject.connect(self.fs, QtCore.SIGNAL("currentIndexChanged(int)"), exportBiquadDialog.updateBiquads)
        QtCore.QObject.connect(self.maxBiquads, QtCore.SIGNAL("valueChanged(int)"), exportBiquadDialog.updateBiquads)
        QtCore.QObject.connect(self.setDefaults, QtCore.SIGNAL("clicked()"), exportBiquadDialog.save)
        QtCore.QObject.connect(self.copyToClipboardBtn, QtCore.SIGNAL("clicked()"), exportBiquadDialog.copyToClipboard)
        QtCore.QObject.connect(self.saveToFile, QtCore.SIGNAL("clicked()"), exportBiquadDialog.export)
        QtCore.QObject.connect(self.outputFormat, QtCore.SIGNAL("currentTextChanged(QString)"), exportBiquadDialog.update_format)
        QtCore.QMetaObject.connectSlotsByName(exportBiquadDialog)

    def retranslateUi(self, exportBiquadDialog):
        exportBiquadDialog.setWindowTitle(QtWidgets.QApplication.translate("exportBiquadDialog", "Export Biquads", None, -1))
        self.maxBiquadsLabel.setText(QtWidgets.QApplication.translate("exportBiquadDialog", "Max Biquads", None, -1))
        self.fsLabel.setText(QtWidgets.QApplication.translate("exportBiquadDialog", "Sample Rate (Hz)", None, -1))
        self.setDefaults.setText(QtWidgets.QApplication.translate("exportBiquadDialog", "...", None, -1))
        self.showHex.setText(QtWidgets.QApplication.translate("exportBiquadDialog", "Show Hex Value?", None, -1))
        self.saveToFile.setText(QtWidgets.QApplication.translate("exportBiquadDialog", "...", None, -1))
        self.copyToClipboardBtn.setText(QtWidgets.QApplication.translate("exportBiquadDialog", "...", None, -1))

