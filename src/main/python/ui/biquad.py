# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'biquad.ui'
##
## Created by: Qt User Interface Compiler version 5.14.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import (QCoreApplication, QMetaObject, QObject, QPoint,
    QRect, QSize, QUrl, Qt)
from PySide2.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
    QFontDatabase, QIcon, QLinearGradient, QPalette, QPainter, QPixmap,
    QRadialGradient)
from PySide2.QtWidgets import *


class Ui_exportBiquadDialog(object):
    def setupUi(self, exportBiquadDialog):
        if exportBiquadDialog.objectName():
            exportBiquadDialog.setObjectName(u"exportBiquadDialog")
        exportBiquadDialog.resize(691, 810)
        self.gridLayout = QGridLayout(exportBiquadDialog)
        self.gridLayout.setObjectName(u"gridLayout")
        self.fs = QComboBox(exportBiquadDialog)
        self.fs.addItem("")
        self.fs.addItem("")
        self.fs.addItem("")
        self.fs.setObjectName(u"fs")

        self.gridLayout.addWidget(self.fs, 2, 1, 1, 1)

        self.maxBiquadsLabel = QLabel(exportBiquadDialog)
        self.maxBiquadsLabel.setObjectName(u"maxBiquadsLabel")

        self.gridLayout.addWidget(self.maxBiquadsLabel, 3, 0, 1, 1)

        self.fsLabel = QLabel(exportBiquadDialog)
        self.fsLabel.setObjectName(u"fsLabel")

        self.gridLayout.addWidget(self.fsLabel, 2, 0, 1, 1)

        self.maxBiquads = QSpinBox(exportBiquadDialog)
        self.maxBiquads.setObjectName(u"maxBiquads")
        self.maxBiquads.setMinimum(1)
        self.maxBiquads.setMaximum(100)
        self.maxBiquads.setValue(10)

        self.gridLayout.addWidget(self.maxBiquads, 3, 1, 1, 1)

        self.biquads = QPlainTextEdit(exportBiquadDialog)
        self.biquads.setObjectName(u"biquads")
        font = QFont()
        font.setFamily(u"Consolas")
        self.biquads.setFont(font)
        self.biquads.setReadOnly(True)
        self.biquads.setTextInteractionFlags(Qt.TextSelectableByKeyboard|Qt.TextSelectableByMouse)

        self.gridLayout.addWidget(self.biquads, 4, 0, 1, 3)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.setDefaults = QToolButton(exportBiquadDialog)
        self.setDefaults.setObjectName(u"setDefaults")

        self.horizontalLayout.addWidget(self.setDefaults)

        self.outputFormat = QComboBox(exportBiquadDialog)
        self.outputFormat.addItem("")
        self.outputFormat.addItem("")
        self.outputFormat.addItem("")
        self.outputFormat.addItem("")
        self.outputFormat.setObjectName(u"outputFormat")

        self.horizontalLayout.addWidget(self.outputFormat)

        self.showHex = QCheckBox(exportBiquadDialog)
        self.showHex.setObjectName(u"showHex")
        self.showHex.setChecked(False)

        self.horizontalLayout.addWidget(self.showHex)

        self.saveToFile = QToolButton(exportBiquadDialog)
        self.saveToFile.setObjectName(u"saveToFile")

        self.horizontalLayout.addWidget(self.saveToFile)

        self.copyToClipboardBtn = QToolButton(exportBiquadDialog)
        self.copyToClipboardBtn.setObjectName(u"copyToClipboardBtn")

        self.horizontalLayout.addWidget(self.copyToClipboardBtn)


        self.gridLayout.addLayout(self.horizontalLayout, 1, 1, 1, 1)

        self.gridLayout.setColumnStretch(0, 1)
        self.gridLayout.setColumnStretch(1, 4)

        self.retranslateUi(exportBiquadDialog)
        self.showHex.clicked.connect(exportBiquadDialog.updateBiquads)
        self.fs.currentIndexChanged.connect(exportBiquadDialog.updateBiquads)
        self.maxBiquads.valueChanged.connect(exportBiquadDialog.updateBiquads)
        self.setDefaults.clicked.connect(exportBiquadDialog.save)
        self.copyToClipboardBtn.clicked.connect(exportBiquadDialog.copyToClipboard)
        self.saveToFile.clicked.connect(exportBiquadDialog.export)
        self.outputFormat.currentTextChanged.connect(exportBiquadDialog.update_format)

        QMetaObject.connectSlotsByName(exportBiquadDialog)
    # setupUi

    def retranslateUi(self, exportBiquadDialog):
        exportBiquadDialog.setWindowTitle(QCoreApplication.translate("exportBiquadDialog", u"Export Biquads", None))
        self.fs.setItemText(0, QCoreApplication.translate("exportBiquadDialog", u"48000", None))
        self.fs.setItemText(1, QCoreApplication.translate("exportBiquadDialog", u"96000", None))
        self.fs.setItemText(2, QCoreApplication.translate("exportBiquadDialog", u"192000", None))

        self.maxBiquadsLabel.setText(QCoreApplication.translate("exportBiquadDialog", u"Max Biquads", None))
        self.fsLabel.setText(QCoreApplication.translate("exportBiquadDialog", u"Sample Rate (Hz)", None))
        self.setDefaults.setText(QCoreApplication.translate("exportBiquadDialog", u"...", None))
        self.outputFormat.setItemText(0, QCoreApplication.translate("exportBiquadDialog", u"Minidsp 2x4HD", None))
        self.outputFormat.setItemText(1, QCoreApplication.translate("exportBiquadDialog", u"Minidsp 10x10HD", None))
        self.outputFormat.setItemText(2, QCoreApplication.translate("exportBiquadDialog", u"Minidsp 2x4", None))
        self.outputFormat.setItemText(3, QCoreApplication.translate("exportBiquadDialog", u"User Selected", None))

        self.showHex.setText(QCoreApplication.translate("exportBiquadDialog", u"Show Hex Value?", None))
        self.saveToFile.setText(QCoreApplication.translate("exportBiquadDialog", u"...", None))
        self.copyToClipboardBtn.setText(QCoreApplication.translate("exportBiquadDialog", u"...", None))
    # retranslateUi

