# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'export.ui'
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


class Ui_exportSignalDialog(object):
    def setupUi(self, exportSignalDialog):
        if exportSignalDialog.objectName():
            exportSignalDialog.setObjectName(u"exportSignalDialog")
        exportSignalDialog.resize(408, 96)
        self.gridLayout = QGridLayout(exportSignalDialog)
        self.gridLayout.setObjectName(u"gridLayout")
        self.buttonBox = QDialogButtonBox(exportSignalDialog)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Save)

        self.gridLayout.addWidget(self.buttonBox, 1, 0, 1, 1)

        self.formLayout = QFormLayout()
        self.formLayout.setObjectName(u"formLayout")
        self.signal = QComboBox(exportSignalDialog)
        self.signal.setObjectName(u"signal")

        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.signal)

        self.signalLabel = QLabel(exportSignalDialog)
        self.signalLabel.setObjectName(u"signalLabel")

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.signalLabel)


        self.gridLayout.addLayout(self.formLayout, 0, 0, 1, 1)


        self.retranslateUi(exportSignalDialog)
        self.buttonBox.accepted.connect(exportSignalDialog.accept)
        self.buttonBox.rejected.connect(exportSignalDialog.reject)

        QMetaObject.connectSlotsByName(exportSignalDialog)
    # setupUi

    def retranslateUi(self, exportSignalDialog):
        exportSignalDialog.setWindowTitle(QCoreApplication.translate("exportSignalDialog", u"Export Signal", None))
        self.signalLabel.setText(QCoreApplication.translate("exportSignalDialog", u"Signal", None))
    # retranslateUi

