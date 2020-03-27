# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'values.ui'
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


class Ui_valuesDialog(object):
    def setupUi(self, valuesDialog):
        if valuesDialog.objectName():
            valuesDialog.setObjectName(u"valuesDialog")
        valuesDialog.resize(400, 393)
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(valuesDialog.sizePolicy().hasHeightForWidth())
        valuesDialog.setSizePolicy(sizePolicy)
        valuesDialog.setSizeGripEnabled(True)
        self.formLayout = QFormLayout(valuesDialog)
        self.formLayout.setObjectName(u"formLayout")
        self.label = QLabel(valuesDialog)
        self.label.setObjectName(u"label")

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.label)

        self.freq = QDoubleSpinBox(valuesDialog)
        self.freq.setObjectName(u"freq")
        self.freq.setDecimals(1)
        self.freq.setMinimum(1.000000000000000)
        self.freq.setMaximum(1000.000000000000000)

        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.freq)


        self.retranslateUi(valuesDialog)
        self.freq.valueChanged.connect(valuesDialog.updateValues)

        QMetaObject.connectSlotsByName(valuesDialog)
    # setupUi

    def retranslateUi(self, valuesDialog):
        valuesDialog.setWindowTitle(QCoreApplication.translate("valuesDialog", u"Values", None))
        self.label.setText(QCoreApplication.translate("valuesDialog", u"Freq (Hz)", None))
    # retranslateUi

