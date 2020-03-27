# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'savechart.ui'
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


class Ui_saveChartDialog(object):
    def setupUi(self, saveChartDialog):
        if saveChartDialog.objectName():
            saveChartDialog.setObjectName(u"saveChartDialog")
        saveChartDialog.setWindowModality(Qt.ApplicationModal)
        saveChartDialog.resize(259, 155)
        saveChartDialog.setModal(True)
        self.gridLayout = QGridLayout(saveChartDialog)
        self.gridLayout.setObjectName(u"gridLayout")
        self.formLayout = QFormLayout()
        self.formLayout.setObjectName(u"formLayout")
        self.widthPixels = QSpinBox(saveChartDialog)
        self.widthPixels.setObjectName(u"widthPixels")
        self.widthPixels.setMinimum(1)
        self.widthPixels.setMaximum(8192)

        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.widthPixels)

        self.heightPixels = QSpinBox(saveChartDialog)
        self.heightPixels.setObjectName(u"heightPixels")
        self.heightPixels.setEnabled(False)
        self.heightPixels.setMinimum(1)
        self.heightPixels.setMaximum(8192)

        self.formLayout.setWidget(2, QFormLayout.FieldRole, self.heightPixels)

        self.label = QLabel(saveChartDialog)
        self.label.setObjectName(u"label")

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.label)

        self.label_2 = QLabel(saveChartDialog)
        self.label_2.setObjectName(u"label_2")

        self.formLayout.setWidget(2, QFormLayout.LabelRole, self.label_2)


        self.gridLayout.addLayout(self.formLayout, 0, 0, 1, 1)

        self.buttonBox = QDialogButtonBox(saveChartDialog)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Save)

        self.gridLayout.addWidget(self.buttonBox, 1, 0, 1, 1)


        self.retranslateUi(saveChartDialog)
        self.buttonBox.accepted.connect(saveChartDialog.accept)
        self.buttonBox.rejected.connect(saveChartDialog.reject)
        self.widthPixels.valueChanged.connect(saveChartDialog.set_height)

        QMetaObject.connectSlotsByName(saveChartDialog)
    # setupUi

    def retranslateUi(self, saveChartDialog):
        saveChartDialog.setWindowTitle(QCoreApplication.translate("saveChartDialog", u"Save Chart", None))
        self.label.setText(QCoreApplication.translate("saveChartDialog", u"Width", None))
        self.label_2.setText(QCoreApplication.translate("saveChartDialog", u"Height", None))
    # retranslateUi

