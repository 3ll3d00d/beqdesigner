# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'limits.ui'
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


class Ui_graphLayoutDialog(object):
    def setupUi(self, graphLayoutDialog):
        if graphLayoutDialog.objectName():
            graphLayoutDialog.setObjectName(u"graphLayoutDialog")
        graphLayoutDialog.resize(328, 166)
        self.gridLayout_2 = QGridLayout(graphLayoutDialog)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.y2Min = QSpinBox(graphLayoutDialog)
        self.y2Min.setObjectName(u"y2Min")
        self.y2Min.setMinimum(-200)
        self.y2Min.setMaximum(200)

        self.gridLayout.addWidget(self.y2Min, 2, 3, 1, 1)

        self.y1Min = QSpinBox(graphLayoutDialog)
        self.y1Min.setObjectName(u"y1Min")
        self.y1Min.setMinimum(-200)
        self.y1Min.setMaximum(200)

        self.gridLayout.addWidget(self.y1Min, 2, 1, 1, 1)

        self.xMax = QSpinBox(graphLayoutDialog)
        self.xMax.setObjectName(u"xMax")
        self.xMax.setMinimum(1)
        self.xMax.setMaximum(20000)
        self.xMax.setValue(250)

        self.gridLayout.addWidget(self.xMax, 1, 4, 1, 1)

        self.y1Max = QSpinBox(graphLayoutDialog)
        self.y1Max.setObjectName(u"y1Max")
        self.y1Max.setMinimum(-200)
        self.y1Max.setMaximum(200)

        self.gridLayout.addWidget(self.y1Max, 0, 1, 1, 1)

        self.xMin = QSpinBox(graphLayoutDialog)
        self.xMin.setObjectName(u"xMin")
        self.xMin.setMinimum(1)
        self.xMin.setMaximum(20000)
        self.xMin.setValue(2)

        self.gridLayout.addWidget(self.xMin, 1, 0, 1, 1)

        self.applyButton = QPushButton(graphLayoutDialog)
        self.applyButton.setObjectName(u"applyButton")

        self.gridLayout.addWidget(self.applyButton, 1, 1, 1, 3)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer_2, 2, 2, 1, 1)

        self.y2Max = QSpinBox(graphLayoutDialog)
        self.y2Max.setObjectName(u"y2Max")

        self.gridLayout.addWidget(self.y2Max, 0, 3, 1, 1)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer, 0, 2, 1, 1)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.hzLog = QCheckBox(graphLayoutDialog)
        self.hzLog.setObjectName(u"hzLog")
        self.hzLog.setChecked(True)

        self.horizontalLayout.addWidget(self.hzLog)

        self.applyFullRangeX = QPushButton(graphLayoutDialog)
        self.applyFullRangeX.setObjectName(u"applyFullRangeX")

        self.horizontalLayout.addWidget(self.applyFullRangeX)

        self.applyBassX = QPushButton(graphLayoutDialog)
        self.applyBassX.setObjectName(u"applyBassX")

        self.horizontalLayout.addWidget(self.applyBassX)


        self.gridLayout.addLayout(self.horizontalLayout, 3, 0, 1, 5)

        self.gridLayout.setColumnStretch(0, 1)
        self.gridLayout.setColumnStretch(1, 1)
        self.gridLayout.setColumnStretch(3, 1)
        self.gridLayout.setColumnStretch(4, 1)

        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)


        self.retranslateUi(graphLayoutDialog)
        self.applyButton.clicked.connect(graphLayoutDialog.changeLimits)
        self.applyFullRangeX.clicked.connect(graphLayoutDialog.fullRangeLimits)
        self.applyBassX.clicked.connect(graphLayoutDialog.bassLimits)

        QMetaObject.connectSlotsByName(graphLayoutDialog)
    # setupUi

    def retranslateUi(self, graphLayoutDialog):
        graphLayoutDialog.setWindowTitle(QCoreApplication.translate("graphLayoutDialog", u"Graph Limits", None))
        self.applyButton.setText(QCoreApplication.translate("graphLayoutDialog", u"Apply", None))
        self.hzLog.setText(QCoreApplication.translate("graphLayoutDialog", u"log scale?", None))
        self.applyFullRangeX.setText(QCoreApplication.translate("graphLayoutDialog", u"20-20k", None))
        self.applyBassX.setText(QCoreApplication.translate("graphLayoutDialog", u"1-160", None))
    # retranslateUi

