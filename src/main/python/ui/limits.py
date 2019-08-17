# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'limits.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_graphLayoutDialog(object):
    def setupUi(self, graphLayoutDialog):
        graphLayoutDialog.setObjectName("graphLayoutDialog")
        graphLayoutDialog.resize(328, 166)
        self.gridLayout_2 = QtWidgets.QGridLayout(graphLayoutDialog)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.y2Min = QtWidgets.QSpinBox(graphLayoutDialog)
        self.y2Min.setMinimum(-200)
        self.y2Min.setMaximum(200)
        self.y2Min.setObjectName("y2Min")
        self.gridLayout.addWidget(self.y2Min, 2, 3, 1, 1)
        self.y1Min = QtWidgets.QSpinBox(graphLayoutDialog)
        self.y1Min.setMinimum(-200)
        self.y1Min.setMaximum(200)
        self.y1Min.setObjectName("y1Min")
        self.gridLayout.addWidget(self.y1Min, 2, 1, 1, 1)
        self.xMax = QtWidgets.QSpinBox(graphLayoutDialog)
        self.xMax.setMinimum(1)
        self.xMax.setMaximum(20000)
        self.xMax.setProperty("value", 250)
        self.xMax.setObjectName("xMax")
        self.gridLayout.addWidget(self.xMax, 1, 4, 1, 1)
        self.y1Max = QtWidgets.QSpinBox(graphLayoutDialog)
        self.y1Max.setMinimum(-200)
        self.y1Max.setMaximum(200)
        self.y1Max.setObjectName("y1Max")
        self.gridLayout.addWidget(self.y1Max, 0, 1, 1, 1)
        self.xMin = QtWidgets.QSpinBox(graphLayoutDialog)
        self.xMin.setMinimum(1)
        self.xMin.setMaximum(20000)
        self.xMin.setProperty("value", 2)
        self.xMin.setObjectName("xMin")
        self.gridLayout.addWidget(self.xMin, 1, 0, 1, 1)
        self.applyButton = QtWidgets.QPushButton(graphLayoutDialog)
        self.applyButton.setObjectName("applyButton")
        self.gridLayout.addWidget(self.applyButton, 1, 1, 1, 3)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 2, 2, 1, 1)
        self.y2Max = QtWidgets.QSpinBox(graphLayoutDialog)
        self.y2Max.setObjectName("y2Max")
        self.gridLayout.addWidget(self.y2Max, 0, 3, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 0, 2, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.hzLog = QtWidgets.QCheckBox(graphLayoutDialog)
        self.hzLog.setChecked(True)
        self.hzLog.setObjectName("hzLog")
        self.horizontalLayout.addWidget(self.hzLog)
        self.applyFullRangeX = QtWidgets.QPushButton(graphLayoutDialog)
        self.applyFullRangeX.setObjectName("applyFullRangeX")
        self.horizontalLayout.addWidget(self.applyFullRangeX)
        self.applyBassX = QtWidgets.QPushButton(graphLayoutDialog)
        self.applyBassX.setObjectName("applyBassX")
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
        # QtCore.QMetaObject.connectSlotsByName(graphLayoutDialog)

    def retranslateUi(self, graphLayoutDialog):
        _translate = QtCore.QCoreApplication.translate
        graphLayoutDialog.setWindowTitle(_translate("graphLayoutDialog", "Graph Limits"))
        self.applyButton.setText(_translate("graphLayoutDialog", "Apply"))
        self.hzLog.setText(_translate("graphLayoutDialog", "log scale?"))
        self.applyFullRangeX.setText(_translate("graphLayoutDialog", "20-20k"))
        self.applyBassX.setText(_translate("graphLayoutDialog", "1-160"))
