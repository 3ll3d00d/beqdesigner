# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'limits.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_graphLayoutDialog(object):
    def setupUi(self, graphLayoutDialog):
        graphLayoutDialog.setObjectName("graphLayoutDialog")
        graphLayoutDialog.resize(317, 165)
        self.gridLayout_2 = QtWidgets.QGridLayout(graphLayoutDialog)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.hzLog = QtWidgets.QCheckBox(graphLayoutDialog)
        self.hzLog.setChecked(True)
        self.hzLog.setObjectName("hzLog")
        self.gridLayout.addWidget(self.hzLog, 3, 0, 1, 4)
        self.yMax = QtWidgets.QSpinBox(graphLayoutDialog)
        self.yMax.setMinimum(-200)
        self.yMax.setMaximum(200)
        self.yMax.setObjectName("yMax")
        self.gridLayout.addWidget(self.yMax, 0, 1, 1, 1)
        self.yMin = QtWidgets.QSpinBox(graphLayoutDialog)
        self.yMin.setMinimum(-200)
        self.yMin.setMaximum(200)
        self.yMin.setObjectName("yMin")
        self.gridLayout.addWidget(self.yMin, 2, 1, 1, 1)
        self.xMax = QtWidgets.QSpinBox(graphLayoutDialog)
        self.xMax.setMinimum(1)
        self.xMax.setMaximum(20000)
        self.xMax.setProperty("value", 250)
        self.xMax.setObjectName("xMax")
        self.gridLayout.addWidget(self.xMax, 1, 3, 1, 1)
        self.xMin = QtWidgets.QSpinBox(graphLayoutDialog)
        self.xMin.setMinimum(1)
        self.xMin.setMaximum(20000)
        self.xMin.setProperty("value", 2)
        self.xMin.setObjectName("xMin")
        self.gridLayout.addWidget(self.xMin, 1, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 0, 2, 1, 1)
        self.applyButton = QtWidgets.QPushButton(graphLayoutDialog)
        self.applyButton.setObjectName("applyButton")
        self.gridLayout.addWidget(self.applyButton, 1, 1, 1, 2)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 2, 2, 1, 1)
        self.gridLayout.setColumnStretch(0, 1)
        self.gridLayout.setColumnStretch(1, 1)
        self.gridLayout.setColumnStretch(3, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)

        self.retranslateUi(graphLayoutDialog)
        self.hzLog.clicked.connect(graphLayoutDialog.toggleLogScale)
        self.applyButton.clicked.connect(graphLayoutDialog.changeLimits)
        QtCore.QMetaObject.connectSlotsByName(graphLayoutDialog)

    def retranslateUi(self, graphLayoutDialog):
        _translate = QtCore.QCoreApplication.translate
        graphLayoutDialog.setWindowTitle(_translate("graphLayoutDialog", "Graph Limits"))
        self.hzLog.setText(_translate("graphLayoutDialog", "log scale?"))
        self.applyButton.setText(_translate("graphLayoutDialog", "Apply"))

