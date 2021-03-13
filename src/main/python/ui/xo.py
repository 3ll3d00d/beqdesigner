# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'xo.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_xoDialog(object):
    def setupUi(self, xoDialog):
        xoDialog.setObjectName("xoDialog")
        xoDialog.resize(886, 820)
        self.verticalLayout = QtWidgets.QVBoxLayout(xoDialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.chartLayout = QtWidgets.QHBoxLayout()
        self.chartLayout.setObjectName("chartLayout")
        self.previewChart = MplWidget(xoDialog)
        self.previewChart.setObjectName("previewChart")
        self.chartLayout.addWidget(self.previewChart)
        self.chartControlsLayout = QtWidgets.QVBoxLayout()
        self.chartControlsLayout.setObjectName("chartControlsLayout")
        self.limitsButton = QtWidgets.QToolButton(xoDialog)
        self.limitsButton.setObjectName("limitsButton")
        self.chartControlsLayout.addWidget(self.limitsButton)
        self.showPhase = QtWidgets.QToolButton(xoDialog)
        self.showPhase.setCheckable(True)
        self.showPhase.setObjectName("showPhase")
        self.chartControlsLayout.addWidget(self.showPhase)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.chartControlsLayout.addItem(spacerItem)
        self.chartLayout.addLayout(self.chartControlsLayout)
        self.verticalLayout.addLayout(self.chartLayout)
        self.xoContainerLayout = QtWidgets.QGridLayout()
        self.xoContainerLayout.setObjectName("xoContainerLayout")
        self.showMatrixButton = QtWidgets.QPushButton(xoDialog)
        self.showMatrixButton.setObjectName("showMatrixButton")
        self.xoContainerLayout.addWidget(self.showMatrixButton, 0, 1, 1, 1)
        self.presetSelector = QtWidgets.QComboBox(xoDialog)
        self.presetSelector.setObjectName("presetSelector")
        self.xoContainerLayout.addWidget(self.presetSelector, 0, 2, 1, 1)
        self.linkChannelsButton = QtWidgets.QPushButton(xoDialog)
        self.linkChannelsButton.setObjectName("linkChannelsButton")
        self.xoContainerLayout.addWidget(self.linkChannelsButton, 0, 0, 1, 1)
        self.peqScrollArea = QtWidgets.QScrollArea(xoDialog)
        self.peqScrollArea.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.peqScrollArea.setWidgetResizable(True)
        self.peqScrollArea.setObjectName("peqScrollArea")
        self.channelsFrame = QtWidgets.QFrame()
        self.channelsFrame.setGeometry(QtCore.QRect(0, 0, 868, 259))
        self.channelsFrame.setObjectName("channelsFrame")
        self.channelsLayout = QtWidgets.QVBoxLayout(self.channelsFrame)
        self.channelsLayout.setObjectName("channelsLayout")
        self.peqScrollArea.setWidget(self.channelsFrame)
        self.xoContainerLayout.addWidget(self.peqScrollArea, 1, 0, 1, 3)
        self.verticalLayout.addLayout(self.xoContainerLayout)
        self.buttonBox = QtWidgets.QDialogButtonBox(xoDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Save)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)
        self.verticalLayout.setStretch(0, 3)
        self.verticalLayout.setStretch(1, 2)

        self.retranslateUi(xoDialog)
        self.buttonBox.accepted.connect(xoDialog.accept)
        self.buttonBox.rejected.connect(xoDialog.reject)
        self.showMatrixButton.clicked.connect(xoDialog.show_matrix)
        QtCore.QMetaObject.connectSlotsByName(xoDialog)

    def retranslateUi(self, xoDialog):
        _translate = QtCore.QCoreApplication.translate
        xoDialog.setWindowTitle(_translate("xoDialog", "Crossover Design"))
        self.limitsButton.setText(_translate("xoDialog", "..."))
        self.showPhase.setText(_translate("xoDialog", "..."))
        self.showMatrixButton.setText(_translate("xoDialog", "Input/Output  Routing"))
        self.linkChannelsButton.setText(_translate("xoDialog", "Link Channels"))
from mpl import MplWidget
