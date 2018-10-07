# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'spectro.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_spectroDialog(object):
    def setupUi(self, spectroDialog):
        spectroDialog.setObjectName("spectroDialog")
        spectroDialog.resize(1065, 603)
        self.verticalLayout = QtWidgets.QVBoxLayout(spectroDialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.resolutionLabel = QtWidgets.QLabel(spectroDialog)
        self.resolutionLabel.setObjectName("resolutionLabel")
        self.horizontalLayout.addWidget(self.resolutionLabel)
        self.resolution = QtWidgets.QComboBox(spectroDialog)
        self.resolution.setObjectName("resolution")
        self.horizontalLayout.addWidget(self.resolution)
        self.vRangeLabel = QtWidgets.QLabel(spectroDialog)
        self.vRangeLabel.setObjectName("vRangeLabel")
        self.horizontalLayout.addWidget(self.vRangeLabel)
        self.vRange = QtWidgets.QSpinBox(spectroDialog)
        self.vRange.setMinimum(1)
        self.vRange.setMaximum(90)
        self.vRange.setProperty("value", 60)
        self.vRange.setObjectName("vRange")
        self.horizontalLayout.addWidget(self.vRange)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.limitsButton = QtWidgets.QToolButton(spectroDialog)
        self.limitsButton.setObjectName("limitsButton")
        self.horizontalLayout.addWidget(self.limitsButton)
        self.horizontalLayout.setStretch(1, 1)
        self.horizontalLayout.setStretch(4, 1)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.spectroChart = MplWidget(spectroDialog)
        self.spectroChart.setObjectName("spectroChart")
        self.verticalLayout.addWidget(self.spectroChart)

        self.retranslateUi(spectroDialog)
        self.resolution.currentIndexChanged['int'].connect(spectroDialog.update_chart)
        self.vRange.valueChanged['int'].connect(spectroDialog.update_chart)
        self.limitsButton.clicked.connect(spectroDialog.show_limits)
        QtCore.QMetaObject.connectSlotsByName(spectroDialog)

    def retranslateUi(self, spectroDialog):
        _translate = QtCore.QCoreApplication.translate
        spectroDialog.setWindowTitle(_translate("spectroDialog", "Spectrogram"))
        self.resolutionLabel.setText(_translate("spectroDialog", "Resolution:"))
        self.vRangeLabel.setText(_translate("spectroDialog", "Colour Range (dB)"))
        self.limitsButton.setText(_translate("spectroDialog", "..."))

from mpl import MplWidget
