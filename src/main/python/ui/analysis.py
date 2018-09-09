# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'analysis.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_analysisDialog(object):
    def setupUi(self, analysisDialog):
        analysisDialog.setObjectName("analysisDialog")
        analysisDialog.resize(1402, 669)
        analysisDialog.setSizeGripEnabled(True)
        self.gridLayout_2 = QtWidgets.QGridLayout(analysisDialog)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.analysisGridLayout = QtWidgets.QGridLayout()
        self.analysisGridLayout.setObjectName("analysisGridLayout")
        self.showLimitsButton = QtWidgets.QToolButton(analysisDialog)
        self.showLimitsButton.setObjectName("showLimitsButton")
        self.analysisGridLayout.addWidget(self.showLimitsButton, 0, 1, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.analysisGridLayout.addItem(spacerItem, 1, 1, 1, 1)
        self.analysisTabs = QtWidgets.QTabWidget(analysisDialog)
        self.analysisTabs.setObjectName("analysisTabs")
        self.spectrumTab = QtWidgets.QWidget()
        self.spectrumTab.setObjectName("spectrumTab")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.spectrumTab)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.spectrumChart = MplWidget(self.spectrumTab)
        self.spectrumChart.setObjectName("spectrumChart")
        self.gridLayout_3.addWidget(self.spectrumChart, 0, 0, 1, 1)
        self.analysisTabs.addTab(self.spectrumTab, "")
        self.waveformTab = QtWidgets.QWidget()
        self.waveformTab.setObjectName("waveformTab")
        self.gridLayout = QtWidgets.QGridLayout(self.waveformTab)
        self.gridLayout.setObjectName("gridLayout")
        self.waveformChart = MplWidget(self.waveformTab)
        self.waveformChart.setObjectName("waveformChart")
        self.gridLayout.addWidget(self.waveformChart, 0, 0, 1, 1)
        self.analysisTabs.addTab(self.waveformTab, "")
        self.analysisGridLayout.addWidget(self.analysisTabs, 0, 2, 2, 1)
        self.frame = QtWidgets.QFrame(analysisDialog)
        self.frame.setFrameShape(QtWidgets.QFrame.Box)
        self.frame.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame.setObjectName("frame")
        self.formLayout = QtWidgets.QGridLayout(self.frame)
        self.formLayout.setObjectName("formLayout")
        self.startLabel = QtWidgets.QLabel(self.frame)
        self.startLabel.setObjectName("startLabel")
        self.formLayout.addWidget(self.startLabel, 2, 1, 1, 1)
        self.channelSelector = QtWidgets.QComboBox(self.frame)
        self.channelSelector.setObjectName("channelSelector")
        self.formLayout.addWidget(self.channelSelector, 1, 2, 1, 1)
        self.endLabel = QtWidgets.QLabel(self.frame)
        self.endLabel.setObjectName("endLabel")
        self.formLayout.addWidget(self.endLabel, 3, 1, 1, 1)
        self.startTime = QtWidgets.QTimeEdit(self.frame)
        self.startTime.setObjectName("startTime")
        self.formLayout.addWidget(self.startTime, 2, 2, 1, 1)
        self.endTime = QtWidgets.QTimeEdit(self.frame)
        self.endTime.setObjectName("endTime")
        self.formLayout.addWidget(self.endTime, 3, 2, 1, 1)
        self.channelSelectorLabel = QtWidgets.QLabel(self.frame)
        self.channelSelectorLabel.setObjectName("channelSelectorLabel")
        self.formLayout.addWidget(self.channelSelectorLabel, 1, 1, 1, 1)
        self.fileLabel = QtWidgets.QLabel(self.frame)
        self.fileLabel.setObjectName("fileLabel")
        self.formLayout.addWidget(self.fileLabel, 0, 1, 1, 1)
        self.filePicker = QtWidgets.QToolButton(self.frame)
        self.filePicker.setObjectName("filePicker")
        self.formLayout.addWidget(self.filePicker, 0, 3, 1, 1)
        self.file = QtWidgets.QLineEdit(self.frame)
        self.file.setEnabled(False)
        self.file.setObjectName("file")
        self.formLayout.addWidget(self.file, 0, 2, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.formLayout.addItem(spacerItem1, 5, 1, 1, 3)
        self.loadButton = QtWidgets.QPushButton(self.frame)
        self.loadButton.setObjectName("loadButton")
        self.formLayout.addWidget(self.loadButton, 4, 1, 1, 3)
        self.analysisGridLayout.addWidget(self.frame, 0, 0, 2, 1)
        self.analysisGridLayout.setColumnStretch(2, 1)
        self.gridLayout_2.addLayout(self.analysisGridLayout, 0, 0, 1, 1)

        self.retranslateUi(analysisDialog)
        self.analysisTabs.setCurrentIndex(0)
        self.filePicker.clicked.connect(analysisDialog.select_wav_file)
        self.loadButton.clicked.connect(analysisDialog.load_file)
        self.analysisTabs.currentChanged['int'].connect(analysisDialog.show_chart)
        self.showLimitsButton.clicked.connect(analysisDialog.show_limits)
        QtCore.QMetaObject.connectSlotsByName(analysisDialog)

    def retranslateUi(self, analysisDialog):
        _translate = QtCore.QCoreApplication.translate
        analysisDialog.setWindowTitle(_translate("analysisDialog", "Dialog"))
        self.showLimitsButton.setText(_translate("analysisDialog", "..."))
        self.analysisTabs.setTabText(self.analysisTabs.indexOf(self.spectrumTab), _translate("analysisDialog", "Spectrum"))
        self.analysisTabs.setTabText(self.analysisTabs.indexOf(self.waveformTab), _translate("analysisDialog", "Waveform"))
        self.startLabel.setText(_translate("analysisDialog", "Start"))
        self.endLabel.setText(_translate("analysisDialog", "End"))
        self.startTime.setDisplayFormat(_translate("analysisDialog", "HH:mm:ss.zzz"))
        self.endTime.setDisplayFormat(_translate("analysisDialog", "HH:mm:ss.zzz"))
        self.channelSelectorLabel.setText(_translate("analysisDialog", "Channel"))
        self.fileLabel.setText(_translate("analysisDialog", "File"))
        self.filePicker.setText(_translate("analysisDialog", "..."))
        self.loadButton.setText(_translate("analysisDialog", "Load"))

from mpl import MplWidget
