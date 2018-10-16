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
        analysisDialog.resize(1369, 720)
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
        self.gridLayout_3.addWidget(self.spectrumChart, 3, 0, 1, 1)
        self.spectrumControlsLayout = QtWidgets.QGridLayout()
        self.spectrumControlsLayout.setObjectName("spectrumControlsLayout")
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.spectrumControlsLayout.addItem(spacerItem1, 0, 7, 1, 1)
        self.dbRange = QtWidgets.QSpinBox(self.spectrumTab)
        self.dbRange.setMinimum(-100)
        self.dbRange.setMaximum(-1)
        self.dbRange.setProperty("value", -12)
        self.dbRange.setObjectName("dbRange")
        self.spectrumControlsLayout.addWidget(self.dbRange, 0, 3, 1, 1)
        self.clipAtAverage = QtWidgets.QCheckBox(self.spectrumTab)
        self.clipAtAverage.setEnabled(False)
        self.clipAtAverage.setChecked(False)
        self.clipAtAverage.setObjectName("clipAtAverage")
        self.spectrumControlsLayout.addWidget(self.clipAtAverage, 0, 0, 1, 1)
        self.colourRangeLabel = QtWidgets.QLabel(self.spectrumTab)
        self.colourRangeLabel.setObjectName("colourRangeLabel")
        self.spectrumControlsLayout.addWidget(self.colourRangeLabel, 0, 4, 1, 1)
        self.colourRange = QtWidgets.QSpinBox(self.spectrumTab)
        self.colourRange.setMinimum(6)
        self.colourRange.setMaximum(120)
        self.colourRange.setProperty("value", 30)
        self.colourRange.setObjectName("colourRange")
        self.spectrumControlsLayout.addWidget(self.colourRange, 0, 5, 1, 1)
        self.showSpectroButton = QtWidgets.QToolButton(self.spectrumTab)
        self.showSpectroButton.setObjectName("showSpectroButton")
        self.spectrumControlsLayout.addWidget(self.showSpectroButton, 0, 8, 1, 1)
        self.clipToAbsolute = QtWidgets.QCheckBox(self.spectrumTab)
        self.clipToAbsolute.setChecked(True)
        self.clipToAbsolute.setObjectName("clipToAbsolute")
        self.spectrumControlsLayout.addWidget(self.clipToAbsolute, 0, 1, 1, 1)
        self.dbRangeLabel = QtWidgets.QLabel(self.spectrumTab)
        self.dbRangeLabel.setObjectName("dbRangeLabel")
        self.spectrumControlsLayout.addWidget(self.dbRangeLabel, 0, 2, 1, 1)
        self.gridLayout_3.addLayout(self.spectrumControlsLayout, 0, 0, 1, 1)
        self.analysisTabs.addTab(self.spectrumTab, "")
        self.waveformTab = QtWidgets.QWidget()
        self.waveformTab.setObjectName("waveformTab")
        self.gridLayout = QtWidgets.QGridLayout(self.waveformTab)
        self.gridLayout.setObjectName("gridLayout")
        self.waveformChart = MplWidget(self.waveformTab)
        self.waveformChart.setObjectName("waveformChart")
        self.gridLayout.addWidget(self.waveformChart, 1, 0, 1, 1)
        self.waveformControls = QtWidgets.QGridLayout()
        self.waveformControls.setObjectName("waveformControls")
        self.magnitudeDecibels = QtWidgets.QCheckBox(self.waveformTab)
        self.magnitudeDecibels.setObjectName("magnitudeDecibels")
        self.waveformControls.addWidget(self.magnitudeDecibels, 0, 0, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.waveformControls.addItem(spacerItem2, 0, 3, 1, 1)
        self.headroomLabel = QtWidgets.QLabel(self.waveformTab)
        self.headroomLabel.setObjectName("headroomLabel")
        self.waveformControls.addWidget(self.headroomLabel, 0, 1, 1, 1)
        self.headroom = QtWidgets.QDoubleSpinBox(self.waveformTab)
        self.headroom.setEnabled(True)
        self.headroom.setDecimals(3)
        self.headroom.setMinimum(-100.0)
        self.headroom.setMaximum(100.0)
        self.headroom.setSingleStep(0.001)
        self.headroom.setObjectName("headroom")
        self.waveformControls.addWidget(self.headroom, 0, 2, 1, 1)
        self.gridLayout.addLayout(self.waveformControls, 0, 0, 1, 1)
        self.analysisTabs.addTab(self.waveformTab, "")
        self.analysisGridLayout.addWidget(self.analysisTabs, 0, 2, 2, 1)
        self.frame = QtWidgets.QFrame(analysisDialog)
        self.frame.setFrameShape(QtWidgets.QFrame.Box)
        self.frame.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame.setObjectName("frame")
        self.formLayout = QtWidgets.QGridLayout(self.frame)
        self.formLayout.setObjectName("formLayout")
        self.line = QtWidgets.QFrame(self.frame)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.formLayout.addWidget(self.line, 5, 1, 1, 2)
        self.copyFilter = QtWidgets.QComboBox(self.frame)
        self.copyFilter.setObjectName("copyFilter")
        self.formLayout.addWidget(self.copyFilter, 6, 2, 1, 1)
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setObjectName("label")
        self.formLayout.addWidget(self.label, 6, 1, 1, 1)
        self.file = QtWidgets.QLineEdit(self.frame)
        self.file.setEnabled(False)
        self.file.setObjectName("file")
        self.formLayout.addWidget(self.file, 0, 2, 1, 1)
        self.startTime = QtWidgets.QTimeEdit(self.frame)
        self.startTime.setObjectName("startTime")
        self.formLayout.addWidget(self.startTime, 2, 2, 1, 1)
        self.loadButton = QtWidgets.QPushButton(self.frame)
        self.loadButton.setObjectName("loadButton")
        self.formLayout.addWidget(self.loadButton, 4, 1, 1, 3)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.formLayout.addItem(spacerItem3, 10, 1, 1, 3)
        self.channelSelectorLabel = QtWidgets.QLabel(self.frame)
        self.channelSelectorLabel.setObjectName("channelSelectorLabel")
        self.formLayout.addWidget(self.channelSelectorLabel, 1, 1, 1, 1)
        self.endTime = QtWidgets.QTimeEdit(self.frame)
        self.endTime.setObjectName("endTime")
        self.formLayout.addWidget(self.endTime, 3, 2, 1, 1)
        self.endLabel = QtWidgets.QLabel(self.frame)
        self.endLabel.setObjectName("endLabel")
        self.formLayout.addWidget(self.endLabel, 3, 1, 1, 1)
        self.fileLabel = QtWidgets.QLabel(self.frame)
        self.fileLabel.setObjectName("fileLabel")
        self.formLayout.addWidget(self.fileLabel, 0, 1, 1, 1)
        self.filePicker = QtWidgets.QToolButton(self.frame)
        self.filePicker.setObjectName("filePicker")
        self.formLayout.addWidget(self.filePicker, 0, 3, 1, 1)
        self.channelSelector = QtWidgets.QComboBox(self.frame)
        self.channelSelector.setObjectName("channelSelector")
        self.formLayout.addWidget(self.channelSelector, 1, 2, 1, 1)
        self.startLabel = QtWidgets.QLabel(self.frame)
        self.startLabel.setObjectName("startLabel")
        self.formLayout.addWidget(self.startLabel, 2, 1, 1, 1)
        self.analysisGridLayout.addWidget(self.frame, 0, 0, 2, 1)
        self.analysisGridLayout.setColumnStretch(2, 1)
        self.gridLayout_2.addLayout(self.analysisGridLayout, 0, 0, 1, 1)

        self.retranslateUi(analysisDialog)
        self.analysisTabs.setCurrentIndex(1)
        self.filePicker.clicked.connect(analysisDialog.select_wav_file)
        self.loadButton.clicked.connect(analysisDialog.load_file)
        self.analysisTabs.currentChanged['int'].connect(analysisDialog.show_chart)
        self.showLimitsButton.clicked.connect(analysisDialog.show_limits)
        self.clipAtAverage.clicked.connect(analysisDialog.show_chart)
        self.dbRange.valueChanged['int'].connect(analysisDialog.show_chart)
        self.clipAtAverage.clicked.connect(analysisDialog.allow_clip_choice)
        self.colourRange.valueChanged['int'].connect(analysisDialog.show_chart)
        self.clipToAbsolute.clicked.connect(analysisDialog.clip_to_abs)
        self.magnitudeDecibels.clicked.connect(analysisDialog.show_chart)
        self.showSpectroButton.clicked.connect(analysisDialog.show_spectro)
        self.copyFilter.currentIndexChanged['int'].connect(analysisDialog.show_chart)
        QtCore.QMetaObject.connectSlotsByName(analysisDialog)

    def retranslateUi(self, analysisDialog):
        _translate = QtCore.QCoreApplication.translate
        analysisDialog.setWindowTitle(_translate("analysisDialog", "Dialog"))
        self.showLimitsButton.setText(_translate("analysisDialog", "..."))
        self.clipAtAverage.setText(_translate("analysisDialog", "Clip to Average Level?"))
        self.colourRangeLabel.setText(_translate("analysisDialog", "Colour Range (dB)"))
        self.showSpectroButton.setText(_translate("analysisDialog", "..."))
        self.clipToAbsolute.setText(_translate("analysisDialog", "Clip to Absolute Peak?"))
        self.dbRangeLabel.setText(_translate("analysisDialog", "Clip At (dB)"))
        self.analysisTabs.setTabText(self.analysisTabs.indexOf(self.spectrumTab), _translate("analysisDialog", "Peak Spectrum"))
        self.magnitudeDecibels.setText(_translate("analysisDialog", "Waveform in dBFS?"))
        self.headroomLabel.setText(_translate("analysisDialog", "Headroom (dB)"))
        self.analysisTabs.setTabText(self.analysisTabs.indexOf(self.waveformTab), _translate("analysisDialog", "Waveform"))
        self.label.setText(_translate("analysisDialog", "Filter"))
        self.startTime.setDisplayFormat(_translate("analysisDialog", "HH:mm:ss.zzz"))
        self.loadButton.setText(_translate("analysisDialog", "Load"))
        self.channelSelectorLabel.setText(_translate("analysisDialog", "Channel"))
        self.endTime.setDisplayFormat(_translate("analysisDialog", "HH:mm:ss.zzz"))
        self.endLabel.setText(_translate("analysisDialog", "End"))
        self.fileLabel.setText(_translate("analysisDialog", "File"))
        self.filePicker.setText(_translate("analysisDialog", "..."))
        self.startLabel.setText(_translate("analysisDialog", "Start"))

from mpl import MplWidget
