# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'analysis.ui'
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

from mpl import MplWidget


class Ui_analysisDialog(object):
    def setupUi(self, analysisDialog):
        if analysisDialog.objectName():
            analysisDialog.setObjectName(u"analysisDialog")
        analysisDialog.resize(1494, 895)
        analysisDialog.setSizeGripEnabled(True)
        self.gridLayout_2 = QGridLayout(analysisDialog)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.analysisGridLayout = QGridLayout()
        self.analysisGridLayout.setObjectName(u"analysisGridLayout")
        self.analysisFrame = QFrame(analysisDialog)
        self.analysisFrame.setObjectName(u"analysisFrame")
        self.analysisFrame.setFrameShape(QFrame.Box)
        self.analysisFrame.setFrameShadow(QFrame.Sunken)
        self.formLayout = QGridLayout(self.analysisFrame)
        self.formLayout.setObjectName(u"formLayout")
        self.line = QFrame(self.analysisFrame)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.formLayout.addWidget(self.line, 5, 1, 1, 2)

        self.copyFilter = QComboBox(self.analysisFrame)
        self.copyFilter.setObjectName(u"copyFilter")

        self.formLayout.addWidget(self.copyFilter, 6, 2, 1, 1)

        self.label = QLabel(self.analysisFrame)
        self.label.setObjectName(u"label")

        self.formLayout.addWidget(self.label, 6, 1, 1, 1)

        self.file = QLineEdit(self.analysisFrame)
        self.file.setObjectName(u"file")
        self.file.setEnabled(False)

        self.formLayout.addWidget(self.file, 0, 2, 1, 1)

        self.startTime = QTimeEdit(self.analysisFrame)
        self.startTime.setObjectName(u"startTime")

        self.formLayout.addWidget(self.startTime, 2, 2, 1, 1)

        self.loadButton = QPushButton(self.analysisFrame)
        self.loadButton.setObjectName(u"loadButton")

        self.formLayout.addWidget(self.loadButton, 4, 1, 1, 3)

        self.endLabel = QLabel(self.analysisFrame)
        self.endLabel.setObjectName(u"endLabel")

        self.formLayout.addWidget(self.endLabel, 3, 1, 1, 1)

        self.fileLabel = QLabel(self.analysisFrame)
        self.fileLabel.setObjectName(u"fileLabel")

        self.formLayout.addWidget(self.fileLabel, 0, 1, 1, 1)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.formLayout.addItem(self.verticalSpacer, 10, 1, 1, 3)

        self.channelSelectorLabel = QLabel(self.analysisFrame)
        self.channelSelectorLabel.setObjectName(u"channelSelectorLabel")

        self.formLayout.addWidget(self.channelSelectorLabel, 1, 1, 1, 1)

        self.endTime = QTimeEdit(self.analysisFrame)
        self.endTime.setObjectName(u"endTime")

        self.formLayout.addWidget(self.endTime, 3, 2, 1, 1)

        self.filePicker = QToolButton(self.analysisFrame)
        self.filePicker.setObjectName(u"filePicker")

        self.formLayout.addWidget(self.filePicker, 0, 3, 1, 1)

        self.channelSelector = QComboBox(self.analysisFrame)
        self.channelSelector.setObjectName(u"channelSelector")

        self.formLayout.addWidget(self.channelSelector, 1, 2, 1, 1)

        self.startLabel = QLabel(self.analysisFrame)
        self.startLabel.setObjectName(u"startLabel")

        self.formLayout.addWidget(self.startLabel, 2, 1, 1, 1)


        self.analysisGridLayout.addWidget(self.analysisFrame, 0, 0, 1, 1)

        self.analysisTabs = QTabWidget(analysisDialog)
        self.analysisTabs.setObjectName(u"analysisTabs")
        self.spectrumTab = QWidget()
        self.spectrumTab.setObjectName(u"spectrumTab")
        self.gridLayout_3 = QGridLayout(self.spectrumTab)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.spectrumControlsLayout = QGridLayout()
        self.spectrumControlsLayout.setObjectName(u"spectrumControlsLayout")
        self.buttonBoxLayout = QVBoxLayout()
        self.buttonBoxLayout.setObjectName(u"buttonBoxLayout")
        self.saveChart = QPushButton(self.spectrumTab)
        self.saveChart.setObjectName(u"saveChart")

        self.buttonBoxLayout.addWidget(self.saveChart)

        self.updateChart = QPushButton(self.spectrumTab)
        self.updateChart.setObjectName(u"updateChart")

        self.buttonBoxLayout.addWidget(self.updateChart)

        self.saveLayout = QPushButton(self.spectrumTab)
        self.saveLayout.setObjectName(u"saveLayout")

        self.buttonBoxLayout.addWidget(self.saveLayout)


        self.spectrumControlsLayout.addLayout(self.buttonBoxLayout, 0, 10, 3, 1)

        self.colourRangeLabel = QLabel(self.spectrumTab)
        self.colourRangeLabel.setObjectName(u"colourRangeLabel")
        font = QFont()
        font.setBold(False)
        font.setWeight(50)
        self.colourRangeLabel.setFont(font)
        self.colourRangeLabel.setAlignment(Qt.AlignCenter)

        self.spectrumControlsLayout.addWidget(self.colourRangeLabel, 0, 7, 1, 1)

        self.colourLowerLimit = QSpinBox(self.spectrumTab)
        self.colourLowerLimit.setObjectName(u"colourLowerLimit")
        self.colourLowerLimit.setMinimum(-120)
        self.colourLowerLimit.setMaximum(0)
        self.colourLowerLimit.setValue(-70)

        self.spectrumControlsLayout.addWidget(self.colourLowerLimit, 2, 7, 1, 1)

        self.colourUpperLimit = QSpinBox(self.spectrumTab)
        self.colourUpperLimit.setObjectName(u"colourUpperLimit")
        self.colourUpperLimit.setMinimum(-99)
        self.colourUpperLimit.setMaximum(0)
        self.colourUpperLimit.setValue(-10)

        self.spectrumControlsLayout.addWidget(self.colourUpperLimit, 1, 7, 1, 1)

        self.markerType = QComboBox(self.spectrumTab)
        self.markerType.setObjectName(u"markerType")

        self.spectrumControlsLayout.addWidget(self.markerType, 0, 1, 1, 1)

        self.magLimitTypeLabel = QLabel(self.spectrumTab)
        self.magLimitTypeLabel.setObjectName(u"magLimitTypeLabel")

        self.spectrumControlsLayout.addWidget(self.magLimitTypeLabel, 2, 0, 1, 1)

        self.markerTypeLabel = QLabel(self.spectrumTab)
        self.markerTypeLabel.setObjectName(u"markerTypeLabel")

        self.spectrumControlsLayout.addWidget(self.markerTypeLabel, 0, 0, 1, 1)

        self.analysisResolutionLabel = QLabel(self.spectrumTab)
        self.analysisResolutionLabel.setObjectName(u"analysisResolutionLabel")

        self.spectrumControlsLayout.addWidget(self.analysisResolutionLabel, 1, 0, 1, 1)

        self.analysisResolution = QComboBox(self.spectrumTab)
        self.analysisResolution.setObjectName(u"analysisResolution")

        self.spectrumControlsLayout.addWidget(self.analysisResolution, 1, 1, 1, 1)

        self.markerSize = QSpinBox(self.spectrumTab)
        self.markerSize.setObjectName(u"markerSize")
        self.markerSize.setMinimum(1)
        self.markerSize.setMaximum(9)

        self.spectrumControlsLayout.addWidget(self.markerSize, 0, 3, 1, 1)

        self.magLimitType = QComboBox(self.spectrumTab)
        self.magLimitType.addItem("")
        self.magLimitType.addItem("")
        self.magLimitType.addItem("")
        self.magLimitType.setObjectName(u"magLimitType")

        self.spectrumControlsLayout.addWidget(self.magLimitType, 2, 1, 1, 1)

        self.markerSizeLabel = QLabel(self.spectrumTab)
        self.markerSizeLabel.setObjectName(u"markerSizeLabel")

        self.spectrumControlsLayout.addWidget(self.markerSizeLabel, 0, 2, 1, 1)

        self.ellipseWidthLabel = QLabel(self.spectrumTab)
        self.ellipseWidthLabel.setObjectName(u"ellipseWidthLabel")

        self.spectrumControlsLayout.addWidget(self.ellipseWidthLabel, 1, 2, 1, 1)

        self.ellipseHeight = QDoubleSpinBox(self.spectrumTab)
        self.ellipseHeight.setObjectName(u"ellipseHeight")
        self.ellipseHeight.setMinimum(0.010000000000000)
        self.ellipseHeight.setMaximum(100.000000000000000)
        self.ellipseHeight.setSingleStep(0.010000000000000)
        self.ellipseHeight.setValue(1.000000000000000)

        self.spectrumControlsLayout.addWidget(self.ellipseHeight, 2, 3, 1, 1)

        self.ellipseWidth = QDoubleSpinBox(self.spectrumTab)
        self.ellipseWidth.setObjectName(u"ellipseWidth")
        self.ellipseWidth.setMinimum(0.010000000000000)
        self.ellipseWidth.setMaximum(100.000000000000000)
        self.ellipseWidth.setSingleStep(0.010000000000000)
        self.ellipseWidth.setValue(3.000000000000000)

        self.spectrumControlsLayout.addWidget(self.ellipseWidth, 1, 3, 1, 1)

        self.ellipseHeightLabel = QLabel(self.spectrumTab)
        self.ellipseHeightLabel.setObjectName(u"ellipseHeightLabel")

        self.spectrumControlsLayout.addWidget(self.ellipseHeightLabel, 2, 2, 1, 1)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.spectrumControlsLayout.addItem(self.horizontalSpacer, 0, 9, 3, 1)

        self.minTime = QTimeEdit(self.spectrumTab)
        self.minTime.setObjectName(u"minTime")

        self.spectrumControlsLayout.addWidget(self.minTime, 1, 5, 1, 1)

        self.magLowerLimit = QDoubleSpinBox(self.spectrumTab)
        self.magLowerLimit.setObjectName(u"magLowerLimit")
        self.magLowerLimit.setMinimum(-120.000000000000000)
        self.magLowerLimit.setMaximum(0.000000000000000)
        self.magLowerLimit.setSingleStep(0.010000000000000)
        self.magLowerLimit.setValue(-70.000000000000000)

        self.spectrumControlsLayout.addWidget(self.magLowerLimit, 2, 8, 1, 1)

        self.signalRangeLabel = QLabel(self.spectrumTab)
        self.signalRangeLabel.setObjectName(u"signalRangeLabel")
        self.signalRangeLabel.setFont(font)
        self.signalRangeLabel.setAlignment(Qt.AlignCenter)

        self.spectrumControlsLayout.addWidget(self.signalRangeLabel, 0, 8, 1, 1)

        self.magUpperLimit = QDoubleSpinBox(self.spectrumTab)
        self.magUpperLimit.setObjectName(u"magUpperLimit")
        self.magUpperLimit.setEnabled(False)
        self.magUpperLimit.setDecimals(2)
        self.magUpperLimit.setMinimum(-99.000000000000000)
        self.magUpperLimit.setMaximum(0.000000000000000)
        self.magUpperLimit.setSingleStep(0.001000000000000)
        self.magUpperLimit.setValue(-10.000000000000000)

        self.spectrumControlsLayout.addWidget(self.magUpperLimit, 1, 8, 1, 1)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.maxFilteredFreq = QSpinBox(self.spectrumTab)
        self.maxFilteredFreq.setObjectName(u"maxFilteredFreq")
        self.maxFilteredFreq.setMinimum(1)
        self.maxFilteredFreq.setMaximum(24000)
        self.maxFilteredFreq.setValue(40)

        self.horizontalLayout.addWidget(self.maxFilteredFreq)

        self.maxUnfilteredFreq = QSpinBox(self.spectrumTab)
        self.maxUnfilteredFreq.setObjectName(u"maxUnfilteredFreq")
        self.maxUnfilteredFreq.setMinimum(10)
        self.maxUnfilteredFreq.setMaximum(24000)
        self.maxUnfilteredFreq.setValue(80)

        self.horizontalLayout.addWidget(self.maxUnfilteredFreq)


        self.spectrumControlsLayout.addLayout(self.horizontalLayout, 2, 6, 1, 1)

        self.timeRangeLabe = QLabel(self.spectrumTab)
        self.timeRangeLabe.setObjectName(u"timeRangeLabe")
        self.timeRangeLabe.setFont(font)
        self.timeRangeLabe.setAlignment(Qt.AlignCenter)

        self.spectrumControlsLayout.addWidget(self.timeRangeLabe, 0, 5, 1, 1)

        self.minLabel = QLabel(self.spectrumTab)
        self.minLabel.setObjectName(u"minLabel")
        self.minLabel.setFont(font)
        self.minLabel.setAlignment(Qt.AlignCenter)

        self.spectrumControlsLayout.addWidget(self.minLabel, 1, 4, 1, 1)

        self.freqRangeLabel = QLabel(self.spectrumTab)
        self.freqRangeLabel.setObjectName(u"freqRangeLabel")
        self.freqRangeLabel.setFont(font)
        self.freqRangeLabel.setAlignment(Qt.AlignCenter)

        self.spectrumControlsLayout.addWidget(self.freqRangeLabel, 0, 6, 1, 1)

        self.maxTime = QTimeEdit(self.spectrumTab)
        self.maxTime.setObjectName(u"maxTime")

        self.spectrumControlsLayout.addWidget(self.maxTime, 2, 5, 1, 1)

        self.minFreq = QSpinBox(self.spectrumTab)
        self.minFreq.setObjectName(u"minFreq")

        self.spectrumControlsLayout.addWidget(self.minFreq, 1, 6, 1, 1)

        self.maxLabel = QLabel(self.spectrumTab)
        self.maxLabel.setObjectName(u"maxLabel")
        self.maxLabel.setFont(font)
        self.maxLabel.setAlignment(Qt.AlignCenter)

        self.spectrumControlsLayout.addWidget(self.maxLabel, 2, 4, 1, 1)


        self.gridLayout_3.addLayout(self.spectrumControlsLayout, 0, 0, 1, 1)

        self.spectrumChart = MplWidget(self.spectrumTab)
        self.spectrumChart.setObjectName(u"spectrumChart")

        self.gridLayout_3.addWidget(self.spectrumChart, 3, 0, 1, 1)

        self.analysisTabs.addTab(self.spectrumTab, "")
        self.waveformTab = QWidget()
        self.waveformTab.setObjectName(u"waveformTab")
        self.gridLayout = QGridLayout(self.waveformTab)
        self.gridLayout.setObjectName(u"gridLayout")
        self.waveformChart = MplWidget(self.waveformTab)
        self.waveformChart.setObjectName(u"waveformChart")

        self.gridLayout.addWidget(self.waveformChart, 1, 0, 1, 1)

        self.waveformControls = QGridLayout()
        self.waveformControls.setObjectName(u"waveformControls")
        self.magnitudeDecibels = QCheckBox(self.waveformTab)
        self.magnitudeDecibels.setObjectName(u"magnitudeDecibels")

        self.waveformControls.addWidget(self.magnitudeDecibels, 0, 0, 1, 1)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.waveformControls.addItem(self.horizontalSpacer_2, 0, 4, 1, 1)

        self.headroomLabel = QLabel(self.waveformTab)
        self.headroomLabel.setObjectName(u"headroomLabel")

        self.waveformControls.addWidget(self.headroomLabel, 0, 1, 1, 1)

        self.headroom = QDoubleSpinBox(self.waveformTab)
        self.headroom.setObjectName(u"headroom")
        self.headroom.setEnabled(True)
        self.headroom.setDecimals(3)
        self.headroom.setMinimum(-100.000000000000000)
        self.headroom.setMaximum(100.000000000000000)
        self.headroom.setSingleStep(0.001000000000000)

        self.waveformControls.addWidget(self.headroom, 0, 2, 1, 1)

        self.showLimitsButton = QToolButton(self.waveformTab)
        self.showLimitsButton.setObjectName(u"showLimitsButton")

        self.waveformControls.addWidget(self.showLimitsButton, 0, 3, 1, 1)


        self.gridLayout.addLayout(self.waveformControls, 0, 0, 1, 1)

        self.analysisTabs.addTab(self.waveformTab, "")

        self.analysisGridLayout.addWidget(self.analysisTabs, 0, 1, 2, 1)

        self.signalFrame = QFrame(analysisDialog)
        self.signalFrame.setObjectName(u"signalFrame")
        self.signalFrame.setFrameShape(QFrame.StyledPanel)
        self.signalFrame.setFrameShadow(QFrame.Raised)
        self.signalLayout = QGridLayout(self.signalFrame)
        self.signalLayout.setObjectName(u"signalLayout")
        self.leftSignal = QComboBox(self.signalFrame)
        self.leftSignal.setObjectName(u"leftSignal")

        self.signalLayout.addWidget(self.leftSignal, 0, 1, 1, 1)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.signalLayout.addItem(self.verticalSpacer_2, 3, 0, 1, 2)

        self.rightSignal = QComboBox(self.signalFrame)
        self.rightSignal.setObjectName(u"rightSignal")

        self.signalLayout.addWidget(self.rightSignal, 1, 1, 1, 1)

        self.leftSignalLabel = QLabel(self.signalFrame)
        self.leftSignalLabel.setObjectName(u"leftSignalLabel")

        self.signalLayout.addWidget(self.leftSignalLabel, 0, 0, 1, 1)

        self.rightSignalLabel = QLabel(self.signalFrame)
        self.rightSignalLabel.setObjectName(u"rightSignalLabel")

        self.signalLayout.addWidget(self.rightSignalLabel, 1, 0, 1, 1)

        self.compareSignalsButton = QPushButton(self.signalFrame)
        self.compareSignalsButton.setObjectName(u"compareSignalsButton")

        self.signalLayout.addWidget(self.compareSignalsButton, 2, 0, 1, 2)

        self.filterLeft = QCheckBox(self.signalFrame)
        self.filterLeft.setObjectName(u"filterLeft")

        self.signalLayout.addWidget(self.filterLeft, 0, 2, 1, 1)

        self.filterRight = QCheckBox(self.signalFrame)
        self.filterRight.setObjectName(u"filterRight")

        self.signalLayout.addWidget(self.filterRight, 1, 2, 1, 1)

        self.signalLayout.setColumnStretch(1, 1)

        self.analysisGridLayout.addWidget(self.signalFrame, 1, 0, 1, 1)

        self.analysisGridLayout.setColumnStretch(1, 1)

        self.gridLayout_2.addLayout(self.analysisGridLayout, 0, 0, 1, 1)


        self.retranslateUi(analysisDialog)
        self.filePicker.clicked.connect(analysisDialog.select_wav_file)
        self.loadButton.clicked.connect(analysisDialog.load_file)
        self.analysisTabs.currentChanged.connect(analysisDialog.show_chart)
        self.showLimitsButton.clicked.connect(analysisDialog.show_limits)
        self.magnitudeDecibels.clicked.connect(analysisDialog.show_chart)
        self.magLimitType.currentIndexChanged.connect(analysisDialog.set_mag_range_type)
        self.copyFilter.currentIndexChanged.connect(analysisDialog.update_filter)
        self.markerType.currentIndexChanged.connect(analysisDialog.update_marker_type)
        self.saveChart.clicked.connect(analysisDialog.save_chart)
        self.updateChart.clicked.connect(analysisDialog.update_chart)
        self.saveLayout.clicked.connect(analysisDialog.save_layout)
        self.leftSignal.currentIndexChanged.connect(analysisDialog.enable_compare)
        self.rightSignal.currentIndexChanged.connect(analysisDialog.enable_compare)
        self.compareSignalsButton.clicked.connect(analysisDialog.compare_signals)
        self.filterLeft.clicked.connect(analysisDialog.enable_compare)
        self.filterRight.clicked.connect(analysisDialog.enable_compare)

        self.analysisTabs.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(analysisDialog)
    # setupUi

    def retranslateUi(self, analysisDialog):
        analysisDialog.setWindowTitle(QCoreApplication.translate("analysisDialog", u"Analysis", None))
        self.label.setText(QCoreApplication.translate("analysisDialog", u"Filter", None))
        self.startTime.setDisplayFormat(QCoreApplication.translate("analysisDialog", u"HH:mm:ss.zzz", None))
        self.loadButton.setText(QCoreApplication.translate("analysisDialog", u"Load", None))
        self.endLabel.setText(QCoreApplication.translate("analysisDialog", u"End", None))
        self.fileLabel.setText(QCoreApplication.translate("analysisDialog", u"File", None))
        self.channelSelectorLabel.setText(QCoreApplication.translate("analysisDialog", u"Channel", None))
        self.endTime.setDisplayFormat(QCoreApplication.translate("analysisDialog", u"HH:mm:ss.zzz", None))
        self.filePicker.setText(QCoreApplication.translate("analysisDialog", u"...", None))
        self.startLabel.setText(QCoreApplication.translate("analysisDialog", u"Start", None))
        self.saveChart.setText(QCoreApplication.translate("analysisDialog", u"Save Chart", None))
        self.updateChart.setText(QCoreApplication.translate("analysisDialog", u"Update", None))
        self.saveLayout.setText(QCoreApplication.translate("analysisDialog", u"Save Layout", None))
        self.colourRangeLabel.setText(QCoreApplication.translate("analysisDialog", u"Colour", None))
        self.magLimitTypeLabel.setText(QCoreApplication.translate("analysisDialog", u"Filter Type", None))
        self.markerTypeLabel.setText(QCoreApplication.translate("analysisDialog", u"Type", None))
        self.analysisResolutionLabel.setText(QCoreApplication.translate("analysisDialog", u"Resolution", None))
        self.magLimitType.setItemText(0, QCoreApplication.translate("analysisDialog", u"Constant", None))
        self.magLimitType.setItemText(1, QCoreApplication.translate("analysisDialog", u"Peak", None))
        self.magLimitType.setItemText(2, QCoreApplication.translate("analysisDialog", u"Average", None))

        self.markerSizeLabel.setText(QCoreApplication.translate("analysisDialog", u"Marker Size", None))
        self.ellipseWidthLabel.setText(QCoreApplication.translate("analysisDialog", u"Ellipse Width", None))
        self.ellipseHeightLabel.setText(QCoreApplication.translate("analysisDialog", u"Ellipse Height", None))
        self.minTime.setDisplayFormat(QCoreApplication.translate("analysisDialog", u"HH:mm:ss", None))
        self.signalRangeLabel.setText(QCoreApplication.translate("analysisDialog", u"Signal", None))
        self.timeRangeLabe.setText(QCoreApplication.translate("analysisDialog", u"Time", None))
        self.minLabel.setText(QCoreApplication.translate("analysisDialog", u"Min", None))
        self.freqRangeLabel.setText(QCoreApplication.translate("analysisDialog", u"Freq", None))
        self.maxTime.setDisplayFormat(QCoreApplication.translate("analysisDialog", u"HH:mm:ss", None))
        self.maxLabel.setText(QCoreApplication.translate("analysisDialog", u"Max", None))
        self.analysisTabs.setTabText(self.analysisTabs.indexOf(self.spectrumTab), QCoreApplication.translate("analysisDialog", u"Peak Spectrum", None))
        self.magnitudeDecibels.setText(QCoreApplication.translate("analysisDialog", u"Waveform in dBFS?", None))
        self.headroomLabel.setText(QCoreApplication.translate("analysisDialog", u"Headroom (dB)", None))
        self.showLimitsButton.setText(QCoreApplication.translate("analysisDialog", u"...", None))
        self.analysisTabs.setTabText(self.analysisTabs.indexOf(self.waveformTab), QCoreApplication.translate("analysisDialog", u"Waveform", None))
        self.leftSignalLabel.setText(QCoreApplication.translate("analysisDialog", u"Left", None))
        self.rightSignalLabel.setText(QCoreApplication.translate("analysisDialog", u"Right", None))
        self.compareSignalsButton.setText(QCoreApplication.translate("analysisDialog", u"Compare", None))
        self.filterLeft.setText("")
        self.filterRight.setText("")
    # retranslateUi

