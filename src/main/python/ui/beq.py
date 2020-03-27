# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'beq.ui'
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
from common import PlotWidgetWithDateAxis


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1569, 1012)
        self.actionShow_Logs = QAction(MainWindow)
        self.actionShow_Logs.setObjectName(u"actionShow_Logs")
        self.actionPreferences = QAction(MainWindow)
        self.actionPreferences.setObjectName(u"actionPreferences")
        self.actionExtract_Audio = QAction(MainWindow)
        self.actionExtract_Audio.setObjectName(u"actionExtract_Audio")
        self.actionPresets = QAction(MainWindow)
        self.actionPresets.setObjectName(u"actionPresets")
        self.actionClear_Preset_3 = QAction(MainWindow)
        self.actionClear_Preset_3.setObjectName(u"actionClear_Preset_3")
        self.actionSave_Chart = QAction(MainWindow)
        self.actionSave_Chart.setObjectName(u"actionSave_Chart")
        self.actionExport_Biquad = QAction(MainWindow)
        self.actionExport_Biquad.setObjectName(u"actionExport_Biquad")
        self.actionSave_Filter = QAction(MainWindow)
        self.actionSave_Filter.setObjectName(u"actionSave_Filter")
        self.actionSave_Filter.setEnabled(False)
        self.actionLoad_Filter = QAction(MainWindow)
        self.actionLoad_Filter.setObjectName(u"actionLoad_Filter")
        self.actionLoad_Filter.setCheckable(False)
        self.actionLoad_Filter.setEnabled(True)
        self.action_load_preset_1 = QAction(MainWindow)
        self.action_load_preset_1.setObjectName(u"action_load_preset_1")
        self.action_load_preset_2 = QAction(MainWindow)
        self.action_load_preset_2.setObjectName(u"action_load_preset_2")
        self.action_load_preset_3 = QAction(MainWindow)
        self.action_load_preset_3.setObjectName(u"action_load_preset_3")
        self.action_clear_preset_1 = QAction(MainWindow)
        self.action_clear_preset_1.setObjectName(u"action_clear_preset_1")
        self.action_clear_preset_2 = QAction(MainWindow)
        self.action_clear_preset_2.setObjectName(u"action_clear_preset_2")
        self.action_clear_preset_3 = QAction(MainWindow)
        self.action_clear_preset_3.setObjectName(u"action_clear_preset_3")
        self.action_store_preset_1 = QAction(MainWindow)
        self.action_store_preset_1.setObjectName(u"action_store_preset_1")
        self.action_store_preset_2 = QAction(MainWindow)
        self.action_store_preset_2.setObjectName(u"action_store_preset_2")
        self.action_store_preset_3 = QAction(MainWindow)
        self.action_store_preset_3.setObjectName(u"action_store_preset_3")
        self.actionExport_FRD = QAction(MainWindow)
        self.actionExport_FRD.setObjectName(u"actionExport_FRD")
        self.action_Save_Project = QAction(MainWindow)
        self.action_Save_Project.setObjectName(u"action_Save_Project")
        self.action_Load_Project = QAction(MainWindow)
        self.action_Load_Project.setObjectName(u"action_Load_Project")
        self.actionSave_Signal = QAction(MainWindow)
        self.actionSave_Signal.setObjectName(u"actionSave_Signal")
        self.actionLoad_Signal = QAction(MainWindow)
        self.actionLoad_Signal.setObjectName(u"actionLoad_Signal")
        self.actionClear_Project = QAction(MainWindow)
        self.actionClear_Project.setObjectName(u"actionClear_Project")
        self.actionAnalyse_Audio = QAction(MainWindow)
        self.actionAnalyse_Audio.setObjectName(u"actionAnalyse_Audio")
        self.action_Batch_Extract = QAction(MainWindow)
        self.action_Batch_Extract.setObjectName(u"action_Batch_Extract")
        self.actionSave_Report = QAction(MainWindow)
        self.actionSave_Report.setObjectName(u"actionSave_Report")
        self.actionAbout = QAction(MainWindow)
        self.actionAbout.setObjectName(u"actionAbout")
        self.action_Remux_Audio = QAction(MainWindow)
        self.action_Remux_Audio.setObjectName(u"action_Remux_Audio")
        self.actionAdd_BEQ_Filter = QAction(MainWindow)
        self.actionAdd_BEQ_Filter.setObjectName(u"actionAdd_BEQ_Filter")
        self.actionClear_Signals = QAction(MainWindow)
        self.actionClear_Signals.setObjectName(u"actionClear_Signals")
        self.action1_1_Smoothing = QAction(MainWindow)
        self.action1_1_Smoothing.setObjectName(u"action1_1_Smoothing")
        self.action1_3_Smoothing = QAction(MainWindow)
        self.action1_3_Smoothing.setObjectName(u"action1_3_Smoothing")
        self.action1_6_Smoothing = QAction(MainWindow)
        self.action1_6_Smoothing.setObjectName(u"action1_6_Smoothing")
        self.action1_1_2_Smoothing = QAction(MainWindow)
        self.action1_1_2_Smoothing.setObjectName(u"action1_1_2_Smoothing")
        self.action1_2_4_Smoothing = QAction(MainWindow)
        self.action1_2_4_Smoothing.setObjectName(u"action1_2_4_Smoothing")
        self.action1_4_8_Smoothing = QAction(MainWindow)
        self.action1_4_8_Smoothing.setObjectName(u"action1_4_8_Smoothing")
        self.action_Remove_Smoothing = QAction(MainWindow)
        self.action_Remove_Smoothing.setObjectName(u"action_Remove_Smoothing")
        self.actionClear_Filters = QAction(MainWindow)
        self.actionClear_Filters.setObjectName(u"actionClear_Filters")
        self.actionMerge_Minidsp_XML = QAction(MainWindow)
        self.actionMerge_Minidsp_XML.setObjectName(u"actionMerge_Minidsp_XML")
        self.actionUser_Guide = QAction(MainWindow)
        self.actionUser_Guide.setObjectName(u"actionUser_Guide")
        self.actionRelease_Notes = QAction(MainWindow)
        self.actionRelease_Notes.setObjectName(u"actionRelease_Notes")
        self.actionExport_BEQ_Filter = QAction(MainWindow)
        self.actionExport_BEQ_Filter.setObjectName(u"actionExport_BEQ_Filter")
        self.actionCreate_AVS_Post = QAction(MainWindow)
        self.actionCreate_AVS_Post.setObjectName(u"actionCreate_AVS_Post")
        self.actionSync_with_HTP_1 = QAction(MainWindow)
        self.actionSync_with_HTP_1.setObjectName(u"actionSync_with_HTP_1")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.widgetGridLayout = QGridLayout(self.centralwidget)
        self.widgetGridLayout.setObjectName(u"widgetGridLayout")
        self.panes = QHBoxLayout()
        self.panes.setSpacing(9)
        self.panes.setObjectName(u"panes")
        self.leftPane = QGridLayout()
        self.leftPane.setObjectName(u"leftPane")
        self.showValuesButton = QToolButton(self.centralwidget)
        self.showValuesButton.setObjectName(u"showValuesButton")

        self.leftPane.addWidget(self.showValuesButton, 0, 6, 1, 1)

        self.filterReference = QComboBox(self.centralwidget)
        self.filterReference.addItem("")
        self.filterReference.setObjectName(u"filterReference")

        self.leftPane.addWidget(self.filterReference, 0, 2, 1, 1)

        self.limitsButton = QToolButton(self.centralwidget)
        self.limitsButton.setObjectName(u"limitsButton")

        self.leftPane.addWidget(self.limitsButton, 0, 7, 1, 1)

        self.signalReference = QComboBox(self.centralwidget)
        self.signalReference.addItem("")
        self.signalReference.setObjectName(u"signalReference")

        self.leftPane.addWidget(self.signalReference, 0, 1, 1, 1)

        self.referenceLabel = QLabel(self.centralwidget)
        self.referenceLabel.setObjectName(u"referenceLabel")

        self.leftPane.addWidget(self.referenceLabel, 0, 0, 1, 1)

        self.chartSplitter = QSplitter(self.centralwidget)
        self.chartSplitter.setObjectName(u"chartSplitter")
        self.chartSplitter.setLineWidth(1)
        self.chartSplitter.setOrientation(Qt.Vertical)
        self.mainChartContainer = QFrame(self.chartSplitter)
        self.mainChartContainer.setObjectName(u"mainChartContainer")
        self.mainChartLayout = QHBoxLayout(self.mainChartContainer)
        self.mainChartLayout.setObjectName(u"mainChartLayout")
        self.mainChartLeftTools = QFrame(self.mainChartContainer)
        self.mainChartLeftTools.setObjectName(u"mainChartLeftTools")
        self.mainChartLeftTools.setLineWidth(0)
        self.mainChartLeftToolLayout = QVBoxLayout(self.mainChartLeftTools)
        self.mainChartLeftToolLayout.setSpacing(0)
        self.mainChartLeftToolLayout.setObjectName(u"mainChartLeftToolLayout")
        self.mainChartLeftToolLayout.setContentsMargins(0, 0, 0, 0)
        self.mainChartLeftToolTopLayout = QVBoxLayout()
        self.mainChartLeftToolTopLayout.setObjectName(u"mainChartLeftToolTopLayout")
        self.y1MaxPlus10Button = QToolButton(self.mainChartLeftTools)
        self.y1MaxPlus10Button.setObjectName(u"y1MaxPlus10Button")

        self.mainChartLeftToolTopLayout.addWidget(self.y1MaxPlus10Button)

        self.y1MaxPlus5Button = QToolButton(self.mainChartLeftTools)
        self.y1MaxPlus5Button.setObjectName(u"y1MaxPlus5Button")

        self.mainChartLeftToolTopLayout.addWidget(self.y1MaxPlus5Button)

        self.y1MaxMinus5Button = QToolButton(self.mainChartLeftTools)
        self.y1MaxMinus5Button.setObjectName(u"y1MaxMinus5Button")

        self.mainChartLeftToolTopLayout.addWidget(self.y1MaxMinus5Button)

        self.y1MaxMinus10Button = QToolButton(self.mainChartLeftTools)
        self.y1MaxMinus10Button.setObjectName(u"y1MaxMinus10Button")

        self.mainChartLeftToolTopLayout.addWidget(self.y1MaxMinus10Button)

        self.leftToolTopSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.mainChartLeftToolTopLayout.addItem(self.leftToolTopSpacer)


        self.mainChartLeftToolLayout.addLayout(self.mainChartLeftToolTopLayout)

        self.mainChartLeftToolMidLayout = QVBoxLayout()
        self.mainChartLeftToolMidLayout.setObjectName(u"mainChartLeftToolMidLayout")
        self.leftToolMidTopSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.mainChartLeftToolMidLayout.addItem(self.leftToolMidTopSpacer)

        self.y1AutoOnButton = QToolButton(self.mainChartLeftTools)
        self.y1AutoOnButton.setObjectName(u"y1AutoOnButton")
        self.y1AutoOnButton.setCheckable(True)

        self.mainChartLeftToolMidLayout.addWidget(self.y1AutoOnButton)

        self.y1AutoOffButton = QToolButton(self.mainChartLeftTools)
        self.y1AutoOffButton.setObjectName(u"y1AutoOffButton")
        self.y1AutoOffButton.setCheckable(True)

        self.mainChartLeftToolMidLayout.addWidget(self.y1AutoOffButton)

        self.leftToolMidBottomSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.mainChartLeftToolMidLayout.addItem(self.leftToolMidBottomSpacer)


        self.mainChartLeftToolLayout.addLayout(self.mainChartLeftToolMidLayout)

        self.mainChartLeftToolBottomLayout = QVBoxLayout()
        self.mainChartLeftToolBottomLayout.setObjectName(u"mainChartLeftToolBottomLayout")
        self.leftToolBottomSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.mainChartLeftToolBottomLayout.addItem(self.leftToolBottomSpacer)

        self.y1MinPlus10Button = QToolButton(self.mainChartLeftTools)
        self.y1MinPlus10Button.setObjectName(u"y1MinPlus10Button")

        self.mainChartLeftToolBottomLayout.addWidget(self.y1MinPlus10Button)

        self.y1MinPlus5Button = QToolButton(self.mainChartLeftTools)
        self.y1MinPlus5Button.setObjectName(u"y1MinPlus5Button")

        self.mainChartLeftToolBottomLayout.addWidget(self.y1MinPlus5Button)

        self.y1MinMinus5Button = QToolButton(self.mainChartLeftTools)
        self.y1MinMinus5Button.setObjectName(u"y1MinMinus5Button")

        self.mainChartLeftToolBottomLayout.addWidget(self.y1MinMinus5Button)

        self.y1MinMinus10Button = QToolButton(self.mainChartLeftTools)
        self.y1MinMinus10Button.setObjectName(u"y1MinMinus10Button")

        self.mainChartLeftToolBottomLayout.addWidget(self.y1MinMinus10Button)


        self.mainChartLeftToolLayout.addLayout(self.mainChartLeftToolBottomLayout)


        self.mainChartLayout.addWidget(self.mainChartLeftTools)

        self.mainChart = MplWidget(self.mainChartContainer)
        self.mainChart.setObjectName(u"mainChart")

        self.mainChartLayout.addWidget(self.mainChart)

        self.mainChartRightTools = QFrame(self.mainChartContainer)
        self.mainChartRightTools.setObjectName(u"mainChartRightTools")
        self.mainChartRightTools.setLineWidth(0)
        self.mainChartRightToolLayout = QVBoxLayout(self.mainChartRightTools)
        self.mainChartRightToolLayout.setSpacing(0)
        self.mainChartRightToolLayout.setObjectName(u"mainChartRightToolLayout")
        self.mainChartRightToolLayout.setContentsMargins(0, 0, 0, 0)
        self.mainChartRightToolTopLayout = QVBoxLayout()
        self.mainChartRightToolTopLayout.setObjectName(u"mainChartRightToolTopLayout")
        self.y2MaxPlus10Button = QToolButton(self.mainChartRightTools)
        self.y2MaxPlus10Button.setObjectName(u"y2MaxPlus10Button")

        self.mainChartRightToolTopLayout.addWidget(self.y2MaxPlus10Button)

        self.y2MaxPlus5Button = QToolButton(self.mainChartRightTools)
        self.y2MaxPlus5Button.setObjectName(u"y2MaxPlus5Button")

        self.mainChartRightToolTopLayout.addWidget(self.y2MaxPlus5Button)

        self.y2MaxMinus5Button = QToolButton(self.mainChartRightTools)
        self.y2MaxMinus5Button.setObjectName(u"y2MaxMinus5Button")

        self.mainChartRightToolTopLayout.addWidget(self.y2MaxMinus5Button)

        self.y2MaxMinus10Button = QToolButton(self.mainChartRightTools)
        self.y2MaxMinus10Button.setObjectName(u"y2MaxMinus10Button")

        self.mainChartRightToolTopLayout.addWidget(self.y2MaxMinus10Button)

        self.rightToolTopSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.mainChartRightToolTopLayout.addItem(self.rightToolTopSpacer)


        self.mainChartRightToolLayout.addLayout(self.mainChartRightToolTopLayout)

        self.mainChartRightToolMidLayout = QVBoxLayout()
        self.mainChartRightToolMidLayout.setObjectName(u"mainChartRightToolMidLayout")
        self.rightToolMidTopSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.mainChartRightToolMidLayout.addItem(self.rightToolMidTopSpacer)

        self.y2AutoOnButton = QToolButton(self.mainChartRightTools)
        self.y2AutoOnButton.setObjectName(u"y2AutoOnButton")
        self.y2AutoOnButton.setCheckable(True)

        self.mainChartRightToolMidLayout.addWidget(self.y2AutoOnButton)

        self.y2AutoOffButton = QToolButton(self.mainChartRightTools)
        self.y2AutoOffButton.setObjectName(u"y2AutoOffButton")
        self.y2AutoOffButton.setCheckable(True)

        self.mainChartRightToolMidLayout.addWidget(self.y2AutoOffButton)

        self.rightToolMidBottomSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.mainChartRightToolMidLayout.addItem(self.rightToolMidBottomSpacer)


        self.mainChartRightToolLayout.addLayout(self.mainChartRightToolMidLayout)

        self.mainChartRightToolBottomLayout = QVBoxLayout()
        self.mainChartRightToolBottomLayout.setObjectName(u"mainChartRightToolBottomLayout")
        self.rightToolBottomSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.mainChartRightToolBottomLayout.addItem(self.rightToolBottomSpacer)

        self.y2MinPlus10Button = QToolButton(self.mainChartRightTools)
        self.y2MinPlus10Button.setObjectName(u"y2MinPlus10Button")

        self.mainChartRightToolBottomLayout.addWidget(self.y2MinPlus10Button)

        self.y2MinPlus5Button = QToolButton(self.mainChartRightTools)
        self.y2MinPlus5Button.setObjectName(u"y2MinPlus5Button")

        self.mainChartRightToolBottomLayout.addWidget(self.y2MinPlus5Button)

        self.y2MinMinus5Button = QToolButton(self.mainChartRightTools)
        self.y2MinMinus5Button.setObjectName(u"y2MinMinus5Button")

        self.mainChartRightToolBottomLayout.addWidget(self.y2MinMinus5Button)

        self.y2MinMinus10Button = QToolButton(self.mainChartRightTools)
        self.y2MinMinus10Button.setObjectName(u"y2MinMinus10Button")

        self.mainChartRightToolBottomLayout.addWidget(self.y2MinMinus10Button)


        self.mainChartRightToolLayout.addLayout(self.mainChartRightToolBottomLayout)


        self.mainChartLayout.addWidget(self.mainChartRightTools)

        self.mainChartLayout.setStretch(1, 1)
        self.chartSplitter.addWidget(self.mainChartContainer)
        self.waveformContainer = QFrame(self.chartSplitter)
        self.waveformContainer.setObjectName(u"waveformContainer")
        self.waveformLayout = QHBoxLayout(self.waveformContainer)
        self.waveformLayout.setObjectName(u"waveformLayout")
        self.waveformControls = QGridLayout()
        self.waveformControls.setObjectName(u"waveformControls")
        self.checkboxContainer = QHBoxLayout()
        self.checkboxContainer.setObjectName(u"checkboxContainer")
        self.waveformIsFiltered = QCheckBox(self.waveformContainer)
        self.waveformIsFiltered.setObjectName(u"waveformIsFiltered")

        self.checkboxContainer.addWidget(self.waveformIsFiltered)

        self.hardClipWaveform = QCheckBox(self.waveformContainer)
        self.hardClipWaveform.setObjectName(u"hardClipWaveform")

        self.checkboxContainer.addWidget(self.hardClipWaveform)

        self.saveWaveformChartButton = QToolButton(self.waveformContainer)
        self.saveWaveformChartButton.setObjectName(u"saveWaveformChartButton")

        self.checkboxContainer.addWidget(self.saveWaveformChartButton)


        self.waveformControls.addLayout(self.checkboxContainer, 3, 0, 1, 2)

        self.signalSelectorLabel = QLabel(self.waveformContainer)
        self.signalSelectorLabel.setObjectName(u"signalSelectorLabel")

        self.waveformControls.addWidget(self.signalSelectorLabel, 1, 0, 1, 1)

        self.headroom = QDoubleSpinBox(self.waveformContainer)
        self.headroom.setObjectName(u"headroom")
        self.headroom.setEnabled(False)
        self.headroom.setMinimum(-120.000000000000000)
        self.headroom.setMaximum(120.000000000000000)
        self.headroom.setSingleStep(0.010000000000000)

        self.waveformControls.addWidget(self.headroom, 4, 1, 1, 1)

        self.sourceFileLayout = QHBoxLayout()
        self.sourceFileLayout.setObjectName(u"sourceFileLayout")
        self.sourceFile = QLineEdit(self.waveformContainer)
        self.sourceFile.setObjectName(u"sourceFile")
        self.sourceFile.setReadOnly(True)

        self.sourceFileLayout.addWidget(self.sourceFile)

        self.loadSignalButton = QToolButton(self.waveformContainer)
        self.loadSignalButton.setObjectName(u"loadSignalButton")

        self.sourceFileLayout.addWidget(self.loadSignalButton)


        self.waveformControls.addLayout(self.sourceFileLayout, 2, 1, 1, 1)

        self.sourceFileLabel = QLabel(self.waveformContainer)
        self.sourceFileLabel.setObjectName(u"sourceFileLabel")

        self.waveformControls.addWidget(self.sourceFileLabel, 2, 0, 1, 1)

        self.startTime = QTimeEdit(self.waveformContainer)
        self.startTime.setObjectName(u"startTime")

        self.waveformControls.addWidget(self.startTime, 6, 1, 1, 1)

        self.analysisLayout = QHBoxLayout()
        self.analysisLayout.setObjectName(u"analysisLayout")
        self.rmsLevel = QDoubleSpinBox(self.waveformContainer)
        self.rmsLevel.setObjectName(u"rmsLevel")
        self.rmsLevel.setEnabled(False)
        self.rmsLevel.setMinimum(-120.000000000000000)
        self.rmsLevel.setMaximum(120.000000000000000)
        self.rmsLevel.setSingleStep(0.010000000000000)

        self.analysisLayout.addWidget(self.rmsLevel)

        self.crestFactor = QDoubleSpinBox(self.waveformContainer)
        self.crestFactor.setObjectName(u"crestFactor")
        self.crestFactor.setEnabled(False)
        self.crestFactor.setMinimum(-120.000000000000000)
        self.crestFactor.setMaximum(120.000000000000000)
        self.crestFactor.setSingleStep(0.010000000000000)

        self.analysisLayout.addWidget(self.crestFactor)


        self.waveformControls.addLayout(self.analysisLayout, 5, 1, 1, 1)

        self.analysisLabel = QLabel(self.waveformContainer)
        self.analysisLabel.setObjectName(u"analysisLabel")

        self.waveformControls.addWidget(self.analysisLabel, 5, 0, 1, 1)

        self.bmHeaderLabel = QLabel(self.waveformContainer)
        self.bmHeaderLabel.setObjectName(u"bmHeaderLabel")
        font = QFont()
        font.setBold(True)
        font.setWeight(75)
        self.bmHeaderLabel.setFont(font)
        self.bmHeaderLabel.setFrameShape(QFrame.Box)
        self.bmHeaderLabel.setFrameShadow(QFrame.Sunken)
        self.bmHeaderLabel.setAlignment(Qt.AlignCenter)

        self.waveformControls.addWidget(self.bmHeaderLabel, 9, 0, 1, 2)

        self.signalSelector = QComboBox(self.waveformContainer)
        self.signalSelector.setObjectName(u"signalSelector")

        self.waveformControls.addWidget(self.signalSelector, 1, 1, 1, 1)

        self.yRangeLayout = QHBoxLayout()
        self.yRangeLayout.setObjectName(u"yRangeLayout")
        self.yMin = QDoubleSpinBox(self.waveformContainer)
        self.yMin.setObjectName(u"yMin")
        self.yMin.setMinimum(-1.000000000000000)
        self.yMin.setMaximum(1.000000000000000)
        self.yMin.setSingleStep(0.010000000000000)
        self.yMin.setValue(-1.000000000000000)

        self.yRangeLayout.addWidget(self.yMin)

        self.yMax = QDoubleSpinBox(self.waveformContainer)
        self.yMax.setObjectName(u"yMax")
        self.yMax.setMinimum(-1.000000000000000)
        self.yMax.setMaximum(1.000000000000000)
        self.yMax.setSingleStep(0.010000000000000)
        self.yMax.setValue(1.000000000000000)

        self.yRangeLayout.addWidget(self.yMax)


        self.waveformControls.addLayout(self.yRangeLayout, 8, 1, 1, 1)

        self.label_2 = QLabel(self.waveformContainer)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setFont(font)
        self.label_2.setFrameShape(QFrame.Box)
        self.label_2.setFrameShadow(QFrame.Sunken)
        self.label_2.setAlignment(Qt.AlignCenter)

        self.waveformControls.addWidget(self.label_2, 0, 0, 1, 2)

        self.endTime = QTimeEdit(self.waveformContainer)
        self.endTime.setObjectName(u"endTime")

        self.waveformControls.addWidget(self.endTime, 7, 1, 1, 1)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.compareSpectrumButton = QToolButton(self.waveformContainer)
        self.compareSpectrumButton.setObjectName(u"compareSpectrumButton")

        self.horizontalLayout.addWidget(self.compareSpectrumButton)

        self.showSpectrumButton = QToolButton(self.waveformContainer)
        self.showSpectrumButton.setObjectName(u"showSpectrumButton")

        self.horizontalLayout.addWidget(self.showSpectrumButton)

        self.hideSpectrumButton = QToolButton(self.waveformContainer)
        self.hideSpectrumButton.setObjectName(u"hideSpectrumButton")

        self.horizontalLayout.addWidget(self.hideSpectrumButton)

        self.filteredSpectrumLimitsButton = QToolButton(self.waveformContainer)
        self.filteredSpectrumLimitsButton.setObjectName(u"filteredSpectrumLimitsButton")

        self.horizontalLayout.addWidget(self.filteredSpectrumLimitsButton)

        self.zoomInButton = QToolButton(self.waveformContainer)
        self.zoomInButton.setObjectName(u"zoomInButton")

        self.horizontalLayout.addWidget(self.zoomInButton)

        self.zoomOutButton = QToolButton(self.waveformContainer)
        self.zoomOutButton.setObjectName(u"zoomOutButton")

        self.horizontalLayout.addWidget(self.zoomOutButton)

        self.showStatsButton = QToolButton(self.waveformContainer)
        self.showStatsButton.setObjectName(u"showStatsButton")

        self.horizontalLayout.addWidget(self.showStatsButton)


        self.waveformControls.addLayout(self.horizontalLayout, 14, 0, 1, 2)

        self.bmClippingLabel = QLabel(self.waveformContainer)
        self.bmClippingLabel.setObjectName(u"bmClippingLabel")

        self.waveformControls.addWidget(self.bmClippingLabel, 11, 0, 1, 1)

        self.yRangeLabel = QLabel(self.waveformContainer)
        self.yRangeLabel.setObjectName(u"yRangeLabel")

        self.waveformControls.addWidget(self.yRangeLabel, 8, 0, 1, 1)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.waveformControls.addItem(self.verticalSpacer, 15, 0, 1, 2)

        self.headroomLabel = QLabel(self.waveformContainer)
        self.headroomLabel.setObjectName(u"headroomLabel")

        self.waveformControls.addWidget(self.headroomLabel, 4, 0, 1, 1)

        self.bmHeadroom = QComboBox(self.waveformContainer)
        self.bmHeadroom.addItem("")
        self.bmHeadroom.addItem("")
        self.bmHeadroom.addItem("")
        self.bmHeadroom.addItem("")
        self.bmHeadroom.addItem("")
        self.bmHeadroom.setObjectName(u"bmHeadroom")

        self.waveformControls.addWidget(self.bmHeadroom, 12, 1, 1, 1)

        self.endTimeLabel = QLabel(self.waveformContainer)
        self.endTimeLabel.setObjectName(u"endTimeLabel")

        self.waveformControls.addWidget(self.endTimeLabel, 7, 0, 1, 1)

        self.bmLayout = QHBoxLayout()
        self.bmLayout.setObjectName(u"bmLayout")
        self.bmlpfPosition = QComboBox(self.waveformContainer)
        self.bmlpfPosition.setObjectName(u"bmlpfPosition")

        self.bmLayout.addWidget(self.bmlpfPosition)

        self.bmhpfOn = QCheckBox(self.waveformContainer)
        self.bmhpfOn.setObjectName(u"bmhpfOn")

        self.bmLayout.addWidget(self.bmhpfOn)


        self.waveformControls.addLayout(self.bmLayout, 10, 1, 1, 1)

        self.startTimeLabel = QLabel(self.waveformContainer)
        self.startTimeLabel.setObjectName(u"startTimeLabel")

        self.waveformControls.addWidget(self.startTimeLabel, 6, 0, 1, 1)

        self.bmlpfPositionLabel = QLabel(self.waveformContainer)
        self.bmlpfPositionLabel.setObjectName(u"bmlpfPositionLabel")

        self.waveformControls.addWidget(self.bmlpfPositionLabel, 10, 0, 1, 1)

        self.bmClippingOptions = QHBoxLayout()
        self.bmClippingOptions.setObjectName(u"bmClippingOptions")
        self.bmClipBefore = QCheckBox(self.waveformContainer)
        self.bmClipBefore.setObjectName(u"bmClipBefore")

        self.bmClippingOptions.addWidget(self.bmClipBefore)

        self.bmClipAfter = QCheckBox(self.waveformContainer)
        self.bmClipAfter.setObjectName(u"bmClipAfter")

        self.bmClippingOptions.addWidget(self.bmClipAfter)


        self.waveformControls.addLayout(self.bmClippingOptions, 11, 1, 1, 1)

        self.bmHeadroomLabel = QLabel(self.waveformContainer)
        self.bmHeadroomLabel.setObjectName(u"bmHeadroomLabel")

        self.waveformControls.addWidget(self.bmHeadroomLabel, 12, 0, 1, 1)

        self.label_3 = QLabel(self.waveformContainer)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setFont(font)
        self.label_3.setFrameShape(QFrame.Box)
        self.label_3.setFrameShadow(QFrame.Sunken)
        self.label_3.setAlignment(Qt.AlignCenter)

        self.waveformControls.addWidget(self.label_3, 13, 0, 1, 2)


        self.waveformLayout.addLayout(self.waveformControls)

        self.waveformChart = PlotWidgetWithDateAxis(self.waveformContainer)
        self.waveformChart.setObjectName(u"waveformChart")

        self.waveformLayout.addWidget(self.waveformChart)

        self.filteredSpectrumChart = MplWidget(self.waveformContainer)
        self.filteredSpectrumChart.setObjectName(u"filteredSpectrumChart")

        self.waveformLayout.addWidget(self.filteredSpectrumChart)

        self.waveformLayout.setStretch(1, 1)
        self.waveformLayout.setStretch(2, 1)
        self.chartSplitter.addWidget(self.waveformContainer)

        self.leftPane.addWidget(self.chartSplitter, 1, 0, 1, 9)

        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")

        self.leftPane.addWidget(self.label, 0, 3, 1, 1)

        self.octaveSmoothing = QComboBox(self.centralwidget)
        self.octaveSmoothing.addItem("")
        self.octaveSmoothing.addItem("")
        self.octaveSmoothing.addItem("")
        self.octaveSmoothing.addItem("")
        self.octaveSmoothing.addItem("")
        self.octaveSmoothing.addItem("")
        self.octaveSmoothing.addItem("")
        self.octaveSmoothing.addItem("")
        self.octaveSmoothing.setObjectName(u"octaveSmoothing")

        self.leftPane.addWidget(self.octaveSmoothing, 0, 4, 1, 1)

        self.smoothAllSignals = QCheckBox(self.centralwidget)
        self.smoothAllSignals.setObjectName(u"smoothAllSignals")

        self.leftPane.addWidget(self.smoothAllSignals, 0, 5, 1, 1)

        self.leftPane.setColumnStretch(1, 1)
        self.leftPane.setColumnStretch(2, 1)

        self.panes.addLayout(self.leftPane)

        self.rightPane = QVBoxLayout()
        self.rightPane.setObjectName(u"rightPane")
        self.signalsLabel = QLabel(self.centralwidget)
        self.signalsLabel.setObjectName(u"signalsLabel")
        self.signalsLabel.setFont(font)
        self.signalsLabel.setFrameShape(QFrame.Box)
        self.signalsLabel.setFrameShadow(QFrame.Sunken)
        self.signalsLabel.setAlignment(Qt.AlignCenter)

        self.rightPane.addWidget(self.signalsLabel)

        self.signalButtonsLayout = QHBoxLayout()
        self.signalButtonsLayout.setObjectName(u"signalButtonsLayout")
        self.addSignalButton = QToolButton(self.centralwidget)
        self.addSignalButton.setObjectName(u"addSignalButton")
        self.addSignalButton.setProperty("flat", False)

        self.signalButtonsLayout.addWidget(self.addSignalButton)

        self.linkSignalButton = QToolButton(self.centralwidget)
        self.linkSignalButton.setObjectName(u"linkSignalButton")
        self.linkSignalButton.setEnabled(False)

        self.signalButtonsLayout.addWidget(self.linkSignalButton)

        self.deleteSignalButton = QToolButton(self.centralwidget)
        self.deleteSignalButton.setObjectName(u"deleteSignalButton")
        self.deleteSignalButton.setEnabled(False)

        self.signalButtonsLayout.addWidget(self.deleteSignalButton)

        self.clearSignalsButton = QToolButton(self.centralwidget)
        self.clearSignalsButton.setObjectName(u"clearSignalsButton")
        self.clearSignalsButton.setEnabled(False)

        self.signalButtonsLayout.addWidget(self.clearSignalsButton)


        self.rightPane.addLayout(self.signalButtonsLayout)

        self.signalView = QTableView(self.centralwidget)
        self.signalView.setObjectName(u"signalView")

        self.rightPane.addWidget(self.signalView)

        self.filtersLabel = QLabel(self.centralwidget)
        self.filtersLabel.setObjectName(u"filtersLabel")
        self.filtersLabel.setFont(font)
        self.filtersLabel.setFrameShape(QFrame.Box)
        self.filtersLabel.setFrameShadow(QFrame.Sunken)
        self.filtersLabel.setAlignment(Qt.AlignCenter)

        self.rightPane.addWidget(self.filtersLabel)

        self.filterToolbarLayout = QHBoxLayout()
        self.filterToolbarLayout.setObjectName(u"filterToolbarLayout")
        self.preset1Button = QToolButton(self.centralwidget)
        self.preset1Button.setObjectName(u"preset1Button")
        self.preset1Button.setEnabled(False)

        self.filterToolbarLayout.addWidget(self.preset1Button)

        self.preset2Button = QToolButton(self.centralwidget)
        self.preset2Button.setObjectName(u"preset2Button")
        self.preset2Button.setEnabled(False)

        self.filterToolbarLayout.addWidget(self.preset2Button)

        self.preset3Button = QToolButton(self.centralwidget)
        self.preset3Button.setObjectName(u"preset3Button")
        self.preset3Button.setEnabled(False)

        self.filterToolbarLayout.addWidget(self.preset3Button)

        self.addFilterButton = QToolButton(self.centralwidget)
        self.addFilterButton.setObjectName(u"addFilterButton")
        self.addFilterButton.setEnabled(True)

        self.filterToolbarLayout.addWidget(self.addFilterButton)

        self.editFilterButton = QToolButton(self.centralwidget)
        self.editFilterButton.setObjectName(u"editFilterButton")
        self.editFilterButton.setEnabled(False)
        self.editFilterButton.setProperty("flat", False)

        self.filterToolbarLayout.addWidget(self.editFilterButton)

        self.deleteFilterButton = QToolButton(self.centralwidget)
        self.deleteFilterButton.setObjectName(u"deleteFilterButton")
        self.deleteFilterButton.setEnabled(False)

        self.filterToolbarLayout.addWidget(self.deleteFilterButton)

        self.clearFiltersButton = QToolButton(self.centralwidget)
        self.clearFiltersButton.setObjectName(u"clearFiltersButton")
        self.clearFiltersButton.setEnabled(False)

        self.filterToolbarLayout.addWidget(self.clearFiltersButton)


        self.rightPane.addLayout(self.filterToolbarLayout)

        self.filterView = QTableView(self.centralwidget)
        self.filterView.setObjectName(u"filterView")

        self.rightPane.addWidget(self.filterView)

        self.selectorsLayout = QGridLayout()
        self.selectorsLayout.setObjectName(u"selectorsLayout")
        self.showSignalsLabel = QLabel(self.centralwidget)
        self.showSignalsLabel.setObjectName(u"showSignalsLabel")

        self.selectorsLayout.addWidget(self.showSignalsLabel, 0, 0, 1, 1)

        self.showFiltersLabel = QLabel(self.centralwidget)
        self.showFiltersLabel.setObjectName(u"showFiltersLabel")

        self.selectorsLayout.addWidget(self.showFiltersLabel, 0, 1, 1, 1)

        self.showFiltersLabel1 = QLabel(self.centralwidget)
        self.showFiltersLabel1.setObjectName(u"showFiltersLabel1")

        self.selectorsLayout.addWidget(self.showFiltersLabel1, 0, 2, 1, 1)

        self.showSignals = QComboBox(self.centralwidget)
        self.showSignals.setObjectName(u"showSignals")

        self.selectorsLayout.addWidget(self.showSignals, 1, 0, 1, 1)

        self.showFilteredSignals = QComboBox(self.centralwidget)
        self.showFilteredSignals.setObjectName(u"showFilteredSignals")

        self.selectorsLayout.addWidget(self.showFilteredSignals, 1, 1, 1, 1)

        self.showFilters = QComboBox(self.centralwidget)
        self.showFilters.setObjectName(u"showFilters")

        self.selectorsLayout.addWidget(self.showFilters, 1, 2, 1, 1)

        self.equalEnergyTilt = QCheckBox(self.centralwidget)
        self.equalEnergyTilt.setObjectName(u"equalEnergyTilt")

        self.selectorsLayout.addWidget(self.equalEnergyTilt, 2, 2, 1, 1)

        self.showLegend = QCheckBox(self.centralwidget)
        self.showLegend.setObjectName(u"showLegend")
        self.showLegend.setChecked(True)

        self.selectorsLayout.addWidget(self.showLegend, 2, 0, 1, 1)


        self.rightPane.addLayout(self.selectorsLayout)


        self.panes.addLayout(self.rightPane)

        self.panes.setStretch(0, 3)
        self.panes.setStretch(1, 1)

        self.widgetGridLayout.addLayout(self.panes, 0, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1569, 21))
        self.menuHelp = QMenu(self.menubar)
        self.menuHelp.setObjectName(u"menuHelp")
        self.menuSettings = QMenu(self.menubar)
        self.menuSettings.setObjectName(u"menuSettings")
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName(u"menuFile")
        self.menuPresets = QMenu(self.menuFile)
        self.menuPresets.setObjectName(u"menuPresets")
        self.menuLoad = QMenu(self.menuPresets)
        self.menuLoad.setObjectName(u"menuLoad")
        self.menu_Clear = QMenu(self.menuPresets)
        self.menu_Clear.setObjectName(u"menu_Clear")
        self.menu_Store = QMenu(self.menuPresets)
        self.menu_Store.setObjectName(u"menu_Store")
        self.menu_Tools = QMenu(self.menubar)
        self.menu_Tools.setObjectName(u"menu_Tools")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuSettings.menuAction())
        self.menubar.addAction(self.menu_Tools.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())
        self.menuHelp.addAction(self.actionUser_Guide)
        self.menuHelp.addAction(self.actionShow_Logs)
        self.menuHelp.addSeparator()
        self.menuHelp.addAction(self.actionRelease_Notes)
        self.menuHelp.addAction(self.actionAbout)
        self.menuSettings.addAction(self.actionPreferences)
        self.menuSettings.addSeparator()
        self.menuFile.addAction(self.action_Load_Project)
        self.menuFile.addAction(self.action_Save_Project)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionLoad_Signal)
        self.menuFile.addAction(self.actionSave_Signal)
        self.menuFile.addAction(self.actionClear_Signals)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionLoad_Filter)
        self.menuFile.addAction(self.actionSave_Filter)
        self.menuFile.addAction(self.actionClear_Filters)
        self.menuFile.addAction(self.menuPresets.menuAction())
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionAdd_BEQ_Filter)
        self.menuFile.addAction(self.actionExport_BEQ_Filter)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExport_FRD)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionSave_Chart)
        self.menuFile.addAction(self.actionSave_Report)
        self.menuPresets.addAction(self.menuLoad.menuAction())
        self.menuPresets.addAction(self.menu_Store.menuAction())
        self.menuPresets.addAction(self.menu_Clear.menuAction())
        self.menuLoad.addAction(self.action_load_preset_1)
        self.menuLoad.addAction(self.action_load_preset_2)
        self.menuLoad.addAction(self.action_load_preset_3)
        self.menu_Clear.addAction(self.action_clear_preset_1)
        self.menu_Clear.addAction(self.action_clear_preset_2)
        self.menu_Clear.addAction(self.action_clear_preset_3)
        self.menu_Store.addAction(self.action_store_preset_1)
        self.menu_Store.addAction(self.action_store_preset_2)
        self.menu_Store.addAction(self.action_store_preset_3)
        self.menu_Tools.addAction(self.actionExtract_Audio)
        self.menu_Tools.addAction(self.action_Batch_Extract)
        self.menu_Tools.addAction(self.action_Remux_Audio)
        self.menu_Tools.addSeparator()
        self.menu_Tools.addAction(self.actionAnalyse_Audio)
        self.menu_Tools.addSeparator()
        self.menu_Tools.addAction(self.actionExport_Biquad)
        self.menu_Tools.addAction(self.actionMerge_Minidsp_XML)
        self.menu_Tools.addAction(self.actionCreate_AVS_Post)
        self.menu_Tools.addSeparator()
        self.menu_Tools.addAction(self.actionSync_with_HTP_1)

        self.retranslateUi(MainWindow)
        self.addFilterButton.clicked.connect(MainWindow.addFilter)
        self.deleteFilterButton.clicked.connect(MainWindow.deleteFilter)
        self.editFilterButton.clicked.connect(MainWindow.editFilter)
        self.addSignalButton.clicked.connect(MainWindow.addSignal)
        self.deleteSignalButton.clicked.connect(MainWindow.deleteSignal)
        self.signalReference.currentIndexChanged.connect(MainWindow.normaliseSignalMagnitude)
        self.limitsButton.clicked.connect(MainWindow.showLimits)
        self.filterReference.currentIndexChanged.connect(MainWindow.normaliseFilterMagnitude)
        self.showValuesButton.clicked.connect(MainWindow.showValues)
        self.showLegend.clicked.connect(MainWindow.changeLegendVisibility)
        self.showFilters.currentTextChanged.connect(MainWindow.changeFilterVisibility)
        self.preset1Button.clicked.connect(MainWindow.applyPreset1)
        self.preset2Button.clicked.connect(MainWindow.applyPreset2)
        self.preset3Button.clicked.connect(MainWindow.applyPreset3)
        self.showSignals.currentTextChanged.connect(MainWindow.changeSignalVisibility)
        self.showFilteredSignals.currentTextChanged.connect(MainWindow.changeSignalFilterVisibility)
        self.linkSignalButton.clicked.connect(MainWindow.linkSignals)
        self.equalEnergyTilt.clicked.connect(MainWindow.toggleTilt)
        self.octaveSmoothing.currentTextChanged.connect(MainWindow.smoothSignals)
        self.smoothAllSignals.clicked.connect(MainWindow.smoothSignals)
        self.clearSignalsButton.clicked.connect(MainWindow.clearSignals)
        self.clearFiltersButton.clicked.connect(MainWindow.clearFilters)
        self.y2MaxPlus10Button.clicked.connect(MainWindow.y2_max_plus_10)
        self.y2MaxMinus10Button.clicked.connect(MainWindow.y2_max_minus_10)
        self.y1MaxPlus10Button.clicked.connect(MainWindow.y1_max_plus_10)
        self.y1MaxPlus5Button.clicked.connect(MainWindow.y1_max_plus_5)
        self.y1AutoOnButton.clicked.connect(MainWindow.y1_auto_on)
        self.y1AutoOffButton.clicked.connect(MainWindow.y1_auto_off)
        self.y1MaxMinus5Button.clicked.connect(MainWindow.y1_max_minus_5)
        self.y1MaxMinus10Button.clicked.connect(MainWindow.y1_max_minus_10)
        self.y2MaxPlus5Button.clicked.connect(MainWindow.y2_max_plus_5)
        self.y2AutoOnButton.clicked.connect(MainWindow.y2_auto_on)
        self.y2AutoOffButton.clicked.connect(MainWindow.y2_auto_off)
        self.y2MaxMinus5Button.clicked.connect(MainWindow.y2_max_minus_5)
        self.y1MinPlus10Button.clicked.connect(MainWindow.y1_min_plus_10)
        self.y1MinPlus5Button.clicked.connect(MainWindow.y1_min_plus_5)
        self.y1MinMinus5Button.clicked.connect(MainWindow.y1_min_minus_5)
        self.y1MinMinus10Button.clicked.connect(MainWindow.y1_min_minus_10)
        self.y2MinPlus10Button.clicked.connect(MainWindow.y2_min_plus_10)
        self.y2MinPlus5Button.clicked.connect(MainWindow.y2_min_plus_5)
        self.y2MinMinus5Button.clicked.connect(MainWindow.y2_min_minus_5)
        self.y2MinMinus10Button.clicked.connect(MainWindow.y2_min_minus_10)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"BEQ Designer", None))
        self.actionShow_Logs.setText(QCoreApplication.translate("MainWindow", u"Show &Logs", None))
#if QT_CONFIG(shortcut)
        self.actionShow_Logs.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+L", None))
#endif // QT_CONFIG(shortcut)
        self.actionPreferences.setText(QCoreApplication.translate("MainWindow", u"&Preferences", None))
#if QT_CONFIG(shortcut)
        self.actionPreferences.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+P", None))
#endif // QT_CONFIG(shortcut)
        self.actionExtract_Audio.setText(QCoreApplication.translate("MainWindow", u"&Extract Audio", None))
#if QT_CONFIG(shortcut)
        self.actionExtract_Audio.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+E", None))
#endif // QT_CONFIG(shortcut)
        self.actionPresets.setText(QCoreApplication.translate("MainWindow", u"Presets", None))
        self.actionClear_Preset_3.setText(QCoreApplication.translate("MainWindow", u"Clear Preset 3", None))
        self.actionSave_Chart.setText(QCoreApplication.translate("MainWindow", u"Save &Chart", None))
#if QT_CONFIG(shortcut)
        self.actionSave_Chart.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+Shift+C", None))
#endif // QT_CONFIG(shortcut)
        self.actionExport_Biquad.setText(QCoreApplication.translate("MainWindow", u"Export &Biquad", None))
#if QT_CONFIG(shortcut)
        self.actionExport_Biquad.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+B", None))
#endif // QT_CONFIG(shortcut)
        self.actionSave_Filter.setText(QCoreApplication.translate("MainWindow", u"Save &Filter", None))
#if QT_CONFIG(shortcut)
        self.actionSave_Filter.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+Alt+S", None))
#endif // QT_CONFIG(shortcut)
        self.actionLoad_Filter.setText(QCoreApplication.translate("MainWindow", u"Load Filte&r", None))
#if QT_CONFIG(shortcut)
        self.actionLoad_Filter.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+Alt+L", None))
#endif // QT_CONFIG(shortcut)
        self.action_load_preset_1.setText(QCoreApplication.translate("MainWindow", u"&1", None))
        self.action_load_preset_2.setText(QCoreApplication.translate("MainWindow", u"&2", None))
        self.action_load_preset_3.setText(QCoreApplication.translate("MainWindow", u"&3", None))
        self.action_clear_preset_1.setText(QCoreApplication.translate("MainWindow", u"&1", None))
        self.action_clear_preset_2.setText(QCoreApplication.translate("MainWindow", u"&2", None))
        self.action_clear_preset_3.setText(QCoreApplication.translate("MainWindow", u"&3", None))
        self.action_store_preset_1.setText(QCoreApplication.translate("MainWindow", u"&1", None))
#if QT_CONFIG(shortcut)
        self.action_store_preset_1.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+Alt+1", None))
#endif // QT_CONFIG(shortcut)
        self.action_store_preset_2.setText(QCoreApplication.translate("MainWindow", u"&2", None))
#if QT_CONFIG(shortcut)
        self.action_store_preset_2.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+Alt+2", None))
#endif // QT_CONFIG(shortcut)
        self.action_store_preset_3.setText(QCoreApplication.translate("MainWindow", u"&3", None))
#if QT_CONFIG(shortcut)
        self.action_store_preset_3.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+Alt+3", None))
#endif // QT_CONFIG(shortcut)
        self.actionExport_FRD.setText(QCoreApplication.translate("MainWindow", u"&Export FRD", None))
        self.action_Save_Project.setText(QCoreApplication.translate("MainWindow", u"&Save Project", None))
#if QT_CONFIG(shortcut)
        self.action_Save_Project.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+S", None))
#endif // QT_CONFIG(shortcut)
        self.action_Load_Project.setText(QCoreApplication.translate("MainWindow", u"&Load Project", None))
#if QT_CONFIG(shortcut)
        self.action_Load_Project.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+O", None))
#endif // QT_CONFIG(shortcut)
        self.actionSave_Signal.setText(QCoreApplication.translate("MainWindow", u"Save Si&gnal", None))
#if QT_CONFIG(shortcut)
        self.actionSave_Signal.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+Shift+S", None))
#endif // QT_CONFIG(shortcut)
        self.actionLoad_Signal.setText(QCoreApplication.translate("MainWindow", u"Loa&d Signal", None))
#if QT_CONFIG(shortcut)
        self.actionLoad_Signal.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+Shift+O", None))
#endif // QT_CONFIG(shortcut)
        self.actionClear_Project.setText(QCoreApplication.translate("MainWindow", u"Clear Project", None))
        self.actionAnalyse_Audio.setText(QCoreApplication.translate("MainWindow", u"&Analyse Audio", None))
#if QT_CONFIG(shortcut)
        self.actionAnalyse_Audio.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+A", None))
#endif // QT_CONFIG(shortcut)
        self.action_Batch_Extract.setText(QCoreApplication.translate("MainWindow", u"&Batch Extract", None))
#if QT_CONFIG(shortcut)
        self.action_Batch_Extract.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+Shift+E", None))
#endif // QT_CONFIG(shortcut)
        self.actionSave_Report.setText(QCoreApplication.translate("MainWindow", u"Save Repor&t", None))
#if QT_CONFIG(shortcut)
        self.actionSave_Report.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+R", None))
#endif // QT_CONFIG(shortcut)
        self.actionAbout.setText(QCoreApplication.translate("MainWindow", u"&About", None))
        self.action_Remux_Audio.setText(QCoreApplication.translate("MainWindow", u"&Remux Audio", None))
#if QT_CONFIG(shortcut)
        self.action_Remux_Audio.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+U", None))
#endif // QT_CONFIG(shortcut)
        self.actionAdd_BEQ_Filter.setText(QCoreApplication.translate("MainWindow", u"Load BE&Q Filter", None))
#if QT_CONFIG(shortcut)
        self.actionAdd_BEQ_Filter.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+Shift+B", None))
#endif // QT_CONFIG(shortcut)
        self.actionClear_Signals.setText(QCoreApplication.translate("MainWindow", u"Clear S&ignals", None))
        self.action1_1_Smoothing.setText(QCoreApplication.translate("MainWindow", u"1/&1 Smoothing", None))
#if QT_CONFIG(shortcut)
        self.action1_1_Smoothing.setShortcut(QCoreApplication.translate("MainWindow", u"Alt+Shift+1", None))
#endif // QT_CONFIG(shortcut)
        self.action1_3_Smoothing.setText(QCoreApplication.translate("MainWindow", u"1/&3 Smoothing", None))
#if QT_CONFIG(shortcut)
        self.action1_3_Smoothing.setShortcut(QCoreApplication.translate("MainWindow", u"Alt+Shift+2", None))
#endif // QT_CONFIG(shortcut)
        self.action1_6_Smoothing.setText(QCoreApplication.translate("MainWindow", u"1/&6 Smoothing", None))
#if QT_CONFIG(shortcut)
        self.action1_6_Smoothing.setShortcut(QCoreApplication.translate("MainWindow", u"Alt+Shift+3", None))
#endif // QT_CONFIG(shortcut)
        self.action1_1_2_Smoothing.setText(QCoreApplication.translate("MainWindow", u"1/1&2 Smoothing", None))
#if QT_CONFIG(shortcut)
        self.action1_1_2_Smoothing.setShortcut(QCoreApplication.translate("MainWindow", u"Alt+Shift+4", None))
#endif // QT_CONFIG(shortcut)
        self.action1_2_4_Smoothing.setText(QCoreApplication.translate("MainWindow", u"1/2&4 Smoothing", None))
#if QT_CONFIG(shortcut)
        self.action1_2_4_Smoothing.setShortcut(QCoreApplication.translate("MainWindow", u"Alt+Shift+5", None))
#endif // QT_CONFIG(shortcut)
        self.action1_4_8_Smoothing.setText(QCoreApplication.translate("MainWindow", u"1/4&8 Smoothing", None))
#if QT_CONFIG(shortcut)
        self.action1_4_8_Smoothing.setShortcut(QCoreApplication.translate("MainWindow", u"Alt+Shift+6", None))
#endif // QT_CONFIG(shortcut)
        self.action_Remove_Smoothing.setText(QCoreApplication.translate("MainWindow", u"&Remove Smoothing", None))
#if QT_CONFIG(shortcut)
        self.action_Remove_Smoothing.setShortcut(QCoreApplication.translate("MainWindow", u"Alt+Shift+0", None))
#endif // QT_CONFIG(shortcut)
        self.actionClear_Filters.setText(QCoreApplication.translate("MainWindow", u"Clear Filters", None))
        self.actionMerge_Minidsp_XML.setText(QCoreApplication.translate("MainWindow", u"Merge Minidsp XML", None))
#if QT_CONFIG(shortcut)
        self.actionMerge_Minidsp_XML.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+X", None))
#endif // QT_CONFIG(shortcut)
        self.actionUser_Guide.setText(QCoreApplication.translate("MainWindow", u"User Guide", None))
#if QT_CONFIG(shortcut)
        self.actionUser_Guide.setShortcut(QCoreApplication.translate("MainWindow", u"F1", None))
#endif // QT_CONFIG(shortcut)
        self.actionRelease_Notes.setText(QCoreApplication.translate("MainWindow", u"Release Notes", None))
        self.actionExport_BEQ_Filter.setText(QCoreApplication.translate("MainWindow", u"Export BEQ Filter", None))
#if QT_CONFIG(shortcut)
        self.actionExport_BEQ_Filter.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+Alt+E", None))
#endif // QT_CONFIG(shortcut)
        self.actionCreate_AVS_Post.setText(QCoreApplication.translate("MainWindow", u"Create AVS Post and XML", None))
#if QT_CONFIG(tooltip)
        self.actionCreate_AVS_Post.setToolTip(QCoreApplication.translate("MainWindow", u"Create AVS Post and XML", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(shortcut)
        self.actionCreate_AVS_Post.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+Shift+P", None))
#endif // QT_CONFIG(shortcut)
        self.actionSync_with_HTP_1.setText(QCoreApplication.translate("MainWindow", u"Manage &HTP-1 Filters", None))
#if QT_CONFIG(shortcut)
        self.actionSync_with_HTP_1.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+M", None))
#endif // QT_CONFIG(shortcut)
        self.showValuesButton.setText(QCoreApplication.translate("MainWindow", u"...", None))
        self.filterReference.setItemText(0, QCoreApplication.translate("MainWindow", u"None", None))

        self.limitsButton.setText(QCoreApplication.translate("MainWindow", u"...", None))
        self.signalReference.setItemText(0, QCoreApplication.translate("MainWindow", u"None", None))

        self.referenceLabel.setText(QCoreApplication.translate("MainWindow", u"Reference:", None))
        self.waveformIsFiltered.setText(QCoreApplication.translate("MainWindow", u"Filtered?", None))
        self.hardClipWaveform.setText(QCoreApplication.translate("MainWindow", u"Clip?", None))
        self.saveWaveformChartButton.setText(QCoreApplication.translate("MainWindow", u"...", None))
        self.signalSelectorLabel.setText(QCoreApplication.translate("MainWindow", u"Signal", None))
        self.loadSignalButton.setText(QCoreApplication.translate("MainWindow", u"...", None))
        self.sourceFileLabel.setText(QCoreApplication.translate("MainWindow", u"Source File", None))
        self.startTime.setDisplayFormat(QCoreApplication.translate("MainWindow", u"HH:mm:ss.zzz", None))
        self.analysisLabel.setText(QCoreApplication.translate("MainWindow", u"RMS / Crest Factor", None))
        self.bmHeaderLabel.setText(QCoreApplication.translate("MainWindow", u"Bass Management", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Waveform", None))
        self.endTime.setDisplayFormat(QCoreApplication.translate("MainWindow", u"HH:mm:ss.zzz", None))
        self.compareSpectrumButton.setText(QCoreApplication.translate("MainWindow", u"...", None))
        self.showSpectrumButton.setText(QCoreApplication.translate("MainWindow", u"...", None))
        self.hideSpectrumButton.setText(QCoreApplication.translate("MainWindow", u"...", None))
        self.filteredSpectrumLimitsButton.setText(QCoreApplication.translate("MainWindow", u"...", None))
        self.zoomInButton.setText(QCoreApplication.translate("MainWindow", u"...", None))
        self.zoomOutButton.setText(QCoreApplication.translate("MainWindow", u"...", None))
        self.showStatsButton.setText(QCoreApplication.translate("MainWindow", u"...", None))
        self.bmClippingLabel.setText(QCoreApplication.translate("MainWindow", u"Clip", None))
        self.yRangeLabel.setText(QCoreApplication.translate("MainWindow", u"Y Range", None))
        self.headroomLabel.setText(QCoreApplication.translate("MainWindow", u"Headroom", None))
        self.bmHeadroom.setItemText(0, QCoreApplication.translate("MainWindow", u"WCS", None))
        self.bmHeadroom.setItemText(1, QCoreApplication.translate("MainWindow", u"-8", None))
        self.bmHeadroom.setItemText(2, QCoreApplication.translate("MainWindow", u"-7", None))
        self.bmHeadroom.setItemText(3, QCoreApplication.translate("MainWindow", u"-6", None))
        self.bmHeadroom.setItemText(4, QCoreApplication.translate("MainWindow", u"-5", None))

        self.endTimeLabel.setText(QCoreApplication.translate("MainWindow", u"End", None))
        self.bmhpfOn.setText(QCoreApplication.translate("MainWindow", u"HPF On?", None))
        self.startTimeLabel.setText(QCoreApplication.translate("MainWindow", u"Start", None))
        self.bmlpfPositionLabel.setText(QCoreApplication.translate("MainWindow", u"XO Filter", None))
        self.bmClipBefore.setText(QCoreApplication.translate("MainWindow", u"Before", None))
        self.bmClipAfter.setText(QCoreApplication.translate("MainWindow", u"After", None))
        self.bmHeadroomLabel.setText(QCoreApplication.translate("MainWindow", u"Headroom", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Chart Controls", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Smoothing:", None))
        self.octaveSmoothing.setItemText(0, QCoreApplication.translate("MainWindow", u"None", None))
        self.octaveSmoothing.setItemText(1, QCoreApplication.translate("MainWindow", u"1/1", None))
        self.octaveSmoothing.setItemText(2, QCoreApplication.translate("MainWindow", u"1/2", None))
        self.octaveSmoothing.setItemText(3, QCoreApplication.translate("MainWindow", u"1/3", None))
        self.octaveSmoothing.setItemText(4, QCoreApplication.translate("MainWindow", u"1/6", None))
        self.octaveSmoothing.setItemText(5, QCoreApplication.translate("MainWindow", u"1/12", None))
        self.octaveSmoothing.setItemText(6, QCoreApplication.translate("MainWindow", u"1/24", None))
        self.octaveSmoothing.setItemText(7, QCoreApplication.translate("MainWindow", u"Savitzky\u2013Golay", None))

        self.smoothAllSignals.setText(QCoreApplication.translate("MainWindow", u"All Signals?", None))
        self.signalsLabel.setText(QCoreApplication.translate("MainWindow", u"Signals", None))
        self.addSignalButton.setText(QCoreApplication.translate("MainWindow", u"Add", None))
#if QT_CONFIG(shortcut)
        self.addSignalButton.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+Shift+=", None))
#endif // QT_CONFIG(shortcut)
        self.linkSignalButton.setText(QCoreApplication.translate("MainWindow", u"Link", None))
#if QT_CONFIG(shortcut)
        self.linkSignalButton.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+Shift+L", None))
#endif // QT_CONFIG(shortcut)
        self.deleteSignalButton.setText(QCoreApplication.translate("MainWindow", u"Delete", None))
        self.clearSignalsButton.setText(QCoreApplication.translate("MainWindow", u"Clear", None))
        self.filtersLabel.setText(QCoreApplication.translate("MainWindow", u"Filters", None))
#if QT_CONFIG(tooltip)
        self.preset1Button.setToolTip("")
#endif // QT_CONFIG(tooltip)
        self.preset1Button.setText(QCoreApplication.translate("MainWindow", u"P1", None))
#if QT_CONFIG(shortcut)
        self.preset1Button.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+Shift+1", None))
#endif // QT_CONFIG(shortcut)
#if QT_CONFIG(tooltip)
        self.preset2Button.setToolTip("")
#endif // QT_CONFIG(tooltip)
        self.preset2Button.setText(QCoreApplication.translate("MainWindow", u"P2", None))
#if QT_CONFIG(shortcut)
        self.preset2Button.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+Shift+2", None))
#endif // QT_CONFIG(shortcut)
#if QT_CONFIG(tooltip)
        self.preset3Button.setToolTip("")
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(statustip)
        self.preset3Button.setStatusTip("")
#endif // QT_CONFIG(statustip)
        self.preset3Button.setText(QCoreApplication.translate("MainWindow", u"P3", None))
#if QT_CONFIG(shortcut)
        self.preset3Button.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+Shift+3", None))
#endif // QT_CONFIG(shortcut)
        self.addFilterButton.setText(QCoreApplication.translate("MainWindow", u"Add", None))
#if QT_CONFIG(shortcut)
        self.addFilterButton.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+=", None))
#endif // QT_CONFIG(shortcut)
        self.editFilterButton.setText(QCoreApplication.translate("MainWindow", u"Edit", None))
        self.deleteFilterButton.setText(QCoreApplication.translate("MainWindow", u"Delete", None))
#if QT_CONFIG(shortcut)
        self.deleteFilterButton.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+-", None))
#endif // QT_CONFIG(shortcut)
        self.clearFiltersButton.setText(QCoreApplication.translate("MainWindow", u"Clear", None))
#if QT_CONFIG(shortcut)
        self.clearFiltersButton.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+Shift+-", None))
#endif // QT_CONFIG(shortcut)
        self.showSignalsLabel.setText(QCoreApplication.translate("MainWindow", u"Peak/Avg", None))
        self.showFiltersLabel.setText(QCoreApplication.translate("MainWindow", u"With Filters?", None))
        self.showFiltersLabel1.setText(QCoreApplication.translate("MainWindow", u"Filter Response?", None))
        self.equalEnergyTilt.setText(QCoreApplication.translate("MainWindow", u"+3dB/octave?", None))
        self.showLegend.setText(QCoreApplication.translate("MainWindow", u"Show Legend", None))
        self.menuHelp.setTitle(QCoreApplication.translate("MainWindow", u"&Help", None))
        self.menuSettings.setTitle(QCoreApplication.translate("MainWindow", u"&Settings", None))
        self.menuFile.setTitle(QCoreApplication.translate("MainWindow", u"&File", None))
        self.menuPresets.setTitle(QCoreApplication.translate("MainWindow", u"&Presets", None))
        self.menuLoad.setTitle(QCoreApplication.translate("MainWindow", u"&Load", None))
        self.menu_Clear.setTitle(QCoreApplication.translate("MainWindow", u"&Clear", None))
        self.menu_Store.setTitle(QCoreApplication.translate("MainWindow", u"&Store", None))
        self.menu_Tools.setTitle(QCoreApplication.translate("MainWindow", u"&Tools", None))
    # retranslateUi

