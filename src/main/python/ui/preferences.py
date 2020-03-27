# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'preferences.ui'
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


class Ui_preferencesDialog(object):
    def setupUi(self, preferencesDialog):
        if preferencesDialog.objectName():
            preferencesDialog.setObjectName(u"preferencesDialog")
        preferencesDialog.resize(375, 485)
        self.verticalLayout = QVBoxLayout(preferencesDialog)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.panes = QVBoxLayout()
        self.panes.setObjectName(u"panes")
        self.toolBox = QToolBox(preferencesDialog)
        self.toolBox.setObjectName(u"toolBox")
        self.binariesPage = QWidget()
        self.binariesPage.setObjectName(u"binariesPage")
        self.binariesPage.setGeometry(QRect(0, 0, 152, 86))
        self.binariesPane = QGridLayout(self.binariesPage)
        self.binariesPane.setObjectName(u"binariesPane")
        self.ffmpegDirectory = QLineEdit(self.binariesPage)
        self.ffmpegDirectory.setObjectName(u"ffmpegDirectory")
        self.ffmpegDirectory.setEnabled(False)

        self.binariesPane.addWidget(self.ffmpegDirectory, 0, 1, 1, 1)

        self.ffmpegLabel = QLabel(self.binariesPage)
        self.ffmpegLabel.setObjectName(u"ffmpegLabel")

        self.binariesPane.addWidget(self.ffmpegLabel, 0, 0, 1, 1)

        self.ffmpegDirectoryPicker = QToolButton(self.binariesPage)
        self.ffmpegDirectoryPicker.setObjectName(u"ffmpegDirectoryPicker")

        self.binariesPane.addWidget(self.ffmpegDirectoryPicker, 0, 2, 1, 1)

        self.ffprobeLabel = QLabel(self.binariesPage)
        self.ffprobeLabel.setObjectName(u"ffprobeLabel")

        self.binariesPane.addWidget(self.ffprobeLabel, 1, 0, 1, 1)

        self.ffprobeDirectory = QLineEdit(self.binariesPage)
        self.ffprobeDirectory.setObjectName(u"ffprobeDirectory")
        self.ffprobeDirectory.setEnabled(False)

        self.binariesPane.addWidget(self.ffprobeDirectory, 1, 1, 1, 1)

        self.ffprobeDirectoryPicker = QToolButton(self.binariesPage)
        self.ffprobeDirectoryPicker.setObjectName(u"ffprobeDirectoryPicker")

        self.binariesPane.addWidget(self.ffprobeDirectoryPicker, 1, 2, 1, 1)

        self.toolBox.addItem(self.binariesPage, u"Binaries")
        self.analysisPage = QWidget()
        self.analysisPage.setObjectName(u"analysisPage")
        self.analysisPage.setGeometry(QRect(0, 0, 185, 158))
        self.analysisPane = QGridLayout(self.analysisPage)
        self.analysisPane.setObjectName(u"analysisPane")
        self.peakAnalysisWindow = QComboBox(self.analysisPage)
        self.peakAnalysisWindow.setObjectName(u"peakAnalysisWindow")

        self.analysisPane.addWidget(self.peakAnalysisWindow, 3, 1, 1, 1)

        self.peakAnalysisWindowLabel = QLabel(self.analysisPage)
        self.peakAnalysisWindowLabel.setObjectName(u"peakAnalysisWindowLabel")

        self.analysisPane.addWidget(self.peakAnalysisWindowLabel, 3, 0, 1, 1)

        self.avgAnalysisWindowLabel = QLabel(self.analysisPage)
        self.avgAnalysisWindowLabel.setObjectName(u"avgAnalysisWindowLabel")

        self.analysisPane.addWidget(self.avgAnalysisWindowLabel, 2, 0, 1, 1)

        self.avgAnalysisWindow = QComboBox(self.analysisPage)
        self.avgAnalysisWindow.setObjectName(u"avgAnalysisWindow")

        self.analysisPane.addWidget(self.avgAnalysisWindow, 2, 1, 1, 1)

        self.targetFsLabel = QLabel(self.analysisPage)
        self.targetFsLabel.setObjectName(u"targetFsLabel")

        self.analysisPane.addWidget(self.targetFsLabel, 0, 0, 1, 1)

        self.targetFs = QComboBox(self.analysisPage)
        self.targetFs.addItem("")
        self.targetFs.addItem("")
        self.targetFs.addItem("")
        self.targetFs.addItem("")
        self.targetFs.addItem("")
        self.targetFs.addItem("")
        self.targetFs.setObjectName(u"targetFs")

        self.analysisPane.addWidget(self.targetFs, 0, 1, 1, 1)

        self.resolutionLabel = QLabel(self.analysisPage)
        self.resolutionLabel.setObjectName(u"resolutionLabel")

        self.analysisPane.addWidget(self.resolutionLabel, 1, 0, 1, 1)

        self.resolutionSelect = QComboBox(self.analysisPage)
        self.resolutionSelect.addItem("")
        self.resolutionSelect.addItem("")
        self.resolutionSelect.addItem("")
        self.resolutionSelect.addItem("")
        self.resolutionSelect.addItem("")
        self.resolutionSelect.setObjectName(u"resolutionSelect")

        self.analysisPane.addWidget(self.resolutionSelect, 1, 1, 1, 1)

        self.analysisPane.setColumnStretch(1, 1)
        self.toolBox.addItem(self.analysisPage, u"Analysis")
        self.widget = QWidget()
        self.widget.setObjectName(u"widget")
        self.widget.setGeometry(QRect(0, 0, 344, 146))
        self.extractPane = QGridLayout(self.widget)
        self.extractPane.setObjectName(u"extractPane")
        self.defaultOutputDirectoryLabel = QLabel(self.widget)
        self.defaultOutputDirectoryLabel.setObjectName(u"defaultOutputDirectoryLabel")

        self.extractPane.addWidget(self.defaultOutputDirectoryLabel, 0, 0, 1, 1)

        self.defaultOutputDirectory = QLineEdit(self.widget)
        self.defaultOutputDirectory.setObjectName(u"defaultOutputDirectory")
        self.defaultOutputDirectory.setEnabled(False)

        self.extractPane.addWidget(self.defaultOutputDirectory, 0, 1, 1, 1)

        self.defaultOutputDirectoryPicker = QToolButton(self.widget)
        self.defaultOutputDirectoryPicker.setObjectName(u"defaultOutputDirectoryPicker")

        self.extractPane.addWidget(self.defaultOutputDirectoryPicker, 0, 2, 1, 1)

        self.extractCompleteAudioFileLabel = QLabel(self.widget)
        self.extractCompleteAudioFileLabel.setObjectName(u"extractCompleteAudioFileLabel")

        self.extractPane.addWidget(self.extractCompleteAudioFileLabel, 1, 0, 1, 1)

        self.extractCompleteAudioFile = QLineEdit(self.widget)
        self.extractCompleteAudioFile.setObjectName(u"extractCompleteAudioFile")
        self.extractCompleteAudioFile.setEnabled(False)

        self.extractPane.addWidget(self.extractCompleteAudioFile, 1, 1, 1, 1)

        self.extractCompleteAudioFilePicker = QToolButton(self.widget)
        self.extractCompleteAudioFilePicker.setObjectName(u"extractCompleteAudioFilePicker")

        self.extractPane.addWidget(self.extractCompleteAudioFilePicker, 1, 2, 1, 1)

        self.extractOptions1 = QHBoxLayout()
        self.extractOptions1.setObjectName(u"extractOptions1")
        self.includeOriginal = QCheckBox(self.widget)
        self.includeOriginal.setObjectName(u"includeOriginal")

        self.extractOptions1.addWidget(self.includeOriginal)

        self.compress = QCheckBox(self.widget)
        self.compress.setObjectName(u"compress")

        self.extractOptions1.addWidget(self.compress)


        self.extractPane.addLayout(self.extractOptions1, 2, 0, 1, 3)

        self.extractOptions2 = QHBoxLayout()
        self.extractOptions2.setObjectName(u"extractOptions2")
        self.monoMix = QCheckBox(self.widget)
        self.monoMix.setObjectName(u"monoMix")

        self.extractOptions2.addWidget(self.monoMix)

        self.decimate = QCheckBox(self.widget)
        self.decimate.setObjectName(u"decimate")

        self.extractOptions2.addWidget(self.decimate)

        self.includeSubtitles = QCheckBox(self.widget)
        self.includeSubtitles.setObjectName(u"includeSubtitles")

        self.extractOptions2.addWidget(self.includeSubtitles)


        self.extractPane.addLayout(self.extractOptions2, 3, 0, 1, 3)

        self.toolBox.addItem(self.widget, u"Extraction")
        self.widget1 = QWidget()
        self.widget1.setObjectName(u"widget1")
        self.widget1.setGeometry(QRect(0, 0, 222, 100))
        self.stylePane = QGridLayout(self.widget1)
        self.stylePane.setObjectName(u"stylePane")
        self.themeLabel = QLabel(self.widget1)
        self.themeLabel.setObjectName(u"themeLabel")

        self.stylePane.addWidget(self.themeLabel, 0, 0, 1, 1)

        self.themePicker = QComboBox(self.widget1)
        self.themePicker.addItem("")
        self.themePicker.setObjectName(u"themePicker")

        self.stylePane.addWidget(self.themePicker, 0, 1, 1, 1)

        self.speclabLineStyle = QCheckBox(self.widget1)
        self.speclabLineStyle.setObjectName(u"speclabLineStyle")

        self.stylePane.addWidget(self.speclabLineStyle, 1, 1, 1, 1)

        self.smoothGraphs = QCheckBox(self.widget1)
        self.smoothGraphs.setObjectName(u"smoothGraphs")

        self.stylePane.addWidget(self.smoothGraphs, 2, 1, 1, 1)

        self.stylePane.setColumnStretch(1, 1)
        self.toolBox.addItem(self.widget1, u"Style")
        self.graphPage = QWidget()
        self.graphPage.setObjectName(u"graphPage")
        self.graphPage.setGeometry(QRect(0, 0, 239, 130))
        self.graphPane = QVBoxLayout(self.graphPage)
        self.graphPane.setObjectName(u"graphPane")
        self.freqIsLogScale = QCheckBox(self.graphPage)
        self.freqIsLogScale.setObjectName(u"freqIsLogScale")
        self.freqIsLogScale.setChecked(True)

        self.graphPane.addWidget(self.freqIsLogScale)

        self.precalcSmoothing = QCheckBox(self.graphPage)
        self.precalcSmoothing.setObjectName(u"precalcSmoothing")
        self.precalcSmoothing.setChecked(True)

        self.graphPane.addWidget(self.precalcSmoothing)

        self.expandYLimits = QCheckBox(self.graphPage)
        self.expandYLimits.setObjectName(u"expandYLimits")

        self.graphPane.addWidget(self.expandYLimits)

        self.xminmaxLayout = QHBoxLayout()
        self.xminmaxLayout.setObjectName(u"xminmaxLayout")
        self.xminmaxLabel = QLabel(self.graphPage)
        self.xminmaxLabel.setObjectName(u"xminmaxLabel")

        self.xminmaxLayout.addWidget(self.xminmaxLabel)

        self.xmin = QSpinBox(self.graphPage)
        self.xmin.setObjectName(u"xmin")
        self.xmin.setMinimum(1)
        self.xmin.setMaximum(23999)
        self.xmin.setValue(2)

        self.xminmaxLayout.addWidget(self.xmin)

        self.xmax = QSpinBox(self.graphPage)
        self.xmax.setObjectName(u"xmax")
        self.xmax.setMinimum(2)
        self.xmax.setMaximum(24000)
        self.xmax.setValue(160)

        self.xminmaxLayout.addWidget(self.xmax)


        self.graphPane.addLayout(self.xminmaxLayout)

        self.toolBox.addItem(self.graphPage, u"Graph")
        self.filterPage = QWidget()
        self.filterPage.setObjectName(u"filterPage")
        self.filterPage.setGeometry(QRect(0, 0, 173, 120))
        self.filterPane = QGridLayout(self.filterPage)
        self.filterPane.setObjectName(u"filterPane")
        self.filterQLabel = QLabel(self.filterPage)
        self.filterQLabel.setObjectName(u"filterQLabel")

        self.filterPane.addWidget(self.filterQLabel, 0, 0, 1, 1)

        self.bmlpfFreq = QSpinBox(self.filterPage)
        self.bmlpfFreq.setObjectName(u"bmlpfFreq")
        self.bmlpfFreq.setMinimum(20)
        self.bmlpfFreq.setMaximum(160)
        self.bmlpfFreq.setValue(80)

        self.filterPane.addWidget(self.bmlpfFreq, 2, 1, 1, 1)

        self.filterFreq = QSpinBox(self.filterPage)
        self.filterFreq.setObjectName(u"filterFreq")
        self.filterFreq.setMinimum(1)
        self.filterFreq.setMaximum(24000)

        self.filterPane.addWidget(self.filterFreq, 1, 1, 1, 1)

        self.filterQ = QDoubleSpinBox(self.filterPage)
        self.filterQ.setObjectName(u"filterQ")
        self.filterQ.setDecimals(3)
        self.filterQ.setMinimum(0.001000000000000)
        self.filterQ.setSingleStep(0.001000000000000)

        self.filterPane.addWidget(self.filterQ, 0, 1, 1, 1)

        self.filterFreqLabel = QLabel(self.filterPage)
        self.filterFreqLabel.setObjectName(u"filterFreqLabel")

        self.filterPane.addWidget(self.filterFreqLabel, 1, 0, 1, 1)

        self.bmlpfLabel = QLabel(self.filterPage)
        self.bmlpfLabel.setObjectName(u"bmlpfLabel")

        self.filterPane.addWidget(self.bmlpfLabel, 2, 0, 1, 1)

        self.filterPane.setColumnStretch(1, 1)
        self.toolBox.addItem(self.filterPage, u"Filters")
        self.systemPage = QWidget()
        self.systemPage.setObjectName(u"systemPage")
        self.systemPage.setGeometry(QRect(0, 0, 225, 62))
        self.systemPane = QGridLayout(self.systemPage)
        self.systemPane.setObjectName(u"systemPane")
        self.checkForBetaUpdates = QCheckBox(self.systemPage)
        self.checkForBetaUpdates.setObjectName(u"checkForBetaUpdates")

        self.systemPane.addWidget(self.checkForBetaUpdates, 1, 0, 1, 1)

        self.checkForUpdates = QCheckBox(self.systemPage)
        self.checkForUpdates.setObjectName(u"checkForUpdates")
        self.checkForUpdates.setChecked(True)

        self.systemPane.addWidget(self.checkForUpdates, 0, 0, 1, 1)

        self.toolBox.addItem(self.systemPage, u"System")
        self.beqPage = QWidget()
        self.beqPage.setObjectName(u"beqPage")
        self.beqPage.setGeometry(QRect(0, 0, 361, 175))
        self.beqPane = QGridLayout(self.beqPage)
        self.beqPane.setObjectName(u"beqPane")
        self.beqDirectoryLabel = QLabel(self.beqPage)
        self.beqDirectoryLabel.setObjectName(u"beqDirectoryLabel")

        self.beqPane.addWidget(self.beqDirectoryLabel, 0, 0, 1, 1)

        self.beqFiltersDir = QLineEdit(self.beqPage)
        self.beqFiltersDir.setObjectName(u"beqFiltersDir")
        self.beqFiltersDir.setReadOnly(True)

        self.beqPane.addWidget(self.beqFiltersDir, 0, 1, 1, 1)

        self.beqDirectoryPicker = QToolButton(self.beqPage)
        self.beqDirectoryPicker.setObjectName(u"beqDirectoryPicker")

        self.beqPane.addWidget(self.beqDirectoryPicker, 0, 2, 1, 1)

        self.repoURLLabel = QLabel(self.beqPage)
        self.repoURLLabel.setObjectName(u"repoURLLabel")

        self.beqPane.addWidget(self.repoURLLabel, 1, 0, 1, 1)

        self.repoURL = QLineEdit(self.beqPage)
        self.repoURL.setObjectName(u"repoURL")

        self.beqPane.addWidget(self.repoURL, 1, 1, 1, 1)

        self.addRepoButton = QToolButton(self.beqPage)
        self.addRepoButton.setObjectName(u"addRepoButton")

        self.beqPane.addWidget(self.addRepoButton, 1, 2, 1, 1)

        self.beqRepos = QComboBox(self.beqPage)
        self.beqRepos.setObjectName(u"beqRepos")

        self.beqPane.addWidget(self.beqRepos, 2, 1, 1, 1)

        self.deleteRepoButton = QToolButton(self.beqPage)
        self.deleteRepoButton.setObjectName(u"deleteRepoButton")

        self.beqPane.addWidget(self.deleteRepoButton, 2, 2, 1, 1)

        self.filteredLoadedLabel = QLabel(self.beqPage)
        self.filteredLoadedLabel.setObjectName(u"filteredLoadedLabel")

        self.beqPane.addWidget(self.filteredLoadedLabel, 3, 0, 1, 1)

        self.beqFiltersCount = QSpinBox(self.beqPage)
        self.beqFiltersCount.setObjectName(u"beqFiltersCount")
        self.beqFiltersCount.setEnabled(False)
        self.beqFiltersCount.setMaximum(100000)

        self.beqPane.addWidget(self.beqFiltersCount, 3, 1, 1, 1)

        self.refreshBeq = QToolButton(self.beqPage)
        self.refreshBeq.setObjectName(u"refreshBeq")

        self.beqPane.addWidget(self.refreshBeq, 3, 2, 1, 1)

        self.beqReposLabel = QLabel(self.beqPage)
        self.beqReposLabel.setObjectName(u"beqReposLabel")

        self.beqPane.addWidget(self.beqReposLabel, 2, 0, 1, 1)

        self.toolBox.addItem(self.beqPage, u"BEQ")

        self.panes.addWidget(self.toolBox)


        self.verticalLayout.addLayout(self.panes)

        self.buttonBox = QDialogButtonBox(preferencesDialog)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.RestoreDefaults|QDialogButtonBox.Save)

        self.verticalLayout.addWidget(self.buttonBox)

        QWidget.setTabOrder(self.ffmpegDirectory, self.ffmpegDirectoryPicker)
        QWidget.setTabOrder(self.ffmpegDirectoryPicker, self.ffprobeDirectory)
        QWidget.setTabOrder(self.ffprobeDirectory, self.ffprobeDirectoryPicker)
        QWidget.setTabOrder(self.ffprobeDirectoryPicker, self.targetFs)
        QWidget.setTabOrder(self.targetFs, self.resolutionSelect)
        QWidget.setTabOrder(self.resolutionSelect, self.avgAnalysisWindow)
        QWidget.setTabOrder(self.avgAnalysisWindow, self.peakAnalysisWindow)
        QWidget.setTabOrder(self.peakAnalysisWindow, self.defaultOutputDirectory)
        QWidget.setTabOrder(self.defaultOutputDirectory, self.defaultOutputDirectoryPicker)
        QWidget.setTabOrder(self.defaultOutputDirectoryPicker, self.extractCompleteAudioFile)
        QWidget.setTabOrder(self.extractCompleteAudioFile, self.extractCompleteAudioFilePicker)
        QWidget.setTabOrder(self.extractCompleteAudioFilePicker, self.includeOriginal)
        QWidget.setTabOrder(self.includeOriginal, self.compress)
        QWidget.setTabOrder(self.compress, self.monoMix)
        QWidget.setTabOrder(self.monoMix, self.decimate)
        QWidget.setTabOrder(self.decimate, self.includeSubtitles)
        QWidget.setTabOrder(self.includeSubtitles, self.themePicker)
        QWidget.setTabOrder(self.themePicker, self.speclabLineStyle)
        QWidget.setTabOrder(self.speclabLineStyle, self.smoothGraphs)
        QWidget.setTabOrder(self.smoothGraphs, self.freqIsLogScale)
        QWidget.setTabOrder(self.freqIsLogScale, self.precalcSmoothing)
        QWidget.setTabOrder(self.precalcSmoothing, self.expandYLimits)
        QWidget.setTabOrder(self.expandYLimits, self.xmin)
        QWidget.setTabOrder(self.xmin, self.xmax)
        QWidget.setTabOrder(self.xmax, self.filterQ)
        QWidget.setTabOrder(self.filterQ, self.filterFreq)
        QWidget.setTabOrder(self.filterFreq, self.bmlpfFreq)
        QWidget.setTabOrder(self.bmlpfFreq, self.checkForUpdates)
        QWidget.setTabOrder(self.checkForUpdates, self.checkForBetaUpdates)
        QWidget.setTabOrder(self.checkForBetaUpdates, self.beqFiltersDir)
        QWidget.setTabOrder(self.beqFiltersDir, self.beqDirectoryPicker)
        QWidget.setTabOrder(self.beqDirectoryPicker, self.beqFiltersCount)
        QWidget.setTabOrder(self.beqFiltersCount, self.refreshBeq)

        self.retranslateUi(preferencesDialog)
        self.buttonBox.accepted.connect(preferencesDialog.accept)
        self.buttonBox.rejected.connect(preferencesDialog.reject)
        self.ffmpegDirectoryPicker.clicked.connect(preferencesDialog.showFfmpegDirectoryPicker)
        self.ffprobeDirectoryPicker.clicked.connect(preferencesDialog.showFfprobeDirectoryPicker)
        self.defaultOutputDirectoryPicker.clicked.connect(preferencesDialog.showDefaultOutputDirectoryPicker)
        self.extractCompleteAudioFilePicker.clicked.connect(preferencesDialog.showExtractCompleteSoundPicker)
        self.beqDirectoryPicker.clicked.connect(preferencesDialog.showBeqDirectoryPicker)
        self.refreshBeq.clicked.connect(preferencesDialog.updateBeq)
        self.addRepoButton.clicked.connect(preferencesDialog.add_beq_repo)
        self.deleteRepoButton.clicked.connect(preferencesDialog.remove_beq_repo)
        self.repoURL.textChanged.connect(preferencesDialog.validate_beq_repo)

        self.toolBox.setCurrentIndex(7)


        QMetaObject.connectSlotsByName(preferencesDialog)
    # setupUi

    def retranslateUi(self, preferencesDialog):
        preferencesDialog.setWindowTitle(QCoreApplication.translate("preferencesDialog", u"Preferences", None))
        self.ffmpegLabel.setText(QCoreApplication.translate("preferencesDialog", u"ffmpeg", None))
        self.ffmpegDirectoryPicker.setText(QCoreApplication.translate("preferencesDialog", u"...", None))
        self.ffprobeLabel.setText(QCoreApplication.translate("preferencesDialog", u"ffprobe", None))
        self.ffprobeDirectoryPicker.setText(QCoreApplication.translate("preferencesDialog", u"...", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.binariesPage), QCoreApplication.translate("preferencesDialog", u"Binaries", None))
        self.peakAnalysisWindowLabel.setText(QCoreApplication.translate("preferencesDialog", u"Peak Window", None))
        self.avgAnalysisWindowLabel.setText(QCoreApplication.translate("preferencesDialog", u"Avg Window", None))
        self.targetFsLabel.setText(QCoreApplication.translate("preferencesDialog", u"Target Fs", None))
        self.targetFs.setItemText(0, QCoreApplication.translate("preferencesDialog", u"250 Hz", None))
        self.targetFs.setItemText(1, QCoreApplication.translate("preferencesDialog", u"500 Hz", None))
        self.targetFs.setItemText(2, QCoreApplication.translate("preferencesDialog", u"1000 Hz", None))
        self.targetFs.setItemText(3, QCoreApplication.translate("preferencesDialog", u"2000 Hz", None))
        self.targetFs.setItemText(4, QCoreApplication.translate("preferencesDialog", u"4000 Hz", None))
        self.targetFs.setItemText(5, QCoreApplication.translate("preferencesDialog", u"8000 Hz", None))

        self.resolutionLabel.setText(QCoreApplication.translate("preferencesDialog", u"Resolution", None))
        self.resolutionSelect.setItemText(0, QCoreApplication.translate("preferencesDialog", u"0.25 Hz", None))
        self.resolutionSelect.setItemText(1, QCoreApplication.translate("preferencesDialog", u"0.5 Hz", None))
        self.resolutionSelect.setItemText(2, QCoreApplication.translate("preferencesDialog", u"1.0 Hz", None))
        self.resolutionSelect.setItemText(3, QCoreApplication.translate("preferencesDialog", u"2.0 Hz", None))
        self.resolutionSelect.setItemText(4, QCoreApplication.translate("preferencesDialog", u"4.0 Hz", None))

        self.resolutionSelect.setCurrentText(QCoreApplication.translate("preferencesDialog", u"0.25 Hz", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.analysisPage), QCoreApplication.translate("preferencesDialog", u"Analysis", None))
        self.defaultOutputDirectoryLabel.setText(QCoreApplication.translate("preferencesDialog", u"Default Output Directory", None))
        self.defaultOutputDirectoryPicker.setText(QCoreApplication.translate("preferencesDialog", u"...", None))
        self.extractCompleteAudioFileLabel.setText(QCoreApplication.translate("preferencesDialog", u"Extract Complete Sound", None))
        self.extractCompleteAudioFilePicker.setText(QCoreApplication.translate("preferencesDialog", u"...", None))
        self.includeOriginal.setText(QCoreApplication.translate("preferencesDialog", u"Add Original Audio?", None))
        self.compress.setText(QCoreApplication.translate("preferencesDialog", u"Compress Audio?", None))
        self.monoMix.setText(QCoreApplication.translate("preferencesDialog", u"Mix to mono?", None))
        self.decimate.setText(QCoreApplication.translate("preferencesDialog", u"Decimate?", None))
        self.includeSubtitles.setText(QCoreApplication.translate("preferencesDialog", u"Add Subtitles?", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.widget), QCoreApplication.translate("preferencesDialog", u"Extraction", None))
        self.themeLabel.setText(QCoreApplication.translate("preferencesDialog", u"Theme", None))
        self.themePicker.setItemText(0, QCoreApplication.translate("preferencesDialog", u"default", None))

        self.speclabLineStyle.setText(QCoreApplication.translate("preferencesDialog", u"Speclab Line Colours?", None))
        self.smoothGraphs.setText(QCoreApplication.translate("preferencesDialog", u"Smooth?", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.widget1), QCoreApplication.translate("preferencesDialog", u"Style", None))
        self.freqIsLogScale.setText(QCoreApplication.translate("preferencesDialog", u"Frequency Axis Log Scale?", None))
        self.precalcSmoothing.setText(QCoreApplication.translate("preferencesDialog", u"Precalculate Octave Smoothing?", None))
        self.expandYLimits.setText(QCoreApplication.translate("preferencesDialog", u"Auto Expand Y Limits?", None))
        self.xminmaxLabel.setText(QCoreApplication.translate("preferencesDialog", u"x min/max", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.graphPage), QCoreApplication.translate("preferencesDialog", u"Graph", None))
        self.filterQLabel.setText(QCoreApplication.translate("preferencesDialog", u"Default Q", None))
        self.bmlpfFreq.setSuffix(QCoreApplication.translate("preferencesDialog", u" Hz", None))
        self.filterFreqLabel.setText(QCoreApplication.translate("preferencesDialog", u"Default Freq", None))
        self.bmlpfLabel.setText(QCoreApplication.translate("preferencesDialog", u"BM LPF", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.filterPage), QCoreApplication.translate("preferencesDialog", u"Filters", None))
        self.checkForBetaUpdates.setText(QCoreApplication.translate("preferencesDialog", u"Include Beta Versions?", None))
        self.checkForUpdates.setText(QCoreApplication.translate("preferencesDialog", u"Check for Updates on startup?", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.systemPage), QCoreApplication.translate("preferencesDialog", u"System", None))
        self.beqDirectoryLabel.setText(QCoreApplication.translate("preferencesDialog", u"Directory", None))
        self.beqDirectoryPicker.setText(QCoreApplication.translate("preferencesDialog", u"...", None))
        self.repoURLLabel.setText(QCoreApplication.translate("preferencesDialog", u"New Repo", None))
        self.addRepoButton.setText(QCoreApplication.translate("preferencesDialog", u"...", None))
        self.deleteRepoButton.setText(QCoreApplication.translate("preferencesDialog", u"...", None))
        self.filteredLoadedLabel.setText(QCoreApplication.translate("preferencesDialog", u"Filter Count", None))
        self.refreshBeq.setText(QCoreApplication.translate("preferencesDialog", u"...", None))
        self.beqReposLabel.setText(QCoreApplication.translate("preferencesDialog", u"Repos", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.beqPage), QCoreApplication.translate("preferencesDialog", u"BEQ", None))
    # retranslateUi

