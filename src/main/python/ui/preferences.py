# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'preferences.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_preferencesDialog(object):
    def setupUi(self, preferencesDialog):
        preferencesDialog.setObjectName("preferencesDialog")
        preferencesDialog.resize(809, 764)
        self.verticalLayout = QtWidgets.QVBoxLayout(preferencesDialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.panes = QtWidgets.QVBoxLayout()
        self.panes.setObjectName("panes")
        self.binariesPane = QtWidgets.QGridLayout()
        self.binariesPane.setObjectName("binariesPane")
        self.ffprobeLabel = QtWidgets.QLabel(preferencesDialog)
        self.ffprobeLabel.setObjectName("ffprobeLabel")
        self.binariesPane.addWidget(self.ffprobeLabel, 2, 0, 1, 1)
        self.binaryLabel = QtWidgets.QLabel(preferencesDialog)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.binaryLabel.setFont(font)
        self.binaryLabel.setFrameShape(QtWidgets.QFrame.Box)
        self.binaryLabel.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.binaryLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.binaryLabel.setObjectName("binaryLabel")
        self.binariesPane.addWidget(self.binaryLabel, 0, 0, 1, 4)
        self.ffprobeDirectoryPicker = QtWidgets.QToolButton(preferencesDialog)
        self.ffprobeDirectoryPicker.setObjectName("ffprobeDirectoryPicker")
        self.binariesPane.addWidget(self.ffprobeDirectoryPicker, 2, 3, 1, 1)
        self.ffmpegLabel = QtWidgets.QLabel(preferencesDialog)
        self.ffmpegLabel.setObjectName("ffmpegLabel")
        self.binariesPane.addWidget(self.ffmpegLabel, 1, 0, 1, 1)
        self.ffmpegDirectory = QtWidgets.QLineEdit(preferencesDialog)
        self.ffmpegDirectory.setEnabled(False)
        self.ffmpegDirectory.setObjectName("ffmpegDirectory")
        self.binariesPane.addWidget(self.ffmpegDirectory, 1, 1, 1, 2)
        self.ffmpegDirectoryPicker = QtWidgets.QToolButton(preferencesDialog)
        self.ffmpegDirectoryPicker.setObjectName("ffmpegDirectoryPicker")
        self.binariesPane.addWidget(self.ffmpegDirectoryPicker, 1, 3, 1, 1)
        self.ffprobeDirectory = QtWidgets.QLineEdit(preferencesDialog)
        self.ffprobeDirectory.setEnabled(False)
        self.ffprobeDirectory.setObjectName("ffprobeDirectory")
        self.binariesPane.addWidget(self.ffprobeDirectory, 2, 1, 1, 2)
        self.panes.addLayout(self.binariesPane)
        self.analysisPane = QtWidgets.QGridLayout()
        self.analysisPane.setObjectName("analysisPane")
        self.targetFs = QtWidgets.QComboBox(preferencesDialog)
        self.targetFs.setObjectName("targetFs")
        self.targetFs.addItem("")
        self.targetFs.addItem("")
        self.targetFs.addItem("")
        self.targetFs.addItem("")
        self.targetFs.addItem("")
        self.targetFs.addItem("")
        self.analysisPane.addWidget(self.targetFs, 1, 1, 1, 1)
        self.resolutionLabel = QtWidgets.QLabel(preferencesDialog)
        self.resolutionLabel.setObjectName("resolutionLabel")
        self.analysisPane.addWidget(self.resolutionLabel, 1, 2, 1, 1)
        self.targetFsLabel = QtWidgets.QLabel(preferencesDialog)
        self.targetFsLabel.setObjectName("targetFsLabel")
        self.analysisPane.addWidget(self.targetFsLabel, 1, 0, 1, 1)
        self.analysisLabel = QtWidgets.QLabel(preferencesDialog)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.analysisLabel.setFont(font)
        self.analysisLabel.setFrameShape(QtWidgets.QFrame.Box)
        self.analysisLabel.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.analysisLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.analysisLabel.setObjectName("analysisLabel")
        self.analysisPane.addWidget(self.analysisLabel, 0, 0, 1, 4)
        self.resolutionSelect = QtWidgets.QComboBox(preferencesDialog)
        self.resolutionSelect.setObjectName("resolutionSelect")
        self.resolutionSelect.addItem("")
        self.resolutionSelect.addItem("")
        self.resolutionSelect.addItem("")
        self.resolutionSelect.addItem("")
        self.resolutionSelect.addItem("")
        self.analysisPane.addWidget(self.resolutionSelect, 1, 3, 1, 1)
        self.avgAnalysisWindowLabel = QtWidgets.QLabel(preferencesDialog)
        self.avgAnalysisWindowLabel.setObjectName("avgAnalysisWindowLabel")
        self.analysisPane.addWidget(self.avgAnalysisWindowLabel, 2, 0, 1, 1)
        self.avgAnalysisWindow = QtWidgets.QComboBox(preferencesDialog)
        self.avgAnalysisWindow.setObjectName("avgAnalysisWindow")
        self.analysisPane.addWidget(self.avgAnalysisWindow, 2, 1, 1, 1)
        self.peakAnalysisWindowLabel = QtWidgets.QLabel(preferencesDialog)
        self.peakAnalysisWindowLabel.setObjectName("peakAnalysisWindowLabel")
        self.analysisPane.addWidget(self.peakAnalysisWindowLabel, 2, 2, 1, 1)
        self.peakAnalysisWindow = QtWidgets.QComboBox(preferencesDialog)
        self.peakAnalysisWindow.setObjectName("peakAnalysisWindow")
        self.analysisPane.addWidget(self.peakAnalysisWindow, 2, 3, 1, 1)
        self.panes.addLayout(self.analysisPane)
        self.extractPane = QtWidgets.QGridLayout()
        self.extractPane.setObjectName("extractPane")
        self.defaultOutputDirectoryLabel = QtWidgets.QLabel(preferencesDialog)
        self.defaultOutputDirectoryLabel.setObjectName("defaultOutputDirectoryLabel")
        self.extractPane.addWidget(self.defaultOutputDirectoryLabel, 1, 0, 1, 1)
        self.defaultOutputDirectoryPicker = QtWidgets.QToolButton(preferencesDialog)
        self.defaultOutputDirectoryPicker.setObjectName("defaultOutputDirectoryPicker")
        self.extractPane.addWidget(self.defaultOutputDirectoryPicker, 1, 2, 1, 1)
        self.extractionLabel = QtWidgets.QLabel(preferencesDialog)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.extractionLabel.setFont(font)
        self.extractionLabel.setFrameShape(QtWidgets.QFrame.Box)
        self.extractionLabel.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.extractionLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.extractionLabel.setObjectName("extractionLabel")
        self.extractPane.addWidget(self.extractionLabel, 0, 0, 1, 3)
        self.defaultOutputDirectory = QtWidgets.QLineEdit(preferencesDialog)
        self.defaultOutputDirectory.setEnabled(False)
        self.defaultOutputDirectory.setObjectName("defaultOutputDirectory")
        self.extractPane.addWidget(self.defaultOutputDirectory, 1, 1, 1, 1)
        self.extractCompleteAudioFileLabel = QtWidgets.QLabel(preferencesDialog)
        self.extractCompleteAudioFileLabel.setObjectName("extractCompleteAudioFileLabel")
        self.extractPane.addWidget(self.extractCompleteAudioFileLabel, 2, 0, 1, 1)
        self.extractCompleteAudioFile = QtWidgets.QLineEdit(preferencesDialog)
        self.extractCompleteAudioFile.setEnabled(False)
        self.extractCompleteAudioFile.setObjectName("extractCompleteAudioFile")
        self.extractPane.addWidget(self.extractCompleteAudioFile, 2, 1, 1, 1)
        self.extractCompleteAudioFilePicker = QtWidgets.QToolButton(preferencesDialog)
        self.extractCompleteAudioFilePicker.setObjectName("extractCompleteAudioFilePicker")
        self.extractPane.addWidget(self.extractCompleteAudioFilePicker, 2, 2, 1, 1)
        self.extractOptions = QtWidgets.QHBoxLayout()
        self.extractOptions.setObjectName("extractOptions")
        self.monoMix = QtWidgets.QCheckBox(preferencesDialog)
        self.monoMix.setObjectName("monoMix")
        self.extractOptions.addWidget(self.monoMix)
        self.decimate = QtWidgets.QCheckBox(preferencesDialog)
        self.decimate.setObjectName("decimate")
        self.extractOptions.addWidget(self.decimate)
        self.includeOriginal = QtWidgets.QCheckBox(preferencesDialog)
        self.includeOriginal.setObjectName("includeOriginal")
        self.extractOptions.addWidget(self.includeOriginal)
        self.compress = QtWidgets.QCheckBox(preferencesDialog)
        self.compress.setObjectName("compress")
        self.extractOptions.addWidget(self.compress)
        self.includeSubtitles = QtWidgets.QCheckBox(preferencesDialog)
        self.includeSubtitles.setObjectName("includeSubtitles")
        self.extractOptions.addWidget(self.includeSubtitles)
        self.extractPane.addLayout(self.extractOptions, 3, 0, 1, 3)
        self.panes.addLayout(self.extractPane)
        self.stylePane = QtWidgets.QGridLayout()
        self.stylePane.setObjectName("stylePane")
        self.themeLabel = QtWidgets.QLabel(preferencesDialog)
        self.themeLabel.setObjectName("themeLabel")
        self.stylePane.addWidget(self.themeLabel, 1, 0, 1, 1)
        self.themePicker = QtWidgets.QComboBox(preferencesDialog)
        self.themePicker.setObjectName("themePicker")
        self.themePicker.addItem("")
        self.stylePane.addWidget(self.themePicker, 1, 1, 1, 1)
        self.styleLabel = QtWidgets.QLabel(preferencesDialog)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.styleLabel.setFont(font)
        self.styleLabel.setFrameShape(QtWidgets.QFrame.Box)
        self.styleLabel.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.styleLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.styleLabel.setObjectName("styleLabel")
        self.stylePane.addWidget(self.styleLabel, 0, 0, 1, 3)
        self.speclabLineStyle = QtWidgets.QCheckBox(preferencesDialog)
        self.speclabLineStyle.setObjectName("speclabLineStyle")
        self.stylePane.addWidget(self.speclabLineStyle, 1, 2, 1, 1)
        self.stylePane.setColumnStretch(1, 1)
        self.panes.addLayout(self.stylePane)
        self.graphPane = QtWidgets.QGridLayout()
        self.graphPane.setObjectName("graphPane")
        self.xmin = QtWidgets.QSpinBox(preferencesDialog)
        self.xmin.setMinimum(1)
        self.xmin.setMaximum(23999)
        self.xmin.setProperty("value", 2)
        self.xmin.setObjectName("xmin")
        self.graphPane.addWidget(self.xmin, 2, 1, 1, 1)
        self.freqIsLogScale = QtWidgets.QCheckBox(preferencesDialog)
        self.freqIsLogScale.setChecked(True)
        self.freqIsLogScale.setObjectName("freqIsLogScale")
        self.graphPane.addWidget(self.freqIsLogScale, 1, 0, 1, 1)
        self.graphLabel = QtWidgets.QLabel(preferencesDialog)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.graphLabel.setFont(font)
        self.graphLabel.setFrameShape(QtWidgets.QFrame.Box)
        self.graphLabel.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.graphLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.graphLabel.setObjectName("graphLabel")
        self.graphPane.addWidget(self.graphLabel, 0, 0, 1, 3)
        self.xminmaxLabel = QtWidgets.QLabel(preferencesDialog)
        self.xminmaxLabel.setObjectName("xminmaxLabel")
        self.graphPane.addWidget(self.xminmaxLabel, 2, 0, 1, 1)
        self.xmax = QtWidgets.QSpinBox(preferencesDialog)
        self.xmax.setMinimum(2)
        self.xmax.setMaximum(24000)
        self.xmax.setProperty("value", 160)
        self.xmax.setObjectName("xmax")
        self.graphPane.addWidget(self.xmax, 2, 2, 1, 1)
        self.smoothFullRange = QtWidgets.QCheckBox(preferencesDialog)
        self.smoothFullRange.setChecked(True)
        self.smoothFullRange.setObjectName("smoothFullRange")
        self.graphPane.addWidget(self.smoothFullRange, 1, 1, 1, 1)
        self.panes.addLayout(self.graphPane)
        self.filterPane = QtWidgets.QGridLayout()
        self.filterPane.setObjectName("filterPane")
        self.filterQLabel = QtWidgets.QLabel(preferencesDialog)
        self.filterQLabel.setObjectName("filterQLabel")
        self.filterPane.addWidget(self.filterQLabel, 1, 0, 1, 1)
        self.filterFreqLabel = QtWidgets.QLabel(preferencesDialog)
        self.filterFreqLabel.setObjectName("filterFreqLabel")
        self.filterPane.addWidget(self.filterFreqLabel, 1, 2, 1, 1)
        self.filterQ = QtWidgets.QDoubleSpinBox(preferencesDialog)
        self.filterQ.setDecimals(3)
        self.filterQ.setMinimum(0.001)
        self.filterQ.setSingleStep(0.001)
        self.filterQ.setObjectName("filterQ")
        self.filterPane.addWidget(self.filterQ, 1, 1, 1, 1)
        self.filterFreq = QtWidgets.QSpinBox(preferencesDialog)
        self.filterFreq.setMinimum(1)
        self.filterFreq.setMaximum(24000)
        self.filterFreq.setObjectName("filterFreq")
        self.filterPane.addWidget(self.filterFreq, 1, 3, 1, 1)
        self.bmlpfFreq = QtWidgets.QSpinBox(preferencesDialog)
        self.bmlpfFreq.setMinimum(20)
        self.bmlpfFreq.setMaximum(160)
        self.bmlpfFreq.setProperty("value", 80)
        self.bmlpfFreq.setObjectName("bmlpfFreq")
        self.filterPane.addWidget(self.bmlpfFreq, 1, 5, 1, 1)
        self.bmlpfLabel = QtWidgets.QLabel(preferencesDialog)
        self.bmlpfLabel.setObjectName("bmlpfLabel")
        self.filterPane.addWidget(self.bmlpfLabel, 1, 4, 1, 1)
        self.filterLayoutLabel = QtWidgets.QLabel(preferencesDialog)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.filterLayoutLabel.setFont(font)
        self.filterLayoutLabel.setFrameShape(QtWidgets.QFrame.Box)
        self.filterLayoutLabel.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.filterLayoutLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.filterLayoutLabel.setObjectName("filterLayoutLabel")
        self.filterPane.addWidget(self.filterLayoutLabel, 0, 0, 1, 6)
        self.panes.addLayout(self.filterPane)
        self.systemPane = QtWidgets.QGridLayout()
        self.systemPane.setObjectName("systemPane")
        self.systemLayoutLabel = QtWidgets.QLabel(preferencesDialog)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.systemLayoutLabel.setFont(font)
        self.systemLayoutLabel.setFrameShape(QtWidgets.QFrame.Box)
        self.systemLayoutLabel.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.systemLayoutLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.systemLayoutLabel.setObjectName("systemLayoutLabel")
        self.systemPane.addWidget(self.systemLayoutLabel, 0, 0, 1, 1)
        self.checkForUpdates = QtWidgets.QCheckBox(preferencesDialog)
        self.checkForUpdates.setChecked(True)
        self.checkForUpdates.setObjectName("checkForUpdates")
        self.systemPane.addWidget(self.checkForUpdates, 1, 0, 1, 1)
        self.panes.addLayout(self.systemPane)
        self.beqPane = QtWidgets.QGridLayout()
        self.beqPane.setObjectName("beqPane")
        self.beqLayoutLabel = QtWidgets.QLabel(preferencesDialog)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.beqLayoutLabel.setFont(font)
        self.beqLayoutLabel.setFrameShape(QtWidgets.QFrame.Box)
        self.beqLayoutLabel.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.beqLayoutLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.beqLayoutLabel.setObjectName("beqLayoutLabel")
        self.beqPane.addWidget(self.beqLayoutLabel, 0, 0, 1, 6)
        self.beqFiltersDir = QtWidgets.QLineEdit(preferencesDialog)
        self.beqFiltersDir.setReadOnly(True)
        self.beqFiltersDir.setObjectName("beqFiltersDir")
        self.beqPane.addWidget(self.beqFiltersDir, 1, 1, 1, 1)
        self.refreshBeq = QtWidgets.QToolButton(preferencesDialog)
        self.refreshBeq.setObjectName("refreshBeq")
        self.beqPane.addWidget(self.refreshBeq, 1, 5, 1, 1)
        self.filteredLoadedLabel = QtWidgets.QLabel(preferencesDialog)
        self.filteredLoadedLabel.setObjectName("filteredLoadedLabel")
        self.beqPane.addWidget(self.filteredLoadedLabel, 1, 3, 1, 1)
        self.beqFiltersCount = QtWidgets.QSpinBox(preferencesDialog)
        self.beqFiltersCount.setEnabled(False)
        self.beqFiltersCount.setMaximum(100000)
        self.beqFiltersCount.setObjectName("beqFiltersCount")
        self.beqPane.addWidget(self.beqFiltersCount, 1, 4, 1, 1)
        self.beqDirectoryLabel = QtWidgets.QLabel(preferencesDialog)
        self.beqDirectoryLabel.setObjectName("beqDirectoryLabel")
        self.beqPane.addWidget(self.beqDirectoryLabel, 1, 0, 1, 1)
        self.beqDirectoryPicker = QtWidgets.QToolButton(preferencesDialog)
        self.beqDirectoryPicker.setObjectName("beqDirectoryPicker")
        self.beqPane.addWidget(self.beqDirectoryPicker, 1, 2, 1, 1)
        self.panes.addLayout(self.beqPane)
        self.verticalLayout.addLayout(self.panes)
        self.buttonBox = QtWidgets.QDialogButtonBox(preferencesDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Save)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(preferencesDialog)
        self.buttonBox.accepted.connect(preferencesDialog.accept)
        self.buttonBox.rejected.connect(preferencesDialog.reject)
        self.ffmpegDirectoryPicker.clicked.connect(preferencesDialog.showFfmpegDirectoryPicker)
        self.ffprobeDirectoryPicker.clicked.connect(preferencesDialog.showFfprobeDirectoryPicker)
        self.defaultOutputDirectoryPicker.clicked.connect(preferencesDialog.showDefaultOutputDirectoryPicker)
        self.extractCompleteAudioFilePicker.clicked.connect(preferencesDialog.showExtractCompleteSoundPicker)
        self.beqDirectoryPicker.clicked.connect(preferencesDialog.showBeqDirectoryPicker)
        self.refreshBeq.clicked.connect(preferencesDialog.updateBeq)
        QtCore.QMetaObject.connectSlotsByName(preferencesDialog)
        preferencesDialog.setTabOrder(self.ffmpegDirectoryPicker, self.ffprobeDirectoryPicker)
        preferencesDialog.setTabOrder(self.ffprobeDirectoryPicker, self.targetFs)
        preferencesDialog.setTabOrder(self.targetFs, self.resolutionSelect)
        preferencesDialog.setTabOrder(self.resolutionSelect, self.avgAnalysisWindow)
        preferencesDialog.setTabOrder(self.avgAnalysisWindow, self.peakAnalysisWindow)
        preferencesDialog.setTabOrder(self.peakAnalysisWindow, self.defaultOutputDirectoryPicker)
        preferencesDialog.setTabOrder(self.defaultOutputDirectoryPicker, self.extractCompleteAudioFilePicker)
        preferencesDialog.setTabOrder(self.extractCompleteAudioFilePicker, self.themePicker)
        preferencesDialog.setTabOrder(self.themePicker, self.freqIsLogScale)
        preferencesDialog.setTabOrder(self.freqIsLogScale, self.xmin)
        preferencesDialog.setTabOrder(self.xmin, self.xmax)
        preferencesDialog.setTabOrder(self.xmax, self.ffmpegDirectory)
        preferencesDialog.setTabOrder(self.ffmpegDirectory, self.defaultOutputDirectory)
        preferencesDialog.setTabOrder(self.defaultOutputDirectory, self.extractCompleteAudioFile)
        preferencesDialog.setTabOrder(self.extractCompleteAudioFile, self.ffprobeDirectory)

    def retranslateUi(self, preferencesDialog):
        _translate = QtCore.QCoreApplication.translate
        preferencesDialog.setWindowTitle(_translate("preferencesDialog", "Preferences"))
        self.ffprobeLabel.setText(_translate("preferencesDialog", "ffprobe"))
        self.binaryLabel.setText(_translate("preferencesDialog", "Binaries"))
        self.ffprobeDirectoryPicker.setText(_translate("preferencesDialog", "..."))
        self.ffmpegLabel.setText(_translate("preferencesDialog", "ffmpeg"))
        self.ffmpegDirectoryPicker.setText(_translate("preferencesDialog", "..."))
        self.targetFs.setItemText(0, _translate("preferencesDialog", "250 Hz"))
        self.targetFs.setItemText(1, _translate("preferencesDialog", "500 Hz"))
        self.targetFs.setItemText(2, _translate("preferencesDialog", "1000 Hz"))
        self.targetFs.setItemText(3, _translate("preferencesDialog", "2000 Hz"))
        self.targetFs.setItemText(4, _translate("preferencesDialog", "4000 Hz"))
        self.targetFs.setItemText(5, _translate("preferencesDialog", "8000 Hz"))
        self.resolutionLabel.setText(_translate("preferencesDialog", "Resolution"))
        self.targetFsLabel.setText(_translate("preferencesDialog", "Target Fs"))
        self.analysisLabel.setText(_translate("preferencesDialog", "Analysis"))
        self.resolutionSelect.setCurrentText(_translate("preferencesDialog", "0.25 Hz"))
        self.resolutionSelect.setItemText(0, _translate("preferencesDialog", "0.25 Hz"))
        self.resolutionSelect.setItemText(1, _translate("preferencesDialog", "0.5 Hz"))
        self.resolutionSelect.setItemText(2, _translate("preferencesDialog", "1.0 Hz"))
        self.resolutionSelect.setItemText(3, _translate("preferencesDialog", "2.0 Hz"))
        self.resolutionSelect.setItemText(4, _translate("preferencesDialog", "4.0 Hz"))
        self.avgAnalysisWindowLabel.setText(_translate("preferencesDialog", "Avg Window"))
        self.peakAnalysisWindowLabel.setText(_translate("preferencesDialog", "Peak Window"))
        self.defaultOutputDirectoryLabel.setText(_translate("preferencesDialog", "Default Output Directory"))
        self.defaultOutputDirectoryPicker.setText(_translate("preferencesDialog", "..."))
        self.extractionLabel.setText(_translate("preferencesDialog", "Extraction"))
        self.extractCompleteAudioFileLabel.setText(_translate("preferencesDialog", "Extract Complete Sound"))
        self.extractCompleteAudioFilePicker.setText(_translate("preferencesDialog", "..."))
        self.monoMix.setText(_translate("preferencesDialog", "Mix to mono?"))
        self.decimate.setText(_translate("preferencesDialog", "Decimate?"))
        self.includeOriginal.setText(_translate("preferencesDialog", "Add Original Audio?"))
        self.compress.setText(_translate("preferencesDialog", "Compress Audio?"))
        self.includeSubtitles.setText(_translate("preferencesDialog", "Add Subtitles?"))
        self.themeLabel.setText(_translate("preferencesDialog", "Theme"))
        self.themePicker.setItemText(0, _translate("preferencesDialog", "default"))
        self.styleLabel.setText(_translate("preferencesDialog", "Style"))
        self.speclabLineStyle.setText(_translate("preferencesDialog", "Speclab Line Colours?"))
        self.freqIsLogScale.setText(_translate("preferencesDialog", "Frequency Axis Log Scale?"))
        self.graphLabel.setText(_translate("preferencesDialog", "Graph"))
        self.xminmaxLabel.setText(_translate("preferencesDialog", "x min/max"))
        self.smoothFullRange.setText(_translate("preferencesDialog", "Smooth Full Range?"))
        self.filterQLabel.setText(_translate("preferencesDialog", "Default Q"))
        self.filterFreqLabel.setText(_translate("preferencesDialog", "Default Freq"))
        self.bmlpfFreq.setSuffix(_translate("preferencesDialog", " Hz"))
        self.bmlpfLabel.setText(_translate("preferencesDialog", "BM LPF"))
        self.filterLayoutLabel.setText(_translate("preferencesDialog", "Filter"))
        self.systemLayoutLabel.setText(_translate("preferencesDialog", "System"))
        self.checkForUpdates.setText(_translate("preferencesDialog", "Check for Updates on startup?"))
        self.beqLayoutLabel.setText(_translate("preferencesDialog", "BEQ Files"))
        self.refreshBeq.setText(_translate("preferencesDialog", "..."))
        self.filteredLoadedLabel.setText(_translate("preferencesDialog", "Filters Loaded"))
        self.beqDirectoryLabel.setText(_translate("preferencesDialog", "Directory"))
        self.beqDirectoryPicker.setText(_translate("preferencesDialog", "..."))

