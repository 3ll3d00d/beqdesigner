# Form implementation generated from reading ui file 'extract.ui'
#
# Created by: PyQt6 UI code generator 6.7.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_extractAudioDialog(object):
    def setupUi(self, extractAudioDialog):
        extractAudioDialog.setObjectName("extractAudioDialog")
        extractAudioDialog.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        extractAudioDialog.resize(880, 1027)
        extractAudioDialog.setSizeGripEnabled(True)
        extractAudioDialog.setModal(False)
        self.boxLayout = QtWidgets.QVBoxLayout(extractAudioDialog)
        self.boxLayout.setObjectName("boxLayout")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.ffmpegOutput = QtWidgets.QPlainTextEdit(parent=extractAudioDialog)
        self.ffmpegOutput.setEnabled(True)
        font = QtGui.QFont()
        font.setFamily("Consolas")
        self.ffmpegOutput.setFont(font)
        self.ffmpegOutput.setReadOnly(True)
        self.ffmpegOutput.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.ffmpegOutput.setObjectName("ffmpegOutput")
        self.gridLayout.addWidget(self.ffmpegOutput, 13, 1, 1, 1)
        self.streamsLabel = QtWidgets.QLabel(parent=extractAudioDialog)
        self.streamsLabel.setObjectName("streamsLabel")
        self.gridLayout.addWidget(self.streamsLabel, 1, 0, 1, 1)
        self.targetDirPicker = QtWidgets.QToolButton(parent=extractAudioDialog)
        self.targetDirPicker.setObjectName("targetDirPicker")
        self.gridLayout.addWidget(self.targetDirPicker, 9, 2, 1, 1)
        self.targetDir = QtWidgets.QLineEdit(parent=extractAudioDialog)
        self.targetDir.setEnabled(False)
        self.targetDir.setObjectName("targetDir")
        self.gridLayout.addWidget(self.targetDir, 9, 1, 1, 1)
        self.showProbeButton = QtWidgets.QToolButton(parent=extractAudioDialog)
        self.showProbeButton.setObjectName("showProbeButton")
        self.gridLayout.addWidget(self.showProbeButton, 1, 2, 2, 1)
        self.signalName = QtWidgets.QLineEdit(parent=extractAudioDialog)
        self.signalName.setEnabled(True)
        self.signalName.setObjectName("signalName")
        self.gridLayout.addWidget(self.signalName, 14, 1, 1, 1)
        self.inputFilePicker = QtWidgets.QToolButton(parent=extractAudioDialog)
        self.inputFilePicker.setObjectName("inputFilePicker")
        self.gridLayout.addWidget(self.inputFilePicker, 0, 2, 1, 1)
        self.showRemuxCommand = QtWidgets.QToolButton(parent=extractAudioDialog)
        self.showRemuxCommand.setObjectName("showRemuxCommand")
        self.gridLayout.addWidget(self.showRemuxCommand, 11, 2, 1, 1, QtCore.Qt.AlignmentFlag.AlignTop)
        self.targetDirectoryLabel = QtWidgets.QLabel(parent=extractAudioDialog)
        self.targetDirectoryLabel.setObjectName("targetDirectoryLabel")
        self.gridLayout.addWidget(self.targetDirectoryLabel, 9, 0, 1, 1)
        self.filterMapping = QtWidgets.QListWidget(parent=extractAudioDialog)
        self.filterMapping.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.filterMapping.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.filterMapping.setObjectName("filterMapping")
        self.gridLayout.addWidget(self.filterMapping, 8, 1, 1, 1)
        self.inputLayout = QtWidgets.QHBoxLayout()
        self.inputLayout.setObjectName("inputLayout")
        self.inputFile = QtWidgets.QLineEdit(parent=extractAudioDialog)
        self.inputFile.setEnabled(False)
        self.inputFile.setObjectName("inputFile")
        self.inputLayout.addWidget(self.inputFile)
        self.inputDrop = DropArea(parent=extractAudioDialog)
        self.inputDrop.setMinimumSize(QtCore.QSize(100, 0))
        self.inputDrop.setAutoFillBackground(False)
        self.inputDrop.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.inputDrop.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.inputDrop.setText("")
        self.inputDrop.setScaledContents(False)
        self.inputDrop.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.inputDrop.setObjectName("inputDrop")
        self.inputLayout.addWidget(self.inputDrop)
        self.inputLayout.setStretch(0, 1)
        self.gridLayout.addLayout(self.inputLayout, 0, 1, 1, 1)
        self.ffmpegCommandLine = QtWidgets.QPlainTextEdit(parent=extractAudioDialog)
        self.ffmpegCommandLine.setEnabled(True)
        font = QtGui.QFont()
        font.setFamily("Consolas")
        self.ffmpegCommandLine.setFont(font)
        self.ffmpegCommandLine.setReadOnly(True)
        self.ffmpegCommandLine.setObjectName("ffmpegCommandLine")
        self.gridLayout.addWidget(self.ffmpegCommandLine, 11, 1, 1, 1)
        self.ffmpegCommandLabel = QtWidgets.QLabel(parent=extractAudioDialog)
        self.ffmpegCommandLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeading|QtCore.Qt.AlignmentFlag.AlignLeft|QtCore.Qt.AlignmentFlag.AlignTop)
        self.ffmpegCommandLabel.setObjectName("ffmpegCommandLabel")
        self.gridLayout.addWidget(self.ffmpegCommandLabel, 11, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.monoMix = QtWidgets.QCheckBox(parent=extractAudioDialog)
        self.monoMix.setEnabled(True)
        self.monoMix.setChecked(True)
        self.monoMix.setObjectName("monoMix")
        self.horizontalLayout.addWidget(self.monoMix)
        self.decimateAudio = QtWidgets.QCheckBox(parent=extractAudioDialog)
        self.decimateAudio.setChecked(True)
        self.decimateAudio.setObjectName("decimateAudio")
        self.horizontalLayout.addWidget(self.decimateAudio)
        self.includeSubtitles = QtWidgets.QCheckBox(parent=extractAudioDialog)
        self.includeSubtitles.setToolTip("")
        self.includeSubtitles.setObjectName("includeSubtitles")
        self.horizontalLayout.addWidget(self.includeSubtitles)
        self.bassManage = QtWidgets.QCheckBox(parent=extractAudioDialog)
        self.bassManage.setObjectName("bassManage")
        self.horizontalLayout.addWidget(self.bassManage)
        self.gridLayout.addLayout(self.horizontalLayout, 6, 1, 1, 1)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.audioStreams = QtWidgets.QComboBox(parent=extractAudioDialog)
        self.audioStreams.setObjectName("audioStreams")
        self.horizontalLayout_3.addWidget(self.audioStreams)
        self.videoStreams = QtWidgets.QComboBox(parent=extractAudioDialog)
        self.videoStreams.setObjectName("videoStreams")
        self.horizontalLayout_3.addWidget(self.videoStreams)
        self.gridLayout.addLayout(self.horizontalLayout_3, 1, 1, 1, 1)
        self.channelsLabel = QtWidgets.QLabel(parent=extractAudioDialog)
        self.channelsLabel.setObjectName("channelsLabel")
        self.gridLayout.addWidget(self.channelsLabel, 3, 0, 1, 1)
        self.signalNameLabel = QtWidgets.QLabel(parent=extractAudioDialog)
        self.signalNameLabel.setObjectName("signalNameLabel")
        self.gridLayout.addWidget(self.signalNameLabel, 14, 0, 1, 1)
        self.rangeLayout = QtWidgets.QHBoxLayout()
        self.rangeLayout.setObjectName("rangeLayout")
        self.limitRange = QtWidgets.QToolButton(parent=extractAudioDialog)
        self.limitRange.setCheckable(True)
        self.limitRange.setObjectName("limitRange")
        self.rangeLayout.addWidget(self.limitRange)
        self.rangeFrom = QtWidgets.QTimeEdit(parent=extractAudioDialog)
        self.rangeFrom.setObjectName("rangeFrom")
        self.rangeLayout.addWidget(self.rangeFrom)
        self.rangeSeparatorLabel = QtWidgets.QLabel(parent=extractAudioDialog)
        self.rangeSeparatorLabel.setObjectName("rangeSeparatorLabel")
        self.rangeLayout.addWidget(self.rangeSeparatorLabel)
        self.rangeTo = QtWidgets.QTimeEdit(parent=extractAudioDialog)
        self.rangeTo.setObjectName("rangeTo")
        self.rangeLayout.addWidget(self.rangeTo)
        self.rangeLayout.setStretch(1, 1)
        self.rangeLayout.setStretch(3, 1)
        self.gridLayout.addLayout(self.rangeLayout, 5, 1, 1, 1)
        self.ffmpegProgress = QtWidgets.QProgressBar(parent=extractAudioDialog)
        self.ffmpegProgress.setProperty("value", 0)
        self.ffmpegProgress.setObjectName("ffmpegProgress")
        self.gridLayout.addWidget(self.ffmpegProgress, 12, 1, 1, 1)
        self.outputFilename = QtWidgets.QLineEdit(parent=extractAudioDialog)
        self.outputFilename.setObjectName("outputFilename")
        self.gridLayout.addWidget(self.outputFilename, 10, 1, 1, 1)
        self.outputFilenameLabel = QtWidgets.QLabel(parent=extractAudioDialog)
        self.outputFilenameLabel.setObjectName("outputFilenameLabel")
        self.gridLayout.addWidget(self.outputFilenameLabel, 10, 0, 1, 1)
        self.ffmpegProgressLabel = QtWidgets.QLabel(parent=extractAudioDialog)
        self.ffmpegProgressLabel.setObjectName("ffmpegProgressLabel")
        self.gridLayout.addWidget(self.ffmpegProgressLabel, 12, 0, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.lfeChannelIndex = QtWidgets.QSpinBox(parent=extractAudioDialog)
        self.lfeChannelIndex.setMinimum(0)
        self.lfeChannelIndex.setObjectName("lfeChannelIndex")
        self.horizontalLayout_2.addWidget(self.lfeChannelIndex)
        self.channelCount = QtWidgets.QSpinBox(parent=extractAudioDialog)
        self.channelCount.setMinimum(1)
        self.channelCount.setObjectName("channelCount")
        self.horizontalLayout_2.addWidget(self.channelCount)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.audioFormatLabel = QtWidgets.QLabel(parent=extractAudioDialog)
        self.audioFormatLabel.setObjectName("audioFormatLabel")
        self.horizontalLayout_2.addWidget(self.audioFormatLabel)
        self.audioFormat = QtWidgets.QComboBox(parent=extractAudioDialog)
        self.audioFormat.setObjectName("audioFormat")
        self.horizontalLayout_2.addWidget(self.audioFormat)
        self.eacBitRate = QtWidgets.QSpinBox(parent=extractAudioDialog)
        self.eacBitRate.setMinimum(32)
        self.eacBitRate.setMaximum(6000)
        self.eacBitRate.setProperty("value", 1500)
        self.eacBitRate.setObjectName("eacBitRate")
        self.horizontalLayout_2.addWidget(self.eacBitRate)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.gridLayout.addLayout(self.horizontalLayout_2, 3, 1, 1, 1)
        self.inputFileLabel = QtWidgets.QLabel(parent=extractAudioDialog)
        self.inputFileLabel.setObjectName("inputFileLabel")
        self.gridLayout.addWidget(self.inputFileLabel, 0, 0, 1, 1)
        self.label = QtWidgets.QLabel(parent=extractAudioDialog)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 5, 0, 1, 1)
        self.ffmpegOutputLabel = QtWidgets.QLabel(parent=extractAudioDialog)
        self.ffmpegOutputLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeading|QtCore.Qt.AlignmentFlag.AlignLeft|QtCore.Qt.AlignmentFlag.AlignTop)
        self.ffmpegOutputLabel.setObjectName("ffmpegOutputLabel")
        self.gridLayout.addWidget(self.ffmpegOutputLabel, 13, 0, 1, 1)
        self.filterMappingLabel = QtWidgets.QLabel(parent=extractAudioDialog)
        self.filterMappingLabel.setObjectName("filterMappingLabel")
        self.gridLayout.addWidget(self.filterMappingLabel, 8, 0, 1, 1)
        self.remuxOptionsLayout = QtWidgets.QHBoxLayout()
        self.remuxOptionsLayout.setObjectName("remuxOptionsLayout")
        self.includeOriginalAudio = QtWidgets.QCheckBox(parent=extractAudioDialog)
        self.includeOriginalAudio.setObjectName("includeOriginalAudio")
        self.remuxOptionsLayout.addWidget(self.includeOriginalAudio)
        self.gainOffsetLabel = QtWidgets.QLabel(parent=extractAudioDialog)
        self.gainOffsetLabel.setObjectName("gainOffsetLabel")
        self.remuxOptionsLayout.addWidget(self.gainOffsetLabel)
        self.gainOffset = QtWidgets.QDoubleSpinBox(parent=extractAudioDialog)
        self.gainOffset.setMinimum(-100.0)
        self.gainOffset.setMaximum(100.0)
        self.gainOffset.setSingleStep(0.01)
        self.gainOffset.setObjectName("gainOffset")
        self.remuxOptionsLayout.addWidget(self.gainOffset)
        self.adjustRemuxedAudio = QtWidgets.QCheckBox(parent=extractAudioDialog)
        self.adjustRemuxedAudio.setChecked(True)
        self.adjustRemuxedAudio.setObjectName("adjustRemuxedAudio")
        self.remuxOptionsLayout.addWidget(self.adjustRemuxedAudio)
        self.remuxedAudioOffset = QtWidgets.QDoubleSpinBox(parent=extractAudioDialog)
        self.remuxedAudioOffset.setMinimum(-120.0)
        self.remuxedAudioOffset.setMaximum(12.0)
        self.remuxedAudioOffset.setSingleStep(0.01)
        self.remuxedAudioOffset.setObjectName("remuxedAudioOffset")
        self.remuxOptionsLayout.addWidget(self.remuxedAudioOffset)
        self.remuxOptionsLayout.setStretch(0, 1)
        self.remuxOptionsLayout.setStretch(3, 1)
        self.gridLayout.addLayout(self.remuxOptionsLayout, 7, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(parent=extractAudioDialog)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 6, 0, 2, 1)
        self.calculateGainAdjustment = QtWidgets.QToolButton(parent=extractAudioDialog)
        self.calculateGainAdjustment.setObjectName("calculateGainAdjustment")
        self.gridLayout.addWidget(self.calculateGainAdjustment, 7, 2, 1, 1)
        self.gridLayout.setRowStretch(0, 1)
        self.boxLayout.addLayout(self.gridLayout)
        self.buttonLayout = QtWidgets.QGridLayout()
        self.buttonLayout.setObjectName("buttonLayout")
        self.buttonBox = QtWidgets.QDialogButtonBox(parent=extractAudioDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.StandardButton.Cancel|QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.buttonLayout.addWidget(self.buttonBox, 0, 0, 1, 1)
        self.boxLayout.addLayout(self.buttonLayout)

        self.retranslateUi(extractAudioDialog)
        self.buttonBox.accepted.connect(extractAudioDialog.accept) # type: ignore
        self.buttonBox.rejected.connect(extractAudioDialog.reject) # type: ignore
        self.inputFilePicker.clicked.connect(extractAudioDialog.selectFile) # type: ignore
        self.showProbeButton.clicked.connect(extractAudioDialog.showProbeInDetail) # type: ignore
        self.targetDirPicker.clicked.connect(extractAudioDialog.setTargetDirectory) # type: ignore
        self.outputFilename.editingFinished.connect(extractAudioDialog.updateOutputFilename) # type: ignore
        self.audioStreams.currentIndexChanged['int'].connect(extractAudioDialog.updateFfmpegSpec) # type: ignore
        self.outputFilename.editingFinished.connect(extractAudioDialog.updateOutputFilename) # type: ignore
        self.monoMix.clicked.connect(extractAudioDialog.toggleMonoMix) # type: ignore
        self.lfeChannelIndex.valueChanged['int'].connect(extractAudioDialog.overrideFfmpegSpec) # type: ignore
        self.channelCount.valueChanged['int'].connect(extractAudioDialog.overrideFfmpegSpec) # type: ignore
        self.decimateAudio.clicked.connect(extractAudioDialog.toggle_decimate_audio) # type: ignore
        self.includeOriginalAudio.clicked.connect(extractAudioDialog.update_original_audio) # type: ignore
        self.rangeTo.timeChanged['QTime'].connect(extractAudioDialog.update_end_time) # type: ignore
        self.rangeFrom.timeChanged['QTime'].connect(extractAudioDialog.update_start_time) # type: ignore
        self.limitRange.clicked.connect(extractAudioDialog.toggle_range) # type: ignore
        self.showRemuxCommand.clicked.connect(extractAudioDialog.show_remux_cmd) # type: ignore
        self.includeSubtitles.clicked.connect(extractAudioDialog.toggle_include_subtitles) # type: ignore
        self.gainOffset.valueChanged['double'].connect(extractAudioDialog.update_original_audio) # type: ignore
        self.videoStreams.currentIndexChanged['int'].connect(extractAudioDialog.onVideoStreamChange) # type: ignore
        self.audioFormat.currentTextChanged['QString'].connect(extractAudioDialog.change_audio_format) # type: ignore
        self.eacBitRate.valueChanged['int'].connect(extractAudioDialog.change_audio_bitrate) # type: ignore
        self.calculateGainAdjustment.clicked.connect(extractAudioDialog.calculate_gain_adjustment) # type: ignore
        self.remuxedAudioOffset.valueChanged['double'].connect(extractAudioDialog.override_filtered_gain_adjustment) # type: ignore
        self.bassManage.clicked.connect(extractAudioDialog.toggle_bass_manage) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(extractAudioDialog)

    def retranslateUi(self, extractAudioDialog):
        _translate = QtCore.QCoreApplication.translate
        extractAudioDialog.setWindowTitle(_translate("extractAudioDialog", "Extract Audio"))
        self.streamsLabel.setText(_translate("extractAudioDialog", "A/V Streams"))
        self.targetDirPicker.setText(_translate("extractAudioDialog", "..."))
        self.showProbeButton.setText(_translate("extractAudioDialog", "..."))
        self.inputFilePicker.setText(_translate("extractAudioDialog", "..."))
        self.showRemuxCommand.setText(_translate("extractAudioDialog", "..."))
        self.targetDirectoryLabel.setText(_translate("extractAudioDialog", "Target Directory"))
        self.ffmpegCommandLabel.setText(_translate("extractAudioDialog", "ffmpeg command "))
        self.monoMix.setText(_translate("extractAudioDialog", "Mix to Mono?"))
        self.decimateAudio.setText(_translate("extractAudioDialog", "Decimate Audio?"))
        self.includeSubtitles.setText(_translate("extractAudioDialog", "Add Subtitles?"))
        self.bassManage.setText(_translate("extractAudioDialog", "Bass Manage?"))
        self.channelsLabel.setText(_translate("extractAudioDialog", "LFE Channel/Total"))
        self.signalNameLabel.setText(_translate("extractAudioDialog", "Signal Name"))
        self.limitRange.setText(_translate("extractAudioDialog", "..."))
        self.rangeFrom.setDisplayFormat(_translate("extractAudioDialog", "HH:mm:ss.zzz"))
        self.rangeSeparatorLabel.setText(_translate("extractAudioDialog", "to"))
        self.rangeTo.setDisplayFormat(_translate("extractAudioDialog", "HH:mm:ss.zzz"))
        self.outputFilenameLabel.setText(_translate("extractAudioDialog", "Output Filename"))
        self.ffmpegProgressLabel.setText(_translate("extractAudioDialog", "Progress"))
        self.lfeChannelIndex.setToolTip(_translate("extractAudioDialog", "0 = No LFE"))
        self.audioFormatLabel.setText(_translate("extractAudioDialog", "Format"))
        self.eacBitRate.setSuffix(_translate("extractAudioDialog", " kbps"))
        self.inputFileLabel.setText(_translate("extractAudioDialog", "File"))
        self.label.setText(_translate("extractAudioDialog", "Range"))
        self.ffmpegOutputLabel.setText(_translate("extractAudioDialog", "ffmpeg output"))
        self.filterMappingLabel.setText(_translate("extractAudioDialog", "Signal Mapping"))
        self.includeOriginalAudio.setText(_translate("extractAudioDialog", "Add Original Audio?"))
        self.gainOffsetLabel.setText(_translate("extractAudioDialog", "Offset:"))
        self.gainOffset.setSuffix(_translate("extractAudioDialog", " dB"))
        self.adjustRemuxedAudio.setText(_translate("extractAudioDialog", "Adjust Remuxed Audio?"))
        self.remuxedAudioOffset.setSuffix(_translate("extractAudioDialog", " dB"))
        self.label_2.setText(_translate("extractAudioDialog", "Options"))
        self.calculateGainAdjustment.setText(_translate("extractAudioDialog", "..."))
from ui.drop import DropArea
