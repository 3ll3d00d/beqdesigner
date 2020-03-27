# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'extract.ui'
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

from ui.drop import DropArea


class Ui_extractAudioDialog(object):
    def setupUi(self, extractAudioDialog):
        if extractAudioDialog.objectName():
            extractAudioDialog.setObjectName(u"extractAudioDialog")
        extractAudioDialog.setWindowModality(Qt.ApplicationModal)
        extractAudioDialog.resize(880, 1027)
        extractAudioDialog.setSizeGripEnabled(True)
        extractAudioDialog.setModal(False)
        self.boxLayout = QVBoxLayout(extractAudioDialog)
        self.boxLayout.setObjectName(u"boxLayout")
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.ffmpegOutput = QPlainTextEdit(extractAudioDialog)
        self.ffmpegOutput.setObjectName(u"ffmpegOutput")
        self.ffmpegOutput.setEnabled(True)
        font = QFont()
        font.setFamily(u"Consolas")
        self.ffmpegOutput.setFont(font)
        self.ffmpegOutput.setReadOnly(True)
        self.ffmpegOutput.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.gridLayout.addWidget(self.ffmpegOutput, 13, 1, 1, 1)

        self.streamsLabel = QLabel(extractAudioDialog)
        self.streamsLabel.setObjectName(u"streamsLabel")

        self.gridLayout.addWidget(self.streamsLabel, 1, 0, 1, 1)

        self.targetDirPicker = QToolButton(extractAudioDialog)
        self.targetDirPicker.setObjectName(u"targetDirPicker")

        self.gridLayout.addWidget(self.targetDirPicker, 9, 2, 1, 1)

        self.targetDir = QLineEdit(extractAudioDialog)
        self.targetDir.setObjectName(u"targetDir")
        self.targetDir.setEnabled(False)

        self.gridLayout.addWidget(self.targetDir, 9, 1, 1, 1)

        self.showProbeButton = QToolButton(extractAudioDialog)
        self.showProbeButton.setObjectName(u"showProbeButton")

        self.gridLayout.addWidget(self.showProbeButton, 1, 2, 2, 1)

        self.signalName = QLineEdit(extractAudioDialog)
        self.signalName.setObjectName(u"signalName")
        self.signalName.setEnabled(True)

        self.gridLayout.addWidget(self.signalName, 14, 1, 1, 1)

        self.inputFilePicker = QToolButton(extractAudioDialog)
        self.inputFilePicker.setObjectName(u"inputFilePicker")

        self.gridLayout.addWidget(self.inputFilePicker, 0, 2, 1, 1)

        self.showRemuxCommand = QToolButton(extractAudioDialog)
        self.showRemuxCommand.setObjectName(u"showRemuxCommand")

        self.gridLayout.addWidget(self.showRemuxCommand, 11, 2, 1, 1, Qt.AlignTop)

        self.targetDirectoryLabel = QLabel(extractAudioDialog)
        self.targetDirectoryLabel.setObjectName(u"targetDirectoryLabel")

        self.gridLayout.addWidget(self.targetDirectoryLabel, 9, 0, 1, 1)

        self.filterMapping = QListWidget(extractAudioDialog)
        self.filterMapping.setObjectName(u"filterMapping")
        self.filterMapping.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.filterMapping.setSelectionMode(QAbstractItemView.NoSelection)

        self.gridLayout.addWidget(self.filterMapping, 8, 1, 1, 1)

        self.inputLayout = QHBoxLayout()
        self.inputLayout.setObjectName(u"inputLayout")
        self.inputFile = QLineEdit(extractAudioDialog)
        self.inputFile.setObjectName(u"inputFile")
        self.inputFile.setEnabled(False)

        self.inputLayout.addWidget(self.inputFile)

        self.inputDrop = DropArea(extractAudioDialog)
        self.inputDrop.setObjectName(u"inputDrop")
        self.inputDrop.setMinimumSize(QSize(100, 0))
        self.inputDrop.setAutoFillBackground(False)
        self.inputDrop.setFrameShape(QFrame.StyledPanel)
        self.inputDrop.setFrameShadow(QFrame.Sunken)
        self.inputDrop.setScaledContents(False)
        self.inputDrop.setAlignment(Qt.AlignCenter)

        self.inputLayout.addWidget(self.inputDrop)

        self.inputLayout.setStretch(0, 1)

        self.gridLayout.addLayout(self.inputLayout, 0, 1, 1, 1)

        self.ffmpegCommandLine = QPlainTextEdit(extractAudioDialog)
        self.ffmpegCommandLine.setObjectName(u"ffmpegCommandLine")
        self.ffmpegCommandLine.setEnabled(True)
        self.ffmpegCommandLine.setFont(font)
        self.ffmpegCommandLine.setReadOnly(True)

        self.gridLayout.addWidget(self.ffmpegCommandLine, 11, 1, 1, 1)

        self.ffmpegCommandLabel = QLabel(extractAudioDialog)
        self.ffmpegCommandLabel.setObjectName(u"ffmpegCommandLabel")
        self.ffmpegCommandLabel.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignTop)

        self.gridLayout.addWidget(self.ffmpegCommandLabel, 11, 0, 1, 1)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.monoMix = QCheckBox(extractAudioDialog)
        self.monoMix.setObjectName(u"monoMix")
        self.monoMix.setEnabled(True)
        self.monoMix.setChecked(True)

        self.horizontalLayout.addWidget(self.monoMix)

        self.decimateAudio = QCheckBox(extractAudioDialog)
        self.decimateAudio.setObjectName(u"decimateAudio")
        self.decimateAudio.setChecked(True)

        self.horizontalLayout.addWidget(self.decimateAudio)

        self.includeSubtitles = QCheckBox(extractAudioDialog)
        self.includeSubtitles.setObjectName(u"includeSubtitles")

        self.horizontalLayout.addWidget(self.includeSubtitles)

        self.bassManage = QCheckBox(extractAudioDialog)
        self.bassManage.setObjectName(u"bassManage")

        self.horizontalLayout.addWidget(self.bassManage)


        self.gridLayout.addLayout(self.horizontalLayout, 6, 1, 1, 1)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.audioStreams = QComboBox(extractAudioDialog)
        self.audioStreams.setObjectName(u"audioStreams")

        self.horizontalLayout_3.addWidget(self.audioStreams)

        self.videoStreams = QComboBox(extractAudioDialog)
        self.videoStreams.setObjectName(u"videoStreams")

        self.horizontalLayout_3.addWidget(self.videoStreams)


        self.gridLayout.addLayout(self.horizontalLayout_3, 1, 1, 1, 1)

        self.channelsLabel = QLabel(extractAudioDialog)
        self.channelsLabel.setObjectName(u"channelsLabel")

        self.gridLayout.addWidget(self.channelsLabel, 3, 0, 1, 1)

        self.signalNameLabel = QLabel(extractAudioDialog)
        self.signalNameLabel.setObjectName(u"signalNameLabel")

        self.gridLayout.addWidget(self.signalNameLabel, 14, 0, 1, 1)

        self.rangeLayout = QHBoxLayout()
        self.rangeLayout.setObjectName(u"rangeLayout")
        self.limitRange = QToolButton(extractAudioDialog)
        self.limitRange.setObjectName(u"limitRange")
        self.limitRange.setCheckable(True)

        self.rangeLayout.addWidget(self.limitRange)

        self.rangeFrom = QTimeEdit(extractAudioDialog)
        self.rangeFrom.setObjectName(u"rangeFrom")

        self.rangeLayout.addWidget(self.rangeFrom)

        self.rangeSeparatorLabel = QLabel(extractAudioDialog)
        self.rangeSeparatorLabel.setObjectName(u"rangeSeparatorLabel")

        self.rangeLayout.addWidget(self.rangeSeparatorLabel)

        self.rangeTo = QTimeEdit(extractAudioDialog)
        self.rangeTo.setObjectName(u"rangeTo")

        self.rangeLayout.addWidget(self.rangeTo)

        self.rangeLayout.setStretch(1, 1)
        self.rangeLayout.setStretch(3, 1)

        self.gridLayout.addLayout(self.rangeLayout, 5, 1, 1, 1)

        self.ffmpegProgress = QProgressBar(extractAudioDialog)
        self.ffmpegProgress.setObjectName(u"ffmpegProgress")
        self.ffmpegProgress.setValue(0)

        self.gridLayout.addWidget(self.ffmpegProgress, 12, 1, 1, 1)

        self.outputFilename = QLineEdit(extractAudioDialog)
        self.outputFilename.setObjectName(u"outputFilename")

        self.gridLayout.addWidget(self.outputFilename, 10, 1, 1, 1)

        self.outputFilenameLabel = QLabel(extractAudioDialog)
        self.outputFilenameLabel.setObjectName(u"outputFilenameLabel")

        self.gridLayout.addWidget(self.outputFilenameLabel, 10, 0, 1, 1)

        self.ffmpegProgressLabel = QLabel(extractAudioDialog)
        self.ffmpegProgressLabel.setObjectName(u"ffmpegProgressLabel")

        self.gridLayout.addWidget(self.ffmpegProgressLabel, 12, 0, 1, 1)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.lfeChannelIndex = QSpinBox(extractAudioDialog)
        self.lfeChannelIndex.setObjectName(u"lfeChannelIndex")
        self.lfeChannelIndex.setMinimum(0)

        self.horizontalLayout_2.addWidget(self.lfeChannelIndex)

        self.channelCount = QSpinBox(extractAudioDialog)
        self.channelCount.setObjectName(u"channelCount")
        self.channelCount.setMinimum(1)

        self.horizontalLayout_2.addWidget(self.channelCount)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer)

        self.audioFormatLabel = QLabel(extractAudioDialog)
        self.audioFormatLabel.setObjectName(u"audioFormatLabel")

        self.horizontalLayout_2.addWidget(self.audioFormatLabel)

        self.audioFormat = QComboBox(extractAudioDialog)
        self.audioFormat.setObjectName(u"audioFormat")

        self.horizontalLayout_2.addWidget(self.audioFormat)

        self.eacBitRate = QSpinBox(extractAudioDialog)
        self.eacBitRate.setObjectName(u"eacBitRate")
        self.eacBitRate.setMinimum(32)
        self.eacBitRate.setMaximum(6000)
        self.eacBitRate.setValue(1500)

        self.horizontalLayout_2.addWidget(self.eacBitRate)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_2)


        self.gridLayout.addLayout(self.horizontalLayout_2, 3, 1, 1, 1)

        self.inputFileLabel = QLabel(extractAudioDialog)
        self.inputFileLabel.setObjectName(u"inputFileLabel")

        self.gridLayout.addWidget(self.inputFileLabel, 0, 0, 1, 1)

        self.label = QLabel(extractAudioDialog)
        self.label.setObjectName(u"label")

        self.gridLayout.addWidget(self.label, 5, 0, 1, 1)

        self.ffmpegOutputLabel = QLabel(extractAudioDialog)
        self.ffmpegOutputLabel.setObjectName(u"ffmpegOutputLabel")
        self.ffmpegOutputLabel.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignTop)

        self.gridLayout.addWidget(self.ffmpegOutputLabel, 13, 0, 1, 1)

        self.filterMappingLabel = QLabel(extractAudioDialog)
        self.filterMappingLabel.setObjectName(u"filterMappingLabel")

        self.gridLayout.addWidget(self.filterMappingLabel, 8, 0, 1, 1)

        self.remuxOptionsLayout = QHBoxLayout()
        self.remuxOptionsLayout.setObjectName(u"remuxOptionsLayout")
        self.includeOriginalAudio = QCheckBox(extractAudioDialog)
        self.includeOriginalAudio.setObjectName(u"includeOriginalAudio")

        self.remuxOptionsLayout.addWidget(self.includeOriginalAudio)

        self.gainOffsetLabel = QLabel(extractAudioDialog)
        self.gainOffsetLabel.setObjectName(u"gainOffsetLabel")

        self.remuxOptionsLayout.addWidget(self.gainOffsetLabel)

        self.gainOffset = QDoubleSpinBox(extractAudioDialog)
        self.gainOffset.setObjectName(u"gainOffset")
        self.gainOffset.setMinimum(-100.000000000000000)
        self.gainOffset.setMaximum(100.000000000000000)
        self.gainOffset.setSingleStep(0.010000000000000)

        self.remuxOptionsLayout.addWidget(self.gainOffset)

        self.adjustRemuxedAudio = QCheckBox(extractAudioDialog)
        self.adjustRemuxedAudio.setObjectName(u"adjustRemuxedAudio")
        self.adjustRemuxedAudio.setChecked(True)

        self.remuxOptionsLayout.addWidget(self.adjustRemuxedAudio)

        self.remuxedAudioOffset = QDoubleSpinBox(extractAudioDialog)
        self.remuxedAudioOffset.setObjectName(u"remuxedAudioOffset")
        self.remuxedAudioOffset.setMinimum(-120.000000000000000)
        self.remuxedAudioOffset.setMaximum(12.000000000000000)
        self.remuxedAudioOffset.setSingleStep(0.010000000000000)

        self.remuxOptionsLayout.addWidget(self.remuxedAudioOffset)

        self.remuxOptionsLayout.setStretch(0, 1)
        self.remuxOptionsLayout.setStretch(3, 1)

        self.gridLayout.addLayout(self.remuxOptionsLayout, 7, 1, 1, 1)

        self.label_2 = QLabel(extractAudioDialog)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout.addWidget(self.label_2, 6, 0, 2, 1)

        self.calculateGainAdjustment = QToolButton(extractAudioDialog)
        self.calculateGainAdjustment.setObjectName(u"calculateGainAdjustment")

        self.gridLayout.addWidget(self.calculateGainAdjustment, 7, 2, 1, 1)

        self.gridLayout.setRowStretch(0, 1)

        self.boxLayout.addLayout(self.gridLayout)

        self.buttonLayout = QGridLayout()
        self.buttonLayout.setObjectName(u"buttonLayout")
        self.buttonBox = QDialogButtonBox(extractAudioDialog)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)

        self.buttonLayout.addWidget(self.buttonBox, 0, 0, 1, 1)


        self.boxLayout.addLayout(self.buttonLayout)


        self.retranslateUi(extractAudioDialog)
        self.buttonBox.accepted.connect(extractAudioDialog.accept)
        self.buttonBox.rejected.connect(extractAudioDialog.reject)
        self.inputFilePicker.clicked.connect(extractAudioDialog.selectFile)
        self.showProbeButton.clicked.connect(extractAudioDialog.showProbeInDetail)
        self.targetDirPicker.clicked.connect(extractAudioDialog.setTargetDirectory)
        self.outputFilename.editingFinished.connect(extractAudioDialog.updateOutputFilename)
        self.audioStreams.currentIndexChanged.connect(extractAudioDialog.updateFfmpegSpec)
        self.outputFilename.editingFinished.connect(extractAudioDialog.updateOutputFilename)
        self.monoMix.clicked.connect(extractAudioDialog.toggleMonoMix)
        self.lfeChannelIndex.valueChanged.connect(extractAudioDialog.overrideFfmpegSpec)
        self.channelCount.valueChanged.connect(extractAudioDialog.overrideFfmpegSpec)
        self.decimateAudio.clicked.connect(extractAudioDialog.toggle_decimate_audio)
        self.includeOriginalAudio.clicked.connect(extractAudioDialog.update_original_audio)
        self.rangeTo.timeChanged.connect(extractAudioDialog.update_end_time)
        self.rangeFrom.timeChanged.connect(extractAudioDialog.update_start_time)
        self.limitRange.clicked.connect(extractAudioDialog.toggle_range)
        self.showRemuxCommand.clicked.connect(extractAudioDialog.show_remux_cmd)
        self.includeSubtitles.clicked.connect(extractAudioDialog.toggle_include_subtitles)
        self.gainOffset.valueChanged.connect(extractAudioDialog.update_original_audio)
        self.videoStreams.currentIndexChanged.connect(extractAudioDialog.onVideoStreamChange)
        self.audioFormat.currentTextChanged.connect(extractAudioDialog.change_audio_format)
        self.eacBitRate.valueChanged.connect(extractAudioDialog.change_audio_bitrate)
        self.calculateGainAdjustment.clicked.connect(extractAudioDialog.calculate_gain_adjustment)
        self.remuxedAudioOffset.valueChanged.connect(extractAudioDialog.override_filtered_gain_adjustment)
        self.bassManage.clicked.connect(extractAudioDialog.toggle_bass_manage)

        QMetaObject.connectSlotsByName(extractAudioDialog)
    # setupUi

    def retranslateUi(self, extractAudioDialog):
        extractAudioDialog.setWindowTitle(QCoreApplication.translate("extractAudioDialog", u"Extract Audio", None))
        self.streamsLabel.setText(QCoreApplication.translate("extractAudioDialog", u"A/V Streams", None))
        self.targetDirPicker.setText(QCoreApplication.translate("extractAudioDialog", u"...", None))
        self.showProbeButton.setText(QCoreApplication.translate("extractAudioDialog", u"...", None))
        self.inputFilePicker.setText(QCoreApplication.translate("extractAudioDialog", u"...", None))
        self.showRemuxCommand.setText(QCoreApplication.translate("extractAudioDialog", u"...", None))
        self.targetDirectoryLabel.setText(QCoreApplication.translate("extractAudioDialog", u"Target Directory", None))
        self.inputDrop.setText("")
        self.ffmpegCommandLabel.setText(QCoreApplication.translate("extractAudioDialog", u"ffmpeg command ", None))
        self.monoMix.setText(QCoreApplication.translate("extractAudioDialog", u"Mix to Mono?", None))
        self.decimateAudio.setText(QCoreApplication.translate("extractAudioDialog", u"Decimate Audio?", None))
#if QT_CONFIG(tooltip)
        self.includeSubtitles.setToolTip("")
#endif // QT_CONFIG(tooltip)
        self.includeSubtitles.setText(QCoreApplication.translate("extractAudioDialog", u"Add Subtitles?", None))
        self.bassManage.setText(QCoreApplication.translate("extractAudioDialog", u"Bass Manage?", None))
        self.channelsLabel.setText(QCoreApplication.translate("extractAudioDialog", u"LFE Channel/Total", None))
        self.signalNameLabel.setText(QCoreApplication.translate("extractAudioDialog", u"Signal Name", None))
        self.limitRange.setText(QCoreApplication.translate("extractAudioDialog", u"...", None))
        self.rangeFrom.setDisplayFormat(QCoreApplication.translate("extractAudioDialog", u"HH:mm:ss.zzz", None))
        self.rangeSeparatorLabel.setText(QCoreApplication.translate("extractAudioDialog", u"to", None))
        self.rangeTo.setDisplayFormat(QCoreApplication.translate("extractAudioDialog", u"HH:mm:ss.zzz", None))
        self.outputFilenameLabel.setText(QCoreApplication.translate("extractAudioDialog", u"Output Filename", None))
        self.ffmpegProgressLabel.setText(QCoreApplication.translate("extractAudioDialog", u"Progress", None))
#if QT_CONFIG(tooltip)
        self.lfeChannelIndex.setToolTip(QCoreApplication.translate("extractAudioDialog", u"0 = No LFE", None))
#endif // QT_CONFIG(tooltip)
        self.audioFormatLabel.setText(QCoreApplication.translate("extractAudioDialog", u"Format", None))
        self.eacBitRate.setSuffix(QCoreApplication.translate("extractAudioDialog", u" kbps", None))
        self.inputFileLabel.setText(QCoreApplication.translate("extractAudioDialog", u"File", None))
        self.label.setText(QCoreApplication.translate("extractAudioDialog", u"Range", None))
        self.ffmpegOutputLabel.setText(QCoreApplication.translate("extractAudioDialog", u"ffmpeg output", None))
        self.filterMappingLabel.setText(QCoreApplication.translate("extractAudioDialog", u"Signal Mapping", None))
        self.includeOriginalAudio.setText(QCoreApplication.translate("extractAudioDialog", u"Add Original Audio?", None))
        self.gainOffsetLabel.setText(QCoreApplication.translate("extractAudioDialog", u"Offset:", None))
        self.gainOffset.setSuffix(QCoreApplication.translate("extractAudioDialog", u" dB", None))
        self.adjustRemuxedAudio.setText(QCoreApplication.translate("extractAudioDialog", u"Adjust Remuxed Audio?", None))
        self.remuxedAudioOffset.setSuffix(QCoreApplication.translate("extractAudioDialog", u" dB", None))
        self.label_2.setText(QCoreApplication.translate("extractAudioDialog", u"Options", None))
        self.calculateGainAdjustment.setText(QCoreApplication.translate("extractAudioDialog", u"...", None))
    # retranslateUi

