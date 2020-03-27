# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'signal.ui'
##
## Created by: Qt User Interface Compiler version 5.14.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import (QCoreApplication, QMetaObject, QObject, QPoint,
                            QRect, QSize, QUrl, Qt, QTime)
from PySide2.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
    QFontDatabase, QIcon, QLinearGradient, QPalette, QPainter, QPixmap,
    QRadialGradient)
from PySide2.QtWidgets import *

from mpl import MplWidget


class Ui_addSignalDialog(object):
    def setupUi(self, addSignalDialog):
        if addSignalDialog.objectName():
            addSignalDialog.setObjectName(u"addSignalDialog")
        addSignalDialog.resize(1392, 748)
        self.verticalLayout = QVBoxLayout(addSignalDialog)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.panesLayout = QGridLayout()
        self.panesLayout.setObjectName(u"panesLayout")
        self.buttonBox = QDialogButtonBox(addSignalDialog)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)

        self.panesLayout.addWidget(self.buttonBox, 4, 0, 1, 1)

        self.linkedSignal = QCheckBox(addSignalDialog)
        self.linkedSignal.setObjectName(u"linkedSignal")
        self.linkedSignal.setEnabled(True)

        self.panesLayout.addWidget(self.linkedSignal, 3, 0, 1, 1)

        self.filterSelectLayout = QGridLayout()
        self.filterSelectLayout.setObjectName(u"filterSelectLayout")
        self.filterSelectLabel = QLabel(addSignalDialog)
        self.filterSelectLabel.setObjectName(u"filterSelectLabel")

        self.filterSelectLayout.addWidget(self.filterSelectLabel, 0, 0, 1, 1)

        self.filterSelect = QComboBox(addSignalDialog)
        self.filterSelect.addItem("")
        self.filterSelect.setObjectName(u"filterSelect")

        self.filterSelectLayout.addWidget(self.filterSelect, 0, 1, 1, 1)

        self.filterSelectLayout.setColumnStretch(1, 1)

        self.panesLayout.addLayout(self.filterSelectLayout, 2, 0, 1, 1)

        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.signalTypeTabs = QTabWidget(addSignalDialog)
        self.signalTypeTabs.setObjectName(u"signalTypeTabs")
        self.wavTab = QWidget()
        self.wavTab.setObjectName(u"wavTab")
        self.gridLayout_3 = QGridLayout(self.wavTab)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.wavGridLayout = QGridLayout()
        self.wavGridLayout.setObjectName(u"wavGridLayout")
        self.wavStartTime = QTimeEdit(self.wavTab)
        self.wavStartTime.setObjectName(u"wavStartTime")
        self.wavStartTime.setEnabled(False)
        self.wavStartTime.setTime(QTime(0, 0, 0))

        self.wavGridLayout.addWidget(self.wavStartTime, 4, 1, 1, 1)

        self.wavFile = QLineEdit(self.wavTab)
        self.wavFile.setObjectName(u"wavFile")
        self.wavFile.setEnabled(False)
        self.wavFile.setReadOnly(True)

        self.wavGridLayout.addWidget(self.wavFile, 1, 1, 1, 1)

        self.wavFs = QLineEdit(self.wavTab)
        self.wavFs.setObjectName(u"wavFs")
        self.wavFs.setEnabled(False)

        self.wavGridLayout.addWidget(self.wavFs, 2, 1, 1, 1)

        self.wavSignalName = QLineEdit(self.wavTab)
        self.wavSignalName.setObjectName(u"wavSignalName")
        self.wavSignalName.setEnabled(False)

        self.wavGridLayout.addWidget(self.wavSignalName, 6, 1, 1, 1)

        self.decimate = QCheckBox(self.wavTab)
        self.decimate.setObjectName(u"decimate")
        self.decimate.setChecked(True)

        self.wavGridLayout.addWidget(self.decimate, 9, 1, 1, 1)

        self.wavFileLabel = QLabel(self.wavTab)
        self.wavFileLabel.setObjectName(u"wavFileLabel")

        self.wavGridLayout.addWidget(self.wavFileLabel, 1, 0, 1, 1)

        self.loadAllChannels = QCheckBox(self.wavTab)
        self.loadAllChannels.setObjectName(u"loadAllChannels")
        self.loadAllChannels.setChecked(True)

        self.wavGridLayout.addWidget(self.loadAllChannels, 8, 1, 1, 1)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.wavGridLayout.addItem(self.verticalSpacer_2, 10, 1, 1, 1)

        self.applyTimeRangeButton = QToolButton(self.wavTab)
        self.applyTimeRangeButton.setObjectName(u"applyTimeRangeButton")

        self.wavGridLayout.addWidget(self.applyTimeRangeButton, 5, 2, 1, 1)

        self.wavStartTimeLabel = QLabel(self.wavTab)
        self.wavStartTimeLabel.setObjectName(u"wavStartTimeLabel")

        self.wavGridLayout.addWidget(self.wavStartTimeLabel, 4, 0, 1, 1)

        self.wavEndTime = QTimeEdit(self.wavTab)
        self.wavEndTime.setObjectName(u"wavEndTime")
        self.wavEndTime.setEnabled(False)

        self.wavGridLayout.addWidget(self.wavEndTime, 5, 1, 1, 1)

        self.wavSignalNameLabel = QLabel(self.wavTab)
        self.wavSignalNameLabel.setObjectName(u"wavSignalNameLabel")

        self.wavGridLayout.addWidget(self.wavSignalNameLabel, 6, 0, 1, 1)

        self.wavFilePicker = QToolButton(self.wavTab)
        self.wavFilePicker.setObjectName(u"wavFilePicker")

        self.wavGridLayout.addWidget(self.wavFilePicker, 1, 2, 1, 1)

        self.wavChannelSelector = QComboBox(self.wavTab)
        self.wavChannelSelector.setObjectName(u"wavChannelSelector")
        self.wavChannelSelector.setEnabled(False)

        self.wavGridLayout.addWidget(self.wavChannelSelector, 3, 1, 1, 1)

        self.wavEndTimeLabel = QLabel(self.wavTab)
        self.wavEndTimeLabel.setObjectName(u"wavEndTimeLabel")

        self.wavGridLayout.addWidget(self.wavEndTimeLabel, 5, 0, 1, 1)

        self.wavFsLabel = QLabel(self.wavTab)
        self.wavFsLabel.setObjectName(u"wavFsLabel")

        self.wavGridLayout.addWidget(self.wavFsLabel, 2, 0, 1, 1)

        self.wavChannelLabel = QLabel(self.wavTab)
        self.wavChannelLabel.setObjectName(u"wavChannelLabel")

        self.wavGridLayout.addWidget(self.wavChannelLabel, 3, 0, 1, 1)

        self.gainOffset = QDoubleSpinBox(self.wavTab)
        self.gainOffset.setObjectName(u"gainOffset")
        self.gainOffset.setEnabled(False)
        self.gainOffset.setMinimum(-100.000000000000000)
        self.gainOffset.setMaximum(100.000000000000000)
        self.gainOffset.setSingleStep(0.010000000000000)

        self.wavGridLayout.addWidget(self.gainOffset, 7, 1, 1, 1)

        self.gainOffsetLabel = QLabel(self.wavTab)
        self.gainOffsetLabel.setObjectName(u"gainOffsetLabel")

        self.wavGridLayout.addWidget(self.gainOffsetLabel, 7, 0, 1, 1)


        self.gridLayout_3.addLayout(self.wavGridLayout, 0, 0, 1, 1)

        self.signalTypeTabs.addTab(self.wavTab, "")
        self.frdTab = QWidget()
        self.frdTab.setObjectName(u"frdTab")
        self.gridLayout_5 = QGridLayout(self.frdTab)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.frdGridLayout = QGridLayout()
        self.frdGridLayout.setObjectName(u"frdGridLayout")
        self.frdFsLabel = QLabel(self.frdTab)
        self.frdFsLabel.setObjectName(u"frdFsLabel")

        self.frdGridLayout.addWidget(self.frdFsLabel, 2, 0, 1, 1)

        self.frdFs = QSpinBox(self.frdTab)
        self.frdFs.setObjectName(u"frdFs")
        self.frdFs.setEnabled(False)
        self.frdFs.setMinimum(100)
        self.frdFs.setMaximum(96000)
        self.frdFs.setSingleStep(100)
        self.frdFs.setValue(48000)

        self.frdGridLayout.addWidget(self.frdFs, 2, 1, 1, 1)

        self.frdAvgFileLabel = QLabel(self.frdTab)
        self.frdAvgFileLabel.setObjectName(u"frdAvgFileLabel")

        self.frdGridLayout.addWidget(self.frdAvgFileLabel, 0, 0, 1, 1)

        self.frdAvgFile = QLineEdit(self.frdTab)
        self.frdAvgFile.setObjectName(u"frdAvgFile")
        self.frdAvgFile.setEnabled(False)

        self.frdGridLayout.addWidget(self.frdAvgFile, 0, 1, 1, 1)

        self.frdAvgFilePicker = QToolButton(self.frdTab)
        self.frdAvgFilePicker.setObjectName(u"frdAvgFilePicker")

        self.frdGridLayout.addWidget(self.frdAvgFilePicker, 0, 2, 1, 1)

        self.frdSignalNameLabel = QLabel(self.frdTab)
        self.frdSignalNameLabel.setObjectName(u"frdSignalNameLabel")

        self.frdGridLayout.addWidget(self.frdSignalNameLabel, 3, 0, 1, 1)

        self.frdSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.frdGridLayout.addItem(self.frdSpacer, 4, 1, 1, 1)

        self.frdSignalName = QLineEdit(self.frdTab)
        self.frdSignalName.setObjectName(u"frdSignalName")
        self.frdSignalName.setEnabled(False)

        self.frdGridLayout.addWidget(self.frdSignalName, 3, 1, 1, 1)

        self.frdPeakFileLabel = QLabel(self.frdTab)
        self.frdPeakFileLabel.setObjectName(u"frdPeakFileLabel")

        self.frdGridLayout.addWidget(self.frdPeakFileLabel, 1, 0, 1, 1)

        self.frdPeakFilePicker = QToolButton(self.frdTab)
        self.frdPeakFilePicker.setObjectName(u"frdPeakFilePicker")

        self.frdGridLayout.addWidget(self.frdPeakFilePicker, 1, 2, 1, 1)

        self.frdPeakFile = QLineEdit(self.frdTab)
        self.frdPeakFile.setObjectName(u"frdPeakFile")
        self.frdPeakFile.setEnabled(False)

        self.frdGridLayout.addWidget(self.frdPeakFile, 1, 1, 1, 1)


        self.gridLayout_5.addLayout(self.frdGridLayout, 0, 0, 1, 1)

        self.signalTypeTabs.addTab(self.frdTab, "")
        self.pulseTab = QWidget()
        self.pulseTab.setObjectName(u"pulseTab")
        self.verticalLayout_2 = QVBoxLayout(self.pulseTab)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.gridLayout_2 = QGridLayout()
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.pulseFsLabel = QLabel(self.pulseTab)
        self.pulseFsLabel.setObjectName(u"pulseFsLabel")

        self.gridLayout_2.addWidget(self.pulseFsLabel, 2, 0, 1, 1)

        self.pulsePrefixLabel = QLabel(self.pulseTab)
        self.pulsePrefixLabel.setObjectName(u"pulsePrefixLabel")

        self.gridLayout_2.addWidget(self.pulsePrefixLabel, 1, 0, 1, 1)

        self.presetsHeaderLabel = QLabel(self.pulseTab)
        self.presetsHeaderLabel.setObjectName(u"presetsHeaderLabel")
        font = QFont()
        font.setBold(True)
        font.setWeight(75)
        self.presetsHeaderLabel.setFont(font)
        self.presetsHeaderLabel.setFrameShape(QFrame.Box)
        self.presetsHeaderLabel.setFrameShadow(QFrame.Sunken)
        self.presetsHeaderLabel.setAlignment(Qt.AlignCenter)

        self.gridLayout_2.addWidget(self.presetsHeaderLabel, 0, 0, 1, 3)

        self.pulseChannels = QListWidget(self.pulseTab)
        self.pulseChannels.setObjectName(u"pulseChannels")

        self.gridLayout_2.addWidget(self.pulseChannels, 3, 0, 1, 3)

        self.pulseFs = QComboBox(self.pulseTab)
        self.pulseFs.addItem("")
        self.pulseFs.addItem("")
        self.pulseFs.setObjectName(u"pulseFs")

        self.gridLayout_2.addWidget(self.pulseFs, 2, 1, 1, 2)

        self.pulsePrefix = QLineEdit(self.pulseTab)
        self.pulsePrefix.setObjectName(u"pulsePrefix")
        self.pulsePrefix.setEnabled(True)
        self.pulsePrefix.setReadOnly(False)

        self.gridLayout_2.addWidget(self.pulsePrefix, 1, 1, 1, 2)


        self.verticalLayout_2.addLayout(self.gridLayout_2)

        self.signalTypeTabs.addTab(self.pulseTab, "")

        self.gridLayout.addWidget(self.signalTypeTabs, 7, 1, 1, 1)


        self.panesLayout.addLayout(self.gridLayout, 0, 0, 1, 1)

        self.previewChart = MplWidget(addSignalDialog)
        self.previewChart.setObjectName(u"previewChart")

        self.panesLayout.addWidget(self.previewChart, 0, 1, 6, 1)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.panesLayout.addItem(self.verticalSpacer, 5, 0, 1, 1)

        self.panesLayout.setColumnStretch(1, 1)

        self.verticalLayout.addLayout(self.panesLayout)

        QWidget.setTabOrder(self.signalTypeTabs, self.wavFilePicker)
        QWidget.setTabOrder(self.wavFilePicker, self.wavChannelSelector)
        QWidget.setTabOrder(self.wavChannelSelector, self.wavStartTime)
        QWidget.setTabOrder(self.wavStartTime, self.wavEndTime)
        QWidget.setTabOrder(self.wavEndTime, self.wavSignalName)
        QWidget.setTabOrder(self.wavSignalName, self.loadAllChannels)
        QWidget.setTabOrder(self.loadAllChannels, self.decimate)
        QWidget.setTabOrder(self.decimate, self.filterSelect)
        QWidget.setTabOrder(self.filterSelect, self.linkedSignal)
        QWidget.setTabOrder(self.linkedSignal, self.wavFile)
        QWidget.setTabOrder(self.wavFile, self.wavFs)
        QWidget.setTabOrder(self.wavFs, self.frdFs)
        QWidget.setTabOrder(self.frdFs, self.frdAvgFile)
        QWidget.setTabOrder(self.frdAvgFile, self.frdAvgFilePicker)
        QWidget.setTabOrder(self.frdAvgFilePicker, self.frdSignalName)
        QWidget.setTabOrder(self.frdSignalName, self.frdPeakFilePicker)
        QWidget.setTabOrder(self.frdPeakFilePicker, self.frdPeakFile)
        QWidget.setTabOrder(self.frdPeakFile, self.previewChart)

        self.retranslateUi(addSignalDialog)
        self.buttonBox.rejected.connect(addSignalDialog.reject)
        self.buttonBox.accepted.connect(addSignalDialog.accept)
        self.wavFilePicker.clicked.connect(addSignalDialog.selectFile)
        self.signalTypeTabs.currentChanged.connect(addSignalDialog.changeLoader)
        self.frdAvgFilePicker.clicked.connect(addSignalDialog.selectAvgFile)
        self.frdPeakFilePicker.clicked.connect(addSignalDialog.selectPeakFile)
        self.wavSignalName.textChanged.connect(addSignalDialog.enableOk)
        self.frdSignalName.textChanged.connect(addSignalDialog.enableOk)
        self.filterSelect.currentIndexChanged.connect(addSignalDialog.masterFilterChanged)
        self.wavChannelSelector.currentTextChanged.connect(addSignalDialog.previewChannel)
        self.applyTimeRangeButton.clicked.connect(addSignalDialog.limitTimeRange)
        self.wavStartTime.timeChanged.connect(addSignalDialog.enableLimitTimeRangeButton)
        self.wavEndTime.timeChanged.connect(addSignalDialog.enableLimitTimeRangeButton)
        self.decimate.stateChanged.connect(addSignalDialog.toggleDecimate)
        self.pulsePrefix.textChanged.connect(addSignalDialog.enableOk)

        self.signalTypeTabs.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(addSignalDialog)
    # setupUi

    def retranslateUi(self, addSignalDialog):
        addSignalDialog.setWindowTitle(QCoreApplication.translate("addSignalDialog", u"Load Signal", None))
        self.linkedSignal.setText(QCoreApplication.translate("addSignalDialog", u"Linked Filter?", None))
        self.filterSelectLabel.setText(QCoreApplication.translate("addSignalDialog", u"Copy Filter", None))
        self.filterSelect.setItemText(0, QCoreApplication.translate("addSignalDialog", u"None", None))

        self.wavStartTime.setDisplayFormat(QCoreApplication.translate("addSignalDialog", u"HH:mm:ss.zzz", None))
        self.decimate.setText(QCoreApplication.translate("addSignalDialog", u"Resample?", None))
        self.wavFileLabel.setText(QCoreApplication.translate("addSignalDialog", u"File", None))
        self.loadAllChannels.setText(QCoreApplication.translate("addSignalDialog", u"Load All Channels?", None))
        self.applyTimeRangeButton.setText(QCoreApplication.translate("addSignalDialog", u"...", None))
        self.wavStartTimeLabel.setText(QCoreApplication.translate("addSignalDialog", u"Start", None))
        self.wavEndTime.setDisplayFormat(QCoreApplication.translate("addSignalDialog", u"HH:mm:ss.zzz", None))
        self.wavSignalNameLabel.setText(QCoreApplication.translate("addSignalDialog", u"Name", None))
        self.wavFilePicker.setText(QCoreApplication.translate("addSignalDialog", u"...", None))
        self.wavEndTimeLabel.setText(QCoreApplication.translate("addSignalDialog", u"End", None))
        self.wavFsLabel.setText(QCoreApplication.translate("addSignalDialog", u"Fs", None))
        self.wavChannelLabel.setText(QCoreApplication.translate("addSignalDialog", u"Channel", None))
        self.gainOffset.setSuffix(QCoreApplication.translate("addSignalDialog", u" dB", None))
        self.gainOffsetLabel.setText(QCoreApplication.translate("addSignalDialog", u"Offset", None))
        self.signalTypeTabs.setTabText(self.signalTypeTabs.indexOf(self.wavTab), QCoreApplication.translate("addSignalDialog", u"AUDIO", None))
        self.frdFsLabel.setText(QCoreApplication.translate("addSignalDialog", u"Fs", None))
        self.frdAvgFileLabel.setText(QCoreApplication.translate("addSignalDialog", u"Avg", None))
        self.frdAvgFilePicker.setText(QCoreApplication.translate("addSignalDialog", u"...", None))
        self.frdSignalNameLabel.setText(QCoreApplication.translate("addSignalDialog", u"Name", None))
        self.frdPeakFileLabel.setText(QCoreApplication.translate("addSignalDialog", u"Peak", None))
        self.frdPeakFilePicker.setText(QCoreApplication.translate("addSignalDialog", u"...", None))
        self.signalTypeTabs.setTabText(self.signalTypeTabs.indexOf(self.frdTab), QCoreApplication.translate("addSignalDialog", u"TXT", None))
        self.pulseFsLabel.setText(QCoreApplication.translate("addSignalDialog", u"Fs", None))
        self.pulsePrefixLabel.setText(QCoreApplication.translate("addSignalDialog", u"Prefix", None))
        self.presetsHeaderLabel.setText(QCoreApplication.translate("addSignalDialog", u"Presets", None))
        self.pulseFs.setItemText(0, QCoreApplication.translate("addSignalDialog", u"48 kHz", None))
        self.pulseFs.setItemText(1, QCoreApplication.translate("addSignalDialog", u"96 kHz", None))

        self.signalTypeTabs.setTabText(self.signalTypeTabs.indexOf(self.pulseTab), QCoreApplication.translate("addSignalDialog", u"PULSE", None))
    # retranslateUi

