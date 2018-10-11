# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'signal.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_addSignalDialog(object):
    def setupUi(self, addSignalDialog):
        addSignalDialog.setObjectName("addSignalDialog")
        addSignalDialog.resize(1392, 748)
        self.verticalLayout = QtWidgets.QVBoxLayout(addSignalDialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.panesLayout = QtWidgets.QGridLayout()
        self.panesLayout.setObjectName("panesLayout")
        self.linkedSignal = QtWidgets.QCheckBox(addSignalDialog)
        self.linkedSignal.setEnabled(True)
        self.linkedSignal.setObjectName("linkedSignal")
        self.panesLayout.addWidget(self.linkedSignal, 3, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.panesLayout.addItem(spacerItem, 5, 0, 1, 1)
        self.previewChart = MplWidget(addSignalDialog)
        self.previewChart.setObjectName("previewChart")
        self.panesLayout.addWidget(self.previewChart, 0, 1, 6, 1)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.signalTypeTabs = QtWidgets.QTabWidget(addSignalDialog)
        self.signalTypeTabs.setObjectName("signalTypeTabs")
        self.wavTab = QtWidgets.QWidget()
        self.wavTab.setObjectName("wavTab")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.wavTab)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.wavGridLayout = QtWidgets.QGridLayout()
        self.wavGridLayout.setObjectName("wavGridLayout")
        self.loadAllChannels = QtWidgets.QCheckBox(self.wavTab)
        self.loadAllChannels.setChecked(True)
        self.loadAllChannels.setObjectName("loadAllChannels")
        self.wavGridLayout.addWidget(self.loadAllChannels, 7, 1, 1, 1)
        self.wavFs = QtWidgets.QLineEdit(self.wavTab)
        self.wavFs.setEnabled(False)
        self.wavFs.setObjectName("wavFs")
        self.wavGridLayout.addWidget(self.wavFs, 2, 1, 1, 1)
        self.wavStartTime = QtWidgets.QTimeEdit(self.wavTab)
        self.wavStartTime.setEnabled(False)
        self.wavStartTime.setTime(QtCore.QTime(0, 0, 0))
        self.wavStartTime.setObjectName("wavStartTime")
        self.wavGridLayout.addWidget(self.wavStartTime, 4, 1, 1, 1)
        self.wavChannelLabel = QtWidgets.QLabel(self.wavTab)
        self.wavChannelLabel.setObjectName("wavChannelLabel")
        self.wavGridLayout.addWidget(self.wavChannelLabel, 3, 0, 1, 1)
        self.wavEndTimeLabel = QtWidgets.QLabel(self.wavTab)
        self.wavEndTimeLabel.setObjectName("wavEndTimeLabel")
        self.wavGridLayout.addWidget(self.wavEndTimeLabel, 5, 0, 1, 1)
        self.wavChannelSelector = QtWidgets.QComboBox(self.wavTab)
        self.wavChannelSelector.setEnabled(False)
        self.wavChannelSelector.setObjectName("wavChannelSelector")
        self.wavGridLayout.addWidget(self.wavChannelSelector, 3, 1, 1, 1)
        self.wavFileLabel = QtWidgets.QLabel(self.wavTab)
        self.wavFileLabel.setObjectName("wavFileLabel")
        self.wavGridLayout.addWidget(self.wavFileLabel, 1, 0, 1, 1)
        self.wavFsLabel = QtWidgets.QLabel(self.wavTab)
        self.wavFsLabel.setObjectName("wavFsLabel")
        self.wavGridLayout.addWidget(self.wavFsLabel, 2, 0, 1, 1)
        self.wavSignalName = QtWidgets.QLineEdit(self.wavTab)
        self.wavSignalName.setEnabled(False)
        self.wavSignalName.setObjectName("wavSignalName")
        self.wavGridLayout.addWidget(self.wavSignalName, 6, 1, 1, 1)
        self.wavEndTime = QtWidgets.QTimeEdit(self.wavTab)
        self.wavEndTime.setEnabled(False)
        self.wavEndTime.setObjectName("wavEndTime")
        self.wavGridLayout.addWidget(self.wavEndTime, 5, 1, 1, 1)
        self.wavFilePicker = QtWidgets.QToolButton(self.wavTab)
        self.wavFilePicker.setObjectName("wavFilePicker")
        self.wavGridLayout.addWidget(self.wavFilePicker, 1, 2, 1, 1)
        self.wavStartTimeLabel = QtWidgets.QLabel(self.wavTab)
        self.wavStartTimeLabel.setObjectName("wavStartTimeLabel")
        self.wavGridLayout.addWidget(self.wavStartTimeLabel, 4, 0, 1, 1)
        self.previewButton = QtWidgets.QPushButton(self.wavTab)
        self.previewButton.setEnabled(False)
        self.previewButton.setObjectName("previewButton")
        self.wavGridLayout.addWidget(self.previewButton, 10, 0, 1, 3)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.wavGridLayout.addItem(spacerItem1, 9, 1, 1, 1)
        self.wavFile = QtWidgets.QLineEdit(self.wavTab)
        self.wavFile.setEnabled(False)
        self.wavFile.setReadOnly(True)
        self.wavFile.setObjectName("wavFile")
        self.wavGridLayout.addWidget(self.wavFile, 1, 1, 1, 1)
        self.wavSignalNameLabel = QtWidgets.QLabel(self.wavTab)
        self.wavSignalNameLabel.setObjectName("wavSignalNameLabel")
        self.wavGridLayout.addWidget(self.wavSignalNameLabel, 6, 0, 1, 1)
        self.decimate = QtWidgets.QCheckBox(self.wavTab)
        self.decimate.setChecked(True)
        self.decimate.setObjectName("decimate")
        self.wavGridLayout.addWidget(self.decimate, 8, 1, 1, 1)
        self.gridLayout_3.addLayout(self.wavGridLayout, 0, 0, 1, 1)
        self.signalTypeTabs.addTab(self.wavTab, "")
        self.frdTab = QtWidgets.QWidget()
        self.frdTab.setObjectName("frdTab")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.frdTab)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.frdGridLayout = QtWidgets.QGridLayout()
        self.frdGridLayout.setObjectName("frdGridLayout")
        self.frdFsLabel = QtWidgets.QLabel(self.frdTab)
        self.frdFsLabel.setObjectName("frdFsLabel")
        self.frdGridLayout.addWidget(self.frdFsLabel, 2, 0, 1, 1)
        self.frdFs = QtWidgets.QSpinBox(self.frdTab)
        self.frdFs.setEnabled(False)
        self.frdFs.setMinimum(100)
        self.frdFs.setMaximum(96000)
        self.frdFs.setSingleStep(100)
        self.frdFs.setProperty("value", 48000)
        self.frdFs.setObjectName("frdFs")
        self.frdGridLayout.addWidget(self.frdFs, 2, 1, 1, 1)
        self.frdAvgFileLabel = QtWidgets.QLabel(self.frdTab)
        self.frdAvgFileLabel.setObjectName("frdAvgFileLabel")
        self.frdGridLayout.addWidget(self.frdAvgFileLabel, 0, 0, 1, 1)
        self.frdAvgFile = QtWidgets.QLineEdit(self.frdTab)
        self.frdAvgFile.setEnabled(False)
        self.frdAvgFile.setObjectName("frdAvgFile")
        self.frdGridLayout.addWidget(self.frdAvgFile, 0, 1, 1, 1)
        self.frdAvgFilePicker = QtWidgets.QToolButton(self.frdTab)
        self.frdAvgFilePicker.setObjectName("frdAvgFilePicker")
        self.frdGridLayout.addWidget(self.frdAvgFilePicker, 0, 2, 1, 1)
        self.frdSignalNameLabel = QtWidgets.QLabel(self.frdTab)
        self.frdSignalNameLabel.setObjectName("frdSignalNameLabel")
        self.frdGridLayout.addWidget(self.frdSignalNameLabel, 3, 0, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.frdGridLayout.addItem(spacerItem2, 4, 1, 1, 1)
        self.frdSignalName = QtWidgets.QLineEdit(self.frdTab)
        self.frdSignalName.setEnabled(False)
        self.frdSignalName.setObjectName("frdSignalName")
        self.frdGridLayout.addWidget(self.frdSignalName, 3, 1, 1, 1)
        self.frdPeakFileLabel = QtWidgets.QLabel(self.frdTab)
        self.frdPeakFileLabel.setObjectName("frdPeakFileLabel")
        self.frdGridLayout.addWidget(self.frdPeakFileLabel, 1, 0, 1, 1)
        self.frdPeakFilePicker = QtWidgets.QToolButton(self.frdTab)
        self.frdPeakFilePicker.setObjectName("frdPeakFilePicker")
        self.frdGridLayout.addWidget(self.frdPeakFilePicker, 1, 2, 1, 1)
        self.frdPeakFile = QtWidgets.QLineEdit(self.frdTab)
        self.frdPeakFile.setEnabled(False)
        self.frdPeakFile.setObjectName("frdPeakFile")
        self.frdGridLayout.addWidget(self.frdPeakFile, 1, 1, 1, 1)
        self.gridLayout_5.addLayout(self.frdGridLayout, 0, 0, 1, 1)
        self.signalTypeTabs.addTab(self.frdTab, "")
        self.gridLayout.addWidget(self.signalTypeTabs, 7, 1, 1, 1)
        self.panesLayout.addLayout(self.gridLayout, 0, 0, 1, 1)
        self.filterSelectLayout = QtWidgets.QGridLayout()
        self.filterSelectLayout.setObjectName("filterSelectLayout")
        self.filterSelectLabel = QtWidgets.QLabel(addSignalDialog)
        self.filterSelectLabel.setObjectName("filterSelectLabel")
        self.filterSelectLayout.addWidget(self.filterSelectLabel, 0, 0, 1, 1)
        self.filterSelect = QtWidgets.QComboBox(addSignalDialog)
        self.filterSelect.setObjectName("filterSelect")
        self.filterSelect.addItem("")
        self.filterSelectLayout.addWidget(self.filterSelect, 0, 1, 1, 1)
        self.filterSelectLayout.setColumnStretch(1, 1)
        self.panesLayout.addLayout(self.filterSelectLayout, 2, 0, 1, 1)
        self.buttonBox = QtWidgets.QDialogButtonBox(addSignalDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.panesLayout.addWidget(self.buttonBox, 4, 0, 1, 1)
        self.panesLayout.setColumnStretch(1, 1)
        self.verticalLayout.addLayout(self.panesLayout)

        self.retranslateUi(addSignalDialog)
        self.signalTypeTabs.setCurrentIndex(0)
        self.buttonBox.rejected.connect(addSignalDialog.reject)
        self.buttonBox.accepted.connect(addSignalDialog.accept)
        self.wavFilePicker.clicked.connect(addSignalDialog.selectFile)
        self.previewButton.clicked.connect(addSignalDialog.prepareSignal)
        self.wavSignalName.textChanged['QString'].connect(addSignalDialog.enablePreview)
        self.signalTypeTabs.currentChanged['int'].connect(addSignalDialog.changeLoader)
        self.frdAvgFilePicker.clicked.connect(addSignalDialog.selectAvgFile)
        self.frdPeakFilePicker.clicked.connect(addSignalDialog.selectPeakFile)
        self.wavSignalName.textChanged['QString'].connect(addSignalDialog.enableOk)
        self.frdSignalName.textChanged['QString'].connect(addSignalDialog.enableOk)
        self.filterSelect.currentIndexChanged['int'].connect(addSignalDialog.masterFilterChanged)
        QtCore.QMetaObject.connectSlotsByName(addSignalDialog)

    def retranslateUi(self, addSignalDialog):
        _translate = QtCore.QCoreApplication.translate
        addSignalDialog.setWindowTitle(_translate("addSignalDialog", "Load Signal"))
        self.linkedSignal.setText(_translate("addSignalDialog", "Linked Filter?"))
        self.loadAllChannels.setText(_translate("addSignalDialog", "Load All Channels?"))
        self.wavStartTime.setDisplayFormat(_translate("addSignalDialog", "HH:mm:ss.zzz"))
        self.wavChannelLabel.setText(_translate("addSignalDialog", "Channel"))
        self.wavEndTimeLabel.setText(_translate("addSignalDialog", "End"))
        self.wavFileLabel.setText(_translate("addSignalDialog", "File"))
        self.wavFsLabel.setText(_translate("addSignalDialog", "Fs"))
        self.wavEndTime.setDisplayFormat(_translate("addSignalDialog", "HH:mm:ss.zzz"))
        self.wavFilePicker.setText(_translate("addSignalDialog", "..."))
        self.wavStartTimeLabel.setText(_translate("addSignalDialog", "Start"))
        self.previewButton.setText(_translate("addSignalDialog", "Preview"))
        self.wavSignalNameLabel.setText(_translate("addSignalDialog", "Name"))
        self.decimate.setText(_translate("addSignalDialog", "Decimate?"))
        self.signalTypeTabs.setTabText(self.signalTypeTabs.indexOf(self.wavTab), _translate("addSignalDialog", "AUDIO"))
        self.frdFsLabel.setText(_translate("addSignalDialog", "Fs"))
        self.frdAvgFileLabel.setText(_translate("addSignalDialog", "Avg"))
        self.frdAvgFilePicker.setText(_translate("addSignalDialog", "..."))
        self.frdSignalNameLabel.setText(_translate("addSignalDialog", "Name"))
        self.frdPeakFileLabel.setText(_translate("addSignalDialog", "Peak"))
        self.frdPeakFilePicker.setText(_translate("addSignalDialog", "..."))
        self.signalTypeTabs.setTabText(self.signalTypeTabs.indexOf(self.frdTab), _translate("addSignalDialog", "TXT"))
        self.filterSelectLabel.setText(_translate("addSignalDialog", "Copy Filter"))
        self.filterSelect.setItemText(0, _translate("addSignalDialog", "None"))

from mpl import MplWidget
