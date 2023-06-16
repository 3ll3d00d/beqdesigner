# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'merge.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_mergeDspDialog(object):
    def setupUi(self, mergeDspDialog):
        mergeDspDialog.setObjectName("mergeDspDialog")
        mergeDspDialog.resize(639, 856)
        self.gridLayout = QtWidgets.QGridLayout(mergeDspDialog)
        self.gridLayout.setObjectName("gridLayout")
        self.innerLayout = QtWidgets.QGridLayout()
        self.innerLayout.setObjectName("innerLayout")
        self.userSourceDirLabel = QtWidgets.QLabel(mergeDspDialog)
        self.userSourceDirLabel.setObjectName("userSourceDirLabel")
        self.innerLayout.addWidget(self.userSourceDirLabel, 0, 0, 1, 1)
        self.userSourceDirLayout = QtWidgets.QHBoxLayout()
        self.userSourceDirLayout.setObjectName("userSourceDirLayout")
        self.userSourceDir = QtWidgets.QLineEdit(mergeDspDialog)
        self.userSourceDir.setReadOnly(True)
        self.userSourceDir.setObjectName("userSourceDir")
        self.userSourceDirLayout.addWidget(self.userSourceDir)
        self.clearUserSourceDir = QtWidgets.QToolButton(mergeDspDialog)
        self.clearUserSourceDir.setObjectName("clearUserSourceDir")
        self.userSourceDirLayout.addWidget(self.clearUserSourceDir)
        self.innerLayout.addLayout(self.userSourceDirLayout, 0, 1, 1, 4)
        self.userSourceDirPicker = QtWidgets.QToolButton(mergeDspDialog)
        self.userSourceDirPicker.setObjectName("userSourceDirPicker")
        self.innerLayout.addWidget(self.userSourceDirPicker, 0, 5, 1, 1)
        self.configFileLabel = QtWidgets.QLabel(mergeDspDialog)
        self.configFileLabel.setObjectName("configFileLabel")
        self.innerLayout.addWidget(self.configFileLabel, 1, 0, 1, 1)
        self.configFile = QtWidgets.QLineEdit(mergeDspDialog)
        self.configFile.setEnabled(False)
        self.configFile.setReadOnly(True)
        self.configFile.setObjectName("configFile")
        self.innerLayout.addWidget(self.configFile, 1, 1, 1, 4)
        self.configFilePicker = QtWidgets.QToolButton(mergeDspDialog)
        self.configFilePicker.setObjectName("configFilePicker")
        self.innerLayout.addWidget(self.configFilePicker, 1, 5, 1, 1)
        self.outputDirectoryLabel = QtWidgets.QLabel(mergeDspDialog)
        self.outputDirectoryLabel.setObjectName("outputDirectoryLabel")
        self.innerLayout.addWidget(self.outputDirectoryLabel, 2, 0, 1, 1)
        self.outputDirectory = QtWidgets.QLineEdit(mergeDspDialog)
        self.outputDirectory.setEnabled(False)
        self.outputDirectory.setReadOnly(True)
        self.outputDirectory.setObjectName("outputDirectory")
        self.innerLayout.addWidget(self.outputDirectory, 2, 1, 1, 4)
        self.outputDirectoryPicker = QtWidgets.QToolButton(mergeDspDialog)
        self.outputDirectoryPicker.setObjectName("outputDirectoryPicker")
        self.innerLayout.addWidget(self.outputDirectoryPicker, 2, 5, 1, 1)
        self.dspTypeLabel = QtWidgets.QLabel(mergeDspDialog)
        self.dspTypeLabel.setObjectName("dspTypeLabel")
        self.innerLayout.addWidget(self.dspTypeLabel, 3, 0, 1, 1)
        self.dspType = QtWidgets.QComboBox(mergeDspDialog)
        self.dspType.setObjectName("dspType")
        self.innerLayout.addWidget(self.dspType, 3, 1, 1, 4)
        self.outputModeLabel = QtWidgets.QLabel(mergeDspDialog)
        self.outputModeLabel.setObjectName("outputModeLabel")
        self.innerLayout.addWidget(self.outputModeLabel, 4, 0, 1, 1)
        self.outputMode = QtWidgets.QComboBox(mergeDspDialog)
        self.outputMode.setObjectName("outputMode")
        self.innerLayout.addWidget(self.outputMode, 4, 1, 1, 3)
        self.outputChannelsLabel = QtWidgets.QLabel(mergeDspDialog)
        self.outputChannelsLabel.setObjectName("outputChannelsLabel")
        self.innerLayout.addWidget(self.outputChannelsLabel, 5, 0, 1, 1)
        self.outputChannels = QtWidgets.QListWidget(mergeDspDialog)
        self.outputChannels.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.outputChannels.setObjectName("outputChannels")
        self.innerLayout.addWidget(self.outputChannels, 5, 1, 1, 3)
        self.totalFiles = QtWidgets.QSpinBox(mergeDspDialog)
        self.totalFiles.setReadOnly(True)
        self.totalFiles.setMaximum(10000)
        self.totalFiles.setObjectName("totalFiles")
        self.innerLayout.addWidget(self.totalFiles, 6, 3, 1, 1)
        self.filesProcessedLabel = QtWidgets.QLabel(mergeDspDialog)
        self.filesProcessedLabel.setObjectName("filesProcessedLabel")
        self.innerLayout.addWidget(self.filesProcessedLabel, 6, 0, 1, 1)
        self.filesProcessed = QtWidgets.QSpinBox(mergeDspDialog)
        self.filesProcessed.setReadOnly(True)
        self.filesProcessed.setMaximum(100000)
        self.filesProcessed.setObjectName("filesProcessed")
        self.innerLayout.addWidget(self.filesProcessed, 6, 1, 1, 1)
        self.ofLabel = QtWidgets.QLabel(mergeDspDialog)
        self.ofLabel.setObjectName("ofLabel")
        self.innerLayout.addWidget(self.ofLabel, 6, 2, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.innerLayout.addItem(spacerItem, 6, 4, 1, 1)
        self.processFiles = QtWidgets.QToolButton(mergeDspDialog)
        self.processFiles.setObjectName("processFiles")
        self.innerLayout.addWidget(self.processFiles, 6, 5, 1, 1)
        self.optimisedLabel = QtWidgets.QLabel(mergeDspDialog)
        self.optimisedLabel.setObjectName("optimisedLabel")
        self.innerLayout.addWidget(self.optimisedLabel, 7, 0, 1, 1)
        self.optimised = QtWidgets.QListWidget(mergeDspDialog)
        self.optimised.setObjectName("optimised")
        self.innerLayout.addWidget(self.optimised, 7, 1, 1, 4)
        self.copyOptimisedButton = QtWidgets.QToolButton(mergeDspDialog)
        self.copyOptimisedButton.setObjectName("copyOptimisedButton")
        self.innerLayout.addWidget(self.copyOptimisedButton, 7, 5, 1, 1)
        self.errorsLabel = QtWidgets.QLabel(mergeDspDialog)
        self.errorsLabel.setObjectName("errorsLabel")
        self.innerLayout.addWidget(self.errorsLabel, 8, 0, 1, 1)
        self.errors = QtWidgets.QListWidget(mergeDspDialog)
        self.errors.setEnabled(False)
        self.errors.setObjectName("errors")
        self.innerLayout.addWidget(self.errors, 8, 1, 1, 4)
        self.copyErrorsButton = QtWidgets.QToolButton(mergeDspDialog)
        self.copyErrorsButton.setObjectName("copyErrorsButton")
        self.innerLayout.addWidget(self.copyErrorsButton, 8, 5, 1, 1)
        self.gitStatusLabel = QtWidgets.QLabel(mergeDspDialog)
        self.gitStatusLabel.setObjectName("gitStatusLabel")
        self.innerLayout.addWidget(self.gitStatusLabel, 9, 0, 1, 1)
        self.gitStatus = QtWidgets.QListWidget(mergeDspDialog)
        self.gitStatus.setObjectName("gitStatus")
        self.innerLayout.addWidget(self.gitStatus, 9, 1, 1, 4)
        self.gridLayout.addLayout(self.innerLayout, 0, 0, 1, 1)
        self.buttonBox = QtWidgets.QDialogButtonBox(mergeDspDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Close)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 1, 0, 1, 1)

        self.retranslateUi(mergeDspDialog)
        self.buttonBox.accepted.connect(mergeDspDialog.accept) # type: ignore
        self.buttonBox.rejected.connect(mergeDspDialog.reject) # type: ignore
        self.configFilePicker.clicked.connect(mergeDspDialog.pick_config_file) # type: ignore
        self.outputDirectoryPicker.clicked.connect(mergeDspDialog.pick_output_dir) # type: ignore
        self.processFiles.clicked.connect(mergeDspDialog.process_files) # type: ignore
        self.userSourceDirPicker.clicked.connect(mergeDspDialog.pick_user_source_dir) # type: ignore
        self.clearUserSourceDir.clicked.connect(mergeDspDialog.clear_user_source_dir) # type: ignore
        self.copyOptimisedButton.clicked.connect(mergeDspDialog.copy_optimised) # type: ignore
        self.copyErrorsButton.clicked.connect(mergeDspDialog.copy_errors) # type: ignore
        self.dspType.currentTextChanged['QString'].connect(mergeDspDialog.dsp_type_changed) # type: ignore
        self.outputMode.currentTextChanged['QString'].connect(mergeDspDialog.output_mode_changed) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(mergeDspDialog)
        mergeDspDialog.setTabOrder(self.userSourceDir, self.clearUserSourceDir)
        mergeDspDialog.setTabOrder(self.clearUserSourceDir, self.userSourceDirPicker)
        mergeDspDialog.setTabOrder(self.userSourceDirPicker, self.configFile)
        mergeDspDialog.setTabOrder(self.configFile, self.configFilePicker)
        mergeDspDialog.setTabOrder(self.configFilePicker, self.outputDirectory)
        mergeDspDialog.setTabOrder(self.outputDirectory, self.outputDirectoryPicker)
        mergeDspDialog.setTabOrder(self.outputDirectoryPicker, self.dspType)
        mergeDspDialog.setTabOrder(self.dspType, self.filesProcessed)
        mergeDspDialog.setTabOrder(self.filesProcessed, self.totalFiles)
        mergeDspDialog.setTabOrder(self.totalFiles, self.processFiles)
        mergeDspDialog.setTabOrder(self.processFiles, self.errors)

    def retranslateUi(self, mergeDspDialog):
        _translate = QtCore.QCoreApplication.translate
        mergeDspDialog.setWindowTitle(_translate("mergeDspDialog", "Merge DSP Config"))
        self.userSourceDirLabel.setText(_translate("mergeDspDialog", "User Source Directory"))
        self.clearUserSourceDir.setText(_translate("mergeDspDialog", "..."))
        self.userSourceDirPicker.setText(_translate("mergeDspDialog", "..."))
        self.configFileLabel.setText(_translate("mergeDspDialog", "Config File"))
        self.configFilePicker.setText(_translate("mergeDspDialog", "..."))
        self.outputDirectoryLabel.setText(_translate("mergeDspDialog", "Output Directory"))
        self.outputDirectoryPicker.setText(_translate("mergeDspDialog", "..."))
        self.dspTypeLabel.setText(_translate("mergeDspDialog", "DSP Type"))
        self.outputModeLabel.setText(_translate("mergeDspDialog", "Output Mode"))
        self.outputChannelsLabel.setText(_translate("mergeDspDialog", "Output Channels"))
        self.filesProcessedLabel.setText(_translate("mergeDspDialog", "Files Processed"))
        self.ofLabel.setText(_translate("mergeDspDialog", "of"))
        self.processFiles.setText(_translate("mergeDspDialog", "..."))
        self.optimisedLabel.setText(_translate("mergeDspDialog", "Optimised"))
        self.copyOptimisedButton.setText(_translate("mergeDspDialog", "..."))
        self.errorsLabel.setText(_translate("mergeDspDialog", "Errors"))
        self.copyErrorsButton.setText(_translate("mergeDspDialog", "..."))
        self.gitStatusLabel.setText(_translate("mergeDspDialog", "Git Status"))
