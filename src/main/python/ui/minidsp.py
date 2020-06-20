# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'minidsp.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_mergeMinidspDialog(object):
    def setupUi(self, mergeMinidspDialog):
        mergeMinidspDialog.setObjectName("mergeMinidspDialog")
        mergeMinidspDialog.resize(639, 655)
        self.gridLayout = QtWidgets.QGridLayout(mergeMinidspDialog)
        self.gridLayout.setObjectName("gridLayout")
        self.buttonBox = QtWidgets.QDialogButtonBox(mergeMinidspDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Close|QtWidgets.QDialogButtonBox.Reset)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 1, 0, 1, 1)
        self.innerLayout = QtWidgets.QGridLayout()
        self.innerLayout.setObjectName("innerLayout")
        self.outputDirectoryLabel = QtWidgets.QLabel(mergeMinidspDialog)
        self.outputDirectoryLabel.setObjectName("outputDirectoryLabel")
        self.innerLayout.addWidget(self.outputDirectoryLabel, 5, 0, 1, 1)
        self.filesProcessedLabel = QtWidgets.QLabel(mergeMinidspDialog)
        self.filesProcessedLabel.setObjectName("filesProcessedLabel")
        self.innerLayout.addWidget(self.filesProcessedLabel, 7, 0, 1, 1)
        self.totalFiles = QtWidgets.QSpinBox(mergeMinidspDialog)
        self.totalFiles.setReadOnly(True)
        self.totalFiles.setMaximum(10000)
        self.totalFiles.setObjectName("totalFiles")
        self.innerLayout.addWidget(self.totalFiles, 7, 3, 1, 1)
        self.configFileLabel = QtWidgets.QLabel(mergeMinidspDialog)
        self.configFileLabel.setObjectName("configFileLabel")
        self.innerLayout.addWidget(self.configFileLabel, 4, 0, 1, 1)
        self.userSourceDirLayout = QtWidgets.QHBoxLayout()
        self.userSourceDirLayout.setObjectName("userSourceDirLayout")
        self.userSourceDir = QtWidgets.QLineEdit(mergeMinidspDialog)
        self.userSourceDir.setReadOnly(True)
        self.userSourceDir.setObjectName("userSourceDir")
        self.userSourceDirLayout.addWidget(self.userSourceDir)
        self.clearUserSourceDir = QtWidgets.QToolButton(mergeMinidspDialog)
        self.clearUserSourceDir.setObjectName("clearUserSourceDir")
        self.userSourceDirLayout.addWidget(self.clearUserSourceDir)
        self.innerLayout.addLayout(self.userSourceDirLayout, 3, 1, 1, 4)
        self.lastCommitDate = QtWidgets.QDateTimeEdit(mergeMinidspDialog)
        self.lastCommitDate.setReadOnly(True)
        self.lastCommitDate.setObjectName("lastCommitDate")
        self.innerLayout.addWidget(self.lastCommitDate, 1, 1, 1, 3)
        self.beqRepos = QtWidgets.QComboBox(mergeMinidspDialog)
        self.beqRepos.setObjectName("beqRepos")
        self.innerLayout.addWidget(self.beqRepos, 0, 1, 1, 4)
        self.minidspType = QtWidgets.QComboBox(mergeMinidspDialog)
        self.minidspType.setObjectName("minidspType")
        self.innerLayout.addWidget(self.minidspType, 6, 1, 1, 4)
        self.errors = QtWidgets.QListWidget(mergeMinidspDialog)
        self.errors.setEnabled(False)
        self.errors.setObjectName("errors")
        self.innerLayout.addWidget(self.errors, 9, 1, 1, 4)
        self.errorsLabel = QtWidgets.QLabel(mergeMinidspDialog)
        self.errorsLabel.setObjectName("errorsLabel")
        self.innerLayout.addWidget(self.errorsLabel, 9, 0, 1, 1)
        self.lastUpdateLabel = QtWidgets.QLabel(mergeMinidspDialog)
        self.lastUpdateLabel.setObjectName("lastUpdateLabel")
        self.innerLayout.addWidget(self.lastUpdateLabel, 1, 0, 1, 1)
        self.outputDirectory = QtWidgets.QLineEdit(mergeMinidspDialog)
        self.outputDirectory.setEnabled(False)
        self.outputDirectory.setReadOnly(True)
        self.outputDirectory.setObjectName("outputDirectory")
        self.innerLayout.addWidget(self.outputDirectory, 5, 1, 1, 4)
        self.processFiles = QtWidgets.QToolButton(mergeMinidspDialog)
        self.processFiles.setObjectName("processFiles")
        self.innerLayout.addWidget(self.processFiles, 7, 5, 1, 1)
        self.minidspTypeLabel = QtWidgets.QLabel(mergeMinidspDialog)
        self.minidspTypeLabel.setObjectName("minidspTypeLabel")
        self.innerLayout.addWidget(self.minidspTypeLabel, 6, 0, 1, 1)
        self.filesProcessed = QtWidgets.QSpinBox(mergeMinidspDialog)
        self.filesProcessed.setReadOnly(True)
        self.filesProcessed.setMaximum(100000)
        self.filesProcessed.setObjectName("filesProcessed")
        self.innerLayout.addWidget(self.filesProcessed, 7, 1, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.innerLayout.addItem(spacerItem, 7, 4, 1, 1)
        self.beqReposLabel = QtWidgets.QLabel(mergeMinidspDialog)
        self.beqReposLabel.setObjectName("beqReposLabel")
        self.innerLayout.addWidget(self.beqReposLabel, 0, 0, 1, 1)
        self.lastCommitMessage = QtWidgets.QPlainTextEdit(mergeMinidspDialog)
        self.lastCommitMessage.setReadOnly(True)
        self.lastCommitMessage.setObjectName("lastCommitMessage")
        self.innerLayout.addWidget(self.lastCommitMessage, 2, 1, 1, 4)
        self.ofLabel = QtWidgets.QLabel(mergeMinidspDialog)
        self.ofLabel.setObjectName("ofLabel")
        self.innerLayout.addWidget(self.ofLabel, 7, 2, 1, 1)
        self.configFile = QtWidgets.QLineEdit(mergeMinidspDialog)
        self.configFile.setEnabled(False)
        self.configFile.setReadOnly(True)
        self.configFile.setObjectName("configFile")
        self.innerLayout.addWidget(self.configFile, 4, 1, 1, 4)
        self.outputDirectoryPicker = QtWidgets.QToolButton(mergeMinidspDialog)
        self.outputDirectoryPicker.setObjectName("outputDirectoryPicker")
        self.innerLayout.addWidget(self.outputDirectoryPicker, 5, 5, 1, 1)
        self.refreshGitRepo = QtWidgets.QToolButton(mergeMinidspDialog)
        self.refreshGitRepo.setObjectName("refreshGitRepo")
        self.innerLayout.addWidget(self.refreshGitRepo, 1, 5, 1, 1)
        self.infoLabel = QtWidgets.QLabel(mergeMinidspDialog)
        self.infoLabel.setText("")
        self.infoLabel.setObjectName("infoLabel")
        self.innerLayout.addWidget(self.infoLabel, 1, 4, 1, 1)
        self.configFilePicker = QtWidgets.QToolButton(mergeMinidspDialog)
        self.configFilePicker.setObjectName("configFilePicker")
        self.innerLayout.addWidget(self.configFilePicker, 4, 5, 1, 1)
        self.userSourceDirLabel = QtWidgets.QLabel(mergeMinidspDialog)
        self.userSourceDirLabel.setObjectName("userSourceDirLabel")
        self.innerLayout.addWidget(self.userSourceDirLabel, 3, 0, 1, 1)
        self.userSourceDirPicker = QtWidgets.QToolButton(mergeMinidspDialog)
        self.userSourceDirPicker.setObjectName("userSourceDirPicker")
        self.innerLayout.addWidget(self.userSourceDirPicker, 3, 5, 1, 1)
        self.optimisedLabel = QtWidgets.QLabel(mergeMinidspDialog)
        self.optimisedLabel.setObjectName("optimisedLabel")
        self.innerLayout.addWidget(self.optimisedLabel, 8, 0, 1, 1)
        self.optimised = QtWidgets.QListWidget(mergeMinidspDialog)
        self.optimised.setObjectName("optimised")
        self.innerLayout.addWidget(self.optimised, 8, 1, 1, 4)
        self.copyOptimisedButton = QtWidgets.QToolButton(mergeMinidspDialog)
        self.copyOptimisedButton.setObjectName("copyOptimisedButton")
        self.innerLayout.addWidget(self.copyOptimisedButton, 8, 5, 1, 1)
        self.copyErrorsButton = QtWidgets.QToolButton(mergeMinidspDialog)
        self.copyErrorsButton.setObjectName("copyErrorsButton")
        self.innerLayout.addWidget(self.copyErrorsButton, 9, 5, 1, 1)
        self.gridLayout.addLayout(self.innerLayout, 0, 0, 1, 1)

        self.retranslateUi(mergeMinidspDialog)
        self.buttonBox.accepted.connect(mergeMinidspDialog.accept)
        self.buttonBox.rejected.connect(mergeMinidspDialog.reject)
        self.configFilePicker.clicked.connect(mergeMinidspDialog.pick_config_file)
        self.outputDirectoryPicker.clicked.connect(mergeMinidspDialog.pick_output_dir)
        self.processFiles.clicked.connect(mergeMinidspDialog.process_files)
        self.refreshGitRepo.clicked.connect(mergeMinidspDialog.refresh_repo)
        self.userSourceDirPicker.clicked.connect(mergeMinidspDialog.pick_user_source_dir)
        self.clearUserSourceDir.clicked.connect(mergeMinidspDialog.clear_user_source_dir)
        self.beqRepos.currentTextChanged['QString'].connect(mergeMinidspDialog.update_beq_repo_status)
        self.copyOptimisedButton.clicked.connect(mergeMinidspDialog.copy_optimised)
        self.copyErrorsButton.clicked.connect(mergeMinidspDialog.copy_errors)
        QtCore.QMetaObject.connectSlotsByName(mergeMinidspDialog)
        mergeMinidspDialog.setTabOrder(self.beqRepos, self.lastCommitDate)
        mergeMinidspDialog.setTabOrder(self.lastCommitDate, self.refreshGitRepo)
        mergeMinidspDialog.setTabOrder(self.refreshGitRepo, self.lastCommitMessage)
        mergeMinidspDialog.setTabOrder(self.lastCommitMessage, self.userSourceDir)
        mergeMinidspDialog.setTabOrder(self.userSourceDir, self.clearUserSourceDir)
        mergeMinidspDialog.setTabOrder(self.clearUserSourceDir, self.userSourceDirPicker)
        mergeMinidspDialog.setTabOrder(self.userSourceDirPicker, self.configFile)
        mergeMinidspDialog.setTabOrder(self.configFile, self.configFilePicker)
        mergeMinidspDialog.setTabOrder(self.configFilePicker, self.outputDirectory)
        mergeMinidspDialog.setTabOrder(self.outputDirectory, self.outputDirectoryPicker)
        mergeMinidspDialog.setTabOrder(self.outputDirectoryPicker, self.minidspType)
        mergeMinidspDialog.setTabOrder(self.minidspType, self.filesProcessed)
        mergeMinidspDialog.setTabOrder(self.filesProcessed, self.totalFiles)
        mergeMinidspDialog.setTabOrder(self.totalFiles, self.processFiles)
        mergeMinidspDialog.setTabOrder(self.processFiles, self.errors)

    def retranslateUi(self, mergeMinidspDialog):
        _translate = QtCore.QCoreApplication.translate
        mergeMinidspDialog.setWindowTitle(_translate("mergeMinidspDialog", "Merge Minidsp Config"))
        self.outputDirectoryLabel.setText(_translate("mergeMinidspDialog", "Output Directory"))
        self.filesProcessedLabel.setText(_translate("mergeMinidspDialog", "Files Processed"))
        self.configFileLabel.setText(_translate("mergeMinidspDialog", "Config File"))
        self.clearUserSourceDir.setText(_translate("mergeMinidspDialog", "..."))
        self.errorsLabel.setText(_translate("mergeMinidspDialog", "Errors"))
        self.lastUpdateLabel.setText(_translate("mergeMinidspDialog", "Last Update"))
        self.processFiles.setText(_translate("mergeMinidspDialog", "..."))
        self.minidspTypeLabel.setText(_translate("mergeMinidspDialog", "Minidsp Type"))
        self.beqReposLabel.setText(_translate("mergeMinidspDialog", "Repo"))
        self.ofLabel.setText(_translate("mergeMinidspDialog", "of"))
        self.outputDirectoryPicker.setText(_translate("mergeMinidspDialog", "..."))
        self.refreshGitRepo.setText(_translate("mergeMinidspDialog", "..."))
        self.configFilePicker.setText(_translate("mergeMinidspDialog", "..."))
        self.userSourceDirLabel.setText(_translate("mergeMinidspDialog", "User Source Directory"))
        self.userSourceDirPicker.setText(_translate("mergeMinidspDialog", "..."))
        self.optimisedLabel.setText(_translate("mergeMinidspDialog", "Optimised"))
        self.copyOptimisedButton.setText(_translate("mergeMinidspDialog", "..."))
        self.copyErrorsButton.setText(_translate("mergeMinidspDialog", "..."))
