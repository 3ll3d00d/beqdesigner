# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'minidsp.ui'
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


class Ui_mergeMinidspDialog(object):
    def setupUi(self, mergeMinidspDialog):
        if mergeMinidspDialog.objectName():
            mergeMinidspDialog.setObjectName(u"mergeMinidspDialog")
        mergeMinidspDialog.resize(639, 655)
        self.gridLayout = QGridLayout(mergeMinidspDialog)
        self.gridLayout.setObjectName(u"gridLayout")
        self.buttonBox = QDialogButtonBox(mergeMinidspDialog)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Close|QDialogButtonBox.Reset)

        self.gridLayout.addWidget(self.buttonBox, 1, 0, 1, 1)

        self.innerLayout = QGridLayout()
        self.innerLayout.setObjectName(u"innerLayout")
        self.outputDirectoryLabel = QLabel(mergeMinidspDialog)
        self.outputDirectoryLabel.setObjectName(u"outputDirectoryLabel")

        self.innerLayout.addWidget(self.outputDirectoryLabel, 5, 0, 1, 1)

        self.filesProcessedLabel = QLabel(mergeMinidspDialog)
        self.filesProcessedLabel.setObjectName(u"filesProcessedLabel")

        self.innerLayout.addWidget(self.filesProcessedLabel, 7, 0, 1, 1)

        self.totalFiles = QSpinBox(mergeMinidspDialog)
        self.totalFiles.setObjectName(u"totalFiles")
        self.totalFiles.setReadOnly(True)
        self.totalFiles.setMaximum(10000)

        self.innerLayout.addWidget(self.totalFiles, 7, 3, 1, 1)

        self.configFileLabel = QLabel(mergeMinidspDialog)
        self.configFileLabel.setObjectName(u"configFileLabel")

        self.innerLayout.addWidget(self.configFileLabel, 4, 0, 1, 1)

        self.userSourceDirLayout = QHBoxLayout()
        self.userSourceDirLayout.setObjectName(u"userSourceDirLayout")
        self.userSourceDir = QLineEdit(mergeMinidspDialog)
        self.userSourceDir.setObjectName(u"userSourceDir")
        self.userSourceDir.setReadOnly(True)

        self.userSourceDirLayout.addWidget(self.userSourceDir)

        self.clearUserSourceDir = QToolButton(mergeMinidspDialog)
        self.clearUserSourceDir.setObjectName(u"clearUserSourceDir")

        self.userSourceDirLayout.addWidget(self.clearUserSourceDir)


        self.innerLayout.addLayout(self.userSourceDirLayout, 3, 1, 1, 4)

        self.lastCommitDate = QDateTimeEdit(mergeMinidspDialog)
        self.lastCommitDate.setObjectName(u"lastCommitDate")
        self.lastCommitDate.setReadOnly(True)

        self.innerLayout.addWidget(self.lastCommitDate, 1, 1, 1, 3)

        self.beqRepos = QComboBox(mergeMinidspDialog)
        self.beqRepos.setObjectName(u"beqRepos")

        self.innerLayout.addWidget(self.beqRepos, 0, 1, 1, 4)

        self.minidspType = QComboBox(mergeMinidspDialog)
        self.minidspType.addItem("")
        self.minidspType.addItem("")
        self.minidspType.addItem("")
        self.minidspType.setObjectName(u"minidspType")

        self.innerLayout.addWidget(self.minidspType, 6, 1, 1, 4)

        self.errors = QListWidget(mergeMinidspDialog)
        self.errors.setObjectName(u"errors")
        self.errors.setEnabled(False)

        self.innerLayout.addWidget(self.errors, 9, 1, 1, 4)

        self.errorsLabel = QLabel(mergeMinidspDialog)
        self.errorsLabel.setObjectName(u"errorsLabel")

        self.innerLayout.addWidget(self.errorsLabel, 9, 0, 1, 1)

        self.lastUpdateLabel = QLabel(mergeMinidspDialog)
        self.lastUpdateLabel.setObjectName(u"lastUpdateLabel")

        self.innerLayout.addWidget(self.lastUpdateLabel, 1, 0, 1, 1)

        self.outputDirectory = QLineEdit(mergeMinidspDialog)
        self.outputDirectory.setObjectName(u"outputDirectory")
        self.outputDirectory.setEnabled(False)
        self.outputDirectory.setReadOnly(True)

        self.innerLayout.addWidget(self.outputDirectory, 5, 1, 1, 4)

        self.processFiles = QToolButton(mergeMinidspDialog)
        self.processFiles.setObjectName(u"processFiles")

        self.innerLayout.addWidget(self.processFiles, 7, 5, 1, 1)

        self.minidspTypeLabel = QLabel(mergeMinidspDialog)
        self.minidspTypeLabel.setObjectName(u"minidspTypeLabel")

        self.innerLayout.addWidget(self.minidspTypeLabel, 6, 0, 1, 1)

        self.filesProcessed = QSpinBox(mergeMinidspDialog)
        self.filesProcessed.setObjectName(u"filesProcessed")
        self.filesProcessed.setReadOnly(True)
        self.filesProcessed.setMaximum(100000)

        self.innerLayout.addWidget(self.filesProcessed, 7, 1, 1, 1)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.innerLayout.addItem(self.horizontalSpacer, 7, 4, 1, 1)

        self.beqReposLabel = QLabel(mergeMinidspDialog)
        self.beqReposLabel.setObjectName(u"beqReposLabel")

        self.innerLayout.addWidget(self.beqReposLabel, 0, 0, 1, 1)

        self.lastCommitMessage = QPlainTextEdit(mergeMinidspDialog)
        self.lastCommitMessage.setObjectName(u"lastCommitMessage")
        self.lastCommitMessage.setReadOnly(True)

        self.innerLayout.addWidget(self.lastCommitMessage, 2, 1, 1, 4)

        self.ofLabel = QLabel(mergeMinidspDialog)
        self.ofLabel.setObjectName(u"ofLabel")

        self.innerLayout.addWidget(self.ofLabel, 7, 2, 1, 1)

        self.configFile = QLineEdit(mergeMinidspDialog)
        self.configFile.setObjectName(u"configFile")
        self.configFile.setEnabled(False)
        self.configFile.setReadOnly(True)

        self.innerLayout.addWidget(self.configFile, 4, 1, 1, 4)

        self.outputDirectoryPicker = QToolButton(mergeMinidspDialog)
        self.outputDirectoryPicker.setObjectName(u"outputDirectoryPicker")

        self.innerLayout.addWidget(self.outputDirectoryPicker, 5, 5, 1, 1)

        self.refreshGitRepo = QToolButton(mergeMinidspDialog)
        self.refreshGitRepo.setObjectName(u"refreshGitRepo")

        self.innerLayout.addWidget(self.refreshGitRepo, 1, 5, 1, 1)

        self.infoLabel = QLabel(mergeMinidspDialog)
        self.infoLabel.setObjectName(u"infoLabel")

        self.innerLayout.addWidget(self.infoLabel, 1, 4, 1, 1)

        self.configFilePicker = QToolButton(mergeMinidspDialog)
        self.configFilePicker.setObjectName(u"configFilePicker")

        self.innerLayout.addWidget(self.configFilePicker, 4, 5, 1, 1)

        self.userSourceDirLabel = QLabel(mergeMinidspDialog)
        self.userSourceDirLabel.setObjectName(u"userSourceDirLabel")

        self.innerLayout.addWidget(self.userSourceDirLabel, 3, 0, 1, 1)

        self.userSourceDirPicker = QToolButton(mergeMinidspDialog)
        self.userSourceDirPicker.setObjectName(u"userSourceDirPicker")

        self.innerLayout.addWidget(self.userSourceDirPicker, 3, 5, 1, 1)

        self.optimisedLabel = QLabel(mergeMinidspDialog)
        self.optimisedLabel.setObjectName(u"optimisedLabel")

        self.innerLayout.addWidget(self.optimisedLabel, 8, 0, 1, 1)

        self.optimised = QListWidget(mergeMinidspDialog)
        self.optimised.setObjectName(u"optimised")

        self.innerLayout.addWidget(self.optimised, 8, 1, 1, 4)

        self.copyOptimisedButton = QToolButton(mergeMinidspDialog)
        self.copyOptimisedButton.setObjectName(u"copyOptimisedButton")

        self.innerLayout.addWidget(self.copyOptimisedButton, 8, 5, 1, 1)

        self.copyErrorsButton = QToolButton(mergeMinidspDialog)
        self.copyErrorsButton.setObjectName(u"copyErrorsButton")

        self.innerLayout.addWidget(self.copyErrorsButton, 9, 5, 1, 1)


        self.gridLayout.addLayout(self.innerLayout, 0, 0, 1, 1)

        QWidget.setTabOrder(self.beqRepos, self.lastCommitDate)
        QWidget.setTabOrder(self.lastCommitDate, self.refreshGitRepo)
        QWidget.setTabOrder(self.refreshGitRepo, self.lastCommitMessage)
        QWidget.setTabOrder(self.lastCommitMessage, self.userSourceDir)
        QWidget.setTabOrder(self.userSourceDir, self.clearUserSourceDir)
        QWidget.setTabOrder(self.clearUserSourceDir, self.userSourceDirPicker)
        QWidget.setTabOrder(self.userSourceDirPicker, self.configFile)
        QWidget.setTabOrder(self.configFile, self.configFilePicker)
        QWidget.setTabOrder(self.configFilePicker, self.outputDirectory)
        QWidget.setTabOrder(self.outputDirectory, self.outputDirectoryPicker)
        QWidget.setTabOrder(self.outputDirectoryPicker, self.minidspType)
        QWidget.setTabOrder(self.minidspType, self.filesProcessed)
        QWidget.setTabOrder(self.filesProcessed, self.totalFiles)
        QWidget.setTabOrder(self.totalFiles, self.processFiles)
        QWidget.setTabOrder(self.processFiles, self.errors)

        self.retranslateUi(mergeMinidspDialog)
        self.buttonBox.accepted.connect(mergeMinidspDialog.accept)
        self.buttonBox.rejected.connect(mergeMinidspDialog.reject)
        self.configFilePicker.clicked.connect(mergeMinidspDialog.pick_config_file)
        self.outputDirectoryPicker.clicked.connect(mergeMinidspDialog.pick_output_dir)
        self.processFiles.clicked.connect(mergeMinidspDialog.process_files)
        self.refreshGitRepo.clicked.connect(mergeMinidspDialog.refresh_repo)
        self.userSourceDirPicker.clicked.connect(mergeMinidspDialog.pick_user_source_dir)
        self.clearUserSourceDir.clicked.connect(mergeMinidspDialog.clear_user_source_dir)
        self.beqRepos.currentTextChanged.connect(mergeMinidspDialog.update_beq_repo_status)
        self.copyOptimisedButton.clicked.connect(mergeMinidspDialog.copy_optimised)
        self.copyErrorsButton.clicked.connect(mergeMinidspDialog.copy_errors)

        QMetaObject.connectSlotsByName(mergeMinidspDialog)
    # setupUi

    def retranslateUi(self, mergeMinidspDialog):
        mergeMinidspDialog.setWindowTitle(QCoreApplication.translate("mergeMinidspDialog", u"Merge Minidsp Config", None))
        self.outputDirectoryLabel.setText(QCoreApplication.translate("mergeMinidspDialog", u"Output Directory", None))
        self.filesProcessedLabel.setText(QCoreApplication.translate("mergeMinidspDialog", u"Files Processed", None))
        self.configFileLabel.setText(QCoreApplication.translate("mergeMinidspDialog", u"Config File", None))
        self.clearUserSourceDir.setText(QCoreApplication.translate("mergeMinidspDialog", u"...", None))
        self.minidspType.setItemText(0, QCoreApplication.translate("mergeMinidspDialog", u"2x4 HD", None))
        self.minidspType.setItemText(1, QCoreApplication.translate("mergeMinidspDialog", u"2x4", None))
        self.minidspType.setItemText(2, QCoreApplication.translate("mergeMinidspDialog", u"10x10 HD", None))

        self.errorsLabel.setText(QCoreApplication.translate("mergeMinidspDialog", u"Errors", None))
        self.lastUpdateLabel.setText(QCoreApplication.translate("mergeMinidspDialog", u"Last Update", None))
        self.processFiles.setText(QCoreApplication.translate("mergeMinidspDialog", u"...", None))
        self.minidspTypeLabel.setText(QCoreApplication.translate("mergeMinidspDialog", u"Minidsp Type", None))
        self.beqReposLabel.setText(QCoreApplication.translate("mergeMinidspDialog", u"Repo", None))
        self.ofLabel.setText(QCoreApplication.translate("mergeMinidspDialog", u"of", None))
        self.outputDirectoryPicker.setText(QCoreApplication.translate("mergeMinidspDialog", u"...", None))
        self.refreshGitRepo.setText(QCoreApplication.translate("mergeMinidspDialog", u"...", None))
        self.infoLabel.setText("")
        self.configFilePicker.setText(QCoreApplication.translate("mergeMinidspDialog", u"...", None))
        self.userSourceDirLabel.setText(QCoreApplication.translate("mergeMinidspDialog", u"User Source Directory", None))
        self.userSourceDirPicker.setText(QCoreApplication.translate("mergeMinidspDialog", u"...", None))
        self.optimisedLabel.setText(QCoreApplication.translate("mergeMinidspDialog", u"Optimised", None))
        self.copyOptimisedButton.setText(QCoreApplication.translate("mergeMinidspDialog", u"...", None))
        self.copyErrorsButton.setText(QCoreApplication.translate("mergeMinidspDialog", u"...", None))
    # retranslateUi

