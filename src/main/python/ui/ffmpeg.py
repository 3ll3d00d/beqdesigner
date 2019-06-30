# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ffmpeg.ui',
# licensing of 'ffmpeg.ui' applies.
#
# Created: Sun Jun 30 22:06:40 2019
#      by: pyside2-uic  running on PySide2 5.13.0
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_ffmpegReportDialog(object):
    def setupUi(self, ffmpegReportDialog):
        ffmpegReportDialog.setObjectName("ffmpegReportDialog")
        ffmpegReportDialog.resize(800, 300)
        self.verticalLayout = QtWidgets.QVBoxLayout(ffmpegReportDialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.message = QtWidgets.QLabel(ffmpegReportDialog)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.message.setFont(font)
        self.message.setText("")
        self.message.setObjectName("message")
        self.verticalLayout.addWidget(self.message)
        self.details = QtWidgets.QPlainTextEdit(ffmpegReportDialog)
        font = QtGui.QFont()
        font.setFamily("Consolas")
        self.details.setFont(font)
        self.details.setObjectName("details")
        self.verticalLayout.addWidget(self.details)
        self.buttonBox = QtWidgets.QDialogButtonBox(ffmpegReportDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Close)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(ffmpegReportDialog)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("accepted()"), ffmpegReportDialog.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("rejected()"), ffmpegReportDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(ffmpegReportDialog)

    def retranslateUi(self, ffmpegReportDialog):
        ffmpegReportDialog.setWindowTitle(QtWidgets.QApplication.translate("ffmpegReportDialog", "ffmpeg", None, -1))

