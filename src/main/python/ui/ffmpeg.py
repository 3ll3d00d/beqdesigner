# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ffmpeg.ui'
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


class Ui_ffmpegReportDialog(object):
    def setupUi(self, ffmpegReportDialog):
        if ffmpegReportDialog.objectName():
            ffmpegReportDialog.setObjectName(u"ffmpegReportDialog")
        ffmpegReportDialog.resize(800, 300)
        self.verticalLayout = QVBoxLayout(ffmpegReportDialog)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.message = QLabel(ffmpegReportDialog)
        self.message.setObjectName(u"message")
        font = QFont()
        font.setPointSize(8)
        self.message.setFont(font)

        self.verticalLayout.addWidget(self.message)

        self.details = QPlainTextEdit(ffmpegReportDialog)
        self.details.setObjectName(u"details")
        font1 = QFont()
        font1.setFamily(u"Consolas")
        self.details.setFont(font1)

        self.verticalLayout.addWidget(self.details)

        self.buttonBox = QDialogButtonBox(ffmpegReportDialog)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Close)

        self.verticalLayout.addWidget(self.buttonBox)


        self.retranslateUi(ffmpegReportDialog)
        self.buttonBox.accepted.connect(ffmpegReportDialog.accept)
        self.buttonBox.rejected.connect(ffmpegReportDialog.reject)

        QMetaObject.connectSlotsByName(ffmpegReportDialog)
    # setupUi

    def retranslateUi(self, ffmpegReportDialog):
        ffmpegReportDialog.setWindowTitle(QCoreApplication.translate("ffmpegReportDialog", u"ffmpeg", None))
        self.message.setText("")
    # retranslateUi

