# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'newversion.ui'
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


class Ui_newVersionDialog(object):
    def setupUi(self, newVersionDialog):
        if newVersionDialog.objectName():
            newVersionDialog.setObjectName(u"newVersionDialog")
        newVersionDialog.resize(586, 544)
        self.verticalLayout = QVBoxLayout(newVersionDialog)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.headerLayout = QHBoxLayout()
        self.headerLayout.setObjectName(u"headerLayout")
        self.message = QLabel(newVersionDialog)
        self.message.setObjectName(u"message")

        self.headerLayout.addWidget(self.message)

        self.headerLayout.setStretch(0, 1)

        self.verticalLayout.addLayout(self.headerLayout)

        self.versionTable = QTableView(newVersionDialog)
        self.versionTable.setObjectName(u"versionTable")
        self.versionTable.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.versionTable.setSelectionMode(QAbstractItemView.MultiSelection)
        self.versionTable.setSelectionBehavior(QAbstractItemView.SelectRows)

        self.verticalLayout.addWidget(self.versionTable)

        self.releaseNotes = QTextBrowser(newVersionDialog)
        self.releaseNotes.setObjectName(u"releaseNotes")
        self.releaseNotes.setOpenExternalLinks(True)

        self.verticalLayout.addWidget(self.releaseNotes)

        self.buttonBox = QDialogButtonBox(newVersionDialog)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Ok)

        self.verticalLayout.addWidget(self.buttonBox)

        self.verticalLayout.setStretch(1, 1)
        self.verticalLayout.setStretch(2, 1)

        self.retranslateUi(newVersionDialog)
        self.buttonBox.accepted.connect(newVersionDialog.accept)
        self.buttonBox.rejected.connect(newVersionDialog.reject)

        QMetaObject.connectSlotsByName(newVersionDialog)
    # setupUi

    def retranslateUi(self, newVersionDialog):
        newVersionDialog.setWindowTitle(QCoreApplication.translate("newVersionDialog", u"New Version Available!", None))
        self.message.setText(QCoreApplication.translate("newVersionDialog", u"Message", None))
    # retranslateUi

