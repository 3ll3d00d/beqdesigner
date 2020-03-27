# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'link.ui'
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


class Ui_linkSignalDialog(object):
    def setupUi(self, linkSignalDialog):
        if linkSignalDialog.objectName():
            linkSignalDialog.setObjectName(u"linkSignalDialog")
        linkSignalDialog.resize(833, 325)
        self.gridLayout = QGridLayout(linkSignalDialog)
        self.gridLayout.setObjectName(u"gridLayout")
        self.addToMaster = QToolButton(linkSignalDialog)
        self.addToMaster.setObjectName(u"addToMaster")

        self.gridLayout.addWidget(self.addToMaster, 0, 2, 1, 1)

        self.masterCandidates = QComboBox(linkSignalDialog)
        self.masterCandidates.setObjectName(u"masterCandidates")

        self.gridLayout.addWidget(self.masterCandidates, 0, 1, 1, 1)

        self.masterCandidatesLabel = QLabel(linkSignalDialog)
        self.masterCandidatesLabel.setObjectName(u"masterCandidatesLabel")

        self.gridLayout.addWidget(self.masterCandidatesLabel, 0, 0, 1, 1)

        self.linkSignals = QTableView(linkSignalDialog)
        self.linkSignals.setObjectName(u"linkSignals")

        self.gridLayout.addWidget(self.linkSignals, 1, 0, 1, 3)

        self.buttonBox = QDialogButtonBox(linkSignalDialog)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Save)

        self.gridLayout.addWidget(self.buttonBox, 2, 0, 1, 3)

        self.gridLayout.setColumnStretch(1, 1)

        self.retranslateUi(linkSignalDialog)
        self.buttonBox.accepted.connect(linkSignalDialog.accept)
        self.buttonBox.rejected.connect(linkSignalDialog.reject)
        self.addToMaster.clicked.connect(linkSignalDialog.addMaster)

        QMetaObject.connectSlotsByName(linkSignalDialog)
    # setupUi

    def retranslateUi(self, linkSignalDialog):
        linkSignalDialog.setWindowTitle(QCoreApplication.translate("linkSignalDialog", u"Link Signals", None))
        self.addToMaster.setText(QCoreApplication.translate("linkSignalDialog", u"...", None))
        self.masterCandidatesLabel.setText(QCoreApplication.translate("linkSignalDialog", u"Make Master", None))
    # retranslateUi

