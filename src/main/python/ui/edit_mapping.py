# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'edit_mapping.ui'
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


class Ui_editMappingDialog(object):
    def setupUi(self, editMappingDialog):
        if editMappingDialog.objectName():
            editMappingDialog.setObjectName(u"editMappingDialog")
        editMappingDialog.resize(415, 347)
        self.gridLayout = QGridLayout(editMappingDialog)
        self.gridLayout.setObjectName(u"gridLayout")
        self.channelLabel = QLabel(editMappingDialog)
        self.channelLabel.setObjectName(u"channelLabel")

        self.gridLayout.addWidget(self.channelLabel, 0, 0, 1, 1)

        self.channels = QListWidget(editMappingDialog)
        self.channels.setObjectName(u"channels")
        self.channels.setSelectionMode(QAbstractItemView.MultiSelection)

        self.gridLayout.addWidget(self.channels, 0, 1, 1, 1)

        self.signalLabel = QLabel(editMappingDialog)
        self.signalLabel.setObjectName(u"signalLabel")

        self.gridLayout.addWidget(self.signalLabel, 1, 0, 1, 1)

        self.signal = QComboBox(editMappingDialog)
        self.signal.setObjectName(u"signal")

        self.gridLayout.addWidget(self.signal, 1, 1, 1, 1)

        self.buttonBox = QDialogButtonBox(editMappingDialog)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Save)

        self.gridLayout.addWidget(self.buttonBox, 2, 1, 1, 1)


        self.retranslateUi(editMappingDialog)
        self.buttonBox.accepted.connect(editMappingDialog.accept)
        self.buttonBox.rejected.connect(editMappingDialog.reject)

        QMetaObject.connectSlotsByName(editMappingDialog)
    # setupUi

    def retranslateUi(self, editMappingDialog):
        editMappingDialog.setWindowTitle(QCoreApplication.translate("editMappingDialog", u"Edit Mapping", None))
        self.channelLabel.setText(QCoreApplication.translate("editMappingDialog", u"Channels", None))
        self.signalLabel.setText(QCoreApplication.translate("editMappingDialog", u"Signal", None))
    # retranslateUi

