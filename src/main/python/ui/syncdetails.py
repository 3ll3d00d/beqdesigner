# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'syncdetails.ui'
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


class Ui_syncDetailsDialog(object):
    def setupUi(self, syncDetailsDialog):
        if syncDetailsDialog.objectName():
            syncDetailsDialog.setObjectName(u"syncDetailsDialog")
        syncDetailsDialog.resize(663, 539)
        self.gridLayout = QGridLayout(syncDetailsDialog)
        self.gridLayout.setObjectName(u"gridLayout")
        self.buttonBox = QDialogButtonBox(syncDetailsDialog)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Close)

        self.gridLayout.addWidget(self.buttonBox, 1, 0, 1, 1)

        self.scrollArea = QScrollArea(syncDetailsDialog)
        self.scrollArea.setObjectName(u"scrollArea")
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 643, 490))
        self.horizontalLayout = QHBoxLayout(self.scrollAreaWidgetContents)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.gridLayout_2 = QGridLayout()
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_2.addItem(self.verticalSpacer, 1, 0, 1, 4)

        self.label_2 = QLabel(self.scrollAreaWidgetContents)
        self.label_2.setObjectName(u"label_2")
        font = QFont()
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setFrameShape(QFrame.Box)
        self.label_2.setFrameShadow(QFrame.Sunken)
        self.label_2.setAlignment(Qt.AlignHCenter|Qt.AlignTop)

        self.gridLayout_2.addWidget(self.label_2, 0, 1, 1, 1)

        self.label_4 = QLabel(self.scrollAreaWidgetContents)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setFont(font)
        self.label_4.setFrameShape(QFrame.Box)
        self.label_4.setFrameShadow(QFrame.Sunken)
        self.label_4.setAlignment(Qt.AlignHCenter|Qt.AlignTop)

        self.gridLayout_2.addWidget(self.label_4, 0, 3, 1, 1)

        self.label = QLabel(self.scrollAreaWidgetContents)
        self.label.setObjectName(u"label")
        self.label.setFont(font)
        self.label.setFrameShape(QFrame.Box)
        self.label.setFrameShadow(QFrame.Sunken)
        self.label.setAlignment(Qt.AlignHCenter|Qt.AlignTop)

        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)

        self.label_3 = QLabel(self.scrollAreaWidgetContents)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setFont(font)
        self.label_3.setFrameShape(QFrame.Box)
        self.label_3.setFrameShadow(QFrame.Sunken)
        self.label_3.setAlignment(Qt.AlignHCenter|Qt.AlignTop)

        self.gridLayout_2.addWidget(self.label_3, 0, 2, 1, 1)


        self.horizontalLayout.addLayout(self.gridLayout_2)

        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        self.gridLayout.addWidget(self.scrollArea, 0, 0, 1, 1)


        self.retranslateUi(syncDetailsDialog)
        self.buttonBox.accepted.connect(syncDetailsDialog.accept)
        self.buttonBox.rejected.connect(syncDetailsDialog.reject)

        QMetaObject.connectSlotsByName(syncDetailsDialog)
    # setupUi

    def retranslateUi(self, syncDetailsDialog):
        syncDetailsDialog.setWindowTitle(QCoreApplication.translate("syncDetailsDialog", u"Sync Details", None))
        self.label_2.setText(QCoreApplication.translate("syncDetailsDialog", u"Requested", None))
        self.label_4.setText(QCoreApplication.translate("syncDetailsDialog", u"Status", None))
        self.label.setText(QCoreApplication.translate("syncDetailsDialog", u"Path", None))
        self.label_3.setText(QCoreApplication.translate("syncDetailsDialog", u"Actual", None))
    # retranslateUi

