# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'logs.ui'
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


class Ui_logsForm(object):
    def setupUi(self, logsForm):
        if logsForm.objectName():
            logsForm.setObjectName(u"logsForm")
        logsForm.setEnabled(True)
        logsForm.resize(960, 768)
        self.centralwidget = QWidget(logsForm)
        self.centralwidget.setObjectName(u"centralwidget")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.logViewer = QPlainTextEdit(self.centralwidget)
        self.logViewer.setObjectName(u"logViewer")
        font = QFont()
        font.setFamily(u"Consolas")
        font.setPointSize(9)
        self.logViewer.setFont(font)
        self.logViewer.setReadOnly(True)

        self.gridLayout.addWidget(self.logViewer, 3, 0, 1, 2)

        self.logLevel = QComboBox(self.centralwidget)
        self.logLevel.addItem("")
        self.logLevel.addItem("")
        self.logLevel.addItem("")
        self.logLevel.addItem("")
        self.logLevel.addItem("")
        self.logLevel.setObjectName(u"logLevel")

        self.gridLayout.addWidget(self.logLevel, 1, 1, 1, 1)

        self.logSizeLabel = QLabel(self.centralwidget)
        self.logSizeLabel.setObjectName(u"logSizeLabel")

        self.gridLayout.addWidget(self.logSizeLabel, 0, 0, 1, 1)

        self.maxRows = QSpinBox(self.centralwidget)
        self.maxRows.setObjectName(u"maxRows")
        self.maxRows.setMinimum(10)
        self.maxRows.setMaximum(5000)
        self.maxRows.setSingleStep(10)
        self.maxRows.setValue(1000)

        self.gridLayout.addWidget(self.maxRows, 0, 1, 1, 1)

        self.logLevelLabel = QLabel(self.centralwidget)
        self.logLevelLabel.setObjectName(u"logLevelLabel")

        self.gridLayout.addWidget(self.logLevelLabel, 1, 0, 1, 1)

        self.excludesLabel = QLabel(self.centralwidget)
        self.excludesLabel.setObjectName(u"excludesLabel")

        self.gridLayout.addWidget(self.excludesLabel, 2, 0, 1, 1)

        self.excludes = QLineEdit(self.centralwidget)
        self.excludes.setObjectName(u"excludes")

        self.gridLayout.addWidget(self.excludes, 2, 1, 1, 1)

        logsForm.setCentralWidget(self.centralwidget)
        QWidget.setTabOrder(self.maxRows, self.logLevel)
        QWidget.setTabOrder(self.logLevel, self.excludes)
        QWidget.setTabOrder(self.excludes, self.logViewer)

        self.retranslateUi(logsForm)
        self.maxRows.valueChanged.connect(logsForm.set_log_size)
        self.logLevel.currentTextChanged.connect(logsForm.set_log_level)
        self.excludes.returnPressed.connect(logsForm.set_excludes)

        QMetaObject.connectSlotsByName(logsForm)
    # setupUi

    def retranslateUi(self, logsForm):
        logsForm.setWindowTitle(QCoreApplication.translate("logsForm", u"Logs", None))
        self.logLevel.setItemText(0, QCoreApplication.translate("logsForm", u"DEBUG", None))
        self.logLevel.setItemText(1, QCoreApplication.translate("logsForm", u"INFO", None))
        self.logLevel.setItemText(2, QCoreApplication.translate("logsForm", u"WARNING", None))
        self.logLevel.setItemText(3, QCoreApplication.translate("logsForm", u"ERROR", None))
        self.logLevel.setItemText(4, QCoreApplication.translate("logsForm", u"CRITICAL", None))

        self.logLevel.setCurrentText(QCoreApplication.translate("logsForm", u"DEBUG", None))
        self.logSizeLabel.setText(QCoreApplication.translate("logsForm", u"Log Size", None))
        self.logLevelLabel.setText(QCoreApplication.translate("logsForm", u"Log Level", None))
        self.excludesLabel.setText(QCoreApplication.translate("logsForm", u"Excludes", None))
    # retranslateUi

