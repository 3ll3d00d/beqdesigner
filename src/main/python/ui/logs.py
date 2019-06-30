# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'logs.ui',
# licensing of 'logs.ui' applies.
#
# Created: Sun Jun 30 22:06:41 2019
#      by: pyside2-uic  running on PySide2 5.13.0
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_logsForm(object):
    def setupUi(self, logsForm):
        logsForm.setObjectName("logsForm")
        logsForm.setEnabled(True)
        logsForm.resize(960, 768)
        self.centralwidget = QtWidgets.QWidget(logsForm)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.maxRows = QtWidgets.QSpinBox(self.centralwidget)
        self.maxRows.setMinimum(10)
        self.maxRows.setMaximum(5000)
        self.maxRows.setSingleStep(10)
        self.maxRows.setProperty("value", 1000)
        self.maxRows.setObjectName("maxRows")
        self.gridLayout.addWidget(self.maxRows, 0, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.logLevel = QtWidgets.QComboBox(self.centralwidget)
        self.logLevel.setObjectName("logLevel")
        self.logLevel.addItem("")
        self.logLevel.addItem("")
        self.logLevel.addItem("")
        self.logLevel.addItem("")
        self.logLevel.addItem("")
        self.gridLayout.addWidget(self.logLevel, 1, 1, 1, 1)
        self.logViewer = QtWidgets.QPlainTextEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(9)
        self.logViewer.setFont(font)
        self.logViewer.setReadOnly(True)
        self.logViewer.setObjectName("logViewer")
        self.gridLayout.addWidget(self.logViewer, 2, 0, 1, 2)
        logsForm.setCentralWidget(self.centralwidget)

        self.retranslateUi(logsForm)
        QtCore.QObject.connect(self.maxRows, QtCore.SIGNAL("valueChanged(int)"), logsForm.setLogSize)
        QtCore.QObject.connect(self.logLevel, QtCore.SIGNAL("currentTextChanged(QString)"), logsForm.setLogLevel)
        QtCore.QMetaObject.connectSlotsByName(logsForm)

    def retranslateUi(self, logsForm):
        logsForm.setWindowTitle(QtWidgets.QApplication.translate("logsForm", "Logs", None, -1))
        self.label.setText(QtWidgets.QApplication.translate("logsForm", "Log Size", None, -1))
        self.label_2.setText(QtWidgets.QApplication.translate("logsForm", "Log Level", None, -1))
        self.logLevel.setCurrentText(QtWidgets.QApplication.translate("logsForm", "DEBUG", None, -1))
        self.logLevel.setItemText(0, QtWidgets.QApplication.translate("logsForm", "DEBUG", None, -1))
        self.logLevel.setItemText(1, QtWidgets.QApplication.translate("logsForm", "INFO", None, -1))
        self.logLevel.setItemText(2, QtWidgets.QApplication.translate("logsForm", "WARNING", None, -1))
        self.logLevel.setItemText(3, QtWidgets.QApplication.translate("logsForm", "ERROR", None, -1))
        self.logLevel.setItemText(4, QtWidgets.QApplication.translate("logsForm", "CRITICAL", None, -1))

