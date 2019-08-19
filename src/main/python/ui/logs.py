# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'logs.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


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
        self.maxRows.valueChanged['int'].connect(logsForm.setLogSize)
        self.logLevel.currentTextChanged['QString'].connect(logsForm.setLogLevel)
        QtCore.QMetaObject.connectSlotsByName(logsForm)

    def retranslateUi(self, logsForm):
        _translate = QtCore.QCoreApplication.translate
        logsForm.setWindowTitle(_translate("logsForm", "Logs"))
        self.label.setText(_translate("logsForm", "Log Size"))
        self.label_2.setText(_translate("logsForm", "Log Level"))
        self.logLevel.setCurrentText(_translate("logsForm", "DEBUG"))
        self.logLevel.setItemText(0, _translate("logsForm", "DEBUG"))
        self.logLevel.setItemText(1, _translate("logsForm", "INFO"))
        self.logLevel.setItemText(2, _translate("logsForm", "WARNING"))
        self.logLevel.setItemText(3, _translate("logsForm", "ERROR"))
        self.logLevel.setItemText(4, _translate("logsForm", "CRITICAL"))
