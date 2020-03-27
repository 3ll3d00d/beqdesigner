# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'stats.ui'
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


class Ui_signalStatsDialog(object):
    def setupUi(self, signalStatsDialog):
        if signalStatsDialog.objectName():
            signalStatsDialog.setObjectName(u"signalStatsDialog")
        signalStatsDialog.resize(306, 561)
        self.gridLayout = QGridLayout(signalStatsDialog)
        self.gridLayout.setObjectName(u"gridLayout")
        self.rms = QDoubleSpinBox(signalStatsDialog)
        self.rms.setObjectName(u"rms")
        self.rms.setReadOnly(True)
        self.rms.setMinimum(-150.000000000000000)
        self.rms.setMaximum(10.000000000000000)
        self.rms.setSingleStep(0.010000000000000)

        self.gridLayout.addWidget(self.rms, 2, 1, 1, 1)

        self.label_7 = QLabel(signalStatsDialog)
        self.label_7.setObjectName(u"label_7")

        self.gridLayout.addWidget(self.label_7, 8, 0, 1, 1)

        self.label_10 = QLabel(signalStatsDialog)
        self.label_10.setObjectName(u"label_10")
        font = QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_10.setFont(font)
        self.label_10.setFrameShape(QFrame.Box)
        self.label_10.setFrameShadow(QFrame.Sunken)
        self.label_10.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.label_10, 10, 0, 1, 3)

        self.slowPeak = QDoubleSpinBox(signalStatsDialog)
        self.slowPeak.setObjectName(u"slowPeak")
        self.slowPeak.setReadOnly(True)
        self.slowPeak.setMinimum(-150.000000000000000)
        self.slowPeak.setMaximum(10.000000000000000)
        self.slowPeak.setSingleStep(0.010000000000000)

        self.gridLayout.addWidget(self.slowPeak, 5, 1, 1, 1)

        self.label_4 = QLabel(signalStatsDialog)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout.addWidget(self.label_4, 5, 0, 1, 1)

        self.label_14 = QLabel(signalStatsDialog)
        self.label_14.setObjectName(u"label_14")

        self.gridLayout.addWidget(self.label_14, 11, 0, 1, 1)

        self.label_17 = QLabel(signalStatsDialog)
        self.label_17.setObjectName(u"label_17")
        self.label_17.setFont(font)

        self.gridLayout.addWidget(self.label_17, 14, 0, 1, 1)

        self.buttonBox = QDialogButtonBox(signalStatsDialog)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Close)

        self.gridLayout.addWidget(self.buttonBox, 18, 0, 1, 3)

        self.label_18 = QLabel(signalStatsDialog)
        self.label_18.setObjectName(u"label_18")
        self.label_18.setFont(font)

        self.gridLayout.addWidget(self.label_18, 17, 0, 1, 1)

        self.includeInstant = QToolButton(signalStatsDialog)
        self.includeInstant.setObjectName(u"includeInstant")
        self.includeInstant.setCheckable(True)
        self.includeInstant.setChecked(True)

        self.gridLayout.addWidget(self.includeInstant, 3, 2, 1, 1)

        self.customPeak = QDoubleSpinBox(signalStatsDialog)
        self.customPeak.setObjectName(u"customPeak")
        self.customPeak.setReadOnly(True)
        self.customPeak.setMinimum(-150.000000000000000)
        self.customPeak.setMaximum(10.000000000000000)
        self.customPeak.setSingleStep(0.010000000000000)

        self.gridLayout.addWidget(self.customPeak, 7, 1, 1, 1)

        self.gridLayout_3 = QGridLayout()
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.peakExtensionX1 = QDoubleSpinBox(signalStatsDialog)
        self.peakExtensionX1.setObjectName(u"peakExtensionX1")
        self.peakExtensionX1.setReadOnly(True)
        self.peakExtensionX1.setDecimals(1)
        self.peakExtensionX1.setMaximum(160.000000000000000)
        self.peakExtensionX1.setSingleStep(0.100000000000000)

        self.gridLayout_3.addWidget(self.peakExtensionX1, 0, 0, 1, 1)

        self.peakExtensionX2 = QDoubleSpinBox(signalStatsDialog)
        self.peakExtensionX2.setObjectName(u"peakExtensionX2")
        self.peakExtensionX2.setReadOnly(True)
        self.peakExtensionX2.setDecimals(1)
        self.peakExtensionX2.setMaximum(160.000000000000000)
        self.peakExtensionX2.setSingleStep(0.100000000000000)

        self.gridLayout_3.addWidget(self.peakExtensionX2, 0, 1, 1, 1)

        self.peakExtensionY1 = QDoubleSpinBox(signalStatsDialog)
        self.peakExtensionY1.setObjectName(u"peakExtensionY1")
        self.peakExtensionY1.setReadOnly(True)
        self.peakExtensionY1.setMinimum(-120.000000000000000)
        self.peakExtensionY1.setMaximum(0.000000000000000)
        self.peakExtensionY1.setSingleStep(0.010000000000000)

        self.gridLayout_3.addWidget(self.peakExtensionY1, 1, 0, 1, 1)

        self.peakExtensionY2 = QDoubleSpinBox(signalStatsDialog)
        self.peakExtensionY2.setObjectName(u"peakExtensionY2")
        self.peakExtensionY2.setReadOnly(True)
        self.peakExtensionY2.setMinimum(-120.000000000000000)
        self.peakExtensionY2.setMaximum(0.000000000000000)
        self.peakExtensionY2.setSingleStep(0.010000000000000)

        self.gridLayout_3.addWidget(self.peakExtensionY2, 1, 1, 1, 1)


        self.gridLayout.addLayout(self.gridLayout_3, 13, 1, 1, 2)

        self.label_6 = QLabel(signalStatsDialog)
        self.label_6.setObjectName(u"label_6")

        self.gridLayout.addWidget(self.label_6, 7, 0, 1, 1)

        self.label_11 = QLabel(signalStatsDialog)
        self.label_11.setObjectName(u"label_11")
        self.label_11.setFont(font)
        self.label_11.setFrameShape(QFrame.Box)
        self.label_11.setFrameShadow(QFrame.Sunken)
        self.label_11.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.label_11, 15, 0, 1, 3)

        self.includeFast = QToolButton(signalStatsDialog)
        self.includeFast.setObjectName(u"includeFast")
        self.includeFast.setCheckable(True)

        self.gridLayout.addWidget(self.includeFast, 4, 2, 1, 1)

        self.includeRMS = QToolButton(signalStatsDialog)
        self.includeRMS.setObjectName(u"includeRMS")
        self.includeRMS.setCheckable(True)
        self.includeRMS.setChecked(True)

        self.gridLayout.addWidget(self.includeRMS, 2, 2, 1, 1)

        self.label_19 = QLabel(signalStatsDialog)
        self.label_19.setObjectName(u"label_19")

        self.gridLayout.addWidget(self.label_19, 12, 0, 1, 1)

        self.label_2 = QLabel(signalStatsDialog)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout.addWidget(self.label_2, 3, 0, 1, 1)

        self.levelRating = QSpinBox(signalStatsDialog)
        self.levelRating.setObjectName(u"levelRating")
        self.levelRating.setReadOnly(True)
        self.levelRating.setMaximum(5)

        self.gridLayout.addWidget(self.levelRating, 9, 1, 1, 1)

        self.label_15 = QLabel(signalStatsDialog)
        self.label_15.setObjectName(u"label_15")

        self.gridLayout.addWidget(self.label_15, 16, 0, 1, 1)

        self.gridLayout_2 = QGridLayout()
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.averageExtensionX1 = QDoubleSpinBox(signalStatsDialog)
        self.averageExtensionX1.setObjectName(u"averageExtensionX1")
        self.averageExtensionX1.setReadOnly(True)
        self.averageExtensionX1.setDecimals(1)
        self.averageExtensionX1.setMaximum(160.000000000000000)
        self.averageExtensionX1.setSingleStep(0.100000000000000)

        self.gridLayout_2.addWidget(self.averageExtensionX1, 0, 0, 1, 1)

        self.averageExtensionX2 = QDoubleSpinBox(signalStatsDialog)
        self.averageExtensionX2.setObjectName(u"averageExtensionX2")
        self.averageExtensionX2.setReadOnly(True)
        self.averageExtensionX2.setDecimals(1)
        self.averageExtensionX2.setMaximum(160.000000000000000)
        self.averageExtensionX2.setSingleStep(0.100000000000000)

        self.gridLayout_2.addWidget(self.averageExtensionX2, 0, 1, 1, 1)

        self.averageExtensionY1 = QDoubleSpinBox(signalStatsDialog)
        self.averageExtensionY1.setObjectName(u"averageExtensionY1")
        self.averageExtensionY1.setReadOnly(True)
        self.averageExtensionY1.setMinimum(-120.000000000000000)
        self.averageExtensionY1.setMaximum(10.000000000000000)
        self.averageExtensionY1.setSingleStep(0.010000000000000)

        self.gridLayout_2.addWidget(self.averageExtensionY1, 1, 0, 1, 1)

        self.averageExtensionY2 = QDoubleSpinBox(signalStatsDialog)
        self.averageExtensionY2.setObjectName(u"averageExtensionY2")
        self.averageExtensionY2.setReadOnly(True)
        self.averageExtensionY2.setMinimum(-120.000000000000000)
        self.averageExtensionY2.setMaximum(10.000000000000000)
        self.averageExtensionY2.setSingleStep(0.010000000000000)

        self.gridLayout_2.addWidget(self.averageExtensionY2, 1, 1, 1, 1)


        self.gridLayout.addLayout(self.gridLayout_2, 12, 1, 1, 2)

        self.includeSlow = QToolButton(signalStatsDialog)
        self.includeSlow.setObjectName(u"includeSlow")
        self.includeSlow.setCheckable(True)
        self.includeSlow.setChecked(True)

        self.gridLayout.addWidget(self.includeSlow, 5, 2, 1, 1)

        self.label_9 = QLabel(signalStatsDialog)
        self.label_9.setObjectName(u"label_9")

        self.gridLayout.addWidget(self.label_9, 1, 2, 1, 1)

        self.label_8 = QLabel(signalStatsDialog)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setFont(font)
        self.label_8.setFrameShape(QFrame.Box)
        self.label_8.setFrameShadow(QFrame.Sunken)
        self.label_8.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.label_8, 0, 0, 1, 3)

        self.compositeLevel = QDoubleSpinBox(signalStatsDialog)
        self.compositeLevel.setObjectName(u"compositeLevel")
        self.compositeLevel.setReadOnly(True)
        self.compositeLevel.setMinimum(-150.000000000000000)
        self.compositeLevel.setMaximum(10.000000000000000)
        self.compositeLevel.setSingleStep(0.010000000000000)

        self.gridLayout.addWidget(self.compositeLevel, 8, 1, 1, 1)

        self.label = QLabel(signalStatsDialog)
        self.label.setObjectName(u"label")

        self.gridLayout.addWidget(self.label, 2, 0, 1, 1)

        self.label_16 = QLabel(signalStatsDialog)
        self.label_16.setObjectName(u"label_16")
        font1 = QFont()
        font1.setBold(True)
        font1.setItalic(False)
        font1.setWeight(75)
        self.label_16.setFont(font1)

        self.gridLayout.addWidget(self.label_16, 9, 0, 1, 1)

        self.instantPeak = QDoubleSpinBox(signalStatsDialog)
        self.instantPeak.setObjectName(u"instantPeak")
        self.instantPeak.setReadOnly(True)
        self.instantPeak.setMinimum(-150.000000000000000)
        self.instantPeak.setMaximum(10.000000000000000)
        self.instantPeak.setSingleStep(0.010000000000000)

        self.gridLayout.addWidget(self.instantPeak, 3, 1, 1, 1)

        self.label_3 = QLabel(signalStatsDialog)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout.addWidget(self.label_3, 4, 0, 1, 1)

        self.extensionLimit = QSpinBox(signalStatsDialog)
        self.extensionLimit.setObjectName(u"extensionLimit")
        self.extensionLimit.setMinimum(-20)
        self.extensionLimit.setMaximum(-1)
        self.extensionLimit.setValue(-10)

        self.gridLayout.addWidget(self.extensionLimit, 11, 1, 1, 2)

        self.label_12 = QLabel(signalStatsDialog)
        self.label_12.setObjectName(u"label_12")

        self.gridLayout.addWidget(self.label_12, 13, 0, 1, 1)

        self.customIntegrationTime = QSpinBox(signalStatsDialog)
        self.customIntegrationTime.setObjectName(u"customIntegrationTime")
        self.customIntegrationTime.setMinimum(1)
        self.customIntegrationTime.setMaximum(60000)
        self.customIntegrationTime.setValue(5000)

        self.gridLayout.addWidget(self.customIntegrationTime, 6, 1, 1, 1)

        self.label_5 = QLabel(signalStatsDialog)
        self.label_5.setObjectName(u"label_5")

        self.gridLayout.addWidget(self.label_5, 6, 0, 1, 1)

        self.dynamics = QDoubleSpinBox(signalStatsDialog)
        self.dynamics.setObjectName(u"dynamics")
        self.dynamics.setReadOnly(True)
        self.dynamics.setSingleStep(0.010000000000000)

        self.gridLayout.addWidget(self.dynamics, 16, 1, 1, 2)

        self.includeCustom = QToolButton(signalStatsDialog)
        self.includeCustom.setObjectName(u"includeCustom")
        self.includeCustom.setCheckable(True)

        self.gridLayout.addWidget(self.includeCustom, 7, 2, 1, 1)

        self.fastPeak = QDoubleSpinBox(signalStatsDialog)
        self.fastPeak.setObjectName(u"fastPeak")
        self.fastPeak.setReadOnly(True)
        self.fastPeak.setMinimum(-150.000000000000000)
        self.fastPeak.setMaximum(10.000000000000000)
        self.fastPeak.setSingleStep(0.010000000000000)

        self.gridLayout.addWidget(self.fastPeak, 4, 1, 1, 1)

        self.dynamicsRating = QSpinBox(signalStatsDialog)
        self.dynamicsRating.setObjectName(u"dynamicsRating")
        self.dynamicsRating.setReadOnly(True)
        self.dynamicsRating.setMaximum(5)

        self.gridLayout.addWidget(self.dynamicsRating, 17, 1, 1, 2)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.extensionSource = QComboBox(signalStatsDialog)
        self.extensionSource.addItem("")
        self.extensionSource.addItem("")
        self.extensionSource.addItem("")
        self.extensionSource.setObjectName(u"extensionSource")
        self.extensionSource.setEnabled(False)

        self.horizontalLayout.addWidget(self.extensionSource)

        self.extensionRating = QSpinBox(signalStatsDialog)
        self.extensionRating.setObjectName(u"extensionRating")
        self.extensionRating.setReadOnly(True)
        self.extensionRating.setMaximum(5)

        self.horizontalLayout.addWidget(self.extensionRating)


        self.gridLayout.addLayout(self.horizontalLayout, 14, 1, 1, 2)


        self.retranslateUi(signalStatsDialog)
        self.buttonBox.accepted.connect(signalStatsDialog.accept)
        self.buttonBox.rejected.connect(signalStatsDialog.reject)
        self.includeRMS.toggled.connect(signalStatsDialog.set_composite)
        self.includeInstant.toggled.connect(signalStatsDialog.set_composite)
        self.includeFast.toggled.connect(signalStatsDialog.set_composite)
        self.includeSlow.toggled.connect(signalStatsDialog.set_composite)
        self.includeCustom.toggled.connect(signalStatsDialog.set_composite)
        self.customIntegrationTime.editingFinished.connect(signalStatsDialog.set_custom_peak)
        self.extensionLimit.valueChanged.connect(signalStatsDialog.set_extension)
        self.compositeLevel.valueChanged.connect(signalStatsDialog.rate_level)

        QMetaObject.connectSlotsByName(signalStatsDialog)
    # setupUi

    def retranslateUi(self, signalStatsDialog):
        signalStatsDialog.setWindowTitle(QCoreApplication.translate("signalStatsDialog", u"Stats", None))
        self.rms.setSuffix(QCoreApplication.translate("signalStatsDialog", u" dB", None))
        self.label_7.setText(QCoreApplication.translate("signalStatsDialog", u"Composite Level", None))
        self.label_10.setText(QCoreApplication.translate("signalStatsDialog", u"Extension", None))
        self.slowPeak.setSuffix(QCoreApplication.translate("signalStatsDialog", u" dB", None))
        self.label_4.setText(QCoreApplication.translate("signalStatsDialog", u"Slow Peak", None))
        self.label_14.setText(QCoreApplication.translate("signalStatsDialog", u"Limit", None))
        self.label_17.setText(QCoreApplication.translate("signalStatsDialog", u"Rating", None))
        self.label_18.setText(QCoreApplication.translate("signalStatsDialog", u"Rating", None))
        self.includeInstant.setText("")
        self.customPeak.setSuffix(QCoreApplication.translate("signalStatsDialog", u" dB", None))
        self.peakExtensionX1.setSuffix(QCoreApplication.translate("signalStatsDialog", u" Hz", None))
        self.peakExtensionX2.setSuffix(QCoreApplication.translate("signalStatsDialog", u" Hz", None))
        self.peakExtensionY1.setSuffix(QCoreApplication.translate("signalStatsDialog", u" dB", None))
        self.peakExtensionY2.setSuffix(QCoreApplication.translate("signalStatsDialog", u" dB", None))
        self.label_6.setText(QCoreApplication.translate("signalStatsDialog", u"Custom Peak", None))
        self.label_11.setText(QCoreApplication.translate("signalStatsDialog", u"Dynamics", None))
        self.includeFast.setText("")
        self.includeRMS.setText("")
        self.label_19.setText(QCoreApplication.translate("signalStatsDialog", u"Average", None))
        self.label_2.setText(QCoreApplication.translate("signalStatsDialog", u"Instantaneous Peak ", None))
        self.label_15.setText(QCoreApplication.translate("signalStatsDialog", u"Range", None))
        self.averageExtensionX1.setSuffix(QCoreApplication.translate("signalStatsDialog", u" Hz", None))
        self.averageExtensionX2.setSuffix(QCoreApplication.translate("signalStatsDialog", u" Hz", None))
        self.averageExtensionY1.setSuffix(QCoreApplication.translate("signalStatsDialog", u" dB", None))
        self.averageExtensionY2.setSuffix(QCoreApplication.translate("signalStatsDialog", u" dB", None))
        self.includeSlow.setText("")
        self.label_9.setText(QCoreApplication.translate("signalStatsDialog", u"Include?", None))
        self.label_8.setText(QCoreApplication.translate("signalStatsDialog", u"Level", None))
        self.compositeLevel.setSuffix(QCoreApplication.translate("signalStatsDialog", u" dB", None))
        self.label.setText(QCoreApplication.translate("signalStatsDialog", u"RMS:", None))
        self.label_16.setText(QCoreApplication.translate("signalStatsDialog", u"Rating", None))
        self.instantPeak.setSuffix(QCoreApplication.translate("signalStatsDialog", u" dB", None))
        self.label_3.setText(QCoreApplication.translate("signalStatsDialog", u"Fast Peak", None))
        self.extensionLimit.setSuffix(QCoreApplication.translate("signalStatsDialog", u" dB", None))
        self.label_12.setText(QCoreApplication.translate("signalStatsDialog", u"Peak", None))
        self.customIntegrationTime.setSuffix(QCoreApplication.translate("signalStatsDialog", u" ms", None))
        self.label_5.setText(QCoreApplication.translate("signalStatsDialog", u"Custom Integration Time", None))
        self.dynamics.setSuffix(QCoreApplication.translate("signalStatsDialog", u" dB", None))
        self.includeCustom.setText("")
        self.fastPeak.setSuffix(QCoreApplication.translate("signalStatsDialog", u" dB", None))
        self.extensionSource.setItemText(0, "")
        self.extensionSource.setItemText(1, QCoreApplication.translate("signalStatsDialog", u"Peak", None))
        self.extensionSource.setItemText(2, QCoreApplication.translate("signalStatsDialog", u"Average", None))

    # retranslateUi

