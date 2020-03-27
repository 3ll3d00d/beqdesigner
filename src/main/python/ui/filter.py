# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'filter.ui'
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

from mpl import MplWidget


class Ui_editFilterDialog(object):
    def setupUi(self, editFilterDialog):
        if editFilterDialog.objectName():
            editFilterDialog.setObjectName(u"editFilterDialog")
        editFilterDialog.resize(1390, 658)
        self.panes = QGridLayout(editFilterDialog)
        self.panes.setObjectName(u"panes")
        self.viewPane = QGridLayout()
        self.viewPane.setObjectName(u"viewPane")
        self.previewChart = MplWidget(editFilterDialog)
        self.previewChart.setObjectName(u"previewChart")

        self.viewPane.addWidget(self.previewChart, 0, 0, 1, 1)


        self.panes.addLayout(self.viewPane, 0, 1, 1, 1)

        self.paramsPane = QGridLayout()
        self.paramsPane.setObjectName(u"paramsPane")
        self.gainLabel = QLabel(editFilterDialog)
        self.gainLabel.setObjectName(u"gainLabel")

        self.paramsPane.addWidget(self.gainLabel, 8, 0, 1, 1)

        self.filterQLabel = QLabel(editFilterDialog)
        self.filterQLabel.setObjectName(u"filterQLabel")

        self.paramsPane.addWidget(self.filterQLabel, 6, 0, 1, 1)

        self.snapLayout = QHBoxLayout()
        self.snapLayout.setObjectName(u"snapLayout")
        self.snapLabel = QLabel(editFilterDialog)
        self.snapLabel.setObjectName(u"snapLabel")
        font = QFont()
        font.setBold(True)
        font.setWeight(75)
        self.snapLabel.setFont(font)

        self.snapLayout.addWidget(self.snapLabel)

        self.snapFilterButton = QToolButton(editFilterDialog)
        self.snapFilterButton.setObjectName(u"snapFilterButton")

        self.snapLayout.addWidget(self.snapFilterButton)

        self.loadSnapButton = QToolButton(editFilterDialog)
        self.loadSnapButton.setObjectName(u"loadSnapButton")

        self.snapLayout.addWidget(self.loadSnapButton)

        self.acceptSnapButton = QToolButton(editFilterDialog)
        self.acceptSnapButton.setObjectName(u"acceptSnapButton")

        self.snapLayout.addWidget(self.acceptSnapButton)

        self.resetButton = QToolButton(editFilterDialog)
        self.resetButton.setObjectName(u"resetButton")

        self.snapLayout.addWidget(self.resetButton)


        self.paramsPane.addLayout(self.snapLayout, 13, 0, 1, 2)

        self.typeLabel = QLabel(editFilterDialog)
        self.typeLabel.setObjectName(u"typeLabel")

        self.paramsPane.addWidget(self.typeLabel, 2, 0, 1, 1)

        self.sLabel = QLabel(editFilterDialog)
        self.sLabel.setObjectName(u"sLabel")

        self.paramsPane.addWidget(self.sLabel, 7, 0, 1, 1)

        self.buttonLayout = QHBoxLayout()
        self.buttonLayout.setObjectName(u"buttonLayout")
        self.saveButton = QToolButton(editFilterDialog)
        self.saveButton.setObjectName(u"saveButton")

        self.buttonLayout.addWidget(self.saveButton)

        self.exitButton = QToolButton(editFilterDialog)
        self.exitButton.setObjectName(u"exitButton")

        self.buttonLayout.addWidget(self.exitButton)

        self.limitsButton = QToolButton(editFilterDialog)
        self.limitsButton.setObjectName(u"limitsButton")

        self.buttonLayout.addWidget(self.limitsButton)


        self.paramsPane.addLayout(self.buttonLayout, 11, 0, 1, 2)

        self.passFilterType = QComboBox(editFilterDialog)
        self.passFilterType.addItem("")
        self.passFilterType.addItem("")
        self.passFilterType.setObjectName(u"passFilterType")
        self.passFilterType.setEnabled(True)

        self.paramsPane.addWidget(self.passFilterType, 3, 1, 1, 1)

        self.snapshotFilterView = QTableView(editFilterDialog)
        self.snapshotFilterView.setObjectName(u"snapshotFilterView")
        self.snapshotFilterView.setSelectionMode(QAbstractItemView.SingleSelection)
        self.snapshotFilterView.setSelectionBehavior(QAbstractItemView.SelectRows)

        self.paramsPane.addWidget(self.snapshotFilterView, 15, 0, 1, 2)

        self.filterCount = QSpinBox(editFilterDialog)
        self.filterCount.setObjectName(u"filterCount")
        self.filterCount.setMinimum(1)
        self.filterCount.setMaximum(20)

        self.paramsPane.addWidget(self.filterCount, 9, 1, 1, 1)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.paramsPane.addItem(self.verticalSpacer_2, 12, 1, 1, 1)

        self.workingFilterView = QTableView(editFilterDialog)
        self.workingFilterView.setObjectName(u"workingFilterView")
        self.workingFilterView.setSelectionMode(QAbstractItemView.SingleSelection)
        self.workingFilterView.setSelectionBehavior(QAbstractItemView.SelectRows)

        self.paramsPane.addWidget(self.workingFilterView, 0, 0, 1, 2)

        self.filterType = QComboBox(editFilterDialog)
        self.filterType.addItem("")
        self.filterType.addItem("")
        self.filterType.addItem("")
        self.filterType.addItem("")
        self.filterType.addItem("")
        self.filterType.addItem("")
        self.filterType.addItem("")
        self.filterType.addItem("")
        self.filterType.setObjectName(u"filterType")

        self.paramsPane.addWidget(self.filterType, 2, 1, 1, 1)

        self.showIndividual = QCheckBox(editFilterDialog)
        self.showIndividual.setObjectName(u"showIndividual")
        self.showIndividual.setChecked(True)

        self.paramsPane.addWidget(self.showIndividual, 10, 0, 1, 2)

        self.qStepButton = QToolButton(editFilterDialog)
        self.qStepButton.setObjectName(u"qStepButton")

        self.paramsPane.addWidget(self.qStepButton, 6, 2, 1, 1)

        self.filterS = QDoubleSpinBox(editFilterDialog)
        self.filterS.setObjectName(u"filterS")
        self.filterS.setEnabled(False)
        self.filterS.setDecimals(4)
        self.filterS.setMinimum(0.100000000000000)
        self.filterS.setMaximum(100.000000000000000)
        self.filterS.setSingleStep(0.000100000000000)
        self.filterS.setValue(1.000000000000000)

        self.paramsPane.addWidget(self.filterS, 7, 1, 1, 1)

        self.freqStepButton = QToolButton(editFilterDialog)
        self.freqStepButton.setObjectName(u"freqStepButton")

        self.paramsPane.addWidget(self.freqStepButton, 5, 2, 1, 1)

        self.freqLabel = QLabel(editFilterDialog)
        self.freqLabel.setObjectName(u"freqLabel")

        self.paramsPane.addWidget(self.freqLabel, 5, 0, 1, 1)

        self.freq = QDoubleSpinBox(editFilterDialog)
        self.freq.setObjectName(u"freq")
        self.freq.setDecimals(1)
        self.freq.setMinimum(1.000000000000000)
        self.freq.setMaximum(500.000000000000000)
        self.freq.setSingleStep(0.100000000000000)
        self.freq.setValue(40.000000000000000)

        self.paramsPane.addWidget(self.freq, 5, 1, 1, 1)

        self.filterCountLabel = QLabel(editFilterDialog)
        self.filterCountLabel.setObjectName(u"filterCountLabel")

        self.paramsPane.addWidget(self.filterCountLabel, 9, 0, 1, 1)

        self.snapshotViewButtonWidget = QWidget(editFilterDialog)
        self.snapshotViewButtonWidget.setObjectName(u"snapshotViewButtonWidget")
        self.snapshotViewButtonLayout = QVBoxLayout(self.snapshotViewButtonWidget)
        self.snapshotViewButtonLayout.setObjectName(u"snapshotViewButtonLayout")
        self.addSnapshotRowButton = QToolButton(self.snapshotViewButtonWidget)
        self.addSnapshotRowButton.setObjectName(u"addSnapshotRowButton")

        self.snapshotViewButtonLayout.addWidget(self.addSnapshotRowButton)

        self.removeSnapshotRowButton = QToolButton(self.snapshotViewButtonWidget)
        self.removeSnapshotRowButton.setObjectName(u"removeSnapshotRowButton")

        self.snapshotViewButtonLayout.addWidget(self.removeSnapshotRowButton)


        self.paramsPane.addWidget(self.snapshotViewButtonWidget, 15, 2, 1, 1)

        self.filterQ = QDoubleSpinBox(editFilterDialog)
        self.filterQ.setObjectName(u"filterQ")
        self.filterQ.setDecimals(4)
        self.filterQ.setMinimum(0.001000000000000)
        self.filterQ.setMaximum(20.000000000000000)
        self.filterQ.setSingleStep(0.000100000000000)
        self.filterQ.setValue(0.707100000000000)

        self.paramsPane.addWidget(self.filterQ, 6, 1, 1, 1)

        self.filterGain = QDoubleSpinBox(editFilterDialog)
        self.filterGain.setObjectName(u"filterGain")
        self.filterGain.setDecimals(1)
        self.filterGain.setMinimum(-30.000000000000000)
        self.filterGain.setMaximum(30.000000000000000)
        self.filterGain.setSingleStep(0.100000000000000)

        self.paramsPane.addWidget(self.filterGain, 8, 1, 1, 1)

        self.gainStepButton = QToolButton(editFilterDialog)
        self.gainStepButton.setObjectName(u"gainStepButton")

        self.paramsPane.addWidget(self.gainStepButton, 8, 2, 1, 1)

        self.orderLabel = QLabel(editFilterDialog)
        self.orderLabel.setObjectName(u"orderLabel")

        self.paramsPane.addWidget(self.orderLabel, 4, 0, 1, 1)

        self.sStepButton = QToolButton(editFilterDialog)
        self.sStepButton.setObjectName(u"sStepButton")

        self.paramsPane.addWidget(self.sStepButton, 7, 2, 1, 1)

        self.optimiseLayout = QHBoxLayout()
        self.optimiseLayout.setObjectName(u"optimiseLayout")
        self.label_2 = QLabel(editFilterDialog)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setFont(font)

        self.optimiseLayout.addWidget(self.label_2)

        self.optimiseButton = QToolButton(editFilterDialog)
        self.optimiseButton.setObjectName(u"optimiseButton")

        self.optimiseLayout.addWidget(self.optimiseButton)

        self.targetBiquadCount = QSpinBox(editFilterDialog)
        self.targetBiquadCount.setObjectName(u"targetBiquadCount")
        self.targetBiquadCount.setMinimum(1)
        self.targetBiquadCount.setMaximum(20)
        self.targetBiquadCount.setValue(6)

        self.optimiseLayout.addWidget(self.targetBiquadCount)


        self.paramsPane.addLayout(self.optimiseLayout, 14, 0, 1, 2)

        self.filterOrder = QSpinBox(editFilterDialog)
        self.filterOrder.setObjectName(u"filterOrder")
        self.filterOrder.setEnabled(True)
        self.filterOrder.setMinimum(1)
        self.filterOrder.setMaximum(24)
        self.filterOrder.setValue(2)

        self.paramsPane.addWidget(self.filterOrder, 4, 1, 1, 1)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.paramsPane.addItem(self.verticalSpacer, 16, 1, 1, 1)

        self.workingViewButtonWidget = QWidget(editFilterDialog)
        self.workingViewButtonWidget.setObjectName(u"workingViewButtonWidget")
        self.workingViewButtonLayout = QVBoxLayout(self.workingViewButtonWidget)
        self.workingViewButtonLayout.setObjectName(u"workingViewButtonLayout")
        self.addWorkingRowButton = QToolButton(self.workingViewButtonWidget)
        self.addWorkingRowButton.setObjectName(u"addWorkingRowButton")

        self.workingViewButtonLayout.addWidget(self.addWorkingRowButton)

        self.removeWorkingRowButton = QToolButton(self.workingViewButtonWidget)
        self.removeWorkingRowButton.setObjectName(u"removeWorkingRowButton")

        self.workingViewButtonLayout.addWidget(self.removeWorkingRowButton)


        self.paramsPane.addWidget(self.workingViewButtonWidget, 0, 2, 1, 1)

        self.headerLabel = QLabel(editFilterDialog)
        self.headerLabel.setObjectName(u"headerLabel")
        self.headerLabel.setFont(font)
        self.headerLabel.setFrameShape(QFrame.Box)
        self.headerLabel.setFrameShadow(QFrame.Sunken)
        self.headerLabel.setAlignment(Qt.AlignCenter)

        self.paramsPane.addWidget(self.headerLabel, 1, 0, 1, 2)

        self.paramsPane.setColumnStretch(0, 1)
        self.paramsPane.setColumnStretch(1, 4)

        self.panes.addLayout(self.paramsPane, 0, 0, 1, 1)

        self.panes.setColumnStretch(0, 1)
        self.panes.setColumnStretch(1, 3)
        QWidget.setTabOrder(self.filterType, self.passFilterType)
        QWidget.setTabOrder(self.passFilterType, self.filterOrder)
        QWidget.setTabOrder(self.filterOrder, self.freq)
        QWidget.setTabOrder(self.freq, self.filterQ)
        QWidget.setTabOrder(self.filterQ, self.filterS)
        QWidget.setTabOrder(self.filterS, self.filterGain)
        QWidget.setTabOrder(self.filterGain, self.filterCount)
        QWidget.setTabOrder(self.filterCount, self.freqStepButton)
        QWidget.setTabOrder(self.freqStepButton, self.qStepButton)
        QWidget.setTabOrder(self.qStepButton, self.gainStepButton)
        QWidget.setTabOrder(self.gainStepButton, self.sStepButton)
        QWidget.setTabOrder(self.sStepButton, self.previewChart)

        self.retranslateUi(editFilterDialog)
        self.filterType.currentTextChanged.connect(editFilterDialog.enableFilterParams)
        self.passFilterType.currentTextChanged.connect(editFilterDialog.changeOrderStep)
        self.filterQ.valueChanged.connect(editFilterDialog.recalcShelfFromQ)
        self.filterGain.valueChanged.connect(editFilterDialog.recalcShelfFromGain)
        self.filterType.currentIndexChanged.connect(editFilterDialog.previewFilter)
        self.passFilterType.currentIndexChanged.connect(editFilterDialog.previewFilter)
        self.filterOrder.valueChanged.connect(editFilterDialog.previewFilter)
        self.freq.valueChanged.connect(editFilterDialog.previewFilter)
        self.filterQ.valueChanged.connect(editFilterDialog.previewFilter)
        self.filterGain.valueChanged.connect(editFilterDialog.previewFilter)
        self.filterCount.valueChanged.connect(editFilterDialog.previewFilter)
        self.filterS.valueChanged.connect(editFilterDialog.recalcShelfFromS)
        self.sStepButton.clicked.connect(editFilterDialog.handleSToolButton)
        self.qStepButton.clicked.connect(editFilterDialog.handleQToolButton)
        self.gainStepButton.clicked.connect(editFilterDialog.handleGainToolButton)
        self.freqStepButton.clicked.connect(editFilterDialog.handleFreqToolButton)
        self.saveButton.clicked.connect(editFilterDialog.accept)
        self.exitButton.clicked.connect(editFilterDialog.reject)
        self.showIndividual.clicked.connect(editFilterDialog.previewFilter)
        self.limitsButton.clicked.connect(editFilterDialog.show_limits)

        QMetaObject.connectSlotsByName(editFilterDialog)
    # setupUi

    def retranslateUi(self, editFilterDialog):
        editFilterDialog.setWindowTitle(QCoreApplication.translate("editFilterDialog", u"Create Filter", None))
        self.gainLabel.setText(QCoreApplication.translate("editFilterDialog", u"Gain", None))
        self.filterQLabel.setText(QCoreApplication.translate("editFilterDialog", u"Q", None))
        self.snapLabel.setText(QCoreApplication.translate("editFilterDialog", u"Compare", None))
        self.snapFilterButton.setText(QCoreApplication.translate("editFilterDialog", u"...", None))
        self.loadSnapButton.setText(QCoreApplication.translate("editFilterDialog", u"...", None))
        self.acceptSnapButton.setText(QCoreApplication.translate("editFilterDialog", u"...", None))
        self.resetButton.setText(QCoreApplication.translate("editFilterDialog", u"...", None))
        self.typeLabel.setText(QCoreApplication.translate("editFilterDialog", u"Type", None))
        self.sLabel.setText(QCoreApplication.translate("editFilterDialog", u"S", None))
        self.saveButton.setText("")
#if QT_CONFIG(shortcut)
        self.saveButton.setShortcut(QCoreApplication.translate("editFilterDialog", u"Return", None))
#endif // QT_CONFIG(shortcut)
        self.exitButton.setText(QCoreApplication.translate("editFilterDialog", u"...", None))
        self.limitsButton.setText(QCoreApplication.translate("editFilterDialog", u"...", None))
        self.passFilterType.setItemText(0, QCoreApplication.translate("editFilterDialog", u"Butterworth", None))
        self.passFilterType.setItemText(1, QCoreApplication.translate("editFilterDialog", u"Linkwitz-Riley", None))

        self.filterType.setItemText(0, QCoreApplication.translate("editFilterDialog", u"Low Shelf", None))
        self.filterType.setItemText(1, QCoreApplication.translate("editFilterDialog", u"High Shelf", None))
        self.filterType.setItemText(2, QCoreApplication.translate("editFilterDialog", u"PEQ", None))
        self.filterType.setItemText(3, QCoreApplication.translate("editFilterDialog", u"Gain", None))
        self.filterType.setItemText(4, QCoreApplication.translate("editFilterDialog", u"Variable Q LPF", None))
        self.filterType.setItemText(5, QCoreApplication.translate("editFilterDialog", u"Variable Q HPF", None))
        self.filterType.setItemText(6, QCoreApplication.translate("editFilterDialog", u"Low Pass", None))
        self.filterType.setItemText(7, QCoreApplication.translate("editFilterDialog", u"High Pass", None))

        self.showIndividual.setText(QCoreApplication.translate("editFilterDialog", u"Show Individual Filters", None))
        self.qStepButton.setText(QCoreApplication.translate("editFilterDialog", u"...", None))
        self.freqStepButton.setText(QCoreApplication.translate("editFilterDialog", u"...", None))
        self.freqLabel.setText(QCoreApplication.translate("editFilterDialog", u"Freq", None))
        self.filterCountLabel.setText(QCoreApplication.translate("editFilterDialog", u"Count", None))
        self.addSnapshotRowButton.setText(QCoreApplication.translate("editFilterDialog", u"...", None))
        self.removeSnapshotRowButton.setText(QCoreApplication.translate("editFilterDialog", u"...", None))
        self.gainStepButton.setText(QCoreApplication.translate("editFilterDialog", u"...", None))
        self.orderLabel.setText(QCoreApplication.translate("editFilterDialog", u"Order", None))
        self.sStepButton.setText(QCoreApplication.translate("editFilterDialog", u"...", None))
        self.label_2.setText(QCoreApplication.translate("editFilterDialog", u"Optimise", None))
        self.optimiseButton.setText(QCoreApplication.translate("editFilterDialog", u"...", None))
        self.addWorkingRowButton.setText(QCoreApplication.translate("editFilterDialog", u"...", None))
        self.removeWorkingRowButton.setText(QCoreApplication.translate("editFilterDialog", u"...", None))
        self.headerLabel.setText("")
    # retranslateUi

