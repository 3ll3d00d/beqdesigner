# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'report.ui'
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


class Ui_saveReportDialog(object):
    def setupUi(self, saveReportDialog):
        if saveReportDialog.objectName():
            saveReportDialog.setObjectName(u"saveReportDialog")
        saveReportDialog.resize(1530, 700)
        self.gridLayout_2 = QGridLayout(saveReportDialog)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.toolboxContainer = QVBoxLayout()
        self.toolboxContainer.setObjectName(u"toolboxContainer")
        self.toolBox = QToolBox(saveReportDialog)
        self.toolBox.setObjectName(u"toolBox")
        font = QFont()
        font.setBold(False)
        font.setWeight(50)
        self.toolBox.setFont(font)
        self.toolBox.setFrameShape(QFrame.NoFrame)
        self.toolBox.setFrameShadow(QFrame.Plain)
        self.page = QWidget()
        self.page.setObjectName(u"page")
        self.page.setGeometry(QRect(0, 0, 368, 342))
        self.gridLayout_4 = QGridLayout(self.page)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.gridLayout_3 = QGridLayout()
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.imageWidthPixels = QSpinBox(self.page)
        self.imageWidthPixels.setObjectName(u"imageWidthPixels")
        self.imageWidthPixels.setEnabled(False)
        self.imageWidthPixels.setMinimum(0)
        self.imageWidthPixels.setMaximum(999999)

        self.horizontalLayout_6.addWidget(self.imageWidthPixels)

        self.label_3 = QLabel(self.page)
        self.label_3.setObjectName(u"label_3")

        self.horizontalLayout_6.addWidget(self.label_3)

        self.imageHeightPixels = QSpinBox(self.page)
        self.imageHeightPixels.setObjectName(u"imageHeightPixels")
        self.imageHeightPixels.setEnabled(False)
        self.imageHeightPixels.setMinimum(0)
        self.imageHeightPixels.setMaximum(999999)

        self.horizontalLayout_6.addWidget(self.imageHeightPixels)

        self.horizontalLayout_6.setStretch(0, 1)
        self.horizontalLayout_6.setStretch(2, 1)

        self.gridLayout_3.addLayout(self.horizontalLayout_6, 6, 1, 1, 1)

        self.imageFileLabel = QLabel(self.page)
        self.imageFileLabel.setObjectName(u"imageFileLabel")

        self.gridLayout_3.addWidget(self.imageFileLabel, 0, 0, 1, 1)

        self.imagePicker = QToolButton(self.page)
        self.imagePicker.setObjectName(u"imagePicker")

        self.gridLayout_3.addWidget(self.imagePicker, 0, 2, 1, 1)

        self.image = QLineEdit(self.page)
        self.image.setObjectName(u"image")
        self.image.setEnabled(True)
        self.image.setReadOnly(True)

        self.gridLayout_3.addWidget(self.image, 0, 1, 1, 1)

        self.titleLabel = QLabel(self.page)
        self.titleLabel.setObjectName(u"titleLabel")

        self.gridLayout_3.addWidget(self.titleLabel, 4, 0, 1, 1)

        self.imageURLLabel = QLabel(self.page)
        self.imageURLLabel.setObjectName(u"imageURLLabel")

        self.gridLayout_3.addWidget(self.imageURLLabel, 1, 0, 1, 1)

        self.imageURL = QLineEdit(self.page)
        self.imageURL.setObjectName(u"imageURL")

        self.gridLayout_3.addWidget(self.imageURL, 1, 1, 1, 1)

        self.label_4 = QLabel(self.page)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout_3.addWidget(self.label_4, 2, 0, 1, 1)

        self.titleFontSizeLabel = QLabel(self.page)
        self.titleFontSizeLabel.setObjectName(u"titleFontSizeLabel")

        self.gridLayout_3.addWidget(self.titleFontSizeLabel, 5, 0, 1, 1)

        self.imageAlphaLabel = QLabel(self.page)
        self.imageAlphaLabel.setObjectName(u"imageAlphaLabel")

        self.gridLayout_3.addWidget(self.imageAlphaLabel, 3, 0, 1, 1)

        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.nativeImageWidth = QSpinBox(self.page)
        self.nativeImageWidth.setObjectName(u"nativeImageWidth")
        self.nativeImageWidth.setEnabled(False)
        self.nativeImageWidth.setMaximum(99999)

        self.horizontalLayout_8.addWidget(self.nativeImageWidth)

        self.label_5 = QLabel(self.page)
        self.label_5.setObjectName(u"label_5")

        self.horizontalLayout_8.addWidget(self.label_5)

        self.nativeImageHeight = QSpinBox(self.page)
        self.nativeImageHeight.setObjectName(u"nativeImageHeight")
        self.nativeImageHeight.setEnabled(False)
        self.nativeImageHeight.setMaximum(99999)

        self.horizontalLayout_8.addWidget(self.nativeImageHeight)

        self.horizontalLayout_8.setStretch(0, 1)
        self.horizontalLayout_8.setStretch(2, 1)

        self.gridLayout_3.addLayout(self.horizontalLayout_8, 2, 1, 1, 1)

        self.imageOpacity = QDoubleSpinBox(self.page)
        self.imageOpacity.setObjectName(u"imageOpacity")
        self.imageOpacity.setDecimals(2)
        self.imageOpacity.setMinimum(0.010000000000000)
        self.imageOpacity.setMaximum(1.000000000000000)
        self.imageOpacity.setSingleStep(0.010000000000000)
        self.imageOpacity.setValue(1.000000000000000)

        self.gridLayout_3.addWidget(self.imageOpacity, 3, 1, 1, 1)

        self.loadURL = QToolButton(self.page)
        self.loadURL.setObjectName(u"loadURL")

        self.gridLayout_3.addWidget(self.loadURL, 1, 2, 1, 1)

        self.titleFontSize = QSpinBox(self.page)
        self.titleFontSize.setObjectName(u"titleFontSize")

        self.gridLayout_3.addWidget(self.titleFontSize, 5, 1, 1, 1)

        self.widthLabel = QLabel(self.page)
        self.widthLabel.setObjectName(u"widthLabel")

        self.gridLayout_3.addWidget(self.widthLabel, 6, 0, 1, 1)

        self.title = QLineEdit(self.page)
        self.title.setObjectName(u"title")

        self.gridLayout_3.addWidget(self.title, 4, 1, 1, 1)

        self.imageBorder = QCheckBox(self.page)
        self.imageBorder.setObjectName(u"imageBorder")

        self.gridLayout_3.addWidget(self.imageBorder, 7, 1, 1, 1)


        self.gridLayout_4.addLayout(self.gridLayout_3, 0, 0, 1, 1)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_4.addItem(self.verticalSpacer, 1, 0, 1, 1)

        self.toolBox.addItem(self.page, u"Image")
        self.page_2 = QWidget()
        self.page_2.setObjectName(u"page_2")
        self.page_2.setGeometry(QRect(0, 0, 368, 310))
        self.gridLayout_6 = QGridLayout(self.page_2)
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.gridLayout_5 = QGridLayout()
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.tableAlphaLabel = QLabel(self.page_2)
        self.tableAlphaLabel.setObjectName(u"tableAlphaLabel")

        self.gridLayout_5.addWidget(self.tableAlphaLabel, 5, 0, 1, 1)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.x0 = QDoubleSpinBox(self.page_2)
        self.x0.setObjectName(u"x0")
        self.x0.setDecimals(3)
        self.x0.setMaximum(0.999000000000000)
        self.x0.setSingleStep(0.001000000000000)
        self.x0.setValue(0.748000000000000)

        self.horizontalLayout_3.addWidget(self.x0)

        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_5)

        self.x1 = QDoubleSpinBox(self.page_2)
        self.x1.setObjectName(u"x1")
        self.x1.setDecimals(3)
        self.x1.setMaximum(1.000000000000000)
        self.x1.setSingleStep(0.001000000000000)
        self.x1.setValue(1.000000000000000)

        self.horizontalLayout_3.addWidget(self.x1)


        self.gridLayout_5.addLayout(self.horizontalLayout_3, 2, 1, 1, 1)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer_4)

        self.y0 = QDoubleSpinBox(self.page_2)
        self.y0.setObjectName(u"y0")
        self.y0.setDecimals(3)
        self.y0.setMaximum(0.999000000000000)
        self.y0.setSingleStep(0.001000000000000)
        self.y0.setValue(0.750000000000000)

        self.horizontalLayout_4.addWidget(self.y0)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer_3)


        self.gridLayout_5.addLayout(self.horizontalLayout_4, 3, 1, 1, 1)

        self.tableRowHeightLabel = QLabel(self.page_2)
        self.tableRowHeightLabel.setObjectName(u"tableRowHeightLabel")

        self.gridLayout_5.addWidget(self.tableRowHeightLabel, 0, 0, 1, 1)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_2)

        self.y1 = QDoubleSpinBox(self.page_2)
        self.y1.setObjectName(u"y1")
        self.y1.setDecimals(3)
        self.y1.setMaximum(1.000000000000000)
        self.y1.setSingleStep(0.001000000000000)
        self.y1.setValue(1.000000000000000)

        self.horizontalLayout_2.addWidget(self.y1)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer)


        self.gridLayout_5.addLayout(self.horizontalLayout_2, 1, 1, 1, 1)

        self.filterRowHeightMultiplier = QDoubleSpinBox(self.page_2)
        self.filterRowHeightMultiplier.setObjectName(u"filterRowHeightMultiplier")
        self.filterRowHeightMultiplier.setMaximum(3.000000000000000)
        self.filterRowHeightMultiplier.setSingleStep(0.010000000000000)
        self.filterRowHeightMultiplier.setValue(1.200000000000000)

        self.gridLayout_5.addWidget(self.filterRowHeightMultiplier, 0, 1, 1, 1)

        self.tableFontSizeLabel = QLabel(self.page_2)
        self.tableFontSizeLabel.setObjectName(u"tableFontSizeLabel")

        self.gridLayout_5.addWidget(self.tableFontSizeLabel, 6, 0, 1, 1)

        self.showTableHeader = QCheckBox(self.page_2)
        self.showTableHeader.setObjectName(u"showTableHeader")

        self.gridLayout_5.addWidget(self.showTableHeader, 4, 1, 1, 1)

        self.tablePositionLabel = QLabel(self.page_2)
        self.tablePositionLabel.setObjectName(u"tablePositionLabel")

        self.gridLayout_5.addWidget(self.tablePositionLabel, 1, 0, 3, 1)

        self.tableAlpha = QDoubleSpinBox(self.page_2)
        self.tableAlpha.setObjectName(u"tableAlpha")
        self.tableAlpha.setMinimum(0.010000000000000)
        self.tableAlpha.setMaximum(1.000000000000000)
        self.tableAlpha.setSingleStep(0.010000000000000)
        self.tableAlpha.setValue(1.000000000000000)

        self.gridLayout_5.addWidget(self.tableAlpha, 5, 1, 1, 1)

        self.tableFontSize = QSpinBox(self.page_2)
        self.tableFontSize.setObjectName(u"tableFontSize")
        self.tableFontSize.setMaximum(24)
        self.tableFontSize.setValue(10)

        self.gridLayout_5.addWidget(self.tableFontSize, 6, 1, 1, 1)


        self.gridLayout_6.addLayout(self.gridLayout_5, 0, 0, 1, 1)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_6.addItem(self.verticalSpacer_2, 1, 0, 1, 1)

        self.toolBox.addItem(self.page_2, u"Filters")
        self.page_3 = QWidget()
        self.page_3.setObjectName(u"page_3")
        self.page_3.setGeometry(QRect(0, 0, 410, 308))
        self.gridLayout_8 = QGridLayout(self.page_3)
        self.gridLayout_8.setObjectName(u"gridLayout_8")
        self.gridLayout_7 = QGridLayout()
        self.gridLayout_7.setObjectName(u"gridLayout_7")
        self.label_6 = QLabel(self.page_3)
        self.label_6.setObjectName(u"label_6")

        self.gridLayout_7.addWidget(self.label_6, 5, 0, 1, 1)

        self.majorSplitRatioLabel = QLabel(self.page_3)
        self.majorSplitRatioLabel.setObjectName(u"majorSplitRatioLabel")

        self.gridLayout_7.addWidget(self.majorSplitRatioLabel, 0, 0, 1, 1)

        self.label_7 = QLabel(self.page_3)
        self.label_7.setObjectName(u"label_7")

        self.gridLayout_7.addWidget(self.label_7, 3, 0, 1, 1)

        self.heightSpacing = QDoubleSpinBox(self.page_3)
        self.heightSpacing.setObjectName(u"heightSpacing")
        self.heightSpacing.setMaximum(1.000000000000000)
        self.heightSpacing.setSingleStep(0.010000000000000)

        self.gridLayout_7.addWidget(self.heightSpacing, 6, 1, 1, 1)

        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_7.addItem(self.verticalSpacer_3, 7, 0, 1, 3)

        self.majorSplitRatio = QDoubleSpinBox(self.page_3)
        self.majorSplitRatio.setObjectName(u"majorSplitRatio")
        self.majorSplitRatio.setDecimals(3)
        self.majorSplitRatio.setMinimum(0.100000000000000)
        self.majorSplitRatio.setMaximum(10.000000000000000)
        self.majorSplitRatio.setSingleStep(0.001000000000000)
        self.majorSplitRatio.setValue(1.000000000000000)

        self.gridLayout_7.addWidget(self.majorSplitRatio, 0, 1, 1, 1)

        self.widthSpacing = QDoubleSpinBox(self.page_3)
        self.widthSpacing.setObjectName(u"widthSpacing")
        self.widthSpacing.setMaximum(1.000000000000000)
        self.widthSpacing.setSingleStep(0.010000000000000)

        self.gridLayout_7.addWidget(self.widthSpacing, 5, 1, 1, 1)

        self.label_8 = QLabel(self.page_3)
        self.label_8.setObjectName(u"label_8")

        self.gridLayout_7.addWidget(self.label_8, 6, 0, 1, 1)

        self.label = QLabel(self.page_3)
        self.label.setObjectName(u"label")

        self.gridLayout_7.addWidget(self.label, 4, 0, 1, 1)

        self.chartSplitLabel = QLabel(self.page_3)
        self.chartSplitLabel.setObjectName(u"chartSplitLabel")

        self.gridLayout_7.addWidget(self.chartSplitLabel, 2, 0, 1, 1)

        self.chartLayout = QComboBox(self.page_3)
        self.chartLayout.addItem("")
        self.chartLayout.addItem("")
        self.chartLayout.addItem("")
        self.chartLayout.addItem("")
        self.chartLayout.addItem("")
        self.chartLayout.addItem("")
        self.chartLayout.addItem("")
        self.chartLayout.addItem("")
        self.chartLayout.addItem("")
        self.chartLayout.addItem("")
        self.chartLayout.addItem("")
        self.chartLayout.addItem("")
        self.chartLayout.addItem("")
        self.chartLayout.addItem("")
        self.chartLayout.addItem("")
        self.chartLayout.addItem("")
        self.chartLayout.addItem("")
        self.chartLayout.setObjectName(u"chartLayout")

        self.gridLayout_7.addWidget(self.chartLayout, 3, 1, 1, 1)

        self.chartSplit = QComboBox(self.page_3)
        self.chartSplit.addItem("")
        self.chartSplit.addItem("")
        self.chartSplit.setObjectName(u"chartSplit")

        self.gridLayout_7.addWidget(self.chartSplit, 2, 1, 1, 1)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.widthPixels = QSpinBox(self.page_3)
        self.widthPixels.setObjectName(u"widthPixels")
        self.widthPixels.setMinimum(512)
        self.widthPixels.setMaximum(8192)

        self.horizontalLayout_7.addWidget(self.widthPixels)

        self.label_2 = QLabel(self.page_3)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout_7.addWidget(self.label_2)

        self.heightPixels = QSpinBox(self.page_3)
        self.heightPixels.setObjectName(u"heightPixels")
        self.heightPixels.setMinimum(512)
        self.heightPixels.setMaximum(8192)

        self.horizontalLayout_7.addWidget(self.heightPixels)

        self.horizontalLayout_7.setStretch(0, 1)
        self.horizontalLayout_7.setStretch(2, 1)

        self.gridLayout_7.addLayout(self.horizontalLayout_7, 4, 1, 1, 1)

        self.minorSplitRatio = QDoubleSpinBox(self.page_3)
        self.minorSplitRatio.setObjectName(u"minorSplitRatio")
        self.minorSplitRatio.setDecimals(3)
        self.minorSplitRatio.setMinimum(0.001000000000000)
        self.minorSplitRatio.setMaximum(10.000000000000000)
        self.minorSplitRatio.setSingleStep(0.001000000000000)
        self.minorSplitRatio.setValue(3.000000000000000)

        self.gridLayout_7.addWidget(self.minorSplitRatio, 1, 1, 1, 1)

        self.label_9 = QLabel(self.page_3)
        self.label_9.setObjectName(u"label_9")

        self.gridLayout_7.addWidget(self.label_9, 1, 0, 1, 1)

        self.snapToImageSize = QToolButton(self.page_3)
        self.snapToImageSize.setObjectName(u"snapToImageSize")

        self.gridLayout_7.addWidget(self.snapToImageSize, 5, 2, 1, 1)


        self.gridLayout_8.addLayout(self.gridLayout_7, 0, 0, 1, 1)

        self.toolBox.addItem(self.page_3, u"Layout")
        self.page_4 = QWidget()
        self.page_4.setObjectName(u"page_4")
        self.page_4.setGeometry(QRect(0, 0, 410, 329))
        self.gridLayout_10 = QGridLayout(self.page_4)
        self.gridLayout_10.setObjectName(u"gridLayout_10")
        self.gridLayout_9 = QGridLayout()
        self.gridLayout_9.setObjectName(u"gridLayout_9")
        self.curves = QListWidget(self.page_4)
        self.curves.setObjectName(u"curves")
        self.curves.setSelectionMode(QAbstractItemView.MultiSelection)

        self.gridLayout_9.addWidget(self.curves, 1, 1, 1, 1)

        self.limitsButton = QToolButton(self.page_4)
        self.limitsButton.setObjectName(u"limitsButton")

        self.gridLayout_9.addWidget(self.limitsButton, 0, 2, 1, 1)

        self.curvesLabel = QLabel(self.page_4)
        self.curvesLabel.setObjectName(u"curvesLabel")

        self.gridLayout_9.addWidget(self.curvesLabel, 1, 0, 1, 1)

        self.gridAlphaLabel = QLabel(self.page_4)
        self.gridAlphaLabel.setObjectName(u"gridAlphaLabel")

        self.gridLayout_9.addWidget(self.gridAlphaLabel, 0, 0, 1, 1)

        self.gridOpacity = QDoubleSpinBox(self.page_4)
        self.gridOpacity.setObjectName(u"gridOpacity")
        self.gridOpacity.setMaximum(1.000000000000000)
        self.gridOpacity.setSingleStep(0.010000000000000)
        self.gridOpacity.setValue(0.400000000000000)

        self.gridLayout_9.addWidget(self.gridOpacity, 0, 1, 1, 1)

        self.saveLayout = QPushButton(self.page_4)
        self.saveLayout.setObjectName(u"saveLayout")

        self.gridLayout_9.addWidget(self.saveLayout, 3, 1, 1, 1)

        self.showLegend = QCheckBox(self.page_4)
        self.showLegend.setObjectName(u"showLegend")

        self.gridLayout_9.addWidget(self.showLegend, 2, 1, 1, 1)


        self.gridLayout_10.addLayout(self.gridLayout_9, 0, 0, 1, 1)

        self.toolBox.addItem(self.page_4, u"Chart")

        self.toolboxContainer.addWidget(self.toolBox)

        self.verticalSpacer_4 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.toolboxContainer.addItem(self.verticalSpacer_4)

        self.buttonBox = QDialogButtonBox(saveReportDialog)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Close|QDialogButtonBox.RestoreDefaults|QDialogButtonBox.Save)

        self.toolboxContainer.addWidget(self.buttonBox)


        self.horizontalLayout.addLayout(self.toolboxContainer)

        self.preview = MplWidget(saveReportDialog)
        self.preview.setObjectName(u"preview")

        self.horizontalLayout.addWidget(self.preview)

        self.horizontalLayout.setStretch(1, 1)

        self.gridLayout_2.addLayout(self.horizontalLayout, 0, 1, 1, 1)

        QWidget.setTabOrder(self.x0, self.y0)
        QWidget.setTabOrder(self.y0, self.x1)
        QWidget.setTabOrder(self.x1, self.y1)
        QWidget.setTabOrder(self.y1, self.preview)

        self.retranslateUi(saveReportDialog)
        self.buttonBox.accepted.connect(saveReportDialog.accept)
        self.buttonBox.rejected.connect(saveReportDialog.reject)
        self.curves.itemSelectionChanged.connect(saveReportDialog.set_selected)
        self.widthPixels.valueChanged.connect(saveReportDialog.update_height)
        self.showLegend.clicked.connect(saveReportDialog.redraw)
        self.imagePicker.clicked.connect(saveReportDialog.choose_image)
        self.title.textChanged.connect(saveReportDialog.set_title)
        self.majorSplitRatio.valueChanged.connect(saveReportDialog.redraw_all_axes)
        self.chartSplit.currentIndexChanged.connect(saveReportDialog.redraw_all_axes)
        self.limitsButton.clicked.connect(saveReportDialog.show_limits)
        self.chartSplit.currentIndexChanged.connect(saveReportDialog.redraw_all_axes)
        self.chartLayout.currentIndexChanged.connect(saveReportDialog.redraw_all_axes)
        self.minorSplitRatio.valueChanged.connect(saveReportDialog.redraw_all_axes)
        self.gridOpacity.valueChanged.connect(saveReportDialog.set_grid_opacity)
        self.imageOpacity.valueChanged.connect(saveReportDialog.set_image_opacity)
        self.filterRowHeightMultiplier.valueChanged.connect(saveReportDialog.replace_table)
        self.y0.valueChanged.connect(saveReportDialog.replace_table)
        self.x0.valueChanged.connect(saveReportDialog.replace_table)
        self.y1.valueChanged.connect(saveReportDialog.replace_table)
        self.x1.valueChanged.connect(saveReportDialog.replace_table)
        self.saveLayout.clicked.connect(saveReportDialog.save_layout)
        self.imageURL.textChanged.connect(saveReportDialog.update_image_url)
        self.loadURL.clicked.connect(saveReportDialog.load_image_from_url)
        self.tableAlpha.valueChanged.connect(saveReportDialog.replace_table)
        self.tableFontSize.valueChanged.connect(saveReportDialog.replace_table)
        self.showTableHeader.clicked.connect(saveReportDialog.replace_table)
        self.imageBorder.clicked.connect(saveReportDialog.set_image_border)
        self.widthSpacing.valueChanged.connect(saveReportDialog.redraw_all_axes)
        self.heightSpacing.valueChanged.connect(saveReportDialog.redraw_all_axes)
        self.snapToImageSize.clicked.connect(saveReportDialog.snap_to_image_size)
        self.titleFontSize.valueChanged.connect(saveReportDialog.set_title_size)

        self.toolBox.setCurrentIndex(0)
        self.chartLayout.setCurrentIndex(0)
        self.chartSplit.setCurrentIndex(1)


        QMetaObject.connectSlotsByName(saveReportDialog)
    # setupUi

    def retranslateUi(self, saveReportDialog):
        saveReportDialog.setWindowTitle(QCoreApplication.translate("saveReportDialog", u"Save Report", None))
        self.label_3.setText(QCoreApplication.translate("saveReportDialog", u"x", None))
        self.imageFileLabel.setText(QCoreApplication.translate("saveReportDialog", u"File", None))
        self.imagePicker.setText(QCoreApplication.translate("saveReportDialog", u"...", None))
        self.titleLabel.setText(QCoreApplication.translate("saveReportDialog", u"Title", None))
        self.imageURLLabel.setText(QCoreApplication.translate("saveReportDialog", u"URL", None))
        self.label_4.setText(QCoreApplication.translate("saveReportDialog", u"Native Size", None))
        self.titleFontSizeLabel.setText(QCoreApplication.translate("saveReportDialog", u"Font Size", None))
        self.imageAlphaLabel.setText(QCoreApplication.translate("saveReportDialog", u"Alpha", None))
        self.label_5.setText(QCoreApplication.translate("saveReportDialog", u"x", None))
        self.loadURL.setText(QCoreApplication.translate("saveReportDialog", u"...", None))
        self.widthLabel.setText(QCoreApplication.translate("saveReportDialog", u"Actual Size", None))
        self.imageBorder.setText(QCoreApplication.translate("saveReportDialog", u"Show Border?", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page), QCoreApplication.translate("saveReportDialog", u"Image", None))
        self.tableAlphaLabel.setText(QCoreApplication.translate("saveReportDialog", u"Alpha", None))
        self.tableRowHeightLabel.setText(QCoreApplication.translate("saveReportDialog", u"Row Height", None))
        self.tableFontSizeLabel.setText(QCoreApplication.translate("saveReportDialog", u"Font Size", None))
        self.showTableHeader.setText(QCoreApplication.translate("saveReportDialog", u"Show Header?", None))
        self.tablePositionLabel.setText(QCoreApplication.translate("saveReportDialog", u"Position", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_2), QCoreApplication.translate("saveReportDialog", u"Filters", None))
        self.label_6.setText(QCoreApplication.translate("saveReportDialog", u"Width Spacing", None))
        self.majorSplitRatioLabel.setText(QCoreApplication.translate("saveReportDialog", u"Major Ratio", None))
        self.label_7.setText(QCoreApplication.translate("saveReportDialog", u"Layout", None))
        self.label_8.setText(QCoreApplication.translate("saveReportDialog", u"Height Spacing", None))
        self.label.setText(QCoreApplication.translate("saveReportDialog", u"Size", None))
        self.chartSplitLabel.setText(QCoreApplication.translate("saveReportDialog", u"Split", None))
        self.chartLayout.setItemText(0, QCoreApplication.translate("saveReportDialog", u"Image | Chart, Filters", None))
        self.chartLayout.setItemText(1, QCoreApplication.translate("saveReportDialog", u"Image | Filters, Chart", None))
        self.chartLayout.setItemText(2, QCoreApplication.translate("saveReportDialog", u"Chart | Image, Filter", None))
        self.chartLayout.setItemText(3, QCoreApplication.translate("saveReportDialog", u"Chart | Filters, Image", None))
        self.chartLayout.setItemText(4, QCoreApplication.translate("saveReportDialog", u"Filters | Image, Chart", None))
        self.chartLayout.setItemText(5, QCoreApplication.translate("saveReportDialog", u"Filters | Chart, Image", None))
        self.chartLayout.setItemText(6, QCoreApplication.translate("saveReportDialog", u"Image, Filters | Chart", None))
        self.chartLayout.setItemText(7, QCoreApplication.translate("saveReportDialog", u"Filters, Image | Chart", None))
        self.chartLayout.setItemText(8, QCoreApplication.translate("saveReportDialog", u"Chart, Image | Filters", None))
        self.chartLayout.setItemText(9, QCoreApplication.translate("saveReportDialog", u"Image, Chart | Filters", None))
        self.chartLayout.setItemText(10, QCoreApplication.translate("saveReportDialog", u"Filters, Chart | Image", None))
        self.chartLayout.setItemText(11, QCoreApplication.translate("saveReportDialog", u"Chart, Filters | Image", None))
        self.chartLayout.setItemText(12, QCoreApplication.translate("saveReportDialog", u"Chart | Filters", None))
        self.chartLayout.setItemText(13, QCoreApplication.translate("saveReportDialog", u"Filters | Chart", None))
        self.chartLayout.setItemText(14, QCoreApplication.translate("saveReportDialog", u"Chart | Image", None))
        self.chartLayout.setItemText(15, QCoreApplication.translate("saveReportDialog", u"Image | Chart", None))
        self.chartLayout.setItemText(16, QCoreApplication.translate("saveReportDialog", u"Pixel Perfect Image | Chart", None))

        self.chartSplit.setItemText(0, QCoreApplication.translate("saveReportDialog", u"Horizontal", None))
        self.chartSplit.setItemText(1, QCoreApplication.translate("saveReportDialog", u"Vertical", None))

        self.label_2.setText(QCoreApplication.translate("saveReportDialog", u"x", None))
        self.label_9.setText(QCoreApplication.translate("saveReportDialog", u"Minor Ratio", None))
        self.snapToImageSize.setText(QCoreApplication.translate("saveReportDialog", u"...", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_3), QCoreApplication.translate("saveReportDialog", u"Layout", None))
        self.limitsButton.setText(QCoreApplication.translate("saveReportDialog", u"...", None))
        self.curvesLabel.setText(QCoreApplication.translate("saveReportDialog", u"Curves", None))
        self.gridAlphaLabel.setText(QCoreApplication.translate("saveReportDialog", u"Grid Alpha", None))
        self.saveLayout.setText(QCoreApplication.translate("saveReportDialog", u"Save Layout", None))
        self.showLegend.setText(QCoreApplication.translate("saveReportDialog", u"Show Legend?", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_4), QCoreApplication.translate("saveReportDialog", u"Chart", None))
    # retranslateUi

