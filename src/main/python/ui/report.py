# Form implementation generated from reading ui file 'report.ui'
#
# Created by: PyQt6 UI code generator 6.7.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_saveReportDialog(object):
    def setupUi(self, saveReportDialog):
        saveReportDialog.setObjectName("saveReportDialog")
        saveReportDialog.resize(1530, 700)
        self.gridLayout_2 = QtWidgets.QGridLayout(saveReportDialog)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.toolboxContainer = QtWidgets.QVBoxLayout()
        self.toolboxContainer.setObjectName("toolboxContainer")
        self.toolBox = QtWidgets.QToolBox(parent=saveReportDialog)
        font = QtGui.QFont()
        font.setBold(False)
        self.toolBox.setFont(font)
        self.toolBox.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.toolBox.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)
        self.toolBox.setObjectName("toolBox")
        self.imagePage = QtWidgets.QWidget()
        self.imagePage.setGeometry(QtCore.QRect(0, 0, 294, 314))
        self.imagePage.setObjectName("imagePage")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.imagePage)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.imageWidthPixels = QtWidgets.QSpinBox(parent=self.imagePage)
        self.imageWidthPixels.setEnabled(False)
        self.imageWidthPixels.setMinimum(0)
        self.imageWidthPixels.setMaximum(999999)
        self.imageWidthPixels.setObjectName("imageWidthPixels")
        self.horizontalLayout_6.addWidget(self.imageWidthPixels)
        self.label_3 = QtWidgets.QLabel(parent=self.imagePage)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_6.addWidget(self.label_3)
        self.imageHeightPixels = QtWidgets.QSpinBox(parent=self.imagePage)
        self.imageHeightPixels.setEnabled(False)
        self.imageHeightPixels.setMinimum(0)
        self.imageHeightPixels.setMaximum(999999)
        self.imageHeightPixels.setObjectName("imageHeightPixels")
        self.horizontalLayout_6.addWidget(self.imageHeightPixels)
        self.horizontalLayout_6.setStretch(0, 1)
        self.horizontalLayout_6.setStretch(2, 1)
        self.gridLayout_3.addLayout(self.horizontalLayout_6, 6, 1, 1, 1)
        self.imageFileLabel = QtWidgets.QLabel(parent=self.imagePage)
        self.imageFileLabel.setObjectName("imageFileLabel")
        self.gridLayout_3.addWidget(self.imageFileLabel, 0, 0, 1, 1)
        self.imagePicker = QtWidgets.QToolButton(parent=self.imagePage)
        self.imagePicker.setObjectName("imagePicker")
        self.gridLayout_3.addWidget(self.imagePicker, 0, 2, 1, 1)
        self.image = QtWidgets.QLineEdit(parent=self.imagePage)
        self.image.setEnabled(True)
        self.image.setReadOnly(True)
        self.image.setObjectName("image")
        self.gridLayout_3.addWidget(self.image, 0, 1, 1, 1)
        self.titleLabel = QtWidgets.QLabel(parent=self.imagePage)
        self.titleLabel.setObjectName("titleLabel")
        self.gridLayout_3.addWidget(self.titleLabel, 4, 0, 1, 1)
        self.imageURLLabel = QtWidgets.QLabel(parent=self.imagePage)
        self.imageURLLabel.setObjectName("imageURLLabel")
        self.gridLayout_3.addWidget(self.imageURLLabel, 1, 0, 1, 1)
        self.imageURL = QtWidgets.QLineEdit(parent=self.imagePage)
        self.imageURL.setObjectName("imageURL")
        self.gridLayout_3.addWidget(self.imageURL, 1, 1, 1, 1)
        self.label_4 = QtWidgets.QLabel(parent=self.imagePage)
        self.label_4.setObjectName("label_4")
        self.gridLayout_3.addWidget(self.label_4, 2, 0, 1, 1)
        self.titleFontSizeLabel = QtWidgets.QLabel(parent=self.imagePage)
        self.titleFontSizeLabel.setObjectName("titleFontSizeLabel")
        self.gridLayout_3.addWidget(self.titleFontSizeLabel, 5, 0, 1, 1)
        self.imageAlphaLabel = QtWidgets.QLabel(parent=self.imagePage)
        self.imageAlphaLabel.setObjectName("imageAlphaLabel")
        self.gridLayout_3.addWidget(self.imageAlphaLabel, 3, 0, 1, 1)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.nativeImageWidth = QtWidgets.QSpinBox(parent=self.imagePage)
        self.nativeImageWidth.setEnabled(False)
        self.nativeImageWidth.setMaximum(99999)
        self.nativeImageWidth.setObjectName("nativeImageWidth")
        self.horizontalLayout_8.addWidget(self.nativeImageWidth)
        self.label_5 = QtWidgets.QLabel(parent=self.imagePage)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_8.addWidget(self.label_5)
        self.nativeImageHeight = QtWidgets.QSpinBox(parent=self.imagePage)
        self.nativeImageHeight.setEnabled(False)
        self.nativeImageHeight.setMaximum(99999)
        self.nativeImageHeight.setObjectName("nativeImageHeight")
        self.horizontalLayout_8.addWidget(self.nativeImageHeight)
        self.horizontalLayout_8.setStretch(0, 1)
        self.horizontalLayout_8.setStretch(2, 1)
        self.gridLayout_3.addLayout(self.horizontalLayout_8, 2, 1, 1, 1)
        self.imageOpacity = QtWidgets.QDoubleSpinBox(parent=self.imagePage)
        self.imageOpacity.setDecimals(2)
        self.imageOpacity.setMinimum(0.01)
        self.imageOpacity.setMaximum(1.0)
        self.imageOpacity.setSingleStep(0.01)
        self.imageOpacity.setProperty("value", 1.0)
        self.imageOpacity.setObjectName("imageOpacity")
        self.gridLayout_3.addWidget(self.imageOpacity, 3, 1, 1, 1)
        self.loadURL = QtWidgets.QToolButton(parent=self.imagePage)
        self.loadURL.setObjectName("loadURL")
        self.gridLayout_3.addWidget(self.loadURL, 1, 2, 1, 1)
        self.titleFontSize = QtWidgets.QSpinBox(parent=self.imagePage)
        self.titleFontSize.setObjectName("titleFontSize")
        self.gridLayout_3.addWidget(self.titleFontSize, 5, 1, 1, 1)
        self.widthLabel = QtWidgets.QLabel(parent=self.imagePage)
        self.widthLabel.setObjectName("widthLabel")
        self.gridLayout_3.addWidget(self.widthLabel, 6, 0, 1, 1)
        self.title = QtWidgets.QLineEdit(parent=self.imagePage)
        self.title.setObjectName("title")
        self.gridLayout_3.addWidget(self.title, 4, 1, 1, 1)
        self.imageBorder = QtWidgets.QCheckBox(parent=self.imagePage)
        self.imageBorder.setObjectName("imageBorder")
        self.gridLayout_3.addWidget(self.imageBorder, 7, 1, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_3, 0, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.gridLayout_4.addItem(spacerItem, 1, 0, 1, 1)
        self.toolBox.addItem(self.imagePage, "")
        self.filtersPage = QtWidgets.QWidget()
        self.filtersPage.setGeometry(QtCore.QRect(0, 0, 294, 287))
        self.filtersPage.setObjectName("filtersPage")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.filtersPage)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.tableAlphaLabel = QtWidgets.QLabel(parent=self.filtersPage)
        self.tableAlphaLabel.setObjectName("tableAlphaLabel")
        self.gridLayout_5.addWidget(self.tableAlphaLabel, 5, 0, 1, 1)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.x0 = QtWidgets.QDoubleSpinBox(parent=self.filtersPage)
        self.x0.setDecimals(3)
        self.x0.setMaximum(0.999)
        self.x0.setSingleStep(0.001)
        self.x0.setProperty("value", 0.748)
        self.x0.setObjectName("x0")
        self.horizontalLayout_3.addWidget(self.x0)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem1)
        self.x1 = QtWidgets.QDoubleSpinBox(parent=self.filtersPage)
        self.x1.setDecimals(3)
        self.x1.setMaximum(1.0)
        self.x1.setSingleStep(0.001)
        self.x1.setProperty("value", 1.0)
        self.x1.setObjectName("x1")
        self.horizontalLayout_3.addWidget(self.x1)
        self.gridLayout_5.addLayout(self.horizontalLayout_3, 2, 1, 1, 1)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem2)
        self.y0 = QtWidgets.QDoubleSpinBox(parent=self.filtersPage)
        self.y0.setDecimals(3)
        self.y0.setMaximum(0.999)
        self.y0.setSingleStep(0.001)
        self.y0.setProperty("value", 0.75)
        self.y0.setObjectName("y0")
        self.horizontalLayout_4.addWidget(self.y0)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem3)
        self.gridLayout_5.addLayout(self.horizontalLayout_4, 3, 1, 1, 1)
        self.tableRowHeightLabel = QtWidgets.QLabel(parent=self.filtersPage)
        self.tableRowHeightLabel.setObjectName("tableRowHeightLabel")
        self.gridLayout_5.addWidget(self.tableRowHeightLabel, 0, 0, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem4)
        self.y1 = QtWidgets.QDoubleSpinBox(parent=self.filtersPage)
        self.y1.setDecimals(3)
        self.y1.setMaximum(1.0)
        self.y1.setSingleStep(0.001)
        self.y1.setProperty("value", 1.0)
        self.y1.setObjectName("y1")
        self.horizontalLayout_2.addWidget(self.y1)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem5)
        self.gridLayout_5.addLayout(self.horizontalLayout_2, 1, 1, 1, 1)
        self.filterRowHeightMultiplier = QtWidgets.QDoubleSpinBox(parent=self.filtersPage)
        self.filterRowHeightMultiplier.setMaximum(3.0)
        self.filterRowHeightMultiplier.setSingleStep(0.01)
        self.filterRowHeightMultiplier.setProperty("value", 1.2)
        self.filterRowHeightMultiplier.setObjectName("filterRowHeightMultiplier")
        self.gridLayout_5.addWidget(self.filterRowHeightMultiplier, 0, 1, 1, 1)
        self.tableFontSizeLabel = QtWidgets.QLabel(parent=self.filtersPage)
        self.tableFontSizeLabel.setObjectName("tableFontSizeLabel")
        self.gridLayout_5.addWidget(self.tableFontSizeLabel, 6, 0, 1, 1)
        self.showTableHeader = QtWidgets.QCheckBox(parent=self.filtersPage)
        self.showTableHeader.setObjectName("showTableHeader")
        self.gridLayout_5.addWidget(self.showTableHeader, 4, 1, 1, 1)
        self.tablePositionLabel = QtWidgets.QLabel(parent=self.filtersPage)
        self.tablePositionLabel.setObjectName("tablePositionLabel")
        self.gridLayout_5.addWidget(self.tablePositionLabel, 1, 0, 3, 1)
        self.tableAlpha = QtWidgets.QDoubleSpinBox(parent=self.filtersPage)
        self.tableAlpha.setMinimum(0.01)
        self.tableAlpha.setMaximum(1.0)
        self.tableAlpha.setSingleStep(0.01)
        self.tableAlpha.setProperty("value", 1.0)
        self.tableAlpha.setObjectName("tableAlpha")
        self.gridLayout_5.addWidget(self.tableAlpha, 5, 1, 1, 1)
        self.tableFontSize = QtWidgets.QSpinBox(parent=self.filtersPage)
        self.tableFontSize.setMaximum(24)
        self.tableFontSize.setProperty("value", 10)
        self.tableFontSize.setObjectName("tableFontSize")
        self.gridLayout_5.addWidget(self.tableFontSize, 6, 1, 1, 1)
        self.gridLayout_6.addLayout(self.gridLayout_5, 0, 0, 1, 1)
        spacerItem6 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.gridLayout_6.addItem(spacerItem6, 1, 0, 1, 1)
        self.toolBox.addItem(self.filtersPage, "")
        self.layoutPage = QtWidgets.QWidget()
        self.layoutPage.setGeometry(QtCore.QRect(0, 0, 325, 284))
        self.layoutPage.setObjectName("layoutPage")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.layoutPage)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.gridLayout_7 = QtWidgets.QGridLayout()
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.label_6 = QtWidgets.QLabel(parent=self.layoutPage)
        self.label_6.setObjectName("label_6")
        self.gridLayout_7.addWidget(self.label_6, 5, 0, 1, 1)
        self.majorSplitRatioLabel = QtWidgets.QLabel(parent=self.layoutPage)
        self.majorSplitRatioLabel.setObjectName("majorSplitRatioLabel")
        self.gridLayout_7.addWidget(self.majorSplitRatioLabel, 0, 0, 1, 1)
        self.label_7 = QtWidgets.QLabel(parent=self.layoutPage)
        self.label_7.setObjectName("label_7")
        self.gridLayout_7.addWidget(self.label_7, 3, 0, 1, 1)
        self.heightSpacing = QtWidgets.QDoubleSpinBox(parent=self.layoutPage)
        self.heightSpacing.setMaximum(1.0)
        self.heightSpacing.setSingleStep(0.01)
        self.heightSpacing.setObjectName("heightSpacing")
        self.gridLayout_7.addWidget(self.heightSpacing, 6, 1, 1, 1)
        spacerItem7 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.gridLayout_7.addItem(spacerItem7, 7, 0, 1, 3)
        self.majorSplitRatio = QtWidgets.QDoubleSpinBox(parent=self.layoutPage)
        self.majorSplitRatio.setDecimals(3)
        self.majorSplitRatio.setMinimum(0.1)
        self.majorSplitRatio.setMaximum(10.0)
        self.majorSplitRatio.setSingleStep(0.001)
        self.majorSplitRatio.setProperty("value", 1.0)
        self.majorSplitRatio.setObjectName("majorSplitRatio")
        self.gridLayout_7.addWidget(self.majorSplitRatio, 0, 1, 1, 1)
        self.widthSpacing = QtWidgets.QDoubleSpinBox(parent=self.layoutPage)
        self.widthSpacing.setMaximum(1.0)
        self.widthSpacing.setSingleStep(0.01)
        self.widthSpacing.setObjectName("widthSpacing")
        self.gridLayout_7.addWidget(self.widthSpacing, 5, 1, 1, 1)
        self.label_8 = QtWidgets.QLabel(parent=self.layoutPage)
        self.label_8.setObjectName("label_8")
        self.gridLayout_7.addWidget(self.label_8, 6, 0, 1, 1)
        self.label = QtWidgets.QLabel(parent=self.layoutPage)
        self.label.setObjectName("label")
        self.gridLayout_7.addWidget(self.label, 4, 0, 1, 1)
        self.chartSplitLabel = QtWidgets.QLabel(parent=self.layoutPage)
        self.chartSplitLabel.setObjectName("chartSplitLabel")
        self.gridLayout_7.addWidget(self.chartSplitLabel, 2, 0, 1, 1)
        self.chartLayout = QtWidgets.QComboBox(parent=self.layoutPage)
        self.chartLayout.setObjectName("chartLayout")
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
        self.gridLayout_7.addWidget(self.chartLayout, 3, 1, 1, 1)
        self.chartSplit = QtWidgets.QComboBox(parent=self.layoutPage)
        self.chartSplit.setObjectName("chartSplit")
        self.chartSplit.addItem("")
        self.chartSplit.addItem("")
        self.gridLayout_7.addWidget(self.chartSplit, 2, 1, 1, 1)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.widthPixels = QtWidgets.QSpinBox(parent=self.layoutPage)
        self.widthPixels.setMinimum(512)
        self.widthPixels.setMaximum(8192)
        self.widthPixels.setObjectName("widthPixels")
        self.horizontalLayout_7.addWidget(self.widthPixels)
        self.label_2 = QtWidgets.QLabel(parent=self.layoutPage)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_7.addWidget(self.label_2)
        self.heightPixels = QtWidgets.QSpinBox(parent=self.layoutPage)
        self.heightPixels.setMinimum(512)
        self.heightPixels.setMaximum(8192)
        self.heightPixels.setObjectName("heightPixels")
        self.horizontalLayout_7.addWidget(self.heightPixels)
        self.horizontalLayout_7.setStretch(0, 1)
        self.horizontalLayout_7.setStretch(2, 1)
        self.gridLayout_7.addLayout(self.horizontalLayout_7, 4, 1, 1, 1)
        self.minorSplitRatio = QtWidgets.QDoubleSpinBox(parent=self.layoutPage)
        self.minorSplitRatio.setDecimals(3)
        self.minorSplitRatio.setMinimum(0.001)
        self.minorSplitRatio.setMaximum(10.0)
        self.minorSplitRatio.setSingleStep(0.001)
        self.minorSplitRatio.setProperty("value", 3.0)
        self.minorSplitRatio.setObjectName("minorSplitRatio")
        self.gridLayout_7.addWidget(self.minorSplitRatio, 1, 1, 1, 1)
        self.label_9 = QtWidgets.QLabel(parent=self.layoutPage)
        self.label_9.setObjectName("label_9")
        self.gridLayout_7.addWidget(self.label_9, 1, 0, 1, 1)
        self.snapToImageSize = QtWidgets.QToolButton(parent=self.layoutPage)
        self.snapToImageSize.setObjectName("snapToImageSize")
        self.gridLayout_7.addWidget(self.snapToImageSize, 5, 2, 1, 1)
        self.gridLayout_8.addLayout(self.gridLayout_7, 0, 0, 1, 1)
        self.toolBox.addItem(self.layoutPage, "")
        self.chartPage = QtWidgets.QWidget()
        self.chartPage.setGeometry(QtCore.QRect(0, 0, 372, 304))
        self.chartPage.setObjectName("chartPage")
        self.gridLayout_10 = QtWidgets.QGridLayout(self.chartPage)
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.gridLayout_9 = QtWidgets.QGridLayout()
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.curves = QtWidgets.QListWidget(parent=self.chartPage)
        self.curves.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)
        self.curves.setObjectName("curves")
        self.gridLayout_9.addWidget(self.curves, 1, 1, 1, 1)
        self.limitsButton = QtWidgets.QToolButton(parent=self.chartPage)
        self.limitsButton.setObjectName("limitsButton")
        self.gridLayout_9.addWidget(self.limitsButton, 0, 2, 1, 1)
        self.curvesLabel = QtWidgets.QLabel(parent=self.chartPage)
        self.curvesLabel.setObjectName("curvesLabel")
        self.gridLayout_9.addWidget(self.curvesLabel, 1, 0, 1, 1)
        self.gridAlphaLabel = QtWidgets.QLabel(parent=self.chartPage)
        self.gridAlphaLabel.setObjectName("gridAlphaLabel")
        self.gridLayout_9.addWidget(self.gridAlphaLabel, 0, 0, 1, 1)
        self.gridOpacity = QtWidgets.QDoubleSpinBox(parent=self.chartPage)
        self.gridOpacity.setMaximum(1.0)
        self.gridOpacity.setSingleStep(0.01)
        self.gridOpacity.setProperty("value", 0.4)
        self.gridOpacity.setObjectName("gridOpacity")
        self.gridLayout_9.addWidget(self.gridOpacity, 0, 1, 1, 1)
        self.saveLayout = QtWidgets.QPushButton(parent=self.chartPage)
        self.saveLayout.setObjectName("saveLayout")
        self.gridLayout_9.addWidget(self.saveLayout, 3, 1, 1, 1)
        self.showLegend = QtWidgets.QCheckBox(parent=self.chartPage)
        self.showLegend.setObjectName("showLegend")
        self.gridLayout_9.addWidget(self.showLegend, 2, 1, 1, 1)
        self.gridLayout_10.addLayout(self.gridLayout_9, 0, 0, 1, 1)
        self.toolBox.addItem(self.chartPage, "")
        self.toolboxContainer.addWidget(self.toolBox)
        spacerItem8 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.toolboxContainer.addItem(spacerItem8)
        self.buttonBox = QtWidgets.QDialogButtonBox(parent=saveReportDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.StandardButton.Close|QtWidgets.QDialogButtonBox.StandardButton.RestoreDefaults|QtWidgets.QDialogButtonBox.StandardButton.Save)
        self.buttonBox.setObjectName("buttonBox")
        self.toolboxContainer.addWidget(self.buttonBox)
        self.horizontalLayout.addLayout(self.toolboxContainer)
        self.preview = MplWidget(parent=saveReportDialog)
        self.preview.setObjectName("preview")
        self.horizontalLayout.addWidget(self.preview)
        self.horizontalLayout.setStretch(1, 1)
        self.gridLayout_2.addLayout(self.horizontalLayout, 0, 1, 1, 1)

        self.retranslateUi(saveReportDialog)
        self.toolBox.setCurrentIndex(0)
        self.chartLayout.setCurrentIndex(0)
        self.chartSplit.setCurrentIndex(1)
        self.buttonBox.accepted.connect(saveReportDialog.accept) # type: ignore
        self.buttonBox.rejected.connect(saveReportDialog.reject) # type: ignore
        self.curves.itemSelectionChanged.connect(saveReportDialog.set_selected) # type: ignore
        self.widthPixels.valueChanged['int'].connect(saveReportDialog.update_height) # type: ignore
        self.showLegend.clicked.connect(saveReportDialog.redraw) # type: ignore
        self.imagePicker.clicked.connect(saveReportDialog.choose_image) # type: ignore
        self.title.textChanged['QString'].connect(saveReportDialog.set_title) # type: ignore
        self.majorSplitRatio.valueChanged['double'].connect(saveReportDialog.redraw_all_axes) # type: ignore
        self.chartSplit.currentIndexChanged['int'].connect(saveReportDialog.redraw_all_axes) # type: ignore
        self.limitsButton.clicked.connect(saveReportDialog.show_limits) # type: ignore
        self.chartSplit.currentIndexChanged['int'].connect(saveReportDialog.redraw_all_axes) # type: ignore
        self.chartLayout.currentIndexChanged['int'].connect(saveReportDialog.redraw_all_axes) # type: ignore
        self.minorSplitRatio.valueChanged['double'].connect(saveReportDialog.redraw_all_axes) # type: ignore
        self.gridOpacity.valueChanged['double'].connect(saveReportDialog.set_grid_opacity) # type: ignore
        self.imageOpacity.valueChanged['double'].connect(saveReportDialog.set_image_opacity) # type: ignore
        self.filterRowHeightMultiplier.valueChanged['double'].connect(saveReportDialog.replace_table) # type: ignore
        self.y0.valueChanged['double'].connect(saveReportDialog.replace_table) # type: ignore
        self.x0.valueChanged['double'].connect(saveReportDialog.replace_table) # type: ignore
        self.y1.valueChanged['double'].connect(saveReportDialog.replace_table) # type: ignore
        self.x1.valueChanged['double'].connect(saveReportDialog.replace_table) # type: ignore
        self.saveLayout.clicked.connect(saveReportDialog.save_layout) # type: ignore
        self.imageURL.textChanged['QString'].connect(saveReportDialog.update_image_url) # type: ignore
        self.loadURL.clicked.connect(saveReportDialog.load_image_from_url) # type: ignore
        self.tableAlpha.valueChanged['double'].connect(saveReportDialog.replace_table) # type: ignore
        self.tableFontSize.valueChanged['int'].connect(saveReportDialog.replace_table) # type: ignore
        self.showTableHeader.clicked.connect(saveReportDialog.replace_table) # type: ignore
        self.imageBorder.clicked.connect(saveReportDialog.set_image_border) # type: ignore
        self.widthSpacing.valueChanged['double'].connect(saveReportDialog.redraw_all_axes) # type: ignore
        self.heightSpacing.valueChanged['double'].connect(saveReportDialog.redraw_all_axes) # type: ignore
        self.snapToImageSize.clicked.connect(saveReportDialog.snap_to_image_size) # type: ignore
        self.titleFontSize.valueChanged['int'].connect(saveReportDialog.set_title_size) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(saveReportDialog)
        saveReportDialog.setTabOrder(self.x0, self.y0)
        saveReportDialog.setTabOrder(self.y0, self.x1)
        saveReportDialog.setTabOrder(self.x1, self.y1)
        saveReportDialog.setTabOrder(self.y1, self.preview)

    def retranslateUi(self, saveReportDialog):
        _translate = QtCore.QCoreApplication.translate
        saveReportDialog.setWindowTitle(_translate("saveReportDialog", "Save Report"))
        self.label_3.setText(_translate("saveReportDialog", "x"))
        self.imageFileLabel.setText(_translate("saveReportDialog", "File"))
        self.imagePicker.setText(_translate("saveReportDialog", "..."))
        self.titleLabel.setText(_translate("saveReportDialog", "Title"))
        self.imageURLLabel.setText(_translate("saveReportDialog", "URL"))
        self.label_4.setText(_translate("saveReportDialog", "Native Size"))
        self.titleFontSizeLabel.setText(_translate("saveReportDialog", "Font Size"))
        self.imageAlphaLabel.setText(_translate("saveReportDialog", "Alpha"))
        self.label_5.setText(_translate("saveReportDialog", "x"))
        self.loadURL.setText(_translate("saveReportDialog", "..."))
        self.widthLabel.setText(_translate("saveReportDialog", "Actual Size"))
        self.imageBorder.setText(_translate("saveReportDialog", "Show Border?"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.imagePage), _translate("saveReportDialog", "Image"))
        self.tableAlphaLabel.setText(_translate("saveReportDialog", "Alpha"))
        self.tableRowHeightLabel.setText(_translate("saveReportDialog", "Row Height"))
        self.tableFontSizeLabel.setText(_translate("saveReportDialog", "Font Size"))
        self.showTableHeader.setText(_translate("saveReportDialog", "Show Header?"))
        self.tablePositionLabel.setText(_translate("saveReportDialog", "Position"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.filtersPage), _translate("saveReportDialog", "Filters"))
        self.label_6.setText(_translate("saveReportDialog", "Width Spacing"))
        self.majorSplitRatioLabel.setText(_translate("saveReportDialog", "Major Ratio"))
        self.label_7.setText(_translate("saveReportDialog", "Layout"))
        self.label_8.setText(_translate("saveReportDialog", "Height Spacing"))
        self.label.setText(_translate("saveReportDialog", "Size"))
        self.chartSplitLabel.setText(_translate("saveReportDialog", "Split"))
        self.chartLayout.setItemText(0, _translate("saveReportDialog", "Image | Chart, Filters"))
        self.chartLayout.setItemText(1, _translate("saveReportDialog", "Image | Filters, Chart"))
        self.chartLayout.setItemText(2, _translate("saveReportDialog", "Chart | Image, Filter"))
        self.chartLayout.setItemText(3, _translate("saveReportDialog", "Chart | Filters, Image"))
        self.chartLayout.setItemText(4, _translate("saveReportDialog", "Filters | Image, Chart"))
        self.chartLayout.setItemText(5, _translate("saveReportDialog", "Filters | Chart, Image"))
        self.chartLayout.setItemText(6, _translate("saveReportDialog", "Image, Filters | Chart"))
        self.chartLayout.setItemText(7, _translate("saveReportDialog", "Filters, Image | Chart"))
        self.chartLayout.setItemText(8, _translate("saveReportDialog", "Chart, Image | Filters"))
        self.chartLayout.setItemText(9, _translate("saveReportDialog", "Image, Chart | Filters"))
        self.chartLayout.setItemText(10, _translate("saveReportDialog", "Filters, Chart | Image"))
        self.chartLayout.setItemText(11, _translate("saveReportDialog", "Chart, Filters | Image"))
        self.chartLayout.setItemText(12, _translate("saveReportDialog", "Chart | Filters"))
        self.chartLayout.setItemText(13, _translate("saveReportDialog", "Filters | Chart"))
        self.chartLayout.setItemText(14, _translate("saveReportDialog", "Chart | Image"))
        self.chartLayout.setItemText(15, _translate("saveReportDialog", "Image | Chart"))
        self.chartLayout.setItemText(16, _translate("saveReportDialog", "Pixel Perfect Image | Chart"))
        self.chartSplit.setItemText(0, _translate("saveReportDialog", "Horizontal"))
        self.chartSplit.setItemText(1, _translate("saveReportDialog", "Vertical"))
        self.label_2.setText(_translate("saveReportDialog", "x"))
        self.label_9.setText(_translate("saveReportDialog", "Minor Ratio"))
        self.snapToImageSize.setText(_translate("saveReportDialog", "..."))
        self.toolBox.setItemText(self.toolBox.indexOf(self.layoutPage), _translate("saveReportDialog", "Layout"))
        self.limitsButton.setText(_translate("saveReportDialog", "..."))
        self.curvesLabel.setText(_translate("saveReportDialog", "Curves"))
        self.gridAlphaLabel.setText(_translate("saveReportDialog", "Grid Alpha"))
        self.saveLayout.setText(_translate("saveReportDialog", "Save Layout"))
        self.showLegend.setText(_translate("saveReportDialog", "Show Legend?"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.chartPage), _translate("saveReportDialog", "Chart"))
from mpl import MplWidget
