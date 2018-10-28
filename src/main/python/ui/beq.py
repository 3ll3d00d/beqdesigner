# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'beq.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1550, 956)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.widgetGridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.widgetGridLayout.setObjectName("widgetGridLayout")
        self.panes = QtWidgets.QHBoxLayout()
        self.panes.setSpacing(9)
        self.panes.setObjectName("panes")
        self.leftPane = QtWidgets.QGridLayout()
        self.leftPane.setObjectName("leftPane")
        self.referenceLabel = QtWidgets.QLabel(self.centralwidget)
        self.referenceLabel.setObjectName("referenceLabel")
        self.leftPane.addWidget(self.referenceLabel, 0, 0, 1, 1)
        self.signalReference = QtWidgets.QComboBox(self.centralwidget)
        self.signalReference.setObjectName("signalReference")
        self.signalReference.addItem("")
        self.leftPane.addWidget(self.signalReference, 0, 1, 1, 1)
        self.limitsButton = QtWidgets.QToolButton(self.centralwidget)
        self.limitsButton.setObjectName("limitsButton")
        self.leftPane.addWidget(self.limitsButton, 0, 4, 1, 1)
        self.filterReference = QtWidgets.QComboBox(self.centralwidget)
        self.filterReference.setObjectName("filterReference")
        self.filterReference.addItem("")
        self.leftPane.addWidget(self.filterReference, 0, 2, 1, 1)
        self.showValuesButton = QtWidgets.QToolButton(self.centralwidget)
        self.showValuesButton.setObjectName("showValuesButton")
        self.leftPane.addWidget(self.showValuesButton, 0, 3, 1, 1)
        self.chartSplitter = QtWidgets.QSplitter(self.centralwidget)
        self.chartSplitter.setLineWidth(1)
        self.chartSplitter.setOrientation(QtCore.Qt.Vertical)
        self.chartSplitter.setObjectName("chartSplitter")
        self.mainChart = MplWidget(self.chartSplitter)
        self.mainChart.setObjectName("mainChart")
        self.waveformContainer = QtWidgets.QFrame(self.chartSplitter)
        self.waveformContainer.setObjectName("waveformContainer")
        self.waveformLayout = QtWidgets.QHBoxLayout(self.waveformContainer)
        self.waveformLayout.setObjectName("waveformLayout")
        self.waveformControls = QtWidgets.QGridLayout()
        self.waveformControls.setObjectName("waveformControls")
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.waveformControls.addItem(spacerItem, 7, 0, 1, 2)
        self.signalSelectorLabel = QtWidgets.QLabel(self.waveformContainer)
        self.signalSelectorLabel.setObjectName("signalSelectorLabel")
        self.waveformControls.addWidget(self.signalSelectorLabel, 0, 0, 1, 1)
        self.headroom = QtWidgets.QDoubleSpinBox(self.waveformContainer)
        self.headroom.setEnabled(False)
        self.headroom.setMinimum(0.0)
        self.headroom.setMaximum(120.0)
        self.headroom.setSingleStep(0.01)
        self.headroom.setObjectName("headroom")
        self.waveformControls.addWidget(self.headroom, 2, 1, 1, 1)
        self.startTimeLabel = QtWidgets.QLabel(self.waveformContainer)
        self.startTimeLabel.setObjectName("startTimeLabel")
        self.waveformControls.addWidget(self.startTimeLabel, 3, 0, 1, 1)
        self.headroomLabel = QtWidgets.QLabel(self.waveformContainer)
        self.headroomLabel.setObjectName("headroomLabel")
        self.waveformControls.addWidget(self.headroomLabel, 2, 0, 1, 1)
        self.signalSelector = QtWidgets.QComboBox(self.waveformContainer)
        self.signalSelector.setObjectName("signalSelector")
        self.waveformControls.addWidget(self.signalSelector, 0, 1, 1, 1)
        self.waveformIsFiltered = QtWidgets.QCheckBox(self.waveformContainer)
        self.waveformIsFiltered.setObjectName("waveformIsFiltered")
        self.waveformControls.addWidget(self.waveformIsFiltered, 1, 0, 1, 2)
        self.startTime = QtWidgets.QTimeEdit(self.waveformContainer)
        self.startTime.setObjectName("startTime")
        self.waveformControls.addWidget(self.startTime, 3, 1, 1, 1)
        self.endTime = QtWidgets.QTimeEdit(self.waveformContainer)
        self.endTime.setObjectName("endTime")
        self.waveformControls.addWidget(self.endTime, 4, 1, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.applyWaveformLimitsButton = QtWidgets.QToolButton(self.waveformContainer)
        self.applyWaveformLimitsButton.setObjectName("applyWaveformLimitsButton")
        self.horizontalLayout.addWidget(self.applyWaveformLimitsButton)
        self.resetWaveformLimitsButton = QtWidgets.QToolButton(self.waveformContainer)
        self.resetWaveformLimitsButton.setObjectName("resetWaveformLimitsButton")
        self.horizontalLayout.addWidget(self.resetWaveformLimitsButton)
        self.zoomInButton = QtWidgets.QToolButton(self.waveformContainer)
        self.zoomInButton.setObjectName("zoomInButton")
        self.horizontalLayout.addWidget(self.zoomInButton)
        self.zoomOutButton = QtWidgets.QToolButton(self.waveformContainer)
        self.zoomOutButton.setObjectName("zoomOutButton")
        self.horizontalLayout.addWidget(self.zoomOutButton)
        self.waveformControls.addLayout(self.horizontalLayout, 6, 0, 1, 2)
        self.endTimeLabel = QtWidgets.QLabel(self.waveformContainer)
        self.endTimeLabel.setObjectName("endTimeLabel")
        self.waveformControls.addWidget(self.endTimeLabel, 4, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.waveformContainer)
        self.label.setObjectName("label")
        self.waveformControls.addWidget(self.label, 5, 0, 1, 1)
        self.yRangeLayout = QtWidgets.QHBoxLayout()
        self.yRangeLayout.setObjectName("yRangeLayout")
        self.yMin = QtWidgets.QDoubleSpinBox(self.waveformContainer)
        self.yMin.setMinimum(-1.0)
        self.yMin.setMaximum(1.0)
        self.yMin.setSingleStep(0.01)
        self.yMin.setProperty("value", -1.0)
        self.yMin.setObjectName("yMin")
        self.yRangeLayout.addWidget(self.yMin)
        self.yMax = QtWidgets.QDoubleSpinBox(self.waveformContainer)
        self.yMax.setMinimum(-1.0)
        self.yMax.setMaximum(1.0)
        self.yMax.setSingleStep(0.01)
        self.yMax.setProperty("value", 1.0)
        self.yMax.setObjectName("yMax")
        self.yRangeLayout.addWidget(self.yMax)
        self.waveformControls.addLayout(self.yRangeLayout, 5, 1, 1, 1)
        self.waveformLayout.addLayout(self.waveformControls)
        self.waveformChart = PlotWidgetWithDateAxis(self.waveformContainer)
        self.waveformChart.setObjectName("waveformChart")
        self.waveformLayout.addWidget(self.waveformChart)
        self.waveformLayout.setStretch(1, 1)
        self.leftPane.addWidget(self.chartSplitter, 1, 0, 1, 6)
        self.leftPane.setColumnStretch(1, 1)
        self.leftPane.setColumnStretch(2, 1)
        self.panes.addLayout(self.leftPane)
        self.rightPane = QtWidgets.QGridLayout()
        self.rightPane.setObjectName("rightPane")
        self.filtersLabel = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.filtersLabel.setFont(font)
        self.filtersLabel.setFrameShape(QtWidgets.QFrame.Box)
        self.filtersLabel.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.filtersLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.filtersLabel.setObjectName("filtersLabel")
        self.rightPane.addWidget(self.filtersLabel, 3, 0, 1, 3)
        self.signalsLabel = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.signalsLabel.setFont(font)
        self.signalsLabel.setFrameShape(QtWidgets.QFrame.Box)
        self.signalsLabel.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.signalsLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.signalsLabel.setObjectName("signalsLabel")
        self.rightPane.addWidget(self.signalsLabel, 0, 0, 1, 3)
        self.addSignalButton = QtWidgets.QPushButton(self.centralwidget)
        self.addSignalButton.setFlat(False)
        self.addSignalButton.setObjectName("addSignalButton")
        self.rightPane.addWidget(self.addSignalButton, 1, 0, 1, 1)
        self.preset1Button = QtWidgets.QPushButton(self.centralwidget)
        self.preset1Button.setEnabled(False)
        self.preset1Button.setObjectName("preset1Button")
        self.rightPane.addWidget(self.preset1Button, 4, 0, 1, 1)
        self.preset2Button = QtWidgets.QPushButton(self.centralwidget)
        self.preset2Button.setEnabled(False)
        self.preset2Button.setObjectName("preset2Button")
        self.rightPane.addWidget(self.preset2Button, 4, 1, 1, 1)
        self.addFilterButton = QtWidgets.QPushButton(self.centralwidget)
        self.addFilterButton.setEnabled(True)
        self.addFilterButton.setObjectName("addFilterButton")
        self.rightPane.addWidget(self.addFilterButton, 5, 0, 1, 1)
        self.signalView = QtWidgets.QTableView(self.centralwidget)
        self.signalView.setObjectName("signalView")
        self.rightPane.addWidget(self.signalView, 2, 0, 1, 3)
        self.preset3Button = QtWidgets.QPushButton(self.centralwidget)
        self.preset3Button.setEnabled(False)
        self.preset3Button.setObjectName("preset3Button")
        self.rightPane.addWidget(self.preset3Button, 4, 2, 1, 1)
        self.editFilterButton = QtWidgets.QPushButton(self.centralwidget)
        self.editFilterButton.setEnabled(False)
        self.editFilterButton.setFlat(False)
        self.editFilterButton.setObjectName("editFilterButton")
        self.rightPane.addWidget(self.editFilterButton, 5, 1, 1, 1)
        self.deleteFilterButton = QtWidgets.QPushButton(self.centralwidget)
        self.deleteFilterButton.setEnabled(False)
        self.deleteFilterButton.setObjectName("deleteFilterButton")
        self.rightPane.addWidget(self.deleteFilterButton, 5, 2, 1, 1)
        self.deleteSignalButton = QtWidgets.QPushButton(self.centralwidget)
        self.deleteSignalButton.setEnabled(False)
        self.deleteSignalButton.setObjectName("deleteSignalButton")
        self.rightPane.addWidget(self.deleteSignalButton, 1, 2, 1, 1)
        self.filterView = QtWidgets.QTableView(self.centralwidget)
        self.filterView.setObjectName("filterView")
        self.rightPane.addWidget(self.filterView, 6, 0, 1, 3)
        self.showFiltersLabel = QtWidgets.QLabel(self.centralwidget)
        self.showFiltersLabel.setObjectName("showFiltersLabel")
        self.rightPane.addWidget(self.showFiltersLabel, 9, 0, 1, 1)
        self.linkSignalButton = QtWidgets.QPushButton(self.centralwidget)
        self.linkSignalButton.setEnabled(False)
        self.linkSignalButton.setObjectName("linkSignalButton")
        self.rightPane.addWidget(self.linkSignalButton, 1, 1, 1, 1)
        self.showFilters = QtWidgets.QComboBox(self.centralwidget)
        self.showFilters.setObjectName("showFilters")
        self.rightPane.addWidget(self.showFilters, 9, 1, 1, 1)
        self.showSignalsLabel = QtWidgets.QLabel(self.centralwidget)
        self.showSignalsLabel.setObjectName("showSignalsLabel")
        self.rightPane.addWidget(self.showSignalsLabel, 7, 0, 1, 1)
        self.showSignals = QtWidgets.QComboBox(self.centralwidget)
        self.showSignals.setObjectName("showSignals")
        self.rightPane.addWidget(self.showSignals, 7, 1, 1, 1)
        self.showFilteredSignals = QtWidgets.QComboBox(self.centralwidget)
        self.showFilteredSignals.setObjectName("showFilteredSignals")
        self.rightPane.addWidget(self.showFilteredSignals, 7, 2, 1, 1)
        self.showLegend = QtWidgets.QCheckBox(self.centralwidget)
        self.showLegend.setChecked(True)
        self.showLegend.setObjectName("showLegend")
        self.rightPane.addWidget(self.showLegend, 9, 2, 1, 1)
        self.equalEnergyTilt = QtWidgets.QCheckBox(self.centralwidget)
        self.equalEnergyTilt.setObjectName("equalEnergyTilt")
        self.rightPane.addWidget(self.equalEnergyTilt, 10, 1, 1, 2)
        self.panes.addLayout(self.rightPane)
        self.panes.setStretch(0, 3)
        self.panes.setStretch(1, 1)
        self.widgetGridLayout.addLayout(self.panes, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1550, 31))
        self.menubar.setObjectName("menubar")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        self.menuSettings = QtWidgets.QMenu(self.menubar)
        self.menuSettings.setObjectName("menuSettings")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuPresets = QtWidgets.QMenu(self.menuFile)
        self.menuPresets.setObjectName("menuPresets")
        self.menuLoad = QtWidgets.QMenu(self.menuPresets)
        self.menuLoad.setObjectName("menuLoad")
        self.menu_Clear = QtWidgets.QMenu(self.menuPresets)
        self.menu_Clear.setObjectName("menu_Clear")
        self.menu_Store = QtWidgets.QMenu(self.menuPresets)
        self.menu_Store.setObjectName("menu_Store")
        self.menu_Tools = QtWidgets.QMenu(self.menubar)
        self.menu_Tools.setObjectName("menu_Tools")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionShow_Logs = QtWidgets.QAction(MainWindow)
        self.actionShow_Logs.setObjectName("actionShow_Logs")
        self.actionPreferences = QtWidgets.QAction(MainWindow)
        self.actionPreferences.setObjectName("actionPreferences")
        self.actionExtract_Audio = QtWidgets.QAction(MainWindow)
        self.actionExtract_Audio.setObjectName("actionExtract_Audio")
        self.actionPresets = QtWidgets.QAction(MainWindow)
        self.actionPresets.setObjectName("actionPresets")
        self.actionClear_Preset_3 = QtWidgets.QAction(MainWindow)
        self.actionClear_Preset_3.setObjectName("actionClear_Preset_3")
        self.actionSave_Chart = QtWidgets.QAction(MainWindow)
        self.actionSave_Chart.setObjectName("actionSave_Chart")
        self.actionExport_Biquad = QtWidgets.QAction(MainWindow)
        self.actionExport_Biquad.setObjectName("actionExport_Biquad")
        self.actionSave_Filter = QtWidgets.QAction(MainWindow)
        self.actionSave_Filter.setEnabled(False)
        self.actionSave_Filter.setObjectName("actionSave_Filter")
        self.actionLoad_Filter = QtWidgets.QAction(MainWindow)
        self.actionLoad_Filter.setCheckable(False)
        self.actionLoad_Filter.setEnabled(True)
        self.actionLoad_Filter.setObjectName("actionLoad_Filter")
        self.action_load_preset_1 = QtWidgets.QAction(MainWindow)
        self.action_load_preset_1.setObjectName("action_load_preset_1")
        self.action_load_preset_2 = QtWidgets.QAction(MainWindow)
        self.action_load_preset_2.setObjectName("action_load_preset_2")
        self.action_load_preset_3 = QtWidgets.QAction(MainWindow)
        self.action_load_preset_3.setObjectName("action_load_preset_3")
        self.action_clear_preset_1 = QtWidgets.QAction(MainWindow)
        self.action_clear_preset_1.setObjectName("action_clear_preset_1")
        self.action_clear_preset_2 = QtWidgets.QAction(MainWindow)
        self.action_clear_preset_2.setObjectName("action_clear_preset_2")
        self.action_clear_preset_3 = QtWidgets.QAction(MainWindow)
        self.action_clear_preset_3.setObjectName("action_clear_preset_3")
        self.action_store_preset_1 = QtWidgets.QAction(MainWindow)
        self.action_store_preset_1.setObjectName("action_store_preset_1")
        self.action_store_preset_2 = QtWidgets.QAction(MainWindow)
        self.action_store_preset_2.setObjectName("action_store_preset_2")
        self.action_store_preset_3 = QtWidgets.QAction(MainWindow)
        self.action_store_preset_3.setObjectName("action_store_preset_3")
        self.actionExport_FRD = QtWidgets.QAction(MainWindow)
        self.actionExport_FRD.setObjectName("actionExport_FRD")
        self.action_Save_Project = QtWidgets.QAction(MainWindow)
        self.action_Save_Project.setObjectName("action_Save_Project")
        self.action_Load_Project = QtWidgets.QAction(MainWindow)
        self.action_Load_Project.setObjectName("action_Load_Project")
        self.actionSave_Signal = QtWidgets.QAction(MainWindow)
        self.actionSave_Signal.setObjectName("actionSave_Signal")
        self.actionLoad_Signal = QtWidgets.QAction(MainWindow)
        self.actionLoad_Signal.setObjectName("actionLoad_Signal")
        self.actionClear_Project = QtWidgets.QAction(MainWindow)
        self.actionClear_Project.setObjectName("actionClear_Project")
        self.actionAnalyse_Audio = QtWidgets.QAction(MainWindow)
        self.actionAnalyse_Audio.setObjectName("actionAnalyse_Audio")
        self.action_Batch_Extract = QtWidgets.QAction(MainWindow)
        self.action_Batch_Extract.setObjectName("action_Batch_Extract")
        self.actionSave_Report = QtWidgets.QAction(MainWindow)
        self.actionSave_Report.setObjectName("actionSave_Report")
        self.actionAbout = QtWidgets.QAction(MainWindow)
        self.actionAbout.setObjectName("actionAbout")
        self.action_Remux_Audio = QtWidgets.QAction(MainWindow)
        self.action_Remux_Audio.setObjectName("action_Remux_Audio")
        self.menuHelp.addAction(self.actionShow_Logs)
        self.menuHelp.addAction(self.actionAbout)
        self.menuSettings.addAction(self.actionPreferences)
        self.menuSettings.addSeparator()
        self.menuLoad.addAction(self.action_load_preset_1)
        self.menuLoad.addAction(self.action_load_preset_2)
        self.menuLoad.addAction(self.action_load_preset_3)
        self.menu_Clear.addAction(self.action_clear_preset_1)
        self.menu_Clear.addAction(self.action_clear_preset_2)
        self.menu_Clear.addAction(self.action_clear_preset_3)
        self.menu_Store.addAction(self.action_store_preset_1)
        self.menu_Store.addAction(self.action_store_preset_2)
        self.menu_Store.addAction(self.action_store_preset_3)
        self.menuPresets.addAction(self.menuLoad.menuAction())
        self.menuPresets.addAction(self.menu_Store.menuAction())
        self.menuPresets.addAction(self.menu_Clear.menuAction())
        self.menuFile.addAction(self.action_Load_Project)
        self.menuFile.addAction(self.action_Save_Project)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionLoad_Signal)
        self.menuFile.addAction(self.actionSave_Signal)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionLoad_Filter)
        self.menuFile.addAction(self.actionSave_Filter)
        self.menuFile.addAction(self.menuPresets.menuAction())
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExport_Biquad)
        self.menuFile.addAction(self.actionExport_FRD)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionSave_Chart)
        self.menuFile.addAction(self.actionSave_Report)
        self.menu_Tools.addAction(self.actionExtract_Audio)
        self.menu_Tools.addAction(self.action_Batch_Extract)
        self.menu_Tools.addAction(self.action_Remux_Audio)
        self.menu_Tools.addSeparator()
        self.menu_Tools.addAction(self.actionAnalyse_Audio)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuSettings.menuAction())
        self.menubar.addAction(self.menu_Tools.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        self.addFilterButton.clicked.connect(MainWindow.addFilter)
        self.deleteFilterButton.clicked.connect(MainWindow.deleteFilter)
        self.editFilterButton.clicked.connect(MainWindow.editFilter)
        self.addSignalButton.clicked.connect(MainWindow.addSignal)
        self.deleteSignalButton.clicked.connect(MainWindow.deleteSignal)
        self.signalReference.currentIndexChanged['int'].connect(MainWindow.normaliseSignalMagnitude)
        self.limitsButton.clicked.connect(MainWindow.showLimits)
        self.filterReference.currentIndexChanged['int'].connect(MainWindow.normaliseFilterMagnitude)
        self.showValuesButton.clicked.connect(MainWindow.showValues)
        self.showLegend.clicked.connect(MainWindow.changeLegendVisibility)
        self.showFilters.currentTextChanged['QString'].connect(MainWindow.changeFilterVisibility)
        self.preset1Button.clicked.connect(MainWindow.applyPreset1)
        self.preset2Button.clicked.connect(MainWindow.applyPreset2)
        self.preset3Button.clicked.connect(MainWindow.applyPreset3)
        self.showSignals.currentTextChanged['QString'].connect(MainWindow.changeSignalVisibility)
        self.showFilteredSignals.currentTextChanged['QString'].connect(MainWindow.changeSignalFilterVisibility)
        self.linkSignalButton.clicked.connect(MainWindow.linkSignals)
        self.equalEnergyTilt.clicked.connect(MainWindow.toggleTilt)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "BEQ Designer"))
        self.referenceLabel.setText(_translate("MainWindow", "Reference:"))
        self.signalReference.setItemText(0, _translate("MainWindow", "None"))
        self.limitsButton.setText(_translate("MainWindow", "..."))
        self.filterReference.setItemText(0, _translate("MainWindow", "None"))
        self.showValuesButton.setText(_translate("MainWindow", "..."))
        self.signalSelectorLabel.setText(_translate("MainWindow", "Signal"))
        self.startTimeLabel.setText(_translate("MainWindow", "Start"))
        self.headroomLabel.setText(_translate("MainWindow", "Headroom"))
        self.waveformIsFiltered.setText(_translate("MainWindow", "Filtered?"))
        self.startTime.setDisplayFormat(_translate("MainWindow", "HH:mm:ss.zzz"))
        self.endTime.setDisplayFormat(_translate("MainWindow", "HH:mm:ss.zzz"))
        self.applyWaveformLimitsButton.setText(_translate("MainWindow", "..."))
        self.resetWaveformLimitsButton.setText(_translate("MainWindow", "..."))
        self.zoomInButton.setText(_translate("MainWindow", "..."))
        self.zoomOutButton.setText(_translate("MainWindow", "..."))
        self.endTimeLabel.setText(_translate("MainWindow", "End"))
        self.label.setText(_translate("MainWindow", "Y Range"))
        self.filtersLabel.setText(_translate("MainWindow", "Filters"))
        self.signalsLabel.setText(_translate("MainWindow", "Signals"))
        self.addSignalButton.setText(_translate("MainWindow", "Add"))
        self.addSignalButton.setShortcut(_translate("MainWindow", "Ctrl+Shift+="))
        self.preset1Button.setText(_translate("MainWindow", "Preset 1"))
        self.preset1Button.setShortcut(_translate("MainWindow", "Ctrl+Shift+1"))
        self.preset2Button.setText(_translate("MainWindow", "Preset 2"))
        self.preset2Button.setShortcut(_translate("MainWindow", "Ctrl+Shift+2"))
        self.addFilterButton.setText(_translate("MainWindow", "Add"))
        self.addFilterButton.setShortcut(_translate("MainWindow", "Ctrl+="))
        self.preset3Button.setText(_translate("MainWindow", "Preset 3"))
        self.preset3Button.setShortcut(_translate("MainWindow", "Ctrl+Shift+3"))
        self.editFilterButton.setText(_translate("MainWindow", "Edit"))
        self.deleteFilterButton.setText(_translate("MainWindow", "Delete"))
        self.deleteFilterButton.setShortcut(_translate("MainWindow", "Ctrl+-"))
        self.deleteSignalButton.setText(_translate("MainWindow", "Delete"))
        self.showFiltersLabel.setText(_translate("MainWindow", "Show Filters?"))
        self.linkSignalButton.setText(_translate("MainWindow", "Link"))
        self.linkSignalButton.setShortcut(_translate("MainWindow", "Ctrl+Shift+L"))
        self.showSignalsLabel.setText(_translate("MainWindow", "Show Signals?"))
        self.showLegend.setText(_translate("MainWindow", "Show Legend"))
        self.equalEnergyTilt.setText(_translate("MainWindow", "+3dB/octave adjustment?"))
        self.menuHelp.setTitle(_translate("MainWindow", "&Help"))
        self.menuSettings.setTitle(_translate("MainWindow", "&Settings"))
        self.menuFile.setTitle(_translate("MainWindow", "&File"))
        self.menuPresets.setTitle(_translate("MainWindow", "&Presets"))
        self.menuLoad.setTitle(_translate("MainWindow", "&Load"))
        self.menu_Clear.setTitle(_translate("MainWindow", "&Clear"))
        self.menu_Store.setTitle(_translate("MainWindow", "&Store"))
        self.menu_Tools.setTitle(_translate("MainWindow", "&Tools"))
        self.actionShow_Logs.setText(_translate("MainWindow", "Show &Logs"))
        self.actionShow_Logs.setShortcut(_translate("MainWindow", "Ctrl+L"))
        self.actionPreferences.setText(_translate("MainWindow", "&Preferences"))
        self.actionPreferences.setShortcut(_translate("MainWindow", "Ctrl+P"))
        self.actionExtract_Audio.setText(_translate("MainWindow", "&Extract Audio"))
        self.actionExtract_Audio.setShortcut(_translate("MainWindow", "Ctrl+E"))
        self.actionPresets.setText(_translate("MainWindow", "Presets"))
        self.actionClear_Preset_3.setText(_translate("MainWindow", "Clear Preset 3"))
        self.actionSave_Chart.setText(_translate("MainWindow", "Save &Chart"))
        self.actionSave_Chart.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.actionExport_Biquad.setText(_translate("MainWindow", "Export &Biquad"))
        self.actionExport_Biquad.setShortcut(_translate("MainWindow", "Ctrl+B"))
        self.actionSave_Filter.setText(_translate("MainWindow", "Save &Filter"))
        self.actionLoad_Filter.setText(_translate("MainWindow", "Load Filte&r"))
        self.action_load_preset_1.setText(_translate("MainWindow", "&1"))
        self.action_load_preset_2.setText(_translate("MainWindow", "&2"))
        self.action_load_preset_3.setText(_translate("MainWindow", "&3"))
        self.action_clear_preset_1.setText(_translate("MainWindow", "&1"))
        self.action_clear_preset_2.setText(_translate("MainWindow", "&2"))
        self.action_clear_preset_3.setText(_translate("MainWindow", "&3"))
        self.action_store_preset_1.setText(_translate("MainWindow", "&1"))
        self.action_store_preset_1.setShortcut(_translate("MainWindow", "Ctrl+Alt+1"))
        self.action_store_preset_2.setText(_translate("MainWindow", "&2"))
        self.action_store_preset_2.setShortcut(_translate("MainWindow", "Ctrl+Alt+2"))
        self.action_store_preset_3.setText(_translate("MainWindow", "&3"))
        self.action_store_preset_3.setShortcut(_translate("MainWindow", "Ctrl+Alt+3"))
        self.actionExport_FRD.setText(_translate("MainWindow", "Export &FRD"))
        self.action_Save_Project.setText(_translate("MainWindow", "&Save Project"))
        self.action_Load_Project.setText(_translate("MainWindow", "&Load Project"))
        self.actionSave_Signal.setText(_translate("MainWindow", "Save Si&gnal"))
        self.actionLoad_Signal.setText(_translate("MainWindow", "Loa&d Signal"))
        self.actionClear_Project.setText(_translate("MainWindow", "Clear Project"))
        self.actionAnalyse_Audio.setText(_translate("MainWindow", "&Analyse Audio"))
        self.actionAnalyse_Audio.setShortcut(_translate("MainWindow", "Ctrl+A"))
        self.action_Batch_Extract.setText(_translate("MainWindow", "&Batch Extract"))
        self.action_Batch_Extract.setShortcut(_translate("MainWindow", "Ctrl+Shift+E"))
        self.actionSave_Report.setText(_translate("MainWindow", "Save Repor&t"))
        self.actionSave_Report.setShortcut(_translate("MainWindow", "Ctrl+R"))
        self.actionAbout.setText(_translate("MainWindow", "About"))
        self.action_Remux_Audio.setText(_translate("MainWindow", "&Remux Audio"))
        self.action_Remux_Audio.setShortcut(_translate("MainWindow", "Ctrl+U"))

from app import PlotWidgetWithDateAxis
from mpl import MplWidget
