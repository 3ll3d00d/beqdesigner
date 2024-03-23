# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'synchtp1.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_syncHtp1Dialog(object):
    def setupUi(self, syncHtp1Dialog):
        syncHtp1Dialog.setObjectName("syncHtp1Dialog")
        syncHtp1Dialog.resize(1416, 700)
        self.gridLayout = QtWidgets.QGridLayout(syncHtp1Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.resyncFilters = QtWidgets.QToolButton(syncHtp1Dialog)
        self.resyncFilters.setObjectName("resyncFilters")
        self.gridLayout_3.addWidget(self.resyncFilters, 1, 2, 1, 1)
        self.showDetailsButton = QtWidgets.QToolButton(syncHtp1Dialog)
        self.showDetailsButton.setObjectName("showDetailsButton")
        self.gridLayout_3.addWidget(self.showDetailsButton, 6, 2, 1, 1)
        self.selectBeqButton = QtWidgets.QToolButton(syncHtp1Dialog)
        self.selectBeqButton.setObjectName("selectBeqButton")
        self.gridLayout_3.addWidget(self.selectBeqButton, 3, 2, 1, 1)
        self.syncLayout = QtWidgets.QHBoxLayout()
        self.syncLayout.setObjectName("syncLayout")
        self.selectNoneButton = QtWidgets.QPushButton(syncHtp1Dialog)
        self.selectNoneButton.setObjectName("selectNoneButton")
        self.syncLayout.addWidget(self.selectNoneButton)
        self.selectAllButton = QtWidgets.QPushButton(syncHtp1Dialog)
        self.selectAllButton.setObjectName("selectAllButton")
        self.syncLayout.addWidget(self.selectAllButton)
        self.applyFiltersButton = QtWidgets.QPushButton(syncHtp1Dialog)
        self.applyFiltersButton.setObjectName("applyFiltersButton")
        self.syncLayout.addWidget(self.applyFiltersButton)
        self.autoSyncButton = QtWidgets.QToolButton(syncHtp1Dialog)
        self.autoSyncButton.setCheckable(True)
        self.autoSyncButton.setObjectName("autoSyncButton")
        self.syncLayout.addWidget(self.autoSyncButton)
        self.syncLayout.setStretch(2, 1)
        self.gridLayout_3.addLayout(self.syncLayout, 6, 1, 1, 1)
        self.filterView = QtWidgets.QTableView(syncHtp1Dialog)
        self.filterView.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.filterView.setObjectName("filterView")
        self.gridLayout_3.addWidget(self.filterView, 2, 1, 1, 1)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.addFilterButton = QtWidgets.QPushButton(syncHtp1Dialog)
        self.addFilterButton.setObjectName("addFilterButton")
        self.gridLayout_2.addWidget(self.addFilterButton, 0, 0, 1, 1)
        self.removeFilterButton = QtWidgets.QPushButton(syncHtp1Dialog)
        self.removeFilterButton.setObjectName("removeFilterButton")
        self.gridLayout_2.addWidget(self.removeFilterButton, 0, 1, 1, 1)
        self.loadFromSignalsButton = QtWidgets.QPushButton(syncHtp1Dialog)
        self.loadFromSignalsButton.setObjectName("loadFromSignalsButton")
        self.gridLayout_2.addWidget(self.loadFromSignalsButton, 0, 2, 1, 1)
        self.gridLayout_3.addLayout(self.gridLayout_2, 4, 1, 1, 1)
        self.createPulsesButton = QtWidgets.QToolButton(syncHtp1Dialog)
        self.createPulsesButton.setToolTip("")
        self.createPulsesButton.setToolTipDuration(5000)
        self.createPulsesButton.setObjectName("createPulsesButton")
        self.gridLayout_3.addWidget(self.createPulsesButton, 4, 2, 1, 1)
        self.ipAddressLabel = QtWidgets.QLabel(syncHtp1Dialog)
        self.ipAddressLabel.setObjectName("ipAddressLabel")
        self.gridLayout_3.addWidget(self.ipAddressLabel, 0, 0, 1, 1)
        self.filterLabel = QtWidgets.QLabel(syncHtp1Dialog)
        self.filterLabel.setObjectName("filterLabel")
        self.gridLayout_3.addWidget(self.filterLabel, 2, 0, 1, 1)
        self.ipAddress = QtWidgets.QLineEdit(syncHtp1Dialog)
        self.ipAddress.setObjectName("ipAddress")
        self.gridLayout_3.addWidget(self.ipAddress, 0, 1, 1, 1)
        self.connectButtonLayout = QtWidgets.QVBoxLayout()
        self.connectButtonLayout.setObjectName("connectButtonLayout")
        self.connectButton = QtWidgets.QToolButton(syncHtp1Dialog)
        self.connectButton.setObjectName("connectButton")
        self.connectButtonLayout.addWidget(self.connectButton)
        self.disconnectButton = QtWidgets.QToolButton(syncHtp1Dialog)
        self.disconnectButton.setObjectName("disconnectButton")
        self.connectButtonLayout.addWidget(self.disconnectButton)
        self.gridLayout_3.addLayout(self.connectButtonLayout, 0, 2, 1, 1)
        self.beqLabel = QtWidgets.QLabel(syncHtp1Dialog)
        self.beqLabel.setObjectName("beqLabel")
        self.gridLayout_3.addWidget(self.beqLabel, 3, 0, 1, 1)
        self.toolButtonsLayout = QtWidgets.QVBoxLayout()
        self.toolButtonsLayout.setObjectName("toolButtonsLayout")
        self.editFilterButton = QtWidgets.QToolButton(syncHtp1Dialog)
        self.editFilterButton.setObjectName("editFilterButton")
        self.toolButtonsLayout.addWidget(self.editFilterButton)
        self.deleteFiltersButton = QtWidgets.QToolButton(syncHtp1Dialog)
        self.deleteFiltersButton.setObjectName("deleteFiltersButton")
        self.toolButtonsLayout.addWidget(self.deleteFiltersButton)
        self.gridLayout_3.addLayout(self.toolButtonsLayout, 2, 2, 1, 1)
        self.filterMapping = QtWidgets.QListWidget(syncHtp1Dialog)
        self.filterMapping.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.filterMapping.setObjectName("filterMapping")
        self.gridLayout_3.addWidget(self.filterMapping, 5, 1, 1, 1)
        self.filtersetLabel = QtWidgets.QLabel(syncHtp1Dialog)
        self.filtersetLabel.setObjectName("filtersetLabel")
        self.gridLayout_3.addWidget(self.filtersetLabel, 1, 0, 1, 1)
        self.beqFile = QtWidgets.QLineEdit(syncHtp1Dialog)
        self.beqFile.setReadOnly(True)
        self.beqFile.setObjectName("beqFile")
        self.gridLayout_3.addWidget(self.beqFile, 3, 1, 1, 1)
        self.filtersetSelector = QtWidgets.QComboBox(syncHtp1Dialog)
        self.filtersetSelector.setObjectName("filtersetSelector")
        self.gridLayout_3.addWidget(self.filtersetSelector, 1, 1, 1, 1)
        self.filterMappingLabel = QtWidgets.QLabel(syncHtp1Dialog)
        self.filterMappingLabel.setObjectName("filterMappingLabel")
        self.gridLayout_3.addWidget(self.filterMappingLabel, 5, 0, 1, 1)
        self.gridLayout.addLayout(self.gridLayout_3, 0, 0, 1, 1)
        self.previewChart = MplWidget(syncHtp1Dialog)
        self.previewChart.setObjectName("previewChart")
        self.gridLayout.addWidget(self.previewChart, 0, 1, 1, 1)
        self.graphButtonsLayout = QtWidgets.QVBoxLayout()
        self.graphButtonsLayout.setObjectName("graphButtonsLayout")
        self.limitsButton = QtWidgets.QToolButton(syncHtp1Dialog)
        self.limitsButton.setObjectName("limitsButton")
        self.graphButtonsLayout.addWidget(self.limitsButton)
        self.fullRangeButton = QtWidgets.QToolButton(syncHtp1Dialog)
        self.fullRangeButton.setObjectName("fullRangeButton")
        self.graphButtonsLayout.addWidget(self.fullRangeButton)
        self.subOnlyButton = QtWidgets.QToolButton(syncHtp1Dialog)
        self.subOnlyButton.setObjectName("subOnlyButton")
        self.graphButtonsLayout.addWidget(self.subOnlyButton)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.graphButtonsLayout.addItem(spacerItem)
        self.gridLayout.addLayout(self.graphButtonsLayout, 0, 2, 1, 1)
        self.buttonBox = QtWidgets.QDialogButtonBox(syncHtp1Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Close)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 1, 0, 1, 3)
        self.gridLayout.setColumnStretch(0, 1)
        self.gridLayout.setColumnStretch(1, 3)

        self.retranslateUi(syncHtp1Dialog)
        self.buttonBox.accepted.connect(syncHtp1Dialog.accept) # type: ignore
        self.buttonBox.rejected.connect(syncHtp1Dialog.reject) # type: ignore
        self.selectBeqButton.clicked.connect(syncHtp1Dialog.select_beq) # type: ignore
        self.deleteFiltersButton.clicked.connect(syncHtp1Dialog.clear_filters) # type: ignore
        self.filtersetSelector.currentTextChanged['QString'].connect(syncHtp1Dialog.display_filterset) # type: ignore
        self.applyFiltersButton.clicked.connect(syncHtp1Dialog.send_filters_to_device) # type: ignore
        self.connectButton.clicked.connect(syncHtp1Dialog.connect_htp1) # type: ignore
        self.disconnectButton.clicked.connect(syncHtp1Dialog.disconnect_htp1) # type: ignore
        self.resyncFilters.clicked.connect(syncHtp1Dialog.resync_filters) # type: ignore
        self.addFilterButton.clicked.connect(syncHtp1Dialog.add_filter) # type: ignore
        self.removeFilterButton.clicked.connect(syncHtp1Dialog.remove_filter) # type: ignore
        self.limitsButton.clicked.connect(syncHtp1Dialog.show_limits) # type: ignore
        self.showDetailsButton.clicked.connect(syncHtp1Dialog.show_sync_details) # type: ignore
        self.createPulsesButton.clicked.connect(syncHtp1Dialog.create_pulses) # type: ignore
        self.fullRangeButton.clicked.connect(syncHtp1Dialog.show_full_range) # type: ignore
        self.subOnlyButton.clicked.connect(syncHtp1Dialog.show_sub_only) # type: ignore
        self.loadFromSignalsButton.clicked.connect(syncHtp1Dialog.load_from_signals) # type: ignore
        self.selectNoneButton.clicked.connect(syncHtp1Dialog.clear_sync_selection) # type: ignore
        self.selectAllButton.clicked.connect(syncHtp1Dialog.select_all_for_sync) # type: ignore
        self.filterMapping.itemSelectionChanged.connect(syncHtp1Dialog.on_signal_selected) # type: ignore
        self.editFilterButton.clicked.connect(syncHtp1Dialog.edit_filter) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(syncHtp1Dialog)

    def retranslateUi(self, syncHtp1Dialog):
        _translate = QtCore.QCoreApplication.translate
        syncHtp1Dialog.setWindowTitle(_translate("syncHtp1Dialog", "HTP-1 Filter Manager"))
        self.resyncFilters.setToolTip(_translate("syncHtp1Dialog", "Resync from HTP-1"))
        self.resyncFilters.setText(_translate("syncHtp1Dialog", "..."))
        self.showDetailsButton.setText(_translate("syncHtp1Dialog", "..."))
        self.selectBeqButton.setText(_translate("syncHtp1Dialog", "..."))
        self.selectNoneButton.setText(_translate("syncHtp1Dialog", "Clear"))
        self.selectAllButton.setText(_translate("syncHtp1Dialog", "Select All"))
        self.applyFiltersButton.setText(_translate("syncHtp1Dialog", "Sync to HTP-1"))
        self.autoSyncButton.setText(_translate("syncHtp1Dialog", "..."))
        self.addFilterButton.setText(_translate("syncHtp1Dialog", "Add Filter"))
        self.removeFilterButton.setText(_translate("syncHtp1Dialog", "Remove Filter"))
        self.loadFromSignalsButton.setText(_translate("syncHtp1Dialog", "Load from Signals"))
        self.createPulsesButton.setText(_translate("syncHtp1Dialog", "..."))
        self.ipAddressLabel.setText(_translate("syncHtp1Dialog", "IP Address"))
        self.filterLabel.setText(_translate("syncHtp1Dialog", "Filters"))
        self.ipAddress.setInputMask(_translate("syncHtp1Dialog", "000.000.000.000:00000"))
        self.connectButton.setToolTip(_translate("syncHtp1Dialog", "Connect"))
        self.connectButton.setText(_translate("syncHtp1Dialog", "..."))
        self.disconnectButton.setToolTip(_translate("syncHtp1Dialog", "Disconnect"))
        self.disconnectButton.setText(_translate("syncHtp1Dialog", "..."))
        self.beqLabel.setText(_translate("syncHtp1Dialog", "BEQ"))
        self.editFilterButton.setText(_translate("syncHtp1Dialog", "..."))
        self.deleteFiltersButton.setText(_translate("syncHtp1Dialog", "..."))
        self.filtersetLabel.setText(_translate("syncHtp1Dialog", "Channel"))
        self.filterMappingLabel.setText(_translate("syncHtp1Dialog", "Signal\n"
"Mapping"))
        self.limitsButton.setText(_translate("syncHtp1Dialog", "..."))
        self.fullRangeButton.setText(_translate("syncHtp1Dialog", "..."))
        self.subOnlyButton.setText(_translate("syncHtp1Dialog", "..."))
from mpl import MplWidget
