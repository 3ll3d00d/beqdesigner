# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'synchtp1.ui'
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


class Ui_syncHtp1Dialog(object):
    def setupUi(self, syncHtp1Dialog):
        if syncHtp1Dialog.objectName():
            syncHtp1Dialog.setObjectName(u"syncHtp1Dialog")
        syncHtp1Dialog.resize(1416, 700)
        self.gridLayout = QGridLayout(syncHtp1Dialog)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout_3 = QGridLayout()
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.resyncFilters = QToolButton(syncHtp1Dialog)
        self.resyncFilters.setObjectName(u"resyncFilters")

        self.gridLayout_3.addWidget(self.resyncFilters, 1, 2, 1, 1)

        self.showDetailsButton = QToolButton(syncHtp1Dialog)
        self.showDetailsButton.setObjectName(u"showDetailsButton")

        self.gridLayout_3.addWidget(self.showDetailsButton, 6, 2, 1, 1)

        self.selectBeqButton = QToolButton(syncHtp1Dialog)
        self.selectBeqButton.setObjectName(u"selectBeqButton")

        self.gridLayout_3.addWidget(self.selectBeqButton, 3, 2, 1, 1)

        self.syncLayout = QHBoxLayout()
        self.syncLayout.setObjectName(u"syncLayout")
        self.selectNoneButton = QPushButton(syncHtp1Dialog)
        self.selectNoneButton.setObjectName(u"selectNoneButton")

        self.syncLayout.addWidget(self.selectNoneButton)

        self.selectAllButton = QPushButton(syncHtp1Dialog)
        self.selectAllButton.setObjectName(u"selectAllButton")

        self.syncLayout.addWidget(self.selectAllButton)

        self.applyFiltersButton = QPushButton(syncHtp1Dialog)
        self.applyFiltersButton.setObjectName(u"applyFiltersButton")

        self.syncLayout.addWidget(self.applyFiltersButton)

        self.syncLayout.setStretch(2, 1)

        self.gridLayout_3.addLayout(self.syncLayout, 6, 1, 1, 1)

        self.filterView = QTableView(syncHtp1Dialog)
        self.filterView.setObjectName(u"filterView")
        self.filterView.setSelectionMode(QAbstractItemView.MultiSelection)

        self.gridLayout_3.addWidget(self.filterView, 2, 1, 1, 1)

        self.gridLayout_2 = QGridLayout()
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.addFilterButton = QPushButton(syncHtp1Dialog)
        self.addFilterButton.setObjectName(u"addFilterButton")

        self.gridLayout_2.addWidget(self.addFilterButton, 0, 0, 1, 1)

        self.removeFilterButton = QPushButton(syncHtp1Dialog)
        self.removeFilterButton.setObjectName(u"removeFilterButton")

        self.gridLayout_2.addWidget(self.removeFilterButton, 0, 1, 1, 1)

        self.loadFromSignalsButton = QPushButton(syncHtp1Dialog)
        self.loadFromSignalsButton.setObjectName(u"loadFromSignalsButton")

        self.gridLayout_2.addWidget(self.loadFromSignalsButton, 0, 2, 1, 1)


        self.gridLayout_3.addLayout(self.gridLayout_2, 4, 1, 1, 1)

        self.createPulsesButton = QToolButton(syncHtp1Dialog)
        self.createPulsesButton.setObjectName(u"createPulsesButton")
        self.createPulsesButton.setToolTipDuration(5000)

        self.gridLayout_3.addWidget(self.createPulsesButton, 4, 2, 1, 1)

        self.ipAddressLabel = QLabel(syncHtp1Dialog)
        self.ipAddressLabel.setObjectName(u"ipAddressLabel")

        self.gridLayout_3.addWidget(self.ipAddressLabel, 0, 0, 1, 1)

        self.filterLabel = QLabel(syncHtp1Dialog)
        self.filterLabel.setObjectName(u"filterLabel")

        self.gridLayout_3.addWidget(self.filterLabel, 2, 0, 1, 1)

        self.ipAddress = QLineEdit(syncHtp1Dialog)
        self.ipAddress.setObjectName(u"ipAddress")

        self.gridLayout_3.addWidget(self.ipAddress, 0, 1, 1, 1)

        self.connectButtonLayout = QVBoxLayout()
        self.connectButtonLayout.setObjectName(u"connectButtonLayout")
        self.connectButton = QToolButton(syncHtp1Dialog)
        self.connectButton.setObjectName(u"connectButton")

        self.connectButtonLayout.addWidget(self.connectButton)

        self.disconnectButton = QToolButton(syncHtp1Dialog)
        self.disconnectButton.setObjectName(u"disconnectButton")

        self.connectButtonLayout.addWidget(self.disconnectButton)


        self.gridLayout_3.addLayout(self.connectButtonLayout, 0, 2, 1, 1)

        self.beqLabel = QLabel(syncHtp1Dialog)
        self.beqLabel.setObjectName(u"beqLabel")

        self.gridLayout_3.addWidget(self.beqLabel, 3, 0, 1, 1)

        self.deleteFiltersButton = QToolButton(syncHtp1Dialog)
        self.deleteFiltersButton.setObjectName(u"deleteFiltersButton")

        self.gridLayout_3.addWidget(self.deleteFiltersButton, 2, 2, 1, 1)

        self.filterMapping = QListWidget(syncHtp1Dialog)
        self.filterMapping.setObjectName(u"filterMapping")
        self.filterMapping.setSelectionMode(QAbstractItemView.MultiSelection)

        self.gridLayout_3.addWidget(self.filterMapping, 5, 1, 1, 1)

        self.filtersetLabel = QLabel(syncHtp1Dialog)
        self.filtersetLabel.setObjectName(u"filtersetLabel")

        self.gridLayout_3.addWidget(self.filtersetLabel, 1, 0, 1, 1)

        self.beqFile = QLineEdit(syncHtp1Dialog)
        self.beqFile.setObjectName(u"beqFile")
        self.beqFile.setReadOnly(True)

        self.gridLayout_3.addWidget(self.beqFile, 3, 1, 1, 1)

        self.filtersetSelector = QComboBox(syncHtp1Dialog)
        self.filtersetSelector.setObjectName(u"filtersetSelector")

        self.gridLayout_3.addWidget(self.filtersetSelector, 1, 1, 1, 1)

        self.filterMappingLabel = QLabel(syncHtp1Dialog)
        self.filterMappingLabel.setObjectName(u"filterMappingLabel")

        self.gridLayout_3.addWidget(self.filterMappingLabel, 5, 0, 1, 1)


        self.gridLayout.addLayout(self.gridLayout_3, 0, 0, 1, 1)

        self.previewChart = MplWidget(syncHtp1Dialog)
        self.previewChart.setObjectName(u"previewChart")

        self.gridLayout.addWidget(self.previewChart, 0, 1, 1, 1)

        self.graphButtonsLayout = QVBoxLayout()
        self.graphButtonsLayout.setObjectName(u"graphButtonsLayout")
        self.limitsButton = QToolButton(syncHtp1Dialog)
        self.limitsButton.setObjectName(u"limitsButton")

        self.graphButtonsLayout.addWidget(self.limitsButton)

        self.fullRangeButton = QToolButton(syncHtp1Dialog)
        self.fullRangeButton.setObjectName(u"fullRangeButton")

        self.graphButtonsLayout.addWidget(self.fullRangeButton)

        self.subOnlyButton = QToolButton(syncHtp1Dialog)
        self.subOnlyButton.setObjectName(u"subOnlyButton")

        self.graphButtonsLayout.addWidget(self.subOnlyButton)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.graphButtonsLayout.addItem(self.verticalSpacer)


        self.gridLayout.addLayout(self.graphButtonsLayout, 0, 2, 1, 1)

        self.buttonBox = QDialogButtonBox(syncHtp1Dialog)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Close)

        self.gridLayout.addWidget(self.buttonBox, 1, 0, 1, 3)

        self.gridLayout.setColumnStretch(0, 1)
        self.gridLayout.setColumnStretch(1, 3)

        self.retranslateUi(syncHtp1Dialog)
        self.buttonBox.accepted.connect(syncHtp1Dialog.accept)
        self.buttonBox.rejected.connect(syncHtp1Dialog.reject)
        self.selectBeqButton.clicked.connect(syncHtp1Dialog.select_beq)
        self.deleteFiltersButton.clicked.connect(syncHtp1Dialog.clear_filters)
        self.filtersetSelector.currentTextChanged.connect(syncHtp1Dialog.display_filterset)
        self.applyFiltersButton.clicked.connect(syncHtp1Dialog.send_filters_to_device)
        self.connectButton.clicked.connect(syncHtp1Dialog.connect_htp1)
        self.disconnectButton.clicked.connect(syncHtp1Dialog.disconnect_htp1)
        self.resyncFilters.clicked.connect(syncHtp1Dialog.resync_filters)
        self.addFilterButton.clicked.connect(syncHtp1Dialog.add_filter)
        self.removeFilterButton.clicked.connect(syncHtp1Dialog.remove_filter)
        self.limitsButton.clicked.connect(syncHtp1Dialog.show_limits)
        self.showDetailsButton.clicked.connect(syncHtp1Dialog.show_sync_details)
        self.createPulsesButton.clicked.connect(syncHtp1Dialog.create_pulses)
        self.fullRangeButton.clicked.connect(syncHtp1Dialog.show_full_range)
        self.subOnlyButton.clicked.connect(syncHtp1Dialog.show_sub_only)
        self.loadFromSignalsButton.clicked.connect(syncHtp1Dialog.load_from_signals)
        self.selectNoneButton.clicked.connect(syncHtp1Dialog.clear_sync_selection)
        self.selectAllButton.clicked.connect(syncHtp1Dialog.select_all_for_sync)
        self.filterMapping.itemSelectionChanged.connect(syncHtp1Dialog.on_signal_selected)

        QMetaObject.connectSlotsByName(syncHtp1Dialog)
    # setupUi

    def retranslateUi(self, syncHtp1Dialog):
        syncHtp1Dialog.setWindowTitle(QCoreApplication.translate("syncHtp1Dialog", u"HTP-1 Filter Manager", None))
#if QT_CONFIG(tooltip)
        self.resyncFilters.setToolTip(QCoreApplication.translate("syncHtp1Dialog", u"Resync from HTP-1", None))
#endif // QT_CONFIG(tooltip)
        self.resyncFilters.setText(QCoreApplication.translate("syncHtp1Dialog", u"...", None))
        self.showDetailsButton.setText(QCoreApplication.translate("syncHtp1Dialog", u"...", None))
        self.selectBeqButton.setText(QCoreApplication.translate("syncHtp1Dialog", u"...", None))
        self.selectNoneButton.setText(QCoreApplication.translate("syncHtp1Dialog", u"Clear", None))
        self.selectAllButton.setText(QCoreApplication.translate("syncHtp1Dialog", u"Select All", None))
        self.applyFiltersButton.setText(QCoreApplication.translate("syncHtp1Dialog", u"Sync to HTP-1", None))
        self.addFilterButton.setText(QCoreApplication.translate("syncHtp1Dialog", u"Add Filter", None))
        self.removeFilterButton.setText(QCoreApplication.translate("syncHtp1Dialog", u"Remove Filter", None))
        self.loadFromSignalsButton.setText(QCoreApplication.translate("syncHtp1Dialog", u"Load from Signals", None))
#if QT_CONFIG(tooltip)
        self.createPulsesButton.setToolTip("")
#endif // QT_CONFIG(tooltip)
        self.createPulsesButton.setText(QCoreApplication.translate("syncHtp1Dialog", u"...", None))
        self.ipAddressLabel.setText(QCoreApplication.translate("syncHtp1Dialog", u"IP Address", None))
        self.filterLabel.setText(QCoreApplication.translate("syncHtp1Dialog", u"Filters", None))
        self.ipAddress.setInputMask(QCoreApplication.translate("syncHtp1Dialog", u"000.000.000.000:00000", None))
#if QT_CONFIG(tooltip)
        self.connectButton.setToolTip(QCoreApplication.translate("syncHtp1Dialog", u"Connect", None))
#endif // QT_CONFIG(tooltip)
        self.connectButton.setText(QCoreApplication.translate("syncHtp1Dialog", u"...", None))
#if QT_CONFIG(tooltip)
        self.disconnectButton.setToolTip(QCoreApplication.translate("syncHtp1Dialog", u"Disconnect", None))
#endif // QT_CONFIG(tooltip)
        self.disconnectButton.setText(QCoreApplication.translate("syncHtp1Dialog", u"...", None))
        self.beqLabel.setText(QCoreApplication.translate("syncHtp1Dialog", u"BEQ", None))
        self.deleteFiltersButton.setText(QCoreApplication.translate("syncHtp1Dialog", u"...", None))
        self.filtersetLabel.setText(QCoreApplication.translate("syncHtp1Dialog", u"Channel", None))
        self.filterMappingLabel.setText(QCoreApplication.translate("syncHtp1Dialog", u"Signal\n"
"Mapping", None))
        self.limitsButton.setText(QCoreApplication.translate("syncHtp1Dialog", u"...", None))
        self.fullRangeButton.setText(QCoreApplication.translate("syncHtp1Dialog", u"...", None))
        self.subOnlyButton.setText(QCoreApplication.translate("syncHtp1Dialog", u"...", None))
    # retranslateUi

