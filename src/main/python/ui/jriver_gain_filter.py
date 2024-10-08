# Form implementation generated from reading ui file 'jriver_gain_filter.ui'
#
# Created by: PyQt6 UI code generator 6.7.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_jriverGainDialog(object):
    def setupUi(self, jriverGainDialog):
        jriverGainDialog.setObjectName("jriverGainDialog")
        jriverGainDialog.resize(262, 276)
        self.verticalLayout = QtWidgets.QVBoxLayout(jriverGainDialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.gain = QtWidgets.QDoubleSpinBox(parent=jriverGainDialog)
        self.gain.setDecimals(2)
        self.gain.setMinimum(-120.0)
        self.gain.setMaximum(120.0)
        self.gain.setSingleStep(0.01)
        self.gain.setObjectName("gain")
        self.gridLayout.addWidget(self.gain, 0, 1, 1, 1)
        self.channelListLabel = QtWidgets.QLabel(parent=jriverGainDialog)
        self.channelListLabel.setObjectName("channelListLabel")
        self.gridLayout.addWidget(self.channelListLabel, 2, 0, 1, 1)
        self.label = QtWidgets.QLabel(parent=jriverGainDialog)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.channelList = QtWidgets.QListWidget(parent=jriverGainDialog)
        self.channelList.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)
        self.channelList.setObjectName("channelList")
        self.gridLayout.addWidget(self.channelList, 2, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.buttonBox = QtWidgets.QDialogButtonBox(parent=jriverGainDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.StandardButton.Cancel|QtWidgets.QDialogButtonBox.StandardButton.Save)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(jriverGainDialog)
        self.buttonBox.accepted.connect(jriverGainDialog.accept) # type: ignore
        self.buttonBox.rejected.connect(jriverGainDialog.reject) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(jriverGainDialog)
        jriverGainDialog.setTabOrder(self.gain, self.channelList)

    def retranslateUi(self, jriverGainDialog):
        _translate = QtCore.QCoreApplication.translate
        jriverGainDialog.setWindowTitle(_translate("jriverGainDialog", "Add/Edit Filter"))
        self.gain.setSuffix(_translate("jriverGainDialog", " dB"))
        self.channelListLabel.setText(_translate("jriverGainDialog", "Channels"))
        self.label.setText(_translate("jriverGainDialog", "Gain"))
