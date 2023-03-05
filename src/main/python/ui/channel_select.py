# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'channel_select.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_channelSelectDialog(object):
    def setupUi(self, channelSelectDialog):
        channelSelectDialog.setObjectName("channelSelectDialog")
        channelSelectDialog.resize(337, 250)
        self.gridLayout = QtWidgets.QGridLayout(channelSelectDialog)
        self.gridLayout.setObjectName("gridLayout")
        self.buttonBox = QtWidgets.QDialogButtonBox(channelSelectDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Save)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 2, 1, 1, 1)
        self.channelListLabel = QtWidgets.QLabel(channelSelectDialog)
        self.channelListLabel.setObjectName("channelListLabel")
        self.gridLayout.addWidget(self.channelListLabel, 1, 0, 1, 1)
        self.channelList = QtWidgets.QListWidget(channelSelectDialog)
        self.channelList.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.channelList.setObjectName("channelList")
        self.gridLayout.addWidget(self.channelList, 1, 1, 1, 1)
        self.lfeChannelLabel = QtWidgets.QLabel(channelSelectDialog)
        self.lfeChannelLabel.setObjectName("lfeChannelLabel")
        self.gridLayout.addWidget(self.lfeChannelLabel, 0, 0, 1, 1)
        self.lfeChannel = QtWidgets.QComboBox(channelSelectDialog)
        self.lfeChannel.setObjectName("lfeChannel")
        self.gridLayout.addWidget(self.lfeChannel, 0, 1, 1, 1)

        self.retranslateUi(channelSelectDialog)
        self.buttonBox.accepted.connect(channelSelectDialog.accept) # type: ignore
        self.buttonBox.rejected.connect(channelSelectDialog.reject) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(channelSelectDialog)

    def retranslateUi(self, channelSelectDialog):
        _translate = QtCore.QCoreApplication.translate
        channelSelectDialog.setWindowTitle(_translate("channelSelectDialog", "Add/Edit Filter"))
        self.channelListLabel.setText(_translate("channelSelectDialog", "Channels"))
        self.lfeChannelLabel.setText(_translate("channelSelectDialog", "LFE"))
