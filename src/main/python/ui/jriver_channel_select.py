# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'jriver_channel_select.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_jriverChannelSelectDialog(object):
    def setupUi(self, jriverChannelSelectDialog):
        jriverChannelSelectDialog.setObjectName("jriverChannelSelectDialog")
        jriverChannelSelectDialog.resize(249, 250)
        self.gridLayout = QtWidgets.QGridLayout(jriverChannelSelectDialog)
        self.gridLayout.setObjectName("gridLayout")
        self.buttonBox = QtWidgets.QDialogButtonBox(jriverChannelSelectDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Save)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 1, 1, 1, 1)
        self.channelListLabel = QtWidgets.QLabel(jriverChannelSelectDialog)
        self.channelListLabel.setObjectName("channelListLabel")
        self.gridLayout.addWidget(self.channelListLabel, 0, 0, 1, 1)
        self.channelList = QtWidgets.QListWidget(jriverChannelSelectDialog)
        self.channelList.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.channelList.setObjectName("channelList")
        self.gridLayout.addWidget(self.channelList, 0, 1, 1, 1)

        self.retranslateUi(jriverChannelSelectDialog)
        self.buttonBox.accepted.connect(jriverChannelSelectDialog.accept)
        self.buttonBox.rejected.connect(jriverChannelSelectDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(jriverChannelSelectDialog)

    def retranslateUi(self, jriverChannelSelectDialog):
        _translate = QtCore.QCoreApplication.translate
        jriverChannelSelectDialog.setWindowTitle(_translate("jriverChannelSelectDialog", "Add/Edit Filter"))
        self.channelListLabel.setText(_translate("jriverChannelSelectDialog", "Channels"))
