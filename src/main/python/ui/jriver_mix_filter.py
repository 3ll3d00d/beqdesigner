# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'jriver_mix_filter.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_jriverMixDialog(object):
    def setupUi(self, jriverMixDialog):
        jriverMixDialog.setObjectName("jriverMixDialog")
        jriverMixDialog.resize(263, 166)
        self.gridLayout = QtWidgets.QGridLayout(jriverMixDialog)
        self.gridLayout.setObjectName("gridLayout")
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 3, 1, 1, 1)
        self.gain = QtWidgets.QDoubleSpinBox(jriverMixDialog)
        self.gain.setMinimum(-120.0)
        self.gain.setMaximum(120.0)
        self.gain.setSingleStep(0.01)
        self.gain.setObjectName("gain")
        self.gridLayout.addWidget(self.gain, 2, 1, 1, 1)
        self.destinationLabel = QtWidgets.QLabel(jriverMixDialog)
        self.destinationLabel.setObjectName("destinationLabel")
        self.gridLayout.addWidget(self.destinationLabel, 1, 0, 1, 1)
        self.sourceLabel = QtWidgets.QLabel(jriverMixDialog)
        self.sourceLabel.setObjectName("sourceLabel")
        self.gridLayout.addWidget(self.sourceLabel, 0, 0, 1, 1)
        self.gainLabel = QtWidgets.QLabel(jriverMixDialog)
        self.gainLabel.setObjectName("gainLabel")
        self.gridLayout.addWidget(self.gainLabel, 2, 0, 1, 1)
        self.buttonBox = QtWidgets.QDialogButtonBox(jriverMixDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Save)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 4, 1, 1, 1)
        self.source = QtWidgets.QComboBox(jriverMixDialog)
        self.source.setObjectName("source")
        self.gridLayout.addWidget(self.source, 0, 1, 1, 1)
        self.destination = QtWidgets.QComboBox(jriverMixDialog)
        self.destination.setObjectName("destination")
        self.gridLayout.addWidget(self.destination, 1, 1, 1, 1)

        self.retranslateUi(jriverMixDialog)
        self.buttonBox.accepted.connect(jriverMixDialog.accept) # type: ignore
        self.buttonBox.rejected.connect(jriverMixDialog.reject) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(jriverMixDialog)
        jriverMixDialog.setTabOrder(self.source, self.destination)

    def retranslateUi(self, jriverMixDialog):
        _translate = QtCore.QCoreApplication.translate
        jriverMixDialog.setWindowTitle(_translate("jriverMixDialog", "Add/Edit Filter"))
        self.gain.setSuffix(_translate("jriverMixDialog", " dB"))
        self.destinationLabel.setText(_translate("jriverMixDialog", "Destination"))
        self.sourceLabel.setText(_translate("jriverMixDialog", "Source"))
        self.gainLabel.setText(_translate("jriverMixDialog", "Gain"))
