# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'timing.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_timingDialog(object):
    def setupUi(self, timingDialog):
        timingDialog.setObjectName("timingDialog")
        timingDialog.resize(1054, 821)
        self.outerLayout = QtWidgets.QVBoxLayout(timingDialog)
        self.outerLayout.setObjectName("outerLayout")
        self.innerLayout = QtWidgets.QHBoxLayout()
        self.innerLayout.setObjectName("innerLayout")
        self.chartView = MplWidget(timingDialog)
        self.chartView.setObjectName("chartView")
        self.innerLayout.addWidget(self.chartView)
        self.buttonLayout = QtWidgets.QVBoxLayout()
        self.buttonLayout.setObjectName("buttonLayout")
        self.limitsButton = QtWidgets.QToolButton(timingDialog)
        self.limitsButton.setObjectName("limitsButton")
        self.buttonLayout.addWidget(self.limitsButton)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.buttonLayout.addItem(spacerItem)
        self.innerLayout.addLayout(self.buttonLayout)
        self.outerLayout.addLayout(self.innerLayout)
        self.bottomLayout = QtWidgets.QHBoxLayout()
        self.bottomLayout.setObjectName("bottomLayout")
        self.selector = QtWidgets.QFrame(timingDialog)
        self.selector.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.selector.setFrameShadow(QtWidgets.QFrame.Raised)
        self.selector.setObjectName("selector")
        self.bottomLayout.addWidget(self.selector)
        self.checkBox = QtWidgets.QCheckBox(timingDialog)
        self.checkBox.setObjectName("checkBox")
        self.bottomLayout.addWidget(self.checkBox)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.bottomLayout.addItem(spacerItem1)
        self.buttonBox = QtWidgets.QDialogButtonBox(timingDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Close)
        self.buttonBox.setObjectName("buttonBox")
        self.bottomLayout.addWidget(self.buttonBox)
        self.outerLayout.addLayout(self.bottomLayout)

        self.retranslateUi(timingDialog)
        self.buttonBox.accepted.connect(timingDialog.accept) # type: ignore
        self.buttonBox.rejected.connect(timingDialog.reject) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(timingDialog)

    def retranslateUi(self, timingDialog):
        _translate = QtCore.QCoreApplication.translate
        timingDialog.setWindowTitle(_translate("timingDialog", "Timing"))
        self.limitsButton.setText(_translate("timingDialog", "..."))
        self.checkBox.setText(_translate("timingDialog", "L"))
from mpl import MplWidget
