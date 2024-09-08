# Form implementation generated from reading ui file 'impulse.ui'
#
# Created by: PyQt6 UI code generator 6.7.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_impulseDialog(object):
    def setupUi(self, impulseDialog):
        impulseDialog.setObjectName("impulseDialog")
        impulseDialog.resize(1070, 638)
        self.verticalLayout = QtWidgets.QVBoxLayout(impulseDialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.previewChart = MplWidget(parent=impulseDialog)
        self.previewChart.setObjectName("previewChart")
        self.horizontalLayout.addWidget(self.previewChart)
        self.toolbarLayout = QtWidgets.QVBoxLayout()
        self.toolbarLayout.setObjectName("toolbarLayout")
        self.limitsButton = QtWidgets.QToolButton(parent=impulseDialog)
        self.limitsButton.setObjectName("limitsButton")
        self.toolbarLayout.addWidget(self.limitsButton)
        self.selectChannelsButton = QtWidgets.QToolButton(parent=impulseDialog)
        self.selectChannelsButton.setObjectName("selectChannelsButton")
        self.toolbarLayout.addWidget(self.selectChannelsButton)
        self.chartToggle = QtWidgets.QToolButton(parent=impulseDialog)
        self.chartToggle.setCheckable(True)
        self.chartToggle.setObjectName("chartToggle")
        self.toolbarLayout.addWidget(self.chartToggle)
        self.zoomInButton = QtWidgets.QToolButton(parent=impulseDialog)
        self.zoomInButton.setObjectName("zoomInButton")
        self.toolbarLayout.addWidget(self.zoomInButton)
        self.zoomOutButton = QtWidgets.QToolButton(parent=impulseDialog)
        self.zoomOutButton.setObjectName("zoomOutButton")
        self.toolbarLayout.addWidget(self.zoomOutButton)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.toolbarLayout.addItem(spacerItem)
        self.horizontalLayout.addLayout(self.toolbarLayout)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.buttonLayout = QtWidgets.QHBoxLayout()
        self.buttonLayout.setObjectName("buttonLayout")
        self.leftTimeLabel = QtWidgets.QLabel(parent=impulseDialog)
        self.leftTimeLabel.setObjectName("leftTimeLabel")
        self.buttonLayout.addWidget(self.leftTimeLabel)
        self.leftTimeValue = QtWidgets.QDoubleSpinBox(parent=impulseDialog)
        self.leftTimeValue.setReadOnly(True)
        self.leftTimeValue.setDecimals(3)
        self.leftTimeValue.setMinimum(-20.0)
        self.leftTimeValue.setSingleStep(0.001)
        self.leftTimeValue.setObjectName("leftTimeValue")
        self.buttonLayout.addWidget(self.leftTimeValue)
        self.rightTimeLabel = QtWidgets.QLabel(parent=impulseDialog)
        self.rightTimeLabel.setObjectName("rightTimeLabel")
        self.buttonLayout.addWidget(self.rightTimeLabel)
        self.rightTimeValue = QtWidgets.QDoubleSpinBox(parent=impulseDialog)
        self.rightTimeValue.setReadOnly(True)
        self.rightTimeValue.setDecimals(3)
        self.rightTimeValue.setMinimum(-20.0)
        self.rightTimeValue.setSingleStep(0.001)
        self.rightTimeValue.setObjectName("rightTimeValue")
        self.buttonLayout.addWidget(self.rightTimeValue)
        self.diffValueLabel = QtWidgets.QLabel(parent=impulseDialog)
        self.diffValueLabel.setObjectName("diffValueLabel")
        self.buttonLayout.addWidget(self.diffValueLabel)
        self.diffValue = QtWidgets.QDoubleSpinBox(parent=impulseDialog)
        self.diffValue.setReadOnly(True)
        self.diffValue.setDecimals(3)
        self.diffValue.setMinimum(-20.0)
        self.diffValue.setSingleStep(0.001)
        self.diffValue.setObjectName("diffValue")
        self.buttonLayout.addWidget(self.diffValue)
        self.buttonBox = QtWidgets.QDialogButtonBox(parent=impulseDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.StandardButton.Close)
        self.buttonBox.setObjectName("buttonBox")
        self.buttonLayout.addWidget(self.buttonBox)
        self.verticalLayout.addLayout(self.buttonLayout)

        self.retranslateUi(impulseDialog)
        self.buttonBox.accepted.connect(impulseDialog.accept) # type: ignore
        self.buttonBox.rejected.connect(impulseDialog.reject) # type: ignore
        self.limitsButton.clicked.connect(impulseDialog.show_limits) # type: ignore
        self.chartToggle.toggled['bool'].connect(impulseDialog.update_chart) # type: ignore
        self.selectChannelsButton.clicked.connect(impulseDialog.select_channels) # type: ignore
        self.zoomOutButton.clicked.connect(impulseDialog.zoom_out) # type: ignore
        self.zoomInButton.clicked.connect(impulseDialog.zoom_in) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(impulseDialog)

    def retranslateUi(self, impulseDialog):
        _translate = QtCore.QCoreApplication.translate
        impulseDialog.setWindowTitle(_translate("impulseDialog", "Impulse/Step Response"))
        self.selectChannelsButton.setText(_translate("impulseDialog", "..."))
        self.chartToggle.setText(_translate("impulseDialog", "..."))
        self.zoomInButton.setText(_translate("impulseDialog", "..."))
        self.zoomOutButton.setText(_translate("impulseDialog", "..."))
        self.leftTimeLabel.setText(_translate("impulseDialog", "Left"))
        self.leftTimeValue.setSuffix(_translate("impulseDialog", " ms"))
        self.rightTimeLabel.setText(_translate("impulseDialog", "Right"))
        self.rightTimeValue.setSuffix(_translate("impulseDialog", " ms"))
        self.diffValueLabel.setText(_translate("impulseDialog", "Diff"))
        self.diffValue.setSuffix(_translate("impulseDialog", " ms"))
from mpl import MplWidget
