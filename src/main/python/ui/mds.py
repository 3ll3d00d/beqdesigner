# Form implementation generated from reading ui file 'mds.ui'
#
# Created by: PyQt6 UI code generator 6.7.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_mdsDialog(object):
    def setupUi(self, mdsDialog):
        mdsDialog.setObjectName("mdsDialog")
        mdsDialog.resize(303, 302)
        self.verticalLayout = QtWidgets.QVBoxLayout(mdsDialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.fieldLayout = QtWidgets.QHBoxLayout()
        self.fieldLayout.setObjectName("fieldLayout")
        self.waysTable = QtWidgets.QTableWidget(parent=mdsDialog)
        self.waysTable.setColumnCount(2)
        self.waysTable.setObjectName("waysTable")
        self.waysTable.setRowCount(0)
        self.fieldLayout.addWidget(self.waysTable)
        self.verticalLayout.addLayout(self.fieldLayout)
        self.buttonBox = QtWidgets.QDialogButtonBox(parent=mdsDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.StandardButton.Apply)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(mdsDialog)
        self.buttonBox.accepted.connect(mdsDialog.accept) # type: ignore
        self.buttonBox.rejected.connect(mdsDialog.reject) # type: ignore
        self.buttonBox.clicked['QAbstractButton*'].connect(mdsDialog.update_mds) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(mdsDialog)

    def retranslateUi(self, mdsDialog):
        _translate = QtCore.QCoreApplication.translate
        mdsDialog.setWindowTitle(_translate("mdsDialog", "MDS Designer"))
