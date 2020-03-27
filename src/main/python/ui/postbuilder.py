# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'postbuilder.ui'
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


class Ui_postbuilder(object):
    def setupUi(self, postbuilder):
        if postbuilder.objectName():
            postbuilder.setObjectName(u"postbuilder")
        postbuilder.resize(628, 715)
        self.gridLayout = QGridLayout(postbuilder)
        self.gridLayout.setObjectName(u"gridLayout")
        self.buttonGrid = QGridLayout()
        self.buttonGrid.setObjectName(u"buttonGrid")
        self.leadingSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.buttonGrid.addItem(self.leadingSpacer, 0, 2, 1, 1)

        self.closeButton = QPushButton(postbuilder)
        self.closeButton.setObjectName(u"closeButton")
        self.closeButton.setAutoDefault(False)

        self.buttonGrid.addWidget(self.closeButton, 0, 3, 1, 1)

        self.middleSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.buttonGrid.addItem(self.middleSpacer, 0, 0, 1, 1)

        self.generateButton = QPushButton(postbuilder)
        self.generateButton.setObjectName(u"generateButton")
        self.generateButton.setAutoDefault(False)

        self.buttonGrid.addWidget(self.generateButton, 0, 1, 1, 1)


        self.gridLayout.addLayout(self.buttonGrid, 3, 0, 1, 1)

        self.comboBox = QComboBox(postbuilder)
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.setObjectName(u"comboBox")
        self.comboBox.setMaximumSize(QSize(250, 16777215))

        self.gridLayout.addWidget(self.comboBox, 0, 0, 1, 1, Qt.AlignRight)

        self.dataGrid = QGridLayout()
        self.dataGrid.setObjectName(u"dataGrid")
        self.spectrumLabel = QLabel(postbuilder)
        self.spectrumLabel.setObjectName(u"spectrumLabel")

        self.dataGrid.addWidget(self.spectrumLabel, 8, 0, 1, 1)

        self.postTextEdit = QPlainTextEdit(postbuilder)
        self.postTextEdit.setObjectName(u"postTextEdit")

        self.dataGrid.addWidget(self.postTextEdit, 11, 2, 1, 2)

        self.sourcePicker = QComboBox(postbuilder)
        self.sourcePicker.addItem("")
        self.sourcePicker.setObjectName(u"sourcePicker")

        self.dataGrid.addWidget(self.sourcePicker, 6, 2, 1, 2)

        self.pvaValid = QToolButton(postbuilder)
        self.pvaValid.setObjectName(u"pvaValid")
        self.pvaValid.setEnabled(True)

        self.dataGrid.addWidget(self.pvaValid, 7, 3, 1, 1)

        self.pvaField = QLineEdit(postbuilder)
        self.pvaField.setObjectName(u"pvaField")

        self.dataGrid.addWidget(self.pvaField, 7, 2, 1, 1)

        self.warningField = QLineEdit(postbuilder)
        self.warningField.setObjectName(u"warningField")

        self.dataGrid.addWidget(self.warningField, 5, 2, 1, 2)

        self.postLabel = QLabel(postbuilder)
        self.postLabel.setObjectName(u"postLabel")

        self.dataGrid.addWidget(self.postLabel, 11, 0, 1, 1)

        self.seasonLabel = QLabel(postbuilder)
        self.seasonLabel.setObjectName(u"seasonLabel")

        self.dataGrid.addWidget(self.seasonLabel, 3, 0, 1, 1)

        self.noteField = QLineEdit(postbuilder)
        self.noteField.setObjectName(u"noteField")

        self.dataGrid.addWidget(self.noteField, 4, 2, 1, 2)

        self.spectrumValid = QToolButton(postbuilder)
        self.spectrumValid.setObjectName(u"spectrumValid")
        self.spectrumValid.setEnabled(True)

        self.dataGrid.addWidget(self.spectrumValid, 8, 3, 1, 1)

        self.yearLabel = QLabel(postbuilder)
        self.yearLabel.setObjectName(u"yearLabel")

        self.dataGrid.addWidget(self.yearLabel, 1, 0, 1, 1)

        self.editionField = QLineEdit(postbuilder)
        self.editionField.setObjectName(u"editionField")

        self.dataGrid.addWidget(self.editionField, 2, 2, 1, 2)

        self.label = QLabel(postbuilder)
        self.label.setObjectName(u"label")

        self.dataGrid.addWidget(self.label, 12, 0, 1, 1)

        self.pvaLabel = QLabel(postbuilder)
        self.pvaLabel.setObjectName(u"pvaLabel")

        self.dataGrid.addWidget(self.pvaLabel, 7, 0, 1, 1)

        self.formatLabel = QLabel(postbuilder)
        self.formatLabel.setObjectName(u"formatLabel")

        self.dataGrid.addWidget(self.formatLabel, 10, 0, 1, 1)

        self.warningLabel = QLabel(postbuilder)
        self.warningLabel.setObjectName(u"warningLabel")

        self.dataGrid.addWidget(self.warningLabel, 5, 0, 1, 1)

        self.noteLabel = QLabel(postbuilder)
        self.noteLabel.setObjectName(u"noteLabel")

        self.dataGrid.addWidget(self.noteLabel, 4, 0, 1, 1)

        self.editionLabel = QLabel(postbuilder)
        self.editionLabel.setObjectName(u"editionLabel")

        self.dataGrid.addWidget(self.editionLabel, 2, 0, 1, 1)

        self.spectrumField = QLineEdit(postbuilder)
        self.spectrumField.setObjectName(u"spectrumField")

        self.dataGrid.addWidget(self.spectrumField, 8, 2, 1, 1)

        self.seasonField = QLineEdit(postbuilder)
        self.seasonField.setObjectName(u"seasonField")

        self.dataGrid.addWidget(self.seasonField, 3, 2, 1, 2)

        self.titleLabel = QLabel(postbuilder)
        self.titleLabel.setObjectName(u"titleLabel")

        self.dataGrid.addWidget(self.titleLabel, 0, 0, 1, 1)

        self.sourceLabel = QLabel(postbuilder)
        self.sourceLabel.setObjectName(u"sourceLabel")

        self.dataGrid.addWidget(self.sourceLabel, 6, 0, 1, 1)

        self.fileName = QLineEdit(postbuilder)
        self.fileName.setObjectName(u"fileName")
        self.fileName.setReadOnly(True)

        self.dataGrid.addWidget(self.fileName, 12, 2, 1, 2)

        self.titleField = QLineEdit(postbuilder)
        self.titleField.setObjectName(u"titleField")

        self.dataGrid.addWidget(self.titleField, 0, 2, 1, 1)

        self.yearField = QLineEdit(postbuilder)
        self.yearField.setObjectName(u"yearField")

        self.dataGrid.addWidget(self.yearField, 1, 2, 1, 1)

        self.titleValid = QToolButton(postbuilder)
        self.titleValid.setObjectName(u"titleValid")

        self.dataGrid.addWidget(self.titleValid, 0, 3, 1, 1)

        self.yearValid = QToolButton(postbuilder)
        self.yearValid.setObjectName(u"yearValid")
        self.yearValid.setEnabled(True)

        self.dataGrid.addWidget(self.yearValid, 1, 3, 1, 1)

        self.formatGrid = QGridLayout()
        self.formatGrid.setObjectName(u"formatGrid")
        self.ddCheckBox = QCheckBox(postbuilder)
        self.ddCheckBox.setObjectName(u"ddCheckBox")

        self.formatGrid.addWidget(self.ddCheckBox, 4, 2, 1, 1)

        self.dts71CheckBox = QCheckBox(postbuilder)
        self.dts71CheckBox.setObjectName(u"dts71CheckBox")

        self.formatGrid.addWidget(self.dts71CheckBox, 3, 1, 1, 1)

        self.truehd51CheckBox = QCheckBox(postbuilder)
        self.truehd51CheckBox.setObjectName(u"truehd51CheckBox")

        self.formatGrid.addWidget(self.truehd51CheckBox, 0, 2, 1, 1)

        self.ddPlusCheckBox = QCheckBox(postbuilder)
        self.ddPlusCheckBox.setObjectName(u"ddPlusCheckBox")

        self.formatGrid.addWidget(self.ddPlusCheckBox, 4, 1, 1, 1)

        self.lpcm51CheckBox = QCheckBox(postbuilder)
        self.lpcm51CheckBox.setObjectName(u"lpcm51CheckBox")

        self.formatGrid.addWidget(self.lpcm51CheckBox, 5, 1, 1, 1)

        self.dts61CheckBox = QCheckBox(postbuilder)
        self.dts61CheckBox.setObjectName(u"dts61CheckBox")

        self.formatGrid.addWidget(self.dts61CheckBox, 3, 2, 1, 1)

        self.lpcm71CheckBox = QCheckBox(postbuilder)
        self.lpcm71CheckBox.setObjectName(u"lpcm71CheckBox")

        self.formatGrid.addWidget(self.lpcm71CheckBox, 5, 0, 1, 1)

        self.truehd71CheckBox = QCheckBox(postbuilder)
        self.truehd71CheckBox.setObjectName(u"truehd71CheckBox")

        self.formatGrid.addWidget(self.truehd71CheckBox, 0, 1, 1, 1)

        self.ddAtmosCheckBox = QCheckBox(postbuilder)
        self.ddAtmosCheckBox.setObjectName(u"ddAtmosCheckBox")

        self.formatGrid.addWidget(self.ddAtmosCheckBox, 4, 0, 1, 1)

        self.atmosCheckBox = QCheckBox(postbuilder)
        self.atmosCheckBox.setObjectName(u"atmosCheckBox")

        self.formatGrid.addWidget(self.atmosCheckBox, 0, 0, 1, 1)

        self.dtsxCheckBox = QCheckBox(postbuilder)
        self.dtsxCheckBox.setObjectName(u"dtsxCheckBox")

        self.formatGrid.addWidget(self.dtsxCheckBox, 3, 0, 1, 1)

        self.dts51CheckBox = QCheckBox(postbuilder)
        self.dts51CheckBox.setObjectName(u"dts51CheckBox")

        self.formatGrid.addWidget(self.dts51CheckBox, 3, 3, 1, 1)


        self.dataGrid.addLayout(self.formatGrid, 10, 2, 1, 1)

        self.audioValid = QToolButton(postbuilder)
        self.audioValid.setObjectName(u"audioValid")

        self.dataGrid.addWidget(self.audioValid, 10, 3, 1, 1)


        self.gridLayout.addLayout(self.dataGrid, 1, 0, 1, 1)

        QWidget.setTabOrder(self.titleField, self.yearField)
        QWidget.setTabOrder(self.yearField, self.editionField)
        QWidget.setTabOrder(self.editionField, self.seasonField)
        QWidget.setTabOrder(self.seasonField, self.noteField)
        QWidget.setTabOrder(self.noteField, self.warningField)
        QWidget.setTabOrder(self.warningField, self.sourcePicker)
        QWidget.setTabOrder(self.sourcePicker, self.pvaField)
        QWidget.setTabOrder(self.pvaField, self.spectrumField)
        QWidget.setTabOrder(self.spectrumField, self.atmosCheckBox)
        QWidget.setTabOrder(self.atmosCheckBox, self.truehd71CheckBox)
        QWidget.setTabOrder(self.truehd71CheckBox, self.truehd51CheckBox)
        QWidget.setTabOrder(self.truehd51CheckBox, self.dtsxCheckBox)
        QWidget.setTabOrder(self.dtsxCheckBox, self.dts71CheckBox)
        QWidget.setTabOrder(self.dts71CheckBox, self.dts61CheckBox)
        QWidget.setTabOrder(self.dts61CheckBox, self.dts51CheckBox)
        QWidget.setTabOrder(self.dts51CheckBox, self.ddAtmosCheckBox)
        QWidget.setTabOrder(self.ddAtmosCheckBox, self.ddPlusCheckBox)
        QWidget.setTabOrder(self.ddPlusCheckBox, self.ddCheckBox)
        QWidget.setTabOrder(self.ddCheckBox, self.lpcm71CheckBox)
        QWidget.setTabOrder(self.lpcm71CheckBox, self.lpcm51CheckBox)
        QWidget.setTabOrder(self.lpcm51CheckBox, self.postTextEdit)
        QWidget.setTabOrder(self.postTextEdit, self.generateButton)
        QWidget.setTabOrder(self.generateButton, self.closeButton)

        self.retranslateUi(postbuilder)
        self.closeButton.clicked.connect(postbuilder.close)
        self.generateButton.clicked.connect(postbuilder.generate_avs_post)
        self.titleField.textChanged.connect(postbuilder.build_avs_post)
        self.yearField.textChanged.connect(postbuilder.build_avs_post)
        self.editionField.textChanged.connect(postbuilder.build_avs_post)
        self.seasonField.textChanged.connect(postbuilder.build_avs_post)
        self.noteField.textChanged.connect(postbuilder.build_avs_post)
        self.warningField.textChanged.connect(postbuilder.build_avs_post)
        self.pvaField.textChanged.connect(postbuilder.build_avs_post)
        self.sourcePicker.currentIndexChanged.connect(postbuilder.build_avs_post)
        self.spectrumField.textChanged.connect(postbuilder.build_avs_post)
        self.atmosCheckBox.toggled.connect(postbuilder.build_avs_post)
        self.dtsxCheckBox.toggled.connect(postbuilder.build_avs_post)
        self.ddAtmosCheckBox.toggled.connect(postbuilder.build_avs_post)
        self.lpcm71CheckBox.toggled.connect(postbuilder.build_avs_post)
        self.truehd71CheckBox.toggled.connect(postbuilder.build_avs_post)
        self.dts71CheckBox.toggled.connect(postbuilder.build_avs_post)
        self.ddPlusCheckBox.toggled.connect(postbuilder.build_avs_post)
        self.lpcm51CheckBox.toggled.connect(postbuilder.build_avs_post)
        self.truehd51CheckBox.toggled.connect(postbuilder.build_avs_post)
        self.dts61CheckBox.toggled.connect(postbuilder.build_avs_post)
        self.ddCheckBox.toggled.connect(postbuilder.build_avs_post)
        self.dts51CheckBox.toggled.connect(postbuilder.build_avs_post)
        self.comboBox.currentIndexChanged.connect(postbuilder.post_type_changed)

        self.closeButton.setDefault(False)


        QMetaObject.connectSlotsByName(postbuilder)
    # setupUi

    def retranslateUi(self, postbuilder):
        postbuilder.setWindowTitle(QCoreApplication.translate("postbuilder", u"AVS Post Builder", None))
        self.closeButton.setText(QCoreApplication.translate("postbuilder", u"Close", None))
        self.generateButton.setText(QCoreApplication.translate("postbuilder", u"Generate 2x4HD XML File", None))
        self.comboBox.setItemText(0, QCoreApplication.translate("postbuilder", u"Movie", None))
        self.comboBox.setItemText(1, QCoreApplication.translate("postbuilder", u"TV Show", None))

        self.spectrumLabel.setText(QCoreApplication.translate("postbuilder", u"Spectrum URL", None))
        self.sourcePicker.setItemText(0, QCoreApplication.translate("postbuilder", u"Disc", None))

        self.pvaValid.setText("")
        self.postLabel.setText(QCoreApplication.translate("postbuilder", u"BEQ Post", None))
        self.seasonLabel.setText(QCoreApplication.translate("postbuilder", u"TV Season", None))
        self.spectrumValid.setText("")
        self.yearLabel.setText(QCoreApplication.translate("postbuilder", u"Year", None))
        self.label.setText(QCoreApplication.translate("postbuilder", u"XML File Name", None))
        self.pvaLabel.setText(QCoreApplication.translate("postbuilder", u"PvA Graph URL", None))
        self.formatLabel.setText(QCoreApplication.translate("postbuilder", u"Audio Format", None))
        self.warningLabel.setText(QCoreApplication.translate("postbuilder", u"Warning", None))
        self.noteLabel.setText(QCoreApplication.translate("postbuilder", u"Note", None))
        self.editionLabel.setText(QCoreApplication.translate("postbuilder", u"Edition", None))
        self.titleLabel.setText(QCoreApplication.translate("postbuilder", u"Title", None))
        self.sourceLabel.setText(QCoreApplication.translate("postbuilder", u"Source", None))
        self.titleValid.setText("")
        self.yearValid.setText("")
        self.ddCheckBox.setText(QCoreApplication.translate("postbuilder", u"DD 5.1", None))
        self.dts71CheckBox.setText(QCoreApplication.translate("postbuilder", u"DTS-HD MA 7.1", None))
        self.truehd51CheckBox.setText(QCoreApplication.translate("postbuilder", u"TrueHD 5.1", None))
        self.ddPlusCheckBox.setText(QCoreApplication.translate("postbuilder", u"DD+", None))
        self.lpcm51CheckBox.setText(QCoreApplication.translate("postbuilder", u"LPCM 5.1", None))
        self.dts61CheckBox.setText(QCoreApplication.translate("postbuilder", u"DTS-HD MA 6.1", None))
        self.lpcm71CheckBox.setText(QCoreApplication.translate("postbuilder", u"LPCM 7.1", None))
        self.truehd71CheckBox.setText(QCoreApplication.translate("postbuilder", u"TrueHD 7.1", None))
        self.ddAtmosCheckBox.setText(QCoreApplication.translate("postbuilder", u"DD+ Atmos", None))
        self.atmosCheckBox.setText(QCoreApplication.translate("postbuilder", u"Atmos", None))
        self.dtsxCheckBox.setText(QCoreApplication.translate("postbuilder", u"DTS:X", None))
        self.dts51CheckBox.setText(QCoreApplication.translate("postbuilder", u"DTS-HD MA 5.1", None))
        self.audioValid.setText("")
    # retranslateUi

