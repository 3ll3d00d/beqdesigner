# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'batch.ui'
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


class Ui_batchExtractDialog(object):
    def setupUi(self, batchExtractDialog):
        if batchExtractDialog.objectName():
            batchExtractDialog.setObjectName(u"batchExtractDialog")
        batchExtractDialog.resize(1727, 925)
        self.verticalLayout = QVBoxLayout(batchExtractDialog)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.controlFrame = QFrame(batchExtractDialog)
        self.controlFrame.setObjectName(u"controlFrame")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.controlFrame.sizePolicy().hasHeightForWidth())
        self.controlFrame.setSizePolicy(sizePolicy)
        self.controlFrame.setFrameShape(QFrame.Panel)
        self.controlFrame.setFrameShadow(QFrame.Sunken)
        self.gridLayout = QGridLayout(self.controlFrame)
        self.gridLayout.setObjectName(u"gridLayout")
        self.controlsLayout = QGridLayout()
        self.controlsLayout.setObjectName(u"controlsLayout")
        self.threads = QSpinBox(self.controlFrame)
        self.threads.setObjectName(u"threads")
        self.threads.setMinimum(1)
        self.threads.setMaximum(64)
        self.threads.setValue(1)

        self.controlsLayout.addWidget(self.threads, 3, 1, 1, 1)

        self.searchButton = QPushButton(self.controlFrame)
        self.searchButton.setObjectName(u"searchButton")
        self.searchButton.setEnabled(False)

        self.controlsLayout.addWidget(self.searchButton, 5, 1, 1, 1)

        self.outputDirLabel = QLabel(self.controlFrame)
        self.outputDirLabel.setObjectName(u"outputDirLabel")

        self.controlsLayout.addWidget(self.outputDirLabel, 1, 0, 1, 1)

        self.threadsLabel = QLabel(self.controlFrame)
        self.threadsLabel.setObjectName(u"threadsLabel")

        self.controlsLayout.addWidget(self.threadsLabel, 3, 0, 1, 1)

        self.filterLabel = QLabel(self.controlFrame)
        self.filterLabel.setObjectName(u"filterLabel")

        self.controlsLayout.addWidget(self.filterLabel, 0, 0, 1, 1)

        self.extractButton = QPushButton(self.controlFrame)
        self.extractButton.setObjectName(u"extractButton")
        self.extractButton.setEnabled(False)

        self.controlsLayout.addWidget(self.extractButton, 5, 2, 1, 1)

        self.resetButton = QPushButton(self.controlFrame)
        self.resetButton.setObjectName(u"resetButton")
        self.resetButton.setEnabled(False)

        self.controlsLayout.addWidget(self.resetButton, 5, 3, 1, 1)

        self.outputDirPicker = QToolButton(self.controlFrame)
        self.outputDirPicker.setObjectName(u"outputDirPicker")

        self.controlsLayout.addWidget(self.outputDirPicker, 1, 4, 1, 1)

        self.outputDir = QLineEdit(self.controlFrame)
        self.outputDir.setObjectName(u"outputDir")
        self.outputDir.setEnabled(False)

        self.controlsLayout.addWidget(self.outputDir, 1, 1, 1, 3)

        self.filter = QLineEdit(self.controlFrame)
        self.filter.setObjectName(u"filter")
        font = QFont()
        font.setFamily(u"Consolas")
        self.filter.setFont(font)

        self.controlsLayout.addWidget(self.filter, 0, 1, 1, 3)

        self.monoMix = QCheckBox(self.controlFrame)
        self.monoMix.setObjectName(u"monoMix")
        self.monoMix.setChecked(True)

        self.controlsLayout.addWidget(self.monoMix, 5, 0, 1, 1)

        self.controlsLayout.setColumnStretch(1, 1)
        self.controlsLayout.setColumnStretch(2, 1)
        self.controlsLayout.setColumnStretch(3, 1)

        self.gridLayout.addLayout(self.controlsLayout, 0, 0, 1, 1)


        self.verticalLayout.addWidget(self.controlFrame)

        self.resultsFrame = QFrame(batchExtractDialog)
        self.resultsFrame.setObjectName(u"resultsFrame")
        sizePolicy.setHeightForWidth(self.resultsFrame.sizePolicy().hasHeightForWidth())
        self.resultsFrame.setSizePolicy(sizePolicy)
        self.resultsFrame.setFrameShape(QFrame.Box)
        self.resultsFrame.setFrameShadow(QFrame.Sunken)
        self.gridLayout_2 = QGridLayout(self.resultsFrame)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.resultsTitle = QLabel(self.resultsFrame)
        self.resultsTitle.setObjectName(u"resultsTitle")
        font1 = QFont()
        font1.setBold(True)
        font1.setWeight(75)
        self.resultsTitle.setFont(font1)
        self.resultsTitle.setFrameShape(QFrame.Box)
        self.resultsTitle.setFrameShadow(QFrame.Sunken)
        self.resultsTitle.setAlignment(Qt.AlignCenter)

        self.gridLayout_2.addWidget(self.resultsTitle, 0, 0, 1, 1)

        self.resultsScrollArea = QScrollArea(self.resultsFrame)
        self.resultsScrollArea.setObjectName(u"resultsScrollArea")
        self.resultsScrollArea.setWidgetResizable(True)
        self.resultsScrollAreaContents = QWidget()
        self.resultsScrollAreaContents.setObjectName(u"resultsScrollAreaContents")
        self.resultsScrollAreaContents.setGeometry(QRect(0, 0, 1669, 660))
        self.resultsScrollLayout = QGridLayout(self.resultsScrollAreaContents)
        self.resultsScrollLayout.setObjectName(u"resultsScrollLayout")
        self.resultsLayout = QGridLayout()
        self.resultsLayout.setObjectName(u"resultsLayout")
        self.statusHeaderLabel = QLabel(self.resultsScrollAreaContents)
        self.statusHeaderLabel.setObjectName(u"statusHeaderLabel")
        font2 = QFont()
        font2.setUnderline(True)
        self.statusHeaderLabel.setFont(font2)
        self.statusHeaderLabel.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignTop)

        self.resultsLayout.addWidget(self.statusHeaderLabel, 0, 0, 1, 1)

        self.probeHeaderLabel = QLabel(self.resultsScrollAreaContents)
        self.probeHeaderLabel.setObjectName(u"probeHeaderLabel")
        font3 = QFont()
        font3.setItalic(True)
        font3.setUnderline(True)
        self.probeHeaderLabel.setFont(font3)
        self.probeHeaderLabel.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignTop)

        self.resultsLayout.addWidget(self.probeHeaderLabel, 0, 2, 1, 1)

        self.streamHeaderLabel = QLabel(self.resultsScrollAreaContents)
        self.streamHeaderLabel.setObjectName(u"streamHeaderLabel")
        font4 = QFont()
        font4.setBold(False)
        font4.setItalic(True)
        font4.setUnderline(True)
        font4.setWeight(50)
        self.streamHeaderLabel.setFont(font4)
        self.streamHeaderLabel.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignTop)

        self.resultsLayout.addWidget(self.streamHeaderLabel, 0, 3, 1, 1)

        self.inputFileHeaderLabel = QLabel(self.resultsScrollAreaContents)
        self.inputFileHeaderLabel.setObjectName(u"inputFileHeaderLabel")
        self.inputFileHeaderLabel.setFont(font4)
        self.inputFileHeaderLabel.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignTop)

        self.resultsLayout.addWidget(self.inputFileHeaderLabel, 0, 1, 1, 1)

        self.channelsHeaderLabel = QLabel(self.resultsScrollAreaContents)
        self.channelsHeaderLabel.setObjectName(u"channelsHeaderLabel")
        self.channelsHeaderLabel.setFont(font4)
        self.channelsHeaderLabel.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignTop)

        self.resultsLayout.addWidget(self.channelsHeaderLabel, 0, 4, 1, 1)

        self.outputFileHeaderLabel = QLabel(self.resultsScrollAreaContents)
        self.outputFileHeaderLabel.setObjectName(u"outputFileHeaderLabel")
        self.outputFileHeaderLabel.setFont(font4)
        self.outputFileHeaderLabel.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignTop)

        self.resultsLayout.addWidget(self.outputFileHeaderLabel, 0, 6, 1, 1)

        self.progressHeaderLabel = QLabel(self.resultsScrollAreaContents)
        self.progressHeaderLabel.setObjectName(u"progressHeaderLabel")
        self.progressHeaderLabel.setFont(font4)
        self.progressHeaderLabel.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignTop)

        self.resultsLayout.addWidget(self.progressHeaderLabel, 0, 8, 1, 1)

        self.lfeHeaderLabel = QLabel(self.resultsScrollAreaContents)
        self.lfeHeaderLabel.setObjectName(u"lfeHeaderLabel")
        self.lfeHeaderLabel.setFont(font4)
        self.lfeHeaderLabel.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignTop)

        self.resultsLayout.addWidget(self.lfeHeaderLabel, 0, 5, 1, 1)

        self.ffmpegCliLabel = QLabel(self.resultsScrollAreaContents)
        self.ffmpegCliLabel.setObjectName(u"ffmpegCliLabel")
        self.ffmpegCliLabel.setFont(font3)
        self.ffmpegCliLabel.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignTop)

        self.resultsLayout.addWidget(self.ffmpegCliLabel, 0, 7, 1, 1)

        self.resultsLayout.setColumnStretch(1, 1)
        self.resultsLayout.setColumnStretch(3, 2)
        self.resultsLayout.setColumnStretch(6, 1)
        self.resultsLayout.setColumnStretch(8, 1)

        self.resultsScrollLayout.addLayout(self.resultsLayout, 0, 0, 1, 1)

        self.resultsScrollArea.setWidget(self.resultsScrollAreaContents)

        self.gridLayout_2.addWidget(self.resultsScrollArea, 1, 0, 1, 1)


        self.verticalLayout.addWidget(self.resultsFrame)


        self.retranslateUi(batchExtractDialog)
        self.searchButton.clicked.connect(batchExtractDialog.search)
        self.extractButton.clicked.connect(batchExtractDialog.extract)
        self.outputDirPicker.clicked.connect(batchExtractDialog.select_output)
        self.filter.textChanged.connect(batchExtractDialog.enable_search)
        self.resetButton.clicked.connect(batchExtractDialog.reset_batch)
        self.threads.valueChanged.connect(batchExtractDialog.change_pool_size)

        QMetaObject.connectSlotsByName(batchExtractDialog)
    # setupUi

    def retranslateUi(self, batchExtractDialog):
        batchExtractDialog.setWindowTitle(QCoreApplication.translate("batchExtractDialog", u"Extract Audio", None))
        self.searchButton.setText(QCoreApplication.translate("batchExtractDialog", u"Search", None))
        self.outputDirLabel.setText(QCoreApplication.translate("batchExtractDialog", u"Output Directory", None))
        self.threadsLabel.setText(QCoreApplication.translate("batchExtractDialog", u"Threads", None))
#if QT_CONFIG(tooltip)
        self.filterLabel.setToolTip("")
#endif // QT_CONFIG(tooltip)
        self.filterLabel.setText(QCoreApplication.translate("batchExtractDialog", u"Search Filter", None))
        self.extractButton.setText(QCoreApplication.translate("batchExtractDialog", u"Extract", None))
        self.resetButton.setText(QCoreApplication.translate("batchExtractDialog", u"Reset", None))
        self.outputDirPicker.setText(QCoreApplication.translate("batchExtractDialog", u"...", None))
        self.filter.setText("")
        self.filter.setPlaceholderText(QCoreApplication.translate("batchExtractDialog", u"Enter 1 or more search filters, e.g. w:/films/*.mkv;y:/videos/**/*.m2ts", None))
        self.monoMix.setText(QCoreApplication.translate("batchExtractDialog", u"Mix to Mono?", None))
        self.resultsTitle.setText(QCoreApplication.translate("batchExtractDialog", u"Results", None))
        self.statusHeaderLabel.setText(QCoreApplication.translate("batchExtractDialog", u"Status", None))
        self.probeHeaderLabel.setText(QCoreApplication.translate("batchExtractDialog", u"Probe", None))
        self.streamHeaderLabel.setText(QCoreApplication.translate("batchExtractDialog", u"Stream", None))
        self.inputFileHeaderLabel.setText(QCoreApplication.translate("batchExtractDialog", u"Input File", None))
        self.channelsHeaderLabel.setText(QCoreApplication.translate("batchExtractDialog", u"Channels", None))
        self.outputFileHeaderLabel.setText(QCoreApplication.translate("batchExtractDialog", u"Output File", None))
        self.progressHeaderLabel.setText(QCoreApplication.translate("batchExtractDialog", u"Progress", None))
        self.lfeHeaderLabel.setText(QCoreApplication.translate("batchExtractDialog", u"LFE", None))
        self.ffmpegCliLabel.setText(QCoreApplication.translate("batchExtractDialog", u"ffmpeg", None))
    # retranslateUi

