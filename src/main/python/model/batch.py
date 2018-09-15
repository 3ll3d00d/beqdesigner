import glob
import logging
import os

import qtawesome as qta
from qtpy import QtWidgets, QtCore
from qtpy.QtCore import Qt, QObject, QRunnable, QThread, Signal, QThreadPool
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QDialog, QStatusBar, QFileDialog, QMessageBox

from model.ffmpeg import Executor, parse_audio_stream, ViewProbeDialog
from model.preferences import EXTRACTION_OUTPUT_DIR, EXTRACTION_BATCH_FILTER
from ui.batch import Ui_batchExtractDialog

logger = logging.getLogger('batch')


class BatchExtractDialog(QDialog, Ui_batchExtractDialog):
    '''
    Allows user to load a signal, processing it if necessary.
    '''

    def __init__(self, parent, preferences):
        super(BatchExtractDialog, self).__init__(parent)
        self.setupUi(self)
        self.setWindowFlags(self.windowFlags() | Qt.WindowSystemMenuHint | Qt.WindowMinMaxButtonsHint)
        self.__candidates = None
        self.__preferences = preferences
        default_output_dir = self.__preferences.get(EXTRACTION_OUTPUT_DIR)
        if os.path.isdir(default_output_dir):
            self.outputDir.setText(default_output_dir)
        filt = self.__preferences.get(EXTRACTION_BATCH_FILTER)
        if filt is not None:
            self.filter.setText(filt)
        self.outputDirPicker.setIcon(qta.icon('fa.folder-open-o'))
        self.statusBar = QStatusBar()
        self.verticalLayout.addWidget(self.statusBar)
        try:
            core_count = QThread.idealThreadCount()
            self.threads.setMaximum(core_count)
            self.threads.setValue(core_count)
        except Exception as e:
            logger.warning(f"Unable to get cpu_count()", e)

    def enable_search(self, search):
        '''
        Selects the root directory from which to search for files matching the filter.
        '''
        self.searchButton.setEnabled(len(search) > 0)

    def __select_dir(self):
        dialog = QFileDialog(parent=self)
        dialog.setFileMode(QFileDialog.DirectoryOnly)
        dialog.setWindowTitle('Select Directory')
        if dialog.exec():
            selected = dialog.selectedFiles()
            if len(selected) > 0:
                return selected[0]
        return ''

    def select_output(self):
        '''
        Selects the output directory.
        '''
        self.outputDir.setText(self.__select_dir())
        self.__enable_extract()

    def reset_batch(self):
        '''
        Removes all candidates.
        '''
        if self.__candidates is not None:
            self.__candidates.reset()

    def __enable_extract(self):
        '''
        Enables the extract button if we're ready to go!
        '''
        pass

    def search(self):
        '''
        Searches for files matching the filter in the input directory.
        '''
        self.__candidates = ExtractCandidates(self)
        globs = self.filter.text().split(';')
        self.__preferences.set(EXTRACTION_BATCH_FILTER, self.filter.text())
        for g in globs:
            from app import wait_cursor
            with wait_cursor(f"Searching {g}"):
                for matching_file in glob.iglob(g, recursive=True):
                    self.__candidates.append(matching_file)
        if len(self.__candidates) > 0:
            self.resetButton.setEnabled(True)
            self.searchButton.setEnabled(False)
            self.__candidates.probe()
        else:
            self.resetButton.setEnabled(False)

    def extract(self):
        '''
        Kicks off the extract.
        '''
        pass


class ExtractCandidates:

    def __init__(self, dialog):
        self.__dialog = dialog
        self.__candidates = []

    def __len__(self):
        return len(self.__candidates)

    def __getitem__(self, idx):
        return self.__candidates[idx]

    def append(self, candidate):
        '''
        Adds a new candidate to the group and renders it on the dialog.
        :param candidate: the candidate.
        '''
        extract_candidate = ExtractCandidate(len(self.__candidates), candidate, self.__dialog)
        self.__candidates.append(extract_candidate)
        extract_candidate.render()

    def reset(self):
        for candidate in self.__candidates:
            candidate.remove()
        self.__candidates = []

    def probe(self):
        '''
        Probes all the candidates.
        '''
        logger.info(f"Probing {len(self.__candidates)} candidates")
        for c in self.__candidates:
            c.probe()
        logger.info(f"Probed {len(self.__candidates)} candidates")


class ExtractCandidate:
    def __init__(self, idx, filename, dialog):
        self.__idx = idx
        self.__filename = filename
        self.__dialog = dialog
        self.__include = False
        self.__pre_probe_icon = None
        self.__stream_duration_micros = []
        self.executor = Executor(self.__filename, self.__dialog.outputDir.text())
        self.actionButton = None
        self.probeButton = None
        self.input = None
        self.audioStreams = None
        self.channelCount = None
        self.lfeChannelIndex = None
        self.ffmpegButton = None
        self.outputFilename = None
        self.progress = None

    def render(self):
        dialog = self.__dialog
        self.actionButton = QtWidgets.QToolButton(dialog.resultsScrollAreaContents)
        self.actionButton.setObjectName(f"actionButton{self.__idx}")
        self.toggle()
        self.actionButton.clicked.connect(self.toggle)
        dialog.resultsLayout.addWidget(self.actionButton, self.__idx + 1, 0, 1, 1, alignment=QtCore.Qt.AlignTop)
        self.input = QtWidgets.QLineEdit(dialog.resultsScrollAreaContents)
        self.input.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.input.setObjectName(f"input{self.__idx}")
        self.input.setText(self.__filename)
        self.input.setCursorPosition(0)
        self.input.setReadOnly(True)
        dialog.resultsLayout.addWidget(self.input, self.__idx + 1, 1, 1, 1, alignment=QtCore.Qt.AlignTop)
        self.probeButton = QtWidgets.QToolButton(dialog.resultsScrollAreaContents)
        self.probeButton.setObjectName(f"probeButton{self.__idx}")
        self.probeButton.setIcon(qta.icon('fa.info'))
        self.probeButton.setEnabled(False)
        self.probeButton.clicked.connect(self.show_probe_detail)
        dialog.resultsLayout.addWidget(self.probeButton, self.__idx + 1, 2, 1, 1, alignment=QtCore.Qt.AlignTop)
        self.audioStreams = QtWidgets.QComboBox(dialog.resultsScrollAreaContents)
        self.audioStreams.setObjectName(f"streams{self.__idx}")
        self.audioStreams.setEnabled(False)
        self.audioStreams.currentIndexChanged['int'].connect(self.recalc_ffmpeg_cmd)
        dialog.resultsLayout.addWidget(self.audioStreams, self.__idx + 1, 3, 1, 1, alignment=QtCore.Qt.AlignTop)
        self.channelCount = QtWidgets.QSpinBox(dialog.resultsScrollAreaContents)
        self.channelCount.setObjectName(f"channels{self.__idx}")
        self.channelCount.setEnabled(False)
        self.channelCount.valueChanged['int'].connect(self.override_ffmpeg_cmd)
        dialog.resultsLayout.addWidget(self.channelCount, self.__idx + 1, 4, 1, 1, alignment=QtCore.Qt.AlignTop)
        self.lfeChannelIndex = QtWidgets.QSpinBox(dialog.resultsScrollAreaContents)
        self.lfeChannelIndex.setObjectName(f"lfeChannel{self.__idx}")
        self.lfeChannelIndex.setEnabled(False)
        self.lfeChannelIndex.valueChanged['int'].connect(self.override_ffmpeg_cmd)
        dialog.resultsLayout.addWidget(self.lfeChannelIndex, self.__idx + 1, 5, 1, 1, alignment=QtCore.Qt.AlignTop)
        self.outputFilename = QtWidgets.QLineEdit(dialog.resultsScrollAreaContents)
        self.outputFilename.setObjectName(f"output{self.__idx}")
        self.outputFilename.setEnabled(False)
        self.outputFilename.textChanged.connect(self.override_output_filename)
        dialog.resultsLayout.addWidget(self.outputFilename, self.__idx + 1, 6, 1, 1, alignment=QtCore.Qt.AlignTop)
        self.ffmpegButton = QtWidgets.QToolButton(dialog.resultsScrollAreaContents)
        self.ffmpegButton.setObjectName(f"ffmpegButton{self.__idx}")
        self.ffmpegButton.setIcon(qta.icon('fa.info'))
        self.ffmpegButton.setEnabled(False)
        self.ffmpegButton.clicked.connect(self.show_ffmpeg_cmd)
        dialog.resultsLayout.addWidget(self.ffmpegButton, self.__idx + 1, 7, 1, 1, alignment=QtCore.Qt.AlignTop)
        self.progress = QtWidgets.QProgressBar(dialog.resultsScrollAreaContents)
        self.progress.setProperty(f"value", 0)
        self.progress.setObjectName(f"progress{self.__idx}")
        self.progress.setEnabled(False)
        dialog.resultsLayout.addWidget(self.progress, self.__idx + 1, 8, 1, 1, alignment=QtCore.Qt.AlignTop)

    def remove(self):
        pass

    def toggle(self):
        '''
        toggles whether this candidate should be included.
        :return:
        '''
        self.__include = not self.__include
        if self.__include is True:
            self.actionButton.setIcon(qta.icon('fa.check', color='green'))
        else:
            self.actionButton.setIcon(qta.icon('fa.times', color='red'))

    def probe(self):
        '''
        Schedules a ProbeJob with the global thread pool.
        '''
        job = ProbeJob(self)
        QThreadPool.globalInstance().start(job)

    def probe_start(self):
        '''
        Updates the UI when the probe starts.
        '''
        self.__pre_probe_icon = self.actionButton.icon()
        self.actionButton.setIcon(qta.icon('fa.spinner', color='blue', animation=qta.Spin(self.actionButton)))

    def probe_failed(self):
        '''
        Updates the UI when the probe fails.
        '''
        self.actionButton.setIcon(qta.icon('fa.exclamation-triangle', color='red'))
        # TODO disable all items
        self.actionButton.setEnabled(False)

    def probe_complete(self):
        '''
        Updates the UI when the probe completes.
        '''
        self.actionButton.setIcon(self.__pre_probe_icon)
        if self.executor.has_audio():
            for a in self.executor.audio_stream_data:
                text, duration_micros = parse_audio_stream(a)
                self.audioStreams.addItem(text)
                self.__stream_duration_micros.append(duration_micros)
            self.probeButton.setEnabled(True)
            self.audioStreams.setEnabled(True)
            self.channelCount.setEnabled(True)
            self.lfeChannelIndex.setEnabled(True)
            self.ffmpegButton.setEnabled(True)
            self.outputFilename.setEnabled(True)
            self.audioStreams.setCurrentIndex(0)
        else:
            self.probe_failed()
            self.actionButton.setTooltip('No audio streams')

    def recalc_ffmpeg_cmd(self):
        '''
        Calculates an ffmpeg cmd for the selected options.
        :return:
        '''
        self.executor.update_spec(self.audioStreams.currentIndex(), self.__dialog.monoMix.isChecked())
        self.lfeChannelIndex.setMaximum(self.executor.channel_count)
        self.lfeChannelIndex.setValue(self.executor.lfe_idx)
        self.channelCount.setMaximum(self.executor.channel_count)
        self.channelCount.setValue(self.executor.channel_count)

    def override_ffmpeg_cmd(self):
        '''
        overrides the ffmpeg command implied by the stream.
        '''
        self.executor.override('custom', self.channelCount.value(), self.lfeChannelIndex.value())
        self.outputFilename.setText(self.executor.output_file_name)
        self.outputFilename.setCursorPosition(0)

    def override_output_filename(self):
        '''
        overrides the output file name from the default.
        '''
        self.executor.output_file_name = self.outputFilename.text()

    def show_probe_detail(self):
        '''
        shows a tree widget containing the contents of the probe to allow the raw probe info to be visible.
        '''
        ViewProbeDialog(self.__filename, self.executor.probe, parent=self.__dialog).show()

    def show_ffmpeg_cmd(self):
        '''
        Pops up a message box containing the command.
        '''
        msg_box = QMessageBox()
        msg_box.setText(f"ffmpeg command line for {self.executor.file}")
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle('ffmpeg command')
        msg_box.setDetailedText(self.executor.ffmpeg_cli)
        font = QFont()
        font.setFamily("Consolas")
        font.setPointSize(8)
        msg_box.setFont(font)
        msg_box.exec()

    def __repr__(self):
        return self.__filename


class JobSignals(QObject):
    started = Signal()
    errored = Signal()
    finished = Signal()


class ProbeJob(QRunnable):
    '''
    Executes ffmpeg probe in the global thread pool.
    '''

    def __init__(self, candidate):
        super().__init__()
        self.__candidate = candidate
        self.__signals = JobSignals()
        self.__signals.started.connect(candidate.probe_start)
        self.__signals.errored.connect(candidate.probe_failed)
        self.__signals.finished.connect(candidate.probe_complete)

    def run(self):
        logger.info(f">> ProbeJob.run {self.__candidate.executor.file}")
        self.__signals.started.emit()
        try:
            self.__candidate.executor.probe_file()
            self.__signals.finished.emit()
        except Exception as e:
            logger.error(f"Probe {self.__candidate.executor.file} failed", e)
            self.__signals.errored.emit()
        logger.info(f"<< ProbeJob.run {self.__candidate.executor.file}")
