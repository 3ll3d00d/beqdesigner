import glob
import logging
import math
import os
from enum import Enum

import qtawesome as qta
from qtpy import QtWidgets, QtCore
from qtpy.QtCore import Qt, QObject, QRunnable, QThread, Signal, QThreadPool
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QDialog, QStatusBar, QFileDialog

from model.ffmpeg import Executor, parse_audio_stream, ViewProbeDialog, SIGNAL_CONNECTED, SIGNAL_ERROR, \
    SIGNAL_COMPLETE, SIGNAL_CANCELLED, FFMpegDetailsDialog
from model.preferences import EXTRACTION_OUTPUT_DIR, EXTRACTION_BATCH_FILTER, ANALYSIS_TARGET_FS
from model.spin import StoppableSpin, stop_spinner
from ui.batch import Ui_batchExtractDialog

logger = logging.getLogger('batch')


class BatchExtractDialog(QDialog, Ui_batchExtractDialog):
    '''
    Allows user to search for files to convert according to some glob patterns and then extract audio from them en masse.
    The workflow is
    * set search filter and search for files that match the filter
    * each matching file will be rendered in the scroll area showing the various configurable parameters for the extraction
    * an ffprobe job is created for each file and executed asynchronously in the global thread pool
    * user can toggle whether the file should be included in the extraction or not & can choose how many threads to put into the job
    * if the user presses reset now, all files are removed and the search can restart
    * if the user presses extract, an ffmpeg job is created for each file and scheduled with the global thread pool
    * if the user presses reset after extract, all not started jobs are cancelled
    '''

    def __init__(self, parent, preferences):
        super(BatchExtractDialog, self).__init__(parent)
        self.setupUi(self)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowSystemMenuHint | Qt.WindowType.WindowMinMaxButtonsHint)
        self.__candidates = None
        self.__preferences = preferences
        self.__search_spinner = None
        default_output_dir = self.__preferences.get(EXTRACTION_OUTPUT_DIR)
        if os.path.isdir(default_output_dir):
            self.outputDir.setText(default_output_dir)
        filt = self.__preferences.get(EXTRACTION_BATCH_FILTER)
        if filt is not None:
            self.filter.setText(filt)
        self.outputDirPicker.setIcon(qta.icon('fa5s.folder-open'))
        self.statusBar = QStatusBar()
        self.verticalLayout.addWidget(self.statusBar)
        try:
            core_count = QThreadPool.globalInstance().maxThreadCount()
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
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOptions(QFileDialog.Option.ShowDirsOnly)
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
        self.enable_extract()

    def reset_batch(self):
        '''
        Removes all candidates if we're not already in progress.
        '''
        if self.__candidates is not None:
            self.__candidates.reset()
            if self.__candidates.is_extracting is False:
                self.__candidates = None
                self.__prepare_search()

    def __prepare_search(self):
        self.enable_search(self.filter.text())
        self.filter.setEnabled(True)
        self.searchButton.blockSignals(False)
        self.searchButton.setIcon(QIcon())
        self.resetButton.setEnabled(False)
        self.extractButton.setEnabled(False)

    def accept(self):
        '''
        Resets the thread pool size back to the default.
        '''
        self.change_pool_size(QThread.idealThreadCount())
        QDialog.accept(self)

    def reject(self):
        '''
        Resets the thread pool size back to the default.
        '''
        self.change_pool_size(QThread.idealThreadCount())
        QDialog.reject(self)

    def enable_extract(self):
        ''';
        Enables the extract button if we're ready to go!
        '''
        self.extractButton.setEnabled(True)
        self.searchButton.setEnabled(False)

    def search(self):
        '''
        Searches for files matching the filter in the input directory.
        '''
        self.filter.setEnabled(False)
        self.__candidates = ExtractCandidates(self, self.__preferences.get(ANALYSIS_TARGET_FS))
        self.__preferences.set(EXTRACTION_BATCH_FILTER, self.filter.text())
        globs = self.filter.text().split(';')
        job = FileSearch(globs)
        job.signals.started.connect(self.__on_search_start)
        job.signals.on_error.connect(self.__on_search_error)
        job.signals.on_match.connect(self.__on_search_match)
        job.signals.finished.connect(self.__on_search_complete)
        QThreadPool.globalInstance().start(job)

    def __on_search_start(self):
        '''
        Reacts to the search job starting by providing a visual indication that it has started.
        '''
        self.searchButton.setText('Searching...')
        self.__search_spinner = StoppableSpin(self.searchButton, 'search')
        spin_icon = qta.icon('fa5s.spinner', color='green', animation=self.__search_spinner)
        self.searchButton.setIcon(spin_icon)
        self.searchButton.blockSignals(True)

    def __on_search_match(self, matching_file):
        '''
        Adds the match to the candidates.
        '''
        if self.__candidates.append(matching_file):
            self.resultsTitle.setText(f"Results - {len(self.__candidates)} matches")

    def __on_search_error(self, bad_glob):
        '''
        Reports on glob failures.
        '''
        self.searchButton.setText(f"Search error in {bad_glob}")

    def __on_search_complete(self):
        stop_spinner(self.__search_spinner, self.searchButton)
        self.searchButton.blockSignals(False)
        self.searchButton.setText('Search')
        self.__search_spinner = None
        if len(self.__candidates) > 0:
            self.resetButton.setEnabled(True)
            self.searchButton.setEnabled(False)
            self.searchButton.setIcon(qta.icon('fa5s.check'))
            self.__candidates.probe()
        else:
            self.resultsTitle.setText(f"Results - no matches, try a different search filter")
            self.__prepare_search()

    def extract(self):
        '''
        Kicks off the extract.
        '''
        self.extractButton.setEnabled(False)
        self.threads.setEnabled(False)
        self.resetButton.setText('Cancel')
        self.resultsTitle.setText('Extracting...')
        self.__candidates.extract()

    def extract_complete(self):
        '''
        Signals that we're all done.
        :return:
        '''
        self.resultsTitle.setText('Extraction Complete')
        self.resetButton.setEnabled(False)

    def change_pool_size(self, size):
        '''
        Changes the pool size.
        :param size: size.
        '''
        logger.info(f"Changing thread pool size to {size}")
        QThreadPool.globalInstance().setMaxThreadCount(size)


class FileSearchSignals(QObject):
    started = Signal()
    on_match = Signal(str, name='on_match')
    on_error = Signal(str, name='on_error')
    finished = Signal()


class FileSearch(QRunnable):
    '''
    Runs the glob in a separate thread as it can be v v slow so UI performance is poor.
    '''
    def __init__(self, globs):
        super().__init__()
        self.signals = FileSearchSignals()
        self.globs = globs

    def run(self):
        self.signals.started.emit()
        for g in self.globs:
            try:
                if os.path.isdir(g):
                    g = f"{g}{os.sep}*"
                for matching_file in glob.iglob(g, recursive=True):
                    self.signals.on_match.emit(matching_file)
            except Exception as e:
                logger.exception(f"Unexpected exception during search of {g}", e)
                self.signals.on_error.emit(g)
        self.signals.finished.emit()


class ExtractCandidates:

    def __init__(self, dialog, decimate_fs=1000):
        self.__dialog = dialog
        self.__candidates = []
        self.__probed = []
        self.__extracting = False
        self.__extracted = []
        self.__decimate_fs = decimate_fs

    @property
    def is_extracting(self):
        return self.__extracting

    def __len__(self):
        return len(self.__candidates)

    def __getitem__(self, idx):
        return self.__candidates[idx]

    def append(self, candidate):
        '''
        Adds a new candidate to the group and renders it on the dialog.
        :param candidate: the candidate.
        '''
        if os.path.isfile(candidate):
            extract_candidate = ExtractCandidate(len(self.__candidates), candidate, self.__dialog,
                                                 self.on_probe_complete, self.on_extract_complete, self.__decimate_fs)
            self.__candidates.append(extract_candidate)
            extract_candidate.render()
            return True
        return False

    def reset(self):
        '''
        Removes all the candidates.
        '''
        if self.__extracting is True:
            for candidate in self.__candidates:
                candidate.toggle()
        else:
            for candidate in self.__candidates:
                candidate.remove()
            self.__candidates = []

    def probe(self):
        '''
        Probes all the candidates.
        '''
        logger.info(f"Probing {len(self)} candidates")
        for c in self.__candidates:
            c.probe()

    def on_probe_complete(self, idx):
        '''
        Registers the completion of a single probe.
        :param idx: the idx of the probed candidate.
        '''
        self.__probed.append(idx)
        if len(self.__probed) == len(self):
            logger.info('All probes complete')
            self.__dialog.enable_extract()

    def extract(self):
        self.__extracting = True
        logger.info(f"Extracting {len(self)} candidates")
        for c in self.__candidates:
            c.extract()

    def on_extract_complete(self, idx):
        '''
        Registers the completion of a single extract.
        :param idx: the idx of the extracted candidate.
        '''
        self.__extracted.append(idx)
        if len(self.__extracted) == len(self):
            logger.info('All probes complete')
            self.__dialog.extract_complete()


class ExtractStatus(Enum):
    NEW = 0
    IN_PROGRESS = 1
    PROBED = 2
    EXCLUDED = 3
    CANCELLED = 4
    COMPLETE = 5
    FAILED = 6

    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


class ExtractCandidate:
    def __init__(self, idx, filename, dialog, on_probe_complete, on_extract_complete, decimate_fs):
        self.__idx = idx
        self.__filename = filename
        self.__dialog = dialog
        self.__in_progress_icon = None
        self.__stream_duration_micros = []
        self.__on_probe_complete = on_probe_complete
        self.__on_extract_complete = on_extract_complete
        self.__result = None
        self.__status = ExtractStatus.NEW
        self.executor = Executor(self.__filename, self.__dialog.outputDir.text(), decimate_fs=decimate_fs)
        self.executor.progress_handler = self.__handle_ffmpeg_process
        self.actionButton = None
        self.probeButton = None
        self.input = None
        self.audioStreams = None
        self.channelCount = None
        self.lfeChannelIndex = None
        self.ffmpegButton = None
        self.outputFilename = None
        self.ffmpegProgress = None

    def render(self):
        dialog = self.__dialog
        self.actionButton = QtWidgets.QToolButton(dialog.resultsScrollAreaContents)
        self.actionButton.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.actionButton.setObjectName(f"actionButton{self.__idx}")
        self.status = ExtractStatus.NEW
        self.actionButton.clicked.connect(self.toggle)
        dialog.resultsLayout.addWidget(self.actionButton, self.__idx + 1, 0, 1, 1, alignment=QtCore.Qt.AlignmentFlag.AlignTop)
        self.input = QtWidgets.QLineEdit(dialog.resultsScrollAreaContents)
        self.input.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.input.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeading | QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        self.input.setObjectName(f"input{self.__idx}")
        self.input.setText(self.__filename)
        self.input.setCursorPosition(0)
        self.input.setReadOnly(True)
        dialog.resultsLayout.addWidget(self.input, self.__idx + 1, 1, 1, 1)
        self.probeButton = QtWidgets.QToolButton(dialog.resultsScrollAreaContents)
        self.probeButton.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        # self.probeButton.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeading | QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        self.probeButton.setObjectName(f"probeButton{self.__idx}")
        self.probeButton.setIcon(qta.icon('fa5s.info'))
        self.probeButton.setEnabled(False)
        self.probeButton.clicked.connect(self.show_probe_detail)
        dialog.resultsLayout.addWidget(self.probeButton, self.__idx + 1, 2, 1, 1)
        self.audioStreams = QtWidgets.QComboBox(dialog.resultsScrollAreaContents)
        self.audioStreams.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        # self.audioStreams.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeading | QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        self.audioStreams.setObjectName(f"streams{self.__idx}")
        self.audioStreams.setEnabled(False)
        self.audioStreams.currentIndexChanged['int'].connect(self.recalc_ffmpeg_cmd)
        dialog.resultsLayout.addWidget(self.audioStreams, self.__idx + 1, 3, 1, 1)
        self.channelCount = QtWidgets.QSpinBox(dialog.resultsScrollAreaContents)
        self.channelCount.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.channelCount.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeading | QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        self.channelCount.setObjectName(f"channels{self.__idx}")
        self.channelCount.setEnabled(False)
        self.channelCount.valueChanged['int'].connect(self.override_ffmpeg_cmd)
        dialog.resultsLayout.addWidget(self.channelCount, self.__idx + 1, 4, 1, 1)
        self.lfeChannelIndex = QtWidgets.QSpinBox(dialog.resultsScrollAreaContents)
        self.lfeChannelIndex.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.lfeChannelIndex.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeading | QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        self.lfeChannelIndex.setObjectName(f"lfeChannel{self.__idx}")
        self.lfeChannelIndex.setEnabled(False)
        self.lfeChannelIndex.valueChanged['int'].connect(self.override_ffmpeg_cmd)
        dialog.resultsLayout.addWidget(self.lfeChannelIndex, self.__idx + 1, 5, 1, 1)
        self.outputFilename = QtWidgets.QLineEdit(dialog.resultsScrollAreaContents)
        self.outputFilename.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.outputFilename.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeading | QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        self.outputFilename.setObjectName(f"output{self.__idx}")
        self.outputFilename.setEnabled(False)
        self.outputFilename.textChanged.connect(self.override_output_filename)
        dialog.resultsLayout.addWidget(self.outputFilename, self.__idx + 1, 6, 1, 1)
        self.ffmpegButton = QtWidgets.QToolButton(dialog.resultsScrollAreaContents)
        self.ffmpegButton.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        # self.ffmpegButton.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeading | QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        self.ffmpegButton.setObjectName(f"ffmpegButton{self.__idx}")
        self.ffmpegButton.setIcon(qta.icon('fa5s.info'))
        self.ffmpegButton.setEnabled(False)
        self.ffmpegButton.clicked.connect(self.show_ffmpeg_cmd)
        dialog.resultsLayout.addWidget(self.ffmpegButton, self.__idx + 1, 7, 1, 1)
        self.ffmpegProgress = QtWidgets.QProgressBar(dialog.resultsScrollAreaContents)
        self.ffmpegProgress.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.ffmpegProgress.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeading | QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        self.ffmpegProgress.setProperty(f"value", 0)
        self.ffmpegProgress.setObjectName(f"progress{self.__idx}")
        self.ffmpegProgress.setEnabled(False)
        dialog.resultsLayout.addWidget(self.ffmpegProgress, self.__idx + 1, 8, 1, 1)

    def remove(self):
        logger.debug(f"Closing widgets for {self.executor.file}")
        self.actionButton.close()
        self.probeButton.close()
        self.input.close()
        self.audioStreams.close()
        self.channelCount.close()
        self.lfeChannelIndex.close()
        self.ffmpegButton.close()
        self.outputFilename.close()
        self.ffmpegProgress.close()
        logger.debug(f"Closed widgets for {self.executor.file}")

    @property
    def status(self):
        return self.__status

    @status.setter
    def status(self, status):
        self.__status = status
        do_stop = False
        if status == ExtractStatus.NEW:
            self.actionButton.setIcon(qta.icon('fa5s.check', color='green'))
        elif status == ExtractStatus.IN_PROGRESS:
            self.actionButton.blockSignals(True)
            self.__in_progress_icon = StoppableSpin(self.actionButton, self.__filename)
            self.actionButton.setIcon(qta.icon('fa5s.spinner', color='blue', animation=self.__in_progress_icon))
            self.probeButton.setEnabled(False)
            self.input.setEnabled(False)
            self.audioStreams.setEnabled(False)
            self.channelCount.setEnabled(False)
            self.lfeChannelIndex.setEnabled(False)
            self.outputFilename.setEnabled(False)
        elif status == ExtractStatus.EXCLUDED:
            self.actionButton.setIcon(qta.icon('fa5s.times', color='green'))
        elif status == ExtractStatus.PROBED:
            self.actionButton.blockSignals(False)
            self.actionButton.setIcon(qta.icon('fa5s.check', color='green'))
            self.probeButton.setEnabled(True)
            self.input.setEnabled(False)
            self.audioStreams.setEnabled(True)
            self.channelCount.setEnabled(True)
            self.lfeChannelIndex.setEnabled(True)
            self.outputFilename.setEnabled(True)
            self.ffmpegButton.setEnabled(True)
            do_stop = True
        elif status == ExtractStatus.FAILED:
            self.actionButton.setIcon(qta.icon('fa5s.exclamation-triangle', color='red'))
            self.actionButton.blockSignals(True)
            self.probeButton.setEnabled(False)
            self.input.setEnabled(False)
            self.audioStreams.setEnabled(False)
            self.channelCount.setEnabled(False)
            self.lfeChannelIndex.setEnabled(False)
            self.outputFilename.setEnabled(False)
            self.ffmpegProgress.setEnabled(False)
            do_stop = True
        elif status == ExtractStatus.CANCELLED:
            self.actionButton.blockSignals(True)
            self.actionButton.setIcon(qta.icon('fa5s.ban', color='green'))
            self.probeButton.setEnabled(False)
            self.input.setEnabled(False)
            self.audioStreams.setEnabled(False)
            self.channelCount.setEnabled(False)
            self.lfeChannelIndex.setEnabled(False)
            self.outputFilename.setEnabled(False)
            self.ffmpegProgress.setEnabled(False)
            do_stop = True
        elif status == ExtractStatus.COMPLETE:
            self.actionButton.blockSignals(True)
            self.actionButton.setIcon(qta.icon('fa5s.check', color='green'))
            self.probeButton.setEnabled(False)
            self.input.setEnabled(False)
            self.audioStreams.setEnabled(False)
            self.channelCount.setEnabled(False)
            self.lfeChannelIndex.setEnabled(False)
            self.outputFilename.setEnabled(False)
            self.ffmpegProgress.setEnabled(False)
            do_stop = True

        if do_stop is True:
            stop_spinner(self.__in_progress_icon, self.actionButton)
            self.__in_progress_icon = None

    def toggle(self):
        '''
        toggles whether this candidate should be excluded from the batch.
        '''
        if self.status < ExtractStatus.CANCELLED:
            if self.status == ExtractStatus.EXCLUDED:
                self.executor.enable()
                if self.executor.probe is not None:
                    self.status = ExtractStatus.PROBED
                else:
                    self.status = ExtractStatus.NEW
            else:
                self.status = ExtractStatus.EXCLUDED
                self.executor.cancel()

    def probe(self):
        '''
        Schedules a ProbeJob with the global thread pool.
        '''
        QThreadPool.globalInstance().start(ProbeJob(self))

    def __handle_ffmpeg_process(self, key, value):
        '''
        Handles progress reports from ffmpeg in order to communicate status via the progress bar. Used as a slot
        connected to a signal emitted by the AudioExtractor.
        :param key: the key.
        :param value: the value.
        '''
        if key == SIGNAL_CONNECTED:
            pass
        elif key == SIGNAL_CANCELLED:
            self.status = ExtractStatus.CANCELLED
            self.__result = value
            self.__on_extract_complete(self.__idx)
        elif key == 'out_time_ms':
            if self.status != ExtractStatus.IN_PROGRESS:
                logger.debug(f"Extraction started for {self}")
                self.status = ExtractStatus.IN_PROGRESS
            out_time_ms = int(value)
            total_micros = self.__stream_duration_micros[self.audioStreams.currentIndex()]
            # logger.debug(f"{self.input.text()} -- {key}={value} vs {total_micros}")
            if total_micros > 0:
                progress = (out_time_ms / total_micros) * 100.0
                self.ffmpegProgress.setValue(math.ceil(progress))
                self.ffmpegProgress.setTextVisible(True)
                self.ffmpegProgress.setFormat(f"{round(progress, 2):.2f}%")
        elif key == SIGNAL_ERROR:
            self.status = ExtractStatus.FAILED
            self.__result = value
            self.__on_extract_complete(self.__idx)
        elif key == SIGNAL_COMPLETE:
            self.ffmpegProgress.setValue(100)
            self.status = ExtractStatus.COMPLETE
            self.__result = value
            self.__on_extract_complete(self.__idx)

    def extract(self):
        '''
        Triggers the extraction.
        '''
        self.executor.execute()

    def probe_start(self):
        '''
        Updates the UI when the probe starts.
        '''
        self.status = ExtractStatus.IN_PROGRESS

    def probe_failed(self):
        '''
        Updates the UI when an ff cmd fails.
        '''
        self.status = ExtractStatus.FAILED
        self.__on_probe_complete(self.__idx)

    def probe_complete(self):
        '''
        Updates the UI when the probe completes.
        '''
        if self.executor.has_audio():
            self.status = ExtractStatus.PROBED
            for a in self.executor.audio_stream_data:
                text, duration_micros = parse_audio_stream(self.executor.probe, a)
                self.audioStreams.addItem(text)
                self.__stream_duration_micros.append(duration_micros)
            self.audioStreams.setCurrentIndex(0)
            self.__on_probe_complete(self.__idx)
        else:
            self.probe_failed()
            self.actionButton.setToolTip('No audio streams')

    def recalc_ffmpeg_cmd(self):
        '''
        Calculates an ffmpeg cmd for the selected options.
        :return:
        '''
        self.executor.update_spec(self.audioStreams.currentIndex(), -1, self.__dialog.monoMix.isChecked())
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
        Pops up a message box containing the command or the result.
        '''
        msg_box = FFMpegDetailsDialog(self.executor.file, self.__dialog)
        if self.__result is None:
            msg_box.message.setText(f"Command")
            msg_box.details.setPlainText(self.executor.ffmpeg_cli)
        else:
            msg_box.message.setText(f"Result")
            msg_box.details.setPlainText(self.__result)
        msg_box.show()

    def __repr__(self):
        return self.__filename


class ProbeJobSignals(QObject):
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
        self.__signals = ProbeJobSignals()
        self.__signals.started.connect(candidate.probe_start)
        self.__signals.errored.connect(candidate.probe_failed)
        self.__signals.finished.connect(candidate.probe_complete)

    def run(self):
        logger.info(f">> ProbeJob.run {self.__candidate.executor.file}")
        self.__signals.started.emit()
        from ffmpeg import Error
        try:
            self.__candidate.executor.probe_file()
            self.__signals.finished.emit()
        except Error as err:
            errorMsg = err.stderr.decode('utf-8') if err.stderr is not None else 'no stderr available'
            logger.error(f"ffprobe {self.__candidate.executor.file} failed [msg: {errorMsg}]")
            self.__signals.errored.emit()
        except Exception as e:
            try:
                logger.exception(f"Probe {self.__candidate.executor.file} failed", e)
            except:
                logger.exception(f"Probe {self.__candidate.executor.file} failed, unable to format exception")
            finally:
                self.__signals.errored.emit()
        logger.info(f"<< ProbeJob.run {self.__candidate.executor.file}")
