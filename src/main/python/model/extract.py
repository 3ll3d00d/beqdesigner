import datetime
import logging
import math
import os
from pathlib import Path

import qtawesome as qta
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor
from qtpy.QtMultimedia import QSound
from qtpy.QtWidgets import QDialog, QFileDialog, QStatusBar, QDialogButtonBox, QMessageBox

from model.ffmpeg import Executor, ViewProbeDialog, SIGNAL_CONNECTED, SIGNAL_ERROR, SIGNAL_COMPLETE, get_duration, \
    parse_audio_stream
from model.preferences import EXTRACTION_OUTPUT_DIR, EXTRACTION_NOTIFICATION_SOUND
from model.signal import AutoWavLoader
from ui.extract import Ui_extractAudioDialog

logger = logging.getLogger('extract')

# copied from https://trac.ffmpeg.org/wiki/AudioChannelManipulation

CHANNEL_LAYOUTS = {
    'stereo': ['FL', 'FR'],
    '2.1': ['FL', 'FR', 'LFE'],
    '3.0': ['FL', 'FR', 'FC'],
    '3.0(back)': ['FL', 'FR', 'BC'],
    '4.0': ['FL', 'FR', 'FC', 'BC'],
    'quad': ['FL', 'FR', 'BL', 'BR'],
    'quad(side)': ['FL', 'FR', 'SL', 'SR'],
    '3.1': ['FL', 'FR', 'FC', 'LFE'],
    '5.0': ['FL', 'FR', 'FC', 'BL', 'BR'],
    '5.0(side)': ['FL', 'FR', 'FC', 'SL', 'SR'],
    '4.1': ['FL', 'FR', 'FC', 'LFE', 'BC'],
    '5.1': ['FL', 'FR', 'FC', 'LFE', 'BL', 'BR'],
    '5.1(side)': ['FL', 'FR', 'FC', 'LFE', 'SL', 'SR'],
    '6.0': ['FL', 'FR', 'FC', 'BC', 'SL', 'SR'],
    '6.0(front)': ['FL', 'FR', 'FLC', 'FRC', 'SL', 'SR'],
    'hexagonal': ['FL', 'FR', 'FC', 'BL', 'BR', 'BC'],
    '6.1': ['FL', 'FR', 'FC', 'LFE', 'BC', 'SL', 'SR'],
    '6.1(back)': ['FL', 'FR', 'FC', 'LFE', 'BL', 'BR', 'BC'],
    '6.1(front)': ['FL', 'FR', 'LFE', 'FLC', 'FRC', 'SL', 'SR'],
    '7.0': ['FL', 'FR', 'FC', 'BL', 'BR', 'SL', 'SR'],
    '7.0(front)': ['FL', 'FR', 'FC', 'FLC', 'FRC', 'SL', 'SR'],
    '7.1': ['FL', 'FR', 'FC', 'LFE', 'BL', 'BR', 'SL', 'SR'],
    '7.1(wide)': ['FL', 'FR', 'FC', 'LFE', 'BL', 'BR', 'FLC', 'FRC'],
    '7.1(wide-side)': ['FL', 'FR', 'FC', 'LFE', 'FLC', 'FRC', 'SL', 'SR'],
    'octagonal': ['FL', 'FR', 'FC', 'BL', 'BR', 'BC', 'SL', 'SR'],
    'hexadecagonal': ['FL', 'FR', 'FC', 'BL', 'BR', 'BC', 'SL', 'SR', 'TFL', 'TFC', 'TFR', 'TBL', 'TBC', 'TBR', 'WL',
                      'WR'],
    'downmix': ['DL', 'DR']
}

UNKNOWN_CHANNEL_LAYOUTS = {
    2: CHANNEL_LAYOUTS['stereo'],
    3: CHANNEL_LAYOUTS['2.1'],
    4: CHANNEL_LAYOUTS['3.1'],
    5: CHANNEL_LAYOUTS['4.1'],
    6: CHANNEL_LAYOUTS['5.1'],
    8: CHANNEL_LAYOUTS['7.1']
}


def get_channel_name(text, channel, channel_count, channel_layout_name='unknown'):
    '''
    Appends a named channel to the given name.
    :param text: the prefix.
    :param channel: the channel idx (0 based)
    :param channel_count: the channel count.
    :param channel_layout_name: the channel layout.
    :return: the named channel.
    '''
    if channel_count == 1:
        return text
    else:
        if channel_layout_name == 'unknown' and channel_count in UNKNOWN_CHANNEL_LAYOUTS:
            return f"{text}_{UNKNOWN_CHANNEL_LAYOUTS[channel_count][channel]}"
        elif channel_layout_name in CHANNEL_LAYOUTS:
            return f"{text}_{CHANNEL_LAYOUTS[channel_layout_name][channel]}"
        else:
            return f"{text}_c{channel+1}"


class ExtractAudioDialog(QDialog, Ui_extractAudioDialog):
    '''
    Allows user to load a signal, processing it if necessary.
    '''

    def __init__(self, parent, preferences, signal_model):
        super(ExtractAudioDialog, self).__init__(parent)
        self.setupUi(self)
        self.showProbeButton.setIcon(qta.icon('fa.info'))
        self.inputFilePicker.setIcon(qta.icon('fa.folder-open-o'))
        self.targetDirPicker.setIcon(qta.icon('fa.folder-open-o'))
        self.statusBar = QStatusBar()
        self.gridLayout.addWidget(self.statusBar, 5, 1, 1, 1)
        self.__preferences = preferences
        self.__signal_model = signal_model
        self.__executor = None
        self.__sound = None
        self.__extracted = False
        defaultOutputDir = self.__preferences.get(EXTRACTION_OUTPUT_DIR)
        if os.path.isdir(defaultOutputDir):
            self.targetDir.setText(defaultOutputDir)
        self.__reinit_fields()

    def selectFile(self):
        self.__reinit_fields()
        dialog = QFileDialog(parent=self)
        dialog.setFileMode(QFileDialog.ExistingFile)
        # dialog.setNameFilter()
        dialog.setWindowTitle('Select Audio or Video File')
        if dialog.exec():
            selected = dialog.selectedFiles()
            if len(selected) > 0:
                self.inputFile.setText(selected[0])
                self.__probe_file()

    def __reinit_fields(self):
        '''
        Resets various fields and temporary state.
        '''
        if self.__sound is not None:
            if not self.__sound.isFinished():
                self.__sound.stop()
                self.__sound = None
        self.audioStreams.clear()
        self.statusBar.clearMessage()
        self.__executor = None
        self.__extracted = False
        self.__stream_duration_micros = []
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(False)
        self.buttonBox.button(QDialogButtonBox.Ok).setText('Extract')
        self.signalName.setEnabled(False)
        self.signalNameLabel.setEnabled(False)
        self.signalName.setText('')
        self.inputFilePicker.setEnabled(True)
        self.audioStreams.setEnabled(False)
        self.channelCount.setEnabled(False)
        self.lfeChannelIndex.setEnabled(False)
        self.monoMix.setEnabled(False)
        self.targetDirPicker.setEnabled(True)
        self.outputFilename.setEnabled(False)
        self.showProbeButton.setEnabled(False)
        self.ffmpegCommandLine.clear()
        self.ffmpegCommandLine.setEnabled(False)
        self.ffmpegOutput.clear()
        self.ffmpegOutput.setEnabled(False)
        self.ffmpegProgress.setEnabled(False)
        self.ffmpegProgressLabel.setEnabled(False)
        self.ffmpegProgress.setValue(0)

    def __probe_file(self):
        '''
        Probes the specified file using ffprobe in order to discover the audio streams.
        '''
        file_name = self.inputFile.text()
        self.__executor = Executor(file_name, self.targetDir.text())
        self.__executor.progress_handler = self.__handle_ffmpeg_process
        from app import wait_cursor
        with wait_cursor(f"Probing {file_name}"):
            self.__executor.probe_file()
            self.showProbeButton.setEnabled(True)
        if self.__executor.has_audio():
            for a in self.__executor.audio_stream_data:
                text, duration_micros = parse_audio_stream(a)
                self.audioStreams.addItem(text)
                self.__stream_duration_micros.append(duration_micros)
            self.audioStreams.setEnabled(True)
            self.channelCount.setEnabled(True)
            self.lfeChannelIndex.setEnabled(True)
            self.monoMix.setEnabled(True)
            self.outputFilename.setEnabled(True)
            self.ffmpegCommandLine.setEnabled(True)
        else:
            self.statusBar.showMessage(f"{file_name} contains no audio streams!")

    def updateFfmpegSpec(self):
        '''
        Creates a new ffmpeg command for the specified channel layout.
        '''
        if self.__executor is not None:
            self.__executor.update_spec(self.audioStreams.currentIndex(), self.monoMix.isChecked())
            self.__init_channel_count_fields(self.__executor.channel_count, lfe_index=self.__executor.lfe_idx)
            self.__display_command_info()
            self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(True)

    def __display_command_info(self):
        self.outputFilename.setText(self.__executor.output_file_name)
        self.ffmpegCommandLine.setPlainText(self.__executor.ffmpeg_cli)

    def updateOutputFilename(self):
        '''
        Updates the output file name.
        '''
        if self.__executor is not None:
            self.__executor.output_file_name = self.outputFilename.text()
            self.__display_command_info()

    def overrideFfmpegSpec(self, _):
        if self.__executor is not None:
            self.__executor.override('custom', self.channelCount.value(), self.lfeChannelIndex.value())
            self.__display_command_info()

    def toggleMonoMix(self):
        '''
        Reacts to the change in mono vs multichannel target.
        '''
        if self.audioStreams.count() > 0 and self.__executor is not None:
            self.__executor.mono_mix = self.monoMix.isChecked()
            self.__display_command_info()

    def __init_channel_count_fields(self, channels, lfe_index=0):
        self.lfeChannelIndex.setMaximum(channels)
        self.lfeChannelIndex.setValue(lfe_index)
        self.channelCount.setMaximum(channels)
        self.channelCount.setValue(channels)

    def reject(self):
        '''
        Stops any sound that is playing and exits.
        '''
        if self.__sound is not None and not self.__sound.isFinished():
            self.__sound.stop()
            self.__sound = None
        QDialog.reject(self)

    def accept(self):
        '''
        Executes the ffmpeg command.
        '''
        if self.__extracted is False:
            self.__extract()
        else:
            if self.__create_signals():
                QDialog.accept(self)

    def __create_signals(self):
        '''
        Creates signals from the output file just created.
        :return: True if we created the signals.
        '''
        loader = AutoWavLoader(self.__preferences)
        output_file = self.__executor.get_output_path()
        if os.path.exists(output_file):
            from app import wait_cursor
            with wait_cursor(f"Creating signals for {output_file}"):
                logger.info(f"Creating signals for {output_file}")
                name_provider = lambda channel, channel_count: get_channel_name(self.signalName.text(), channel,
                                                                                channel_count,
                                                                                channel_layout_name=self.__executor.channel_layout_name)
                signals = loader.auto_load(output_file, name_provider)
                if len(signals) > 0:
                    for s in signals:
                        logger.info(f"Adding signal {s.name}")
                        self.__signal_model.add(s)
                    return True
        else:
            msg_box = QMessageBox()
            msg_box.setText(f"Extracted audio file does not exist at: \n\n {output_file}")
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setWindowTitle('Unexpected Error')
            msg_box.exec()
        return False

    def __extract(self):
        '''
        Triggers the ffmpeg command.
        '''
        if self.__executor is not None:
            logger.info(f"Extracting {self.outputFilename.text()} from {self.inputFile.text()}")
            self.__executor.execute()

    def __handle_ffmpeg_process(self, key, value):
        '''
        Handles progress reports from ffmpeg in order to communicate status via the progress bar. Used as a slot
        connected to a signal emitted by the AudioExtractor.
        :param key: the key.
        :param value: the value.
        '''
        if key == SIGNAL_CONNECTED:
            self.__extract_started()
        elif key == 'out_time_ms':
            out_time_ms = int(value)
            total_micros = self.__stream_duration_micros[self.audioStreams.currentIndex()]
            logger.debug(f"{self.inputFile.text()} -- {key}={value} vs {total_micros}")
            if total_micros > 0:
                progress = math.ceil((out_time_ms / total_micros) * 100.0)
                self.ffmpegProgress.setValue(progress)
                self.ffmpegProgress.setTextVisible(True)
                self.ffmpegProgress.setFormat(f"{round(out_time_ms/1000000)} of {round(total_micros/1000000)}s")
        elif key == SIGNAL_ERROR:
            self.__extract_complete(value, False)
        elif key == SIGNAL_COMPLETE:
            self.__extract_complete(value, True)

    def __extract_started(self):
        '''
        Changes the UI to signal that extraction has started
        '''
        self.inputFilePicker.setEnabled(False)
        self.audioStreams.setEnabled(False)
        self.channelCount.setEnabled(False)
        self.lfeChannelIndex.setEnabled(False)
        self.monoMix.setEnabled(False)
        self.targetDirPicker.setEnabled(False)
        self.outputFilename.setEnabled(False)
        self.ffmpegOutput.setEnabled(True)
        self.ffmpegProgress.setEnabled(True)
        self.ffmpegProgressLabel.setEnabled(True)
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(False)
        palette = QPalette(self.ffmpegProgress.palette())
        palette.setColor(QPalette.Highlight, QColor(Qt.green))
        self.ffmpegProgress.setPalette(palette)

    def __extract_complete(self, result, success):
        '''
        triggered when the extraction thread completes.
        '''
        if self.__executor is not None:
            if success:
                logger.info(f"Extraction complete for {self.outputFilename.text()}")
                self.ffmpegProgress.setValue(100)
                self.__extracted = True
                self.signalName.setEnabled(True)
                self.signalNameLabel.setEnabled(True)
                self.signalName.setText(Path(self.outputFilename.text()).resolve().stem)
                self.buttonBox.button(QDialogButtonBox.Ok).setText('Create Signals')
            else:
                logger.error(f"Extraction failed for {self.outputFilename.text()}")
                palette = QPalette(self.ffmpegProgress.palette())
                palette.setColor(QPalette.Highlight, QColor(Qt.red))
                self.ffmpegProgress.setPalette(palette)
                self.statusBar.showMessage('Extraction failed', 5000)

            self.ffmpegOutput.setPlainText(result)
            self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(True)
            audio = self.__preferences.get(EXTRACTION_NOTIFICATION_SOUND)
            if audio is not None:
                logger.debug(f"Playing {audio}")
                self.__sound = QSound(audio)
                self.__sound.play()

    def showProbeInDetail(self):
        '''
        shows a tree widget containing the contents of the probe to allow the raw probe info to be visible.
        '''
        if self.__executor is not None:
            ViewProbeDialog(self.inputFile.text(), self.__executor.probe, parent=self).exec()

    def setTargetDirectory(self):
        '''
        Sets the target directory based on the user selection.
        '''
        dialog = QFileDialog(parent=self)
        dialog.setFileMode(QFileDialog.DirectoryOnly)
        dialog.setWindowTitle(f"Select Output Directory")
        if dialog.exec():
            selected = dialog.selectedFiles()
            if len(selected) > 0:
                self.targetDir.setText(selected[0])
                if self.__executor is not None:
                    self.__executor.target_dir = selected[0]
                    self.__display_command_info()
