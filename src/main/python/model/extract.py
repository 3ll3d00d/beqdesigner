import logging
import math
import os
from pathlib import Path

import qtawesome as qta
from qtpy.QtCore import Qt
from qtpy.QtGui import QPalette, QColor
from qtpy.QtMultimedia import QSound
from qtpy.QtWidgets import QDialog, QFileDialog, QStatusBar, QDialogButtonBox, QMessageBox

from model.ffmpeg import Executor, ViewProbeDialog, SIGNAL_CONNECTED, SIGNAL_ERROR, SIGNAL_COMPLETE, parse_audio_stream, \
    get_channel_name, parse_video_stream
from model.preferences import EXTRACTION_OUTPUT_DIR, EXTRACTION_NOTIFICATION_SOUND
from model.signal import AutoWavLoader
from ui.edit_mapping import Ui_editMappingDialog
from ui.extract import Ui_extractAudioDialog

logger = logging.getLogger('extract')


class ExtractAudioDialog(QDialog, Ui_extractAudioDialog):
    '''
    Allows user to load a signal, processing it if necessary.
    '''

    def __init__(self, parent, preferences, signal_model, is_remux=False):
        super(ExtractAudioDialog, self).__init__(parent)
        self.setupUi(self)
        self.showProbeButton.setIcon(qta.icon('fa.info'))
        self.inputFilePicker.setIcon(qta.icon('fa.folder-open-o'))
        self.targetDirPicker.setIcon(qta.icon('fa.folder-open-o'))
        self.statusBar = QStatusBar()
        self.statusBar.setSizeGripEnabled(False)
        self.boxLayout.addWidget(self.statusBar)
        self.__preferences = preferences
        self.__signal_model = signal_model
        self.__executor = None
        self.__sound = None
        self.__extracted = False
        self.__stream_duration_micros = []
        self.__is_remux = is_remux
        if self.__is_remux:
            self.setWindowTitle('Remux Audio')
        defaultOutputDir = self.__preferences.get(EXTRACTION_OUTPUT_DIR)
        if os.path.isdir(defaultOutputDir):
            self.targetDir.setText(defaultOutputDir)
        self.__reinit_fields()
        self.filterMapping.itemDoubleClicked.connect(self.show_mapping_dialog)

    def show_mapping_dialog(self, item):
        ''' Shows the edit mapping dialog '''
        if len(self.__signal_model) > 0:
            channel_idx = self.filterMapping.indexFromItem(item).row()
            mapped_filter = self.__executor.channel_to_filter.get(channel_idx, None)
            EditMappingDialog(self, channel_idx, self.__signal_model, mapped_filter,
                              self.map_filter_to_channel).exec()

    def map_filter_to_channel(self, channel_idx, signal_name):
        ''' updates the mapping of the given signal to the specified channel idx '''
        if self.audioStreams.count() > 0 and self.__executor is not None:
            self.__executor.map_filter_to_channel(channel_idx, signal_name)
            self.__display_command_info()

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
        self.videoStreams.clear()
        self.statusBar.clearMessage()
        self.__executor = None
        self.__extracted = False
        self.__stream_duration_micros = []
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(False)
        self.buttonBox.button(QDialogButtonBox.Ok).setText('Remux' if self.__is_remux else 'Extract')
        if self.__is_remux:
            self.signalName.setVisible(False)
            self.signalNameLabel.setVisible(False)
            self.filterMapping.setVisible(True)
            self.filterMappingLabel.setVisible(True)
            self.includeOriginalAudio.setVisible(True)
        else:
            self.signalName.setEnabled(True)
            self.signalNameLabel.setEnabled(True)
            self.signalName.setText('')
            self.filterMapping.setVisible(False)
            self.filterMappingLabel.setVisible(False)
            self.includeOriginalAudio.setVisible(False)
        self.monoMix.setChecked(False)
        self.decimateAudio.setChecked(False)
        self.includeOriginalAudio.setChecked(False)
        self.compressAudio.setChecked(False)
        self.monoMix.setEnabled(False)
        self.decimateAudio.setEnabled(False)
        self.compressAudio.setEnabled(False)
        self.includeOriginalAudio.setEnabled(False)
        self.inputFilePicker.setEnabled(True)
        self.audioStreams.setEnabled(False)
        self.videoStreams.setEnabled(False)
        self.channelCount.setEnabled(False)
        self.lfeChannelIndex.setEnabled(False)
        self.targetDirPicker.setEnabled(True)
        self.outputFilename.setEnabled(False)
        self.showProbeButton.setEnabled(False)
        self.filterMapping.setEnabled(False)
        self.filterMapping.clear()
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
        self.__executor = Executor(file_name, self.targetDir.text(), self.monoMix.isChecked(),
                                   self.decimateAudio.isChecked(), self.compressAudio.isChecked(),
                                   self.includeOriginalAudio.isChecked(),
                                   signal_model=self.__signal_model if self.__is_remux else None)
        self.__executor.progress_handler = self.__handle_ffmpeg_process
        from app import wait_cursor
        with wait_cursor(f"Probing {file_name}"):
            self.__executor.probe_file()
            self.showProbeButton.setEnabled(True)
        if self.__executor.has_audio():
            for a in self.__executor.audio_stream_data:
                text, duration_micros = parse_audio_stream(self.__executor.probe, a)
                self.audioStreams.addItem(text)
                self.__stream_duration_micros.append(duration_micros)
            self.videoStreams.addItem('No Video')
            for a in self.__executor.video_stream_data:
                self.videoStreams.addItem(parse_video_stream(self.__executor.probe, a))
            self.audioStreams.setEnabled(True)
            self.videoStreams.setEnabled(True)
            self.channelCount.setEnabled(True)
            self.lfeChannelIndex.setEnabled(True)
            self.monoMix.setEnabled(True)
            self.decimateAudio.setEnabled(True)
            self.compressAudio.setEnabled(True)
            self.includeOriginalAudio.setEnabled(True)
            self.outputFilename.setEnabled(True)
            self.ffmpegCommandLine.setEnabled(True)
            self.filterMapping.setEnabled(True)
        else:
            self.statusBar.showMessage(f"{file_name} contains no audio streams!")

    def updateFfmpegSpec(self):
        '''
        Creates a new ffmpeg command for the specified channel layout.
        '''
        if self.__executor is not None:
            self.__executor.update_spec(self.audioStreams.currentIndex(), self.videoStreams.currentIndex() - 1,
                                        self.monoMix.isChecked())
            self.__init_channel_count_fields(self.__executor.channel_count, lfe_index=self.__executor.lfe_idx)
            self.__display_command_info()
            self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(True)

    def __display_command_info(self):
        self.outputFilename.setText(self.__executor.output_file_name)
        self.ffmpegCommandLine.setPlainText(self.__executor.ffmpeg_cli)
        self.filterMapping.clear()
        for channel_idx, signal in self.__executor.channel_to_filter.items():
            self.filterMapping.addItem(f"Channel {channel_idx+1} -> {signal.name if signal else 'Passthrough'}")

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

    def toggle_decimate_audio(self):
        '''
        Reacts to the change in decimation.
        '''
        if self.audioStreams.count() > 0 and self.__executor is not None:
            self.__executor.decimate_audio = self.decimateAudio.isChecked()
            self.__display_command_info()

    def toggle_compress_audio(self):
        '''
        Reacts to the change in decimation.
        '''
        if self.audioStreams.count() > 0 and self.__executor is not None:
            self.__executor.compress_audio = self.compressAudio.isChecked()
            self.__display_command_info()

    def toggle_include_original_audio(self):
        '''
        Reacts to the change in original audio selection.
        '''
        if self.audioStreams.count() > 0 and self.__executor is not None:
            self.__executor.include_original_audio = self.includeOriginalAudio.isChecked()
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
                progress = (out_time_ms / total_micros) * 100.0
                self.ffmpegProgress.setValue(math.ceil(progress))
                self.ffmpegProgress.setTextVisible(True)
                self.ffmpegProgress.setFormat(f"{round(progress, 2):.2f}%")
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
        self.videoStreams.setEnabled(False)
        self.channelCount.setEnabled(False)
        self.lfeChannelIndex.setEnabled(False)
        self.monoMix.setEnabled(False)
        self.decimateAudio.setEnabled(False)
        self.compressAudio.setEnabled(False)
        self.includeOriginalAudio.setEnabled(False)
        self.targetDirPicker.setEnabled(False)
        self.outputFilename.setEnabled(False)
        self.filterMapping.setEnabled(False)
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
                if not self.__is_remux:
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


class EditMappingDialog(QDialog, Ui_editMappingDialog):
    ''' Allows the user to override the signal to channel mapping '''

    def __init__(self, parent, channel_idx, signal_model, selected_signal, on_change_handler):
        super(EditMappingDialog, self).__init__(parent)
        self.setupUi(self)
        self.channelIdx.setText(str(channel_idx+1))
        for idx, s in enumerate(signal_model):
            self.signal.addItem(s.name)
            if s.name == selected_signal.name:
                self.signal.setCurrentIndex(idx)

        def pass_idx_with_text(text):
            on_change_handler(channel_idx, text)

        self.signal.currentTextChanged.connect(pass_idx_with_text)
