import logging
import math
import os
from pathlib import Path

import qtawesome as qta
from qtpy.QtCore import Qt, QTime
from qtpy.QtGui import QPalette, QColor, QFont
from qtpy.QtMultimedia import QSound
from qtpy.QtWidgets import QDialog, QFileDialog, QStatusBar, QDialogButtonBox, QMessageBox

from model.ffmpeg import Executor, ViewProbeDialog, SIGNAL_CONNECTED, SIGNAL_ERROR, SIGNAL_COMPLETE, parse_audio_stream, \
    get_channel_name, parse_video_stream
from model.preferences import EXTRACTION_OUTPUT_DIR, EXTRACTION_NOTIFICATION_SOUND, ANALYSIS_TARGET_FS, \
    EXTRACTION_MIX_MONO, EXTRACTION_DECIMATE, EXTRACTION_INCLUDE_ORIGINAL, EXTRACTION_INCLUDE_SUBTITLES, \
    EXTRACTION_COMPRESS, COMPRESS_FORMAT_OPTIONS, COMPRESS_FORMAT_FLAC, COMPRESS_FORMAT_NATIVE, COMPRESS_FORMAT_EAC3
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
        for f in COMPRESS_FORMAT_OPTIONS:
            self.audioFormat.addItem(f)
        self.showProbeButton.setIcon(qta.icon('fa5s.info'))
        self.showRemuxCommand.setIcon(qta.icon('fa5s.info'))
        self.inputFilePicker.setIcon(qta.icon('fa5s.folder-open'))
        self.targetDirPicker.setIcon(qta.icon('fa5s.folder-open'))
        self.limitRange.setIcon(qta.icon('fa5s.cut'))
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
        self.showRemuxCommand.setVisible(self.__is_remux)
        defaultOutputDir = self.__preferences.get(EXTRACTION_OUTPUT_DIR)
        if os.path.isdir(defaultOutputDir):
            self.targetDir.setText(defaultOutputDir)
        self.__reinit_fields()
        self.filterMapping.itemDoubleClicked.connect(self.show_mapping_dialog)

    def show_remux_cmd(self):
        ''' Pops the ffmpeg command into a message box '''
        if self.__executor is not None and self.__executor.filter_complex_script_content is not None:
            msg_box = QMessageBox()
            font = QFont()
            font.setFamily("Consolas")
            font.setPointSize(8)
            msg_box.setFont(font)
            msg_box.setText(self.__executor.filter_complex_script_content.replace(';', ';\n'))
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setWindowTitle('Remux Script')
            msg_box.exec()

    def show_mapping_dialog(self, item):
        ''' Shows the edit mapping dialog '''
        if len(self.__signal_model) > 0:
            channel_idx = self.filterMapping.indexFromItem(item).row()
            mapped_filter = self.__executor.channel_to_filter.get(channel_idx, None)
            EditMappingDialog(self, channel_idx, self.__signal_model, mapped_filter, self.filterMapping.count(),
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
            self.includeSubtitles.setVisible(True)
            self.gainOffset.setVisible(True)
            self.gainOffsetLabel.setVisible(True)
            self.gainOffset.setEnabled(False)
        else:
            self.signalName.setText('')
            self.filterMapping.setVisible(False)
            self.filterMappingLabel.setVisible(False)
            self.includeOriginalAudio.setVisible(False)
            self.includeSubtitles.setVisible(False)
            self.gainOffset.setVisible(False)
            self.gainOffsetLabel.setVisible(False)
        self.eacBitRate.setVisible(False)
        self.monoMix.setChecked(self.__preferences.get(EXTRACTION_MIX_MONO))
        self.decimateAudio.setChecked(self.__preferences.get(EXTRACTION_DECIMATE))
        self.includeOriginalAudio.setChecked(self.__preferences.get(EXTRACTION_INCLUDE_ORIGINAL))
        self.includeSubtitles.setChecked(self.__preferences.get(EXTRACTION_INCLUDE_SUBTITLES))
        if self.__preferences.get(EXTRACTION_COMPRESS):
            self.audioFormat.setCurrentText(COMPRESS_FORMAT_FLAC)
        else:
            self.audioFormat.setCurrentText(COMPRESS_FORMAT_NATIVE)
        self.monoMix.setEnabled(False)
        self.decimateAudio.setEnabled(False)
        self.audioFormat.setEnabled(False)
        self.eacBitRate.setEnabled(False)
        self.includeOriginalAudio.setEnabled(False)
        self.includeSubtitles.setEnabled(False)
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
        self.rangeFrom.setEnabled(False)
        self.rangeSeparatorLabel.setEnabled(False)
        self.rangeTo.setEnabled(False)
        self.limitRange.setEnabled(False)
        self.signalName.setEnabled(False)
        self.signalNameLabel.setEnabled(False)
        self.showRemuxCommand.setEnabled(False)

    def __probe_file(self):
        '''
        Probes the specified file using ffprobe in order to discover the audio streams.
        '''
        file_name = self.inputFile.text()
        self.__executor = Executor(file_name, self.targetDir.text(),
                                   mono_mix=self.monoMix.isChecked(),
                                   decimate_audio=self.decimateAudio.isChecked(),
                                   audio_format=self.audioFormat.currentText(),
                                   audio_bitrate=self.eacBitRate.value(),
                                   include_original=self.includeOriginalAudio.isChecked(),
                                   include_subtitles=self.includeSubtitles.isChecked(),
                                   signal_model=self.__signal_model if self.__is_remux else None,
                                   decimate_fs=self.__preferences.get(ANALYSIS_TARGET_FS))
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
            if self.__is_remux and self.videoStreams.count() > 1:
                if self.audioFormat.findText(COMPRESS_FORMAT_EAC3) == -1:
                    self.audioFormat.addItem(COMPRESS_FORMAT_EAC3)
                if self.__preferences.get(EXTRACTION_COMPRESS):
                    self.audioFormat.setCurrentText(COMPRESS_FORMAT_EAC3)
                    self.eacBitRate.setVisible(True)
                else:
                    self.audioFormat.setCurrentText(COMPRESS_FORMAT_NATIVE)
                    self.eacBitRate.setVisible(False)
                self.videoStreams.setCurrentIndex(1)
            self.audioStreams.setEnabled(True)
            self.videoStreams.setEnabled(True)
            self.channelCount.setEnabled(True)
            self.lfeChannelIndex.setEnabled(True)
            self.monoMix.setEnabled(True)
            self.decimateAudio.setEnabled(True)
            self.audioFormat.setEnabled(True)
            self.eacBitRate.setEnabled(True)
            self.includeOriginalAudio.setEnabled(True)
            self.outputFilename.setEnabled(True)
            self.ffmpegCommandLine.setEnabled(True)
            self.filterMapping.setEnabled(True)
            self.limitRange.setEnabled(True)
            self.showRemuxCommand.setEnabled(True)
            self.__fit_options_to_selected()
        else:
            self.statusBar.showMessage(f"{file_name} contains no audio streams!")

    def onVideoStreamChange(self, idx):
        if idx == 0:
            eac_idx = self.audioFormat.findText(COMPRESS_FORMAT_EAC3)
            if eac_idx > -1:
                self.audioFormat.removeItem(eac_idx)
            if self.__preferences.get(EXTRACTION_COMPRESS):
                self.audioFormat.setCurrentText(COMPRESS_FORMAT_FLAC)
            else:
                self.audioFormat.setCurrentText(COMPRESS_FORMAT_NATIVE)
        else:
            if self.audioFormat.findText(COMPRESS_FORMAT_EAC3) == -1:
                self.audioFormat.addItem(COMPRESS_FORMAT_EAC3)
            if self.__preferences.get(EXTRACTION_COMPRESS):
                self.audioFormat.setCurrentText(COMPRESS_FORMAT_EAC3)
            else:
                self.audioFormat.setCurrentText(COMPRESS_FORMAT_NATIVE)
        self.updateFfmpegSpec()

    def updateFfmpegSpec(self):
        '''
        Creates a new ffmpeg command for the specified channel layout.
        '''
        if self.__executor is not None:
            self.__executor.update_spec(self.audioStreams.currentIndex(), self.videoStreams.currentIndex() - 1,
                                        self.monoMix.isChecked())

            self.__init_channel_count_fields(self.__executor.channel_count, lfe_index=self.__executor.lfe_idx)
            self.__fit_options_to_selected()
            self.__display_command_info()
            self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(True)

    def __fit_options_to_selected(self):
        # if we have no video then the output cannot contain multiple streams
        if self.videoStreams.currentIndex() == 0:
            self.includeOriginalAudio.setChecked(False)
            self.includeOriginalAudio.setEnabled(False)
            self.includeSubtitles.setChecked(False)
            self.includeSubtitles.setEnabled(False)
        else:
            self.includeOriginalAudio.setEnabled(True)
            self.includeSubtitles.setEnabled(True)
        # don't allow mono mix option if the stream is mono
        if self.channelCount.value() == 1:
            self.monoMix.setChecked(False)
            self.monoMix.setEnabled(False)
        else:
            self.monoMix.setEnabled(True)

    def __display_command_info(self):
        self.outputFilename.setText(self.__executor.output_file_name)
        self.ffmpegCommandLine.setPlainText(self.__executor.ffmpeg_cli)
        self.filterMapping.clear()
        for channel_idx, signal in self.__executor.channel_to_filter.items():
            self.filterMapping.addItem(f"Channel {channel_idx + 1} -> {signal.name if signal else 'Passthrough'}")

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
            self.__fit_options_to_selected()
            self.__display_command_info()

    def toggle_decimate_audio(self):
        '''
        Reacts to the change in decimation.
        '''
        if self.audioStreams.count() > 0 and self.__executor is not None:
            self.__executor.decimate_audio = self.decimateAudio.isChecked()
            self.__display_command_info()

    def change_audio_format(self, audio_format):
        '''
        Reacts to the change in audio format.
        '''
        if self.audioStreams.count() > 0 and self.__executor is not None:
            self.__executor.audio_format = audio_format
            if audio_format == COMPRESS_FORMAT_EAC3:
                self.eacBitRate.setVisible(True)
                self.__executor.audio_bitrate = self.eacBitRate.value()
            else:
                self.eacBitRate.setVisible(False)
            self.__display_command_info()

    def change_audio_bitrate(self, bitrate):
        ''' Allows the bitrate to be updated '''
        if self.__executor is not None:
            self.__executor.audio_bitrate = bitrate
            self.__display_command_info()

    def update_original_audio(self):
        '''
        Reacts to the change in original audio selection.
        '''
        if self.audioStreams.count() > 0 and self.__executor is not None:
            if self.includeOriginalAudio.isChecked():
                self.__executor.include_original_audio = True
                self.__executor.original_audio_offset = self.gainOffset.value()
                self.gainOffset.setEnabled(True)
            else:
                self.__executor.include_original_audio = False
                self.__executor.original_audio_offset = 0.0
                self.gainOffset.setEnabled(False)
            self.__display_command_info()

    def toggle_include_subtitles(self):
        '''
        Reacts to the change in subtitles selection.
        '''
        if self.audioStreams.count() > 0 and self.__executor is not None:
            self.__executor.include_subtitles = self.includeSubtitles.isChecked()
            self.__display_command_info()

    def toggleMonoMix(self):
        '''
        Reacts to the change in mono vs multichannel target.
        '''
        if self.audioStreams.count() > 0 and self.__executor is not None:
            self.__executor.mono_mix = self.monoMix.isChecked()
            self.__display_command_info()

    def toggle_range(self):
        ''' toggles whether the range is enabled or not '''
        if self.limitRange.isChecked():
            self.limitRange.setText('Cut')
            if self.audioStreams.count() > 0:
                duration_ms = int(self.__stream_duration_micros[self.audioStreams.currentIndex()] / 1000)
                if duration_ms > 1:
                    from model.report import block_signals
                    with block_signals(self.rangeFrom):
                        self.rangeFrom.setTimeRange(QTime.fromMSecsSinceStartOfDay(0),
                                                    QTime.fromMSecsSinceStartOfDay(duration_ms - 1))
                        self.rangeFrom.setTime(QTime.fromMSecsSinceStartOfDay(0))
                        self.rangeFrom.setEnabled(True)
                    self.rangeSeparatorLabel.setEnabled(True)
                    with block_signals(self.rangeTo):
                        self.rangeTo.setEnabled(True)
                        self.rangeTo.setTimeRange(QTime.fromMSecsSinceStartOfDay(1),
                                                  QTime.fromMSecsSinceStartOfDay(duration_ms))
                        self.rangeTo.setTime(QTime.fromMSecsSinceStartOfDay(duration_ms))
        else:
            self.limitRange.setText('Enable')
            self.rangeFrom.setEnabled(False)
            self.rangeSeparatorLabel.setEnabled(False)
            self.rangeTo.setEnabled(False)
            if self.__executor is not None:
                self.__executor.start_time_ms = 0
                self.__executor.end_time_ms = 0

    def update_start_time(self, time):
        ''' Reacts to start time changes '''
        self.__executor.start_time_ms = time.msecsSinceStartOfDay()
        self.__display_command_info()

    def update_end_time(self, time):
        ''' Reacts to end time changes '''
        msecs = time.msecsSinceStartOfDay()
        duration_ms = int(self.__stream_duration_micros[self.audioStreams.currentIndex()] / 1000)
        self.__executor.end_time_ms = msecs if msecs != duration_ms else 0
        self.__display_command_info()

    def __init_channel_count_fields(self, channels, lfe_index=0):
        from model.report import block_signals
        with block_signals(self.lfeChannelIndex):
            self.lfeChannelIndex.setMaximum(channels)
            self.lfeChannelIndex.setValue(lfe_index)
        with block_signals(self.channelCount):
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
            if not self.__is_remux:
                self.signalName.setEnabled(True)
                self.signalNameLabel.setEnabled(True)
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
                loader.load(output_file)
                signal = loader.auto_load(name_provider, self.decimateAudio.isChecked())
                self.__signal_model.add(signal)
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
            if self.__executor.start_time_ms > 0 and self.__executor.end_time_ms > 0:
                total_micros = (self.__executor.end_time_ms - self.__executor.start_time_ms) * 1000
            elif self.__executor.end_time_ms > 0:
                total_micros = self.__executor.end_time_ms * 1000
            elif self.__executor.start_time_ms > 0:
                total_micros = self.__stream_duration_micros[self.audioStreams.currentIndex()] - (self.__executor.start_time_ms * 1000)
            else:
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
        self.audioFormat.setEnabled(False)
        self.eacBitRate.setEnabled(False)
        self.includeOriginalAudio.setEnabled(False)
        self.includeSubtitles.setEnabled(False)
        self.targetDirPicker.setEnabled(False)
        self.outputFilename.setEnabled(False)
        self.filterMapping.setEnabled(False)
        self.gainOffset.setEnabled(False)
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

    def __init__(self, parent, channel_idx, signal_model, selected_signal, channel_count, on_change_handler):
        super(EditMappingDialog, self).__init__(parent)
        self.setupUi(self)
        self.channel_idx = channel_idx
        self.channelIdx.setText(str(channel_idx + 1))
        self.signal.addItem('Passthrough')
        self.channel_count = channel_count
        for idx, s in enumerate(signal_model):
            self.signal.addItem(s.name)
        if selected_signal is not None:
            self.signal.setCurrentText(selected_signal.name)
        self.on_change_handler = on_change_handler

    def accept(self):
        filt = None if self.signal.currentText() == 'Passthrough' else self.signal.currentText()
        if self.applyToAll.isChecked():
            for idx in range(0, self.channel_count):
                self.on_change_handler(idx, filt)
        else:
            self.on_change_handler(self.channel_idx, filt)
        super().accept()


