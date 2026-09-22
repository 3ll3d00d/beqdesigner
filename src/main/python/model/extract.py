import datetime
import logging
import math
import os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import qtawesome as qta
from qtpy.QtCore import Qt, QTime, QThreadPool
from qtpy.QtGui import QPalette, QColor, QFont
from qtpy.QtMultimedia import QSoundEffect
from qtpy.QtWidgets import QDialog, QFileDialog, QStatusBar, QDialogButtonBox, QMessageBox, QVBoxLayout, \
    QTableWidget, QTableWidgetItem, QAbstractItemView, QHeaderView

from model.bdmv import is_bdmv_root, list_playlists, resolve_title
from model.ffmpeg import Executor, ViewProbeDialog, SIGNAL_CONNECTED, SIGNAL_ERROR, SIGNAL_COMPLETE, parse_audio_stream, \
    get_channel_name, parse_video_stream
from model.preferences import EXTRACTION_OUTPUT_DIR, EXTRACTION_NOTIFICATION_SOUND, ANALYSIS_TARGET_FS, \
    EXTRACTION_MIX_MONO, EXTRACTION_DECIMATE, EXTRACTION_INCLUDE_ORIGINAL, EXTRACTION_INCLUDE_SUBTITLES, \
    EXTRACTION_COMPRESS, COMPRESS_FORMAT_OPTIONS, COMPRESS_FORMAT_FLAC, COMPRESS_FORMAT_NATIVE, COMPRESS_FORMAT_EAC3, \
    BASS_MANAGEMENT_LPF_FS, COMPRESS_FORMAT_AC3, EXTRACTION_GEOMETRY, DESIGNER_DEFAULT, DESIGNER_QUEUE_DIR, Preferences
from model.signal import AutoWavLoader
from ui.edit_mapping import Ui_editMappingDialog
from ui.extract import Ui_extractAudioDialog

# model.batch/model.worklist_review/pipeline.* are imported lazily (inside the functions that need them), matching
# model/batch.py's own convention -- pipeline.orchestrate transitively imports model.merge -> model.sync ->
# model.batch, so a module-level import of either here risks the same circularity model/batch.py documents.
logger = logging.getLogger('extract')


class ExtractAudioDialog(QDialog, Ui_extractAudioDialog):
    '''
    Allows user to load a signal, processing it if necessary.
    '''

    def __init__(self, parent, preferences: Preferences, signal_model, default_signal=None, is_remux=False):
        super(ExtractAudioDialog, self).__init__(parent)
        self.setupUi(self)
        for f in COMPRESS_FORMAT_OPTIONS:
            self.audioFormat.addItem(f)
        self.showProbeButton.setIcon(qta.icon('fa5s.info'))
        self.showRemuxCommand.setIcon(qta.icon('fa5s.info'))
        self.inputFilePicker.setIcon(qta.icon('fa5s.folder-open'))
        self.inputBdFolderPicker.setIcon(qta.icon('fa5s.compact-disc'))
        self.targetDirPicker.setIcon(qta.icon('fa5s.folder-open'))
        self.calculateGainAdjustment.setIcon(qta.icon('fa5s.sliders-h'))
        self.limitRange.setIcon(qta.icon('fa5s.cut'))
        self.statusBar = QStatusBar()
        self.statusBar.setSizeGripEnabled(False)
        self.boxLayout.addWidget(self.statusBar)
        self.__preferences = preferences
        self.__signal_model = signal_model
        self.__default_signal = default_signal
        self.__executor = None
        self.__sound = None
        self.__extracted = False
        self.__stream_duration_micros = []
        self.__is_remux = is_remux
        self.__session = None
        self.__design_entry = None
        if self.__is_remux:
            self.setWindowTitle('Remux Audio')
        self.showRemuxCommand.setVisible(self.__is_remux)
        default_output_dir = self.__preferences.get(EXTRACTION_OUTPUT_DIR)
        if os.path.isdir(default_output_dir):
            self.targetDir.setText(default_output_dir)

        from pipeline.designer.registry import registered_designers
        designers = registered_designers()
        # design controls (design/designer-interface.md) are only meaningful once at least one designer is
        # registered -- which only happens via Preferences' HTTP endpoints table (register_configured_designers)
        self.has_designers = len(designers) > 0
        self.designerCombo.addItems(designers)

        self.__restore_geometry()
        self.__reinit_fields()
        self.filterMapping.itemDoubleClicked.connect(self.show_mapping_dialog)
        self.inputDrop.callback = self.__handle_drop
        self.finished.connect(self.__on_finished)

        default_designer = self.__preferences.get(DESIGNER_DEFAULT)
        if default_designer:
            idx = self.designerCombo.findText(default_designer)
            if idx != -1:
                self.designerCombo.setCurrentIndex(idx)
        default_queue_dir = self.__preferences.get(DESIGNER_QUEUE_DIR)
        if default_queue_dir and os.path.isdir(default_queue_dir):
            self.queueDirEdit.setText(default_queue_dir)
        self.designEnabled.toggled.connect(self.__toggle_design)
        self.browseQueueDirButton.clicked.connect(self.__select_queue_dir)

    def __on_finished(self):
        self.__preferences.set(EXTRACTION_GEOMETRY, self.saveGeometry())

    def __restore_geometry(self):
        ''' loads the saved window size '''
        geometry = self.__preferences.get(EXTRACTION_GEOMETRY)
        if geometry is not None:
            self.restoreGeometry(geometry)

    def __handle_drop(self, file):
        if file.startswith('file:/'):
            file = url2pathname(urlparse(file).path)
        if os.path.exists(file) and os.path.isfile(file):
            self.__reinit_fields()
            self.inputFile.setText(file)
            self.__probe_file()

    def show_remux_cmd(self):
        ''' Pops the ffmpeg command into a message box '''
        if self.__executor is not None and self.__executor.filter_complex_script_content is not None:
            msg_box = QMessageBox()
            font = QFont()
            font.setFamily("Consolas")
            font.setPointSize(8)
            msg_box.setFont(font)
            msg_box.setText(self.__executor.filter_complex_script_content.replace(';', ';\n'))
            msg_box.setIcon(QMessageBox.Icon.Information)
            msg_box.setWindowTitle('Remux Script')
            msg_box.exec()

    def show_mapping_dialog(self, item):
        ''' Shows the edit mapping dialog '''
        if len(self.__signal_model) > 0 or self.__default_signal is not None:
            channel_idx = self.filterMapping.indexFromItem(item).row()
            mapped_filter = self.__executor.channel_to_filter.get(channel_idx, None)
            EditMappingDialog(self, channel_idx, self.__signal_model, self.__default_signal,
                              mapped_filter, self.filterMapping.count(), self.map_filter_to_channel).exec()

    def map_filter_to_channel(self, channel_idx, signal):
        ''' updates the mapping of the given signal to the specified channel idx '''
        if self.audioStreams.count() > 0 and self.__executor is not None:
            self.__executor.map_filter_to_channel(channel_idx, signal)
            self.__display_command_info()

    def selectFile(self):
        self.__reinit_fields()
        dialog = QFileDialog(parent=self)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setWindowTitle('Select Audio or Video File')
        if dialog.exec():
            selected = dialog.selectedFiles()
            if len(selected) > 0:
                self.inputFile.setText(selected[0])
                self.__probe_file()

    def selectBdFolder(self):
        '''
        Lets the user pick a BD disc rip folder (containing a BDMV structure), choose which title (playlist) is
        the one to extract from and resolves that to a concrete ffmpeg input before probing it as normal.
        '''
        self.__reinit_fields()
        dialog = QFileDialog(parent=self)
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setWindowTitle('Select BD Disc Folder')
        if not dialog.exec():
            return
        selected = dialog.selectedFiles()
        if len(selected) == 0:
            return
        bdmv_root = selected[0]
        if not is_bdmv_root(bdmv_root):
            QMessageBox.warning(self, 'Not a BD disc',
                               f"{bdmv_root} does not look like a BD disc rip (expected a BDMV folder "
                               f"containing index.bdmv).")
            return
        playlists = list_playlists(bdmv_root)
        if len(playlists) == 0:
            QMessageBox.warning(self, 'No titles found', f"No playable titles were found under {bdmv_root}.")
            return
        picker = BdmvTitlePickerDialog(self, playlists)
        if not picker.exec():
            return
        try:
            resolved = resolve_title(bdmv_root, picker.selected_playlist)
        except FileNotFoundError as e:
            QMessageBox.critical(self, 'Unable to resolve title', str(e))
            return
        self.inputFile.setText(f"{resolved.display_name}  [{bdmv_root}]")
        self.__probe_file(file_name=resolved.ffmpeg_input, display_name=resolved.display_name,
                          duration_override_s=resolved.playlist.extraction_duration_s)

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
        self.buttonBox.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        self.buttonBox.button(QDialogButtonBox.StandardButton.Ok).setText('Remux' if self.__is_remux else 'Extract')
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
            self.gainOffsetLabel.setEnabled(False)
            self.calculateGainAdjustment.setVisible(True)
            self.calculateGainAdjustment.setEnabled(False)
            self.adjustRemuxedAudio.setVisible(True)
            self.remuxedAudioOffset.setVisible(True)
            self.adjustRemuxedAudio.setEnabled(False)
            self.remuxedAudioOffset.setEnabled(False)
            # remux applies an already-designed filter -- there is nothing to design here
            self.designEnabled.setVisible(False)
            self.designerCombo.setVisible(False)
            self.queueDirLabel.setVisible(False)
            self.queueDirEdit.setVisible(False)
            self.browseQueueDirButton.setVisible(False)
        else:
            self.signalName.setText('')
            self.filterMapping.setVisible(False)
            self.filterMappingLabel.setVisible(False)
            self.includeOriginalAudio.setVisible(False)
            self.includeSubtitles.setVisible(False)
            self.gainOffset.setVisible(False)
            self.gainOffsetLabel.setVisible(False)
            self.calculateGainAdjustment.setVisible(False)
            self.adjustRemuxedAudio.setVisible(False)
            self.remuxedAudioOffset.setVisible(False)
            self.designEnabled.setVisible(self.has_designers)
            self.designerCombo.setVisible(self.has_designers)
            self.queueDirLabel.setVisible(self.has_designers)
            self.queueDirEdit.setVisible(self.has_designers)
            self.browseQueueDirButton.setVisible(self.has_designers)
        self.eacBitRate.setVisible(False)
        self.designEnabled.setChecked(False)
        self.monoMix.setChecked(self.__preferences.get(EXTRACTION_MIX_MONO))
        self.bassManage.setChecked(False)
        self.decimateAudio.setChecked(self.__preferences.get(EXTRACTION_DECIMATE))
        self.includeOriginalAudio.setChecked(self.__preferences.get(EXTRACTION_INCLUDE_ORIGINAL))
        self.includeSubtitles.setChecked(self.__preferences.get(EXTRACTION_INCLUDE_SUBTITLES))
        if self.__preferences.get(EXTRACTION_COMPRESS):
            self.audioFormat.setCurrentText(COMPRESS_FORMAT_FLAC)
        else:
            self.audioFormat.setCurrentText(COMPRESS_FORMAT_NATIVE)
        self.monoMix.setEnabled(False)
        self.bassManage.setEnabled(False)
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

    def __probe_file(self, file_name=None, display_name=None, duration_override_s=None):
        '''
        Probes the specified file using ffprobe in order to discover the audio streams.
        :param file_name: overrides the ffmpeg input (defaults to the input file field's text), used when the
        input was resolved from a BD disc rip rather than typed/dropped/picked directly.
        :param display_name: see Executor.
        :param duration_override_s: see Executor.
        '''
        if file_name is None:
            file_name = self.inputFile.text()
        self.__executor = Executor(file_name, self.targetDir.text(),
                                   mono_mix=self.monoMix.isChecked(),
                                   decimate_audio=self.decimateAudio.isChecked(),
                                   audio_format=self.audioFormat.currentText(),
                                   audio_bitrate=self.eacBitRate.value(),
                                   include_original=self.includeOriginalAudio.isChecked(),
                                   include_subtitles=self.includeSubtitles.isChecked(),
                                   signal_model=self.__signal_model if self.__is_remux else None,
                                   decimate_fs=self.__preferences.get(ANALYSIS_TARGET_FS),
                                   bm_fs=self.__preferences.get(BASS_MANAGEMENT_LPF_FS),
                                   display_name=display_name,
                                   duration_override_s=duration_override_s)
        self.__executor.progress_handler = self.__handle_ffmpeg_process
        from app import wait_cursor
        try:
            with wait_cursor(f"Probing {display_name if display_name else file_name}"):
                self.__executor.probe_file()
                self.showProbeButton.setEnabled(True)
        except FileNotFoundError as e:
            QMessageBox.critical(self, 'ffmpeg not found', str(e))
            return
        except Exception as error:
            # ffmpeg-python's Error.__str__ only says "see stdout/stderr";
            # this desktop app owns neither stream, so surface the diagnostic
            # where the person can actually act on it.
            stderr = getattr(error, 'stderr', None)
            stdout = getattr(error, 'stdout', None)
            detail = stderr or stdout
            if isinstance(detail, bytes):
                detail = detail.decode('utf-8', errors='replace')
            detail = (detail or str(error)).strip()
            QMessageBox.critical(self, 'ffprobe failed', detail)
            logger.error('ffprobe failed for %s: %s', file_name, detail)
            return
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
                if self.audioFormat.findText(COMPRESS_FORMAT_AC3) == -1:
                    self.audioFormat.addItem(COMPRESS_FORMAT_AC3)
                if self.__preferences.get(EXTRACTION_COMPRESS):
                    self.audioFormat.setCurrentText(COMPRESS_FORMAT_EAC3)
                else:
                    self.audioFormat.setCurrentText(COMPRESS_FORMAT_NATIVE)
                self.videoStreams.setCurrentIndex(1)
                self.adjustRemuxedAudio.setEnabled(True)
                self.remuxedAudioOffset.setEnabled(True)
                self.gainOffsetLabel.setEnabled(True)
                self.calculateGainAdjustment.setEnabled(True)
            self.audioStreams.setEnabled(True)
            self.videoStreams.setEnabled(True)
            self.channelCount.setEnabled(True)
            self.lfeChannelIndex.setEnabled(True)
            self.monoMix.setEnabled(True)
            self.bassManage.setEnabled(True)
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
            ac_idx = self.audioFormat.findText(COMPRESS_FORMAT_AC3)
            if ac_idx > -1:
                self.audioFormat.removeItem(ac_idx)
            if self.__preferences.get(EXTRACTION_COMPRESS):
                self.audioFormat.setCurrentText(COMPRESS_FORMAT_FLAC)
            else:
                self.audioFormat.setCurrentText(COMPRESS_FORMAT_NATIVE)
        else:
            if self.audioFormat.findText(COMPRESS_FORMAT_EAC3) == -1:
                self.audioFormat.addItem(COMPRESS_FORMAT_EAC3)
            if self.audioFormat.findText(COMPRESS_FORMAT_AC3) == -1:
                self.audioFormat.addItem(COMPRESS_FORMAT_AC3)
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
            self.__update_bitrate_for_format(self.audioFormat.currentText())
            self.__fit_options_to_selected()
            self.__display_command_info()
            self.buttonBox.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)

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
            self.bassManage.setChecked(False)
            self.bassManage.setEnabled(False)
        else:
            self.monoMix.setEnabled(True)
            # only allow bass management if we have an LFE channel
            if self.__executor.lfe_idx == 0:
                self.bassManage.setChecked(False)
                self.bassManage.setEnabled(False)
            else:
                self.bassManage.setEnabled(True)

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

    def toggle_bass_manage(self):
        '''
        Reacts to the change in bass management.
        '''
        if self.audioStreams.count() > 0 and self.__executor is not None:
            self.__executor.bass_manage = self.bassManage.isChecked()
            self.__display_command_info()

    def change_audio_format(self, audio_format):
        '''
        Reacts to the change in audio format.
        '''
        if self.audioStreams.count() > 0 and self.__executor is not None:
            self.__executor.audio_format = audio_format
            self.__update_bitrate_for_format(audio_format)
            self.__display_command_info()

    def __update_bitrate_for_format(self, audio_format):
        '''
        Presets the bitrate spinbox to the source stream's bitrate, when ffprobe reports one, otherwise falls
        back to a sane default for the selected compressed format.
        '''
        if audio_format == COMPRESS_FORMAT_EAC3 or audio_format == COMPRESS_FORMAT_AC3:
            self.eacBitRate.setVisible(True)
            default_bitrate = 1500 if audio_format == COMPRESS_FORMAT_EAC3 else 640
            source_bitrate = self.__executor.source_bit_rate_kbps if self.__executor is not None else None
            bitrate = int(source_bitrate) if source_bitrate else int(default_bitrate)
            bitrate = min(max(bitrate, self.eacBitRate.minimum()), self.eacBitRate.maximum())
            self.eacBitRate.setValue(bitrate)
            if self.__executor is not None:
                self.__executor.audio_bitrate = self.eacBitRate.value()
        else:
            self.eacBitRate.setVisible(False)

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
                self.gainOffsetLabel.setEnabled(True)
            else:
                self.__executor.include_original_audio = False
                self.__executor.original_audio_offset = 0.0
                self.gainOffset.setEnabled(False)
                self.gainOffsetLabel.setEnabled(False)
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
            msg_box.setIcon(QMessageBox.Icon.Critical)
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
            if value is None or value == 'N/A':
                return
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
        self.bassManage.setEnabled(False)
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
        self.buttonBox.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        palette = QPalette(self.ffmpegProgress.palette())
        palette.setColor(QPalette.ColorRole.Highlight, QColor(Qt.GlobalColor.green))
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
                    self.signalName.setText(Path(self.outputFilename.text()).resolve().name)
                    self.buttonBox.button(QDialogButtonBox.StandardButton.Ok).setText('Create Signals')
            else:
                logger.error(f"Extraction failed for {self.outputFilename.text()}")
                palette = QPalette(self.ffmpegProgress.palette())
                palette.setColor(QPalette.ColorRole.Highlight, QColor(Qt.GlobalColor.red))
                self.ffmpegProgress.setPalette(palette)
                self.statusBar.showMessage('Extraction failed', 5000)

            self.ffmpegOutput.setPlainText(result)
            self.buttonBox.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)
            audio = self.__preferences.get(EXTRACTION_NOTIFICATION_SOUND)
            if audio is not None:
                logger.debug(f"Playing {audio}")
                self.__sound = QSoundEffect(audio)
                self.__sound.play()
            if success and not self.__is_remux and self.designEnabled.isChecked():
                self.__design()

    def __toggle_design(self, checked):
        '''
        Enables/disables the design-related controls in lockstep with the "Design filters?" checkbox.
        '''
        self.designerCombo.setEnabled(checked)
        self.queueDirEdit.setEnabled(checked)
        self.browseQueueDirButton.setEnabled(checked)

    def __select_queue_dir(self):
        '''
        Selects the queue directory that the design result is written to, and remembers it as the
        DESIGNER_QUEUE_DIR default for next time.
        '''
        dialog = QFileDialog(parent=self)
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setWindowTitle('Select Queue Directory')
        if dialog.exec():
            selected = dialog.selectedFiles()
            if len(selected) > 0:
                self.queueDirEdit.setText(selected[0])
                self.__preferences.set(DESIGNER_QUEUE_DIR, selected[0])

    def __get_session(self):
        '''
        Lazily builds the (Qt-free) pipeline Session used for the design step.
        '''
        if self.__session is None:
            from pipeline.config import AnalysisConfig
            from pipeline.orchestrate import Session
            self.__session = Session(AnalysisConfig(target_fs=self.__preferences.get(ANALYSIS_TARGET_FS)))
        return self.__session

    def __design(self):
        '''
        Schedules a DesignJob for the just-extracted file, reusing model/batch.py's DesignJob -- same
        mono-downmix-plus-optional-per-channel-diagnostic behaviour as the batch dialog's "Design filters?"
        step (see ExtractCandidate.design()'s docstring there): if the kept extraction is already mono
        (monoMix checked), it's used directly; otherwise a second mono-only extraction is made for design
        and the kept multichannel file is additionally decomposed into per-channel arrays
        (Session.load_channels()) and sent alongside as DesignRequest.channels.
        '''
        from model.batch import DesignJob
        self.statusBar.showMessage('Designing...')
        entry_id = os.path.splitext(os.path.basename(self.__executor.get_output_path()))[0]
        already_mono_wav_path = self.__executor.get_output_path() if self.monoMix.isChecked() else None
        multichannel_wav_path = None if already_mono_wav_path is not None else self.__executor.get_output_path()
        job = DesignJob(self, self.__get_session(), entry_id, already_mono_wav_path, multichannel_wav_path,
                        self.__executor.channel_layout_name, self.__executor.file,
                        self.audioStreams.currentIndex(), self.targetDir.text(), self.designerCombo.currentText(),
                        self.queueDirEdit.text())
        QThreadPool.globalInstance().start(job)

    def design_started(self):
        ''' DesignJob callback -- design has started. '''
        self.statusBar.showMessage('Designing...')

    def design_complete(self, entry):
        '''
        DesignJob callback -- design has completed (entry.candidates is empty on a decline). Offers to open
        the Review folder window (model/worklist_review.py) on the queue directory.
        '''
        self.__design_entry = entry
        if entry.candidates:
            top = entry.candidates[0]
            self.statusBar.showMessage(f"Designed: confidence={top.confidence:.2f} method={top.method}", 5000)
        else:
            self.statusBar.showMessage(f"Design declined: {entry.decline_reason}", 5000)
        answer = QMessageBox.question(self, 'Design complete',
                                      f"Wrote a queue entry to {self.queueDirEdit.text()}.\n\n"
                                      f"Open it for review now?")
        if answer == QMessageBox.StandardButton.Yes:
            # the Review folder window (model/worklist_review.py) on the directory the entry was just written to
            from model.worklist_review import open_review_folder
            open_review_folder(self, self.__preferences, self.queueDirEdit.text())

    def design_failed(self, msg):
        ''' DesignJob callback -- design raised. '''
        self.statusBar.showMessage(f"Design failed: {msg}", 5000)
        QMessageBox.critical(self, 'Design failed', msg)

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
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setWindowTitle(f"Select Output Directory")
        if dialog.exec():
            selected = dialog.selectedFiles()
            if len(selected) > 0:
                self.targetDir.setText(selected[0])
                if self.__executor is not None:
                    self.__executor.target_dir = selected[0]
                    self.__display_command_info()

    def override_filtered_gain_adjustment(self, val):
        ''' forces the gain adjustment to a specific value. '''
        if self.__executor is not None:
            self.__executor.filtered_audio_offset = val

    def calculate_gain_adjustment(self):
        '''
        Based on the filters applied, calculates the gain adjustment that is required to avoid clipping.
        '''
        filts = list(set(self.__executor.channel_to_filter.values()))
        if len(filts) > 1 or filts[0] is not None:
            from app import wait_cursor
            with wait_cursor():
                headroom = min([min(self.__calc_headroom(x.filter_signal(filt=True, clip=False)), 0.0)
                                for x in filts if x is not None])
            self.remuxedAudioOffset.setValue(headroom)

    @staticmethod
    def __calc_headroom(filtered_signal):
        from pipeline.stats import signal_stats
        return signal_stats(filtered_signal.samples, filtered_signal.fs).headroom


class BdmvTitlePickerDialog(QDialog):
    '''
    Lets the user choose which BD playlist (title) to extract from, showing each candidate's duration and clip
    count since the automatic "longest playlist" heuristic can pick a bonus feature or the wrong angle.
    '''

    def __init__(self, parent, playlists):
        super(BdmvTitlePickerDialog, self).__init__(parent)
        self.setWindowTitle('Select BD Title')
        self.__playlists = playlists
        layout = QVBoxLayout(self)
        self.table = QTableWidget(len(playlists), 3, self)
        self.table.setHorizontalHeaderLabels(['Playlist', 'Duration', 'Clips'])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        for row, playlist in enumerate(playlists):
            self.table.setItem(row, 0, QTableWidgetItem(playlist.name))
            duration_str = str(datetime.timedelta(seconds=round(playlist.duration_s)))
            self.table.setItem(row, 1, QTableWidgetItem(duration_str))
            self.table.setItem(row, 2, QTableWidgetItem(str(len(playlist.play_items))))
        if playlists:
            self.table.selectRow(0)
        self.table.doubleClicked.connect(self.accept)
        layout.addWidget(self.table)
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
                                      parent=self)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        self.resize(480, 320)

    @property
    def selected_playlist(self):
        rows = self.table.selectionModel().selectedRows()
        return self.__playlists[rows[0].row()] if rows else None


class EditMappingDialog(QDialog, Ui_editMappingDialog):
    ''' Allows the user to override the signal to channel mapping '''

    def __init__(self, parent, channel_idx, signal_model, default_signal, selected_signal, channel_count, on_change_handler):
        super(EditMappingDialog, self).__init__(parent)
        self.setupUi(self)
        self.__signal_model = signal_model
        self.__default_signal = default_signal
        for i in range(channel_count):
            self.channels.addItem(str(i + 1))
            if i == channel_idx:
                self.channels.setCurrentRow(i)
        self.signal.addItem('Passthrough')
        self.channel_count = channel_count
        if len(signal_model) > 0:
            for idx, s in enumerate(signal_model):
                self.signal.addItem(s.name)
        elif default_signal is not None:
            self.signal.addItem(default_signal.name)
        if selected_signal is not None:
            self.signal.setCurrentText(selected_signal.name)
        self.on_change_handler = on_change_handler

    def accept(self):
        signal_name = None if self.signal.currentText() == 'Passthrough' else self.signal.currentText()
        signal = None
        if len(self.__signal_model) > 0:
            signal = next((s for s in self.__signal_model if s.name == signal_name), None)
        elif self.__default_signal is not None:
            signal = self.__default_signal if signal_name == self.__default_signal.name else None
        for c in self.channels.selectedItems():
            self.on_change_handler(int(c.text())-1, signal)
        super().accept()
