import datetime
import logging
import os
import time
from pathlib import Path

import ffmpeg
from ffmpeg.nodes import filter_operator, FilterNode
from qtpy import QtWidgets
from qtpy.QtWidgets import QDialog, QFileDialog, QStatusBar, QTreeWidget, QTreeWidgetItem, QDialogButtonBox

from model.preferences import EXTRACTION_OUTPUT_DIR
from ui.extract import Ui_extractAudioDialog

logger = logging.getLogger('extract')


class ExtractAudioDialog(QDialog, Ui_extractAudioDialog):
    MAIN = str(10 ** (-20.2 / 20.0))
    LFE = str(10 ** (-10.2 / 20.0))
    '''
    Allows user to load a signal, processing it if necessary.
    '''

    def __init__(self, preferences, parent=None):
        if parent is not None:
            super(ExtractAudioDialog, self).__init__(parent)
        else:
            super(ExtractAudioDialog, self).__init__()
        self.setupUi(self)
        self.statusBar = QStatusBar()
        self.gridLayout.addWidget(self.statusBar, 5, 1, 1, 1)
        self.__preferences = preferences
        self.__mono_mix_spec = ''
        self.__probe = None
        self.__audio_stream_data = []
        self.__ffmpegCommand = None
        self.__channel_layout_name = None
        defaultOutputDir = self.__preferences.get(EXTRACTION_OUTPUT_DIR)
        if os.path.isdir(defaultOutputDir):
            self.targetDir.setText(defaultOutputDir)
        self.buttonBox.button(QDialogButtonBox.Ok).setText('Extract')
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
        self.audioStreams.clear()
        self.statusBar.clearMessage()
        self.__probe = None
        self.__audio_stream_data = []
        self.showProbeButton.setEnabled(False)
        self.ffmpegCommandLine.clear()
        self.ffmpegOutput.clear()
        self.__mono_mix_spec = ''
        self.__ffmpegCommand = None
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(False)

    def __probe_file(self):
        '''
        Probes the specified file using ffprobe in order to discover the audio streams.
        '''
        logger.info(f"Probing {self.inputFile.text()}")
        start = time.time()
        from app import wait_cursor
        with wait_cursor(f"Probing {self.inputFile.text()}"):
            self.__probe = ffmpeg.probe(self.inputFile.text())
            self.showProbeButton.setEnabled(True)
        end = time.time()
        logger.info(f"Probed {self.inputFile.text()} in {round(end-start, 3)}s")
        self.__audio_stream_data = [s for s in self.__probe.get('streams', []) if s['codec_type'] == 'audio']
        if len(self.__audio_stream_data) == 0:
            self.statusBar.showMessage(f"{self.inputFile.text()} contains no audio streams!")
        else:
            for a in self.__audio_stream_data:
                self.__add_stream(a)
            self.audioStreams.setEnabled(True)

    def __add_stream(self, audio_stream):
        '''
        Adds the specified audio stream to the combo box to allow the user to choose a value.
        :param audio_stream: the stream.
        '''
        duration = self.__format_duration(audio_stream)
        if duration is None:
            duration = ''
        text = f"{audio_stream['index']}: {audio_stream['codec_long_name']} - {audio_stream['sample_rate']}Hz"
        if 'channel_layout' in audio_stream:
            text += f" {audio_stream['channel_layout']} "
        elif 'channels' in audio_stream:
            text += f" {audio_stream['channels']} channels "
        text += f"{duration}"
        self.audioStreams.addItem(text)

    def __format_duration(self, audio_stream):
        '''
        Looks for a duration field and formats it into hh:mm:ss.zzz format.
        :param audio_stream: the stream data.
        :return: the duration, if any.
        '''
        duration = None
        durationSecs = audio_stream.get('duration', None)
        if durationSecs is not None:
            duration = str(datetime.timedelta(seconds=float(durationSecs)))
        else:
            tags = audio_stream.get('tags', None)
            if tags is not None:
                duration = audio_stream.get('DURATION', None)
        if duration is not None:
            duration = ' - ' + duration
        return duration

    def updateFfmpegSpec(self):
        '''
        Creates a new ffmpeg command for the specified channel layout.
        '''
        selected_stream = self.__audio_stream_data[self.audioStreams.currentIndex()]
        channel_layout = None
        channel_layout_name = None
        mono_mix = None
        channel_count = 0
        lfe_idx = 0
        if 'channel_layout' in selected_stream:
            channel_layout = selected_stream['channel_layout']
        else:
            channel_count = selected_stream['channels']
            if channel_count == 1:
                channel_layout = 'mono'
            elif channel_count == 2:
                channel_layout = 'stereo'
            elif channel_count == 3:
                channel_layout = '2.1'
            elif channel_count == 4:
                channel_layout = '3.1'
            elif channel_count == 5:
                channel_layout = '4.1'
            elif channel_count == 6:
                channel_layout = '5.1'
            elif channel_count == 8:
                channel_layout = '7.1'
            else:
                channel_layout_name = f"{channel_count} channels"
                mono_mix = self.__get_no_lfe_mono_mix(channel_count)
        if channel_layout is not None:
            if channel_layout == 'mono':
                # TODO is this necessary?
                mono_mix = 'pan=mono|c0=c0'
                channel_count = 1
            elif channel_layout == 'stereo':
                mono_mix = self.__get_no_lfe_mono_mix(2)
                channel_count = 2
            elif channel_layout.startswith('3.0'):
                mono_mix = self.__get_no_lfe_mono_mix(3)
                channel_count = 3
            elif channel_layout == '4.0':
                mono_mix = self.__get_no_lfe_mono_mix(4)
                channel_count = 4
            elif channel_layout.startswith('quad'):
                mono_mix = self.__get_no_lfe_mono_mix(4)
                channel_count = 4
            elif channel_layout.startswith('5.0'):
                mono_mix = self.__get_no_lfe_mono_mix(5)
                channel_count = 5
            elif channel_layout.startswith('6.0'):
                mono_mix = self.__get_no_lfe_mono_mix(6)
                channel_count = 6
            elif channel_layout == 'hexagonal':
                mono_mix = self.__get_no_lfe_mono_mix(6)
                channel_count = 6
            elif channel_layout.startswith('7.0'):
                mono_mix = self.__get_no_lfe_mono_mix(7)
                channel_count = 7
            elif channel_layout == 'octagonal':
                mono_mix = self.__get_no_lfe_mono_mix(8)
                channel_count = 8
            elif channel_layout == 'downmix':
                mono_mix = self.__get_no_lfe_mono_mix(2)
                channel_count = 2
            elif channel_layout == '2.1':
                mono_mix = self.__get_lfe_mono_mix(3, 2)
                channel_count = 3
                lfe_idx = 3
            elif channel_layout == '3.1':
                mono_mix = self.__get_lfe_mono_mix(4, 3)
                channel_count = 4
                lfe_idx = 4
            elif channel_layout == '4.1':
                mono_mix = self.__get_lfe_mono_mix(5, 3)
                channel_count = 5
                lfe_idx = 4
            elif channel_layout.startswith('5.1'):
                mono_mix = self.__get_lfe_mono_mix(6, 3)
                channel_count = 6
                lfe_idx = 4
            elif channel_layout == '6.1':
                mono_mix = self.__get_lfe_mono_mix(7, 3)
                channel_count = 7
                lfe_idx = 4
            elif channel_layout == '6.1(front)':
                mono_mix = self.__get_lfe_mono_mix(7, 2)
                channel_count = 7
                lfe_idx = 4
            elif channel_layout.startswith('7.1'):
                mono_mix = self.__get_lfe_mono_mix(8, 3)
                channel_count = 8
                lfe_idx = 4
            channel_layout_name = channel_layout
        else:
            if channel_layout_name is None:
                channel_layout_name = 'unknown'
        self.__init_channel_count_fields(channel_count, lfe_index=lfe_idx)
        self.__channel_layout_name = channel_layout_name
        self.__mono_mix_spec = mono_mix
        self.updateOutputFileName()
        self.updateFfmpegCommand()

    def overrideFfmpegSpec(self, _):
        self.__channel_layout_name = 'custom'
        if self.lfeChannelIndex.value() > 0:
            self.__mono_mix_spec = self.__get_lfe_mono_mix(self.channelCount.value(), self.lfeChannelIndex.value() - 1)
        else:
            self.__mono_mix_spec = self.__get_no_lfe_mono_mix(self.channelCount.value())
        self.updateOutputFileName()
        self.updateFfmpegCommand()

    def toggleMonoMix(self):
        '''
        Reacts to the change in mono vs multichannel target.
        '''
        if self.audioStreams.count() > 0:
            self.updateOutputFileName()
            self.updateFfmpegCommand()

    def updateOutputFileName(self):
        '''
        Creates a new output file name based on the currently selected stream
        :return:
        '''
        stream_idx = str(self.audioStreams.currentIndex() + 1)
        channel_layout = self.__channel_layout_name
        output_file_name = f"{Path(self.inputFile.text()).resolve().stem}_s{stream_idx}_{channel_layout}"
        if self.monoMix.isChecked():
            output_file_name += '_to_mono'
        output_file_name += '.wav'
        self.outputFilename.setText(output_file_name)

    def updateFfmpegCommand(self):
        '''
        Creates a new ffmpeg and puts the compiled output into the text field.
        '''
        output_file = os.path.join(self.targetDir.text(), self.outputFilename.text())
        input_stream = ffmpeg.input(self.inputFile.text())
        if self.monoMix.isChecked():
            self.__ffmpegCommand = \
                input_stream \
                    .filter('pan', **{'mono|c0': self.__mono_mix_spec}) \
                    .filter('aresample', '1000', resampler='soxr') \
                    .output(output_file, acodec='pcm_s24le')
        else:
            stream_idx = f"{self.audioStreams.currentIndex()+1}"
            selected_stream = self.__audio_stream_data[self.audioStreams.currentIndex()]
            channel_count = int(selected_stream['channels'])
            if channel_count > 1:
                merge_streams = [input_stream[f"{stream_idx}:{x}"] for x in range(1, int(selected_stream['channels']))]
                filtered_stream = input_stream[f"{stream_idx}:0"].amerge(*merge_streams, inputs=channel_count)
            else:
                filtered_stream = input_stream
            self.__ffmpegCommand = \
                filtered_stream.filter('aresample', '1000', resampler='soxr').output(output_file, acodec='pcm_s24le')
        command_args = self.__ffmpegCommand.compile(overwrite_output=True)
        self.ffmpegCommandLine.setPlainText(
            ' '.join([s if s == 'ffmpeg' or s.startswith('-') else f"\"{s}\"" for s in command_args]))
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(True)

    def __get_lfe_mono_mix(self, channels, lfe_idx):
        '''
        Gets a pan filter spec that will mix to mono while respecting bass management requirements.
        :param channels: the no of channels.
        :param lfe_idx: the channel index for the LFE channel.
        :return: the spec.
        '''
        chan_gain = {lfe_idx: self.LFE}
        gains = '+'.join([f"{chan_gain.get(a, self.MAIN)}*c{a}" for a in range(0, channels)])
        return gains

    def __get_no_lfe_mono_mix(self, channels):
        '''
        Gets a pan filter spec that will mix to mono giving equal weight to each channel.
        :param channels: the no of channels.
        :return: the spec.
        '''
        ratio = 1 / channels
        gains = '+'.join([f"{ratio}*c{a}" for a in range(0, channels)])
        return gains

    def __init_channel_count_fields(self, channels, lfe_index=0):
        self.lfeChannelIndex.setMaximum(channels)
        self.lfeChannelIndex.setValue(lfe_index)
        self.channelCount.setMaximum(channels)
        self.channelCount.setValue(channels)

    def accept(self):
        '''
        Executes the ffmpeg command.
        '''
        if self.__ffmpegCommand is not None:
            from app import wait_cursor
            with wait_cursor(f"Extracting audio from {self.inputFile.text()}"):
                start = time.time()
                try:
                    out, err = self.__ffmpegCommand.run(overwrite_output=True, quiet=True)
                    end = time.time()
                    elapsed = round(end - start, 3)
                    logger.info(f"Executed ffmpeg command in {elapsed}s")
                    result = f"Command completed normally in {elapsed}s" + os.linesep + os.linesep
                    result = self.append_out_err(err, out, result)
                    self.ffmpegOutput.setPlainText(result)
                except ffmpeg.Error as e:
                    end = time.time()
                    elapsed = round(end - start, 3)
                    logger.info(f"FAILED to execute ffmpeg command in {elapsed}s")
                    result = f"Command FAILED in {elapsed}s" + os.linesep + os.linesep
                    result = self.append_out_err(e.stderr, e.stdout, result)
                    self.ffmpegOutput.setPlainText(result)

    def append_out_err(self, err, out, result):
        result += 'STDOUT' + os.linesep + '------' + os.linesep + os.linesep
        if out is not None:
            result += out.decode() + os.linesep
        result += 'STDERR' + os.linesep + '------' + os.linesep + os.linesep
        if err is not None:
            result += err.decode() + os.linesep
        return result

    def showProbeInDetail(self):
        '''
        shows a tree widget containing the contents of the probe to allow the raw probe info to be visible.
        '''
        if self.__probe is not None:
            ViewProbeDialog(self.inputFile.text(), self.__probe, parent=self).exec()

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


@filter_operator()
def join(*streams, **kwargs):
    '''
    implementation of the ffmpeg join filter for merging multiple input streams, e.g.

    i1 = ffmpeg.input(file)
    s1 = i1['1:0'].join(i1['1:1'], inputs=2, channel_layout='stereo', map='0.0-FL|1.0-FR')
    s1.output('d:/junk/test_join.wav', acodec='pcm_s24le').run()

    :param streams: the streams to join.
    :param kwargs: args to pass to join.
    :return: the resulting stream.
    '''
    return FilterNode(streams, join.__name__, kwargs=kwargs, max_inputs=None).stream()


@filter_operator()
def amerge(*streams, **kwargs):
    '''
    implementation of the ffmpeg amerge filter for merging multiple input streams, e.g.

    i1 = ffmpeg.input(file)
    s1 = i1['1:0'].amerge(i1['1:1'], inputs=6)
    s1.output('d:/junk/test_join.wav', acodec='pcm_s24le').run()

    :param streams: the streams to join.
    :param kwargs: args to pass to join.
    :return: the resulting stream.
    '''
    return FilterNode(streams, amerge.__name__, kwargs=kwargs, max_inputs=None).stream()


class ViewProbeDialog(QDialog):
    '''
    Shows the tree widget in a separate dialog.
    '''

    def __init__(self, name, probe, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle(f"ffprobe data {name}")
        self.resize(400, 600)
        self.gridLayout = QtWidgets.QGridLayout(self)
        self.gridLayout.setObjectName("gridLayout")
        self.probeTree = ViewTree(probe)
        self.gridLayout.addWidget(self.probeTree, 1, 1, 1, 1)


class ViewTree(QTreeWidget):
    '''
    Renders a dict as a tree, taken from https://stackoverflow.com/a/46096319/123054
    '''

    def __init__(self, value):
        super().__init__()

        def fill_item(item, value):
            def new_item(parent, text, val=None):
                child = QTreeWidgetItem([text])
                fill_item(child, val)
                parent.addChild(child)
                child.setExpanded(True)

            if value is None:
                return
            elif isinstance(value, dict):
                for key, val in sorted(value.items()):
                    new_item(item, str(key), val)
            elif isinstance(value, (list, tuple)):
                for val in value:
                    text = (str(val) if not isinstance(val, (dict, list, tuple))
                            else '[%s]' % type(val).__name__)
                    new_item(item, text, val)
            else:
                new_item(item, str(value))

        fill_item(self.invisibleRootItem(), value)
