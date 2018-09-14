import logging
import os
import socketserver
import time
from pathlib import Path

import ffmpeg
from ffmpeg.nodes import filter_operator, FilterNode
from qtpy import QtWidgets
from qtpy.QtCore import QThread
from qtpy.QtWidgets import QDialog, QTreeWidget, QTreeWidgetItem

logger = logging.getLogger('progress')

MAIN = str(10 ** (-20.2 / 20.0))
LFE = str(10 ** (-10.2 / 20.0))


class Executor:
    '''
    Deals with calculating an ffmpeg command for extracting audio from some input file.
    '''

    def __init__(self, file, target_dir):
        self.file = file
        self.__target_dir = target_dir
        self.__probe = None
        self.__audio_stream_data = []
        self.__channel_count = 0
        self.__lfe_idx = 0
        self.__channel_layout_name = None
        self.__mono_mix = True
        self.__mono_mix_spec = None
        self.__selected_stream_idx = 0
        self.__output_file_name = None
        self.__ffmpeg_cmd = None
        self.__ffmpeg_cli = None

    @property
    def target_dir(self):
        return self.__target_dir

    @target_dir.setter
    def target_dir(self, target_dir):
        self.__target_dir = target_dir
        self.__calculate_ffmpeg_cmd()

    @property
    def probe(self):
        return self.__probe

    @property
    def audio_stream_data(self):
        return self.__audio_stream_data

    @property
    def channel_count(self):
        return self.__channel_count

    @property
    def lfe_idx(self):
        return self.__lfe_idx

    @property
    def channel_layout_name(self):
        return self.__channel_layout_name

    @property
    def mono_mix(self):
        return self.__mono_mix

    @mono_mix.setter
    def mono_mix(self, mono_mix):
        self.__mono_mix = mono_mix
        self.__calculate_output()

    @property
    def mono_mix_spec(self):
        return self.__mono_mix_spec

    @property
    def selected_stream_idx(self):
        return self.__selected_stream_idx

    @property
    def output_file_name(self):
        return self.__output_file_name

    @output_file_name.setter
    def output_file_name(self, output_file_name):
        self.__output_file_name = output_file_name
        self.__calculate_ffmpeg_cmd()

    @property
    def ffmpeg_cli(self):
        return self.__ffmpeg_cli

    def probe_file(self):
        '''
        Calls ffprobe.
        :param file: the file.
        '''
        start = time.time()
        self.__probe = ffmpeg.probe(self.file)
        self.__audio_stream_data = [s for s in self.__probe.get('streams', []) if s['codec_type'] == 'audio']
        end = time.time()
        logger.info(f"Probed {self.file} in {round(end-start, 3)}s")

    def has_audio(self):
        '''
        :return: True if we have one or more audio streams in the file.
        '''
        return len(self.__audio_stream_data) > 0

    def override(self, channel_layout, channel_count, lfe_idx):
        '''
        Overrides the channel layout, count and lfe position. Overrides do not survive an update_spec call.
        :param channel_layout: the specified channel layout.
        :param channel_count: the channel count.
        :param lfe_idx: the lfe channel index.
        '''
        self.__channel_layout_name = channel_layout
        if lfe_idx > 0:
            self.__mono_mix_spec = self.__get_lfe_mono_mix(channel_count, lfe_idx - 1)
        else:
            self.__mono_mix_spec = self.__get_no_lfe_mono_mix(channel_count)
        self.__calculate_output()

    def update_spec(self, selected_stream_idx, mono_mix):
        '''
        Creates a new ffmpeg command for the specified channel layout.
        :param selected_stream_idx: the stream idx we want to extract.
        :param mono_mix: whether to mix to mono
        '''
        self.__selected_stream_idx = selected_stream_idx
        selected_stream = self.__audio_stream_data[selected_stream_idx]
        self.__mono_mix = mono_mix
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
            elif channel_layout == 'hexadecagonal':
                mono_mix = self.__get_no_lfe_mono_mix(16)
                channel_count = 16
            channel_layout_name = channel_layout
        else:
            if channel_layout_name is None:
                channel_layout_name = 'unknown'
        self.__channel_count = channel_count
        self.__lfe_idx = lfe_idx
        self.__channel_layout_name = channel_layout_name
        self.__mono_mix_spec = mono_mix
        self.__calculate_output()

    def __calculate_output(self):
        '''
        Translates the current configuration into the output commands.
        '''
        self.__calculate_output_file_name()
        self.__calculate_ffmpeg_cmd()

    def __calculate_output_file_name(self):
        '''
        Creates a new output file name based on the currently selected stream
        :return:
        '''
        stream_idx = str(self.__selected_stream_idx + 1)
        channel_layout = self.__channel_layout_name
        output_file_name = f"{Path(self.file).resolve().stem}_s{stream_idx}_{channel_layout}"
        if self.__mono_mix is True:
            output_file_name += '_to_mono'
        output_file_name += '.wav'
        self.__output_file_name = output_file_name

    def __calculate_ffmpeg_cmd(self):
        output_file = self.get_output_path()
        input_stream = ffmpeg.input(self.file)
        if self.__mono_mix:
            self.__ffmpeg_cmd = \
                input_stream \
                    .filter('pan', **{'mono|c0': self.__mono_mix_spec}) \
                    .filter('aresample', '1000', resampler='soxr') \
                    .output(output_file, acodec='pcm_s24le')
        else:
            filtered_stream = input_stream[f"{self.__selected_stream_idx+1}"]
            self.__ffmpeg_cmd = \
                filtered_stream.filter('aresample', '1000', resampler='soxr').output(output_file, acodec='pcm_s24le')
        command_args = self.__ffmpeg_cmd.compile(overwrite_output=True)
        self.__ffmpeg_cli = ' '.join([s if s == 'ffmpeg' or s.startswith('-') else f"\"{s}\"" for s in command_args])

    def get_output_path(self):
        '''
        :return: the path to the output file.
        '''
        if len(self.__target_dir) > 0:
            output_file = os.path.join(self.__target_dir, self.__output_file_name)
        else:
            output_file = self.__output_file_name
        return output_file

    def __get_lfe_mono_mix(self, channels, lfe_idx):
        '''
        Gets a pan filter spec that will mix to mono while respecting bass management requirements.
        :param channels: the no of channels.
        :param lfe_idx: the channel index for the LFE channel.
        :return: the spec.
        '''
        chan_gain = {lfe_idx: LFE}
        gains = '+'.join([f"{chan_gain.get(a, MAIN)}*c{a}" for a in range(0, channels)])
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

    def execute(self, on_progress_callback=None, on_complete_callback=None):
        '''
        Executes the command.
        :param on_progress_callback: called when the command makes progress.
        :param on_complete_callback: called when the command completes.
        '''
        if self.__ffmpeg_cmd is not None:
            # TODO update the cmd with the progress arg
            self.__extract_thread = AudioExtractor(self.__ffmpeg_cmd)

            def pass_result_on_complete():
                on_complete_callback(self.__extract_thread.result)

            if on_complete_callback:
                self.__extract_thread.finished.connect(pass_result_on_complete)
            self.__extract_thread.start()


class AudioExtractor(QThread):
    '''
    Allows audio extraction to be performed outside the main UI thread.
    '''

    def __init__(self, ffmpeg_cmd):
        QThread.__init__(self)
        self.__ffmpeg_cmd = ffmpeg_cmd
        self.result = None

    def __del__(self):
        self.wait()

    def run(self):
        start = time.time()
        try:
            logger.info("Starting ffmpeg command")
            out, err = self.__ffmpeg_cmd.run(overwrite_output=True, quiet=True)
            end = time.time()
            elapsed = round(end - start, 3)
            logger.info(f"Executed ffmpeg command in {elapsed}s")
            result = f"Command completed normally in {elapsed}s" + os.linesep + os.linesep
            self.result = self.append_out_err(err, out, result)
        except ffmpeg.Error as e:
            end = time.time()
            elapsed = round(end - start, 3)
            logger.info(f"FAILED to execute ffmpeg command in {elapsed}s")
            result = f"Command FAILED in {elapsed}s" + os.linesep + os.linesep
            self.result = self.append_out_err(e.stderr, e.stdout, result)

    def append_out_err(self, err, out, result):
        result += 'STDOUT' + os.linesep + '------' + os.linesep + os.linesep
        if out is not None:
            result += out.decode() + os.linesep
        result += 'STDERR' + os.linesep + '------' + os.linesep + os.linesep
        if err is not None:
            result += err.decode() + os.linesep
        return result


class FfmpegProgressBridge:
    '''
    A socket server which bridges progress reports from ffmpeg to the given handler.
    '''

    def __init__(self, handler, host='localhost', port=8000):
        self.__host = host
        self.__port = port
        self.__handler = handler
        self.__server = None

    def start(self):
        '''
        Starts the server and connects the handler to it.
        '''
        if self.__server is None:
            request_handler = self.__handler

            class SocketHandler(socketserver.BaseRequestHandler):
                '''
                a socket request handler which breaks the content into key-value pairs to pass to the handler.
                '''

                def handle(self):
                    data = self.request[0].strip()
                    for line in data.decode("utf-8").splitlines()[:-1]:
                        parts = line.split('=')
                        key = parts[0] if len(parts) > 0 else None
                        value = parts[1] if len(parts) > 1 else None
                        request_handler(key, value)

            self.__server = socketserver.UDPServer((self.__host, self.__port), SocketHandler)
            try:
                self.__server.serve_forever(poll_interval=0.25)
            except:
                self.__server = None

    def stop(self):
        if self.__server is not None:
            self.__server.shutdown()


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
