import datetime
import logging
import math
import os
import platform
import socketserver
import tempfile
import subprocess
import threading
import time
from pathlib import Path

import ffmpeg
from ffmpeg.nodes import filter_operator, FilterNode
from qtpy import QtWidgets
from qtpy.QtCore import Signal, QRunnable, QObject, QThreadPool
from qtpy.QtWidgets import QDialog, QTreeWidget, QTreeWidgetItem

from model.iir import Passthrough
from model.preferences import COMPRESS_FORMAT_NATIVE, COMPRESS_FORMAT_FLAC, COMPRESS_FORMAT_EAC3
from ui.ffmpeg import Ui_ffmpegReportDialog

logger = logging.getLogger('progress')

MAIN = str(10 ** (-20.2 / 20.0))
LFE = str(10 ** (-10.2 / 20.0))

SIGNAL_CONNECTED = 'signal_connected'
SIGNAL_ERROR = 'signal_error'
SIGNAL_COMPLETE = 'signal_complete'
SIGNAL_CANCELLED = 'signal_cancel'
NEXT_PORT = 12000

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
    6: CHANNEL_LAYOUTS['5.1(side)'],
    8: CHANNEL_LAYOUTS['7.1']
}


def format_biquad(f):
    return f.format_biquads(False, separator=':')


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
        prefix = f"{text}_" if text is not None else ''
        if channel_layout_name == 'unknown' and channel_count in UNKNOWN_CHANNEL_LAYOUTS:
            return f"{prefix}{UNKNOWN_CHANNEL_LAYOUTS[channel_count][channel]}"
        elif channel_layout_name in CHANNEL_LAYOUTS:
            return f"{prefix}{CHANNEL_LAYOUTS[channel_layout_name][channel]}"
        else:
            return f"{prefix}c{channel + 1}"


def get_next_port():
    # v dodgy hack! wot no AtomicInt
    global NEXT_PORT
    NEXT_PORT += 1
    return NEXT_PORT


class Executor:
    '''
    Deals with calculating an ffmpeg command for extracting audio from some input file.
    '''

    def __init__(self, file, target_dir, mono_mix=True, decimate_audio=True, audio_format=COMPRESS_FORMAT_NATIVE,
                 audio_bitrate=1500, include_original=False, include_subtitles=False, signal_model=None,
                 decimate_fs=1000):
        self.file = file
        self.__target_dir = target_dir
        self.__probe = None
        self.__audio_stream_data = []
        self.__video_stream_data = []
        self.__channel_to_filter = {}
        self.__channel_count = 0
        self.__lfe_idx = 0
        self.__channel_layout_name = None
        self.__mono_mix = mono_mix
        self.__decimate_audio = decimate_audio
        self.__audio_format = audio_format
        self.__audio_bitrate = audio_bitrate
        self.__include_original_audio = include_original
        self.__original_audio_offset = 0.0
        self.__filtered_audio_offset = 0.0
        self.__include_subtitles = include_subtitles
        self.__mono_mix_spec = None
        self.__selected_audio_stream_idx = -1
        self.__selected_video_stream_idx = -1
        self.__output_file_name = None
        self.__output_file_name_overridden = False
        self.__ffmpeg_cmd = None
        self.__ffmpeg_cli = None
        self.__filter_complex_script_content = None
        self.__progress_handler = None
        self.__progress_port = get_next_port()
        self.__extractor = None
        self.__cancel = False
        self.__sample_rate = None
        self.__filter_complex_filter = None
        self.__is_remux = signal_model is not None
        self.__signal_model = signal_model
        self.__decimate_fs = int(decimate_fs)
        self.__start_time_ms = 0
        self.__end_time_ms = 0

    @property
    def filter_complex_script_content(self):
        return self.__filter_complex_script_content

    @property
    def start_time_ms(self):
        return self.__start_time_ms

    @start_time_ms.setter
    def start_time_ms(self, start_time_ms):
        self.__start_time_ms = start_time_ms
        self.__calculate_output()

    @property
    def end_time_ms(self):
        return self.__end_time_ms

    @end_time_ms.setter
    def end_time_ms(self, end_time_ms):
        self.__end_time_ms = end_time_ms
        self.__calculate_output()

    @property
    def channel_to_filter(self):
        return self.__channel_to_filter

    @property
    def target_dir(self):
        return self.__target_dir

    @target_dir.setter
    def target_dir(self, target_dir):
        self.__target_dir = target_dir
        self.__calculate_ffmpeg_cmd()

    @property
    def progress_handler(self):
        return self.__progress_handler

    @progress_handler.setter
    def progress_handler(self, progress_handler):
        '''
        Sets the progress handler, should be a callable which accepts 2 parameters (key, value)
        :param progress_handler: the handler.
        '''
        self.__progress_handler = progress_handler

    @property
    def probe(self):
        return self.__probe

    @property
    def audio_stream_data(self):
        return self.__audio_stream_data

    @property
    def video_stream_data(self):
        return self.__video_stream_data

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
    def decimate_audio(self):
        return self.__decimate_audio

    @decimate_audio.setter
    def decimate_audio(self, decimate_audio):
        self.__decimate_audio = decimate_audio
        self.__calculate_output()

    @property
    def audio_format(self):
        return self.__audio_format

    @audio_format.setter
    def audio_format(self, audio_format):
        self.__audio_format = audio_format
        self.__calculate_output()

    @property
    def audio_bitrate(self):
        return self.__audio_bitrate

    @audio_bitrate.setter
    def audio_bitrate(self, audio_bitrate):
        self.__audio_bitrate = audio_bitrate
        self.__calculate_output()

    @property
    def include_original_audio(self):
        return self.__include_original_audio

    @include_original_audio.setter
    def include_original_audio(self, include_original_audio):
        self.__include_original_audio = include_original_audio
        self.__calculate_output()

    @property
    def original_audio_offset(self):
        return self.__original_audio_offset

    @original_audio_offset.setter
    def original_audio_offset(self, original_audio_offset):
        self.__original_audio_offset = original_audio_offset
        self.__calculate_output()

    @property
    def filtered_audio_offset(self):
        return self.__filtered_audio_offset

    @filtered_audio_offset.setter
    def filtered_audio_offset(self, filtered_audio_offset):
        self.__filtered_audio_offset = filtered_audio_offset
        self.__calculate_output()

    @property
    def include_subtitles(self):
        return self.__include_subtitles

    @include_subtitles.setter
    def include_subtitles(self, include_subtitles):
        self.__include_subtitles = include_subtitles
        self.__calculate_output()

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
        return self.__selected_audio_stream_idx

    @property
    def output_file_name(self):
        return self.__output_file_name

    @output_file_name.setter
    def output_file_name(self, output_file_name):
        self.__output_file_name = output_file_name
        self.__output_file_name_overridden = True
        self.__calculate_ffmpeg_cmd()

    @property
    def ffmpeg_cli(self):
        return self.__ffmpeg_cli

    def map_filter_to_channel(self, channel_idx, signal):
        ''' updates the mapping of the given signal to the specified channel idx '''
        self.__channel_to_filter[channel_idx] = signal
        self.__calculate_ffmpeg_cmd()

    def probe_file(self):
        '''
        Calls ffprobe.
        '''
        logger.info(f"Probing {self.file}")
        start = time.time()
        self.__probe = ffmpeg.probe(self.file)
        self.__audio_stream_data = [s for s in self.__probe.get('streams', []) if s['codec_type'] == 'audio']
        self.__video_stream_data = [s for s in self.__probe.get('streams', []) if s['codec_type'] == 'video']
        end = time.time()
        logger.info(f"Probed {self.file} in {round(end - start, 3)}s")

    def has_audio(self):
        '''
        :return: True if we have one or more audio streams in the file.
        '''
        return len(self.__audio_stream_data) > 0

    def has_video(self):
        '''
        :return: True if we have one or more video streams in the file.
        '''
        return len(self.__video_stream_data) > 0

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

    def update_spec(self, audio_stream_idx, video_stream_idx, mono_mix):
        '''
        Creates a new ffmpeg command for the specified channel layout.
        :param audio_stream_idx: the stream idx we want to extract.
        :param video_stream_idx: the video idx we want to extract, if -1 no video is extracted.
        :param mono_mix: whether to mix to mono
        '''
        recalc_signal_mapping = self.__selected_audio_stream_idx != audio_stream_idx and self.__is_remux is True
        self.__selected_audio_stream_idx = audio_stream_idx
        self.__selected_video_stream_idx = video_stream_idx
        selected_audio_stream = self.__audio_stream_data[audio_stream_idx]
        self.__mono_mix = mono_mix
        self.__sample_rate = selected_audio_stream['sample_rate'] if 'sample_rate' in selected_audio_stream else 48000
        channel_layout = None
        channel_layout_name = None
        mono_mix = None
        channel_count = 0
        lfe_idx = 0
        if 'channel_layout' in selected_audio_stream:
            channel_layout = selected_audio_stream['channel_layout']
        else:
            channel_count = selected_audio_stream['channels']
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
        if recalc_signal_mapping:
            self.__channel_to_filter = {}
            for channel_idx in range(0, self.channel_count):
                c_name = get_channel_name(None, channel_idx, self.channel_count, self.channel_layout_name)
                sig = next((x for x in self.__signal_model if x.name.endswith(f"_{c_name}")), None)
                self.__channel_to_filter[channel_idx] = sig
        self.__calculate_output()

    def __calculate_output(self):
        '''
        Translates the current configuration into the output commands.
        '''
        start = time.time()
        self.__calculate_output_file_name()
        self.__calculate_ffmpeg_cmd()
        end = time.time()
        logger.debug(f"Recalculated output in {round(end - start, 3)}s")

    def __calculate_output_file_name(self):
        '''
        Creates a new output file name based on the currently selected stream
        '''
        if self.__output_file_name_overridden is False:
            stream_idx = str(self.__selected_audio_stream_idx + 1)
            channel_layout = self.__channel_layout_name
            output_file_name = f"{Path(self.file).resolve().stem}_s{stream_idx}_{channel_layout}"
            if self.__mono_mix is True:
                output_file_name += '_to_mono'
            self.__output_file_name = output_file_name

    def __write_filter_complex(self):
        filts = []
        ch_outs = []
        audio_input = f"[a:{self.selected_stream_idx}]"
        for channel_idx in range(0, self.channel_count):
            c_name = get_channel_name(None, channel_idx, self.channel_count, self.channel_layout_name)
            sig = self.__channel_to_filter[channel_idx]
            f_idx = 0
            # extract a single channel, convert it to dbl sample format
            # pipe it through each biquad in the matching signal
            # convert back to 32bit into the original fs and place in a named mono stream
            #   [0:a]pan=1c|c0=c<channel_num>[<c_name>];[L]aformat=sample_fmts=dbl[Lin];[Lin]biquad=b0=1.00080054343984:b1=-1.99150225042309:b2=0.990752403334702:a0=1.0:a1=-1.99150965346467:a2=0.991545543732965[L1];[L1]biquad=b0=0.999200096917323:b1=-1.98991663875369:b2=0.990752403395918:a0=1.0:a1=-1.98990924163382:a2=0.989959897433105[L2];[L2]aformat=sample_fmts=s32:sample_rates=48000:channel_layouts=mono[Lout];
            filt = f"{audio_input}pan=1c|c0=c{channel_idx}[{c_name}];"
            filt += f"[{c_name}]aformat=sample_fmts=dbl[{c_name}_{f_idx}];"
            if sig is not None:
                sig_filter = sig.active_filter.resample(int(self.__sample_rate), copy_listener=False)
                for f in sig_filter.filters:
                    for bq in format_biquad(f):
                        filt += f"[{c_name}_{f_idx}]biquad={bq}[{c_name}_{f_idx + 1}];"
                        f_idx += 1
            else:
                filt += f"[{c_name}_{f_idx}]biquad={format_biquad(Passthrough())[0]}[{c_name}_{f_idx + 1}];"
                f_idx += 1

            if not math.isclose(self.filtered_audio_offset, 0.0):
                filt += f"[{c_name}_{f_idx}]volume=volume={self.filtered_audio_offset:+g}dB[{c_name}_{f_idx + 1}];"
                f_idx += 1

            filt += f"[{c_name}_{f_idx}]aformat=sample_fmts=s32:sample_rates={self.__sample_rate}"
            filt += f":channel_layouts=mono[{c_name}out]"
            filts.append(filt)
            ch_outs.append(f"[{c_name}out]")
        if self.__mono_mix:
            # merge the filtered mono streams down into a single mono mix at the usual sample rate
            merge_filt = "".join(ch_outs)
            merge_filt += f"amerge=inputs={len(ch_outs)},pan=mono|c0={self.__mono_mix_spec}"
        else:
            # merge the filtered mono streams back into a single multichannel stream
            # [Lout][Rin]amerge=inputs=2,pan=stereo|c0=c0|c1=c1[Wout]
            merge_filt = "".join(ch_outs)
            merge_filt += f"amerge=inputs={len(ch_outs)},pan={len(ch_outs)}c"
            merge_filt += ''.join([f"|c{idx}=c{idx}" for idx, x in enumerate(ch_outs)])

        if self.decimate_audio is True:
            merge_filt += f"[m0];[m0]aresample={self.__decimate_fs}:resampler=soxr[s0]"
        else:
            merge_filt += '[s0]'

        new_filt = f"{';'.join(filts)};{merge_filt}"

        if self.include_original_audio and not math.isclose(self.original_audio_offset, 0.0):
            new_filt += f";{audio_input}volume=volume={self.original_audio_offset:+g}dB[s1]"

        if self.__filter_complex_script_content is None or self.__filter_complex_script_content != new_filt:
            self.__filter_complex_script_content = new_filt

            if self.__filter_complex_filter is not None:
                os.remove(self.__filter_complex_filter.name)
            self.__filter_complex_filter = tempfile.NamedTemporaryFile(delete=False, suffix='.filt', mode='w')

            print(self.__filter_complex_script_content, file=self.__filter_complex_filter)
            self.__filter_complex_filter.close()
        return self.__filter_complex_filter.name.replace('\\', '/')

    def __calculate_ffmpeg_cmd(self):
        output_file = self.get_output_path()
        if self.__is_remux:
            self.__calculate_remux_cmd(output_file)
        else:
            self.__calculate_extract_command(output_file)

    def __calculate_input_kwargs(self):
        args = {'drc_scale': 0}
        if self.start_time_ms > 0:
            args['ss'] = round(self.start_time_ms / 1000, 3)
        if self.end_time_ms > 0:
            args['t'] = round((self.end_time_ms - self.start_time_ms) / 1000, 3)
        return args

    def __calculate_extract_command(self, output_file):
        ''' calculates the command required to extract the audio to the specified output file '''
        input_stream = ffmpeg.input(self.file, **self.__calculate_input_kwargs())
        acodec = self.__get_acodec()
        if self.__mono_mix:
            self.__ffmpeg_cmd = self.__calculate_extract_mono_cmd(acodec, input_stream, output_file)
        else:
            self.__ffmpeg_cmd = self.__calculate_extract_multi_cmd(acodec, input_stream, output_file)
        if self.progress_handler is not None:
            self.__ffmpeg_cmd = self.__ffmpeg_cmd.global_args('-progress',
                                                              f"udp://127.0.0.1:{self.__progress_port}")
        command_args = self.__ffmpeg_cmd.compile(overwrite_output=True)
        self.__ffmpeg_cli = ' '.join(
            [s if s == 'ffmpeg' or s.startswith('-') else f"\"{s}\"" for s in command_args])

    def __get_acodec(self):
        if self.audio_format == COMPRESS_FORMAT_NATIVE:
            return 'pcm_s24le'
        elif self.audio_format == COMPRESS_FORMAT_FLAC:
            return 'flac'
        elif self.audio_format == COMPRESS_FORMAT_EAC3:
            return 'eac3'
        else:
            raise ValueError(f"Unknown audio format {self.audio_format}")

    def __get_audio_ext(self):
        if self.audio_format == COMPRESS_FORMAT_NATIVE:
            return 'wav'
        elif self.audio_format == COMPRESS_FORMAT_FLAC:
            return 'flac'
        elif self.audio_format == COMPRESS_FORMAT_EAC3:
            return 'mkv'
        else:
            raise ValueError(f"Unknown audio format {self.audio_format}")

    def __calculate_extract_multi_cmd(self, acodec, input_stream, output_file):
        ''' Calculates an ffmpeg-python cmd for extracting multichannel audio from an input stream. '''
        audio_filter = input_stream[f"a:{self.__selected_audio_stream_idx}"]
        if self.decimate_audio is True:
            audio_filter = audio_filter.filter('aresample', str(self.__decimate_fs), resampler='soxr')
        if self.__selected_video_stream_idx != -1:
            return ffmpeg.output(input_stream[f"v:{self.__selected_video_stream_idx}"], audio_filter, output_file,
                                 acodec=acodec, vcodec='copy')
        else:
            return audio_filter.output(output_file, acodec=acodec)

    def __calculate_extract_mono_cmd(self, acodec, input_stream, output_file):
        ''' Calculates an ffmpeg-python cmd for extracting mono (bass managed) audio from an input stream. '''
        if self.__selected_video_stream_idx != -1:
            audio_filter = input_stream[f"a:{self.__selected_audio_stream_idx}"].filter('pan', **{'mono|c0': self.__mono_mix_spec})
            if self.decimate_audio is True:
                audio_filter = audio_filter.filter('aresample', str(self.__decimate_fs), resampler='soxr')
            return ffmpeg.output(input_stream[f"v:{self.__selected_video_stream_idx}"], audio_filter, output_file,
                                 acodec=acodec, vcodec='copy')
        else:
            mix = input_stream[f"a:{self.__selected_audio_stream_idx}"].filter('pan', **{'mono|c0': self.__mono_mix_spec})
            if self.decimate_audio is True:
                mix = mix.filter('aresample', str(self.__decimate_fs), resampler='soxr')
            return mix.output(output_file, acodec=acodec)

    def __calculate_remux_cmd(self, output_file):
        ''' calculates the command filter the audio and remux it back into the output file '''
        filename = self.file.replace('\\', '/')
        # TODO only write the filter file when it actually changes
        exe = ".exe" if platform.system() == 'Windows' else ""
        self.__ffmpeg_cmd = [f"ffmpeg{exe}"]
        for key, value in self.__calculate_input_kwargs().items():
            self.__ffmpeg_cmd += [f"-{key}", str(value)]
        self.__ffmpeg_cmd += [
            '-i', filename,
            '-filter_complex_script', self.__write_filter_complex()
        ]
        if self.__selected_video_stream_idx != -1:
            self.__ffmpeg_cmd += ['-map', f"0:v:{self.__selected_video_stream_idx}" , '-c:v', 'copy']
        if self.include_subtitles:
            self.__ffmpeg_cmd += ['-map', '0:s', '-c:s', 'copy']
        acodec = self.__get_acodec()
        if self.include_original_audio:
            if math.isclose(self.original_audio_offset, 0.0):
                self.__ffmpeg_cmd += [
                    '-map',
                    f"0:a:{self.__selected_audio_stream_idx}",
                    f"-c:a:{self.__selected_audio_stream_idx}",
                    'copy'
                ]
            else:
                self.__ffmpeg_cmd += ['-map', '[s1]']
        self.__ffmpeg_cmd += ['-map', '[s0]', '-acodec',  acodec]
        if self.audio_format == COMPRESS_FORMAT_EAC3:
            self.__ffmpeg_cmd += ['-ab', f"{self.audio_bitrate}k"]
        self.__ffmpeg_cmd += [output_file]
        if self.progress_handler is not None:
            self.__ffmpeg_cmd += ['-progress', f"udp://127.0.0.1:{self.__progress_port}"]
        self.__ffmpeg_cmd += ['-y']
        self.__ffmpeg_cli = ' '.join(self.__ffmpeg_cmd)

    def get_output_path(self):
        '''
        :return: the path to the output file.
        '''
        if len(self.__target_dir) > 0:
            output_file = os.path.join(self.__target_dir, self.__output_file_name)
        else:
            output_file = self.__output_file_name
        if self.__selected_video_stream_idx != -1:
            return f"{output_file}{os.path.splitext(self.file)[1]}"
        else:
            return f"{output_file}.{self.__get_audio_ext()}"

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

    def execute(self):
        '''
        Executes the command.
        '''
        if self.__ffmpeg_cmd is not None:
            self.__extractor = AudioExtractor(self.__ffmpeg_cmd, port=self.__progress_port, cancel=self.__cancel,
                                              progress_handler=self.progress_handler, is_remux=self.__is_remux)
            QThreadPool.globalInstance().start(self.__extractor)

    def cancel(self):
        '''
        Attempts to cancel the extractor.
        '''
        self.__cancel = True
        if self.__extractor is not None:
            self.__extractor.cancel()

    def enable(self):
        '''
        Revokes a previously issued cancel.
        '''
        self.__cancel = False
        if self.__extractor is not None:
            self.__extractor.enable()


class JobSignals(QObject):
    on_progress = Signal(str, str, name='on_progress')


class AudioExtractor(QRunnable):
    '''
    Allows audio extraction to be performed outside the main UI thread.
    '''

    def __init__(self, ffmpeg_cmd, port=None, progress_handler=None, cancel=False, is_remux=False):
        super().__init__()
        self.__ffmpeg_cmd = ffmpeg_cmd
        self.__is_remux = is_remux
        self.__progress_handler = progress_handler
        self.__signals = JobSignals()
        if self.__progress_handler is not None:
            self.__signals.on_progress.connect(self.__progress_handler)
            self.__signals.on_progress.emit(SIGNAL_CONNECTED, '')
        self.__socket_server = None
        self.__port = port
        self.__cancel = cancel

    def __del__(self):
        self.__stop_socket_server()

    def cancel(self):
        '''
        Attempts to cancel the job. Currently only has any effect if it hasn't already started.
        '''
        self.__cancel = True

    def enable(self):
        '''
        Enables the job. Currently only has any effect if it hasn't already started.
        '''
        self.__cancel = False

    def run(self):
        '''
        Executes the ffmpeg command.
        '''
        if self.__cancel is True:
            self.__signals.on_progress.emit(SIGNAL_CANCELLED, 'Cancelled')
        else:
            self.__start_socket_server()
            start = time.time()
            try:
                logger.info("Starting ffmpeg command")
                if self.__is_remux:
                    out, err = self.__execute_ffmpeg()
                else:
                    out, err = self.__ffmpeg_cmd.run(overwrite_output=True, quiet=True)
                end = time.time()
                elapsed = round(end - start, 3)
                logger.info(f"Executed ffmpeg command in {elapsed}s")
                result = f"Command completed normally in {elapsed}s" + os.linesep + os.linesep
                result = self.__append_out_err(err, out, result)
                self.__signals.on_progress.emit(SIGNAL_COMPLETE, result)
            except ffmpeg.Error as e:
                end = time.time()
                elapsed = round(end - start, 3)
                logger.info(f"FAILED to execute ffmpeg command in {elapsed}s")
                result = f"Command FAILED in {elapsed}s" + os.linesep + os.linesep
                result = self.__append_out_err(e.stderr, e.stdout, result)
                self.__signals.on_progress.emit(SIGNAL_ERROR, result)
            finally:
                self.__stop_socket_server()

    def __execute_ffmpeg(self):
        p = subprocess.Popen(self.__ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        retcode = p.poll()
        if retcode:
            raise ffmpeg.Error('ffmpeg', out, err)
        return out, err

    def __append_out_err(self, err, out, result):
        result += 'STDOUT' + os.linesep + '------' + os.linesep + os.linesep
        if out is not None:
            result += out.decode() + os.linesep
        result += 'STDERR' + os.linesep + '------' + os.linesep + os.linesep
        if err is not None:
            result += err.decode() + os.linesep
        return result

    def __start_socket_server(self):
        if self.__progress_handler is not None:
            self.__socket_server = FfmpegProgressBridge(self.handle_progress_event, port=self.__port, auto=True)

    def __stop_socket_server(self):
        if self.__socket_server is not None:
            self.__socket_server.stop()

    def handle_progress_event(self, key, value):
        '''
        Callback from ffmpeg -progress, passes the value straight to the __progress_handler
        :param key: the key.
        :param value: the value.
        '''
        logger.debug(f"Received -- 127.0.0.1:{self.__port} -- {key}={value}")
        self.__signals.on_progress.emit(key, value)


class ThreadedUDPServer(socketserver.ThreadingMixIn, socketserver.UDPServer):
    pass


class FfmpegProgressBridge:
    '''
    A socket server which bridges progress reports from ffmpeg to the given handler.
    '''

    def __init__(self, handler, host='127.0.0.1', port=8000, auto=False):
        self.__host = host
        self.__port = port
        self.__handler = handler
        self.__server = None
        if auto is True:
            self.start()

    def start(self):
        '''
        Starts the server and connects the handler to it.
        '''
        if self.__server is None:
            logger.info(f"Starting progress bridge on {self.__host}:{self.__port}")
            request_handler = self.__handler

            class SocketHandler(socketserver.BaseRequestHandler):
                '''
                a socket request handler which breaks the content into key-value pairs to pass to the handler.
                '''

                def handle(self):
                    data = self.request[0].strip()
                    decode = data.decode("utf-8")
                    for line in decode.splitlines():
                        parts = line.split('=')
                        key = parts[0] if len(parts) > 0 else None
                        value = parts[1] if len(parts) > 1 else None
                        request_handler(key, value)

            self.__server = socketserver.UDPServer((self.__host, self.__port), SocketHandler)

            def serve_forever():
                logger.info('Server loop running')
                self.__server.serve_forever(poll_interval=0.25)

            server_thread = threading.Thread(target=serve_forever)
            server_thread.daemon = True
            server_thread.start()
            # give the thread a bit of time to start before we kick off
            time.sleep(0.25)

    def stop(self):
        '''
        Stops the server if it is running.
        '''
        if self.__server is not None:
            logger.info(f"Stopping progress bridge on {self.__host}:{self.__port}")
            self.__server.shutdown()
            logger.info(f"Stopped progress bridge on {self.__host}:{self.__port}")


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


def parse_video_stream(probe, video_stream):
    '''
    Parses key details from the specified audio stream.
    :param probe: the probe data.
    :param video_stream: the stream data.
    :return: friendly description of the stream
    '''
    desc = f"{video_stream['index']}: {video_stream['codec_name']}"
    if 'height' in video_stream and 'width' in video_stream:
        desc += f" ({video_stream['width']}x{video_stream['height']})"
    return desc


def parse_audio_stream(probe, audio_stream):
    '''
    Parses key details from the specified audio stream.
    :param probe: the probe data.
    :param audio_stream: the stream data.
    :return: friendly description of the stream, duration_micros
    '''
    duration, duration_ms = get_duration(probe, audio_stream)
    if duration is None:
        duration = ''
    if 'codec_name' in audio_stream and audio_stream['codec_name'] == 'dts' and 'profile' in audio_stream:
        codec = audio_stream['profile']
    else:
        codec = audio_stream['codec_long_name']
    if 'tags' in audio_stream and 'language' in audio_stream['tags']:
        lang = f"{audio_stream['tags']['language']} - "
    else:
        lang = ''
    text = f"{audio_stream['index']}: {codec} - {lang}{audio_stream['sample_rate']}Hz"
    if 'channel_layout' in audio_stream:
        text += f" {audio_stream['channel_layout']} "
    elif 'channels' in audio_stream:
        text += f" {audio_stream['channels']} channels "
    text += f"{duration}"
    return text, duration_ms


def get_duration(probe, audio_stream):
    '''
    Looks for a duration field and formats it into hh:mm:ss.zzz format.
    :param probe: the probe data.
    :param audio_stream: the stream data.
    :return: the duration, if any.
    '''
    # default to an hour is a complete hack but I have no idea what turn up in that DURATION tag
    duration_micros = 60 * 60 * 1000
    duration = None
    duration_secs = None
    if 'format' in probe:
        if 'duration' in probe['format']:
            duration_secs = probe['format']['duration']
    if duration_secs is None:
        duration_secs = audio_stream.get('duration', None)
    if duration_secs is not None:
        duration = str(datetime.timedelta(seconds=float(duration_secs)))
        duration_micros = float(duration_secs) * 1000000
    else:
        tags = audio_stream.get('tags', None)
        if tags is not None:
            duration = tags.get('DURATION', None)
    if duration is not None:
        tmp = extract_duration_micros(duration)
        if tmp is not None:
            duration_micros = tmp
        duration = ' - ' + duration
    return duration, duration_micros


def extract_duration_micros(duration):
    '''
    Attempts to convert a duration string of unknown format into a millisecond value.
    :param duration: the duration str.
    :return: duration in millis if we could convert.
    '''
    duration_millis = None
    try:
        subsecond_pos = duration.rfind('.')
        duration_str = duration
        if subsecond_pos > 0:
            subsecond_len = len(duration) - subsecond_pos
            if subsecond_len > 6:
                duration_str = duration[:6 - subsecond_len]
            elif subsecond_len == -1:
                duration_str += '.000000'
            elif subsecond_len < 6:
                duration_str += ('0' * subsecond_len)
            x = datetime.datetime.strptime(duration_str, '%H:%M:%S.%f')
            duration_millis = datetime.timedelta(hours=x.hour, minutes=x.minute, seconds=x.second,
                                                 microseconds=x.microsecond).total_seconds() * 1000000
    except Exception as e:
        logger.exception(f"Unable to extract duration_millis from {duration}", e)
    return duration_millis


class FFMpegDetailsDialog(QDialog, Ui_ffmpegReportDialog):
    def __init__(self, name, parent):
        super().__init__(parent=parent)
        self.setupUi(self)
        self.setWindowTitle(f"ffmpeg: {name}")
