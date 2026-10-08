'''pipeline.library.streams: a title's audio streams in words, for choosing one (TODO W2).'''
import pytest

from pipeline.library.streams import describe_stream, describe_streams


@pytest.mark.parametrize('detail, said', [
    ({'codec': 'DTS-HD MA', 'channels': '6', 'sample_rate': '48000', 'language': 'English'},
     '1: DTS-HD MA 5.1, English, 48 kHz'),
    ({'codec': 'AC-3', 'channels': '2', 'bitrate': '192', 'sample_rate': '48000', 'language': 'French'},
     '1: AC-3 stereo, French, 48 kHz, 192 kbps'),
    ({'codec': 'TrueHD Atmos', 'channels': '8', 'title': 'Original mix'}, '1: TrueHD Atmos 7.1, "Original mix"'),
    ({'codec': 'PCM_BLURAY', 'channels': '3', 'sample_rate': '96000'}, '1: PCM_BLURAY 3 channels, 96 kHz'),
    ({'codec': 'AAC LC', 'channels': '1', 'sample_rate': '44100'}, '1: AAC LC mono, 44.1 kHz'),
    ({'codec': '', 'channels': ''}, '1: unknown codec'),
    ({'codec': 'AC-3', 'channels': 'n/a', 'bitrate': 'n/a'}, '1: AC-3'),
])
def test_a_stream_is_said_with_whatever_the_source_knows(detail, said):
    assert describe_stream(detail, 0) == said


def test_streams_are_numbered_from_one_in_the_sources_order():
    assert describe_streams([{'codec': 'A', 'channels': '2'}, {'codec': 'B', 'channels': '6'}]) == \
        ['1: A stereo', '2: B 5.1']


# --- a title whose source listed no streams: read from its file (W2) -------------------------------------------------

import os
import shutil
import subprocess

import numpy as np
import soundfile as sf

from pipeline.library.streams import details_from_ffprobe


def test_ffprobes_streams_are_described_as_a_source_describes_them():
    streams = [
        {'codec_name': 'truehd', 'profile': 'Dolby TrueHD + Dolby Atmos', 'channels': 8, 'sample_rate': '48000',
         'tags': {'language': 'eng', 'title': 'Original'}},
        {'codec_name': 'dts', 'profile': 'DTS-HD MA', 'channels': 6, 'sample_rate': '48000', 'tags': {'language': 'fre'}},
        {'codec_name': 'ac3', 'channels': 6, 'sample_rate': '48000', 'bit_rate': '640000'},
        {'codec_name': 'dts', 'profile': 'DTS', 'channels': 6, 'sample_rate': '48000', 'bit_rate': '1509000'},
        {'codec_name': 'eac3', 'profile': 'Dolby Digital Plus + Dolby Atmos', 'channels': 6, 'sample_rate': '48000'},
    ]

    details = details_from_ffprobe(streams)

    assert [d['codec'] for d in details] == ['TrueHD Atmos', 'DTS-HD MA', 'AC-3', 'DTS', 'E-AC3 Atmos']
    assert [d['audio_types'] for d in details] == [('Atmos',), ('DTS-HD MA 5.1',), ('DD 5.1',), (), ('DD+ Atmos',)]
    assert describe_streams(details) == ['1: TrueHD Atmos 7.1, "Original", eng, 48 kHz', '2: DTS-HD MA 5.1, fre, 48 kHz',
                                         '3: AC-3 5.1, 48 kHz, 640 kbps', '4: DTS 5.1, 48 kHz, 1509 kbps',
                                         '5: E-AC3 Atmos 5.1, 48 kHz']


needs_ffmpeg = pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'),
                                  reason='ffmpeg and ffprobe are optional')


def _two_stream_file(tmp_path) -> str:
    ''' Two seconds: audio stream 1 is stereo 100 Hz, audio stream 2 six channels of 40 Hz. '''
    path = str(tmp_path / 'two streams.mkv')
    subprocess.run(['ffmpeg', '-v', 'error', '-y',
                    '-f', 'lavfi', '-i', 'sine=frequency=100:sample_rate=48000:duration=2',
                    '-f', 'lavfi', '-i', 'sine=frequency=40:sample_rate=48000:duration=2',
                    '-filter_complex', '[0:a]pan=stereo|c0=c0|c1=c0[a0];[1:a]pan=5.1|c0=c0|c1=c0|c2=c0|c3=c0|c4=c0|c5=c0[a1]',
                    '-map', '[a0]', '-map', '[a1]', '-c:a', 'flac', '-metadata:s:a:1', 'language=fre', path],
                   check=True)
    return path


@needs_ffmpeg
def test_a_files_audio_streams_are_probed_as_extraction_opens_it(tmp_path):
    from pipeline.config import AnalysisConfig
    from pipeline.orchestrate import Session

    details = details_from_ffprobe(Session(AnalysisConfig()).probe_audio_streams(_two_stream_file(tmp_path)))

    assert [(d['codec'], d['channels'], d['sample_rate']) for d in details] == [('FLAC', '2', '48000'),
                                                                               ('FLAC', '6', '48000')]
    assert details[1]['language'] == 'fre'
    assert list(tmp_path.iterdir()) == [tmp_path / 'two streams.mkv']   # probing writes nothing


@needs_ffmpeg
def test_the_chosen_stream_is_extracted_and_its_mono_and_channels_share_rate_and_length(tmp_path):
    ''' AGENTS.md's extraction parity contract, for a stream other than the first: both at the analysis rate. '''
    from pipeline.library.run import LibraryRunConfig, LibraryRunReport, run_unit
    from pipeline.library.source import LibraryItem
    from pipeline.orchestrate import Session

    source = _two_stream_file(tmp_path)
    item = LibraryItem(id='two', source_path=source, display_name='Two', fingerprint='fp', audio_stream=1)
    config = LibraryRunConfig(work_dir=str(tmp_path / 'work'), queue_dir=str(tmp_path / 'queue'), designer='none',
                              keep_multichannel=True)
    report = LibraryRunReport()

    work = run_unit(Session(config.config), item, config, report, through='extract')

    assert report.failed == [] and work is not None
    mono, mono_fs = sf.read(work.wav_path)
    channels, channels_fs = sf.read(work.multichannel_wav_path)
    assert channels.shape[1] == 6                                   # stream 2, not the stereo first stream
    assert mono_fs == channels_fs == config.config.target_fs
    assert len(mono) == channels.shape[0]
    spectrum = np.abs(np.fft.rfft(mono))
    assert abs(np.argmax(spectrum) * mono_fs / len(mono) - 40) < 2   # its 40 Hz, not the first stream's 100 Hz
