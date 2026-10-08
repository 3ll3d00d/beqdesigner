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
