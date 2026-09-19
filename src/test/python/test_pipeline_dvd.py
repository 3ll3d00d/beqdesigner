'''
DVD support in the headless pipeline (plan §11.8): Executor input options, Session.extract's DVD branch, and the
filesystem and JRiver library sources. The dvdvideo demuxer itself needs a real disc, so what is checked here is
what this code asks ffmpeg for.
'''
import pytest

from dvd_fixtures import dvd_time, write_disc
from model.ffmpeg import Executor
from pipeline.config import AnalysisConfig
from pipeline.library.filesystem import FilesystemLibrarySource
from pipeline.library.jriver import JRiverLibrarySource
from pipeline.library.pathmap import PathMapping
from pipeline.orchestrate import Session

FEATURE = dvd_time(1, 40)


def _disc(root):
    return write_disc(root, [(1, 1, 12), (1, 2, 1)], {1: ([[(1, 1)], [(2, 1)]], [FEATURE, dvd_time(0, 3)])})


AUDIO = {'format': {'duration': '6000'},
         'streams': [{'codec_type': 'audio', 'channels': 6, 'channel_layout': '5.1(side)', 'sample_rate': '48000'}]}


# --- Executor ----------------------------------------------------------------------------------------------------

def test_input_options_reach_the_probe_and_the_extract_command(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr('model.ffmpeg.ffmpeg.probe', lambda file, **kwargs: seen.update(kwargs) or AUDIO)
    executor = Executor(str(tmp_path), str(tmp_path / 'out'), display_name='Disc_t03',
                        input_options={'f': 'dvdvideo', 'title': 3})

    executor.probe_file()
    executor.update_spec(0, -1, True)

    assert seen == {'f': 'dvdvideo', 'title': 3}
    before_input = executor.ffmpeg_cli.split(' -i "')[0]  # input options only count if they precede -i
    assert '-f "dvdvideo"' in before_input and '-title "3"' in before_input


def test_a_plain_executor_adds_no_input_options(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr('model.ffmpeg.ffmpeg.probe', lambda file, **kwargs: seen.update(kwargs) or AUDIO)
    executor = Executor(str(tmp_path / 'a.mkv'), str(tmp_path / 'out'))

    executor.probe_file()
    executor.update_spec(0, -1, True)

    assert seen == {}
    assert '-f "dvdvideo"' not in executor.ffmpeg_cli


def test_an_ffmpeg_without_the_dvdvideo_demuxer_gets_a_clear_error(monkeypatch, tmp_path):
    import ffmpeg

    def refuse(file, **kwargs):
        raise ffmpeg.Error('ffprobe', b'', b"Unknown input format: 'dvdvideo'")

    monkeypatch.setattr('model.ffmpeg.ffmpeg.probe', refuse)
    executor = Executor(str(tmp_path), str(tmp_path / 'out'), display_name='x',
                        input_options={'f': 'dvdvideo', 'title': 1})

    with pytest.raises(ValueError, match='cannot read DVDs.*dvdvideo demuxer'):
        executor.probe_file()


def test_any_other_probe_failure_is_left_alone(monkeypatch, tmp_path):
    import ffmpeg

    def refuse(file, **kwargs):
        raise ffmpeg.Error('ffprobe', b'', b'Invalid data found when processing input')

    monkeypatch.setattr('model.ffmpeg.ffmpeg.probe', refuse)
    executor = Executor(str(tmp_path), str(tmp_path / 'out'), display_name='x',
                        input_options={'f': 'dvdvideo', 'title': 1})

    with pytest.raises(ffmpeg.Error):
        executor.probe_file()


# --- Session.extract ----------------------------------------------------------------------------------------------

class _Executor:
    ''' Records what Session asked for, and pretends to have extracted. '''
    instances = []

    def __init__(self, src, target_dir, **kwargs):
        self.src, self.kwargs = src, kwargs
        self.output_file_name = None
        self.channel_layout_name, self.channel_count = '5.1', 6
        _Executor.instances.append(self)

    def probe_file(self): pass
    def has_audio(self): return True
    def update_spec(self, *args): pass
    def run_sync(self): pass
    def get_output_path(self): return '/out/x.wav'


@pytest.fixture
def executor(monkeypatch):
    _Executor.instances = []
    monkeypatch.setattr('pipeline.orchestrate.Executor', _Executor)
    return _Executor


def test_extract_reads_a_dvds_longest_title_through_dvdvideo(executor, tmp_path):
    disc = _disc(tmp_path / 'Some Film')

    Session(AnalysisConfig()).extract_with_layout(disc, str(tmp_path / 'out'))

    call = executor.instances[0]
    assert call.src == disc
    assert call.kwargs['input_options'] == {'f': 'dvdvideo', 'title': 1}
    assert call.kwargs['display_name'] == 'Some Film_t01'
    assert call.kwargs['duration_override_s'] == pytest.approx(6000)


def test_extract_honours_a_named_title_and_the_video_ts_folder(executor, tmp_path):
    disc = _disc(tmp_path / 'Some Film')

    Session(AnalysisConfig()).extract_with_layout(disc + '/VIDEO_TS', str(tmp_path / 'out'), playlist_name='2')

    call = executor.instances[0]
    assert call.src == disc
    assert call.kwargs['input_options'] == {'f': 'dvdvideo', 'title': 2}


def test_extract_rejects_an_unknown_title(executor, tmp_path):
    with pytest.raises(ValueError, match='No title numbered 7'):
        Session(AnalysisConfig()).extract_with_layout(_disc(tmp_path / 'd'), str(tmp_path / 'out'), playlist_name='7')


def test_extract_leaves_a_plain_file_without_options(executor, tmp_path):
    plain = tmp_path / 'a.mkv'
    plain.write_bytes(b'x')

    Session(AnalysisConfig()).extract_with_layout(str(plain), str(tmp_path / 'out'))

    assert executor.instances[0].kwargs['input_options'] is None


# --- library sources ----------------------------------------------------------------------------------------------

def test_the_filesystem_source_yields_a_dvd_once_and_not_its_files(tmp_path):
    disc = _disc(tmp_path / 'lib' / 'Some Film')
    (tmp_path / 'lib' / 'Some Film' / 'VIDEO_TS' / 'VTS_01_1.VOB').write_bytes(b'x')
    plain = tmp_path / 'lib' / 'a.mkv'
    plain.write_bytes(b'x')

    items = list(FilesystemLibrarySource([str(tmp_path / 'lib' / '**' / '*')]).list_items())

    assert sorted(i.source_path for i in items) == sorted([disc, str(plain)])
    dvd = next(i for i in items if i.source_path == disc)
    assert dvd.display_name == 'Some Film'
    assert dvd.fingerprint  # stat of VIDEO_TS.IFO


def test_the_filesystem_source_finds_a_dvd_whatever_the_case_of_video_ts(tmp_path):
    disc = write_disc(tmp_path / 'lib' / 'Lower', [(1, 1, 1)], {1: ([[(1, 1)]], [FEATURE])}, video_ts='video_ts')

    items = list(FilesystemLibrarySource([str(tmp_path / 'lib')]).list_items())

    assert [i.source_path for i in items] == [disc]
    assert items[0].fingerprint


def test_the_filesystem_source_skips_a_folder_that_only_looks_like_a_disc(tmp_path):
    (tmp_path / 'lib' / 'Odd' / 'VIDEO_TS').mkdir(parents=True)  # no VIDEO_TS.IFO

    assert list(FilesystemLibrarySource([str(tmp_path / 'lib')]).list_items()) == []


def _source(**kwargs):
    return JRiverLibrarySource('JRiver.local', 52199, 42, **kwargs)


def _row(**overrides):
    return {'Key': 7, 'Name': 'Chapter 1', 'Filename': 'W:\\Show\\S01\\D1\\VIDEO_TS\\VIDEO_TS.dvd;1',
            'Media Sub Type': 'TV Show', **overrides}


def test_a_jriver_dvd_pseudo_file_becomes_the_translated_disc_folder(tmp_path):
    source = _source(path_mappings=[PathMapping('W:\\', str(tmp_path))])

    item = source._map_row(_row())

    assert item.source_path == str(tmp_path / 'Show' / 'S01' / 'D1')


def test_a_bluray_3d_pseudo_file_becomes_the_disc_folder_too():
    item = _source()._map_row(_row(Filename='W:\\Blade Runner 2049\\BDMV\\index.bluray3d;1'))

    assert item.source_path == 'W:\\Blade Runner 2049'
