'''
The batch/bulk extract dialog's DVD support (model/batch.py's ExtractCandidates.append()): a glob match that is a
DVD rip (the disc folder, or its VIDEO_TS folder) becomes one candidate whose main title is resolved automatically.
'''
import pytest
from qtpy.QtCore import QSettings

from dvd_fixtures import dvd_time, write_disc
from model.batch import BatchExtractDialog, ExtractCandidates
from model.preferences import Preferences


@pytest.fixture
def dialog(qtbot, tmp_path):
    d = BatchExtractDialog(None, Preferences(QSettings(str(tmp_path / 'settings.ini'), QSettings.Format.IniFormat)))
    qtbot.addWidget(d)
    return d


def _disc(tmp_path, name='Some Film'):
    return write_disc(tmp_path / name, [(1, 1, 12), (1, 2, 1)],
                      {1: ([[(1, 1)], [(2, 1)]], [dvd_time(1, 40), dvd_time(0, 3)])})


def test_append_resolves_a_dvd_folder_to_its_main_title(dialog, tmp_path):
    candidates = ExtractCandidates(dialog)

    assert candidates.append(_disc(tmp_path)) is True

    assert len(candidates) == 1
    executor = candidates[0].executor
    assert executor.file == str(tmp_path / 'Some Film')  # read through dvdvideo, so the disc folder itself
    assert executor._Executor__duration_override_s == 6000.0
    assert 'Some Film_t01' in candidates[0].input.text()
    assert candidates[0].entry_id == 'Some Film'


def test_append_accepts_the_video_ts_folder_and_uses_the_disc_for_the_name(dialog, tmp_path):
    disc = _disc(tmp_path)
    candidates = ExtractCandidates(dialog)

    assert candidates.append(disc + '/VIDEO_TS') is True

    assert candidates[0].executor.file == disc
    assert candidates[0].entry_id == 'Some Film'  # not 'VIDEO_TS'


def test_the_executor_reads_the_title_through_the_dvdvideo_demuxer(dialog, tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr('model.ffmpeg.ffmpeg.probe', lambda file, **kwargs: captured.update(kwargs, file=file) or {
        'format': {}, 'streams': [{'codec_type': 'audio', 'channels': 6, 'channel_layout': '5.1(side)'}]})
    candidates = ExtractCandidates(dialog)
    candidates.append(_disc(tmp_path))

    candidates[0].executor.probe_file()

    assert captured['f'] == 'dvdvideo' and captured['title'] == 1


def test_append_skips_a_dvd_with_no_readable_titles(dialog, tmp_path):
    root = write_disc(tmp_path / 'empty', [(9, 1, 1)], {})
    candidates = ExtractCandidates(dialog)

    assert candidates.append(root) is False
    assert len(candidates) == 0


def test_append_still_skips_a_folder_that_is_not_a_disc(dialog, tmp_path):
    (tmp_path / 'plain').mkdir()
    candidates = ExtractCandidates(dialog)

    assert candidates.append(str(tmp_path / 'plain')) is False
