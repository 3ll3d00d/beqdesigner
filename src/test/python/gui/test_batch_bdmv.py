'''
Coverage for the batch/bulk extract dialog's BD disc rip support (model/batch.py's ExtractCandidates.append()):
a glob match that is a BDMV disc folder should be added as a single candidate, with its main feature (longest
playlist) resolved automatically -- there is no interactive title picker in batch mode.
'''
import os
import struct

import pytest
from qtpy.QtCore import QSettings

from model.batch import BatchExtractDialog, ExtractCandidates
from model.preferences import Preferences


def _make_preferences(tmp_path):
    settings = QSettings(str(tmp_path / 'settings.ini'), QSettings.Format.IniFormat)
    return Preferences(settings)


def _build_playitem_bytes(clip_id, in_time, out_time):
    body = clip_id.encode('ascii')
    body += b'M2TS'
    body += b'\x00\x00'
    body += b'\x00'
    body += struct.pack('>II', in_time, out_time)
    return struct.pack('>H', len(body)) + body


def _build_mpls_bytes(play_items, playlist_start=40):
    items_bytes = b''.join(_build_playitem_bytes(c, i, o) for c, i, o in play_items)
    playlist_length = 6 + len(items_bytes)
    playlist_block = struct.pack('>IHHH', playlist_length, 0, len(play_items), 0) + items_bytes
    header = b'MPLS0200'
    header += struct.pack('>I', playlist_start)
    header += struct.pack('>I', 0)
    header += struct.pack('>I', 0)
    header += b'\x00' * (playlist_start - len(header))
    return header + playlist_block


def _make_bdmv_disc(root, clip_ids_and_durations, playlist_items):
    stream_dir = root / 'BDMV' / 'STREAM'
    stream_dir.mkdir(parents=True)
    (root / 'BDMV' / 'PLAYLIST').mkdir(parents=True)
    (root / 'BDMV' / 'index.bdmv').write_bytes(b'INDX0100')
    for clip_id in clip_ids_and_durations:
        (stream_dir / f"{clip_id}.m2ts").write_bytes(b'\x00' * 16)
    with open(root / 'BDMV' / 'PLAYLIST' / '00800.mpls', 'wb') as f:
        f.write(_build_mpls_bytes(playlist_items))
    return str(root)


@pytest.fixture
def dialog(qtbot, tmp_path):
    d = BatchExtractDialog(None, _make_preferences(tmp_path))
    qtbot.addWidget(d)
    return d


def test_append_accepts_a_plain_file(dialog, tmp_path):
    plain_file = tmp_path / 'movie.mkv'
    plain_file.write_bytes(b'\x00')
    candidates = ExtractCandidates(dialog)

    added = candidates.append(str(plain_file))

    assert added is True
    assert len(candidates) == 1
    assert candidates[0].executor.file == str(plain_file)


def test_append_resolves_a_bdmv_folder_to_its_main_feature(dialog, tmp_path):
    disc_root = _make_bdmv_disc(tmp_path / 'disc', ['00001', '00002'], [
        ('00001', 0, 45000 * 60),
        ('00002', 0, 45000 * 40),
    ])
    candidates = ExtractCandidates(dialog)

    added = candidates.append(disc_root)

    assert added is True
    assert len(candidates) == 1
    executor_file = candidates[0].executor.file
    assert executor_file.startswith('concat:')
    assert '00001.m2ts' in executor_file and '00002.m2ts' in executor_file
    # friendly display text (used for the row's input field / logging), not the raw concat spec
    assert 'disc_00800' in candidates[0].input.text()


def test_append_skips_a_folder_that_is_not_a_bdmv_disc(dialog, tmp_path):
    plain_dir = tmp_path / 'not_a_disc'
    plain_dir.mkdir()
    candidates = ExtractCandidates(dialog)

    added = candidates.append(str(plain_dir))

    assert added is False
    assert len(candidates) == 0


def test_append_skips_a_bdmv_disc_with_no_playlists(dialog, tmp_path):
    disc_root = tmp_path / 'empty_disc'
    (disc_root / 'BDMV' / 'PLAYLIST').mkdir(parents=True)
    (disc_root / 'BDMV' / 'index.bdmv').write_bytes(b'INDX0100')
    candidates = ExtractCandidates(dialog)

    added = candidates.append(str(disc_root))

    assert added is False
    assert len(candidates) == 0
