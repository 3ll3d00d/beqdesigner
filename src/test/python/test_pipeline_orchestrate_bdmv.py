'''
Coverage for pipeline.orchestrate.Session.extract()'s BD disc rip support: when src is a BDMV structure
rather than a single container file, it must resolve the main feature (or a named playlist) to a concrete
ffmpeg input -- a single clip, or a `concat:` spec for a multi-clip title -- headlessly, with no title picker
(that's ui/extract.py's BdmvTitlePickerDialog, not available/appropriate in this unattended context).
'''
import os
import struct
import subprocess
import wave

import pytest

from pipeline.config import AnalysisConfig
from pipeline.orchestrate import Session


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


def _write_mpls(path, play_items):
    with open(path, 'wb') as f:
        f.write(_build_mpls_bytes(play_items))


def _write_m2ts_clip(path, tone_hz, duration_s):
    ''' A real, ffprobe/ffmpeg-decodable BDAV-style mpegts clip -- a synthetic wav is not a valid BD clip. '''
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi',
                    '-i', f"sine=frequency={tone_hz}:duration={duration_s}",
                    '-c:a', 'mp2', '-f', 'mpegts', path, '-y'], check=True)


@pytest.fixture
def bdmv_disc(tmp_path):
    root = tmp_path / 'disc'
    stream_dir = root / 'BDMV' / 'STREAM'
    stream_dir.mkdir(parents=True)
    (root / 'BDMV' / 'PLAYLIST').mkdir(parents=True)
    (root / 'BDMV' / 'index.bdmv').write_bytes(b'INDX0100')
    _write_m2ts_clip(str(stream_dir / '00001.m2ts'), 1000, 2.0)
    _write_m2ts_clip(str(stream_dir / '00002.m2ts'), 2000, 1.0)
    _write_m2ts_clip(str(stream_dir / '00003.m2ts'), 3000, 1.0)
    # main feature: 00001+00002 (longest, 3.0s combined, spanning two clips -> exercises concat:); a shorter
    # bonus feature: 00003 alone (1.0s) at 00801, selectable via playlist_name. Each PlayItem's IN/OUT here
    # spans its whole clip since resolve_title() always uses whole clips (see its docstring).
    _write_mpls(str(root / 'BDMV' / 'PLAYLIST' / '00800.mpls'), [
        ('00001', 0, 45000 * 2),
        ('00002', 0, 45000 * 1),
    ])
    _write_mpls(str(root / 'BDMV' / 'PLAYLIST' / '00801.mpls'), [('00003', 0, 45000 * 1)])
    return str(root)


def test_extract_auto_selects_longest_playlist_and_concatenates_clips(bdmv_disc, tmp_path):
    session = Session(AnalysisConfig())
    target_dir = str(tmp_path / 'out')

    output_path = session.extract(bdmv_disc, target_dir, mono_mix=False, decimate=False)

    assert os.path.isfile(output_path)
    assert os.path.basename(output_path).startswith('disc_00800')
    with wave.open(output_path, 'rb') as w:
        duration_s = w.getnframes() / w.getframerate()
    assert duration_s == pytest.approx(3.0, abs=0.1)  # 00001 (2s) + 00002 (1s), proving concat: read both clips


def test_extract_honours_an_explicit_playlist_name(bdmv_disc, tmp_path):
    session = Session(AnalysisConfig())
    target_dir = str(tmp_path / 'out')

    output_path = session.extract(bdmv_disc, target_dir, mono_mix=False, decimate=False, playlist_name='00801')

    assert os.path.basename(output_path).startswith('disc_00801')
    with wave.open(output_path, 'rb') as w:
        duration_s = w.getnframes() / w.getframerate()
    assert duration_s == pytest.approx(1.0, abs=0.1)  # 00003 alone


def test_extract_raises_for_unknown_playlist_name(bdmv_disc, tmp_path):
    session = Session(AnalysisConfig())

    with pytest.raises(ValueError, match='No playlist named'):
        session.extract(bdmv_disc, str(tmp_path / 'out'), playlist_name='99999')


def test_extract_raises_when_disc_has_no_playlists(tmp_path):
    root = tmp_path / 'empty_disc'
    (root / 'BDMV').mkdir(parents=True)
    (root / 'BDMV' / 'PLAYLIST').mkdir()
    (root / 'BDMV' / 'index.bdmv').write_bytes(b'INDX0100')
    session = Session(AnalysisConfig())

    with pytest.raises(ValueError, match='No playable titles'):
        session.extract(str(root), str(tmp_path / 'out'))


def test_extract_still_handles_a_plain_file_unchanged(tmp_path):
    ''' Non-BD src (the pre-existing behaviour) must be untouched by the BDMV detection added to extract(). '''
    clip = tmp_path / 'plain.m2ts'
    _write_m2ts_clip(str(clip), 1000, 1.0)
    session = Session(AnalysisConfig())

    output_path = session.extract(str(clip), str(tmp_path / 'out'), mono_mix=False, decimate=False)

    assert os.path.isfile(output_path)
    assert os.path.basename(output_path).startswith('plain_')
