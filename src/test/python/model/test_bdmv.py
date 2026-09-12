'''
Tests for model/bdmv.py, the BD disc rip (BDMV/PLAYLIST/*.mpls) title resolver that lets
extract/remux work against a full BD disc folder rather than a single container file.

Builds minimal synthetic .mpls files matching the field layout documented at
https://github.com/lw/BluRay/wiki/MPLS (and its linked PlayList/PlayItem pages) rather than
relying on a real disc image.
'''
import os
import struct

import pytest

from model.bdmv import is_bdmv_root, parse_mpls, list_playlists, resolve_title


def _build_playitem_bytes(clip_id, in_time, out_time):
    body = clip_id.encode('ascii')
    body += b'M2TS'
    body += b'\x00\x00'  # reserved(11) + is_multi_angle(1) + connection_condition(4)
    body += b'\x00'  # ref_to_STC_id
    body += struct.pack('>II', in_time, out_time)
    return struct.pack('>H', len(body)) + body


def _build_mpls_bytes(play_items, playlist_start=40):
    items_bytes = b''.join(_build_playitem_bytes(c, i, o) for c, i, o in play_items)
    playlist_length = 6 + len(items_bytes)
    playlist_block = struct.pack('>IHHH', playlist_length, 0, len(play_items), 0) + items_bytes
    header = b'MPLS0200'
    header += struct.pack('>I', playlist_start)  # PlayListStartAddress
    header += struct.pack('>I', 0)  # PlayListMarkStartAddress (unused by parser)
    header += struct.pack('>I', 0)  # ExtensionDataStartAddress (unused by parser)
    header += b'\x00' * (playlist_start - len(header))
    return header + playlist_block


def _write_mpls(path, play_items):
    with open(path, 'wb') as f:
        f.write(_build_mpls_bytes(play_items))


def _make_disc(tmp_path, clip_ids=('00001',)):
    root = tmp_path / 'disc'
    stream_dir = root / 'BDMV' / 'STREAM'
    stream_dir.mkdir(parents=True)
    (root / 'BDMV' / 'PLAYLIST').mkdir(parents=True)
    (root / 'BDMV' / 'index.bdmv').write_bytes(b'INDX0100')
    for clip_id in clip_ids:
        (stream_dir / f"{clip_id}.m2ts").write_bytes(b'\x00' * 16)
    return root


def test_is_bdmv_root_true_when_index_bdmv_present(tmp_path):
    root = _make_disc(tmp_path)
    assert is_bdmv_root(str(root))


def test_is_bdmv_root_false_for_plain_directory(tmp_path):
    assert not is_bdmv_root(str(tmp_path))


def test_parse_mpls_single_play_item():
    playlist_bytes = _build_mpls_bytes([('00001', 0, 45000 * 100)])
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.mpls', delete=False) as f:
        f.write(playlist_bytes)
        path = f.name
    try:
        playlist = parse_mpls(path)
        assert playlist.clip_ids == ['00001']
        assert playlist.duration_s == pytest.approx(100.0)
    finally:
        os.remove(path)


def test_parse_mpls_multiple_play_items_sums_duration():
    playlist_bytes = _build_mpls_bytes([
        ('00001', 0, 45000 * 60),
        ('00002', 45000 * 5, 45000 * 65),
    ])
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.mpls', delete=False) as f:
        f.write(playlist_bytes)
        path = f.name
    try:
        playlist = parse_mpls(path)
        assert playlist.clip_ids == ['00001', '00002']
        assert playlist.duration_s == pytest.approx(60.0 + 60.0)
    finally:
        os.remove(path)


def test_list_playlists_orders_longest_first_and_skips_empty(tmp_path):
    root = _make_disc(tmp_path, clip_ids=('00001', '00002', '00003'))
    playlist_dir = root / 'BDMV' / 'PLAYLIST'
    _write_mpls(str(playlist_dir / '00000.mpls'), [('00001', 0, 45000 * 10)])
    _write_mpls(str(playlist_dir / '00800.mpls'), [('00002', 0, 45000 * 7200)])
    _write_mpls(str(playlist_dir / '00801.mpls'), [])  # no clips -> excluded

    playlists = list_playlists(str(root))

    assert [p.name for p in playlists] == ['00800', '00000']
    assert playlists[0].duration_s == pytest.approx(7200.0)


def test_resolve_title_single_clip_returns_plain_path(tmp_path):
    root = _make_disc(tmp_path, clip_ids=('00001',))
    _write_mpls(str(root / 'BDMV' / 'PLAYLIST' / '00800.mpls'), [('00001', 0, 45000 * 100)])
    playlist = list_playlists(str(root))[0]

    resolved = resolve_title(str(root), playlist)

    expected_clip = os.path.join(str(root), 'BDMV', 'STREAM', '00001.m2ts')
    assert resolved.ffmpeg_input == expected_clip
    assert resolved.clip_paths == [expected_clip]
    assert not resolved.ffmpeg_input.startswith('concat:')
    assert resolved.display_name == 'disc_00800'


def test_resolve_title_multi_clip_uses_concat_protocol(tmp_path):
    root = _make_disc(tmp_path, clip_ids=('00001', '00002'))
    _write_mpls(str(root / 'BDMV' / 'PLAYLIST' / '00800.mpls'), [
        ('00001', 0, 45000 * 60),
        ('00002', 0, 45000 * 40),
    ])
    playlist = list_playlists(str(root))[0]

    resolved = resolve_title(str(root), playlist)

    assert resolved.ffmpeg_input.startswith('concat:')
    clip1 = os.path.join(str(root), 'BDMV', 'STREAM', '00001.m2ts').replace('\\', '/')
    clip2 = os.path.join(str(root), 'BDMV', 'STREAM', '00002.m2ts').replace('\\', '/')
    assert resolved.ffmpeg_input == f"concat:{clip1}|{clip2}"


def test_resolve_title_raises_when_clip_missing(tmp_path):
    root = _make_disc(tmp_path, clip_ids=('00001',))
    _write_mpls(str(root / 'BDMV' / 'PLAYLIST' / '00800.mpls'), [('00099', 0, 45000 * 100)])
    playlist = list_playlists(str(root))[0]

    with pytest.raises(FileNotFoundError):
        resolve_title(str(root), playlist)
