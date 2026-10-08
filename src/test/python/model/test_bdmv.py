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

from model.bdmv import bdmv_root_of, is_bdmv_root, parse_mpls, list_playlists, resolve_title


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


def test_bdmv_root_of_a_disc_root_is_itself(tmp_path):
    root = _make_disc(tmp_path)
    assert bdmv_root_of(str(root)) == str(root)


def test_bdmv_root_of_the_index_bdmv_file_is_the_disc_root(tmp_path):
    root = _make_disc(tmp_path)
    assert bdmv_root_of(str(root / 'BDMV' / 'index.bdmv')) == str(root)


def test_bdmv_root_of_anything_else_is_none(tmp_path):
    root = _make_disc(tmp_path)
    assert bdmv_root_of(str(tmp_path)) is None
    assert bdmv_root_of(str(root / 'BDMV' / 'STREAM' / '00001.m2ts')) is None
    assert bdmv_root_of(str(tmp_path / 'missing' / 'BDMV' / 'index.bdmv')) is None


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


def test_repeated_clip_run_is_collapsed_for_extraction_and_title_selection(tmp_path):
    root = _make_disc(tmp_path, clip_ids=('00001', '00002', '00173'))
    playlist_dir = root / 'BDMV' / 'PLAYLIST'
    # A decoy playlist can make the same stream look much longer than the feature by repeating it.
    _write_mpls(str(playlist_dir / '00001.mpls'), [('00173', 0, 45000 * 60)] * 200)
    _write_mpls(str(playlist_dir / '00800.mpls'), [
        ('00001', 0, 45000 * 90),
        ('00002', 0, 45000 * 90),
    ])

    playlists = list_playlists(str(root))

    assert [p.name for p in playlists] == ['00800', '00001']
    repeated = playlists[1]
    assert repeated.duration_s == pytest.approx(12000.0)
    assert repeated.extraction_duration_s == pytest.approx(60.0)
    resolved = resolve_title(str(root), repeated)
    expected_clip = os.path.join(str(root), 'BDMV', 'STREAM', '00173.m2ts')
    assert resolved.clip_paths == [expected_clip]
    assert resolved.ffmpeg_input == expected_clip


def test_resolve_title_raises_when_clip_missing(tmp_path):
    root = _make_disc(tmp_path, clip_ids=('00001',))
    _write_mpls(str(root / 'BDMV' / 'PLAYLIST' / '00800.mpls'), [('00099', 0, 45000 * 100)])
    playlist = list_playlists(str(root))[0]

    with pytest.raises(FileNotFoundError):
        resolve_title(str(root), playlist)


# --- the title a library plays, and its streams from the feature (E4) -----------------------------------------------

from model.bdmv import TitleHint, codec_family, resolve_main_title  # noqa: E402

S = 45000   # clock ticks a second


def _layouts(**by_clip):
    ''' A fake ffprobe: each clip's audio, as `codec,channels` lines. '''
    def layout(path):
        return by_clip.get(os.path.splitext(os.path.basename(path))[0], ('ac3,2',))
    return layout


def _disc_with(tmp_path, playlists):
    clips = sorted({c for items in playlists.values() for c, _ in items})
    root = _make_disc(tmp_path, clips)
    for name, items in playlists.items():
        _write_mpls(os.path.join(root, 'BDMV', 'PLAYLIST', f'{name}.mpls'), [(c, 0, int(d * S)) for c, d in items])
    return root


def test_a_short_intro_with_other_audio_is_left_out_so_the_streams_are_the_features(tmp_path):
    ''' "A Star Is Born": a 22 s logo with stereo AC-3, then the DTS-HD MA feature; joined, it read as AC-3. '''
    root = _disc_with(tmp_path, {'00100': [('00064', 22), ('00020', 10555)]})

    title = resolve_main_title(root, layout=_layouts(**{'00064': ('ac3,2',), '00020': ('dts,6', 'ac3,2')}))

    assert title.dropped == ['00064'] and title.ffmpeg_input.endswith('00020.m2ts')
    assert title.duration_s == pytest.approx(10555)


@pytest.mark.parametrize('first, seconds, dropped', [
    (('dts,6', 'ac3,2'), 22, []),          # the same audio: a part of the feature
    (('ac3,2',), 300, []),                 # long: not an intro
])
def test_a_clip_is_kept_when_its_audio_matches_or_it_is_long(tmp_path, first, seconds, dropped):
    root = _disc_with(tmp_path, {'00100': [('00064', seconds), ('00020', 5000)]})
    title = resolve_main_title(root, layout=_layouts(**{'00064': first, '00020': ('dts,6', 'ac3,2')}))
    assert title.dropped == dropped


def test_a_short_clip_at_the_end_with_other_audio_is_left_out_too(tmp_path):
    root = _disc_with(tmp_path, {'00100': [('00020', 5000), ('00099', 10)]})
    assert resolve_main_title(root, layout=_layouts(**{'00020': ('dts,6',), '00099': ('ac3,2',)})).dropped == ['00099']


def test_the_playlist_the_library_names_is_the_one_resolved(tmp_path):
    root = _disc_with(tmp_path, {'00001': [('00001', 7000)], '00034': [('00034', 6500)]})
    layout = _layouts()
    assert resolve_main_title(root, '00034', layout=layout).playlist.name == '00034'
    assert resolve_main_title(root, '00034.mpls', layout=layout).playlist.name == '00034'
    with pytest.raises(ValueError, match='No playlist named 00999'):
        resolve_main_title(root, '00999', layout=layout)   # a person's choice is not guessed at


def test_without_a_name_the_title_is_the_one_as_long_as_the_library_says(tmp_path):
    ''' Wall-E: the library plays 00081 (5891.9 s), a branch of the longer 00082. '''
    root = _disc_with(tmp_path, {'00081': [('00081', 5891.9)], '00082': [('00082', 5922.0)]})
    assert resolve_main_title(root, duration_s=5891, layout=_layouts()).playlist.name == '00081'
    assert resolve_main_title(root, layout=_layouts()).playlist.name == '00082'   # no hint: the longest, as before


def test_two_titles_of_that_length_are_told_apart_by_the_first_audio_stream(tmp_path):
    ''' Glory: a stereo AC-3 decoy 0.2 s from the TrueHD Atmos feature. '''
    root = _disc_with(tmp_path, {'00246': [('00337', 7334.1)], '00001': [('00001', 7334.3)]})
    layout = _layouts(**{'00337': ('ac3,2', 'ac3,2'), '00001': ('truehd,8', 'ac3,6')})

    assert resolve_main_title(root, duration_s=7334, layout=layout, first_audio='TrueHD Atmos').playlist.name == '00001'
    assert resolve_main_title(root, duration_s=7334, layout=layout).playlist.name == '00246'   # closest, unaided


def test_a_named_playlist_missing_a_clip_falls_back_to_one_of_that_length(tmp_path):
    root = _disc_with(tmp_path, {'00801': [('00010', 6000)], '00800': [('00011', 6000)]})
    os.remove(os.path.join(root, 'BDMV', 'STREAM', '00010.m2ts'))
    assert resolve_main_title(root, '00801', duration_s=6000, layout=_layouts()).playlist.name == '00800'


def test_an_incomplete_rip_says_so_rather_than_extracting_some_other_title(tmp_path):
    ''' RoboCop: every feature-length playlist names a missing clip; a 3-minute extra is not the film. '''
    root = _disc_with(tmp_path, {'00800': [('01571', 6200)], '00801': [('01571', 6201)], '00302': [('00302', 180)]})
    os.remove(os.path.join(root, 'BDMV', 'STREAM', '01571.m2ts'))
    for kwargs in ({'playlist_name': '00801', 'duration_s': 6200}, {}):
        with pytest.raises(ValueError, match='an incomplete rip'):
            resolve_main_title(root, layout=_layouts(), **kwargs)


@pytest.mark.parametrize('name, family', [('TrueHD Atmos', 'truehd'), ('DTS-HD MA + DTS:X', 'dts'), ('AC-3', 'ac3'),
                                          ('E-AC3', 'eac3'), ('PCM_BLURAY', 'pcm_bluray'), ('ac3', 'ac3')])
def test_a_codec_is_compared_by_its_family(name, family):
    assert codec_family(name) == family
