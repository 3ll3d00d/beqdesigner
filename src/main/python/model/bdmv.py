import glob
import logging
import os
import struct
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger('bdmv')

MPLS_MAGIC = b'MPLS'
CLOCK_TICKS_PER_SECOND = 45000.0


@dataclass
class PlayItem:
    '''
    A single clip reference within a playlist (BDMV/STREAM/<clip_id>.m2ts plus the portion of it to play).
    '''
    clip_id: str
    in_time: int
    out_time: int

    @property
    def duration_s(self) -> float:
        return max(0, self.out_time - self.in_time) / CLOCK_TICKS_PER_SECOND


@dataclass
class Playlist:
    '''
    A parsed BDMV/PLAYLIST/*.mpls file, i.e. an ordered list of clips that make up a single "title".
    '''
    mpls_path: str
    play_items: List[PlayItem] = field(default_factory=list)

    @property
    def name(self) -> str:
        return os.path.splitext(os.path.basename(self.mpls_path))[0]

    @property
    def duration_s(self) -> float:
        return sum(pi.duration_s for pi in self.play_items)

    @property
    def clip_ids(self) -> List[str]:
        return [pi.clip_id for pi in self.play_items]


@dataclass
class ResolvedTitle:
    '''
    A playlist resolved to a concrete ffmpeg input spec.
    '''
    playlist: Playlist
    ffmpeg_input: str
    display_name: str
    clip_paths: List[str]


def is_bdmv_root(path: str) -> bool:
    '''
    :param path: a candidate disc root directory.
    :return: True if this looks like a BD disc rip (i.e. it contains BDMV/index.bdmv).
    '''
    return os.path.isfile(os.path.join(path, 'BDMV', 'index.bdmv'))


def parse_mpls(mpls_path: str) -> Playlist:
    '''
    Parses a .mpls playlist file to discover the ordered list of clips (and their in/out points) that make up
    the title. Only the fields required to identify and concatenate clips are extracted; the rest of the format
    (menus, chapters, STN tables, sub paths etc) is not needed for extraction and is ignored. Field offsets are
    per the (reverse engineered) MPLS/PlayList/PlayItem structures documented at
    https://github.com/lw/BluRay/wiki/MPLS.
    :param mpls_path: path to the .mpls file.
    :return: the parsed playlist.
    '''
    with open(mpls_path, 'rb') as f:
        data = f.read()
    if data[0:4] != MPLS_MAGIC:
        raise ValueError(f"{mpls_path} is not a valid mpls file")
    playlist_start = struct.unpack('>I', data[8:12])[0]
    pos = playlist_start
    # PlayList(): length(4), reserved(2), number_of_PlayItems(2), number_of_SubPaths(2)
    number_of_play_items = struct.unpack('>H', data[pos + 6:pos + 8])[0]
    pos += 10
    play_items = []
    for _ in range(number_of_play_items):
        item_len = struct.unpack('>H', data[pos:pos + 2])[0]
        item_start = pos + 2
        # PlayItem(): length(2, already consumed) then Clip_Information_filename(5), Clip_codec_identifier(4),
        # is_multi_angle/connection_condition(2), ref_to_STC_id(1), IN_time(4), OUT_time(4)
        clip_id = data[item_start:item_start + 5].decode('ascii', errors='replace')
        times_offset = item_start + 5 + 4 + 2 + 1
        in_time, out_time = struct.unpack('>II', data[times_offset:times_offset + 8])
        play_items.append(PlayItem(clip_id=clip_id, in_time=in_time, out_time=out_time))
        pos = pos + 2 + item_len
    return Playlist(mpls_path=mpls_path, play_items=play_items)


def list_playlists(bdmv_root: str) -> List[Playlist]:
    '''
    :param bdmv_root: the disc root directory (containing a BDMV subfolder).
    :return: every playlist found under BDMV/PLAYLIST that references at least one clip, longest duration first.
    '''
    playlists = []
    for mpls_path in sorted(glob.glob(os.path.join(bdmv_root, 'BDMV', 'PLAYLIST', '*.mpls'))):
        try:
            playlist = parse_mpls(mpls_path)
        except Exception as e:
            logger.warning(f"Unable to parse {mpls_path}, skipping: {e}")
            continue
        if playlist.play_items:
            playlists.append(playlist)
    return sorted(playlists, key=lambda p: p.duration_s, reverse=True)


def resolve_title(bdmv_root: str, playlist: Playlist) -> ResolvedTitle:
    '''
    Resolves a playlist to a concrete ffmpeg input spec, concatenating clips via the concat protocol when the
    title spans more than one .m2ts clip.

    Each PlayItem's IN_time/OUT_time is used only to compute the reported duration (ResolvedTitle.playlist.
    duration_s) -- the whole of each referenced clip is used for extraction, not just the [IN_time, OUT_time)
    slice. This matches the overwhelmingly common case (a main feature's PlayItems reference whole, dedicated
    clips) but is not correct for a title that only plays a sub-range of a clip (e.g. a menu, an excerpt, or a
    seamless-branching angle segment): trimming that correctly would require translating IN_time/OUT_time
    (PTS on the clip's own timeline, which does not necessarily start at 0) into stream-relative offsets, which
    needs data (e.g. the clip's first PTS, found in its .clpi) this module does not parse.
    :param bdmv_root: the disc root directory.
    :param playlist: the playlist to resolve.
    :return: the resolved title.
    '''
    clip_paths = [os.path.join(bdmv_root, 'BDMV', 'STREAM', f"{clip_id}.m2ts") for clip_id in playlist.clip_ids]
    missing = [p for p in clip_paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"Clip(s) referenced by {playlist.mpls_path} not found: {', '.join(missing)}")
    if len(clip_paths) == 1:
        ffmpeg_input = clip_paths[0]
    else:
        ffmpeg_input = 'concat:' + '|'.join(p.replace('\\', '/') for p in clip_paths)
    disc_name = os.path.basename(os.path.normpath(bdmv_root))
    display_name = f"{disc_name}_{playlist.name}"
    return ResolvedTitle(playlist=playlist, ffmpeg_input=ffmpeg_input, display_name=display_name,
                         clip_paths=clip_paths)


def resolve_main_title(bdmv_root: str, playlist_name: str = None) -> ResolvedTitle:
    '''
    Convenience for unattended callers (batch/pipeline) that can't prompt the user to choose a title: resolves
    the main feature (the longest playlist) by default, or a specific playlist by name when given.
    :param bdmv_root: the disc root directory.
    :param playlist_name: the .mpls basename (without extension) to resolve, e.g. '00800'; the longest playlist
    is used when omitted.
    :return: the resolved title.
    :raises ValueError: if no playlists are found, or playlist_name doesn't match any of them.
    '''
    playlists = list_playlists(bdmv_root)
    if not playlists:
        raise ValueError(f"No playable titles found under {bdmv_root}")
    if playlist_name is not None:
        playlist = next((p for p in playlists if p.name == playlist_name), None)
        if playlist is None:
            raise ValueError(f"No playlist named {playlist_name} found under {bdmv_root}")
    else:
        playlist = playlists[0]
    return resolve_title(bdmv_root, playlist)
