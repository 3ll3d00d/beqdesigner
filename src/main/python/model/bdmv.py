import glob
import logging
import os
import struct
import subprocess
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger('bdmv')

MPLS_MAGIC = b'MPLS'
CLOCK_TICKS_PER_SECOND = 45000.0
# a clip at the start or end of a title this short, whose audio is not the feature's (a studio logo, a warning), is left
# out: ffmpeg takes a joined input's streams from its first clip, so the feature's would be read as the logo's
INTRO_LIMIT_S = 120.0


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

    @property
    def extraction_play_items(self) -> List[PlayItem]:
        '''
        The play items that can safely be represented by this module's whole-clip extraction.

        This resolver does not yet trim a clip to an item's IN/OUT points.  Consequently, feeding two
        consecutive references to the same clip to ffmpeg would append the *entire* m2ts twice.  Some discs
        (and particularly their protection/decoy playlists) contain very long runs of those references, which
        creates an enormous ``concat:`` URL and is never the intended whole-clip input.  Retain the first
        reference in each run; a later occurrence after another clip is retained because it may be a genuine
        branch back to that clip.
        '''
        items = []
        for item in self.play_items:
            if not items or items[-1].clip_id != item.clip_id:
                items.append(item)
        return items

    @property
    def extraction_duration_s(self) -> float:
        '''Duration represented by :attr:`extraction_play_items`, used for ffmpeg progress reporting.'''
        return sum(pi.duration_s for pi in self.extraction_play_items)

    @property
    def extraction_clip_ids(self) -> List[str]:
        return [pi.clip_id for pi in self.extraction_play_items]


@dataclass(frozen=True)
class TitleHint:
    ''' What a media library knows of a disc title, to choose its playlist (resolve_main_title()). '''
    duration_s: Optional[float] = None
    first_audio: Optional[str] = None   # the codec of its first audio stream, as the library names it


@dataclass
class ResolvedTitle:
    '''
    A playlist resolved to a concrete ffmpeg input spec.
    '''
    playlist: Playlist
    ffmpeg_input: str
    display_name: str
    clip_paths: List[str]
    input_options: dict = field(default_factory=dict)  # none for a BD; see model.dvd.ResolvedDvdTitle
    dropped: List[str] = field(default_factory=list)   # clips of the playlist left out (short, with other audio)

    @property
    def duration_s(self) -> float:
        ''' What is extracted: the playlist's whole-clip duration less any clip left out. '''
        return sum(pi.duration_s for pi in self.playlist.extraction_play_items if pi.clip_id not in self.dropped)


def is_bdmv_root(path: str) -> bool:
    '''
    :param path: a candidate disc root directory.
    :return: True if this looks like a BD disc rip (i.e. it contains BDMV/index.bdmv).
    '''
    return os.path.isfile(os.path.join(path, 'BDMV', 'index.bdmv'))


def bdmv_root_of(path: str) -> Optional[str]:
    '''
    :param path: a disc root directory, or the disc's own BDMV/index.bdmv (what a media library such as JRiver may
    report for a disc rip).
    :return: the disc root if path names a BD disc rip either way, else None.
    '''
    if is_bdmv_root(path):
        return path
    folder, name = os.path.split(os.path.normpath(path))
    if name.lower() == 'index.bdmv' and os.path.basename(folder).upper() == 'BDMV' and os.path.isfile(path):
        root = os.path.dirname(folder)
        if is_bdmv_root(root):
            return root
    return None


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
    :return: every playlist found under BDMV/PLAYLIST that references at least one clip, longest usable
    whole-clip extraction first.
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
    return sorted(playlists, key=lambda p: p.extraction_duration_s, reverse=True)


def audio_layout(clip_path: str) -> Tuple[str, ...]:
    '''
    A clip's audio streams as ffprobe sees them, (codec:channels, ...): enough to tell a logo's from a feature's. Empty
    when that cannot be known -- ffprobe is not installed (the app treats it as optional) or cannot read the clip -- so
    nothing is left out on a guess.
    '''
    try:
        out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a', '-show_entries',
                              'stream=codec_name,channels', '-of', 'csv=p=0', clip_path],
                             capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as error:
        logger.info(f"Cannot read the audio of {clip_path} ({error}); no clip is left out of its title")
        return ()
    return tuple(line.strip() for line in out.stdout.splitlines() if line.strip())


def _clips_to_drop(items: List[PlayItem], clip_path: Callable[[str], str],
                   layout: Callable[[str], Tuple[str, ...]]) -> List[str]:
    '''
    Short clips at either end of a title whose audio differs from its longest clip's ("A Star Is Born": a 22 s logo
    with stereo AC-3 before a DTS-HD MA feature, which a joined input would otherwise read as AC-3).
    '''
    if len(items) < 2:
        return []
    feature = max(items, key=lambda pi: pi.duration_s)
    expected = layout(clip_path(feature.clip_id))
    if not expected:   # the feature's audio is unknown: nothing to compare a clip with
        return []
    dropped = []
    for ends in (items, list(reversed(items))):
        for item in ends:
            if item is feature or item.duration_s >= INTRO_LIMIT_S or item.clip_id in dropped:
                break
            found = layout(clip_path(item.clip_id))
            if not found or found == expected:   # unknown is not different
                break
            dropped.append(item.clip_id)
    return dropped


def resolve_title(bdmv_root: str, playlist: Playlist,
                  layout: Callable[[str], Tuple[str, ...]] = audio_layout) -> ResolvedTitle:
    '''
    Resolves a playlist to a concrete ffmpeg input spec, concatenating clips via the concat protocol when the
    title spans more than one .m2ts clip.

    Each PlayItem's IN_time/OUT_time is used only to compute the reported duration (``extraction_duration_s``)
    -- the whole of each referenced clip is used for extraction, not just the [IN_time, OUT_time) slice. This
    matches the overwhelmingly common case (a main feature's PlayItems reference whole, dedicated clips) but is
    not correct for a title that only plays a sub-range of a clip (e.g. a menu, an excerpt, or a seamless-
    branching angle segment): trimming that correctly would require translating IN_time/OUT_time (PTS on the
    clip's own timeline, which does not necessarily start at 0) into stream-relative offsets, which needs data
    (e.g. the clip's first PTS, found in its .clpi) this module does not parse. Consecutive references to the
    same clip are collapsed for this same whole-clip limitation.
    :param bdmv_root: the disc root directory.
    :param playlist: the playlist to resolve.
    :return: the resolved title.
    '''
    def path_of(clip_id: str) -> str:
        return os.path.join(bdmv_root, 'BDMV', 'STREAM', f"{clip_id}.m2ts")
    clip_paths = [path_of(clip_id) for clip_id in playlist.extraction_clip_ids]
    missing = [p for p in clip_paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"Clip(s) referenced by {playlist.mpls_path} not found: {', '.join(missing)}")
    dropped = _clips_to_drop(playlist.extraction_play_items, path_of, layout)
    if dropped:
        logger.info(f"Leaving {', '.join(dropped)} out of {playlist.mpls_path}: short, with audio that is not the feature's")
        clip_paths = [path_of(c) for c in playlist.extraction_clip_ids if c not in dropped]
    if len(clip_paths) == 1:
        ffmpeg_input = clip_paths[0]
    else:
        ffmpeg_input = 'concat:' + '|'.join(p.replace('\\', '/') for p in clip_paths)
    disc_name = os.path.basename(os.path.normpath(bdmv_root))
    display_name = f"{disc_name}_{playlist.name}"
    return ResolvedTitle(playlist=playlist, ffmpeg_input=ffmpeg_input, display_name=display_name,
                         clip_paths=clip_paths, dropped=dropped)


def codec_family(name: str) -> str:
    ''' A codec as ffprobe names it (`dts`, `truehd`, `ac3`, `eac3`, `pcm_bluray`...), from its or a library's name. '''
    text = (name or '').strip().upper()
    for prefix, family in (('TRUEHD', 'truehd'), ('DTS', 'dts'), ('E-AC3', 'eac3'), ('EAC3', 'eac3'), ('AC-3', 'ac3'),
                           ('AC3', 'ac3'), ('PCM', 'pcm_bluray'), ('LPCM', 'pcm_bluray'), ('AAC', 'aac'), ('FLAC', 'flac')):
        if text.startswith(prefix):
            return family
    return text.lower()


def playlist_of_duration(bdmv_root: str, playlists: List[Playlist], duration_s: float,
                         first_audio: Optional[str] = None,
                         layout: Callable[[str], Tuple[str, ...]] = audio_layout) -> Optional[Playlist]:
    '''
    The playlist as long as `duration_s` (a media library's duration of the title), within a couple of seconds or
    0.2%, else None. Two can be that long ("Glory": a decoy of stereo AC-3 and the feature, 0.2 s apart), so where the
    library says what its first audio stream is (`first_audio`, a codec name), a playlist whose longest clip starts with
    a different codec is passed over. Then the closest in length.
    '''
    tolerance = max(2.0, duration_s * 0.002)
    close = sorted((p for p in playlists if abs(p.duration_s - duration_s) <= tolerance),
                   key=lambda p: abs(p.duration_s - duration_s))
    if len(close) > 1 and first_audio:
        wanted = codec_family(first_audio)

        def starts_with_it(playlist: Playlist) -> bool:
            feature = max(playlist.extraction_play_items, key=lambda pi: pi.duration_s)
            found = layout(os.path.join(bdmv_root, 'BDMV', 'STREAM', f'{feature.clip_id}.m2ts'))
            return bool(found) and codec_family(found[0].split(',')[0]) == wanted
        close = [p for p in close if starts_with_it(p)] or close
    return close[0] if close else None


def resolve_main_title(bdmv_root: str, playlist_name: str = None, duration_s: Optional[float] = None,
                       layout: Callable[[str], Tuple[str, ...]] = audio_layout,
                       first_audio: Optional[str] = None) -> ResolvedTitle:
    '''
    Convenience for unattended callers (batch/pipeline) that can't prompt the user to choose a title: resolves
    a specific playlist by name when given; else the one as long as `duration_s` when one is; else the main feature
    (the longest playlist).
    :param bdmv_root: the disc root directory.
    :param playlist_name: the .mpls basename (without extension) to resolve, e.g. '00800' (`.mpls` is allowed).
    :param duration_s: the title's duration as a media library reports it (JRiver's `Duration`).
    :param first_audio: the codec of the title's first audio stream as the library reports it, to tell apart two
        playlists of that duration (playlist_of_duration()).
    :return: the resolved title.
    :raises ValueError: if no playlists are found, or playlist_name doesn't match any of them and nothing else is
        known of the title (a caller naming a playlist with no duration: a person's choice, which must not be guessed).
    '''
    playlists = list_playlists(bdmv_root)
    if not playlists:
        raise ValueError(f"No playable titles found under {bdmv_root}")
    if playlist_name is not None:
        name = playlist_name[:-5] if playlist_name.lower().endswith('.mpls') else playlist_name
        playlist = next((p for p in playlists if p.name == name), None)
        if playlist is None and duration_s is None:
            raise ValueError(f"No playlist named {playlist_name} found under {bdmv_root}")
        if playlist is not None:
            try:
                return resolve_title(bdmv_root, playlist, layout)
            except FileNotFoundError as error:   # a library's playlist missing a clip (RoboCop): choose as below
                if duration_s is None:
                    raise
                logger.warning(f"{error}; choosing the title by its duration instead")
        else:
            logger.warning(f"No playlist named {playlist_name} under {bdmv_root}; choosing the title by its duration")
    # only titles as long as the one wanted: an incomplete rip (RoboCop, whose feature playlists all name a missing
    # clip) must fail, saying so, rather than fall back to an unrelated short title
    found = playlist_of_duration(bdmv_root, playlists, duration_s, first_audio, layout) if duration_s else None
    if duration_s and found is not None:
        tolerance = max(2.0, duration_s * 0.002)
        alike = [found] + [p for p in playlists if p is not found and abs(p.duration_s - duration_s) <= tolerance]
    else:
        longest = playlists[0].extraction_duration_s
        alike = [p for p in playlists if p.extraction_duration_s >= longest * 0.9]
    missing = []
    for playlist in alike:
        try:
            return resolve_title(bdmv_root, playlist, layout)
        except FileNotFoundError as error:
            logger.warning(f"{error}; trying another title of that length")
            missing.append(str(error))
    raise ValueError(f"The main title of {bdmv_root} cannot be read, an incomplete rip? {missing[0]}")
