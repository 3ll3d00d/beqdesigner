'''
DVD-Video disc rips (a folder holding VIDEO_TS/VIDEO_TS.IFO), the counterpart of model/bdmv.py.

The disc's own tables say which titles exist and how long each plays, so a main feature can be chosen unattended:
VIDEO_TS.IFO lists the titles (which title set, and which title within it), and each title set's VTS_nn_0.IFO gives
the playback time of the program chain a title plays. Only what that needs is parsed -- no cell or VOBU tables,
no menus, no navigation commands.

Reading the audio is left to ffmpeg's `dvdvideo` demuxer (`-f dvdvideo -title N`, which needs an ffmpeg built with
libdvdread/libdvdnav), because a DVD title is a program chain that may play cells out of order or skip some, so
concatenating VTS_nn_m.VOB files would not reliably reproduce it (and would mix a multi-episode disc together).
Encrypted (CSS) discs are not readable without a CSS library; rips are normally already decrypted.

Field offsets follow libdvdread's ifo_types.h. All integers are big-endian; addresses in the headers are in
2048-byte sectors.
'''
import logging
import os
import re
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger('dvd')

SECTOR = 2048
VMG_MAGIC = b'DVDVIDEO-VMG'
VTS_MAGIC = b'DVDVIDEO-VTS'
_VMG_TT_SRPT = 0xC4  # sector of the title table in VIDEO_TS.IFO
_VTS_PTT_SRPT = 0xC8  # sector of the title -> (program chain, program) table in VTS_nn_0.IFO
_VTS_PGCIT = 0xCC  # sector of the program chain table in VTS_nn_0.IFO


@dataclass
class DvdTitle:
    ''' One entry of the disc's title table. `number` is what ffmpeg's -title option takes. '''
    number: int
    vts: int  # title set: VTS_<vts>_*.VOB
    vts_title: int  # this title's number within its title set
    chapters: int
    angles: int
    duration_s: float

    @property
    def name(self) -> str:
        ''' The selector callers pass to resolve_main_title(), as a BD playlist's basename is. '''
        return str(self.number)


@dataclass
class ResolvedDvdTitle:
    '''
    A title resolved to an ffmpeg input. Shaped like model.bdmv.ResolvedTitle (`playlist` carries `duration_s`
    and `name`) so callers that handle either can treat them alike; `input_options` are ffmpeg input options
    (format, title) that an ordinary path cannot express.
    '''
    playlist: DvdTitle
    ffmpeg_input: str
    display_name: str
    input_options: Dict[str, object] = field(default_factory=dict)


def dvd_root(path: str) -> Optional[str]:
    '''
    :param path: a candidate disc root, or the VIDEO_TS folder itself.
    :return: the disc root (the folder holding VIDEO_TS), or None if `path` is not a DVD-Video rip.
    '''
    normal = os.path.normpath(path)
    if os.path.basename(normal).upper() == 'VIDEO_TS':
        normal = os.path.dirname(normal)
    return normal if _video_ts(normal) is not None else None


def is_dvd_root(path: str) -> bool:
    return dvd_root(path) is not None


def _video_ts(root: str) -> Optional[str]:
    ''' The VIDEO_TS folder under `root` whatever its case (the spec says upper, rips vary), if it holds the VMG. '''
    try:
        entries = os.listdir(root)
    except OSError:
        return None
    for entry in entries:
        if entry.upper() == 'VIDEO_TS':
            folder = os.path.join(root, entry)
            if _find_file(folder, 'VIDEO_TS.IFO') is not None:
                return folder
    return None


def _find_file(folder: str, name: str) -> Optional[str]:
    try:
        for entry in os.listdir(folder):
            if entry.upper() == name.upper():
                return os.path.join(folder, entry)
    except OSError:
        pass
    return None


def _u16(data: bytes, offset: int) -> int:
    return struct.unpack_from('>H', data, offset)[0]


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from('>I', data, offset)[0]


def _bcd(value: int) -> int:
    return (value >> 4) * 10 + (value & 0x0F)


def _playback_seconds(data: bytes, offset: int) -> float:
    ''' A dvd_time_t: BCD hour, minute, second, then a frame count whose top two bits give the frame rate. '''
    hours, minutes, seconds, frames = data[offset:offset + 4]
    rate = {1: 25.0, 3: 29.97}.get(frames >> 6)
    fraction = _bcd(frames & 0x3F) / rate if rate else 0.0
    return _bcd(hours) * 3600 + _bcd(minutes) * 60 + _bcd(seconds) + fraction


def _read(path: str) -> bytes:
    with open(path, 'rb') as f:
        return f.read()


def _title_table(vmg: bytes) -> List[dict]:
    if vmg[:12] != VMG_MAGIC:
        raise ValueError('VIDEO_TS.IFO is not a DVD-Video manager file')
    start = _u32(vmg, _VMG_TT_SRPT) * SECTOR
    count = _u16(vmg, start)
    titles = []
    for i in range(count):
        entry = start + 8 + i * 12
        titles.append({'angles': vmg[entry + 1], 'chapters': _u16(vmg, entry + 2), 'vts': vmg[entry + 6],
                       'vts_title': vmg[entry + 7]})
    return titles


def _pgc_durations(vts: bytes) -> List[float]:
    ''' Playback time of each program chain of a title set, in PGCN order (index 0 is PGC number 1). '''
    if vts[:12] != VTS_MAGIC:
        raise ValueError('not a DVD-Video title set file')
    table = _u32(vts, _VTS_PGCIT) * SECTOR
    count = _u16(vts, table)
    durations = []
    for i in range(count):
        pgc_start = table + _u32(vts, table + 8 + i * 8 + 4)
        durations.append(_playback_seconds(vts, pgc_start + 4))
    return durations


def _title_pgcn(vts: bytes, vts_title: int) -> Optional[int]:
    ''' The program chain a title starts with, from VTS_PTT_SRPT (title -> its chapters' (PGCN, program)). '''
    table = _u32(vts, _VTS_PTT_SRPT) * SECTOR
    count = _u16(vts, table)
    if not 1 <= vts_title <= count:
        return None
    first_ptt = table + _u32(vts, table + 8 + (vts_title - 1) * 4)
    return _u16(vts, first_ptt)


def list_titles(root: str) -> List[DvdTitle]:
    '''
    :param root: the disc root.
    :return: every title on the disc that has a readable duration, longest first (so the first is the main
        feature, as with model.bdmv.list_playlists()). Titles whose title set cannot be read are skipped.
    '''
    folder = _video_ts(root)
    if folder is None:
        raise ValueError(f"{root} does not look like a DVD-Video rip (no VIDEO_TS/VIDEO_TS.IFO)")
    entries = _title_table(_read(_find_file(folder, 'VIDEO_TS.IFO')))
    titles: List[DvdTitle] = []
    cache: Dict[int, Optional[bytes]] = {}
    for number, entry in enumerate(entries, start=1):
        vts_no = entry['vts']
        if vts_no not in cache:
            path = _find_file(folder, f"VTS_{vts_no:02d}_0.IFO")
            cache[vts_no] = _read(path) if path else None
        vts = cache[vts_no]
        if vts is None:
            logger.warning(f"Title {number} of {root}: VTS_{vts_no:02d}_0.IFO is missing")
            continue
        try:
            pgcn = _title_pgcn(vts, entry['vts_title'])
            durations = _pgc_durations(vts)
            if pgcn is None or not 1 <= pgcn <= len(durations):
                continue
            titles.append(DvdTitle(number, vts_no, entry['vts_title'], entry['chapters'], entry['angles'],
                                   durations[pgcn - 1]))
        except (ValueError, struct.error) as e:
            logger.warning(f"Title {number} of {root} unreadable: {e}")
    return sorted(titles, key=lambda t: (-t.duration_s, t.number))


def resolve_title(root: str, title: DvdTitle) -> ResolvedDvdTitle:
    disc_name = os.path.basename(os.path.normpath(root))
    return ResolvedDvdTitle(playlist=title, ffmpeg_input=root, display_name=f"{disc_name}_t{title.number:02d}",
                            input_options={'f': 'dvdvideo', 'title': title.number})


def resolve_main_title(root: str, title_name: Optional[str] = None) -> ResolvedDvdTitle:
    '''
    Unattended title choice, as model.bdmv.resolve_main_title(): the longest title, or a specific one by number.
    :param root: the disc root (or its VIDEO_TS folder).
    :param title_name: a title number as text, e.g. '3'; the longest title is used when omitted.
    :raises ValueError: if this is not a DVD, has no readable titles, or title_name matches none.
    '''
    disc = dvd_root(root)
    if disc is None:
        raise ValueError(f"{root} does not look like a DVD-Video rip (no VIDEO_TS/VIDEO_TS.IFO)")
    titles = list_titles(disc)
    if not titles:
        raise ValueError(f"No playable titles found under {disc}")
    if title_name is not None:
        title = next((t for t in titles if t.name == str(title_name)), None)
        if title is None:
            raise ValueError(f"No title numbered {title_name} found under {disc}")
    else:
        title = titles[0]
    return resolve_title(disc, title)


_DVD_PSEUDO_FILE = re.compile(r'^(?P<root>.+?)[\\/]VIDEO_TS[\\/]VIDEO_TS\.dvd;\d+$', re.IGNORECASE)


def pseudo_file_root(filename: str) -> Optional[str]:
    ''' The disc folder named by a JRiver-style `<disc>\\VIDEO_TS\\VIDEO_TS.dvd;N` pseudo-file, else None. '''
    match = _DVD_PSEUDO_FILE.match(filename)
    if not match:
        return None
    root = match.group('root')
    return root + filename[len(root)] if root.endswith(':') else root
