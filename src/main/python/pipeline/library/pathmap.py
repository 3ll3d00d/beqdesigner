r'''
Translating the paths a library server reports into paths this machine can open --
design/library-sync-pipeline-plan.md §11.6.

JRiver reports the paths as its host sees them, and does so in Windows form (`W:\Films\x.mkv`, or a UNC
`\\nas\media\x.mkv`) whatever the client is; ffmpeg here needs a local path. A mapping says "the folder the server
calls X is folder Y on this machine".
'''
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PathMapping:
    source: str  # a folder as the server names it, e.g. 'W:\\Films' or '\\\\nas\\media'
    target: str  # the same folder on this machine, e.g. '/mnt/films'


_WINDOWS_DRIVE_PATH = re.compile(r'^[A-Za-z]:[\\/]')


def unmapped_path_problem(path: str, mappings: Sequence[PathMapping]) -> str | None:
    '''
    Explain a Windows path that this non-Windows process cannot open because no mapping claimed it.

    A normal POSIX path is deliberately never diagnosed, and Windows itself leaves its native paths alone.  The
    message contains the next useful action but not the server path: it is safe to put in an index/failure result.
    '''
    if os.name == 'nt' or not (_WINDOWS_DRIVE_PATH.match(path) or path.startswith('\\\\')):
        return None
    if translate_path(path, mappings) != path:
        return None
    return 'JRiver reported a Windows path with no local mapping; add one in Preferences > JRiver.'


def _normal(path: str) -> str:
    ''' Windows and POSIX separators are the same thing for matching; a trailing one is not significant. '''
    return path.replace('\\', '/').rstrip('/')


def translate_path(path: str, mappings: Sequence[PathMapping]) -> str:
    r'''
    Rewrites `path` using the mapping with the longest source prefix that contains it. Matching ignores case
    (Windows paths) and treats `\` and `/` alike, and a prefix only matches whole path components: `W:\Film`
    does not claim `W:\Films\x.mkv`. A path no mapping contains is returned unchanged.
    '''
    normal = path.replace('\\', '/')
    best = None
    for mapping in mappings:
        prefix = _normal(mapping.source)
        if not prefix and not mapping.source:
            continue  # an empty source would match everything
        if normal[:len(prefix)].lower() != prefix.lower():
            continue
        rest = normal[len(prefix):]
        if rest and not rest.startswith('/'):
            continue  # part-way through a component
        if best is None or len(prefix) > len(best[0]):
            best = (prefix, mapping, rest)
    if best is None:
        return path
    _, mapping, rest = best
    parts = [p for p in rest.split('/') if p]
    return os.path.join(mapping.target, *parts) if parts else mapping.target


def mappings_from_config(entries: Iterable | None) -> list[PathMapping]:
    r'''
    :param entries: from a config file: [{'from': 'W:\\Films', 'to': '/mnt/films'}, ...]; a `[from, to]` pair or a
        `'from=to'` string is accepted too (the last is what the CLI's --path-map gives).
    :raises ValueError: for anything else.
    '''
    mappings = []
    for entry in entries or []:
        if isinstance(entry, PathMapping):
            mappings.append(entry)
        elif isinstance(entry, Mapping) and 'from' in entry and 'to' in entry:
            mappings.append(PathMapping(str(entry['from']), str(entry['to'])))
        elif isinstance(entry, str) and '=' in entry:
            source, _, target = entry.partition('=')
            mappings.append(PathMapping(source, target))
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            mappings.append(PathMapping(str(entry[0]), str(entry[1])))
        else:
            raise ValueError(f'path mapping {entry!r} must be {{from, to}}, [from, to] or "from=to"')
    return mappings
