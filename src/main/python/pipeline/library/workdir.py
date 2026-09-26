'''Readable extraction directories without changing stable library title IDs.'''
import os
import re
import threading
from functools import lru_cache
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pipeline.library.source import LibraryItem

TITLE_ID_MARKER = '.beq-title-id'
_MARKER = TITLE_ID_MARKER
_LOCK = threading.RLock()
_HASHED_ID = re.compile(r'^(?:fs-[0-9a-f]{16}|jriver-[0-9a-f]{12}-.+)$')


def _safe_name(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name).strip(' .')[:100].rstrip(' .')
    if name.upper().split('.')[0] in {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)),
                                      *(f'LPT{i}' for i in range(1, 10))}:
        name = '_' + name
    return name or 'Untitled track'


@lru_cache(maxsize=32)
def _named_dirs(root: str, mtime_ns: int) -> dict[str, str]:
    found = {}
    with os.scandir(root) as entries:
        for entry in entries:
            if not entry.is_dir() or entry.name.startswith('.'):
                continue
            try:
                with open(os.path.join(entry.path, _MARKER), encoding='utf-8') as marker:
                    identifier = marker.read().strip()
            except OSError:
                continue
            if identifier:
                found[identifier] = entry.path
    return found


def named_ids(root: Optional[str]) -> set[str]:
    '''Stable IDs claimed by readable extraction folders.'''
    if not root or not os.path.isdir(root):
        return set()
    return set(_named_dirs(root, os.stat(root).st_mtime_ns))


def work_ids(root: Optional[str]) -> set[str]:
    '''IDs claimed by both readable folders and older ID-named folders.'''
    if not root or not os.path.isdir(root):
        return set()
    named = _named_dirs(root, os.stat(root).st_mtime_ns)
    readable_names = {os.path.basename(path) for path in named.values()}
    return set(named) | {name for name in os.listdir(root) if name not in readable_names and
                         not name.startswith('.') and os.path.isdir(os.path.join(root, name))}


def entry_directory(root: str, identifier: str) -> str:
    '''Find a readable directory by its ID marker, or an older ID-named directory.'''
    old = os.path.join(root, identifier)
    if os.path.isdir(old):
        return old
    if os.path.isdir(root):
        return _named_dirs(root, os.stat(root).st_mtime_ns).get(identifier, old)
    return old


def item_directory(root: str, item: 'LibraryItem', *, create: bool = False) -> str:
    '''Choose a stable readable folder for this track; `create` reserves it before extraction.'''
    with _LOCK:
        found = entry_directory(root, item.id)
        if os.path.isdir(found) or not create or not _HASHED_ID.match(item.id):
            return found
        os.makedirs(root, exist_ok=True)
        title = _safe_name(item.display_name or os.path.splitext(os.path.basename(item.source_path))[0])[:75].rstrip(' .')
        base = _safe_name(f'{title} - audio {item.audio_stream + 1}')
        for suffix in range(1, 10000):
            name = base if suffix == 1 else f'{base} ({suffix})'
            path = os.path.join(root, name)
            try:
                os.mkdir(path)
            except FileExistsError:
                continue
            try:
                with open(os.path.join(path, _MARKER), 'x', encoding='utf-8') as marker:
                    marker.write(item.id)
            except BaseException:
                os.rmdir(path)
                raise
            _named_dirs.cache_clear()
            return path
        raise FileExistsError(f'no free extraction directory for {title!r}')
