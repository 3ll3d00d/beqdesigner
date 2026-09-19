'''
A LibrarySource over plain files on disk -- what Batch Extract's file search does, made a source the library
pipeline (extract cache, design cache, review queue, sync) can run against. design/library-sync-pipeline-plan.md
§11.2.

Matching follows model/batch.py's FileSearch: each entry is a glob (recursive, so `**` works); an entry that is a
directory means "everything directly in it". A match is a media file, or a Blu-ray disc rip root (a folder holding
BDMV/index.bdmv), whose main feature the pipeline resolves at extract time.
'''
import glob
import hashlib
import os
from collections.abc import Collection, Iterable, Sequence
from typing import Any, Optional

from model.bdmv import is_bdmv_root
from pipeline.library.source import LibraryItem

# what a directory glob is narrowed to by default: a folder of films usually holds artwork, subtitles and nfo
# files too, and each would otherwise be a failed item in an unattended run
DEFAULT_MEDIA_EXTENSIONS = frozenset({
    '.mkv', '.mp4', '.m4v', '.avi', '.mov', '.ts', '.m2ts', '.mts', '.mpg', '.mpeg', '.wmv', '.webm', '.iso',
    '.flac', '.wav', '.mka', '.ac3', '.eac3', '.dts', '.thd', '.truehd', '.aac', '.mp3', '.opus', '.ogg',
})


class FilesystemLibrarySource:
    '''
    Identity is `fs-<hash of the resolved path>`: a filesystem has no key that survives a rename, so a moved or
    renamed file is a new title (JRiver's Key does not have this limitation).
    '''

    def __init__(self, globs: Sequence[str], extensions: Optional[Collection[str]] = DEFAULT_MEDIA_EXTENSIONS):
        '''
        :param globs: glob patterns and/or directories.
        :param extensions: lower-case suffixes (with the dot) a matched *file* must have; None accepts any file.
        '''
        if not globs:
            raise ValueError('at least one glob or directory is required')
        self.globs = list(globs)
        self.extensions = None if extensions is None else {e.lower() for e in extensions}

    def list_items(self, **query: Any) -> Iterable[LibraryItem]:
        if query:
            raise TypeError('FilesystemLibrarySource is configured by its globs; list_items accepts no query')
        items = {}
        for match in self._matches():
            item = self._to_item(match)
            if item is not None:
                items.setdefault(item.id, item)
        return sorted(items.values(), key=lambda i: i.source_path)

    def _matches(self) -> Iterable[str]:
        for pattern in self.globs:
            if os.path.isdir(pattern):
                pattern = os.path.join(pattern, '*')
            yield from glob.iglob(pattern, recursive=True)

    def _to_item(self, path: str) -> Optional[LibraryItem]:
        if 'BDMV' in path.replace('\\', '/').split('/'):
            return None  # a file inside a disc rip: the disc root is the item, not its clips
        if os.path.isdir(path):
            if not is_bdmv_root(path):
                return None
            display_name = os.path.basename(os.path.normpath(path))
            fingerprint = _stat_fingerprint(os.path.join(path, 'BDMV', 'index.bdmv'))
        elif os.path.isfile(path):
            if self.extensions is not None and os.path.splitext(path)[1].lower() not in self.extensions:
                return None
            display_name = os.path.splitext(os.path.basename(path))[0]
            fingerprint = ''  # the extract cache's own mtime/size fallback applies
        else:
            return None
        return LibraryItem(id=_item_id(path), source_path=path, display_name=display_name, fingerprint=fingerprint)


def _item_id(path: str) -> str:
    resolved = os.path.normcase(os.path.realpath(path))
    return f"fs-{hashlib.sha256(resolved.encode('utf-8')).hexdigest()[:16]}"


def _stat_fingerprint(path: str) -> str:
    try:
        stat = os.stat(path)
    except OSError:
        return ''
    return f"{stat.st_mtime_ns}:{stat.st_size}"
