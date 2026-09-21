'''
Artwork for the title page (design/library-sync/workflow-rework/design.md §12.10, chunk 27b): the poster a title's report
image is drawn with (`QueueEntry.art_path`), chosen by a person -- a file from disk, or an image downloaded from a URL.

What is here is everything about *getting* an image that needs no widget: checking that a file or a download really is an
image (a png or a jpeg: what the report can use), where a download is kept (only ever `<queue dir>/_art_cache/`, under a
name made from the title's id, so a download can never write outside that directory), and the `QRunnable` that does the
download off the UI thread. Writing the entry (`art_path`, `art_overridden`) is `model.worklist_metadata`'s job, on the
UI thread, once this has succeeded.
'''
import logging
import os
import re
import tempfile
from typing import Callable, Optional

import requests
from qtpy.QtCore import QBuffer, QIODevice, QObject, QRunnable, Signal
from qtpy.QtGui import QImageReader

logger = logging.getLogger('worklist')

ART_CACHE = '_art_cache'
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024
TIMEOUT_SECONDS = 30
_EXTENSIONS = {'png': '.png', 'jpeg': '.jpg', 'jpg': '.jpg'}


class ArtworkError(Exception):
    ''' Why an image could not be used, in words for the person (never a traceback). '''


def _format_of(reader: QImageReader) -> str:
    return bytes(reader.format()).decode('ascii', 'ignore').lower()


def image_extension(data: bytes) -> str:
    '''
    :return: '.png' or '.jpg' for `data` if it is a complete png or jpeg image.
    :raises ArtworkError: for anything else (a web page, a truncated download, a gif).
    '''
    buffer = QBuffer()
    buffer.setData(data)
    buffer.open(QIODevice.OpenModeFlag.ReadOnly)
    reader = QImageReader(buffer)
    extension = _EXTENSIONS.get(_format_of(reader))
    if extension is None:
        raise ArtworkError('that is not a png or jpeg image')
    if reader.read().isNull():
        raise ArtworkError('the image is damaged or incomplete')
    return extension


def check_local_image(path: str) -> None:
    '''
    :raises ArtworkError: if `path` is not a readable png or jpeg file.
    '''
    if not os.path.isfile(path):
        raise ArtworkError(f'{path} is not a file')
    reader = QImageReader(path)
    if _format_of(reader) not in _EXTENSIONS:
        raise ArtworkError('that file is not a png or jpeg image')
    if reader.read().isNull():
        raise ArtworkError('the image is damaged or incomplete')


def cache_path(queue_dir: str, entry_id: str, extension: str) -> str:
    '''
    :return: where a download for `entry_id` is kept: `<queue_dir>/_art_cache/<id><extension>`, the id reduced to
        characters that are safe in a file name. Nothing else is ever written by a download.
    :raises ArtworkError: if the result would not be directly inside the cache directory.
    '''
    cache_dir = os.path.join(queue_dir, ART_CACHE)
    name = re.sub(r'[^\w.\-]', '_', entry_id).lstrip('.') or 'artwork'
    destination = os.path.join(cache_dir, name + extension)
    if os.path.dirname(os.path.realpath(destination)) != os.path.realpath(cache_dir):
        raise ArtworkError(f'{entry_id!r} cannot be used to name a file')
    return destination


def download_artwork(url: str, queue_dir: str, entry_id: str, get: Optional[Callable] = None) -> str:
    '''
    Downloads the image at `url` into the art cache.
    :param get: `requests.get` unless given.
    :return: the path written.
    :raises ArtworkError: for a URL that is not http(s), a failed request, a download that is too big, or one that is not
        a png or jpeg image. Nothing is written in any of those cases.
    '''
    url = url.strip()
    if not url.lower().startswith(('http://', 'https://')):
        raise ArtworkError('the address must start with http:// or https://')
    try:
        response = (get or requests.get)(url, timeout=TIMEOUT_SECONDS, stream=True)
        response.raise_for_status()
        chunks, size = [], 0
        for chunk in response.iter_content(chunk_size=65536):
            size += len(chunk)
            if size > MAX_DOWNLOAD_BYTES:
                raise ArtworkError(f'the download is bigger than {MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB')
            chunks.append(chunk)
    except requests.RequestException as error:
        raise ArtworkError(f'the download failed: {error}') from error
    data = b''.join(chunks)
    destination = cache_path(queue_dir, entry_id, image_extension(data))
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix='.download.', suffix='.tmp', dir=os.path.dirname(destination))
    try:
        with os.fdopen(handle, 'wb') as f:
            f.write(data)
        os.replace(temporary, destination)   # a reader sees the old file or the new one, never half of one
    except BaseException:
        try:
            os.remove(temporary)
        except OSError:
            pass
        raise
    return destination


class DownloadSignals(QObject):
    finished = Signal(str)      # the path written
    failed = Signal(str)        # what to tell the person


class DownloadJob(QRunnable):
    ''' `download_artwork()` on the global thread pool (the house pattern: a `QRunnable` and a paired signals object). '''

    def __init__(self, url: str, queue_dir: str, entry_id: str):
        super().__init__()
        self.signals = DownloadSignals()
        self._url, self._queue_dir, self._entry_id = url, queue_dir, entry_id

    def run(self):
        try:
            self.signals.finished.emit(download_artwork(self._url, self._queue_dir, self._entry_id))
        except ArtworkError as error:
            self.signals.failed.emit(str(error))
        except Exception as error:   # a full disk, a permission: shown, never raised into the pool
            logger.exception('Could not download artwork for %s', self._entry_id)
            self.signals.failed.emit(f'{type(error).__name__}: {error}')
