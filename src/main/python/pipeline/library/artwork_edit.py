'''
A title's artwork, chosen by a person (no widgets): the poster its report image is drawn with (`QueueEntry.art_path`) -- a
file, an image downloaded from a URL, or one sent to the service. Shared by the desktop title page
(`model.worklist_artwork`, `model.worklist_metadata`) and the service's artwork routes (design/web-parity.md WP4).

Checking that a file or a download really is an image (a png or a jpeg: what the report can use) decodes it whole with
Pillow. A download or an upload is kept only ever in `<queue dir>/_art_cache/`, under a name made from the title's id, so
it can never write outside that directory.
'''
import io
import logging
import os
import re
import tempfile
from typing import Callable, Optional

import requests
from PIL import Image, UnidentifiedImageError

from pipeline.review import QueueEntry, update_entry

logger = logging.getLogger('worklist')

ART_CACHE = '_art_cache'
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024
TIMEOUT_SECONDS = 30
_EXTENSIONS = {'png': '.png', 'jpeg': '.jpg', 'jpg': '.jpg'}


class ArtworkError(Exception):
    ''' Why an image could not be used, in words for the person (never a traceback). '''


def _extension_of(source, what: str) -> str:
    ''' The extension of a complete png or jpeg image read from `source` (a path or a file object). '''
    try:
        with Image.open(source) as image:
            extension = _EXTENSIONS.get(str(image.format or '').lower())
            if extension is None:
                raise ArtworkError(f'{what} is not a png or jpeg image')
            image.load()   # decodes it all: a truncated or damaged image raises here
    except ArtworkError:
        raise
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as error:
        if isinstance(error, UnidentifiedImageError):
            raise ArtworkError(f'{what} is not a png or jpeg image') from None
        raise ArtworkError('the image is damaged or incomplete') from None
    return extension


def image_extension(data: bytes) -> str:
    '''
    :return: '.png' or '.jpg' for `data` if it is a complete png or jpeg image.
    :raises ArtworkError: for anything else (a web page, a truncated download, a gif).
    '''
    return _extension_of(io.BytesIO(data), 'that')


def check_local_image(path: str) -> None:
    '''
    :raises ArtworkError: if `path` is not a readable png or jpeg file.
    '''
    if not os.path.isfile(path):
        raise ArtworkError(f'{path} is not a file')
    _extension_of(path, 'that file')


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
    return store_image(b''.join(chunks), queue_dir, entry_id)


def store_image(data: bytes, queue_dir: str, entry_id: str) -> str:
    '''
    Keeps an image -- downloaded, or sent to the service -- in the art cache, replacing the title's last one atomically.
    :return: the path written.
    :raises ArtworkError: if it is too big, or not a complete png or jpeg image; nothing is written then.
    '''
    if len(data) > MAX_DOWNLOAD_BYTES:
        raise ArtworkError(f'the image is bigger than {MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB')
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


def set_artwork(queue_dir: str, entry_id: str, path: Optional[str]) -> QueueEntry:
    '''
    Points the title's entry at `path` (a checked image), or at none. `art_overridden` is True once a person has set it:
    automatic artwork never overwrites that.
    :raises FileNotFoundError: if the title has no queue entry.
    '''
    return update_entry(queue_dir, entry_id, art_path=path, art_overridden=path is not None)
