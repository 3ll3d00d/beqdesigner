'''
Artwork for the title page (design/archive/library-sync/workflow-rework/design.md §12.10, chunk 27b): the poster a title's report
image is drawn with (`QueueEntry.art_path`), chosen by a person -- a file from disk, or an image downloaded from a URL.

Checking an image and where a download is kept are `pipeline.library.artwork_edit`'s, shared with the service; what is here
is the `QRunnable` that does a download off the UI thread. Writing the entry (`art_path`, `art_overridden`) is `model.worklist_metadata`'s job, on the
UI thread, once this has succeeded.
'''
import logging

from qtpy.QtCore import QObject, QRunnable, Signal

from pipeline.library.artwork_edit import ART_CACHE, MAX_DOWNLOAD_BYTES, ArtworkError, cache_path, check_local_image, \
    download_artwork, image_extension, set_artwork, store_image  # noqa: F401 (the page's callers and tests)

logger = logging.getLogger('worklist')


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
