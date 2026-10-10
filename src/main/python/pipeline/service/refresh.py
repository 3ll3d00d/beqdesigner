'''
Bringing the discovery index up to date after a person decides a title over HTTP (design/web-review.md §3): a decision writes
only the queue entry, and the index learns of it by reading the outputs again (`LibraryIndex.refresh()`), as the work list's
title page has it done when it is left.

The refresh runs on a thread of its own, after the response, and requests made while one is going are folded into the next.
It is done only when no run is going -- the service's own job, or the work list's or a command-line run holding the
work-directory lease -- because a run refreshes the index after every title and when it ends, which takes the decision in;
a refresh of ours alongside it could write a moment-old view over the run's.
'''
import logging
import os
import threading
from typing import Callable, Optional

from pipeline.library.index import LibraryIndex, index_path
from pipeline.service.lease import read_lease

logger = logging.getLogger('service_refresh')


class IndexRefresher:
    '''
    :param load: the job context loader (`load_context`) for the profile; read again per refresh.
    :param busy: whether a job of the service is queued or running.
    '''

    def __init__(self, load: Callable[[], object], busy: Callable[[], bool]):
        self.__load, self.__busy = load, busy
        self.__wanted = threading.Event()
        self.__idle = threading.Event()
        self.__idle.set()
        self.__lock = threading.Lock()
        self.__thread: Optional[threading.Thread] = None
        self.refreshed = 0          # how many refreshes were done (for tests and logs)
        self.skipped = 0            # how many were left to a run going

    def request(self) -> None:
        ''' Asks for a refresh; returns at once. '''
        with self.__lock:
            self.__idle.clear()
            self.__wanted.set()
            if self.__thread is None or not self.__thread.is_alive():
                self.__thread = threading.Thread(target=self.__work, name='index-refresh', daemon=True)
                self.__thread.start()

    def wait_idle(self, timeout: float = 30.0) -> bool:
        ''' Waits until no refresh is wanted or going; True if that happened in time. '''
        return self.__idle.wait(timeout)

    def refresh_now(self) -> bool:
        ''' Refreshes on this thread unless a run is going. :return: whether it refreshed. '''
        try:
            ctx = self.__load()
        except (OSError, ValueError) as error:
            logger.warning('index not refreshed after a decision: the profile cannot be read: %s', error)
            return False
        path = index_path(ctx.work_dir) if ctx.work_dir else ''
        if not path or not os.path.isfile(path):
            return False
        if self.__busy() or read_lease(ctx.work_dir) is not None:
            self.skipped += 1    # the run's own refreshes take the decision in
            return False
        with LibraryIndex(path) as index:
            index.refresh(ctx.profile, ctx.scan_settings())
        self.refreshed += 1
        return True

    def __work(self) -> None:
        while True:
            with self.__lock:
                if not self.__wanted.is_set():
                    self.__idle.set()
                    self.__thread = None
                    return
                self.__wanted.clear()
            try:
                self.refresh_now()
            except Exception:
                logger.exception('index refresh after a decision failed; the next run or scan brings it up to date')
