'''
Leaving the app while background work (a QRunnable on the global thread pool) is still going.

`app.exec()` returning is not the end: Python then finalises, and only afterwards does `~QCoreApplication` wait for the
pool. A runnable that comes back into Python after finalisation can never take the GIL again, so that wait never ended --
seen with a library scan held up writing its index over a share: the window was gone and the process, holding the index
locked, stayed. The work is waited for here instead, while Python is alive, and for a bounded time.
'''
import logging
import os
from typing import Callable

from qtpy.QtCore import QThreadPool

logger = logging.getLogger('shutdown')

FINISH_WAIT_MS = 60_000


def finish_background_work(pool: QThreadPool = None, wait_ms: int = FINISH_WAIT_MS,
                           exit_now: Callable[[int], None] = os._exit) -> bool:
    '''
    Waits for the pool's runnables to finish, at most `wait_ms`.
    :param exit_now: called with 0 if they have not finished, to end the process now rather than hang in Qt's teardown
        (whatever they were writing is left for its own recovery: the index's journal is rolled back on its next open).
    :return: True if everything finished.
    '''
    pool = pool or QThreadPool.globalInstance()
    if pool.waitForDone(wait_ms):
        return True
    logger.error('Background work did not finish within %d s of closing; ending the process', wait_ms // 1000)
    logging.shutdown()
    exit_now(0)
    return False
