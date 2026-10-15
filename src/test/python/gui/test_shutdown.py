'''
model/shutdown.py: background work is waited for while Python is alive, and a runnable that does not finish ends the
process rather than hanging it in Qt's teardown.
'''
import threading

from qtpy.QtCore import QRunnable, QThreadPool

from model.shutdown import finish_background_work


class _Job(QRunnable):
    def __init__(self, release: threading.Event):
        super().__init__()
        self.release, self.done = release, threading.Event()

    def run(self):
        self.release.wait(10)
        self.done.set()


def test_work_that_finishes_is_waited_for(qtbot):
    pool, release = QThreadPool(), threading.Event()
    job = _Job(release)
    pool.start(job)
    threading.Timer(0.2, release.set).start()
    exits = []

    assert finish_background_work(pool, 5000, exits.append)

    assert job.done.is_set() and exits == []


def test_work_that_does_not_finish_ends_the_process_instead_of_hanging(qtbot):
    pool, release = QThreadPool(), threading.Event()
    pool.start(_Job(release))
    exits = []
    try:
        assert not finish_background_work(pool, 200, exits.append)
        assert exits == [0]
    finally:
        release.set()
        pool.waitForDone(5000)
