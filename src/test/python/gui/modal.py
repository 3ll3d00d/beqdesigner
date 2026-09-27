'''
Answering the modal dialog an action opens, as a person would: `when_modal(ConfirmDialog, respond)` just before the action.

A `QTimer.singleShot(0, respond)` fired from the first event loop pass, and under load that pass can come before the dialog's
own loop is running: `activeModalWidget()` was then None, the answer never came and the run waited out its timeout (seen on CI).
This polls until a dialog of that type is showing, then answers it once.
'''
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QApplication

_LIVE = []   # the polling timers, kept alive until they answer or give up


def when_modal(kind, respond, timeout_ms: int = 10000, interval_ms: int = 10) -> None:
    ''' Calls respond(dialog) once a modal widget of this kind is active; gives up quietly after timeout_ms. '''
    timer = QTimer()
    timer.setInterval(interval_ms)
    left = [max(1, timeout_ms // interval_ms)]
    _LIVE.append(timer)

    def poll():
        dialog = QApplication.activeModalWidget()
        left[0] -= 1
        if isinstance(dialog, kind) or left[0] <= 0:
            timer.stop()
            _LIVE.remove(timer)
            if isinstance(dialog, kind):
                respond(dialog)

    timer.timeout.connect(poll)
    timer.start()
