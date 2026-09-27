'''
Fixtures for every test in `gui/`.

**No test may raise a modal question nobody can answer.** The title page asks "Discard what you typed?" (Discard / Cancel) when it
is left with an edit that cannot be saved -- and that includes the window closing at the end of a test. A test that types into the
episodes or another box and neither saves nor discards would then wait for a person at teardown (a hang was reproduced). So the
default question is answered here: Discard, without a dialog. A test about the question itself takes `real_ask_discard` and puts
it back; a test that wants a different answer sets `page.confirm_discard`, as before.

The work list's own close asks "A run is in progress. Cancel it and close?" when a test ends with a run still going (pytest-qt
closes the windows before any fixture tears down, so a fixture cannot cancel the run first; a hang was reproduced). It is answered
Yes -- cancel and close -- without a dialog; a test about that question takes `real_ask_cancel_run`.
'''
import gc

import pytest

_MESSAGE_BOXES = ('critical', 'warning', 'information', 'question')
_REAL_BOXES = {}


def pytest_configure(config):
    from qtpy.QtWidgets import QMessageBox
    _REAL_BOXES.update({kind: getattr(QMessageBox, kind) for kind in _MESSAGE_BOXES})


@pytest.fixture(autouse=True)
def _no_unanswered_message_box(monkeypatch):
    '''
    **A message box no test answers fails the test, with its words, instead of waiting for a person.** QMessageBox's static
    boxes are modal: shown in a test they block until the job's timeout, and CI says only where (a missing ffmpeg and a failed
    design both hung macOS and Windows that way). A test that expects a box patches it, as before, and that wins.
    '''
    from qtpy.QtWidgets import QMessageBox

    def unanswered(kind):
        def box(parent, title, text, *args, **kwargs):
            raise AssertionError(f'unanswered QMessageBox.{kind}: {title}: {text}')
        return staticmethod(box)

    for kind in _MESSAGE_BOXES:
        monkeypatch.setattr(QMessageBox, kind, unanswered(kind))


@pytest.fixture
def real_message_boxes(monkeypatch):
    ''' Puts back QMessageBox's real boxes, for a test that answers the one it opens (a QTimer that clicks a button). '''
    from qtpy.QtWidgets import QMessageBox
    for kind in _MESSAGE_BOXES:
        monkeypatch.setattr(QMessageBox, kind, _REAL_BOXES[kind])


@pytest.fixture(autouse=True)
def _no_answer_outlives_its_test():
    ''' A `when_modal` answer whose dialog never opened (the action was refused) went on to answer the next test's. '''
    yield
    from modal import cancel_all
    cancel_all()


@pytest.fixture(autouse=True)
def _no_modal_discard_question(monkeypatch):
    import ui.beq  # noqa: F401 (AGENTS.md gotcha 3: first)
    from model.worklist_title import TitlePage
    original = TitlePage._ask_discard

    def discard(self, reason: str) -> bool:
        return True

    discard.original = original
    monkeypatch.setattr(TitlePage, '_ask_discard', discard)


@pytest.fixture
def real_ask_discard():
    ''' The page's real question (a QMessageBox), for the test that is about it. '''
    from model.worklist_title import TitlePage
    return TitlePage._ask_discard.original


@pytest.fixture(autouse=True)
def _no_modal_cancel_run_question(monkeypatch):
    import ui.beq  # noqa: F401 (AGENTS.md gotcha 3: first)
    from model.worklist import WorkListWindow
    original = WorkListWindow._ask_cancel_run

    def cancel_and_close(self) -> bool:
        return True

    cancel_and_close.original = original
    monkeypatch.setattr(WorkListWindow, '_ask_cancel_run', cancel_and_close)


@pytest.fixture
def real_ask_cancel_run(monkeypatch, real_message_boxes):
    ''' Puts back the work list's real "Run in progress" question (a QMessageBox), for the tests that are about it. '''
    from model.worklist import WorkListWindow
    monkeypatch.setattr(WorkListWindow, '_ask_cancel_run', WorkListWindow._ask_cancel_run.original)


@pytest.fixture(autouse=True)
def _delete_windows_deterministically():
    '''
    **A window a test is done with is deleted here, not whenever the garbage collector gets to it.** Left to the collector, the
    window is destroyed in the middle of a later test's event loop; its widgets lose the keyboard as they go, `editingFinished`
    fires into a lambda whose closure the collector has already cleared, and the process dies with a segfault (reproduced in
    `test_worklist_metadata_fixes.py`). Deleting the C++ objects first, while their Python side is intact, makes that harmless.
    '''
    yield
    from qtpy.QtCore import QCoreApplication, QEvent
    from qtpy.QtWidgets import QApplication
    if QApplication.instance() is None:
        return
    for widget in QApplication.topLevelWidgets():
        widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    QApplication.processEvents()
    gc.collect()
