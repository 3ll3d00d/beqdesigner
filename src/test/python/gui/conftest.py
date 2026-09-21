'''
Fixtures for every test in `gui/`.

**No test may raise a modal question nobody can answer.** The title page asks "Discard what you typed?" (Discard / Cancel) when it
is left with an edit that cannot be saved -- and that includes the window closing at the end of a test. A test that types into the
episodes or another box and neither saves nor discards would then wait for a person at teardown (a hang was reproduced). So the
default question is answered here: Discard, without a dialog. A test about the question itself takes `real_ask_discard` and puts
it back; a test that wants a different answer sets `page.confirm_discard`, as before.
'''
import pytest


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
