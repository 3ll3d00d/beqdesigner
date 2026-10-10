'''
Rejected designs on the title page (designer contract 1.1, design/outstanding.md W4): the designs the designer judged unfit to
publish are listed after the candidates under a heading that cannot be picked, numbered on from them; picking one shows why it
was rejected, its commentary and its filter on the chart; and accepting one is an override that asks first -- Cancel writes
nothing -- and is then recorded on the entry and said on the page. `import ui.beq` first: see AGENTS.md gotcha 3.
'''
import ui.beq  # noqa: F401 (must come first)

from qtpy.QtCore import Qt

from model.worklist_title import candidate_text, override_question, state_text
from pipeline.review import read_entry
from test_worklist_title import _click, _designer, _open, _queue, _window  # noqa: F401 (a fixture)
from review_entry_fixture import rejected_designs, write_entry

WITH_REJECTED = [('r-alien', {'rejected': 2}), ('r-arrival', {}), ('r-sicario', {})]


def _asked(page, answer):
    ''' Answers the override question with `answer`, and keeps what was asked. '''
    questions = []
    page.confirm_override = lambda question: questions.append(question) or answer
    return questions


# --- what is said, with no widgets ----------------------------------------------------------------------------------------

def test_a_rejected_design_says_so_and_how_many_reasons_were_given():
    (design,) = rejected_designs()

    assert candidate_text(2, design).startswith('3: REJECTED (2 reasons) confidence=0.95 method=non_parametric')


def test_the_override_question_lists_the_designers_reasons():
    (design,) = rejected_designs()

    question = override_question(2, design)

    assert question.startswith('The designer rejected design 3 as unfit to publish:')
    assert '\n  - introduces a cliff of 53 dB/oct at 17 Hz\n  - corrected only down to 24.7 Hz; content continues to 16.7 Hz' \
        in question
    assert 'override' in question


def test_an_accepted_override_is_said_in_the_state_line(tmp_path):
    entry = write_entry(str(tmp_path), 'a', status='accepted', rejected=1, chosen=2)

    assert state_text(entry, None) == 'Accepted design 3, which the designer rejected (your override)'
    assert state_text(write_entry(str(tmp_path), 'b', status='accepted', rejected=1), None) == 'Accepted'


# --- the page -------------------------------------------------------------------------------------------------------------

def test_rejected_designs_are_listed_after_the_candidates_under_a_heading_that_cannot_be_picked(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, WITH_REJECTED)
    page = _open(qtbot, window, 'r-alien')
    rows = [page.candidateList.item(i) for i in range(page.candidateList.count())]

    assert [r.text()[:13] for r in rows] == ['1: confidence', '2: confidence',
                                             'Rejected by t', '3: REJECTED (', '4: REJECTED (']
    assert rows[2].text() == 'Rejected by the designer (2) -- for review; accepting one overrides the designer'
    assert rows[2].flags() == Qt.ItemFlag.NoItemFlags
    assert rows[3].font().italic() and 'introduces a cliff of 53 dB/oct' in rows[3].toolTip()
    assert page.picked == 0 and page.acceptButton.text() == 'Accept && next'   # the top pick, as ever


def test_picking_a_rejected_design_shows_why_and_its_filter_and_the_accept_button_says_override(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, WITH_REJECTED)
    page = _open(qtbot, window, 'r-alien')

    qtbot.keyClick(page.candidateList, Qt.Key.Key_3)   # the digits count on through the rejected designs

    assert page.picked == 2 and page.candidateList.currentRow() == 3
    assert page.commentaryHeading.text() == 'Rejected by the designer'
    # a reason is one bullet however it is punctuated: beqforge's own reasons contain '; ' (commentary_html would split them)
    assert page.commentaryText.toPlainText().split('\n')[:5] == [
        'Why the designer rejected it', 'introduces a cliff of 53 dB/oct at 17 Hz',
        'corrected only down to 24.7 Hz; content continues to 16.7 Hz', 'Strategy', 'flatten']
    assert page.acceptButton.text() == 'Override && accept...' and page.acceptButton.isEnabled()
    assert 'accepting it overrides the designer' in page.decisionLabel.text()
    assert any(name.startswith('Filtered') for name in page._magnitude.get_curve_names())   # its filter, on the chart

    qtbot.keyClick(page.candidateList, Qt.Key.Key_1)

    assert page.picked == 0 and page.commentaryHeading.text() == 'Commentary'
    assert page.acceptButton.text() == 'Accept && next'


def test_arrow_keys_step_over_the_heading(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, WITH_REJECTED)
    page = _open(qtbot, window, 'r-alien')
    page.pick_candidate(1)

    qtbot.keyClick(page.candidateList, Qt.Key.Key_Down)

    assert page.picked == 2 and page.candidateList.currentRow() == 3


def test_accepting_a_rejected_design_asks_first_and_cancel_writes_nothing(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, WITH_REJECTED)
    page = _open(qtbot, window, 'r-alien')
    questions = _asked(page, False)
    page.pick_candidate(3)

    assert page.accept() is False

    assert questions == [override_question(3, read_entry(_queue(tmp_path), 'r-alien').offered[3])]
    assert read_entry(_queue(tmp_path), 'r-alien').status == 'pending'
    assert page.current_id == 'r-alien' and 'not overridden' in page.decisionLabel.text()


def test_an_override_confirmed_is_accepted_recorded_and_moves_on(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, WITH_REJECTED)
    page = _open(qtbot, window, 'r-alien')
    questions = _asked(page, True)
    page.pick_candidate(2)

    _click(qtbot, page.acceptButton)

    entry = read_entry(_queue(tmp_path), 'r-alien')
    assert len(questions) == 1
    assert entry.status == 'accepted' and entry.chosen_candidate_index == 2 and entry.overrides_rejection
    assert page.current_id == 'r-arrival'    # and on, as any accept

    page.show_title('r-alien')

    assert page.picked == 2 and page.candidateList.currentRow() == 3
    assert page.decisionLabel.text() == 'Accepted design 3, which the designer rejected: your override.'
    assert 'which the designer rejected (your override)' in page.stateLabel.text()


def test_accepting_a_candidate_asks_nothing(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, WITH_REJECTED)
    page = _open(qtbot, window, 'r-alien')
    questions = _asked(page, False)

    assert page.accept() is True

    assert questions == [] and not read_entry(_queue(tmp_path), 'r-alien').overrides_rejection


def test_a_declined_titles_rejected_designs_can_be_viewed_and_one_accepted_over_the_decline(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, [('r-alien', {'decline': True, 'rejected': 1}), ('r-arrival', {})])
    page = _open(qtbot, window, 'r-alien')
    _asked(page, True)

    assert page.candidateList.count() == 3   # the flat candidate, the heading, the rejected design
    assert page.commentaryHeading.text() == 'Why the designer declined'
    page.pick_candidate(1)
    assert page.commentaryHeading.text() == 'Rejected by the designer'

    assert page.accept() is True

    entry = read_entry(_queue(tmp_path), 'r-alien')
    assert entry.declined and entry.chosen_candidate_index == 1 and entry.overrides_rejection


def test_a_redesign_that_changes_only_the_rejected_designs_is_not_accepted_blind(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, WITH_REJECTED)
    page = _open(qtbot, window, 'r-alien')
    _asked(page, True)
    page.pick_candidate(2)
    write_entry(_queue(tmp_path), 'r-alien', rejected=1)   # redesigned under the page: one rejected design fewer

    assert page.accept() is False

    assert read_entry(_queue(tmp_path), 'r-alien').status == 'pending'
    assert 'changed while it was open' in page.decisionLabel.text()
