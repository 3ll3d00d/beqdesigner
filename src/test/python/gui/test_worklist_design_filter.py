'''
model/worklist_design_filter.py (TODO L3, design/design-version.md "Caller behaviour" 3): the work list's *Design* filter
(Any / Declined / Out of date) and the out-of-date count on the drift banner. The designer is a registered HTTP designer whose
`/health` is answered by a fake `check_designer`, so the identity goes through the same `designer_identity()` the service uses.
`import ui.beq` first: see AGENTS.md gotcha 3.
'''
import ui.beq  # noqa: F401 (must come first)

import json

import pytest

from model.preferences import DESIGNER_DEFAULT
from model.worklist import WorkListWindow
from model.worklist_design_filter import NO_IDENTITY
from model.worklist_model import DESIGN_ANY, DESIGN_DECLINED, DESIGN_OUT_OF_DATE
from pipeline.designer.registry import register_designer, unregister_designer
from pipeline.library.index import LibraryIndex
from test_worklist_title import NOW, SOURCES, _prefs, _row
from worklist_fixture import make_index

URL = 'http://designer.invalid:8420/design'


def _summary(version, settings='abc'):
    return json.dumps({'design_version': version, 'design_settings': settings, 'designer_build': 'test'})


def _rows():
    return [
        _row('r-old', 'Old', 'review', 9, review_state='pending', detail='conf 0.90 - 1 candidate', entry_summary=_summary(6)),
        _row('r-new', 'New', 'review', 8, review_state='pending', detail='conf 0.90 - 1 candidate', entry_summary=_summary(8)),
        _row('r-decl', 'Declined', 'review', 7, review_state='pending', detail='designer declined: no loud events',
             entry_summary=_summary(8)),
        _row('r-other', 'Other settings', 'review', 6, review_state='pending', detail='conf 0.80 - 1 candidate',
             entry_summary=_summary(8, 'xyz')),
        _row('x-gravity', 'Gravity', 'extract', 1, detail='new'),
    ]


@pytest.fixture
def http_designer():
    register_designer(URL, lambda request: None)
    yield URL
    unregister_designer(URL)


def _window(qtbot, tmp_path, monkeypatch, health, designer=URL):
    asked = []

    def check(url, **kwargs):
        asked.append(url)
        return health

    monkeypatch.setattr('pipeline.library.setup.check_designer', check)
    prefs = _prefs(tmp_path)
    prefs.set(DESIGNER_DEFAULT, designer)
    make_index(tmp_path / 'work', _rows(), SOURCES, generation=2, last_scan_at=NOW - 900)
    window = WorkListWindow(None, prefs, auto_scan=False, clock=lambda: NOW)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    return window, asked


def _out_of_date_item(window):
    combo = window.designFilterCombo
    return combo.model().item(combo.findData(DESIGN_OUT_OF_DATE))


def test_out_of_date_lists_the_designs_a_redesign_now_could_answer_differently(qtbot, tmp_path, monkeypatch, http_designer):
    window, asked = _window(qtbot, tmp_path, monkeypatch, {'design_version': 8, 'design_settings': 'abc'})
    qtbot.waitUntil(lambda: window.current_identity is not None, timeout=5000)
    assert asked == [URL] and window.current_identity.version == 8
    assert _out_of_date_item(window).isEnabled()

    window.set_design_filter(DESIGN_OUT_OF_DATE)

    assert window.listed_ids() == ['r-old', 'r-other']      # an older version, and the same version with other settings
    selection = window.current_selection()
    assert selection.out_of_date is True
    with LibraryIndex(str(tmp_path / 'work' / 'library-index.sqlite')) as index:   # what the buttons act on is what is listed
        assert [r.id for r in selection.rows(index, include_done=False)] == window.listed_ids()
    assert window.chip_counts()['Review'] == 2 and window.chip_counts()['Extract'] == 0


def test_declined_lists_only_the_designer_declined_titles(qtbot, tmp_path, monkeypatch, http_designer):
    window, _ = _window(qtbot, tmp_path, monkeypatch, {'design_version': 8, 'design_settings': 'abc'})

    window.set_design_filter(DESIGN_DECLINED)

    assert window.listed_ids() == ['r-decl']
    assert window.current_selection().declined is True
    window.set_design_filter(DESIGN_ANY)
    assert len(window.listed_ids()) == 5


def test_the_banner_counts_the_out_of_date_designs_and_shows_them(qtbot, tmp_path, monkeypatch, http_designer):
    window, _ = _window(qtbot, tmp_path, monkeypatch, {'design_version': 8, 'design_settings': 'abc'})
    qtbot.waitUntil(lambda: window.driftBanner.isVisible(), timeout=5000)

    assert window.driftBannerLabel.text().startswith('2 designs were made with another design version or settings')
    assert 'design version 8' in window.driftBannerLabel.text()
    assert window.driftOutOfDateButton.isVisible() and not window.driftBannerButton.isVisible()   # nothing accepted drifted

    window.driftOutOfDateButton.click()
    assert window.design_filter == DESIGN_OUT_OF_DATE and window.listed_ids() == ['r-old', 'r-other']

    window.driftDismissButton.click()
    assert not window.driftBanner.isVisible()


def test_a_designer_that_says_no_version_leaves_out_of_date_unoffered(qtbot, tmp_path, monkeypatch, http_designer):
    window, asked = _window(qtbot, tmp_path, monkeypatch, {})   # a pre-1.4 designer
    qtbot.waitUntil(lambda: asked == [URL], timeout=5000)
    qtbot.waitUntil(lambda: not window._identity_jobs, timeout=5000)

    assert window.current_identity is None
    assert not _out_of_date_item(window).isEnabled() and _out_of_date_item(window).toolTip() == NO_IDENTITY
    with pytest.raises(ValueError):
        window.set_design_filter(DESIGN_OUT_OF_DATE)
    assert window.out_of_date_ids() == [] and not window.driftBanner.isVisible()


def test_a_designer_with_no_address_is_not_asked(qtbot, tmp_path, monkeypatch):
    window, asked = _window(qtbot, tmp_path, monkeypatch, {'design_version': 8}, designer='')   # the manual designer

    assert asked == [] and not window._identity_jobs
    assert window.current_identity is None and not _out_of_date_item(window).isEnabled()


def test_out_of_date_falls_back_to_any_when_the_designer_stops_saying(qtbot, tmp_path, monkeypatch, http_designer):
    window, _ = _window(qtbot, tmp_path, monkeypatch, {'design_version': 8, 'design_settings': 'abc'})
    qtbot.waitUntil(lambda: window.current_identity is not None, timeout=5000)
    window.set_design_filter(DESIGN_OUT_OF_DATE)

    monkeypatch.setattr('pipeline.library.setup.check_designer', lambda url, **kwargs: {})
    window.refresh_identity()
    qtbot.waitUntil(lambda: window.current_identity is None, timeout=5000)

    assert window.design_filter == DESIGN_ANY and len(window.listed_ids()) == 5
