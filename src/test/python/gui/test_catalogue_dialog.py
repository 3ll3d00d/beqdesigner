'''
The catalogue browser's author filter: every author in the catalogue, as a multi-select list (none selected: all),
where it was five fixed radio buttons that left out authors such as mikejl and kaelaria.
'''
import ui.beq  # noqa: F401 -- must come before model.catalogue (-> model.report -> app); see test_filter_dialog.py

import json

from qtpy.QtCore import QRunnable, QSettings

import model.catalogue as catalogue
from model.preferences import BEQ_DOWNLOAD_DIR, Preferences

ENTRIES = [
    {'title': 'Alien', 'year': '1979', 'author': 'aron7awol'},
    {'title': 'Blade Runner', 'year': '1982', 'author': 'mikejl'},
    {'title': 'Contact', 'year': '1997', 'author': 'kaelaria'},
    {'title': 'Dune', 'year': '2021', 'author': 'mobe1969'},
    {'title': 'Dune', 'year': '2021', 'author': 'mikejl'},
]


class _NoDownload(QRunnable):
    def __init__(self, *args):
        super().__init__()

    def run(self):
        pass


def _dialog(qtbot, tmp_path, monkeypatch):
    prefs = Preferences(QSettings(str(tmp_path / 'settings.ini'), QSettings.Format.IniFormat))
    prefs.set(BEQ_DOWNLOAD_DIR, str(tmp_path / 'beq'))
    monkeypatch.setattr(catalogue, 'DatabaseDownloader', _NoDownload)
    dialog = catalogue.CatalogueDialog(None, prefs, lambda *args: None)
    qtbot.addWidget(dialog)
    (tmp_path / 'beq' / 'database.json').write_text(json.dumps(ENTRIES))
    dialog._CatalogueDialog__on_database_load(True)
    return dialog


def _authors_listed(dialog):
    return sorted(dialog.resultsList.item(i).text().split(' (')[-1].rstrip(')')
                  for i in range(dialog.resultsList.count()))


def test_every_author_in_the_catalogue_is_offered_in_order(qtbot, tmp_path, monkeypatch):
    dialog = _dialog(qtbot, tmp_path, monkeypatch)

    assert [dialog.authorFilter.item(i).text() for i in range(dialog.authorFilter.count())] == \
        ['aron7awol', 'kaelaria', 'mikejl', 'mobe1969']
    assert not hasattr(dialog, 'allReposRadioButton')


def test_with_no_author_selected_every_entry_shows(qtbot, tmp_path, monkeypatch):
    dialog = _dialog(qtbot, tmp_path, monkeypatch)
    dialog.apply_filter()
    assert dialog.resultsList.count() == len(ENTRIES)


def test_selecting_authors_shows_only_theirs(qtbot, tmp_path, monkeypatch):
    dialog = _dialog(qtbot, tmp_path, monkeypatch)

    dialog.authorFilter.item(2).setSelected(True)   # mikejl: one of the authors the radio buttons left out

    assert _authors_listed(dialog) == ['mikejl', 'mikejl']

    dialog.authorFilter.item(1).setSelected(True)   # and kaelaria
    assert _authors_listed(dialog) == ['kaelaria', 'mikejl', 'mikejl']

    for i in range(dialog.authorFilter.count()):
        dialog.authorFilter.item(i).setSelected(False)
    assert dialog.resultsList.count() == len(ENTRIES)
