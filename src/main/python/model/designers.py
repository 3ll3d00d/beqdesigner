'''
model/designers.py: design/http-designer-binding-plan.md phase 3 -- a
durable, named list of HTTP designer endpoints a user configures once
(Tools > Designers), registered automatically on every app startup (and
immediately on save) so they're available to pipeline.review.batch_design()'s
`designer` name parameter without writing any Python.
'''
import json
import logging

from qtpy.QtWidgets import QDialog, QMessageBox, QTableWidgetItem

from model.preferences import DESIGNER_HTTP_ENDPOINTS
from pipeline.designer.http_binding import http_designer
from pipeline.designer.registry import register_designer, registered_designers, unregister_designer
from ui.designers import Ui_designersDialog

logger = logging.getLogger('designers')

# distinguishes preference-configured designers from ad-hoc/in-process ones (e.g. registered by a test or a
# script), so re-registering on every startup or Designers-dialog save replaces cleanly rather than accumulating
_REGISTERED_PREFIX = 'http:'


def register_configured_designers(preferences):
    ''' Registers every endpoint in DESIGNER_HTTP_ENDPOINTS -- call once at app startup. '''
    for entry in preferences.get(DESIGNER_HTTP_ENDPOINTS):
        name = f"{_REGISTERED_PREFIX}{entry['name']}"
        register_designer(name, http_designer(entry['url'], headers=entry.get('headers') or None))


def _unregister_configured_designers():
    for name in list(registered_designers()):
        if name.startswith(_REGISTERED_PREFIX):
            unregister_designer(name)


class DesignersDialog(QDialog, Ui_designersDialog):

    def __init__(self, parent, preferences):
        super().__init__(parent)
        self.setupUi(self)
        self.__preferences = preferences
        self.addButton.clicked.connect(self.__add_row)
        self.removeButton.clicked.connect(self.__remove_selected_row)
        self.buttonBox.accepted.connect(self.__save)
        self.__load()

    def __load(self):
        self.designersTable.setRowCount(0)
        for entry in self.__preferences.get(DESIGNER_HTTP_ENDPOINTS):
            self.__append_row(entry.get('name', ''), entry.get('url', ''), entry.get('headers') or {})

    def __append_row(self, name='', url='', headers=None):
        row = self.designersTable.rowCount()
        self.designersTable.insertRow(row)
        self.designersTable.setItem(row, 0, QTableWidgetItem(name))
        self.designersTable.setItem(row, 1, QTableWidgetItem(url))
        self.designersTable.setItem(row, 2, QTableWidgetItem(json.dumps(headers or {})))

    def __add_row(self):
        self.__append_row()

    def __remove_selected_row(self):
        selection = self.designersTable.selectionModel()
        if selection.hasSelection():
            self.designersTable.removeRow(selection.selectedRows()[0].row())

    def __cell_text(self, row, col):
        item = self.designersTable.item(row, col)
        return item.text().strip() if item is not None else ''

    def __save(self):
        entries = []
        problems = []
        for row in range(self.designersTable.rowCount()):
            name = self.__cell_text(row, 0)
            url = self.__cell_text(row, 1)
            headers_text = self.__cell_text(row, 2)
            if not name and not url:
                continue  # a blank row added then left empty -- not an error, just skipped
            if not name or not url:
                problems.append(f"row {row + 1}: both name and URL are required")
                continue
            try:
                headers = json.loads(headers_text) if headers_text else {}
                if not isinstance(headers, dict):
                    raise ValueError('headers must be a JSON object')
            except ValueError as e:
                problems.append(f"row {row + 1} ({name}): invalid headers -- {e}")
                continue
            entries.append({'name': name, 'url': url, 'headers': headers})

        names = [e['name'] for e in entries]
        if len(names) != len(set(names)):
            problems.append('designer names must be unique')

        if problems:
            QMessageBox.critical(self, 'Cannot save', '\n'.join(problems))
            return

        self.__preferences.set(DESIGNER_HTTP_ENDPOINTS, entries)
        _unregister_configured_designers()
        register_configured_designers(self.__preferences)
        self.accept()
