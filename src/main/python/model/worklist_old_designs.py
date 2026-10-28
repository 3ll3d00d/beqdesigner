'''
Deleting a title's old designs from the work list's title page: its earlier design runs (`QueueEntry.variants`, kept by
each redesign), listed one a line to tick, with *Delete selected* and *Delete all*. The current design is not listed: it
is what the page shows and what a decision is made on. The deleting is `pipeline.library.variants.discard_variants()`,
which the service's `DELETE /v1/titles/{id}/variants` shares.
'''
from datetime import datetime
from typing import List, Optional, Sequence

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout


def describe_run(variant: dict) -> str:
    ''' An earlier design run in a line: its id, when, what it made, and the designer build. '''
    at = variant.get('designed_at') or ''
    try:
        at = datetime.fromisoformat(str(at).replace('Z', '+00:00')).astimezone().strftime('%Y-%m-%d %H:%M')
    except ValueError:
        at = at or 'when not recorded'
    designs = len(variant.get('candidates') or ())
    made = 'declined' if variant.get('declined') or (not designs and variant.get('decline_reason')) else \
        f'{designs} design{"" if designs == 1 else "s"}'
    build = variant.get('designer_build') or ''
    return ' · '.join(part for part in (str(variant.get('variant_id') or '?'), at, made, build) if part)


class OldDesignsDialog(QDialog):
    ''' The earlier design runs to tick; `chosen` is the ids to delete once accepted. '''

    def __init__(self, parent, title: str, variants: Sequence[dict]):
        super().__init__(parent)
        self.setWindowTitle(f'Old designs of {title}')
        self.chosen: List[str] = []
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f'{len(variants)} earlier design run{"" if len(variants) == 1 else "s"} of {title}. '
                                'The current design is kept.', self))
        self.runs = QListWidget(self)
        for variant in reversed(list(variants)):   # newest first, as the browser lists them
            item = QListWidgetItem(describe_run(variant), self.runs)
            item.setData(Qt.ItemDataRole.UserRole, variant.get('variant_id'))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
        self.runs.itemChanged.connect(lambda *_: self._enable())
        layout.addWidget(self.runs)
        buttons = QHBoxLayout()
        self.deleteSelectedButton = QPushButton('Delete selected', self)
        self.deleteAllButton = QPushButton(f'Delete all {len(variants)}', self)
        self.cancelButton = QPushButton('Cancel', self)
        for button in (self.deleteSelectedButton, self.deleteAllButton, self.cancelButton):
            button.setAutoDefault(False)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.deleteSelectedButton.clicked.connect(lambda: self._finish(self.ticked()))
        self.deleteAllButton.clicked.connect(lambda: self._finish(self.all_ids()))
        self.cancelButton.clicked.connect(self.reject)
        self._enable()

    def _items(self) -> List[QListWidgetItem]:
        return [self.runs.item(i) for i in range(self.runs.count())]

    def all_ids(self) -> List[str]:
        return [item.data(Qt.ItemDataRole.UserRole) for item in self._items()]

    def ticked(self) -> List[str]:
        return [item.data(Qt.ItemDataRole.UserRole) for item in self._items()
                if item.checkState() == Qt.CheckState.Checked]

    def _enable(self) -> None:
        self.deleteSelectedButton.setEnabled(bool(self.ticked()))

    def _finish(self, ids: List[str]) -> None:
        self.chosen = ids
        self.accept()


def ask_old_designs(parent, title: str, variants: Sequence[dict]) -> Optional[List[str]]:
    ''' The ids of the earlier runs to delete, or None if the person cancelled. '''
    dialog = OldDesignsDialog(parent, title, variants)
    return list(dialog.chosen) if dialog.exec() and dialog.chosen else None
