'''
A tree dialog for choosing the JRiver browse node a library source reads -- design/library-sync-pipeline-plan.md
§11.4. Children are fetched only when a node is expanded, off the UI thread, so a large library costs nothing until
the user opens it.

The fetch function is injected (`fetch(node_id) -> list[BrowseNode]`) so the dialog knows nothing about MCWS and
can be driven by a fake.
'''
import logging
from typing import Callable, Optional

from qtpy.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from qtpy.QtWidgets import QDialog, QDialogButtonBox, QLabel, QTreeWidget, QTreeWidgetItem, QVBoxLayout

from pipeline.library.jriver import BrowseNode

logger = logging.getLogger('browse_node_picker')

_NODE_ID = Qt.ItemDataRole.UserRole + 1
_LOADED = Qt.ItemDataRole.UserRole + 2
ROOT_LABEL = 'Whole library (root)'


class _FetchSignals(QObject):
    fetched = Signal(int, object)
    failed = Signal(int, str)


class _FetchJob(QRunnable):
    def __init__(self, fetch: Callable[[int], list], node_id: int):
        super().__init__()
        self.signals = _FetchSignals()
        self.__fetch = fetch
        self.__node_id = node_id

    def run(self):
        try:
            self.signals.fetched.emit(self.__node_id, self.__fetch(self.__node_id))
        except Exception as error:
            logger.exception('Unable to list browse node %s', self.__node_id)
            self.signals.failed.emit(self.__node_id, f'{type(error).__name__}: {error}')


class JRiverBrowseNodePicker(QDialog):
    '''
    After exec(), `selected_node_id` and `selected_path` describe the choice (both None if cancelled).
    '''

    def __init__(self, parent, fetch: Callable[[int], list[BrowseNode]], current_id: int = -1):
        super().__init__(parent)
        self.setWindowTitle('Choose a JRiver browse node')
        self.setMinimumSize(420, 480)
        self.__fetch = fetch
        self.__jobs = {}
        self.__items: dict[int, QTreeWidgetItem] = {}
        self.selected_node_id: Optional[int] = None
        self.selected_path: Optional[str] = None
        self.__current_id = current_id

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.statusLabel = QLabel('')
        self.statusLabel.setWordWrap(True)
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttonBox.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        layout = QVBoxLayout(self)
        layout.addWidget(self.tree)
        layout.addWidget(self.statusLabel)
        layout.addWidget(self.buttonBox)

        root = self.__make_item(None, BrowseNode(-1, ROOT_LABEL))
        self.tree.itemExpanded.connect(self.__on_expanded)
        self.tree.itemSelectionChanged.connect(self.__on_selection_changed)
        self.tree.itemDoubleClicked.connect(lambda *_: self.__accept_selection())
        self.buttonBox.accepted.connect(self.__accept_selection)
        self.buttonBox.rejected.connect(self.reject)
        root.setExpanded(True)
        self.tree.setCurrentItem(root)

    def __make_item(self, parent: Optional[QTreeWidgetItem], node: BrowseNode) -> QTreeWidgetItem:
        item = QTreeWidgetItem(self.tree if parent is None else parent, [node.name])
        item.setData(0, _NODE_ID, node.id)
        item.setData(0, _LOADED, False)
        # every node is expandable until its children have been fetched and found to be none
        item.setChildIndicatorPolicy(QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator)
        self.__items[node.id] = item
        return item

    def __on_expanded(self, item: QTreeWidgetItem):
        node_id = item.data(0, _NODE_ID)
        if item.data(0, _LOADED) or node_id in self.__jobs:
            return
        job = _FetchJob(self.__fetch, node_id)
        job.signals.fetched.connect(self.__on_fetched)
        job.signals.failed.connect(self.__on_failed)
        self.__jobs[node_id] = job
        self.statusLabel.setText(f'Loading {item.text(0)}...')
        QThreadPool.globalInstance().start(job)

    def __on_fetched(self, node_id: int, children: list):
        self.__jobs.pop(node_id, None)
        self.statusLabel.setText('')
        item = self.__items.get(node_id)
        if item is None:
            return
        item.setData(0, _LOADED, True)
        for child in children:
            self.__make_item(item, child)
        if not children:
            item.setChildIndicatorPolicy(QTreeWidgetItem.ChildIndicatorPolicy.DontShowIndicator)
        self.__reveal_current(item)

    def __on_failed(self, node_id: int, message: str):
        self.__jobs.pop(node_id, None)
        self.statusLabel.setText(f'Unable to load this node ({message}). Close and enter the id by hand if needed.')
        item = self.__items.get(node_id)
        if item is not None:
            item.setExpanded(False)  # so expanding it again retries

    def __reveal_current(self, parent: QTreeWidgetItem):
        ''' Once a level has loaded, select the node that was already chosen if it is in it. '''
        current = self.__items.get(self.__current_id)
        if current is not None and current.parent() is parent:
            self.tree.setCurrentItem(current)
            self.__current_id = -1

    def __on_selection_changed(self):
        self.buttonBox.button(QDialogButtonBox.StandardButton.Ok).setEnabled(len(self.tree.selectedItems()) == 1)

    def __path(self, item: QTreeWidgetItem) -> str:
        names = []
        while item is not None and item.parent() is not None:
            names.append(item.text(0))
            item = item.parent()
        return ' > '.join(reversed(names)) or ROOT_LABEL

    def __accept_selection(self):
        selected = self.tree.selectedItems()
        if len(selected) == 1:
            self.selected_node_id = selected[0].data(0, _NODE_ID)
            self.selected_path = self.__path(selected[0])
            self.accept()
