import qtawesome as qta
from qtpy.QtCore import Signal, QMimeData, QSize
from qtpy.QtGui import QPalette
from qtpy.QtWidgets import QLabel


class DropArea(QLabel):

    changed = Signal(QMimeData)

    def __init__(self, parent = None):
        super(DropArea, self).__init__(parent)

        self.callback = None
        height = self.frameGeometry().height()
        self.setPixmap(qta.icon('fa5s.file-import').pixmap(QSize(int(height/2), int(height/2))))
        self.setAcceptDrops(True)
        self.clear()

    def dragEnterEvent(self, event):
        self.setBackgroundRole(QPalette.Highlight)
        event.acceptProposedAction()
        self.changed.emit(event.mimeData())

    def dragMoveEvent(self, event):
        event.acceptProposedAction()

    def dropEvent(self, event):
        mime_data = event.mimeData()
        if mime_data.hasText():
            if self.callback is not None:
                self.callback(mime_data.text())

        self.setBackgroundRole(QPalette.Dark)
        event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self.clear()
        event.accept()

    def clear(self):
        self.setBackgroundRole(QPalette.Dark)
        self.changed.emit(None)
