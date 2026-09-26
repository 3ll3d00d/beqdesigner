"""The per-title Details view and run-status table delegate for the library work list."""
from collections import deque
import json
import os
import tempfile
from typing import Optional

from qtpy.QtCore import QEvent, QSize, Qt
from qtpy.QtGui import QGuiApplication
from qtpy.QtWidgets import QDialog, QDialogButtonBox, QPlainTextEdit, QPushButton, QStyle, \
    QStyleOptionButton, QStyledItemDelegate, QVBoxLayout

from model.execution_events import ExecutionEvent, event_text, redacted_event

MAX_EVENTS = 500
MAX_BUFFER_CHARS = 256 * 1024

DETAILS_FILE = '.worklist-run-details.json'


def load_run_details(work_dir: Optional[str]) -> dict[str, str]:
    '''Read the last run's already-redacted Details text; a damaged cache is simply ignored.'''
    if not work_dir:
        return {}
    try:
        with open(os.path.join(work_dir, DETAILS_FILE), encoding='utf-8') as source:
            data = json.load(source)
    except (OSError, ValueError):
        return {}
    return {key: value for key, value in data.items() if isinstance(key, str) and isinstance(value, str)} \
        if isinstance(data, dict) else {}


def save_run_details(work_dir: Optional[str], details: dict[str, str]) -> None:
    '''Atomically replace the last run's Details cache (including an empty new-run reset).'''
    if not work_dir:
        return
    os.makedirs(work_dir, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix='.run-details-', suffix='.tmp', dir=work_dir)
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as target:
            json.dump(details, target)
        os.replace(temporary, os.path.join(work_dir, DETAILS_FILE))
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


class EventBuffer:
    """Bounded, title-scoped event history; its redacted text is saved when a run ends."""

    def __init__(self):
        self._events = deque()
        self._chars = 0
        self.trimmed = 0

    def append(self, event: ExecutionEvent) -> None:
        event = redacted_event(event)
        rendered = event_text(event)
        self._events.append((event, rendered))
        self._chars += len(rendered)
        while len(self._events) > MAX_EVENTS or self._chars > MAX_BUFFER_CHARS:
            _, old = self._events.popleft()
            self._chars -= len(old)
            self.trimmed += 1

    def events(self):
        return tuple(event for event, _ in self._events)

    def text(self) -> str:
        prefix = f'[Earlier events trimmed: {self.trimmed}]\n\n' if self.trimmed else ''
        return prefix + '\n\n'.join(rendered for _, rendered in self._events)


class RunDetailsDialog(QDialog):
    def __init__(self, title: str, title_id: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f'Run details — {title}')
        self.resize(760, 520)
        self.title_id = title_id
        layout = QVBoxLayout(self)
        self.output = QPlainTextEdit(self)
        self.output.setReadOnly(True)
        self.output.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self.output)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        self.copyButton = QPushButton('Copy all', self)
        buttons.addButton(self.copyButton, QDialogButtonBox.ButtonRole.ActionRole)
        self.copyButton.clicked.connect(self.copy_all)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

    def set_text(self, text: str) -> None:
        old_text = self.output.toPlainText()
        if old_text == text:
            return
        scrollbar = self.output.verticalScrollBar()
        old_scroll = scrollbar.value()
        follow = old_scroll >= scrollbar.maximum()
        cursor = self.output.textCursor()
        old_position, old_anchor = cursor.position(), cursor.anchor()
        old_selection = cursor.selectedText().replace('\u2029', '\n')

        if text.startswith(old_text):
            # Appended output is inserted at the end so the reader's cursor,
            # selection and scroll position remain stable.
            suffix = text[len(old_text):]
            cursor.movePosition(cursor.MoveOperation.End)
            cursor.insertText(suffix)
            if follow:
                cursor.movePosition(cursor.MoveOperation.End)
                self.output.setTextCursor(cursor)
                scrollbar.setValue(scrollbar.maximum())
            else:
                cursor = self.output.textCursor()
                cursor.setPosition(old_anchor)
                cursor.setPosition(old_position, cursor.MoveMode.KeepAnchor)
                self.output.setTextCursor(cursor)
                scrollbar.setValue(old_scroll)
            return

        # The bounded event buffer dropped its prefix. Rebuild, then anchor
        # the reader at the nearest line that survived the trim.
        old_lines = old_text.splitlines(keepends=True)
        line_start = old_text.rfind('\n', 0, old_position) + 1
        line_number = old_text.count('\n', 0, line_start)
        candidates = old_lines[line_number:] + list(reversed(old_lines[:line_number]))
        self.output.setPlainText(text)
        cursor = self.output.textCursor()
        restored = False
        if old_selection and (selection_start := text.find(old_selection)) >= 0:
            cursor.setPosition(selection_start)
            cursor.setPosition(selection_start + len(old_selection), cursor.MoveMode.KeepAnchor)
            restored = True
        else:
            for line in candidates:
                anchor = line.strip()
                if len(anchor) < 8:
                    continue
                position = text.find(anchor)
                if position >= 0:
                    offset = min(max(0, old_position - line_start), len(anchor))
                    cursor.setPosition(position + offset)
                    restored = True
                    break
        if not restored:
            cursor.setPosition(min(old_position, len(text)))
        self.output.setTextCursor(cursor)
        if follow:
            cursor.movePosition(cursor.MoveOperation.End)
            self.output.setTextCursor(cursor)
            scrollbar.setValue(scrollbar.maximum())
        else:
            scrollbar.setValue(min(old_scroll, scrollbar.maximum()))

    def copy_all(self) -> None:
        QGuiApplication.clipboard().setText(self.output.toPlainText())


class RunStatusDelegate(QStyledItemDelegate):
    """Paints a progress bar or a Details button in their dedicated table columns."""

    def __init__(self, progress_column: int, details_column: int, open_details, parent=None):
        super().__init__(parent)
        self.progress_column, self.details_column = progress_column, details_column
        self.open_details = open_details

    def sizeHint(self, option, index):
        return QSize(170 if index.column() == self.progress_column else 82, 30)

    def paint(self, painter, option, index):
        state = index.data(Qt.ItemDataRole.UserRole + 20) or {}
        if index.column() == self.progress_column:
            if not state:
                return
            rect = option.rect.adjusted(4, 5, -4, -5)
            progress = QStyleOptionButton()
            progress.rect = rect
            progress.text = state.get('text', 'Queued' if state.get('queued') else '')
            progress.state = QStyle.StateFlag.State_Enabled
            painter.save()
            painter.setPen(option.palette.color(option.palette.ColorRole.Mid))
            painter.setBrush(option.palette.color(option.palette.ColorRole.Base))
            painter.drawRoundedRect(rect, 3, 3)
            if state.get('current') is not None and state.get('total'):
                width = max(0, min(rect.width(), int(rect.width() * state['current'] / state['total'])))
                fill = rect.adjusted(0, 0, -(rect.width() - width), 0)
                painter.fillRect(fill, option.palette.color(option.palette.ColorRole.Highlight))
            painter.setPen(option.palette.color(option.palette.ColorRole.Text))
            painter.drawText(rect.adjusted(5, 0, -5, 0), int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                             progress.text)
            painter.restore()
            return
        button = QStyleOptionButton()
        button.rect = option.rect.adjusted(5, 3, -5, -3)
        button.text = 'Details'
        button.state = QStyle.StateFlag.State_Raised
        if state.get('has_details'):
            button.state |= QStyle.StateFlag.State_Enabled
        style = option.widget.style() if option.widget else None
        if style:
            style.drawControl(QStyle.ControlElement.CE_PushButton, button, painter, option.widget)

    def editorEvent(self, event, model, option, index):
        state = index.data(Qt.ItemDataRole.UserRole + 20) or {}
        if index.column() != self.details_column or not state.get('has_details'):
            return False
        if event.type() == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
            self.open_details(index.data(Qt.ItemDataRole.UserRole + 2))
            return True
        if event.type() == QEvent.Type.KeyPress and event.key() in (Qt.Key.Key_Space, Qt.Key.Key_Return,
                                                                    Qt.Key.Key_Enter):
            self.open_details(index.data(Qt.ItemDataRole.UserRole + 2))
            return True
        return False
