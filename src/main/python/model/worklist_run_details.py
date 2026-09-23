"""The per-title Details view and run-status table delegate for the library work list."""
from collections import deque
from dataclasses import dataclass
import re
import shlex
import time
from typing import Optional

from qtpy.QtCore import QEvent, QSize, Qt
from qtpy.QtGui import QGuiApplication
from qtpy.QtWidgets import QDialog, QDialogButtonBox, QPlainTextEdit, QPushButton, QStyle, \
    QStyleOptionButton, QStyledItemDelegate, QVBoxLayout

from model.execution_events import ExecutionEvent

MAX_EVENTS = 500
MAX_EVENT_TEXT = 8192
MAX_BUFFER_CHARS = 256 * 1024

_SECRET = re.compile(r'(?i)(api[_-]?key|access[_-]?token|password|secret)([=: ]+)([^\s&]+)')
_URL_CREDENTIALS = re.compile(r'(https?://)[^/@\s]+:[^/@\s]+@', re.IGNORECASE)
_AUTH_HEADER = re.compile(r'(?i)(authorization\s*[:=]\s*(?:bearer|basic)\s+)[^\s,;]+')
_COOKIE_HEADER = re.compile(r'(?i)((?:set-)?cookie\s*[:=]\s*)[^\r\n]+')
_SECRET_ARG = re.compile(r'^--?(?:api[_-]?key|access[_-]?token|password|secret)$', re.IGNORECASE)
MAX_COMMAND_ARGS = 256
MAX_COMMAND_CHARS = 32 * 1024


def _redact(text: str) -> str:
    text = _URL_CREDENTIALS.sub(r'\1[REDACTED]@', text)
    text = _AUTH_HEADER.sub(r'\1[REDACTED]', text)
    text = _COOKIE_HEADER.sub(r'\1[REDACTED]', text)
    return _SECRET.sub(r'\1\2[REDACTED]', text)


def _redact_argv(argv) -> tuple:
    safe, redact_next = [], False
    remaining = MAX_COMMAND_CHARS
    for raw in list(argv)[:MAX_COMMAND_ARGS]:
        arg = str(raw)
        if redact_next:
            safe.append('[REDACTED]')
            redact_next = False
            continue
        arg = _redact(arg)[:min(MAX_EVENT_TEXT, remaining)]
        safe.append(arg)
        remaining -= len(arg)
        if remaining <= 0:
            safe[-1] += '…'
            break
        if _SECRET_ARG.match(arg):
            redact_next = True
    if len(argv) > MAX_COMMAND_ARGS:
        safe.append('[additional arguments trimmed]')
    return tuple(safe)


def _event_text(event: ExecutionEvent) -> str:
    stamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(event.timestamp))
    heading = ' '.join(part for part in (stamp, event.stage, event.kind) if part)
    lines = [f'[{heading}] {event.message}' if event.message else f'[{heading}]']
    if event.command:
        lines.append('Command: ' + shlex.join(_redact(arg) for arg in event.command))
    if event.exit_code is not None:
        lines.append(f'Exit code: {event.exit_code}')
    if event.stdout:
        lines.append('stdout:\n' + _redact(event.stdout))
    if event.stderr:
        lines.append('stderr:\n' + _redact(event.stderr))
    if event.current is not None or event.total is not None:
        lines.append(f'Progress: {event.current or 0} / {event.total or 0}')
    return '\n'.join(lines)


class EventBuffer:
    """Bounded, title-scoped in-memory event history for the current/most recent run."""

    def __init__(self):
        self._events = deque()
        self._chars = 0
        self.trimmed = 0

    def append(self, event: ExecutionEvent) -> None:
        if event.stdout or event.stderr or event.message or event.command:
            event = ExecutionEvent(event.run_id, event.title_id, event.stage, event.kind, event.timestamp,
                                   _redact(event.message)[:MAX_EVENT_TEXT],
                                   tuple(arg[:MAX_EVENT_TEXT] for arg in _redact_argv(event.command)),
                                   _redact(event.stdout)[:MAX_EVENT_TEXT], _redact(event.stderr)[:MAX_EVENT_TEXT],
                                   event.exit_code, event.current, event.total)
        rendered = _event_text(event)
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
        self.output.setPlainText(text)
        cursor = self.output.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.output.setTextCursor(cursor)

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
