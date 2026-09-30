"""Persisted failure views, separate from the current run's event history."""
from qtpy.QtGui import QGuiApplication
from qtpy.QtWidgets import QLabel, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget

from model.execution_events import redact_text


def failed_stage(row) -> str:
    if row is None:
        return ''
    return 'extract' if row.extract_state == 'failed' else 'design' if row.design_state == 'failed' else ''


def retry_label(stages, count=None) -> str:
    stages = set(stages) - {''}
    if not stages:
        return f'Retry {count:,} failed' if count is not None else 'Retry failed'
    word = 'extraction' if stages == {'extract'} else 'design' if stages == {'design'} else 'extraction/design'
    return f'Retry failed {word}' if count is None else f'Retry {count:,} failed {word}'


def failure_text(stage: str, message: str, attempting: bool = False) -> str:
    if not message:
        return ''
    prefix = 'Previous indexed failure' if attempting else 'Indexed failure'
    return f'{prefix} ({stage}):\n{redact_text(message)}'


class FailurePanel(QWidget):
    """A full, selectable failure and an explicit current-attempt status on the title page."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.attemptLabel = QLabel(self)
        self.attemptLabel.setWordWrap(True)
        layout.addWidget(self.attemptLabel)
        self.output = QPlainTextEdit(self)
        self.output.setReadOnly(True)
        layout.addWidget(self.output)
        self.copyButton = QPushButton('Copy failure', self)
        self.copyButton.setAutoDefault(False)
        self.copyButton.clicked.connect(lambda: QGuiApplication.clipboard().setText(self.output.toPlainText()))
        layout.addWidget(self.copyButton)

    def show_failure(self, info: dict) -> None:
        attempting = info.get('attempting', False)
        self.attemptLabel.setText('Current attempt: ' + redact_text(info.get('attempt') or 'Updating result...') if attempting else
                                  'This failure is retained in the index until a refreshed result replaces it.')
        text = failure_text(info.get('stage', ''), info.get('message', ''), attempting)
        if text != self.output.toPlainText():
            self.output.setPlainText(text)
        self.copyButton.setEnabled(bool(text))
