"""Lazy title-page spectrum comparison, rendered exactly like the publication heatmap."""
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

from qtpy.QtCore import QObject, QRunnable, QThreadPool, Signal
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from model.worklist_image import ScaledImage
from model.codec import filter_from_json
from model.execution_events import redact_text
from pipeline.config import AnalysisConfig
from pipeline.orchestrate import Session
from pipeline.publish.heatmap import HeatmapSpec, heatmap_for, spec_from_preferences
from pipeline.publish.project import ProjectFilterConflict, preview_published_projects
from pipeline.review import project_paths

logger = logging.getLogger('worklist')


@dataclass(frozen=True)
class ComparisonRequest:
    title_id: str
    title: str
    mono_wav: str
    mono_project: str
    multichannel_project: Optional[str]
    filters: str
    config: AnalysisConfig
    spec: HeatmapSpec
    files: tuple


def _stamp(path):
    try:
        stat = os.stat(path) if path else None
        return (stat.st_mtime_ns, stat.st_size) if stat else None
    except OSError:
        return None


def comparison_request(work_dir, entry, picked, title, config, preferences) -> ComparisonRequest:
    if entry is None or not 0 <= picked < len(entry.offered):
        raise ValueError('No design to compare yet. Extract and design this title first.')
    if not work_dir:
        raise ValueError('No work directory is set. The comparison needs the extracted mono audio.')
    directory, mono, multichannel, _ = project_paths(work_dir, entry.id)
    wav = os.path.join(directory, 'mono.wav')
    if not os.path.isfile(wav):
        raise ValueError('No extracted mono audio is available for this title. Extract it again to view the comparison.')
    return ComparisonRequest(entry.id, title, wav, mono, multichannel,
                             json.dumps(entry.offered[picked].filters, sort_keys=True), config,
                             spec_from_preferences(preferences), tuple(_stamp(p) for p in (wav, mono, multichannel)))


def render_comparison(request: ComparisonRequest) -> tuple[bytes, str]:
    """Publication's loading, project resolution and renderer, without writing any projects or repositories."""
    candidate = filter_from_json(json.loads(request.filters))
    try:
        published = preview_published_projects(request.mono_project, request.multichannel_project, candidate)
    except ProjectFilterConflict as error:
        raise ValueError('The edited mono and multichannel projects disagree. Resolve their filters before comparing.') from error
    signal = Session(request.config).load(request.mono_wav)
    png = heatmap_for(signal, published.filter, request.spec, title=request.title)
    source = f'Saved {published.edited_side} project edits' if published.edited_side else 'Selected design'
    return png, f'{source}. Filtered on the left; unfiltered on the right. Both panes reach 40 Hz.'


class _ComparisonSignals(QObject):
    finished = Signal(object, bytes, str)
    failed = Signal(object, str)


class _ComparisonJob(QRunnable):
    def __init__(self, request):
        super().__init__()
        self.request = request
        self.signals = _ComparisonSignals()

    def run(self):
        try:
            png, note = render_comparison(self.request)
            self.signals.finished.emit(self.request, png, note)
        except Exception as error:
            logger.exception('Could not render spectrum comparison for %s', self.request.title_id)
            self.signals.failed.emit(self.request, redact_text(f'{type(error).__name__}: {error}'))


class SpectrumPanel(QWidget):
    """One worker and one cached image; changed titles/candidates coalesce and late results are discarded."""
    refresh_requested = Signal()
    ready = Signal(bytes)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._request = self._cached = self._job = None
        self.png = b''
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.statusLabel = QLabel(self)
        self.statusLabel.setWordWrap(True)
        controls.addWidget(self.statusLabel, 1)
        self.refreshButton = QPushButton('Refresh comparison', self)
        self.refreshButton.setAutoDefault(False)
        self.refreshButton.clicked.connect(self.refresh_requested.emit)
        controls.addWidget(self.refreshButton)
        layout.addLayout(controls)
        self.image = ScaledImage(self)
        layout.addWidget(self.image, 1)
        self.unavailable('No design to compare yet.')

    def unavailable(self, message):
        self._request = self._cached = None
        self.png = b''
        self.image.set_image(QPixmap())
        self.statusLabel.setText(message)
        self.refreshButton.setEnabled(self._job is None)

    def set_request(self, request, force=False):
        if force or request != self._request:
            self._request = request
            self._cached = None
            self.png = b''
            self.image.set_image(QPixmap())
            self.statusLabel.setText('Open this tab to generate the spectrum comparison.')
        self._start()

    def _start(self):
        if self._job is not None or self._request is None or self._request == self._cached or not self.isVisible():
            return
        self.statusLabel.setText('Generating spectrum comparison…')
        self.refreshButton.setEnabled(False)
        self._job = _ComparisonJob(self._request)
        self._job.signals.finished.connect(self._finished)
        self._job.signals.failed.connect(self._failed)
        QThreadPool.globalInstance().start(self._job)

    def _finished(self, request, png, note):
        self._job = None
        self.refreshButton.setEnabled(True)
        if request == self._request:
            image = QPixmap()
            if image.loadFromData(png, 'PNG'):
                self._cached = request
                self.png = png
                self.image.set_image(image)
                self.statusLabel.setText(note)
                self.ready.emit(png)
            else:
                self.statusLabel.setText('The spectrum comparison image could not be read. Try Refresh comparison.')
        else:
            self._start()

    def _failed(self, request, message):
        self._job = None
        self.refreshButton.setEnabled(True)
        if request == self._request:
            self.statusLabel.setText(f'Could not generate spectrum comparison: {message}. Try Refresh comparison.')
        else:
            self._start()

    def showEvent(self, event):
        super().showEvent(event)
        self._start()
