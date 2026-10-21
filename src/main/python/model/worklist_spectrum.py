"""Precomputed title-page spectrum comparison, rendered exactly like the publication heatmap."""
import logging
import time

from qtpy.QtCore import QObject, QRunnable, QThreadPool, Signal
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from model.worklist_image import ScaledImage
from model.execution_events import redact_text
from pipeline.library.spectrum import ComparisonRequest, comparison_request, render_comparison  # noqa: F401 (callers, tests)

logger = logging.getLogger('worklist')


class _ComparisonSignals(QObject):
    finished = Signal(object, bytes, str)
    failed = Signal(object, str)


class _ComparisonJob(QRunnable):
    def __init__(self, request):
        super().__init__()
        self.request = request
        self.signals = _ComparisonSignals()

    def run(self):
        started = time.perf_counter()
        try:
            png, note = render_comparison(self.request)
            logger.info("Spectrum comparison for %s generated in %.2f seconds",
                        self.request.title_id, time.perf_counter() - started)
            self.signals.finished.emit(self.request, png, note)
        except Exception as error:
            logger.exception('Could not render spectrum comparison for %s', self.request.title_id)
            self.signals.failed.emit(self.request, redact_text(f'{type(error).__name__}: {error}'))


class SpectrumPanel(QWidget):
    """Precompute on one worker, retaining up to four images; changed requests coalesce."""
    refresh_requested = Signal()
    ready = Signal(bytes)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._request = self._cached = self._job = None
        self._images = []
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
            self.statusLabel.setText('Generating spectrum comparison…')
        if force:
            self._images = [item for item in self._images if item[0] != request]
        else:
            if request == self._cached:
                return
            for item in self._images:
                if item[0] == request:
                    self._images.remove(item)
                    self._images.append(item)
                    self._display(*item)
                    return
        self._start()

    def _start(self):
        if self._job is not None or self._request is None or self._request == self._cached:
            return
        self.statusLabel.setText('Generating spectrum comparison…')
        self.refreshButton.setEnabled(False)
        self._job = _ComparisonJob(self._request)
        self._job.signals.finished.connect(self._finished)
        self._job.signals.failed.connect(self._failed)
        QThreadPool.globalInstance().start(self._job)

    def _display(self, request, png, note):
        image = QPixmap()
        if not image.loadFromData(png, 'PNG'):
            self.statusLabel.setText('The spectrum comparison image could not be read. Try Refresh comparison.')
            return False
        self._cached = request
        self.png = png
        self.image.set_image(image)
        self.statusLabel.setText(note)
        self.ready.emit(png)
        return True

    def _finished(self, request, png, note):
        self._job = None
        self.refreshButton.setEnabled(True)
        if request == self._request:
            if self._display(request, png, note):
                self._images = [item for item in self._images if item[0] != request]
                self._images.append((request, png, note))
                self._images = self._images[-4:]
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
