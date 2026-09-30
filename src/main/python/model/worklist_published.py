"""Published BEQs for a library title, with the catalogue's images in one pane."""
import hashlib
import json
import logging
import os
import tempfile

import requests
from qtpy.QtCore import QObject, QRunnable, QThreadPool, QUrl, Signal
from qtpy.QtGui import QDesktopServices, QPixmap
from qtpy.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget

from model.catalogue import CatalogueEntry, DatabaseDownloader, load_catalogue
from model.execution_events import redact_text
from model.preferences import BEQ_DOWNLOAD_DIR
from model.worklist_artwork import ArtworkError, cache_path, check_local_image, download_artwork
from model.worklist_catalogue import matching_entries
from model.worklist_image import ScaledImage

logger = logging.getLogger('worklist')
MAX_DATABASE_BYTES = 100 * 1024 * 1024


def published_catalogue(cache_dir, refresh=False):
    """Share Browse Catalogue's database cache; validate downloads before replacing it atomically."""
    path = os.path.join(cache_dir, 'database.json')
    if os.path.isfile(path) and not refresh:
        return load_catalogue(path), 'Cached catalogue; Refresh catalogue checks for updates.'
    try:
        response = requests.get(DatabaseDownloader.DATABASE_URL, timeout=20, stream=True)
        try:
            response.raise_for_status()
            chunks, size = [], 0
            for chunk in response.iter_content(65536):
                size += len(chunk)
                if size > MAX_DATABASE_BYTES:
                    raise ValueError('the catalogue download is too large')
                chunks.append(chunk)
            records = json.loads(b''.join(chunks))
        finally:
            response.close()
        if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
            raise ValueError('the catalogue is not a list of entries')
        entries = [CatalogueEntry(str(i), record) for i, record in enumerate(records)]
        os.makedirs(cache_dir, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix='.catalogue-', suffix='.json', dir=cache_dir)
        try:
            with os.fdopen(handle, 'w', encoding='utf-8') as target:
                json.dump(records, target)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return entries, 'Catalogue updated.'
    except Exception as error:
        if os.path.isfile(path):
            return load_catalogue(path), f'Could not refresh catalogue ({redact_text(str(error))}); showing cached entries.'
        raise


def published_image(cache_dir, url):
    """Cache by complete URL, so equal basenames from different authors cannot collide."""
    root = os.path.join(cache_dir, '.published')
    key = hashlib.sha256(url.encode('utf-8')).hexdigest()
    for extension in ('.png', '.jpg'):
        path = cache_path(root, key, extension)
        if os.path.isfile(path):
            try:
                check_local_image(path)
            except ArtworkError:
                continue
            else:
                with open(path, 'rb') as source:
                    return source.read()
    path = download_artwork(url, root, key)
    with open(path, 'rb') as source:
        return source.read()


class _LookupSignals(QObject):
    loaded = Signal(object, str)
    image = Signal(str, bytes)
    failed = Signal(str, str)


class _LookupJob(QRunnable):
    def __init__(self, cache_dir, url='', refresh=False):
        super().__init__()
        self.signals = _LookupSignals()
        self.cache_dir, self.url, self.refresh = cache_dir, url, refresh

    def run(self):
        try:
            if self.url:
                self.signals.image.emit(self.url, published_image(self.cache_dir, self.url))
            else:
                entries, note = published_catalogue(self.cache_dir, self.refresh)
                self.signals.loaded.emit(entries, note)
        except Exception as error:
            logger.exception('Could not load published BEQs')
            self.signals.failed.emit(self.url, redact_text(f'{type(error).__name__}: {error}'))


class PublishedPanel(QWidget):
    """Author/entry and image selectors above one image, with lazy lookup and stale-image protection."""
    def __init__(self, preferences, parent=None):
        super().__init__(parent)
        self._preferences = preferences
        self._target = self._entries = self._catalogue_job = self._image_job = None
        self._wanted_url = ''
        self._note = ''
        self.png = b''
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.otherTracks = QCheckBox('Include other tracks / editions', self)
        self.otherTracks.toggled.connect(self._populate)
        controls.addWidget(self.otherTracks)
        controls.addStretch()
        self.refreshButton = QPushButton('Refresh catalogue', self)
        self.refreshButton.setAutoDefault(False)
        self.refreshButton.clicked.connect(lambda: self._load(refresh=True))
        controls.addWidget(self.refreshButton)
        layout.addLayout(controls)
        self.statusLabel = QLabel(self)
        self.statusLabel.setWordWrap(True)
        layout.addWidget(self.statusLabel)
        selectors = QHBoxLayout()
        self.entryChoice = QComboBox(self)
        self.entryChoice.setMinimumContentsLength(15)
        self.entryChoice.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.entryChoice.currentIndexChanged.connect(self._select_entry)
        selectors.addWidget(self.entryChoice, 2)
        self.imageChoice = QComboBox(self)
        self.imageChoice.currentIndexChanged.connect(self._select_image)
        self.imageChoice.activated.connect(self._select_image)
        selectors.addWidget(self.imageChoice, 1)
        self.catalogueButton = QPushButton('Open catalogue page', self)
        self.catalogueButton.setAutoDefault(False)
        self.catalogueButton.clicked.connect(self._open_catalogue)
        selectors.addWidget(self.catalogueButton)
        layout.addLayout(selectors)
        self.details = QPlainTextEdit(self)
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(130)
        layout.addWidget(self.details)
        self.image = ScaledImage(self)
        layout.addWidget(self.image, 1)
        self.set_target(None)

    def set_target(self, target):
        if target == self._target and target is not None:
            return
        self._target = target
        self._populate()
        if self.isVisible():
            self._load()

    def _load(self, refresh=False):
        if self._target is None or not self._target.usable:
            return
        if self._catalogue_job is not None or (self._entries is not None and not refresh):
            return
        self.statusLabel.setText('Looking for published BEQs…')
        self.refreshButton.setEnabled(False)
        self._catalogue_job = _LookupJob(self._preferences.get(BEQ_DOWNLOAD_DIR), refresh=refresh)
        self._catalogue_job.signals.loaded.connect(self._loaded)
        self._catalogue_job.signals.failed.connect(self._failed)
        QThreadPool.globalInstance().start(self._catalogue_job)

    def _loaded(self, entries, note):
        self._catalogue_job = None
        self._entries, self._note = entries, note
        self.refreshButton.setEnabled(True)
        self._populate()

    def _populate(self, *_):
        previous = self.entryChoice.currentData()
        matches = matching_entries(self._target, self._entries or [], self.otherTracks.isChecked())
        self.entryChoice.blockSignals(True)
        self.entryChoice.clear()
        chosen = 0
        for i, entry in enumerate(matches):
            audio = ', '.join(entry.audio_types) or 'audio unspecified'
            label = f'{entry.author or "Unknown author"} — {entry.formatted_title} · {audio}'
            if entry.edition:
                label += f' · {entry.edition}'
            self.entryChoice.addItem(label, entry)
            if previous is not None and entry.idx == previous.idx:
                chosen = i
        self.entryChoice.setCurrentIndex(chosen if matches else -1)
        self.entryChoice.blockSignals(False)
        if self._target is None:
            status = 'Open a title to look for published BEQs.'
        elif not self._target.usable:
            status = 'Set a TMDB ID, or title and year, in Metadata to find published BEQs.'
        elif self._entries is None:
            status = 'Open this tab to look for published BEQs.'
        else:
            authors = len({entry.author for entry in matches})
            status = f'{len(matches)} published BEQ(s) by {authors} author(s). '
            if not matches and not self.otherTracks.isChecked():
                status += 'Try Include other tracks / editions. '
            status += 'Matches use catalogue metadata; check edition, language and source. ' + self._note
        self.statusLabel.setText(status)
        self._select_entry()

    def _select_entry(self, *_):
        entry = self.entryChoice.currentData()
        self.imageChoice.blockSignals(True)
        self.imageChoice.clear()
        self.details.clear()
        self.catalogueButton.setEnabled(bool(entry and entry.beqc_url.startswith(('https://', 'http://'))))
        if entry is not None:
            lines = [f'{entry.formatted_title} — {entry.author}', f'Audio: {", ".join(entry.audio_types)}',
                     f'Edition: {entry.edition or "unspecified"} · Language: {entry.language or "unspecified"} · Source: {entry.source or "unspecified"}',
                     f'MV adjustment: {entry.mv_adjust:+g} dB']
            if entry.note:
                lines.append(f'Note: {entry.note}')
            if entry.warning:
                warning = '; '.join(entry.warning) if isinstance(entry.warning, list) else entry.warning
                lines.append(f'Warning: {warning}')
            self.details.setPlainText('\n'.join(lines))
            images = [url for url in entry.images if isinstance(url, str) and url.startswith(('https://', 'http://'))]
            for i, url in enumerate(images):
                self.imageChoice.addItem(f'Image {i + 1}' + (' — heatmap' if 'heatmap' in url.casefold() else ''), url)
        self.imageChoice.blockSignals(False)
        self._select_image()

    def _select_image(self, *_):
        self._wanted_url = self.imageChoice.currentData() or ''
        self.png = b''
        self.image.set_image(QPixmap())
        if not self._wanted_url:
            self.image.setText('No published images for this entry.' if self.entryChoice.currentData() else '')
        self._start_image()

    def _start_image(self):
        if self._image_job is not None or self.png or not self._wanted_url or not self.isVisible():
            return
        self.image.setText('Loading published image…')
        self._image_job = _LookupJob(self._preferences.get(BEQ_DOWNLOAD_DIR), self._wanted_url)
        self._image_job.signals.image.connect(self._image_loaded)
        self._image_job.signals.failed.connect(self._failed)
        QThreadPool.globalInstance().start(self._image_job)

    def _image_loaded(self, url, data):
        self._image_job = None
        if url == self._wanted_url:
            pixmap = QPixmap()
            if pixmap.loadFromData(data):
                self.png = data
                self.image.set_image(pixmap)
            else:
                self.image.setText('The published image could not be read.')
        else:
            self._start_image()

    def _failed(self, url, message):
        if url:
            self._image_job = None
            if url == self._wanted_url:
                self.image.setText(f'Could not load published image: {message}. Select it again to retry.')
            else:
                self._start_image()
        else:
            self._catalogue_job = None
            self.refreshButton.setEnabled(True)
            if self._target is not None:
                self.statusLabel.setText(f'Could not load catalogue: {message}. Try Refresh catalogue.')

    def _open_catalogue(self):
        entry = self.entryChoice.currentData()
        if entry and entry.beqc_url.startswith(('https://', 'http://')):
            QDesktopServices.openUrl(QUrl(entry.beqc_url))

    def showEvent(self, event):
        super().showEvent(event)
        self._load()
        self._start_image()
