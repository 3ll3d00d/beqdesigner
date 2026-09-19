'''Qt wrapper for the headless library-run and explicit-sync functions.'''
import logging

from qtpy.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from qtpy.QtWidgets import QDialog, QMessageBox

from model.preferences import DESIGNER_DEFAULT, DESIGNER_QUEUE_DIR, JRIVER_MCWS_CONNECTIONS, LIBRARY_IMAGES_REPO, \
    LIBRARY_JRIVER_BROWSE_NODE, LIBRARY_WORK_DIR, LIBRARY_XML_REPO, TMDB_API_KEY
from pipeline.config import AnalysisConfig
from pipeline.library.jriver import JRiverLibrarySource
from pipeline.library.run import LibraryRunConfig, run_library
from pipeline.library.sync import sync_library
from pipeline.publish.git import RepoTarget
from pipeline.review import describe_publish_error, split_publish_results
from ui.library_sync import Ui_librarySyncDialog

logger = logging.getLogger('library_sync')


class _RunSignals(QObject):
    item_done = Signal(str)
    finished = Signal(object)
    errored = Signal(str)


class _RunJob(QRunnable):
    def __init__(self, source, config):
        super().__init__()
        self.signals = _RunSignals()
        self.__source = source
        self.__config = config

    def run(self):
        try:
            report = run_library(self.__source, self.__config, self.signals.item_done.emit)
            self.signals.finished.emit(report)
        except Exception as error:
            logger.exception('Library run failed')
            self.signals.errored.emit(str(error))


class _SyncSignals(QObject):
    finished = Signal(object)
    errored = Signal(str)


class _SyncJob(QRunnable):
    def __init__(self, queue_dir, xml_repo, images_repo, work_dir):
        super().__init__()
        self.signals = _SyncSignals()
        self.__queue_dir = queue_dir
        self.__xml_repo = xml_repo
        self.__images_repo = images_repo
        self.__work_dir = work_dir

    def run(self):
        try:
            results = sync_library(self.__queue_dir, RepoTarget(self.__xml_repo),
                                   images_repo=RepoTarget(self.__images_repo) if self.__images_repo else None,
                                   config=AnalysisConfig(), work_dir=self.__work_dir)
            self.signals.finished.emit(results)
        except Exception as error:
            logger.exception('Library sync failed')
            self.signals.errored.emit(str(error))


class LibrarySyncDialog(QDialog, Ui_librarySyncDialog):
    '''Configure a JRiver browse source, run it in the thread pool, then review or explicitly sync.'''

    def __init__(self, parent, preferences):
        super().__init__(parent)
        self.setupUi(self)
        self.__preferences = preferences
        self.__review = None
        self.__active_job = None
        self.__load_preferences()
        self.__load_designers()
        self.runButton.clicked.connect(self.__run)
        self.syncButton.clicked.connect(self.__sync)
        self.progressBar.setVisible(False)

    def __load_preferences(self):
        self.workDirEdit.setText(self.__preferences.get(LIBRARY_WORK_DIR))
        self.queueDirEdit.setText(self.__preferences.get(DESIGNER_QUEUE_DIR))
        self.xmlRepoEdit.setText(self.__preferences.get(LIBRARY_XML_REPO))
        self.imagesRepoEdit.setText(self.__preferences.get(LIBRARY_IMAGES_REPO))
        self.browseNodeSpin.setValue(self.__preferences.get(LIBRARY_JRIVER_BROWSE_NODE))
        connections = self.__preferences.get(JRIVER_MCWS_CONNECTIONS)
        if connections:
            endpoint, saved = next(iter(connections.items()))
            host, separator, port = endpoint.rpartition(':')
            self.serverEdit.setText(host if separator else endpoint)
            if separator and port.isdigit():
                self.portSpin.setValue(int(port))
            auth, secure = saved if len(saved) == 2 else (None, False)
            if auth:
                self.usernameEdit.setText(auth[0])
                self.passwordEdit.setText(auth[1])
            self.sslCheck.setChecked(bool(secure))

    def __load_designers(self):
        from pipeline.designer.registry import registered_designers
        self.designerCombo.addItems(registered_designers())
        default = self.__preferences.get(DESIGNER_DEFAULT)
        index = self.designerCombo.findText(default)
        if index >= 0:
            self.designerCombo.setCurrentIndex(index)

    def __persist_preferences(self):
        self.__preferences.set(LIBRARY_WORK_DIR, self.workDirEdit.text())
        self.__preferences.set(DESIGNER_QUEUE_DIR, self.queueDirEdit.text())
        self.__preferences.set(LIBRARY_XML_REPO, self.xmlRepoEdit.text())
        self.__preferences.set(LIBRARY_IMAGES_REPO, self.imagesRepoEdit.text())
        self.__preferences.set(LIBRARY_JRIVER_BROWSE_NODE, self.browseNodeSpin.value())

    def __source_and_config(self):
        host = self.serverEdit.text().strip()
        queue_dir = self.queueDirEdit.text().strip()
        work_dir = self.workDirEdit.text().strip()
        designer = self.designerCombo.currentText()
        if not host or not queue_dir or not work_dir or not designer:
            raise ValueError('Server, work directory, review queue, and designer are required')
        source = JRiverLibrarySource(
            host, self.portSpin.value(), self.browseNodeSpin.value(),
            username=self.usernameEdit.text().strip() or None,
            password=self.passwordEdit.text() or None, ssl=self.sslCheck.isChecked(),
        )
        config = LibraryRunConfig(work_dir=work_dir, queue_dir=queue_dir, designer=designer,
                                  keep_multichannel=self.keepMultichannelCheck.isChecked(),
                                  tmdb_api_key=self.__preferences.get(TMDB_API_KEY) or None)
        return source, config

    def __set_busy(self, busy, message=''):
        self.runButton.setEnabled(not busy)
        self.syncButton.setEnabled(not busy)
        self.progressBar.setVisible(busy)
        self.progressBar.setRange(0, 0 if busy else 100)
        self.statusLabel.setText(message)

    def __run(self):
        try:
            source, config = self.__source_and_config()
        except ValueError as error:
            QMessageBox.warning(self, 'Library Sync', str(error))
            return
        self.__persist_preferences()
        job = _RunJob(source, config)
        self.__active_job = job
        self.__set_busy(True, 'Listing and processing library items...')
        job.signals.item_done.connect(lambda item_id: self.statusLabel.setText(f'Processed {item_id}'))
        job.signals.finished.connect(self.__run_finished)
        job.signals.errored.connect(self.__job_failed)
        QThreadPool.globalInstance().start(job)

    def __run_finished(self, report):
        self.__set_busy(False, f'Designed {len(report.designed)}, cached {len(report.design_cached)}, '
                               f'failed {len(report.failed)}')
        from model.review import ReviewQueueDialog
        if self.__review is None:
            self.__review = ReviewQueueDialog(self, self.__preferences)
            self.__review.setWindowFlags(Qt.WindowType.Widget)
            self.reviewLayout.addWidget(self.__review)
        self.__review.load_queue_dir(self.queueDirEdit.text())
        self.mainTabs.setCurrentWidget(self.reviewTab)

    def __sync(self):
        queue_dir = self.queueDirEdit.text().strip()
        xml_repo = self.xmlRepoEdit.text().strip()
        if not queue_dir or not xml_repo:
            QMessageBox.warning(self, 'Library Sync', 'Review queue and XML repository are required')
            return
        self.__persist_preferences()
        job = _SyncJob(queue_dir, xml_repo, self.imagesRepoEdit.text().strip(), self.workDirEdit.text().strip())
        self.__active_job = job
        self.__set_busy(True, 'Publishing accepted entries...')
        job.signals.finished.connect(self.__sync_finished)
        job.signals.errored.connect(self.__job_failed)
        QThreadPool.globalInstance().start(job)

    def __sync_finished(self, results):
        published, needs_attention = split_publish_results(results)
        message = f'Published {len(published)} accepted entries'
        if needs_attention:
            message += f', {len(needs_attention)} need attention'
        self.__set_busy(False, message)
        if self.__review is not None:
            self.__review.load_queue_dir(self.queueDirEdit.text())  # show the new 'published' statuses
        if needs_attention:
            QMessageBox.warning(self, 'Library Sync', 'These entries were not published:\n\n'
                                + '\n'.join(describe_publish_error(r) for r in needs_attention))

    def __job_failed(self, message):
        self.__set_busy(False, message)
        QMessageBox.critical(self, 'Library Sync', message)
