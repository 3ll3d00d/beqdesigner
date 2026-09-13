'''
model/batch_design.py: design/http-designer-binding-plan.md phase 4 -- the
previously-missing GUI entry point for pipeline.review.batch_design(). Pick
source files, pick a registered designer (an in-process one, or one of
model/designers.py's configured HTTP endpoints), pick a queue directory,
run -- writing pending queue entries for model/review.py's ReviewQueueDialog
to work through.
'''
import logging
import os

from qtpy.QtCore import QObject, QRunnable, QThreadPool, Signal
from qtpy.QtWidgets import QDialog, QFileDialog, QMessageBox

from pipeline.config import AnalysisConfig
from pipeline.designer.registry import registered_designers
from pipeline.review import batch_design
from ui.batch_design import Ui_batchDesignDialog

logger = logging.getLogger('batch_design')


class _BatchDesignSignals(QObject):
    item_done = Signal(str)
    finished = Signal(list)
    errored = Signal(str)


class _BatchDesignJob(QRunnable):
    ''' Runs batch_design() off the UI thread -- same QThreadPool pattern as model/batch.py/model/review.py. '''

    def __init__(self, items, designer, queue_dir, work_dir):
        super().__init__()
        self.signals = _BatchDesignSignals()
        self.__items = items
        self.__designer = designer
        self.__queue_dir = queue_dir
        self.__work_dir = work_dir

    def run(self):
        try:
            written = batch_design(self.__items, self.__designer, self.__queue_dir, self.__work_dir,
                                    config=AnalysisConfig(), on_item_done=self.signals.item_done.emit)
            self.signals.finished.emit(written)
        except Exception as e:
            logger.exception('Batch design failed')
            self.signals.errored.emit(str(e))


class BatchDesignDialog(QDialog, Ui_batchDesignDialog):

    def __init__(self, parent, preferences):
        super().__init__(parent)
        self.setupUi(self)
        self.__preferences = preferences
        self.__queue_dir = None
        self.__work_dir = None

        self.designerCombo.addItems(registered_designers())
        self.addFilesButton.clicked.connect(self.__add_files)
        self.removeFileButton.clicked.connect(self.__remove_selected_file)
        self.browseQueueDirButton.clicked.connect(self.__browse_queue_dir)
        self.browseWorkDirButton.clicked.connect(self.__browse_work_dir)
        self.runButton.clicked.connect(self.__run)
        self.runProgress.setRange(0, 1)
        self.runProgress.setValue(0)

    def __add_files(self):
        dialog = QFileDialog(parent=self)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dialog.setWindowTitle('Select Source Files')
        if dialog.exec():
            for path in dialog.selectedFiles():
                self.filesList.addItem(path)

    def __remove_selected_file(self):
        for item in self.filesList.selectedItems():
            self.filesList.takeItem(self.filesList.row(item))

    def __browse_queue_dir(self):
        selected = self.__select_dir('Select Queue Directory')
        if selected:
            self.__queue_dir = selected
            self.queueDirEdit.setText(selected)

    def __browse_work_dir(self):
        selected = self.__select_dir('Select Work Directory')
        if selected:
            self.__work_dir = selected
            self.workDirEdit.setText(selected)

    def __select_dir(self, title):
        dialog = QFileDialog(parent=self)
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOptions(QFileDialog.Option.ShowDirsOnly)
        dialog.setWindowTitle(title)
        if dialog.exec():
            selected = dialog.selectedFiles()
            if selected:
                return selected[0]
        return None

    def __paths(self):
        return [self.filesList.item(i).text() for i in range(self.filesList.count())]

    def __run(self):
        paths = self.__paths()
        designer = self.designerCombo.currentText()
        if not paths:
            QMessageBox.critical(self, 'Cannot run', 'Add at least one source file')
            return
        if not designer:
            QMessageBox.critical(self, 'Cannot run', 'No designer registered -- configure one under Tools > Designers')
            return
        if not self.__queue_dir or not self.__work_dir:
            QMessageBox.critical(self, 'Cannot run', 'Select a queue directory and a work directory')
            return

        ids = [os.path.splitext(os.path.basename(p))[0] for p in paths]
        if len(ids) != len(set(ids)):
            QMessageBox.critical(self, 'Cannot run', 'Two or more source files share the same filename stem -- '
                                                     'rename one or remove the duplicate')
            return

        items = list(zip(ids, paths, [None] * len(paths)))
        self.runButton.setEnabled(False)
        self.runProgress.setRange(0, len(items))
        self.runProgress.setValue(0)
        job = _BatchDesignJob(items, designer, self.__queue_dir, self.__work_dir)
        job.signals.item_done.connect(self.__on_item_done)
        job.signals.finished.connect(self.__on_finished)
        job.signals.errored.connect(self.__on_errored)
        QThreadPool.globalInstance().start(job)

    def __on_item_done(self, entry_id):
        self.runProgress.setValue(self.runProgress.value() + 1)

    def __on_finished(self, written):
        self.runButton.setEnabled(True)
        answer = QMessageBox.question(self, 'Batch design complete',
                                      f"Wrote {len(written)} queue entr{'y' if len(written) == 1 else 'ies'} to "
                                      f"{self.__queue_dir}.\n\nOpen it for review now?")
        if answer == QMessageBox.StandardButton.Yes:
            from model.review import ReviewQueueDialog
            dialog = ReviewQueueDialog(self.parent(), self.__preferences)
            dialog.load_queue_dir(self.__queue_dir)
            dialog.show()
        self.accept()

    def __on_errored(self, message):
        self.runButton.setEnabled(True)
        QMessageBox.critical(self, 'Batch design failed', message)
