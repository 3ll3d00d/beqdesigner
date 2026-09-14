'''
model/review.py: the beqd-side of design/candidate-review-plan.md phase 3 --
a standalone dialog (ReviewQueueDialog) for working through a
pipeline.review queue (produced by an unattended pipeline.review.batch_design()
run) and picking a candidate per title, fast, before anything publishes.

Deliberately independent of whatever's loaded in the main window's signal
table -- this dialog reads/writes its own queue directory via
pipeline.review, which is Qt-free; this module is the only place that
wires it to Qt widgets.
'''
import logging
import os

import qtawesome as qta
from qtpy.QtCore import QAbstractTableModel, QModelIndex, QObject, QRunnable, Qt, QThreadPool, Signal
from qtpy.QtGui import QKeySequence, QShortcut
from qtpy.QtWidgets import QDialog, QFileDialog, QMessageBox, QStatusBar, QTableWidgetItem

from model.codec import filter_from_json, xydata_from_json
from model.magnitude import MagnitudeModel
from model.preferences import DESIGNER_QUEUE_DIR
from pipeline.config import AnalysisConfig
from pipeline.publish.git import RepoTarget
from pipeline.review import read_queue, update_entry, publish_reviewed_queue
from ui.review import Ui_reviewQueueDialog

logger = logging.getLogger('review')

_STATUS_COLUMN_HEADERS = ['Title', 'Status', 'Confidence', 'Method / Decline Reason']


class _QueueTableModel(QAbstractTableModel):
    '''
    A thin, read-only view over pipeline.review.read_queue()'s result --
    no editing happens through this model, only through the dialog's
    accept/skip/reject actions (which write via pipeline.review.update_entry
    and then reload).
    '''

    def __init__(self, parent=None):
        super().__init__(parent)
        self.entries = []

    def set_entries(self, entries):
        self.beginResetModel()
        self.entries = entries
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.entries)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(_STATUS_COLUMN_HEADERS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return _STATUS_COLUMN_HEADERS[section]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        entry = self.entries[index.row()]
        col = index.column()
        if col == 0:
            return entry.meta.get('title', entry.id)
        elif col == 1:
            return entry.status
        elif col == 2:
            return f"{entry.candidates[0].confidence:.2f}" if entry.candidates else ''
        elif col == 3:
            return entry.candidates[0].method if entry.candidates else (entry.decline_reason or '')
        return None


class _PublishSignals(QObject):
    finished = Signal(list)
    errored = Signal(str)


class _PublishJob(QRunnable):
    ''' Runs publish_reviewed_queue() off the UI thread -- same QThreadPool pattern as model/batch.py. '''

    def __init__(self, queue_dir, xml_repo):
        super().__init__()
        self.signals = _PublishSignals()
        self.__queue_dir = queue_dir
        self.__xml_repo = xml_repo

    def run(self):
        try:
            results = publish_reviewed_queue(self.__queue_dir, self.__xml_repo, config=AnalysisConfig())
            self.signals.finished.emit(results)
        except Exception as e:
            logger.exception('Publish failed')
            self.signals.errored.emit(str(e))


class ReviewQueueDialog(QDialog, Ui_reviewQueueDialog):

    def __init__(self, parent, preferences):
        super().__init__(parent)
        self.setupUi(self)
        self.__preferences = preferences
        self.__queue_dir = None
        self.__selected_candidate_index = 0
        self.__table_model = _QueueTableModel(self)
        self.queueTable.setModel(self.__table_model)
        self.statusBar = QStatusBar()
        self.mainLayout.addWidget(self.statusBar)

        self.browseQueueDirButton.setIcon(qta.icon('fa5s.folder-open'))
        self.browseQueueDirButton.clicked.connect(self.__browse_queue_dir)
        self.refreshButton.clicked.connect(self.__reload_queue)
        self.queueTable.selectionModel().selectionChanged.connect(self.__on_row_selected)
        self.candidateList.currentRowChanged.connect(self.__on_candidate_picked)
        self.acceptButton.clicked.connect(self.__accept_current)
        self.skipButton.clicked.connect(self.__skip_current)
        self.rejectButton.clicked.connect(self.__reject_current)
        self.publishButton.clicked.connect(self.__publish_accepted)

        self.__magnitude_model = MagnitudeModel('review', self.previewChart, preferences, self.__get_chart_data,
                                                 'Filter', fill_primary=True)
        self.__install_shortcuts()
        self.__update_action_buttons()

        default_queue_dir = self.__preferences.get(DESIGNER_QUEUE_DIR)
        if default_queue_dir and os.path.isdir(default_queue_dir):
            self.load_queue_dir(default_queue_dir)

    def reject(self):
        '''
        QDialog's default Escape-key behaviour calls reject(), which hides this dialog -- fine for a modal
        popup, but this one is a persistent workspace (embedded as model/batch.py's Review tab, or shown
        non-modally standalone), so hiding it leaves an apparently-blank tab/window rather than closing
        anything meaningful. No-op; close via the window's own controls instead.
        '''
        pass

    def __install_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key.Key_Return), self, activated=self.__accept_current)
        QShortcut(QKeySequence(Qt.Key.Key_Enter), self, activated=self.__accept_current)
        QShortcut(QKeySequence('A'), self, activated=self.__accept_current)
        QShortcut(QKeySequence('S'), self, activated=self.__skip_current)
        QShortcut(QKeySequence('R'), self, activated=self.__reject_current)
        for i in range(1, 10):
            QShortcut(QKeySequence(str(i)), self, activated=lambda idx=i - 1: self.__pick_candidate(idx))

    # --- queue loading ----------------------------------------------------

    def __browse_queue_dir(self):
        dialog = QFileDialog(parent=self)
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOptions(QFileDialog.Option.ShowDirsOnly)
        dialog.setWindowTitle('Select Queue Directory')
        if dialog.exec():
            selected = dialog.selectedFiles()
            if len(selected) > 0:
                self.load_queue_dir(selected[0])

    def load_queue_dir(self, queue_dir):
        '''
        Points this dialog at a queue directory and (re)loads it -- also usable from outside (tests, app.py).
        Remembers queue_dir as the DESIGNER_QUEUE_DIR default for next time.
        '''
        self.__queue_dir = queue_dir
        self.queueDirEdit.setText(queue_dir)
        self.__preferences.set(DESIGNER_QUEUE_DIR, queue_dir)
        self.__reload_queue()

    def __reload_queue(self):
        if self.__queue_dir is None:
            return
        entries = read_queue(self.__queue_dir)
        self.__table_model.set_entries(entries)
        if entries:
            self.queueTable.selectRow(0)
        else:
            self.__on_row_selected()
        self.statusBar.showMessage(
            f"{len(entries)} entries, {sum(1 for e in entries if e.status == 'pending')} pending")

    # --- selection / preview -----------------------------------------------

    def __current_entry(self):
        selection = self.queueTable.selectionModel()
        if selection is None or not selection.hasSelection():
            return None
        return self.__table_model.entries[selection.selectedRows()[0].row()]

    def __on_row_selected(self, *_):
        entry = self.__current_entry()
        self.__selected_candidate_index = 0
        self.candidateList.blockSignals(True)
        self.candidateList.clear()
        if entry is not None:
            self.titleLabel.setText(entry.meta.get('title', entry.id))
            self.declineReasonLabel.setVisible(bool(entry.decline_reason))
            if entry.decline_reason:
                self.declineReasonLabel.setText(
                    f"Declined: {entry.decline_reason} -- {entry.decline_message or ''}")
            for i, candidate in enumerate(entry.candidates):
                self.candidateList.addItem(
                    f"{i}: confidence={candidate.confidence:.2f} method={candidate.method} "
                    f"mv_adjust_db={candidate.mv_adjust_db:+.1f} gain_reduction_db="
                    f"{candidate.gain_reduction_db if candidate.gain_reduction_db is not None else 'n/a'}")
            if entry.status == 'accepted' and entry.chosen_candidate_index is not None:
                self.__selected_candidate_index = entry.chosen_candidate_index
            if entry.candidates:
                self.candidateList.setCurrentRow(self.__selected_candidate_index)
        else:
            self.titleLabel.setText('')
            self.declineReasonLabel.setVisible(False)
        self.candidateList.blockSignals(False)
        self.__refresh_commentary_table()
        self.__update_action_buttons()
        self.__magnitude_model.redraw()

    def __on_candidate_picked(self, row):
        if row < 0:
            return
        self.__selected_candidate_index = row
        self.__refresh_commentary_table()
        self.__magnitude_model.redraw()

    def __pick_candidate(self, index):
        if 0 <= index < self.candidateList.count():
            self.candidateList.setCurrentRow(index)

    def __refresh_commentary_table(self):
        entry = self.__current_entry()
        commentary = {}
        if entry is not None and entry.candidates and 0 <= self.__selected_candidate_index < len(entry.candidates):
            commentary = entry.candidates[self.__selected_candidate_index].commentary or {}
        self.commentaryTable.setRowCount(len(commentary))
        for row, (key, value) in enumerate(commentary.items()):
            self.commentaryTable.setItem(row, 0, QTableWidgetItem(str(key)))
            self.commentaryTable.setItem(row, 1, QTableWidgetItem(str(value)))

    def __get_chart_data(self, reference=None):
        entry = self.__current_entry()
        if entry is None or not entry.candidates:
            return []
        unfiltered = xydata_from_json(entry.curve)
        unfiltered.colour = 'grey'
        result = [unfiltered]
        idx = self.__selected_candidate_index
        if 0 <= idx < len(entry.candidates):
            complete_filter = filter_from_json(entry.candidates[idx].filters)
            filter_mag = complete_filter.get_transfer_function().get_magnitude()
            filtered = unfiltered.filter(filter_mag)
            filtered.colour = 'red'
            result.append(filtered)
        return result

    def __update_action_buttons(self):
        entry = self.__current_entry()
        has_candidates = entry is not None and len(entry.candidates) > 0
        self.acceptButton.setEnabled(has_candidates)
        self.skipButton.setEnabled(entry is not None)
        self.rejectButton.setEnabled(entry is not None)

    # --- triage actions ------------------------------------------------------

    def __accept_current(self):
        entry = self.__current_entry()
        if entry is None or not entry.candidates:
            return
        self.__apply_decision(entry.id, status='accepted', chosen_candidate_index=self.__selected_candidate_index)

    def __skip_current(self):
        entry = self.__current_entry()
        if entry is None:
            return
        self.__apply_decision(entry.id, status='skipped')

    def __reject_current(self):
        entry = self.__current_entry()
        if entry is None:
            return
        self.__apply_decision(entry.id, status='rejected')

    def __apply_decision(self, entry_id, **fields):
        update_entry(self.__queue_dir, entry_id, **fields)
        self.__advance_to_next_pending(entry_id)

    def __advance_to_next_pending(self, processed_id):
        '''
        Selects the next *pending* row after the one just processed, in the
        table's current order, wrapping around -- never jumps back to row 0
        regardless of where the processed entry was (same principle as
        app.py's deleteSignal() fix: advance from where you were, don't
        reset to the top).
        '''
        old_ids = [e.id for e in self.__table_model.entries]
        try:
            start = old_ids.index(processed_id)
        except ValueError:
            start = -1
        probe_order = old_ids[start + 1:] + old_ids[:start + 1]

        entries = read_queue(self.__queue_dir)
        self.__table_model.set_entries(entries)
        by_id = {e.id: (i, e) for i, e in enumerate(entries)}
        for candidate_id in probe_order:
            hit = by_id.get(candidate_id)
            if hit is not None and hit[1].status == 'pending':
                self.queueTable.selectRow(hit[0])
                self.statusBar.showMessage(
                    f"{sum(1 for e in entries if e.status == 'pending')} entries still pending")
                return
        self.statusBar.showMessage('No more pending entries')
        self.__on_row_selected()

    # --- publish ------------------------------------------------------------

    def __publish_accepted(self):
        if self.__queue_dir is None:
            return
        dialog = QFileDialog(parent=self)
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOptions(QFileDialog.Option.ShowDirsOnly)
        dialog.setWindowTitle('Select XML repo (local clone)')
        if not dialog.exec():
            return
        selected = dialog.selectedFiles()
        if not selected:
            return
        xml_repo = RepoTarget(local_path=selected[0])
        self.publishButton.setEnabled(False)
        self.publishProgress.setRange(0, 0)  # busy indicator -- publish_reviewed_queue reports no per-item progress
        job = _PublishJob(self.__queue_dir, xml_repo)
        job.signals.finished.connect(self.__on_publish_finished)
        job.signals.errored.connect(self.__on_publish_errored)
        QThreadPool.globalInstance().start(job)

    def __on_publish_finished(self, results):
        self.publishProgress.setRange(0, 1)
        self.publishProgress.setValue(1)
        self.publishButton.setEnabled(True)
        self.statusBar.showMessage(f"Published {len(results)} title(s)")
        self.__reload_queue()

    def __on_publish_errored(self, message):
        self.publishProgress.setRange(0, 1)
        self.publishButton.setEnabled(True)
        QMessageBox.critical(self, 'Publish failed', message)
