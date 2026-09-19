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
import requests
from qtpy.QtCore import QAbstractTableModel, QModelIndex, QObject, QRunnable, Qt, QThreadPool, Signal
from qtpy.QtGui import QKeySequence, QPixmap, QShortcut
from qtpy.QtWidgets import QDialog, QFileDialog, QLineEdit, QMessageBox, QPushButton, QStatusBar, QTableWidgetItem

from model.codec import filter_from_json, xydata_from_json
from model.magnitude import MagnitudeModel
from model.preferences import DESIGNER_QUEUE_DIR
from pipeline.config import AnalysisConfig
from pipeline.metadata import format_episodes, parse_episodes
from pipeline.publish.git import RepoTarget
from pipeline.review import describe_publish_error, publish_reviewed_queue, read_queue, split_publish_results, \
    update_entry
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
        self.__keep_candidate = None  # (entry id, candidate index) to restore when that entry is re-selected
        self.__table_model = _QueueTableModel(self)
        self.queueTable.setModel(self.__table_model)
        self.statusBar = QStatusBar()
        self.mainLayout.addWidget(self.statusBar)

        # Enter in a QLineEdit clicks a QDialog's default button, and every QPushButton is autoDefault: none of these
        # may fire from the keyboard (Enter on the table/candidate list is handled by the shortcuts below).
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)
            button.setDefault(False)

        self.browseQueueDirButton.setIcon(qta.icon('fa5s.folder-open'))
        self.browseQueueDirButton.clicked.connect(self.__browse_queue_dir)
        self.refreshButton.clicked.connect(lambda: self.__reload_queue(keep_current=True))
        self.queueTable.selectionModel().selectionChanged.connect(self.__on_row_selected)
        self.candidateList.currentRowChanged.connect(self.__on_candidate_picked)
        self.acceptButton.clicked.connect(self.__accept_current)
        self.skipButton.clicked.connect(self.__skip_current)
        self.rejectButton.clicked.connect(self.__reject_current)
        self.reopenButton.clicked.connect(self.__reopen_current)
        self.publishButton.clicked.connect(self.__publish_accepted)
        self.saveMetadataButton.clicked.connect(self.__save_metadata)
        self.reloadTmdbButton.clicked.connect(self.__reload_tmdb)
        self.browseArtButton.clicked.connect(self.__browse_art)
        self.downloadArtButton.clicked.connect(self.__download_art)
        self.clearArtButton.clicked.connect(self.__clear_art)
        self.__pending_tmdb_extras = {}
        self.__metadata_dirty = False
        for field in self.metadataTab.findChildren(QLineEdit):
            field.textEdited.connect(self.__mark_metadata_dirty)  # textEdited: a person typing, not setText()

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
        # Enter accepts only while the queue table or candidate list has focus. As a window-wide shortcut it also
        # fired from inside a metadata text field, accepting the entry (design/library-sync-pipeline-plan.md §12.1).
        # The letter and digit shortcuts below need no such scoping: a QLineEdit claims printable keys itself.
        for widget in (self.queueTable, self.candidateList):
            for key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                QShortcut(QKeySequence(key), widget, activated=self.__accept_current,
                          context=Qt.ShortcutContext.WidgetShortcut)
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

    def __reload_queue(self, keep_current=False):
        '''
        :param keep_current: stay on the entry that is selected now if it is still in the queue, rather than jumping to
            the first row -- what an edit to that entry (metadata, artwork, reopen) needs.
        '''
        if self.__queue_dir is None:
            return
        current = self.__current_entry() if keep_current else None
        if current is not None:
            self.__keep_candidate = (current.id, self.__selected_candidate_index)
        entries = read_queue(self.__queue_dir)
        self.__table_model.set_entries(entries)
        row = next((i for i, e in enumerate(entries) if current is not None and e.id == current.id), 0)
        if entries:
            self.queueTable.selectRow(row)
        else:
            self.__on_row_selected()
        self.__keep_candidate = None
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
            if self.__keep_candidate is not None and self.__keep_candidate[0] == entry.id:
                self.__selected_candidate_index = self.__keep_candidate[1]  # same entry after an edit: keep the pick
            if entry.status == 'accepted' and entry.chosen_candidate_index is not None:
                self.__selected_candidate_index = entry.chosen_candidate_index
            if not 0 <= self.__selected_candidate_index < len(entry.candidates):
                self.__selected_candidate_index = 0
            if entry.candidates:
                self.candidateList.setCurrentRow(self.__selected_candidate_index)
        else:
            self.titleLabel.setText('')
            self.declineReasonLabel.setVisible(False)
        self.candidateList.blockSignals(False)
        self.__refresh_commentary_table()
        self.__update_action_buttons()
        self.__magnitude_model.redraw()
        self.__load_metadata_form(entry)
        self.__load_artwork_section(entry)

    def __load_metadata_form(self, entry):
        self.__pending_tmdb_extras = {}
        self.__metadata_dirty = False
        meta = entry.meta if entry is not None else {}
        self.titleField.setText(meta.get('title', ''))
        self.altTitleField.setText(meta.get('alt_title', ''))
        self.sortTitleField.setText(meta.get('sort_title', ''))
        self.yearField.setText(meta.get('year', ''))
        self.audioTypesField.setText(', '.join(meta.get('audio_types', [])))
        self.editionField.setText(meta.get('edition', ''))
        self.seasonField.setText(meta.get('season', ''))
        self.episodesField.setText(format_episodes(meta.get('episodes') or []))
        self.noteField.setText(meta.get('note', ''))
        self.warningField.setText(meta.get('warning', ''))
        self.languageField.setText(meta.get('language', ''))
        self.sourceField.setText(meta.get('source', ''))
        self.ratingField.setText(meta.get('rating', ''))
        self.authorField.setText(meta.get('author', ''))
        self.avsField.setText(meta.get('avs', ''))
        self.runtimeField.setText(meta.get('runtime', ''))
        self.gainField.setText(meta.get('gain') or '')
        self.movieDbIdField.setText(meta.get('the_movie_db', ''))
        genres = meta.get('genres') or []
        self.genresLabel.setText(', '.join(g.get('name', '') for g in genres))
        self.metadataStatusLabel.setText('')
        editable = entry is not None and entry.status in ('pending', 'skipped')
        self.metadataTab.setEnabled(editable)  # whole tab greyed out once accepted/published

    def __load_artwork_section(self, entry):
        art_path = entry.art_path if entry is not None else None
        self.artPathField.setText(art_path or '')
        if art_path and os.path.isfile(art_path):
            self.artPreviewLabel.setPixmap(QPixmap(art_path).scaledToWidth(160, Qt.TransformationMode.SmoothTransformation))
        else:
            self.artPreviewLabel.clear()

    # --- metadata / artwork editing -----------------------------------------

    def __mark_metadata_dirty(self, *_):
        self.__metadata_dirty = True

    def __save_metadata(self):
        entry = self.__current_entry()
        if entry is None:
            return
        fields = {
            'title': self.titleField.text().strip(),
            'year': self.yearField.text().strip(),
            'audio_types': [t.strip() for t in self.audioTypesField.text().split(',') if t.strip()],
        }
        optional = {
            'alt_title': self.altTitleField.text().strip(),
            'sort_title': self.sortTitleField.text().strip(),
            'edition': self.editionField.text().strip(),
            'season': self.seasonField.text().strip(),
            'note': self.noteField.text().strip(),
            'warning': self.warningField.text().strip(),
            'language': self.languageField.text().strip(),
            'source': self.sourceField.text().strip(),
            'rating': self.ratingField.text().strip(),
            'author': self.authorField.text().strip(),
            'avs': self.avsField.text().strip(),
            'runtime': self.runtimeField.text().strip(),
            'gain': self.gainField.text().strip(),
            'the_movie_db': self.movieDbIdField.text().strip(),
            'overview': self.__pending_tmdb_extras.get('overview', ''),
        }
        fields.update({k: v for k, v in optional.items() if v})  # blank = leave unset, don't stomp a BeqMetadata default
        try:
            # unlike the text fields a blank box is a real answer here (no episodes in scope), so it is written
            fields['episodes'] = parse_episodes(self.episodesField.text())
        except ValueError as error:
            self.metadataStatusLabel.setText(f"Episodes: {error} (use numbers and ranges such as 1-3, 5)")
            return
        if 'genres' in self.__pending_tmdb_extras:
            fields['genres'] = self.__pending_tmdb_extras['genres']
        if 'collection' in self.__pending_tmdb_extras:
            fields['collection'] = self.__pending_tmdb_extras['collection']
        merged = {**entry.meta, **fields}
        update_entry(self.__queue_dir, entry.id, meta=merged)
        self.__reload_queue(keep_current=True)
        self.metadataStatusLabel.setText('Saved')

    def __reload_tmdb(self):
        from pipeline.metadata import tmdb_details_by_id, tmdb_lookup
        from model.preferences import TMDB_API_KEY
        api_key = self.__preferences.get(TMDB_API_KEY)
        tmdb_id = self.movieDbIdField.text().strip()
        kind = 'tv' if self.seasonField.text().strip() else 'movie'
        try:
            if tmdb_id:
                meta = tmdb_details_by_id(tmdb_id, api_key, kind=kind)
            else:
                meta = tmdb_lookup(self.titleField.text().strip(), self.yearField.text().strip(), api_key,
                                   kind=kind)
            self.titleField.setText(meta.title)
            self.altTitleField.setText(meta.alt_title)
            self.yearField.setText(meta.year)
            self.ratingField.setText(meta.rating)
            self.runtimeField.setText(meta.runtime)
            self.movieDbIdField.setText(meta.the_movie_db)
            self.genresLabel.setText(', '.join(g.get('name', '') for g in meta.genres))
            self.__pending_tmdb_extras = {'poster': meta.poster, 'overview': meta.overview,
                                          'genres': meta.genres, 'collection': meta.collection}
            self.__metadata_dirty = True
        except requests.RequestException as e:  # HTTPError, ConnectionError, Timeout...
            QMessageBox.critical(self, 'TMDB lookup failed', str(e))

    def __browse_art(self):
        entry = self.__current_entry()
        if entry is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, 'Choose artwork', filter='Images (*.png *.jpg *.jpeg)')
        if path:
            update_entry(self.__queue_dir, entry.id, art_path=path, art_overridden=True)
            self.__reload_queue(keep_current=True)

    def __download_art(self):
        entry = self.__current_entry()
        url = self.artUrlField.text().strip()
        if entry is None or not url:
            return
        try:
            resp = requests.get(url)
            resp.raise_for_status()
        except Exception as e:
            QMessageBox.critical(self, 'Download failed', str(e))
            return
        cache_dir = os.path.join(self.__queue_dir, '_art_cache')
        os.makedirs(cache_dir, exist_ok=True)
        ext = os.path.splitext(url)[1] or '.jpg'
        dest = os.path.join(cache_dir, f"{entry.id}{ext}")
        with open(dest, 'wb') as f:
            f.write(resp.content)
        update_entry(self.__queue_dir, entry.id, art_path=dest, art_overridden=True)
        self.artUrlField.clear()
        self.__reload_queue(keep_current=True)

    def __clear_art(self):
        entry = self.__current_entry()
        if entry is None:
            return
        update_entry(self.__queue_dir, entry.id, art_path=None, art_overridden=False)
        self.__reload_queue(keep_current=True)

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
        self.acceptButton.setEnabled(has_candidates and entry.status in ('pending', 'skipped'))
        self.skipButton.setEnabled(entry is not None)
        self.rejectButton.setEnabled(entry is not None)
        self.reopenButton.setEnabled(entry is not None and entry.status == 'accepted')

    # --- triage actions ------------------------------------------------------

    def __accept_current(self):
        entry = self.__current_entry()
        if entry is None or not entry.candidates:
            return
        if entry.status in ('accepted', 'published'):
            return  # nothing to decide; also stops Enter re-accepting (or, worse, un-publishing) a decided entry
        picked = self.__selected_candidate_index  # before the prompt: saving reloads the queue and reselects the row
        if self.__metadata_dirty and not self.__resolve_unsaved_metadata():
            return
        self.__apply_decision(entry.id, status='accepted', chosen_candidate_index=picked)

    def __resolve_unsaved_metadata(self):
        '''
        Accepting with edits still in the metadata form would publish the entry without them, silently. Ask instead.
        :return: True if accepting should go ahead (edits saved, or knowingly discarded), False to stay put.
        '''
        answer = QMessageBox.question(
            self, 'Unsaved metadata', 'The metadata for this entry has been edited but not saved. Save it before accepting?',
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save)
        if answer == QMessageBox.StandardButton.Cancel:
            return False
        if answer == QMessageBox.StandardButton.Save:
            self.__save_metadata()
            if self.metadataStatusLabel.text() != 'Saved':
                return False  # e.g. an unparseable episodes list: the reason is on screen, so don't accept
        return True

    def __reopen_current(self):
        ''' Undoes an Accept that has not been published: the entry goes back to pending, on the same row. '''
        entry = self.__current_entry()
        if entry is None or entry.status != 'accepted':
            return
        update_entry(self.__queue_dir, entry.id, status='pending', chosen_candidate_index=None)
        self.__reload_queue(keep_current=True)

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
        published, needs_attention = split_publish_results(results)
        self.__reload_queue()  # first: it replaces the status message with the queue summary
        self.statusBar.showMessage(f"Published {len(published)} title(s)"
                                   + (f", {len(needs_attention)} need attention" if needs_attention else ''))
        if needs_attention:
            QMessageBox.warning(self, 'Publish', 'These entries were not published:\n\n'
                                + '\n'.join(describe_publish_error(r) for r in needs_attention))

    def __on_publish_errored(self, message):
        self.publishProgress.setRange(0, 1)
        self.publishButton.setEnabled(True)
        QMessageBox.critical(self, 'Publish failed', message)
