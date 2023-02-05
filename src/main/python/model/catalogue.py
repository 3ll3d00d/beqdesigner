import json
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, List, Dict
from urllib.parse import urlparse

import qtawesome as qta
import requests
from dateutil.parser import parse as parsedate
from qtpy.QtCore import Signal, QRunnable, QObject, QThreadPool, QUrl, Qt, QAbstractTableModel, QModelIndex, QVariant
from qtpy.QtGui import QDesktopServices, QImageReader, QPixmap
from qtpy.QtWidgets import QDialog, QMessageBox, QSizePolicy, QListWidgetItem, QHeaderView, QMenu, QAction, QPushButton
from sortedcontainers import SortedSet

from model.iir import HighShelf, LowShelf, PeakingEQ, BiquadWithQGain
from model.minidsp import load_filter_file, FilterPublisher, FilterPublisherSignals
from model.preferences import BEQ_DOWNLOAD_DIR, BINARIES_MINIDSP_RS, MINIDSP_RS_OPTIONS
from model.report import block_signals
from ui.browse_catalogue import Ui_catalogueViewerDialog
from ui.catalogue import Ui_catalogueDialog
from ui.imgviewer import Ui_imgViewerDialog

logger = logging.getLogger('catalogue')

TWO_WEEKS_AGO_SECONDS = 2 * 7 * 24 * 60 * 60
ENTRY_ID_ROLE = Qt.UserRole + 1


class CatalogueDialog(QDialog, Ui_catalogueDialog):

    def __init__(self, parent, prefs, filter_loader):
        super(CatalogueDialog, self).__init__(parent=parent)
        self.__filter_loader = filter_loader
        self.__preferences = prefs
        minidsp_rs_path = prefs.get(BINARIES_MINIDSP_RS)
        self.__minidsp_rs_exe = None
        if minidsp_rs_path:
            minidsp_rs_exe = os.path.join(minidsp_rs_path, 'minidsp')
            if os.path.isfile(minidsp_rs_exe):
                self.__minidsp_rs_exe = minidsp_rs_exe
            else:
                minidsp_rs_exe = os.path.join(minidsp_rs_path, 'minidsp.exe')
                if os.path.isfile(minidsp_rs_exe):
                    self.__minidsp_rs_exe = minidsp_rs_exe
        self.__minidsp_rs_options = None
        if self.__minidsp_rs_exe:
            self.__minidsp_rs_options = prefs.get(MINIDSP_RS_OPTIONS)
        self.__catalogue: List[CatalogueEntry] = []
        self.__catalogue_by_idx: Dict[str, CatalogueEntry] = {}
        self.setupUi(self)
        self.sendToMinidspButton.setMenu(self.__make_minidsp_menu(self.send_filter_to_minidsp))
        self.bypassMinidspButton.setMenu(self.__make_minidsp_menu(self.clear_filter_from_minidsp))
        self.__beq_dir = self.__preferences.get(BEQ_DOWNLOAD_DIR)
        self.__beq_file = os.path.join(self.__beq_dir, 'database.json')
        self.browseCatalogueButton.setEnabled(False)
        self.browseCatalogueButton.setIcon(qta.icon('fa5s.folder-open'))
        QThreadPool.globalInstance().start(DatabaseDownloader(self.__on_database_load,
                                                              self.__alert_on_database_load_error,
                                                              self.__beq_file))
        self.loadFilterButton.setEnabled(False)
        self.showInfoButton.setEnabled(False)
        self.openAvsButton.setEnabled(False)
        self.openCatalogueButton.setEnabled(False)

    def __make_minidsp_menu(self, func):
        menu = QMenu(self)
        current_config = QAction(menu)
        current_config.setText('Current')
        current_config.triggered.connect(func)
        menu.addAction(current_config)
        for i in range(4):
            self.__add_send_action(i, menu, func)
        return menu

    @staticmethod
    def __add_send_action(slot_idx, menu, func):
        a = QAction(menu)
        a.setText(f"Slot {slot_idx + 1}")
        menu.addAction(a)
        a.triggered.connect(lambda: func(slot=slot_idx))

    def __on_database_load(self, database):
        if database is True:
            with open(self.__beq_file, 'r') as infile:
                self.__catalogue = [CatalogueEntry(f"{idx}", c) for idx, c in enumerate(json.load(infile))]
                self.__catalogue_by_idx = {c.idx: c for c in self.__catalogue}
            self.browseCatalogueButton.setEnabled(True)
            years = SortedSet({c.year for c in self.__catalogue})
            for y in reversed(years):
                self.yearFilter.addItem(str(y))
            self.yearMinFilter.setMinimum(int(years[0]))
            self.yearMinFilter.setMaximum(int(years[-1]) - 1)
            self.yearMaxFilter.setMinimum(int(years[0]) + 1)
            self.yearMaxFilter.setMaximum(int(years[-1]))
            self.yearMinFilter.setValue(int(years[0]))
            self.filter_min_year(self.yearMinFilter.value())
            self.yearMaxFilter.setValue(int(years[-1]))
            self.filter_max_year(self.yearMaxFilter.value())
            content_types = SortedSet({c.content_type for c in self.__catalogue})
            for c in content_types:
                self.contentTypeFilter.addItem(c)
            self.filter_content_type('')
            self.totalCount.setValue(len(self.__catalogue))

    @staticmethod
    def __alert_on_database_load_error(message):
        '''
        Shows an alert if we can't load the database.
        :param message: the message.
        '''
        show_alert('Unable to Load BEQCatalogue Database', message)

    def apply_filter(self):
        '''
        Updates the matching count.
        '''
        years = [int(i.text()) for i in self.yearFilter.selectedItems()]
        content_types = [i.text() for i in self.contentTypeFilter.selectedItems()]
        name_filter = self.nameFilter.text().casefold()
        count = 0
        selected_text = self.resultsList.selectedItems()[0].text() if len(
            self.resultsList.selectedItems()) > 0 else None
        self.resultsList.clear()
        self.on_result_selection_changed()
        matches: List[CatalogueEntry] = []
        for beq in self.__catalogue:
            if not years or beq.year in years:
                if not content_types or beq.content_type in content_types:
                    if not name_filter or beq.title.find(name_filter) > -1:
                        if self.__included_in_catalogue_filter(beq):
                            count += 1
                            matches.append(beq)
        row_to_set = -1
        for idx, m in enumerate(matches):
            suffix = f" ({','.join(m.audio_types)})" if m.audio_types else ''
            item = QListWidgetItem(f"{m.formatted_title} ({m.author}){suffix}")
            item.setData(ENTRY_ID_ROLE, m.idx)
            self.resultsList.addItem(item)
            if selected_text and m == selected_text:
                row_to_set = idx
        if row_to_set > -1:
            self.resultsList.setCurrentRow(row_to_set)
        self.matchCount.setValue(count)

    def __included_in_catalogue_filter(self, beq: 'CatalogueEntry'):
        ''' if this beq can be found in the catalogue. '''
        include = False
        if self.allRadioButton.isChecked():
            include = True
        elif self.inCatalogueOnlyRadioButton.isChecked():
            include = beq.title in self.__catalogue_by_idx
        elif self.missingFromCatalogueRadioButton.isChecked():
            include = beq.title not in self.__catalogue_by_idx

        if include:
            if self.allReposRadioButton.isChecked():
                return True
            elif self.aron7awolRepoButton.isChecked():
                return beq.author == 'aron7awol'
            elif self.mobe1969RepoButton.isChecked():
                return beq.author == 'mobe1969'
        else:
            return False

    def load_filter(self):
        '''
        Loads the currently selected filter into the model.
        '''
        beq = self.__get_entry_from_results()
        self.__filter_loader(beq.title, beq.iir_filters)

    def send_filter_to_minidsp(self, slot=None):
        '''
        Sends the currently selected filter to the filter publisher.
        '''
        beq = self.__get_entry_from_results()
        fp = FilterPublisher(beq.iir_filters, slot, self.__minidsp_rs_exe, self.__minidsp_rs_options,
                             lambda c: self.__on_send_filter_event(c, self.sendToMinidspButton))
        QThreadPool.globalInstance().start(fp)

    def clear_filter_from_minidsp(self, slot=None):
        '''
        Sets the config to bypass.
        '''
        fp = FilterPublisher([], slot, self.__minidsp_rs_exe, self.__minidsp_rs_options,
                             lambda c: self.__on_send_filter_event(c, self.bypassMinidspButton))
        QThreadPool.globalInstance().start(fp)

    def __on_send_filter_event(self, code: int, btn: QPushButton):
        if code == FilterPublisherSignals.ON_START:
            self.__process_spinner = qta.Spin(btn)
            spin_icon = qta.icon('fa5s.spinner', color='green', animation=self.__process_spinner)
            btn.setIcon(spin_icon)
            btn.setEnabled(False)
            pass
        elif code == FilterPublisherSignals.ON_COMPLETE:
            from model.batch import stop_spinner
            stop_spinner(self.__process_spinner, btn)
            self.__process_spinner = None
            btn.setIcon(qta.icon('fa5s.check', color='green'))
            btn.setEnabled(True)
        elif code == FilterPublisherSignals.ON_ERROR:
            from model.batch import stop_spinner
            stop_spinner(self.__process_spinner, btn)
            self.__process_spinner = None
            btn.setIcon(qta.icon('fa5s.times', color='red'))
            btn.setEnabled(True)
        else:
            logger.warning(f"Unknown code received from FilterPublisher - {code}")

    def __get_entry_from_results(self) -> Optional['CatalogueEntry']:
        selected = self.resultsList.selectedItems()[0]
        return self.__catalogue_by_idx.get(selected.data(ENTRY_ID_ROLE), None)

    def on_result_selection_changed(self):
        '''
        enables or disables the result buttons.
        '''
        selected_items = self.resultsList.selectedItems()
        selected = len(selected_items) > 0
        self.loadFilterButton.setEnabled(False)
        self.sendToMinidspButton.setEnabled(False)
        self.showInfoButton.setEnabled(False)
        self.openCatalogueButton.setEnabled(False)
        self.openAvsButton.setEnabled(False)
        if selected:
            cat = self.__get_entry_from_results()
            if cat:
                self.showInfoButton.setEnabled(True)
                self.openCatalogueButton.setEnabled(True)
                self.openAvsButton.setEnabled(len(cat.avs_url) > 0)
                self.loadFilterButton.setEnabled(True)
                self.sendToMinidspButton.setEnabled(self.__minidsp_rs_exe is not None)

    def show_info(self):
        ''' Displays the info about the BEQ from the catalogue '''
        cat = self.__get_entry_from_results()
        if cat.images:
            for i in cat.images:
                ImageViewerDialog(self, self.__preferences, self.__beq_dir, cat.author, i).show()

    def goto_catalogue(self):
        ''' Opens the catalogue page in a browser '''
        QDesktopServices.openUrl(QUrl(self.__get_entry_from_results().beqc_url))

    def goto_avs(self):
        ''' Open the corresponding AVS post. '''
        QDesktopServices.openUrl(QUrl(self.__get_entry_from_results().avs_url))

    def filter_min_year(self, min_year: int):
        ''' sets the min year filter '''
        max_year = self.yearMaxFilter.value()
        with block_signals(self.yearFilter):
            for i in range(self.yearFilter.count()):
                item: QListWidgetItem = self.yearFilter.item(i)
                val = int(item.text())
                item.setSelected(min_year <= val <= max_year)
            self.yearMaxFilter.setMinimum(min_year + 1)
        self.apply_filter()

    def filter_max_year(self, max_year: int):
        ''' sets the max year filter '''
        min_year = self.yearMinFilter.value()
        with block_signals(self.yearFilter):
            for i in range(self.yearFilter.count()):
                item: QListWidgetItem = self.yearFilter.item(i)
                val = int(item.text())
                item.setSelected(min_year <= val <= max_year)
            self.yearMinFilter.setMaximum(max_year - 1)
        self.apply_filter()

    def filter_content_type(self, txt: str):
        ''' filters the selected content types to match the given string '''
        with block_signals(self.contentTypeFilter):
            for i in range(self.contentTypeFilter.count()):
                item: QListWidgetItem = self.contentTypeFilter.item(i)
                item.setSelected(len(txt.strip()) == 0 or item.text().casefold().find(txt.casefold()) > -1)
        self.apply_filter()

    def browse_catalogue(self):
        ''' Show the DB csv browser. '''
        dialog = BrowseCatalogueDialog(self, self.__beq_db, filt=self.nameFilter.text())
        dialog.show()


class DatabaseDownloadSignals(QObject):
    on_load = Signal(bool, name='on_load')
    on_error = Signal(str, name='on_error')


class DatabaseDownloader(QRunnable):
    DATABASE_URL = 'http://beqcatalogue.readthedocs.io/en/latest/database.json'

    def __init__(self, on_load_handler, on_error_handler, cached_file):
        super().__init__()
        self.__signals = DatabaseDownloadSignals()
        self.__cached = cached_file
        self.__signals.on_load.connect(on_load_handler)
        self.__signals.on_error.connect(on_error_handler)

    def run(self):
        '''
        Hit the BEQ Catalogue database and compare to the local cached version.
        if there is an updated database then download it.
        '''
        mod_date = self.__get_mod_date()
        cached_date = datetime.fromtimestamp(os.path.getmtime(self.__cached)).astimezone() if os.path.exists(
            self.__cached) else None
        if mod_date is None or cached_date is None or mod_date > cached_date:
            r = requests.get(self.DATABASE_URL, allow_redirects=True)
            if r.status_code == 200:
                with open(self.__cached, 'wb') as f:
                    f.write(r.content)
                modified = time.mktime(mod_date.timetuple())
                now = time.mktime(datetime.today().timetuple())
                os.utime(self.__cached, (now, modified))
        self.__signals.on_load.emit(os.path.exists(self.__cached))

    def __get_mod_date(self):
        '''
        HEADs the database to find the last modified date.
        :return: the date.
        '''
        try:
            r = requests.head(self.DATABASE_URL, allow_redirects=True)
            if r.status_code == 200:
                if 'Last-Modified' in r.headers:
                    return parsedate(r.headers['Last-Modified']).astimezone()
            else:
                self.__signals.on_error.emit(f"Unable to hit BEQCatalogue, response was {r.status_code}")
        except:
            logger.exception('Failed to hit BEQCatalogue')
            self.__signals.on_error.emit(f"Unable to contact BEQCatalogue at: \n\n {self.DATABASE_URL}")
        return None


class ImgDownloadSignals(QObject):
    on_load = Signal(str, name='on_load')
    on_error = Signal(str, name='on_error')


class ImageViewerDialog(QDialog, Ui_imgViewerDialog):

    def __init__(self, parent, prefs, cache_dir, repo, img):
        super(ImageViewerDialog, self).__init__(parent=parent)
        self.__img = img
        self.__preferences = prefs
        self.__pm = None
        self.setupUi(self)
        self.scrollArea.setWidgetResizable(True)
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label.setScaledContents(True)
        QThreadPool.globalInstance().start(ImgDownloader(cache_dir, repo, self.__on_load, self.__on_error, self.__img))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        logger.info(f"Resizing to {event.size().width()} x {event.size().height()}")
        self.__update_image()

    def __on_load(self, file_path):
        logger.info(f"Loading {file_path}")
        ir = QImageReader(file_path)
        ir.setAutoTransform(True)
        img = ir.read()
        if not img:
            show_alert('Unable to Display Image', f"Unable to display {file_path}")
        else:
            self.__pm = QPixmap.fromImage(img)
            self.__update_image()

    def __update_image(self):
        if self.__pm:
            scaled_pm = self.__pm.scaled(0.95 * self.scrollArea.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label.setPixmap(scaled_pm)
            self.label.resize(scaled_pm.size())

    @staticmethod
    def __on_error(msg):
        show_alert('Unable to Load Image', msg)


class ImgDownloader(QRunnable):
    def __init__(self, cache_root_dir, repo, on_load_handler, on_error_handler, img_url):
        super().__init__()
        self.__cache_dir = os.path.join(cache_root_dir, '.img_cache', repo)
        Path(self.__cache_dir).mkdir(parents=True, exist_ok=True)
        self.__signals = ImgDownloadSignals()
        self.__img_url = img_url
        self.__signals.on_load.connect(on_load_handler)
        self.__signals.on_error.connect(on_error_handler)

    def run(self):
        '''
        Download and cache the image located at the URL if we don't have it already.
        '''
        file_name = os.path.basename(urlparse(self.__img_url).path)
        file_path = os.path.join(self.__cache_dir, file_name)
        if not os.path.isfile(file_path) or Path(file_path).stat().st_size == 0:
            ok = False
            with open(file_path, 'wb') as f:
                try:
                    f.write(requests.get(self.__img_url).content)
                    ok = True
                except:
                    logger.exception(f"Unable to download {self.__img_url}")
            if ok:
                self.__signals.on_load.emit(file_path)
            else:
                self.__signals.on_error.emit(f"Unable to download {self.__img_url}")
                os.remove(file_path)
        else:
            self.__signals.on_load.emit(file_path)


def show_alert(title, message):
    '''
    Shows an alert.
    :param title: the title
    :param message: the message.
    '''
    msg_box = QMessageBox()
    msg_box.setText(message)
    msg_box.setIcon(QMessageBox.Warning)
    msg_box.setWindowTitle(title)
    msg_box.exec()


class BrowseCatalogueDialog(QDialog, Ui_catalogueViewerDialog):

    def __init__(self, parent, catalogue, filt: Optional[str] = None):
        super(BrowseCatalogueDialog, self).__init__(parent=parent)
        self.__catalogue = catalogue
        super().setupUi(self)
        self.__model = CatalogueTableModel(catalogue, parent=parent)
        self.tableView.setModel(self.__model)
        self.tableView.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.nameFilter.setText(filt)

    def apply_filter(self, txt: str):
        self.__model.filter(txt)


class CatalogueTableModel(QAbstractTableModel):

    def __init__(self, data: dict, filt: Optional[str] = None, parent=None):
        super().__init__(parent=parent)
        self.__raw_data = list(data.values())
        self.__data = None
        self.__cols = ['Title', 'Format', 'Author']
        self.filter(filt)

    def rowCount(self, parent=None):
        return len(self.__data)

    def columnCount(self, parent=None):
        return len(self.__cols)

    def data(self, index: QModelIndex, role: int = ...) -> Any:
        if not index.isValid() or role != Qt.DisplayRole:
            return QVariant()
        else:
            return QVariant(self.__data[index.row()][self.__cols[index.column()]])

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = ...) -> Any:
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return QVariant(self.__cols[section])
        return QVariant()

    def filter(self, txt: str):
        self.beginResetModel()
        if txt is None or len(txt.strip()) == 0:
            self.__data = self.__raw_data
        else:
            match_txt = txt.casefold()
            self.__data = [d for d in self.__raw_data if d['Title'].casefold().find(match_txt) > -1]
        self.endResetModel()


class CatalogueEntry:

    def __init__(self, idx: str, vals: dict):
        self.idx = idx
        self.title = vals.get('title', '')
        y = 0
        try:
            y = int(vals.get('year', 0))
        except:
            logger.error(f"Invalid year {vals.get('year', 0)} in {self.title}")
        self.year = y
        self.audio_types = vals.get('audioTypes', [])
        self.content_type = vals.get('content_type', 'film')
        self.author = vals.get('author', '')
        self.beqc_url = vals.get('catalogue_url', '')
        self.filters: List[dict] = vals.get('filters', [])
        self.images = vals.get('images', [])
        self.warning = vals.get('warning', [])
        self.season = vals.get('season', '')
        self.episodes = vals.get('episode', '')
        self.avs_url = vals.get('avs', '')
        self.sort_title = vals.get('sortTitle', '')
        self.edition = vals.get('edition', '')
        self.note = vals.get('note', '')
        self.language = vals.get('language', '')
        self.source = vals.get('source', '')
        self.overview = vals.get('overview', '')
        self.the_movie_db = vals.get('theMovieDB', '')
        self.rating = vals.get('rating', '')
        self.genres = vals.get('genres', [])
        self.altTitle = vals.get('altTitle', '')
        self.created_at = vals.get('created_at', 0)
        self.updated_at = vals.get('updated_at', 0)
        self.digest = vals.get('digest', '')
        self.collection = vals.get('collection', {})
        self.formatted_title = self.__format_title()
        now = time.time()
        if self.created_at >= (now - TWO_WEEKS_AGO_SECONDS):
            self.freshness = 'Fresh'
        elif self.updated_at >= (now - TWO_WEEKS_AGO_SECONDS):
            self.freshness = 'Updated'
        else:
            self.freshness = 'Stale'
        try:
            r = int(vals.get('runtime', 0))
        except:
            logger.error(f"Invalid runtime {vals.get('runtime', 0)} in {self.title}")
            r = 0
        self.runtime = r
        self.mv_adjust = 0.0
        if 'mv' in vals:
            v = vals['mv']
            try:
                self.mv_adjust = float(v)
            except:
                logger.error(f"Unknown mv_adjust value in {self.title} - {vals['mv']}")
                pass
        self.for_search = {
            'id': self.idx,
            'title': self.title,
            'year': self.year,
            'sortTitle': self.sort_title,
            'audioTypes': self.audio_types,
            'contentType': self.content_type,
            'author': self.author,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'freshness': self.freshness,
            'digest': self.digest,
            'formattedTitle': self.formatted_title
        }
        if self.beqc_url:
            self.for_search['beqcUrl'] = self.beqc_url
        if self.images:
            self.for_search['images'] = self.images
        if self.warning:
            self.for_search['warning'] = self.warning
        if self.season:
            self.for_search['season'] = self.season
        if self.episodes:
            self.for_search['episodes'] = self.episodes
        if self.mv_adjust:
            self.for_search['mvAdjust'] = self.mv_adjust
        if self.avs_url:
            self.for_search['avsUrl'] = self.avs_url
        if self.edition:
            self.for_search['edition'] = self.edition
        if self.note:
            self.for_search['note'] = self.note
        if self.language:
            self.for_search['language'] = self.language
        if self.source:
            self.for_search['source'] = self.source
        if self.overview:
            self.for_search['overview'] = self.overview
        if self.the_movie_db:
            self.for_search['theMovieDB'] = self.the_movie_db
        if self.rating:
            self.for_search['rating'] = self.rating
        if self.runtime:
            self.for_search['runtime'] = self.runtime
        if self.genres:
            self.for_search['genres'] = self.genres
        if self.altTitle:
            self.for_search['altTitle'] = self.altTitle
        if self.note:
            self.for_search['note'] = self.note
        if self.warning:
            self.for_search['warning'] = self.warning
        if self.collection and 'name' in self.collection:
            self.for_search['collection'] = self.collection['name']

    def matches(self, authors: List[str], years: List[int], audio_types: List[str], content_types: List[str]):
        if not authors or self.author in authors:
            if not years or self.year in years:
                if not audio_types or any(a_t in audio_types for a_t in self.audio_types):
                    return not content_types or self.content_type in content_types
        return False

    def __repr__(self):
        return f"[{self.content_type}] {self.title} / {self.audio_types} / {self.year}"

    @staticmethod
    def __format_episodes(formatted, working):
        val = ''
        if len(formatted) > 1:
            val += ', '
        if len(working) == 1:
            val += working[0]
        else:
            val += f"{working[0]}-{working[-1]}"
        return val

    def __format_tv_meta(self):
        season = f"S{self.season}" if self.season else ''
        episodes = self.episodes.split(',') if self.episodes else None
        if episodes:
            formatted = 'E'
            if len(episodes) > 1:
                working = []
                last_value = 0
                for ep in episodes:
                    if len(working) == 0:
                        working.append(ep)
                        last_value = int(ep)
                    else:
                        current = int(ep)
                        if last_value == current - 1:
                            working.append(ep)
                            last_value = current
                        else:
                            formatted += self.__format_episodes(formatted, working)
                            working = []
                if len(working) > 0:
                    formatted += self.__format_episodes(formatted, working)
            else:
                formatted += f"{self.episodes}"
            return f"{season}{formatted}"
        return season

    def __format_title(self) -> str:
        if self.content_type == 'TV':
            return f"{self.title} {self.__format_tv_meta()}"
        return self.title

    @property
    def iir_filters(self) -> List[BiquadWithQGain]:
        return [self.__convert(i, f) for i, f in enumerate(self.filters)]

    @staticmethod
    def __convert(i: int, f: dict) -> BiquadWithQGain:
        t = f['type']
        freq = f['freq']
        gain = f['gain']
        q = f['q']
        if t == 'PeakingEQ':
            return PeakingEQ(96000, freq, q, gain, f_id=i)
        elif t == 'LowShelf':
            return LowShelf(96000, freq, q, gain, f_id=i)
        elif t == 'HighShelf':
            return HighShelf(96000, freq, q, gain, f_id=i)
        else:
            raise ValueError(f"Unknown filt_type {t}")
