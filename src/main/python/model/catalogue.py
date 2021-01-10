import csv
import glob
import logging
import os
import re
import time
from typing import Optional, Any
from datetime import datetime
from itertools import groupby
from pathlib import Path
from urllib.parse import urlparse

import qtawesome as qta
import requests
from dateutil.parser import parse as parsedate
from qtpy.QtCore import Signal, QRunnable, QObject, QThreadPool, QUrl, Qt, QAbstractTableModel, QModelIndex, QVariant
from qtpy.QtGui import QDesktopServices, QImageReader, QPixmap
from qtpy.QtWidgets import QDialog, QMessageBox, QSizePolicy, QListWidgetItem, QHeaderView
from sortedcontainers import SortedSet

from model.minidsp import get_repo_subdir, load_filter_file, FilterPublisher, FilterPublisherSignals
from model.preferences import BEQ_DOWNLOAD_DIR, BEQ_REPOS, BINARIES_MINIDSP_RS, MINIDSP_RS_OPTIONS
from model.report import block_signals
from ui.browse_catalogue import Ui_catalogueViewerDialog
from ui.catalogue import Ui_catalogueDialog
from ui.imgviewer import Ui_imgViewerDialog

logger = logging.getLogger('catalogue')


class BEQ:

    def __init__(self, repo, filename):
        self.repo = repo
        self.filename = filename
        self.name = None
        self.year = None
        self.content_type = None
        match = re.match(r"(.*)\((\d{4})\)(?:.*BEQ )?(.*)", os.path.basename(filename))
        if match:
            self.name = match.group(1).strip()
            self.year = match.group(2)
            self.content_type = match.group(3).strip()[:-4]

    def __repr__(self):
        return f"{self.repo} / {self.name} / {self.content_type}"


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
        self.__catalogue = {}
        self.setupUi(self)
        self.__beq_dir = self.__preferences.get(BEQ_DOWNLOAD_DIR)
        self.__db_csv_file = os.path.join(self.__beq_dir, 'database.csv')
        self.__db_csv = {}
        self.browseCatalogueButton.setEnabled(False)
        self.browseCatalogueButton.setIcon(qta.icon('fa5s.folder-open'))
        QThreadPool.globalInstance().start(DatabaseDownloader(self.__on_database_load,
                                                              self.__alert_on_database_load_error,
                                                              self.__db_csv_file))
        self.loadFilterButton.setEnabled(False)
        self.showInfoButton.setEnabled(False)
        self.openAvsButton.setEnabled(False)
        self.openCatalogueButton.setEnabled(False)
        for r in self.__preferences.get(BEQ_REPOS).split('|'):
            self.__populate_catalogue(r)
        years = SortedSet({c.year for c in self.__catalogue.values()})
        for y in reversed(years):
            self.yearFilter.addItem(y)
        self.yearMinFilter.setMinimum(int(years[0]))
        self.yearMinFilter.setMaximum(int(years[-1]) - 1)
        self.yearMaxFilter.setMinimum(int(years[0]) + 1)
        self.yearMaxFilter.setMaximum(int(years[-1]))
        self.yearMinFilter.setValue(int(years[0]))
        self.filter_min_year(self.yearMinFilter.value())
        self.yearMaxFilter.setValue(int(years[-1]))
        self.filter_max_year(self.yearMaxFilter.value())
        content_types = SortedSet({c.content_type for c in self.__catalogue.values()})
        for c in content_types:
            self.contentTypeFilter.addItem(c)
        self.filter_content_type('')
        self.totalCount.setValue(len(self.__catalogue))

    def __on_database_load(self, database):
        if database is True:
            db_csv = []
            db_csv_dict = {}
            with open(self.__db_csv_file) as f:
                indices = {
                    'Title': -1,
                    'Format': -1,
                    'Author': -1,
                    'AVS': -1,
                    'Catalogue': -1
                }
                last_idx = -1
                for i, x in enumerate(csv.reader(f)):
                    if i == 0:
                        for j, arg in enumerate(x):
                            if arg in indices:
                                indices[arg] = j
                        last_idx = len(x)
                    else:
                        vals = {k: x[v] for k, v in indices.items()}
                        vals['imgs'] = x[last_idx:]
                        db_csv.append(vals)
            for name, cats in groupby(db_csv, lambda x: x['Title']):
                cats_list = list(cats)
                for c in cats_list:
                    k = c['Title'] if len(cats_list) == 1 else f"{c['Title']} -- {c['Format']}"
                    db_csv_dict[k] = c
            self.__db_csv = db_csv_dict
            self.browseCatalogueButton.setEnabled(True)

    @staticmethod
    def __alert_on_database_load_error(message):
        '''
        Shows an alert if we can't load the database.
        :param message: the message.
        '''
        show_alert('Unable to Load BEQCatalogue Database', message)

    def __populate_catalogue(self, repo_url):
        subdir = get_repo_subdir(repo_url)
        if os.path.exists(self.__beq_dir) and os.path.exists(os.path.join(self.__beq_dir, subdir, '.git')):
            search_path = f"{self.__beq_dir}{os.sep}{subdir}{os.sep}**{os.sep}*.xml"
            beqs = sorted(filter(lambda x: x.name is not None, [BEQ(subdir, x) for x in glob.glob(search_path, recursive=True)]),
                         key=lambda x: x.name)
            for name, grouped in groupby(beqs, lambda x: x.name):
                beq_list = list(grouped)
                for beq in beq_list:
                    k = self.__make_key(beq) if len(beq_list) > 1 else beq.name
                    if k in self.__catalogue:
                        logger.warning(f"Duplicate catalogue entry found at {beq.filename} vs {self.__catalogue[k]}")
                    else:
                        self.__catalogue[k] = beq

    def apply_filter(self):
        '''
        Updates the matching count.
        '''
        years = [i.text() for i in self.yearFilter.selectedItems()]
        content_types = [i.text() for i in self.contentTypeFilter.selectedItems()]
        name_filter = self.nameFilter.text().casefold()
        count = 0
        selected_text = self.resultsList.selectedItems()[0].text() if len(self.resultsList.selectedItems()) > 0 else None
        self.resultsList.clear()
        self.on_result_selection_changed()
        matches = []
        for key, beq in self.__catalogue.items():
            name = key.split(' -- ', maxsplit=1)[0].casefold()
            if not years or beq.year in years:
                if not content_types or beq.content_type in content_types:
                    if not name_filter or name.find(name_filter) > -1:
                        if self.__included_in_catalogue_filter(beq):
                            count += 1
                            matches.append(key)
        row_to_set = -1
        for idx, m in enumerate(sorted(matches, key=str.casefold)):
            self.resultsList.addItem(m)
            if selected_text and m == selected_text:
                row_to_set = idx
        if row_to_set > -1:
            self.resultsList.setCurrentRow(row_to_set)
        self.matchCount.setValue(count)

    def __included_in_catalogue_filter(self, beq):
        ''' if this beq can be found in the catalogue. '''
        include = False
        if self.allRadioButton.isChecked():
            include = True
        elif self.inCatalogueOnlyRadioButton.isChecked():
            include = self.__make_key(beq) in self.__db_csv or beq.name in self.__db_csv
        elif self.missingFromCatalogueRadioButton.isChecked():
            include = self.__make_key(beq) not in self.__db_csv and beq.name not in self.__db_csv

        if include:
            if self.allReposRadioButton.isChecked():
                return True
            elif self.aron7awolRepoButton.isChecked():
                return beq.repo == 'bmiller_miniDSPBEQ'
            elif self.mobe1969RepoButton.isChecked():
                return beq.repo == 'Mobe1969_miniDSPBEQ'
        else:
            return False

    @staticmethod
    def __make_key(beq):
        return f"{beq.name} -- {beq.content_type}"

    def load_filter(self):
        '''
        Loads the currently selected filter into the model.
        '''
        beq = self.__get_beq_from_results()
        filt = load_filter_file(beq.filename, 48000)
        self.__filter_loader(beq.name, filt)

    def send_filter_to_minidsp(self):
        '''
        Sends the currently selected filter to the filter publisher.
        :return:
        '''
        beq = self.__get_beq_from_results()
        filt = load_filter_file(beq.filename, 96000)
        fp = FilterPublisher(filt, self.__minidsp_rs_exe, self.__minidsp_rs_options, self.__on_send_filter_event)
        QThreadPool.globalInstance().start(fp)

    def __on_send_filter_event(self, code: int):
        if code == FilterPublisherSignals.ON_START:
            self.__process_spinner = qta.Spin(self.sendToMinidspButton)
            spin_icon = qta.icon('fa5s.spinner', color='green', animation=self.__process_spinner)
            self.sendToMinidspButton.setIcon(spin_icon)
            self.sendToMinidspButton.setEnabled(False)
            pass
        elif code == FilterPublisherSignals.ON_COMPLETE:
            from model.batch import stop_spinner
            stop_spinner(self.__process_spinner, self.sendToMinidspButton)
            self.__process_spinner = None
            self.sendToMinidspButton.setIcon(qta.icon('fa5s.check', color='green'))
            self.sendToMinidspButton.setEnabled(True)
        elif code == FilterPublisherSignals.ON_ERROR:
            from model.batch import stop_spinner
            stop_spinner(self.__process_spinner, self.sendToMinidspButton)
            self.__process_spinner = None
            self.sendToMinidspButton.setIcon(qta.icon('fa5s.times', color='red'))
            self.sendToMinidspButton.setEnabled(True)
        else:
            logger.warning(f"Unknown code received from FilterPublisher - {code}")

    def __get_beq_from_results(self) -> BEQ:
        return self.__catalogue[self.resultsList.selectedItems()[0].text()]

    def __get_catalogue_from_results(self):
        return self.__db_csv.get(self.resultsList.selectedItems()[0].text(), None)

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
        self.path.clear()
        if selected:
            self.path.setText(self.__get_beq_from_results().filename)
            cat = self.__get_catalogue_from_results()
            if self.__db_csv and cat:
                self.showInfoButton.setEnabled(True)
                self.openCatalogueButton.setEnabled(True)
                self.openAvsButton.setEnabled('AVS' in cat and len(cat['AVS']) > 0)
                self.loadFilterButton.setEnabled(True)
                self.sendToMinidspButton.setEnabled(self.__minidsp_rs_exe is not None)

    def show_info(self):
        ''' Displays the info about the BEQ from the catalogue '''
        cat = self.__get_catalogue_from_results()
        if cat['imgs']:
            for i in cat['imgs']:
                ImageViewerDialog(self, self.__preferences, self.__beq_dir, cat['Author'], i).show()

    def goto_catalogue(self):
        ''' Opens the catalogue page in a browser '''
        QDesktopServices.openUrl(QUrl(self.__get_catalogue_from_results()['Catalogue']))

    def goto_avs(self):
        ''' Open the corresponding AVS post. '''
        QDesktopServices.openUrl(QUrl(self.__get_catalogue_from_results()['AVS']))

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
        dialog = BrowseCatalogueDialog(self, self.__db_csv, filt=self.nameFilter.text())
        dialog.show()


class DatabaseDownloadSignals(QObject):
    on_load = Signal(bool, name='on_load')
    on_error = Signal(str, name='on_error')


class DatabaseDownloader(QRunnable):
    DATABASE_CSV = 'http://beqcatalogue.readthedocs.io/en/latest/database.csv'

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
        cached_date = datetime.fromtimestamp(os.path.getmtime(self.__cached)).astimezone() if os.path.exists(self.__cached) else None
        if mod_date is None or cached_date is None or mod_date > cached_date:
            r = requests.get(self.DATABASE_CSV, allow_redirects=True)
            if r.status_code == 200:
                with open(self.__cached, 'wb') as f:
                    f.write(r.content)
                modified = time.mktime(mod_date.timetuple())
                now = time.mktime(datetime.today().timetuple())
                os.utime(self.__cached, (now, modified))
        self.__signals.on_load.emit(os.path.exists(self.__cached))

    def __get_mod_date(self):
        '''
        HEADs the database.csv to find the last modified date.
        :return: the date.
        '''
        try:
            r = requests.head(self.DATABASE_CSV, allow_redirects=True)
            if r.status_code == 200:
                if 'Last-Modified' in r.headers:
                    return parsedate(r.headers['Last-Modified']).astimezone()
            else:
                self.__signals.on_error.emit(f"Unable to hit BEQCatalogue, response was {r.status_code}")
        except:
            logger.exception('Failed to hit BEQCatalogue')
            self.__signals.on_error.emit(f"Unable to contact BEQCatalogue at: \n\n {self.DATABASE_CSV}")
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
