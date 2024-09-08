import glob
import os
import shutil
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import List

import qtawesome as qta
from qtpy.QtCore import QThreadPool, QRunnable, QObject, Signal
from qtpy.QtGui import QGuiApplication
from qtpy.QtWidgets import QDialog, QMessageBox, QFileDialog, QListWidgetItem, QDialogButtonBox
from sanitize_filename import sanitize

from model.catalogue import DatabaseDownloader, show_alert, load_catalogue, CatalogueEntry
from model.jriver.common import JRIVER_CHANNELS
from model.minidsp import logger, TwoByFourXmlParser, HDXmlParser, \
    xml_to_filt
from model.preferences import BEQ_CONFIG_FILE, BEQ_MERGE_DIR, BEQ_MINIDSP_TYPE, BEQ_DOWNLOAD_DIR, BEQ_EXTRA_DIR, \
    BEQ_OUTPUT_CHANNELS, BEQ_OUTPUT_MODE
from model.sync import HTP1Parser
from ui.merge import Ui_mergeDspDialog


class MergeFiltersDialog(QDialog, Ui_mergeDspDialog):
    '''
    Merge Filters dialog
    '''

    def __init__(self, parent, prefs, statusbar):
        super(MergeFiltersDialog, self).__init__(parent)
        self.setupUi(self)
        self.buttonBox.button(QDialogButtonBox.StandardButton.Reset).clicked.connect(self.__reset_index)
        self.__spinner = None
        self.__catalogue: List[CatalogueEntry] = []
        self.configFilePicker.setIcon(qta.icon('fa5s.folder-open'))
        self.outputDirectoryPicker.setIcon(qta.icon('fa5s.folder-open'))
        self.processFiles.setIcon(qta.icon('fa5s.save'))
        self.userSourceDirPicker.setIcon(qta.icon('fa5s.folder-open'))
        self.clearUserSourceDir.setIcon(qta.icon('fa5s.times', color='red'))
        self.copyOptimisedButton.setIcon(qta.icon('fa5s.copy'))
        self.copyErrorsButton.setIcon(qta.icon('fa5s.copy'))
        self.__preferences = prefs
        self.statusbar = statusbar
        self.optimised.setVisible(False)
        self.copyOptimisedButton.setVisible(False)
        self.optimisedLabel.setVisible(False)
        config_file = self.__preferences.get(BEQ_CONFIG_FILE)
        if config_file is not None and len(config_file) > 0 and os.path.exists(config_file):
            self.configFile.setText(os.path.abspath(config_file))
        self.outputDirectory.setText(self.__preferences.get(BEQ_MERGE_DIR))
        os.makedirs(self.outputDirectory.text(), exist_ok=True)
        for t in DspType:
            self.dspType.addItem(t.display_name)
        dsp_type = self.__preferences.get(BEQ_MINIDSP_TYPE)
        if dsp_type is not None and len(dsp_type) > 0:
            self.dspType.setCurrentText(dsp_type)
        self.__beq_dir = self.__preferences.get(BEQ_DOWNLOAD_DIR)
        os.makedirs(self.__beq_dir, exist_ok=True)
        self.__beq_file = os.path.join(self.__beq_dir, 'database.json')
        QThreadPool.globalInstance().start(DatabaseDownloader(self.__on_database_load,
                                                              self.__alert_on_database_load_error,
                                                              self.__beq_file))
        extra_dir = self.__preferences.get(BEQ_EXTRA_DIR)
        if extra_dir is not None and len(extra_dir) > 0 and os.path.exists(extra_dir):
            self.userSourceDir.setText(os.path.abspath(extra_dir))

    def __on_database_load(self, database: bool):
        if database is True:
            catalogue = load_catalogue(self.__beq_file)
            full_size = len(catalogue)
            p = Path(self.outputDirectory.text(), '.index')
            if p.is_file():
                try:
                    last_updated = int(p.read_text())
                except:
                    last_updated = 0
                    p.unlink()
                catalogue = [c for c in catalogue if c.updated_at > last_updated]
                if len(catalogue) == 0:
                    from datetime import datetime
                    last_formatted = datetime.fromtimestamp(last_updated).strftime('%c')
                    show_alert('BEQ Merge',
                               f'Merged files are up to date\n\nCatalogue last updated at {last_formatted} and has {full_size} entries')
            self.__catalogue = catalogue
            self.update_beq_count()
            self.__enable_process()

    def __reset_index(self):
        index_file = f"{self.outputDirectory.text()}/.index"
        if os.path.exists(index_file):
            result = QMessageBox.question(self,
                                          'Reset Index',
                                          f"All generated config files will be deleted from "
                                          f"{self.outputDirectory.text()}\nAre you sure you want to continue?",
                                          QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                          QMessageBox.StandardButton.No)
            if result == QMessageBox.StandardButton.Yes:
                from app import wait_cursor
                with wait_cursor():
                    self.statusbar.showMessage(f"Deleting {index_file}", 2000)
                    os.remove(index_file)
                    self.statusbar.showMessage(f"Deleting {self.outputDirectory.text()}", 2000)
                    shutil.rmtree(self.outputDirectory.text())
                    os.makedirs(self.outputDirectory.text(), exist_ok=True)
                    self.statusbar.showMessage(f"Deleted {self.outputDirectory.text()}", 10000)
                    self.__on_database_load(True)
                    self.filesProcessed.setValue(0)

    @staticmethod
    def __alert_on_database_load_error(message):
        '''
        Shows an alert if we can't load the database.
        :param message: the message.
        '''
        show_alert('Unable to Load BEQCatalogue Database', message)

    def update_beq_count(self):
        user_files = 0
        if len(self.userSourceDir.text().strip()) > 0 and os.path.exists(self.userSourceDir.text()):
            user_files = len(glob.glob(f"{self.userSourceDir.text()}{os.sep}**{os.sep}*.xml", recursive=True))

        if user_files > 0 or len(self.__catalogue) > 0:
            self.__show_or_hide(user_files > 0, len(self.__catalogue) > 0)
            self.totalFiles.setValue(user_files + len(self.__catalogue))

    def __show_or_hide(self, has_user_files, has_catalogue):
        has_any_files = has_user_files or has_catalogue
        self.totalFiles.setVisible(has_any_files)
        self.filesProcessed.setVisible(has_any_files)
        self.filesProcessedLabel.setVisible(has_any_files)
        self.ofLabel.setVisible(has_any_files)
        self.dspType.setVisible(has_any_files)
        self.dspTypeLabel.setVisible(has_any_files)
        self.processFiles.setVisible(has_any_files)
        self.errors.setVisible(has_any_files)
        self.errorsLabel.setVisible(has_any_files)

    def process_files(self):
        '''
        Creates the output content.
        '''
        self.__preferences.set(BEQ_CONFIG_FILE, self.configFile.text())
        self.__preferences.set(BEQ_MERGE_DIR, self.outputDirectory.text())
        self.__preferences.set(BEQ_MINIDSP_TYPE, self.dspType.currentText())
        self.__preferences.set(BEQ_EXTRA_DIR, self.userSourceDir.text())
        selected_channels = [item.text() for item in self.outputChannels.selectedItems()]
        self.__preferences.set(BEQ_OUTPUT_CHANNELS, "|".join(selected_channels))
        if self.outputMode.isVisible():
            self.__preferences.set(BEQ_OUTPUT_MODE, self.outputMode.currentText())
        self.filesProcessed.setValue(0)
        optimise_filters = False
        dsp_type = DspType.parse(self.dspType.currentText())
        should_process = True
        if dsp_type.is_minidsp and dsp_type.is_fixed_point_hardware():
            result = QMessageBox.question(self,
                                          'Are you feeling lucky?',
                                          f"Do you want to automatically optimise filters to fit in the 6 biquad limit? \n\n"
                                          f"Note this feature is experimental. \n"
                                          f"You are strongly encouraged to review the generated filters to ensure they are safe to use.\n"
                                          f"USE AT YOUR OWN RISK!\n\n"
                                          f"Are you sure you want to continue?",
                                          QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                          QMessageBox.StandardButton.No)
            optimise_filters = result == QMessageBox.StandardButton.Yes
        elif dsp_type.is_experimental:
            result = QMessageBox.question(self,
                                          'Generate HTP-1 Config Files?',
                                          f"Support for HTP-1 config files is experimental and currently untested on an actual device. \n\n"
                                          f"USE AT YOUR OWN RISK!\n\n"
                                          f"Are you sure you want to continue?",
                                          QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                          QMessageBox.StandardButton.No)
            should_process = result == QMessageBox.StandardButton.Yes
        if should_process:
            self.__start_spinning()
            self.errors.clear()
            self.errors.setEnabled(False)
            self.copyErrorsButton.setEnabled(False)
            self.optimised.clear()
            self.optimised.setEnabled(False)
            self.copyOptimisedButton.setEnabled(False)
            self.optimised.setVisible(optimise_filters)
            self.copyOptimisedButton.setVisible(optimise_filters)
            self.optimisedLabel.setVisible(optimise_filters)
            in_out_split = None
            if self.outputMode.isVisible() and not self.outputChannels.isVisible():
                import re
                m = re.search('Input ([1-9]) / Output ([1-9])', self.outputMode.currentText())
                if m:
                    in_out_split = (m.group(1), m.group(2))
            QThreadPool.globalInstance().start(XmlProcessor(self.__beq_dir,
                                                            self.__catalogue,
                                                            self.userSourceDir.text(),
                                                            self.outputDirectory.text(),
                                                            self.configFile.text(),
                                                            dsp_type,
                                                            self.__on_file_fail,
                                                            self.__on_file_ok,
                                                            self.__on_complete,
                                                            self.__on_optimised,
                                                            optimise_filters,
                                                            selected_channels,
                                                            in_out_split))

    def __on_file_fail(self, dir_name, file, message):
        self.errors.setEnabled(True)
        self.copyErrorsButton.setEnabled(True)
        self.errors.addItem(f"{dir_name} - {file} - {message}")

    def __on_file_ok(self):
        self.filesProcessed.setValue(self.filesProcessed.value() + 1)

    def __on_complete(self, last_updated: int):
        self.__stop_spinning()
        if last_updated:
            Path(self.outputDirectory.text(), '.index').write_text(str(last_updated))

    def __on_optimised(self, dir_name, file):
        self.optimised.setEnabled(True)
        self.copyOptimisedButton.setEnabled(True)
        self.optimised.addItem(f"{dir_name} - {file}")
        self.filesProcessed.setValue(self.filesProcessed.value() + 1)

    def __stop_spinning(self):
        from model.batch import stop_spinner
        stop_spinner(self.__spinner, self.processFiles)
        self.__spinner = None
        self.processFiles.setIcon(qta.icon('fa5s.save'))
        self.processFiles.setEnabled(True)

    def __start_spinning(self):
        self.__spinner = qta.Spin(self.processFiles)
        spin_icon = qta.icon('fa5s.spinner', color='green', animation=self.__spinner)
        self.processFiles.setIcon(spin_icon)
        self.processFiles.setEnabled(False)

    def __clear_output_directory(self):
        '''
        Empties the output directory if required
        '''
        import glob
        dsp_type: DspType = DspType.parse(self.dspType.currentText())
        matching_files = glob.glob(f"{self.outputDirectory.text()}/**/*.{dsp_type.extension}", recursive=True)
        if len(matching_files) > 0:
            result = QMessageBox.question(self,
                                          'Clear Directory',
                                          f"All generated config files will be deleted from "
                                          f"{self.outputDirectory.text()}\nAre you sure you want to continue?",
                                          QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                          QMessageBox.StandardButton.No)
            if result == QMessageBox.StandardButton.Yes:
                for file in matching_files:
                    self.statusbar.showMessage(f"Deleting {file}", 2000)
                    os.remove(file)
                    self.statusbar.showMessage(f"Deleted {file}", 2000)
                self.statusbar.showMessage(
                    f"Cleared {len(matching_files)} config files from {self.outputDirectory.text()}", 5000)
                return True
            else:
                return False
        else:
            return True

    def pick_output_dir(self):
        '''
        Sets the output directory.
        '''
        dialog = QFileDialog(parent=self)
        dialog.setFileMode(QFileDialog.FileMode.DirectoryOnly)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly)
        dialog.setWindowTitle('Select a location to store the generated dsp config files')
        if dialog.exec():
            selected = dialog.selectedFiles()
            if len(selected) > 0:
                if os.path.abspath(selected[0]) == os.path.abspath(self.__beq_dir):
                    warning = f"Output directory cannot be inside the input directory, choose a different folder"
                    QMessageBox.critical(self, '', warning, QMessageBox.Ok)
                else:
                    suffix = 'minidsp' if DspType.parse(self.dspType.currentText()).is_minidsp else 'config'
                    abspath = os.path.abspath(f"{selected[0]}{os.path.sep}beq_{suffix}")
                    if not os.path.exists(abspath):
                        try:
                            os.mkdir(abspath)
                        except:
                            QMessageBox.critical(self, '', f"Unable to create directory - {abspath}", QMessageBox.Ok)
                    if os.path.exists(abspath):
                        self.outputDirectory.setText(abspath)
        self.__enable_process()

    def clear_user_source_dir(self):
        self.userSourceDir.clear()
        self.update_beq_count()

    def pick_user_source_dir(self):
        '''
        Sets the user source directory.
        '''
        dialog = QFileDialog(parent=self)
        dialog.setFileMode(QFileDialog.FileMode.DirectoryOnly)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly)
        dialog.setWindowTitle('Choose a directory which holds your own BEQ files')
        if dialog.exec():
            selected = dialog.selectedFiles()
            if len(selected) > 0:
                if os.path.abspath(selected[0]) == os.path.abspath(self.__beq_dir):
                    QMessageBox.critical(self, '',
                                         f"User directory cannot be inside the input directory, choose a different folder",
                                         QMessageBox.Ok)
                else:
                    self.userSourceDir.setText(selected[0])
                    self.update_beq_count()

    def pick_config_file(self):
        '''
        Picks the master config file.
        :return: the file.
        '''
        kwargs = {}
        if self.configFile.text() is not None and len(self.configFile.text()) > 0:
            kwargs['directory'] = str(Path(self.configFile.text()).parent.resolve())
        dsp_type = DspType.parse(self.dspType.currentText())
        if dsp_type.is_minidsp is True:
            kwargs['caption'] = 'Select Minidsp XML Filter'
            kwargs['filter'] = 'Filter (*.xml)'
        elif dsp_type.name.startswith('JRIVER'):
            kwargs['caption'] = 'Select JRiver Media Centre DSP File'
            kwargs['filter'] = 'DSP (*.dsp)'
        else:
            kwargs['caption'] = 'Select HTP-1 Config File'
            kwargs['filter'] = 'Config (*.json)'
        selected = QFileDialog.getOpenFileName(parent=self, **kwargs)
        if selected is not None and len(selected[0]) > 0:
            self.configFile.setText(os.path.abspath(selected[0]))
        self.__enable_process()

    def __enable_process(self):
        '''
        Enables the process button if we're ready to go.
        '''
        enable = os.path.exists(self.configFile.text()) and os.path.exists(self.outputDirectory.text())
        self.processFiles.setEnabled(enable)

    def copy_optimised(self):
        '''
        Copy the optimised files to the clipboard
        '''
        self.__to_clipboard(self.optimised)

    def copy_errors(self):
        '''
        Copy the errored files to the clipboard
        '''
        self.__to_clipboard(self.errors)

    @staticmethod
    def __to_clipboard(widget):
        QGuiApplication.clipboard().setText('\n'.join([widget.item(i).text() for i in range(widget.count())]))

    def dsp_type_changed(self, selected):
        '''
        Show or hide the output channel selector.
        :param selected: the dsp type
        '''
        dsp_type = DspType.parse(selected)
        if dsp_type in OUTPUT_CHANNELS_BY_DEVICE:
            self.outputChannels.clear()
            for c in OUTPUT_CHANNELS_BY_DEVICE[dsp_type]:
                self.outputChannels.addItem(c)
            self.outputChannels.setVisible(True)
            self.outputChannelsLabel.setVisible(True)
        else:
            self.outputChannels.setVisible(False)
            self.outputChannelsLabel.setVisible(False)
        saved_channels = self.__preferences.get(BEQ_OUTPUT_CHANNELS) if selected == self.__preferences.get(
            BEQ_MINIDSP_TYPE) else None
        if saved_channels is not None:
            selected_channels = saved_channels.split('|')
        elif dsp_type in DEFAULT_OUTPUT_CHANNELS_BY_DEVICE:
            selected_channels = DEFAULT_OUTPUT_CHANNELS_BY_DEVICE[dsp_type]
        else:
            selected_channels = []
        for i in range(self.outputChannels.count()):
            w: QListWidgetItem = self.outputChannels.item(i)
            if w.text() in selected_channels:
                w.setSelected(True)
        if dsp_type.can_split is True:
            self.outputMode.setVisible(True)
            self.outputModeLabel.setVisible(True)
            self.outputMode.clear()
            self.outputMode.addItem('Overwrite')
            for i in reversed(range(1, dsp_type.filters_required)):
                self.outputMode.addItem(f"Input {i} / Output {dsp_type.filters_required - i}")
            saved = self.__preferences.get(BEQ_OUTPUT_MODE)
            if saved is not None:
                self.outputMode.setCurrentText(saved)
            else:
                self.outputMode.setCurrentText('Overwrite')
        else:
            self.outputMode.setVisible(False)
            self.outputModeLabel.setVisible(False)

    def output_mode_changed(self, txt):
        if txt == 'Overwrite':
            self.outputChannels.setVisible(True)
            self.outputChannelsLabel.setVisible(True)
        else:
            self.outputChannels.setVisible(False)
            self.outputChannelsLabel.setVisible(False)


class DspType(Enum):
    MINIDSP_TWO_BY_FOUR_HD = ('2x4 HD', True, True, False, (('1', '2'), ('3', '4', '5', '6')), 'xml')
    MINIDSP_TWO_BY_FOUR = ('2x4', False, True, False, None, 'xml')
    MINIDSP_TEN_BY_TEN = ('10x10', False, True, False, None, 'xml')
    MINIDSP_SHD = ('SHD', True, True, False, None, 'xml')
    MINIDSP_EIGHTY_EIGHT_BM = ('88BM', True, True, False, None, 'xml')
    MINIDSP_HTX = ('HTx', True, True, False, None, 'xml')
    MONOPRICE_HTP1 = ('HTP-1', False, False, False, None, 'json')
    JRIVER_PEQ1 = ('JRiver PEQ1', False, False, False, None, 'dsp')
    JRIVER_PEQ2 = ('JRiver PEQ2', False, False, False, None, 'dsp')

    def __init__(self, display_name, hd_compatible, is_minidsp, is_experimental, split_channels, ext):
        self.display_name = display_name
        self.hd_compatible = hd_compatible
        self.is_minidsp = is_minidsp
        self.is_experimental = is_experimental
        self.split_channels = split_channels
        self.extension = ext

    @property
    def can_split(self):
        return self.split_channels is not None

    def is_fixed_point_hardware(self):
        return not self.hd_compatible

    @property
    def filters_required(self):
        '''
        :return: the no of filter slots expected.
        '''
        return 10 if self.hd_compatible else 6

    @property
    def target_fs(self):
        '''
        :return: the fs for the selected minidsp.
        '''
        return 96000 if self.hd_compatible else 48000

    @classmethod
    def parse(cls, dsp_type):
        return next((t for t in cls if t.display_name == dsp_type), None)

    @property
    def filter_channels(self):
        '''
        :return: list of valid channels.
        '''
        if self == DspType.MINIDSP_TEN_BY_TEN:
            return [str(x) for x in range(11, 19)]
        elif self == DspType.MINIDSP_SHD:
            return ['1', '2', '3', '4']
        elif self == DspType.MINIDSP_EIGHTY_EIGHT_BM:
            return ['3']
        elif self == DspType.MINIDSP_HTX:
            return [str(x) for x in range(1, 9)]
        else:
            return ['1', '2']

    @property
    def input_channel_count(self):
        return 0 if self == DspType.MINIDSP_HTX else 2


OUTPUT_CHANNELS_BY_DEVICE = {
    DspType.MONOPRICE_HTP1: ['sub1', 'sub2', 'sub3', 'sub4', 'sub5', 'lf', 'rf', 'c', 'ls', 'rs', 'lb', 'rb', 'ltf',
                             'rtf', 'ltm', 'rtm', 'ltr', 'rtr', 'lw', 'rw', 'lfh', 'rfh', 'lhb', 'rhb'],
    DspType.MINIDSP_TWO_BY_FOUR_HD: ['Input 1', 'Input 2', 'Output 1', 'Output 2', 'Output 3', 'Output 4'],
    DspType.MINIDSP_EIGHTY_EIGHT_BM: [str(i + 1) for i in range(8)],
    DspType.JRIVER_PEQ1: JRIVER_CHANNELS,
    DspType.JRIVER_PEQ2: JRIVER_CHANNELS,
    DspType.MINIDSP_HTX: [f'Output {i}' for i in range(1, 9)]
}

DEFAULT_OUTPUT_CHANNELS_BY_DEVICE = {
    DspType.MONOPRICE_HTP1: ['sub1'],
    DspType.MINIDSP_TWO_BY_FOUR_HD: ['Input 1', 'Input 2'],
    DspType.MINIDSP_EIGHTY_EIGHT_BM: ['3'],
    DspType.JRIVER_PEQ1: ['Subwoofer'],
    DspType.JRIVER_PEQ2: ['Subwoofer'],
    DspType.MINIDSP_HTX: ['Output 1', 'Output 2']
}


class XmlProcessor(QRunnable):
    '''
    Completes the batch conversion of config files in a separate thread.
    '''

    def __init__(self, beq_dir, catalogue: List[CatalogueEntry], user_source_dir, output_dir,
                 config_file, dsp_type, failure_handler, success_handler, complete_handler, optimise_handler,
                 optimise_filters, selected_channels, in_out_split):
        super().__init__()
        self.__optimise_filters = optimise_filters
        self.__catalogue = catalogue
        self.__beq_dir = beq_dir
        self.__user_source_dir = user_source_dir
        self.__output_dir = output_dir
        self.__config_file = config_file
        self.__dsp_type = dsp_type
        if DspType.MONOPRICE_HTP1 == self.__dsp_type:
            self.__parser = HTP1Parser(selected_channels)
        elif DspType.JRIVER_PEQ1 == self.__dsp_type:
            from model.jriver.parser import JRiverParser
            self.__parser = JRiverParser(0, selected_channels)
        elif DspType.JRIVER_PEQ2 == self.__dsp_type:
            from model.jriver.parser import JRiverParser
            self.__parser = JRiverParser(1, selected_channels)
        elif DspType.MINIDSP_TWO_BY_FOUR == self.__dsp_type:
            self.__parser = TwoByFourXmlParser(self.__dsp_type, self.__optimise_filters)
        else:
            self.__parser = HDXmlParser(self.__dsp_type, self.__optimise_filters, selected_channels=selected_channels,
                                        in_out_split=in_out_split)
        self.__signals = ProcessSignals()
        self.__signals.on_failure.connect(failure_handler)
        self.__signals.on_success.connect(success_handler)
        self.__signals.on_complete.connect(complete_handler)
        self.__signals.on_optimised.connect(optimise_handler)

    def run(self):
        self.__process_dir(self.__user_source_dir)
        last_updated = self.__process_catalogue(self.__catalogue)
        self.__signals.on_complete.emit(last_updated)

    def __process_dir(self, src_dir):
        if len(src_dir) > 0:
            beq_dir = Path(src_dir)
            base_parts_idx = len(beq_dir.parts)
            for xml in beq_dir.glob(f"**{os.sep}*.xml"):
                self.__process_file(base_parts_idx, xml)

    def __process_catalogue(self, catalogue: List[CatalogueEntry]) -> int:
        by_author_by_filename = defaultdict(lambda: defaultdict(list))
        last_updated = 0
        for entry in catalogue:
            year_suffix = f'_{entry.year}' if entry.year else ''
            edition_suffix = f'_{entry.edition.strip()}' if entry.edition.strip() else ''
            fn = f"{entry.formatted_title}{edition_suffix}{year_suffix}"
            if entry.audio_types:
                fn = f"{fn}_{'_'.join(entry.audio_types)}"
            by_author_by_filename[entry.author][fn.replace('/', '_').replace('.', '')].append(entry)
            last_updated = max(entry.updated_at, last_updated)

        for author, values in by_author_by_filename.items():
            for filename, entries in values.items():
                file_output_dir = os.path.join(self.__output_dir, entries[0].content_type, author,
                                               entries[0].formatted_title[0].upper())
                os.makedirs(file_output_dir, exist_ok=True)
                val_provider = None
                if len(entries) > 1:
                    unique_vals = {e.source for e in entries}
                    if len(unique_vals) != len(entries):
                        unique_vals = {e.mv_adjust for e in entries}
                        if len(unique_vals) != len(entries):
                            unique_vals = {e.note for e in entries}
                            if len(unique_vals) != len(entries):
                                unique_vals = {e.language for e in entries}
                                if len(unique_vals) != len(entries):
                                    pass
                                else:
                                    val_provider = lambda e: e.language
                            else:
                                val_provider = lambda e: e.note
                        else:
                            val_provider = lambda e: f'MV {e.mv_adjust}'
                    else:
                        val_provider = lambda e: e.source
                    for i, e in enumerate(entries):
                        if val_provider:
                            suffix = val_provider(e)
                        else:
                            suffix = i
                            self.__signals.on_failure.emit(e.author, e.formatted_title, f"Duplicate entry")
                        self.__write_to(Path(file_output_dir).joinpath(sanitize(f'{filename}_{suffix}')).with_suffix(
                            self.__parser.file_extension()), e)
                else:
                    self.__write_to(
                        Path(file_output_dir).joinpath(sanitize(filename)).with_suffix(self.__parser.file_extension()),
                        entries[0])
        return last_updated

    def __write_to(self, dst: Path, entry: CatalogueEntry):
        try:
            if dst.is_file():
                self.__signals.on_failure.emit(entry.author, entry.formatted_title, f"File exists at {dst}")
            else:
                logger.info(f"Copying {self.__config_file} to {dst}")
                dst = shutil.copy2(self.__config_file, dst.resolve())
                output_config, was_optimised = self.__parser.convert(str(dst),
                                                                     entry.iir_filters(self.__dsp_type.target_fs))
                with dst.open('w', newline=self.__parser.newline(), encoding='utf8') as dst_file:
                    dst_file.write(output_config)
                if was_optimised is False:
                    self.__signals.on_success.emit()
                else:
                    self.__signals.on_optimised.emit(entry.author, entry.formatted_title)
        except Exception as e:
            logger.exception(f"Unexpected failure during processing of {entry.author}/{entry.formatted_title}")
            self.__signals.on_failure.emit(entry.author, entry.formatted_title, str(e))

    def __process_file(self, base_parts_idx, xml):
        '''
        Processes an individual file.
        :param base_parts_idx: the path index to start from.
        :param xml: the source xml.
        '''
        dir_parts = []
        try:
            dir_parts = xml.parts[base_parts_idx:-1]
            file_output_dir = os.path.join(self.__output_dir, *dir_parts)
            os.makedirs(file_output_dir, exist_ok=True)
            dst = Path(file_output_dir).joinpath(xml.name).with_suffix(self.__parser.file_extension())
            logger.info(f"Copying {self.__config_file} to {dst}")
            dst = shutil.copy2(self.__config_file, dst.resolve())
            filt = xml_to_filt(str(xml), fs=self.__dsp_type.target_fs)
            output_config, was_optimised = self.__parser.convert(str(dst), filt)
            with dst.open('w', newline=self.__parser.newline(), encoding='utf8') as dst_file:
                dst_file.write(output_config)
            if was_optimised is False:
                self.__signals.on_success.emit()
            else:
                self.__signals.on_optimised.emit(' - '.join(dir_parts), xml.name)
        except Exception as e:
            logger.exception(f"Unexpected failure during processing of {xml}")
            self.__signals.on_failure.emit(' - '.join(dir_parts), xml.name, str(e))


class ProcessSignals(QObject):
    on_failure = Signal(str, str, str)
    on_success = Signal()
    on_complete = Signal(int)
    on_optimised = Signal(str, str)
