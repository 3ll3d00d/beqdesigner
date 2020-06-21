import glob
import os
import shutil
from enum import Enum
from pathlib import Path

import qtawesome as qta
from qtpy.QtCore import QThreadPool, QDateTime, Qt, QRunnable, QObject, Signal
from qtpy.QtGui import QGuiApplication
from qtpy.QtWidgets import QDialog, QDialogButtonBox, QMessageBox, QFileDialog, QListWidgetItem

from model.minidsp import logger, RepoRefresher, get_repo_subdir, get_commit_url, TwoByFourXmlParser, HDXmlParser, \
    xml_to_filt
from model.preferences import BEQ_CONFIG_FILE, BEQ_MERGE_DIR, BEQ_MINIDSP_TYPE, BEQ_DOWNLOAD_DIR, BEQ_EXTRA_DIR, \
    BEQ_REPOS, BEQ_DEFAULT_REPO, BEQ_OUTPUT_CHANNELS
from model.sync import HTP1Parser
from ui.merge import Ui_mergeDspDialog


class MergeFiltersDialog(QDialog, Ui_mergeDspDialog):
    '''
    Merge Filters dialog
    '''

    def __init__(self, parent, prefs, statusbar):
        super(MergeFiltersDialog, self).__init__(parent)
        self.setupUi(self)
        self.buttonBox.button(QDialogButtonBox.Reset).clicked.connect(self.__force_clone)
        self.__process_spinner = None
        self.configFilePicker.setIcon(qta.icon('fa5s.folder-open'))
        self.outputDirectoryPicker.setIcon(qta.icon('fa5s.folder-open'))
        self.processFiles.setIcon(qta.icon('fa5s.save'))
        self.refreshGitRepo.setIcon(qta.icon('fa5s.sync'))
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
        extra_dir = self.__preferences.get(BEQ_EXTRA_DIR)
        if extra_dir is not None and len(extra_dir) > 0 and os.path.exists(extra_dir):
            self.userSourceDir.setText(os.path.abspath(extra_dir))
        self.__delete_legacy_dir()
        self.__beq_repos = self.__preferences.get(BEQ_REPOS).split('|')
        for r in self.__beq_repos:
            self.beqRepos.addItem(r)
        self.update_beq_count()
        self.__enable_process()

    def __delete_legacy_dir(self):
        git_metadata_dir = os.path.abspath(os.path.join(self.__beq_dir, '.git'))
        move_it = False
        if os.path.exists(git_metadata_dir):
            from dulwich import porcelain
            with porcelain.open_repo_closing(self.__beq_dir) as local_repo:
                config = local_repo.get_config()
                remote_url = config.get(('remote', 'origin'), 'url').decode()
                if remote_url == BEQ_DEFAULT_REPO:
                    move_it = True
        if move_it is True:
            logger.info(f"Migrating legacy repo location from {self.__beq_dir}")
            target_dir = os.path.abspath(os.path.join(self.__beq_dir, 'bmiller_miniDSPBEQ'))
            os.mkdir(target_dir)
            for d in os.listdir(self.__beq_dir):
                if d != 'bmiller_miniDSPBEQ':
                    src = os.path.abspath(os.path.join(self.__beq_dir, d))
                    if os.path.isdir(src):
                        dst = os.path.abspath(os.path.join(target_dir, d))
                        logger.info(f"Migrating {src} to {dst}")
                        shutil.move(src, dst)
                    else:
                        logger.info(f"Migrating {src} to {target_dir}")
                        shutil.move(src, target_dir)

    def __force_clone(self):
        if not os.path.exists(self.__beq_dir):
            refresh = True
        else:
            result = QMessageBox.question(self,
                                          'Get Clean Copy?',
                                          f"Do you want to delete {self.__beq_dir} and download a fresh copy?"
                                          f"\n\nEverything in {self.__beq_dir} will be deleted. "
                                          f"\n\nThis action is irreversible!",
                                          QMessageBox.Yes | QMessageBox.No,
                                          QMessageBox.No)
            refresh = result == QMessageBox.Yes
        if refresh is True:
            shutil.rmtree(self.__beq_dir)
            self.refresh_repo()

    def refresh_repo(self):
        from app import wait_cursor
        with wait_cursor():
            refresher = RepoRefresher(self.__beq_dir, self.__beq_repos)
            refresher.signals.on_end.connect(self.__refresh_complete)
            QThreadPool.globalInstance().start(refresher)

    def __refresh_complete(self):
        self.update_beq_repo_status(self.beqRepos.currentText())
        self.update_beq_count()
        self.filesProcessed.setValue(0)

    def update_beq_repo_status(self, repo_url):
        ''' updates the displayed state of the selected beq repo '''
        subdir = get_repo_subdir(repo_url)
        if os.path.exists(self.__beq_dir) and os.path.exists(os.path.join(self.__beq_dir, subdir, '.git')):
            self.__load_repo_metadata(repo_url)
        else:
            self.__beq_dir_not_exists(repo_url)

    def update_beq_count(self):
        git_files = 0
        user_files = 0
        for repo_url in self.__beq_repos:
            subdir = get_repo_subdir(repo_url)
            if os.path.exists(self.__beq_dir) and os.path.exists(os.path.join(self.__beq_dir, subdir, '.git')):
                git_files += len(glob.glob(f"{self.__beq_dir}{os.sep}{subdir}{os.sep}**{os.sep}*.xml", recursive=True))

        if len(self.userSourceDir.text().strip()) > 0 and os.path.exists(self.userSourceDir.text()):
            user_files = len(glob.glob(f"{self.userSourceDir.text()}{os.sep}**{os.sep}*.xml", recursive=True))

        if user_files > 0 or git_files > 0:
            self.__show_or_hide(git_files > 0, user_files > 0)
            self.totalFiles.setValue(user_files + git_files)

    def __load_repo_metadata(self, repo_url):
        from dulwich import porcelain
        subdir = get_repo_subdir(repo_url)
        repo_dir = os.path.join(self.__beq_dir, subdir)
        commit_url = get_commit_url(repo_url)
        try:
            with porcelain.open_repo_closing(repo_dir) as local_repo:
                last_commit = local_repo[local_repo.head()]
                last_commit_time_utc = last_commit.commit_time
                last_commit_qdt = QDateTime()
                last_commit_qdt.setTime_t(last_commit_time_utc)
                self.lastCommitDate.setDateTime(last_commit_qdt)
                from datetime import datetime
                import calendar
                d = datetime.utcnow()
                now_utc = calendar.timegm(d.utctimetuple())
                days_since_commit = (now_utc - last_commit_time_utc) / 60 / 60 / 24
                warning_msg = ''
                if days_since_commit > 7.0:
                    warning_msg = f"&nbsp;was {round(days_since_commit)} days ago, press the button to update -->"
                commit_link = f"{commit_url}/{last_commit.id.decode('utf-8')}"
                self.infoLabel.setText(f"<a href=\"{commit_link}\">Last Commit</a>{warning_msg}")
                self.infoLabel.setTextFormat(Qt.RichText)
                self.infoLabel.setTextInteractionFlags(Qt.TextBrowserInteraction)
                self.infoLabel.setOpenExternalLinks(True)
                self.lastCommitMessage.setPlainText(
                    f"Author: {last_commit.author.decode('utf-8')}\n\n{last_commit.message.decode('utf-8')}")
        except:
            logger.exception(f"Unable to open git repo in {self.__beq_dir}")
            self.__beq_dir_not_exists(repo_url)

    def __beq_dir_not_exists(self, repo_url):
        target_path = os.path.abspath(os.path.join(self.__beq_dir, get_repo_subdir(repo_url)))
        self.infoLabel.setText(
            f"BEQ repo not found in {target_path}, press the button to clone the repository -->")
        self.lastCommitMessage.clear()
        time = QDateTime()
        time.setMSecsSinceEpoch(0)
        self.lastCommitDate.setDateTime(time)

    def __show_or_hide(self, has_git_files, has_user_files):
        self.lastCommitDate.setVisible(has_git_files)
        has_any_files = has_git_files or has_user_files
        self.totalFiles.setVisible(has_any_files)
        self.lastCommitMessage.setVisible(has_git_files)
        self.lastUpdateLabel.setVisible(has_git_files)
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
        if self.__clear_output_directory():
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
                                              QMessageBox.Yes | QMessageBox.No,
                                              QMessageBox.No)
                optimise_filters = result == QMessageBox.Yes
            elif dsp_type.is_experimental:
                result = QMessageBox.question(self,
                                              'Generate HTP-1 Config Files?',
                                              f"Support for HTP-1 config files is experimental and currently untested on an actual device. \n\n"
                                              f"USE AT YOUR OWN RISK!\n\n"
                                              f"Are you sure you want to continue?",
                                              QMessageBox.Yes | QMessageBox.No,
                                              QMessageBox.No)
                should_process = result == QMessageBox.Yes
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
                QThreadPool.globalInstance().start(XmlProcessor(self.__beq_dir,
                                                                self.userSourceDir.text(),
                                                                self.outputDirectory.text(),
                                                                self.configFile.text(),
                                                                dsp_type,
                                                                self.__on_file_fail,
                                                                self.__on_file_ok,
                                                                self.__on_complete,
                                                                self.__on_optimised,
                                                                optimise_filters,
                                                                selected_channels))

    def __on_file_fail(self, dir_name, file, message):
        self.errors.setEnabled(True)
        self.copyErrorsButton.setEnabled(True)
        self.errors.addItem(f"{dir_name} - {file} - {message}")

    def __on_file_ok(self):
        self.filesProcessed.setValue(self.filesProcessed.value()+1)

    def __on_complete(self):
        self.__stop_spinning()

    def __on_optimised(self, dir_name, file):
        self.optimised.setEnabled(True)
        self.copyOptimisedButton.setEnabled(True)
        self.optimised.addItem(f"{dir_name} - {file}")
        self.filesProcessed.setValue(self.filesProcessed.value()+1)

    def __stop_spinning(self):
        from model.batch import stop_spinner
        stop_spinner(self.__process_spinner, self.processFiles)
        self.__process_spinner = None
        self.processFiles.setIcon(qta.icon('fa5s.save'))
        self.processFiles.setEnabled(True)

    def __start_spinning(self):
        self.__process_spinner = qta.Spin(self.processFiles)
        spin_icon = qta.icon('fa5s.spinner', color='green', animation=self.__process_spinner)
        self.processFiles.setIcon(spin_icon)
        self.processFiles.setEnabled(False)

    def __clear_output_directory(self):
        '''
        Empties the output directory if required
        '''
        import glob
        matching_files = glob.glob(f"{self.outputDirectory.text()}/**/*.xml", recursive=True)
        if len(matching_files) > 0:
            result = QMessageBox.question(self,
                                          'Clear Directory',
                                          f"All generated config files will be deleted from {self.outputDirectory.text()}\nAre you sure you want to continue?",
                                          QMessageBox.Yes | QMessageBox.No,
                                          QMessageBox.No)
            if result == QMessageBox.Yes:
                for file in matching_files:
                    self.statusbar.showMessage(f"Deleting {file}", 2000)
                    os.remove(file)
                    self.statusbar.showMessage(f"Deleted {file}", 2000)
                self.statusbar.showMessage(f"Cleared {len(matching_files)} config files from {self.outputDirectory.text()}", 5000)
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
        dialog.setFileMode(QFileDialog.DirectoryOnly)
        dialog.setOption(QFileDialog.ShowDirsOnly)
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
        dialog.setFileMode(QFileDialog.DirectoryOnly)
        dialog.setOption(QFileDialog.ShowDirsOnly)
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
        if DspType.parse(self.dspType.currentText()).is_minidsp is True:
            kwargs['caption'] = 'Select Minidsp XML Filter'
            kwargs['filter'] = 'Filter (*.xml)'
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
        if dsp_type == DspType.MONOPRICE_HTP1:
            self.outputChannels.clear()
            self.outputChannels.addItem('sub1')
            self.outputChannels.addItem('sub2')
            self.outputChannels.addItem('sub3')
            self.outputChannels.addItem('sub4')
            self.outputChannels.addItem('sub5')
            self.outputChannels.addItem('lf')
            self.outputChannels.addItem('rf')
            self.outputChannels.addItem('c')
            self.outputChannels.addItem('ls')
            self.outputChannels.addItem('rs')
            self.outputChannels.addItem('lb')
            self.outputChannels.addItem('rb')
            self.outputChannels.addItem('ltf')
            self.outputChannels.addItem('rtf')
            self.outputChannels.addItem('ltm')
            self.outputChannels.addItem('rtm')
            self.outputChannels.addItem('ltr')
            self.outputChannels.addItem('rtr')
            self.outputChannels.addItem('lw')
            self.outputChannels.addItem('rw')
            self.outputChannels.addItem('lfh')
            self.outputChannels.addItem('rfh')
            self.outputChannels.addItem('lhb')
            self.outputChannels.addItem('rhb')
            self.outputChannels.setVisible(True)
            self.outputChannelsLabel.setVisible(True)
        else:
            self.outputChannels.setVisible(False)
            self.outputChannelsLabel.setVisible(False)
        saved_channels = self.__preferences.get(BEQ_OUTPUT_CHANNELS)
        if saved_channels is not None:
            selected_channels = saved_channels.split('|')
            for i in range(self.outputChannels.count()):
                w: QListWidgetItem = self.outputChannels.item(i)
                if w.text() in selected_channels:
                    w.setSelected(True)


class DspType(Enum):
    MINIDSP_TWO_BY_FOUR_HD = ('2x4 HD', True, True, False)
    MINIDSP_TWO_BY_FOUR = ('2x4', False, True, False)
    MINIDSP_TEN_BY_TEN = ('10x10', False, True, False)
    MINIDSP_SHD = ('SHD', True, True, False)
    MINIDSP_EIGHTY_EIGHT_BM = ('88BM', True, True, False)
    MONOPRICE_HTP1 = ('HTP-1', False, False, True)

    def __init__(self, display_name, hd_compatible, is_minidsp, is_experimental):
        self.display_name = display_name
        self.hd_compatible = hd_compatible
        self.is_minidsp = is_minidsp
        self.is_experimental = is_experimental

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
            return [str(x) for x in range(11, 21)]
        elif self == DspType.MINIDSP_SHD:
            return ['1', '2', '3', '4']
        elif self == DspType.MINIDSP_EIGHTY_EIGHT_BM:
            return ['3']
        else:
            return ['1', '2']


class XmlProcessor(QRunnable):
    '''
    Completes the batch conversion of config files in a separate thread.
    '''
    def __init__(self, beq_dir, user_source_dir, output_dir, config_file, dsp_type, failure_handler,
                 success_handler, complete_handler, optimise_handler, optimise_filters, selected_channels):
        super().__init__()
        self.__optimise_filters = optimise_filters
        self.__beq_dir = beq_dir
        self.__user_source_dir = user_source_dir
        self.__output_dir = output_dir
        self.__config_file = config_file
        self.__dsp_type = dsp_type
        if DspType.MONOPRICE_HTP1 == self.__dsp_type:
            self.__parser = HTP1Parser(selected_channels)
        elif DspType.MINIDSP_TWO_BY_FOUR == self.__dsp_type:
            self.__parser = TwoByFourXmlParser(self.__dsp_type, self.__optimise_filters)
        else:
            self.__parser = HDXmlParser(self.__dsp_type, self.__optimise_filters)
        self.__signals = ProcessSignals()
        self.__signals.on_failure.connect(failure_handler)
        self.__signals.on_success.connect(success_handler)
        self.__signals.on_complete.connect(complete_handler)
        self.__signals.on_optimised.connect(optimise_handler)

    def run(self):
        self.__process_dir(self.__beq_dir)
        self.__process_dir(self.__user_source_dir)
        self.__signals.on_complete.emit()

    def __process_dir(self, src_dir):
        if len(src_dir) > 0:
            beq_dir = Path(src_dir)
            base_parts_idx = len(beq_dir.parts)
            for xml in beq_dir.glob(f"**{os.sep}*.xml"):
                self.__process_file(base_parts_idx, xml)

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
            output_config, was_optimised = self.__parser.convert(dst, filt)
            with dst.open('w') as dst_file:
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
    on_complete = Signal()
    on_optimised = Signal(str, str)