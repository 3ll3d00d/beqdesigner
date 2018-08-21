import os

from PyQt5.QtWidgets import QDialog, QFileDialog

from model.signal import WINDOWS
from ui.preferences import Ui_preferencesDialog

EXTRACTION_OUTPUT_DIR = 'extraction/output_dir'
ANALYSIS_RESOLUTION = 'analysis/resolution'
ANALYSIS_TARGET_FS = 'analysis/target_fs'
ANALYSIS_AVG_WINDOW = 'analysis/avg_window'
ANALYSIS_PEAK_WINDOW = 'analysis/peak_window'
ANALYSIS_WINDOW_DEFAULT = 'Default'
BINARIES_FFPROBE = 'binaries/ffprobe'
BINARIES_FFMPEG = 'binaries/ffmpeg'

BINARIES = ['ffprobe', 'ffmpeg']


class PreferencesDialog(QDialog, Ui_preferencesDialog):
    '''
    Allows user to set some basic preferences.
    '''

    def __init__(self, settings, parent=None):
        super(PreferencesDialog, self).__init__(parent)
        self.setupUi(self)
        self.__init_analysis_window(self.avgAnalysisWindow)
        self.__init_analysis_window(self.peakAnalysisWindow)
        self.settings = settings

        ffmpegLoc = self.settings.value(BINARIES_FFMPEG)
        if ffmpegLoc:
            if os.path.isdir(ffmpegLoc):
                self.ffmpegDirectory.setText(ffmpegLoc)

        ffprobeLoc = self.settings.value(BINARIES_FFPROBE)
        if ffprobeLoc:
            if os.path.isdir(ffprobeLoc):
                self.ffprobeDirectory.setText(ffprobeLoc)

        self.init_combo(ANALYSIS_TARGET_FS, 1000, self.targetFs,
                        translater=lambda a: 'Full Range' if a == 0 else str(a) + ' Hz')
        self.init_combo(ANALYSIS_RESOLUTION, 1, self.resolutionSelect, translater=lambda a: str(a) + ' Hz')
        self.init_combo(ANALYSIS_AVG_WINDOW, ANALYSIS_WINDOW_DEFAULT, self.avgAnalysisWindow)
        self.init_combo(ANALYSIS_PEAK_WINDOW, ANALYSIS_WINDOW_DEFAULT, self.peakAnalysisWindow)

        outputDir = self.settings.value(EXTRACTION_OUTPUT_DIR)
        if outputDir:
            if os.path.isdir(outputDir):
                self.defaultOutputDirectory.setText(outputDir)

    def __init_analysis_window(self, combo):
        '''
        Adds the supported windows to the combo.
        :param combo: the combo.
        '''
        combo.addItem(ANALYSIS_WINDOW_DEFAULT)
        for w in WINDOWS:
            combo.addItem(w)

    def init_combo(self, key, default, combo, translater=lambda a: a):
        '''
        Initialises a combo box from either settings or a default value.
        :param key: the settings key.
        :param default: the default value.
        :param combo: the combo box.
        :param translater: a lambda to translate from the stored value to the display name.
        '''
        stored_value = self.settings.value(key)
        idx = -1
        if stored_value is not None:
            idx = combo.findText(translater(stored_value))
        else:
            self.settings.setValue(key, default)
        if idx == -1:
            idx = combo.findText(translater(default))
        if idx != -1:
            combo.setCurrentIndex(idx)

    def accept(self):
        '''
        Saves the locations if they exist.
        '''
        ffmpegLoc = self.ffmpegDirectory.text()
        if os.path.isdir(ffmpegLoc):
            self.settings.setValue(BINARIES_FFMPEG, ffmpegLoc)
        ffprobeLoc = self.ffprobeDirectory.text()
        if os.path.isdir(ffprobeLoc):
            self.settings.setValue(BINARIES_FFPROBE, ffprobeLoc)
        outputDir = self.defaultOutputDirectory.text()
        if os.path.isdir(outputDir):
            self.settings.setValue(EXTRACTION_OUTPUT_DIR, outputDir)
        text = self.targetFs.currentText()
        if text == 'Full Range':
            self.settings.setValue(ANALYSIS_TARGET_FS, 0)
        else:
            self.settings.setValue(ANALYSIS_TARGET_FS, int(text.split(' ')[0]))
        self.settings.setValue(ANALYSIS_RESOLUTION, float(self.resolutionSelect.currentText().split(' ')[0]))
        self.settings.setValue(ANALYSIS_AVG_WINDOW, self.avgAnalysisWindow.currentText())
        self.settings.setValue(ANALYSIS_PEAK_WINDOW, self.peakAnalysisWindow.currentText())
        QDialog.accept(self)

    def __get_directory(self, name):
        dialog = QFileDialog(parent=self)
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setNameFilter(f"{name}.exe")
        dialog.setWindowTitle(f"Select {name}.exe")
        if dialog.exec():
            selected = dialog.selectedFiles()
            if len(selected) > 0:
                return selected[0]
        return None

    def showFfmpegDirectoryPicker(self):
        loc = self.__get_directory('ffmpeg')
        if loc is not None:
            dirname = os.path.dirname(loc)
            self.ffmpegDirectory.setText(dirname)
            if os.path.exists(os.path.join(dirname, 'ffprobe.exe')):
                self.ffprobeDirectory.setText(dirname)

    def showFfprobeDirectoryPicker(self):
        loc = self.__get_directory('ffprobe')
        if loc is not None:
            dirname = os.path.dirname(loc)
            self.ffprobeDirectory.setText(dirname)
            if os.path.exists(os.path.join(dirname, 'ffmpeg.exe')):
                self.ffmpegDirectory.setText(dirname)

    def showDefaultOutputDirectoryPicker(self):
        dialog = QFileDialog(parent=self)
        dialog.setFileMode(QFileDialog.DirectoryOnly)
        dialog.setWindowTitle(f"Select Extract Audio Output Directory")
        if dialog.exec():
            selected = dialog.selectedFiles()
            if len(selected) > 0:
                self.defaultOutputDirectory.setText(selected[0])
