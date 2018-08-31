import glob
import os
from pathlib import Path

import matplotlib.style as style
from qtpy.QtWidgets import QDialog, QFileDialog, QMessageBox

from model.signal import WINDOWS
from ui.preferences import Ui_preferencesDialog

EXTRACTION_OUTPUT_DIR = 'extraction/output_dir'
ANALYSIS_RESOLUTION = 'analysis/resolution'
ANALYSIS_TARGET_FS = 'analysis/target_fs'
ANALYSIS_WINDOW_DEFAULT = 'Default'
ANALYSIS_AVG_WINDOW = 'analysis/avg_window'
ANALYSIS_PEAK_WINDOW = 'analysis/peak_window'
BINARIES_GROUP = 'binaries'
BINARIES_FFPROBE = f"{BINARIES_GROUP}/ffprobe"
BINARIES_FFMPEG = f"{BINARIES_GROUP}/ffmpeg"
FILTERS_PRESET_x = 'filters/preset_%d'
SCREEN_GEOMETRY = 'screen/geometry'
SCREEN_WINDOW_STATE = 'screen/window_state'
STYLE_MATPLOTLIB_THEME_DEFAULT = 'default'
STYLE_MATPLOTLIB_THEME = 'style/matplotlib_theme'

DEFAULT_PREFS = {
    ANALYSIS_RESOLUTION: 1,
    ANALYSIS_TARGET_FS: 1000,
    ANALYSIS_AVG_WINDOW: ANALYSIS_WINDOW_DEFAULT,
    ANALYSIS_PEAK_WINDOW: ANALYSIS_WINDOW_DEFAULT,
    STYLE_MATPLOTLIB_THEME: STYLE_MATPLOTLIB_THEME_DEFAULT
}


class Preferences:
    def __init__(self, settings):
        self.__settings = settings

    def has(self, key):
        '''
        checks for existence of a value.
        :param key: the key.
        :return: True if we have a value.
        '''
        return self.get(key) is not None

    def get(self, key, default_if_unset=True):
        '''
        Gets the value, if any.
        :param key: the settings key.
        :param default_if_unset: if true, return a default value.
        :return: the value.
        '''
        value = self.__settings.value(key)
        if value is None and default_if_unset is True:
            return DEFAULT_PREFS.get(key, None)
        return value

    def get_all(self, prefix):
        '''
        Get all values with the given prefix.
        :param prefix: the prefix.
        :return: the values, if any.
        '''
        self.__settings.beginGroup(prefix)
        try:
            return set(filter(None.__ne__, [self.__settings.value(x) for x in self.__settings.childKeys()]))
        finally:
            self.__settings.endGroup()

    def set(self, key, value):
        '''
        sets a new value.
        :param key: the key.
        :param value:  the value.
        '''
        if value is None:
            self.__settings.remove(key)
        else:
            self.__settings.setValue(key, value)


class PreferencesDialog(QDialog, Ui_preferencesDialog):
    '''
    Allows user to set some basic preferences.
    '''

    def __init__(self, preferences, style_root, parent=None):
        super(PreferencesDialog, self).__init__(parent)
        self.__style_root = style_root
        self.setupUi(self)
        self.__init_analysis_window(self.avgAnalysisWindow)
        self.__init_analysis_window(self.peakAnalysisWindow)
        self.__init_themes()
        self.__preferences = preferences

        ffmpegLoc = self.__preferences.get(BINARIES_FFMPEG)
        if ffmpegLoc:
            if os.path.isdir(ffmpegLoc):
                self.ffmpegDirectory.setText(ffmpegLoc)

        ffprobeLoc = self.__preferences.get(BINARIES_FFPROBE)
        if ffprobeLoc:
            if os.path.isdir(ffprobeLoc):
                self.ffprobeDirectory.setText(ffprobeLoc)

        self.init_combo(ANALYSIS_TARGET_FS, self.targetFs,
                        translater=lambda a: 'Full Range' if a == 0 else str(a) + ' Hz')
        self.init_combo(ANALYSIS_RESOLUTION, self.resolutionSelect, translater=lambda a: str(a) + ' Hz')
        self.init_combo(ANALYSIS_AVG_WINDOW, self.avgAnalysisWindow)
        self.init_combo(ANALYSIS_PEAK_WINDOW, self.peakAnalysisWindow)
        self.init_combo(STYLE_MATPLOTLIB_THEME, self.themePicker)

        outputDir = self.__preferences.get(EXTRACTION_OUTPUT_DIR)
        if outputDir:
            if os.path.isdir(outputDir):
                self.defaultOutputDirectory.setText(outputDir)

    def __init_themes(self):
        '''
        Adds all the available matplotlib theme names to a combo along with our internal theme names.
        '''
        for p in glob.iglob(f"{self.__style_root}/style/mpl/*.mplstyle"):
            self.themePicker.addItem(Path(p).resolve().stem)
        for style_name in sorted(style.library.keys()):
            self.themePicker.addItem(style_name)

    def __init_analysis_window(self, combo):
        '''
        Adds the supported windows to the combo.
        :param combo: the combo.
        '''
        combo.addItem(ANALYSIS_WINDOW_DEFAULT)
        for w in WINDOWS:
            combo.addItem(w)

    def init_combo(self, key, combo, translater=lambda a: a):
        '''
        Initialises a combo box from either settings or a default value.
        :param key: the settings key.
        :param combo: the combo box.
        :param translater: a lambda to translate from the stored value to the display name.
        '''
        stored_value = self.__preferences.get(key)
        idx = -1
        if stored_value is not None:
            idx = combo.findText(translater(stored_value))
        if idx != -1:
            combo.setCurrentIndex(idx)

    def accept(self):
        '''
        Saves the locations if they exist.
        '''
        ffmpegLoc = self.ffmpegDirectory.text()
        if os.path.isdir(ffmpegLoc):
            self.__preferences.set(BINARIES_FFMPEG, ffmpegLoc)
        ffprobeLoc = self.ffprobeDirectory.text()
        if os.path.isdir(ffprobeLoc):
            self.__preferences.set(BINARIES_FFPROBE, ffprobeLoc)
        outputDir = self.defaultOutputDirectory.text()
        if os.path.isdir(outputDir):
            self.__preferences.set(EXTRACTION_OUTPUT_DIR, outputDir)
        text = self.targetFs.currentText()
        if text == 'Full Range':
            self.__preferences.set(ANALYSIS_TARGET_FS, 0)
        else:
            self.__preferences.set(ANALYSIS_TARGET_FS, int(text.split(' ')[0]))
        self.__preferences.set(ANALYSIS_RESOLUTION, float(self.resolutionSelect.currentText().split(' ')[0]))
        self.__preferences.set(ANALYSIS_AVG_WINDOW, self.avgAnalysisWindow.currentText())
        self.__preferences.set(ANALYSIS_PEAK_WINDOW, self.peakAnalysisWindow.currentText())
        current_theme = self.__preferences.get(STYLE_MATPLOTLIB_THEME)
        if current_theme is not None and current_theme != self.themePicker.currentText():
            msg_box = QMessageBox()
            msg_box.setText('Theme change will not take effect until the application is restarted')
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle('Theme Change Detected')
            msg_box.exec()
        self.__preferences.set(STYLE_MATPLOTLIB_THEME, self.themePicker.currentText())

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
