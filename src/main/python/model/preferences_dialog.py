'''
The Preferences dialog. It lives apart from `model.preferences`, whose keys, defaults and `Preferences` wrapper are
Qt-free so the headless pipeline can import them without Qt (design/pipeline-service/docker.md §10.1).
'''
import glob
import json
import os
from pathlib import Path
from typing import Optional, Callable

import matplotlib.style as style
import qtawesome as qta
from qtpy.QtWidgets import QDialog, QFileDialog, QFrame, QMessageBox, QDialogButtonBox, QLineEdit, QScrollArea, \
    QTableWidgetItem

from model.preferences import (
    ANALYSIS_AVG_WINDOW, ANALYSIS_PEAK_WINDOW, ANALYSIS_RESOLUTION, ANALYSIS_TARGET_FS, ANALYSIS_WINDOW_DEFAULT,
    BASS_MANAGEMENT_LPF_FS, BEQ_DOWNLOAD_DIR, BINARIES_FFMPEG, BINARIES_FFPROBE, BINARIES_MINIDSP_RS,
    DESIGNER_DEFAULT, DESIGNER_HTTP_ENDPOINTS, DESIGNER_QUEUE_DIR, DISPLAY_LINE_STYLE, DISPLAY_SMOOTH_GRAPHS,
    DISPLAY_SMOOTH_PRECALC, EXTRACTION_COMPRESS, EXTRACTION_DECIMATE, EXTRACTION_INCLUDE_ORIGINAL,
    EXTRACTION_INCLUDE_SUBTITLES, EXTRACTION_MIX_MONO, EXTRACTION_NOTIFICATION_SOUND, EXTRACTION_OUTPUT_DIR,
    FILTERS_DEFAULT_FREQ, FILTERS_DEFAULT_HS_FREQ, FILTERS_DEFAULT_HS_Q, FILTERS_DEFAULT_PEAK_FREQ,
    FILTERS_DEFAULT_PEAK_Q, FILTERS_DEFAULT_Q, GRAPH_EXPAND_Y, GRAPH_X_AXIS_SCALE, GRAPH_X_MAX, GRAPH_X_MIN,
    MINIDSP_RS_OPTIONS, STYLE_IMAGE_FORMAT_DEFAULT, STYLE_MATPLOTLIB_THEME, SYSTEM_CHECK_FOR_BETA_UPDATES,
    SYSTEM_CHECK_FOR_UPDATES, WINDOWS, _unregister_configured_designers, register_configured_designers)
from pipeline.designer.registry import registered_designers
from ui.preferences import Ui_preferencesDialog


class PreferencesDialog(QDialog, Ui_preferencesDialog):
    '''
    Allows user to set some basic preferences.
    '''

    def __init__(self, preferences, style_root, main_chart_limits, parent=None):
        super(PreferencesDialog, self).__init__(parent)
        self.__style_root = style_root
        self.setupUi(self)
        self.__init_analysis_window(self.avgAnalysisWindow)
        self.__init_analysis_window(self.peakAnalysisWindow)
        self.__init_themes()
        self.__preferences = preferences
        self.__main_chart_limits = main_chart_limits
        self.buttonBox.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(self.__reset)

        self.beqDirectoryPicker.setIcon(qta.icon('fa5s.folder-open'))
        self.defaultOutputDirectoryPicker.setIcon(qta.icon('fa5s.folder-open'))
        self.ffmpegDirectoryPicker.setIcon(qta.icon('fa5s.folder-open'))
        self.ffprobeDirectoryPicker.setIcon(qta.icon('fa5s.folder-open'))
        self.minidspRsPathPicker.setIcon(qta.icon('fa5s.folder-open'))
        self.extractCompleteAudioFilePicker.setIcon(qta.icon('fa5s.folder-open'))

        from model.jriver.connections import JRiverConnectionsWidget
        jriver_connections = JRiverConnectionsWidget(self.__preferences)
        jriver_scroll = QScrollArea()  # the page holds a list, a form and two editors: more than a tool box page shows
        jriver_scroll.setWidgetResizable(True)
        jriver_scroll.setFrameShape(QFrame.Shape.NoFrame)
        jriver_scroll.setWidget(jriver_connections)
        self.jriverPane.addWidget(jriver_scroll)
        jriver_connections.refresh_aliases()

        self.__init_field(BINARIES_FFMPEG, os.path.isdir, self.ffmpegDirectory)
        self.__init_field(BINARIES_FFPROBE, os.path.isdir, self.ffprobeDirectory)
        self.__init_field(BINARIES_MINIDSP_RS, os.path.isdir, self.minidspRsPath)
        self.__init_field(EXTRACTION_NOTIFICATION_SOUND, os.path.isfile, self.extractCompleteAudioFile)
        minidsp_rs_options_txt = self.__preferences.get(MINIDSP_RS_OPTIONS)
        if minidsp_rs_options_txt:
            self.minidspRsOptions.setText(minidsp_rs_options_txt)
        self.init_combo(ANALYSIS_TARGET_FS, self.targetFs,
                        translater=lambda a: 'Full Range' if a == 0 else str(a) + ' Hz')
        self.init_combo(ANALYSIS_RESOLUTION, self.resolutionSelect, translater=lambda a: str(a) + ' Hz')
        self.init_combo(ANALYSIS_AVG_WINDOW, self.avgAnalysisWindow)
        self.init_combo(ANALYSIS_PEAK_WINDOW, self.peakAnalysisWindow)
        self.init_combo(STYLE_MATPLOTLIB_THEME, self.themePicker)

        self.__init_field(EXTRACTION_OUTPUT_DIR, os.path.isdir, self.defaultOutputDirectory)

        freq_is_log = self.__preferences.get(GRAPH_X_AXIS_SCALE)
        self.freqIsLogScale.setChecked(freq_is_log == 'log')
        self.xmin.setValue(self.__preferences.get(GRAPH_X_MIN))
        self.xmax.setValue(self.__preferences.get(GRAPH_X_MAX))
        self.expandYLimits.setChecked(self.__preferences.get(GRAPH_EXPAND_Y))

        self.speclabLineStyle.setChecked(self.__preferences.get(DISPLAY_LINE_STYLE))
        self.checkForUpdates.setChecked(self.__preferences.get(SYSTEM_CHECK_FOR_UPDATES))
        self.checkForBetaUpdates.setChecked(self.__preferences.get(SYSTEM_CHECK_FOR_BETA_UPDATES))
        self.smoothGraphs.setChecked(self.__preferences.get(DISPLAY_SMOOTH_GRAPHS))
        self.imageFormat.setCurrentText(self.__preferences.get(STYLE_IMAGE_FORMAT_DEFAULT))

        self.monoMix.setChecked(self.__preferences.get(EXTRACTION_MIX_MONO))
        self.decimate.setChecked(self.__preferences.get(EXTRACTION_DECIMATE))
        self.includeOriginal.setChecked(self.__preferences.get(EXTRACTION_INCLUDE_ORIGINAL))
        self.includeSubtitles.setChecked(self.__preferences.get(EXTRACTION_INCLUDE_SUBTITLES))
        self.compress.setChecked(self.__preferences.get(EXTRACTION_COMPRESS))

        self.lsQ.setValue(self.__preferences.get(FILTERS_DEFAULT_Q))
        self.lsFreq.setValue(self.__preferences.get(FILTERS_DEFAULT_FREQ))

        self.hsQ.setValue(self.__preferences.get(FILTERS_DEFAULT_HS_Q))
        self.hsFreq.setValue(self.__preferences.get(FILTERS_DEFAULT_HS_FREQ))

        self.peakQ.setValue(self.__preferences.get(FILTERS_DEFAULT_PEAK_Q))
        self.peakFreq.setValue(self.__preferences.get(FILTERS_DEFAULT_PEAK_FREQ))

        self.beqFiltersDir.setText(self.__preferences.get(BEQ_DOWNLOAD_DIR))

        self.bmlpfFreq.setValue(self.__preferences.get(BASS_MANAGEMENT_LPF_FS))

        self.precalcSmoothing.setChecked(self.__preferences.get(DISPLAY_SMOOTH_PRECALC))

        self.designQueueDirPicker.setIcon(qta.icon('fa5s.folder-open'))
        self.__init_field(DESIGNER_QUEUE_DIR, os.path.isdir, self.designQueueDir)
        self.designersAddButton.clicked.connect(self.add_designer_row)
        self.designersRemoveButton.clicked.connect(self.remove_selected_designer_row)
        self.__load_designers()
        self.defaultDesignerCombo.addItems(registered_designers())
        self.init_combo(DESIGNER_DEFAULT, self.defaultDesignerCombo)

    def __load_designers(self):
        self.designersTable.setRowCount(0)
        for entry in self.__preferences.get(DESIGNER_HTTP_ENDPOINTS):
            self.__append_designer_row(entry.get('name', ''), entry.get('url', ''), entry.get('headers') or {})

    def __append_designer_row(self, name='', url='', headers=None):
        row = self.designersTable.rowCount()
        self.designersTable.insertRow(row)
        self.designersTable.setItem(row, 0, QTableWidgetItem(name))
        self.designersTable.setItem(row, 1, QTableWidgetItem(url))
        self.designersTable.setItem(row, 2, QTableWidgetItem(json.dumps(headers or {})))

    def add_designer_row(self):
        self.__append_designer_row()

    def remove_selected_designer_row(self):
        selection = self.designersTable.selectionModel()
        if selection.hasSelection():
            self.designersTable.removeRow(selection.selectedRows()[0].row())

    def __designer_cell_text(self, row, col):
        item = self.designersTable.item(row, col)
        return item.text().strip() if item is not None else ''

    def showDesignQueueDirPicker(self):
        dialog = QFileDialog(parent=self)
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setWindowTitle('Select Review Queue Directory')
        if dialog.exec():
            selected = dialog.selectedFiles()
            if len(selected) > 0:
                self.designQueueDir.setText(selected[0])

    def __init_field(self, pref_key: str, lookup: Callable[[str], bool], field: QLineEdit):
        loc = self.__preferences.get(pref_key)
        if loc and lookup(loc):
            field.setText(loc)

    def __reset(self):
        '''
        Reset all settings
        '''
        result = QMessageBox.question(self,
                                      'Reset Preferences?',
                                      f"All preferences will be restored to their default values. This action is irreversible.\nAre you sure you want to continue?",
                                      QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                      QMessageBox.StandardButton.No)
        if result == QMessageBox.StandardButton.Yes:
            self.__preferences.reset()
            self.alert_on_change('Defaults Restored')
            self.reject()

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

    def __save_loc(self, field: QLineEdit, test: Callable[[str], bool], pref_key: str, clear_if_unset: bool = False):
        loc = field.text()
        if len(loc) > 0 and test(loc):
            self.__preferences.set(pref_key, loc)
        elif clear_if_unset:
            self.__preferences.set(pref_key, None)

    def accept(self):
        '''
        Saves the locations if they exist.
        '''
        self.__save_loc(self.ffmpegDirectory, os.path.isdir, BINARIES_FFMPEG)
        self.__save_loc(self.ffprobeDirectory, os.path.isdir, BINARIES_FFPROBE)
        self.__save_loc(self.minidspRsPath, os.path.isdir, BINARIES_MINIDSP_RS)
        self.__save_loc(self.defaultOutputDirectory, os.path.isdir, EXTRACTION_OUTPUT_DIR)
        self.__save_loc(self.extractCompleteAudioFile, os.path.isfile, EXTRACTION_NOTIFICATION_SOUND, clear_if_unset=True)
        text = self.minidspRsOptions.text()
        if len(text) > 0:
            self.__preferences.set(MINIDSP_RS_OPTIONS, text)
        else:
            self.__preferences.set(MINIDSP_RS_OPTIONS, None)
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
            self.alert_on_change('Theme Change')
        self.__preferences.set(STYLE_MATPLOTLIB_THEME, self.themePicker.currentText())
        new_x_scale = 'log' if self.freqIsLogScale.isChecked() else 'linear'
        update_limits = False
        if self.__preferences.get(GRAPH_X_AXIS_SCALE) != new_x_scale:
            update_limits = True
        self.__preferences.set(GRAPH_X_AXIS_SCALE, new_x_scale)
        if self.xmin.value() < self.xmax.value():
            if self.__preferences.get(GRAPH_X_MIN) != self.xmin.value():
                update_limits = True
                self.__preferences.set(GRAPH_X_MIN, self.xmin.value())
            if self.__preferences.get(GRAPH_X_MAX) != self.xmax.value():
                update_limits = True
                self.__preferences.set(GRAPH_X_MAX, self.xmax.value())
        else:
            self.alert_on_change('X Axis Invalid', text='Invalid values: x_min must be less than x_max',
                                 icon=QMessageBox.Icon.Critical)
        if self.__preferences.get(GRAPH_EXPAND_Y) != self.expandYLimits.isChecked():
            update_limits = True
            self.__preferences.set(GRAPH_EXPAND_Y, self.expandYLimits.isChecked())
            self.__main_chart_limits.set_expand_y(self.expandYLimits.isChecked())
        if update_limits:
            self.__main_chart_limits.update(x_min=self.xmin.value(), x_max=self.xmax.value(),
                                            x_scale=new_x_scale, draw=True)
        self.__preferences.set(SYSTEM_CHECK_FOR_UPDATES, self.checkForUpdates.isChecked())
        self.__preferences.set(SYSTEM_CHECK_FOR_BETA_UPDATES, self.checkForBetaUpdates.isChecked())
        self.__preferences.set(DISPLAY_LINE_STYLE, self.speclabLineStyle.isChecked())
        self.__preferences.set(DISPLAY_SMOOTH_GRAPHS, self.smoothGraphs.isChecked())
        self.__preferences.set(EXTRACTION_MIX_MONO, self.monoMix.isChecked())
        self.__preferences.set(EXTRACTION_DECIMATE, self.decimate.isChecked())
        self.__preferences.set(EXTRACTION_INCLUDE_ORIGINAL, self.includeOriginal.isChecked())
        self.__preferences.set(EXTRACTION_INCLUDE_SUBTITLES, self.includeSubtitles.isChecked())
        self.__preferences.set(EXTRACTION_COMPRESS, self.compress.isChecked())
        self.__preferences.set(FILTERS_DEFAULT_FREQ, self.lsFreq.value())
        self.__preferences.set(FILTERS_DEFAULT_Q, self.lsQ.value())
        self.__preferences.set(FILTERS_DEFAULT_HS_FREQ, self.hsFreq.value())
        self.__preferences.set(FILTERS_DEFAULT_HS_Q, self.hsQ.value())
        self.__preferences.set(FILTERS_DEFAULT_PEAK_FREQ, self.peakFreq.value())
        self.__preferences.set(FILTERS_DEFAULT_PEAK_Q, self.peakQ.value())
        self.__preferences.set(BEQ_DOWNLOAD_DIR, self.beqFiltersDir.text())
        self.__preferences.set(BASS_MANAGEMENT_LPF_FS, self.bmlpfFreq.value())
        self.__preferences.set(DISPLAY_SMOOTH_PRECALC, self.precalcSmoothing.isChecked())
        self.__preferences.set(STYLE_IMAGE_FORMAT_DEFAULT, self.imageFormat.currentText())
        self.__save_loc(self.designQueueDir, os.path.isdir, DESIGNER_QUEUE_DIR)
        self.__preferences.set(DESIGNER_DEFAULT, self.defaultDesignerCombo.currentText())
        self.__save_designers()

        QDialog.accept(self)

    def __save_designers(self):
        '''
        Validates and saves the designer endpoints table -- same rule as the old standalone DesignersDialog. On
        a problem, alerts and leaves the existing DESIGNER_HTTP_ENDPOINTS preference untouched (same
        "warn and skip this section, still save everything else" convention as the X Axis Invalid check above)
        rather than blocking the whole Preferences dialog from closing.
        '''
        entries = []
        problems = []
        for row in range(self.designersTable.rowCount()):
            name = self.__designer_cell_text(row, 0)
            url = self.__designer_cell_text(row, 1)
            headers_text = self.__designer_cell_text(row, 2)
            if not name and not url:
                continue  # a blank row added then left empty -- not an error, just skipped
            if not name or not url:
                problems.append(f"row {row + 1}: both name and URL are required")
                continue
            try:
                headers = json.loads(headers_text) if headers_text else {}
                if not isinstance(headers, dict):
                    raise ValueError('headers must be a JSON object')
            except ValueError as e:
                problems.append(f"row {row + 1} ({name}): invalid headers -- {e}")
                continue
            entries.append({'name': name, 'url': url, 'headers': headers})

        names = [e['name'] for e in entries]
        if len(names) != len(set(names)):
            problems.append('designer names must be unique')

        if problems:
            QMessageBox.critical(self, 'Designer endpoints not saved', '\n'.join(problems))
            return

        self.__preferences.set(DESIGNER_HTTP_ENDPOINTS, entries)
        _unregister_configured_designers()
        register_configured_designers(self.__preferences)

    def alert_on_change(self, title, text='Change will not take effect until the application is restarted',
                        icon=QMessageBox.Icon.Warning):
        msg_box = QMessageBox()
        msg_box.setText(text)
        msg_box.setIcon(icon)
        msg_box.setWindowTitle(title)
        msg_box.exec()

    def __get_directory(self, name):
        dialog = QFileDialog(parent=self)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilter(f"{name} ({name}.exe {name})")
        dialog.setWindowTitle(f"Select {name}")
        if dialog.exec():
            selected = dialog.selectedFiles()
            if len(selected) > 0:
                return selected[0]
        return None

    def showFfmpegDirectoryPicker(self):
        dirname = self.__show_and_set_picker('ffmpeg', self.ffmpegDirectory)
        if dirname:
            if os.path.exists(os.path.join(dirname, 'ffprobe.exe')) or os.path.exists(os.path.join(dirname, 'ffprobe')):
                self.ffprobeDirectory.setText(dirname)

    def showFfprobeDirectoryPicker(self):
        dirname = self.__show_and_set_picker('ffprobe', self.ffprobeDirectory)
        if dirname:
            if os.path.exists(os.path.join(dirname, 'ffmpeg.exe')) or os.path.exists(os.path.join(dirname, 'ffmpeg')):
                self.ffmpegDirectory.setText(dirname)

    def show_minidsp_rs_picker(self):
        self.__show_and_set_picker('minidsp', self.minidspRsPath)

    def __show_and_set_picker(self, name: str, widget: QLineEdit) -> Optional[str]:
        loc = self.__get_directory(name)
        if loc is not None:
            dirname = os.path.dirname(loc)
            widget.setText(dirname)
            return dirname
        return None

    def showDefaultOutputDirectoryPicker(self):
        dialog = QFileDialog(parent=self)
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setWindowTitle(f"Select Extract Audio Output Directory")
        if dialog.exec():
            selected = dialog.selectedFiles()
            if len(selected) > 0:
                self.defaultOutputDirectory.setText(selected[0])

    def showExtractCompleteSoundPicker(self):
        dialog = QFileDialog(parent=self)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilter("Audio (*.wav)")
        dialog.setWindowTitle(f"Select Notification Sound")
        if dialog.exec():
            selected = dialog.selectedFiles()
            if len(selected) > 0:
                self.extractCompleteAudioFile.setText(selected[0])
            else:
                self.extractCompleteAudioFile.setText('')
        else:
            self.extractCompleteAudioFile.setText('')

    def showBeqDirectoryPicker(self):
        ''' selects an output directory for the beq files '''
        dialog = QFileDialog(parent=self)
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setWindowTitle(f"Select BEQ Files Download Directory")
        if dialog.exec():
            selected = dialog.selectedFiles()
            if len(selected) > 0:
                self.beqFiltersDir.setText(selected[0])
