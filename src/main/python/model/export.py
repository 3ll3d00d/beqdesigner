import gzip
import json
import logging
import os
from enum import Enum

import numpy as np
from qtpy.QtWidgets import QDialog, QFileDialog, QDialogButtonBox

from model.codec import signaldata_to_json
from model.preferences import EXTRACTION_OUTPUT_DIR
from ui.export import Ui_exportSignalDialog

logger = logging.getLogger('export')


class Mode(Enum):
    FRD = 1
    SIGNAL = 2


class ExportSignalDialog(QDialog, Ui_exportSignalDialog):
    def __init__(self, preferences, signalModel, parent, statusbar, mode=Mode.FRD):
        super(ExportSignalDialog, self).__init__(parent)
        self.setupUi(self)
        self.__preferences = preferences
        self.__signal_model = signalModel
        self.__mode = mode
        self.__statusbar = statusbar
        for s in self.__signal_model:
            self.signal.addItem(s.name)
        if len(self.__signal_model) == 0:
            self.buttonBox.button(QDialogButtonBox.Save).setEnabled(False)

    def accept(self):
        idx = self.signal.currentIndex()
        to_export = self.__signal_model[idx]
        if self.__mode == Mode.FRD:
            self.export_frd(to_export)
        elif self.__mode == Mode.SIGNAL:
            self.export_signal(to_export)
        QDialog.accept(self)

    def export_signal(self, signal):
        '''
        Exports the signal to json.
        '''
        file_name = QFileDialog(self).getSaveFileName(self, 'Export Signal', f"{signal.name}.signal",
                                                      "BEQ Signal (*.signal)")
        file_name = str(file_name[0]).strip()
        if len(file_name) > 0:
            out = signaldata_to_json(signal)
            if not file_name.endswith('.signal'):
                file_name += '.signal'
            with gzip.open(file_name, 'wb+') as outfile:
                outfile.write(json.dumps(out).encode('utf-8'))
            self.__statusbar.showMessage(f"Saved signal {signal.name} to {file_name}")

    def export_frd(self, signal):
        '''
        Exports the signal as a set o FRDs.
        '''
        dir_name = QFileDialog(self).getExistingDirectory(self, 'Export FRD',
                                                          self.__preferences.get(EXTRACTION_OUTPUT_DIR),
                                                          QFileDialog.ShowDirsOnly)
        if len(dir_name) > 0:
            def __file_name(suffix):
                return os.path.join(dir_name, f"{signal.name}_{suffix}.frd")

            header = self.__make_header(signal)
            # TODO add phase if we have it
            xy = signal.raw[0]
            np.savetxt(__file_name('avg'), np.transpose([xy.x, xy.y]), fmt='%8.3f', header=header)
            xy = signal.raw[1]
            np.savetxt(__file_name('peak'), np.transpose([xy.x, xy.y]), fmt='%8.3f', header=header)
            if len(signal.filtered) > 0:
                xy = signal.filtered[0]
                np.savetxt(__file_name('filter_avg'), np.transpose([xy.x, xy.y]), fmt='%8.3f', header=header)
                xy = signal.filtered[1]
                np.savetxt(__file_name('filter_peak'), np.transpose([xy.x, xy.y]), fmt='%8.3f', header=header)
            QDialog.accept(self)

    def __make_header(self, to_export):
        header_lines = [
            'Exported by BEQ Designer',
            f"Signal Name: {to_export.name}",
            f"Fs: {to_export.fs}",
        ]
        if to_export.duration_hhmmss is not None:
            header_lines += [
                "from wav file",
                f"duration: {to_export.duration_hhmmss}",
                f"start: {to_export.start_hhmmss}",
                f"end: {to_export.end_hhmmss}"
            ]
        header_lines.append('frequency magnitude')
        header = '\n'.join(header_lines)
        return header
