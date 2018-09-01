import logging
import os

import numpy as np
from qtpy.QtWidgets import QDialog, QFileDialog, QDialogButtonBox

from model.preferences import EXTRACTION_OUTPUT_DIR
from ui.export import Ui_exportFRDDialog

logger = logging.getLogger('export')


class ExportFRDDialog(QDialog, Ui_exportFRDDialog):
    def __init__(self, preferences, signalModel, parent):
        super(ExportFRDDialog, self).__init__(parent)
        self.setupUi(self)
        self.__preferences = preferences
        self.__signal_model = signalModel
        for s in self.__signal_model:
            self.series.addItem(s.name)
        if len(self.__signal_model) == 0:
            self.__dialog.buttonBox.button(QDialogButtonBox.Save).setEnabled(False)

    def accept(self):
        '''
        Creates the FRD file.
        '''
        idx = self.series.currentIndex()
        to_export = self.__signal_model[idx]
        dir_name = QFileDialog(self).getExistingDirectory(self, 'Export FRD',
                                                          self.__preferences.get(EXTRACTION_OUTPUT_DIR),
                                                          QFileDialog.ShowDirsOnly)
        if len(dir_name) > 0:
            def __file_name(suffix):
                return os.path.join(dir_name, f"{to_export.name}_{suffix}.frd")

            header = self.__make_header(to_export)
            # TODO add phase if we have it
            xy = to_export.raw[0]
            np.savetxt(__file_name('avg'), np.transpose([xy.x, xy.y]), fmt='%8.3f', header=header)
            xy = to_export.raw[1]
            np.savetxt(__file_name('peak'), np.transpose([xy.x, xy.y]), fmt='%8.3f', header=header)
            if len(to_export.filtered) > 0:
                xy = to_export.filtered[0]
                np.savetxt(__file_name('filter_avg'), np.transpose([xy.x, xy.y]), fmt='%8.3f', header=header)
                xy = to_export.filtered[1]
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
