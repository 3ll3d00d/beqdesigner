import logging
import math
import os
import sys

import qtawesome as qta
from qtpy.QtCore import QRegExp, Qt, QCoreApplication
from qtpy.QtGui import QRegExpValidator, QValidator, QIcon
from qtpy.QtWidgets import QDialog, QFileDialog

from model.iir import Gain
from model.minidsp import HDXmlParser, pad_with_passthrough
from model.preferences import BEQ_DOWNLOAD_DIR
from ui.postbuilder import Ui_postbuilder

logger = logging.getLogger('postbuilder')

URL_REGEX_FRAGMENTS = [
    r'^(?:http|ftp)s?://',  # http:// or https://
    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|',  # domain...
    r'localhost|',  # localhost...
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})',  # ...or ip
    r'(?::\d+)?',  # optional port
    r'(?:/?|[/?]\S+)$'
]


class CreateAVSPostDialog(QDialog, Ui_postbuilder):
    '''
    Create AVS Post dialog
    '''

    def __init__(self, parent, prefs, filter_model, selected_signal):
        super(CreateAVSPostDialog, self).__init__(parent)
        self.setupUi(self)
        self.__build_source_picker()
        self.pvaField.setValidator(UrlValidator(self.pvaField, self.pvaValid))
        self.spectrumField.setValidator(UrlValidator(self.spectrumField, self.spectrumValid))
        self.__preferences = prefs
        self.__beq_dir = self.__preferences.get(BEQ_DOWNLOAD_DIR)
        self.__filter_model = filter_model
        self.__selected_signal = selected_signal
        self.post_type_changed(0)
        self.generateButton.setEnabled(False)

    def __build_source_picker(self):
        _translate = QCoreApplication.translate
        sources = ["Apple TV+", "Amazon Prime", "Disney+", "Hulu", "iTunes", "Netflix"]
        for source in sources:
            self.sourcePicker.addItem(_translate("postbuilder", source))

    def generate_avs_post(self):
        '''
        Creates the output content.
        '''
        metadata = self.__build_metadata()
        save_name = self.__get_file_name(metadata)
        file_name = QFileDialog(self).getSaveFileName(self, 'Export BEQ Filter', f'{save_name}.xml', 'XML (*.xml)')
        if file_name:
            file_name = str(file_name[0]).strip()
            if len(file_name) > 0:
                if getattr(sys, 'frozen', False):
                    file_path = os.path.join(sys._MEIPASS, 'flat24hd.xml')
                else:
                    file_path = os.path.abspath(os.path.join(os.path.dirname('__file__'), '../xml/flat24hd.xml'))
                filters = pad_with_passthrough(self.__filter_model.filter, 96000, 10)
                output_xml = HDXmlParser('2x4 HD').overwrite(filters, file_path, metadata)
                with open(file_name, 'w') as f:
                    f.write(output_xml)

    def __get_file_name(self, metadata):
        save_name = f"{metadata['beq_title']}"
        if len(metadata['beq_year']) > 0:
            save_name += f" ({metadata['beq_year']})"
        if len(metadata['beq_season']) > 0:
            save_name += f" (Season {metadata['beq_season']})"
        if len(metadata['beq_edition']) > 0:
            save_name += f" ({metadata['beq_edition']})"
        if metadata['beq_source'] != 'Disc':
            save_name += f" ({metadata['beq_source']})"
        if self.__selected_signal is not None and not math.isclose(self.__selected_signal.offset, 0.0):
            save_name += f' ({self.__selected_signal.offset:+.1f} gain)'
            metadata['beq_gain'] = f'{self.__selected_signal.offset:+.1f}'
        if len(metadata["beq_audioTypes"]) > 0:
            save_name += f' BEQ {metadata["beq_audioTypes"][0]}'
        save_name = save_name.replace(':', '-')
        return save_name

    def build_avs_post(self):
        '''
        Creates the output content.
        '''
        metadata = self.__build_metadata()

        self.generateButton.setEnabled(self.__validate_metadata(metadata) is True)
        save_name = self.__get_file_name(metadata)
        self.fileName.setText(save_name)

        post = f"[CENTER][B]BassEQ {metadata['beq_title']}"

        if len(metadata['beq_year']) > 0:
            post += f" ({metadata['beq_year']})"

        if len(metadata['beq_season']) > 0:
            post += f" Season {metadata['beq_season']}"

        if len(metadata['beq_edition']) > 0:
            post += f" {metadata['beq_edition']}"

        if metadata['beq_source'] != 'Disc':
            post += f" {metadata['beq_source']}"

        audio_display = ' / '.join(metadata['beq_audioTypes'])
        post += f" {audio_display}[/B][/CENTER]\n\n"

        if len(metadata['beq_warning']) > 0:
            post += f'\n[SIZE="3"][COLOR="DarkRed"][B]WARNING: {metadata["beq_warning"]}[/B][/COLOR][/SIZE]\n\n'

        if len(metadata['beq_pvaURL']) > 0:
            post += f"[IMG]{metadata['beq_pvaURL']}[/IMG]\n"

        if len(metadata['beq_spectrumURL']) > 0:
            post += f"[IMG]{metadata['beq_spectrumURL']}[/IMG]"

        self.postTextEdit.clear()
        self.postTextEdit.insertPlainText(post)

    def post_type_changed(self, index):
        is_hidden = False
        if index == 1:
            is_hidden = True

        self.seasonField.setVisible(is_hidden)
        self.seasonLabel.setVisible(is_hidden)

    def __build_metadata(self):
        return {
            'beq_title': self.titleField.text().strip(),
            'beq_year': self.yearField.text().strip(),
            'beq_spectrumURL': self.spectrumField.text().strip(),
            'beq_pvaURL': self.pvaField.text().strip(),
            'beq_edition': self.editionField.text().strip(),
            'beq_season': self.seasonField.text().strip(),
            'beq_note': self.noteField.text().strip(),
            'beq_source': str(self.sourcePicker.currentText().strip()),
            'beq_warning': self.warningField.text().strip(),
            'beq_audioTypes': self.__build_audio_list()
        }

    def __build_audio_list(self):
        audio_types = []
        self.__add_audio(audio_types, self.atmosCheckBox)
        self.__add_audio(audio_types, self.truehd51CheckBox)
        self.__add_audio(audio_types, self.truehd71CheckBox)
        self.__add_audio(audio_types, self.dtsxCheckBox)
        self.__add_audio(audio_types, self.dts51CheckBox)
        self.__add_audio(audio_types, self.dts61CheckBox)
        self.__add_audio(audio_types, self.dts71CheckBox)
        self.__add_audio(audio_types, self.ddAtmosCheckBox)
        self.__add_audio(audio_types, self.ddCheckBox)
        self.__add_audio(audio_types, self.ddPlusCheckBox)
        self.__add_audio(audio_types, self.lpcm51CheckBox)
        self.__add_audio(audio_types, self.lpcm71CheckBox)

        return audio_types

    @staticmethod
    def __add_audio(audio_types, audio):
        if audio.isChecked():
            audio_types.append(audio.text())

    @staticmethod
    def __find_gain(filters):
        for filt in filters:
            if isinstance(filt, Gain):
                return filt

    def __validate_metadata(self, metadata):
        valid = True
        if len(metadata['beq_title']) < 1:
            self.titleValid.setIcon(qta.icon('fa5s.times', color='red'))
            valid = False
        else:
            self.titleValid.setIcon(qta.icon('fa5s.check', color='green'))

        if len(metadata['beq_year']) < 1:
            self.yearValid.setIcon(qta.icon('fa5s.times', color='red'))
            valid = False
        else:
            self.yearValid.setIcon(qta.icon('fa5s.check', color='green'))

        if len(metadata['beq_audioTypes']) < 1:
            self.audioValid.setIcon(qta.icon('fa5s.times', color='red'))
            valid = False
        else:
            self.audioValid.setIcon(qta.icon('fa5s.check', color='green'))

        return valid


class UrlValidator(QRegExpValidator):
    def __init__(self, parent, button):
        regex = QRegExp(''.join(URL_REGEX_FRAGMENTS))
        regex.setCaseSensitivity(Qt.CaseInsensitive)
        self.__button = button
        super().__init__(regex, parent)

    def validate(self, p_str, p_int):
        validate = super().validate(p_str, p_int)
        if validate:
            if validate[0] == QValidator.Invalid:
                self.__button.setIcon(qta.icon('fa5s.times', color='red'))
            elif validate[0] == QValidator.Acceptable:
                self.__button.setIcon(qta.icon('fa5s.check', color='green'))
            elif validate[0] == QValidator.Intermediate:
                self.__button.setIcon(QIcon())
        return validate
