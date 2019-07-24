import logging
import os
import sys
import re

from qtpy.QtWidgets import QDialog, QFileDialog, QMessageBox

from model.iir import Gain
from model.minidsp import HDXmlParser, pad_with_passthrough
from model.preferences import BEQ_DOWNLOAD_DIR
from ui.postbuilder import Ui_postbuilder

logger = logging.getLogger('postbuilder')


class CreateAVSPostDialog(QDialog, Ui_postbuilder):
    '''
    Create AVS Post dialog
    '''

    def __init__(self, parent, prefs, filter_model):
        super(CreateAVSPostDialog, self).__init__(parent)
        self.setupUi(self)
        self.__preferences = prefs
        self.__beq_dir = self.__preferences.get(BEQ_DOWNLOAD_DIR)
        self.__filter_model = filter_model
        self.post_type_changed(0)

    def generate_avs_post(self):
        '''
        Creates the output content.
        '''
        metadata = self.__build_metadata()

        if not self.__validate_metadata(metadata):
            return

        if metadata['beq_source'] != 'Disc':
            save_name += f" ({metadata['beq_source']})"

        gain_filter = self.__find_gain(self.__filter_model.filter)

        if gain_filter is not None:
            save_name += f' ({gain_filter.gain:+.1f} gain)'
            metadata['beq_gain'] = f'{gain_filter.gain:+.1f}'

        save_name += f' BEQ {metadata["beq_audioTypes"][0]}'
        save_name = save_name.replace(':', '-')

        file_name = QFileDialog(self).getSaveFileName(self, 'Export BEQ Filter', f'{save_name}.xml', 'XML (*.xml)')
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

    def build_avs_post(self):
        '''
        Creates the output content.
        '''
        metadata = self.__build_metadata()

        if not self.__validate_metadata_urls(metadata):
            return

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
        post += f"{audio_display}[/B][/CENTER]\n"

        if len(metadata['beq_warning']) > 0:
            post += f'\n[SIZE="3"][COLOR="DarkRed"][B]WARNING: {metadata["beq_warning"]}[/B][/COLOR][/SIZE]\n\n'

        if len(metadata['beq_pvaURL']) > 0:
            post += f"[IMG]{metadata['beq_pvaURL']}[/IMG]\n"

        if len(metadata['beq_spectrumURL']) > 0:
            post += f"[IMG]{metadata['beq_spectrumURL']}[/IMG]"

        self.postTextEdit.clear()
        self.postTextEdit.insertPlainText(post)

    def post_type_changed(self, index):
        isHidden = False
        if index == 1:
            isHidden = True

        self.seasonField.setVisible(isHidden)
        self.seasonLabel.setVisible(isHidden)




    def __build_metadata(self):
        metadata = {'beq_title': self.titleField.text().strip(), 'beq_year': self.yearField.text().strip(),
                    'beq_spectrumURL': self.spectrumField.text().strip(), 'beq_pvaURL': self.pvaField.text().strip(),
                    'beq_edition': self.editionField.text().strip(), 'beq_season': self.seasonField.text().strip(),
                    'beq_note': self.noteField.text().strip(), 'beq_source': str(self.sourcePicker.currentText().strip()),
                    'beq_warning': self.warningField.text().strip(), 'beq_audioTypes': self.__build_audio_list()}

        return metadata

    def __build_audio_list(self):
        audioTypes = []
        self.__add_audio(audioTypes, self.atmosCheckBox)
        self.__add_audio(audioTypes, self.truehd51CheckBox)
        self.__add_audio(audioTypes, self.truehd71CheckBox)
        self.__add_audio(audioTypes, self.dtsxCheckBox)
        self.__add_audio(audioTypes, self.dts51CheckBox)
        self.__add_audio(audioTypes, self.dts61CheckBox)
        self.__add_audio(audioTypes, self.dts71CheckBox)
        self.__add_audio(audioTypes, self.ddAtmosCheckBox)
        self.__add_audio(audioTypes, self.ddCheckBox)
        self.__add_audio(audioTypes, self.ddPlusCheckBox)
        self.__add_audio(audioTypes, self.lpcm51CheckBox)
        self.__add_audio(audioTypes, self.lpcm71CheckBox)

        return audioTypes

    def __add_audio(self, audioTypes, audio):
        if audio.isChecked():
            audioTypes.append(audio.text())

    def __find_gain(self, filters):
        for filt in filters:
            if isinstance(filt, Gain):
                return filt

    def __validate_metadata(self, metadata):
        if len(metadata['beq_title']) < 1:
            QMessageBox.about(self, "Input Error", "Please enter a Title")
            return False
        elif len(metadata['beq_year']) < 1:
            QMessageBox.about(self, "Input Error", "Please enter a Year")
            return False
        elif not self.__validate_url(metadata['beq_pvaURL']):
            QMessageBox.about(self, "Input Error", "Please enter a valid PvA Graph URL")
            return False
        elif not self.__validate_url(metadata['beq_spectrumURL']):
            QMessageBox.about(self, "Input Error", "Please enter a valid Spectrum Graph URL")
            return False
        elif len(metadata['beq_audioTypes']) < 1:
            QMessageBox.about(self, "Input Error", "Please select an audio format")
            return False

        return True

    def __validate_metadata_urls(self, metadata):
        if len(metadata['beq_pvaURL']) > 0 and not self.__validate_url(metadata['beq_pvaURL']):
            QMessageBox.about(self, "Input Error", "Please enter a valid PvA Graph URL")
            self.pvaField.setFocus()
            return False
        elif len(metadata['beq_spectrumURL']) > 0 and not self.__validate_url(metadata['beq_spectrumURL']):
            QMessageBox.about(self, "Input Error", "Please enter a valid Spectrum Graph URL")
            self.spectrumField.setFocus()
            return False

        return True

    def __validate_url(self, url):
        regex = re.compile(
            r'^(?:http|ftp)s?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)

        return re.match(regex, url) is not None

