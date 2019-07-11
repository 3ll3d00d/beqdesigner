import logging
import os
import sys

from qtpy.QtWidgets import QDialog, QFileDialog

from model.iir import Gain
from model.minidsp import HDXmlParser, pad_with_passthrough
from model.preferences import BEQ_DOWNLOAD_DIR
from ui.postbuilder import Ui_postbuilder

logger = logging.getLogger('postbuilder')


class CreatePostDialog(QDialog, Ui_postbuilder):
    '''
    Create Post dialog
    '''

    def __init__(self, parent, prefs, filter_model):
        super(CreatePostDialog, self).__init__(parent)
        self.setupUi(self)
        self.__preferences = prefs
        self.__beq_dir = self.__preferences.get(BEQ_DOWNLOAD_DIR)
        self.__filter_model = filter_model

    def generate_post(self):
        '''
        Creates the output content.
        '''
        metadata = self.__build_metadata()
        audio_display = ' / '.join(metadata['beq_audioTypes'])

        season_display = ''
        source_display = ''
        post_warning = ''

        save_name = f"{metadata['beq_title']} ({metadata['beq_year']})"

        if len(metadata['beq_season']) > 0:
            season_display = f"Season {metadata['beq_season']}"
            save_name += f' ({season_display})'

        if metadata['beq_source'] != 'Disc':
            source_display = metadata['beq_source']
            save_name += f" ({metadata['beq_source']})"

        if len(metadata['beq_warning']) > 0:
            post_warning = f'\n[SIZE="3"][COLOR="DarkRed"][B]WARNING: {metadata["beq_warning"]}[/B][/COLOR][/SIZE]\n\n'

        post = f"[CENTER][B]BassEQ {metadata['beq_title']} ({metadata['beq_year']}) {season_display} {metadata['beq_edition']} {source_display} {audio_display}[/B][/CENTER]\n{post_warning}[IMG]{metadata['beq_pvaURL']}[/IMG]\n[IMG]{metadata['beq_spectrumURL']}[/IMG]"

        self.postTextEdit.clear()
        self.postTextEdit.insertPlainText(post)

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

    def __build_metadata(self):
        metadata = {'beq_title': self.titleField.text()}
        metadata['beq_year'] = self.yearField.text()
        metadata['beq_spectrumURL'] = self.spectrumField.text()
        metadata['beq_pvaURL'] = self.pvaField.text()
        metadata['beq_edition'] = self.editionField.text()
        metadata['beq_season'] = self.seasonField.text()
        metadata['beq_note'] = self.noteField.text()
        metadata['beq_source'] = str(self.sourcePicker.currentText())
        metadata['beq_warning'] = self.warningField.text()
        metadata['beq_audioTypes'] = self.__build_audio_list()

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

