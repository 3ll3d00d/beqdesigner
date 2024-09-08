import json
import logging
import math
import os
import sys

import qtawesome as qta
import requests
from qtpy.QtCore import QRegularExpression, Qt, QCoreApplication
from qtpy.QtGui import QRegularExpressionValidator, QValidator, QIcon
from qtpy.QtWidgets import QDialog, QFileDialog

from model.iir import Gain
from model.merge import DspType
from model.minidsp import HDXmlParser
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
        self.__build_language_picker()
        self.pvaField.setValidator(UrlValidator(self.pvaField, self.pvaValid))
        self.spectrumField.setValidator(UrlValidator(self.spectrumField, self.spectrumValid))
        self.__preferences = prefs
        self.__beq_dir = self.__preferences.get(BEQ_DOWNLOAD_DIR)
        self.__filter_model = filter_model
        self.__selected_signal = selected_signal
        self.post_type_changed(0)
        self.generateButton.setEnabled(False)
        self.posterURL = None
        self.overview = None
        self.genres = None
        self.collection = None
        self.__tmdb_spinner = None

    def __build_source_picker(self):
        _translate = QCoreApplication.translate
        sources = ["Apple TV+", "Amazon", "DC Universe", "Disney+", "HBOMax", "Hulu", "iTunes", "Netflix", "Peacock"]
        for source in sources:
            self.sourcePicker.addItem(_translate("postbuilder", source))
            
    def __build_language_picker(self):
        _translate = QCoreApplication.translate
        languages = ["Cantonese", "Danish", "English", "French", "German", "Hebrew", "Indonesian", "Italian", "Japanese", "Korean", "Mandarin", "Mayan", "Norwegian", "Portuguese", "Russian", "Spanish", "Swahili", "Vietnamese"]
        for language in languages:
            self.languagePicker.addItem(_translate("postbuilder", language))

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
                filt = self.__filter_model.filter
                output_xml, _ = HDXmlParser(DspType.MINIDSP_TWO_BY_FOUR_HD, False).convert(str(file_path), filt, metadata)
                with open(file_name, 'w+', encoding='utf-8') as f:
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
            metadata['beq_gain'] = f'{self.__selected_signal.offset:.1f}'
        if len(metadata["beq_audioTypes"]) > 0:
            save_name += f' BEQ {metadata["beq_audioTypes"][0]}'
        save_name = save_name.replace(':', '-')
        return save_name

    def autofillSortTitle(self):
        title = self.titleField.text().strip().lower()
        title = title[title.startswith("the ") and len("the "):]
        self.sortField.setText(title)

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

    def load_tmdb_info(self):
        tmdbID = self.movidDBIDField.text().strip()
        if tmdbID == '':
            self.__search_tmdb()
        else:
            self.__get_tmdb_details(tmdbID)

    def __search_tmdb(self):
        self.__start_spinning()
        url = 'https://api.themoviedb.org/3/search/movie'

        if self.postTypePicker.currentIndex() == 1:
            url = 'https://api.themoviedb.org/3/search/tv'
            params = {
                "api_key": "5e23b4412adb55e7cca19cfb9d0196b6",
                "query": self.titleField.text().strip(),
                "first_air_date_year": self.yearField.text().strip(),
                "include_adult": 'false'
            }
        else:
            params = {
                "api_key": "5e23b4412adb55e7cca19cfb9d0196b6",
                "query": self.titleField.text().strip(),
                "year": self.yearField.text().strip(),
                "include_adult": 'false'
            }

        r = requests.get(url=url, params=params)
        logger.info(r)
        self.__stop_spinning()
        if r.status_code == 200:
            jsonResutls = r.json()
            logger.info(json.dumps(jsonResutls, indent=4, sort_keys=False))
            results = jsonResutls.get("results")
            if results is not None and len(results) > 0:
                first = results[0]
                theID = first.get("id")
                if theID is not None:
                    self.movidDBIDField.setText(str(theID))
                    self.__get_tmdb_details(theID)

    def __get_tmdb_details(self, movieID):
        self.__start_spinning()
        url = f"https://api.themoviedb.org/3/movie/{movieID}"

        if self.postTypePicker.currentIndex() == 1:
            url = f"https://api.themoviedb.org/3/tv/{movieID}"
            params = {
                "api_key": "5e23b4412adb55e7cca19cfb9d0196b6",
                "append_to_response": "content_ratings",
            }
        else:
            params = {
                "api_key": "5e23b4412adb55e7cca19cfb9d0196b6",
                "append_to_response": "release_dates"
            }

        r = requests.get(url=url, params=params)
        logger.info(r)
        self.__stop_spinning()
        if r.status_code == 200:
            result = r.json()
            logger.info(json.dumps(result, indent=4, sort_keys=False))
            self.posterURL = result["poster_path"]
            self.overview = result["overview"]
            self.genres = result["genres"]

            if self.postTypePicker.currentIndex() == 1:
                title = result["name"]
                alt = result["original_name"]
                if alt != title: self.altTitleField.setText(alt)
                cr = result.get("content_ratings")
                if cr is not None:
                    results = cr["results"]
                    for item in results:
                        code = item.get("iso_3166_1")
                        if code == "US":
                            self.ratingField.setText(item.get("rating"))
                            break
            else:
                title = result["title"]
                alt = result["original_title"]
                if alt != title: self.altTitleField.setText(alt)
                self.collection = result.get("belongs_to_collection")
                runtime = result["runtime"]
                if runtime is not None: self.runtimeField.setText(str(runtime))
                cr = result.get("release_dates")
                if cr is not None:
                    results = cr["results"]
                    for result in results:
                        code = result.get("iso_3166_1")
                        if code == "US":
                            releases = result.get("release_dates")
                            for release in releases:
                                type = release.get("type")
                                if type == 3 or type == 4:
                                    self.ratingField.setText(release.get("certification"))
                                    break
                            break

            self.titleField.setText(title)
            self.autofillSortTitle()

    def __stop_spinning(self):
        from model.batch import stop_spinner
        stop_spinner(self.__tmdb_spinner, self.tmdbButton)
        self.__tmdb_spinner = None
        self.tmdbButton.setIcon(QIcon())
        self.tmdbButton.setText('Load TMDB Info')
        self.tmdbButton.setEnabled(True)

    def __start_spinning(self):
        self.__tmdb_spinner = qta.Spin(self.tmdbButton)
        spin_icon = qta.icon('fa5s.spinner', color='green', animation=self.__tmdb_spinner)
        self.tmdbButton.setIcon(spin_icon)
        self.tmdbButton.setText('Loading...')
        self.tmdbButton.setEnabled(False)

    def __build_metadata(self):
        return {
            'beq_title': self.titleField.text().strip(),
            'beq_alt_title': self.altTitleField.text().strip(),
            'beq_sortTitle': self.sortField.text().strip(),
            'beq_year': self.yearField.text().strip(),
            'beq_spectrumURL': self.spectrumField.text().strip(),
            'beq_pvaURL': self.pvaField.text().strip(),
            'beq_edition': self.editionField.text().strip(),
            'beq_season': self.seasonField.text().strip(),
            'beq_note': self.noteField.text().strip(),
            'beq_warning': self.warningField.text().strip(),
            'beq_gain': None,
            'beq_language': str(self.languagePicker.currentText().strip()),
            'beq_source': str(self.sourcePicker.currentText().strip()),
            'beq_overview': self.overview,
            'beq_rating': self.ratingField.text().strip(),
            'beq_author': self.authorField.text().strip(),
            'beq_avs': self.avsURLField.text().strip(),
            'beq_theMovieDB': self.movidDBIDField.text().strip(),
            'beq_poster': self.posterURL,
            'beq_runtime': self.runtimeField.text().strip(),
            'beq_collection': self.collection,
            'beq_audioTypes': self.__build_audio_list(),
            'beq_genres': self.genres
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


class UrlValidator(QRegularExpressionValidator):
    def __init__(self, parent, button):
        regex = QRegularExpression(''.join(URL_REGEX_FRAGMENTS))
        regex.setPatternOptions(QRegularExpression.PatternOption.CaseInsensitive)
        self.__button = button
        super().__init__(regex, parent)

    def validate(self, p_str, p_int):
        validate = super().validate(p_str, p_int)
        if validate:
            if validate[0] == QValidator.State.Invalid:
                self.__button.setIcon(qta.icon('fa5s.times', color='red'))
            elif validate[0] == QValidator.State.Acceptable:
                self.__button.setIcon(qta.icon('fa5s.check', color='green'))
            elif validate[0] == QValidator.State.Intermediate:
                self.__button.setIcon(QIcon())
        return validate
