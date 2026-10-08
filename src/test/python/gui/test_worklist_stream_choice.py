'''
TODO W2: the title page's audio-stream choice lists each stream as the source described it -- codec, channel layout,
title, language, sample rate, bitrate -- not just codec and channel count.

`import ui.beq` first: see AGENTS.md gotcha 3.
'''
import ui.beq  # noqa: F401 (must come first)

import json

from qtpy.QtWidgets import QInputDialog, QMessageBox

from pipeline.library.index import item_to_json
from pipeline.library.source import LibraryItem
from test_worklist_revise import _window
from test_worklist_title import _row

STREAMS = ({'codec': 'TrueHD Atmos', 'channels': '8', 'audio_types': ('Atmos',), 'sample_rate': '48000',
            'language': 'English', 'title': 'Original mix'},
           {'codec': 'AC-3', 'channels': '6', 'audio_types': ('DD 5.1',), 'sample_rate': '48000', 'bitrate': '640',
            'language': 'French'})


def test_the_choice_lists_every_stream_as_the_source_described_it(qtbot, tmp_path, monkeypatch):
    item = LibraryItem(id='s-one', source_path='/films/one.mkv', display_name='One', title='One',
                       audio_stream_details=STREAMS)
    window = _window(qtbot, tmp_path, entries=(), rows=[_row('s-one', 'One', 'extract', 1,
                                                             items=json.dumps([item_to_json(item)]))])
    asked = []

    def get_int(parent, title, label, value, low, high):
        asked.append((label, value, low, high))
        return value, False   # cancelled: nothing changes
    monkeypatch.setattr(QInputDialog, 'getInt', staticmethod(get_int))

    assert window._choose_audio_stream('s-one') is False

    label, value, low, high = asked[0]
    assert (value, low, high) == (1, 1, 2)
    assert label.splitlines()[1:] == ['1: TrueHD Atmos 7.1, "Original mix", English, 48 kHz',
                                      '2: AC-3 5.1, French, 48 kHz, 640 kbps']


def _listed_without_streams(qtbot, tmp_path, source_path):
    item = LibraryItem(id='s-two', source_path=source_path, display_name='Two', title='Two')
    return _window(qtbot, tmp_path, entries=(), rows=[_row('s-two', 'Two', 'extract', 1,
                                                           items=json.dumps([item_to_json(item)]))])


def test_a_source_that_listed_no_streams_has_them_read_from_the_file_then_asks(qtbot, tmp_path, monkeypatch):
    media = tmp_path / 'Two.mkv'
    media.write_bytes(b'')
    probed = [{'codec_name': 'ac3', 'channels': 2, 'sample_rate': '48000'},
              {'codec_name': 'truehd', 'profile': 'Dolby TrueHD + Dolby Atmos', 'channels': 8, 'sample_rate': '48000'}]
    monkeypatch.setattr('pipeline.orchestrate.Session.probe_audio_streams', lambda self, src, playlist_name=None: probed)
    window = _listed_without_streams(qtbot, tmp_path, str(media))
    asked = []
    monkeypatch.setattr(QInputDialog, 'getInt', staticmethod(
        lambda parent, title, label, value, low, high: (asked.append(label), (value, False))[1]))

    assert window._choose_audio_stream('s-two') is False   # nothing to choose from yet: the file is read first

    qtbot.waitUntil(lambda: bool(asked), timeout=5000)
    assert asked[0].splitlines()[1:] == ['1: AC-3 stereo, 48 kHz', '2: TrueHD Atmos 7.1, 48 kHz']
    assert len(window._index.units(['s-two'])['s-two'].audio_stream_details) == 2   # kept for next time


def test_a_file_that_cannot_be_found_is_named_with_its_title(qtbot, tmp_path, monkeypatch):
    missing = str(tmp_path / 'films' / 'Two.mkv')
    window = _listed_without_streams(qtbot, tmp_path, missing)
    warned = []
    monkeypatch.setattr(QMessageBox, 'warning', staticmethod(lambda parent, title, text: warned.append((title, text))))

    window._choose_audio_stream('s-two')

    qtbot.waitUntil(lambda: bool(warned), timeout=5000)
    title, text = warned[0]
    assert title == 'Audio streams not read'
    assert 'Two:' in text and missing in text and 'cannot be found' in text
