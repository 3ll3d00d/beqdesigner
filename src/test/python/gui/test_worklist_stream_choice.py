'''
TODO W2: the title page's audio-stream choice lists each stream as the source described it -- codec, channel layout,
title, language, sample rate, bitrate -- not just codec and channel count.

`import ui.beq` first: see AGENTS.md gotcha 3.
'''
import ui.beq  # noqa: F401 (must come first)

import json

from qtpy.QtWidgets import QInputDialog

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
