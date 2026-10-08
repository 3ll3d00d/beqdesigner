'''
TODO R8: the playback chain is sent to the designer -- a profile's `run.bass_management` on the library path, the app's
crossover on Batch Design's -- and recorded in the queue entry for the reviewer.
'''
import os

import numpy as np
import pytest
import soundfile as sf

from pipeline.designer.contract import DesignResponse
from pipeline.designer.registry import register_designer
from pipeline.library.bass import DEFAULTS, bass_management, describe, from_preferences
from pipeline.library.run import LibraryRunConfig, run_library
from pipeline.library.source import LibraryItem
from pipeline.review import read_entry


def test_a_profile_says_what_it_differs_in_and_takes_the_contracts_defaults_for_the_rest():
    assert bass_management(None) is None
    assert bass_management({}) == DEFAULTS
    assert bass_management({'lpf_fs': 120, 'lpf_position': 'after', 'headroom_type': -10, 'clip_after': True}) == \
        {'lpf_fs': 120.0, 'lpf_position': 'After', 'headroom_type': '-10', 'clip_before': False, 'clip_after': True}


@pytest.mark.parametrize('value, says', [
    ({'lpf_fs': 0}, 'lpf_fs'), ({'lpf_fs': '80'}, 'lpf_fs'), ({'lpf_position': 'Sideways'}, 'lpf_position'),
    ({'headroom_type': 'loud'}, 'headroom_type'), ({'clip_before': 'yes'}, 'clip_before'), ({'lfe_gain': 10}, 'lfe_gain'),
    ('80 Hz', 'mapping'),
])
def test_a_wrong_setting_is_named(value, says):
    with pytest.raises(ValueError, match=says):
        bass_management(value)


def test_what_was_sent_is_said_for_a_reviewer():
    assert describe(None) == 'none sent: the designer assumed its own playback chain'
    assert describe(bass_management({'lpf_fs': 90, 'clip_before': True})) == \
        '90 Hz crossover, low-pass before, headroom WCS, clipping before'
    assert from_preferences(80, 'Off') == {**DEFAULTS, 'lpf_position': 'Off'}


class _Source:
    def __init__(self, items):
        self.items = items

    def list_items(self, **query):
        return self.items


@pytest.fixture
def capturing_designer():
    seen = []

    def design(request):
        seen.append(request)
        return DesignResponse('1.0', decline_reason='no_rolloff_detected', decline_message='nothing to restore')
    register_designer('capture', design)
    return seen


def _real_audio(monkeypatch):
    ''' Extraction is not under test: it leaves a real wav where a run expects one. '''
    def extract(session, item, item_dir, config, mono_mix=True, force=False, on_progress=None):
        os.makedirs(item_dir, exist_ok=True)
        path = os.path.join(item_dir, 'mono.wav')
        sf.write(path, np.random.default_rng(0).normal(0, 0.05, (1000 * 5, 1)), 1000, subtype='PCM_24')
        return path, False
    monkeypatch.setattr('pipeline.library.run.extract_if_needed', extract)


@pytest.mark.parametrize('profile, sent', [
    ({'lpf_fs': 100, 'headroom_type': 'WCS'}, {**DEFAULTS, 'lpf_fs': 100.0}),
    (None, None),
])
def test_the_library_sends_the_profiles_bass_management_and_the_entry_records_it(tmp_path, monkeypatch,
                                                                                capturing_designer, profile, sent):
    _real_audio(monkeypatch)
    config = LibraryRunConfig(work_dir=str(tmp_path / 'work'), queue_dir=str(tmp_path / 'queue'), designer='capture',
                              bass_management=bass_management(profile))
    item = LibraryItem(id='one', source_path='/films/one.mkv', display_name='One', fingerprint='fp')

    report = run_library(_Source([item]), config)

    assert report.designed == ['one'] and report.failed == []
    assert capturing_designer[0].bass_management == sent
    assert read_entry(str(tmp_path / 'queue'), 'one').bass_management == sent


def test_the_profiles_section_reaches_the_run(tmp_path):
    from pipeline.library.setup import run_config_from_values
    values = {'work_dir': '/w', 'queue_dir': '/q', 'designer': 'capture', 'bass_management': {'lpf_fs': 70}}
    register_designer('capture', lambda request: None)

    assert run_config_from_values(values, {}).bass_management == {**DEFAULTS, 'lpf_fs': 70.0}
    assert run_config_from_values({**values, 'bass_management': None}, {}).bass_management is None
