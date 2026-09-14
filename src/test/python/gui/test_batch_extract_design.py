'''
Safety net for model/batch.py's BatchExtractDialog -- specifically the merged Batch Extract & Design dialog
(the old, separate model/batch_design.py's BatchDesignDialog was retired and folded into this dialog as an
optional per-candidate design step + an embedded Review tab; see pipeline/README.md's "Batch design + review").
Drives the real search->extract->design flow like a user would, with the real QThreadPool (same pattern as
test_ffmpeg_execute.py/the old test_batch_design_dialog.py: no mocking the thread pool, qtbot.waitUntil() pumps
the event loop for cross-thread signals).
'''
import os
import wave

import numpy as np
import pytest
from qtpy.QtCore import QSettings
from qtpy.QtWidgets import QMessageBox

from model.batch import BatchExtractDialog
from model.preferences import Preferences
from pipeline.designer.contract import BiquadSpec, DesignCandidate, DesignResponse
from pipeline.designer.registry import register_designer, unregister_designer
from pipeline.review import read_queue

DESIGNER_NAME = 'test.batch_extract_design'
RECORDING_DESIGNER_NAME = 'test.batch_extract_design.recording'


def _fake_designer(request):
    return DesignResponse(contract_version='1.0', candidates=[
        DesignCandidate(filters=[BiquadSpec(type='low_shelf', freq_hz=18.0, gain_db=4.5, q=0.7)],
                        confidence=0.9, mv_adjust_db=4.0, method='fitted'),
    ])


@pytest.fixture(autouse=True)
def _designer():
    register_designer(DESIGNER_NAME, _fake_designer)
    yield
    unregister_designer(DESIGNER_NAME)


@pytest.fixture
def recording_designer():
    ''' Registers a designer that records every DesignRequest it's called with, for channels-field assertions. '''
    seen_requests = []

    def _recording_designer(request):
        seen_requests.append(request)
        return DesignResponse(contract_version='1.0', candidates=[
            DesignCandidate(filters=[BiquadSpec(type='low_shelf', freq_hz=18.0, gain_db=4.5, q=0.7)],
                            confidence=0.9, mv_adjust_db=4.0, method='fitted'),
        ])

    register_designer(RECORDING_DESIGNER_NAME, _recording_designer)
    yield seen_requests
    unregister_designer(RECORDING_DESIGNER_NAME)


def _make_preferences(tmp_path):
    settings = QSettings(str(tmp_path / 'settings.ini'), QSettings.Format.IniFormat)
    return Preferences(settings)


def _write_synthetic_wav(path, fs=48000, duration_s=0.25, channel_values=(1000, 2000, 3000, 4000, 5000, 6000)):
    n_frames = int(fs * duration_s)
    frame = np.array(channel_values, dtype=np.int16)
    data = np.tile(frame, (n_frames, 1)).astype('<i2').tobytes()
    with wave.open(path, 'wb') as w:
        w.setnchannels(len(channel_values))
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(data)


@pytest.fixture
def dialog(qtbot, tmp_path):
    d = BatchExtractDialog(None, _make_preferences(tmp_path))
    qtbot.addWidget(d)
    return d


def test_main_tabs_has_run_and_review_tabs(dialog):
    assert dialog.mainTabs.count() == 2
    assert dialog.mainTabs.tabText(0) == 'Run'
    assert dialog.mainTabs.tabText(1) == 'Review'
    assert dialog.mainTabs.currentIndex() == 0


def test_designer_combo_lists_registered_designers(dialog):
    names = [dialog.designerCombo.itemText(i) for i in range(dialog.designerCombo.count())]
    assert DESIGNER_NAME in names


def test_queue_dir_defaults_from_preference_on_open(qtbot, tmp_path):
    from model.preferences import DESIGNER_QUEUE_DIR
    prefs = _make_preferences(tmp_path)
    prefs.set(DESIGNER_QUEUE_DIR, str(tmp_path))

    d = BatchExtractDialog(None, prefs)
    qtbot.addWidget(d)

    assert d.queueDirEdit.text() == str(tmp_path)


def test_select_queue_dir_persists_the_choice(dialog, monkeypatch, tmp_path):
    from model.preferences import DESIGNER_QUEUE_DIR
    monkeypatch.setattr(dialog, '_BatchExtractDialog__select_dir', lambda: str(tmp_path))

    dialog.select_queue_dir()

    assert dialog.queueDirEdit.text() == str(tmp_path)
    assert dialog._BatchExtractDialog__preferences.get(DESIGNER_QUEUE_DIR) == str(tmp_path)


def test_design_controls_start_disabled_and_toggle_with_the_checkbox(dialog):
    assert dialog.designerCombo.isEnabled() is False
    assert dialog.queueDirEdit.isEnabled() is False
    assert dialog.browseQueueDirButton.isEnabled() is False

    dialog.designEnabled.setChecked(True)

    assert dialog.designerCombo.isEnabled() is True
    assert dialog.queueDirEdit.isEnabled() is True
    assert dialog.browseQueueDirButton.isEnabled() is True

    dialog.designEnabled.setChecked(False)

    assert dialog.designerCombo.isEnabled() is False
    assert dialog.queueDirEdit.isEnabled() is False
    assert dialog.browseQueueDirButton.isEnabled() is False


def test_extract_with_design_disabled_does_not_require_a_queue_dir(qtbot, dialog, tmp_path):
    ''' Design off is the pre-existing behaviour -- extract must not be gated on a queue dir in that case. '''
    source = str(tmp_path / 'in' / 'a-movie.wav')
    os.makedirs(os.path.dirname(source), exist_ok=True)
    _write_synthetic_wav(source)
    output_dir = str(tmp_path / 'out')
    os.makedirs(output_dir, exist_ok=True)
    dialog.outputDir.setText(output_dir)
    dialog.filter.setText(str(tmp_path / 'in' / '*.wav'))

    dialog.search()
    qtbot.waitUntil(lambda: dialog.extractButton.isEnabled(), timeout=15000)

    dialog.extract()

    qtbot.waitUntil(lambda: not dialog.resetButton.isEnabled(), timeout=15000)
    assert dialog.mainTabs.currentIndex() == 0
    candidate = dialog._BatchExtractDialog__candidates[0]
    assert candidate.status.name == 'COMPLETE'
    assert os.path.isfile(candidate.executor.get_output_path())


def test_extract_with_design_requires_a_queue_dir(qtbot, dialog, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(QMessageBox, 'critical', lambda *a, **k: calls.append(a))
    source = str(tmp_path / 'in' / 'a-movie.wav')
    os.makedirs(os.path.dirname(source), exist_ok=True)
    _write_synthetic_wav(source)
    dialog.outputDir.setText(str(tmp_path / 'out'))
    dialog.designEnabled.setChecked(True)
    dialog.designerCombo.setCurrentText(DESIGNER_NAME)
    dialog.filter.setText(str(tmp_path / 'in' / '*.wav'))

    dialog.search()
    qtbot.waitUntil(lambda: dialog.extractButton.isEnabled(), timeout=15000)

    dialog.extract()

    assert len(calls) == 1


def test_duplicate_filename_stems_rejected_when_design_enabled(qtbot, dialog, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(QMessageBox, 'critical', lambda *a, **k: calls.append(a))
    (tmp_path / 'a').mkdir()
    (tmp_path / 'b').mkdir()
    source_a = str(tmp_path / 'a' / 'title.wav')
    source_b = str(tmp_path / 'b' / 'title.wav')  # same stem, different directory
    _write_synthetic_wav(source_a)
    _write_synthetic_wav(source_b)
    dialog.outputDir.setText(str(tmp_path / 'out'))
    dialog.designEnabled.setChecked(True)
    dialog.designerCombo.setCurrentText(DESIGNER_NAME)
    dialog.queueDirEdit.setText(str(tmp_path / 'queue'))
    dialog.filter.setText(str(tmp_path / '*' / '*.wav'))

    dialog.search()
    qtbot.waitUntil(lambda: dialog.extractButton.isEnabled(), timeout=15000)

    dialog.extract()

    assert len(calls) == 1
    assert 'stem' in calls[0][2]


def test_extract_with_design_writes_a_pending_queue_entry_and_switches_to_the_review_tab(qtbot, dialog, tmp_path):
    source = str(tmp_path / 'in' / 'ready-player-one.wav')
    os.makedirs(os.path.dirname(source), exist_ok=True)
    _write_synthetic_wav(source)
    queue_dir = str(tmp_path / 'queue')
    output_dir = str(tmp_path / 'out')
    os.makedirs(output_dir, exist_ok=True)
    dialog.outputDir.setText(output_dir)
    dialog.designEnabled.setChecked(True)
    dialog.designerCombo.setCurrentText(DESIGNER_NAME)
    dialog.queueDirEdit.setText(queue_dir)
    dialog.filter.setText(str(tmp_path / 'in' / '*.wav'))

    dialog.search()
    qtbot.waitUntil(lambda: dialog.extractButton.isEnabled(), timeout=15000)

    dialog.extract()

    qtbot.waitUntil(lambda: dialog.mainTabs.currentIndex() == 1, timeout=30000)

    entries = read_queue(queue_dir)
    assert len(entries) == 1
    assert entries[0].id == 'ready-player-one'
    assert entries[0].status == 'pending'
    assert entries[0].candidates[0].confidence == 0.9

    # the Review tab is the same embedded ReviewQueueDialog, already pointed at queue_dir
    review = dialog._BatchExtractDialog__review
    assert review.queueDirEdit.text() == queue_dir
    assert review._ReviewQueueDialog__table_model.rowCount() == 1


def test_designer_combo_preselects_the_default_designer_preference(qtbot, tmp_path):
    from model.preferences import DESIGNER_DEFAULT
    prefs = _make_preferences(tmp_path)
    prefs.set(DESIGNER_DEFAULT, DESIGNER_NAME)

    d = BatchExtractDialog(None, prefs)
    qtbot.addWidget(d)

    assert d.designerCombo.currentText() == DESIGNER_NAME


def test_design_uses_a_mono_downmix_even_when_the_kept_file_is_multichannel(qtbot, dialog, tmp_path):
    '''
    Mix to Mono? governs only the kept extraction -- Design must not be derived from (or blocked by) it, since
    we commonly want both a multichannel file to keep and a mono one to design from.
    '''
    source = str(tmp_path / 'in' / 'ready-player-two.wav')
    os.makedirs(os.path.dirname(source), exist_ok=True)
    _write_synthetic_wav(source)
    queue_dir = str(tmp_path / 'queue')
    output_dir = str(tmp_path / 'out')
    os.makedirs(output_dir, exist_ok=True)
    dialog.outputDir.setText(output_dir)
    dialog.monoMix.setChecked(False)
    dialog.designEnabled.setChecked(True)
    dialog.designerCombo.setCurrentText(DESIGNER_NAME)
    dialog.queueDirEdit.setText(queue_dir)
    dialog.filter.setText(str(tmp_path / 'in' / '*.wav'))

    dialog.search()
    qtbot.waitUntil(lambda: dialog.extractButton.isEnabled(), timeout=15000)

    dialog.extract()

    qtbot.waitUntil(lambda: dialog.mainTabs.currentIndex() == 1, timeout=30000)

    candidate = dialog._BatchExtractDialog__candidates[0]
    with wave.open(candidate.executor.get_output_path(), 'rb') as w:
        assert w.getnchannels() > 1  # the kept file stayed multichannel, as requested

    entries = read_queue(queue_dir)
    assert len(entries) == 1
    assert entries[0].candidates[0].confidence == 0.9  # designed successfully anyway, from its own mono downmix


def test_design_sends_per_channel_data_when_the_kept_file_is_multichannel(qtbot, recording_designer, tmp_path):
    '''
    A multichannel kept extraction shouldn't just get downmixed and have its per-channel detail thrown away --
    design/designer-interface.md §2's DesignRequest.channels lets a designer tell a mastering-wide rolloff
    apart from a one-channel authoring decision, so it should travel through whenever it's available for free
    (no second ffmpeg run -- decomposed straight from the multichannel file already on disk). recording_designer
    must be registered *before* the dialog is constructed so its designerCombo lists it -- built directly here
    rather than via the shared `dialog` fixture, whose construction order relative to other fixtures isn't
    something to rely on.
    '''
    d = BatchExtractDialog(None, _make_preferences(tmp_path))
    qtbot.addWidget(d)
    source = str(tmp_path / 'in' / 'ready-player-three.wav')
    os.makedirs(os.path.dirname(source), exist_ok=True)
    _write_synthetic_wav(source)
    queue_dir = str(tmp_path / 'queue')
    output_dir = str(tmp_path / 'out')
    os.makedirs(output_dir, exist_ok=True)
    d.outputDir.setText(output_dir)
    d.monoMix.setChecked(False)
    d.designEnabled.setChecked(True)
    d.designerCombo.setCurrentText(RECORDING_DESIGNER_NAME)
    d.queueDirEdit.setText(queue_dir)
    d.filter.setText(str(tmp_path / 'in' / '*.wav'))

    d.search()
    qtbot.waitUntil(lambda: d.extractButton.isEnabled(), timeout=15000)

    d.extract()

    qtbot.waitUntil(lambda: d.mainTabs.currentIndex() == 1, timeout=30000)

    assert len(recording_designer) == 1
    channels = recording_designer[0].channels
    assert channels is not None
    assert len(channels) == 6  # the synthetic source is 6-channel
    assert all(len(samples) > 0 for samples in channels.values())


def test_design_sends_no_channels_when_the_kept_file_is_mono(qtbot, recording_designer, tmp_path):
    ''' Mix to Mono? checked -- no multichannel source was ever extracted, so there's nothing to decompose. '''
    d = BatchExtractDialog(None, _make_preferences(tmp_path))
    qtbot.addWidget(d)
    source = str(tmp_path / 'in' / 'ready-player-four.wav')
    os.makedirs(os.path.dirname(source), exist_ok=True)
    _write_synthetic_wav(source)
    queue_dir = str(tmp_path / 'queue')
    output_dir = str(tmp_path / 'out')
    os.makedirs(output_dir, exist_ok=True)
    d.outputDir.setText(output_dir)
    assert d.monoMix.isChecked() is True  # default
    d.designEnabled.setChecked(True)
    d.designerCombo.setCurrentText(RECORDING_DESIGNER_NAME)
    d.queueDirEdit.setText(queue_dir)
    d.filter.setText(str(tmp_path / 'in' / '*.wav'))

    d.search()
    qtbot.waitUntil(lambda: d.extractButton.isEnabled(), timeout=15000)

    d.extract()

    qtbot.waitUntil(lambda: d.mainTabs.currentIndex() == 1, timeout=30000)

    assert len(recording_designer) == 1
    assert recording_designer[0].channels is None
