'''
Safety net for model/extract.py's ExtractAudioDialog -- specifically the "Design filters?" option added
alongside model/batch.py's Batch Extract & Design dialog (the single-file extract dialog gets the same
optional design step: extract, then optionally design_and_queue() the result and offer to open Review).
Reuses model/batch.py's DesignJob directly (see ExtractAudioDialog.__design()'s docstring), so the mono-
downmix/per-channel-diagnostic behaviour itself is exercised by test_batch_extract_design.py; this file
only exercises this dialog's own wiring -- the checkbox/combo/queue-dir controls and the extract-then-design
sequencing -- with the real QThreadPool/ffmpeg (same pattern as test_batch_extract_design.py).
'''
import ui.beq  # noqa: F401 -- see test_filter_dialog.py's comment: must be imported before
               # model.extract (-> app.wait_cursor) to resolve their mid-file circular import.

import os
import wave

import numpy as np
import pytest
from qtpy.QtCore import QSettings
from qtpy.QtWidgets import QMessageBox

from model.extract import ExtractAudioDialog
from model.preferences import BASS_MANAGEMENT_LPF_FS, BASS_MANAGEMENT_LPF_POSITION, Preferences
from pipeline.designer.contract import BiquadSpec, DesignCandidate, DesignResponse
from pipeline.designer.registry import register_designer, unregister_designer
from pipeline.review import read_queue

DESIGNER_NAME = 'test.extract_design'


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


@pytest.fixture(autouse=True)
def _no_blocking_question(monkeypatch):
    ''' design_complete()'s "open for review now?" prompt would hang a headless test. '''
    monkeypatch.setattr(QMessageBox, 'question', lambda *a, **k: QMessageBox.StandardButton.No)


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
    from unittest.mock import MagicMock
    d = ExtractAudioDialog(None, _make_preferences(tmp_path), MagicMock())
    qtbot.addWidget(d)
    return d


def test_design_controls_start_disabled_and_toggle_with_the_checkbox(dialog):
    assert dialog.designerCombo.isEnabled() is False
    assert dialog.queueDirEdit.isEnabled() is False
    assert dialog.browseQueueDirButton.isEnabled() is False

    dialog.designEnabled.setChecked(True)

    assert dialog.designerCombo.isEnabled() is True
    assert dialog.queueDirEdit.isEnabled() is True
    assert dialog.browseQueueDirButton.isEnabled() is True


def test_design_session_uses_the_current_bass_management_settings(dialog):
    prefs = dialog._ExtractAudioDialog__preferences
    prefs.set(BASS_MANAGEMENT_LPF_FS, 90)
    prefs.set(BASS_MANAGEMENT_LPF_POSITION, 'After')
    session = dialog._ExtractAudioDialog__get_session()
    assert session.preferences.get(BASS_MANAGEMENT_LPF_FS) == 90
    assert session.preferences.get(BASS_MANAGEMENT_LPF_POSITION) == 'After'


def test_design_controls_are_hidden_in_remux_mode(qtbot, tmp_path):
    from unittest.mock import MagicMock
    d = ExtractAudioDialog(None, _make_preferences(tmp_path), MagicMock(), is_remux=True)
    qtbot.addWidget(d)

    assert d.designEnabled.isVisible() is False


def test_design_controls_hidden_when_no_designers_are_registered(qtbot, tmp_path):
    ''' The design/review fields only make sense once Preferences has at least one designer configured. '''
    from unittest.mock import MagicMock
    from pipeline.designer.registry import unregister_designer
    unregister_designer(DESIGNER_NAME)  # undoes the autouse _designer fixture for this test only

    d = ExtractAudioDialog(None, _make_preferences(tmp_path), MagicMock())
    qtbot.addWidget(d)
    d.show()

    assert d.has_designers is False
    assert d.designEnabled.isVisible() is False
    assert d.designerCombo.isVisible() is False
    assert d.queueDirLabel.isVisible() is False
    assert d.queueDirEdit.isVisible() is False
    assert d.browseQueueDirButton.isVisible() is False


def test_design_controls_shown_when_a_designer_is_registered(qtbot, dialog):
    dialog.show()

    assert dialog.has_designers is True
    assert dialog.designEnabled.isVisible() is True
    assert dialog.designerCombo.isVisible() is True
    assert dialog.queueDirLabel.isVisible() is True
    assert dialog.queueDirEdit.isVisible() is True
    assert dialog.browseQueueDirButton.isVisible() is True


def test_designer_combo_lists_registered_designers(dialog):
    names = [dialog.designerCombo.itemText(i) for i in range(dialog.designerCombo.count())]
    assert DESIGNER_NAME in names


def test_extract_with_design_enabled_writes_a_pending_queue_entry(qtbot, dialog, tmp_path):
    source = str(tmp_path / 'in' / 'ready-player-one.wav')
    os.makedirs(os.path.dirname(source), exist_ok=True)
    _write_synthetic_wav(source)
    output_dir = str(tmp_path / 'out')
    os.makedirs(output_dir, exist_ok=True)
    queue_dir = str(tmp_path / 'queue')

    dialog.targetDir.setText(output_dir)
    dialog._ExtractAudioDialog__handle_drop(source)
    qtbot.waitUntil(lambda: dialog.audioStreams.count() > 0, timeout=15000)

    dialog.designEnabled.setChecked(True)
    dialog.designerCombo.setCurrentText(DESIGNER_NAME)
    dialog.queueDirEdit.setText(queue_dir)

    dialog.accept()

    qtbot.waitUntil(lambda: dialog._ExtractAudioDialog__design_entry is not None, timeout=30000)

    entries = read_queue(queue_dir)
    assert len(entries) == 1
    assert entries[0].status == 'pending'
    assert entries[0].candidates[0].confidence == 0.9


def test_extract_with_design_disabled_does_not_design(qtbot, dialog, tmp_path):
    source = str(tmp_path / 'in' / 'ready-player-two.wav')
    os.makedirs(os.path.dirname(source), exist_ok=True)
    _write_synthetic_wav(source)
    output_dir = str(tmp_path / 'out')
    os.makedirs(output_dir, exist_ok=True)

    dialog.targetDir.setText(output_dir)
    dialog._ExtractAudioDialog__handle_drop(source)
    qtbot.waitUntil(lambda: dialog.audioStreams.count() > 0, timeout=15000)

    assert dialog.designEnabled.isChecked() is False

    dialog.accept()

    qtbot.waitUntil(lambda: dialog._ExtractAudioDialog__extracted is True, timeout=15000)

    assert dialog._ExtractAudioDialog__design_entry is None
