'''
Safety net for model/batch_design.py's BatchDesignDialog (design/http-
designer-binding-plan.md phase 4) -- the previously-missing GUI entry
point for pipeline.review.batch_design(). Runs the real QThreadPool-backed
job (same pattern as test_ffmpeg_execute.py: no mocking the thread pool,
qtbot.waitUntil() pumps the event loop for the cross-thread signal).
'''
import wave

import numpy as np
import pytest
from qtpy.QtCore import QSettings
from qtpy.QtWidgets import QMessageBox

from model.batch_design import BatchDesignDialog
from model.preferences import Preferences
from pipeline.designer.contract import BiquadSpec, DesignCandidate, DesignResponse
from pipeline.designer.registry import register_designer, unregister_designer
from pipeline.review import read_queue

DESIGNER_NAME = 'test.batch_design'


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
    prefs = _make_preferences(tmp_path)
    d = BatchDesignDialog(None, prefs)
    qtbot.addWidget(d)
    return d


def test_designer_combo_lists_registered_designers(dialog):
    names = [dialog.designerCombo.itemText(i) for i in range(dialog.designerCombo.count())]
    assert DESIGNER_NAME in names


def test_run_without_files_shows_an_error(dialog, monkeypatch):
    calls = []
    monkeypatch.setattr(QMessageBox, 'critical', lambda *a, **k: calls.append(a))

    dialog._BatchDesignDialog__run()

    assert len(calls) == 1


def test_run_without_queue_or_work_dir_shows_an_error(dialog, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(QMessageBox, 'critical', lambda *a, **k: calls.append(a))
    source = str(tmp_path / 'title.wav')
    _write_synthetic_wav(source)
    dialog.filesList.addItem(source)

    dialog._BatchDesignDialog__run()

    assert len(calls) == 1


def test_duplicate_filename_stems_rejected(dialog, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(QMessageBox, 'critical', lambda *a, **k: calls.append(a))
    (tmp_path / 'a').mkdir()
    (tmp_path / 'b').mkdir()
    source_a = str(tmp_path / 'a' / 'title.wav')
    source_b = str(tmp_path / 'b' / 'title.wav')  # same stem, different directory
    _write_synthetic_wav(source_a)
    _write_synthetic_wav(source_b)
    dialog.filesList.addItem(source_a)
    dialog.filesList.addItem(source_b)
    dialog._BatchDesignDialog__queue_dir = str(tmp_path / 'queue')
    dialog._BatchDesignDialog__work_dir = str(tmp_path / 'work')

    dialog._BatchDesignDialog__run()

    assert len(calls) == 1
    assert 'stem' in calls[0][2]


def test_remove_selected_file(dialog, tmp_path):
    dialog.filesList.addItem(str(tmp_path / 'one.wav'))
    dialog.filesList.addItem(str(tmp_path / 'two.wav'))
    dialog.filesList.setCurrentRow(0)

    dialog._BatchDesignDialog__remove_selected_file()

    assert dialog.filesList.count() == 1
    assert dialog.filesList.item(0).text() == str(tmp_path / 'two.wav')


def test_successful_run_writes_a_pending_queue_entry(qtbot, dialog, monkeypatch, tmp_path):
    monkeypatch.setattr(QMessageBox, 'question', lambda *a, **k: QMessageBox.StandardButton.No)
    source = str(tmp_path / 'ready-player-one.wav')
    _write_synthetic_wav(source)
    dialog.filesList.addItem(source)
    queue_dir = str(tmp_path / 'queue')
    dialog._BatchDesignDialog__queue_dir = queue_dir
    dialog._BatchDesignDialog__work_dir = str(tmp_path / 'work')
    dialog.designerCombo.setCurrentText(DESIGNER_NAME)

    dialog._BatchDesignDialog__run()

    qtbot.waitUntil(lambda: dialog.runButton.isEnabled(), timeout=15000)
    entries = read_queue(queue_dir)
    assert len(entries) == 1
    assert entries[0].id == 'ready-player-one'
    assert entries[0].status == 'pending'
    assert entries[0].candidates[0].confidence == 0.9


def test_progress_bar_advances_per_item(qtbot, dialog, monkeypatch, tmp_path):
    monkeypatch.setattr(QMessageBox, 'question', lambda *a, **k: QMessageBox.StandardButton.No)
    source_one = str(tmp_path / 'title-one.wav')
    source_two = str(tmp_path / 'title-two.wav')
    _write_synthetic_wav(source_one)
    _write_synthetic_wav(source_two)
    dialog.filesList.addItem(source_one)
    dialog.filesList.addItem(source_two)
    dialog._BatchDesignDialog__queue_dir = str(tmp_path / 'queue')
    dialog._BatchDesignDialog__work_dir = str(tmp_path / 'work')
    dialog.designerCombo.setCurrentText(DESIGNER_NAME)

    dialog._BatchDesignDialog__run()

    assert dialog.runProgress.maximum() == 2
    qtbot.waitUntil(lambda: dialog.runProgress.value() == 2, timeout=15000)
