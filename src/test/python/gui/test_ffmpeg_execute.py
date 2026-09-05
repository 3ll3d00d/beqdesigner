'''
Safety net for model/ffmpeg.py's Executor.execute() -- the QThreadPool-backed
GUI path -- written BEFORE rewiring AudioExtractor.run() to call
Executor.run_sync() instead of duplicating its subprocess-dispatch logic
(the same "if is_remux: Popen+communicate else: ffmpeg_cmd.run(...)" shape
exists in both places today).

Needs a real Qt event loop to receive the cross-thread progress signal --
qtbot.waitUntil() pumps one.
'''
import os
import wave

import numpy as np

from model.ffmpeg import SIGNAL_COMPLETE, SIGNAL_ERROR, Executor


def _write_synthetic_5_1_wav(path, fs=48000, duration_s=1.0, channel_values=(1000, 2000, 3000, 4000, 5000, 6000)):
    n_frames = int(fs * duration_s)
    frame = np.array(channel_values, dtype=np.int16)
    data = np.tile(frame, (n_frames, 1)).astype('<i2').tobytes()
    with wave.open(path, 'wb') as w:
        w.setnchannels(6)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(data)


def test_execute_runs_ffmpeg_on_a_background_thread_and_reports_completion(qtbot, tmp_path):
    source = str(tmp_path / 'synthetic_5_1.wav')
    _write_synthetic_5_1_wav(source)
    target_dir = str(tmp_path / 'out')
    os.makedirs(target_dir, exist_ok=True)

    ex = Executor(source, target_dir, mono_mix=True, decimate_audio=True, decimate_fs=1000)
    ex.probe_file()
    ex.update_spec(0, -1, True)

    events = []
    ex.progress_handler = lambda key, value: events.append((key, value))

    ex.execute()

    qtbot.waitUntil(lambda: any(key in (SIGNAL_COMPLETE, SIGNAL_ERROR) for key, _ in events), timeout=15000)

    assert events[-1][0] == SIGNAL_COMPLETE, events
    assert os.path.isfile(ex.get_output_path())

    with wave.open(ex.get_output_path(), 'rb') as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == 1000
