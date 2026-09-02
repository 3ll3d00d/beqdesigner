'''
Phase 1 (B2) of design/pipeline-implementation-plan.md: Executor.run_sync()
runs the built ffmpeg command synchronously, with no QThreadPool/QRunnable
involved, so a headless caller doesn't need `ex._Executor__ffmpeg_cmd.run(...)`
via name mangling any more.

Builds a real synthetic 6-channel (5.1) wav, extracts+downmixes+decimates it
via run_sync(), and checks the result against the LFE-weighted downmix
formula (MAIN=10**(-20.2/20)=0.0977, LFE=10**(-10.2/20)=0.3090) that
design/api-headless-pipeline.md §1 verified manually.
'''
import math
import os
import wave

import numpy as np
import pytest

MAIN = 10 ** (-20.2 / 20.0)
LFE = 10 ** (-10.2 / 20.0)


def _write_synthetic_5_1_wav(path, fs=48000, duration_s=1.0, channel_values=(1000, 2000, 3000, 4000, 5000, 6000)):
    '''
    A plain 6-channel PCM16 wav with a constant sample value per channel
    (FL, FR, FC, LFE, BL, BR order) and no explicit channel mask -- ffprobe
    then reports a bare `channels: 6` stream, which Executor.update_spec
    infers as '5.1' (model/ffmpeg.py's fallback when channel_layout is
    absent from the probe).
    '''
    n_frames = int(fs * duration_s)
    frame = np.array(channel_values, dtype=np.int16)
    data = np.tile(frame, (n_frames, 1)).astype('<i2').tobytes()
    with wave.open(path, 'wb') as w:
        w.setnchannels(6)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(data)


@pytest.fixture
def synthetic_5_1_wav(tmp_path):
    path = str(tmp_path / 'synthetic_5_1.wav')
    _write_synthetic_5_1_wav(path)
    return path


def test_run_sync_extracts_mono_decimated_headlessly(synthetic_5_1_wav, tmp_path):
    '''
    No QApplication anywhere in this test -- run_sync() must not need one.
    '''
    from model.ffmpeg import Executor

    target_dir = str(tmp_path / 'out')
    os.makedirs(target_dir, exist_ok=True)

    ex = Executor(synthetic_5_1_wav, target_dir, mono_mix=True, decimate_audio=True, decimate_fs=1000)
    ex.probe_file()
    assert ex.has_audio()
    ex.update_spec(0, -1, True)

    assert ex.ffmpeg_cmd is not None

    out, err = ex.run_sync()
    assert out is not None or err is not None  # ffmpeg-python returns (stdout, stderr) bytes

    output_path = ex.get_output_path()
    assert os.path.isfile(output_path)

    with wave.open(output_path, 'rb') as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == 1000
        sampwidth = w.getsampwidth()  # pcm_s24le -> 3 bytes/sample
        raw = w.readframes(w.getnframes())
        samples = _decode_pcm(raw, sampwidth)

    full_scale_in = 2 ** 15  # source wav is PCM16
    full_scale_out = 2 ** (8 * sampwidth - 1)

    # constant input per channel -> constant (weighted-sum) output, modulo
    # resample/codec rounding -- compare in normalised (-1..1) units so the
    # input's 16-bit and output's 24-bit full scales don't need to match.
    expected = (MAIN * (1000 + 2000 + 3000 + 5000 + 6000) + LFE * 4000) / full_scale_in
    unweighted = (sum([1000, 2000, 3000, 4000, 5000, 6000]) / 6) / full_scale_in
    actual = float(np.mean(samples[len(samples) // 4:3 * len(samples) // 4])) / full_scale_out  # steady-state

    assert math.isclose(actual, expected, rel_tol=0.05), (
        f"expected LFE-weighted downmix ~{expected:.4f}, got {actual:.4f} "
        f"(unweighted average would be {unweighted:.4f})")


def _decode_pcm(raw, sampwidth):
    ''' Decodes little-endian signed PCM of arbitrary (1/2/3/4-byte) width into an int32 numpy array. '''
    if sampwidth == 2:
        return np.frombuffer(raw, dtype='<i2').astype(np.int32)
    elif sampwidth == 3:
        buf = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        padded = np.zeros((buf.shape[0], 4), dtype=np.uint8)
        padded[:, :3] = buf
        padded[:, 3] = np.where(buf[:, 2] >= 0x80, 0xFF, 0x00)
        return padded.view('<i4').flatten()
    elif sampwidth == 4:
        return np.frombuffer(raw, dtype='<i4')
    else:
        raise ValueError(f"Unsupported sample width {sampwidth}")
