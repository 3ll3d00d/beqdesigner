'''
The probe and layout failures the first full library run met (design/library-run-gaps.md, L2): a stream ffprobe gives
no codec_type (Elf), audio probed with no channel count on UHD discs whose audio starts late (Le Samourai, Parasite,
House Party, Lost Highway), an AC-3 layout ffmpeg can only describe by its channels (Prizzi's Honor, Silkwood) and a
disc backup still AACS-encrypted (John Wick 3, a Dark Knight backup).
'''
import os
import subprocess

import ffmpeg
import pytest

from model.bdmv import AACS_UNIT_BYTES, first_clip, is_aacs_encrypted
from model.ffmpeg import LONG_PROBE_OPTIONS, Executor, get_channel_name, layout_channels

FOUR = '4 channels (FL+FR+LFE+BC)'


def _probe(*streams):
    return {'format': {'duration': '60'}, 'streams': list(streams)}


def _audio(**fields):
    return {'codec_type': 'audio', 'codec_name': 'ac3', 'sample_rate': '48000', **fields}


def _executor(tmp_path, file='film.m2ts'):
    return Executor(str(tmp_path / file), str(tmp_path))


def test_a_stream_without_a_codec_type_is_passed_over(monkeypatch, tmp_path):
    probe = _probe({'index': 0, 'codec_type': 'video'}, _audio(index=1, channels=6, channel_layout='5.1(side)'),
                   {'index': 8})   # what ffprobe reports for a disc stream it cannot identify
    monkeypatch.setattr('model.ffmpeg.ffmpeg.probe', lambda file, **kwargs: probe)
    executor = _executor(tmp_path)

    executor.probe_file()

    assert executor.has_audio() and executor.has_video()
    assert [s['index'] for s in executor.audio_stream_data] == [1]


def test_audio_with_no_channel_count_is_probed_again_for_longer_and_extracted_the_same_way(monkeypatch, tmp_path):
    calls = []

    def probe(file, **kwargs):
        calls.append(kwargs)
        channels = 6 if kwargs.get('analyzeduration') else 0
        return _probe({'index': 0, 'codec_type': 'video'}, _audio(index=1, channels=channels))

    monkeypatch.setattr('model.ffmpeg.ffmpeg.probe', probe)
    executor = _executor(tmp_path)

    executor.probe_file()
    executor.update_spec(0, -1, True)

    assert calls == [{}, LONG_PROBE_OPTIONS]
    assert executor.channel_count == 6
    assert '-analyzeduration "100M"' in executor.ffmpeg_cli and '-probesize "200M"' in executor.ffmpeg_cli


def test_audio_with_a_channel_count_is_probed_once(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr('model.ffmpeg.ffmpeg.probe',
                        lambda file, **kwargs: calls.append(kwargs) or _probe(_audio(index=1, channels=2)))

    _executor(tmp_path).probe_file()

    assert calls == [{}]


def test_a_stream_still_without_a_channel_count_is_refused_in_words_not_a_division_by_zero(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr('model.ffmpeg.ffmpeg.probe',
                        lambda file, **kwargs: calls.append(kwargs) or _probe(_audio(index=1, channels=0)))
    executor = _executor(tmp_path)
    executor.probe_file()

    with pytest.raises(ValueError, match='no channel count for audio stream 0'):
        executor.update_spec(0, -1, True)
    assert len(calls) == 2   # the longer probe was tried once, not again and again


@pytest.mark.parametrize('name, channels', [
    (FOUR, ['FL', 'FR', 'LFE', 'BC']),
    ('FL+FR+LFE+BC', ['FL', 'FR', 'LFE', 'BC']),
    ('5.1(side)', ['FL', 'FR', 'FC', 'LFE', 'SL', 'SR']),
    ('unknown', None),
    ('3 channels', None),
    ('', None),
])
def test_a_layout_names_its_channels(name, channels):
    assert layout_channels(name) == channels


def test_a_custom_layout_is_mixed_with_its_lfe_where_it_is_and_its_channels_named(monkeypatch, tmp_path):
    monkeypatch.setattr('model.ffmpeg.ffmpeg.probe',
                        lambda file, **kwargs: _probe(_audio(index=1, channels=4, channel_layout=FOUR)))
    executor = _executor(tmp_path, 'film.mkv')
    executor.probe_file()

    executor.update_spec(0, -1, True)

    assert executor.channel_count == 4 and executor.lfe_idx == 3   # 1-based, as for every other layout
    gains = [float(term.split('*')[0]) for term in executor.mono_mix_spec.split('+')]
    assert gains[2] > gains[0] == gains[1] == gains[3]   # the LFE (c2) carries the LFE gain, not a main's
    assert [get_channel_name(None, i, 4, FOUR) for i in range(4)] == ['FL', 'FR', 'LFE', 'BC']


def test_a_layout_that_names_no_channels_is_mixed_from_the_channel_count(monkeypatch, tmp_path):
    monkeypatch.setattr('model.ffmpeg.ffmpeg.probe',
                        lambda file, **kwargs: _probe(_audio(index=1, channels=3, channel_layout='3 channels')))
    executor = _executor(tmp_path, 'film.mkv')
    executor.probe_file()

    executor.update_spec(0, -1, True)

    assert executor.channel_count == 3 and executor.mono_mix_spec.count('*c') == 3


@pytest.mark.requires_ffmpeg
def test_an_ac3_track_with_a_custom_layout_extracts_to_mono_and_to_channels(tmp_path):
    source = str(tmp_path / 'custom.mka')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', *sum((['-f', 'lavfi', '-i', f'sine=f={f}:d=1'] for f in (40, 50, 30, 60)), []),
                    '-filter_complex', '[0][1][2][3]join=inputs=4:channel_layout=FL+FR+LFE+BC', '-c:a', 'ac3', source],
                   check=True)
    for mono in (True, False):
        executor = Executor(source, str(tmp_path / ('mono' if mono else 'multi')), mono_mix=mono)
        os.makedirs(str(tmp_path / ('mono' if mono else 'multi')))
        executor.probe_file()
        executor.update_spec(0, -1, mono)

        executor.run_sync()

        written = ffmpeg.probe(executor.get_output_path())['streams'][0]
        assert written['channels'] == (1 if mono else 4)


# --- an AACS-encrypted Blu-ray backup --------------------------------------------------------------------------------

def _clip(path, units, encrypted):
    ''' Aligned units of 32 source packets; an encrypted unit keeps only its first packet's sync byte in the clear. '''
    unit = bytearray(os.urandom(AACS_UNIT_BYTES))
    for i in range(AACS_UNIT_BYTES // 192):
        unit[i * 192 + 4] = 0x47 if i == 0 or not encrypted else (unit[i * 192 + 4] % 0x40)
    with open(path, 'wb') as out:
        out.write(bytes(unit) * units)
    return str(path)


def test_an_encrypted_clip_is_told_from_a_decrypted_one(tmp_path):
    assert is_aacs_encrypted(_clip(tmp_path / '00001.m2ts', 20, encrypted=True))
    assert not is_aacs_encrypted(_clip(tmp_path / '00002.m2ts', 20, encrypted=False))
    assert is_aacs_encrypted(_clip(tmp_path / '00003.m2ts', 1, encrypted=True))   # shorter than the unit it reads
    assert not is_aacs_encrypted(_clip(tmp_path / 'film.mkv', 20, encrypted=True))
    assert not is_aacs_encrypted(str(tmp_path / 'missing.m2ts'))
    (tmp_path / 'empty.m2ts').write_bytes(b'')
    assert not is_aacs_encrypted(str(tmp_path / 'empty.m2ts'))


def test_the_first_clip_of_a_joined_title_is_the_one_checked():
    assert first_clip('concat:/d/BDMV/STREAM/00026.m2ts|/d/BDMV/STREAM/00000.m2ts') == '/d/BDMV/STREAM/00026.m2ts'
    assert first_clip('/d/BDMV/STREAM/00000.m2ts') == '/d/BDMV/STREAM/00000.m2ts'


def _refuse(file, **kwargs):
    raise ffmpeg.Error('ffprobe', b'', b'Invalid data found when processing input')


def test_a_probe_that_fails_on_an_encrypted_clip_says_so(monkeypatch, tmp_path):
    clip = _clip(tmp_path / '00030.m2ts', 20, encrypted=True)
    monkeypatch.setattr('model.ffmpeg.ffmpeg.probe', _refuse)

    with pytest.raises(ValueError, match='AACS-encrypted') as raised:
        Executor(f'concat:{clip}|{tmp_path / "00031.m2ts"}', str(tmp_path), display_name='t').probe_file()
    assert isinstance(raised.value.__cause__, ffmpeg.Error)


def test_a_probe_that_fails_on_a_decrypted_clip_is_left_alone(monkeypatch, tmp_path):
    clip = _clip(tmp_path / '00030.m2ts', 20, encrypted=False)
    monkeypatch.setattr('model.ffmpeg.ffmpeg.probe', _refuse)

    with pytest.raises(ffmpeg.Error):
        Executor(clip, str(tmp_path)).probe_file()
