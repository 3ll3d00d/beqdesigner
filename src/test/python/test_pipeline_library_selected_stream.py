'''
TODO J2: the audio stream JRiver plays is the one extracted. Its Playback Info `Streams` names the video, audio and
subtitle streams by ffprobe's global index (fixtures/jriver/streams.json: 67 real titles probed, see the README); the
audio one is matched to the file's audio streams when the title is extracted, and a scan keeps the choice -- and a
reviewer's -- rather than listing the first stream again.
'''
import json
import os
import pathlib
import shutil
import subprocess

import pytest

from pipeline.library.index import carried_choice
from pipeline.library.jriver import JRiverLibrarySource
from pipeline.library.source import LibraryItem
from pipeline.library.streams import audio_ordinal, selected_streams
from test_pipeline_library_index import _item, _scan, env  # noqa: F401

FIXTURE = pathlib.Path(__file__).parent / 'fixtures' / 'jriver'
EVIDENCE = json.loads((FIXTURE / 'streams.json').read_text(encoding='utf-8'))
FILES = json.loads((FIXTURE / 'files.json').read_text(encoding='utf-8'))

def _family(jriver_codec: str) -> str:
    ''' JRiver's name for a stream's codec ('DTS-HD MA + DTS:X', 'TrueHD Atmos') as ffprobe's codec_name. '''
    name = jriver_codec.upper()
    for prefix, codec in (('TRUEHD', 'truehd'), ('DTS', 'dts'), ('E-AC3', 'eac3'), ('AC-3', 'ac3'),
                          ('PCM_BLURAY', 'pcm_bluray'), ('AAC', 'aac'), ('FLAC', 'flac')):
        if name.startswith(prefix):
            return codec
    return jriver_codec.lower()


@pytest.mark.parametrize('value, streams', [
    ('(1:2)(7:Streams)(5:0,1,3)(2:CC)(1:0)', (0, 1, 3)),
    ('(1:3)(14:BlurayPlaylist)(10:00002.mpls)(7:Streams)(6:0,2,37)(2:CC)(1:0)', (0, 2, 37)),
    ('(1:2)(7:Streams)(3:0,1)(9:Subtitles)(4:None)', (0, 1)),
    ('(1:1)', None),
    ('(1:2)(12:CenterOffset)(3:0,0)(11:ZoomPercent)(2:99)', None),
    ('(1:1)(7:Streams)(1:0)', None),          # no audio stream named
    ('(1:1)(7:Streams)(5:0,x,3)', None),
    ('(1:2)(7:Streams', None),                # cut short
    ('', None),
])
def test_the_streams_jriver_plays_are_read_from_its_playback_info(value, streams):
    assert selected_streams(value) == streams


def test_every_captured_playback_info_parses():
    assert all(selected_streams(row['Playback Info']) for row in FILES if 'Streams' in row.get('Playback Info', ''))


def test_the_matched_stream_is_the_one_jriver_lists_at_that_place_and_a_doubtful_one_is_refused():
    matched, refused, disagree = [], [], []
    for title in EVIDENCE:
        selected = tuple(int(n) for n in title['Streams'].split(','))
        ordinal, why = audio_ordinal(title['ffprobe'], selected)
        if ordinal is None:
            refused.append((title['Name'], why))
            continue
        codecs = [_family(c.strip()) for c in (title['Audio Codec'] or '').split(';')]
        audio = [s['codec_name'] for s in title['ffprobe'] if s['codec_type'] == 'audio']
        if codecs == audio:   # where JRiver and ffprobe describe the same streams, the match is JRiver's stream
            assert codecs[ordinal] == audio[ordinal], title['Name']
        else:
            disagree.append(title['Name'])
        matched.append(ordinal)
    assert len(matched) == 66 and sum(1 for o in matched if o) == 8   # eight titles play a stream other than the first
    assert refused == [('The Town', 'stream 1 of the file is not a video stream')]
    # where the two do not describe the same streams: JRiver lists 1 of Rocky's 13, and for A Star Is Born ffprobe sees
    # three AC-3 streams in the main title the disc resolves to where JRiver lists DTS-HD MA first (an E4 question)
    assert sorted(disagree) == ['A Star Is Born', 'Rocky']


def test_listing_records_the_selection_and_still_lists_the_first_stream():
    rows = [row for row in FILES if row['Key'] in {t['Key'] for t in EVIDENCE}]
    items = JRiverLibrarySource('127.0.0.1', 1, 1004)._map_rows(rows)
    by_key = {item.id.rpartition('-')[2]: item for item in items}
    for title in EVIDENCE:
        item = by_key[str(title['Key'])]
        assert item.selected_streams == tuple(int(n) for n in title['Streams'].split(','))
        assert item.audio_stream == 0 and item.audio_stream_source == ''   # resolved when the file is probed


# --- a scan keeps the stream chosen ----------------------------------------------------------------------------------

DETAILS = ({'codec': 'AC-3', 'channels': '2', 'audio_types': ()},
           {'codec': 'TrueHD Atmos', 'channels': '8', 'audio_types': ('Atmos',)})


def test_a_reviewers_choice_survives_a_rescan(env):
    ''' The regression J2 found: a rescan listed the first stream again, and the next run extracted it. '''
    item = _item('a', audio_stream_details=DETAILS)
    _scan(env, item)
    env.index.select_audio_stream('fs-a', 1)

    _scan(env, item)

    chosen = env.index.units(['fs-a'])['fs-a']
    assert (chosen.audio_stream, chosen.audio_stream_source, chosen.meta['audio_types']) == (1, 'manual', ['Atmos'])


def test_a_resolved_selection_is_kept_while_the_source_selects_the_same_streams(env):
    item = _item('a', audio_stream_details=DETAILS, selected_streams=(0, 2, 5))
    _scan(env, item)
    env.index.resolve_audio_stream('fs-a', 1)

    _scan(env, item)
    assert env.index.units(['fs-a'])['fs-a'].audio_stream == 1

    _scan(env, _item('a', audio_stream_details=DETAILS, selected_streams=(0, 1, 5)))   # MC now plays another
    assert env.index.units(['fs-a'])['fs-a'].audio_stream == 0                        # resolved again next run


def test_a_resolved_selection_never_replaces_a_reviewers_choice(env):
    _scan(env, _item('a', audio_stream_details=DETAILS, selected_streams=(0, 1)))
    env.index.select_audio_stream('fs-a', 1)

    assert env.index.resolve_audio_stream('fs-a', 0) is None
    assert env.index.units(['fs-a'])['fs-a'].audio_stream == 1


def test_a_choice_the_new_list_lacks_is_dropped():
    before = LibraryItem(id='x', source_path='/x', display_name='x', audio_stream=3, audio_stream_source='manual')
    now = LibraryItem(id='x', source_path='/x', display_name='x', audio_stream_details=DETAILS)
    assert carried_choice(now, before).audio_stream == 0


# --- resolved when the title is extracted ----------------------------------------------------------------------------

needs_ffmpeg = pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'),
                                  reason='ffmpeg and ffprobe are optional')


def _video_and_two_audio(tmp_path) -> str:
    ''' Global streams: 0 video, 1 stereo 100 Hz, 2 six channels of 40 Hz. '''
    path = str(tmp_path / 'film.mkv')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'color=black:size=32x32:duration=2:rate=5',
                    '-f', 'lavfi', '-i', 'sine=frequency=100:sample_rate=48000:duration=2',
                    '-f', 'lavfi', '-i', 'sine=frequency=40:sample_rate=48000:duration=2',
                    '-filter_complex', '[1:a]pan=stereo|c0=c0|c1=c0[a0];[2:a]pan=5.1|c0=c0|c1=c0|c2=c0|c3=c0|c4=c0|c5=c0[a1]',
                    '-map', '0:v', '-map', '[a0]', '-map', '[a1]', '-c:v', 'ffv1', '-c:a', 'flac', path], check=True)
    return path


@needs_ffmpeg
def test_jrivers_selection_is_the_stream_extracted_and_a_rescan_keeps_it(env):
    from model.execution_events import execution_event_context
    from pipeline.library.run import LibraryRunConfig, LibraryRunReport, run_unit
    from pipeline.orchestrate import Session
    source = _video_and_two_audio(env.tmp)
    item = LibraryItem(id='fs-film', source_path=source, display_name='Film', fingerprint='fp',
                       selected_streams=(0, 2, 7), audio_stream_details=DETAILS)
    _scan(env, item)
    config = LibraryRunConfig(work_dir=env.work, queue_dir=env.queue, designer='none', keep_multichannel=True)
    events = []
    context = execution_event_context('run', events.append)
    context.__enter__()
    try:
        work = run_unit(Session(config.config), env.index.units(['fs-film'])['fs-film'], config, LibraryRunReport(),
                        env.index, through='extract')
    finally:
        context.__exit__(None, None, None)

    import soundfile as sf
    assert sf.info(work.multichannel_wav_path).channels == 6        # global stream 2: the second audio stream
    assert any(e.message.startswith('Extracting audio stream 2: TrueHD Atmos 7.1 (as the library plays it)')
               for e in events if e.kind == 'stage_started')
    _scan(env, item)
    assert env.index.units(['fs-film'])['fs-film'].audio_stream == 1   # and the next scan keeps it


@needs_ffmpeg
def test_a_selection_that_does_not_match_the_file_falls_back_to_the_first_stream_and_says_so(env):
    from model.execution_events import execution_event_context
    from pipeline.library.run import LibraryRunConfig, LibraryRunReport, run_unit
    from pipeline.orchestrate import Session
    item = LibraryItem(id='fs-film', source_path=_video_and_two_audio(env.tmp), display_name='Film', fingerprint='fp',
                       selected_streams=(1, 2))   # stream 1 is audio, not video: numbered some other way
    config = LibraryRunConfig(work_dir=env.work, queue_dir=env.queue, designer='none')
    events = []
    context = execution_event_context('run', events.append)
    context.__enter__()
    try:
        run_unit(Session(config.config), item, config, LibraryRunReport(), through='extract')
    finally:
        context.__exit__(None, None, None)

    notes = [e.message for e in events if e.kind == 'note']
    assert notes == ["The source's stream selection (1,2) does not match the file: stream 1 of the file is not a video "
                     "stream; the first audio stream is used"]
