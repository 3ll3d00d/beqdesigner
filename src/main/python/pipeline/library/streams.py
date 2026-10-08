'''
A title's audio streams in words, for a person choosing which one is extracted (TODO W2).

A source lists each stream as a detail dict, in the container's audio-stream order (its ordinal is the
`LibraryItem.audio_stream` that selects it): `codec`, `channels` and `audio_types` always, and `sample_rate` (Hz),
`bitrate` (kbps), `language` and `title` when the source knows them. No Qt: the pipeline and the work list share it.
'''
import re
from typing import Mapping, Optional, Sequence


def audio_types(codec: str, channels: str) -> tuple[str, ...]:
    '''Translate a stream's codec name (JRiver's, or details_from_ffprobe()'s) into the metadata panel's choices.'''
    text, count = codec.lower(), str(channels).strip()
    if 'atmos' in text and 'truehd' in text:
        return ('Atmos',)
    if 'dts:x' in text:
        return ('DTS:X',)
    if 'truehd' in text:
        return (f'TrueHD {"7.1" if count in ("8", "7") else "5.1"}',)
    if 'dts-hd' in text or 'dts hd' in text:
        return (f'DTS-HD MA {"7.1" if count in ("8", "7") else "6.1" if count == "7" else "5.1"}',)
    if 'e-ac3' in text or 'eac3' in text:
        return ('DD+ Atmos' if 'atmos' in text else 'DD+',)
    if 'ac-3' in text or 'ac3' in text:
        return ('DD 5.1',) if count == '6' else ()
    if 'pcm' in text:
        return (f'LPCM {"7.1" if count in ("8", "7") else "5.1"}',) if count in ('6', '7', '8') else ()
    return ()


def _channels(value: str) -> str:
    try:
        count = int(str(value).strip())
    except ValueError:
        return ''
    return {1: 'mono', 2: 'stereo', 6: '5.1', 7: '6.1', 8: '7.1'}.get(count, f'{count} channels')


def _number(value) -> float:
    try:
        return float(str(value).strip())
    except ValueError:
        return 0.0


def describe_stream(detail: Mapping, ordinal: int) -> str:
    '''
    "2: DTS-HD MA 5.1, English, 48 kHz" -- the one-based number a person picks, then what is known of the stream.
    :param ordinal: the stream's zero-based position in the source's list.
    '''
    head = ' '.join(part for part in (str(detail.get('codec') or '').strip() or 'unknown codec',
                                      _channels(detail.get('channels', ''))) if part)
    parts = [head]
    if detail.get('title'):
        parts.append(f'"{detail["title"]}"')
    if detail.get('language'):
        parts.append(str(detail['language']))
    rate = _number(detail.get('sample_rate', ''))
    if rate:
        parts.append(f'{rate / 1000:g} kHz')
    bitrate = _number(detail.get('bitrate', ''))
    if bitrate:
        parts.append(f'{bitrate:g} kbps')
    return f'{ordinal + 1}: ' + ', '.join(parts)


def describe_streams(details: Sequence[Mapping]) -> list[str]:
    return [describe_stream(detail, i) for i, detail in enumerate(details)]


# ffprobe's codec_name -> the name JRiver uses, which audio_types() and a person read
_CODEC_NAMES = {'truehd': 'TrueHD', 'ac3': 'AC-3', 'eac3': 'E-AC3', 'dts': 'DTS', 'aac': 'AAC', 'flac': 'FLAC',
                'opus': 'Opus', 'mp3': 'MP3', 'pcm_bluray': 'PCM_BLURAY', 'pcm_dvd': 'PCM_DVD'}


def details_from_ffprobe(streams: Sequence[Mapping]) -> tuple[dict, ...]:
    '''
    ffprobe's audio streams (`-show_streams`, already filtered to audio, in order) as the detail dicts a source lists:
    for a title whose source supplied none, so a person still chooses from a described list.
    '''
    details = []
    for stream in streams:
        name = str(stream.get('codec_name') or '')
        profile = str(stream.get('profile') or '')
        codec = _CODEC_NAMES.get(name, name.upper())
        if name == 'dts' and profile and profile.upper() != 'DTS':
            codec = profile   # 'DTS-HD MA', 'DTS-HD HRA', 'DTS-ES'
        if 'atmos' in profile.lower():
            codec = f'{codec} Atmos'
        channels = str(stream.get('channels') or '')
        detail = {'codec': codec, 'channels': channels, 'audio_types': audio_types(codec, channels)}
        tags = stream.get('tags') or {}
        rate = str(stream.get('sample_rate') or '')
        if rate:
            detail['sample_rate'] = rate
        try:
            detail['bitrate'] = str(int(stream['bit_rate']) // 1000)
        except (KeyError, TypeError, ValueError):
            pass
        for key in ('language', 'title'):
            if tags.get(key):
                detail[key] = str(tags[key])
        details.append(detail)
    return tuple(details)


def stream_choice(details: Sequence[Mapping], audio_stream: int, keep_multichannel: bool, chosen_by: str = '') -> str:
    '''
    What a run extracts of a title, said before it starts: "audio stream 2: DTS-HD MA 5.1, English, 48 kHz; multichannel
    kept" -- or, for a source that listed no streams, the stream by number alone.
    '''
    if 0 <= audio_stream < len(details):
        what = f'audio stream {describe_stream(details[audio_stream], audio_stream)}'
    else:
        what = f'audio stream {audio_stream + 1} (the source lists none)'
    by = {'manual': ' (your choice)', 'source': ' (as the library plays it)'}.get(chosen_by, '')
    return f'{what}{by}; multichannel {"kept" if keep_multichannel else "not kept"}'


def channels_found(count, layout: str = '') -> str:
    ''' What extraction found the stream to be: "6 channels (5.1(side))", or '' if it did not record it. '''
    try:
        number = int(count)
    except (TypeError, ValueError):
        return ''
    said = f'{number} channel{"" if number == 1 else "s"}'
    return f'{said} ({layout})' if layout and layout != 'unknown' else said


def playback_records(value: str) -> Optional[list]:
    ''' JRiver's length-prefixed `(n:text)` records (`Playback Info`), or None if the value does not parse. '''
    parts, i = [], 0
    while i < len(value):
        match = re.match(r'\((\d+):', value[i:])
        if not match:
            return None
        start = i + match.end()
        end = start + int(match.group(1))
        if end >= len(value) + 1 or value[end:end + 1] != ')':
            return None
        parts.append(value[start:end])
        i = end + 1
    return parts


def selected_streams(playback_info: str) -> Optional[tuple[int, ...]]:
    '''
    The container streams JRiver plays, from its Playback Info: `(1:N)` then N name/value records, one of them
    `Streams`, whose value is the video, audio and (optionally) subtitle streams as ffprobe's global indices, e.g.
    `0,1,3` (src/test/python/fixtures/jriver/README.md). None when it is absent or does not parse.
    '''
    records = playback_records(playback_info or '')
    if not records or len(records) % 2 == 0:
        return None
    names, values = records[1::2], records[2::2]
    if 'Streams' not in names:
        return None
    try:
        numbers = tuple(int(part) for part in values[names.index('Streams')].split(','))
    except ValueError:
        return None
    return numbers if len(numbers) >= 2 else None


def audio_ordinal(streams: Sequence[Mapping], selected: Sequence[int]) -> tuple[Optional[int], str]:
    '''
    Which audio stream (its position among the file's audio streams) a source's selection is, from ffprobe's streams of
    the file (`-show_streams`, all of them, with `index` and `codec_type`).
    :return: (the ordinal, '') or (None, why it does not match): the selection is trusted only when its first stream is
        a video stream and its second an audio stream, as every title the fixture checked was; a disc whose numbering
        is off cannot otherwise be told from a deliberate choice.
    '''
    by_index = {stream.get('index'): stream for stream in streams}
    video, audio = selected[0], selected[1]
    if by_index.get(video, {}).get('codec_type') != 'video':
        return None, f'stream {video} of the file is not a video stream'
    if by_index.get(audio, {}).get('codec_type') != 'audio':
        return None, f'stream {audio} of the file is not an audio stream'
    ordered = [stream.get('index') for stream in streams if stream.get('codec_type') == 'audio']
    return ordered.index(audio), ''
