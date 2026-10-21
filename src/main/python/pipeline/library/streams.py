'''
A title's audio streams in words, for a person choosing which one is extracted (TODO W2).

A source lists each stream as a detail dict, in the container's audio-stream order (its ordinal is the
`LibraryItem.audio_stream` that selects it): `codec`, `channels` and `audio_types` always, and `sample_rate` (Hz),
`bitrate` (kbps), `language` and `title` when the source knows them. No Qt: the pipeline and the work list share it.
`preferred_stream()` is the one extracted when nobody has chosen; Batch Extract defaults to it too.
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


_LOSSLESS = ('truehd', 'dts-hd ma', 'pcm', 'flac', 'mlp', 'alac')


def is_lossless(codec: str) -> bool:
    ''' True for a codec name (JRiver's, or details_from_ffprobe()'s) that carries the master unreduced. '''
    text = str(codec).lower()
    return any(name in text for name in _LOSSLESS)


def _channel_count(detail: Mapping) -> int:
    try:
        return int(str(detail.get('channels', '')).strip())
    except ValueError:
        return 0


def preferred_stream(details: Sequence[Mapping]) -> int:
    '''
    The audio stream extracted when nobody has chosen one: the most channels, lossless before lossy on a tie, then the
    earlier. Discs often put a stereo or mono track first (Free Solo: LPCM 2.0 before DTS-HD MA 5.1; Jaws: DTS 2.0
    before TrueHD Atmos), and taking the first designed those from the wrong track. 0 for a source that listed none.

    Only streams in the main track's language compete, so a dub never wins: the language of the first lossless stream,
    since a disc's main track is lossless and its dubs usually are not (Universal's discs list a Japanese DTS track
    first: Back to the Future, The Breakfast Club), or of the first stream when none is lossless (a DVD). A stream with
    no language competes either way.
    '''
    if not details:
        return 0
    lossless = [d for d in details if is_lossless(d.get('codec', ''))]
    language = str((lossless or details)[0].get('language') or '').strip().casefold()
    candidates = [i for i, d in enumerate(details)
                  if not language or str(d.get('language') or '').strip().casefold() in ('', language)]
    return max(candidates, key=lambda i: (_channel_count(details[i]), is_lossless(details[i].get('codec', '')), -i))


def poorer_than_preferred(details: Sequence[Mapping], audio_stream: int) -> str:
    '''
    Why the source's selection is not used, or '': a selected stream with fewer channels than preferred_stream()'s
    (JRiver remembers the stream last played, which can be a stereo or commentary track: Thor: The Dark World played
    its AC-3 2.0 ahead of the DTS-HD MA 7.1). Unknown when the source lists no streams: the selection then stands.
    '''
    if not 0 <= audio_stream < len(details):
        return ''
    best = preferred_stream(details)
    if _channel_count(details[audio_stream]) >= _channel_count(details[best]):
        return ''
    return (f"the source plays audio stream {describe_stream(details[audio_stream], audio_stream)}, which has fewer "
            f"channels than audio stream {describe_stream(details[best], best)}, so that is used")


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


def probe_stream_details(item) -> tuple:
    '''
    A title's audio streams read from its file with ffprobe, as a source lists them (`details_from_ffprobe`), for a source
    that listed none: opened as extraction opens it (its playlist, its title hint). Blocks for as long as ffprobe takes.
    The desktop's title page and the service (its streams route and the job that extracts a chosen stream) share it.
    :param item: a `LibraryItem`.
    :raises ValueError: with the reason in words: the file is not here, has no audio, or ffprobe failed (its last line).
    '''
    import os
    from pipeline.config import AnalysisConfig
    from pipeline.library.extract_cache import title_hint
    from pipeline.orchestrate import Session
    path = item.source_path
    if item.source_path_problem:
        raise ValueError(item.source_path_problem)
    if not os.path.exists(path):
        raise ValueError(f'{path} cannot be found (is the media share mounted here?)')
    try:
        streams = Session(AnalysisConfig()).probe_audio_streams(path, item.playlist_name, title_hint(item))
    except Exception as error:
        stderr = getattr(error, 'stderr', None)
        reason = (stderr.decode('utf-8', 'replace').strip().splitlines() or [str(error)])[-1] \
            if isinstance(stderr, bytes) else str(error)
        raise ValueError(reason) from None
    if not streams:
        raise ValueError(f'{path} has no audio stream')
    return details_from_ffprobe(streams)
