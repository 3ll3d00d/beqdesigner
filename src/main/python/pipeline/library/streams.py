'''
A title's audio streams in words, for a person choosing which one is extracted (TODO W2).

A source lists each stream as a detail dict, in the container's audio-stream order (its ordinal is the
`LibraryItem.audio_stream` that selects it): `codec`, `channels` and `audio_types` always, and `sample_rate` (Hz),
`bitrate` (kbps), `language` and `title` when the source knows them. No Qt: the pipeline and the work list share it.
'''
from typing import Mapping, Sequence


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
