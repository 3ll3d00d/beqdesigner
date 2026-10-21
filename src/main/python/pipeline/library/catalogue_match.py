"""Metadata matching of a library title to published BEQs (no widgets): the title page's *Published BEQs* tab and the service."""
import re
import unicodedata
from dataclasses import dataclass

from model.iir import CompleteFilter, HighShelf, LowShelf, PeakingEQ


def _key(value):
    text = unicodedata.normalize('NFKD', str(value or '')).casefold()
    return ''.join(c for c in text if c.isalnum() or c == '+')


def _audio(value):
    text = _key(value)
    return {'dolbyatmos': 'atmos', 'dolbytruehd': 'truehd', 'dtshdmasteraudio': 'dtshdma',
            'dolbydigitalplus': 'dd+', 'dolbydigital': 'dd'}.get(text, text)


def _audio_set(values):
    """Codec sets accept catalogue lists and combined metadata strings, preserving DD+."""
    if isinstance(values, str):
        values = [values]
    return {_audio(codec) for value in (values or [])
            for codec in re.split(r'[,;/|]|\s+\+\s+', str(value)) if codec.strip()}


def _tmdb(value):
    text = str(value or '').strip().rstrip('/')
    if text.isdecimal():
        return str(int(text))
    match = re.search(r'/(?:movie|tv)/(\d+)(?:[-/?#]|$)', text)
    return str(int(match[1])) if match else ''


def _year(value):
    try:
        return str(int(value)) if value else ''
    except (ValueError, TypeError):
        return ''


def _numbers(value):
    result = set()
    for first, last in re.findall(r'(\d+)(?:\s*-\s*(\d+))?', str(value or '')):
        start, end = int(first), int(last or first)
        if 0 <= start <= end <= 1000:
            result.update(range(start, end + 1))
    return result


@dataclass(frozen=True)
class CatalogueTarget:
    title: str = ''
    year: str = ''
    tmdb: str = ''
    is_tv: bool = False
    audio_types: tuple = ()
    edition: str = ''
    language: str = ''
    source: str = ''
    season: str = ''
    episodes: str = ''

    @property
    def usable(self):
        return bool(self.tmdb or (self.title and self.year))


def catalogue_target(entry, row, defaults=None):
    meta = {**(defaults or {}), **(entry.meta if entry else {})}
    ids = getattr(row, 'external_ids', {}) or {}
    audio = meta.get('audio_types') or []
    if isinstance(audio, str):
        audio = [audio]
    return CatalogueTarget(
        title=str(meta.get('title') or getattr(row, 'title', '') or ''),
        year=_year(meta.get('year') or getattr(row, 'year', '')),
        tmdb=_tmdb(meta.get('the_movie_db') or ids.get('tmdb')),
        is_tv=bool(meta.get('season')) or getattr(row, 'kind', '') in ('tv', 'season', 'episode'),
        audio_types=tuple(str(a) for a in audio), edition=str(meta.get('edition') or ''),
        language=str(meta.get('language') or ''), source=str(meta.get('source') or ''),
        season=str(meta.get('season') or ''), episodes=str(meta.get('episodes') or ''))


def matches_title(target, entry):
    if target.is_tv != (str(entry.content_type).casefold() == 'tv'):
        return False
    tmdb = _tmdb(entry.the_movie_db)
    if target.tmdb and tmdb:
        return target.tmdb == tmdb
    titles = (_key(entry.title), _key(getattr(entry, 'altTitle', '')))
    return bool(target.title and target.year and _key(target.title) in titles and target.year == _year(entry.year))


def matches_track(target, entry):
    expected_audio = _audio_set(target.audio_types)
    if expected_audio and not expected_audio.intersection(_audio_set(entry.audio_types)):
        return False
    for field in ('edition', 'language', 'source'):
        expected, actual = getattr(target, field), getattr(entry, field)
        if expected and actual and _key(expected) != _key(actual):
            return False
    if target.is_tv:
        for expected, actual in ((target.season, entry.season), (target.episodes, entry.episodes)):
            if expected and actual and not _numbers(expected).intersection(_numbers(actual)):
                return False
    return True


def matching_entries(target, entries, include_other_tracks=False):
    if target is None or not target.usable:
        return []
    return sorted((entry for entry in entries if matches_title(target, entry) and
                   (include_other_tracks or matches_track(target, entry))),
                  key=lambda e: (e.author.casefold(), e.formatted_title.casefold(), str(e.idx)))


_TYPES = {'PeakingEQ': PeakingEQ, 'LowShelf': LowShelf, 'HighShelf': HighShelf}


def catalogue_biquads(filters, fs=96000):
    """
    A catalogue entry's filters (`{type, freq, gain, q}`, and in beqcatalogue's `compare/` files a `count` of repeats) as
    biquads, in order.
    :raises ValueError: for a filter type the catalogue does not use.
    """
    result = []
    for f in filters or []:
        kind = _TYPES.get(f.get('type'))
        if kind is None:
            raise ValueError(f"Unknown filt_type {f.get('type')}")
        for _ in range(max(1, int(f.get('count') or 1))):
            result.append(kind(fs, f['freq'], f['q'], f['gain'], f_id=len(result)))
    return result


def catalogue_filter(filters, fs=96000, description='Published BEQ'):
    """ A catalogue entry's filters as one `CompleteFilter`, for its response or to apply it to a measured curve. """
    return CompleteFilter(fs=fs, filters=catalogue_biquads(filters, fs), description=description, sort_by_id=True)
