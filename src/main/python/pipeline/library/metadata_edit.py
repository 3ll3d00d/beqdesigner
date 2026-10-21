'''
Editing a title's metadata (no widgets): what the desktop title page's Metadata tab (`model.worklist_metadata`) and the
pipeline service's metadata routes both say and write -- design/web-parity.md WP0/WP4.

The rules are the title page's (its module docstring has the why): an edit is merged into the entry **as it is on disk
now**, only the keys a person touched; a blank field unsets its key (`REMOVE`) so a default applies again, except episodes,
where a blank is a real "no episodes" (`[]`); a change of the season or the TMDB id drops the TMDB season a run looked up.
'''
import logging
import traceback
from typing import Any, Dict, List, Mapping, Optional, Tuple

import requests

from pipeline.metadata import BeqMetadata, format_episodes, parse_episodes, redact, tmdb_details_by_id, tmdb_lookup
from pipeline.review import QueueEntry, read_entry, update_entry

logger = logging.getLogger('worklist')

REMOVE = object()   # a change that unsets a key of `meta`

# the meta keys a person edits as text, in the order they are shown: Essentials, then More
ESSENTIAL_FIELDS = ('title', 'year', 'audio_types', 'edition', 'season', 'episodes', 'note', 'warning', 'the_movie_db')
MORE_FIELDS = ('alt_title', 'sort_title', 'language', 'source', 'rating', 'author', 'avs', 'runtime', 'gain')
TEXT_FIELDS = ESSENTIAL_FIELDS + MORE_FIELDS
TMDB_EXTRAS = ('overview', 'genres', 'collection')   # from a TMDB reload: kept, not shown in a box of their own
FILL_FROM_TMDB = ('title', 'alt_title', 'year', 'rating', 'runtime', 'the_movie_db')
INDEXED_STATUSES = ('pending', 'accepted', 'published')   # the index says "metadata incomplete" of these, and of no other
AUDIO_TYPES = ('Atmos', 'TrueHD 7.1', 'TrueHD 5.1', 'DTS:X', 'DTS-HD MA 7.1', 'DTS-HD MA 6.1',
               'DTS-HD MA 5.1', 'DD+ Atmos', 'DD+', 'DD 5.1', 'LPCM 7.1', 'LPCM 5.1')
SOURCES = ('Disc', 'Apple TV+', 'Amazon', 'DC Universe', 'Disney+', 'HBOMax', 'Hulu', 'iTunes', 'Netflix',
           'Paramount+', 'Peacock')
LANGUAGES = ('English', 'Danish', 'French', 'German', 'Italian', 'Norwegian', 'Portuguese', 'Russian', 'Spanish',
             'Arabic', 'Farsi', 'Hebrew', 'Turkish', 'Hindi', 'Tamil', 'Telugu', 'Bengal', 'Cantonese', 'Japanese',
             'Korean', 'Mandarin', 'Indonesian', 'Thai', 'Vietnamese', 'Swahili', 'Mayan')


def field_texts(meta: Mapping[str, Any]) -> Dict[str, str]:
    ''' What each text box shows for `meta`: audio types comma separated, episodes as ranges ("1-3, 5"). '''
    texts = {}
    for key in TEXT_FIELDS:
        value = meta.get(key)
        if key == 'audio_types':
            texts[key] = ', '.join(str(v) for v in value) if isinstance(value, (list, tuple)) else str(value or '')
        elif key == 'episodes':
            try:
                texts[key] = format_episodes(value or [])
            except TypeError:   # not a list of numbers: show it as it is rather than lose it
                texts[key] = str(value)
        else:
            texts[key] = '' if value is None else str(value)
    return texts


def parse_audio_types(text: str) -> List[str]:
    return [t.strip() for t in text.split(',') if t.strip()]


def changes_from_texts(texts: Mapping[str, str], extras: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    '''
    :param texts: what the person typed, by `meta` key (only the keys they touched).
    :param extras: TMDB_EXTRAS values a reload filled, by key, taken as they are.
    :return: the changes (see `apply_changes`): a blank text is `REMOVE`, but blank episodes are `[]`.
    :raises ValueError: (with a sentence for the person) if the episodes are not numbers and ranges, or a key is not one
        of `TEXT_FIELDS`/`TMDB_EXTRAS`.
    '''
    changes: Dict[str, Any] = {}
    for key, value in (extras or {}).items():
        if key not in TMDB_EXTRAS:
            raise ValueError(f'{key} is not something a reload fills')
        changes[key] = value
    for key, text in texts.items():
        if key not in TEXT_FIELDS:
            raise ValueError(f'{key} is not a metadata field')
        text = str(text or '').strip()
        if key == 'episodes':
            try:
                changes[key] = parse_episodes(text)
            except ValueError as error:
                raise ValueError(f'Episodes: {error} (use numbers and ranges such as 1-3, 5)') from None
        elif key == 'audio_types':
            changes[key] = parse_audio_types(text) or REMOVE
        else:
            changes[key] = text or REMOVE
    return changes


def apply_changes(meta: Mapping[str, Any], changes: Mapping[str, Any]) -> dict:
    '''
    :return: a copy of `meta` with `changes` applied: a value replaces the key's, `REMOVE` unsets it. A change of the season
        *or of the TMDB id* also drops `season_id` and `season_episode_count`: they are the old season's, of the old series.
    '''
    merged = dict(meta)
    for key, value in changes.items():
        if value is REMOVE:
            merged.pop(key, None)
        else:
            merged[key] = value
    if any(_differs(meta, changes, key) for key in ('season', 'the_movie_db')):
        merged.pop('season_id', None)
        merged.pop('season_episode_count', None)
    return merged


def _differs(meta: Mapping[str, Any], changes: Mapping[str, Any], key: str) -> bool:
    ''' Whether `changes` sets `key` to something other than what `meta` has (unset and blank are the same). '''
    if key not in changes:
        return False
    new = '' if changes[key] is REMOVE else str(changes[key] or '')
    return new != str(meta.get(key) or '')


def save_meta_changes(queue_dir: str, entry_id: str, changes: Mapping[str, Any]) -> Tuple[QueueEntry, bool]:
    '''
    Applies `changes` to the entry **as it is on disk now** (not as it was when the page showed it).
    :return: (the entry, whether anything was written): an edit that leaves the metadata as it is writes nothing.
    :raises FileNotFoundError: if the title has no queue entry.
    '''
    fresh = read_entry(queue_dir, entry_id)
    merged = apply_changes(fresh.meta, changes)
    if merged == fresh.meta:
        return fresh, False
    return update_entry(queue_dir, entry_id, meta=merged), True


def badge_alarms(problems: List[str], status: str) -> bool:
    '''
    Whether the badge should warn. The index judges the metadata of a pending, accepted or published title only; a skipped or
    rejected one is not headed for the catalogue, so what is missing is said there without the alarm.
    '''
    return bool(problems) and status in INDEXED_STATUSES


def badge_text(problems: List[str], status: str, unsaved: bool) -> str:
    '''
    The header line: what is missing, or that the metadata is complete. "Ready to publish" is only said of a title that has
    been accepted; before that the metadata being complete is all that is known. A skipped or rejected title is not "not
    ready to publish" -- it is not going to be published as it is -- so its gaps are only listed (`badge_alarms`).
    '''
    if problems:
        text = ('Not ready to publish: ' if badge_alarms(problems, status) else 'Metadata incomplete: ') + '; '.join(problems)
    else:
        text = 'Ready to publish' if status in ('accepted', 'published') else 'Metadata complete'
    return text + (' (unsaved edits)' if unsaved else '')


class TmdbError(Exception):
    ''' A TMDB lookup that failed; its text is for a person, with the API key taken out. '''


def tmdb_reload(api_key: str, tmdb_id: str, title: str, year: str, kind: str) -> BeqMetadata:
    '''
    Looks a title up on TMDB by its id if it has one, else by title and year. Blocks: call it off the UI thread.
    :raises TmdbError: on any failure, its message redacted of the key (`requests` puts the URL, `?api_key=...`, in its
        errors' text, and so may any exception).
    '''
    try:
        return tmdb_details_by_id(tmdb_id, api_key, kind=kind) if tmdb_id else tmdb_lookup(title, year, api_key, kind=kind)
    except requests.RequestException as error:      # HTTPError, ConnectionError, Timeout...
        raise TmdbError(redact(f'TMDB lookup failed: {error}', api_key)) from None
    except Exception as error:                      # an answer that is not what TMDB says it is
        logger.error('TMDB lookup for %s failed: %s', tmdb_id or title,
                     redact(''.join(traceback.format_exception(error)), api_key))
        raise TmdbError(redact(f'TMDB lookup failed: {type(error).__name__}: {error}', api_key)) from None


def tmdb_fill(found: BeqMetadata) -> Tuple[Dict[str, str], Dict[str, Any]]:
    '''
    :return: (texts, extras) a reload fills: the FILL_FROM_TMDB boxes and TMDB_EXTRAS TMDB knows, nothing it does not.
        Both empty when TMDB found no match.
    '''
    if not found.the_movie_db:
        return {}, {}
    texts = {key: str(getattr(found, key)) for key in FILL_FROM_TMDB if getattr(found, key)}
    extras = {key: getattr(found, key) for key in TMDB_EXTRAS if getattr(found, key)}
    return texts, extras
