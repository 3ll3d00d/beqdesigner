'''Resolve a LibraryItem into canonical TMDB-backed BEQ metadata.'''
import logging
from dataclasses import asdict
from typing import Optional, Sequence

import requests

from pipeline.library.source import LibraryItem
from pipeline.metadata import tmdb_details_by_id, tmdb_find_by_external_id, tmdb_find_by_imdb_id, tmdb_lookup, tmdb_season_info

logger = logging.getLogger('library_metadata')


def library_meta(item: LibraryItem) -> dict:
    '''
    What the library itself says about an item, needing no TMDB: its title (falling back to the display name, so
    an entry is never anonymous) and year, plus the season and episodes. This is what an item is designed with
    when TMDB is not configured or cannot be reached; `item.meta` is layered over it by the caller.
    '''
    meta = {'title': item.title or item.display_name}
    if item.year:
        meta['year'] = item.year
    return {**meta, **season_meta(item)}


def season_meta(item: LibraryItem) -> dict:
    '''
    What the library itself says about a TV item's season and the episodes in scope (BeqMetadata's `season` and
    `episodes`). Needs no TMDB, so it is also what an item gets when TMDB can't be reached.
    '''
    meta = {}
    if item.season:
        meta['season'] = item.season
    if item.episodes:
        meta['episodes'] = list(item.episodes)
    return meta


def resolve_meta(item: LibraryItem, api_key: str, audio_types: Optional[Sequence[str]] = None) -> dict:
    '''
    Return BeqMetadata constructor kwargs for one library item.

    A library-provided TMDB id takes precedence, followed by an IMDb id and,
    for TV, an opted-in numeric TVDB series id resolved through TMDB. If none
    produces an id, retain the existing title/year TMDB search. item.meta then supplies BEQ-specific fields
    such as edition, which TMDB does not own.

    For a TV item with a season, the season and the episodes in scope come from the library, and TMDB is asked
    for the season's id and episode count, which beqcatalogue's structured season needs. If that lookup fails or
    finds nothing the item keeps just the season and episodes, which are still written (as the plain season and
    an episode note).
    '''
    audio_types = list(audio_types or [])
    tmdb_id = item.external_ids.get('tmdb')
    if not tmdb_id:
        imdb_id = item.external_ids.get('imdb')
        if imdb_id:
            tmdb_id = tmdb_find_by_imdb_id(imdb_id, api_key, item.kind)
    if not tmdb_id and item.kind == 'tv':
        tvdb_id = str(item.external_ids.get('tvdb') or '').strip()
        if tvdb_id.isdecimal():
            tmdb_id = tmdb_find_by_external_id(tvdb_id, api_key, 'tvdb_id', item.kind)

    if tmdb_id:
        metadata = tmdb_details_by_id(tmdb_id, api_key, item.kind, audio_types)
    else:
        metadata = tmdb_lookup(item.title or item.display_name, item.year or '', api_key, item.kind, audio_types)
    meta = {**asdict(metadata), **season_meta(item)}
    if item.kind == 'tv' and item.season and metadata.the_movie_db:
        try:
            info = tmdb_season_info(metadata.the_movie_db, item.season, api_key)
        except requests.RequestException as error:
            logger.warning('No TMDB season details for %s season %s: %s', item.id, item.season, error)
            info = None
        if info is not None:
            meta['season_id'] = info.id
            meta['season_episode_count'] = info.episode_count
    return {**meta, **item.meta}
