'''Resolve a LibraryItem into canonical TMDB-backed BEQ metadata.'''
from dataclasses import asdict
from typing import Optional, Sequence

from pipeline.library.source import LibraryItem
from pipeline.metadata import tmdb_details_by_id, tmdb_find_by_imdb_id, tmdb_lookup


def resolve_meta(item: LibraryItem, api_key: str, audio_types: Optional[Sequence[str]] = None) -> dict:
    '''
    Return BeqMetadata constructor kwargs for one library item.

    A library-provided TMDB id takes precedence, followed by an IMDb id
    resolved through TMDB. If neither produces an id, retain the existing
    title/year TMDB search. item.meta then supplies BEQ-specific fields
    such as season and edition, which TMDB does not own.
    '''
    audio_types = list(audio_types or [])
    tmdb_id = item.external_ids.get('tmdb')
    if not tmdb_id:
        imdb_id = item.external_ids.get('imdb')
        if imdb_id:
            tmdb_id = tmdb_find_by_imdb_id(imdb_id, api_key, item.kind)

    if tmdb_id:
        metadata = tmdb_details_by_id(tmdb_id, api_key, item.kind, audio_types)
    else:
        metadata = tmdb_lookup(item.title or item.display_name, item.year or '', api_key, item.kind, audio_types)
    return {**asdict(metadata), **item.meta}
