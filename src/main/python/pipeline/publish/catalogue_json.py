'''Version-1 BEQCatalogue filter records.

The record is deliberately independent of BEQDesigner project JSON.  It is
the small, stable interchange object described by beqcatalogue's
``filter-record-contract.md``.
'''
import decimal
import hashlib
import json
import time
from typing import Iterable, Mapping, Optional

from model.iir import HighShelf, LowShelf, PeakingEQ, Shelf
from pipeline.metadata import BeqMetadata


_CONTEXT = decimal.Context(prec=17)


def _number(value: float) -> str:
    '''Use the same non-exponent decimal spelling as BEQCatalogue.'''
    return format(_CONTEXT.create_decimal(repr(float(value))), 'f')


def _biquads(filters: Iterable) -> list[dict]:
    output = []
    for original in filters:
        if not isinstance(original, (PeakingEQ, LowShelf, HighShelf)):
            raise ValueError(f'{type(original).__name__} cannot be published to BEQCatalogue')
        # beqcatalogue rounds Q to four places before it calculates the
        # coefficients.  Reconstruct rather than use ``resample`` directly
        # so the two implementations are byte-for-byte compatible.
        count = original.count if isinstance(original, Shelf) else 1
        if isinstance(original, Shelf):
            sampled = type(original)(96000, original.freq, round(original.q, 4), original.gain, 1)
        else:
            sampled = type(original)(96000, original.freq, round(original.q, 4), original.gain)
        item = {
            'type': type(sampled).__name__, 'freq': sampled.freq, 'gain': sampled.gain, 'q': round(sampled.q, 4),
            'biquads': {'96000': {'b': [_number(value) for value in sampled.b],
                                  'a': [_number(-value) for value in sampled.a[1:]]}}
        }
        if isinstance(sampled, Shelf):
            item['count'] = 1
        output.extend(item.copy() for _ in range(count))
    return output


def catalogue_digest(record: Mapping) -> str:
    '''The historic BEQCatalogue digest; insertion order is part of its format.'''
    relevant = {key: record[key] for key in ('title', 'filters', 'mv', 'season', 'episode') if key in record}
    return hashlib.sha256(json.dumps(relevant).encode('utf-8')).hexdigest()


def filter_record(filters: Iterable, meta: BeqMetadata, *, existing: Optional[Mapping] = None,
                  now: Optional[int] = None) -> dict:
    '''Build one source record, preserving its creation time across updates.'''
    record = {
        'title': meta.title, 'year': meta.year, 'audioTypes': meta.audio_types,
        'content_type': 'TV' if meta.season else 'film', 'author': meta.author,
        'filters': _biquads(filters), 'mv': meta.gain or '0',
    }
    optional = {
        'altTitle': meta.alt_title, 'sortTitle': meta.sort_title, 'edition': meta.edition,
        'season': meta.season_element(), 'note': meta.note_with_episodes(), 'warning': meta.warning,
        'language': meta.language, 'source': meta.source, 'overview': meta.overview, 'rating': meta.rating,
        'runtime': meta.runtime, 'collection': meta.collection, 'genres': meta.genres, 'avs': meta.avs,
        'theMovieDB': meta.the_movie_db, 'images': [url for url in (meta.pva_url, meta.spectrum_url) if url],
    }
    record.update({key: value for key, value in optional.items() if value not in ('', None, [], {})})
    record['digest'] = catalogue_digest(record)
    timestamp = int(time.time() if now is None else now)
    if existing and existing.get('digest') == record['digest']:
        record['created_at'] = existing.get('created_at', timestamp)
        record['updated_at'] = existing.get('updated_at', timestamp)
    else:
        record['created_at'] = existing.get('created_at', timestamp) if existing else timestamp
        record['updated_at'] = timestamp
    return record


def aggregate(records: Mapping[str, dict]) -> bytes:
    '''Derived ``database.json`` content in stable path order.'''
    return (json.dumps([records[path] for path in sorted(records)], indent=2, ensure_ascii=False) + '\n').encode('utf-8')
