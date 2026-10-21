'''
Published BEQs for one title, read on demand from beqcatalogue's comparison files (design/web-parity.md WP3), so the
service never downloads the whole catalogue (`database.json`, about 38 MB): `compare/titles.json` (about 1 MB: each
title's key, title, year, content type and authors) and `compare/<key>.json` (about 1 KB: each author's entry with its
audio types, edition, season/episode and filters).

Both are cached under a folder of their own with their `ETag`s and asked again, conditionally, at most once a day; when
the catalogue cannot be reached the cached copy is used. Matching is `catalogue_match`'s, by TMDB id first (a key ends in
`_<tmdb id>`), then title and year.
'''
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import requests

from pipeline.library.catalogue_match import CatalogueTarget, _key, _year, matches_track

CATALOGUE_URL = 'https://beqcatalogue.readthedocs.io/en/latest/'
MAX_AGE_SECONDS = 24 * 3600
MAX_BYTES = 8 * 1024 * 1024
TIMEOUT_SECONDS = 20
_SAFE_KEY = re.compile(r'^[a-z0-9][a-z0-9._-]*$')


class CatalogueUnavailable(Exception):
    ''' The catalogue could not be read and nothing is cached: said in words. '''


@dataclass(frozen=True)
class PublishedBeq:
    ''' One author's BEQ of a title, as the comparison file lists it. '''
    index: int
    author: str
    audio_types: Tuple[str, ...]
    edition: str
    season: str
    episodes: str
    filters: list = field(default_factory=list, compare=False)
    catalogue_url: str = ''
    language: str = ''      # not in the comparison files: matches_track() then does not judge them
    source: str = ''


@dataclass(frozen=True)
class TitleMatch:
    key: str
    title: str
    year: str
    matched_by: str          # 'tmdb' or 'title'
    entries: List[PublishedBeq]


class CatalogueCache:
    '''
    :param folder: where the files are kept.
    :param get: `requests.get` unless given.
    :param now: the clock.
    '''

    def __init__(self, folder: str, base_url: str = CATALOGUE_URL, get: Optional[Callable] = None,
                 now: Callable[[], float] = time.time):
        self._folder, self._base = folder, base_url.rstrip('/') + '/'
        self._get, self._now = get or requests.get, now

    def _read(self, name: str) -> bytes:
        ''' The file `compare/<name>`, from the cache when it is fresh, else asked for (conditionally). '''
        path = os.path.join(self._folder, name)
        meta_path = path + '.meta'
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (OSError, ValueError):
            meta = {}
        cached = os.path.isfile(path)
        if cached and self._now() - meta.get('fetched_at', 0) < MAX_AGE_SECONDS:
            with open(path, 'rb') as f:
                return f.read()
        headers = {'If-None-Match': meta['etag']} if cached and meta.get('etag') else {}
        try:
            response = self._get(self._base + 'compare/' + name, headers=headers, timeout=TIMEOUT_SECONDS, stream=True)
            if response.status_code == 304 and cached:
                data = None
            else:
                response.raise_for_status()
                chunks, size = [], 0
                for chunk in response.iter_content(chunk_size=65536):
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise CatalogueUnavailable(f'compare/{name} is bigger than {MAX_BYTES // (1024 * 1024)} MB')
                    chunks.append(chunk)
                data = b''.join(chunks)
                json.loads(data)   # a page that is not JSON (an error page) is never kept
        except (requests.RequestException, ValueError, CatalogueUnavailable) as error:
            if cached:
                with open(path, 'rb') as f:
                    return f.read()
            if isinstance(error, requests.HTTPError) and error.response is not None and error.response.status_code == 404:
                raise FileNotFoundError(name) from None
            raise CatalogueUnavailable(f'the catalogue could not be read: {error}') from None
        os.makedirs(self._folder, exist_ok=True)
        if data is not None:
            self._write(path, data)
        etag = response.headers.get('ETag') if hasattr(response, 'headers') and response.headers else None
        self._write(meta_path, json.dumps({'fetched_at': self._now(), 'etag': etag or meta.get('etag')}).encode())
        with open(path, 'rb') as f:
            return f.read()

    @staticmethod
    def _write(path: str, data: bytes) -> None:
        handle, temporary = tempfile.mkstemp(prefix='.catalogue.', dir=os.path.dirname(path))
        with os.fdopen(handle, 'wb') as f:
            f.write(data)
        os.replace(temporary, path)

    def titles(self) -> list:
        return json.loads(self._read('titles.json'))

    def entries(self, key: str) -> List[PublishedBeq]:
        if not _SAFE_KEY.match(key):
            raise ValueError(f'{key!r} is not a catalogue key')
        try:
            listed = json.loads(self._read(f'{key}.json'))
        except FileNotFoundError:
            return []
        return [PublishedBeq(index=i, author=str(e.get('author') or ''),
                             audio_types=tuple(str(a) for a in e.get('audioTypes') or ()),
                             edition=str(e.get('edition') or ''), season=str(e.get('season') or ''),
                             episodes=str(e.get('episode') or ''), filters=list(e.get('filters') or []),
                             catalogue_url=str(e.get('catalogue_url') or ''))
                for i, e in enumerate(listed) if isinstance(e, dict)]

    def find(self, target: CatalogueTarget) -> Optional[TitleMatch]:
        ''' The title the catalogue has for `target`, by TMDB id, else by title and year (movie and TV kept apart). '''
        if target is None or not target.usable:
            return None
        kind = 'tv' if target.is_tv else 'film'
        listed = [t for t in self.titles() if isinstance(t, dict) and str(t.get('content_type', '')).casefold() == kind]
        found, by = None, ''
        if target.tmdb:
            found = next((t for t in listed if str(t.get('key', '')).rsplit('_', 1)[-1] == target.tmdb), None)
            by = 'tmdb'
        if found is None and target.title and target.year:
            found = next((t for t in listed if _key(t.get('title')) == _key(target.title)
                          and _year(t.get('year')) == target.year), None)
            by = 'title'
        if found is None:
            return None
        key = str(found['key'])
        return TitleMatch(key, str(found.get('title') or ''), str(found.get('year') or ''), by, self.entries(key))


def same_track(target: CatalogueTarget, entry: PublishedBeq) -> bool:
    ''' Whether a published BEQ is for the same audio, edition and episodes as the title (`catalogue_match`'s test). '''
    return matches_track(target, entry)

