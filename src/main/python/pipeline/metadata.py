'''
BeqMetadata + tmdb_lookup(): a plain, Qt-free replacement for the GUI's
widget-bound TMDB lookup and metadata dict (model/postbuilder.py's
__search_tmdb/__get_tmdb_details/__build_metadata/__validate_metadata) --
design/pipeline-implementation-plan.md phase 2 (item 5).

Does not read model.preferences (which imports qtpy at module level) --
callers supply the TMDB api_key explicitly, the same pattern
pipeline/config.py's AnalysisConfig establishes for other GUI-preference-
backed values.
'''
from dataclasses import dataclass, field
from typing import List, Optional

import requests

TMDB_BASE_URL = 'https://api.themoviedb.org/3'


@dataclass
class BeqMetadata:
    title: str
    year: str
    audio_types: List[str] = field(default_factory=list)
    genres: List[dict] = field(default_factory=list)  # [{'id': 28, 'name': 'Action'}, ...]
    alt_title: str = ''
    sort_title: str = ''
    spectrum_url: str = ''
    pva_url: str = ''
    edition: str = ''
    season: str = ''
    note: str = ''
    warning: str = ''
    gain: Optional[str] = None  # beq_gain -- this module does not derive it; see D2, item 7
    language: str = 'English'
    source: str = 'Disc'
    overview: str = ''
    rating: str = ''
    author: str = ''
    avs: str = ''
    the_movie_db: str = ''
    poster: str = ''
    runtime: str = ''
    collection: Optional[dict] = None

    def __post_init__(self):
        if not self.sort_title:
            self.sort_title = default_sort_title(self.title)

    def to_dict(self) -> dict:
        '''
        :return: the beq_-prefixed dict HDXmlParser.convert()'s metadata
            argument expects (model/postbuilder.py's __build_metadata shape).
        '''
        return {
            'beq_title': self.title,
            'beq_alt_title': self.alt_title,
            'beq_sortTitle': self.sort_title,
            'beq_year': self.year,
            'beq_spectrumURL': self.spectrum_url,
            'beq_pvaURL': self.pva_url,
            'beq_edition': self.edition,
            'beq_season': self.season,
            'beq_note': self.note,
            'beq_warning': self.warning,
            'beq_gain': self.gain,
            'beq_language': self.language,
            'beq_source': self.source,
            'beq_overview': self.overview,
            'beq_rating': self.rating,
            'beq_author': self.author,
            'beq_avs': self.avs,
            'beq_theMovieDB': self.the_movie_db,
            'beq_poster': self.poster,
            'beq_runtime': self.runtime,
            'beq_collection': self.collection,
            'beq_audioTypes': self.audio_types,
            'beq_genres': self.genres,
        }


def default_sort_title(title: str) -> str:
    ''' Lowercased, with a leading "the " stripped -- matches the GUI's autofillSortTitle. '''
    lowered = title.strip().lower()
    return lowered[len('the '):] if lowered.startswith('the ') else lowered


def validate(metadata: BeqMetadata) -> List[str]:
    '''
    :return: a list of problems, empty if metadata is valid enough to publish.
        Mirrors the GUI's __validate_metadata (title, year, at least one
        audio type) -- not a claim that these are the only fields worth
        checking, just the ones the existing workflow already enforces.
    '''
    problems = []
    if len(metadata.title) < 1:
        problems.append('title is required')
    if len(metadata.year) < 1:
        problems.append('year is required')
    if len(metadata.audio_types) < 1:
        problems.append('at least one audio type is required')
    return problems


def tmdb_lookup(title: str, year: str, api_key: str, kind: str = 'movie',
                audio_types: Optional[List[str]] = None) -> BeqMetadata:
    '''
    :param title: the title to search for.
    :param year: the release year (movie) or first air date year (tv).
    :param api_key: the TMDB API key.
    :param kind: 'movie' or 'tv'.
    :param audio_types: carried straight into the result -- TMDB has no
        concept of this, it is BEQ-specific.
    :return: the metadata found; a mostly-empty BeqMetadata if TMDB has no match.
    :raises ValueError: if kind is not 'movie' or 'tv'.
    :raises requests.HTTPError: on a non-2xx response from TMDB.
    '''
    if kind not in ('movie', 'tv'):
        raise ValueError(f"kind must be 'movie' or 'tv', got {kind!r}")

    search_params = {'api_key': api_key, 'query': title, 'include_adult': 'false'}
    if kind == 'tv':
        search_params['first_air_date_year'] = year
    else:
        search_params['year'] = year

    r = requests.get(url=f'{TMDB_BASE_URL}/search/{"tv" if kind == "tv" else "movie"}', params=search_params)
    r.raise_for_status()
    results = r.json().get('results')
    if not results:
        return BeqMetadata(title=title, year=year, audio_types=audio_types or [])

    the_movie_db_id = results[0].get('id')
    return tmdb_details_by_id(the_movie_db_id, api_key, kind, audio_types or [])


def tmdb_details_by_id(the_movie_db_id, api_key: str, kind: str = 'movie',
                       audio_types: Optional[List[str]] = None) -> BeqMetadata:
    '''
    Looks up a title's details when its TMDB id is already known, skipping
    the search step -- the GUI's "paste an ID directly" path.
    :raises requests.HTTPError: on a non-2xx response from TMDB.
    '''
    return _tmdb_details(the_movie_db_id, api_key, kind, audio_types or [])


def _tmdb_details(the_movie_db_id, api_key: str, kind: str, audio_types: List[str]) -> BeqMetadata:
    params = {'api_key': api_key}
    if kind == 'tv':
        url = f'{TMDB_BASE_URL}/tv/{the_movie_db_id}'
        params['append_to_response'] = 'content_ratings'
    else:
        url = f'{TMDB_BASE_URL}/movie/{the_movie_db_id}'
        params['append_to_response'] = 'release_dates'

    r = requests.get(url=url, params=params)
    r.raise_for_status()
    result = r.json()

    poster = result.get('poster_path') or ''
    overview = result.get('overview') or ''
    genres = result.get('genres') or []
    rating = ''
    collection = None
    runtime = ''
    year = ''

    if kind == 'tv':
        title = result['name']
        alt_title = result.get('original_name', '')
        if result.get('first_air_date'):
            year = result['first_air_date'][:4]
        ratings = (result.get('content_ratings') or {}).get('results', [])
        rating = _find_rating(ratings, 'US', key='rating') or ''
    else:
        title = result['title']
        alt_title = result.get('original_title', '')
        collection = result.get('belongs_to_collection')
        if result.get('runtime') is not None:
            runtime = str(result['runtime'])
        if result.get('release_date'):
            year = result['release_date'][:4]
        release_dates = (result.get('release_dates') or {}).get('results', [])
        rating = _find_us_certification(release_dates)

    if alt_title == title:
        alt_title = ''

    return BeqMetadata(title=title, year=year, audio_types=audio_types, genres=genres, alt_title=alt_title,
                       overview=overview, rating=rating, poster=poster, runtime=runtime,
                       the_movie_db=str(the_movie_db_id), collection=collection)


def _find_rating(results: List[dict], iso_3166_1: str, key: str) -> Optional[str]:
    for item in results:
        if item.get('iso_3166_1') == iso_3166_1:
            return item.get(key)
    return None


def _find_us_certification(release_dates: List[dict]) -> str:
    for entry in release_dates:
        if entry.get('iso_3166_1') != 'US':
            continue
        for release in entry.get('release_dates', []):
            if release.get('type') in (3, 4):
                return release.get('certification', '')
    return ''
