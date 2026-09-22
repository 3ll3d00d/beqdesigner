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
import re
from dataclasses import dataclass, field
from typing import List, Optional
from urllib.parse import quote, quote_plus

import requests

TMDB_BASE_URL = 'https://api.themoviedb.org/3'
_API_KEY_PARAMETER = re.compile(r"(api_key=)[^&\s'\")\]]*", re.IGNORECASE)
TMDB_TIMEOUT_SECONDS = 20   # a hung connection must end in an error rather than wait for ever (search and details requests)


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
    # Which episodes of `season` this filter covers -- a whole season's worth, or a single episode. Empty means
    # unspecified. See season_element() for how it reaches the catalogue.
    episodes: List[int] = field(default_factory=list)
    season_id: str = ''  # TMDB's id for the season; beqcatalogue's structured season needs it
    season_episode_count: int = 0  # how many episodes the whole season has (TMDB); 0 if unknown

    def __post_init__(self):
        if not self.sort_title:
            self.sort_title = default_sort_title(self.title)

    @property
    def structured_season(self) -> bool:
        '''
        True if the season can be written in beqcatalogue's structured form, which needs TMDB's season id and
        the season's episode count (the catalogue calls a season complete when every one of `count` is listed).
        '''
        return bool(self.season and self.season_id and self.season_episode_count > 0)

    def season_element(self):
        '''
        :return: the beq_season value for to_dict(): beqcatalogue's structured season
            `{'id', 'number', 'episode_count', 'episodes'}` when it can be written (see structured_season), else the
            plain season text. `episodes` is omitted when none are given, which the catalogue reads as the whole
            season.
        '''
        if not self.structured_season:
            return self.season
        element = {'id': self.season_id, 'number': self.season}
        episodes = sorted(set(self.episodes))
        if episodes:
            element['episode_count'] = self.season_episode_count
            element['episodes'] = ','.join(str(e) for e in episodes)
        return element

    def note_with_episodes(self) -> str:
        '''
        :return: the beq_note. Where the structured season is not available the catalogue can still read a
            contiguous episode range from the note (`E3`, `E1-8`), so that is written -- but only into an empty
            note, since a note somebody wrote must not be replaced. Non-contiguous episodes cannot be expressed
            that way and are left out.
        '''
        if self.note or self.structured_season or not self.season or not self.episodes:
            return self.note
        episodes = sorted(set(self.episodes))
        if episodes[-1] - episodes[0] + 1 != len(episodes):
            return self.note
        return f"E{episodes[0]}" if len(episodes) == 1 else f"E{episodes[0]}-{episodes[-1]}"

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
            'beq_season': self.season_element(),
            'beq_note': self.note_with_episodes(),
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

    r = requests.get(url=f'{TMDB_BASE_URL}/search/{"tv" if kind == "tv" else "movie"}', params=search_params,
                     timeout=TMDB_TIMEOUT_SECONDS)
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


def tmdb_find_by_external_id(external_id: str, api_key: str, external_source: str, kind: str = 'movie') -> Optional[str]:
    '''
    Resolve an external id through TMDB's external-id endpoint.

    :return: the matching TMDB id, or None when TMDB has no result for this
        media kind. The caller can then use the normal title/year search.
    :raises ValueError: if kind is not movie or tv.
    :raises requests.HTTPError: on a non-2xx TMDB response.
    '''
    if kind not in ('movie', 'tv'):
        raise ValueError(f"kind must be 'movie' or 'tv', got {kind!r}")
    r = requests.get(url=f'{TMDB_BASE_URL}/find/{external_id}',
                     params={'api_key': api_key, 'external_source': external_source})
    r.raise_for_status()
    results = r.json().get('movie_results' if kind == 'movie' else 'tv_results') or []
    return str(results[0]['id']) if results else None


def tmdb_find_by_imdb_id(imdb_id: str, api_key: str, kind: str = 'movie') -> Optional[str]:
    '''Resolve an IMDb id through TMDB's external-id endpoint.'''
    return tmdb_find_by_external_id(imdb_id, api_key, 'imdb_id', kind)


@dataclass(frozen=True)
class SeasonInfo:
    id: str  # TMDB's id for the season (beqcatalogue's <beq_season id>)
    episode_count: int


def tmdb_season_info(series_id, season_number, api_key: str) -> Optional[SeasonInfo]:
    '''
    A TV season's TMDB id and how many episodes it has, which beqcatalogue's structured season needs.
    :param series_id: TMDB's id for the series (BeqMetadata.the_movie_db).
    :return: None if TMDB has no such season.
    :raises requests.HTTPError: on any other non-2xx response.
    '''
    r = requests.get(url=f'{TMDB_BASE_URL}/tv/{series_id}/season/{season_number}', params={'api_key': api_key})
    if r.status_code == 404:
        return None
    r.raise_for_status()
    result = r.json()
    if result.get('id') is None:
        return None
    return SeasonInfo(id=str(result['id']), episode_count=len(result.get('episodes') or []))


def parse_episodes(text: str) -> List[int]:
    '''
    :param text: episode numbers and ranges, e.g. "1-3, 5".
    :return: the distinct episode numbers, ascending; [] for blank text.
    :raises ValueError: for anything that is not positive numbers and ascending ranges.
    '''
    episodes = set()
    for part in (p.strip() for p in text.split(',')):
        if not part:
            continue
        low, dash, high = part.partition('-')
        try:
            first, last = int(low), int(high) if dash else int(low)
        except ValueError:
            raise ValueError(f"'{part}' is not an episode number or range") from None
        if first < 1 or last < first:
            raise ValueError(f"'{part}' is not a valid episode number or range")
        episodes.update(range(first, last + 1))
    return sorted(episodes)


def format_episodes(episodes) -> str:
    ''' The inverse of parse_episodes(): [1, 2, 3, 5] -> "1-3, 5". '''
    ranges, run = [], []
    for episode in sorted(set(episodes)):
        if run and episode == run[-1] + 1:
            run.append(episode)
        else:
            if run:
                ranges.append(run)
            run = [episode]
    if run:
        ranges.append(run)
    return ', '.join(str(r[0]) if len(r) == 1 else f"{r[0]}-{r[-1]}" for r in ranges)


def _tmdb_details(the_movie_db_id, api_key: str, kind: str, audio_types: List[str]) -> BeqMetadata:
    params = {'api_key': api_key}
    if kind == 'tv':
        url = f'{TMDB_BASE_URL}/tv/{the_movie_db_id}'
        params['append_to_response'] = 'content_ratings'
    else:
        url = f'{TMDB_BASE_URL}/movie/{the_movie_db_id}'
        params['append_to_response'] = 'release_dates'

    r = requests.get(url=url, params=params, timeout=TMDB_TIMEOUT_SECONDS)
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


def redact(text: str, secret: str = '') -> str:
    '''
    `text` with the TMDB API key taken out. `requests` puts the whole URL -- `?api_key=...` -- in the text of an HTTPError or a
    ConnectionError, and that text goes to a status line, a report and the log. Any `api_key=` parameter is masked, and so is
    `secret` itself, plain or URL-encoded, wherever it appears (from 8 characters: a real key is 32; a shorter "secret" would
    mask ordinary words).
    '''
    if len(secret) >= 8:
        for form in {secret, quote(secret, safe=''), quote(secret), quote_plus(secret)}:
            text = text.replace(form, '***')
    return _API_KEY_PARAMETER.sub(r'\1***', text)
