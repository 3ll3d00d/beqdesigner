'''
Phase 2 (item 5) of design/pipeline-implementation-plan.md: BeqMetadata +
tmdb_lookup(), with the TMDB HTTP call mocked -- no network access required.
'''
import pytest


class _FakeResponse:
    def __init__(self, json_body, status_code=200):
        self._json = json_body
        self.status_code = status_code

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"{self.status_code}")


def test_beq_metadata_default_sort_title_strips_leading_the():
    from pipeline.metadata import BeqMetadata

    m = BeqMetadata(title='The Matrix', year='1999')
    assert m.sort_title == 'matrix'

    m2 = BeqMetadata(title='Inception', year='2010')
    assert m2.sort_title == 'inception'

    m3 = BeqMetadata(title='The Matrix', year='1999', sort_title='custom')
    assert m3.sort_title == 'custom'


def test_beq_metadata_to_dict_matches_publish_shape():
    from pipeline.metadata import BeqMetadata

    m = BeqMetadata(title='Ready Player One', year='2018', audio_types=['Atmos'],
                    genres=[{'id': 28, 'name': 'Action'}])
    d = m.to_dict()
    assert d['beq_title'] == 'Ready Player One'
    assert d['beq_year'] == '2018'
    assert d['beq_audioTypes'] == ['Atmos']
    assert d['beq_genres'] == [{'id': 28, 'name': 'Action'}]
    assert d['beq_sortTitle'] == 'ready player one'
    assert d['beq_gain'] is None


def test_validate_requires_title_year_and_audio_type():
    from pipeline.metadata import BeqMetadata, validate

    assert validate(BeqMetadata(title='X', year='2020', audio_types=['Atmos'])) == []

    problems = validate(BeqMetadata(title='', year='', audio_types=[]))
    assert 'title is required' in problems
    assert 'year is required' in problems
    assert 'at least one audio type is required' in problems


def test_tmdb_lookup_movie(monkeypatch):
    from pipeline import metadata as md

    search_response = _FakeResponse({'results': [{'id': 335984}]})
    details_response = _FakeResponse({
        'title': 'Ready Player One',
        'original_title': 'Ready Player One',
        'poster_path': '/abc.jpg',
        'overview': 'A VR adventure.',
        'genres': [{'id': 28, 'name': 'Action'}, {'id': 878, 'name': 'Science Fiction'}],
        'belongs_to_collection': None,
        'runtime': 140,
        'release_date': '2018-03-29',
        'release_dates': {'results': [
            {'iso_3166_1': 'US', 'release_dates': [{'type': 3, 'certification': 'PG-13'}]},
        ]},
    })

    calls = []

    def fake_get(url, params, timeout=None):
        calls.append((url, params))
        return search_response if 'search' in url else details_response

    monkeypatch.setattr('requests.get', fake_get)

    result = md.tmdb_lookup('Ready Player One', '2018', api_key='dummy-key', kind='movie',
                            audio_types=['Atmos'])

    assert result.title == 'Ready Player One'
    assert result.year == '2018'
    assert result.alt_title == ''  # same as title -> cleared
    assert result.overview == 'A VR adventure.'
    assert result.poster == '/abc.jpg'
    assert result.runtime == '140'
    assert result.rating == 'PG-13'
    assert result.the_movie_db == '335984'
    assert result.audio_types == ['Atmos']
    assert {'id': 28, 'name': 'Action'} in result.genres

    assert calls[0][1]['api_key'] == 'dummy-key'
    assert 'search/movie' in calls[0][0]
    assert 'movie/335984' in calls[1][0]


def test_tmdb_details_by_id_skips_the_search_step(monkeypatch):
    from pipeline import metadata as md

    details_response = _FakeResponse({
        'title': 'Ready Player One',
        'original_title': 'Ready Player One',
        'poster_path': '/abc.jpg',
        'overview': 'A VR adventure.',
        'genres': [{'id': 28, 'name': 'Action'}],
        'belongs_to_collection': None,
        'runtime': 140,
        'release_date': '2018-03-29',
        'release_dates': {'results': []},
    })

    calls = []

    def fake_get(url, params, timeout=None):
        calls.append((url, params))
        return details_response

    monkeypatch.setattr('requests.get', fake_get)

    result = md.tmdb_details_by_id(335984, api_key='dummy-key', kind='movie', audio_types=['Atmos'])

    assert result.title == 'Ready Player One'
    assert result.the_movie_db == '335984'
    assert result.audio_types == ['Atmos']
    assert len(calls) == 1  # no search call
    assert 'movie/335984' in calls[0][0]


def test_tmdb_lookup_tv(monkeypatch):
    from pipeline import metadata as md

    search_response = _FakeResponse({'results': [{'id': 42}]})
    details_response = _FakeResponse({
        'name': 'Some Show',
        'original_name': 'Some Show Original',
        'poster_path': '/xyz.jpg',
        'overview': 'A show.',
        'genres': [],
        'first_air_date': '2020-05-01',
        'content_ratings': {'results': [{'iso_3166_1': 'US', 'rating': 'TV-14'}]},
    })

    def fake_get(url, params, timeout=None):
        return search_response if 'search' in url else details_response

    monkeypatch.setattr('requests.get', fake_get)

    result = md.tmdb_lookup('Some Show', '2020', api_key='dummy-key', kind='tv')

    assert result.title == 'Some Show'
    assert result.alt_title == 'Some Show Original'
    assert result.year == '2020'
    assert result.rating == 'TV-14'


def test_tmdb_lookup_no_results_returns_minimal_metadata(monkeypatch):
    from pipeline import metadata as md

    monkeypatch.setattr('requests.get', lambda url, params, timeout=None: _FakeResponse({'results': []}))

    result = md.tmdb_lookup('Nonexistent Title', '2099', api_key='dummy-key', audio_types=['DTS-X'])
    assert result.title == 'Nonexistent Title'
    assert result.year == '2099'
    assert result.audio_types == ['DTS-X']
    assert result.the_movie_db == ''


def test_tmdb_lookup_invalid_kind_raises():
    from pipeline.metadata import tmdb_lookup
    with pytest.raises(ValueError, match="kind must be"):
        tmdb_lookup('X', '2020', api_key='k', kind='podcast')


def test_tmdb_lookup_raises_on_http_error(monkeypatch):
    from pipeline import metadata as md

    monkeypatch.setattr('requests.get', lambda url, params, timeout=None: _FakeResponse({}, status_code=500))

    import requests
    with pytest.raises(requests.HTTPError):
        md.tmdb_lookup('X', '2020', api_key='k')


@pytest.mark.parametrize('kind', ['movie', 'tv'])
def test_the_tmdb_search_and_details_requests_have_a_timeout(monkeypatch, kind):
    ''' A hung connection must end in an error: the title page's Reload waits on it, and is disabled meanwhile. '''
    from pipeline import metadata as md

    seen = []

    def fake_get(url, params, timeout=None):
        seen.append((url, timeout))
        body = {'results': [{'id': 7}]} if '/search/' in url else {'name': 'A', 'title': 'A', 'release_dates': {'results': []}}
        return _FakeResponse(body)

    monkeypatch.setattr('requests.get', fake_get)

    md.tmdb_lookup('A', '2020', api_key='k', kind=kind)
    md.tmdb_details_by_id(7, api_key='k', kind=kind)

    assert len(seen) == 3 and [t for _, t in seen] == [md.TMDB_TIMEOUT_SECONDS] * 3
    assert 0 < md.TMDB_TIMEOUT_SECONDS <= 30


def test_pipeline_metadata_module_has_no_qtpy_import():
    import ast
    import pathlib
    source = (pathlib.Path(__file__).parent / '..' / '..' / 'main' / 'python' / 'pipeline' / 'metadata.py').resolve()
    tree = ast.parse(source.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(n.name.startswith('qtpy') for n in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith('qtpy')


# --- TV seasons (plan §11.9) --------------------------------------------------------------------------------

def test_season_info_reads_the_season_id_and_counts_its_episodes(monkeypatch):
    import pipeline.metadata as metadata

    class _Response:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {'id': 92137, 'season_number': 1, 'episodes': [{'episode_number': n} for n in range(1, 9)]}

    seen = {}
    monkeypatch.setattr(metadata.requests, 'get', lambda url, params: seen.update(url=url) or _Response())

    info = metadata.tmdb_season_info(66292, 1, 'key')

    assert info == metadata.SeasonInfo(id='92137', episode_count=8)
    assert seen['url'].endswith('/tv/66292/season/1')


def test_season_info_is_none_for_an_unknown_season(monkeypatch):
    import pipeline.metadata as metadata

    class _Missing:
        status_code = 404
        def raise_for_status(self): raise AssertionError('a 404 is an answer, not an error')

    monkeypatch.setattr(metadata.requests, 'get', lambda url, params: _Missing())

    assert metadata.tmdb_season_info(1, 99, 'key') is None


def test_season_info_raises_on_a_server_error(monkeypatch):
    import pipeline.metadata as metadata
    import requests

    class _Broken:
        status_code = 500
        def raise_for_status(self): raise requests.HTTPError('500')

    monkeypatch.setattr(metadata.requests, 'get', lambda url, params: _Broken())

    with pytest.raises(requests.HTTPError):
        metadata.tmdb_season_info(1, 1, 'key')


@pytest.mark.parametrize('text, episodes', [
    ('', []), ('  ', []), ('3', [3]), ('1-3', [1, 2, 3]), ('1-3, 5', [1, 2, 3, 5]), ('5, 1 ,3-4', [1, 3, 4, 5]),
    ('2, 2, 1-2', [1, 2]), ('1,,2', [1, 2]),
])
def test_parse_episodes(text, episodes):
    from pipeline.metadata import parse_episodes
    assert parse_episodes(text) == episodes


@pytest.mark.parametrize('text', ['x', '0', '3-1', '1-', '-2', '1.5', 'E3'])
def test_parse_episodes_rejects_junk(text):
    from pipeline.metadata import parse_episodes
    with pytest.raises(ValueError):
        parse_episodes(text)


def test_format_episodes_compresses_runs_and_round_trips():
    from pipeline.metadata import format_episodes, parse_episodes
    assert format_episodes([1, 2, 3, 5]) == '1-3, 5'
    assert format_episodes([5, 3, 4, 1]) == '1, 3-5'
    assert format_episodes([]) == ''
    assert parse_episodes(format_episodes([2, 3, 7, 8, 9])) == [2, 3, 7, 8, 9]
