'''Tests for metadata resolution from a library item's external identifiers.'''
import pytest
import requests

from pipeline.library.library_metadata import resolve_meta, season_meta
from pipeline.metadata import BeqMetadata, SeasonInfo
from pipeline.library.source import LibraryItem


def _item(**overrides):
    values = {
        'id': 'example',
        'source_path': '/media/example.mkv',
        'display_name': 'Example',
        'title': 'Example',
        'year': '2024',
        'meta': {'season': '2'},
    }
    values.update(overrides)
    return LibraryItem(**values)


def test_resolve_meta_prefers_a_direct_tmdb_id(monkeypatch):
    calls = []

    def details(tmdb_id, api_key, kind, audio_types):
        calls.append((tmdb_id, api_key, kind, audio_types))
        from pipeline.metadata import BeqMetadata
        return BeqMetadata(title='Canonical', year='2023', audio_types=audio_types)

    monkeypatch.setattr('pipeline.library.library_metadata.tmdb_details_by_id', details)
    monkeypatch.setattr('pipeline.library.library_metadata.tmdb_find_by_imdb_id',
                        lambda *args: (_ for _ in ()).throw(AssertionError('IMDb must not be used')))

    meta = resolve_meta(_item(external_ids={'tmdb': '603', 'imdb': 'tt0133093'}), 'key', ['Atmos'])

    assert calls == [('603', 'key', 'movie', ['Atmos'])]
    assert meta['title'] == 'Canonical'
    assert meta['season'] == '2'


def test_resolve_meta_uses_imdb_then_tmdb_details(monkeypatch):
    calls = []

    monkeypatch.setattr('pipeline.library.library_metadata.tmdb_find_by_imdb_id',
                        lambda imdb_id, api_key, kind: calls.append(('find', imdb_id, kind)) or '42')

    def details(tmdb_id, api_key, kind, audio_types):
        calls.append(('details', tmdb_id, kind))
        from pipeline.metadata import BeqMetadata
        return BeqMetadata(title='Canonical Show', year='2020', audio_types=audio_types)

    monkeypatch.setattr('pipeline.library.library_metadata.tmdb_details_by_id', details)

    meta = resolve_meta(_item(kind='tv', external_ids={'imdb': 'tt7654321'}), 'key')

    assert calls == [('find', 'tt7654321', 'tv'), ('details', '42', 'tv')]
    assert meta['title'] == 'Canonical Show'


def test_resolve_meta_falls_back_to_title_year_search(monkeypatch):
    calls = []

    def lookup(title, year, api_key, kind, audio_types):
        calls.append((title, year, api_key, kind, audio_types))
        from pipeline.metadata import BeqMetadata
        return BeqMetadata(title=title, year=year, audio_types=audio_types)

    monkeypatch.setattr('pipeline.library.library_metadata.tmdb_find_by_imdb_id', lambda *args: None)
    monkeypatch.setattr('pipeline.library.library_metadata.tmdb_lookup', lookup)

    meta = resolve_meta(_item(external_ids={'imdb': 'tt-no-match'}), 'key', ['DTS:X'])

    assert calls == [('Example', '2024', 'key', 'movie', ['DTS:X'])]
    assert meta['title'] == 'Example'


def test_a_tvdb_series_id_is_used_after_imdb_and_before_title_search(monkeypatch):
    calls = []
    monkeypatch.setattr('pipeline.library.library_metadata.tmdb_find_by_imdb_id', lambda *args: None)
    monkeypatch.setattr('pipeline.library.library_metadata.tmdb_find_by_external_id',
                        lambda *args: calls.append(args) or '42')
    monkeypatch.setattr('pipeline.library.library_metadata.tmdb_details_by_id',
                        lambda tmdb_id, *args: BeqMetadata(title='Canonical', year='2024', the_movie_db=tmdb_id))

    meta = resolve_meta(_item(kind='tv', external_ids={'imdb': 'tt-no-match', 'tvdb': '121361'}), 'key')

    assert calls == [('121361', 'key', 'tvdb_id', 'tv')]
    assert meta['title'] == 'Canonical'


def test_an_invalid_or_missing_tvdb_id_keeps_the_title_year_fallback(monkeypatch):
    monkeypatch.setattr('pipeline.library.library_metadata.tmdb_find_by_external_id',
                        lambda *args: pytest.fail('invalid TVDB id must not be requested'))
    monkeypatch.setattr('pipeline.library.library_metadata.tmdb_lookup',
                        lambda title, year, *args: BeqMetadata(title=title, year=year))

    assert resolve_meta(_item(kind='tv', external_ids={'tvdb': 'not-an-id'}), 'key')['title'] == 'Example'


def test_tmdb_find_by_imdb_id_selects_the_requested_media_kind(monkeypatch):
    from pipeline import metadata

    class Response:
        def json(self):
            return {'movie_results': [{'id': 1}], 'tv_results': [{'id': 2}]}

        def raise_for_status(self):
            pass

    calls = []
    monkeypatch.setattr('requests.get', lambda url, params: calls.append((url, params)) or Response())

    assert metadata.tmdb_find_by_imdb_id('tt123', 'key', 'movie') == '1'
    assert metadata.tmdb_find_by_imdb_id('tt123', 'key', 'tv') == '2'
    assert calls[0][0].endswith('/find/tt123')
    assert calls[0][1] == {'api_key': 'key', 'external_source': 'imdb_id'}


def test_tmdb_find_by_tvdb_id_binds_the_external_source(monkeypatch):
    from pipeline import metadata

    class Response:
        def json(self):
            return {'tv_results': [{'id': 2}]}

        def raise_for_status(self):
            pass

    calls = []
    monkeypatch.setattr('requests.get', lambda url, params: calls.append((url, params)) or Response())

    assert metadata.tmdb_find_by_external_id('121361', 'key', 'tvdb_id', 'tv') == '2'
    assert calls == [(f'{metadata.TMDB_BASE_URL}/find/121361', {'api_key': 'key', 'external_source': 'tvdb_id'})]


# --- TV seasons (plan §11.9) --------------------------------------------------------------------------------

def _tv(**overrides):
    values = {'kind': 'tv', 'title': 'Some Show', 'season': '2', 'episodes': (3,), 'meta': {},
              'external_ids': {'tmdb': '66292'}}
    values.update(overrides)
    return _item(**values)


def _patch_show(monkeypatch, season_info=SeasonInfo('92137', 8), season_calls=None):
    monkeypatch.setattr('pipeline.library.library_metadata.tmdb_details_by_id',
                        lambda tmdb_id, api_key, kind, audio_types: BeqMetadata(
                            title='Some Show', year='2015', audio_types=audio_types, the_movie_db=str(tmdb_id)))

    def info(series_id, season, api_key):
        if season_calls is not None:
            season_calls.append((series_id, season, api_key))
        if isinstance(season_info, Exception):
            raise season_info
        return season_info

    monkeypatch.setattr('pipeline.library.library_metadata.tmdb_season_info', info)


def test_a_tv_items_season_episodes_and_tmdb_season_details_are_resolved(monkeypatch):
    calls = []
    _patch_show(monkeypatch, season_calls=calls)

    meta = resolve_meta(_tv(), 'key')

    assert calls == [('66292', '2', 'key')]
    assert (meta['season'], meta['episodes']) == ('2', [3])
    assert (meta['season_id'], meta['season_episode_count']) == ('92137', 8)
    assert BeqMetadata(**meta).structured_season


def test_a_whole_seasons_episodes_are_all_in_scope(monkeypatch):
    _patch_show(monkeypatch)

    assert resolve_meta(_tv(episodes=(1, 2, 3, 4)), 'key')['episodes'] == [1, 2, 3, 4]


def test_a_failed_season_lookup_keeps_the_library_season_and_episodes(monkeypatch):
    _patch_show(monkeypatch, season_info=requests.HTTPError('500'))

    meta = resolve_meta(_tv(), 'key')

    assert (meta['season'], meta['episodes']) == ('2', [3])
    assert not meta['season_id']
    assert not BeqMetadata(**meta).structured_season  # so it is written as a plain season and an episode note


def test_a_season_tmdb_does_not_know_keeps_the_library_values(monkeypatch):
    _patch_show(monkeypatch, season_info=None)

    assert not resolve_meta(_tv(), 'key')['season_id']


def test_no_season_lookup_for_a_film_or_a_show_without_a_season(monkeypatch):
    calls = []
    _patch_show(monkeypatch, season_calls=calls)

    film = resolve_meta(_item(kind='movie', external_ids={'tmdb': '603'}), 'key')
    show = resolve_meta(_tv(season=None, episodes=()), 'key')

    assert calls == []
    assert not film['season_id'] and not show['season'] and not show['episodes']


def test_item_meta_still_wins_over_everything_resolved(monkeypatch):
    _patch_show(monkeypatch)

    meta = resolve_meta(_tv(meta={'season': '9', 'season_id': 'mine'}), 'key')

    assert meta['season'] == '9' and meta['season_id'] == 'mine'


def test_season_meta_is_just_what_the_library_says():
    assert season_meta(_tv()) == {'season': '2', 'episodes': [3]}
    assert season_meta(_item()) == {}
    assert season_meta(_tv(episodes=())) == {'season': '2'}
