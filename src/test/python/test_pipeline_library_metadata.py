'''Tests for metadata resolution from a library item's external identifiers.'''
from pipeline.library.library_metadata import resolve_meta
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
