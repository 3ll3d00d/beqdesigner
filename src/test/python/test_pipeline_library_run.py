'''Tests for library-source orchestration and explicit sync delegation.'''
from dataclasses import dataclass

import requests

from pipeline.library.design_cache import DesignCacheResult
from pipeline.library.run import LibraryRunConfig, run_library
from pipeline.library.source import LibraryItem
from pipeline.library.sync import sync_library
from pipeline.review import QueueEntry


@dataclass
class _Source:
    items: list[LibraryItem]
    query: dict | None = None

    def list_items(self, **query):
        self.query = query
        return self.items


class _Session:
    def __init__(self):
        self.channel_calls = []

    def load_channels(self, path, layout):
        self.channel_calls.append((path, layout))
        return {'LFE': [1, 2]} if path.endswith('multichannel.wav') else {}


def _item(identifier):
    return LibraryItem(id=identifier, source_path=f'/media/{identifier}.mkv', display_name=identifier,
                       fingerprint=f'fingerprint-{identifier}', meta={'season': '2'})


def test_run_library_composes_caches_resolves_metadata_and_threads_multichannel(tmp_path, monkeypatch):
    source = _Source([_item('one')])
    session = _Session()
    extract_calls = []
    design_calls = []

    monkeypatch.setattr('pipeline.library.run.Session', lambda config: session)

    def extract(fake_session, item, item_dir, config, mono_mix, force):
        extract_calls.append((item.id, mono_mix, force))
        return f'{item_dir}/{"mono" if mono_mix else "multichannel"}.wav', mono_mix

    monkeypatch.setattr('pipeline.library.run.extract_if_needed', extract)
    monkeypatch.setattr('pipeline.library.run.read_channel_layout_name', lambda path: '5.1')
    monkeypatch.setattr('pipeline.library.run.resolve_meta',
                        lambda item, key, audio_types: {'title': item.title or item.display_name,
                                                        'audio_types': list(audio_types), **item.meta})

    def design(fake_session, item, wav_path, designer, queue_dir, config, **kwargs):
        design_calls.append((item, wav_path, designer, queue_dir, config, kwargs))
        return DesignCacheResult(QueueEntry(id=item.id, fs=1000, meta={}, curve={}), designed=True)

    monkeypatch.setattr('pipeline.library.run.design_if_needed', design)
    completed = []
    config = LibraryRunConfig(work_dir=str(tmp_path / 'work'), queue_dir=str(tmp_path / 'queue'),
                              designer='test', keep_multichannel=True, tmdb_api_key='key',
                              audio_types=('Atmos',))

    report = run_library(source, config, lambda item_id: completed.append(item_id), content_type='movie')

    assert source.query == {'content_type': 'movie'}
    assert extract_calls == [('one', True, False), ('one', False, False)]
    assert report.extracted == ['one']
    assert report.cached == []
    assert report.designed == ['one']
    assert report.design_cached == []
    assert report.failed == []
    assert completed == ['one']
    assert session.channel_calls == [(f'{tmp_path}/work/one/multichannel.wav', '5.1')]
    assert design_calls[0][5]['meta']() == {'title': 'one', 'audio_types': ['Atmos'], 'season': '2'}
    assert design_calls[0][5]['channels'] == {'LFE': [1, 2]}
    assert design_calls[0][5]['multichannel_wav_path'].endswith('multichannel.wav')
    assert design_calls[0][5]['project_dir'] == f'{tmp_path}/work/one'


def test_run_library_marks_a_fully_cached_item_and_isolates_a_failure(tmp_path, monkeypatch):
    source = _Source([_item('cached'), _item('broken')])
    monkeypatch.setattr('pipeline.library.run.Session', lambda config: _Session())
    completed = []

    def extract(session, item, item_dir, config, mono_mix, force):
        if item.id == 'broken':
            raise ValueError('unreadable audio')
        return f'{item_dir}/mono.wav', True

    monkeypatch.setattr('pipeline.library.run.extract_if_needed', extract)
    monkeypatch.setattr('pipeline.library.run.design_if_needed',
                        lambda *args, **kwargs: DesignCacheResult(
                            QueueEntry(id='cached', fs=1000, meta={}, curve={}), designed=False))

    report = run_library(source, LibraryRunConfig(work_dir=str(tmp_path / 'work'),
                                                   queue_dir=str(tmp_path / 'queue'), designer='test'),
                         completed.append)

    assert report.cached == ['cached']
    assert report.design_cached == ['cached']
    assert report.extracted == []
    assert report.designed == []
    assert report.failed == [('broken', 'ValueError: unreadable audio')]
    assert completed == ['cached', 'broken']


def _run_with_meta(tmp_path, monkeypatch, resolver, **config_kwargs):
    ''' Runs one item whose design_if_needed() invokes (or not) whatever meta it was handed. '''
    monkeypatch.setattr('pipeline.library.run.Session', lambda config: _Session())
    monkeypatch.setattr('pipeline.library.run.extract_if_needed',
                        lambda session, item, item_dir, config, mono_mix, force: (f'{item_dir}/mono.wav', True))
    monkeypatch.setattr('pipeline.library.run.resolve_meta', resolver)
    seen = []

    def design(session, item, wav_path, designer, queue_dir, config, meta=None, **kwargs):
        if config_kwargs.pop('designs', True):
            seen.append(meta() if callable(meta) else meta)
        return DesignCacheResult(QueueEntry(id=item.id, fs=1000, meta={}, curve={}), designed=bool(seen))

    monkeypatch.setattr('pipeline.library.run.design_if_needed', design)
    config = LibraryRunConfig(work_dir=str(tmp_path / 'work'), queue_dir=str(tmp_path / 'queue'), designer='test',
                              tmdb_api_key='key')
    return run_library(_Source([_item('one')]), config), seen


def test_run_library_does_not_resolve_metadata_for_an_item_that_is_not_designed(tmp_path, monkeypatch):
    lookups = []
    report, seen = _run_with_meta(tmp_path, monkeypatch, lambda *args: lookups.append(args) or {}, designs=False)

    assert lookups == []
    assert seen == []
    assert report.failed == []


def test_run_library_degrades_to_library_metadata_when_tmdb_fails(tmp_path, monkeypatch):
    def failing(item, key, audio_types):
        raise requests.HTTPError('401 Unauthorized')

    report, seen = _run_with_meta(tmp_path, monkeypatch, failing)

    assert seen == [{'season': '2'}]
    assert report.failed == []
    assert report.meta_unresolved == [('one', 'HTTPError: 401 Unauthorized')]


def test_sync_library_is_a_parameter_preserving_publish_call_through(monkeypatch):
    calls = []
    monkeypatch.setattr('pipeline.library.sync.publish_reviewed_queue',
                        lambda *args, **kwargs: calls.append((args, kwargs)) or [{'id': 'one'}])
    xml_repo = object()
    images_repo = object()

    result = sync_library('queue', xml_repo, meta_defaults={'source': 'Disc'}, images_repo=images_repo,
                          image_owner='owner', image_repo_name='images', xml_dir='xml', image_dir='img',
                          work_dir='work')

    assert result == [{'id': 'one'}]
    assert calls[0][0] == ('queue', xml_repo)
    assert calls[0][1]['meta_defaults'] == {'source': 'Disc'}
    assert calls[0][1]['images_repo'] is images_repo
    assert calls[0][1]['work_dir'] == 'work'
