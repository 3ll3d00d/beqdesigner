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

    assert seen == [{'title': 'one', 'season': '2'}]
    assert report.failed == []
    assert report.meta_unresolved == [('one', 'HTTPError: 401 Unauthorized')]


def _keep_multichannel_run(tmp_path, monkeypatch, channel_count):
    monkeypatch.setattr('pipeline.library.run.Session', lambda config: _Session())
    extract_calls = []

    def extract(session, item, item_dir, config, mono_mix, force):
        extract_calls.append(mono_mix)
        return f'{item_dir}/{"mono" if mono_mix else "multichannel"}.wav', False

    monkeypatch.setattr('pipeline.library.run.extract_if_needed', extract)
    monkeypatch.setattr('pipeline.library.run.read_source_channel_count', lambda item_dir: channel_count)
    monkeypatch.setattr('pipeline.library.run.read_channel_layout_name', lambda item_dir: '5.1')
    monkeypatch.setattr('pipeline.library.run.design_if_needed',
                        lambda *args, **kwargs: DesignCacheResult(
                            QueueEntry(id='one', fs=1000, meta={}, curve={}), designed=True))
    config = LibraryRunConfig(work_dir=str(tmp_path / 'work'), queue_dir=str(tmp_path / 'queue'), designer='test',
                              keep_multichannel=True)
    report = run_library(_Source([_item('one')]), config)
    return extract_calls, report


def test_run_library_skips_the_kept_extraction_for_a_known_mono_source(tmp_path, monkeypatch):
    extract_calls, report = _keep_multichannel_run(tmp_path, monkeypatch, channel_count=1)

    assert extract_calls == [True]
    assert report.failed == []
    assert report.designed == ['one']


def test_run_library_still_keeps_multichannel_when_the_channel_count_is_unknown_or_greater_than_one(
        tmp_path, monkeypatch):
    for channel_count in (None, 2, 6):
        extract_calls, _ = _keep_multichannel_run(tmp_path, monkeypatch, channel_count=channel_count)
        assert extract_calls == [True, False]


def test_run_library_reports_items_whose_edited_project_was_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr('pipeline.library.run.Session', lambda config: _Session())
    monkeypatch.setattr('pipeline.library.run.extract_if_needed',
                        lambda session, item, item_dir, config, mono_mix, force: (f'{item_dir}/mono.wav', True))

    def design(session, item, *args, **kwargs):
        projects = {'mono': item.id != 'edited', 'multichannel': None}
        return DesignCacheResult(QueueEntry(id=item.id, fs=1000, meta={}, curve={}), designed=True, projects=projects)

    monkeypatch.setattr('pipeline.library.run.design_if_needed', design)
    config = LibraryRunConfig(work_dir=str(tmp_path / 'work'), queue_dir=str(tmp_path / 'queue'), designer='test')

    report = run_library(_Source([_item('plain'), _item('edited')]), config)

    assert report.designed == ['plain', 'edited']
    assert report.project_edit_preserved == ['edited']


def test_sync_library_publishes_without_pushing_then_commits_the_batch(monkeypatch):
    from pipeline.library.commit import CatalogueCommit, RepoCommit
    calls = []
    commits = []
    monkeypatch.setattr('pipeline.library.sync.publish_reviewed_queue',
                        lambda *args, **kwargs: calls.append((args, kwargs)) or [{'id': 'one'}])
    monkeypatch.setattr('pipeline.library.sync.commit_catalogue', lambda *args, **kwargs: commits.append((args, kwargs))
                        or CatalogueCommit(xml=RepoCommit('xml', ['xml/one.xml'], 'abc', True)))
    xml_repo = object()
    images_repo = object()

    result = sync_library('queue', xml_repo, meta_defaults={'source': 'Disc'}, images_repo=images_repo,
                          image_owner='owner', image_repo_name='images', xml_dir='xml', image_dir='img',
                          work_dir='work')

    assert result == [{'id': 'one', 'xml_commit': 'abc'}]
    assert calls[0][0] == ('queue', xml_repo)
    assert calls[0][1]['meta_defaults'] == {'source': 'Disc'}
    assert calls[0][1]['images_repo'] is images_repo
    assert calls[0][1]['work_dir'] == 'work'
    assert calls[0][1]['push'] is False  # written only; the batch commit below does the git work
    assert commits[0][0] == ('queue', xml_repo, images_repo)
    assert commits[0][1] == {'xml_dir': 'xml', 'image_dir': 'img', 'push': True}


# --- TV seasons (plan §11.9) --------------------------------------------------------------------------------

import os

import numpy as np
import pytest
import soundfile as sf


def _episode(series, season, episode):
    return LibraryItem(id=f'{series}-{season}-{episode}', source_path=f'/tv/{series}/{episode}.mkv',
                       display_name=f'{series} S{season}E{episode}', title=series, kind='tv', season=str(season),
                       episodes=(episode,), fingerprint=f'fp-{series}-{season}-{episode}')


def _resolved(meta):
    return meta() if callable(meta) else meta


def _season_run(tmp_path, monkeypatch, items, tv_mode='season', fail=(), **config):
    ''' run_library() with real audio for the extract stage, so the season track really is joined. '''
    monkeypatch.setattr('pipeline.library.run.Session', lambda cfg: _Session())
    extracted, designs = [], []

    def extract(session, item, item_dir, cfg, mono_mix, force):
        if item.id in fail:
            raise FileNotFoundError(f'{item.id} is missing')
        extracted.append(item.id)
        path = os.path.join(item_dir, 'mono.wav')
        os.makedirs(item_dir, exist_ok=True)
        length = (item.episodes or (1,))[0]  # a film has no episode number
        sf.write(path, np.full((1000 * length, 1), length / 10), 1000, subtype='PCM_24')
        return path, False

    def design(session, item, wav_path, designer, queue_dir, cfg, **kwargs):
        designs.append({'item': item, 'wav': wav_path, 'meta': kwargs['meta'], 'kwargs': kwargs})
        return DesignCacheResult(QueueEntry(id=item.id, fs=1000, meta={}, curve={}), designed=True)

    monkeypatch.setattr('pipeline.library.run.extract_if_needed', extract)
    monkeypatch.setattr('pipeline.library.run.design_if_needed', design)
    run_config = LibraryRunConfig(work_dir=str(tmp_path / 'work'), queue_dir=str(tmp_path / 'queue'), designer='test',
                                  tv_mode=tv_mode, **config)
    done = []
    report = run_library(_Source(items), run_config, done.append)
    return report, extracted, designs, done


def test_episode_mode_designs_each_episode_separately_with_its_own_metadata(tmp_path, monkeypatch):
    items = [_episode('Show', 1, e) for e in (1, 2)]

    report, extracted, designs, done = _season_run(tmp_path, monkeypatch, items, tv_mode='episode')

    assert [d['item'].id for d in designs] == ['Show-1-1', 'Show-1-2']
    assert [_resolved(d['meta'])['episodes'] for d in designs] == [[1], [2]]
    assert report.seasons == {} and done == ['Show-1-1', 'Show-1-2']


def test_season_mode_extracts_every_episode_joins_them_and_designs_once(tmp_path, monkeypatch):
    items = [_episode('Show', 1, e) for e in (3, 1, 2)]

    report, extracted, designs, done = _season_run(tmp_path, monkeypatch, items)

    assert sorted(extracted) == ['Show-1-1', 'Show-1-2', 'Show-1-3']
    assert len(designs) == 1
    design = designs[0]
    assert design['item'].display_name == 'Show Season 1'
    assert design['item'].episodes == (1, 2, 3)
    assert design['wav'] == str(tmp_path / 'work' / design['item'].id / 'mono.wav')
    assert len(sf.read(design['wav'])[0]) == 6000  # 1 + 2 + 3 seconds at 1 kHz
    assert design['kwargs']['project_dir'] == str(tmp_path / 'work' / design['item'].id)
    assert report.designed == [design['item'].id]
    assert report.seasons == {design['item'].id: ['Show-1-1', 'Show-1-2', 'Show-1-3']}
    assert report.extracted == extracted and report.failed == []
    assert done == [design['item'].id]  # one progress tick per season, not per episode


def test_season_mode_marks_the_episodes_in_scope_in_the_metadata(tmp_path, monkeypatch):
    _, _, designs, _ = _season_run(tmp_path, monkeypatch, [_episode('Show', 1, e) for e in (1, 2)])

    meta = _resolved(designs[0]['meta'])

    assert meta['season'] == '1' and meta['episodes'] == [1, 2]


def test_season_mode_keeps_films_and_other_series_as_they_were(tmp_path, monkeypatch):
    film = LibraryItem(id='film', source_path='/f.mkv', display_name='Film', title='Film', fingerprint='fp')
    items = [_episode('Show', 1, 1), film, _episode('Other', 2, 1)]

    report, _, designs, _ = _season_run(tmp_path, monkeypatch, items)

    assert [d['item'].display_name for d in designs] == ['Show Season 1', 'Film', 'Other Season 2']
    assert set(report.seasons) == {designs[0]['item'].id, designs[2]['item'].id}


def test_an_episode_that_will_not_extract_is_reported_and_left_out_of_scope(tmp_path, monkeypatch):
    items = [_episode('Show', 1, e) for e in (1, 2, 3)]

    report, _, designs, _ = _season_run(tmp_path, monkeypatch, items, fail={'Show-1-2'})

    assert report.failed == [('Show-1-2', 'FileNotFoundError: Show-1-2 is missing')]
    assert designs[0]['item'].episodes == (1, 3)  # the season is marked with what really went into it
    assert _resolved(designs[0]['meta'])['episodes'] == [1, 3]
    assert list(report.seasons.values()) == [['Show-1-1', 'Show-1-3']]
    assert len(sf.read(designs[0]['wav'])[0]) == 4000


def test_a_season_none_of_whose_episodes_extract_fails_as_one_item(tmp_path, monkeypatch):
    items = [_episode('Show', 1, e) for e in (1, 2)]

    report, _, designs, done = _season_run(tmp_path, monkeypatch, items, fail={'Show-1-1', 'Show-1-2'})

    assert designs == []
    assert (done[0], 'ValueError: none of the 2 episodes could be extracted') in report.failed
    assert len(report.failed) == 3  # each episode, and the season


def test_season_mode_does_not_keep_multichannel(tmp_path, monkeypatch):
    _, _, designs, _ = _season_run(tmp_path, monkeypatch, [_episode('Show', 1, 1)], keep_multichannel=True)

    assert designs[0]['kwargs']['multichannel_wav_path'] is None
    assert designs[0]['kwargs']['channels'] is None


def test_season_mode_needs_no_tmdb_key_to_mark_the_episodes(tmp_path, monkeypatch):
    _, _, designs, _ = _season_run(tmp_path, monkeypatch, [_episode('Show', 1, e) for e in (2, 3)])

    assert _resolved(designs[0]['meta']) == {'title': 'Show', 'season': '1', 'episodes': [2, 3]}


def test_the_season_keeps_one_stable_id_across_runs(tmp_path, monkeypatch):
    items = [_episode('Show', 1, e) for e in (1, 2)]
    _, _, first, _ = _season_run(tmp_path, monkeypatch, items)
    _, _, second, _ = _season_run(tmp_path, monkeypatch, items[:1])

    assert second[0]['item'].id == first[0]['item'].id  # so the same review-queue entry and cache are reused


def test_an_unknown_tv_mode_is_rejected_up_front():
    with pytest.raises(ValueError, match='tv_mode'):
        LibraryRunConfig(work_dir='/w', queue_dir='/q', designer='x', tv_mode='series')


def test_run_library_names_the_entry_from_the_library_when_no_tmdb_key_is_configured(tmp_path, monkeypatch):
    monkeypatch.setattr('pipeline.library.run.Session', lambda config: _Session())
    monkeypatch.setattr('pipeline.library.run.extract_if_needed',
                        lambda session, item, item_dir, config, mono_mix, force: (f'{item_dir}/mono.wav', True))
    seen = []

    def design(session, item, wav_path, designer, queue_dir, config, meta=None, **kwargs):
        seen.append(meta() if callable(meta) else meta)
        return DesignCacheResult(QueueEntry(id=item.id, fs=1000, meta={}, curve={}), designed=True)

    monkeypatch.setattr('pipeline.library.run.design_if_needed', design)
    titled = LibraryItem(id='jriver-3fa9c2-1234', source_path='/media/heat.mkv', display_name='Heat (1995)',
                         title='Heat', year='1995', fingerprint='f1')
    untitled = LibraryItem(id='jriver-3fa9c2-5678', source_path='/media/x.mkv', display_name='x.mkv', fingerprint='f2')
    config = LibraryRunConfig(work_dir=str(tmp_path / 'work'), queue_dir=str(tmp_path / 'queue'), designer='test')

    run_library(_Source([titled, untitled]), config)

    assert seen == [{'title': 'Heat', 'year': '1995'}, {'title': 'x.mkv'}]


def test_library_meta_carries_the_title_year_season_and_episodes():
    from pipeline.library.library_metadata import library_meta
    item = LibraryItem(id='a', source_path='/a.mkv', display_name='a', title='Show', year='2019', season='1',
                       episodes=(2,))

    assert library_meta(item) == {'title': 'Show', 'year': '2019', 'season': '1', 'episodes': [2]}
