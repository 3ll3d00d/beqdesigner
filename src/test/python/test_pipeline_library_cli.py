'''Tests for the JSON/YAML, Qt-free library command-line entry point.'''
import json
import os

import pytest

from pipeline.designer.registry import register_designer, registered_designers, unregister_designer
from pipeline.library import cli
from pipeline.library.design_cache import DesignCacheResult
from pipeline.library.run import LibraryRunReport
from pipeline.review import QueueEntry

_FAKE_DESIGNERS = ('x', 'old', 'test.designer', 'new')


@pytest.fixture(autouse=True)
def _designers():
    ''' The CLI refuses a designer that is not registered, so the names these tests use must be. '''
    before = set(registered_designers())
    for name in _FAKE_DESIGNERS:
        register_designer(name, lambda request: None)
    yield
    for name in set(registered_designers()) - before:
        unregister_designer(name)


def test_run_uses_yaml_config_and_writes_a_machine_readable_report(tmp_path, monkeypatch, capsys):
    from pipeline.library import cli

    config = tmp_path / 'library.yaml'
    config.write_text('''
sources:
  jriver:
    host: media.local
    port: 52199
    browse_node_id: 42
run:
  source: jriver
  work_dir: /work
  queue_dir: /queue
  designer: test.designer
  keep_multichannel: true
  audio_types: [Atmos]
  analysis:
    target_fs: 500
''')
    seen = {}

    class Source:
        def __init__(self, host, port, browse_node_id, **kwargs):
            seen['source'] = (host, port, browse_node_id, kwargs)

    def run(source, run_config, **_):
        seen['config'] = run_config
        return LibraryRunReport(extracted=['one'], designed=['one'])

    monkeypatch.setattr('pipeline.library.profile.JRiverLibrarySource', Source)
    monkeypatch.setattr(cli, 'run_library', run)

    assert cli.main(['--config', str(config), 'run']) == 0
    assert seen['source'] == ('media.local', 52199, 42,
                              {'username': None, 'password': None, 'ssl': False, 'timeout': 5,
                               'external_id_fields': None, 'path_mappings': []})
    assert seen['config'].work_dir == '/work'
    assert seen['config'].keep_multichannel is True
    assert seen['config'].audio_types == ('Atmos',)
    assert seen['config'].config.target_fs == 500
    assert json.loads(capsys.readouterr().out) == {
        'cached': [], 'design_cached': [], 'designed': ['one'], 'extracted': ['one'], 'failed': [], 'failed_earlier': [], 'meta_unresolved': [], 'project_edit_preserved': [], 'seasons': {},
    }


def test_run_flags_override_json_config_and_failure_exits_nonzero(tmp_path, monkeypatch, capsys):
    from pipeline.library import cli

    config = tmp_path / 'library.json'
    config.write_text(json.dumps({
        'sources': {'jriver': {'host': 'from-config', 'port': 1, 'browse_node_id': 2}},
        'run': {'source': 'jriver', 'work_dir': '/work', 'queue_dir': '/queue', 'designer': 'old'},
    }))
    seen = {}

    class Source:
        def __init__(self, host, port, browse_node_id, **kwargs):
            seen['host'] = host

    monkeypatch.setattr('pipeline.library.profile.JRiverLibrarySource', Source)
    def run(source, run_config, **_):
        seen['designer'] = run_config.designer
        return LibraryRunReport(failed=[('one', 'bad input')])

    monkeypatch.setattr(cli, 'run_library', run)

    assert cli.main(['--config', str(config), 'run', '--host', 'from-flag', '--designer', 'new']) == 1
    assert seen == {'host': 'from-flag', 'designer': 'new'}
    assert json.loads(capsys.readouterr().out)['failed'] == [['one', 'bad input']]


def test_sync_builds_repo_targets_and_returns_publish_errors(monkeypatch, capsys):
    from pipeline.library import cli

    calls = []
    monkeypatch.setattr(cli, 'sync_library', lambda *args, **kwargs: calls.append((args, kwargs))
                        or [{'id': 'one'}, {'id': 'two', 'error': 'project_conflict'}])

    assert cli.main(['sync', '--queue-dir', '/queue', '--xml-repo', '/xml', '--images-repo', '/images',
                     '--work-dir', '/work', '--xml-dir', 'xml']) == 1
    assert calls[0][0][0] == '/queue'
    assert calls[0][0][1].local_path == '/xml'
    assert calls[0][1]['images_repo'].local_path == '/images'
    assert calls[0][1]['work_dir'] == '/work'
    assert json.loads(capsys.readouterr().out)[1]['error'] == 'project_conflict'


def test_unknown_source_is_a_cli_error(tmp_path, monkeypatch):
    from pipeline.library import cli

    with pytest.raises(SystemExit) as error:
        cli.main(['run', '--source', 'plex', '--work-dir', '/work', '--queue-dir', '/queue', '--designer', 'x'])
    assert error.value.code == 2


def test_filesystem_source_takes_globs_from_flags_or_config(tmp_path, monkeypatch, capsys):
    from pipeline.library import cli
    (tmp_path / 'films').mkdir()
    (tmp_path / 'films' / 'a.mkv').write_bytes(b'x')
    seen = []
    monkeypatch.setattr(cli, 'run_library', lambda source, run_config, **_: seen.append(list(source.list_items()))
                        or LibraryRunReport())
    base = ['run', '--source', 'filesystem', '--work-dir', '/work', '--queue-dir', '/queue', '--designer', 'x']

    assert cli.main(base + ['--glob', str(tmp_path / 'films')]) == 0
    config = tmp_path / 'library.json'
    config.write_text(json.dumps({'sources': {'filesystem': {'globs': [str(tmp_path / 'films')]}}}))
    assert cli.main(['--config', str(config)] + base) == 0

    assert [[i.display_name for i in items] for items in seen] == [['a'], ['a']]


def test_filesystem_source_without_a_glob_is_a_cli_error(monkeypatch):
    from pipeline.library import cli

    with pytest.raises(SystemExit):
        cli.main(['run', '--source', 'filesystem', '--work-dir', '/work', '--queue-dir', '/queue',
                  '--designer', 'x'])


def _jriver_paths_seen(monkeypatch, argv, config=None, tmp_path=None):
    from pipeline.library import cli
    from pipeline.library.jriver import JRiverLibrarySource
    seen = []
    monkeypatch.setattr(cli, 'run_library', lambda source, run_config, **_: seen.append(source) or LibraryRunReport())
    base = ['run', '--source', 'jriver', '--host', 'media.local', '--port', '52199', '--browse-node-id', '7',
            '--work-dir', '/work', '--queue-dir', '/queue', '--designer', 'x']
    prefix = []
    if config is not None:
        path = tmp_path / 'library.json'
        path.write_text(json.dumps(config))
        prefix = ['--config', str(path)]
    assert cli.main(prefix + base + argv) == 0
    assert isinstance(seen[0], JRiverLibrarySource)
    return seen[0].path_mappings


def test_path_map_flags_build_server_to_local_mappings(monkeypatch):
    from pipeline.library.pathmap import PathMapping

    mappings = _jriver_paths_seen(monkeypatch, ['--path-map', 'W:\\Films=/mnt/films', '--path-map', 'W:\\TV=/mnt/tv'])

    assert mappings == (PathMapping('W:\\Films', '/mnt/films'), PathMapping('W:\\TV', '/mnt/tv'))


def test_path_mappings_can_come_from_the_config_file_and_flags_replace_them(monkeypatch, tmp_path):
    from pipeline.library.pathmap import PathMapping
    config = {'sources': {'jriver': {'path_mappings': [{'from': 'W:\\Films', 'to': '/mnt/films'}]}}}

    assert _jriver_paths_seen(monkeypatch, [], config, tmp_path) == (PathMapping('W:\\Films', '/mnt/films'),)
    assert _jriver_paths_seen(monkeypatch, ['--path-map', 'X:\\=/mnt/x'], config, tmp_path) \
        == (PathMapping('X:\\', '/mnt/x'),)


def test_a_malformed_path_map_is_a_cli_error(monkeypatch):
    from pipeline.library import cli
    monkeypatch.setattr(cli, 'run_library', lambda *args, **_: LibraryRunReport())

    with pytest.raises(SystemExit):
        cli.main(['run', '--source', 'jriver', '--host', 'h', '--port', '1', '--browse-node-id', '1',
                  '--work-dir', '/w', '--queue-dir', '/q', '--designer', 'x', '--path-map', 'no-equals'])


def test_tv_mode_defaults_to_episode_and_can_be_set_by_flag_or_config(monkeypatch, tmp_path):
    from pipeline.library import cli
    seen = []
    monkeypatch.setattr(cli, 'run_library', lambda source, run_config, **_: seen.append(run_config.tv_mode)
                        or LibraryRunReport())
    base = ['run', '--source', 'filesystem', '--glob', str(tmp_path), '--work-dir', '/w', '--queue-dir', '/q',
            '--designer', 'x']
    config = tmp_path / 'library.json'
    config.write_text(json.dumps({'run': {'tv_mode': 'season'}}))

    cli.main(base)
    cli.main(base + ['--tv-mode', 'season'])
    cli.main(['--config', str(config)] + base)
    cli.main(['--config', str(config)] + base + ['--tv-mode', 'episode'])  # a flag overrides the file

    assert seen == ['episode', 'season', 'season', 'episode']


def test_an_unknown_tv_mode_flag_is_a_cli_error(tmp_path):
    from pipeline.library import cli

    with pytest.raises(SystemExit):
        cli.main(['run', '--source', 'filesystem', '--glob', str(tmp_path), '--work-dir', '/w', '--queue-dir', '/q',
                  '--designer', 'x', '--tv-mode', 'series'])


# --- registering designers (the CLI has no GUI preferences to do it) ---------------------------------------------

def _run_args(designer, *extra, tmp_path=None):
    return ['run', '--source', 'filesystem', '--glob', '/films', '--work-dir', '/w', '--queue-dir', '/q',
            '--designer', designer, *extra]


def test_a_designer_declared_in_the_config_file_is_registered_before_the_run(tmp_path, monkeypatch):
    from pipeline.library import cli
    seen = []
    monkeypatch.setattr(cli, 'run_library', lambda source, run_config, **_: seen.append(registered_designers())
                        or LibraryRunReport())
    config = tmp_path / 'library.json'
    config.write_text(json.dumps({'designers': {
        'rolloff': 'http://designer.local:8080/design',
        'bearer': {'url': 'http://other.local/design', 'timeout': 30, 'headers': {'Authorization': 'Bearer t'}},
    }}))

    assert cli.main(['--config', str(config)] + _run_args('rolloff')) == 0

    assert 'rolloff' in seen[0] and 'bearer' in seen[0]


def test_a_declared_designer_is_built_with_its_url_timeout_and_headers(tmp_path, monkeypatch):
    from pipeline.library import cli
    from pipeline.designer.registry import get_designer
    made = []
    monkeypatch.setattr(cli, 'http_designer', lambda url, timeout, headers: made.append((url, timeout, headers)) or (
        lambda request: None))
    monkeypatch.setattr(cli, 'run_library', lambda source, run_config, **_: LibraryRunReport())
    config = tmp_path / 'library.json'
    config.write_text(json.dumps({'designers': {'a': 'http://a/d', 'b': {'url': 'http://b/d', 'timeout': 30,
                                                                         'headers': {'X': 'y'}}}}))

    cli.main(['--config', str(config)] + _run_args('a'))

    assert made == [('http://a/d', 300.0, None), ('http://b/d', 30.0, {'X': 'y'})]
    assert get_designer('a') is not None


def test_designer_url_flags_register_a_designer_and_win_over_the_file(tmp_path, monkeypatch):
    from pipeline.library import cli
    made = []
    monkeypatch.setattr(cli, 'http_designer', lambda url, timeout, headers: made.append(url) or (lambda r: None))
    monkeypatch.setattr(cli, 'run_library', lambda source, run_config, **_: LibraryRunReport())
    config = tmp_path / 'library.json'
    config.write_text(json.dumps({'designers': {'mine': 'http://from-file/d'}}))

    assert cli.main(['--config', str(config)] + _run_args('mine', '--designer-url', 'mine=http://from-flag/d')) == 0
    assert cli.main(_run_args('other', '--designer-url', 'other=http://other/d')) == 0

    assert made == ['http://from-flag/d', 'http://other/d']


def test_a_url_given_as_the_designer_is_registered_under_that_url(monkeypatch):
    from pipeline.library import cli
    seen = []
    monkeypatch.setattr(cli, 'run_library', lambda source, run_config, **_: seen.append(run_config.designer)
                        or LibraryRunReport())

    assert cli.main(_run_args('http://designer.local/design')) == 0

    assert seen == ['http://designer.local/design']
    assert 'http://designer.local/design' in registered_designers()


def test_an_unregistered_designer_stops_the_run_with_a_clear_message(monkeypatch, capsys):
    from pipeline.library import cli
    monkeypatch.setattr(cli, 'run_library', lambda *args: pytest.fail('must not run'))

    with pytest.raises(SystemExit):
        cli.main(_run_args('nobody'))

    error = capsys.readouterr().err
    assert "designer 'nobody' is not registered" in error and '--designer-url nobody=URL' in error


@pytest.mark.parametrize('bad_flag', ['no-equals', '=http://x', 'name='])
def test_a_malformed_designer_url_flag_is_a_cli_error(bad_flag, monkeypatch):
    from pipeline.library import cli
    monkeypatch.setattr(cli, 'run_library', lambda *args: pytest.fail('must not run'))

    with pytest.raises(SystemExit):
        cli.main(_run_args('x', '--designer-url', bad_flag))


def test_a_declared_designer_without_a_url_is_a_cli_error(tmp_path, monkeypatch):
    from pipeline.library import cli
    monkeypatch.setattr(cli, 'run_library', lambda *args: pytest.fail('must not run'))
    config = tmp_path / 'library.json'
    config.write_text(json.dumps({'designers': {'broken': {'timeout': 5}}}))

    with pytest.raises(SystemExit):
        cli.main(['--config', str(config)] + _run_args('broken'))


def test_publish_only_writes_and_reads_the_shared_sync_section(tmp_path, monkeypatch, capsys):
    from pipeline.library import cli
    config = tmp_path / 'c.json'
    config.write_text(json.dumps({'sync': {'queue_dir': '/queue', 'xml_repo': '/xml', 'xml_dir': 'filters'}}))
    calls = []
    monkeypatch.setattr(cli, 'publish_library', lambda *args, **kwargs: calls.append((args, kwargs)) or [{'id': 'one'}])
    monkeypatch.setattr(cli, 'commit_library', lambda *args, **kwargs: pytest.fail('publish must not commit'))
    monkeypatch.setattr(cli, 'sync_library', lambda *args, **kwargs: pytest.fail('publish is not sync'))

    assert cli.main(['--config', str(config), 'publish', '--images-repo', '/images']) == 0

    (args, kwargs), = calls
    assert (args[0], args[1].local_path, kwargs['images_repo'].local_path, kwargs['xml_dir']) == \
        ('/queue', '/xml', '/images', 'filters')
    assert json.loads(capsys.readouterr().out) == [{'id': 'one'}]


def test_commit_pushes_by_default_and_can_be_told_not_to(monkeypatch, capsys):
    from pipeline.library import cli
    from pipeline.library.commit import CatalogueCommit, RepoCommit
    calls = []
    monkeypatch.setattr(cli, 'commit_library', lambda *args, **kwargs: calls.append((args, kwargs)) or CatalogueCommit(
        xml=RepoCommit('/xml', ['a.xml'], 'abc', True), images=RepoCommit('/images', ['a.png'], 'def', True)))

    assert cli.main(['commit', '--queue-dir', '/queue', '--xml-repo', '/xml', '--images-repo', '/images']) == 0
    assert cli.main(['commit', '--queue-dir', '/queue', '--xml-repo', '/xml', '--no-push']) == 0

    assert [kwargs['push'] for _, kwargs in calls] == [True, False]
    assert calls[0][0][1].local_path == '/xml' and calls[0][1]['images_repo'].local_path == '/images'
    assert calls[1][1]['images_repo'] is None
    first = json.loads(capsys.readouterr().out.splitlines()[0])
    assert first['xml'] == {'repo': '/xml', 'paths': ['a.xml'], 'commit': 'abc', 'pushed': True}
    assert first['missing'] == []


def test_commit_exits_nonzero_when_a_published_entry_has_no_file(monkeypatch, capsys):
    from pipeline.library import cli
    from pipeline.library.commit import CatalogueCommit, RepoCommit
    monkeypatch.setattr(cli, 'commit_library', lambda *args, **kwargs: CatalogueCommit(
        xml=RepoCommit('/xml', []), missing=['gone.xml']))

    assert cli.main(['commit', '--queue-dir', '/queue', '--xml-repo', '/xml']) == 1


def test_sync_passes_push_through(monkeypatch, capsys):
    from pipeline.library import cli
    calls = []
    monkeypatch.setattr(cli, 'sync_library', lambda *args, **kwargs: calls.append(kwargs) or [])

    cli.main(['sync', '--queue-dir', '/queue', '--xml-repo', '/xml'])
    cli.main(['sync', '--queue-dir', '/queue', '--xml-repo', '/xml', '--no-push'])

    assert [kwargs['push'] for kwargs in calls] == [True, False]


def test_commit_without_a_repository_is_a_cli_error(capsys):
    from pipeline.library import cli
    with pytest.raises(SystemExit) as raised:
        cli.main(['commit', '--queue-dir', '/queue'])

    assert raised.value.code == 2
    assert 'xml-repo is required' in capsys.readouterr().err


def test_revise_sends_each_id_back_and_reports_the_ones_it_could_not(tmp_path, capsys):
    from pipeline.library import cli
    from pipeline.review import CandidateSummary, QueueEntry, read_entry, write_queue_entry
    queue_dir = str(tmp_path / 'queue')
    candidate = CandidateSummary(filters={}, confidence=0.9, method='fitted', mv_adjust_db=1.0, gain_reduction_db=0.0,
                                 commentary={})
    for entry_id in ('one', 'two'):
        write_queue_entry(queue_dir, QueueEntry(id=entry_id, fs=1000, meta={}, curve={}, candidates=[candidate],
                                                status='accepted', chosen_candidate_index=0))

    code = cli.main(['revise', '--queue-dir', queue_dir, '--to', 'review', '--reason', 'check the poster',
                     '--id', 'one', '--id', 'missing', '--id', 'two'])

    out = json.loads(capsys.readouterr().out)
    assert code == 1
    assert [(r['id'], r.get('status')) for r in out] == [('one', 'pending'), ('missing', None), ('two', 'pending')]
    assert 'missing' in out[1]['error']
    assert read_entry(queue_dir, 'two').reviewer_note == 'Reopened for review: check the poster'


def test_revise_reads_the_shared_sync_section_and_needs_an_id_and_a_target(tmp_path, capsys):
    from pipeline.library import cli
    config = tmp_path / 'c.json'
    config.write_text(json.dumps({'sync': {'queue_dir': str(tmp_path / 'queue'), 'work_dir': str(tmp_path / 'work')}}))

    for argv, message in ((['revise', '--to', 'review'], 'ids is required'), (['revise', '--id', 'one'], 'to is required')):
        with pytest.raises(SystemExit) as raised:
            cli.main(['--config', str(config), *argv])
        assert raised.value.code == 2
        assert message in capsys.readouterr().err


def test_revise_to_extract_without_a_work_dir_is_reported_per_id(tmp_path, capsys):
    from pipeline.library import cli
    from pipeline.review import QueueEntry, write_queue_entry
    queue_dir = str(tmp_path / 'queue')
    write_queue_entry(queue_dir, QueueEntry(id='one', fs=1000, meta={}, curve={}))

    assert cli.main(['revise', '--queue-dir', queue_dir, '--to', 'extract', '--id', 'one']) == 1

    assert 'work_dir is required' in json.loads(capsys.readouterr().out)[0]['error']


def _profile_file(tmp_path, **extra):
    import yaml
    config = {
        'sources': [{'name': 'films', 'kind': 'filesystem', 'globs': [str(tmp_path / 'films')]},
                    {'name': 'more', 'kind': 'filesystem', 'globs': [str(tmp_path / 'more')]}],
        'ignore': [{'kind': 'tv'}],
        'run': {'work_dir': str(tmp_path / 'work'), 'designer': 'x'},
        'sync': {'queue_dir': str(tmp_path / 'queue')},  # the profile may keep the queue under `sync:`
        **extra,
    }
    path = tmp_path / 'catalogue.yaml'
    path.write_text(yaml.safe_dump(config))
    return str(path)


def test_run_with_a_profile_runs_the_union_of_its_sources(tmp_path, monkeypatch, capsys):
    from pipeline.library import cli
    from pipeline.library.union import UnionLibrarySource
    seen = {}
    monkeypatch.setattr(cli, 'run_library', lambda source, run_config, **_: seen.update(source=source, config=run_config)
                        or LibraryRunReport())

    assert cli.main(['run', '--profile', _profile_file(tmp_path)]) == 0

    assert isinstance(seen['source'], UnionLibrarySource)
    assert [s.name for s in seen['source'].profile.sources] == ['films', 'more']
    assert [r.kind for r in seen['source'].profile.ignore] == ['tv']
    assert (seen['config'].work_dir, seen['config'].queue_dir) == (str(tmp_path / 'work'), str(tmp_path / 'queue'))
    assert seen['config'].designer == 'x'


def test_run_flags_still_override_a_profile(tmp_path, monkeypatch):
    from pipeline.library import cli
    seen = {}
    monkeypatch.setattr(cli, 'run_library', lambda source, run_config, **_: seen.update(config=run_config)
                        or LibraryRunReport())

    cli.main(['run', '--profile', _profile_file(tmp_path), '--designer', 'old', '--tv-mode', 'season'])

    assert (seen['config'].designer, seen['config'].tv_mode) == ('old', 'season')


def test_a_profile_with_no_sources_is_refused_and_so_is_combining_it_with_config(tmp_path, capsys):
    import yaml
    from pipeline.library import cli
    empty = tmp_path / 'empty.yaml'
    empty.write_text(yaml.safe_dump({'run': {'work_dir': '/w', 'queue_dir': '/q', 'designer': 'x'}}))

    for argv, message in ((['run', '--profile', str(empty)], 'lists no sources'),
                          (['--config', str(empty), 'run', '--profile', str(empty)], 'replaces --config')):
        with pytest.raises(SystemExit) as raised:
            cli.main(argv)
        assert raised.value.code == 2 and message in capsys.readouterr().err


def test_a_malformed_ignore_rule_in_a_profile_is_a_cli_error_not_a_silent_no_op(tmp_path, capsys):
    from pipeline.library import cli
    path = _profile_file(tmp_path, ignore=[{'colour': 'red'}])

    with pytest.raises(SystemExit) as raised:
        cli.main(['run', '--profile', path])

    assert raised.value.code == 2 and 'unknown ignore rule key' in capsys.readouterr().err


def test_the_old_config_still_runs_through_the_single_source_path(tmp_path, monkeypatch):
    from pipeline.library import cli
    from pipeline.library.filesystem import FilesystemLibrarySource
    seen = {}
    monkeypatch.setattr(cli, 'run_library', lambda source, run_config, **_: seen.update(source=source)
                        or LibraryRunReport())

    cli.main(['run', '--source', 'filesystem', '--glob', str(tmp_path), '--work-dir', '/w', '--queue-dir', '/q',
              '--designer', 'x'])

    assert isinstance(seen['source'], FilesystemLibrarySource)


# --- scan and status (discovery) ---------------------------------------------------------------------------------

def _discovery_config(tmp_path, films=2):
    import yaml
    media = tmp_path / 'films'
    media.mkdir()
    for n in range(films):
        (media / f'film-{n}.mkv').write_bytes(b'x' * (n + 1))
    config = tmp_path / 'library.yaml'
    config.write_text(yaml.safe_dump({
        'sources': [{'name': 'disk', 'kind': 'filesystem', 'globs': [str(media)]}],
        'run': {'work_dir': str(tmp_path / 'work'), 'queue_dir': str(tmp_path / 'queue'), 'designer': 'rolloff'}}))
    return config


def test_scan_writes_the_index_and_status_counts_what_it_found(tmp_path, capsys):
    config = _discovery_config(tmp_path, films=3)

    assert cli.main(['--config', str(config), 'scan']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['titles'] == 3 and result['counts']['extract'] == 3 and result['errors'] == {}
    assert (tmp_path / 'work' / 'library-index.sqlite').is_file()

    assert cli.main(['--config', str(config), 'status', '--json']) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary['counts']['extract'] == 3 and summary['titles'] == 3 and summary['new'] == 3
    assert [s['name'] for s in summary['sources']] == ['disk']

    assert cli.main(['--config', str(config), 'status']) == 0
    text = capsys.readouterr().out
    assert '3 titles' in text and 'extract' in text and 'source disk: ok, 3 items' in text


def test_scan_takes_a_profile_and_flags_that_override_it(tmp_path, capsys):
    config = _discovery_config(tmp_path)
    other_work = tmp_path / 'elsewhere'

    assert cli.main(['scan', '--profile', str(config), '--work-dir', str(other_work)]) == 0
    capsys.readouterr()

    assert (other_work / 'library-index.sqlite').is_file() and not (tmp_path / 'work').exists()
    assert cli.main(['status', '--profile', str(config), '--work-dir', str(other_work), '--json']) == 0


def test_status_before_any_scan_says_so_and_exits_nonzero(tmp_path, capsys):
    config = _discovery_config(tmp_path)

    assert cli.main(['--config', str(config), 'status']) == 1

    assert 'run `scan` first' in capsys.readouterr().out
    assert not (tmp_path / 'work').exists()  # status never creates the index


def test_scan_reports_a_source_that_cannot_be_listed_and_exits_nonzero(tmp_path, capsys, monkeypatch):
    config = _discovery_config(tmp_path)
    assert cli.main(['--config', str(config), 'scan']) == 0
    capsys.readouterr()

    def down(self, **query):
        raise ConnectionError('server down')

    monkeypatch.setattr('pipeline.library.filesystem.FilesystemLibrarySource.list_items', down)
    assert cli.main(['--config', str(config), 'scan']) == 1

    result = json.loads(capsys.readouterr().out)
    assert 'server down' in result['errors']['disk'] and result['titles'] == 2  # the titles are still there
    assert cli.main(['--config', str(config), 'status']) == 0
    assert 'FAILED' in capsys.readouterr().out


def test_scan_of_an_unknown_source_name_is_a_cli_error(tmp_path):
    config = _discovery_config(tmp_path)

    with pytest.raises(SystemExit) as exit_info:
        cli.main(['--config', str(config), 'scan', '--source', 'nope'])

    assert exit_info.value.code == 2


def test_scan_needs_a_profile_with_sources_and_a_work_dir(tmp_path):
    empty = tmp_path / 'empty.json'
    empty.write_text('{"run": {"work_dir": "/w"}}')
    no_dir = tmp_path / 'nodir.json'
    no_dir.write_text('{"sources": [{"name": "d", "kind": "filesystem", "globs": ["/x"]}]}')

    for config in (empty, no_dir):
        with pytest.raises(SystemExit) as exit_info:
            cli.main(['--config', str(config), 'scan'])
        assert exit_info.value.code == 2


def test_scan_from_outputs_rebuilds_without_listing(tmp_path, capsys, monkeypatch):
    config = _discovery_config(tmp_path)
    monkeypatch.setattr('pipeline.library.filesystem.FilesystemLibrarySource.list_items',
                        lambda self, **q: pytest.fail('a rebuild must not list the sources'))

    assert cli.main(['--config', str(config), 'scan', '--from-outputs']) == 0

    assert json.loads(capsys.readouterr().out) == {'rebuilt': 0}


def test_run_leaves_a_failure_for_status_to_report(tmp_path, capsys, monkeypatch):
    from pipeline.designer.registry import register_designer, unregister_designer
    config = _discovery_config(tmp_path, films=1)
    register_designer('rolloff', lambda request: None)
    monkeypatch.setattr('pipeline.library.run.extract_if_needed',
                        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError('no such file')))
    try:
        assert cli.main(['run', '--profile', str(config)]) == 1
    finally:
        unregister_designer('rolloff')
    capsys.readouterr()

    assert cli.main(['--config', str(config), 'scan']) == 0
    assert json.loads(capsys.readouterr().out)['counts']['attention'] == 1


# --- selectors, --through, --retry-failed, publish --republish, accept (chunk 25) -------------------------------------

def test_run_without_a_selector_still_lists_and_runs_everything(tmp_path, monkeypatch, capsys):
    config = _discovery_config(tmp_path)
    seen = {}
    monkeypatch.setattr(cli, 'run_library', lambda source, run_config, **kw: seen.update(kw) or LibraryRunReport())
    monkeypatch.setattr(cli, 'run_stages', lambda *a, **k: pytest.fail('no selector: the old path'))

    assert cli.main(['run', '--profile', str(config), '--designer', 'test.designer']) == 0

    assert seen['retry_failed'] is False


def test_retry_failed_reaches_the_unselected_run_too(tmp_path, monkeypatch, capsys):
    config = _discovery_config(tmp_path)
    seen = {}
    monkeypatch.setattr(cli, 'run_library', lambda source, run_config, **kw: seen.update(kw) or LibraryRunReport())

    cli.main(['run', '--profile', str(config), '--designer', 'test.designer', '--retry-failed'])

    assert seen['retry_failed'] is True


@pytest.mark.parametrize('flags, expected', [
    (['--needs', 'extract', '--needs', 'design'], dict(needs=('extract', 'design'))),
    (['--match', 'film-1'], dict(match='film-1')),
    (['--id', 'x', '--id', 'y'], dict(ids=('x', 'y'))),
    (['--new-since-scan'], dict(new_since_scan=True)),
    (['--source', 'disk'], dict(source='disk')),
    (['--through', 'extract'], {}),
])
def test_a_selector_hands_the_index_and_a_selection_to_run_stages(tmp_path, monkeypatch, capsys, flags, expected):
    from pipeline.library.selection import Selection
    from pipeline.library.stages import StagesReport
    config = _discovery_config(tmp_path)
    seen = {}

    def run_stages(profile, selection, through, **kwargs):
        seen.update(profile=profile, selection=selection, through=through, generation=kwargs['index'].generation,
                    retry=kwargs['retry_failed'], publish=kwargs['publish'])
        return StagesReport(through, 0)

    monkeypatch.setattr(cli, 'run_stages', run_stages)

    assert cli.main(['run', '--profile', str(config), '--designer', 'test.designer', *flags]) == 0

    assert seen['selection'] == Selection(**expected)
    assert seen['through'] == (flags[1] if flags[0] == '--through' else 'design')
    assert seen['generation'] == 1  # nothing had been scanned, so it scanned first
    assert [s.name for s in seen['profile'].sources] == ['disk'] and seen['publish'] is None and seen['retry'] is False
    assert json.loads(capsys.readouterr().out)['through'] == seen['through']


def test_through_publish_needs_a_repository(tmp_path, monkeypatch):
    config = _discovery_config(tmp_path)
    monkeypatch.setattr(cli, 'run_stages', lambda *a, **k: pytest.fail('refused before running'))

    with pytest.raises(SystemExit) as raised:
        cli.main(['run', '--profile', str(config), '--designer', 'test.designer', '--through', 'publish'])

    assert raised.value.code == 2


def test_through_publish_takes_the_repositories_from_the_sync_section_and_flags(tmp_path, monkeypatch, capsys):
    import yaml
    from pipeline.library.stages import StagesReport
    config = _discovery_config(tmp_path)
    data = yaml.safe_load(config.read_text())
    data['sync'] = {'xml_repo': '/xml', 'xml_dir': 'filters', 'meta_defaults': {'source': 'Disc'}}
    config.write_text(yaml.safe_dump(data))
    seen = {}
    monkeypatch.setattr(cli, 'run_stages', lambda profile, selection, through, **kw: seen.update(kw) or StagesReport(through, 0))

    cli.main(['run', '--profile', str(config), '--designer', 'test.designer', '--through', 'commit',
              '--images-repo', '/images', '--no-push', '--image-owner', 'me'])

    publish = seen['publish']
    assert (publish.xml_repo.local_path, publish.images_repo.local_path, publish.xml_dir, publish.image_owner,
            publish.push, publish.meta_defaults) == ('/xml', '/images', 'filters', 'me', False, {'source': 'Disc'})
    assert seen['settings'].meta_defaults == {'source': 'Disc'}  # the index is refreshed with what publish is given


def test_a_selector_run_without_a_profile_uses_the_one_source_the_flags_describe(tmp_path, monkeypatch, capsys):
    from pipeline.library.stages import StagesReport
    seen = {}
    monkeypatch.setattr(cli, 'run_stages', lambda profile, selection, through, **kw: seen.update(
        profile=profile, index=kw['index']) or StagesReport(through, 0))
    (tmp_path / 'films').mkdir()

    cli.main(['run', '--source', 'filesystem', '--glob', str(tmp_path / 'films'), '--work-dir', str(tmp_path / 'w'),
              '--queue-dir', str(tmp_path / 'q'), '--designer', 'test.designer', '--needs', 'extract'])

    (spec,) = seen['profile'].sources
    assert (spec.name, spec.kind, spec.settings['globs']) == ('filesystem', 'filesystem', [str(tmp_path / 'films')])


def test_publish_and_commit_pass_id_and_republish_through(monkeypatch, capsys):
    published, committed = [], []
    monkeypatch.setattr(cli, 'publish_library', lambda *a, **k: published.append(k) or [])
    monkeypatch.setattr(cli, 'sync_library', lambda *a, **k: published.append(k) or [])
    monkeypatch.setattr(cli, 'commit_library', lambda *a, **k: committed.append(k) or __import__(
        'pipeline.library.commit', fromlist=['x']).CatalogueCommit(xml=__import__(
            'pipeline.library.commit', fromlist=['x']).RepoCommit('/xml', [])))

    cli.main(['publish', '--queue-dir', '/q', '--xml-repo', '/xml', '--id', 'a', '--id', 'b', '--republish'])
    cli.main(['sync', '--queue-dir', '/q', '--xml-repo', '/xml', '--republish'])
    cli.main(['publish', '--queue-dir', '/q', '--xml-repo', '/xml'])
    cli.main(['commit', '--queue-dir', '/q', '--xml-repo', '/xml', '--id', 'a'])

    assert [(k['ids'], k['republish']) for k in published] == [(['a', 'b'], True), (None, True), (None, False)]
    assert committed[0]['ids'] == ['a']


def test_the_options_are_documented_by_the_help(capsys):
    with pytest.raises(SystemExit):
        cli.main(['accept', '-h'])
    text = capsys.readouterr().out
    assert '--threshold' in text and 'bulk accepted' in text and '--dry-run' in text


# --- the whole workflow, headless (milestone M2) ----------------------------------------------------------------------

@pytest.fixture
def workflow(tmp_path, monkeypatch):
    ''' A profile over three films, temp git repos, and fake extraction and design that leave real outputs. '''
    import subprocess
    from dataclasses import replace
    from types import SimpleNamespace

    import numpy as np
    import soundfile as sf
    import yaml
    from pipeline.library.extract_cache import source_fingerprint
    from test_pipeline_library_commit import _repo
    from test_pipeline_library_index import _entry, _extracted

    config = _discovery_config(tmp_path, films=3)
    xml, xml_bare = _repo(tmp_path, 'xml')
    images, images_bare = _repo(tmp_path, 'images')
    for repo in (xml, images):
        (tmp_path / repo.local_path.rsplit('/', 1)[1] / 'README').write_text('catalogue')
        subprocess.run(['git', '-C', repo.local_path, 'add', 'README'], check=True, capture_output=True)
        subprocess.run(['git', '-C', repo.local_path, 'commit', '-q', '-m', 'first'], check=True, capture_output=True)
        subprocess.run(['git', '-C', repo.local_path, 'push', '-q', '-u', 'origin', 'HEAD'], check=True,
                       capture_output=True)
    data = yaml.safe_load(config.read_text())
    data['run']['designer'] = 'test.designer'
    data['sync'] = {'xml_repo': xml.local_path, 'xml_dir': 'xml', 'images_repo': images.local_path, 'image_dir': 'img',
                    'image_owner': 'me', 'image_repo_name': 'images'}
    config.write_text(yaml.safe_dump(data))
    env = SimpleNamespace(work=str(tmp_path / 'work'), queue=str(tmp_path / 'queue'))
    confidence = {'film-0': 0.95, 'film-1': 0.6, 'film-2': 0.99}

    def extract(session, item, item_dir, cfg, mono_mix=True, force=False):
        _extracted(env, item, fingerprint=source_fingerprint(item))
        sf.write(os.path.join(item_dir, 'mono.wav'), np.random.default_rng(1).normal(0, 0.1, 4000), 1000)
        return os.path.join(item_dir, 'mono.wav'), False

    def design(session, item, wav_path, designer, queue_dir, cfg, **kwargs):
        name = os.path.basename(item.source_path)[:-4]
        _entry(env, replace(item, title=name), fingerprint=source_fingerprint(item), confidence=confidence[name])
        return DesignCacheResult(QueueEntry(id=item.id, fs=1000, meta={}, curve={}), designed=True)

    monkeypatch.setattr('pipeline.library.run.extract_if_needed', extract)
    monkeypatch.setattr('pipeline.library.run.design_if_needed', design)
    return SimpleNamespace(config=config, tmp=tmp_path, xml=xml, xml_bare=xml_bare, images=images, queue=env.queue,
                           work=env.work)


def _cli(capsys, *argv):
    code = cli.main(list(argv))
    out = capsys.readouterr().out
    try:
        return code, json.loads(out)
    except ValueError:
        return code, out


def test_the_whole_workflow_runs_headless_scan_design_accept_publish_commit(workflow, capsys):
    from pipeline.review import read_queue
    profile = ['--profile', str(workflow.config)]

    code, scanned = _cli(capsys, 'scan', *profile)
    assert code == 0 and scanned['counts']['extract'] == 3

    code, run = _cli(capsys, 'run', *profile, '--through', 'design')  # what cron does: machine work only
    assert code == 0, run['run']['failed']
    assert len(run['run']['designed']) == 3 and run['counts']['review'] == 3
    assert all(e.status == 'pending' for e in read_queue(workflow.queue))  # it never accepts
    assert not os.path.isdir(os.path.join(workflow.xml.local_path, 'xml'))  # ... nor publishes

    code, again = _cli(capsys, 'run', *profile, '--through', 'design')  # and is idempotent
    assert again['run']['designed'] == [] and again['attempted'] == [] and len(again['skipped']) == 3

    code, dry = _cli(capsys, 'accept', *profile, '--dry-run')
    assert len(dry['eligible']) == 2 and dry['below_threshold'] == 1
    assert {e.status for e in read_queue(workflow.queue)} == {'pending'}

    code, accepted = _cli(capsys, 'accept', *profile)
    assert code == 0 and sorted(accepted['accepted']) == sorted(dry['eligible'])
    assert accepted['note'] == 'bulk accepted, confidence >= 0.90'

    code, status = _cli(capsys, 'status', *profile, '--json')
    assert (status['counts']['publish'], status['counts']['review']) == (2, 1)

    code, published = _cli(capsys, 'run', *profile, '--needs', 'publish', '--through', 'publish')
    assert code == 0 and len(published['published']) == 2 and published['counts']['commit'] == 2

    code, committed = _cli(capsys, 'run', *profile, '--needs', 'commit', '--through', 'commit')
    assert code == 0 and len(committed['committed']['xml']['paths']) == 2 and committed['counts']['done'] == 2
    code, final = _cli(capsys, 'status', *profile, '--json')
    assert final['counts']['done'] == 2 and final['counts']['review'] == 1  # the unconfident title still waits


def test_a_failed_title_is_reported_once_then_skipped_until_retry_failed(workflow, capsys, monkeypatch):
    profile = ['--profile', str(workflow.config)]

    def broken(session, item, *args, **kwargs):
        raise FileNotFoundError('no such file (path mapping?)')

    monkeypatch.setattr('pipeline.library.run.extract_if_needed', broken)

    code, first = _cli(capsys, 'run', *profile, '--through', 'design')
    assert code == 1 and len(first['run']['failed']) == 3 and first['counts']['attention'] == 3

    code, second = _cli(capsys, 'run', *profile, '--through', 'design')  # the nightly job does not fail again
    assert code == 0 and second['run']['failed'] == [] and 'retry failed' in second['skipped'][0]['reason']

    code, third = _cli(capsys, 'run', *profile, '--through', 'design', '--retry-failed')
    assert code == 1 and len(third['run']['failed']) == 3
