'''Tests for the JSON/YAML, Qt-free library command-line entry point.'''
import json

import pytest

from pipeline.designer.registry import register_designer, registered_designers, unregister_designer
from pipeline.library.run import LibraryRunReport

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

    def run(source, run_config):
        seen['config'] = run_config
        return LibraryRunReport(extracted=['one'], designed=['one'])

    monkeypatch.setattr(cli, 'JRiverLibrarySource', Source)
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
        'cached': [], 'design_cached': [], 'designed': ['one'], 'extracted': ['one'], 'failed': [], 'meta_unresolved': [], 'project_edit_preserved': [], 'seasons': {},
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

    monkeypatch.setattr(cli, 'JRiverLibrarySource', Source)
    def run(source, run_config):
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
    monkeypatch.setattr(cli, 'run_library', lambda source, run_config: seen.append(list(source.list_items()))
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
    monkeypatch.setattr(cli, 'run_library', lambda source, run_config: seen.append(source) or LibraryRunReport())
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
    monkeypatch.setattr(cli, 'run_library', lambda *args: LibraryRunReport())

    with pytest.raises(SystemExit):
        cli.main(['run', '--source', 'jriver', '--host', 'h', '--port', '1', '--browse-node-id', '1',
                  '--work-dir', '/w', '--queue-dir', '/q', '--designer', 'x', '--path-map', 'no-equals'])


def test_tv_mode_defaults_to_episode_and_can_be_set_by_flag_or_config(monkeypatch, tmp_path):
    from pipeline.library import cli
    seen = []
    monkeypatch.setattr(cli, 'run_library', lambda source, run_config: seen.append(run_config.tv_mode)
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
    monkeypatch.setattr(cli, 'run_library', lambda source, run_config: seen.append(registered_designers())
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
    monkeypatch.setattr(cli, 'run_library', lambda source, run_config: LibraryRunReport())
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
    monkeypatch.setattr(cli, 'run_library', lambda source, run_config: LibraryRunReport())
    config = tmp_path / 'library.json'
    config.write_text(json.dumps({'designers': {'mine': 'http://from-file/d'}}))

    assert cli.main(['--config', str(config)] + _run_args('mine', '--designer-url', 'mine=http://from-flag/d')) == 0
    assert cli.main(_run_args('other', '--designer-url', 'other=http://other/d')) == 0

    assert made == ['http://from-flag/d', 'http://other/d']


def test_a_url_given_as_the_designer_is_registered_under_that_url(monkeypatch):
    from pipeline.library import cli
    seen = []
    monkeypatch.setattr(cli, 'run_library', lambda source, run_config: seen.append(run_config.designer)
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
