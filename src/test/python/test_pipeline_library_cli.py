'''Tests for the JSON/YAML, Qt-free library command-line entry point.'''
import json

import pytest

from pipeline.library.run import LibraryRunReport


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
                               'external_id_fields': None})
    assert seen['config'].work_dir == '/work'
    assert seen['config'].keep_multichannel is True
    assert seen['config'].audio_types == ('Atmos',)
    assert seen['config'].config.target_fs == 500
    assert json.loads(capsys.readouterr().out) == {
        'cached': [], 'design_cached': [], 'designed': ['one'], 'extracted': ['one'], 'failed': [],
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
