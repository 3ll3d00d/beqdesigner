'''
Keeps the library CLI documented: every option explains itself in --help and is named in pipeline/README.md, and the
README's example config loads and is understood by the CLI it describes.
'''
import argparse
import os
import re

import pytest
import yaml

from pipeline.library import cli
from pipeline.library.run import LibraryRunReport

README = os.path.join(os.path.dirname(cli.__file__), '..', 'README.md')


def _subparsers():
    parser = cli.build_parser()
    (action,) = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    return action.choices


def _options(subparser):
    return [a for a in subparser._actions if a.option_strings and not isinstance(a, argparse._HelpAction)]


def _readme():
    with open(README, encoding='utf-8') as f:
        return f.read()


@pytest.mark.parametrize('command', ['run', 'sync'])
def test_every_option_has_help_text(command):
    undocumented = [a.option_strings[0] for a in _options(_subparsers()[command]) if not (a.help or '').strip()]

    assert undocumented == []


def test_the_top_level_options_and_commands_are_described():
    parser = cli.build_parser()

    assert parser.description
    assert next(a for a in parser._actions if '--config' in a.option_strings).help
    assert all(a.help for a in next(a for a in parser._actions
                                    if isinstance(a, argparse._SubParsersAction))._choices_actions)


@pytest.mark.parametrize('command', ['run', 'sync'])
def test_every_option_is_named_in_the_readme(command):
    readme = _readme()
    missing = [flag for a in _options(_subparsers()[command]) for flag in a.option_strings
               if flag.startswith('--') and not flag.startswith('--no-') and flag not in readme]

    assert missing == []


def test_each_command_says_how_the_config_file_maps_to_its_flags():
    for command, section in (('run', '`run:`'), ('sync', '`sync:`')):
        text = _subparsers()[command].format_help()
        assert section in text and 'flag overrides the file' in text


def _readme_config():
    match = re.search(r"```yaml\n(.*?)```", _readme(), re.S)
    return yaml.safe_load(match.group(1))


def test_the_readmes_example_config_is_valid_for_the_cli(tmp_path, monkeypatch):
    config = _readme_config()
    path = tmp_path / 'library.yaml'
    path.write_text(yaml.safe_dump(config))
    seen = {}

    class Source:
        def __init__(self, host, port, browse_node_id, **kwargs):
            seen['source'] = (host, port, browse_node_id, kwargs)

    monkeypatch.setattr(cli, 'JRiverLibrarySource', Source)
    monkeypatch.setattr(cli, 'run_library', lambda source, run_config: seen.update(config=run_config)
                        or LibraryRunReport())
    monkeypatch.setattr(cli, 'sync_library', lambda *args, **kwargs: seen.update(sync=(args, kwargs)) or [])
    try:
        assert cli.main(['--config', str(path), 'run']) == 0
        assert cli.main(['--config', str(path), 'sync']) == 0
    finally:
        from pipeline.designer.registry import unregister_designer
        for name in ('rolloff', 'private'):
            unregister_designer(name)

    host, port, node, kwargs = seen['source']
    assert (host, port, node) == ('media.local', 52199, 1007)
    assert [(m.source, m.target) for m in kwargs['path_mappings']] == [('W:\\', '/media/films')]
    assert kwargs['external_id_fields']['movie']['tmdb'] == ['TheMovieDB Movie ID', 'TMDb ID']
    run = seen['config']
    assert (run.designer, run.tv_mode, run.tmdb_api_key) == ('rolloff', 'season', 'XXXX')
    assert run.audio_types == ('DTS-HD MA 5.1',) and run.config.target_fs == 1000
    _, sync = seen['sync']
    assert sync['meta_defaults'] == {'source': 'Disc', 'author': 'me'}
    assert sync['xml_dir'] == 'filters' and sync['image_dir'] == 'images'
