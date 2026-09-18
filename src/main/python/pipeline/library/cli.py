'''Command-line entry point for library runs and explicit publishing.'''
import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from pipeline.config import AnalysisConfig
from pipeline.library.jriver import JRiverLibrarySource
from pipeline.library.run import LibraryRunConfig, run_library
from pipeline.library.sync import sync_library
from pipeline.publish.git import RepoTarget


def _load_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    content = Path(path).read_text(encoding='utf-8')
    loaded = yaml.safe_load(content) if Path(path).suffix.lower() in {'.yaml', '.yml'} else json.loads(content)
    if not isinstance(loaded, dict):
        raise ValueError('configuration root must be an object')
    return loaded


def _configured_values(args: argparse.Namespace, config: dict[str, Any], section: str) -> dict[str, Any]:
    values = dict(config.get(section, {}))
    values.update({key: value for key, value in vars(args).items()
                   if key not in {'command', 'config'} and value is not None})
    return values


def _required(values: dict[str, Any], name: str) -> Any:
    value = values.get(name)
    if value in (None, ''):
        raise ValueError(f'{name.replace("_", "-")} is required')
    return value


def _source(values: dict[str, Any], config: dict[str, Any]):
    source_name = _required(values, 'source')
    source_values = dict(config.get('sources', {}).get(source_name, {}))
    source_values.update({key: values[key] for key in
                          ('host', 'port', 'browse_node_id', 'username', 'password', 'ssl', 'timeout',
                           'external_id_fields') if values.get(key) is not None})
    if source_name != 'jriver':
        raise ValueError(f'unsupported source {source_name!r}; only jriver is currently available')
    return JRiverLibrarySource(
        _required(source_values, 'host'), int(_required(source_values, 'port')),
        int(_required(source_values, 'browse_node_id')), username=source_values.get('username'),
        password=source_values.get('password'), ssl=bool(source_values.get('ssl', False)),
        timeout=int(source_values.get('timeout', 5)),
        external_id_fields=source_values.get('external_id_fields'),
    )


def _analysis_config(values: dict[str, Any]) -> AnalysisConfig:
    configured = values.get('analysis', {})
    fields = {name: values.get(name, configured.get(name))
              for name in ('target_fs', 'resolution', 'avg_window', 'peak_window')}
    return AnalysisConfig(**{name: value for name, value in fields.items() if value is not None})


def _run(args: argparse.Namespace, config: dict[str, Any]) -> int:
    values = _configured_values(args, config, 'run')
    run_config = LibraryRunConfig(
        work_dir=_required(values, 'work_dir'), queue_dir=_required(values, 'queue_dir'),
        designer=_required(values, 'designer'), config=_analysis_config(values),
        coverage=values.get('coverage', 'complete_programme'),
        keep_multichannel=bool(values.get('keep_multichannel', False)),
        force_extract=bool(values.get('force_extract', False)),
        force_design=bool(values.get('force_design', False)),
        tmdb_api_key=values.get('tmdb_api_key'),
        audio_types=tuple(values.get('audio_types', ())),
    )
    report = run_library(_source(values, config), run_config)
    print(json.dumps(asdict(report), sort_keys=True))
    return 1 if report.failed else 0


def _sync(args: argparse.Namespace, config: dict[str, Any]) -> int:
    values = _configured_values(args, config, 'sync')
    result = sync_library(
        _required(values, 'queue_dir'), RepoTarget(_required(values, 'xml_repo')),
        meta_defaults=values.get('meta_defaults'), images_repo=RepoTarget(values['images_repo'])
        if values.get('images_repo') else None, image_owner=values.get('image_owner'),
        image_repo_name=values.get('image_repo_name'), xml_dir=values.get('xml_dir', ''),
        image_dir=values.get('image_dir', ''), config=_analysis_config(values), work_dir=values.get('work_dir'),
    )
    print(json.dumps(result, sort_keys=True))
    return 1 if any('error' in item for item in result) else 0


def _add_analysis_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--target-fs', type=int)
    parser.add_argument('--resolution', type=float)
    parser.add_argument('--avg-window')
    parser.add_argument('--peak-window')


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--source')
    parser.add_argument('--host')
    parser.add_argument('--port', type=int)
    parser.add_argument('--browse-node-id', type=int)
    parser.add_argument('--username')
    parser.add_argument('--password')
    parser.add_argument('--ssl', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--timeout', type=int)
    parser.add_argument('--work-dir')
    parser.add_argument('--queue-dir')
    parser.add_argument('--designer')
    parser.add_argument('--coverage', choices=('complete_programme', 'representative_segment'))
    parser.add_argument('--keep-multichannel', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--force-extract', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--force-design', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--tmdb-api-key')
    parser.add_argument('--audio-type', dest='audio_types', action='append')
    _add_analysis_options(parser)


def _add_sync_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--queue-dir')
    parser.add_argument('--work-dir')
    parser.add_argument('--xml-repo')
    parser.add_argument('--images-repo')
    parser.add_argument('--image-owner')
    parser.add_argument('--image-repo-name')
    parser.add_argument('--xml-dir')
    parser.add_argument('--image-dir')
    _add_analysis_options(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run or publish a BEQDesigner library source')
    parser.add_argument('--config', help='JSON or YAML configuration file')
    commands = parser.add_subparsers(dest='command', required=True)
    _add_run_options(commands.add_parser('run', help='extract and design, without publishing'))
    _add_sync_options(commands.add_parser('sync', help='publish accepted review entries'))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _load_config(args.config)
    try:
        return _run(args, config) if args.command == 'run' else _sync(args, config)
    except ValueError as error:
        build_parser().error(str(error))
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
