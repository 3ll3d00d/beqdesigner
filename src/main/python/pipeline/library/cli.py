'''Command-line entry point for library runs and explicit publishing.'''
import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from pipeline.config import AnalysisConfig
from pipeline.designer.http_binding import http_designer
from pipeline.designer.registry import register_designer, registered_designers
from pipeline.library.filesystem import FilesystemLibrarySource
from pipeline.library.jriver import JRiverLibrarySource
from pipeline.library.pathmap import mappings_from_config
from pipeline.library.run import LibraryRunConfig, run_library
from pipeline.library.season import DEFAULT_TV_MODE, TV_MODES
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
    if source_name == 'filesystem':
        globs = values.get('globs') or source_values.get('globs')
        if not globs:
            raise ValueError('glob is required for the filesystem source')
        return FilesystemLibrarySource(list(globs))
    source_values.update({key: values[key] for key in
                          ('host', 'port', 'browse_node_id', 'username', 'password', 'ssl', 'timeout',
                           'external_id_fields') if values.get(key) is not None})
    if source_name != 'jriver':
        raise ValueError(f'unsupported source {source_name!r}; available: filesystem, jriver')
    return JRiverLibrarySource(
        _required(source_values, 'host'), int(_required(source_values, 'port')),
        int(_required(source_values, 'browse_node_id')), username=source_values.get('username'),
        password=source_values.get('password'), ssl=bool(source_values.get('ssl', False)),
        timeout=int(source_values.get('timeout', 5)),
        external_id_fields=source_values.get('external_id_fields'),
        # flags replace the config file's rules rather than adding to them, like every other option
        path_mappings=mappings_from_config(values.get('path_maps') or source_values.get('path_mappings')),
    )


def _analysis_config(values: dict[str, Any]) -> AnalysisConfig:
    configured = values.get('analysis', {})
    fields = {name: values.get(name, configured.get(name))
              for name in ('target_fs', 'resolution', 'avg_window', 'peak_window')}
    return AnalysisConfig(**{name: value for name, value in fields.items() if value is not None})


def _register_designers(values: dict[str, Any], config: dict[str, Any]) -> None:
    '''
    A library run needs the designer registered in this process, and unlike the GUI (which registers the endpoints
    saved in Preferences at startup) the CLI has nothing else to do it. Designers come from the config file's
    `designers` mapping (name -> URL, or name -> {url, timeout, headers}) and `--designer-url NAME=URL`; the flag
    wins for a name in both. A `--designer` that is itself an http(s) URL is registered under that URL.
    '''
    declared: dict[str, Any] = dict(config.get('designers') or {})
    for entry in values.get('designer_urls') or []:
        name, separator, url = entry.partition('=')
        if not separator or not name or not url:
            raise ValueError(f'--designer-url {entry!r} must be NAME=URL')
        declared[name] = url
    designer = values.get('designer')
    if designer and designer.lower().startswith(('http://', 'https://')) and designer not in declared:
        declared[designer] = designer
    for name, spec in declared.items():
        if isinstance(spec, str):
            spec = {'url': spec}
        if not isinstance(spec, dict) or not spec.get('url'):
            raise ValueError(f"designer {name!r} needs a url")
        register_designer(name, http_designer(spec['url'], timeout=float(spec.get('timeout', 300.0)),
                                              headers=spec.get('headers') or None))


def _run(args: argparse.Namespace, config: dict[str, Any]) -> int:
    values = _configured_values(args, config, 'run')
    _register_designers(values, config)
    designer = _required(values, 'designer')
    if designer not in registered_designers():
        raise ValueError(f"designer {designer!r} is not registered; declare it under `designers` in the config "
                         f"file or with --designer-url {designer}=URL (registered: {', '.join(registered_designers()) or 'none'})")
    run_config = LibraryRunConfig(
        work_dir=_required(values, 'work_dir'), queue_dir=_required(values, 'queue_dir'),
        designer=designer, config=_analysis_config(values),
        coverage=values.get('coverage', 'complete_programme'),
        keep_multichannel=bool(values.get('keep_multichannel', False)),
        force_extract=bool(values.get('force_extract', False)),
        force_design=bool(values.get('force_design', False)),
        tmdb_api_key=values.get('tmdb_api_key'),
        audio_types=tuple(values.get('audio_types', ())),
        tv_mode=values.get('tv_mode', DEFAULT_TV_MODE),
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
    group = parser.add_argument_group('analysis', 'How audio is analysed. The defaults are the app\'s own; in a config '
                                                  'file these go under an `analysis:` key inside `run:`/`sync:`.')
    group.add_argument('--target-fs', type=int, help='analysis sample rate in Hz (default 1000)')
    group.add_argument('--resolution', type=float, help='analysis frequency resolution in Hz (default 1.0)')
    group.add_argument('--avg-window', help="window used for the average spectrum (default 'Default')")
    group.add_argument('--peak-window', help="window used for the peak spectrum (default 'Default')")


_RUN_EPILOG = """\
Every option can also be set in the config file's `run:` section, under the same name with underscores
(--work-dir is `work_dir`); a flag overrides the file. A source's own settings may be under `sources.<name>:`
instead. Repeatable flags (--glob, --path-map, --designer-url, --audio-type) replace, rather than add to, the file's
list. Exit status: 0, 1 if any item failed, 2 for a bad option or config. Prints the run report as JSON.
"""

_SYNC_EPILOG = """\
Every option can also be set in the config file's `sync:` section, under the same name with underscores
(--xml-repo is `xml_repo`); a flag overrides the file. `sync.meta_defaults` (a mapping of BeqMetadata fields, such as
`source: Disc`) has no flag. Exit status: 0, 1 if any entry could not be published, 2 for a bad option or config.
Prints one JSON result per published or refused entry. Never extracts or designs.
"""


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    source = parser.add_argument_group('library source')
    source.add_argument('--source', help="which library to read: 'jriver' or 'filesystem' (required)")
    source.add_argument('--glob', dest='globs', action='append', metavar='GLOB',
                        help='filesystem source: a folder (its direct contents) or a glob such as /films/**/*.mkv; '
                             'repeatable. DVD and Blu-ray rip folders are each one title')
    source.add_argument('--host', help='jriver source: the Media Center server host or IP')
    source.add_argument('--port', type=int, help='jriver source: the Media Center web service port (usually 52199)')
    source.add_argument('--browse-node-id', type=int,
                        help="jriver source: the browse node whose files are the library (-1 is the root)")
    source.add_argument('--username', help='jriver source: username, if the server requires authentication')
    source.add_argument('--password', help='jriver source: password (prefer the config file to the command line)')
    source.add_argument('--ssl', action=argparse.BooleanOptionalAction, default=None,
                        help='jriver source: use HTTPS (default: no)')
    source.add_argument('--timeout', type=int, help='jriver source: request timeout in seconds (default 5)')
    source.add_argument('--path-map', dest='path_maps', action='append', metavar='SERVER=LOCAL',
                        help=r'jriver source: translate a folder as the server reports it to the same folder on '
                             r'this machine, e.g. W:\Films=/mnt/films; repeatable, longest match wins')

    output = parser.add_argument_group('where things go')
    output.add_argument('--work-dir', help='directory for extracted audio, caches and .beq project files (required)')
    output.add_argument('--queue-dir', help='review queue directory the designed entries are written to (required)')

    design = parser.add_argument_group('design')
    design.add_argument('--designer',
                        help='name of the designer to run (required); it must be declared with --designer-url or '
                             'the config file\'s `designers:`, or be an http(s) URL itself')
    design.add_argument('--designer-url', dest='designer_urls', action='append', metavar='NAME=URL',
                        help='declare an HTTP designer; repeatable. The config file\'s `designers:` mapping '
                             '(name: URL, or name: {url, timeout, headers}) does the same and also sets a timeout '
                             'and headers')
    design.add_argument('--coverage', choices=('complete_programme', 'representative_segment'),
                        help='how much of the programme the designer analyses (default complete_programme)')
    design.add_argument('--keep-multichannel', action=argparse.BooleanOptionalAction, default=None,
                        help='also keep the full-quality multichannel extraction, give the designer the per-channel '
                             'audio and write a multichannel .beq project (default: no; ignored for TV seasons)')
    design.add_argument('--tv-mode', choices=TV_MODES,
                        help='episode: a filter per TV episode (default); season: join each season into one track, '
                             'design it once and mark every episode in scope')

    redo = parser.add_argument_group('redoing work')
    redo.add_argument('--force-extract', action=argparse.BooleanOptionalAction, default=None,
                      help='extract again even if the cached audio is up to date')
    redo.add_argument('--force-design', action=argparse.BooleanOptionalAction, default=None,
                      help='design again even if unchanged; entries a reviewer has accepted or published are '
                           'never redesigned')

    metadata = parser.add_argument_group('metadata')
    metadata.add_argument('--tmdb-api-key',
                          help='TMDB API key, to fill in title, genres, poster and (for TV) season details; '
                               'without one only what the library itself says is recorded')
    metadata.add_argument('--audio-type', dest='audio_types', action='append', metavar='TYPE',
                          help='audio format to record, e.g. "DTS-HD MA 5.1"; repeatable')
    _add_analysis_options(parser)


def _add_sync_options(parser: argparse.ArgumentParser) -> None:
    where = parser.add_argument_group('what to publish')
    where.add_argument('--queue-dir', help='review queue directory to publish from (required)')
    where.add_argument('--work-dir',
                       help='the run\'s work directory: publish the filter from each title\'s .beq project, so a '
                            'hand edit is what ships, rather than the designer\'s original pick')
    repos = parser.add_argument_group('repositories')
    repos.add_argument('--xml-repo', help='local clone of the repository the filter XML is pushed to (required)')
    repos.add_argument('--xml-dir', help='folder within the XML repository to put the files in (default: its root)')
    repos.add_argument('--images-repo', help='local clone of the repository report images are pushed to; without '
                                              'one no image is made')
    repos.add_argument('--image-dir', help='folder within the images repository to put images in (default: its root)')
    repos.add_argument('--image-owner', help="GitHub owner used to build image URLs (default: read from the images "
                                             "repository's remote)")
    repos.add_argument('--image-repo-name', help="GitHub repository name used to build image URLs (default: read "
                                                 "from the images repository's remote)")
    _add_analysis_options(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run or publish a BEQDesigner library source: `run` extracts audio and designs filters into a '
                    'review queue; `sync` publishes the entries a person has accepted. They are separate on '
                    'purpose, so an unattended `run` never publishes.')
    parser.add_argument('--config', help='JSON or YAML configuration file (give it before the command)')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('run', help='extract and design, without publishing', epilog=_RUN_EPILOG,
                        formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_run_options(commands.choices['run'])
    commands.add_parser('sync', help='publish accepted review entries', epilog=_SYNC_EPILOG,
                        formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_sync_options(commands.choices['sync'])
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
