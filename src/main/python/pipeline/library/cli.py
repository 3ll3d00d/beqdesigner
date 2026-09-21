'''Command-line entry point for library runs and explicit publishing.'''
import argparse
import json
import logging
import os
import sqlite3
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import datetime
from typing import Any


from pipeline.config import AnalysisConfig
from pipeline.designer.http_binding import register_declared_designers
from pipeline.designer.registry import registered_designers
from pipeline.designer.manual import MANUAL_DESIGNER
from pipeline.library.bulk import DEFAULT_ACCEPT_THRESHOLD, accept_top_pick, plan_accept
from pipeline.library.index import IndexFileError, LibraryIndex, index_path
from pipeline.library.profile import Profile, SourceSpec, build_source, profile_from_config, read_config_file
from pipeline.library.revise import REVISE_TARGETS, revise_entry
from pipeline.library.run import LibraryRunConfig, run_library
from pipeline.library.season import DEFAULT_TV_MODE, TV_MODES
from pipeline.library.selection import THROUGH, Selection
from pipeline.library.stages import PublishSettings, run_stages
from pipeline.library.state import NEEDS
from pipeline.library.status import ScanSettings, analysis_from_values, report_spec_from_values
from pipeline.library.sync import commit_library, publish_library, sync_library
from pipeline.library.union import UnionLibrarySource
from pipeline.review import describe_publish_error
from pipeline.publish.git import RepoTarget
from pipeline.publish.report import ReportSpec


GIT_FAILED = 3   # exit status: git refused (a rejected push, a repository that is not one), or a file it will not commit


def _load_config(path: str | None) -> dict[str, Any]:
    return {} if path is None else read_config_file(path)


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


def _source_settings(values: dict[str, Any], config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    ''' (the source's kind, its settings) from the flags over the config file's `sources.<kind>:` mapping. '''
    source_name = _required(values, 'source')
    settings = dict(config.get('sources', {}).get(source_name, {}))
    if source_name == 'filesystem':
        settings['globs'] = values.get('globs') or settings.get('globs')
    else:
        settings.update({key: values[key] for key in
                         ('host', 'port', 'browse_node_id', 'username', 'password', 'ssl', 'timeout',
                          'external_id_fields') if values.get(key) is not None})
        # flags replace the config file's rules rather than adding to them, like every other option
        settings['path_mappings'] = values.get('path_maps') or settings.get('path_mappings')
    return source_name, settings


def _source(values: dict[str, Any], config: dict[str, Any]):
    kind, settings = _source_settings(values, config)
    return build_source(kind, settings)


def _analysis_config(values: dict[str, Any]) -> AnalysisConfig:
    return analysis_from_values(values)


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
    register_declared_designers(declared)


def _open_index(work_dir: str) -> LibraryIndex | None:
    ''' The work directory's index, or None if it cannot be opened: remembering failures must never stop a run. '''
    try:
        return LibraryIndex(index_path(work_dir))
    except (OSError, sqlite3.Error) as error:
        logging.getLogger('library_cli').warning('not remembering failures, the index cannot be opened: %s', error)
        return None


_SELECTOR_FLAGS = ('needs', 'match', 'ids', 'new_since_scan', 'through')


def _selection(args: argparse.Namespace, source: str | None) -> Selection:
    ''' The shared selector flags as a Selection (see pipeline.library.selection). '''
    return Selection(needs=tuple(getattr(args, 'needs', None) or ()), source=source, match=args.match,
                     ids=tuple(args.ids or ()),
                     new_since_scan=bool(args.new_since_scan))


_PROFILE_PATHS = ('work_dir', 'queue_dir', 'xml_repo', 'xml_dir', 'images_repo', 'image_dir')


def _effective_profile(profile: Profile, values: dict[str, Any]) -> Profile:
    '''
    The profile with the directories and repositories the run is actually using -- the flags, else `run:`/`sync:` --
    so that what reads them off the profile (the union's sticky claims come from `profile.work_dir` and `queue_dir`)
    sees what the rest of the command sees, and does not lose a claim because a cron job gave its directories by flag.
    '''
    changes = {name: str(values[name]) for name in _PROFILE_PATHS if values.get(name)}
    return replace(profile, **changes) if changes else profile


def _warn_failed_earlier(failed_earlier: list) -> None:
    if failed_earlier:
        _say(f'warning: {len(failed_earlier)} title{"" if len(failed_earlier) == 1 else "s"} skipped: failed earlier '
             f'(same source and settings); use --retry-failed to try again')


def _run_profile(args: argparse.Namespace, config: dict[str, Any], values: dict[str, Any]) -> Profile:
    '''
    The profile a selector `run` works on: the one given (`--profile`, or a config file whose `sources:` is a list), or
    else the one source the older flags and config describe, named by its kind.
    '''
    profile = _effective_profile(profile_from_config(config), values)
    if args.profile or isinstance(config.get('sources'), list):
        return profile
    kind, settings = _source_settings(values, config)
    return replace(profile, sources=(SourceSpec(kind, kind, settings),))


def _run_stages(args: argparse.Namespace, config: dict[str, Any], values: dict[str, Any],
                run_config: LibraryRunConfig) -> int:
    '''`run` with a selector or `--through`: the titles come from the discovery index, not from a fresh listing.'''
    profile = _run_profile(args, config, values)
    if not profile.sources:
        raise ValueError('the profile lists no sources')
    everything = {**dict(config.get('sync') or {}), **values}   # publish and commit take the `sync:` options
    for name in ('xml_repo', 'xml_dir', 'images_repo', 'image_dir'):
        if not everything.get(name) and getattr(profile, name):
            everything[name] = getattr(profile, name)
    settings = ScanSettings.from_values(everything)
    through = args.through or 'design'
    publish = None
    if through in ('publish', 'commit'):
        publish = PublishSettings.from_scan_settings(
            settings, image_owner=everything.get('image_owner'), image_repo_name=everything.get('image_repo_name'),
            push=bool(everything.get('push', True)))
    selection = _selection(args, args.source)
    with LibraryIndex(index_path(run_config.work_dir)) as index:
        if not index.generation:  # never scanned: there is nothing to select from
            index.scan(profile, settings)
        report = run_stages(profile, selection, through, run_config=run_config, index=index, publish=publish,
                            settings=settings, retry_failed=bool(args.retry_failed))
    print(json.dumps(asdict(report), sort_keys=True))
    _warn_failed_earlier(report.run.failed_earlier)
    if report.commit_error:
        _say(f'error: {report.commit_error}')
        return GIT_FAILED
    return 1 if report.failed else 0


def _run(args: argparse.Namespace, config: dict[str, Any]) -> int:
    values = _configured_values(args, config, 'run')
    profile = profile_from_config(config) if args.profile else None
    if profile is not None:
        if not profile.sources:
            raise ValueError('the profile lists no sources')
        for name in ('work_dir', 'queue_dir'):  # the profile may keep them under `sync:` instead of `run:`
            if not values.get(name) and getattr(profile, name):
                values[name] = getattr(profile, name)
        profile = _effective_profile(profile, values)  # the flags' directories, not only the file's
    _register_designers(values, config)
    designer = _required(values, 'designer')
    if designer != MANUAL_DESIGNER and designer not in registered_designers():
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
    if any(getattr(args, name) for name in _SELECTOR_FLAGS) or (profile is not None and args.source):
        return _run_stages(args, config, values, run_config)
    index = _open_index(run_config.work_dir)  # remembers what failed, for `status`
    try:
        report = run_library(UnionLibrarySource(profile) if profile is not None else _source(values, config),
                             run_config, index=index, retry_failed=bool(args.retry_failed))
    finally:
        if index is not None:
            index.close()
    print(json.dumps(asdict(report), sort_keys=True))
    _warn_failed_earlier(report.failed_earlier)
    return 1 if report.failed else 0


def _repo(values: dict[str, Any], name: str) -> RepoTarget | None:
    return RepoTarget(values[name]) if values.get(name) else None


def _publish_values(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    '''
    The `sync:` options, over which the flags. `work_dir` is also taken from the config's `run:` section if `sync:` has
    none: publishing from the .beq projects (so a hand edit is what ships, and what is compared with what was
    published) needs it, and `run:` is where a config that also runs keeps it. Without it a republish would write
    the designer's own pick over an edited project.
    '''
    values = _configured_values(args, config, 'sync')
    if not values.get('work_dir'):
        from_run = (config.get('run') or {}).get('work_dir')
        if from_run:
            values['work_dir'] = from_run
    return values


def _publish_kwargs(values: dict[str, Any]) -> dict[str, Any]:
    return dict(
        meta_defaults=values.get('meta_defaults'), images_repo=_repo(values, 'images_repo'),
        image_owner=values.get('image_owner'), image_repo_name=values.get('image_repo_name'),
        xml_dir=values.get('xml_dir', ''), image_dir=values.get('image_dir', ''), config=_analysis_config(values),
        work_dir=values.get('work_dir'), ids=values.get('ids') or None, republish=bool(values.get('republish', False)),
        report_spec=report_spec_from_values(values) or ReportSpec())


def _print_results(results: list[dict]) -> None:
    ''' The publish results as JSON, without each entry's whole XML (it is in the repository, and in the return value). '''
    print(json.dumps([{k: v for k, v in result.items() if k != 'xml'} for result in results], sort_keys=True))


def _publish_status(results: list[dict]) -> int:
    if any(item.get('error') == 'git_failed' for item in results):
        return GIT_FAILED
    return 1 if any('error' in item for item in results) else 0


def _say(message: str) -> None:
    print(message, file=sys.stderr)


def _publish(args: argparse.Namespace, config: dict[str, Any]) -> int:
    values = _publish_values(args, config)
    result = publish_library(_required(values, 'queue_dir'), RepoTarget(_required(values, 'xml_repo')),
                             **_publish_kwargs(values))
    _print_results(result)
    for item in result:
        if 'error' in item:
            _say(describe_publish_error(item))
    return _publish_status(result)


def _commit_status(result) -> int:
    '''0; 3 if a published file is in the tree but git will not commit it; 1 if a published entry has no file.'''
    for problem in result.not_committed:
        _say(f'error: {problem} is published but git will not commit it (ignored by a .gitignore rule?)')
    for warning in result.warnings:
        _say(f'warning: {warning}')
    if result.not_committed:
        return GIT_FAILED
    return 1 if result.missing else 0


def _commit(args: argparse.Namespace, config: dict[str, Any]) -> int:
    values = _configured_values(args, config, 'sync')
    try:
        result = commit_library(
            _required(values, 'queue_dir'), RepoTarget(_required(values, 'xml_repo')),
            images_repo=_repo(values, 'images_repo'), xml_dir=values.get('xml_dir', ''),
            image_dir=values.get('image_dir', ''), push=bool(values.get('push', True)), ids=values.get('ids') or None)
    except (subprocess.CalledProcessError, OSError) as error:
        _say(f'error: {error}')
        partial = getattr(error, 'partial', None)   # what was committed before git refused stays committed
        print(json.dumps({**(asdict(partial) if partial is not None else {}), 'error': str(error)}, sort_keys=True))
        return GIT_FAILED
    print(json.dumps(asdict(result), sort_keys=True))
    return _commit_status(result)


def _sync(args: argparse.Namespace, config: dict[str, Any]) -> int:
    values = _publish_values(args, config)
    committed = []
    try:
        results = sync_library(_required(values, 'queue_dir'), RepoTarget(_required(values, 'xml_repo')),
                               push=bool(values.get('push', True)), on_committed=committed.append,
                               **_publish_kwargs(values))
    except (subprocess.CalledProcessError, OSError) as error:
        _say(f'error: {error}')
        _print_results(getattr(error, 'results', []))   # published, and committed as far as git got
        return GIT_FAILED
    _print_results(results)
    for item in results:
        if 'error' in item:
            _say(describe_publish_error(item))
    return max(_publish_status(results), _commit_status(committed[0]) if committed else 0)


def _revise(args: argparse.Namespace, config: dict[str, Any]) -> int:
    values = _configured_values(args, config, 'sync')
    queue_dir = _required(values, 'queue_dir')
    ids = _required(values, 'ids')
    to = _required(values, 'to')
    results, failed = [], False
    for entry_id in ids:
        try:
            done = revise_entry(
                queue_dir, entry_id, to, values.get('reason') or '', work_dir=values.get('work_dir'),
                xml_repo=_repo(values, 'xml_repo'), images_repo=_repo(values, 'images_repo'),
                xml_dir=values.get('xml_dir', ''), image_dir=values.get('image_dir', ''))
        except (ValueError, subprocess.CalledProcessError, OSError) as error:  # one bad id must not stop the others
            results.append({'id': entry_id, 'error': str(error)})
            failed = True
        else:
            results.append({'id': entry_id, 'status': done.entry.status, 'revision': done.entry.revision,
                            'reverted': done.reverted, 'extract_invalidated': done.extract_invalidated})
    print(json.dumps(results, sort_keys=True))
    return 1 if failed else 0


def _scan_values(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    ''' `run:` over `sync:` (each with the flags on top): a scan needs the settings of both. '''
    return {**_configured_values(args, config, 'sync'), **_configured_values(args, config, 'run')}


def _scan(args: argparse.Namespace, config: dict[str, Any]) -> int:
    profile = profile_from_config(config)
    if not profile.sources:
        raise ValueError('the profile lists no sources')
    values = _scan_values(args, config)
    settings = ScanSettings.from_values({**values, 'work_dir': values.get('work_dir') or profile.work_dir,
                                         'queue_dir': values.get('queue_dir') or profile.queue_dir})
    if not settings.work_dir:
        raise ValueError('work-dir is required')
    profile = _effective_profile(profile, values)
    with LibraryIndex(index_path(settings.work_dir)) as index:
        if args.from_outputs:
            print(json.dumps({'rebuilt': index.rebuild_from_outputs(settings)}, sort_keys=True))
            return 0
        unknown = set(args.only_sources or ()) - {spec.name for spec in profile.sources}
        if unknown:
            raise ValueError(f"no such source: {', '.join(sorted(unknown))} "
                             f"(the profile has: {', '.join(spec.name for spec in profile.sources)})")
        result = index.scan(profile, settings, only=args.only_sources, allow_empty=bool(args.allow_empty))
    print(json.dumps(asdict(result), sort_keys=True))
    return 1 if result.errors else 0


def _accept(args: argparse.Namespace, config: dict[str, Any]) -> int:
    values = _scan_values(args, config)
    profile = profile_from_config(config)
    if not profile.sources:
        raise ValueError('the profile lists no sources')
    settings = ScanSettings.from_values({**values, 'work_dir': values.get('work_dir') or profile.work_dir,
                                         'queue_dir': values.get('queue_dir') or profile.queue_dir})
    if not settings.work_dir or not settings.queue_dir:
        raise ValueError('work-dir and queue-dir are required')
    profile = _effective_profile(profile, values)
    path = index_path(settings.work_dir)
    if not os.path.isfile(path):
        raise ValueError(f'no index at {path}: run `scan` first')
    threshold = float(values.get('threshold', DEFAULT_ACCEPT_THRESHOLD))
    if not 0 <= threshold <= 1:
        raise ValueError('threshold must be between 0 and 1')
    selection = _selection(args, args.source)
    with LibraryIndex(path) as index:
        options = dict(queue_dir=settings.queue_dir, meta_defaults=settings.meta_defaults, work_dir=settings.work_dir)
        if args.dry_run:
            print(json.dumps(asdict(plan_accept(index, selection, threshold, **options)), sort_keys=True))
            return 0
        report = accept_top_pick(index, selection, threshold, **options)
        index.refresh(profile, settings)
    print(json.dumps(asdict(report), sort_keys=True))
    return 0


def _when(timestamp: float | None) -> str:
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S') if timestamp else 'never'


def _status(args: argparse.Namespace, config: dict[str, Any]) -> int:
    values = _scan_values(args, config)
    work_dir = values.get('work_dir') or profile_from_config(config).work_dir
    if not work_dir:
        raise ValueError('work-dir is required')
    path = index_path(work_dir)
    if not os.path.isfile(path):
        print(f'no index at {path}: run `scan` first')
        return 1
    try:
        with LibraryIndex(path, readonly=True) as index:   # status never creates, migrates or drops an index
            summary = index.summary()
    except IndexFileError as error:
        print(f'{error}')
        return 1
    if args.json:
        print(json.dumps(asdict(summary), sort_keys=True))
        return 0
    if not summary.generation:
        print(f'{path} has never been scanned: run `scan`')
        return 1
    width = max(len(needs) for needs in NEEDS)
    print(f'{summary.titles} titles, last scanned {_when(summary.last_scan_at)}')
    for needs in NEEDS:
        print(f'  {needs:<{width}}  {summary.counts[needs]}')
    print(f'new since the previous scan: {summary.new}')
    flags = ', '.join(f'{name} {count}' for name, count in summary.flags.items() if count)
    print(f'flags: {flags or "none"}')
    for source in summary.sources:
        state = f'FAILED ({source.last_error})' if source.last_error else 'ok'
        print(f'source {source.name}: {state}, {source.item_count} items, last scanned {_when(source.last_scanned)}')
    return 0


_COMMANDS = {'run': _run, 'publish': _publish, 'commit': _commit, 'sync': _sync, 'revise': _revise,
             'scan': _scan, 'status': _status, 'accept': _accept}


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
list. Exit status: 0, 1 if any item failed, 2 for a bad option or config, 3 if git refused while committing (`--through
commit`). Prints the run report as JSON.

With none of the selector flags (--needs --match --id --new-since-scan --through, or --source with --profile) it lists
the source and extracts and designs every title, as it always has. With any of them it works from the last `scan`
(taking one first if there has never been one) on just the titles selected, runs every stage up to --through that each
one still needs, and prints a report of what it did and what it skipped and why. A title whose extraction or design
failed is not tried again while its source and settings are unchanged; --retry-failed tries it again.
"""

_SHARED_SECTION = """\
Every option can also be set in the config file's `sync:` section, under the same name with underscores
(--xml-repo is `xml_repo`); a flag overrides the file. `publish`, `commit`, `sync` and `revise` share that one
section, so a single set of repositories serves them all."""

_PUBLISH_EPILOG = _SHARED_SECTION + """ `sync.meta_defaults` (a mapping of BeqMetadata fields, such as
`source: Disc`) has no flag. Exit status: 0, 1 if any entry could not be published, 2 for a bad option or config,
3 if git refused (an images repository that is not a git repository, say). Prints one JSON result per published or
refused entry (without its XML, which is in the repository); the reason for each refusal is on stderr. Writes files
only -- run `commit` to commit and push them. Never extracts or designs.

The filter is published from each title's .beq project, so a hand edit is what ships, when the work directory is known:
`--work-dir`, or `work_dir` in the `sync:` section, or failing that in the `run:` section. Without one the designer's
own pick is published, which would write over a hand-edited filter, so give it whenever titles have been reviewed by editing their projects. A changed `xml_dir` is a new
location, not a change to a published title: the file at the old one is left behind, for a person to remove.
"""

_COMMIT_EPILOG = _SHARED_SECTION + """ Exit status: 0, 1 if a published entry has no file in its repository
(run `publish` again), 2 for a bad option or config, 3 if git refused (a rejected push, say) or will not commit a
published file (a .gitignore rule matches it). Prints what was committed and pushed per repository -- after a git
failure too, with an `error` key, and what was committed before it stays committed; git's own message is on stderr.
What is already committed or pushed is read from git, so running it again only does what is left. Warns on stderr
when an XML that names a report image is committed without --images-repo (the image is not committed with it).
"""

_REVISE_EPILOG = _SHARED_SECTION + """ A published entry's files are put back as git
has them if they were never committed, and left alone (the title becomes a revision, rewritten at the same path on
the next `publish`) if they were, so give it the repositories. It only changes state: run `run`, or `publish` and
`commit`, afterwards to do the work. Exit status: 0, 1 if any id could not be revised, 2 for a bad option or config.
Prints one JSON result per id.
"""

_SCAN_EPILOG = """\
Reads each source's listing and the outputs (extract manifests, review queue, .beq projects, the two repositories) and
records, for every title, what it needs next -- without extracting, designing or publishing anything and without
reading a media file. The result is a disposable SQLite index in the work directory (`library-index.sqlite`); deleting
it costs only a rescan. Reads the same config or profile file as `run` (`--config FILE` before the command, or
`--profile FILE`): `sources:`, `ignore:`, `run:` and `sync:`. Every option can also be set in those sections under the
same name with underscores; a flag overrides the file. A source that cannot be listed keeps the titles it had and is
reported; so does one that lists nothing when it listed some last time (an unmounted share), unless `--allow-empty`. Exit status: 0, 1 if a source could not be listed, 2 for a bad option or config. Prints the result as JSON.
"""

_STATUS_EPILOG = """\
Prints how many titles need each thing -- attention, review, extract, design, publish, commit -- and how many are done,
from the index the last `scan` wrote (`scan` first; `status` never lists a source). Suitable for a scheduled job to say
what is waiting for a person. Reads the work directory from `--work-dir`, or from the same config or profile file as
`scan`. Exit status: 0, 1 if there is no scanned index, 2 for a bad option or config.
"""

_ACCEPT_EPILOG = """\
Accepts the designer's top pick for the titles that are waiting for review and whose top pick is at least --threshold
confident, then updates the index. It leaves out, and reports, any that a person should still look at: incomplete
metadata, a designer decline, or a `.beq` project someone has edited since it was designed. Each title accepted gets a
reviewer note ("bulk accepted, confidence >= 0.90"). It does not publish: run `run --needs publish --through publish`
(or `publish`) afterwards. Works from the last `scan` and reads the same config or profile file as `scan`; every option
can also be set in `run:`/`sync:` under the same name with underscores, and a flag overrides the file. `--dry-run`
prints what would be accepted and what would be left out, and changes nothing. Exit status: 0, 2 for a bad option,
config or missing index. Prints the result as JSON.
"""

_SYNC_EPILOG = _SHARED_SECTION + """ `sync.meta_defaults` (a mapping of BeqMetadata fields, such as
`source: Disc`) has no flag. `sync` is `publish` followed by `commit`. Exit status: 0, 1 if any entry could not be
published or has no file to commit, 2 for a bad option or config, 3 if git refused or will not commit a published file
(what `commit` says). Prints one JSON result per published or refused entry (without its XML), with the commit
shas of what was committed even if a later push failed. Never extracts or designs. `--work-dir` as for `publish`.
"""


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    source = parser.add_argument_group('library source')
    source.add_argument('--profile', metavar='FILE',
                        help='read a catalogue profile (JSON or YAML) instead of --config: its ordered list of `sources:` '
                             'is merged into one catalogue, a title in two sources is designed once, and its `ignore:` '
                             'rules and `ignore_titles:` are honoured. Its `run:` section supplies every other option; '
                             'flags still override. Not combined with --config, and replaces --source and the '
                             'source options below')
    source.add_argument('--source', help="which library to read: 'jriver' or 'filesystem' (required unless --profile); "
                                         "with --profile it is the name of one of the profile's sources, to run only "
                                         "the titles it owns")
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

    _add_selector_options(parser)
    _add_repo_options(parser, with_image_url_options=True, xml_repo_required=False)   # for --through publish or commit
    _add_push_option(parser)

    metadata = parser.add_argument_group('metadata')
    metadata.add_argument('--tmdb-api-key',
                          help='TMDB API key, to fill in title, genres, poster and (for TV) season details; '
                               'without one only what the library itself says is recorded')
    metadata.add_argument('--audio-type', dest='audio_types', action='append', metavar='TYPE',
                          help='audio format to record, e.g. "DTS-HD MA 5.1"; repeatable')
    _add_analysis_options(parser)


def _add_selector_options(parser: argparse.ArgumentParser, *, needs: bool = True) -> None:
    group = parser.add_argument_group(
        'which titles', 'The selector vocabulary shared with the work list: every one given must hold. They read the '
                        'index the last `scan` wrote.')
    if needs:
        group.add_argument('--needs', action='append', choices=NEEDS, metavar='NEEDS',
                           help='only titles whose next need is this (one of ' + ', '.join(NEEDS) + '); repeatable')
    group.add_argument('--match', help='only titles whose title, name, id or path contains this text (ignoring case)')
    group.add_argument('--id', dest='ids', action='append', metavar='ID',
                       help='only this title, by its catalogue id; repeatable')
    group.add_argument('--new-since-scan', action='store_true', default=None,
                       help='only titles first seen by the latest scan')
    if needs:
        group.add_argument('--through', choices=THROUGH,
                           help='run every stage up to and including this one that each title still needs (default '
                                'design): extract; design (extracts first); publish (writes the accepted titles, and '
                                'published ones that are out of date, into the repositories); commit (also commits '
                                'and pushes them). A person reviews between design and publish, so a title never '
                                'goes past design on its own')
        group.add_argument('--retry-failed', action='store_true', default=None,
                           help='also run titles whose extraction or design failed earlier and whose source and '
                                'settings have not changed since (normally skipped, so a failure is not repeated '
                                'every night)')


def _add_repo_options(parser: argparse.ArgumentParser, with_image_url_options: bool,
                      xml_repo_required: bool = True) -> None:
    repos = parser.add_argument_group('repositories')
    repos.add_argument('--xml-repo', help='local clone of the repository the filter XML goes to'
                                          + (' (required)' if xml_repo_required else ' (needed only for a published entry)'))
    repos.add_argument('--xml-dir', help='folder within the XML repository to put the files in (default: its root)')
    repos.add_argument('--images-repo', help='local clone of the repository report images go to; without one no '
                                              f"image is {'made' if with_image_url_options else 'touched'}")
    repos.add_argument('--image-dir', help='folder within the images repository to put images in (default: its root)')
    if with_image_url_options:
        repos.add_argument('--image-owner', help="GitHub owner used to build image URLs (default: read from the images "
                                                 "repository's remote)")
        repos.add_argument('--image-repo-name', help="GitHub repository name used to build image URLs (default: read "
                                                     "from the images repository's remote)")


def _add_push_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--push', action=argparse.BooleanOptionalAction, default=None,
                        help='push the commits to the remotes, images repository first (default: yes); --no-push '
                             'commits locally only, to look before anything leaves this machine')


def _add_publish_options(parser: argparse.ArgumentParser) -> None:
    where = parser.add_argument_group('what to publish')
    where.add_argument('--id', dest='ids', action='append', metavar='ID',
                       help='publish only this entry, by its id (the file name in the queue, without .json); '
                            'repeatable; default: every accepted entry')
    where.add_argument('--republish', action='store_true', default=None,
                       help='also write again each published entry whose catalogue copy is out of date -- its metadata, '
                            'poster, report style or filter changed since it was published, or its XML is missing '
                            'from the repository -- at the same path, without a second review. Needs the work '
                            'directory (--work-dir, or `work_dir` in the config) to see a hand-edited filter, else '
                            'it reverts it; a changed --xml-dir is a new location, so the old file is left behind')
    where.add_argument('--queue-dir', help='review queue directory to publish from (required)')
    where.add_argument('--work-dir',
                       help='the run\'s work directory (default: `work_dir` in the config file\'s `sync:`, else `run:` '
                            'section): publish the filter from each title\'s .beq project, so a hand edit is what '
                            'ships. Without a work directory the designer\'s original pick is published, which '
                            'writes over a hand-edited filter')
    _add_repo_options(parser, with_image_url_options=True)
    _add_analysis_options(parser)


def _add_commit_options(parser: argparse.ArgumentParser) -> None:
    what = parser.add_argument_group('what to commit')
    what.add_argument('--queue-dir', help='review queue directory whose published entries are committed (required)')
    what.add_argument('--id', dest='ids', action='append', metavar='ID',
                      help='commit only this published entry\'s files, by its id; repeatable; default: every '
                           'published entry')
    _add_repo_options(parser, with_image_url_options=False)
    _add_push_option(parser)


def _add_revise_options(parser: argparse.ArgumentParser) -> None:
    what = parser.add_argument_group('what to send back')
    what.add_argument('--queue-dir', help='review queue directory the entries are in (required)')
    what.add_argument('--id', dest='ids', action='append', metavar='ID',
                      help='an entry id (the file name in the queue, without .json); repeatable (required)')
    what.add_argument('--to', choices=REVISE_TARGETS,
                      help='how far back: review = pick again; design = also design again on the next run; extract = '
                           'also extract the audio again (required)')
    what.add_argument('--reason', help='recorded in the entry\'s reviewer note')
    what.add_argument('--work-dir', help='the run\'s work directory (required for --to extract)')
    _add_repo_options(parser, with_image_url_options=False, xml_repo_required=False)


def _add_scan_options(parser: argparse.ArgumentParser) -> None:
    what = parser.add_argument_group('what to scan')
    what.add_argument('--profile', metavar='FILE',
                      help='read a catalogue profile (JSON or YAML) instead of --config; not combined with --config')
    what.add_argument('--source', dest='only_sources', action='append', metavar='NAME',
                      help='rescan only this source, by the name the profile gives it (repeatable); the others keep '
                           'what the last scan found')
    what.add_argument('--from-outputs', action='store_true',
                      help='instead of listing any source, rebuild the index from the outputs alone (the review '
                           'queue, extract manifests, projects and repositories); what has no queue entry reappears '
                           'at the next scan')
    what.add_argument('--allow-empty', action='store_true',
                      help='accept a source that lists no items at all even though it listed some at the last scan '
                           '(normally that is taken to be an unmounted share or a failed query, so the previous listing '
                           'is kept and the source reported as failed)')
    where = parser.add_argument_group('where things are')
    where.add_argument('--work-dir', help='directory for extracted audio, caches and the index (required)')
    where.add_argument('--queue-dir', help='review queue directory')
    _add_settings_options(parser)


def _add_settings_options(parser: argparse.ArgumentParser) -> None:
    settings = parser.add_argument_group('settings that decide what is up to date',
                                         'Give the same values as `run` and `publish`, or the index describes work '
                                         'they would not do.')
    settings.add_argument('--designer', help='the designer `run` uses; a title designed with another is stale')
    settings.add_argument('--coverage', choices=('complete_programme', 'representative_segment'),
                          help='how much of the programme the designer analyses (default complete_programme)')
    settings.add_argument('--keep-multichannel', action=argparse.BooleanOptionalAction, default=None,
                          help='whether `run` also keeps the multichannel extraction (default: no)')
    settings.add_argument('--tv-mode', choices=TV_MODES,
                          help='episode: a title per TV episode (default); season: one per season')
    _add_repo_options(parser, with_image_url_options=True, xml_repo_required=False)   # the image owner/repo are in the digest
    _add_analysis_options(parser)


def _add_accept_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--profile', metavar='FILE',
                        help='read a catalogue profile (JSON or YAML) instead of --config; not combined with --config')
    what = parser.add_argument_group('what to accept')
    what.add_argument('--source', help='only titles owned by this source of the profile, by name')
    _add_selector_options(parser, needs=False)
    what.add_argument('--threshold', type=float,
                      help=f'the smallest confidence of the designer\'s top pick to accept, 0 to 1 (default '
                           f'{DEFAULT_ACCEPT_THRESHOLD:.2f})')
    what.add_argument('--dry-run', action='store_true',
                      help='change nothing: print the titles that would be accepted, the ones left out and why')
    where = parser.add_argument_group('where things are')
    where.add_argument('--work-dir', help='the work directory: its index and the .beq projects (required)')
    where.add_argument('--queue-dir', help='review queue directory (required)')
    _add_settings_options(parser)


def _add_status_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--profile', metavar='FILE',
                        help='read a catalogue profile (JSON or YAML) instead of --config, to find the work directory')
    parser.add_argument('--work-dir', help='the work directory whose index to read')
    parser.add_argument('--json', action='store_true',
                        help='print the summary as JSON: generation, last_scan_at, titles, counts per needs, new, '
                             'flags and each source\'s last scan and error')


def _add_sync_options(parser: argparse.ArgumentParser) -> None:
    _add_publish_options(parser)
    _add_push_option(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run or publish a BEQDesigner library source: `run` extracts audio and designs filters into a '
                    'review queue; `publish` writes the entries a person has accepted into the catalogue '
                    'repositories and `commit` commits and pushes them (`sync` does both). They are separate on '
                    'purpose, so an unattended `run` never publishes.')
    parser.add_argument('--config', help='JSON or YAML configuration file (give it before the command)')
    commands = parser.add_subparsers(dest='command', required=True)
    for name, help_text, epilog, add_options in (
            ('run', 'extract and design, without publishing', _RUN_EPILOG, _add_run_options),
            ('publish', 'write accepted review entries into the repositories, without committing', _PUBLISH_EPILOG,
             _add_publish_options),
            ('commit', 'commit and push what publish wrote, one commit and one push per repository', _COMMIT_EPILOG,
             _add_commit_options),
            ('sync', 'publish accepted review entries, then commit and push them', _SYNC_EPILOG,
             _add_sync_options),
            ('revise', 'send entries back for another review, design or extraction', _REVISE_EPILOG,
             _add_revise_options),
            ('scan', 'discover what each title needs, without doing any of it', _SCAN_EPILOG, _add_scan_options),
            ('accept', "accept the designer's top pick for confident titles waiting for review", _ACCEPT_EPILOG,
             _add_accept_options),
            ('status', 'count the titles that need each thing, from the last scan', _STATUS_EPILOG,
             _add_status_options)):
        add_options(commands.add_parser(name, help=help_text, epilog=epilog,
                                        formatter_class=argparse.RawDescriptionHelpFormatter))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, 'profile', None) and args.config:
        build_parser().error('--profile replaces --config; give one')
    try:
        config = read_config_file(args.profile) if getattr(args, 'profile', None) else _load_config(args.config)
    except (OSError, ValueError) as error:   # a missing or malformed file is a bad option: exit 2, not a traceback
        build_parser().error(f'cannot read {args.profile or args.config}: {error}')
    try:
        return _COMMANDS[args.command](args, config)
    except (ValueError, IndexFileError) as error:   # the latter: the index file is not one this may overwrite
        build_parser().error(str(error))
    except subprocess.CalledProcessError as error:   # git, where a command has not already handled it
        _say(f'error: {error}')
        return GIT_FAILED
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
