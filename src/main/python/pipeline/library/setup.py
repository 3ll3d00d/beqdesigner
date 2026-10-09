'''
From a profile or config file (and any options over it) to what a library run needs: the profile with the directories in
use, the `LibraryRunConfig`, the `ScanSettings` the index is judged by and, to publish, the `PublishSettings`.

The command line (`pipeline.library.cli`) and the pipeline service (`pipeline.service`) both build their runs here, so the
same file means the same run from either (design/pipeline-service.md §4). `overrides` are the options given on top of the
file: the command line's flags, or nothing for the service. No argparse, no Qt.
'''
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional, Tuple

from pipeline.designer.http_binding import register_declared_designers
from pipeline.designer.manual import MANUAL_DESIGNER
from pipeline.designer.registry import registered_designers
from pipeline.library.bass import bass_management
from pipeline.library.profile import Profile, profile_from_config
from pipeline.library.disks import disk_limit
from pipeline.library.retention import min_free_gb
from pipeline.library.run import LibraryRunConfig, stage_parallelism, stop_after_unavailable
from pipeline.library.season import DEFAULT_TV_MODE
from pipeline.library.stages import PublishSettings
from pipeline.library.status import ScanSettings, analysis_from_values

PROFILE_PATHS = ('work_dir', 'queue_dir', 'xml_repo', 'xml_dir', 'images_repo', 'image_dir')


def configured_values(config: Mapping[str, Any], section: str,
                      overrides: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    '''
    A section of the file (`run:` or `sync:`) with the overrides on top; an override of None is not given. The current
    `filter_repo`/`filter_dir` names are read into the legacy `xml_repo`/`xml_dir` the code uses.
    :raises ValueError: if the current and legacy names disagree.
    '''
    values = dict(config.get(section, {}) or {})
    for current, old in (('filter_repo', 'xml_repo'), ('filter_dir', 'xml_dir')):
        if values.get(current) and values.get(old) and values[current] != values[old]:
            raise ValueError(f'{section}.{current} and legacy {section}.{old} disagree; keep one value')
        values[old] = values.pop(current, None) or values.get(old)
    values.update({key: value for key, value in (overrides or {}).items() if value is not None})
    return values


def scan_values(config: Mapping[str, Any], overrides: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    ''' `run:` over `sync:` (each with the overrides on top): a scan needs the settings of both. '''
    return {**configured_values(config, 'sync', overrides), **configured_values(config, 'run', overrides)}


def required(values: Mapping[str, Any], name: str) -> Any:
    value = values.get(name)
    if value in (None, ''):
        raise ValueError(f'{name.replace("_", "-")} is required')
    return value


def effective_profile(profile: Profile, values: Mapping[str, Any]) -> Profile:
    '''
    The profile with the directories and repositories the run is actually using -- the overrides, else `run:`/`sync:` --
    so that what reads them off the profile (the union's sticky claims come from `profile.work_dir` and `queue_dir`)
    sees what the rest of the run sees, and does not lose a claim because a cron job gave its directories by flag.
    '''
    changes = {name: str(values[name]) for name in PROFILE_PATHS if values.get(name)}
    return replace(profile, **changes) if changes else profile


def register_designers(values: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    '''
    A library run needs the designer registered in this process, and unlike the GUI (which registers the endpoints
    saved in Preferences at startup) nothing else does it here. Designers come from the file's `designers` mapping
    (name -> URL, or name -> {url, timeout, headers, by_reference}) and `designer_urls` (`NAME=URL` entries, as
    --designer-url gives them); the latter wins for a name in both. A `designer` that is itself an http(s) URL is
    registered under that URL. A `by_reference` designer shares the run's `work_dir` (design/designer-interface.md §7.1).
    '''
    register_declared_designers(_declared_designers(values, config), shared_root=values.get('work_dir') or None,
                                queue_depth=stage_parallelism(values.get('parallelism'))['design'])


def _declared_designers(values: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    declared: dict[str, Any] = dict(config.get('designers') or {})
    for entry in values.get('designer_urls') or []:
        name, separator, url = entry.partition('=')
        if not separator or not name or not url:
            raise ValueError(f'--designer-url {entry!r} must be NAME=URL')
        declared[name] = url
    designer = values.get('designer')
    if designer and designer.lower().startswith(('http://', 'https://')) and designer not in declared:
        declared[designer] = designer
    return declared


@dataclass(frozen=True)
class DesignerEndpoint:
    ''' Where the run's designer answers, as the file declares it. '''
    name: str
    url: str
    by_reference: bool = False
    headers: Optional[dict] = None


def designer_endpoint(values: Mapping[str, Any], config: Mapping[str, Any]) -> Optional[DesignerEndpoint]:
    '''
    The run's designer's HTTP endpoint, or None if it has none this process can ask (the manual designer, or one the
    file does not declare: the run itself then says it is not registered).
    :raises ValueError: for a malformed `--designer-url` entry.
    '''
    name = values.get('designer')
    spec = _declared_designers(values, config).get(name) if name else None
    if isinstance(spec, str):
        spec = {'url': spec}
    if not isinstance(spec, Mapping) or not spec.get('url'):
        return None
    return DesignerEndpoint(name, spec['url'], spec.get('by_reference') is True, spec.get('headers') or None)


def run_profile(config: Mapping[str, Any], values: dict[str, Any]) -> Profile:
    '''
    The profile a run works on, with its directories filled in: `values` lacking `work_dir`/`queue_dir` take the
    profile's (which may keep them under `sync:` rather than `run:`), and the profile takes the values'.
    :raises ValueError: if the profile lists no sources.
    '''
    profile = profile_from_config(config)
    if not profile.sources:
        raise ValueError('the profile lists no sources')
    for name in ('work_dir', 'queue_dir'):
        if not values.get(name) and getattr(profile, name):
            values[name] = getattr(profile, name)
    return effective_profile(profile, values)


def run_config_from_values(values: Mapping[str, Any], config: Mapping[str, Any]) -> LibraryRunConfig:
    '''
    The run's config, after registering the designers the file and values declare.
    :raises ValueError: for a missing directory or designer, or a designer that is not registered.
    '''
    register_designers(values, config)
    designer = required(values, 'designer')
    if designer != MANUAL_DESIGNER and designer not in registered_designers():
        raise ValueError(f"designer {designer!r} is not registered; declare it under `designers` in the config "
                         f"file or with --designer-url {designer}=URL (registered: {', '.join(registered_designers()) or 'none'})")
    parallelism = stage_parallelism(values.get('parallelism'))
    return LibraryRunConfig(
        work_dir=required(values, 'work_dir'), queue_dir=required(values, 'queue_dir'),
        designer=designer, config=analysis_from_values(values),
        coverage=values.get('coverage', 'complete_programme'),
        keep_multichannel=bool(values.get('keep_multichannel', False)),
        force_extract=bool(values.get('force_extract', False)),
        force_design=bool(values.get('force_design', False)),
        tmdb_api_key=values.get('tmdb_api_key'),
        audio_types=tuple(values.get('audio_types', ())),
        tv_mode=values.get('tv_mode', DEFAULT_TV_MODE),
        extract_parallelism=parallelism['extract'], design_parallelism=parallelism['design'],
        stop_after_unavailable=stop_after_unavailable(values.get('stop_after_unavailable')),
        bass_management=bass_management(values.get('bass_management')),
        min_free_gb=min_free_gb(values.get('min_free_gb')),
        disks=disk_limit(values.get('disks')),
    )


def stage_settings(profile: Profile, config: Mapping[str, Any], values: Mapping[str, Any],
                   through: str) -> Tuple[ScanSettings, Optional[PublishSettings]]:
    '''
    What `run_stages` is given besides the run config: the settings the index is refreshed with (`sync:` under the run's
    values, the profile's repositories where neither names them) and, when `through` is publish or commit, how to publish.
    '''
    everything = {**dict(config.get('sync') or {}), **values}   # publish and commit take the `sync:` options
    for current in ('filter_repo', 'filter_dir'):
        everything.pop(current, None)  # profile_from_config already checked and resolved the file's aliases
    for name in ('xml_repo', 'xml_dir', 'images_repo', 'image_dir'):
        if not everything.get(name) and getattr(profile, name):
            everything[name] = getattr(profile, name)
    settings = ScanSettings.from_values(everything)
    publish = None
    if through in ('publish', 'commit'):
        publish = PublishSettings.from_scan_settings(
            settings, image_owner=everything.get('image_owner'), image_repo_name=everything.get('image_repo_name'),
            push=bool(everything.get('push', True)))
    return settings, publish


def index_settings(profile: Profile, values: Mapping[str, Any]) -> ScanSettings:
    ''' The settings a scan or a bulk accept judges titles by: `values` (see scan_values), the profile's directories under. '''
    return ScanSettings.from_values({**values, 'work_dir': values.get('work_dir') or profile.work_dir,
                                     'queue_dir': values.get('queue_dir') or profile.queue_dir})
