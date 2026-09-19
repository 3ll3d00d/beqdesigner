'''
The catalogue profile -- design/library-sync/workflow-rework/design.md §12.4: everything one catalogue needs, in one
file the GUI and the CLI both read and write.

It is the CLI's existing config file (`run:`, `sync:`, `designers:`, `sources.<name>:`) with three additions:

    sources:                       # an ORDERED list; earlier wins when two sources have the same file
      - {name: films, kind: jriver, host: media.local, port: 52199, browse_node_id: 1007,
         path_mappings: [{from: 'W:\\', to: /media/films}]}
      - {name: disk, kind: filesystem, globs: [/mnt/extra/**/*.mkv]}
    ignore:                        # see pipeline.library.ignore
      - {path: /media/films/Kids/**}
      - {kind: tv, reason: not doing TV}
    ignore_titles:                 # a single title, by id, with an optional reason
      jriver-3fa9c2-1234: rip is broken

The old shape still loads: `sources` may be the mapping `{jriver: {...}}`, in which case `run.source` names the one
source in use. Nothing in the file is required to be a profile -- a config with no list of sources loads as a profile
of at most one source.
'''
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from pipeline.library.filesystem import FilesystemLibrarySource
from pipeline.library.ignore import IgnoreRule, rules_from_config
from pipeline.library.jriver import JRiverLibrarySource
from pipeline.library.pathmap import mappings_from_config
from pipeline.library.source import LibrarySource

SOURCE_KINDS = ('filesystem', 'jriver')
_JRIVER_KEYS = ('host', 'port', 'browse_node_id', 'username', 'password', 'ssl', 'timeout', 'external_id_fields',
                'path_mappings')


def read_config_file(path: str) -> Dict[str, Any]:
    ''' JSON, or YAML if the suffix says so. :raises ValueError: if the root is not a mapping. '''
    content = Path(path).read_text(encoding='utf-8')
    if Path(path).suffix.lower() in {'.yaml', '.yml'}:
        import yaml
        loaded = yaml.safe_load(content)
    else:
        loaded = json.loads(content)
    if not isinstance(loaded, dict):
        raise ValueError('configuration root must be an object')
    return loaded


def write_config_file(path: str, config: Mapping[str, Any]) -> None:
    if Path(path).suffix.lower() in {'.yaml', '.yml'}:
        import yaml
        text = yaml.safe_dump(dict(config), sort_keys=False)
    else:
        text = json.dumps(config, indent=2)
    Path(path).write_text(text, encoding='utf-8')


@dataclass(frozen=True)
class SourceSpec:
    name: str                 # unique within the profile; what an ignore rule's `source` and "also in" refer to
    kind: str                 # one of SOURCE_KINDS
    settings: Dict[str, Any] = field(default_factory=dict, compare=True, hash=False)

    def __post_init__(self):
        if not self.name or not isinstance(self.name, str):
            raise ValueError('every source needs a name')
        if self.kind not in SOURCE_KINDS:
            raise ValueError(f"source {self.name!r}: kind must be one of {SOURCE_KINDS}, got {self.kind!r}")

    def to_config(self) -> Dict[str, Any]:
        return {'name': self.name, 'kind': self.kind, **self.settings}


def build_source(kind: str, settings: Mapping[str, Any]) -> LibrarySource:
    '''
    :param settings: a source's settings: `globs` for a filesystem source; `host`, `port` and `browse_node_id`
        (required) and `username`, `password`, `ssl`, `timeout`, `external_id_fields`, `path_mappings` for JRiver.
    :raises ValueError: if a required setting is missing or the kind is unknown.
    '''
    if kind == 'filesystem':
        globs = settings.get('globs')
        if not globs:
            raise ValueError('glob is required for the filesystem source')
        return FilesystemLibrarySource(list(globs))
    if kind != 'jriver':
        raise ValueError(f'unsupported source {kind!r}; available: {", ".join(SOURCE_KINDS)}')

    def required(name: str) -> Any:
        value = settings.get(name)
        if value in (None, ''):
            raise ValueError(f'{name.replace("_", "-")} is required')
        return value

    return JRiverLibrarySource(
        required('host'), int(required('port')), int(required('browse_node_id')), username=settings.get('username'),
        password=settings.get('password'), ssl=bool(settings.get('ssl', False)),
        timeout=int(settings.get('timeout', 5)), external_id_fields=settings.get('external_id_fields'),
        path_mappings=mappings_from_config(settings.get('path_mappings')))


@dataclass(frozen=True)
class Profile:
    sources: Tuple[SourceSpec, ...] = ()            # priority order, first wins
    ignore: Tuple[IgnoreRule, ...] = ()
    ignored_titles: Dict[str, str] = field(default_factory=dict, compare=True, hash=False)  # title id -> reason
    work_dir: str = ''
    queue_dir: str = ''
    xml_repo: str = ''
    xml_dir: str = ''
    images_repo: str = ''
    image_dir: str = ''
    config: Dict[str, Any] = field(default_factory=dict, compare=False, hash=False)  # the whole file, as loaded

    def __post_init__(self):
        names = [spec.name for spec in self.sources]
        if len(set(names)) != len(names):
            raise ValueError(f'source names must be unique, got {names}')

    def source(self, name: str) -> SourceSpec:
        return next(spec for spec in self.sources if spec.name == name)

    def to_config(self) -> Dict[str, Any]:
        '''
        The file's content: everything this profile does not manage (`designers`, the rest of `run:`/`sync:`) is kept as
        loaded, and the parts it does manage are written from the profile's own fields.
        '''
        config = json.loads(json.dumps(self.config))  # a deep copy, and proof it is plain data
        config['sources'] = [spec.to_config() for spec in self.sources]
        _set_or_drop(config, 'ignore', [rule.to_config() for rule in self.ignore])
        _set_or_drop(config, 'ignore_titles', dict(self.ignored_titles))
        for section, names in (('run', ('work_dir', 'queue_dir')),
                               ('sync', ('work_dir', 'queue_dir', 'xml_repo', 'xml_dir', 'images_repo', 'image_dir'))):
            for name in names:
                value = getattr(self, name)
                block = config.setdefault(section, {})
                if value:
                    block[name] = value
                else:
                    block.pop(name, None)
            if not config[section]:
                del config[section]
        return config


def _set_or_drop(config: Dict[str, Any], key: str, value) -> None:
    if value:
        config[key] = value
    else:
        config.pop(key, None)


def _sources_from_config(config: Mapping[str, Any]) -> List[SourceSpec]:
    declared = config.get('sources')
    if declared is None:
        declared = {}  # the older shape with nothing under `sources:`, e.g. a filesystem source given only in `run:`
    if isinstance(declared, Mapping):
        return _legacy_sources(config, declared)
    if not isinstance(declared, list):
        raise ValueError('`sources` must be a list of sources (or, in the older shape, a mapping)')
    specs = []
    for entry in declared:
        if not isinstance(entry, Mapping) or 'name' not in entry or 'kind' not in entry:
            raise ValueError(f'each source needs a name and a kind, got {entry!r}')
        settings = {k: v for k, v in entry.items() if k not in ('name', 'kind')}
        specs.append(SourceSpec(entry['name'], entry['kind'], settings))
    return specs


def _legacy_sources(config: Mapping[str, Any], declared: Mapping[str, Any]) -> List[SourceSpec]:
    '''
    The older shape: `sources.<name>: {...}` plus `run.source` naming the one in use. Its filesystem globs may also sit
    in `run:`, since the flags that override them are run options.
    '''
    run = config.get('run') or {}
    chosen = run.get('source')
    if not chosen:
        return []
    settings = dict(declared.get(chosen) or {})
    if chosen == 'filesystem' and run.get('globs') and not settings.get('globs'):
        settings['globs'] = run['globs']
    return [SourceSpec(chosen, chosen, settings)]


def _ignored_titles_from_config(entries) -> Dict[str, str]:
    if not entries:
        return {}
    if isinstance(entries, Mapping):
        return {str(k): str(v or '') for k, v in entries.items()}
    ignored = {}
    for entry in entries:
        if isinstance(entry, str):
            ignored[entry] = ''
        elif isinstance(entry, Mapping) and entry.get('id'):
            ignored[str(entry['id'])] = str(entry.get('reason') or '')
        else:
            raise ValueError(f'ignore_titles entries are ids or {{id, reason}}, got {entry!r}')
    return ignored


def profile_from_config(config: Mapping[str, Any]) -> Profile:
    '''
    :raises ValueError: for a malformed source, ignore rule or duplicate source name.
    '''
    run, sync = config.get('run') or {}, config.get('sync') or {}

    def path(name: str, *sections: Mapping[str, Any]) -> str:
        return next((str(s[name]) for s in sections if s.get(name)), '')

    return Profile(
        sources=tuple(_sources_from_config(config)), ignore=tuple(rules_from_config(config.get('ignore'))),
        ignored_titles=_ignored_titles_from_config(config.get('ignore_titles')),
        work_dir=path('work_dir', run, sync), queue_dir=path('queue_dir', run, sync),
        xml_repo=path('xml_repo', sync), xml_dir=path('xml_dir', sync),
        images_repo=path('images_repo', sync), image_dir=path('image_dir', sync),
        config=json.loads(json.dumps(dict(config))))


def load_profile(path: str) -> Profile:
    return profile_from_config(read_config_file(path))


def save_profile(profile: Profile, path: str) -> None:
    write_config_file(path, profile.to_config())
