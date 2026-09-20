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
import datetime
import json
import os
import shutil
import tempfile
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
    ''' JSON, or YAML if the suffix says so. :raises ValueError: if it is not valid or its root is not a mapping. '''
    content = Path(path).read_text(encoding='utf-8')
    if Path(path).suffix.lower() in {'.yaml', '.yml'}:
        import yaml
        try:
            loaded = yaml.safe_load(content)
        except yaml.YAMLError as error:
            raise ValueError(f'not valid YAML: {error}')
    else:
        loaded = json.loads(content)  # a JSONDecodeError is a ValueError
    if not isinstance(loaded, dict):
        raise ValueError('configuration root must be an object')
    return loaded


def render_config(path: str, config: Mapping[str, Any]) -> str:
    ''' The file's text: YAML if the suffix says so, else JSON. '''
    if Path(path).suffix.lower() in {'.yaml', '.yml'}:
        import yaml
        return yaml.safe_dump(dict(config), sort_keys=False)
    return json.dumps(config, indent=2)


def write_config_file(path: str, config: Mapping[str, Any]) -> None:
    '''
    Writes the file **atomically**: the text goes to a temporary file in the same folder and replaces the file in one step,
    so a failure part way (a full disk, a crash) leaves the earlier file exactly as it was, never a half-written one.
    :raises OSError: if it cannot be written (the earlier file is untouched).
    '''
    text = render_config(path, config)
    directory = os.path.dirname(os.path.abspath(path))
    handle, temporary = tempfile.mkstemp(dir=directory, prefix=f'.{os.path.basename(path)}.', suffix='.tmp')
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as out:
            out.write(text)
        if os.path.exists(path):
            shutil.copymode(path, temporary)   # a replaced file keeps its permissions
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


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
        # a profile built by hand (or read from a key left empty in YAML) may hold None for what it has none of
        object.__setattr__(self, 'sources', tuple(self.sources or ()))
        object.__setattr__(self, 'ignore', tuple(self.ignore or ()))
        object.__setattr__(self, 'ignored_titles', dict(self.ignored_titles or {}))
        names = [spec.name for spec in self.sources]
        if len(set(names)) != len(names):
            raise ValueError(f'source names must be unique, got {names}')

    def source(self, name: str) -> SourceSpec:
        return next(spec for spec in self.sources if spec.name == name)

    def to_config(self) -> Dict[str, Any]:
        '''
        The file's content: everything this profile does not manage (`designers`, the rest of `run:`/`sync:`) is kept as
        loaded, and the parts it does manage are written from the profile's own fields. A directory or repository the
        profile still has as it was loaded is left exactly as the file had it (`run.work_dir` and `sync.work_dir` may
        differ: the profile holds only the one that wins); one that changed is written where the file already had it.

        :raises ValueError: if the config holds a value that is not plain data (YAML reads an unquoted date as a date;
            those are written back as ISO text), or if converting a file in the older mapping shape to a list of sources
            would lose sources the profile does not use (see below).
        '''
        config = _plain(self.config)
        config['sources'] = self._sources_to_config(config)
        _set_or_drop(config, 'ignore', [rule.to_config() for rule in self.ignore])
        _set_or_drop(config, 'ignore_titles', dict(self.ignored_titles))
        for name, sections in _MANAGED_PATHS:
            value = getattr(self, name)
            if value == _first_path(config, name, *sections):
                continue   # unchanged: the file's own (possibly differing) values stay
            present = [section for section in sections if (config.get(section) or {}).get(name)]
            for section in (present or sections[:1]) if value else sections:
                block = config.setdefault(section, {})
                if value:
                    block[name] = value
                else:
                    block.pop(name, None)
        for section in ('run', 'sync'):
            if section in config and not config[section]:
                del config[section]
        return config

    def _sources_to_config(self, config: Dict[str, Any]):
        '''
        The `sources:` value to write. A file in the older shape (`sources: {jriver: {...}}` plus `run.source`) stays in
        it while the profile still fits it (at most one source, named by its kind), so the settings of the sources it
        does not use are kept; otherwise it becomes a list, which cannot hold an unused source, so that is refused when it
        would drop one rather than done silently.
        '''
        declared = config.get('sources')
        if not isinstance(declared, Mapping):
            return [spec.to_config() for spec in self.sources]
        if len(self.sources) <= 1 and all(spec.name == spec.kind for spec in self.sources):
            merged = {name: dict(settings or {}) for name, settings in declared.items()}
            run = config.setdefault('run', {})
            if self.sources:
                (spec,) = self.sources
                merged[spec.name] = dict(spec.settings)
                run['source'] = spec.name
            else:
                run.pop('source', None)
            return merged
        used = {spec.name for spec in self.sources} | {(config.get('run') or {}).get('source')}
        lost = sorted(name for name, settings in declared.items() if settings and name not in used)
        if lost:
            raise ValueError(f"this profile's sources cannot be written in the older `sources:` mapping shape, and a list "
                             f"cannot hold the unused source(s) {', '.join(lost)}; remove them from the file, or "
                             f"convert the file to a list of sources by hand")
        return [spec.to_config() for spec in self.sources]


# where each managed path lives in the file, the winner first: (name, the sections it is read from)
_MANAGED_PATHS = (('work_dir', ('run', 'sync')), ('queue_dir', ('run', 'sync')), ('xml_repo', ('sync',)),
                  ('xml_dir', ('sync',)), ('images_repo', ('sync',)), ('image_dir', ('sync',)))


def _first_path(config: Mapping[str, Any], name: str, *sections: str) -> str:
    ''' What the file says `name` is: the first of the sections that has it. '''
    for section in sections:
        block = config.get(section) or {}
        if block.get(name):
            return str(block[name])
    return ''


def _plain(value: Any) -> Any:
    ''' A deep copy that is proof the value is plain data. YAML reads an unquoted date as one: it becomes ISO text. '''
    def default(obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        raise ValueError(f'the configuration holds {obj!r} ({type(obj).__name__}), which cannot be saved; quote it or '
                         f'use plain text, numbers, lists and mappings')
    try:
        return json.loads(json.dumps(value, default=default))
    except TypeError as error:   # a key that is not text, say a date used as a key
        raise ValueError(f'the configuration cannot be saved: {error}')


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
    return Profile(
        sources=tuple(_sources_from_config(config)), ignore=tuple(rules_from_config(config.get('ignore'))),
        ignored_titles=_ignored_titles_from_config(config.get('ignore_titles')),
        **{name: _first_path(config, name, *sections) for name, sections in _MANAGED_PATHS},
        config=_plain(dict(config)))


def load_profile(path: str) -> Profile:
    return profile_from_config(read_config_file(path))


def save_profile(profile: Profile, path: str) -> None:
    '''
    Writes the profile's file (`Profile.to_config()`, so what the profile does not manage is kept) -- but only if what
    would be written reads back as a valid profile, and atomically (`write_config_file`), so the file is never left
    half-valid.
    :raises ValueError: if the profile would not read back (nothing is written).
    :raises OSError: if the file cannot be written (the earlier file is untouched).
    '''
    config = profile.to_config()
    profile_from_config(config)   # the round trip: what is written must load
    write_config_file(path, config)
