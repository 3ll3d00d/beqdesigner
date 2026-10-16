"""
Where the work list gets its catalogue profile.

An explicit `LIBRARY_PROFILE_PATH` selects the file. If the preference is unset, an existing default
`library-profile.yaml` is loaded and its path remembered. An explicitly blank path (Reset to preferences),
or no default file, uses `bootstrap_profile()` to build a single-source profile from legacy library preferences.

This module reads Qt preferences, Qt configuration paths and the saved JRiver server list.
"""
import os
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, Optional, Tuple

from qtpy.QtCore import QStandardPaths

from model.jriver.connections import load_connections
from model.library_sources import jriver_source_settings
from model.preferences import DESIGNER_DEFAULT, DESIGNER_QUEUE_DIR, LIBRARY_FILESYSTEM_GLOBS, LIBRARY_IMAGES_REPO, \
    LIBRARY_JRIVER_BROWSE_NODE, LIBRARY_JRIVER_CONNECTION, LIBRARY_PROFILE_PATH, LIBRARY_SOURCE_DEFAULT, \
    LIBRARY_TV_MODE, LIBRARY_WORK_DIR, LIBRARY_XML_REPO, LIBRARY_FILTER_REPO
from pipeline.library.index import index_path
from pipeline.library.profile import Profile, SourceSpec, load_profile, profile_from_config
from pipeline.library.status import ScanSettings
from pipeline.designer.manual import MANUAL_DESIGNER
from pipeline.service.config import jriver_password

ORIGIN_FILE = 'file'
ORIGIN_PREFERENCES = 'preferences'


PROFILE_FILE_NAME = 'library-profile.yaml'


def default_profile_path() -> str:
    '''
    Where a new profile file is offered: **the organisation's configuration folder**, not the work directory. The profile is the one
    durable record of what the catalogue is made from, and the work directory is where extracted audio and the disposable
    index live, which a person may clear to reclaim space.
    '''
    app_root = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppConfigLocation)
    # QSettings' file is in the organisation folder; AppConfigLocation adds the application name beneath it.
    root = os.path.dirname(app_root) if app_root else os.path.join(os.path.expanduser('~'), '3ll3d00d')
    return os.path.join(root, PROFILE_FILE_NAME)


def _bootstrap_source(prefs) -> Optional[Dict[str, Any]]:
    ''' The one source the saved preferences describe, in the profile file's shape; None if it is not set up. '''
    kind = prefs.get(LIBRARY_SOURCE_DEFAULT)
    if kind == 'filesystem':
        globs = [g for g in (prefs.get(LIBRARY_FILESYSTEM_GLOBS) or []) if g]
        return {'name': 'filesystem', 'kind': 'filesystem', 'globs': globs} if globs else None
    if kind == 'jriver':
        wanted = prefs.get(LIBRARY_JRIVER_CONNECTION)
        connection = next((c for c in load_connections(prefs) if c.endpoint == wanted), None)
        if connection is None or connection.port is None:
            return None
        source: Dict[str, Any] = {'name': 'jriver', 'kind': 'jriver',
                                  **jriver_source_settings(connection, int(prefs.get(LIBRARY_JRIVER_BROWSE_NODE)))}
        return source
    return None


def default_designer(prefs) -> str:
    ''' The saved default designer if it is still registered, else the first that is, else ''. '''
    from pipeline.designer.registry import registered_designers
    available = registered_designers()
    wanted = prefs.get(DESIGNER_DEFAULT)
    return wanted if wanted in available else (available[0] if available else MANUAL_DESIGNER)


def bootstrap_profile(prefs, designer: str = '') -> Profile:
    '''
    The profile the saved library preferences amount to: a single source, no ignore rules.
    :param designer: the designer to record; empty means none is chosen (setup_problems() says so).
    '''
    run: Dict[str, Any] = {'work_dir': prefs.get(LIBRARY_WORK_DIR), 'queue_dir': prefs.get(DESIGNER_QUEUE_DIR),
                           'designer': designer, 'tv_mode': prefs.get(LIBRARY_TV_MODE)}
    sync = {'filter_repo': prefs.get(LIBRARY_FILTER_REPO) or prefs.get(LIBRARY_XML_REPO),
            'images_repo': prefs.get(LIBRARY_IMAGES_REPO)}
    source = _bootstrap_source(prefs)
    config: Dict[str, Any] = {'sources': [source] if source else [],
                              'run': {k: v for k, v in run.items() if v},
                              'sync': {k: v for k, v in sync.items() if v}}
    return profile_from_config({k: v for k, v in config.items() if v or k == 'sources'})


def setup_problems(profile: Profile, settings: ScanSettings) -> List[str]:
    ''' What must be set before a scan means anything, each as a sentence fit to show; empty when it can run. '''
    problems = []
    if not profile.sources:
        problems.append('No library source is set up: add folders to search, or choose a JRiver server and browse '
                        'node.')
    if not settings.work_dir:
        problems.append('No work directory is chosen.')
    elif not os.path.isdir(settings.work_dir):
        problems.append(f'The work directory {settings.work_dir} does not exist.')
    if settings.index_dir and not os.path.isdir(settings.index_dir):
        problems.append(f'The index folder {settings.index_dir} does not exist.')
    if not settings.queue_dir:
        problems.append('No review queue directory is chosen.')
    if not settings.designer:
        problems.append('No designer is available.')
    else:
        from pipeline.designer.registry import registered_designers
        if settings.designer != MANUAL_DESIGNER and settings.designer not in registered_designers():
            problems.append(f'The designer {settings.designer!r} is not available: declare it under `designers:` in the '
                            f'profile file, or add it in Preferences.')
    return problems


@dataclass(frozen=True)
class WorkListSetup:
    '''
    :param profile: None only if a profile file was named but could not be read (`error` says why).
    :param settings: what a scan must be given (matches what run and publish are given).
    :param origin: ORIGIN_FILE or ORIGIN_PREFERENCES.
    :param problems: what is missing; a scan needs this empty.
    '''
    profile: Optional[Profile]
    settings: Optional[ScanSettings]
    origin: str
    path: str = ''
    problems: Tuple[str, ...] = ()
    error: str = ''

    @property
    def ready(self) -> bool:
        return self.profile is not None and not self.problems

    @property
    def index_file(self) -> Optional[str]:
        ''' The discovery index of this profile's work directory, if there is one to look in (it is not created). '''
        if self.settings is None or not self.settings.work_dir or not os.path.isdir(self.settings.work_dir):
            return None
        if self.settings.index_dir and not os.path.isdir(self.settings.index_dir):
            return None
        return index_path(self.settings.work_dir, self.settings.index_dir)

    @property
    def source_names(self) -> List[str]:
        return [s.name for s in self.profile.sources] if self.profile else []


def load_setup(prefs) -> WorkListSetup:
    '''
    The profile the work list works from, with the scan settings and what is missing. Never raises: a profile file
    that cannot be read is an `error`, shown by the window as its empty state.
    '''
    designer = default_designer(prefs)
    saved_path = prefs.get(LIBRARY_PROFILE_PATH, default_if_unset=False)
    path = (saved_path or '').strip()
    if saved_path is None:
        default = default_profile_path()
        if os.path.isfile(default):
            path = default
    extra: List[str] = []
    if path:
        try:
            profile = load_profile(path)
        except FileNotFoundError:
            # A profile is a convenient durable override, not a lock-in.  If
            # it was deliberately removed outside the app, discard only the
            # stale pointer and resume from Preferences on this same open.
            prefs.set(LIBRARY_PROFILE_PATH, '')
            profile = bootstrap_profile(prefs, designer)
            origin = ORIGIN_PREFERENCES
        except Exception as failure:  # missing file, bad JSON, a YAML library's own errors, a malformed source
            return WorkListSetup(None, None, ORIGIN_FILE, path, (), f'{type(failure).__name__}: {failure}')
        else:
            origin = ORIGIN_FILE
            extra = register_profile_designers(profile)
            if saved_path is None:
                prefs.set(LIBRARY_PROFILE_PATH, path)
    else:
        profile = bootstrap_profile(prefs, designer)
        origin = ORIGIN_PREFERENCES
    settings = ScanSettings.from_profile(profile)
    if not settings.designer and designer:
        settings = replace(settings, designer=designer)
    return WorkListSetup(profile, settings, origin, path, tuple(extra + setup_problems(profile, settings)))


def profile_with_secrets(profile: Profile, prefs, env: Optional[Mapping[str, str]] = None) -> Profile:
    '''
    The profile a scan or run lists its sources with: a JRiver source that names a user and no password gets one, from the
    environment the pipeline service reads (`JRIVER_PASSWORD_<SOURCE>`, `JRIVER_PASSWORD`), else from the server saved in
    Preferences > JRiver with the same address and user. A profile shared with the service leaves the password out, and
    this copy is never written back to the file: the settings drawer edits the profile as it was read.
    '''
    env = os.environ if env is None else env
    connections = None
    sources = []
    for spec in profile.sources:
        settings = spec.settings
        if spec.kind == 'jriver' and settings.get('username') and not settings.get('password'):
            password = jriver_password(env, spec.name)
            if not password:
                if connections is None:
                    connections = load_connections(prefs)
                endpoint = f"{settings.get('host', '')}:{settings.get('port', '')}"
                password = next((c.password for c in connections
                                 if c.endpoint == endpoint and c.username == settings['username'] and c.password), None)
            if password:
                spec = SourceSpec(spec.name, spec.kind, {**settings, 'password': password})
        sources.append(spec)
    return replace(profile, sources=tuple(sources))


def register_profile_designers(profile: Profile) -> List[str]:
    '''
    Registers the HTTP designers the profile's `designers:` section declares (name -> URL, or name -> {url, timeout,
    headers}), the way the CLI does for its config file: a run finds its designer in the registry, and the GUI registers
    only the endpoints saved in Preferences. Called whenever the profile is read.
    :return: what is wrong with the section, each as a sentence fit to show (empty if it is fine).
    '''
    from pipeline.designer.http_binding import register_declared_designers
    from pipeline.library.run import stage_parallelism
    try:
        run = profile.config.get('run') or {}
        register_declared_designers(profile.config.get('designers'), shared_root=profile.work_dir or None,
                                    queue_depth=stage_parallelism(run.get('parallelism'))['design'])
    except (ValueError, TypeError, AttributeError) as error:
        return [f"The profile's `designers:` section is not usable: {error}"]
    return []
