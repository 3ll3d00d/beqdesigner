'''
Where the work list gets its catalogue profile -- design/library-sync/workflow-rework/implementation-order.md, chunk 26a.

The settings editor is chunk 26c. Until then nobody should have to hand-write a profile file, so:

* if the `LIBRARY_PROFILE_PATH` preference names a file, that file is the profile (`pipeline.library.profile.load_profile`);
* otherwise the profile is **built from the preferences Library Sync already keeps** -- one source (the kind in
  `LIBRARY_SOURCE_DEFAULT` with that kind's saved settings), the work and queue directories, the two repositories, the
  designer and the TV mode; no ignore rules.

`bootstrap_profile()` is the whole of that second case, in one place, so chunk 26c can replace it with an editor that
writes a real profile file and leaves `load_setup()` reading it.

Everything here is UI-free apart from reading the JRiver server list, which lives in a Qt-side module.
'''
import os
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

from model.jriver.connections import load_connections
from model.library_sources import jriver_source_settings
from model.preferences import DESIGNER_DEFAULT, DESIGNER_QUEUE_DIR, LIBRARY_FILESYSTEM_GLOBS, LIBRARY_IMAGES_REPO, \
    LIBRARY_JRIVER_BROWSE_NODE, LIBRARY_JRIVER_CONNECTION, LIBRARY_PROFILE_PATH, LIBRARY_SOURCE_DEFAULT, \
    LIBRARY_TV_MODE, LIBRARY_WORK_DIR, LIBRARY_XML_REPO
from pipeline.library.index import index_path
from pipeline.library.profile import Profile, load_profile, profile_from_config
from pipeline.library.status import ScanSettings

ORIGIN_FILE = 'file'
ORIGIN_PREFERENCES = 'preferences'


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
    return wanted if wanted in available else (available[0] if available else '')


def bootstrap_profile(prefs, designer: str = '') -> Profile:
    '''
    The profile Library Sync's saved preferences amount to: a single source, no ignore rules.
    :param designer: the designer to record; empty means none is chosen (setup_problems() says so).
    '''
    run: Dict[str, Any] = {'work_dir': prefs.get(LIBRARY_WORK_DIR), 'queue_dir': prefs.get(DESIGNER_QUEUE_DIR),
                           'designer': designer, 'tv_mode': prefs.get(LIBRARY_TV_MODE)}
    sync = {'xml_repo': prefs.get(LIBRARY_XML_REPO), 'images_repo': prefs.get(LIBRARY_IMAGES_REPO)}
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
    if not settings.queue_dir:
        problems.append('No review queue directory is chosen.')
    if not settings.designer:
        problems.append('No designer is available.')
    else:
        from pipeline.designer.registry import registered_designers
        if settings.designer not in registered_designers():
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
        return index_path(self.settings.work_dir)

    @property
    def source_names(self) -> List[str]:
        return [s.name for s in self.profile.sources] if self.profile else []


def load_setup(prefs) -> WorkListSetup:
    '''
    The profile the work list works from, with the scan settings and what is missing. Never raises: a profile file
    that cannot be read is an `error`, shown by the window as its empty state.
    '''
    designer = default_designer(prefs)
    path = (prefs.get(LIBRARY_PROFILE_PATH) or '').strip()
    extra: List[str] = []
    if path:
        try:
            profile = load_profile(path)
        except Exception as failure:  # missing file, bad JSON, a YAML library's own errors, a malformed source
            return WorkListSetup(None, None, ORIGIN_FILE, path, (), f'{type(failure).__name__}: {failure}')
        origin = ORIGIN_FILE
        extra = register_profile_designers(profile)
    else:
        profile = bootstrap_profile(prefs, designer)
        origin = ORIGIN_PREFERENCES
    settings = ScanSettings.from_profile(profile)
    if not settings.designer and designer:
        settings = replace(settings, designer=designer)
    return WorkListSetup(profile, settings, origin, path, tuple(extra + setup_problems(profile, settings)))


def register_profile_designers(profile: Profile) -> List[str]:
    '''
    Registers the HTTP designers the profile's `designers:` section declares (name -> URL, or name -> {url, timeout,
    headers}), the way the CLI does for its config file: a run finds its designer in the registry, and the GUI registers
    only the endpoints saved in Preferences. Called whenever the profile is read.
    :return: what is wrong with the section, each as a sentence fit to show (empty if it is fine).
    '''
    from pipeline.designer.http_binding import register_declared_designers
    try:
        register_declared_designers(profile.config.get('designers'))
    except (ValueError, TypeError, AttributeError) as error:
        return [f"The profile's `designers:` section is not usable: {error}"]
    return []
