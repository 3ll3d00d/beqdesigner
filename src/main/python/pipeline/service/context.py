'''
A job's view of the catalogue profile (design/pipeline-service.md §4): read again at the start of every job, so an edit to
the mounted file takes effect on the next job without a restart, and built into a run by `pipeline.library.setup` -- the
same resolution `run --profile FILE` uses, with no options over the file.

Secrets are applied from the environment over the file (§7), so none has to be written into a profile that other people
and the app read:

    TMDB_API_KEY                       run.tmdb_api_key
    JRIVER_PASSWORD_<SOURCE>           that JRiver source's password (the source's name, upper case, _ for the rest)
    JRIVER_PASSWORD                    every JRiver source's password that the variable above does not give
    BEQ_DESIGNER_HEADERS_<NAME>        a designer's HTTP headers, as a JSON object, over the file's

each also as `NAME_FILE` (a Docker secret).
'''
import copy
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from pipeline.designer.http_binding import DesignerUnavailable, check_designer
from pipeline.library.profile import Profile, read_config_file
from pipeline.library.run import LibraryRunConfig
from pipeline.library.setup import DesignerEndpoint, configured_values, designer_endpoint, index_settings, \
    run_config_from_values, run_profile, scan_values, stage_settings
from pipeline.library.stages import PublishSettings
from pipeline.library.status import ScanSettings
from pipeline.service.config import secret


def env_name(name: str) -> str:
    ''' A source's or designer's name as it appears in an environment variable: upper case, _ for anything else. '''
    return re.sub(r'[^A-Za-z0-9]', '_', name).upper()


def apply_secrets(config: Mapping[str, Any], env: Mapping[str, str]) -> Dict[str, Any]:
    ''' A copy of the profile with the environment's secrets in it. :raises ValueError: for headers that are not JSON. '''
    config = copy.deepcopy(dict(config))
    tmdb = secret(env, 'TMDB_API_KEY')
    if tmdb:
        config['run'] = {**(config.get('run') or {}), 'tmdb_api_key': tmdb}
    sources = config.get('sources')
    named = sources if isinstance(sources, list) else \
        [dict(settings or {}, name=kind, kind=kind) for kind, settings in (sources or {}).items()]
    for entry in named:
        if entry.get('kind') != 'jriver':
            continue
        password = secret(env, f"JRIVER_PASSWORD_{env_name(str(entry.get('name', '')))}") or secret(env, 'JRIVER_PASSWORD')
        if password:
            if isinstance(sources, list):
                entry['password'] = password
            else:
                sources[entry['name']] = {**(sources[entry['name']] or {}), 'password': password}
    designers = dict(config.get('designers') or {})
    for name, declared in list(designers.items()):
        headers = secret(env, f'BEQ_DESIGNER_HEADERS_{env_name(name)}')
        if not headers:
            continue
        try:
            extra = json.loads(headers)
        except json.JSONDecodeError as error:
            raise ValueError(f'BEQ_DESIGNER_HEADERS_{env_name(name)} is not JSON: {error.msg}')
        if not isinstance(extra, dict):
            raise ValueError(f'BEQ_DESIGNER_HEADERS_{env_name(name)} must be a JSON object')
        entry = {'url': declared} if isinstance(declared, str) else dict(declared)
        entry['headers'] = {**(entry.get('headers') or {}), **extra}
        designers[name] = entry
    if designers:
        config['designers'] = designers
    return config


@dataclass
class JobContext:
    ''' One job's profile, and the settings built from it. '''
    config: Dict[str, Any]
    profile: Profile
    values: Dict[str, Any]           # the `run:` section, the profile's directories filled in

    @property
    def work_dir(self) -> str:
        return str(self.values.get('work_dir') or self.profile.work_dir or '')

    def scan_settings(self) -> ScanSettings:
        ''' What a scan or a bulk accept judges titles by (`run:` over `sync:`). '''
        return index_settings(self.profile, scan_values(self.config))

    def run_config(self) -> LibraryRunConfig:
        ''' :raises ValueError: for a missing directory or a designer that is not declared. '''
        return run_config_from_values(self.values, self.config)

    def stage_settings(self, through: str) -> Tuple[ScanSettings, Optional[PublishSettings]]:
        return stage_settings(self.profile, self.config, self.values, through)

    def designer_endpoint(self) -> Optional[DesignerEndpoint]:
        ''' The run's designer's HTTP endpoint, or None for one that has none (the manual designer). '''
        return designer_endpoint(self.values, self.config)

    def designer_unavailable(self, timeout: float = 10.0) -> str:
        ''' Why the run's designer cannot be used now (its `/health` did not answer, ...), or '' if it can. '''
        try:
            endpoint = self.designer_endpoint()
        except ValueError as error:
            return str(error)
        if endpoint is None:
            return ''
        try:
            check_designer(endpoint.url, by_reference=endpoint.by_reference, timeout=timeout, headers=endpoint.headers)
        except DesignerUnavailable as error:
            return str(error)
        return ''


def load_context(profile_path: str, env: Optional[Mapping[str, str]] = None) -> JobContext:
    '''
    :raises ValueError: for a profile that cannot be read or lists no sources, or bad secrets.
    :raises OSError: for a profile file that is not there.
    '''
    config = apply_secrets(read_config_file(profile_path), os.environ if env is None else env)
    values = configured_values(config, 'run')
    profile = run_profile(config, values)
    return JobContext(config, profile, values)
