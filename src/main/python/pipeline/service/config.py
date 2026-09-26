'''
The pipeline service's own settings (design/pipeline-service.md §7): `service.yaml`, kept apart from the catalogue profile
so the app's Settings drawer, which rewrites the profile, never meets them. Secrets come from the environment, not a file
in the image: `NAME` or, for Docker secrets, `NAME_FILE` naming a file that holds it.

    listen: {host: 0.0.0.0, port: 8080}
    allow_repository_writes: false     # publish, commit and bulk accept over HTTP (never from the schedule)
    history_limit: 200                 # finished jobs kept
    shutdown_grace_seconds: 120        # how long a stop waits for the running job's title in hand
    state_dir: /work/service           # jobs.json, the schedule, the lease's heartbeat; default <work_dir>/service
    schedule: {...}                    # auto mode (chunk S4)
    notify: [...]                      # webhooks (chunk S6)
'''
import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

from pipeline.library.profile import read_config_file

TOKEN_ENV = 'BEQ_SERVICE_TOKEN'
_KEYS = ('listen', 'allow_repository_writes', 'history_limit', 'shutdown_grace_seconds', 'state_dir', 'schedule',
         'notify')


def secret(env: Mapping[str, str], name: str) -> Optional[str]:
    '''
    `name` from the environment, else the contents of the file `name_FILE` names (a Docker secret), stripped; None if
    neither is set or the value is blank.
    :raises ValueError: if `name_FILE` names a file that cannot be read.
    '''
    value = env.get(name)
    if not value and env.get(f'{name}_FILE'):
        path = env[f'{name}_FILE']
        try:
            with open(path, encoding='utf-8') as f:
                value = f.read()
        except OSError as error:
            raise ValueError(f'{name}_FILE names {path}, which cannot be read: {error.strerror}')
    value = (value or '').strip()
    return value or None


@dataclass(frozen=True)
class ServiceConfig:
    profile_path: str
    host: str = '0.0.0.0'
    port: int = 8080
    allow_repository_writes: bool = False
    history_limit: int = 200
    shutdown_grace_seconds: float = 120.0
    state_dir: str = ''                 # '' until the entry point fills it in from the profile's work directory
    schedule: Mapping[str, Any] = field(default_factory=dict)
    notify: Tuple[Mapping[str, Any], ...] = ()
    token: Optional[str] = field(default=None, repr=False)


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f'service config: {message}')


def service_config_from_values(values: Mapping[str, Any], profile_path: str,
                               env: Optional[Mapping[str, str]] = None) -> ServiceConfig:
    '''
    :param values: the parsed service.yaml ({} for none).
    :param env: the environment (default os.environ), for the token.
    :raises ValueError: for an unknown key or a value of the wrong kind -- a typo must not silently change nothing.
    '''
    env = os.environ if env is None else env
    _check(isinstance(values, Mapping), 'the file must be a mapping')
    unknown = sorted(set(values) - set(_KEYS))
    _check(not unknown, f"unknown key{'s' if len(unknown) > 1 else ''} {', '.join(unknown)} (known: {', '.join(_KEYS)})")
    listen = values.get('listen') or {}
    _check(isinstance(listen, Mapping) and set(listen) <= {'host', 'port'}, 'listen takes host and port')
    host = listen.get('host', '0.0.0.0')
    port = listen.get('port', 8080)
    _check(isinstance(host, str) and host != '', 'listen.host must be a host name or address')
    _check(isinstance(port, int) and not isinstance(port, bool) and 0 < port < 65536, 'listen.port must be 1-65535')
    writes = values.get('allow_repository_writes', False)
    _check(isinstance(writes, bool), 'allow_repository_writes must be true or false')
    history = values.get('history_limit', 200)
    _check(isinstance(history, int) and not isinstance(history, bool) and history >= 1, 'history_limit must be 1 or more')
    grace = values.get('shutdown_grace_seconds', 120)
    _check(isinstance(grace, (int, float)) and not isinstance(grace, bool) and grace >= 0,
           'shutdown_grace_seconds must be 0 or more')
    state_dir = values.get('state_dir') or ''
    _check(isinstance(state_dir, str), 'state_dir must be a directory')
    schedule = values.get('schedule') or {}
    _check(isinstance(schedule, Mapping), 'schedule must be a mapping')
    notify = values.get('notify') or []
    _check(isinstance(notify, list) and all(isinstance(n, Mapping) for n in notify), 'notify must be a list of targets')
    return ServiceConfig(profile_path=profile_path, host=host, port=port, allow_repository_writes=writes,
                         history_limit=history, shutdown_grace_seconds=float(grace), state_dir=state_dir,
                         schedule=dict(schedule), notify=tuple(dict(n) for n in notify), token=secret(env, TOKEN_ENV))


def load_service_config(path: Optional[str], profile_path: str,
                        env: Optional[Mapping[str, str]] = None) -> ServiceConfig:
    ''' From the file at `path` (JSON or YAML), or the defaults if None. '''
    values = read_config_file(path) if path else {}
    return service_config_from_values(values or {}, profile_path, env)
