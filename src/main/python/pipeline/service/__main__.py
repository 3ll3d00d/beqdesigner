'''
Runs the pipeline service (design/pipeline-service.md):

    python -m pipeline.service --profile /config/profile.yaml [--service-config /config/service.yaml]

It serves the HTTP interface (the OpenAPI document at /openapi.json, Swagger UI at /docs) until stopped; SIGTERM or Ctrl-C
stops taking jobs, cancels the running one (the title in hand finishes, for up to `shutdown_grace_seconds`) and exits.

Without a token (BEQ_SERVICE_TOKEN, or BEQ_SERVICE_TOKEN_FILE) it refuses to start, unless given --no-auth and bound to the
loopback address. Exit status 2 is a bad option, profile or service config.
'''
import argparse
import ipaddress
import logging
import os
import signal
import sys
import threading
from contextlib import contextmanager
from typing import Any, List, Mapping, Optional, Tuple

from pipeline.service.config import ServiceConfig, load_service_config
from pipeline.service.context import load_context
from pipeline.service.designer import DesignerProbe
from pipeline.service.jobs import JobManager
from pipeline.service.scheduler import AutoScheduler
from pipeline.service.notify import Notifier
from pipeline.service.work import executor, job_failed

logger = logging.getLogger('pipeline_service')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='python -m pipeline.service', description=__doc__.split('\n\n')[0].strip(),
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--profile', default=os.environ.get('BEQ_PROFILE'),
                        help='the catalogue profile (JSON or YAML) the jobs run on; default $BEQ_PROFILE')
    parser.add_argument('--service-config', default=os.environ.get('BEQ_SERVICE_CONFIG'),
                        help="the service's own settings (service.yaml); default $BEQ_SERVICE_CONFIG, else none")
    parser.add_argument('--host', help='listen on this address (over service.yaml listen.host)')
    parser.add_argument('--port', type=int, help='listen on this port (over service.yaml listen.port)')
    parser.add_argument('--static-dir', default=os.environ.get('BEQ_SERVICE_STATIC'),
                        help='a local copy of the API pages\' scripts (swagger-ui-dist, redoc); default '
                             '$BEQ_SERVICE_STATIC, else they load from a CDN')
    parser.add_argument('--ui-dir', default=os.environ.get('BEQ_SERVICE_UI'),
                        help='the built browser app (src/main/web, npm run build), served at /ui; default $BEQ_SERVICE_UI, '
                             'else /ui says how to build it')
    parser.add_argument('--no-auth', action='store_true',
                        help='serve without a token; only allowed on a loopback address (127.0.0.1, ::1)')
    return parser


def _loopback(host: str) -> bool:
    if host == 'localhost':
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def build(argv: Optional[List[str]] = None, env: Optional[Mapping[str, str]] = None) -> Tuple[Any, JobManager, ServiceConfig]:
    '''
    The app, its job manager and its settings, from the command line.
    :raises SystemExit: (2) for a bad option, profile or service config.
    '''
    parser = build_parser()
    args = parser.parse_args(argv)
    env = os.environ if env is None else env
    if not args.profile:
        parser.error('--profile (or $BEQ_PROFILE) is required')
    try:
        config = load_service_config(args.service_config, args.profile, env)
        work_dir = load_context(args.profile, env).work_dir
    except (OSError, ValueError) as error:
        parser.error(str(error))
    changes = {name: value for name, value in (('host', args.host), ('port', args.port)) if value is not None}
    if not config.state_dir:
        if not work_dir:
            parser.error('the profile names no work directory (run.work_dir), and service.yaml no state_dir')
        changes['state_dir'] = os.path.join(work_dir, 'service')
    if changes:
        from dataclasses import replace
        config = replace(config, **changes)
    if args.no_auth and not _loopback(config.host):
        parser.error(f'--no-auth is only allowed on a loopback address, not {config.host}')
    if not config.token and not args.no_auth:
        parser.error('no token: set BEQ_SERVICE_TOKEN (or BEQ_SERVICE_TOKEN_FILE), or use --no-auth on 127.0.0.1')
    from pipeline.service.api import create_app
    manager = JobManager(executor(args.profile, env), state_dir=config.state_dir, history_limit=config.history_limit,
                         allow_repository_writes=config.allow_repository_writes, failed=job_failed)
    designer = DesignerProbe(args.profile, env)
    notifier = None

    def designer_down(reason: str) -> None:
        if notifier is not None:
            notifier.designer_unavailable(designer.last.name if designer.last else '', reason)
    try:
        scheduler = AutoScheduler(manager, config.state_dir, dict(config.schedule), designer=designer.unavailable,
                                  on_designer_down=designer_down)
    except (OSError, ValueError) as error:
        manager.stop(grace_seconds=1)
        parser.error(f'schedule: {error}')
    try:
        notifier = Notifier(manager, config.profile_path, config.notify, env=env)
    except (OSError, ValueError) as error:
        scheduler.stop()
        manager.stop(grace_seconds=1)
        parser.error(f'notify: {error}')
    try:
        app = create_app(manager, config, require_token=not args.no_auth, env=env, static_dir=args.static_dir,
                         ui_dir=args.ui_dir, scheduler=scheduler, notifier=notifier, designer=designer)
    except ValueError as error:   # a --ui-dir that is not a built app
        notifier.stop()
        scheduler.stop()
        manager.stop(grace_seconds=1)
        parser.error(str(error))
    app.state.scheduler = scheduler
    app.state.notifier = notifier
    return app, manager, config


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
    app, manager, config = build(argv)
    import uvicorn

    class Server(uvicorn.Server):
        @contextmanager
        def capture_signals(self):
            '''
            SIGTERM and SIGINT stop the server, as uvicorn's own do -- but uvicorn then raises the signal again, which ends
            the process before the running job is stopped. Here it is not raised again: main() stops the job and exits 0.
            '''
            if threading.current_thread() is not threading.main_thread():
                yield
                return
            previous = {sig: signal.signal(sig, self.handle_exit) for sig in (signal.SIGINT, signal.SIGTERM)}
            try:
                yield
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)

    logger.info('serving %s on %s:%s; jobs kept in %s', config.profile_path, config.host, config.port, config.state_dir)
    # an open event stream must not hold a stop up past the job's own grace
    server = Server(uvicorn.Config(app, host=config.host, port=config.port, timeout_graceful_shutdown=5, log_level='info'))
    try:
        server.run()
    finally:
        app.state.scheduler.stop()
        manager.stop(grace_seconds=config.shutdown_grace_seconds)
        app.state.notifier.stop()
    return 0


if __name__ == '__main__':
    sys.exit(main())
