'''
The pipeline service's HTTP interface (design/pipeline-service.md §6): FastAPI over the JobManager. Route handlers never do
pipeline work -- they read the index (read-only, a connection per request) or queue a job -- and the OpenAPI document,
Swagger UI (/docs, with Try it out and Authorize) and ReDoc (/redoc) come from the same types the routes use.
'''
import asyncio
import hmac
import os
from contextlib import contextmanager
from typing import Annotated, Callable, Iterator, List, Mapping, Optional

from fastapi import APIRouter, Depends, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from pipeline.library.index import IndexFileError, LibraryIndex, index_path
from pipeline.library.selection import plan_stages
from pipeline.service import models
from pipeline.service.config import ServiceConfig
from pipeline.service.context import JobContext, load_context
from pipeline.service.jobs import FINISHED, AcceptRequest, JobFinished, JobManager, JobNotFound, RepositoryWritesRefused, \
    RunRequest, ScanRequest

PROBLEM_JSON = 'application/problem+json'


class ServiceProblem(Exception):
    def __init__(self, status: int, title: str, detail: str = ''):
        super().__init__(detail or title)
        self.status, self.title, self.detail = status, title, detail


def _problem(status: int, title: str, detail: str = '', errors=None) -> JSONResponse:
    body = models.Problem(title=title, status=status, detail=detail, errors=errors)
    return JSONResponse(body.model_dump(exclude_none=True), status_code=status, media_type=PROBLEM_JSON)


def _responses(*statuses: int) -> dict:
    return {status: {'model': models.Problem, 'content': {PROBLEM_JSON: {}}} for status in statuses}


def read_version() -> str:
    ''' The release this is (`src/main/python/VERSION`, written by CI), as the app reads it. '''
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'VERSION')
    try:
        with open(path, encoding='utf-8') as f:
            return f.read().strip() or '0.0.0-alpha.1'
    except OSError:
        return '0.0.0-alpha.1'


def create_app(manager: JobManager, config: ServiceConfig, *, require_token: bool = True,
               load: Callable[..., JobContext] = load_context, env: Optional[Mapping[str, str]] = None,
               static_dir: Optional[str] = None, version: Optional[str] = None,
               checks: Optional[Callable[[], List[models.Check]]] = None) -> FastAPI:
    '''
    :param require_token: False only for a service bound to loopback and started with --no-auth.
    :param static_dir: a local copy of swagger-ui-dist and redoc, served at /static (the Docker image has one); None loads
        them from a CDN.
    :param checks: what /ready checks (default: the profile, the work directory, ffmpeg and the designer).
    '''
    version = version or read_version()
    if require_token and not config.token:
        raise ValueError('a token is required: set BEQ_SERVICE_TOKEN (or BEQ_SERVICE_TOKEN_FILE)')
    app = FastAPI(title='BEQDesigner pipeline service', version=models.API_VERSION, docs_url=None, redoc_url=None,
                  summary='Scan a library, extract and design BEQ filters for titles chosen by filter, on demand or on a '
                          'schedule, and follow the jobs that do it.',
                  description='Reviewing a design stays in the BEQDesigner app. Every /v1 route needs the bearer token '
                              '(Authorize, above). `GET /health` says which release this is.')
    bearer = HTTPBearer(auto_error=False, description='The service token (BEQ_SERVICE_TOKEN).')

    def authorised(credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(bearer)]) -> None:
        if not require_token:
            return
        if credentials is None or not hmac.compare_digest(credentials.credentials.encode(), config.token.encode()):
            raise ServiceProblem(401, 'Unauthorized', 'a valid bearer token is required')

    def context() -> JobContext:
        try:
            return load(config.profile_path, env)
        except (OSError, ValueError) as error:
            raise ServiceProblem(503, 'Profile unavailable', f'{config.profile_path}: {error}')

    @contextmanager
    def read_index(ctx: JobContext) -> Iterator[LibraryIndex]:
        path = index_path(ctx.work_dir) if ctx.work_dir else ''
        if not path or not os.path.isfile(path):
            raise ServiceProblem(409, 'Not scanned', 'the library has not been scanned yet: POST /v1/jobs/scan')
        try:
            with LibraryIndex(path, readonly=True) as index:
                yield index
        except IndexFileError as error:
            raise ServiceProblem(503, 'Index unavailable', str(error))

    def check_sources(ctx: JobContext, *filters: models.TitleFilter, names: tuple = ()) -> None:
        known = [spec.name for spec in ctx.profile.sources]
        asked = set(names) | {f.source for f in filters if f.source}
        unknown = sorted(asked - set(known))
        if unknown:
            raise ServiceProblem(422, 'Unknown source', f"no such source: {', '.join(unknown)} "
                                                        f"(the profile has: {', '.join(known)})")

    # --- errors -------------------------------------------------------------------------------------------------------

    @app.exception_handler(ServiceProblem)
    async def _service_problem(request: Request, error: ServiceProblem):
        response = _problem(error.status, error.title, error.detail)
        if error.status == 401:
            response.headers['WWW-Authenticate'] = 'Bearer'
        return response

    @app.exception_handler(RequestValidationError)
    async def _invalid(request: Request, error: RequestValidationError):
        errors = [{'loc': list(e.get('loc', ())), 'msg': e.get('msg', ''), 'type': e.get('type', '')} for e in error.errors()]
        return _problem(422, 'Invalid request', '; '.join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in errors),
                        errors)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, error: StarletteHTTPException):
        return _problem(error.status_code, str(error.detail), '')

    @app.exception_handler(JobNotFound)
    async def _not_found(request: Request, error: JobNotFound):
        return _problem(404, 'No such job', f'no job {error.args[0]}')

    @app.exception_handler(JobFinished)
    async def _finished(request: Request, error: JobFinished):
        return _problem(409, 'Job finished', str(error))

    @app.exception_handler(RepositoryWritesRefused)
    async def _refused(request: Request, error: RepositoryWritesRefused):
        return _problem(403, 'Repository writes are off', str(error))

    # --- unauthenticated ------------------------------------------------------------------------------------------------

    @app.get('/health', response_model=models.Health, tags=['service'], summary='Liveness')
    def health() -> models.Health:
        return models.Health(version=version)

    def default_checks() -> List[models.Check]:
        from model.ffmpeg import find_missing_ffmpeg_tools
        found: List[models.Check] = []
        try:
            ctx = load(config.profile_path, env)
            found.append(models.Check(name='profile', ok=True, detail=config.profile_path))
        except (OSError, ValueError) as error:
            return [models.Check(name='profile', ok=False, detail=str(error))]
        work = ctx.work_dir
        writable = bool(work) and os.access(work if os.path.isdir(work) else os.path.dirname(work) or '.', os.W_OK)
        found.append(models.Check(name='work_dir', ok=writable, detail=work or 'the profile names no work directory'))
        missing = find_missing_ffmpeg_tools()
        found.append(models.Check(name='ffmpeg', ok=not missing, detail=f"not found: {', '.join(missing)}" if missing else ''))
        try:
            found.append(models.Check(name='designer', ok=True, detail=ctx.run_config().designer))
        except ValueError as error:
            found.append(models.Check(name='designer', ok=False, detail=str(error)))
        return found

    @app.get('/ready', response_model=models.Readiness, tags=['service'], summary='Readiness',
             responses={503: {'model': models.Readiness}})
    def ready():
        found = (checks or default_checks)()
        body = models.Readiness(ready=all(c.ok for c in found), checks=found)
        return body if body.ready else JSONResponse(body.model_dump(), status_code=503)

    # --- the documentation, served here so Try it out works on the LAN ------------------------------------------------

    assets = '/static' if static_dir else None
    if static_dir:
        app.mount('/static', StaticFiles(directory=static_dir), name='static')

    @app.get('/docs', include_in_schema=False)
    def docs():
        extra = dict(swagger_js_url=f'{assets}/swagger-ui-bundle.js', swagger_css_url=f'{assets}/swagger-ui.css',
                     swagger_favicon_url=f'{assets}/favicon-32x32.png') if assets else {}
        return get_swagger_ui_html(openapi_url=app.openapi_url, title=f'{app.title} - Swagger UI',
                                   swagger_ui_parameters={'persistAuthorization': True, 'tryItOutEnabled': True},
                                   **extra)

    @app.get('/redoc', include_in_schema=False)
    def redoc():
        extra = dict(redoc_js_url=f'{assets}/redoc.standalone.js', with_google_fonts=False) if assets else {}
        return get_redoc_html(openapi_url=app.openapi_url, title=f'{app.title} - ReDoc', **extra)

    # --- /v1 ------------------------------------------------------------------------------------------------------------

    v1 = APIRouter(prefix='/v1', dependencies=[Depends(authorised)], responses=_responses(401))

    @v1.get('/status', response_model=models.ServiceStatus, tags=['service'], summary='What the service is doing',
            responses=_responses(503))
    def status() -> models.ServiceStatus:
        ctx = context()
        path = index_path(ctx.work_dir) if ctx.work_dir else ''
        index = None
        if path and os.path.isfile(path):
            with LibraryIndex(path, readonly=True) as opened:
                summary = opened.summary()
            index = models.IndexStatus(
                generation=summary.generation, last_scan_at=models.timestamp(summary.last_scan_at), titles=summary.titles,
                counts=summary.counts, new=summary.new, flags=summary.flags,
                sources=[models.SourceStatus(name=s.name, kind=s.kind, last_scanned=models.timestamp(s.last_scanned),
                                             last_ok=models.timestamp(s.last_ok), last_error=s.last_error,
                                             item_count=s.item_count) for s in summary.sources])
        current = manager.current
        return models.ServiceStatus(version=version, index=index, queued=len(manager.queued),
                                    current_job=models.job_model(current) if current else None)

    @v1.get('/titles', response_model=models.TitlePage, tags=['titles'], summary='List titles, filtered',
            responses=_responses(409, 422, 503))
    def titles(query: Annotated[models.TitleQuery, Query()]) -> models.TitlePage:
        ctx = context()
        check_sources(ctx, query)
        with read_index(ctx) as index:
            rows = query.to_selection().rows(index, include_done=query.include_done)
        page = rows[query.offset:query.offset + query.limit]
        return models.TitlePage(total=len(rows), offset=query.offset, limit=query.limit,
                                titles=[models.Title.of(row) for row in page])

    @v1.get('/titles/{title_id}', response_model=models.Title, tags=['titles'], summary='One title',
            responses=_responses(404, 409, 503))
    def title(title_id: str) -> models.Title:
        with read_index(context()) as index:
            rows = index.titles(ids=[title_id])
        if not rows:
            raise ServiceProblem(404, 'No such title', f'no title {title_id}')
        return models.Title.of(rows[0])

    @v1.post('/plan', response_model=models.PlanPreview, tags=['jobs'], summary='What a run would do (changes nothing)',
             responses=_responses(409, 422, 503))
    def plan(body: models.RunJobRequest) -> models.PlanPreview:
        ctx = context()
        check_sources(ctx, body.filter)
        with read_index(ctx) as index:
            planned = plan_stages(body.filter.to_selection().rows(index), body.through.value,
                                  retry_failed=body.retry_failed)
        return models.PlanPreview(
            through=body.through, label=planned.label,
            planned=[models.PlannedTitle(id=p.row.id, title=p.row.title or p.row.display_name or p.row.id,
                                         stages=list(p.stages)) for p in planned.planned],
            skipped=[models.SkippedTitle(id=s.id, title=s.title, reason=s.reason) for s in planned.skipped])

    def accepted(job) -> JSONResponse:
        body = models.job_model(job)
        return JSONResponse(body.model_dump(mode='json'), status_code=202, headers={'Location': f'/v1/jobs/{job.id}'})

    submitted = {202: {'description': 'Queued; `Location` names the job.'}, **_responses(403, 422, 503)}

    @v1.post('/jobs/scan', status_code=202, response_model=models.Job, tags=['jobs'], summary='List the sources again',
             responses=submitted)
    def submit_scan(body: models.ScanJobRequest):
        check_sources(context(), names=tuple(body.sources))
        return accepted(manager.submit(ScanRequest(sources=tuple(body.sources), allow_empty=body.allow_empty)))

    @v1.post('/jobs/run', status_code=202, response_model=models.Job, tags=['jobs'],
             summary='Extract, design (and, if allowed, publish or commit) the titles a filter selects', responses=submitted)
    def submit_run(body: models.RunJobRequest):
        check_sources(context(), body.filter)
        return accepted(manager.submit(RunRequest(body.filter.to_selection(), body.through.value, body.scan_first,
                                                  body.retry_failed)))

    @v1.post('/jobs/accept', status_code=202, response_model=models.Job, tags=['jobs'],
             summary="Accept the designer's top pick for confident titles (needs allow_repository_writes, unless a dry run)",
             responses=submitted)
    def submit_accept(body: models.AcceptJobRequest):
        check_sources(context(), body.filter)
        return accepted(manager.submit(AcceptRequest(body.filter.to_selection(), body.threshold, body.dry_run)))

    @v1.get('/jobs', response_model=models.JobList, tags=['jobs'], summary='Jobs, newest first')
    def jobs(state: Optional[models.JobState] = None, kind: Optional[models.JobKind] = None,
             limit: Annotated[int, Query(ge=1, le=500)] = 50) -> models.JobList:
        found = [job for job in manager.jobs() if (state is None or job.state == state.value)
                 and (kind is None or job.kind == kind.value)]
        return models.JobList(jobs=[models.job_model(job) for job in found[:limit]])

    @v1.get('/jobs/{job_id}', response_model=models.Job, tags=['jobs'], summary='One job', responses=_responses(404))
    def job(job_id: str):
        return models.job_model(manager.get(job_id))

    @v1.post('/jobs/{job_id}/cancel', response_model=models.Job, tags=['jobs'],
             summary='Cancel: a queued job is dropped, a running one stops before its next title',
             responses=_responses(404, 409))
    def cancel(job_id: str):
        return models.job_model(manager.cancel(job_id))

    @v1.get('/jobs/{job_id}/log', response_model=List[models.JobEvent], tags=['jobs'],
            summary="What the job has reported so far (the last 500 events)", responses=_responses(404))
    def log(job_id: str, after: Annotated[int, Query(ge=0, description='Only events after this seq.')] = 0):
        return [models.JobEvent.of(event) for event in manager.events(job_id, after)]

    @v1.get('/jobs/{job_id}/events', tags=['jobs'], summary='Follow a job: its events as they happen (Server-Sent Events)',
            response_class=StreamingResponse,
            responses={200: {'description': 'A stream of JobEvent, one per `data:` line, until the job ends.',
                             'content': {'text/event-stream': {'schema': {'$ref': '#/components/schemas/JobEvent'}}}},
                       **_responses(404)})
    async def events(job_id: str, request: Request):
        job_ = manager.get(job_id)
        after = int(request.headers.get('last-event-id') or 0)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def listener(job, event):
            if job.id == job_id:
                loop.call_soon_threadsafe(queue.put_nowait, event)
        unsubscribe = manager.subscribe(listener)
        backlog = manager.events(job_id, after)   # taken after subscribing, so nothing falls between the two

        def frame(event) -> str:
            return f"id: {event['seq']}\nevent: {event['type']}\ndata: {models.JobEvent.of(event).model_dump_json()}\n\n"

        def ends(event) -> bool:
            return event['type'] == 'state' and event.get('state') in FINISHED

        async def stream():
            last = after
            try:
                for event in backlog:
                    last = event['seq']
                    yield frame(event)
                    if ends(event):
                        return
                if job_.finished:
                    return
                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=15)
                    except asyncio.TimeoutError:
                        if await request.is_disconnected():
                            return
                        yield ': keep-alive\n\n'
                        continue
                    if event['seq'] <= last:
                        continue
                    last = event['seq']
                    yield frame(event)
                    if ends(event):
                        return
            finally:
                unsubscribe()
        return StreamingResponse(stream(), media_type='text/event-stream', headers={'Cache-Control': 'no-cache'})

    app.include_router(v1)

    def openapi() -> dict:
        '''
        The generated document, corrected to say what the service sends: every error is a Problem (FastAPI would
        describe its 422 as its own HTTPValidationError, which the handler above replaces), as application/problem+json.
        '''
        if app.openapi_schema is None:
            schema = get_openapi(title=app.title, version=app.version, summary=app.summary, description=app.description,
                                 routes=app.routes)
            problem = {PROBLEM_JSON: {'schema': {'$ref': '#/components/schemas/Problem'}}}
            for operations in schema['paths'].values():
                for operation in operations.values():
                    for status, response in operation.get('responses', {}).items():
                        if status == '422' or 'Problem' in str(response.get('content', {})):
                            response['content'] = problem
                            response.setdefault('description', 'A problem')
            for name in ('HTTPValidationError', 'ValidationError'):
                schema['components']['schemas'].pop(name, None)
            app.openapi_schema = schema
        return app.openapi_schema
    app.openapi = openapi
    return app


def openapi_document() -> dict:
    ''' The interface as published (docs/schema/service.openapi.json): the document of an app with no jobs and no profile. '''
    manager = JobManager(lambda job, control: None)
    try:
        return create_app(manager, ServiceConfig(profile_path=''), require_token=False, version='0').openapi()
    finally:
        manager.stop(grace_seconds=1)


if __name__ == '__main__':
    import json
    import sys
    json.dump(openapi_document(), sys.stdout, indent=2, sort_keys=False)
    sys.stdout.write('\n')
