'''
The pipeline service's HTTP interface as types (design/pipeline-service.md §6.5): every request and response body is a named
pydantic model, so each is a schema in the published OpenAPI document and a generated client gets real types. The models
convert to and from the pipeline's own dataclasses (`Selection`, `StagesReport`, ...), which know nothing of pydantic.

Inputs forbid unknown fields: a misspelt filter field is a 422, never a silently wider selection.
'''
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from pipeline.library.bulk import DEFAULT_ACCEPT_THRESHOLD
from pipeline.library.selection import Selection
from pipeline.library.state import FLAG_DUPLICATE, FLAG_GONE, FLAG_IGNORED, FLAG_IN_CATALOGUE, FLAG_SHADOWED
from pipeline.library.year import YEAR_PATTERN, YearRange

API_VERSION = '1.2.0'


class Needs(str, Enum):
    attention = 'attention'
    review = 'review'
    extract = 'extract'
    design = 'design'
    publish = 'publish'
    commit = 'commit'
    done = 'done'


class Through(str, Enum):
    extract = 'extract'
    design = 'design'
    publish = 'publish'
    commit = 'commit'


class AutoThrough(str, Enum):
    extract = 'extract'
    design = 'design'


class Kind(str, Enum):
    movie = 'movie'
    tv = 'tv'


class Tier(str, Enum):
    attention = 'attention'
    human = 'human'
    machine = 'machine'
    done = 'done'


class Flag(str, Enum):
    ignored = FLAG_IGNORED
    shadowed = FLAG_SHADOWED
    gone = FLAG_GONE
    possible_duplicate = FLAG_DUPLICATE
    already_in_catalogue = FLAG_IN_CATALOGUE


class JobKind(str, Enum):
    scan = 'scan'
    run = 'run'
    accept = 'accept'


class JobState(str, Enum):
    queued = 'queued'
    running = 'running'
    succeeded = 'succeeded'
    failed = 'failed'
    cancelled = 'cancelled'
    interrupted = 'interrupted'


class JobOrigin(str, Enum):
    api = 'api'
    schedule = 'schedule'


def _year_range(expression: str) -> str:
    YearRange.parse(expression, allow_empty=False)   # a reversed range passes the pattern; it selects nothing, so refuse it
    return expression.strip()


YearExpression = Annotated[str, Field(pattern=YEAR_PATTERN, examples=['2026', '>=2020', '<1960', '1990-1999'],
                                      description='A year (2026), a comparison (<1960, <=1960, >1999, >=1999) or an '
                                                  'inclusive range (1990-1999). A title with no year never matches.'),
                           AfterValidator(_year_range)]


class Input(BaseModel):
    model_config = ConfigDict(extra='forbid')


def timestamp(value: Optional[float]) -> Optional[datetime]:
    return None if value is None else datetime.fromtimestamp(value, tz=timezone.utc)


# --- the filter ---------------------------------------------------------------------------------------------------------

class TitleFilter(Input):
    ''' Which titles: the work list's filters. Every field given must hold; none is every title. '''
    needs: List[Needs] = Field(default_factory=list, description='What the title needs next (any of these).')
    new_since_scan: bool = Field(False, description='Only titles first seen by the latest scan.')
    source: Optional[str] = Field(None, description="Only titles from this source of the profile, by its name.")
    match: Optional[str] = Field(None, description='Case-insensitive text in the title, name, id or path.')
    ids: List[str] = Field(default_factory=list, description='Only these titles, by catalogue id.')
    kind: Optional[Kind] = None
    year: Optional[YearExpression] = None

    def to_selection(self) -> Selection:
        return Selection(needs=tuple(n.value for n in self.needs), source=self.source, match=self.match,
                         ids=tuple(self.ids), new_since_scan=self.new_since_scan,
                         kind=self.kind.value if self.kind else None, year=self.year)

    @classmethod
    def of(cls, selection: Any) -> 'TitleFilter':
        ''' From a Selection, or from a job history's plain dict of one. '''
        data = selection if isinstance(selection, dict) else selection.__dict__
        return cls(needs=list(data.get('needs') or ()), new_since_scan=bool(data.get('new_since_scan')),
                   source=data.get('source'), match=data.get('match'), ids=list(data.get('ids') or ()),
                   kind=data.get('kind'), year=data.get('year'))


class TitleQuery(TitleFilter):
    ''' The filter as query parameters, with paging. '''
    include_done: bool = Field(False, description='Include titles that need nothing (as the work list does only on Done).')
    limit: int = Field(100, ge=1, le=1000)
    offset: int = Field(0, ge=0)


# --- titles -------------------------------------------------------------------------------------------------------------

class Title(BaseModel):
    id: str
    title: str
    display_name: str
    year: str
    kind: Kind
    source: str
    path: str
    season: str
    episodes: List[int]
    external_ids: Dict[str, str]
    needs: Needs
    tier: Tier
    detail: str
    flags: List[Flag]
    extract_state: str
    design_state: str
    review_state: str
    publish_state: str
    commit_state: str
    confidence: Optional[float]
    candidate_count: int
    failure: str
    is_new: bool
    state_since: Optional[datetime]
    last_seen: Optional[datetime]

    @classmethod
    def of(cls, row) -> 'Title':
        return cls(id=row.id, title=row.title, display_name=row.display_name, year=row.year, kind=row.kind or 'movie',
                   source=row.source, path=row.path, season=row.season, episodes=list(row.episodes),
                   external_ids=dict(row.external_ids), needs=row.needs, tier=row.tier, detail=row.detail,
                   flags=row.flags, extract_state=row.extract_state, design_state=row.design_state,
                   review_state=row.review_state, publish_state=row.publish_state, commit_state=row.commit_state,
                   confidence=row.confidence, candidate_count=row.candidate_count, failure=row.failure,
                   is_new=row.is_new, state_since=timestamp(row.state_since), last_seen=timestamp(row.last_seen))


class TitlePage(BaseModel):
    total: int = Field(description='Titles the filter matches.')
    offset: int
    limit: int
    titles: List[Title]


# --- plans and job requests ---------------------------------------------------------------------------------------------

class ScanJobRequest(Input):
    sources: List[str] = Field(default_factory=list, description="Sources to list again, by name; none is all of them.")
    allow_empty: bool = Field(False, description='A source that now lists nothing replaces its titles (else it keeps them: '
                                                 'an unmounted share).')


class RunJobRequest(Input):
    filter: TitleFilter = Field(default_factory=TitleFilter)
    through: Through = Field(Through.design, description='Run every stage up to this one that each title still needs. '
                                                         'A title waiting for review is never taken past design.')
    scan_first: bool = Field(True, description='List the sources before selecting, so new titles are seen.')
    retry_failed: bool = Field(False, description='Also run titles whose extraction or design failed before.')


class AcceptJobRequest(Input):
    filter: TitleFilter = Field(default_factory=TitleFilter)
    threshold: float = Field(DEFAULT_ACCEPT_THRESHOLD, ge=0, le=1, description="Accept the designer's top pick at least "
                                                                               'this confident.')
    dry_run: bool = Field(False, description='Say what would be accepted and change nothing.')


class PlannedTitle(BaseModel):
    id: str
    title: str
    stages: List[Through]


class SkippedTitle(BaseModel):
    id: str
    title: str
    reason: str


class PlanPreview(BaseModel):
    through: Through
    label: str = Field(description="What the work list's action button would say.")
    planned: List[PlannedTitle]
    skipped: List[SkippedTitle]


# --- results ------------------------------------------------------------------------------------------------------------

class ScanResult(BaseModel):
    generation: int
    titles: int
    new: List[str]
    gone: List[str]
    dropped: List[str]
    errors: Dict[str, str] = Field(description='Source name -> why it could not be listed (its titles are kept).')
    counts: Dict[str, int] = Field(description='Titles per needs.')
    superseded: List[str] = Field(default_factory=list)


class TitleMessage(BaseModel):
    id: str
    message: str


class PublishedTitle(BaseModel):
    model_config = ConfigDict(extra='allow')
    id: str
    image_url: Optional[str] = None
    republished: Optional[bool] = None


class PublishError(BaseModel):
    model_config = ConfigDict(extra='allow')
    id: str
    error: str
    problems: Optional[List[str]] = None
    message: Optional[str] = None


class RepoCommit(BaseModel):
    repo: str
    paths: List[str]
    commit: Optional[str]
    pushed: bool


class CatalogueCommit(BaseModel):
    xml: RepoCommit
    images: Optional[RepoCommit] = None
    missing: List[str]
    not_committed: List[str]
    warnings: List[str]


class RunResult(BaseModel):
    scan: Optional[ScanResult] = Field(None, description='The scan taken first, if one was.')
    through: Through
    selected: int
    extracted: List[str]
    cached: List[str]
    designed: List[str]
    design_cached: List[str]
    failed: List[TitleMessage]
    failed_earlier: List[TitleMessage] = Field(description='Not tried: failed before with the same source and settings.')
    unavailable: List[TitleMessage] = Field(
        default_factory=list,
        description='Not done because something it depends on (the designer, the media storage, JRiver) was '
                    'unavailable. Not remembered as failed: the next run tries it again.')
    meta_unresolved: List[TitleMessage]
    project_edit_preserved: List[str]
    seasons: Dict[str, List[str]]
    published: List[PublishedTitle]
    publish_errors: List[PublishError]
    committed: Optional[CatalogueCommit]
    commit_error: str
    skipped: List[SkippedTitle]
    cancelled: bool
    stopped: str = Field('', description='Why the run stopped before its selection was done, though nobody cancelled '
                                         'it: too many titles in a row met an unavailable dependency.')
    attempted: List[str]
    not_run: List[str]
    counts: Dict[str, int]


class AcceptResult(BaseModel):
    dry_run: bool
    threshold: float
    note: str = ''
    accepted: List[str] = Field(description='Accepted (or, in a dry run, that would be).')
    excluded: List[SkippedTitle] = Field(description='Confident enough, but left for a person, and why.')
    below_threshold: int
    not_for_review: int


def _pairs(values) -> List[TitleMessage]:
    return [TitleMessage(id=v[0], message=v[1]) if not isinstance(v, dict) else TitleMessage(**v) for v in values]


def _data(value: Any) -> Any:
    ''' A dataclass as a dict (a history job's result is already one). '''
    from dataclasses import asdict, is_dataclass
    return asdict(value) if is_dataclass(value) and not isinstance(value, type) else value


def scan_result(value: Any) -> ScanResult:
    return ScanResult(**_data(value))


def run_result(value: Any) -> RunResult:
    ''' From a work.RunOutcome, or a history job's dict of one. '''
    data = _data(value)
    report, scanned = data['report'], data.get('scan')
    run = report.get('run') or {}
    return RunResult(
        scan=scan_result(scanned) if scanned else None, through=report['through'], selected=report['selected'],
        extracted=run.get('extracted', []), cached=run.get('cached', []), designed=run.get('designed', []),
        design_cached=run.get('design_cached', []), failed=_pairs(run.get('failed', [])),
        failed_earlier=_pairs(run.get('failed_earlier', [])), unavailable=_pairs(run.get('unavailable', [])),
        meta_unresolved=_pairs(run.get('meta_unresolved', [])),
        project_edit_preserved=run.get('project_edit_preserved', []), seasons=run.get('seasons', {}),
        published=report.get('published', []), publish_errors=report.get('publish_errors', []),
        committed=report.get('committed'), commit_error=report.get('commit_error', ''),
        skipped=report.get('skipped', []), cancelled=report.get('cancelled', False),
        stopped=report.get('stopped', ''), attempted=report.get('attempted', []),
        not_run=report.get('not_run', []), counts=report.get('counts', {}))


def accept_result(value: Any) -> AcceptResult:
    ''' From an AcceptPlan (a dry run) or AcceptReport, or a history job's dict of either. '''
    data = _data(value)
    dry_run = 'eligible' in data
    return AcceptResult(dry_run=dry_run, threshold=data['threshold'], note=data.get('note', ''),
                        accepted=data['eligible'] if dry_run else data['accepted'], excluded=data.get('excluded', []),
                        below_threshold=data.get('below_threshold', 0), not_for_review=data.get('not_for_review', 0))


# --- jobs ---------------------------------------------------------------------------------------------------------------

class Progress(BaseModel):
    done: int = Field(description='Title-stages finished.')
    total: int
    title: str = Field(description='The title starting now.')
    stage: str
    id: str = ''


class JobBase(BaseModel):
    id: str
    origin: JobOrigin
    state: JobState
    submitted_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    progress: Optional[Progress]
    error: Optional[str] = Field(None, description='Why the job itself failed (a title that failed is in the result).')
    joined_to: Optional[str] = Field(None, description='The run job this one joined: its titles went into that run, '
                                                       'and it ends when that run does, with its result.')


class ScanJob(JobBase):
    kind: Literal['scan'] = 'scan'
    request: ScanJobRequest
    result: Optional[ScanResult]


class RunJob(JobBase):
    kind: Literal['run'] = 'run'
    request: RunJobRequest
    result: Optional[RunResult]


class AcceptJob(JobBase):
    kind: Literal['accept'] = 'accept'
    request: AcceptJobRequest
    result: Optional[AcceptResult]


Job = Annotated[Union[ScanJob, RunJob, AcceptJob], Field(discriminator='kind')]


class JobList(BaseModel):
    jobs: List[Job]


def _request_data(request: Any) -> Dict[str, Any]:
    return _data(request) or {}


def job_model(job) -> Union[ScanJob, RunJob, AcceptJob]:
    ''' A jobs.Job (live, or read back from the history) as its typed model. '''
    common = dict(id=job.id, origin=job.origin, state=job.state, submitted_at=timestamp(job.submitted_at),
                  started_at=timestamp(job.started_at), finished_at=timestamp(job.finished_at),
                  progress=Progress(**_data(job.progress)) if job.progress else None, error=job.error or None,
                  joined_to=getattr(job, 'joined_to', None))
    request = _request_data(job.request)
    finished_ok = job.result is not None
    if job.kind == 'scan':
        return ScanJob(request=ScanJobRequest(sources=list(request.get('sources') or ()),
                                              allow_empty=bool(request.get('allow_empty'))),
                       result=scan_result(job.result) if finished_ok else None, **common)
    selection = TitleFilter.of(request.get('selection') or {})
    if job.kind == 'run':
        return RunJob(request=RunJobRequest(filter=selection, through=request.get('through', 'design'),
                                            scan_first=bool(request.get('scan_first', True)),
                                            retry_failed=bool(request.get('retry_failed'))),
                      result=run_result(job.result) if finished_ok else None, **common)
    return AcceptJob(request=AcceptJobRequest(filter=selection, threshold=request.get('threshold', DEFAULT_ACCEPT_THRESHOLD),
                                              dry_run=bool(request.get('dry_run'))),
                     result=accept_result(job.result) if finished_ok else None, **common)


class JobEvent(BaseModel):
    ''' One thing a job reported, in order (`seq`); `type` says which fields it has. '''
    seq: int
    at: datetime
    type: Literal['state', 'progress', 'ffmpeg', 'event']
    state: Optional[str] = Field(None, description='type state: the job state it entered (or cancelling).')
    done: Optional[int] = None
    total: Optional[int] = None
    title: Optional[str] = None
    stage: Optional[str] = None
    id: Optional[str] = Field(None, description='The title id (progress, ffmpeg).')
    out_time_micros: Optional[int] = None
    total_micros: Optional[int] = None
    title_id: Optional[str] = None
    kind: Optional[str] = Field(None, description='type event: what happened (stage_started, command_finished, ...).')
    text: Optional[str] = Field(None, description='type event: the redacted text a person reads.')

    @classmethod
    def of(cls, event: Dict[str, Any]) -> 'JobEvent':
        return cls(**{**event, 'at': timestamp(event['at'])})


# --- the service --------------------------------------------------------------------------------------------------------

class SourceStatus(BaseModel):
    name: str
    kind: str
    last_scanned: Optional[datetime]
    last_ok: Optional[datetime]
    last_error: str
    item_count: int


class IndexStatus(BaseModel):
    generation: int
    last_scan_at: Optional[datetime]
    titles: int
    counts: Dict[str, int] = Field(description='Titles per needs.')
    new: int
    flags: Dict[str, int]
    sources: List[SourceStatus]


class ServiceStatus(BaseModel):
    version: str
    index: Optional[IndexStatus] = Field(description='None until the first scan.')
    current_job: Optional[Job]
    queued: int
    schedule: Optional['Schedule'] = None
    notify: List['NotifyOutcome'] = Field(default_factory=list)


class ScheduleUpdate(Input):
    enabled: bool = False
    interval_minutes: int = Field(60, ge=5)
    filter: TitleFilter = Field(default_factory=TitleFilter)
    through: AutoThrough = AutoThrough.design
    retry_failed: bool = False


class Schedule(ScheduleUpdate):
    next_run_at: Optional[datetime] = None
    last_run: Optional['ScheduleLastRun'] = None
    last_skip: Optional[str] = None


class ScheduleLastRun(BaseModel):
    job_id: str
    state: JobState
    finished_at: Optional[datetime]


class NotifyEvent(str, Enum):
    review_waiting = 'review_waiting'
    failed = 'failed'
    job_finished = 'job_finished'


class NotifyTest(Input):
    target: str


class NotifyJob(BaseModel):
    id: str
    kind: JobKind
    origin: JobOrigin
    state: JobState
    started_at: Optional[datetime]
    finished_at: Optional[datetime]


class NotifyTitle(BaseModel):
    id: str
    title: str
    year: str
    kind: Kind
    confidence: Optional[float]


class NotifyFailure(BaseModel):
    id: str
    title: str
    message: str


class NotifyLinks(BaseModel):
    job: str
    docs: str


class Notification(BaseModel):
    event: NotifyEvent
    job: NotifyJob
    designed: List[NotifyTitle]
    failed: List[NotifyFailure]
    review_waiting: int
    links: NotifyLinks


class NotifyOutcome(BaseModel):
    name: str
    last_event: Optional[NotifyEvent] = None
    last_attempt_at: Optional[datetime] = None
    ok: Optional[bool] = None
    message: str = ''


class Health(BaseModel):
    status: Literal['ok'] = 'ok'
    version: str


class Check(BaseModel):
    name: str
    ok: bool
    detail: str = ''


class Readiness(BaseModel):
    ready: bool
    checks: List[Check]


class Problem(BaseModel):
    ''' An error, as RFC 9457 problem details. '''
    type: str = 'about:blank'
    title: str
    status: int
    detail: str = ''
    errors: Optional[List[Dict[str, Any]]] = Field(None, description='Field-level validation errors.')
