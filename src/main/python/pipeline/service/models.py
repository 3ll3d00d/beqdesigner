'''
The pipeline service's HTTP interface as types (design/pipeline-service.md §6.5): every request and response body is a named
pydantic model, so each is a schema in the published OpenAPI document and a generated client gets real types. The models
convert to and from the pipeline's own dataclasses (`Selection`, `StagesReport`, ...), which know nothing of pydantic.

Inputs forbid unknown fields: a misspelt filter field is a 422, never a silently wider selection.
'''
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, RootModel, model_validator

from pipeline.library.bulk import DEFAULT_ACCEPT_THRESHOLD
from pipeline.library.selection import Selection
from pipeline.service.executions import MAX_REDESIGN_IDS, redesign_problem
from pipeline.service.window import window_problem
from pipeline.library.state import NEEDS, FLAG_DUPLICATE, FLAG_GONE, FLAG_IGNORED, FLAG_IN_CATALOGUE, FLAG_SHADOWED
from pipeline.library.year import YEAR_PATTERN, YearRange, numeric_year

API_VERSION = '1.17.0'


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


class ExecutionState(str, Enum):
    queued = 'queued'
    running = 'running'
    succeeded = 'succeeded'
    failed = 'failed'
    cancelled = 'cancelled'
    stopped = 'stopped'
    interrupted = 'interrupted'


class StoppedBecause(str, Enum):
    window = 'window'
    paused = 'paused'
    removed = 'removed'


class JobState(str, Enum):
    active = 'active'
    paused = 'paused'
    done = 'done'
    removed = 'removed'


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


# --- active windows (design/job-windows.md) ------------------------------------------------------------------------------

MAX_WINDOW_RANGES = 24


def _ranges(value: Optional[List[str]]) -> Optional[List[str]]:
    problem = window_problem(value) if value is not None else ''
    if problem:
        raise ValueError(problem)
    return value


Ranges = Annotated[List[str], Field(max_length=MAX_WINDOW_RANGES), AfterValidator(_ranges)]
WINDOW_DESCRIPTION = ('The times of day it may run: daily "HH:MM-HH:MM" ranges in the service\'s time zone (one may cross '
                      'midnight, 22:00-06:00); outside them it waits, and a running execution stops between titles: its '
                      'job carries on when they next open. null: the default window (/v1/job-window); []: any time.')


class Windowed(Input):
    window: Optional[Ranges] = Field(None, description=WINDOW_DESCRIPTION)


class DefaultWindowUpdate(Input):
    window: Ranges = Field(description='The window of every job given none: daily "HH:MM-HH:MM" ranges; [] is any time.')


class DefaultWindow(DefaultWindowUpdate):
    time_zone: str = Field(description="The service's time zone, which every window is in.")
    open_now: bool
    next_change: Optional[datetime] = Field(description='When it next opens (or closes, while open); null: never.')


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
    detail: Optional[str] = Field(None, description="Case-insensitive text in the title's detail (what it waits for).")
    flags: List[Flag] = Field(default_factory=list, description='Only titles carrying every one of these flags.')
    declined: Optional[bool] = Field(None, description='true: only titles whose design the designer declined (waiting for '
                                                       'review); false: only those it did not; unset: either.')
    out_of_date: Optional[bool] = Field(
        None, description='true: only designed titles a redesign now could answer differently -- designed with another '
                          'design version, or other designer settings, than the designers that are up now design with '
                          '(contract 1.4; a title designed before it takes its version from its build). false: every '
                          'other title; unset: either. Refused (409) while no designer that is up says its design version.')
    adjusted: Optional[bool] = Field(None, description='true: only titles whose current design was asked to change '
                                                       'something (contract 1.5); false: only those whose was not.')
    unmet: Optional[bool] = Field(None, description='true: only titles where the designer did not honour every '
                                                    'adjustment asked (adapted, declined, not understood or unanswered).')
    several_runs: Optional[bool] = Field(None, description='true: only titles keeping earlier design runs (variants); '
                                                           'false: only titles with one.')

    def to_selection(self) -> Selection:
        return Selection(needs=tuple(n.value for n in self.needs), source=self.source, match=self.match,
                         ids=tuple(self.ids), new_since_scan=self.new_since_scan,
                         kind=self.kind.value if self.kind else None, year=self.year, detail=self.detail,
                         flags=tuple(f.value for f in self.flags), declined=self.declined, out_of_date=self.out_of_date,
                         adjusted=self.adjusted, unmet=self.unmet, several_runs=self.several_runs)

    @classmethod
    def of(cls, selection: Any) -> 'TitleFilter':
        ''' From a Selection, or from a job history's plain dict of one. '''
        data = selection if isinstance(selection, dict) else selection.__dict__
        return cls(needs=list(data.get('needs') or ()), new_since_scan=bool(data.get('new_since_scan')),
                   source=data.get('source'), match=data.get('match'), ids=list(data.get('ids') or ()),
                   kind=data.get('kind'), year=data.get('year'), detail=data.get('detail'),
                   flags=list(data.get('flags') or ()), declined=data.get('declined'), out_of_date=data.get('out_of_date'),
                   adjusted=data.get('adjusted'), unmet=data.get('unmet'), several_runs=data.get('several_runs'))


class TitleSort(str, Enum):
    ''' A column of the titles table to order by; without one, work-list order (tier, then oldest waiting first). '''
    title = 'title'
    year = 'year'
    kind = 'kind'
    source = 'source'
    needs = 'needs'
    detail = 'detail'
    confidence = 'confidence'
    candidate_count = 'candidate_count'
    flags = 'flags'


class TitleQuery(TitleFilter):
    ''' The filter as query parameters, with ordering and paging. '''
    include_done: bool = Field(False, description='Include titles that need nothing (as the work list does only on Done).')
    sort: Optional[TitleSort] = Field(None, description='Order by this column (work-list order within equal values); '
                                                        'a title without a year or confidence comes last either way.')
    descending: bool = Field(False, description='Order `sort` high to low.')

    limit: int = Field(100, ge=1, le=1000)
    offset: int = Field(0, ge=0)

    def order(self, rows: List[Any]) -> List[Any]:
        ''' The index rows (in work-list order) in the order asked for; a missing year or confidence goes last. '''
        if self.sort is None:
            return rows
        key = _SORT_KEYS[self.sort]
        present = [r for r in rows if key(r) is not None]
        missing = [r for r in rows if key(r) is None]
        return sorted(present, key=key, reverse=self.descending) + missing


def _title_text(row) -> str:
    return (row.title or row.display_name or row.id).casefold()


# each returns None for a value the title does not have, which sorts last whichever way round
_SORT_KEYS = {
    TitleSort.title: _title_text,
    TitleSort.year: lambda r: numeric_year(r.year),
    TitleSort.kind: lambda r: r.kind or 'movie',
    TitleSort.source: lambda r: r.source.casefold(),
    TitleSort.needs: lambda r: NEEDS.index(r.needs) if r.needs in NEEDS else len(NEEDS),
    TitleSort.detail: lambda r: r.detail.casefold() or None,
    TitleSort.confidence: lambda r: r.confidence,
    TitleSort.candidate_count: lambda r: r.choices,
    TitleSort.flags: lambda r: ', '.join(r.flags) or None,
}


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
    design_version: Optional[int] = Field(None, description='The design version it was designed with (contract 1.4, or '
                                                            'from its build before it); null if unknown or not designed.')
    adjusted: bool = Field(False, description='Its current design was asked to change something (contract 1.5).')
    unmet_adjustments: int = Field(0, description='How many of those the designer did not simply honour.')
    design_runs: int = Field(0, description='How many design runs it keeps (0: not designed).')
    design_choices: int = Field(0, description='The designs to choose from in every design run it keeps (candidates; the '
                                               'designer\'s rejected ones are not counted).')
    reviewed_at: List['DesignIdentityView'] = Field([], description='The identities a person kept its design at rather '
                                                                    'than redesign it (contract 1.6): it is not out of '
                                                                    'date against them.')

    @classmethod
    def of(cls, row) -> 'Title':
        return cls(id=row.id, title=row.title, display_name=row.display_name, year=row.year, kind=row.kind or 'movie',
                   source=row.source, path=row.path, season=row.season, episodes=list(row.episodes),
                   external_ids=dict(row.external_ids), needs=row.needs, tier=row.tier, detail=row.detail,
                   flags=row.flags, extract_state=row.extract_state, design_state=row.design_state,
                   review_state=row.review_state, publish_state=row.publish_state, commit_state=row.commit_state,
                   confidence=row.confidence, candidate_count=row.candidate_count, failure=row.failure,
                   is_new=row.is_new, state_since=timestamp(row.state_since), last_seen=timestamp(row.last_seen),
                   design_version=row.design_identity.version if row.design_identity else None,
                   adjusted=row.adjusted, unmet_adjustments=row.unmet_adjustments, design_runs=row.design_runs,
                   design_choices=row.choices,
                   reviewed_at=[DesignIdentityView(design_version=v, design_settings=s)
                                for v, s in (row.design_identity.reviewed if row.design_identity else ())])


class DesignIdentityView(BaseModel):
    ''' A design identity (contract 1.4): what a designer answers with. '''
    design_version: int
    design_settings: Optional[str] = None


Title.model_rebuild()


class TitlePage(BaseModel):
    total: int = Field(description='Titles the filter matches.')
    offset: int
    limit: int
    titles: List[Title]


# --- review (design/web-app.md §3) -----------------------------------------------------------------------------------

class QueueStatus(str, Enum):
    pending = 'pending'
    accepted = 'accepted'
    skipped = 'skipped'
    rejected = 'rejected'
    published = 'published'


class DecisionKind(str, Enum):
    accept = 'accept'
    reject = 'reject'


class CandidateView(BaseModel):
    ''' One design a person may pick: a candidate, or one the designer rejected (`rejection_reasons`). '''
    index: int = Field(description='Its position in the designs offered (candidates, then rejected designs): what a '
                                   'Decision names.')
    rejected: bool = Field(description='The designer judged it unfit to publish: accepting it overrides the designer.')
    method: str
    confidence: Optional[float] = Field(description='None only for a decline\'s flat "does not require BEQ" design.')
    mv_adjust_db: float
    gain_reduction_db: Optional[float] = None
    residual_db: Optional[float] = None
    residual_band_hz: Optional[List[float]] = None
    residual_at_hz: Optional[float] = Field(None, description='1.3: where the worst error (residual_db) is.')
    residual_direction: Optional[str] = Field(None, description="1.3: 'over' or 'under' the target there.")
    residual_rms_db: Optional[float] = Field(None, description='1.3: boost-weighted RMS error over residual_band_hz.')
    residual_within_tolerance: Optional[float] = Field(None, description='1.3: share of the requested boost within '
                                                                         'residual_tolerance_db of the target.')
    residual_tolerance_db: Optional[float] = None
    fit: str = Field('', description='How well its filters match their target, in words (pipeline.review.fit_text); '
                                     "'' when the designer measured nothing. Not whether the target is right.")
    commentary: Optional[Dict[str, Any]] = None
    rejection_reasons: List[str] = Field(default_factory=list)
    filters: Dict[str, Any] = Field(description='The design as a .filter document (docs/schema/filter.schema.json).')


class Decline(BaseModel):
    reason: str
    message: str


class Blocked(BaseModel):
    ''' Why each decision is not offered now; empty when it is. '''
    accept: str
    reject: str


class Review(BaseModel):
    ''' What a person decides a title on, and whether they can. '''
    id: str
    title: str
    year: str
    status: QueueStatus
    status_text: str
    digest: str = Field(description='Fingerprint of the designs offered: send it back in a Decision, which is refused if '
                                    'a redesign has changed them since.')
    candidates: List[CandidateView]
    rejected: List[CandidateView]
    chosen_index: Optional[int] = Field(description='The design accepted or published, into the designs offered.')
    declined: Optional[Decline] = Field(description='The designer found nothing to correct; its one candidate is flat.')
    metadata: Dict[str, Any]
    metadata_problems: List[str] = Field(description='What publish would refuse: Accept is not offered until it is fixed.')
    blocked: Blocked
    in_flight: bool = Field(description='A run is working on this title now.')
    playback: str = Field(description='The playback chain the designer was told about.')
    designer: Optional[str]
    designer_build: Optional[str]
    needs: Needs
    detail: str
    reviewer_note: Optional[str]
    metadata_digest: str = Field('', description='What the metadata and artwork were read at: send it back with an edit, '
                                                 'which is refused if they have changed since.')
    metadata_texts: Dict[str, str] = Field({}, description='Each editable field as text, by key (`GET /v1/metadata/fields`).')
    adjustments: List[Dict[str, Any]] = Field([], description='What the next design is asked to change (contract 1.5): '
                                                             '{id, kind, params, text, by, at, reason}.')
    variant_id: Optional[str] = Field(None, description='Which design run this is (v1, v2...); none for one designed '
                                                        'before they were numbered.')
    designed_at: Optional[datetime] = None
    variants: List['VariantSummary'] = Field([], description="The title's other design runs, kept until discarded: "
                                                            'use one again, or chart it beside this one.')
    adjusted_with: List[Dict[str, Any]] = Field([], description='The adjustments the design here was made with '
                                                               '(adjustments is what the next design is asked).')
    adjustment_outcomes: List[Dict[str, Any]] = Field([], description="The designer's answer to each: {id, outcome, "
                                                                     "applied, reason}; none for one it did not answer "
                                                                     "(a designer before 1.5).")
    artwork: Optional['ArtworkState'] = None


class VariantSummary(BaseModel):
    variant_id: str
    designed_at: Optional[datetime]
    designer: Optional[str]
    designer_build: Optional[str]
    design_version: Optional[int]
    designs: int = Field(description='How many designs it offered (candidates, then rejected).')
    declined: bool
    adjusted_with: List[Dict[str, Any]]
    adjustment_outcomes: List[Dict[str, Any]]
    candidates: List[CandidateView] = Field(default_factory=list, description="Its designs, as the Review's own: an index "
                                                                              "is into this run's offered designs.")
    rejected: List[CandidateView] = Field(default_factory=list, description='The designs the designer rejected in it.')


class EditedFilter(Input):
    type: Literal['low_shelf', 'high_shelf', 'peaking_eq']
    freq_hz: float
    gain_db: float
    q: float
    count: int = Field(1, ge=1, le=10, description='How many of it in a row (a stacked shelf).')


class EditedDesign(Input):
    digest: str = Field(description="The Review's digest: what was looked at.")
    source: int = Field(ge=0, description='The design edited, by its index in the Review.')
    filters: List[EditedFilter] = Field(min_length=1, max_length=10)


class VariantChoice(Input):
    digest: str = Field(description="The Review's digest: what was looked at.")


class ArtworkState(BaseModel):
    present: bool = Field(description='The poster file is there to show (`GET /v1/titles/{id}/artwork`).')
    overridden: bool = Field(description='A person chose it (or cleared it): a run never replaces it.')
    name: Optional[str] = Field(None, description="The poster's file name.")


Review.model_rebuild()




class MetadataField(BaseModel):
    key: str
    label: str
    group: Literal['essentials', 'more']
    tip: str
    kind: Literal['text', 'choice', 'choices'] = Field(description='`choices`: several of `choices`, as a comma-separated '
                                                                   'text; `choice`: one, typed or picked.')
    choices: List[str]


class MetadataEdit(Input):
    digest: str = Field(description="The Review's metadata_digest.")
    texts: Dict[str, str] = Field({}, description='The fields changed, as typed, by key; a blank one unsets it (blank '
                                                  'episodes mean none).')
    extras: Dict[str, Any] = Field({}, description='What a TMDB reload filled that has no field of its own (overview, '
                                                   'genres, collection), as the reload gave it.')


class TmdbLookup(Input):
    tmdb_id: Optional[str] = Field(None, description='Look this TMDB id up; else search by title and year.')
    title: Optional[str] = None
    year: Optional[str] = None
    season: Optional[str] = Field(None, description='A season makes it a TV search.')


class TmdbProposal(BaseModel):
    found: bool
    texts: Dict[str, str] = Field(description='The fields TMDB filled, as texts: apply them with a MetadataEdit.')
    extras: Dict[str, Any] = Field(description='overview, genres and collection, to send back as MetadataEdit.extras.')
    message: str


class ArtworkChoice(Input):
    digest: str = Field(description="The Review's metadata_digest.")
    url: Optional[str] = Field(None, description='Download the png or jpeg image here (http or https).')
    data_base64: Optional[str] = Field(None, description='Or use this png or jpeg image, base64-encoded (at most 20 MB).')


class ChartSeries(BaseModel):
    name: str
    kind: Literal['average', 'peak', 'filter'] = Field(description="A measured curve, or (`filter`) a design's own response.")
    filtered: bool = Field(description='After a design (always true of a `filter`).')
    candidate: Optional[int] = Field(None, description='The design it is after, or the response of; none for the measured '
                                                       'curves.')
    variant: Optional[str] = Field(None, description='The design run it is from, when the chart was asked for another.')
    x: List[float] = Field(description='Frequency, Hz.')
    y: List[float] = Field(description='Magnitude, dB.')


class Chart(BaseModel):
    candidate: Optional[int] = Field(description='The one design asked for with `candidate`; none when several or none were.')
    candidates: List[int] = Field([], description='Every design charted, in the order asked.')
    smoothing: Optional[str] = Field(None, description='The smoothing applied to the measured and filtered curves.')
    series: List[ChartSeries]


class Decision(Input):
    decision: DecisionKind
    candidate: Optional[int] = Field(None, ge=0, description='Required to accept: the index of a design in the Review.')
    digest: str = Field(description="The Review's digest: what was looked at.")
    override_rejection: bool = Field(False, description='Accept a design the designer rejected (otherwise a 409).')
    variant: Optional[str] = Field(None, description="Accept a design of this earlier design run (Review.variants): the run "
                                                     "is made current first, the current one kept as a variant, and "
                                                     "`candidate` indexes its designs. Absent: the current run's.")


class NextQuery(TitleFilter):
    ''' The filter as query parameters, and where in its list to look from. '''
    after: Optional[str] = Field(None, description='The title just decided (or open): the next one after it is found. '
                                                   'Absent: the first waiting title in the list.')


class NextTitle(BaseModel):
    id: str
    title: str


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

    redesign: bool = Field(False, description='Invalidate the existing designs before running, then design them in one run '
                                            f'(the whole designer pool at once). Requires through=design and either 1 to '
                                            f'{MAX_REDESIGN_IDS} ids and no other filter, or out_of_date=true (with any '
                                            'other filter but ids): every designed title it selects that is not accepted '
                                            'or published. The titles must be reviewed again.')

    revise_to: Literal['design', 'extract'] = Field('design', description='With redesign: send each title back to '
                                                                         'design (the default), or to extract it again.')
    audio_stream: Optional[int] = Field(None, ge=0, description='With redesign, revise_to=extract and one id: the audio '
                                                                'stream (zero-based) to extract from now on.')

    @model_validator(mode='after')
    def validate_redesign(self):
        if self.redesign:
            redesign_problem(self.filter.to_selection(), self.through.value, raise_it=True)
        if self.audio_stream is not None and not (self.redesign and self.revise_to == 'extract'
                                                  and len(self.filter.ids or []) == 1):
            raise ValueError('audio_stream is for one title, with redesign and revise_to=extract')
        return self


class AdjustmentAsk(Input):
    ''' A change asked of the designer (contract 1.5, design/designer-interface.md §2): a kind it lists, or any other. '''
    kind: str = Field(min_length=1, description='exclude_band, floor, max_boost, goal_tilt, note, or one the designer '
                                                'lists on /v1/status (designer.adjustment_kinds).')
    params: Dict[str, Any] = Field({}, description="The kind's parameters, by name.")
    text: Optional[str] = Field(None, description='What the person said, in their words (a note is only this).')


class ReviseRequest(Windowed):
    ''' Send titles back: to review (decide again), design (designed again) or extract (extracted and designed again). '''
    ids: List[str] = Field(min_length=1, max_length=MAX_REDESIGN_IDS)
    to: Literal['review', 'design', 'extract']
    reason: str = Field('', description='Kept in the reviewer note.')
    run: bool = Field(False, description='With design or extract: do it now, in one run job (refused for review). '
                                         'Without: the titles wait for the next run.')
    adjustments: Optional[List[AdjustmentAsk]] = Field(
        None, description='With design or extract: what the designer is asked to change, from now on, for every title '
                          'sent (replacing what they had; [] asks for none). Absent: each keeps its own.')

    @model_validator(mode='after')
    def window_needs_a_run(self):
        if self.window is not None and not self.run:
            raise ValueError('a window is for the run job: give it with run')
        return self


class RevisedTitle(BaseModel):
    id: str
    reverted: List[str] = Field(description='Catalogue files whose unpublished changes were put back as git has them.')
    extract_invalidated: bool


class ReviseFailure(BaseModel):
    id: str
    reason: str


class ReviseOutcome(BaseModel):
    revised: List[RevisedTitle] = Field(description='Sent back now (none when a job does it).')
    failed: List[ReviseFailure]
    job: Optional['Job'] = Field(None, description='The one-off redesign job doing it, when run was asked for.')
    execution: Optional['RunExecution'] = Field(None, description='Its first execution.')


class AudioStream(BaseModel):
    index: int = Field(description='Zero-based: what a stream choice sends.')
    text: str = Field(description='As a person picks it: "2: DTS-HD MA 5.1, English, 48 kHz".')
    preferred: bool = Field(description='The one extracted when nobody has chosen.')


class PublishedBeq(BaseModel):
    index: int = Field(description='Its place in the catalogue\'s list for the title: what its chart is asked by.')
    author: str
    audio_types: List[str]
    edition: str
    season: str
    episodes: str
    same_track: bool = Field(description='For the same audio, edition and episodes as this title, as far as is known.')
    catalogue_url: Optional[str] = Field(None, description='Its page in the catalogue, when the catalogue says.')
    filter_count: int


class PublishedBeqs(BaseModel):
    key: Optional[str] = Field(None, description="The catalogue's key for the title; none when it has none.")
    matched_by: Optional[Literal['tmdb', 'title']] = None
    compare_url: Optional[str] = Field(None, description="The catalogue's page comparing every author's BEQ of it.")
    entries: List[PublishedBeq]
    message: str = ''


class ProjectView(BaseModel):
    kind: Literal['mono', 'multichannel']
    name: str = Field(description='The file name, in the title\'s folder of the work directory.')
    exists: bool = Field(description='It is written when the title is extracted.')
    edited: bool = Field(description='Its filter is no longer what the pipeline wrote: what is saved in it is what gets '
                                     'published, and a redesign keeps it.')
    flat: bool = Field(description='It has no filter (not designed yet, or the designer declined).')
    error: str = Field(description='Why it could not be read; empty when it was.')


class TitleProjects(BaseModel):
    projects: List[ProjectView]
    summary: str = Field(description='In a line, as the desktop title page\'s badge says it.')
    warn: bool = Field(description='The summary is a warning (a project edited, or unreadable).')
    note: str = Field(description='What an edited project means for the title.')


class StreamChoice(Input):
    audio_stream: int = Field(ge=0, description='Zero-based, as TitleStreams lists them.')


class TitleStreams(BaseModel):
    chosen: int = Field(description='The stream extracted now (zero-based).')
    source: str = Field(description='Who chose it: `manual` (a person), `source` (the library), or empty (preferred).')
    probed: bool = Field(False, description='The source listed none: read from the file now, and kept when one is chosen.')
    streams: List[AudioStream]


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
    per_hour: Optional[float] = Field(None, description='Title-stages finished per hour so far (a running job, once one '
                                                        'is done).')
    remaining_seconds: Optional[float] = Field(None, description='At that rate, how long the rest will take.')
    estimated_finish: Optional[datetime] = Field(None, description='At that rate, when it will end.')


def with_rate(progress: Optional['Progress'], started_at: Optional[float], now: float) -> Optional['Progress']:
    ''' A running job's progress with its rate and the time the rest will take at that rate, once a title-stage is done. '''
    if progress is None or started_at is None or progress.done <= 0 or now <= started_at:
        return progress
    elapsed = now - started_at
    remaining = elapsed / progress.done * max(progress.total - progress.done, 0)
    return progress.model_copy(update={'per_hour': round(progress.done / elapsed * 3600, 2),
                                       'remaining_seconds': round(remaining, 1),
                                       'estimated_finish': timestamp(now + remaining)})


class StageFigures(BaseModel):
    count: int = Field(description='How many titles the figures are over.')
    mean: float = Field(description='Seconds.')
    p50: float
    p90: float
    max: float


class ReasonCount(BaseModel):
    reason: str
    count: int


class DesignerFiguresView(BaseModel):
    endpoint: str
    release: str = Field(description="The release its /health named, when the design was sent; empty if it named none.")
    version: str = Field(description='The design version it named.')
    designed: int
    declined: int
    failed: int
    seconds: Optional[StageFigures] = Field(None, description='Each design at the designer.')


class ExecutionTitle(BaseModel):
    id: str
    title: str
    outcome: Literal['designed', 'declined', 'extracted', 'already_done', 'failed', 'unavailable', 'skipped',
                     'not_reached']
    reason: str = Field('', description='Why it failed, was unavailable, or was skipped.')
    extract_seconds: Optional[float] = None
    waiting_seconds: Optional[float] = Field(None, description='From waiting for a design slot to the designer being sent it.')
    designer_seconds: Optional[float] = Field(None, description='At the designer.')
    writing_seconds: Optional[float] = Field(None, description="From the designer's answer to the title's end in the run.")
    endpoint: str = ''
    release: str = ''
    version: str = ''


class ExecutionTitles(BaseModel):
    titles: List[ExecutionTitle]


class ExecutionStatistics(BaseModel):
    ''' How an execution ran (design/jobs-and-executions.md §2.3); a running one's grow as titles finish. '''
    outcomes: Dict[str, int] = Field(description='Titles by outcome: designed, declined, extracted, already_done, failed, '
                                                 'unavailable, skipped, not_reached (only those with any).')
    failures: List[ReasonCount] = Field(description='Why titles failed or were unavailable, grouped, most first.')
    skips: List[ReasonCount] = Field(description='Why titles were skipped, grouped, most first.')
    designers: List[DesignerFiguresView] = Field(description='Each endpoint that designed for it, with the release and '
                                                             'design version it named.')
    designers_differ: bool = Field(description='Its designs came from more than one release or design version.')
    wall_seconds: Optional[float] = None
    per_hour: Optional[float] = Field(None, description='Titles reached an hour.')
    stages: Dict[str, StageFigures] = Field(description='extract, waiting (for a design slot), designer (the design '
                                                        'itself) and writing (its result): seconds a title.')
    slowest: List[ExecutionTitle] = Field(description='The titles in hand longest.')


def _figures(value) -> Optional[StageFigures]:
    return StageFigures(**_data(value)) if value is not None else None


def statistics_model(execution) -> ExecutionStatistics:
    from pipeline.service.statistics import statistics
    found = statistics(execution)
    return ExecutionStatistics(
        outcomes=found.outcomes, failures=[ReasonCount(reason=r, count=n) for r, n in found.failures],
        skips=[ReasonCount(reason=r, count=n) for r, n in found.skips],
        designers=[DesignerFiguresView(endpoint=d.endpoint, release=d.release, version=d.version, designed=d.designed,
                                       declined=d.declined, failed=d.failed, seconds=_figures(d.seconds))
                   for d in found.designers],
        designers_differ=found.designers_differ, wall_seconds=found.wall_seconds, per_hour=found.per_hour,
        stages={k: _figures(v) for k, v in found.stages.items()},
        slowest=[ExecutionTitle(**_data(t)) for t in found.slowest])


class ExecutionBase(BaseModel):
    id: str
    job_id: Optional[str] = Field(None, description='The job it is an execution of.')
    job_name: Optional[str] = Field(None, description="That job's name, while the job is kept.")
    number: int = Field(0, description="Which of its job's executions it is: 1, 2, ...")
    origin: JobOrigin
    state: ExecutionState
    stopped_because: Optional[StoppedBecause] = Field(None, description='stopped: its window closed, or its job was '
                                                                        'paused or removed. Its job carries on with the '
                                                                        'titles it had not started.')
    submitted_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    progress: Optional[Progress]
    error: Optional[str] = Field(None, description='Why it failed (a title that failed is in the result).')
    joined_to: Optional[str] = Field(None, description='The execution this one joined: its titles went into that run, '
                                                       'and it ends when that run does, with its result.')
    window: Optional[List[str]] = Field(None, description="Its job's window (null: the default's; []: any time).")
    window_open: Optional[bool] = Field(None, description='Whether it may run now, with the default applied.')
    waiting_until: Optional[datetime] = Field(None, description='A queued execution whose window is closed: when it '
                                                                'next opens.')
    statistics: Optional[ExecutionStatistics] = Field(None, description='How it ran: on GET /v1/executions/{id} only.')


class ScanExecution(ExecutionBase):
    kind: Literal['scan'] = 'scan'
    request: ScanJobRequest
    result: Optional[ScanResult]


class RunExecution(ExecutionBase):
    kind: Literal['run'] = 'run'
    request: RunJobRequest
    result: Optional[RunResult]


class AcceptExecution(ExecutionBase):
    kind: Literal['accept'] = 'accept'
    request: AcceptJobRequest
    result: Optional[AcceptResult]


Execution = Annotated[Union[ScanExecution, RunExecution, AcceptExecution], Field(discriminator='kind')]


class NextExecution(BaseModel):
    job_id: str
    job_name: str
    at: Optional[datetime] = Field(description='When it may start: null when it waits only for the one running.')


class ExecutionList(BaseModel):
    running: Optional[Execution] = Field(None, description='The execution running now.')
    next: List[NextExecution] = Field(default_factory=list, description="Each active job's next execution, soonest first.")
    executions: List[Execution] = Field(description='Every execution kept, newest first (the running one included).')


# --- jobs (design/jobs-and-executions.md) -------------------------------------------------------------------------------

EVERY_DESCRIPTION = 'Minutes from the end of one execution to the start of the next (at least 5); null: once.'


class _JobCreateBase(Input):
    name: str = Field('', max_length=200, description='What a person calls it; empty: what it does.')
    every: Optional[int] = Field(None, ge=5, description=EVERY_DESCRIPTION + ' Only a run through extract or design '
                                                                             'recurs.')
    window: Optional[Ranges] = Field(None, description=WINDOW_DESCRIPTION)


class ScanJobCreate(_JobCreateBase):
    kind: Literal['scan']
    request: ScanJobRequest = Field(default_factory=ScanJobRequest)


class RunJobCreate(_JobCreateBase):
    kind: Literal['run']
    request: RunJobRequest = Field(default_factory=RunJobRequest)


class AcceptJobCreate(_JobCreateBase):
    kind: Literal['accept']
    request: AcceptJobRequest = Field(default_factory=AcceptJobRequest)


JobCreate = Annotated[Union[ScanJobCreate, RunJobCreate, AcceptJobCreate], Field(discriminator='kind')]


class JobUpdate(Input):
    ''' What changes: only the fields sent. A one-off's request is fixed; `every` and `request` are a recurring job's. '''
    name: Optional[str] = Field(None, max_length=200)
    window: Optional[Ranges] = Field(None, description=WINDOW_DESCRIPTION + ' Send null for the default.')
    every: Optional[int] = Field(None, ge=5, description=EVERY_DESCRIPTION)
    request: Optional[RunJobRequest] = Field(None, description="A recurring job's run, from its next execution.")


class JobProgress(BaseModel):
    done: int = Field(description='Title-stages done by its executions so far.')
    total: Optional[int] = Field(None, description='How many its first execution set out to do.')


class JobBase(BaseModel):
    id: str
    name: str
    every: Optional[int] = Field(None, description=EVERY_DESCRIPTION)
    window: Optional[List[str]] = Field(None, description="Its window as given (null: the default's; []: any time).")
    state: JobState
    created_at: datetime
    next_at: Optional[datetime] = Field(None, description='When its next execution may start; null while paused, done, '
                                                          'removed or running.')
    progress: Optional[JobProgress] = Field(None, description='A one-off: over every execution so far.')
    remaining: Optional[int] = Field(None, description='A one-off run that stopped: titles its next execution does.')
    executions: int = Field(description='How many it has had.')
    last_execution: Optional[str] = Field(None, description='The id of its latest execution.')
    running_execution: Optional[Execution] = Field(None, description='Its execution queued or running.')
    last_finished_at: Optional[datetime] = None
    last_skip: Optional[str] = Field(None, description='A recurring job: why its last due tick did not start one (busy, '
                                                       'or the designer did not answer).')
    ended: Optional[str] = Field(None, description='Why it is done, when it ended by itself (a recurring job whose ids '
                                                   'are all done).')
    outcomes: Optional[Dict[str, int]] = Field(None, description='Its titles by outcome, summed over the executions '
                                                                 'kept: on GET /v1/jobs/{id} only.')


class ScanJob(JobBase):
    kind: Literal['scan'] = 'scan'
    request: ScanJobRequest


class RunJob(JobBase):
    kind: Literal['run'] = 'run'
    request: RunJobRequest


class AcceptJob(JobBase):
    kind: Literal['accept'] = 'accept'
    request: AcceptJobRequest


Job = Annotated[Union[ScanJob, RunJob, AcceptJob], Field(discriminator='kind')]


class JobList(BaseModel):
    jobs: List[Job]


def _request_data(request: Any) -> Dict[str, Any]:
    return _data(request) or {}


ReviseOutcome.model_rebuild()


# --- redesign scope (contract 1.6, design/redesign-scope.md) ------------------------------------------------------------

class ImpactAsk(Input):
    ''' Ask the designer which designs a redesign now could move. '''
    ids: List[str] = Field([], max_length=5000, description='The titles to ask about; empty: every title whose design is '
                                                           'out of date.')


class ImpactReasonView(BaseModel):
    change: Optional[str] = Field(description="The change it names (an id in the check's changes), or null.")
    feature: Optional[str] = Field(description="The part of the design it reached, in the designer's words.")
    detail: str = Field(description='For a person.')


class ImpactTitle(BaseModel):
    id: str
    title: str
    year: str
    tier: Literal['will_move', 'likely', 'possible', 'unaffected', 'unknown'] = Field(
        description='unknown is read as will_move unless a person decides otherwise.')
    score: Optional[float] = Field(description='The estimated probability that a redesign now moves it, so a sum is the '
                                               'expected number that move; orders titles within a tier, highest first; '
                                               'null for unknown.')
    from_design_version: Optional[int]
    reasons: List[ImpactReasonView]
    unknown_reason: Optional[str] = Field(None, description="Unknown only: why, as the designer's code (not_found, "
                                                            "unknown_build, ...), or no_record when the design names no "
                                                            "run record.")


class ImpactBasisView(BaseModel):
    ''' What a change's scores were measured on, for one outcome class. '''
    cls: str = Field(alias='class', description="The designer's name for the outcome class.")
    measured: int
    moved: int

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)


class ImpactChangeView(BaseModel):
    design_version: int
    id: str
    summary: str
    moved: Optional[str] = Field(None, description='What the designer counts as moved, for display.')
    basis: List[ImpactBasisView] = []


class ImpactCheck(BaseModel):
    ''' The last check of what a redesign now would move, or the one running. '''
    state: Literal['idle', 'running', 'succeeded', 'failed']
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    done: int = Field(0, description='Titles answered so far.')
    total: int = Field(0, description='Titles asked about.')
    error: str = ''
    designer_url: str = Field('', description='The designer endpoint asked.')
    design_version: Optional[int] = Field(None, description='The identity the answer is for: a check made at another '
                                                            'identity than the designers report now is stale.')
    design_settings: Optional[str] = None
    changes: List[ImpactChangeView] = []
    titles: List[ImpactTitle] = Field([], description='As a person reads them: will move and unknown first, then likely, '
                                                      'possible, unaffected; highest score first within each.')


class ReviewedAsk(Input):
    ''' Keep these titles' designs at the identity a redesign now would use, rather than redesign them. '''
    ids: List[str] = Field(min_length=1, max_length=MAX_REDESIGN_IDS)


class ReviewedTitle(BaseModel):
    id: str
    reviewed: bool
    reason: str = Field('', description='Why it was not kept, when it was not.')


class ReviewedOutcome(BaseModel):
    design_version: int
    design_settings: Optional[str]
    titles: List[ReviewedTitle]


def _request_model(kind: str, request: Any) -> Union[ScanJobRequest, RunJobRequest, AcceptJobRequest]:
    data = _request_data(request)
    if kind == 'scan':
        return ScanJobRequest(sources=list(data.get('sources') or ()), allow_empty=bool(data.get('allow_empty')))
    selection = TitleFilter.of(data.get('selection') or {})
    if kind == 'run':
        return RunJobRequest(filter=selection, through=data.get('through', 'design'),
                             scan_first=bool(data.get('scan_first', True)), retry_failed=bool(data.get('retry_failed')),
                             redesign=bool(data.get('redesign')), revise_to=data.get('revise_to') or 'design',
                             audio_stream=data.get('audio_stream'))
    return AcceptJobRequest(filter=selection, threshold=data.get('threshold', DEFAULT_ACCEPT_THRESHOLD),
                            dry_run=bool(data.get('dry_run')))


def execution_model(execution, now: Optional[float] = None, queue=None, job_name: Optional[str] = None,
                    with_statistics: bool = False) -> Union[ScanExecution, RunExecution, AcceptExecution]:
    '''
    An executions.Execution (live, or read back from the history) as its typed model.
    :param now: the time, for a running execution's rate and estimate (R9); default the clock.
    :param queue: the ExecutionQueue, which says whether its window is open now (else that is left null).
    :param job_name: its job's name, while the job is kept.
    :param with_statistics: say how it ran (design/jobs-and-executions.md §2.3).
    '''
    import time
    progress = Progress(**_data(execution.progress)) if execution.progress else None
    if execution.state == 'running':
        progress = with_rate(progress, execution.started_at, time.time() if now is None else now)
    common = dict(id=execution.id, job_id=execution.job_id, job_name=job_name, number=execution.number,
                  origin=execution.origin, state=execution.state, stopped_because=execution.stopped_because or None,
                  submitted_at=timestamp(execution.submitted_at), started_at=timestamp(execution.started_at),
                  finished_at=timestamp(execution.finished_at), progress=progress, error=execution.error or None,
                  joined_to=execution.joined_to, window=execution.window,
                  statistics=statistics_model(execution) if with_statistics and execution.kind == 'run' else None)
    if queue is not None and not execution.finished:
        common.update(window_open=queue.window_of(execution).is_open(time.time() if now is None else now),
                      waiting_until=timestamp(queue.waiting_until(execution)))
    request = _request_model(execution.kind, execution.request)
    result = execution.result
    if execution.kind == 'scan':
        return ScanExecution(request=request, result=scan_result(result) if result is not None else None, **common)
    if execution.kind == 'run':
        return RunExecution(request=request, result=run_result(result) if result is not None else None, **common)
    return AcceptExecution(request=request, result=accept_result(result) if result is not None else None, **common)


def _job_outcomes(job, queue) -> Dict[str, int]:
    ''' Its titles by outcome over the executions the history keeps (a title done in several counts in each). '''
    from collections import Counter
    from pipeline.service.statistics import statistics
    total: Counter = Counter()
    for execution_id in job.executions:
        if queue.has_job(execution_id):
            total.update(statistics(queue.get(execution_id)).outcomes)
    return dict(total)


def job_model(job, book, queue, now: Optional[float] = None, with_outcomes: bool = False) \
        -> Union[ScanJob, RunJob, AcceptJob]:
    ''' A jobbook.Job as its typed model, with its next and running execution (and, asked, its titles by outcome). '''
    running = None
    if job.pending is not None:
        try:
            running = execution_model(queue.get(job.pending), now, queue, job.name)
        except KeyError:
            running = None
    done, total = book.progress(job)
    common = dict(id=job.id, name=job.name, every=job.every, window=job.window, state=job.state,
                  created_at=timestamp(job.created_at), next_at=timestamp(book.next_at(job)),
                  progress=None if job.recurring else JobProgress(done=done, total=total),
                  remaining=len(job.remaining) if job.remaining is not None and not job.recurring else None,
                  executions=len(job.executions), last_execution=job.executions[-1] if job.executions else None,
                  running_execution=running, last_finished_at=timestamp(job.last_finished_at),
                  last_skip=job.last_skip or None, ended=job.ended or None, request=_request_model(job.kind, job.request),
                  outcomes=_job_outcomes(job, queue) if with_outcomes and job.kind == 'run' else None)
    return {'scan': ScanJob, 'run': RunJob, 'accept': AcceptJob}[job.kind](**common)


class _JobEventBase(BaseModel):
    seq: int = Field(description='Its place in the job: events come in this order, and `after` resumes after one.')
    at: datetime
    text: str = Field(description='What a person reads (redacted).')


class StateEvent(_JobEventBase):
    ''' The execution entered a state; the stream ends with a finished one. '''
    type: Literal['state']
    state: str = Field(description='queued, running, cancelling, stopping, succeeded, failed, cancelled, stopped or '
                                   'interrupted.')
    because: Optional[str] = Field(None, description='stopping: its window closed, or its job was paused or removed.')


class RunProgressEvent(_JobEventBase):
    ''' The run started a title-stage: `done` of `total` are finished. The last has no title: the run is over. '''
    type: Literal['run_progress']
    done: int
    total: int
    stage: Optional[str] = Field(None, description='extract, design, publish or commit; absent once the run is over.')
    title: Optional[str] = None
    title_id: Optional[str] = None


class ExtractProgressEvent(_JobEventBase):
    ''' How far ffmpeg has got extracting a title (one every 5 seconds or so, and the last). '''
    type: Literal['extract_progress']
    title: str
    title_id: str
    done_ms: int = Field(description='Audio extracted so far, in milliseconds.')
    total_ms: Optional[int] = Field(None, description="The audio's length, in milliseconds; absent if ffmpeg did not say.")
    percent: Optional[int] = Field(None, description='done_ms of total_ms, 0 to 100; absent without total_ms.')


class TitleEvent(_JobEventBase):
    ''' Something that happened to a title: a stage started or finished, a command ran, it failed. '''
    type: Literal['event']
    title_id: str
    stage: str
    kind: str = Field(description='stage_started, command_finished, ...')


class WindowEvent(_JobEventBase):
    ''' The job's window was changed (design/job-windows.md). '''
    type: Literal['window']
    window: Optional[List[str]] = Field(None, description='The new window: absent for the default, [] for any time.')


class JobEvent(RootModel[Annotated[Union[StateEvent, RunProgressEvent, ExtractProgressEvent, TitleEvent, WindowEvent],
                                   Field(discriminator='type')]]):
    ''' One thing a job reported, in order (`seq`). `type` says which fields it has, and every one has `text`. '''

    @classmethod
    def of(cls, event: Dict[str, Any]) -> 'JobEvent':
        ''' From the job manager's kept event, which has the same fields but the time as epoch seconds and no text yet. '''
        data = {k: v for k, v in event.items() if v is not None}
        model = cls.model_validate({**data, 'at': timestamp(event['at']), 'text': event.get('text') or ''})
        if not model.root.text:
            model.root.text = _event_text(model.root.model_dump())
        return model


def _clock(ms: int) -> str:
    seconds = max(ms // 1000, 0)
    return f'{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}'


def _event_text(event: Dict[str, Any]) -> str:
    ''' The words for an event that carries only numbers: a state, a step of the run, ffmpeg's position. '''
    kind = event.get('type')
    if kind == 'state':
        because = {'window': 'its window closed', 'paused': 'its job was paused',
                   'removed': 'its job was removed'}.get(event.get('because') or '')
        if because:
            return f"{event['state']}: {because}; it stops before its next title"
        return f"execution {event['state']}"
    if kind == 'run_progress':
        doing = f"{event['stage']}: {event.get('title')}" if event.get('stage') else 'finished'
        return f"{event['done']} of {event['total']} done; {doing}"
    if kind == 'extract_progress':
        if event.get('total_ms'):
            return f"{event['title']}: extracted {_clock(event['done_ms'])} of {_clock(event['total_ms'])} ({event['percent']}%)"
        return f"{event['title']}: extracted {_clock(event['done_ms'])}"
    if kind == 'window':
        window = event.get('window')
        return 'window: the default' if window is None else f"window: {', '.join(window)}" if window else \
            'window: any time'
    return ''


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
    running_execution: Optional[Execution] = Field(None, description='The execution running now.')
    waiting: int = Field(0, description='Executions queued behind it, or waiting for their window.')
    notify: List['NotifyOutcome'] = Field(default_factory=list)
    designer: Optional['DesignerStatus'] = None
    tmdb: Optional[bool] = Field(None, description='Whether a TMDB key is set (TMDB_API_KEY): without one, titles are '
                                                   "designed with the library's own metadata only.")
    repository_writes: bool = Field(False, description='Whether publish, commit and bulk accept are allowed '
                                                       '(allow_repository_writes in the service config).')
    repositories_configured: bool = Field(False, description='Whether the profile names the filter repository that publish '
                                                             'and commit write to.')


class EndpointDesign(BaseModel):
    title_id: str = Field(description='The title the design is for.')
    title: str = Field(description='Its name, as the run knew it; empty when it had none.')
    started_at: datetime = Field(description='When the design was sent.')


class DesignerEndpointStatus(BaseModel):
    url: str
    parallel: int = Field(description='How many designs it is sent at once.')
    reachable: bool = Field(description='Whether its /health answered (and it can take what it is declared to).')
    detail: str = Field(description='Its URL when it answered, else why not.')
    checked_at: Optional[datetime] = Field(None, description='When its /health was last asked.')
    designing: List[EndpointDesign] = Field(
        default_factory=list, description='What a run has out on it now, oldest first; empty when nothing is.')
    design_version: Optional[int] = Field(None, description="The design version its /health reports (contract 1.4); "
                                                            "null when it reports none or did not answer.")
    design_settings_detail: Optional[Dict[str, Any]] = Field(
        None, description='Its settings that differ from the defaults, as its /health lists them; null when it does not.')
    designer_build: Optional[str] = Field(None, description='The build its /health names, for people.')
    designer_release: Optional[str] = Field(
        None, description='The release it is (e.g. 0.2.10), as its /health names it or its build starts with; null when '
                          'neither says.')


class DesignerStatus(BaseModel):
    name: str
    reachable: Optional[bool] = Field(description="Whether any of its endpoints' /health answered; null when there is "
                                                  "nothing to ask (the manual designer) or the profile cannot be read.")
    detail: str = Field(description='The URLs that answered (and how many did not), else why none did.')
    checked_at: Optional[datetime]
    endpoints: List[DesignerEndpointStatus] = Field(default_factory=list,
                                                    description='Each endpoint the designer is served from (§7.2).')
    design_version: Optional[int] = Field(
        None, description='What a redesign now would be made with: the highest design version among the endpoints that '
                          'are up. A title designed with another (or with other settings) is out of date. Null when none '
                          'reports one, and then no title can be judged out of date.')
    design_disagreement: List[str] = Field(
        default_factory=list, description='The endpoints that are up but would design differently, each in words; empty '
                                          'when they agree. A redesign then answers differently by endpoint.')
    adjustment_kinds: Dict[str, Any] = Field(
        default_factory=dict, description="The adjustments (contract 1.5) every endpoint up lists on its /health, by kind: "
                                          "{description, params: {name: {type, unit, minimum, maximum, default, "
                                          "description}}}. Empty when it takes none (a designer before 1.5).")


class ScheduleUpdate(Input):
    ''' An older service's schedule (schedule.json, or service.yaml's `schedule`): migrated to a recurring job. '''
    enabled: bool = False
    interval_minutes: int = Field(60, ge=5)
    filter: TitleFilter = Field(default_factory=TitleFilter)
    through: AutoThrough = AutoThrough.design
    retry_failed: bool = False


class NotifyEvent(str, Enum):
    review_waiting = 'review_waiting'
    failed = 'failed'
    job_finished = 'job_finished'


class NotifyTest(Input):
    target: str


class NotifyJob(BaseModel):
    ''' The execution a notification is about, and its job. '''
    id: str
    job_id: Optional[str] = None
    job_name: Optional[str] = None
    kind: JobKind
    origin: JobOrigin
    state: ExecutionState
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
    job: Optional[NotifyJob] = Field(description='The execution it is about; null for a scheduled run that was skipped '
                                                 '(the designer did not answer), which `failed` names instead.')
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
    required: bool = Field(True, description='False: reported, but not a reason for `ready` to be false '
                                             '(a designer that is down does not make the service unready).')


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
