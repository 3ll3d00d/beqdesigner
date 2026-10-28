'''
What an execution says about how it ran (design/jobs-and-executions.md §2.3), from its result and the record the queue
keeps of each title (`Execution.titles`): the titles by outcome, the designers that designed for it with their counts and
times, and each stage's times. Plain data in and out, so it is tested without a service.
'''
import math
from collections import Counter
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, Iterable, List, Optional

# what became of a title, in the order a person reads them
OUTCOMES = ('designed', 'declined', 'extracted', 'already_done', 'failed', 'unavailable', 'skipped', 'not_reached')
SLOWEST = 5


@dataclass(frozen=True)
class Figures:
    count: int
    mean: float
    p50: float
    p90: float
    max: float


@dataclass(frozen=True)
class TitleOutcome:
    id: str
    title: str
    outcome: str
    reason: str = ''
    extract_seconds: Optional[float] = None
    waiting_seconds: Optional[float] = None     # from waiting for a design slot to the designer being sent it
    designer_seconds: Optional[float] = None    # at the designer
    writing_seconds: Optional[float] = None     # from the designer's answer to the title's end in the run
    endpoint: str = ''
    release: str = ''
    version: str = ''


@dataclass(frozen=True)
class DesignerFigures:
    endpoint: str
    release: str
    version: str
    designed: int
    declined: int
    failed: int
    seconds: Optional[Figures]


@dataclass(frozen=True)
class Statistics:
    outcomes: Dict[str, int]
    failures: List[tuple]                       # (reason, count), most first
    skips: List[tuple]
    designers: List[DesignerFigures]
    designers_differ: bool                      # the designs were made by more than one release or design version
    wall_seconds: Optional[float]
    per_hour: Optional[float]                   # titles reached an hour
    stages: Dict[str, Figures] = field(default_factory=dict)   # extract, waiting, designer, writing
    slowest: List[TitleOutcome] = field(default_factory=list)


def figures(values: Iterable[float]) -> Optional[Figures]:
    ''' Mean, median, 90th percentile (nearest rank) and max; None for none. '''
    ordered = sorted(v for v in values if v is not None and v >= 0)
    if not ordered:
        return None

    def rank(p: float) -> float:
        return ordered[max(0, math.ceil(p * len(ordered)) - 1)]
    return Figures(len(ordered), round(sum(ordered) / len(ordered), 1), round(rank(0.5), 1), round(rank(0.9), 1),
                   round(ordered[-1], 1))


def _plain(result: Any) -> Dict[str, Any]:
    if is_dataclass(result) and not isinstance(result, type):
        result = asdict(result)
    return result if isinstance(result, dict) else {}


def _ids(pairs) -> Dict[str, str]:
    ''' (id, message) pairs, or {id, message} from the history, as {id: message}. '''
    found = {}
    for pair in pairs or ():
        if isinstance(pair, dict):
            found[str(pair.get('id', ''))] = str(pair.get('message') or pair.get('reason') or '')
        elif isinstance(pair, (list, tuple)) and pair:
            found[str(pair[0])] = str(pair[1]) if len(pair) > 1 else ''
    return found


def _skips(skipped) -> Dict[str, str]:
    found = {}
    for skip in skipped or ():
        if isinstance(skip, dict):
            found[str(skip.get('id', ''))] = str(skip.get('reason') or '')
        elif hasattr(skip, 'id'):
            found[skip.id] = getattr(skip, 'reason', '')
    return found


def _between(record: Dict[str, Any], start: str, end: str) -> Optional[float]:
    if record.get(start) is None or record.get(end) is None:
        return None
    return round(float(record[end]) - float(record[start]), 3)


def _reason(text: str) -> str:
    ''' A message as a reason to group by: its first line, trimmed. '''
    return (text or '').strip().splitlines()[0][:160] if (text or '').strip() else 'no reason given'


def _so_far(record: Dict[str, Any]) -> tuple:
    ''' A title's outcome from its events alone, while its run has no result yet: (outcome, reason). '''
    end = record.get('end')
    if end == 'failed':
        return 'failed', record.get('message', '')
    if end == 'skipped':
        return 'skipped', record.get('message', '')
    if end == 'title_completed':
        call = record.get('call')
        return ('declined' if call == 'declined' else 'designed' if call == 'designed' else 'already_done'), ''
    return 'not_reached', ''


def title_outcomes(result: Any, records: Dict[str, Dict[str, Any]]) -> List[TitleOutcome]:
    ''' Every title the run selected or reached, with what became of it and how long each part took. '''
    report = _plain(result).get('report') or {}
    run = report.get('run') or {}
    failed, unavailable = _ids(run.get('failed')), _ids(run.get('unavailable'))
    skipped = {**{k: f'failed before: {v}' for k, v in _ids(run.get('failed_earlier')).items()},
               **_skips(report.get('skipped'))}
    designed, design_cached = set(run.get('designed') or ()), set(run.get('design_cached') or ())
    extracted, cached = set(run.get('extracted') or ()), set(run.get('cached') or ())
    not_run = list(report.get('not_run') or ())
    ordered = list(dict.fromkeys([*records, *(report.get('attempted') or ()), *skipped, *not_run]))
    outcomes = []
    for title_id in ordered:
        record = records.get(title_id, {})
        reason = ''
        if title_id in failed:
            outcome, reason = 'failed', failed[title_id]
        elif title_id in unavailable:
            outcome, reason = 'unavailable', unavailable[title_id]
        elif title_id in designed:
            outcome = 'declined' if record.get('call') == 'declined' else 'designed'
        elif title_id in design_cached:
            outcome = 'already_done'
        elif title_id in extracted:
            outcome = 'extracted'
        elif title_id in cached:
            outcome = 'already_done'
        elif title_id in skipped:
            outcome, reason = 'skipped', skipped[title_id]
        elif not report:   # running: what its own events say so far
            outcome, reason = _so_far(record)
        else:
            outcome = 'not_reached'
        waiting_from = record.get('design_queued') or record.get('design_start')
        outcomes.append(TitleOutcome(
            title_id, record.get('title') or title_id, outcome, reason,
            extract_seconds=_between(record, 'extract_start', 'extract_end'),
            waiting_seconds=round(float(record['sent']) - float(waiting_from), 3)
            if waiting_from is not None and record.get('sent') is not None else None,
            designer_seconds=_between(record, 'sent', 'answered'),
            writing_seconds=_between(record, 'answered', 'ended'),
            endpoint=str(record.get('endpoint') or ''), release=str(record.get('release') or ''),
            version=str(record.get('version') or '')))
    return outcomes


def statistics(execution) -> Statistics:
    ''' How an execution ran, from what it has reported so far (a running one's figures grow as titles finish). '''
    titles = title_outcomes(execution.result, getattr(execution, 'titles', {}) or {})
    counts = Counter(t.outcome for t in titles)
    by_endpoint: Dict[tuple, List[TitleOutcome]] = {}
    for title in titles:
        if title.endpoint:
            by_endpoint.setdefault((title.endpoint, title.release, title.version), []).append(title)
    records = getattr(execution, 'titles', {}) or {}
    designers = []
    for (endpoint, release, version), designed in by_endpoint.items():
        calls = Counter(records.get(t.id, {}).get('call', '') for t in designed)
        designers.append(DesignerFigures(endpoint, release, version, calls['designed'], calls['declined'],
                                         calls['failed'], figures(t.designer_seconds for t in designed)))
    started, finished = execution.started_at, execution.finished_at
    ends = [float(r['ended']) for r in records.values() if r.get('ended') is not None]
    end = finished or (max(ends) if ends else None)
    wall = round(end - started, 1) if started is not None and end is not None and end > started else None
    reached = sum(1 for t in titles if t.outcome not in ('not_reached', 'skipped'))
    stages = {name: f for name, f in (
        ('extract', figures(t.extract_seconds for t in titles)),
        ('waiting', figures(t.waiting_seconds for t in titles)),
        ('designer', figures(t.designer_seconds for t in titles)),
        ('writing', figures(t.writing_seconds for t in titles))) if f is not None}

    def in_hand(t: TitleOutcome) -> float:
        return sum(v for v in (t.extract_seconds, t.waiting_seconds, t.designer_seconds, t.writing_seconds) if v)
    slowest = sorted((t for t in titles if in_hand(t) > 0), key=in_hand, reverse=True)[:SLOWEST]
    return Statistics(
        outcomes={o: counts[o] for o in OUTCOMES if counts[o]},
        failures=Counter(_reason(t.reason) for t in titles if t.outcome in ('failed', 'unavailable')).most_common(),
        skips=Counter(_reason(t.reason) for t in titles if t.outcome == 'skipped').most_common(),
        designers=sorted(designers, key=lambda d: (-(d.designed + d.declined + d.failed), d.endpoint)),
        designers_differ=len({(d.release, d.version) for d in designers}) > 1,
        wall_seconds=wall, per_hour=round(reached / wall * 3600, 1) if wall and reached else None,
        stages=stages, slowest=slowest)
