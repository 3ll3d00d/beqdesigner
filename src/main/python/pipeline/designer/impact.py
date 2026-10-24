'''
Asking a designer which designs a redesign now could move (contract 1.6, design/designer-interface.md §7.1 "Impact";
design/redesign-scope.md step 3). The designer reads its own run records -- each design's `run_record`, sent back as it
was given -- and answers, per record, a tier (will move, likely, possible, unaffected, unknown), a score that orders
records within a tier, and the reasons. The answer is advisory: a person chooses what to redesign and what to keep.

Only a designer whose `/health` says `impact: true` and the identity a redesign would use is asked (`impact_endpoint`),
in requests of at most 500 records, one after another (`ask_impact`). An answer out of the contract's shape is an
`ImpactError`, never a guess; a design with no run record is not sent and is `unknown` (`no_record`).
'''
import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlsplit, urlunsplit

import requests

from pipeline.designer.identity import CurrentIdentity, EndpointIdentity, valid_text, valid_version

IMPACT_VERSION = '1.6'
MAX_RECORDS = 500
IMPACT_TIMEOUT = 120.0   # beqforge: about 10-30 s for 500 records over the share
TIERS = ('will_move', 'likely', 'possible', 'unaffected', 'unknown')
# the order a person reads them in: unknown is treated as will_move unless a person decides otherwise (§7.1)
_RANK = {'will_move': 0, 'unknown': 0, 'likely': 1, 'possible': 2, 'unaffected': 3}


class ImpactError(RuntimeError):
    ''' The designer could not be asked, or answered outside the contract: nothing is shown as its answer. '''


@dataclass(frozen=True)
class ImpactReason:
    change: Optional[str]    # an id from the answer's changes, or None
    feature: Optional[str]   # the part of the design it reached, the designer's own name, for display
    detail: str              # for a person; never parsed


@dataclass(frozen=True)
class ImpactResult:
    id: str
    tier: str
    score: Optional[float]               # orders records within a tier, highest first; None for unknown
    from_design_version: Optional[int]
    reasons: Tuple[ImpactReason, ...] = ()

    @property
    def rank(self) -> int:
        ''' Where a person reads it: will move and unknown first, unaffected last. '''
        return _RANK[self.tier]


@dataclass(frozen=True)
class ImpactChange:
    design_version: int
    id: str
    summary: str


@dataclass(frozen=True)
class ImpactAnswer:
    ''' What one designer said of a set of records, its requests merged. '''
    design_version: int
    design_settings: Optional[str]
    designer_build: Optional[str]
    changes: Tuple[ImpactChange, ...]
    results: Tuple[ImpactResult, ...]

    def ranked(self) -> List[ImpactResult]:
        ''' The results as a person reads them: by tier, then highest score first, then by id. '''
        return sorted(self.results, key=lambda r: (r.rank, -(r.score or 0.0), r.id))


def no_record(record_id: str) -> ImpactResult:
    ''' The answer for a design with no run record: it cannot be asked about, so a person treats it as moving. '''
    return ImpactResult(record_id, 'unknown', None, None,
                        (ImpactReason(None, None, 'the designer named no run record for this design'),))


def impact_url(design_url: str) -> str:
    ''' `POST /impact` on the design endpoint's origin (§7.1). '''
    parts = urlsplit(design_url)
    return urlunsplit((parts.scheme, parts.netloc, '/impact', '', ''))


def impact_endpoint(endpoints: Iterable[Optional[EndpointIdentity]],
                    current: Optional[CurrentIdentity]) -> Optional[EndpointIdentity]:
    '''
    The endpoint to ask: one that is up (each of `endpoints` is), says it answers /impact, and reports the identity a
    redesign would use -- its version and settings; its build is never compared (§3). None when there is none, or the
    endpoints up disagree (a redesign could be made with either, so neither answer is the one that matters).
    '''
    if current is None or current.disagreement:
        return None
    for endpoint in endpoints:
        if endpoint is not None and endpoint.impact and \
                (endpoint.version, endpoint.settings) == (current.version, current.settings):
            return endpoint
    return None


def ask_impact(url: str, records: Sequence[Tuple[str, str]], *, timeout: float = IMPACT_TIMEOUT,
               headers: Optional[dict] = None, post: Callable[..., Any] = requests.post,
               on_batch: Optional[Callable[[int, int], None]] = None) -> ImpactAnswer:
    '''
    Asks the designer at `url` (its design endpoint; /impact is on the same origin) about `records`.
    :param records: (id, run record path) pairs, ids unique.
    :param on_batch: told (records done, records in all) after each request.
    :return: every request's answer merged: one result per record, in the order given.
    :raises ImpactError: the designer could not be asked, refused, or answered outside the contract.
    '''
    ids = [i for i, _ in records]
    if len(set(ids)) != len(ids):
        raise ImpactError('two records share an id')
    if not records:
        raise ImpactError('no records to ask about')
    target = impact_url(url)
    answers = []
    for start in range(0, len(records), MAX_RECORDS):
        batch = records[start:start + MAX_RECORDS]
        body = {'contract_version': IMPACT_VERSION, 'records': [{'id': i, 'path': p} for i, p in batch]}
        answers.append(_answer(_post(post, target, body, timeout, headers), [i for i, _ in batch]))
        if on_batch:
            on_batch(start + len(batch), len(records))
    first = answers[0]
    for later in answers[1:]:
        if (later.design_version, later.design_settings) != (first.design_version, first.design_settings):
            raise ImpactError(f'{target} answered as design version {later.design_version} after '
                              f'{first.design_version}: its identity changed between requests')
    changes = {}
    for answer in answers:
        for change in answer.changes:
            changes.setdefault((change.design_version, change.id), change)
    return ImpactAnswer(first.design_version, first.design_settings, first.designer_build,
                        tuple(sorted(changes.values(), key=lambda c: c.design_version)),
                        tuple(r for answer in answers for r in answer.results))


def _post(post, target: str, body: dict, timeout: float, headers: Optional[dict]) -> Any:
    try:
        response = post(target, json=body, headers=headers, timeout=timeout)
    except requests.RequestException as e:
        raise ImpactError(f'POST {target} failed: {e}') from e
    if not response.ok:
        detail = (response.text or '').strip()[:500]
        raise ImpactError(f'POST {target} answered HTTP {response.status_code}' + (f': {detail}' if detail else ''))
    try:
        return response.json()
    except ValueError as e:
        raise ImpactError(f'{target} did not answer JSON: {e}') from e


def _answer(body: Any, ids: List[str]) -> ImpactAnswer:
    ''' One response, checked against the contract (§7.1) and the request it answers. '''
    if not isinstance(body, Mapping):
        raise ImpactError('the impact answer is not an object')
    if body.get('contract_version') != IMPACT_VERSION:
        raise ImpactError(f"the impact answer says contract {body.get('contract_version')!r}, not {IMPACT_VERSION}")
    version = valid_version(body.get('design_version'))
    if version is None:
        raise ImpactError(f"the impact answer's design_version is {body.get('design_version')!r}")
    results = body.get('results')
    if not isinstance(results, list) or [r.get('id') if isinstance(r, Mapping) else None for r in results] != ids:
        raise ImpactError('the impact answer does not hold one result per record, in the order asked')
    changes = body.get('changes')
    if not isinstance(changes, list):
        raise ImpactError('the impact answer has no list of changes')
    return ImpactAnswer(version, valid_text(body.get('design_settings')), valid_text(body.get('designer_build')),
                        tuple(_change(c) for c in changes), tuple(_result(r) for r in results))


def _change(c: Any) -> ImpactChange:
    if not isinstance(c, Mapping) or valid_version(c.get('design_version')) is None or not valid_text(c.get('id')) \
            or not valid_text(c.get('summary')):
        raise ImpactError(f'an impact change is out of shape: {c!r}')
    return ImpactChange(c['design_version'], c['id'], c['summary'])


def _result(r: Mapping) -> ImpactResult:
    name = f"the impact result for {r.get('id')!r}"
    tier = r.get('tier')
    if tier not in TIERS:
        raise ImpactError(f'{name} has tier {tier!r}')
    score = r.get('score')
    if tier == 'unknown':
        if score is not None:
            raise ImpactError(f'{name} is unknown but scored')
    elif isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) \
            or not 0 <= score <= 1:
        raise ImpactError(f'{name} has score {score!r}, not a number in [0, 1]')
    from_version = r.get('from_design_version')
    if from_version is not None and valid_version(from_version) is None:
        raise ImpactError(f'{name} has from_design_version {from_version!r}')
    reasons = r.get('reasons')
    if not isinstance(reasons, list) or (tier != 'unaffected' and not reasons):
        raise ImpactError(f'{name} gives no reasons')
    return ImpactResult(r['id'], tier, None if score is None else float(score), from_version,
                        tuple(_reason(name, x) for x in reasons))


def _reason(name: str, x: Any) -> ImpactReason:
    if not isinstance(x, Mapping) or not valid_text(x.get('detail')):
        raise ImpactError(f'{name} has a reason out of shape: {x!r}')
    return ImpactReason(valid_text(x.get('change')), valid_text(x.get('feature')), x['detail'])
