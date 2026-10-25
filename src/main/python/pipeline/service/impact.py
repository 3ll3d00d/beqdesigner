'''
Which out-of-date designs to redesign, over HTTP (contract 1.6; design/redesign-scope.md step 4).

`POST /v1/impact` asks the designer -- one that is up, answers /impact and reports the identity a redesign would use --
about the titles whose design is out of date (or those named), in the background: it reads records, not audio, but a
library takes a minute or so, so it is not a request's work, and it is not a job either (it would wait behind a run for
hours, and touches nothing). `GET /v1/impact` is the last check, or the one running, kept in `<state_dir>/impact.json`
across restarts. A title with no run record is not sent: it is `unknown`, as the designer says of a record it cannot read.

`POST /v1/reviewed` is the person's answer for the titles they keep: each design is stamped reviewed at the identity a
redesign now would use (`pipeline.library.reviewed`), and stops counting as out of date. Those they redesign go through
`/v1/revise` as any redesign does. Nothing here redesigns or keeps a title on the designer's answer alone.
'''
import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from pipeline.designer.identity import CurrentIdentity
from pipeline.designer.impact import ImpactChange, ImpactError, ImpactResult, ask_impact, impact_endpoint, no_record
from pipeline.library.reviewed import ReviewRefused, mark_reviewed
from pipeline.library.selection import Selection
from pipeline.service import models

logger = logging.getLogger('service_impact')

STATE_FILE = 'impact.json'


class CheckRunning(RuntimeError):
    ''' A check is already running: one at a time. '''


class ImpactChecker:
    '''
    One check at a time, in a thread of its own; the last one's answer kept.
    :param state_dir: where the last answer is kept; None keeps it in memory only.
    :param ask: asks the designer (`pipeline.designer.impact.ask_impact`).
    :param run: starts the work; a thread by default (a test runs it in place).
    '''

    def __init__(self, state_dir: Optional[str] = None, *, ask: Callable = ask_impact,
                 clock: Callable[[], float] = time.time, run: Optional[Callable[[Callable[[], None]], None]] = None):
        self.state_dir, self.ask, self.clock = state_dir, ask, clock
        self.run = run or (lambda work: threading.Thread(target=work, name='impact-check', daemon=True).start())
        self.lock = threading.Lock()
        self.state = self.__load()

    def __path(self) -> Optional[str]:
        return os.path.join(self.state_dir, STATE_FILE) if self.state_dir else None

    def __load(self) -> dict:
        path = self.__path()
        try:
            with open(path) as f:
                state = json.load(f)
        except (TypeError, OSError, ValueError):
            return {'state': 'idle'}
        if state.get('state') == 'running':   # the service stopped part way: nothing is still asking
            state.update(state='failed', error='the service restarted before the check finished',
                         finished_at=state.get('started_at'))
        return state

    def __save(self) -> None:
        path = self.__path()
        if not path:
            return
        try:
            os.makedirs(self.state_dir, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix='.impact-', suffix='.json', dir=self.state_dir)
            with os.fdopen(fd, 'w') as f:
                json.dump(self.state, f)
            os.replace(temporary, path)
        except OSError:
            logger.exception('the impact check could not be saved in %s', self.state_dir)

    def snapshot(self) -> dict:
        with self.lock:
            return json.loads(json.dumps(self.state))

    def start(self, url: str, headers: Optional[dict], current: CurrentIdentity, records: List[Tuple[str, str]],
              unrecorded: List[str], titles: Dict[str, Tuple[str, str]]) -> dict:
        '''
        Starts a check of `records` ((title id, run record) pairs) and `unrecorded` (ids with no run record).
        :param titles: each id's (title, year), for display.
        :raises CheckRunning: one is running.
        '''
        with self.lock:
            if self.state.get('state') == 'running':
                raise CheckRunning('a check of what a redesign would move is running already')
            self.state = {'state': 'running', 'started_at': self.clock(), 'done': 0,
                          'total': len(records) + len(unrecorded), 'designer_url': url,
                          'design_version': current.version, 'design_settings': current.settings}
            self.__save()
            started = dict(self.state)

        def progress(done: int, _: int) -> None:
            with self.lock:
                self.state['done'] = done

        def work() -> None:
            try:
                answer = self.ask(url, records, headers=headers, on_batch=progress) if records else None
            except ImpactError as error:
                self.__finish(state='failed', error=str(error))
                return
            except Exception as error:   # the check fails; the service does not
                logger.exception('the impact check failed')
                self.__finish(state='failed', error=f'{type(error).__name__}: {error}')
                return
            if answer is not None and (answer.design_version, answer.design_settings) != \
                    (current.version, current.settings):
                self.__finish(state='failed', error=f'{url} answered for design version {answer.design_version} '
                                                    f'(settings {answer.design_settings}), not the {current.describe()} '
                                                    f'a redesign would use')
                return
            results = list(answer.results if answer else ()) + [no_record(i) for i in unrecorded]
            ranked = sorted(results, key=lambda r: (r.rank, -(r.score or 0.0), r.id))
            self.__finish(state='succeeded', done=len(results),
                          changes=[_change(c) for c in (answer.changes if answer else ())],
                          titles=[_title(r, titles.get(r.id, ('', ''))) for r in ranked])
        self.run(work)
        return started

    def __finish(self, **fields) -> None:
        with self.lock:
            self.state.update(fields, finished_at=self.clock())
            self.__save()


def _change(change: ImpactChange) -> dict:
    return {'design_version': change.design_version, 'id': change.id, 'summary': change.summary,
            'moved': change.moved,
            'basis': [{'class': b.cls, 'measured': b.measured, 'moved': b.moved} for b in change.basis]}


def _title(result: ImpactResult, named: Tuple[str, str]) -> dict:
    return {'id': result.id, 'title': named[0], 'year': named[1], 'tier': result.tier, 'score': result.score,
            'from_design_version': result.from_design_version,
            'reasons': [{'change': x.change, 'feature': x.feature, 'detail': x.detail} for x in result.reasons],
            'unknown_reason': result.unknown_reason}


def check_model(state: dict) -> models.ImpactCheck:
    return models.ImpactCheck(
        state=state.get('state', 'idle'), started_at=models.timestamp(state.get('started_at')),
        finished_at=models.timestamp(state.get('finished_at')), done=state.get('done', 0), total=state.get('total', 0),
        error=state.get('error', ''), designer_url=state.get('designer_url', ''),
        design_version=state.get('design_version'), design_settings=state.get('design_settings'),
        changes=[models.ImpactChangeView(**c) for c in state.get('changes') or []],
        titles=[models.ImpactTitle(**t) for t in state.get('titles') or []])


@dataclass(frozen=True)
class ImpactRoutes:
    ''' What the routes need of the app. '''
    context: Callable
    read_index: Callable
    designer: object          # the DesignerProbe
    checker: ImpactChecker
    in_flight: Callable
    refresh: Callable[[], None]
    problem: type             # ServiceProblem
    responses: Callable


def register(v1: APIRouter, deps: ImpactRoutes) -> None:
    Problem, responses = deps.problem, deps.responses

    def identity() -> Tuple[CurrentIdentity, object]:
        ''' What a redesign now would use, asked now, and the probe's answer. '''
        state = deps.designer.check()
        current = state.identity
        if current is None:
            raise Problem(409, 'Design version unknown', 'no designer that is up says its design version (contract '
                                                         '1.4), so no design can be judged out of date')
        if current.disagreement:
            raise Problem(409, 'Designers disagree', 'the designers that are up would design differently: '
                                                     + '; '.join(current.disagreement))
        return current, state

    @v1.get('/impact', response_model=models.ImpactCheck, tags=['redesign'],
            summary='The last check of what a redesign now would move (or the one running)')
    def get_impact() -> models.ImpactCheck:
        return check_model(deps.checker.snapshot())

    @v1.post('/impact', response_model=models.ImpactCheck, status_code=202, tags=['redesign'],
             summary="Ask the designer which out-of-date designs a redesign now could move (in the background; "
                     "GET /v1/impact follows it)", responses=responses(409, 422, 503))
    def post_impact(body: models.ImpactAsk) -> JSONResponse:
        current, state = identity()
        endpoint = impact_endpoint([e.identity for e in state.endpoints if e.reachable], current)
        if endpoint is None:
            raise Problem(409, 'No designer answers impact',
                          f'no designer that is up answers /impact (contract 1.6) at {current.describe()}')
        ctx = deps.context()
        with deps.read_index(ctx) as index:
            if body.ids:
                rows = index.titles(ids=list(dict.fromkeys(body.ids)))
                missing = sorted(set(body.ids) - {r.id for r in rows})
                if missing:
                    raise Problem(422, 'No such title', f"no title {', '.join(missing[:20])}")
                rows = [r for r in rows if r.design_identity is not None]
            else:
                rows = Selection(out_of_date=True, current=current).rows(index)
        if not rows:
            raise Problem(409, 'Nothing to check', 'no design is out of date' if not body.ids else
                          'none of those titles has a design')
        declared = ctx.designer_endpoints()
        headers = next((e.headers for e in (declared.endpoints if declared else ()) if e.url == endpoint.url), None)
        try:
            started = deps.checker.start(endpoint.url, headers, current,
                                         [(r.id, r.run_record) for r in rows if r.run_record],
                                         [r.id for r in rows if not r.run_record],
                                         {r.id: (r.title or r.display_name, r.year) for r in rows})
        except CheckRunning as error:
            raise Problem(409, 'Check running', str(error))
        return JSONResponse(check_model(started).model_dump(mode='json'), status_code=202)

    @v1.post('/reviewed', response_model=models.ReviewedOutcome, tags=['redesign'],
             summary="Keep titles' designs at the identity a redesign now would use, rather than redesign them",
             responses=responses(409, 503))
    def reviewed(body: models.ReviewedAsk) -> models.ReviewedOutcome:
        current, _ = identity()
        ctx = deps.context()
        settings = ctx.scan_settings()
        with deps.read_index(ctx) as index:
            rows = {r.id: r for r in index.titles(ids=list(dict.fromkeys(body.ids)))}
        out = []
        for title_id in dict.fromkeys(body.ids):
            row = rows.get(title_id)
            if row is None:
                out.append(models.ReviewedTitle(id=title_id, reviewed=False, reason='no such title'))
                continue
            if deps.in_flight(ctx, row):
                out.append(models.ReviewedTitle(id=title_id, reviewed=False,
                                                reason='a run is working on this title now'))
                continue
            try:
                mark_reviewed(settings.queue_dir, title_id, current)
            except ReviewRefused as error:
                out.append(models.ReviewedTitle(id=title_id, reviewed=False, reason=str(error)))
            except FileNotFoundError:
                out.append(models.ReviewedTitle(id=title_id, reviewed=False, reason='it has no queue entry'))
            except OSError as error:
                out.append(models.ReviewedTitle(id=title_id, reviewed=False,
                                                reason=f'not saved: {type(error).__name__}: {error}'))
            else:
                out.append(models.ReviewedTitle(id=title_id, reviewed=True))
        if any(t.reviewed for t in out):
            deps.refresh()
        return models.ReviewedOutcome(design_version=current.version, design_settings=current.settings, titles=out)
