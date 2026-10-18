'''
One designer served by several HTTP endpoints, each taking up to its own number of designs at once, and each used only
while it is up (design/designer-interface.md §7.2).

A profile declares the endpoints under the designer's name:

    designers:
      beqforge:
        timeout: 900
        endpoints:
          - {url: http://10.151.167.10:8420/design, by_reference: true}
          - {url: http://10.151.167.20:8420/design, parallel: 2}

The pool is registered under that one name, so a design's fingerprint (which names the designer) does not depend on which
endpoint answered: the endpoints must run the same designer build for that to be true. A design goes to the endpoint with
the most room; an endpoint that does not answer (its `/health`, or a request that cannot connect) is left out until a
later check finds it up again, and the design is sent to another. Only when no endpoint is up does the call fail with
DesignerUnavailable, which a run counts towards `stop_after_unavailable` as before.

How many designs an endpoint is sent at once is its `parallel` in the profile, or else what its /health reports
(`parallel`, beqforge 0.2.4 on), or else 1. It is read again at every check, so a designer restarted with more workers is
used to the full within a minute, and `capacity` (what a run keeps in hand) counts only the endpoints that are up.
'''
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import requests

from pipeline.designer.contract import DesignRequest, DesignResponse
from pipeline.designer.http_binding import DesignerUnavailable, reported_parallel

logger = logging.getLogger('pipeline.designer.pool')

# how long an answer from an endpoint's /health is trusted, and how long one that did not answer is left alone
RECHECK_SECONDS = 60.0


@dataclass
class PoolEndpoint:
    url: str
    call: Callable[..., DesignResponse]   # the endpoint's http_designer(); also takes sources= if takes_sources
    health: Callable[[], Optional[dict]]  # its /health answer; raises DesignerUnavailable when it cannot be used
    parallel: Optional[int] = None        # from the profile; None: what /health reports
    takes_sources: bool = False
    reported: Optional[int] = None        # `parallel` from its last /health
    in_use: int = 0
    up: bool = False
    checked_at: Optional[float] = None
    checking: bool = False
    reason: str = 'not checked yet'

    @property
    def limit(self) -> int:
        ''' How many designs it is sent at once: the profile's `parallel`, else its /health's, else 1. '''
        return self.parallel or self.reported or 1


@dataclass(frozen=True)
class EndpointState:
    url: str
    parallel: int
    in_use: int
    up: Optional[bool]   # None: not asked yet
    detail: str


def cannot_reach(error: BaseException) -> bool:
    '''
    True if `error` says the endpoint could not be reached -- not that it refused or failed this design. Only these move a
    design to another endpoint: a 500 or a design that outlives its timeout is the title's, and sending it elsewhere would
    only repeat it (and leave every endpoint marked down by one bad title).
    '''
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, DesignerUnavailable):
            return True
        if isinstance(error, (requests.ConnectionError, requests.ConnectTimeout)) \
                and not isinstance(error, requests.ReadTimeout):
            return True
        if isinstance(error, requests.HTTPError):
            return getattr(error.response, 'status_code', None) in (502, 503, 504)
        error = error.__cause__ or error.__context__
    return False


@dataclass
class DesignerPool:
    name: str
    endpoints: Sequence[PoolEndpoint]
    clock: Callable[[], float] = time.monotonic
    recheck_seconds: float = RECHECK_SECONDS
    _condition: threading.Condition = field(default_factory=threading.Condition, init=False, repr=False)

    @property
    def capacity(self) -> int:
        '''
        How many designs the pool takes at once now: the endpoints that are up (or not asked yet), each at its limit. A
        run keeps this many in hand. At least 1, so that with none up a design is still tried and found unavailable.
        '''
        with self._condition:
            return max(1, sum(e.limit for e in self.endpoints if e.up or e.checked_at is None))

    @property
    def takes_sources(self) -> bool:
        return any(e.takes_sources for e in self.endpoints)

    def states(self) -> List[EndpointState]:
        with self._condition:
            return [EndpointState(e.url, e.limit, e.in_use, None if e.checked_at is None else e.up,
                                  e.url if e.up else e.reason) for e in self.endpoints]

    def __call__(self, request: DesignRequest, sources=None) -> DesignResponse:
        tried: set = set()
        while True:
            endpoint = self._acquire(tried)
            try:
                logger.info('%s: designing on %s', self.name, endpoint.url)
                return endpoint.call(request, sources) if endpoint.takes_sources else endpoint.call(request)
            except Exception as error:
                if not cannot_reach(error):
                    raise
                self._down(endpoint, str(error))
                tried.add(endpoint.url)
                logger.warning('%s: %s could not be reached (%s); trying another endpoint', self.name, endpoint.url, error)
                if len(tried) == len(self.endpoints):
                    raise
            finally:
                self._release(endpoint)

    def _acquire(self, tried: set) -> PoolEndpoint:
        '''
        A slot on an endpoint that is up, waiting for one to come free while every live endpoint is busy.
        :raises DesignerUnavailable: if no endpoint not already tried for this design is up.
        '''
        with self._condition:
            while True:
                candidates = [e for e in self.endpoints if e.url not in tried]
                due = [e for e in candidates if not e.checking and
                       (e.checked_at is None or self.clock() - e.checked_at >= self.recheck_seconds)]
                if due:
                    for e in due:
                        e.checking = True
                    self._condition.release()
                    try:
                        results = [(e, self._ask(e)) for e in due]
                    finally:
                        self._condition.acquire()
                    for e, (reason, body) in results:
                        e.checking, e.checked_at, e.up, e.reason = False, self.clock(), not reason, reason or ''
                        if not reason:
                            e.reported = reported_parallel(body)
                    self._condition.notify_all()
                    continue
                up = [e for e in candidates if e.up]
                if not up and not any(e.checking for e in candidates):
                    reasons = '; '.join(f'{e.url}: {e.reason}' for e in candidates) or 'every endpoint was tried'
                    raise DesignerUnavailable(f'no endpoint of designer {self.name!r} is up ({reasons})')
                free = [e for e in up if e.in_use < e.limit]
                if free:
                    chosen = max(free, key=lambda e: (e.limit - e.in_use) / e.limit)
                    chosen.in_use += 1
                    return chosen
                # every live endpoint is busy (or one is being asked): a release, or the next recheck, frees one
                self._condition.wait(timeout=self.recheck_seconds)

    @staticmethod
    def _ask(endpoint: PoolEndpoint):
        ''' :return: (why it cannot be used, or ''; what its /health answered). '''
        try:
            return '', endpoint.health()
        except DesignerUnavailable as error:
            return str(error), None

    def _down(self, endpoint: PoolEndpoint, reason: str) -> None:
        with self._condition:
            endpoint.up, endpoint.checked_at, endpoint.reason = False, self.clock(), reason

    def _release(self, endpoint: PoolEndpoint) -> None:
        with self._condition:
            endpoint.in_use -= 1
            self._condition.notify_all()
