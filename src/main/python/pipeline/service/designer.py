'''
Is the profile's designer up? The scheduler asks before each tick (an unanswered designer skips the tick rather than
starting a run whose every design would fail), and `/ready` and `/v1/status` report the last answer.
'''
import threading
import time
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Tuple

from pipeline.service.context import JobContext, load_context

MAX_AGE_SECONDS = 30.0   # how long /ready and /v1/status reuse an answer, so polling them does not poll the designer


@dataclass(frozen=True)
class EndpointState:
    url: str
    parallel: int
    reachable: bool
    detail: str   # the URL when it answered, else why not


@dataclass(frozen=True)
class DesignerState:
    name: str
    reachable: Optional[bool]    # None: there is nothing to ask (the manual designer) or the profile is unreadable
    detail: str
    checked_at: Optional[float]
    endpoints: Tuple[EndpointState, ...] = ()   # each endpoint; the designer is reachable if any of them is


class DesignerProbe:
    def __init__(self, profile_path: str, env: Optional[Mapping[str, str]] = None,
                 load: Callable[..., JobContext] = load_context, *, clock: Callable[[], float] = time.time,
                 max_age: float = MAX_AGE_SECONDS, timeout: float = 10.0):
        self.profile_path, self.env, self.load = profile_path, env, load
        self.clock, self.max_age, self.timeout = clock, max_age, timeout
        self.lock = threading.Lock()
        self.last: Optional[DesignerState] = None

    def check(self) -> DesignerState:
        ''' Asks the designer now. '''
        try:
            ctx = self.load(self.profile_path, self.env)
            declared = ctx.designer_endpoints()
            asked = ctx.designer_endpoint_states(self.timeout) if declared is not None else []
        except (OSError, ValueError) as error:
            state = DesignerState('', None, f'the profile cannot be read: {error}', self.clock())
        else:
            if declared is None:
                name = str(ctx.values.get('designer') or '')
                state = DesignerState(name, None, 'not an HTTP designer: nothing to ask', self.clock())
            else:
                endpoints = tuple(EndpointState(url, parallel, not reason, reason or url) for url, parallel, reason in asked)
                up = [e for e in endpoints if e.reachable]
                detail = ', '.join(e.url for e in up) if up else '; '.join(e.detail for e in endpoints)
                if up and len(up) < len(endpoints):
                    detail += f' ({len(endpoints) - len(up)} of {len(endpoints)} endpoints down)'
                state = DesignerState(declared.name, bool(up), detail, self.clock(), endpoints)
        with self.lock:
            self.last = state
        return state

    def current(self) -> DesignerState:
        ''' The last answer if it is recent, else a new one. '''
        with self.lock:
            last = self.last
        if last is not None and last.checked_at is not None and self.clock() - last.checked_at < self.max_age:
            return last
        return self.check()

    def unavailable(self) -> str:
        ''' Asks now: why the designer cannot be used, or '' if it can (or there is nothing to ask). '''
        state = self.check()
        return state.detail if state.reachable is False else ''
