'''
Extra work offered to a run in progress -- design/archive/library-sync/worklist-feedback.md F5.

A run's machine phase (extract and design) takes more titles while it lasts, so a person who starts another extraction does
not wait for the first to finish. The run (`run_stages(join=...)`) calls `take()` whenever it looks for work; each request
taken is planned `through` extract or design, never past it, and its titles go to the back of the queue. Once the machine
phase is over the run calls `close()`: nothing more is taken, `offer()` refuses, and `close()` hands back what was offered
but not taken, for the offerer to run on its own.

Besides what is offered in this process, a queue may take requests from other sources (`sources`): a callable returning
what it has for the run, called under the queue's lock and only while it is open (the work directory's inbox, through which
another process joins, is one).
'''
import threading
import uuid
from dataclasses import dataclass, field
from typing import Callable, List, Sequence

from pipeline.library.selection import Selection

JOINABLE = ('extract', 'design')   # a run's machine phase: publish and commit are not joined, they wait for the run to end


@dataclass(frozen=True)
class JoinRequest:
    selection: Selection
    through: str = 'design'
    retry_failed: bool = False
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self):
        if self.through not in JOINABLE:
            raise ValueError(f"only {' or '.join(JOINABLE)} can join a run in progress; got {self.through!r}")


class JoinQueue:
    ''' Thread-safe: offered from any thread, taken by the running run. '''

    def __init__(self, sources: Sequence[Callable[[], List[JoinRequest]]] = ()):
        self.__lock = threading.Lock()
        self.__pending: List[JoinRequest] = []
        self.__sources = list(sources)
        self.__closed = False
        self.__taken: List[str] = []

    def offer(self, request: JoinRequest) -> bool:
        ''' :return: True if the run will take it; False if its machine phase is over (run it on its own). '''
        with self.__lock:
            if self.__closed:
                return False
            self.__pending.append(request)
            return True

    def take(self) -> List[JoinRequest]:
        ''' What has been offered since the last call, and what the other sources have; nothing once closed. '''
        with self.__lock:
            if self.__closed:
                return []
            taken, self.__pending = self.__pending, []
            for source in self.__sources:
                taken.extend(source())
            self.__taken.extend(r.id for r in taken)
            return taken

    def close(self) -> List[JoinRequest]:
        ''' Takes no more. :return: what was offered and not taken, which the run will not do. '''
        with self.__lock:
            self.__closed = True
            left, self.__pending = self.__pending, []
            return left

    @property
    def closed(self) -> bool:
        with self.__lock:
            return self.__closed

    @property
    def taken(self) -> List[str]:
        ''' The ids of the requests the run has taken. '''
        with self.__lock:
            return list(self.__taken)
