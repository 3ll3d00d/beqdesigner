'''
The work-directory lease (design/pipeline-service.md §5.1). A person reviews in the app, which reads the same index and
review queue as the service; reading while a job runs is fine, but two *runs* at once -- the service's job and the work
list's Run, Publish or Commit -- would both write the index, the queue and the repositories. So every service job holds a
lease on the work directory, `<work_dir>/service/lease.json`: who holds it (host, process, job) and a heartbeat rewritten
every HEARTBEAT_SECONDS. Since design/worklist-feedback.md F5 the work list's and the command line's runs hold it too
(run_lease()), and extract and design work asked for while a lease is fresh is handed to the holder's run; otherwise the
work list refuses to start while a lease is fresh; one whose heartbeat is older than
STALE_AFTER_SECONDS (a service that died) is ignored and taken over.

Two services on one work directory are refused the same way. Taking the lease is not atomic across machines (a network
share gives no such guarantee); it is a guard against the ordinary mistake, not a lock manager.
'''
import json
import logging
import os
import socket
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Callable, Optional

logger = logging.getLogger('service_lease')

LEASE_DIR = 'service'
LEASE_FILE = 'lease.json'
HEARTBEAT_SECONDS = 30.0
WORKLIST, COMMAND_LINE = 'worklist', 'cli'   # run_lease()'s kinds: the service's own leases are named by their job id
STALE_AFTER_SECONDS = 3 * HEARTBEAT_SECONDS


@dataclass(frozen=True)
class LeaseHolder:
    host: str
    pid: int
    job_id: str
    heartbeat_at: float

    def who(self) -> str:
        ''' Who is running: the service's job, the work list or a command-line run (run_lease()), and where. '''
        if self.job_id.startswith(f'{WORKLIST}-'):
            return f'the work list on {self.host}'
        if self.job_id.startswith(f'{COMMAND_LINE}-'):
            return f'a command-line run on {self.host}'
        return f'the pipeline service on {self.host} ({self.job_id[:8]})'

    def describe(self) -> str:
        if self.job_id.startswith((f'{WORKLIST}-', f'{COMMAND_LINE}-')):
            return f'{self.who()} is running in this work directory; try again when it has finished'
        return (f'the pipeline service on {self.host} is running a job ({self.job_id[:8]}) in this work directory; '
                f'try again when it has finished')


class LeaseHeld(RuntimeError):
    ''' Someone else holds a fresh lease on the work directory. '''


def lease_path(work_dir: str) -> str:
    return os.path.join(work_dir, LEASE_DIR, LEASE_FILE)


def _read(work_dir: str) -> Optional[LeaseHolder]:
    try:
        with open(lease_path(work_dir), encoding='utf-8') as f:
            data = json.load(f)
        return LeaseHolder(str(data['host']), int(data['pid']), str(data['job_id']), float(data['heartbeat_at']))
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError):
        logger.warning('the lease at %s cannot be read; treating it as not held', lease_path(work_dir))
        return None


def read_lease(work_dir: Optional[str], now: Optional[float] = None) -> Optional[LeaseHolder]:
    ''' The holder of a fresh lease on `work_dir`, or None if there is none, it is stale, or there is no work directory. '''
    if not work_dir:
        return None
    holder = _read(work_dir)
    if holder is None:
        return None
    return holder if (time.time() if now is None else now) - holder.heartbeat_at <= STALE_AFTER_SECONDS else None


def run_lease(work_dir: str, kind: str) -> 'WorkDirLease':
    '''
    The lease a run of the work list or of the command line holds while it runs (design/worklist-feedback.md F5), as the
    service's jobs do, so each of them knows another run is going: extract and design work is handed to that run (the join
    inbox, pipeline.library.inbox) rather than run alongside it.
    '''
    return WorkDirLease(work_dir, f'{kind}-{uuid.uuid4().hex}')


class WorkDirLease:
    '''
    Held for one job: `with WorkDirLease(work_dir, job_id):`.
    :raises LeaseHeld: on entry, if another process holds a fresh lease.
    '''

    def __init__(self, work_dir: str, job_id: str, *, heartbeat_seconds: float = HEARTBEAT_SECONDS,
                 clock: Callable[[], float] = time.time, host: Optional[str] = None, pid: Optional[int] = None):
        self.__work_dir = work_dir
        self.__job_id = job_id
        self.__heartbeat_seconds = heartbeat_seconds
        self.__clock = clock
        self.__host = host or socket.gethostname()
        self.__pid = os.getpid() if pid is None else pid
        self.__stop = threading.Event()
        self.__beat: Optional[threading.Thread] = None

    def __mine(self, holder: Optional[LeaseHolder]) -> bool:
        return holder is not None and (holder.host, holder.pid, holder.job_id) == (self.__host, self.__pid, self.__job_id)

    def __write(self) -> None:
        holder = LeaseHolder(self.__host, self.__pid, self.__job_id, self.__clock())
        directory = os.path.dirname(lease_path(self.__work_dir))
        os.makedirs(directory, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix='.lease-', suffix='.json', dir=directory)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(asdict(holder), f)
            os.replace(temporary, lease_path(self.__work_dir))
        except BaseException:
            if os.path.exists(temporary):
                os.unlink(temporary)
            raise

    def __heartbeat(self) -> None:
        while not self.__stop.wait(self.__heartbeat_seconds):
            try:
                if self.__mine(_read(self.__work_dir)):
                    self.__write()
            except OSError:
                logger.exception('the lease heartbeat could not be written')

    def __enter__(self) -> 'WorkDirLease':
        holder = read_lease(self.__work_dir, self.__clock())
        if holder is not None and (holder.host, holder.pid) != (self.__host, self.__pid):
            raise LeaseHeld(holder.describe())
        self.__write()
        self.__beat = threading.Thread(target=self.__heartbeat, name='pipeline-service-lease', daemon=True)
        self.__beat.start()
        return self

    def __exit__(self, *exc) -> None:
        self.__stop.set()
        if self.__beat is not None:
            self.__beat.join()
        if self.__mine(_read(self.__work_dir)):   # never removes a lease someone took over from a stale one of ours
            try:
                os.unlink(lease_path(self.__work_dir))
            except FileNotFoundError:
                pass
