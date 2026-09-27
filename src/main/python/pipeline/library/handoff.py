'''
Handing extract and design work to the run in progress in another process -- design/worklist-feedback.md F5.

A command-line `run` that finds the work directory's lease held (by the service, the work list or another `run`) does not
run alongside it -- both would write the index and the review queue -- nor refuse: it posts its titles to the join inbox
(pipeline.library.inbox) and waits. Once the holder's run claims them, it waits for that run to end (the lease is released)
and says what became of its titles, from the index. If the holder's run ends first without claiming them, it withdraws
them and runs them itself.
'''
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from pipeline.library.inbox import PENDING, TAKEN, WorkDirInbox
from pipeline.library.index import LibraryIndex
from pipeline.library.join import JoinRequest
from pipeline.service.lease import LeaseHolder, read_lease

JOINED, WITHDRAWN = 'joined', 'withdrawn'


def hand_off(work_dir: str, request: JoinRequest, *, poll_seconds: float = 1.0,
             sleep: Callable[[float], None] = time.sleep,
             on_claimed: Optional[Callable[[], None]] = None) -> str:
    '''
    Posts `request` and waits. :return: `joined` once the run that claimed it has ended, or `withdrawn` if no run claimed
    it (the lease was released with it still waiting): the caller then runs it itself. Interrupted, a request not yet
    claimed is withdrawn before the interrupt goes on.
    '''
    inbox = WorkDirInbox(work_dir)
    inbox.post(request)
    claimed = False
    try:
        while True:
            state = inbox.state(request.id)
            holder = read_lease(work_dir)
            if state == TAKEN:
                if not claimed:
                    claimed = True
                    if on_claimed is not None:
                        on_claimed()
                if holder is None:
                    inbox.forget(request.id)
                    return JOINED
            elif state == PENDING:
                if holder is None and inbox.withdraw(request.id):
                    inbox.forget(request.id)
                    return WITHDRAWN
                if holder is None:
                    continue   # claimed between the two looks: follow it
            else:   # withdrawn or gone: nobody will run it but the caller
                inbox.forget(request.id)
                return WITHDRAWN
            sleep(poll_seconds)
    except BaseException:
        inbox.withdraw(request.id)   # (does nothing to one a run has claimed)
        raise


@dataclass(frozen=True)
class TitleOutcome:
    needs: str
    detail: str
    failed: bool   # it failed, or it was not done (still needs extract or design)


def outcomes(index: LibraryIndex, ids: List[str]) -> Dict[str, TitleOutcome]:
    ''' What the handed-off titles need now: each one still needing extract or design, or failed, counts as failed. '''
    rows = {row.id: row for row in index.titles(ids=ids)}
    result = {}
    for title_id in ids:
        row = rows.get(title_id)
        if row is None:
            result[title_id] = TitleOutcome('', 'not in the index', True)
            continue
        failed = row.extract_state == 'failed' or row.design_state == 'failed' or row.needs in ('extract', 'design')
        result[title_id] = TitleOutcome(row.needs, row.detail, failed)
    return result


def who(holder: Optional[LeaseHolder]) -> str:
    return holder.who() if holder is not None else 'another run'
