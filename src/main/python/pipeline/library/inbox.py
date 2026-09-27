'''
The work directory's join inbox -- design/archive/library-sync/worklist-feedback.md F5: how a run in one process (the
work list, a command-line `run`, the service) takes extract and design work from another.

A request is a JSON file, `<work_dir>/service/join/<id>.json`, written atomically. The run holding the work directory's
lease claims it (`claim()`, a source of its `JoinQueue`) by renaming it to `<id>.taken`; the poster, if it gives up waiting
(the run ended without claiming it), withdraws it by renaming it to `<id>.withdrawn`. The two renames are what decide: the
file can only go one way, so a request is either run by the holder or taken back by its poster, never both and never
neither. A file that cannot be read is renamed `<id>.bad` and left for a person.
'''
import json
import logging
import os
import tempfile
from dataclasses import asdict
from typing import List, Optional

from pipeline.library.join import JoinRequest
from pipeline.library.selection import Selection

logger = logging.getLogger('library_inbox')

INBOX_DIR = os.path.join('service', 'join')
PENDING, TAKEN, WITHDRAWN = 'pending', 'taken', 'withdrawn'


def inbox_path(work_dir: str) -> str:
    return os.path.join(work_dir, INBOX_DIR)


def request_to_json(request: JoinRequest) -> dict:
    return {'id': request.id, 'through': request.through, 'retry_failed': request.retry_failed,
            'selection': {name: list(value) if isinstance(value, tuple) else value
                          for name, value in asdict(request.selection).items()}}


def request_from_json(data: dict) -> JoinRequest:
    selection = {name: tuple(value) if isinstance(value, list) else value for name, value in data['selection'].items()}
    return JoinRequest(Selection(**selection), data['through'], bool(data.get('retry_failed')), id=str(data['id']))


class WorkDirInbox:

    def __init__(self, work_dir: str):
        self.__dir = inbox_path(work_dir)

    def __path(self, request_id: str, suffix: str) -> str:
        return os.path.join(self.__dir, f'{request_id}.{suffix}')

    def post(self, request: JoinRequest) -> None:
        ''' Offers the request to whichever run holds the lease; it is claimed the next time that run looks for work. '''
        os.makedirs(self.__dir, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=f'.{request.id}.', suffix='.tmp', dir=self.__dir)
        try:
            with os.fdopen(handle, 'w', encoding='utf-8') as f:
                json.dump(request_to_json(request), f)
            os.replace(temporary, self.__path(request.id, 'json'))
        except BaseException:
            if os.path.exists(temporary):
                os.unlink(temporary)
            raise

    def claim(self) -> List[JoinRequest]:
        ''' Every request waiting, oldest first, each now `taken` by the caller (a run holding the lease). '''
        try:
            names = [n for n in os.listdir(self.__dir) if n.endswith('.json')]
        except FileNotFoundError:
            return []
        paths = sorted((os.path.join(self.__dir, n) for n in names), key=_mtime)
        claimed = []
        for path in paths:
            request_id = os.path.basename(path)[:-len('.json')]
            taken = self.__path(request_id, TAKEN)
            try:
                os.rename(path, taken)
            except FileNotFoundError:   # withdrawn (or claimed) in the meantime
                continue
            try:
                with open(taken, encoding='utf-8') as f:
                    claimed.append(request_from_json(json.load(f)))
            except (OSError, ValueError, KeyError, TypeError) as error:
                logger.warning('the join request %s cannot be read, leaving it as .bad: %s', taken, error)
                os.replace(taken, self.__path(request_id, 'bad'))
        return claimed

    def withdraw(self, request_id: str) -> bool:
        ''' Takes back a request no run has claimed. :return: False if a run claimed it first. '''
        try:
            os.rename(self.__path(request_id, 'json'), self.__path(request_id, WITHDRAWN))
        except FileNotFoundError:
            return False
        return True

    def state(self, request_id: str) -> Optional[str]:
        ''' `pending`, `taken`, `withdrawn`, or None if there is no such request. '''
        for state, suffix in ((PENDING, 'json'), (TAKEN, TAKEN), (WITHDRAWN, WITHDRAWN)):
            if os.path.isfile(self.__path(request_id, suffix)):
                return state
        return None

    def forget(self, request_id: str) -> None:
        ''' The poster is done with it: its `.taken` or `.withdrawn` file goes. '''
        for suffix in (TAKEN, WITHDRAWN):
            try:
                os.unlink(self.__path(request_id, suffix))
            except FileNotFoundError:
                pass


def _mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0
