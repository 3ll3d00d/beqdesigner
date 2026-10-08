'''
Small file operations made safe against Windows' sharing rules: there a rename over a file another thread or process has
open, or an open of a file being renamed over, fails for a moment with PermissionError. Elsewhere they never do, and these
behave exactly as the plain calls.
'''
import os
import time
from typing import Callable, TypeVar

T = TypeVar('T')

ATTEMPTS, PAUSE_S = 20, 0.05   # a second in all: a reader or writer holds a small file for far less


def retrying(action: Callable[[], T], attempts: int = ATTEMPTS, pause: float = PAUSE_S) -> T:
    ''' `action()`, tried again on Windows while it fails with PermissionError. '''
    for attempt in range(attempts):
        try:
            return action()
        except PermissionError:
            if os.name != 'nt' or attempt == attempts - 1:
                raise
            time.sleep(pause)


def replace(source: str, target: str) -> None:
    ''' os.replace(), retried on Windows while a reader has the target open. '''
    retrying(lambda: os.replace(source, target))
