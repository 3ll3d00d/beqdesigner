'''
Is a failure the title's, or something the title depends on that was not there?

A run remembers a title's failure against its source and settings, and does not try it again until one of them
changes (run.py). That is right for a title that cannot be done -- an undecodable stream, a designer that answers 4xx
or declines, an ffmpeg error on a file it can read -- and wrong for one that met a designer that was down or slow, a
dropped media mount or a JRiver that did not answer: remembered, an unattended schedule would never come back to it.
`unavailable_reason()` tells the two apart; a run reports the second kind but does not remember it.
'''
import errno
import os
from typing import Optional

import requests

from pipeline.designer.http_binding import DesignerUnavailable

# the network or remote-filesystem errors an OSError can carry when what is at the other end went away
_UNAVAILABLE_ERRNOS = frozenset(getattr(errno, name) for name in (
    'ECONNREFUSED', 'ECONNRESET', 'ECONNABORTED', 'ETIMEDOUT', 'EHOSTDOWN', 'EHOSTUNREACH', 'ENETDOWN',
    'ENETUNREACH', 'ENETRESET', 'ENOTCONN', 'ESTALE', 'EREMOTEIO') if hasattr(errno, name))


class Unavailable(Exception):
    ''' Raised when something a title depends on is known not to be there: never remembered against the title. '''


def _chain(error: BaseException):
    ''' The error and those it was raised from or while handling, each once. '''
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        yield error
        error = error.__cause__ or error.__context__


def _unavailable_error(error: BaseException) -> Optional[str]:
    if isinstance(error, (Unavailable, DesignerUnavailable)):
        return str(error) or type(error).__name__
    if isinstance(error, requests.Timeout):
        return 'timed out'
    if isinstance(error, requests.ConnectionError):
        return 'could not connect'
    if isinstance(error, requests.HTTPError):
        status = getattr(error.response, 'status_code', None)
        return f'answered HTTP {status}' if status is not None and status >= 500 else None
    if isinstance(error, TimeoutError):
        return 'timed out'
    if isinstance(error, ConnectionError):
        return 'could not connect'
    if isinstance(error, OSError) and error.errno in _UNAVAILABLE_ERRNOS:
        return os.strerror(error.errno)
    return None


def missing_mount(path: Optional[str]) -> Optional[str]:
    '''
    Why `path` cannot be read because the storage it is on is not there, or None if it is there (or the path's absence
    is the title's own problem). A missing file under a populated folder is the title's: it was moved or deleted. A
    missing file whose nearest existing folder is empty is a share or mount that is not mounted (a dropped mount
    leaves its empty mount point), and one whose root is not there (a Windows drive letter or UNC share) is storage
    that is not connected. A POSIX path with nothing above it but `/` is a wrong path, not a missing mount: a mount
    point exists whether or not anything is mounted on it.
    '''
    if not path or os.path.exists(path):
        return None
    parent = os.path.dirname(os.path.abspath(path))
    while not os.path.isdir(parent):
        up = os.path.dirname(parent)
        if up == parent:
            return 'the media storage is not available (its drive or share is not connected)'
        parent = up
    if parent == os.path.dirname(parent):
        return None
    try:
        empty = not any(True for _ in os.scandir(parent))
    except OSError as error:
        return f'the media storage is not available ({parent}: {error.strerror or error})'
    if empty:
        return f'the media storage is not available ({parent} is empty: is it mounted?)'
    return None


def unavailable_reason(error: BaseException, source_path: Optional[str] = None) -> Optional[str]:
    '''
    :param source_path: the title's media, if the failure might have come from reading it.
    :return: why a dependency of the title was unavailable, if that is what `error` was; None if it was the title's own
        failure, which is remembered.
    '''
    for link in _chain(error):
        if isinstance(link, requests.HTTPError) and _unavailable_error(link) is None:
            return None   # a 4xx: the designer (or JRiver) answered, and refused this title
        reason = _unavailable_error(link)
        if reason is not None:
            return reason
    return missing_mount(source_path)
