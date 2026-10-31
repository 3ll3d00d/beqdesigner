'''
The release this is: `VERSION` beside `app.py` (in the bundle, `sys._MEIPASS`), written by CI from the release tag. Running
from source there is none, which is expected: the version is then what `git describe` says of the checkout (the last tag,
the commits since it and the commit, e.g. `2.2.0-alpha.35-3-gd770e80`, with `-dirty` for uncommitted changes), and
DEFAULT_VERSION only where that cannot be had either. Qt-free: the pipeline service reads it too.
'''
import logging
import os
import subprocess
from typing import Optional

logger = logging.getLogger('version')

DEFAULT_VERSION = '0.0.0-alpha.1'
GIT_TIMEOUT_S = 5


def describe_checkout(root: str) -> Optional[str]:
    ''' `git describe --tags --always --dirty` of the checkout `root` is in; None if git or the checkout is not there. '''
    try:
        result = subprocess.run(['git', '-C', root, 'describe', '--tags', '--always', '--dirty'], capture_output=True,
                                text=True, timeout=GIT_TIMEOUT_S, check=True,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except (OSError, subprocess.SubprocessError) as error:
        logger.debug('No version from git in %s: %s', root, error)
        return None
    return result.stdout.strip() or None


def read_version(root: str) -> str:
    '''
    The version in `<root>/VERSION`; with no such file (running from source), the checkout's `git describe`; else
    DEFAULT_VERSION. An empty or unreadable file is DEFAULT_VERSION: it was written, so it is not a checkout's to answer.
    '''
    path = os.path.abspath(os.path.join(root, 'VERSION'))
    try:
        with open(path, encoding='utf-8') as f:
            return f.read().strip() or DEFAULT_VERSION
    except FileNotFoundError:
        described = describe_checkout(os.path.dirname(path))
        logger.debug('No %s (running from source): the version is %s', path, described or DEFAULT_VERSION)
        return described or DEFAULT_VERSION
    except OSError as error:
        logger.warning('Unable to read %s (%s): the version is %s', path, error, DEFAULT_VERSION)
    return DEFAULT_VERSION
