'''
The release this is: `VERSION` beside `app.py` (in the bundle, `sys._MEIPASS`), written by CI from the release tag. Running
from source there is none, which is expected, and the default is used. Qt-free: the pipeline service reads it too.
'''
import logging
import os

logger = logging.getLogger('version')

DEFAULT_VERSION = '0.0.0-alpha.1'


def read_version(root: str) -> str:
    ''' The version in `<root>/VERSION`; DEFAULT_VERSION if there is no such file, or it is empty or cannot be read. '''
    path = os.path.abspath(os.path.join(root, 'VERSION'))
    try:
        with open(path, encoding='utf-8') as f:
            return f.read().strip() or DEFAULT_VERSION
    except FileNotFoundError:
        logger.debug('No %s (running from source): the version is %s', path, DEFAULT_VERSION)
    except OSError as error:
        logger.warning('Unable to read %s (%s): the version is %s', path, error, DEFAULT_VERSION)
    return DEFAULT_VERSION
