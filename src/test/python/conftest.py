import logging
import os
import shutil

import pytest

# The gui tests are written for the offscreen platform: on a real display windows take focus and events, and
# they stall until their waits time out. An explicit QT_QPA_PLATFORM still wins.
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


def pytest_configure(config):
    config.addinivalue_line('markers', 'requires_ffmpeg: runs the real ffmpeg/ffprobe; skipped where they are not on the PATH')


def pytest_collection_modifyitems(config, items):
    ''' ffmpeg is optional (AGENTS.md): without it these tests skip, rather than fail or wait on an "ffmpeg not found" box. '''
    if shutil.which('ffmpeg') and shutil.which('ffprobe'):
        return
    skip = pytest.mark.skip(reason='ffmpeg and ffprobe are not on the PATH')
    for item in items:
        if 'requires_ffmpeg' in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session", autouse=True)
def logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


@pytest.fixture
def tmpdirPath(tmpdir):
    yield str(tmpdir)
    # required due to https://github.com/pytest-dev/pytest/issues/1120
    shutil.rmtree(str(tmpdir))
