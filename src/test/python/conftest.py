import logging
import os
import shutil

import pytest

# The gui tests are written for the offscreen platform: on a real display windows take focus and events, and
# they stall until their waits time out. An explicit QT_QPA_PLATFORM still wins.
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


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
