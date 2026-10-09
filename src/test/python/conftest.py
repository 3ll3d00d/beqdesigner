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


@pytest.fixture(autouse=True)
def _plenty_of_free_space(monkeypatch):
    '''
    **No test depends on the free space of the machine it runs on.** A library run stops below `run.min_free_gb` (10 GB
    by default) free in the work directory, and a CI runner can have less (ubuntu-26.04 had 8.3 GB), which failed every
    test that extracts. A test about the floor sets `disk_usage` itself, which replaces this.
    '''
    real = shutil.disk_usage
    monkeypatch.setattr(shutil, 'disk_usage', lambda path: real(path)._replace(free=10 ** 15))


@pytest.fixture(autouse=True)
def _designer_registry_restored():
    '''
    **Every test leaves the designer registry as it found it.** The registry is process-wide, and a service job or a CLI
    run registers its profile's designers there and never removes them, so under `-n auto` whichever test a worker ran
    before decided what the next one saw (the batch dialog's "no designers are registered" test failed after a service
    test had left 'rolloff' behind).
    '''
    from pipeline.designer import registry
    designers, takes_sources = dict(registry._registry), set(registry._takes_sources)
    yield
    registry._registry.clear()
    registry._registry.update(designers)
    registry._takes_sources.clear()
    registry._takes_sources.update(takes_sources)
