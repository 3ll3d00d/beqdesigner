'''
pipeline.library.failure: telling a title's own failure (remembered, not tried again) from a dependency that was
unavailable (reported, tried again next run).
'''
import errno
import os

import pytest
import requests

from pipeline.designer.http_binding import DesignerUnavailable, HttpDesignerError
from pipeline.library.failure import Unavailable, missing_mount, unavailable_reason


def _raised_from(cause: BaseException, outer=HttpDesignerError) -> BaseException:
    ''' What the HTTP binding raises: its own error, raised from the requests one. '''
    try:
        try:
            raise cause
        except BaseException as inner:
            raise outer(f'POST http://designer failed: {inner}') from inner
    except BaseException as error:
        return error


def _http_error(status: int) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status
    return requests.HTTPError(f'{status} error', response=response)


@pytest.mark.parametrize('cause, reason', [
    (requests.ReadTimeout('read timed out'), 'timed out'),
    (requests.ConnectTimeout('connect timed out'), 'timed out'),
    (requests.ConnectionError('connection refused'), 'could not connect'),
    (_http_error(503), 'answered HTTP 503'),
    (_http_error(500), 'answered HTTP 500'),
])
def test_a_designer_that_is_down_slow_or_broken_is_unavailable(cause, reason):
    assert unavailable_reason(_raised_from(cause)) == reason


@pytest.mark.parametrize('status', [400, 404, 422])
def test_a_designer_that_answered_and_refused_the_title_is_the_titles_failure(status):
    assert unavailable_reason(_raised_from(_http_error(status))) is None


def test_a_designer_that_cannot_take_references_is_unavailable_whatever_the_title():
    assert unavailable_reason(DesignerUnavailable('it cannot take arrays by reference'))


@pytest.mark.parametrize('error', [
    ConnectionRefusedError('refused'),
    TimeoutError('timed out'),
    OSError(errno.ENOTCONN, 'Transport endpoint is not connected'),
    OSError(errno.ESTALE, 'Stale file handle'),
    Unavailable('JRiver did not answer'),
])
def test_network_and_remote_filesystem_errors_are_unavailable(error):
    assert unavailable_reason(error)


@pytest.mark.parametrize('error', [
    RuntimeError('ffmpeg exploded'),
    ValueError('no audio stream to extract'),
    OSError(errno.EACCES, 'Permission denied'),
    HttpDesignerError('did not return a valid JSON body'),
])
def test_anything_else_is_the_titles_own_failure(error):
    assert unavailable_reason(error) is None


def test_a_file_missing_from_an_empty_folder_is_a_mount_that_is_not_there(tmp_path):
    mount = tmp_path / 'films'
    mount.mkdir()
    source = str(mount / 'Heat (1995)' / 'Heat.mkv')

    assert 'is empty' in missing_mount(source)
    assert unavailable_reason(RuntimeError('ffmpeg: No such file or directory'), source)


def test_a_file_missing_from_a_populated_folder_was_moved_or_deleted(tmp_path):
    (tmp_path / 'films').mkdir()
    (tmp_path / 'films' / 'Alien.mkv').write_bytes(b'')
    source = str(tmp_path / 'films' / 'Heat.mkv')

    assert missing_mount(source) is None
    assert unavailable_reason(RuntimeError('ffmpeg: No such file or directory'), source) is None


def test_a_file_that_is_there_is_not_a_missing_mount(tmp_path):
    source = tmp_path / 'Heat.mkv'
    source.write_bytes(b'')

    assert missing_mount(str(source)) is None


def test_a_posix_path_with_nothing_but_the_root_above_it_is_a_wrong_path():
    assert missing_mount('/nonexistent-beq-root/films/Heat.mkv') is None


@pytest.mark.skipif(os.name != 'nt', reason='drive letters are Windows paths')
def test_a_drive_that_is_not_connected_is_a_mount_that_is_not_there():
    used = {d for d in 'DEFGHIJKLMNOPQRSTUVWXYZ' if os.path.exists(f'{d}:\\')}
    free = next(d for d in reversed('DEFGHIJKLMNOPQRSTUVWXYZ') if d not in used)
    assert 'not connected' in missing_mount(f'{free}:\\films\\Heat.mkv')
