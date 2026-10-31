'''
model/version.py: the app's and the service's version, from the VERSION file CI writes. From source there is none, which is
expected: the default is used and nothing is logged as an error (it was a traceback at every start).
'''
import logging

from model.version import DEFAULT_VERSION, read_version


def test_the_version_file_is_read(tmp_path):
    (tmp_path / 'VERSION').write_text('2.2.0-alpha.36\n')
    assert read_version(str(tmp_path)) == '2.2.0-alpha.36'


def test_an_empty_version_file_gives_the_default(tmp_path):
    (tmp_path / 'VERSION').write_text('\n')
    assert read_version(str(tmp_path)) == DEFAULT_VERSION


def test_no_version_file_gives_the_default_without_a_warning_or_traceback(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    assert read_version(str(tmp_path)) == DEFAULT_VERSION
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
    assert all(r.exc_info is None for r in caplog.records)


def test_a_version_file_that_cannot_be_read_is_a_warning(tmp_path, caplog):
    (tmp_path / 'VERSION').mkdir()   # there, but not a file
    assert read_version(str(tmp_path)) == DEFAULT_VERSION
    assert [r.levelno for r in caplog.records] == [logging.WARNING]
    assert all(r.exc_info is None for r in caplog.records)
