'''
pipeline.fileops: Windows refuses, for a moment, a rename over a file another thread or process has open and an open of a
file being renamed over. Queue entries and the work-directory lease are written and read through these, so a reader never
loses an entry's write -- nor takes a held lease for a free one (CI found that on Windows).
'''
import json

import pytest

from pipeline import fileops
from pipeline.service import lease as lease_module
from pipeline.service.lease import WorkDirLease, read_lease


@pytest.fixture
def windows(monkeypatch):
    monkeypatch.setattr(fileops.os, 'name', 'nt')
    monkeypatch.setattr(fileops.time, 'sleep', lambda seconds: None)


def _refused(times, then):
    calls = []

    def action(*args):
        calls.append(args)
        if len(calls) <= times:
            raise PermissionError(13, 'The process cannot access the file because it is being used by another process')
        return then(*args)
    return action, calls


def test_a_rename_refused_for_a_moment_on_windows_is_tried_again(tmp_path, monkeypatch, windows):
    real = fileops.os.replace
    replace, calls = _refused(2, real)
    monkeypatch.setattr(fileops.os, 'replace', replace)
    (tmp_path / 'new').write_text('new')

    fileops.replace(str(tmp_path / 'new'), str(tmp_path / 'entry.json'))

    assert len(calls) == 3 and (tmp_path / 'entry.json').read_text() == 'new'


def test_a_refusal_that_lasts_or_happens_elsewhere_is_raised(monkeypatch, windows):
    def refuse():
        raise PermissionError(13, 'denied')
    with pytest.raises(PermissionError):
        fileops.retrying(refuse, attempts=3)
    monkeypatch.setattr(fileops.os, 'name', 'posix')   # only Windows refuses for a reader
    with pytest.raises(PermissionError):
        fileops.retrying(refuse)


def test_a_held_lease_read_during_its_heartbeat_is_still_held(tmp_path, monkeypatch, windows):
    ''' The CI failure (windows-2022): the lease read as None while the holder's heartbeat renamed over it. '''
    with WorkDirLease(str(tmp_path), 'job', host='box', pid=7, heartbeat_seconds=3600):   # only this test reads it
        real = lease_module._load
        load, calls = _refused(2, real)
        monkeypatch.setattr(lease_module, '_load', load)

        holder = read_lease(str(tmp_path))
        reads = len(calls)   # leaving reads it once more, to check it is still ours

    assert holder is not None and holder.job_id == 'job' and reads == 3


def test_a_damaged_lease_is_still_no_lease(tmp_path):
    (tmp_path / 'service').mkdir()
    lease_path = lease_module.lease_path(str(tmp_path))
    with open(lease_path, 'w') as f:
        json.dump({'host': 'box'}, f)   # no job or heartbeat
    assert read_lease(str(tmp_path)) is None
