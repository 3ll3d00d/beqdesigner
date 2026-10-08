'''A queue entry written while a reader has it open (Windows refuses the rename for a moment) is written, not lost.'''
import pytest

from pipeline import review


def test_a_rename_refused_for_a_moment_on_windows_is_tried_again(tmp_path, monkeypatch):
    calls = []
    real = review.os.replace

    def replace(source, target):
        calls.append(target)
        if len(calls) < 3:
            raise PermissionError(13, 'The process cannot access the file because it is being used by another process')
        real(source, target)
    monkeypatch.setattr(review.os, 'replace', replace)
    monkeypatch.setattr(review.os, 'name', 'nt')
    monkeypatch.setattr(review.time, 'sleep', lambda seconds: None)
    (tmp_path / 'new').write_text('new')

    review._replace(str(tmp_path / 'new'), str(tmp_path / 'entry.json'))

    assert len(calls) == 3 and (tmp_path / 'entry.json').read_text() == 'new'


def test_a_rename_that_stays_refused_or_is_refused_elsewhere_is_raised(tmp_path, monkeypatch):
    def refuse(source, target):
        raise PermissionError(13, 'denied')
    monkeypatch.setattr(review.os, 'replace', refuse)
    monkeypatch.setattr(review.time, 'sleep', lambda seconds: None)
    monkeypatch.setattr(review.os, 'name', 'nt')
    with pytest.raises(PermissionError):
        review._replace('a', 'b', attempts=3)
    monkeypatch.setattr(review.os, 'name', 'posix')   # only Windows refuses a rename for a reader
    with pytest.raises(PermissionError):
        review._replace('a', 'b')
