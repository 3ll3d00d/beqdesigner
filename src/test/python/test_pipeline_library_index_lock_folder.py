'''
An index in a folder that refuses byte-range locks (the Linux SMB3 client, an Unraid share through /mnt/user) is locked
with a lock folder instead -- SQLite's unix-dotfile VFS -- and every program using it makes the same choice, read from the
`.locking` file beside it. A lock folder left by a program killed mid-write is removed once it is stale.
'''
import os
import sqlite3
import time

import pytest

from pipeline.library import index as index_module
from pipeline.library.index import LibraryIndex, byte_range_locks_work, clear_stale_lock, folder_can_lock, \
    lock_folder, lock_folder_works, lock_held_for, locking_file, uses_lock_folder

pytestmark = pytest.mark.skipif(os.name == 'nt', reason='SQLite has no unix-dotfile VFS on Windows')


@pytest.fixture(autouse=True)
def _forget_probes(monkeypatch):
    monkeypatch.setattr(index_module, '_ORDINARY_LOCKS_WORK', set())


def _refuse_byte_range_locks(monkeypatch):
    monkeypatch.setattr(index_module, 'byte_range_locks_work', lambda directory: False)


def _lock_folder_index(tmp_path):
    path = str(tmp_path / 'library-index.sqlite')
    with open(locking_file(path), 'w') as f:
        f.write('dotfile\n')
    return path


def _age(path, seconds):
    then = time.time() - seconds
    os.utime(path, (then, then))


def test_both_probes_work_in_an_ordinary_folder_and_leave_nothing_behind(tmp_path):
    assert byte_range_locks_work(str(tmp_path)) and lock_folder_works(str(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_a_folder_that_takes_only_a_lock_folder_can_still_hold_the_index(tmp_path, monkeypatch):
    _refuse_byte_range_locks(monkeypatch)
    assert folder_can_lock(str(tmp_path))
    monkeypatch.setattr(index_module, 'lock_folder_works', lambda directory: False)
    assert not folder_can_lock(str(tmp_path))


def test_ordinary_locks_are_used_where_they_work_and_nothing_is_written(tmp_path):
    path = str(tmp_path / 'library-index.sqlite')
    assert not uses_lock_folder(path)
    assert not os.path.exists(locking_file(path))
    assert str(tmp_path) in index_module._ORDINARY_LOCKS_WORK   # probed once per process


def test_the_first_program_to_find_ordinary_locks_refused_decides_for_every_other(tmp_path, monkeypatch):
    path = str(tmp_path / 'library-index.sqlite')
    _refuse_byte_range_locks(monkeypatch)
    assert not uses_lock_folder(path, decide=False)   # a reader never decides
    assert not os.path.exists(locking_file(path))

    assert uses_lock_folder(path)
    assert open(locking_file(path)).read() == 'dotfile\n'

    monkeypatch.setattr(index_module, 'byte_range_locks_work', lambda directory: True)
    assert uses_lock_folder(path) and uses_lock_folder(path, decide=False)   # whatever this program's view of the folder


def test_a_folder_that_refuses_both_kinds_of_lock_decides_nothing(tmp_path, monkeypatch):
    path = str(tmp_path / 'library-index.sqlite')
    _refuse_byte_range_locks(monkeypatch)
    monkeypatch.setattr(index_module, 'lock_folder_works', lambda directory: False)
    assert not uses_lock_folder(path)
    assert not os.path.exists(locking_file(path))


def test_an_index_locked_with_a_lock_folder_holds_it_only_while_it_writes(tmp_path):
    path = _lock_folder_index(tmp_path)
    with LibraryIndex(path) as index:
        assert index.generation == 0
        assert lock_held_for(path) is None
        db = index._LibraryIndex__db
        db.execute('BEGIN EXCLUSIVE')
        assert os.path.isdir(lock_folder(path)) and lock_held_for(path) >= 0
        db.execute('ROLLBACK')
        assert not os.path.exists(lock_folder(path))
    with LibraryIndex(path, readonly=True) as reader:
        assert reader.generation == 0


def test_a_second_program_waits_for_the_lock_folder_and_gives_up(tmp_path, monkeypatch):
    path = _lock_folder_index(tmp_path)
    LibraryIndex(path).close()
    monkeypatch.setattr(index_module, 'BUSY_TIMEOUT_S', 0.2)
    with LibraryIndex(path) as holder, LibraryIndex(path) as other:
        holder._LibraryIndex__db.execute('BEGIN EXCLUSIVE')
        try:
            with pytest.raises(sqlite3.OperationalError, match='locked'):
                other._LibraryIndex__db.execute("INSERT OR REPLACE INTO meta VALUES ('x', '1')")
        finally:
            holder._LibraryIndex__db.execute('ROLLBACK')
        other._LibraryIndex__db.execute("INSERT OR REPLACE INTO meta VALUES ('x', '1')")   # free again
        other._LibraryIndex__db.commit()


def test_a_lock_folder_left_by_a_program_killed_mid_write_is_removed_and_the_write_goes_ahead(tmp_path, monkeypatch):
    path = _lock_folder_index(tmp_path)
    monkeypatch.setattr(index_module, 'BUSY_TIMEOUT_S', 0.2)
    with LibraryIndex(path) as index:
        os.mkdir(lock_folder(path))   # left after the index was opened, so only the retry can clear it
        _age(lock_folder(path), index_module.STALE_LOCK_S + 10)
        db = index._LibraryIndex__db
        db.execute("INSERT OR REPLACE INTO meta VALUES ('x', '1')")
        db.commit()
        assert not os.path.exists(lock_folder(path))


def test_a_fresh_lock_folder_is_another_program_writing_and_is_left_alone(tmp_path):
    path = str(tmp_path / 'library-index.sqlite')
    assert clear_stale_lock(path)   # none there: the caller may go ahead
    os.mkdir(lock_folder(path))
    assert not clear_stale_lock(path)
    assert os.path.isdir(lock_folder(path))
    assert lock_held_for(path, now=lambda: os.stat(lock_folder(path)).st_mtime + 30) == pytest.approx(30)

    _age(lock_folder(path), index_module.STALE_LOCK_S + 1)
    assert clear_stale_lock(path)
    assert not os.path.exists(lock_folder(path)) and lock_held_for(path) is None
