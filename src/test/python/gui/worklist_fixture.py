'''
A fixture discovery index for the work list tests: the real `LibraryIndex` creates the file (so the schema is the frozen
one), then rows are inserted directly, so a test can say exactly what each title needs, since when, and whether it is new,
without building outputs on disk for each. (`test_pipeline_library_index.py` covers how a scan arrives at those values.)
'''
import sqlite3
import time
from typing import Iterable, Optional, Sequence, Tuple

from pipeline.library.index import LibraryIndex, index_path
from pipeline.library.state import TIER_OF_NEEDS

# name, kind, last_scanned, last_ok, last_error, item_count
SourceFixture = Tuple[str, str, Optional[float], Optional[float], str, int]


def title_row(title_id: str, title: str = '', needs: str = 'review', **fields) -> dict:
    ''' One `titles` row; every column not given is the schema's default. `tier` follows `needs` unless given. '''
    row = dict(id=title_id, title=title or title_id, display_name=title or title_id, needs=needs,
               tier=TIER_OF_NEEDS[needs], state_since=time.time() - 3600, first_seen_generation=1, source='films',
               year='2001', detail=needs)
    row.update(fields)
    return row


def make_index(work_dir, rows: Iterable[dict], sources: Sequence[SourceFixture] = (), generation: int = 2,
               last_scan_at: Optional[float] = None) -> str:
    '''
    Writes `<work_dir>/library-index.sqlite` with these rows and sources.
    :param generation: the scan generation; a row whose `first_seen_generation` equals it is "new since the last scan".
    :return: the index file.
    '''
    path = index_path(str(work_dir))
    with LibraryIndex(path):  # creates the file and the frozen schema
        pass
    db = sqlite3.connect(path)
    with db:
        db.execute("INSERT OR REPLACE INTO meta VALUES ('generation', ?)", (str(generation),))
        if last_scan_at is not None:
            db.execute("INSERT OR REPLACE INTO meta VALUES ('last_scan_at', ?)", (str(last_scan_at),))
        for position, (name, kind, scanned, ok, error, count) in enumerate(sources):
            db.execute('INSERT INTO sources VALUES (?, ?, ?, ?, ?, ?, ?)',
                       (name, position, kind, scanned, ok, error, count))
        for row in rows:
            names = list(row)
            db.execute(f'INSERT INTO titles ({",".join(names)}) VALUES ({",".join("?" * len(names))})',
                       [row[n] for n in names])
    db.close()
    return path
