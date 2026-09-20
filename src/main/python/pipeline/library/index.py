'''
The discovery index -- design/library-sync/workflow-rework/design.md §12.5 (schema frozen there, at `SCHEMA_VERSION`).

A disposable SQLite cache, one row per title, of what `scan()` found and what each title needs next. It answers "what
work exists?" without doing any of it: a scan lists each source, merges them (union.union_of), reads the outputs (status.py)
and never extracts or designs. Delete the file and the cost is a rescan; nothing else is lost, because claims are
reconstructed from the queue and work directories and ignores live in the profile. The one thing a rebuild cannot
restore is `state_since`, which restarts at the rebuild time.

    with LibraryIndex(index_path(work_dir)) as index:
        result = index.scan(profile)               # list, diff, write
        for row in index.titles(needs='review'):   # the work list
            ...

Stdlib `sqlite3` only, and Qt-free like the rest of `pipeline/`.
'''
import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from pipeline.library.catalogue_scan import XmlRecord, scan_xml_repo, tmdb_index
from pipeline.library.profile import Profile, SourceSpec, build_source
from pipeline.library.season import SeasonGroup, Unit, plan_units
from pipeline.library.source import LibraryItem, LibrarySource
from pipeline.library.state import FLAG_DUPLICATE, FLAG_GONE, FLAG_IGNORED, FLAG_IN_CATALOGUE, FLAG_SHADOWED, NEEDS, \
    TIER_ORDER, TIER_OF_NEEDS, StageStates, derive_needs
from pipeline.library.status import Evaluation, Evaluator, FailureMemory, ScanSettings, failure_applies, \
    safe_fingerprint
from pipeline.library.union import UnionTitle, ignore_label, reconstruct_claims, union_of

logger = logging.getLogger('library_index')

INDEX_FILE_NAME = 'library-index.sqlite'
_ID_BATCH = 500   # ids per query, as units() does
SCHEMA_VERSION = 1  # PRAGMA user_version. A different one means drop the file's tables and rescan: it is a cache.

SCHEMA = f'''
CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE sources (
    name         TEXT PRIMARY KEY,
    position     INTEGER NOT NULL,
    kind         TEXT NOT NULL,
    last_scanned REAL,
    last_ok      REAL,
    last_error   TEXT NOT NULL DEFAULT '',
    item_count   INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE titles (
    id                    TEXT PRIMARY KEY,
    unit                  TEXT NOT NULL DEFAULT 'item',
    source                TEXT NOT NULL DEFAULT '',
    item_id               TEXT NOT NULL DEFAULT '',
    members               TEXT NOT NULL DEFAULT '[]',
    path                  TEXT NOT NULL DEFAULT '',
    display_name          TEXT NOT NULL DEFAULT '',
    title                 TEXT NOT NULL DEFAULT '',
    year                  TEXT NOT NULL DEFAULT '',
    kind                  TEXT NOT NULL DEFAULT 'movie',
    season                TEXT NOT NULL DEFAULT '',
    episodes              TEXT NOT NULL DEFAULT '[]',
    external_ids          TEXT NOT NULL DEFAULT '{{}}',
    items                 TEXT NOT NULL DEFAULT '[]',
    fingerprint           TEXT NOT NULL DEFAULT '',
    first_seen_generation INTEGER NOT NULL DEFAULT 0,
    last_seen             REAL NOT NULL DEFAULT 0,
    also_in               TEXT NOT NULL DEFAULT '[]',
    shadowed_by           TEXT NOT NULL DEFAULT '',
    ignored               TEXT NOT NULL DEFAULT '',
    gone                  INTEGER NOT NULL DEFAULT 0,
    duplicates            TEXT NOT NULL DEFAULT '[]',
    in_catalogue          INTEGER NOT NULL DEFAULT 0,
    extract_state         TEXT NOT NULL DEFAULT 'none',
    design_state          TEXT NOT NULL DEFAULT 'none',
    review_state          TEXT NOT NULL DEFAULT 'none',
    publish_state         TEXT NOT NULL DEFAULT 'none',
    commit_state          TEXT NOT NULL DEFAULT 'none',
    needs                 TEXT NOT NULL,
    tier                  TEXT NOT NULL,
    detail                TEXT NOT NULL DEFAULT '',
    state_since           REAL NOT NULL,
    confidence            REAL,
    candidate_count       INTEGER NOT NULL DEFAULT 0,
    failure               TEXT NOT NULL DEFAULT '',
    entry_summary         TEXT NOT NULL DEFAULT '',
    digest_key            TEXT NOT NULL DEFAULT '',
    current_digest        TEXT NOT NULL DEFAULT '',
    conflict              INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX titles_by_needs  ON titles (needs);
CREATE INDEX titles_by_tier   ON titles (tier, state_since);
CREATE INDEX titles_by_source ON titles (source);
CREATE TABLE failures (
    id          TEXT PRIMARY KEY,
    stage       TEXT NOT NULL,
    message     TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    key         TEXT NOT NULL,
    at          REAL NOT NULL
);
CREATE TABLE repo_xml (
    path     TEXT PRIMARY KEY,
    mtime_ns INTEGER NOT NULL,
    size     INTEGER NOT NULL,
    tmdb     TEXT NOT NULL,
    is_tv    INTEGER NOT NULL
);
'''

_TABLES = ('meta', 'sources', 'titles', 'failures', 'repo_xml')
_TITLE_COLUMNS = tuple(
    line.split()[0] for line in SCHEMA[SCHEMA.index('CREATE TABLE titles'):SCHEMA.index('CREATE INDEX')].splitlines()[1:]
    if line.startswith('    ') and not line.startswith('     ') and line.split()[0] not in (')', ');'))


def index_path(work_dir: str) -> str:
    ''' Where a work directory's index lives. '''
    return os.path.join(work_dir, INDEX_FILE_NAME)


# --- items <-> JSON -------------------------------------------------------------------------------------------------

def item_to_json(item: LibraryItem) -> dict:
    return asdict(item)


def item_from_json(data: Mapping[str, Any]) -> LibraryItem:
    data = dict(data)
    data['episodes'] = tuple(data.get('episodes') or ())
    data['art_candidates'] = tuple(data.get('art_candidates') or ())
    return LibraryItem(**data)


# --- rows -------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TitleRow:
    ''' One row of the index, decoded. See the schema for what each field is. '''
    id: str
    unit: str
    source: str
    item_id: str
    members: Tuple[str, ...]
    path: str
    display_name: str
    title: str
    year: str
    kind: str
    season: str
    episodes: Tuple[int, ...]
    external_ids: Dict[str, str]
    fingerprint: str
    first_seen_generation: int
    last_seen: float
    also_in: Tuple[str, ...]
    shadowed_by: str
    ignored: str
    gone: bool
    duplicates: Tuple[str, ...]
    in_catalogue: bool
    extract_state: str
    design_state: str
    review_state: str
    publish_state: str
    commit_state: str
    needs: str
    tier: str
    detail: str
    state_since: float
    confidence: Optional[float]
    candidate_count: int
    failure: str
    is_new: bool = False  # first seen by the latest scan

    @property
    def flags(self) -> List[str]:
        ''' The orthogonal flags that apply (state.FLAG_*), in a fixed order. '''
        return [name for name, on in ((FLAG_IGNORED, bool(self.ignored)), (FLAG_SHADOWED, bool(self.shadowed_by)),
                                      (FLAG_GONE, self.gone), (FLAG_DUPLICATE, bool(self.duplicates)),
                                      (FLAG_IN_CATALOGUE, self.in_catalogue)) if on]

    @classmethod
    def of(cls, row: sqlite3.Row, generation: int) -> 'TitleRow':
        return cls(
            id=row['id'], unit=row['unit'], source=row['source'], item_id=row['item_id'],
            members=tuple(json.loads(row['members'])), path=row['path'], display_name=row['display_name'],
            title=row['title'], year=row['year'], kind=row['kind'], season=row['season'],
            episodes=tuple(json.loads(row['episodes'])), external_ids=json.loads(row['external_ids']),
            fingerprint=row['fingerprint'], first_seen_generation=row['first_seen_generation'],
            last_seen=row['last_seen'], also_in=tuple(json.loads(row['also_in'])), shadowed_by=row['shadowed_by'],
            ignored=row['ignored'], gone=bool(row['gone']), duplicates=tuple(json.loads(row['duplicates'])),
            in_catalogue=bool(row['in_catalogue']), extract_state=row['extract_state'],
            design_state=row['design_state'], review_state=row['review_state'], publish_state=row['publish_state'],
            commit_state=row['commit_state'], needs=row['needs'], tier=row['tier'], detail=row['detail'],
            state_since=row['state_since'], confidence=row['confidence'], candidate_count=row['candidate_count'],
            failure=row['failure'], is_new=row['first_seen_generation'] == generation and generation > 0)


@dataclass(frozen=True)
class SourceRow:
    name: str
    position: int
    kind: str
    last_scanned: Optional[float]   # when it was last listed (or attempted)
    last_ok: Optional[float]        # when it was last listed successfully
    last_error: str                 # '' if the last attempt worked; else why it did not (its titles are kept as they were)
    item_count: int


@dataclass(frozen=True)
class ScanResult:
    generation: int
    titles: int
    new: List[str] = field(default_factory=list)          # ids first seen by this scan
    gone: List[str] = field(default_factory=list)         # ids that left their source, and were kept for their outputs
    dropped: List[str] = field(default_factory=list)      # ids that left their source with no outputs, and were removed
    errors: Dict[str, str] = field(default_factory=dict)  # source name -> why it could not be listed
    counts: Dict[str, int] = field(default_factory=dict)  # needs -> titles
    superseded: List[str] = field(default_factory=list)   # ids now grouped under another row (tv_mode changed), kept for a person


@dataclass(frozen=True)
class IndexSummary:
    generation: int
    last_scan_at: Optional[float]
    titles: int
    counts: Dict[str, int]      # needs -> titles, every needs present (0 if none)
    new: int                    # titles first seen by the latest scan
    flags: Dict[str, int]       # state.FLAG_* -> titles
    sources: List[SourceRow]


# --- the index ----------------------------------------------------------------------------------------------------------

class IndexFileError(sqlite3.DatabaseError):
    ''' The file is not this index, or is one of another version, and is not to be overwritten or migrated by this call. '''


def _casefold(value: Any) -> Any:
    return value.casefold() if isinstance(value, str) else value


def _ascii_lower(text: str) -> str:
    ''' What SQLite's COLLATE NOCASE compares by: only A-Z fold. '''
    return ''.join(chr(ord(c) + 32) if 'A' <= c <= 'Z' else c for c in text)


class LibraryIndex:
    def __init__(self, path: str, *, readonly: bool = False):
        '''
        :param path: the SQLite file (or ':memory:'). Created if missing. A file that is not a database at all, or that
            is this index with another SCHEMA_VERSION, is discarded and started afresh: the index is a cache, and
            `generation == 0` says it has never been scanned. A SQLite file that is *not* this index (none of its
            tables) is left alone: it raises IndexFileError.
        :param readonly: open an existing index for reading only -- it never creates, migrates or drops anything, and
            raises IndexFileError if the file is not an index of this version (for `status`).
        '''
        self.path = path
        self.__lock = threading.RLock()            # the connection: held only for the moment of a query or write
        self.__scan_guard = threading.RLock()      # one scan (or rebuild) at a time; never held by a reader
        if path != ':memory:' and not readonly:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.__db = self.__open_readonly() if readonly else self.__open()

    def __enter__(self) -> 'LibraryIndex':
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        with self.__lock:
            self.__db.close()

    def __connect(self, target: Optional[str] = None, uri: bool = False) -> sqlite3.Connection:
        db = sqlite3.connect(target or self.path, timeout=30, check_same_thread=False, uri=uri)
        db.row_factory = sqlite3.Row
        db.create_function('casefold', 1, _casefold, deterministic=True)  # what the work list's search box folds with
        return db

    def __open_readonly(self) -> sqlite3.Connection:
        if self.path != ':memory:' and not os.path.isfile(self.path):
            raise IndexFileError(f'no index at {self.path}')
        try:
            db = self.__connect(Path(self.path).resolve().as_uri() + '?mode=ro', uri=True) \
                if self.path != ':memory:' else self.__connect()
            version = db.execute('PRAGMA user_version').fetchone()[0]
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        except sqlite3.DatabaseError as error:
            raise IndexFileError(f'{self.path} is not a readable index: {error}')
        if version != SCHEMA_VERSION or not set(_TABLES) <= tables:
            db.close()
            raise IndexFileError(f'{self.path} is not an index of this version (schema {version}, want '
                                 f'{SCHEMA_VERSION}): run `scan`')
        return db

    def __open(self) -> sqlite3.Connection:
        db = self.__connect()
        try:
            version = db.execute('PRAGMA user_version').fetchone()[0]
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        except sqlite3.DatabaseError:  # not a database at all
            db.close()
            logger.warning('%s is not a readable index; starting again', self.path)
            if os.path.isfile(self.path):
                os.remove(self.path)
            db = self.__connect()
            version, tables = 0, set()
        ours = tables & set(_TABLES)
        if tables and not ours:   # somebody else's database: do not drop its tables, do not add ours to it
            db.close()
            raise IndexFileError(f'{self.path} is a SQLite database but not a library index (it has tables '
                                 f'{", ".join(sorted(tables))}); refusing to overwrite it')
        if version != SCHEMA_VERSION or not set(_TABLES) <= tables:
            if ours:
                logger.info('index %s has schema version %s, want %s; dropping it (it is a cache)', self.path,
                            version, SCHEMA_VERSION)
            with db:
                for name in ours:  # only our own tables
                    db.execute(f'DROP TABLE IF EXISTS "{name}"')
                db.executescript(SCHEMA)
                db.execute(f'PRAGMA user_version = {SCHEMA_VERSION}')
                db.execute("INSERT INTO meta VALUES ('generation', '0')")
        return db

    # meta ---------------------------------------------------------------------------------------------------------

    def __meta(self, key: str, default: str = '') -> str:
        row = self.__db.execute('SELECT value FROM meta WHERE key = ?', (key,)).fetchone()
        return row[0] if row else default

    def __set_meta(self, key: str, value: Any) -> None:
        self.__db.execute('INSERT OR REPLACE INTO meta VALUES (?, ?)', (key, str(value)))

    @property
    def generation(self) -> int:
        ''' How many scans have run; 0 means never scanned (a new, reset or rebuilt index). '''
        with self.__lock:
            return int(self.__meta('generation', '0'))

    # queries ------------------------------------------------------------------------------------------------------

    def titles(self, *, needs: Optional[Sequence[str] | str] = None, tier: Optional[str] = None,
               source: Optional[str] = None, match: Optional[str] = None, ids: Optional[Iterable[str]] = None,
               new_only: bool = False, include_done: bool = True) -> List[TitleRow]:
        '''
        The work list: attention first, then human, machine and done, each oldest `state_since` first (then by title).
        :param needs: one of state.NEEDS, or several.
        :param match: case-insensitive text found in the title, display name, id or path.
        :param new_only: only titles first seen by the latest scan.
        :param include_done: False leaves out `done` (ignored, shadowed, gone, skipped, rejected and pushed titles).
        '''
        where, args = [], []
        if needs is not None:
            wanted = [needs] if isinstance(needs, str) else list(needs)
            where.append(f'needs IN ({",".join("?" * len(wanted))})')
            args += wanted
        if tier is not None:
            where.append('tier = ?')
            args.append(tier)
        if source is not None:
            where.append('source = ?')
            args.append(source)
        if match:
            # instr over casefold(), not LIKE: LIKE folds only ASCII, and the work list's search box folds with casefold
            where.append('(instr(casefold(title), ?) > 0 OR instr(casefold(display_name), ?) > 0 '
                         'OR instr(casefold(id), ?) > 0 OR instr(casefold(path), ?) > 0)')
            args += [match.casefold()] * 4
        if not include_done:
            where.append("tier != 'done'")
        unique_ids = None if ids is None else list(dict.fromkeys(ids))
        batches = [None] if unique_ids is None else \
            [unique_ids[i:i + _ID_BATCH] for i in range(0, len(unique_ids), _ID_BATCH)] or [[]]
        with self.__lock:
            generation = self.generation
            if new_only:
                where.append('first_seen_generation = ?')
                args.append(generation)
            tier_order = ' '.join(f"WHEN '{t}' THEN {r}" for t, r in TIER_ORDER.items())
            rows = []
            for batch in batches:   # a long list of ids is looked up in batches (SQLite limits the variables in a query)
                clause = where + ([f'id IN ({",".join("?" * len(batch))})'] if batch is not None else [])
                rows += self.__db.execute(
                    f'SELECT * FROM titles {"WHERE " + " AND ".join(clause) if clause else ""} '
                    f'ORDER BY CASE tier {tier_order} END, state_since, title COLLATE NOCASE, id',
                    args + (batch or [])).fetchall()
        if len(batches) > 1:   # each batch came back in order; put them in one order
            rows.sort(key=lambda r: (TIER_ORDER[r['tier']], r['state_since'], _ascii_lower(r['title']), r['id']))
        return [TitleRow.of(row, generation) for row in rows]

    def title(self, title_id: str) -> Optional[TitleRow]:
        found = self.titles(ids=[title_id])
        return found[0] if found else None

    def units(self, ids: Iterable[str]) -> Dict[str, Unit]:
        '''
        The work units the given titles stand for, rebuilt from what the last scan listed (the `items` column): a
        LibraryItem for an `item` row and a SeasonGroup for a `season` row, exactly what a run of that scan's listing
        would have worked on. This is what lets `run_stages` do its work without listing any source again. A title
        the index does not hold, or one with no items (a row rebuilt from outputs alone), is left out.
        '''
        wanted = list(dict.fromkeys(ids))
        found: Dict[str, Unit] = {}
        for start in range(0, len(wanted), 500):
            batch = wanted[start:start + 500]
            with self.__lock:
                rows = self.__db.execute(
                    f'SELECT id, unit, items FROM titles WHERE id IN ({",".join("?" * len(batch))})', batch).fetchall()
            for row in rows:
                items = [item_from_json(data) for data in json.loads(row['items'])]
                if not items:
                    continue
                if row['unit'] == 'season':
                    (group,) = plan_units(items, 'season', lambda _first, season_id=row['id']: season_id)
                    if not isinstance(group, SeasonGroup):
                        raise ValueError(f"{row['id']} is a season row whose episodes do not form a season")
                    found[row['id']] = group
                else:
                    found[row['id']] = items[0]
        return found

    def sources(self) -> List[SourceRow]:
        with self.__lock:
            return [SourceRow(**dict(row)) for row in
                    self.__db.execute('SELECT * FROM sources ORDER BY position').fetchall()]

    def summary(self) -> IndexSummary:
        with self.__lock:
            generation = self.generation
            counts = {needs: 0 for needs in NEEDS}
            counts.update({r['needs']: r['n'] for r in
                           self.__db.execute('SELECT needs, COUNT(*) AS n FROM titles GROUP BY needs')})
            flag = self.__db.execute(
                "SELECT SUM(ignored != ''), SUM(shadowed_by != ''), SUM(gone), SUM(duplicates != '[]'), "
                "SUM(in_catalogue), SUM(first_seen_generation = ?) FROM titles", (generation,)).fetchone()
            last = self.__meta('last_scan_at')
            return IndexSummary(
                generation, float(last) if last else None, sum(counts.values()), counts,
                int(flag[5] or 0) if generation else 0,
                dict(zip((FLAG_IGNORED, FLAG_SHADOWED, FLAG_GONE, FLAG_DUPLICATE, FLAG_IN_CATALOGUE),
                         (int(v or 0) for v in flag[:5]))), self.sources())

    # failure memory -----------------------------------------------------------------------------------------------

    def record_failure(self, title_id: str, stage: str, message: str, fingerprint: str, key: str,
                       at: Optional[float] = None) -> None:
        '''
        Remembers that `stage` (`extract` or `design`) failed for a title against this source fingerprint and these
        settings (status.failure_key()). It is remembered only while both are unchanged, so a changed source or
        setting retries it, and nothing else does except clear_failure() (Retry failed).
        '''
        if stage not in ('extract', 'design'):
            raise ValueError(f"stage must be 'extract' or 'design', got {stage!r}")
        with self.__lock, self.__db:
            self.__db.execute('INSERT OR REPLACE INTO failures VALUES (?, ?, ?, ?, ?, ?)',
                              (title_id, stage, message, fingerprint, key, time.time() if at is None else at))

    def clear_failure(self, *title_ids: str) -> None:
        ''' Forgets the failures of the given titles, or of every title if none is named. '''
        with self.__lock, self.__db:
            if title_ids:
                self.__db.executemany('DELETE FROM failures WHERE id = ?', [(i,) for i in title_ids])
            else:
                self.__db.execute('DELETE FROM failures')

    def failure(self, title_id: str) -> Optional[FailureMemory]:
        with self.__lock:
            row = self.__db.execute('SELECT * FROM failures WHERE id = ?', (title_id,)).fetchone()
        return FailureMemory(row['stage'], row['message'], row['fingerprint'], row['key']) if row else None

    def failures(self) -> Dict[str, FailureMemory]:
        with self.__lock:
            return {r['id']: FailureMemory(r['stage'], r['message'], r['fingerprint'], r['key'])
                    for r in self.__db.execute('SELECT * FROM failures')}

    # scan ---------------------------------------------------------------------------------------------------------

    def __previous(self) -> Dict[str, dict]:
        return {r['id']: dict(r) for r in self.__db.execute('SELECT * FROM titles')}

    @staticmethod
    def __listing_from(previous: Mapping[str, dict], source: str) -> List[LibraryItem]:
        ''' What a source listed last time, for a source that is not being rescanned or could not be listed. '''
        items: List[LibraryItem] = []
        for row in previous.values():
            if row['source'] == source and not row['gone']:
                items += [item_from_json(data) for data in json.loads(row['items'])]
        return items

    def refresh(self, profile: Profile, settings: Optional[ScanSettings] = None, *, now: Optional[float] = None
                ) -> ScanResult:
        '''
        Re-reads the outputs of every title from the *last listing* of each source, without listing any source again:
        the cheap update after a run, a publish or an accept has changed what the titles need. It is not a scan --
        nothing new can appear, the generation (so the new-since-scan marker) and "last scanned" do not move.

        The sources are the ones the index itself recorded at its last scan, **not** the profile's: a profile that names
        its sources differently (an ad-hoc one built for a command line, say) must not make every title look as if its
        source had gone. An index that has recorded none (never scanned, or rebuilt from outputs) is left as it is.
        The profile supplies the ignore rules and per-title ignores, and nothing is ever marked gone or dropped.
        '''
        return self.scan(profile, settings, only=(), now=now, refresh=True)

    def scan(self, profile: Profile, settings: Optional[ScanSettings] = None, *,
             only: Optional[Iterable[str]] = None, sources: Optional[Mapping[str, LibrarySource]] = None,
             now: Optional[float] = None, refresh: bool = False, allow_empty: bool = False) -> ScanResult:
        '''
        Lists the sources, merges them, reads the outputs and rewrites the index. Never extracts, designs, publishes or
        commits, and never reads a media file (design.md §12.5).

        A source that cannot be listed keeps the titles it had, and its `last_error` says why -- one source being down
        must not make its titles vanish. So does a source that lists **nothing** when it listed some last time (an
        unmounted share matches no files and raises nothing): the previous listing is kept and the scan reports it,
        unless `allow_empty`. `only` rescans just those sources (by name) and leaves the rest as they were.

        Sources are listed **without** holding the index's lock, so a reader (the work list, `status`) is never kept
        waiting for a slow source: it sees the last scan until this one has finished. Two scans do not overlap.

        :param settings: default ScanSettings.from_profile(profile); it must match what `run`/`publish` are given.
        :param sources: already-built sources by name (tests, or a caller that has them); any the profile names that
            are not here are built from their settings.
        :param refresh: see refresh(): keep the generation and the "last scan" time.
        :param allow_empty: accept a source's empty listing as the truth even though it listed some titles before.
        '''
        settings = settings or ScanSettings.from_profile(profile)
        now = time.time() if now is None else now
        rescan = set() if refresh else None if only is None else set(only)
        built = dict(sources or {})
        with self.__scan_guard:
            with self.__lock:   # what the last scan left: a moment of local reads
                previous = self.__previous()
                generation = self.generation + (0 if refresh else 1)
                failures = self.failures()
                recorded = self.sources()
                source_state: Dict[str, Tuple[Optional[float], Optional[float], str, int]] = {
                    s.name: (s.last_scanned, s.last_ok, s.last_error, s.item_count) for s in recorded}
                known_xml = {r['path']: XmlRecord(r['mtime_ns'], r['size'], r['tmdb'], bool(r['is_tv']))
                             for r in self.__db.execute('SELECT * FROM repo_xml')}
            if refresh:
                if not recorded:
                    return self.__unchanged()
                specs = [SourceSpec(row.name, row.kind) for row in recorded]
            else:
                specs = list(profile.sources)

            errors: Dict[str, str] = {}
            listed_ok: set = set()
            listings: List[Tuple[str, List[LibraryItem]]] = []
            for spec in specs:
                scanned, ok, error, count = source_state.get(spec.name, (None, None, '', 0))
                items: Optional[List[LibraryItem]] = None
                if rescan is None or spec.name in rescan:
                    try:
                        source = built.get(spec.name) or build_source(spec.kind, spec.settings)
                        items = list(source.list_items())
                    except Exception as failure:  # a source that is down must not take its titles with it
                        logger.warning('scan: source %r could not be listed: %s', spec.name, failure)
                        errors[spec.name] = f'{type(failure).__name__}: {failure}'
                    else:
                        if not items and count and not allow_empty:
                            errors[spec.name] = (f'listed 0 items (previously {count}); keeping the previous listing '
                                                 f'(an unmounted share looks like this; --allow-empty if it really is empty)')
                            logger.warning('scan: source %r %s', spec.name, errors[spec.name])
                            items = None
                    if items is not None:
                        scanned, ok, error, count = now, now, '', len(items)
                        listed_ok.add(spec.name)
                    else:
                        scanned, error = now, errors[spec.name]
                if items is None:
                    items = self.__listing_from(previous, spec.name)
                source_state[spec.name] = (scanned, ok, error, count)
                listings.append((spec.name, items))

            claims = reconstruct_claims(settings.work_dir, settings.queue_dir)
            union = union_of(listings, ignore=profile.ignore, ignored_titles=profile.ignored_titles, claims=claims)
            units = plan_units([t.item for t in union.active()], settings.tv_mode, claims.season_id)
            by_id = {t.id: t for t in union.titles}
            member_ids = {m.id for u in units for m in (u.members if isinstance(u, SeasonGroup) else (u,))}
            left_out = {t.id for t in union.active()} - member_ids   # e.g. a second copy of an episode, in season mode
            ignored_titles = [t for t in union.titles if t.ignored]

            records = scan_xml_repo(settings.xml_repo, known_xml)
            own = frozenset(claims.ids) | frozenset(by_id) | {(u.item if isinstance(u, SeasonGroup) else u).id
                                                             for u in units}
            evaluator = Evaluator(settings, xml_index=tmdb_index(records), own_ids=own)

            rows: Dict[str, dict] = {}
            new_ids, cleared = [], []
            order: List[Tuple[Unit, str]] = []
            for unit in units:
                item = unit.item if isinstance(unit, SeasonGroup) else unit
                # a per-title ignore of a season names the row the work list shows, which exists only after grouping
                order.append((unit, ignore_label(profile.ignored_titles[item.id])
                              if isinstance(unit, SeasonGroup) and item.id in profile.ignored_titles else ''))
            order += [(t.item, t.ignored) for t in ignored_titles]
            for unit, ignored in order:
                item = unit.item if isinstance(unit, SeasonGroup) else unit
                before = previous.get(item.id)
                evaluation = evaluator.evaluate(unit, before, failures.get(item.id))
                if evaluation.clear_failure:
                    cleared.append(item.id)
                if isinstance(unit, SeasonGroup):  # an episode's own failure (it is not a row) lapses like any other
                    cleared += [m.id for m in unit.members if m.id in failures and not failure_applies(
                        failures[m.id], m, safe_fingerprint(m), config=settings.config, designer=settings.designer,
                        coverage=settings.coverage, keep_multichannel=settings.keep_multichannel)]
                rows[item.id] = self.__unit_row(unit, item, evaluation, ignored, by_id, before, generation, now,
                                                listed_ok, left_out)
                if before is None:
                    new_ids.append(item.id)
            for shadow in union.shadowed:
                if shadow.item.id in rows:
                    continue
                before = previous.get(shadow.item.id)
                rows[shadow.item.id] = self.__shadow_row(shadow, before, generation, now, listed_ok)
                if before is None:
                    new_ids.append(shadow.item.id)

            row_of_item = {data['id']: row_id for row_id, row in rows.items() for data in json.loads(row['items'])}
            gone, dropped, superseded = [], [], []
            outputs = claims.ids
            for title_id, before in previous.items():
                if title_id in rows:
                    continue
                if refresh:   # a refresh only re-reads outputs: whatever it did not re-emit stays exactly as it was
                    rows[title_id] = dict(before)
                    continue
                new_owner = self.__superseded_by(before, row_of_item, title_id)
                if new_owner:  # its items are still listed, now grouped under another row (tv_mode changed)
                    kept = self.__superseded_row(before, new_owner, evaluator, settings, now)
                    if kept is not None:
                        rows[title_id] = kept
                        superseded.append(title_id)
                    else:
                        dropped.append(title_id)  # no review or decision to keep; its extraction serves the new row
                elif title_id in outputs:
                    rows[title_id] = self.__gone_row(before, now)
                    if not before['gone']:
                        gone.append(title_id)
                else:
                    dropped.append(title_id)

            with self.__lock:
                with self.__db:
                    self.__db.execute('DELETE FROM titles')
                    self.__db.executemany(
                        f'INSERT INTO titles ({",".join(_TITLE_COLUMNS)}) VALUES ({",".join("?" * len(_TITLE_COLUMNS))})',
                        [[row[c] for c in _TITLE_COLUMNS] for row in rows.values()])
                    # only the failures as this scan read them: one recorded meanwhile, by a run, is a newer fact
                    self.__db.executemany('DELETE FROM failures WHERE id = ? AND fingerprint = ? AND key = ?',
                                          [(i, failures[i].fingerprint, failures[i].key) for i in cleared
                                           if i in failures])
                    self.__db.execute('DELETE FROM repo_xml')
                    self.__db.executemany('INSERT INTO repo_xml VALUES (?, ?, ?, ?, ?)',
                                          [(p, r.mtime_ns, r.size, r.tmdb, int(r.is_tv)) for p, r in records.items()])
                    self.__db.execute('DELETE FROM sources')
                    self.__db.executemany('INSERT INTO sources VALUES (?, ?, ?, ?, ?, ?, ?)', [
                        (spec.name, position, spec.kind, *source_state[spec.name])
                        for position, spec in enumerate(specs)])
                    if not refresh:
                        self.__set_meta('generation', generation)
                        self.__set_meta('last_scan_at', now)
            counts = {needs: 0 for needs in NEEDS}
            for row in rows.values():
                counts[row['needs']] += 1
        return ScanResult(generation, len(rows), new_ids, gone, dropped, errors=errors, counts=counts,
                          superseded=superseded)

    def __unchanged(self) -> ScanResult:
        ''' What a scan that did nothing reports: the index as it is. '''
        summary = self.summary()
        return ScanResult(summary.generation, summary.titles, counts=dict(summary.counts))

    @staticmethod
    def __superseded_by(before: dict, row_of_item: Mapping[str, str], title_id: str) -> str:
        ''' The row that now holds one of this row's items, or '' if none of them is listed any more. '''
        if before['gone']:
            return ''
        for data in json.loads(before['items']):
            owner = row_of_item.get(data['id'])
            if owner and owner != title_id:
                return owner
        return ''

    def __superseded_row(self, before: dict, new_owner: str, evaluator: Evaluator, settings: ScanSettings,
                         now: float) -> Optional[dict]:
        '''
        A row whose items are still listed but now grouped differently (an episode row after `tv_mode` became `season`,
        or a season row after it became `episode`). With a queue entry it is a review or decision a person may not have
        made yet, so it stays, read from its outputs like any title, and says what replaced it -- neither *gone* nor
        hidden from the tier it needs. With none, there is nothing to lose: None.
        '''
        if not settings.queue_dir or not os.path.isfile(os.path.join(settings.queue_dir, f"{before['id']}.json")):
            return None
        evaluation = evaluator.evaluate_from_outputs(before['id'], before)
        if evaluation is None:
            return None
        needs = derive_needs(evaluation.states)
        needs = replace(needs, detail=f'{needs.detail} - superseded by {new_owner}' if needs.detail
                        else f'superseded by {new_owner}')
        return self.__row(
            before['id'], evaluation, needs,
            state_since=before['state_since'] if before['needs'] == needs.needs else now,
            first_seen_generation=before['first_seen_generation'], last_seen=before['last_seen'], unit=before['unit'],
            source=before['source'], item_id=before['item_id'], members=json.loads(before['members']),
            path=before['path'], display_name=before['display_name'], kind=before['kind'], season=before['season'],
            episodes=json.loads(before['episodes']), external_ids=json.loads(before['external_ids']),
            items=json.loads(before['items']), also_in=[], shadowed_by='', ignored='', gone=0, duplicates=[])

    def rebuild_from_outputs(self, settings: ScanSettings, *, now: Optional[float] = None) -> int:
        '''
        Recreates the titles of a lost or reset index from the outputs alone -- every queue entry, its extract
        manifest, its projects and the state of the repos -- without listing any source. The states come out as a
        scan would have put them, provided the sources have not changed; the next scan puts right anything that
        has, and gives each title its source.

        What is lost, and accepted: `state_since` restarts at `now` for every title; nothing knows a title's source
        until the next scan; and a title with no queue entry (extracted but not yet designed, or not started) has no
        output to rebuild from, so it reappears only at the next scan. On an index that is *live* it starts over the
        same way: the generation goes back to 0 and the recorded sources and "last scan" are forgotten, so that the
        next command that works from the index (`run --needs ...`) scans first rather than select from rows that have
        no items to work on.
        :return: the number of titles rebuilt.
        '''
        now = time.time() if now is None else now
        with self.__scan_guard:
            with self.__lock:
                previous = self.__previous()
                known_xml = {r['path']: XmlRecord(r['mtime_ns'], r['size'], r['tmdb'], bool(r['is_tv']))
                             for r in self.__db.execute('SELECT * FROM repo_xml')}
            records = scan_xml_repo(settings.xml_repo, known_xml)
            claims = reconstruct_claims(settings.work_dir, settings.queue_dir)
            evaluator = Evaluator(settings, xml_index=tmdb_index(records), own_ids=frozenset(claims.ids))
            rows: Dict[str, dict] = {}
            for entry_id in sorted(claims.ids):
                evaluation = evaluator.evaluate_from_outputs(entry_id, previous.get(entry_id))
                if evaluation is None:
                    continue
                needs = derive_needs(evaluation.states)
                rows[entry_id] = self.__row(
                    entry_id, evaluation, needs, state_since=now, first_seen_generation=0, last_seen=now,
                    unit='item', source='', item_id=entry_id, members=[], path='', display_name=evaluation.title,
                    kind='tv' if evaluation.facts and evaluation.facts.meta.get('season') else 'movie',
                    season=str((evaluation.facts.meta.get('season') if evaluation.facts else '') or ''),
                    episodes=list(evaluation.facts.meta.get('episodes') or []) if evaluation.facts else [],
                    external_ids={}, items=[], also_in=[], shadowed_by='', ignored='', gone=0, duplicates=[])
            with self.__lock, self.__db:
                self.__db.execute('DELETE FROM titles')
                self.__db.executemany(
                    f'INSERT INTO titles ({",".join(_TITLE_COLUMNS)}) VALUES ({",".join("?" * len(_TITLE_COLUMNS))})',
                    [[row[c] for c in _TITLE_COLUMNS] for row in rows.values()])
                self.__db.execute('DELETE FROM repo_xml')
                self.__db.executemany('INSERT INTO repo_xml VALUES (?, ?, ?, ?, ?)',
                                      [(p, r.mtime_ns, r.size, r.tmdb, int(r.is_tv)) for p, r in records.items()])
                self.__db.execute('DELETE FROM sources')
                self.__db.execute("DELETE FROM meta WHERE key = 'last_scan_at'")
                self.__set_meta('generation', 0)
        return len(rows)

    # row building -------------------------------------------------------------------------------------------------

    @staticmethod
    def __row(title_id: str, evaluation: Evaluation, needs, *, state_since: float, first_seen_generation: int,
              last_seen: float, **fields: Any) -> dict:
        states = evaluation.states
        facts = evaluation.facts
        row = dict(
            id=title_id, fingerprint=evaluation.fingerprint, title=evaluation.title, year=evaluation.year,
            in_catalogue=int(evaluation.in_catalogue), extract_state=states.extract, design_state=states.design,
            review_state=states.review, publish_state=states.publish, commit_state=states.commit,
            needs=needs.needs, tier=needs.tier, detail=needs.detail, state_since=state_since,
            first_seen_generation=first_seen_generation, last_seen=last_seen, confidence=states.confidence,
            candidate_count=states.candidates, failure=states.failure,
            entry_summary=facts.to_json() if facts is not None else '', digest_key=evaluation.digest_key,
            current_digest=evaluation.current_digest, conflict=int(states.project_conflict))
        for name, value in fields.items():
            row[name] = json.dumps(value) if name in ('members', 'episodes', 'external_ids', 'items', 'also_in',
                                                     'duplicates') else value
        return row

    def __unit_row(self, unit: Unit, item: LibraryItem, evaluation: Evaluation, ignored: str,
                   by_id: Mapping[str, UnionTitle], before: Optional[dict], generation: int, now: float,
                   listed_ok: set, left_out: set) -> dict:
        season = isinstance(unit, SeasonGroup)
        members = list(unit.members) if season else [item]
        owners = [by_id[m.id] for m in members if m.id in by_id]
        first = owners[0] if owners else None
        also_in = list(dict.fromkeys(s for o in owners for s in o.also_in))
        member_ids = {m.id for m in members}
        duplicates = list(dict.fromkeys(d for o in owners for d in o.duplicates if d not in member_ids))
        # in season mode a second copy of an episode is dropped from the season (it is no row of its own): say so here
        extra = [d for d in duplicates if d in left_out]
        duplicates = [d for d in duplicates if d not in left_out]
        states = evaluation.states
        if ignored:
            states = replace(states, ignored=ignored)
            evaluation.states = states
        needs = derive_needs(states)
        if extra:
            needs = replace(needs, detail=f'{needs.detail} - left out of the season (a second copy of an episode): '
                                          f'{", ".join(extra)}' if needs.detail else
                            f'left out of the season (a second copy of an episode): {", ".join(extra)}')
        since = before['state_since'] if before is not None and before['needs'] == needs.needs else now
        source = first.source if first else ''
        return self.__row(
            item.id, evaluation, needs, state_since=since,
            first_seen_generation=before['first_seen_generation'] if before is not None else generation,
            last_seen=self.__last_seen(before, source, listed_ok, now), unit='season' if season else 'item',
            source=source, item_id=members[0].id, members=[m.id for m in members] if season else [],
            path=item.source_path, display_name=item.display_name, kind=item.kind, season=item.season or '',
            episodes=list(item.episodes), external_ids=item.external_ids, items=[item_to_json(m) for m in members],
            also_in=also_in, shadowed_by='', ignored=ignored, gone=0, duplicates=duplicates)

    @staticmethod
    def __last_seen(before: Optional[dict], source: str, listed_ok: set, now: float) -> float:
        ''' `now` only if the title's source was actually listed this time; a source that was down did not see it. '''
        if before is None or source in listed_ok:
            return now
        return before['last_seen']

    def __shadow_row(self, shadow, before: Optional[dict], generation: int, now: float, listed_ok: set) -> dict:
        item, source, owner = shadow.item, shadow.source, shadow.owner
        evaluation = Evaluation(StageStates(shadowed_by=owner), safe_fingerprint(item), title=item.title or
                                item.display_name or '', year=item.year or '')
        needs = derive_needs(evaluation.states)
        if shadow.claimed:  # a copy that already has outputs of its own must not be hidden without saying so
            needs = replace(needs, detail=f'{needs.detail} (it has its own review or outputs: see both)')
        return self.__row(
            item.id, evaluation, needs,
            state_since=before['state_since'] if before is not None and before['needs'] == needs.needs else now,
            first_seen_generation=before['first_seen_generation'] if before is not None else generation,
            last_seen=self.__last_seen(before, source, listed_ok, now), unit='item', source=source, item_id=item.id,
            members=[], path=item.source_path, display_name=item.display_name, kind=item.kind,
            season=item.season or '', episodes=list(item.episodes), external_ids=item.external_ids,
            items=[item_to_json(item)], also_in=[], shadowed_by=owner, ignored='', gone=0, duplicates=[])

    @staticmethod
    def __gone_row(before: dict, now: float) -> dict:
        ''' A title that has left its source but has outputs: as it was, done, and labelled. '''
        row = {c: before[c] for c in _TITLE_COLUMNS}
        needs = derive_needs(StageStates(gone=True))
        row.update(gone=1, needs=needs.needs, tier=needs.tier, detail=needs.detail,
                   state_since=before['state_since'] if before['gone'] else now)
        return row


assert set(TIER_OF_NEEDS) == set(NEEDS)
