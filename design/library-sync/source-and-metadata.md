# Library sync plan -- library source, metadata and artwork

> Part of the library sync plan -- **start at the index**: [`../library-sync-pipeline-plan.md`](../library-sync-pipeline-plan.md).
> Contains §3, §3.1-§3.2. Section numbers are global across the plan; the index maps every `§` to its file.
> Status of what this file describes: **built** (see `status-and-open-items.md` §10 for the exceptions)

## 3. `LibrarySource` abstraction

New package `pipeline/library/`.

```python
# pipeline/library/source.py
@dataclass(frozen=True)
class LibraryItem:
    id: str                      # stable across runs -- also the QueueEntry.id / cache key.
                                  # Must survive a rename/re-scan; a source should prefer its own
                                  # persistent key (e.g. JRiver's Media ID) over a filename-derived one.
    source_path: str             # ffmpeg input: a container file path, or a BDMV root directory
    display_name: str
    title: Optional[str] = None
    year: Optional[str] = None
    kind: str = 'movie'                      # 'movie' or 'tv' -- forwarded to TMDB resolution
    external_ids: dict = field(default_factory=dict)  # {'tmdb': '603', 'imdb': 'tt0133093'} -- whatever
                                              # identifiers the source can supply; empty if it can't. See
                                              # "Metadata resolution" below -- this is what lets a library
                                              # run skip the fuzzy title/year TMDB search entirely.
    audio_stream: int = 0
    playlist_name: Optional[str] = None      # BD only, forwarded to Session.extract()
    art_path: Optional[str] = None           # a local poster/cover file the source already has, if any --
                                              # see "Artwork resolution and override" (3.1.3); unverified
                                              # whether JRiver exposes this until the spike checks
    meta: dict = field(default_factory=dict) # extra BeqMetadata ctor kwargs the source can supply directly
                                              # (e.g. beq-specific fields TMDB has no concept of) -- merged
                                              # into, and losing to, whatever TMDB resolution produces
    fingerprint: str = ''                    # source-specific change marker (see 4.1) -- opaque to callers

class LibrarySource(Protocol):
    def list_items(self, **query) -> Iterable[LibraryItem]: ...
```

Registry mirrors `pipeline.designer.registry` exactly
(`register_source(name, source)` / `get_source(name)`) -- same
in-process-callable pattern. **As built, nothing calls it**: the CLI
(`cli._source()`) and `LibrarySyncDialog` both construct a
`JRiverLibrarySource` directly, and only `test_pipeline_library_source.py`
exercises the registry. It is kept as the seam for a second source
(Kodi/Plex) but is currently unused production code.

### 3.1 JRiver implementation (`pipeline/library/jriver.py`)

The JRiver source is a **configured MCWS browse node**, not a global
library search. The user selects a node in JRiver's browse hierarchy
(for example a curated `Films needing BEQ` view); every file under
that node is the source catalogue for a run. JRiver's own browse rules,
categories, and filtering therefore decide which titles BEQDesigner sees.

Use the PyPI package [`hamcws`](https://pypi.org/project/hamcws/), rather
than adding library methods to `model.jriver.mcws.MediaServer`. The two
clients have different jobs: the existing client is a synchronous,
DSP-specific UI integration; `hamcws` is an async MCWS library client
which already supports the required endpoint. Add a compatible `hamcws`
dependency to `pyproject.toml` (currently `0.2.7`; pin the minor line,
for example `>=0.2.7,<0.3.0`, when this chunk is implemented).

`hamcws.MediaServer.browse_files(base_id, fields)` calls:

```
GET /MCWS/v1/Browse/Files?ID=<browse_node_id>&Action=JSON&Fields=<comma-separated fields>
```

and returns a `list[dict]`. Its default field list includes `Key`,
`Name`, `Media Type`, `Media Sub Type`, `Series`, `Season`,
`Episode`, `Dimensions`, `HDR Format`, and `Duration`. The source
appends:

```
Filename, Year, Date Modified, File Size, Image File, IMDB, TheMovieDB
```

Missing optional fields are absent; only `Filename` and `Key` are
required. The GUI may use `hamcws.MediaServer.browse_children()`, starting
at `-1`, to present a browse-node selector. Persist the selected integer
`browse_node_id`, rather than a renameable display name, alongside the
host, credentials, and HTTPS setting already kept in
`JRIVER_MCWS_CONNECTIONS`. Each run creates a fresh `hamcws` connection;
it does not share the Qt UI's synchronous `requests` client.

`JRiverLibrarySource.list_items()` owns the async boundary: it opens a
`hamcws` connection, awaits `browse_files(self.browse_node_id,
self.FIELDS)`, maps the results, closes in a `finally`, and yields normal
synchronous `LibraryItem` values. A CLI or `QRunnable` caller need not
know about `aiohttp`. Reject calls from an already-running event loop with
a clear error instead of nesting `asyncio.run()`.

| MCWS field | `LibraryItem` field | Rule |
|---|---|---|
| `Key` | `id` | `jriver-<stable hash of server identity>-<Key>`. This is filesystem-safe on Windows as well as POSIX. The raw `Key` is the stable JRiver media identifier; never derive an id from `Filename`. The server identity prevents two servers sharing numeric keys and a cache directory. |
| `Filename` | `source_path` | Required. Pass through to ffmpeg, recognising BDMV roots with `is_bdmv_root()`. The path still has to be locally mounted on the BEQDesigner host. |
| `Name` | `display_name`, `title` | Display fallback; leave `title` unset if empty. |
| `Year` | `year` | Optional string; fuzzy TMDB resolution remains the fallback. **Live-verified:** JRiver answers a request for `Year` with the key `Date (year)`, so both are read (the requested name wins). |
| `Media Sub Type`, `Series`, `Season`, `Episode` | `kind`, `meta` | **As built (live-verified):** `Media Sub Type` decides when set -- `TV Show` is `tv`, anything else `movie`. Without one, `tv` only if `Series`, `Season` *and* `Episode` are all set. (The first version treated any of the three as TV, which classed every film franchise -- a `Series` value -- as television.) `meta['season']` is kept only for `tv` items. |
| `Date Modified`, `File Size` | `fingerprint` | Stable serialisation of the values actually supplied. Leave it empty when both are absent, allowing the extract cache's filesystem-stat fallback. |
| id fields (default `IMDb ID`, `TheMovieDB Movie ID`/`TMDb ID`; TV: `IMDb Series ID`, `TheMovieDB Series ID`) | `external_ids` | Non-empty values (an unset numeric id reads `0`, which is ignored) normalised to `imdb` and `tmdb`. **Configurable per kind** (§11.7): the names above are only defaults, verified against a live library; the first draft's `IMDB`/`TheMovieDB` do not exist there. |
| `Image File` | `art_path` | Only accept a locally readable path. `INTERNAL` means JRiver-managed art, not a file path; downloading `File/GetImage` is a later enhancement if needed. |

Deduplicate exact repeated `Key` rows before yielding; fail the run when
one key has conflicting filenames. A browse node is a source selection, so
every valid unique file reaches the normal extract/design stages.

Tests (`test_pipeline_library_jriver.py`) use hand-written row dicts and a
stubbed `hamcws` `MediaServer`, covering the selected node and field list,
normal mapping, optional fields/TV metadata, id stability across a rename,
distinct server identities, duplicate/conflicting keys, and rejection of a
`query`/running event loop. `Image File` is now turned into `LibraryItem.art_candidates` without touching the disk (chunk 24;
`artwork.resolve_art()` checks them at design time), and `INTERNAL` artwork and the listing's no-`stat` guarantee are tested. **Still outstanding:** the
sanitised *real-server* response fixture -- external-ID aliases
(`IMDB`/`TheMovieDB` and the `IMDb`/`TMDB`/`TMDb` fallbacks in
`DEFAULT_EXTERNAL_ID_FIELDS`) and the `Browse/Children` shape are still
educated guesses, not verified against a live library (chunk 3).

### 3.1.1 Metadata resolution -- pull identity from the library, not a fuzzy search

The user's observation (2026-09-17): beqcatalogue is ultimately driven by
`beq_meta` (`BeqMetadata`/`pipeline.metadata.py`), which today gets
populated one of two ways -- the AVS post builder's manual TMDB search
(`model/postbuilder.py`'s `__search_tmdb`, title/year -> pick a result),
or pasting a known TMDB id directly (`tmdb_details_by_id()`, already
factored out and Qt-free -- `pipeline.metadata.tmdb_details_by_id`).
Neither is wired into `model/batch.py`'s batch design flow at all today
(`QueueEntry.meta` stays `{}` unless a caller passes it in) -- metadata
resolution is currently left entirely for later, manual, AVS-post-builder
work.

A JRiver library commonly already carries the identity that today's fuzzy
`tmdb_lookup(title, year, ...)` search is trying to reconstruct from
scratch -- many metadata plugins (and JRiver's own online lookup) tag a
file with a TMDB and/or IMDB id directly. If so, `JRiverLibrarySource` can
put that id straight into `LibraryItem.external_ids`, and metadata
resolution becomes:

1. `external_ids['tmdb']` present -> `tmdb_details_by_id()` directly.
   Deterministic -- no title-collision risk, no extra search request.
2. Else `external_ids['imdb']` present -> **new**, small addition to
   `pipeline/metadata.py`: `tmdb_find_by_imdb_id(imdb_id, api_key) -> str`
   (TMDB's `/find/{imdb_id}?external_source=imdb_id`, no equivalent exists
   in this codebase yet), then `tmdb_details_by_id()` on the result.
   IMDB ids are the more common tag in practice -- many JRiver metadata
   plugins populate an "IMDB" field but not a TMDB one.
3. Else fall back to today's `tmdb_lookup(item.title, item.year, ...)`
   fuzzy search -- still required regardless, since not every title in a
   library will have an id tagged (manual entries, obscure titles, a
   plugin that only writes some fields).

This resolution (`pipeline/library/library_metadata.py::resolve_meta(item,
api_key, audio_types)`, returning `BeqMetadata` ctor kwargs) runs from
`run_library()` (not from `design_if_needed()`, as originally drafted) **only
when `LibraryRunConfig.tmdb_api_key` is set** -- otherwise `meta` is just
`item.meta`. It is resolved **lazily** (commit `efb300f`):
`design_if_needed()` takes a callable for `meta` and invokes it only when it
actually designs, so a cached or protected item costs no TMDB round-trip. A
`requests.RequestException` degrades to `item.meta` and is recorded in
`LibraryRunReport.meta_unresolved` rather than failing the item. So `QueueEntry.meta` arrives
at the review queue already populated for anything the library could
identify -- closing the metadata gap `model/batch.py` leaves manual today,
not just avoiding a second network round-trip. TMDB stays the source of
truth for the fields beqcatalogue actually needs in canonical form
(genres as TMDB's own `{id, name}` scheme, overview, poster, certification
rating) -- the library's own tags are used only to *identify* the title,
not to replace TMDB's data, since beqcatalogue's genre filtering etc.
is built around TMDB's id space, not any given library tool's own
category tags. `item.meta` (BEQ-specific fields with no TMDB equivalent --
edition, note, warning, season) is merged in on top, losing to nothing
since TMDB has no opinion on them.

### 3.1.2 Reviewer override -- the metadata editor

The user's follow-up (2026-09-17): auto-resolution (3.1.1, or the fuzzy
fallback) can get the match wrong, or simply has nothing to say about
fields no automated source covers at all -- `edition`/`season`/`note`/
`warning`/`gain` override/`language`/`source`/`author`/`avs` have no TMDB
or JRiver equivalent and are, today, filled in by hand in the AVS post
builder only. `ReviewQueueDialog` currently has **no editing capability
whatsoever** -- it only ever displays `entry.meta.get('title', ...)`,
read-only (`model/review.py`). Since `publish_reviewed_queue()` builds
`BeqMetadata(**{**meta_defaults, **entry.meta})` and `validate()` requires
at least title/year/audio_types, and nothing upstream of this plan ever
populated most of those fields for a library-driven run, this is a
**blocking gap** for the sync step to be usable end to end, not optional
polish -- without it a human has no way to either correct a wrong
auto-match or supply the fields nothing automates.

Fix: extend `ReviewQueueDialog`'s detail pane (`model/review.py`/
`ui/review.py`) with an editable metadata form -- the same fields and the
same "paste an id, or fall back to title/year search" pattern
`CreateAVSPostDialog.load_tmdb_info()`/`__apply_tmdb_metadata()` already
implements for the AVS post builder, reused rather than reinvented since
it is already exactly this problem solved once. Any edit persists
via `pipeline.review.update_entry(queue_dir, entry.id, meta={**entry.meta,
**edited_fields})` -- `QueueEntry.meta` is already a plain dict, so this
needs no schema change, just UI wiring onto the existing read-modify-write
helper. The editor is meaningful on a `pending` entry the same way the
existing candidate picker/keyboard triage (Enter/A/digit-keys/S/R) already
is; once an entry is `accepted`/`published` further edits should be
blocked or at least visibly stale, matching how the rest of the dialog
already treats those statuses as settled.

This directly replaces what would otherwise be an open question about
correcting a wrong id-based match (3.1.1) -- it is the same fix, and it
benefits the *existing* manual `model/batch.py` batch-design flow equally,
not just this plan's new library-driven path, since both write into the
same `QueueEntry`/`ReviewQueueDialog`. Worth landing independently of the
JRiver-specific work (it does not depend on anything in 3.1's spike) --
see Chunk 1 / Appendix A below.

### 3.1.3 Artwork resolution and override

The user's follow-up (2026-09-17): the report image needs poster artwork
too, and the same three sources apply -- the library source might already
have it, TMDB can supply it, and the user may want to override either
with their own image, exactly as `SaveReportDialog` (the interactive
report builder) already lets them today via `choose_image()` (browse a
local file) and `load_image_from_url()`/`__download_image()` (paste a URL,
download it).

**Existing gap, independent of this plan**: the headless side of this is
already half-built but never wired end to end.
`pipeline.publish.art.fetch_poster()` (downloads a TMDB `poster_path`
fragment to a local file) and `pipeline.publish.report.compose_with_poster()`/
`render_report(poster_path=...)` (stacks that poster above the chart into
the final PNG) both exist and are unit-tested in isolation
(`test_pipeline_publish_art.py`, `test_pipeline_publish_report.py`) -- but
`pipeline.review.publish_reviewed_queue()` calls `session.report(...)`
**without ever passing `poster_path`**, so every report published through
the review-queue flow today renders chart-only, poster art silently
dropped. This is worth fixing regardless of the library-source work below.

Resolution order (same override-wins shape as 3.1.1's metadata
resolution, and the same "resolve once, cache, never silently clobber a
human's choice" shape as the extract cache in 4.1):

1. **User override** -- checked first, and once set, sticky across
   reruns (never silently replaced). Set via the metadata editor's (3.1.2)
   new Artwork section: browse a local file, or paste a URL and download
   it -- literally the same two actions `SaveReportDialog.choose_image()`/
   `load_image_from_url()` already implement, reused rather than
   reinvented.
2. **Library source**, if it has one -- e.g. JRiver commonly caches a
   local poster/cover file per library item; `LibraryItem` could carry an
   optional `art_path` (a local file JRiver already has on disk, if MCWS
   exposes it) alongside `external_ids`. Same caveat as the TMDB/IMDB id
   question in 3.1.1: entirely dependent on the user's library/plugin
   setup, unverified until the spike checks for it.
3. **TMDB fallback** -- once `resolve_meta()` (3.1.1) has a `poster` path
   fragment, `pipeline.publish.art.fetch_poster()` downloads it. This is
   the piece that already exists but isn't wired -- closing that wiring
   gap *is* this fallback, not new logic.
4. **None** -- `render_report(poster_path=None)` already renders chart-only;
   an acceptable outcome when nothing above produced anything.

Intended: resolved once at design time (alongside metadata resolution)
rather than at publish time, so a TMDB download only ever happens once per
item -- stored as the additive `QueueEntry` fields `art_path` and
`art_overridden` (shipped in chunk 1), so a rerun's auto-resolution
(tiers 2/3) never overwrites a tier-1 human choice. `publish_reviewed_queue()`
passes `poster_path=entry.art_path` (shipped, chunk 1).

**Implementation status -- all four tiers now exist.** Tier 1 (the reviewer's
Browse/Download/Clear controls) and the publish wiring shipped in chunk 1.
Tiers 2 and 3 live in `pipeline/library/artwork.py::resolve_art()`, called
from `design_if_needed()` **only when a design runs**:

- an entry whose `art_overridden` is set, or whose automatic `art_path` still
  exists on disk, is left alone (so a reviewer's choice is never replaced and
  a poster is downloaded once);
- otherwise `LibraryItem.art_path` is used in place if the file is readable
  (tier 2), else TMDB's `poster` fragment is downloaded by `fetch_poster()`
  to `<work_dir>/<item.id>/poster.<ext>` (tier 3; needs the durable
  `project_dir`, so it is skipped without one);
- a failed download is logged and means "no artwork" -- it never fails the
  item.

A cache hit (fingerprint match) does not touch artwork, so an entry designed
before this existed only gains a poster when it is next redesigned. The
dialog's **Clear** sets `art_overridden` back to `False` (it means "return to
automatic", not "no artwork"), so a redesign after a Clear re-resolves; the
schema description was corrected to say so.

### 3.2 Kodi / Plex

Out of scope for this plan -- named only so `LibrarySource` isn't
JRiver-shaped by accident. Kodi's `VideoLibrary.GetMovies` JSON-RPC and
Plex's `/library/sections/<id>/all` REST endpoint are the obvious
future implementations; not stubbed as dead code, just documented here
as the intended shape to implement against later.
