# Library sync pipeline -- plan

Status: **chunks 1-2 and 4-10 implemented (2026-09-18); full suite green
(471 passed, 2026-09-19 review). Chunk 3 (real-server spike) is not done,
and a small set of design items in §3.1.3, §3.3.1 and §4 remain unbuilt --
see §10 "Implementation status vs. this plan" for the authoritative list.**
Written 2026-09-17. Builds on the
headless pipeline in `pipeline/` (see `pipeline/README.md` and
`design/api-headless-pipeline.md`/`pipeline-implementation-plan.md`,
all shipped) and the existing GUI Batch Extract & Design workflow
(`model/batch.py`).

## 1. Goal

An end-to-end, source-agnostic workflow:

> point at a library of films -> idempotently extract audio for
> everything new -> idempotently design filters for everything that
> needs it -> a human reviews the queue as today -> sync accepted
> entries to a beqcatalogue XML+images repo pair.

The "library" side is pluggable. One implementation ships now (JRiver
Media Center, via MCWS); Kodi/Plex are named as future implementations
of the same interface, not built here.

There are three distinct outputs, and the plan must cover all three
(the user's clarification, 2026-09-18 -- output 1 was under-specified
in earlier drafts of this plan and is fleshed out in §3.3 below):

1. **Local artifacts, per title** -- the extracted audio file(s), the
   designer's raw output (already covered: `QueueEntry`'s
   `candidates[].filters`), and a beqdesigner `.beq` project file
   linking it all together for opening/tweaking in the interactive app
   -- **one project for the mono track**, and, whenever the kept
   extraction is multichannel, **a second project for the multichannel
   track** with the designed filter linked (`SignalData.enslave()`)
   across every channel and the LFE channel correctly identified (the
   `_LFE` name suffix `model/signal.py` already keys bass-management
   behaviour off). Not published anywhere -- these live on local disk
   alongside the extracted wav.
2. **beqcatalogue's filter (XML) repo** -- unchanged, `publish_reviewed_queue()`.
3. **beqcatalogue's image repo** -- unchanged, `publish_reviewed_queue()`.

Outputs 2 and 3 are the "sync" step (§4.3/§6); output 1 is produced
earlier, as part of extract+design (§3.3), and does not depend on
anything ever being accepted or synced -- a human may want to open a
title's project in the full interactive app as part of *deciding*
whether to accept it, not only after.

Three decisions already made (2026-09-17, asked of the user up front
since each reshapes the design):

| Decision | Resolution |
|---|---|
| Review gate | **Kept.** Extract+design only ever populate the review queue; publishing to beqcatalogue still requires a human to mark an entry `accepted` first (`pipeline.review.publish_reviewed_queue`, unchanged). A "sync" run is a separate, explicit step from a "run library" (extract+design) run -- never bundled, so an unattended nightly job has nothing to auto-publish. |
| Entry point | **Both**, CLI-callable function first, GUI wraps it -- same layering `pipeline.review.batch_design()` / `model/batch.py` already use. |
| JRiver source | **Browse a configured MCWS node**, not filesystem glob or an ad-hoc search -- JRiver's browse rules define the source catalogue and return per-title metadata. `hamcws` provides the required `Browse/Files` client; only the user's configured field aliases remain to verify. |

## 2. What already exists (reuse, don't rebuild)

| Need | Existing piece |
|---|---|
| Extract one file/BD folder to wav | `pipeline.orchestrate.Session.extract()` |
| Design + write one queue entry | `pipeline.review.design_and_queue()` |
| Batch design over many items (extract+design, no cache) | `pipeline.review.batch_design()` -- **the "bulk extract+design" the user means**; this plan adds idempotency and a pluggable item source in front of it, it does not replace it |
| Review queue storage/format | `pipeline.review.QueueEntry`/`read_queue`/`update_entry`, `docs/schema/review_queue.schema.json` |
| Human review UI | `model/review.py`/`ui/review.py` `ReviewQueueDialog` (embeddable, keyboard-first triage) |
| Publish accepted entries to beqcatalogue (XML + images repos) | `pipeline.review.publish_reviewed_queue()` -- **already idempotent**: only touches `status='accepted'` entries, marks them `published`, a re-run skips anything already `published`. This *is* the sync step; no new publish logic needed, just wiring/config and (small) GUI exposure of `images_repo` (README notes the GUI currently only wires XML) |
| BD folder -> main-feature resolution | `model.bdmv.is_bdmv_root()`/`resolve_main_title()` |
| JRiver MCWS client | [`hamcws`](https://pypi.org/project/hamcws/) -- its async `MediaServer.browse_files()` already wraps MCWS `Browse/Files` and returns JSON dictionaries. The existing `model.jriver.mcws.MediaServer` remains the synchronous DSP/zone client; it is not the library client. |
| Existing library "search" UX | `model/batch.py`'s `BatchExtractDialog` -- filesystem glob (`FileSearch`), not a real library query; stays as-is for the manual/ad-hoc case, this plan adds a second, library-driven path alongside it |
| `.beq` project file format (gzip+JSON) | `app.py`'s `exportProject()`/`importProject()` (`SignalModel.to_json()`/`model.codec.signalmodel_from_json()`) -- what output 1's per-title projects (§3.3) are written as, so they open directly in the interactive app |
| Per-signal filter linking, master/slave | `model.signal.SingleChannelSignalData.enslave()`/`.free()` (and `model.codec.signaldata_to_json()`'s `master_name`/`slave_names`, already round-tripped) -- pure Python, no Qt dependency at the object level; this is the "filters linked so they apply to all channels" mechanism §3.3 reuses |
| LFE channel identification | `model/signal.py` keys bass-management behaviour off a signal's name ending in `_LFE` -- the same convention `model.ffmpeg.get_channel_name()` already produces (`Session.load_channels()` uses it today for `DesignRequest.channels`) |

So the genuinely new work is: (a) a `LibrarySource` abstraction + a
JRiver implementation of it, (b) an idempotency layer in front of
extract and design (neither `Session.extract()` nor
`design_and_queue()` currently skip anything), (c) the orchestration
that strings source -> cached extract -> cached design together, (d) a
CLI entry point, (e) a GUI dialog wrapping it, (f) writing output 1's
`.beq` project files at all -- nothing today builds one outside the
interactive app's own signal table (§3.3).

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
| `Year` | `year` | Optional string; fuzzy TMDB resolution remains the fallback. |
| `Series`, `Season`, `Episode` | `kind`, `meta` | **As built:** `kind='tv'` if any of the three is non-empty, else `'movie'`; only `Season` is retained (as `meta['season']`). `Media Type`/`Media Sub Type` are fetched by `hamcws` but **not consulted** -- a non-film, non-episodic item (music video, concert) is treated as a movie. |
| `Date Modified`, `File Size` | `fingerprint` | Stable serialisation of the values actually supplied. Leave it empty when both are absent, allowing the extract cache's filesystem-stat fallback. |
| `IMDB`, `TheMovieDB` | `external_ids` | Normalise non-empty values to `imdb` and `tmdb`. Use configurable field aliases because metadata plugins vary. |
| `Image File` | `art_path` | Only accept a locally readable path. `INTERNAL` means JRiver-managed art, not a file path; downloading `File/GetImage` is a later enhancement if needed. |

Deduplicate exact repeated `Key` rows before yielding; fail the run when
one key has conflicting filenames. A browse node is a source selection, so
every valid unique file reaches the normal extract/design stages.

Tests (`test_pipeline_library_jriver.py`) use hand-written row dicts and a
stubbed `hamcws` `MediaServer`, covering the selected node and field list,
normal mapping, optional fields/TV metadata, id stability across a rename,
distinct server identities, duplicate/conflicting keys, and rejection of a
`query`/running event loop. **Not covered:** `INTERNAL` artwork is handled in
code (`_local_art_path`) but has no dedicated test. **Still outstanding:** the
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
`item.meta`. Two behaviours worth knowing (both open, see §10): it runs
*before* the design-cache check, so a fully cached item still costs a TMDB
round-trip on every rerun, and a TMDB failure (`HTTPError`, no search hit)
fails the whole item -- no extract-only fallback. So `QueueEntry.meta` arrives
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

**Implementation status -- only tiers 1 and 4 exist.** Tier 1 (the reviewer's
Browse/Download/Clear controls) and the publish wiring are built. **Tiers 2
and 3 are not**: `LibraryItem.art_path` is populated by
`JRiverLibrarySource` but nothing reads it; `run_library()`/`design_if_needed()`
never write `QueueEntry.art_path`; and `resolve_meta()`'s `poster` fragment is
never passed to `pipeline.publish.art.fetch_poster()`. A library-driven entry
therefore still publishes chart-only unless a human sets artwork by hand --
the very gap §3.1.3 set out to close, for the library path. (The existing
manual-batch path is no worse than before.)

### 3.2 Kodi / Plex

Out of scope for this plan -- named only so `LibrarySource` isn't
JRiver-shaped by accident. Kodi's `VideoLibrary.GetMovies` JSON-RPC and
Plex's `/library/sections/<id>/all` REST endpoint are the obvious
future implementations; not stubbed as dead code, just documented here
as the intended shape to implement against later.

### 3.3 Output 1 -- local artifacts: extracted audio, designer output, and `.beq` project files

The user's clarification (2026-09-18): the extracted wav(s) and the
designer's raw filter output aren't the end of output 1 -- a beqdesigner
project file (`.beq`) needs producing too, so a human can open the
result directly in the interactive app. Two projects per title, not
one: **a mono project** (the signal design actually ran against) and,
whenever the kept extraction is multichannel, **a multichannel
project** with the same designed filter linked across every channel
and the LFE channel correctly identified as LFE -- this was not covered
by anything written so far in this plan and needed adding.

**This is independent of the library-source work**, same shape as
Chunk 1 (Appendix A) -- `model/batch.py`'s *existing* manual
batch-design flow already produces exactly the mono-wav-plus-optional-
multichannel-wav pair per candidate (pipeline/README.md's "Batch design
+ review" section: "a candidate whose kept file is multichannel gets a
second, mono-only extraction made just for the design step... That same
multichannel kept file is also decomposed... and sent alongside as
DesignRequest.channels"), so this belongs in `pipeline.review`
(Qt-free, already shipped) rather than being new library-specific code
-- both the existing manual flow and this plan's new library-driven
flow benefit equally, exactly as Chunk 1's metadata/artwork editor
does.

**What already exists, reused as-is:**
- The `.beq` file format itself -- gzip'd JSON, `app.py.exportProject()`/
  `importProject()`, built from `model.codec.signaldata_to_json()` per
  signal (no raw audio embedded -- only the analysed avg/peak/median
  curves, the filter, and master/slave names, so the project stays
  self-contained even if the source wav later moves/is deleted).
- Filter linking -- `SingleChannelSignalData.enslave(other)` (pure
  Python, no Qt): appends to `.slaves`, sets `other.master = self`, and
  immediately applies `self`'s current filter to `other`
  (`signal.on_filter_change`). `model.codec.signaldata_to_json()`
  already round-trips `master_name`/`slave_names`, and
  `signalmodel_from_json()` already reconstructs the links on load --
  nothing new needed on the serialisation side.
- LFE identification -- `model.ffmpeg.get_channel_name(prefix, idx,
  count, layout)` already produces names like `"<prefix>_LFE"`, the
  exact suffix `model/signal.py`'s bass-management code already keys
  off (`signal.name.endswith('_LFE')`). `Session.load_channels()`
  already uses this same function for the same labelling today (for
  `DesignRequest.channels`), just returning raw arrays instead of full
  signal objects -- see below for why this plan adds a second,
  parallel loader rather than reusing that one.

**New, small addition to `pipeline/orchestrate.py::Session`** --
`load_channels()` stays as-is (`DesignRequest.channels`, dict-of-arrays,
a different consumer/shape, design/designer-interface.md §2). A second
method returns full signal objects instead:

```python
def load_channel_signals(self, path: str, name: Optional[str] = None,
                         channel_layout_name: str = 'unknown',
                         decimate: bool = True) -> List[SingleChannelSignalData]:
    '''
    Loads every channel of a (possibly multichannel) wav as its own SingleChannelSignalData, named
    "<name>_<channel-label>" (model.ffmpeg.get_channel_name -- same labelling load_channels() uses) --
    ready for set_filters()/enslave(), unlike load_channels()'s raw decomposed arrays. A single-element
    list if path is actually mono.
    '''
```

Implementation calls `model.signal.AutoWavLoader.prepare()`/`get_signal()`
directly, once per channel, rather than going through `auto_load()`
(what `Session.load()` uses for the mono case) -- `auto_load()` wraps a
multichannel result in a `BassManagedSignalData`, which exists for
bass-management headroom calculations this method has no use for, and
which would need `BASS_MANAGEMENT_LPF_FS`/`_POSITION` out of the
`_ConfigPreferences` stand-in (`pipeline/orchestrate.py`, not real
QSettings) for a purpose it's not actually being used for here.
Calling `prepare()`/`get_signal()` directly -- the same two calls
`auto_load()` itself makes internally, per channel, in a loop -- avoids
that dependency entirely rather than working around it:

```python
def load_channel_signals(self, path, name=None, channel_layout_name='unknown', decimate=True):
    from model.ffmpeg import get_channel_name
    default_name = name or os.path.splitext(os.path.basename(path))[0]
    loader = AutoWavLoader(self.__preferences)
    loader.load(path)
    channel_count = loader.info.channels
    signals = []
    for idx in range(channel_count):
        channel_name = get_channel_name(default_name, idx, channel_count, channel_layout_name=channel_layout_name)
        loader.prepare(channel=idx + 1, name=channel_name, channel_count=channel_count, decimate=decimate)
        signals.append(loader.get_signal(idx + 1, channel_name))
    return signals
```

(`get_channel_name(text, idx, count, layout)` returns `text` unchanged
when `count == 1`, so a mono `path` naturally comes back as a single
unsuffixed-name signal -- no special-casing needed.) This removes what
was previously flagged as an implementation risk by construction, not
by verifying it away.

**New module, `pipeline/publish/project.py`:**

```python
def write_project(path: str, signals: Sequence[SingleChannelSignalData]) -> None:
    ''' Same gzip+JSON shape as app.py's exportProject() -- model.codec.signaldata_to_json() per signal,
    no BassManagedSignalData wrapper (this is a designed-filter project, not a bass-management one). '''

def write_title_projects(session: Session, mono_wav_path: str, filters: CompleteFilter,
                         multichannel_wav_path: Optional[str], channel_layout_name: str,
                         mono_out_path: str, multichannel_out_path: Optional[str]) -> None:
    # NB: as built this is `write_title_projects_if_safe()` (hash-gated, returns a per-target
    # written/skipped dict) plus `write_mono_project()`/`write_multichannel_project()` -- see Appendix B.3.
    '''
    Writes the mono project unconditionally (session.load(mono_wav_path), set_filters(), write_project()).
    If multichannel_wav_path is given, also loads every channel via load_channel_signals(),
    applies `filters` to the first channel (the master) and enslave()s every other channel -- including the
    LFE one, which needs no special handling here beyond already being named correctly -- to it, then writes
    that project too.
    '''
```

**On-disk convention** (no new `QueueEntry` fields needed for this --
everything is derivable from `work_dir` + `item.id`, keeping this
additive-free):

```
<work_dir>/<item.id>/
    mono.wav                    -- Session.extract(mono_mix=True), used for design + the mono project
    multichannel.wav            -- the kept extraction, only present when one was requested and the
                                    source is actually multichannel (mirrors model/batch.py's existing
                                    "Mix to Mono?" toggle -- see §4.1's extract-cache note)
    manifest.json                -- extract cache's fingerprint record (§4.1)
    <item.id>.mono.beq           -- always written once a candidate is designed
    <item.id>.multichannel.beq   -- only written when multichannel.wav exists
```

**When produced, and idempotency**: at design time
(`design_if_needed()`, §4.2), from the top-pick candidate
(`QueueEntry.candidates[0]`) -- cheap to redo (no audio decode, just
filter application + JSON), so it rides the same design-cache
fingerprint with no separate cache of its own: whenever
`design_if_needed()` actually (re)designs, it also (re)writes both
project files; when it skips (fingerprint match), the existing project
files are left alone since nothing about them would differ.

**Regenerating for a reviewer's actual pick**: the project files above
are built from the designer's top pick, but a reviewer may `accept` a
different (`chosen_candidate_index != 0`) candidate -- at that point
the on-disk projects would mismatch the human's actual decision unless
regenerated. See §3.3.1 immediately below for exactly when that
regeneration is (and, critically, is *not*) allowed to happen.

### 3.3.1 The `.beq` project is what gets published, not the raw designer output

The user's key correction (2026-09-18): once output 1's mono project
file exists, a human can open it in the full interactive app and change
the filter directly -- add a manual biquad, nudge a Q, whatever the
auto-designer didn't quite get right. **That edited filter is what must
end up in beqcatalogue**, not `QueueEntry.candidates[chosen_candidate_index].filters`
(the designer's untouched raw output) as originally drafted above. This
inverts a piece of the data flow this plan had assumed throughout: the
project file stops being a one-way, disposable *derivative* of the
queue entry and becomes the actual editable source of truth for the
published filter, with the queue entry demoted to "the designer's
original proposal, plus review status" -- still useful (audit trail,
what the designer actually thought), but no longer what
`publish_reviewed_queue()` reads to build the XML.

This immediately raises the follow-up problem the user identified next:
**how does the pipeline tell "still the designer's raw output" apart
from "a human opened this and changed it"?** -- since the answer
controls two different things that must not be conflated: (a) whether
`publish_reviewed_queue()` should trust the project file's filter over
the queue entry's, and (b) whether design-time/accept-time regeneration
(the paragraph above) is safe to run at all, or would silently clobber
a human's edit.

**Fix: a content hash, stored inside the project file itself, that only
the pipeline ever writes.** `write_project()`/`write_title_projects()`
compute `sha256(json.dumps(complete_filter.to_json(), sort_keys=True))`
for whichever filter they just applied, and stash it as an extra key --
`pipeline_filter_hash` -- on the *master* signal's dict, alongside
whatever `model.codec.signaldata_to_json()` already produces (an
additive key; `signaldata_from_json()`/`signalmodel_from_json()` only
ever reads the keys they know about, so this is silently ignored by the
interactive app's own load path -- verified against `model/codec.py`,
not assumed). The tell is what happens on the way back out: if a human
opens the project in the app, edits the filter, and re-exports via
`app.py`'s `exportProject()`, that path calls the *generic*
`signaldata_to_json()`, which has no concept of `pipeline_filter_hash`
and will not re-emit it -- so a human-resaved project always comes back
with either no hash at all, or (if a first save happened to preserve an
unrelated stale one) a hash that no longer matches the filter actually
stored. Either way, recomputing the hash from the project's *current*
`filter_presets[active_filter_preset]` and comparing:

```python
def read_project_filter(path: str) -> Tuple[CompleteFilter, bool]:
    ''' :return: (the master signal's current filter, True if pipeline_filter_hash matches a fresh hash of
    it -- i.e. this project is still exactly what the pipeline last wrote, False if a human has changed
    it since (or the file predates this mechanism and never had a hash at all). '''
```

- **`publish_reviewed_queue()`** reads the mono project's filter via
  `read_project_filter()` and publishes *that* filter -- regardless of
  whether it matches (a matching hash just means "the human's edit is
  the designer's own output," the common case, so this is a strict
  superset of today's behaviour, not a divergent path). `entry.candidates`/
  `apply_reviewed_entry()` are unchanged and still readable (still the
  audit trail of what the designer originally proposed), but stop being
  what `publish_reviewed_queue()` actually reads the filter from. If no
  project file exists yet (`work_dir` wasn't threaded through, or an
  older entry predates chunk 2), it falls back to today's
  `apply_reviewed_entry()` path -- backward compatible.
- **Design-time/accept-time regeneration** (the paragraph above this
  subsection) is now gated on the hash: before overwriting an *existing*
  project file, check whether its current filter's hash still matches
  its stored `pipeline_filter_hash`. A match means it's safe (still
  pure pipeline output, nothing to lose) -- overwrite as planned. A
  mismatch means a human edited it -- **do not overwrite**; the human's
  edit now outranks the designer's top pick and even a reviewer's
  candidate switch. Intended to be surfaced as a new
  `LibraryRunReport`/publish-result field (e.g. `project_edit_preserved:
  List[str]`, item ids) rather than silently swallowed. **Not built:** the
  hash gate itself works (`write_title_projects_if_safe()` skips an edited
  file and returns `False` for it), but `design_and_queue()` discards that
  return value, `publish_reviewed_queue()` ignores it, and `LibraryRunReport`
  has no such field -- a preserved edit is silent. Only the both-edited
  *conflict* case is reported (`{'id', 'error': 'project_conflict'}`).

**Both projects are legitimate places to design/edit a filter; only
mono is ever what gets *published*.** The user's correction
(2026-09-18): mono vs. multichannel is a distinction about what
*analysis* a filter was designed against -- a human (or, per the
already-shipped designer contract's per-channel `channels`/
`channel_scope` diagnostic input, a designer itself) may reasonably do
real design/refinement work in the multichannel project rather than the
mono one, e.g. checking a candidate against the LFE channel
specifically. It is not merely a personal-use copy of whatever mono
produced, so publish cannot simply always prefer mono's file and ignore
multichannel edits -- it needs to notice which one actually carries the
human's edit, whichever that is. What *is* fixed, regardless of which
signal a filter was analysed against, is the shape of the thing that
gets published: a single `CompleteFilter` written into beqcatalogue's
XML exactly as today -- this pipeline has never produced a genuinely
per-channel filter, so "publishing a multichannel-derived edit" needs
no conversion, just reading the right file.

`read_project_filter()` (above) therefore checks **both** project
files' hashes, not just mono's:
- Neither shows a human edit -> both still equal the last
  pipeline-designed filter; publish either (they're identical).
- Exactly one shows a human edit -> that project's filter (the master
  channel's, for multichannel) is authoritative; publish it, and write
  it back into the *other* project too (same mechanism as the
  accept-time regeneration above) so both stay consistent with what
  was actually published rather than one silently going stale.
  **Not built:** `resolve_published_filter()` returns the edited side's
  filter, but nothing writes it back into the other project, so the two
  files can stay divergent after a publish.
- **Both show edits, and they differ** -- a genuine conflict this plan
  cannot silently resolve by picking one. Publish refuses and surfaces
  it (the same `project_edit_preserved`-style reporting as the
  clobber-prevention case above), rather than guessing which of two
  independent human edits should win. Exactly how a human resolves that
  conflict (which project to treat as authoritative, or hand-merge) is
  left to UI/CLI-error-message design in the relevant chunk, not
  decided here.

## 4. Idempotency

Two independent caches, one per stage -- extraction is expensive
(ffmpeg decode of a whole film) and design is comparatively cheap but
still shouldn't silently re-run and perturb an in-review queue entry
underneath a reviewer.

### 4.1 Extract cache

New `pipeline/library/extract_cache.py`.

- One manifest file per item, `<work_dir>/<item.id>/manifest.json`, with
  **flat, prefixed keys** (this replaces the single-record shape drafted
  earlier; Appendix D.4 has the exact shape): `mono_source_fingerprint`,
  `mono_params_hash`, `mono_extracted_at`, the same three with a
  `multichannel_` prefix, and a top-level `channel_layout_name`. That last
  key is read by chunk 2's `publish_reviewed_queue()` regeneration step (and
  by `run_library()` via `extract_cache.read_channel_layout_name()`) to
  correctly label a multichannel project's channels
  (`model.ffmpeg.get_channel_name()`). It is only written by the multichannel
  ("kept") extraction; absent (or the whole manifest missing) degrades to
  `'unknown'`, which `get_channel_name()` already handles sanely by channel
  count. There is no stored `wav_path` -- it is always
  `<target_dir>/{mono,multichannel}.wav`.
- `params_hash` = hash of everything that changes ffmpeg's output for
  this item (audio_stream, mono_mix, decimate/target_fs,
  playlist_name) -- a config change must invalidate the cache even if
  the source file hasn't changed.
- `source_fingerprint` = `item.fingerprint` if the source supplies one
  (JRiver's own modified-date/size are more trustworthy than a local
  stat, since the source is the library's system of record), else
  fall back to `(os.path.getmtime, os.path.getsize)` of `source_path`.
- `extract_if_needed(session, item, target_dir, config, mono_mix, force=False) ->
  (wav_path, cached: bool)`: compares the current
  `(source_fingerprint, params_hash)` against the manifest; matches
  and `wav_path` still exists on disk -> skip ffmpeg, return the
  cached path; otherwise runs `session.extract()` (existing, unchanged)
  and rewrites the manifest. `force=True` always re-extracts.
- Called **twice per item** whenever `LibraryRunConfig.keep_multichannel`
  is set (**as built, the second call is not gated on the source actually
  being multichannel** -- `run_library()` extracts the kept file first and
  only then discovers via `load_channels()` that it is mono, at which point
  it is discarded from the design inputs but the redundant
  `multichannel.wav` and manifest entry remain) -- once with
  `mono_mix=True` into `<work_dir>/<item.id>/mono.wav` (design always
  needs this one -- `Session.design()`'s `mono_mix`) and once with
  `mono_mix=False` into `.../multichannel.wav` (the "kept" file, and
  the source for output 1's multichannel `.beq` project, §3.3) --
  mirrors `model/batch.py`'s existing "Mix to Mono?" toggle and its
  `ExtractCandidate.design()` docstring exactly ("a candidate whose kept
  file is multichannel gets a second, mono-only extraction made just
  for the design step"). Each call gets its own manifest entry (keyed
  by `mono_mix` as part of `params_hash`), so the two are independently
  cacheable -- redoing one doesn't force redoing the other.
- **Decimation only ever applies to the mono extraction.** `decimate`
  is not a separate, caller-chosen setting -- it's `mono_mix` itself
  (`decimate=mono_mix`): the mono extraction is decimated to
  `AnalysisConfig.target_fs` (1 kHz by default) because that's what
  `Session.design()`'s analysis needs to run fast, but the
  multichannel "kept" file must stay full quality (it's what a human
  actually links into output 1's multichannel `.beq` project and would
  use for real) -- resampling it to 1 kHz would silently wreck it.
  Same rule `model/batch.py`'s existing dual-extraction already
  follows, just made explicit here since chunk 5 has to encode it in
  code rather than a human choosing sensibly by hand.
- **Fixed output filenames, not ffmpeg's auto-derived ones.**
  `pipeline.review.publish_reviewed_queue()` (chunk 2, already shipped
  -- commit `407bd91`) hardcodes `<project_dir>/mono.wav` and
  `.../multichannel.wav` when reading `work_dir` -- so this chunk's
  extractions must land at exactly those paths, not whatever filename
  `Session.extract()` would otherwise derive from the source's own
  name. See Appendix D for the small, backward-compatible addition to
  `Session` this requires.

### 4.2 Design cache

`design_and_queue()` currently always overwrites
`<entry_id>.json`. Add a thin wrapper,
`pipeline/library/design_cache.py::design_if_needed()`:

- Skip re-designing when a queue entry for this id already exists
  *and* its recorded fingerprint (new, additive `QueueEntry.design_fingerprint:
  Optional[str] = None` field -- small, backward-compatible schema
  addition to `docs/schema/review_queue.schema.json`, existing entries
  without it just never match and get one designed on next touch)
  matches `hash(source_fingerprint, designer_name, AnalysisConfig,
  coverage)` for the current run.
- **Never** silently redesigns an entry whose `status` is `accepted`
  or `published` -- a rerun must not clobber a human's decision (or an
  already-published result) out from under them. `force=True` redesigns
  `pending`/`skipped`/`rejected` entries regardless of fingerprint match
  ("told to redo"); redoing an `accepted`/`published` entry requires
  the caller to explicitly reset its status first (a deliberate,
  single-item action, not something a library-wide `--force` flag does
  by accident).
- **Fingerprint scope (as built):** `design_fingerprint()` hashes exactly the
  four inputs above. It does *not* cover `keep_multichannel`, the resolved
  metadata, or `item.audio_stream`, so toggling "keep multichannel" or
  correcting a library tag does not trigger a redesign of a `pending`
  entry, and an existing entry never gains the multichannel project it
  would now get on a fresh design. `force_design` is the only way through.
- Whenever this actually (re)designs (not on a skip), output 1's local
  `.beq` project file(s) get written for the top-pick candidate too.
  This lives in core `pipeline.review.design_and_queue()` itself, not
  in this wrapper (§3.3: it benefits `model/batch.py`'s existing manual
  flow equally, so it belongs in the shared, already-shipped module) --
  `design_and_queue()` gains optional `multichannel_wav_path`/
  `channel_layout_name`/`project_dir` parameters and calls
  `pipeline.publish.project.write_title_projects()` internally
  whenever `project_dir` is given (`None`, the default, skips it --
  backward compatible for every existing caller). `design_if_needed()`
  just threads `project_dir=os.path.join(work_dir, item.id)` through on
  a run/redesign, and passes nothing through on a skip -- a skip leaves
  existing project files untouched, same as it leaves the existing
  `QueueEntry` untouched.

### 4.3 Sync

Idempotency for the publish step itself is unchanged in shape --
`publish_reviewed_queue()` already is: only `accepted` entries publish,
each is marked `published` on success, so a rerun after a partial
failure (e.g. a git push failure partway through the queue) only
retries what's still `accepted`. **What changes (§3.3.1) is *which*
filter gets published**: no longer unconditionally
`entry.candidates[chosen_candidate_index].filters`, but the mono
project's *current* filter (`read_project_filter()`), falling back to
the old candidate-based path only when no project file is available.
This makes "publish" pick up a human's project-file edit even though
nothing about the queue entry's own status/fields changed -- a
re-publish is no longer purely a function of `QueueEntry`, it also
depends on live state in a second file. Still idempotent in the sense
that mattered before (accepted-but-already-published entries are
skipped either way); just worth being explicit that the *content*
published for a given entry can now change between two runs with an
identical queue directory, if the project file changed in between.
Project-file regeneration itself (the hash-gated overwrite in §3.3.1)
is idempotent for the same reason it always was: a pipeline-pure
project regenerates identically every time; a human-edited one is
never touched at all.

This plan adds `pipeline/library/sync.py::sync_library()` purely as a
same-shaped verb alongside `run_library()` for CLI/GUI symmetry -- it
is a documented, near-literal call-through to
`pipeline.review.publish_reviewed_queue()`, no new logic.

## 5. Orchestration

New `pipeline/library/run.py`:

```python
@dataclass(frozen=True)
class LibraryRunConfig:
    work_dir: str
    queue_dir: str
    designer: str
    config: AnalysisConfig = field(default_factory=AnalysisConfig)
    coverage: Coverage = 'complete_programme'
    keep_multichannel: bool = False  # mirrors model/batch.py's "Mix to Mono?" -- see §4.1
    force_extract: bool = False
    force_design: bool = False
    tmdb_api_key: Optional[str] = None   # added: resolve_meta() (§3.1.1) only runs when set
    audio_types: Sequence[str] = ()      # added: forwarded to resolve_meta() as BeqMetadata.audio_types

@dataclass(frozen=True)
class LibraryRunReport:
    extracted: List[str]   # item ids that ran ffmpeg this run (either extraction)
    cached: List[str]      # item ids whose extraction(s) were all cached
    designed: List[str]    # item ids that ran a designer this run
    design_cached: List[str]  # also holds accepted/published entries skipped as protected -- not distinguished
    failed: List[Tuple[str, str]]  # (item id, "ExcType: message") -- one item's failure never aborts the rest
    # (all five default to empty lists; no project_edit_preserved field -- see §3.3.1)

def run_library(source: LibrarySource, run_config: LibraryRunConfig,
                on_item_done: Optional[Callable[[str], None]] = None,
                **source_query) -> LibraryRunReport:
    ''' source.list_items(**source_query) -> extract_if_needed -> design_if_needed, per item.
    Never calls publish -- see sync_library(). '''
```

Per-item failures (bad file, ffmpeg error, designer exception) are
caught and recorded in `LibraryRunReport.failed`, not raised -- a
library-scale run over hundreds of titles must not abort on the first
bad one (matches `Applied`/`Declined`'s existing "expected outcomes
are values, not exceptions" convention).

## 6. CLI entry point

New `pipeline/library/cli.py` (still Qt-free -- lives inside
`pipeline/`, same import-direction rule as everything else in the
package) with two subcommands:

```
python -m pipeline.library.cli run  --source jriver --work-dir ... --queue-dir ... --designer ...
python -m pipeline.library.cli sync --queue-dir ... --work-dir ... --xml-repo ... --images-repo ...
```

`sync`'s `--work-dir` is what lets `publish_reviewed_queue()` regenerate
output 1's project files for a reviewer's actual pick when it differs
from the top pick (§3.3) -- optional; omitting it just skips that
regeneration (backward compatible).

Source connection details (JRiver ip/auth) and repo targets come from
a config file (YAML/JSON, a plain dict-to-dataclass load, no
QSettings/Preferences dependency -- consistent with `AnalysisConfig`'s
"explicit input, not a process-wide singleton" rule) or CLI flags (flags
override the file). This is what makes the workflow cron-able.

As built: the config file has `run:`, `sync:`, `sources.<name>:` and
`analysis:` sections (`pyyaml` added as a dependency); `run` prints
`LibraryRunReport` as JSON and exits 1 if any item failed; `sync` prints
the publish results and exits 1 if any carries an `'error'`; only
`--source jriver` is accepted; `--tmdb-api-key`/`--audio-type` exist for
metadata resolution. `sync` has no way to pass `meta_defaults` except via
the config file's `sync.meta_defaults`.

## 7. GUI integration

New dialog, `model/library_sync.py` / `ui/library_sync.py`
(`LibrarySyncDialog`, reachable via a new Tools menu entry), following
`BatchExtractDialog`'s existing shape rather than inventing a new one:

- A source picker (registered `LibrarySource`s; JRiver connection
  reuses the existing `JRIVER_MCWS_CONNECTIONS` preference, same one
  the DSP-push feature already maintains) and a query/filter field.
  **As built:** no picker and no query field -- the dialog is JRiver-only,
  takes the *first* saved MCWS connection, and selects the browse node
  with a plain integer spin box (`browseNodeSpin`, default `-1`); the
  `browse_children()` node selector described in §3.1 is not built.
- **Run** tab: calls `run_library()` on a background `QRunnable` (same
  `QThreadPool` pattern as `ProbeJob`/`DesignJob`), streams
  `on_item_done` progress into the UI, switches to the **Review** tab
  (embedded `ReviewQueueDialog`, exactly as `BatchExtractDialog` does
  today, now carrying 3.1.2's metadata editor -- shared, not
  duplicated, so both this flow and the existing manual batch-design
  flow get it) when done.
- **Sync** action/button: calls `sync_library()` explicitly -- never
  auto-triggered after Run, per the review-gate decision. Also the
  natural place to finally wire `images_repo` into the GUI publish
  path, which `ReviewQueueDialog` doesn't do today (README: "XML-only
  from the dialog today").
- New preferences: xml repo / images repo local paths + owner/repo
  names (nothing persists these today -- `ReviewQueueDialog` asks via
  a file picker each time) and a `LIBRARY_SOURCE_DEFAULT` combo,
  following the existing `DESIGNER_QUEUE_DIR`/`DESIGNER_DEFAULT`
  pattern in `model/preferences.py`. **As built:** `LIBRARY_WORK_DIR`,
  `LIBRARY_XML_REPO`, `LIBRARY_IMAGES_REPO`, `LIBRARY_JRIVER_BROWSE_NODE`
  are defined *and* used; `LIBRARY_IMAGE_OWNER`/`LIBRARY_IMAGE_REPO_NAME`
  are defined but never read or written, so `_SyncJob` never passes
  `image_owner`/`image_repo_name` to `sync_library()`; there is no
  `LIBRARY_SOURCE_DEFAULT`. The queue dir reuses `DESIGNER_QUEUE_DIR`.
  Sync results are only counted in the status label -- `'error'` entries
  (e.g. `project_conflict`) are not surfaced to the reviewer.
- **Filtering the library view.** The user's note (2026-09-18): once
  `source.list_items()` can return results at library scale (hundreds
  of titles), the Run tab needs a way to filter/narrow that list before
  running anything against it -- at minimum by idempotency status
  (e.g. "stale" -- a cached extraction/design whose source fingerprint
  or config no longer matches, per §4.1/§4.2 -- vs. "new"/never-processed
  vs. "up to date"), and by ordinary library metadata (name, year,
  content type, and whatever else turns out to matter). The exact field
  list and how "stale" gets computed/labelled for display is explicitly
  **deferred to UI design time** (chunk 10) rather than fixed here --
  this bullet exists so the requirement itself isn't lost, not to
  pre-specify the filter bar.

## 8. Implementation chunks

Mirrors how `pipeline-implementation-plan.md` phased the original
headless pipeline -- small, independently testable, one commit (or a
small handful) each. Chunks 1 and 2 fix pre-existing gaps in the
*current*, already-shipped `pipeline.review`/`model/batch.py` flow and
are fully independent of the library-source track (chunks 4-8 below);
chunks 4-7 are themselves JRiver-independent and can be built/tested
against a fake `LibrarySource` before chunk 3 captures a real-server
fixture -- only chunk 8 is blocked on that mapping.

| # | Chunk | Depends on | Status |
|---|---|---|---|
| 1 | Review queue metadata + artwork editor (3.1.2, 3.1.3) -- `ReviewQueueDialog` gets editable metadata fields, a "reload from TMDB" control, an artwork browse/download/override section, and `publish_reviewed_queue()` actually wires `poster_path` through. Fixes two pre-existing gaps in the *current* manual `model/batch.py` flow, independent of everything else here. | nothing | **Implemented -- commit `964f36e`** |
| 2 | Output 1 -- local `.beq` project files, and making them the actual published source (§3.3, §3.3.1): `Session.load_channel_signals()`, new `pipeline/publish/project.py` (`write_project`/`write_title_projects_if_safe`/`read_project_filter`/`resolve_published_filter`, with the `pipeline_filter_hash` edit-detection mechanism and `ProjectFilterConflict`), optional new parameters on core `pipeline.review.design_and_queue()` (writes the mono + multichannel projects when designing, hash-gated against overwriting a human edit) and on `publish_reviewed_queue()` (reads whichever project's *current* filter is authoritative to publish instead of the raw candidate, falling back to today's behaviour when no project exists). Fixes gaps in the *current* manual flow too, independent of everything else here. | nothing | **Implemented -- commit `407bd91`** |
| 3 | Spike: capture a sanitised real-server `Browse/Files` response for a selected browse node, verify `hamcws`'s `browse_files()` mapping and the configured external-ID/artwork field aliases, and record the `Browse/Children` shape needed for node selection (§3.1). Write-up + fixture only, no product code. | nothing | **Not done** -- no real-server fixture exists; the adapter was built and tested against hand-written rows, so the external-ID/artwork field aliases and `Browse/Children` shape remain unverified |
| 4 | `pipeline/library/source.py`/`registry.py` -- `LibraryItem`, `LibrarySource` protocol, registry (mirrors `pipeline.designer.registry`). Pure interface, no implementation yet. | nothing | **Implemented -- commit `7d0bac5`** |
| 5 | `pipeline/library/extract_cache.py` -- idempotent extract (§4.1): manifest keyed on source fingerprint + params hash, skip ffmpeg on a hit; handles the mono + optional multichannel pair per item, at the fixed filenames chunk 2 already depends on. Small, backward-compatible addition to `Session` (`extract_with_layout()`) to get a fixed output filename + the channel layout in one call. | 4 | **Implemented -- commit `dd5dd56`** |
| 6 | `pipeline/library/design_cache.py` + additive `QueueEntry.design_fingerprint` field -- idempotent design (§4.2), never clobbers `accepted`/`published`; threads `project_dir` into `design_and_queue()` (chunk 2). | 2, 4 | **Implemented -- commit `d870d6d`** |
| 7 | `pipeline/library/run.py` (`run_library`) + `pipeline/library/sync.py` (`sync_library`) -- composition, per-item failure isolation (§5); threads `work_dir` into `publish_reviewed_queue()` (chunk 2). | 2, 4, 5, 6 | **Implemented -- commit `bc8179b`** |
| 8 | `pipeline/library/jriver.py`, built on `hamcws.MediaServer.browse_files()` and the configured browse-node id (§3.1), plus the `hamcws` dependency. If an id field exists, also `pipeline.metadata.tmdb_find_by_imdb_id()` + `pipeline/library/library_metadata.py::resolve_meta()` (§3.1.1). | 3, 4 | **Implemented -- commit `d870d6d`; `run_library()` calls `resolve_meta()` when `tmdb_api_key` is set. Artwork tiers 2/3 (§3.1.3) not wired** |
| 9 | `pipeline/library/cli.py` -- CLI entry point (§6). | 7, 8 | **Implemented -- commit `e23e03d`** |
| 10 | GUI: `model/library_sync.py`/`ui/library_sync.py` + `model/preferences.py` additions (§7), including the library-view filter bar (status + name/year/content-type, exact fields decided at UI design time), with a `pytest-qt` safety-net test before wiring, per this repo's established practice for touching a dialog. | 7, 8, 9 | **Implemented -- commit `9da7aea`; deferred: library-view filter bar, source picker/query field, browse-node selector, image owner/repo prefs wiring** |

---

## Appendix A -- Chunk 1 handoff spec: review queue metadata + artwork editor

Self-contained enough to implement without reading the rest of this
document (though §3.1.2/§3.1.3 above have the full rationale). Touches
only existing files -- no new modules, no JRiver/library-source
dependency.

### A.1 Why

Two things are true of the review queue *today*, independent of any
library-source work:

1. `pipeline.review.publish_reviewed_queue()` requires a valid
   `BeqMetadata` (title/year/audio_types at minimum --
   `pipeline.metadata.validate()`), but `model/batch.py`'s batch-design
   flow never populates `QueueEntry.meta` at all (`design_and_queue()`
   is called with no `meta` argument) -- so a human currently cannot
   get a batch-designed title through review and publish without some
   other, unbuilt path supplying metadata.
2. `publish_reviewed_queue()` calls `session.report(...)` without ever
   passing `poster_path` -- `pipeline.publish.art.fetch_poster()` and
   `pipeline.publish.report.compose_with_poster()` both already exist
   and are unit-tested standalone, but nothing wires them together, so
   every report published through the queue today renders chart-only.

`ReviewQueueDialog` (`model/review.py`/`ui/review.py`) has no editing
capability at all today -- it only ever displays
`entry.meta.get('title', entry.id)`, read-only. This chunk fixes both
gaps by adding an editing UI there, reusing patterns that already exist
elsewhere in this codebase rather than inventing new ones:
`CreateAVSPostDialog.load_tmdb_info()`/`__apply_tmdb_metadata()`
(`model/postbuilder.py`) for the "paste an id or search by title/year"
TMDB flow, and `SaveReportDialog.choose_image()`/`load_image_from_url()`
(`model/report.py`) for the artwork browse/download flow.

### A.2 Scope

**In scope:**
- Two new additive fields on `pipeline.review.QueueEntry`:
  `art_path: Optional[str] = None`, `art_overridden: bool = False`.
- A "Metadata" tab in `ReviewQueueDialog`'s detail pane with editable
  `BeqMetadata` fields, a "Reload from TMDB" action, and a "Save
  Metadata" action that writes to `entry.meta` via
  `pipeline.review.update_entry()`.
- An "Artwork" section (can live in the same tab, below the metadata
  form) to browse a local image file or download one from a pasted
  URL, writing to `entry.art_path`/`entry.art_overridden`.
- Wiring `pipeline.review.publish_reviewed_queue()` to pass
  `poster_path=entry.art_path` into `session.report(...)`.
- Locking edits once an entry is `accepted`/`published` (read-only,
  not hidden).
- Tests: `pytest-qt` coverage in `gui/test_review_dialog.py`, plain
  `pytest` coverage in `test_pipeline_review.py`.

**Explicitly out of scope** (deferred, either to a later chunk or
indefinitely -- do not build these now):
- Anything JRiver/library-source (chunks 3-10) -- this chunk only adds
  the *editing* surface; nothing here auto-populates `meta`/`art_path`
  from a library yet. Auto-resolution lands in chunk 8.
- TV (`kind='tv'`) support in the reload-from-TMDB control -- default
  to `kind='movie'` only, same trim `pipeline.metadata.tmdb_lookup()`'s
  `kind` param would need a UI control for; add later if wanted.
- `genres`/`collection` as editable fields -- these are TMDB-shaped
  structures (`[{'id':.., 'name':..}]` / a dict), not sensibly hand-edited
  as text. Show `genres` read-only (comma-joined names) after a TMDB
  reload; leave `collection` invisible (still round-trips through
  `entry.meta` if TMDB set it, just not surfaced in the form).
- Combo-box pickers for `language`/`source` (postbuilder has fixed
  lists for these) -- plain text fields for v1, upgradeable later.
- Per-field autosave -- edits batch behind an explicit "Save Metadata"
  button (single-shot actions -- browse/download artwork, TMDB reload
  -- still take effect immediately on their own button, matching how
  accept/skip/reject already write immediately).
- Copying a browsed local artwork file into a managed cache directory
  -- `art_path` may point anywhere on disk the user chose; if they
  later move/delete it, publish fails loudly at that point (acceptable
  for v1, same risk `SaveReportDialog.choose_image()` already has).
  Only a *downloaded* URL gets written into a durable location (see
  A.5) since there is no original local file to reference.

### A.3 Data model changes

`pipeline/review.py`:

```python
@dataclass
class QueueEntry:
    id: str
    fs: int
    meta: dict
    curve: dict
    candidates: List[CandidateSummary] = field(default_factory=list)
    decline_reason: Optional[str] = None
    decline_message: Optional[str] = None
    status: str = 'pending'
    chosen_candidate_index: Optional[int] = None
    reviewer_note: Optional[str] = None
    art_path: Optional[str] = None       # NEW -- local image file used as this entry's report poster
    art_overridden: bool = False         # NEW -- True once a human has explicitly set/cleared art_path;
                                          # future auto-resolution (chunk 8) must never overwrite it
```

Add both fields after `reviewer_note`, at the end -- `_entry_from_dict()`
(`QueueEntry(**d)`) already tolerates missing keys via the dataclass
defaults, so existing on-disk entries written before this chunk load
fine with `art_path=None, art_overridden=False`. No migration needed.

Update `docs/schema/review_queue.schema.json` to document them (the
schema has `"additionalProperties": true` so this isn't required for
anything to keep working, but the schema is a published reference and
should stay accurate):

```json
"art_path": {
  "oneOf": [{ "type": "string" }, { "type": "null" }],
  "description": "local image file path used as this entry's report poster, if any -- set by a human via the review dialog's Artwork section, or (later) auto-resolved from a library source or TMDB. null if none."
},
"art_overridden": {
  "type": "boolean",
  "description": "true once a human has explicitly set or cleared art_path -- auto-resolution must never silently overwrite it."
}
```
(add both to the `properties` object; `required` stays unchanged --
these are optional.)

### A.4 `pipeline/review.py` -- publish wiring

In `publish_reviewed_queue()`, the existing call:

```python
image_png = session.report([unfiltered, filtered], complete_filter, meta=meta, spec=report_spec,
                           mv_offset=chosen.mv_adjust_db)
```

becomes:

```python
image_png = session.report([unfiltered, filtered], complete_filter, meta=meta, poster_path=entry.art_path,
                           spec=report_spec, mv_offset=chosen.mv_adjust_db)
```

That's the entire pipeline-layer change -- `Session.report()` already
forwards `poster_path` to `render_report()`, which already forwards it
to `compose_with_poster()`. `entry.art_path` is `None` for any entry
that never had artwork set, and `render_report(poster_path=None)`
already renders chart-only, so this is backward compatible for every
existing entry.

### A.5 `review.ui` changes

Wrap the existing detail-pane content in a `QTabWidget` (`detailTabs`)
with two tabs, so the metadata form doesn't have to compete for space
with the candidate list/chart. The accept/skip/reject action buttons
stay **outside** the tab widget (in `actionButtonsLayout`, unchanged
position) since they act on the entry regardless of which tab is open.

```
detailPane (QWidget, unchanged)
  detailLayout (QVBoxLayout, unchanged)
    titleLabel                          -- unchanged, stays above the tabs
    declineReasonLabel                  -- unchanged
    detailTabs (QTabWidget)             -- NEW, wraps everything below
      "Candidates" tab (existing widgets, moved in as-is, unchanged behaviour):
        candidateList
        commentaryTable
        previewChart
      "Metadata" tab (metadataTab, QWidget)  -- NEW
        metadataForm (QFormLayout):
          "Title:"        titleField        (QLineEdit)
          "Alt title:"     altTitleField     (QLineEdit)
          "Sort title:"    sortTitleField    (QLineEdit)
          "Year:"          yearField         (QLineEdit)
          "Audio types:"   audioTypesField   (QLineEdit)   -- comma-separated, e.g. "Atmos, TrueHD 7.1"
          "Edition:"       editionField      (QLineEdit)
          "Season:"        seasonField       (QLineEdit)
          "Note:"          noteField         (QLineEdit)
          "Warning:"       warningField      (QLineEdit)
          "Language:"      languageField     (QLineEdit)
          "Source:"        sourceField       (QLineEdit)
          "Rating:"        ratingField       (QLineEdit)
          "Author:"        authorField       (QLineEdit)
          "AVS post URL:"  avsField          (QLineEdit)
          "Runtime:"       runtimeField      (QLineEdit)
          "Gain override:" gainField         (QLineEdit)   -- blank = use the chosen candidate's mv_adjust_db (today's default)
          "Genres:"        genresLabel       (QLabel, read-only -- set only by Reload from TMDB)
          "TMDB id:"       movieDbIdField    (QLineEdit)
                           reloadTmdbButton  (QPushButton, "Reload from TMDB") -- same row as movieDbIdField
        saveMetadataButton (QPushButton, "Save Metadata")
        metadataStatusLabel (QLabel)         -- transient feedback: "Saved" / a validation error
        --- Artwork ---
        artworkGroupLabel (QLabel, "Artwork")
        artPathField      (QLineEdit, read-only -- shows the current art_path, or empty)
        browseArtButton   (QPushButton, "Browse...")
        artUrlField       (QLineEdit, placeholder "Paste an image URL")
        downloadArtButton (QPushButton, "Download")
        clearArtButton    (QPushButton, "Clear")
        artPreviewLabel   (QLabel -- shows a small QPixmap thumbnail of art_path if it exists and is readable; blank otherwise)
    (candidateList/commentaryTable/previewChart end -- moved into the Candidates tab, not duplicated)
    actionButtonsLayout                  -- unchanged: acceptButton, skipButton, rejectButton
```

Regenerate with `cd src/main/python/ui && ../../../../.venv/bin/pyuic6 review.ui -o review.py` (or run
`./convert.sh` which does the same for every `.ui` in that directory).

### A.6 `model/review.py` changes

**Loading a selected entry** -- extend the existing `__on_row_selected`
to also populate the new tab:

```python
def __on_row_selected(self, *_):
    entry = self.__current_entry()
    ... # existing candidate/decline/chart logic, unchanged
    self.__load_metadata_form(entry)
    self.__load_artwork_section(entry)

def __load_metadata_form(self, entry):
    meta = entry.meta if entry is not None else {}
    self.titleField.setText(meta.get('title', ''))
    self.altTitleField.setText(meta.get('alt_title', ''))
    self.sortTitleField.setText(meta.get('sort_title', ''))
    self.yearField.setText(meta.get('year', ''))
    self.audioTypesField.setText(', '.join(meta.get('audio_types', [])))
    self.editionField.setText(meta.get('edition', ''))
    self.seasonField.setText(meta.get('season', ''))
    self.noteField.setText(meta.get('note', ''))
    self.warningField.setText(meta.get('warning', ''))
    self.languageField.setText(meta.get('language', ''))
    self.sourceField.setText(meta.get('source', ''))
    self.ratingField.setText(meta.get('rating', ''))
    self.authorField.setText(meta.get('author', ''))
    self.avsField.setText(meta.get('avs', ''))
    self.runtimeField.setText(meta.get('runtime', ''))
    self.gainField.setText(meta.get('gain') or '')
    self.movieDbIdField.setText(meta.get('the_movie_db', ''))
    genres = meta.get('genres') or []
    self.genresLabel.setText(', '.join(g.get('name', '') for g in genres))
    self.metadataStatusLabel.setText('')
    editable = entry is not None and entry.status in ('pending', 'skipped')
    self.metadataTab.setEnabled(editable)  # whole tab greyed out once accepted/published
```

`__load_artwork_section`:

```python
def __load_artwork_section(self, entry):
    art_path = entry.art_path if entry is not None else None
    self.artPathField.setText(art_path or '')
    if art_path and os.path.isfile(art_path):
        self.artPreviewLabel.setPixmap(QPixmap(art_path).scaledToWidth(160, Qt.TransformationMode.SmoothTransformation))
    else:
        self.artPreviewLabel.clear()
```

(`from qtpy.QtGui import QPixmap` added to the existing import block.)

**Saving metadata** -- a new method, wired to `saveMetadataButton.clicked`:

```python
def __save_metadata(self):
    entry = self.__current_entry()
    if entry is None:
        return
    fields = {
        'title': self.titleField.text().strip(),
        'year': self.yearField.text().strip(),
        'audio_types': [t.strip() for t in self.audioTypesField.text().split(',') if t.strip()],
    }
    optional = {
        'alt_title': self.altTitleField.text().strip(),
        'sort_title': self.sortTitleField.text().strip(),
        'edition': self.editionField.text().strip(),
        'season': self.seasonField.text().strip(),
        'note': self.noteField.text().strip(),
        'warning': self.warningField.text().strip(),
        'language': self.languageField.text().strip(),
        'source': self.sourceField.text().strip(),
        'rating': self.ratingField.text().strip(),
        'author': self.authorField.text().strip(),
        'avs': self.avsField.text().strip(),
        'runtime': self.runtimeField.text().strip(),
        'gain': self.gainField.text().strip(),
        'the_movie_db': self.movieDbIdField.text().strip(),
    }
    fields.update({k: v for k, v in optional.items() if v})  # blank = leave unset, don't stomp a BeqMetadata default
    merged = {**entry.meta, **fields}
    update_entry(self.__queue_dir, entry.id, meta=merged)
    self.metadataStatusLabel.setText('Saved')
    self.__reload_queue()
```

Note the blank-means-absent rule: an empty `languageField`/`sourceField`
must not write `''` into `entry.meta` (that would override
`BeqMetadata`'s `'English'`/`'Disc'` defaults at publish time with an
empty string) -- only non-empty optional fields are merged in.
`title`/`year`/`audio_types` are always written (including empty/`[]`)
since they're required by `pipeline.metadata.validate()` and a reviewer
clearing them back out is a legitimate (if unusual) action.

**Reload from TMDB** -- wired to `reloadTmdbButton.clicked`, mirrors
`CreateAVSPostDialog.load_tmdb_info()` exactly:

```python
def __reload_tmdb(self):
    from pipeline.metadata import tmdb_details_by_id, tmdb_lookup
    from model.preferences import TMDB_API_KEY
    api_key = self.__preferences.get(TMDB_API_KEY)
    tmdb_id = self.movieDbIdField.text().strip()
    try:
        if tmdb_id:
            meta = tmdb_details_by_id(tmdb_id, api_key, kind='movie')
        else:
            meta = tmdb_lookup(self.titleField.text().strip(), self.yearField.text().strip(), api_key, kind='movie')
        self.titleField.setText(meta.title)
        self.altTitleField.setText(meta.alt_title)
        self.yearField.setText(meta.year)
        self.ratingField.setText(meta.rating)
        self.runtimeField.setText(meta.runtime)
        self.movieDbIdField.setText(meta.the_movie_db)
        self.genresLabel.setText(', '.join(g.get('name', '') for g in meta.genres))
        self.__pending_tmdb_extras = {'poster': meta.poster, 'overview': meta.overview,
                                      'genres': meta.genres, 'collection': meta.collection}
    except requests.HTTPError as e:
        QMessageBox.critical(self, 'TMDB lookup failed', str(e))
```

`meta.poster`/`meta.overview`/`meta.genres`/`meta.collection` aren't
backed by a visible text field (poster goes through the Artwork
section instead, per A.2's scope trim on genres/collection) --
stash them on `self.__pending_tmdb_extras` and fold them into the
`optional` dict in `__save_metadata()` (`optional['overview'] =
self.__pending_tmdb_extras.get('overview', '')`, plus `genres`/
`collection` merged in directly, not through the blank-means-absent
filter since they're not strings) so a TMDB reload's fuller result
still reaches `entry.meta` on the next Save. Reset
`self.__pending_tmdb_extras = {}` in `__load_metadata_form()` so
switching entries doesn't leak one title's TMDB extras onto another's
save.

**Artwork actions**, immediate-write (no separate Save step, matching
accept/skip/reject's immediacy):

```python
def __browse_art(self):
    entry = self.__current_entry()
    if entry is None:
        return
    path, _ = QFileDialog.getOpenFileName(self, 'Choose artwork', filter='Images (*.png *.jpg *.jpeg)')
    if path:
        update_entry(self.__queue_dir, entry.id, art_path=path, art_overridden=True)
        self.__reload_queue()

def __download_art(self):
    entry = self.__current_entry()
    url = self.artUrlField.text().strip()
    if entry is None or not url:
        return
    try:
        resp = requests.get(url)
        resp.raise_for_status()
    except Exception as e:
        QMessageBox.critical(self, 'Download failed', str(e))
        return
    cache_dir = os.path.join(self.__queue_dir, '_art_cache')
    os.makedirs(cache_dir, exist_ok=True)
    ext = os.path.splitext(url)[1] or '.jpg'
    dest = os.path.join(cache_dir, f"{entry.id}{ext}")
    with open(dest, 'wb') as f:
        f.write(resp.content)
    update_entry(self.__queue_dir, entry.id, art_path=dest, art_overridden=True)
    self.artUrlField.clear()
    self.__reload_queue()

def __clear_art(self):
    entry = self.__current_entry()
    if entry is None:
        return
    update_entry(self.__queue_dir, entry.id, art_path=None, art_overridden=False)
    self.__reload_queue()
```

`_art_cache/` under the queue directory (not a tempfile, unlike
`SaveReportDialog.__download_image()`) -- a downloaded URL has no
original local file to reference, and `art_path` must remain valid
across app restarts and future `sync_library()` runs, so it needs a
durable location. Naming by `entry.id` means a second download for the
same entry simply overwrites the first (fine -- there's only ever one
current artwork per entry).

`requests` needs a module-level `import requests` added to
`model/review.py` (not currently imported there).

**Constructor wiring** -- add to `ReviewQueueDialog.__init__`, alongside
the existing button connections:

```python
self.saveMetadataButton.clicked.connect(self.__save_metadata)
self.reloadTmdbButton.clicked.connect(self.__reload_tmdb)
self.browseArtButton.clicked.connect(self.__browse_art)
self.downloadArtButton.clicked.connect(self.__download_art)
self.clearArtButton.clicked.connect(self.__clear_art)
self.__pending_tmdb_extras = {}
```

### A.7 Tests to add

`src/test/python/gui/test_review_dialog.py` (extend the existing
`_write_entry` helper to accept `meta=None`/`art_path=None` overrides
rather than hardcoding `meta={'title': entry_id}`, then add):

- `test_selecting_a_row_populates_the_metadata_form` -- write an entry
  with a full `meta` dict, select it, assert `dialog.titleField.text()`
  etc. match.
- `test_save_metadata_persists_edited_fields` -- select an entry, set
  `dialog.editionField.setText('Extended Cut')`, call
  `dialog._ReviewQueueDialog__save_metadata()`, assert
  `read_entry(queue_dir, entry_id).meta['edition'] == 'Extended Cut'`.
- `test_save_metadata_does_not_overwrite_defaults_with_blank_fields` --
  leave `languageField`/`sourceField` blank, save, assert `'language'
  not in read_entry(...).meta` (or, equivalently, that
  `BeqMetadata(**read_entry(...).meta).language == 'English'`).
- `test_metadata_tab_is_read_only_once_accepted` -- write an entry with
  `status='accepted'`, select it, assert
  `dialog.metadataTab.isEnabled() is False`.
- `test_browse_art_sets_art_path_and_marks_overridden` -- monkeypatch
  `QFileDialog.getOpenFileName` to return a fixed path (standard
  pytest-qt pattern for file dialogs -- check how other dialogs in this
  suite already do this, e.g. `test_batch_*`/`test_extract_*`, for the
  exact monkeypatch target), call `dialog._ReviewQueueDialog__browse_art()`,
  assert `read_entry(...).art_path == path` and `art_overridden is True`.
- `test_clear_art_resets_override` -- set `art_path` via `update_entry`
  directly, call `__clear_art()`, assert `art_path is None` and
  `art_overridden is False`.
- `test_reload_tmdb_by_id_populates_form_without_saving` -- monkeypatch
  `pipeline.metadata.tmdb_details_by_id` (imported inside the method, so
  patch `model.review.tmdb_details_by_id` after the `from ... import`
  runs, or patch at the `pipeline.metadata` source and call through --
  confirm the exact monkeypatch target once the import style in A.6 is
  finalised) to return a known `BeqMetadata`, call
  `dialog._ReviewQueueDialog__reload_tmdb()`, assert the form fields
  updated **and** `read_entry(...).meta` is unchanged (reload doesn't
  auto-save).

`src/test/python/test_pipeline_review.py` (pure pipeline layer, no Qt):

- `test_queue_entry_round_trips_art_path_and_overridden` -- write an
  entry with `art_path='/tmp/x.jpg', art_overridden=True`, read it back,
  assert both fields survive.
- `test_reading_a_pre_existing_entry_without_art_fields_defaults_them`
  -- hand-write a JSON file *without* `art_path`/`art_overridden` keys
  (simulating an entry written before this chunk), read it via
  `read_entry()`, assert `art_path is None` and `art_overridden is False`
  (backward compatibility).
- Extend `test_publish_reviewed_queue_with_image` (or add a sibling
  test) -- set `art_path` on the entry (via `update_entry`) to a real
  small JPEG written with PIL (same pattern
  `test_pipeline_publish_report.py` already uses:
  `Image.new('RGB', (300, 450), color=(10, 20, 30)).save(poster_path,
  format='JPEG')`), publish with `images_repo` given, fetch the pushed
  PNG, and assert its height is greater than a chart-only baseline (or
  decode it and check the top-left pixel matches the poster's fill
  colour) -- proof `poster_path` actually reached `render_report()`.

### A.8 Manual verification

Since this touches a real dialog: run the app
(`PYTHONPATH=./src/main/python QT_QPA_PLATFORM=offscreen` is for tests
only -- run normally for manual verification), open **Tools -> Review
Batch Designs...**, point it at a queue directory with at least one
pending entry (or create one via `model/batch.py`'s Batch Extract &
Design dialog first), and confirm: the Metadata tab shows blank fields
for an entry with empty `meta`, typing a title/year and clicking Save
persists across a Refresh, Reload from TMDB (with a real or test TMDB
key) populates the form without saving, Browse/Download artwork shows
a thumbnail and persists `art_path`, and the tab greys out once the
entry is Accepted.

### A.9 Acceptance checklist

- [x] `QueueEntry.art_path`/`art_overridden` added, schema doc updated.
- [x] `publish_reviewed_queue()` passes `poster_path=entry.art_path`.
- [x] `review.ui` has the new `detailTabs`/`metadataTab` structure,
      recompiled to `review.py`.
- [x] `ReviewQueueDialog` wires save/reload/browse/download/clear and
      locks the tab on `accepted`/`published`.
- [x] All new tests pass; full suite still green
      (`PYTHONPATH=./src/main/python QT_QPA_PLATFORM=offscreen uv run pytest src/test/python -q`).
- [ ] Manual verification (A.8) done in the real app. (not verifiable from code/tests -- unconfirmed)

---

## Appendix B -- Chunk 2 handoff spec: `.beq` project files as the published source

Self-contained enough to implement without reading the rest of this
document, though §3.3/§3.3.1 above have the full rationale and this
spec assumes you've read them once. Entirely within `pipeline/`
(Qt-free) plus `pipeline/review.py`'s two public entry points -- no
Qt/GUI work in this chunk. Independent of Chunk 1 (Appendix A) and of
every JRiver/library-source chunk.

### B.1 Why

Output 1 (§3.3) is a mono `.beq` project file per title, and
(optionally) a linked multichannel one. §3.3.1 established the
non-obvious part: once that project exists, a human editing it in the
interactive app is what must reach beqcatalogue, not
`QueueEntry.candidates[...]`'s frozen designer output -- and the
pipeline needs a way to tell "still exactly what the designer produced"
apart from "a human changed this" so it never silently clobbers an
edit, and so `publish_reviewed_queue()` knows which of (up to two)
project files to actually read from. This chunk builds all of that:
the project-writing itself, the edit-detection hash, the two-sided
resolution (mono vs. multichannel) including conflict detection, and
wiring both into the two existing `pipeline.review` entry points that
need it.

### B.2 Scope

**In scope:**
- `Session.load_channel_signals()` (`pipeline/orchestrate.py`) -- exact
  code already given in §3.3 above.
- New module `pipeline/publish/project.py` -- all functions in B.3.
- `pipeline.review.design_and_queue()` gains three new optional
  parameters (`multichannel_wav_path`, `channel_layout_name`,
  `project_dir`) and writes output 1's project files when
  `project_dir` is given and the outcome was `Applied`.
- `pipeline.review.publish_reviewed_queue()` gains one new optional
  parameter (`work_dir`) and, when given, regenerates (hash-gated) and
  then reads the published filter from the project file(s) rather than
  unconditionally from `apply_reviewed_entry()`.
- The extract-cache manifest convention (§4.1, not yet built in this
  chunk) gets one more field reserved for it: `channel_layout_name`,
  read back by `publish_reviewed_queue()`'s regeneration step. This
  chunk doesn't write the manifest (that's chunk 5) -- it only needs to
  agree on the field name now so chunk 5 doesn't have to revisit this
  chunk later. See B.5's note on this.

**Explicitly out of scope:**
- Anything that decides *how* a conflict (§3.3.1's "both edited and
  disagree" case) is presented to a human -- `publish_reviewed_queue()`
  raising/reporting it is in scope; a GUI dialog or CLI message
  explaining it is not (deferred to chunks 9/10).
- The extract cache itself (chunk 5) and `design_if_needed()`'s
  fingerprint-skip logic (chunk 6) -- this chunk only defines the
  manifest field name they'll need to also write.
- Detecting an edit to an individually-`free()`'d slave channel (§9's
  open question) -- out of scope, not solved by this chunk's hash
  mechanism, which only ever inspects the master/mono signal.

### B.3 New module: `pipeline/publish/project.py`

```python
def _filter_hash(filters: CompleteFilter) -> str:
    ''' sha256 of the filter's canonical JSON -- what pipeline_filter_hash stores/compares against. '''
    import hashlib, json
    return hashlib.sha256(json.dumps(filters.to_json(), sort_keys=True).encode('utf-8')).hexdigest()


def write_project(path: str, signals: Sequence[SingleChannelSignalData], filter_hash: Optional[str] = None) -> None:
    '''
    Writes a .beq project -- the same gzip+JSON shape app.py's exportProject()/importProject() use
    (model.codec.signaldata_to_json() per signal, no BassManagedSignalData wrapper). If filter_hash is
    given, stamps it as an extra 'pipeline_filter_hash' key on signals[0]'s dict -- an additive key
    signaldata_from_json()/signalmodel_from_json() simply don't look for, so it round-trips fine through
    the interactive app's own load path but is never re-emitted by a human's re-export (app.py's
    exportProject() only ever calls the generic signaldata_to_json(), which has no concept of this key) --
    that asymmetry is the edit-detection mechanism (see read_project_filter()).
    '''
    import gzip, json
    from model.codec import signaldata_to_json
    output = [signaldata_to_json(s) for s in signals]
    if filter_hash is not None and output:
        output[0]['pipeline_filter_hash'] = filter_hash
    with gzip.open(path, 'wb') as f:
        f.write(json.dumps(output).encode('utf-8'))


def write_mono_project(session: Session, mono_wav_path: str, filters: CompleteFilter, out_path: str) -> None:
    sig = session.load(mono_wav_path)
    session.set_filters(sig, filters)
    write_project(out_path, [sig], filter_hash=_filter_hash(filters))


def write_multichannel_project(session: Session, multichannel_wav_path: str, filters: CompleteFilter,
                               channel_layout_name: str, out_path: str) -> None:
    ''' Loads every channel (Session.load_channel_signals()), applies `filters` to the first (the master),
    and enslave()s every other channel -- including LFE, which needs no special handling beyond already
    being named correctly by load_channel_signals() -- to it. '''
    channels = session.load_channel_signals(multichannel_wav_path, channel_layout_name=channel_layout_name)
    master = channels[0]
    session.set_filters(master, filters)
    for slave in channels[1:]:
        master.enslave(slave)
    write_project(out_path, channels, filter_hash=_filter_hash(filters))


def _is_safe_to_overwrite(path: Optional[str]) -> bool:
    ''' True if path doesn't exist yet, or exists and is still pipeline-pure (no human edit to lose). '''
    if path is None or not os.path.isfile(path):
        return True
    _, is_pure = read_project_filter(path)
    return is_pure


def write_title_projects_if_safe(session: Session, mono_wav_path: str, filters: CompleteFilter, mono_out_path: str,
                                 multichannel_wav_path: Optional[str] = None, channel_layout_name: str = 'unknown',
                                 multichannel_out_path: Optional[str] = None) -> dict:
    '''
    The one entry point both design_and_queue() and publish_reviewed_queue() call. Writes each project
    only if it's safe to (§3.3.1's hash gate) -- an existing human-edited file is left untouched, the other
    one (if any) still gets written/updated normally. Both targets are independent; one being unsafe never
    blocks the other.
    :return: {'mono': True|False, 'multichannel': True|False|None} -- True=written, False=skipped (existing
        file was human-edited), None=not applicable (no multichannel_wav_path given).
    '''
    result = {'mono': False, 'multichannel': None}
    if _is_safe_to_overwrite(mono_out_path):
        write_mono_project(session, mono_wav_path, filters, mono_out_path)
        result['mono'] = True
    if multichannel_wav_path is not None:
        if _is_safe_to_overwrite(multichannel_out_path):
            write_multichannel_project(session, multichannel_wav_path, filters, channel_layout_name, multichannel_out_path)
            result['multichannel'] = True
        else:
            result['multichannel'] = False
    return result


def read_project_filter(path: str) -> Tuple[CompleteFilter, bool]:
    '''
    :return: (the master/mono signal's current filter, True if pipeline_filter_hash matches a fresh hash
        of it -- i.e. still exactly what the pipeline last wrote -- False if a human has changed it since,
        or the file predates this mechanism and never had a hash).
    '''
    import gzip, json
    from model.codec import filter_from_json
    with gzip.open(path, 'rb') as f:
        data = json.loads(f.read().decode('utf-8'))
    master = data[0]
    filt = filter_from_json(master['filter_presets'][master['active_filter_preset']])
    stored = master.get('pipeline_filter_hash')
    return filt, stored is not None and stored == _filter_hash(filt)


class ProjectFilterConflict(Exception):
    ''' Raised by resolve_published_filter() when both the mono and multichannel projects were
    independently edited and now disagree -- this plan deliberately does not guess which one wins. '''
    def __init__(self, mono_filter: CompleteFilter, multichannel_filter: CompleteFilter):
        super().__init__('mono and multichannel projects have independently edited, conflicting filters')
        self.mono_filter = mono_filter
        self.multichannel_filter = multichannel_filter


def resolve_published_filter(mono_path: str, multichannel_path: Optional[str] = None) -> Tuple[CompleteFilter, bool]:
    '''
    §3.3.1's three-way resolution. :return: (the filter to publish, True if it came from a human edit on
    either side, False if both projects are still pipeline-pure).
    :raises ProjectFilterConflict: if both projects were independently edited and disagree.
    '''
    mono_filter, mono_pure = read_project_filter(mono_path)
    if multichannel_path is None or not os.path.isfile(multichannel_path):
        return mono_filter, not mono_pure
    mc_filter, mc_pure = read_project_filter(multichannel_path)
    if not mono_pure and not mc_pure:
        if mono_filter.to_json() != mc_filter.to_json():
            raise ProjectFilterConflict(mono_filter, mc_filter)
        return mono_filter, True
    if not mc_pure:
        return mc_filter, True
    return mono_filter, not mono_pure
```

### B.4 `Session.load_channel_signals()`

Exact code already given in §3.3 above (the `AutoWavLoader.prepare()`/
`get_signal()`-direct version, not the `auto_load()`/
`BassManagedSignalData` one) -- add it to `pipeline/orchestrate.py::Session`
next to `load_channels()`, with `from model.ffmpeg import get_channel_name`
imported inside the method (matching how `load_channels()` already does
this same lazy import).

### B.5 `pipeline/review.py` changes

`design_and_queue()` -- three new optional parameters, and a call to
`write_title_projects_if_safe()` right after `write_queue_entry()`,
only when `project_dir` is given and the outcome was `Applied` (a
`Declined` outcome has no filter to write, matching "candidates empty
on decline"):

```python
def design_and_queue(session, entry_id, wav_path, designer, queue_dir, meta=None, coverage='complete_programme',
                     bass_management=None, channels=None,
                     multichannel_wav_path: Optional[str] = None, channel_layout_name: str = 'unknown',
                     project_dir: Optional[str] = None) -> QueueEntry:
    ...  # unchanged up to and including write_queue_entry(queue_dir, entry)
    if project_dir is not None and isinstance(outcome, Applied):
        from pipeline.publish.project import write_title_projects_if_safe
        mono_out = os.path.join(project_dir, f"{entry_id}.mono.beq")
        mc_out = os.path.join(project_dir, f"{entry_id}.multichannel.beq") if multichannel_wav_path else None
        write_title_projects_if_safe(session, wav_path, outcome.filters, mono_out,
                                     multichannel_wav_path=multichannel_wav_path,
                                     channel_layout_name=channel_layout_name, multichannel_out_path=mc_out)
    return entry
```

Note `wav_path` is already documented as "an already-extracted (mono)
wav file" -- it *is* the mono project's source, no new parameter needed
for that half.

`publish_reviewed_queue()` -- one new optional parameter. When given,
regenerate (hash-gated, via the same `write_title_projects_if_safe()`)
before reading, then read the filter to actually publish from the
project file(s) instead of `apply_reviewed_entry()`'s raw candidate --
`apply_reviewed_entry()`/`chosen` stay exactly as they are today (still
needed for `meta.gain`'s default), just no longer the source of the
*filter itself* when a project directory is available:

```python
def publish_reviewed_queue(queue_dir, xml_repo, meta_defaults=None, images_repo=None, image_owner=None,
                           image_repo_name=None, xml_dir='', image_dir='', report_spec=ReportSpec(),
                           config=AnalysisConfig(), work_dir: Optional[str] = None) -> List[dict]:
    from pipeline.publish.project import write_title_projects_if_safe, resolve_published_filter, ProjectFilterConflict
    ...
    for entry in read_queue(queue_dir):
        if entry.status != 'accepted':
            continue
        chosen = _chosen_candidate(entry)
        complete_filter = apply_reviewed_entry(entry)  # unchanged -- still drives meta.gain's default below
        meta = BeqMetadata(**{**(meta_defaults or {}), **entry.meta})
        if meta.gain is None:
            meta.gain = f"{chosen.mv_adjust_db:+g}"

        if work_dir is not None:
            project_dir = os.path.join(work_dir, entry.id)
            mono_path = os.path.join(project_dir, f"{entry.id}.mono.beq")
            mc_wav = os.path.join(project_dir, 'multichannel.wav')
            mc_path = os.path.join(project_dir, f"{entry.id}.multichannel.beq") if os.path.isfile(mc_wav) else None
            layout = _read_channel_layout_name(project_dir)  # manifest.json's channel_layout_name, 'unknown' if absent/missing
            write_title_projects_if_safe(session, os.path.join(project_dir, 'mono.wav'), complete_filter, mono_path,
                                         multichannel_wav_path=mc_wav if mc_path else None,
                                         channel_layout_name=layout, multichannel_out_path=mc_path)
            try:
                complete_filter, _ = resolve_published_filter(mono_path, mc_path)
            except ProjectFilterConflict:
                results.append({'id': entry.id, 'error': 'project_conflict'})
                continue

        image_png = None
        # ...unchanged from here: image_png = session.report(..., complete_filter, ...), xml = self.to_beq_xml(complete_filter, meta), etc.
```

`_read_channel_layout_name(project_dir)` -- a small helper, this
chunk's only piece of forward-looking coupling to chunk 5's (not yet
built) extract-cache manifest: reads
`os.path.join(project_dir, 'manifest.json')`'s `channel_layout_name`
key if the file and key exist, else returns `'unknown'`
(`get_channel_name()`'s own fallback already handles an unknown layout
sanely by channel count). Chunk 5 needs to actually write that key when
it builds the manifest -- noted here so it doesn't have to revisit this
chunk later; this chunk's own tests (B.6) can just hand-write a minimal
`manifest.json` with that key to exercise the read side.

An entry that hits `ProjectFilterConflict` is recorded in `results`
with an `'error'` key and **not** marked `'published'` -- a rerun will
retry it, same "only accepted-and-not-yet-published entries do
anything" idempotency the rest of this function already has.

### B.6 Tests to add

All in `src/test/python/` (pipeline layer, no Qt) -- a new
`test_pipeline_publish_project.py` for the new module, plus additions
to the existing `test_pipeline_review.py`:

**`test_pipeline_publish_project.py`:**
- `test_write_and_read_project_round_trips_the_filter` -- write a mono
  project for a known `CompleteFilter`, `read_project_filter()` it
  back, assert the filter matches and `is_pure is True`.
- `test_read_project_filter_detects_a_human_edit` -- write a project,
  then hand-edit the on-disk JSON's `filter_presets`/`active_filter_preset`
  to a different filter (simulating a resave through the interactive
  app, which wouldn't preserve `pipeline_filter_hash`), read it back,
  assert `is_pure is False` and the filter returned is the *edited* one.
- `test_write_multichannel_project_enslaves_every_channel_to_the_master`
  -- write a multichannel project from a real multichannel wav fixture
  (or a small synthetic one, matching how other pipeline tests build
  fixtures), read it back via `model.codec.signalmodel_from_json`
  directly (not `read_project_filter()`, which only looks at the
  master), assert every non-master signal's `master_name` in the raw
  JSON points at the master and the master's `slave_names` lists all of
  them, including the one named with the `_LFE` suffix.
- `test_write_title_projects_if_safe_skips_an_edited_target` -- write a
  project, hand-edit it (as above), call
  `write_title_projects_if_safe()` again with a different filter,
  assert the result dict says `'mono': False` and the file's filter is
  still the hand-edited one, not the new one.
- `test_write_title_projects_if_safe_writes_the_other_target_independently`
  -- edit only the mono project, call with both mono+multichannel
  targets, assert `{'mono': False, 'multichannel': True}` and the
  multichannel file did get the new filter.
- `test_resolve_published_filter_prefers_the_edited_side` -- mono
  edited, multichannel still pure (or absent) -> resolves to the mono
  edit; and the mirror case (multichannel edited, mono pure) ->
  resolves to the multichannel edit.
- `test_resolve_published_filter_raises_on_disagreeing_edits` -- edit
  both projects to *different* filters, assert
  `resolve_published_filter()` raises `ProjectFilterConflict` carrying
  both filters.

**`test_pipeline_review.py` additions:**
- `test_design_and_queue_writes_a_mono_project_when_project_dir_given`
  -- call with `project_dir=str(tmp_path / 'projects')`, assert the
  `<entry_id>.mono.beq` file exists and its filter matches
  `entry.candidates[0].filters`.
- `test_design_and_queue_writes_a_multichannel_project_when_given_one`
  -- same, plus `multichannel_wav_path`, assert the `.multichannel.beq`
  file exists too.
- `test_design_and_queue_does_not_write_projects_without_project_dir`
  -- omitting `project_dir` (today's call shape, unchanged) writes no
  `.beq` files -- backward compatibility.
- `test_publish_reviewed_queue_reads_the_edited_mono_project_when_work_dir_given`
  -- set up an accepted entry with a `work_dir`/project layout, hand-edit
  the mono project's filter after design, call `publish_reviewed_queue(...,
  work_dir=...)`, assert the published XML reflects the *edited* filter,
  not `entry.candidates[0]`'s.
- `test_publish_reviewed_queue_without_work_dir_uses_apply_reviewed_entry_as_before`
  -- omitting `work_dir` (today's call shape) publishes exactly what it
  does today -- backward compatibility, protects every existing
  `test_publish_reviewed_queue_*` test in this file from regressing.
- `test_publish_reviewed_queue_reports_a_project_conflict_without_publishing`
  -- both projects edited to disagree, assert the entry is *not* marked
  `published`, and the result list contains an `'error'` entry for it.

### B.7 Acceptance checklist

- [x] `pipeline/publish/project.py` created with all of B.3's functions.
- [x] `Session.load_channel_signals()` added (B.4).
- [x] `design_and_queue()`/`publish_reviewed_queue()` updated per B.5,
      both fully backward compatible when their new parameters are omitted.
- [x] All new tests (B.6) pass; full suite still green
      (`PYTHONPATH=./src/main/python QT_QPA_PLATFORM=offscreen uv run pytest src/test/python -q`).
- [x] `pipeline_qt_boundary` test (this package's AST scan for stray
      `qtpy` imports) still passes -- `pipeline/publish/project.py` must
      not import qtpy, same rule as every other module in `pipeline/`.

---

## Appendix C -- Chunk 4 handoff spec: `LibrarySource` abstraction

Self-contained enough to implement without reading the rest of this
document (§3 above has the same shapes plus the JRiver-specific
rationale, but nothing in this chunk depends on JRiver at all). This is
the smallest chunk in the plan: a pure interface + an in-process
registry, no implementation behind it yet -- chunk 8 (JRiver, pending
chunk 3's real-server mapping fixture) is the first real implementation, but chunks 5-7
(extract cache, design cache, orchestration) only need this interface
to exist, not anything concrete implementing it, so this unblocks all
of them.

### C.1 Why

`run_library()` (chunk 7) needs to iterate "whatever a library gives
us" without knowing whether that's JRiver, Kodi, Plex, or a test
double. This chunk defines that contract -- a `LibraryItem` (the data
one title's row needs to carry through extract/design/publish) and a
`LibrarySource` protocol (`list_items(**query) -> Iterable[LibraryItem]`)
-- plus a registry to bind a configured instance to a name, the exact
same shape `pipeline.designer.registry` already uses for designers (a
GUI registers configured instances at startup, a CLI registers from a
config file at process start -- see `pipeline/README.md`'s "In the GUI,
Preferences -> Designers... registered under a `http:`-prefixed name on
every app startup").

### C.2 Scope

**In scope:**
- New package `pipeline/library/` (`__init__.py`, empty -- matches
  `pipeline/designer/__init__.py`'s convention).
- `pipeline/library/source.py` -- `LibraryItem`, `LibrarySource`.
- `pipeline/library/registry.py` -- `register_source`/`unregister_source`/
  `get_source`/`registered_sources`.

**Explicitly out of scope:**
- Any real `LibrarySource` implementation (JRiver -- chunk 8, blocked;
  Kodi/Plex -- not built at all, per §3.2).
- Wiring this into `run_library()`/`design_if_needed()`/anything else
  -- those are chunks 5-7's job; this chunk only needs to exist for
  them to import against, and this chunk's own tests use a small
  fake/test-only `LibrarySource` to exercise the shape, not a real one.

### C.3 Exact code

`pipeline/library/source.py`:

```python
'''
LibrarySource: the pluggable contract a library-scale extract+design run (pipeline/library/run.py, not yet
built) iterates over. One implementation ships against JRiver (blocked on a wire-format spike); Kodi/Plex
are named future implementations of this same interface, not built here (design/library-sync-pipeline-plan.md
§3.2).
'''
from dataclasses import dataclass, field
from typing import Iterable, Optional, Protocol


@dataclass(frozen=True)
class LibraryItem:
    '''
    One title as a library source sees it -- enough for extract+design+metadata resolution to run against
    without needing to know which concrete source produced it.
    '''
    id: str                                  # stable across runs -- also the QueueEntry.id / cache key.
                                              # Must survive a rename/re-scan; a source should prefer its
                                              # own persistent key (e.g. JRiver's Media ID) over a
                                              # filename-derived one.
    source_path: str                         # ffmpeg input: a container file path, or a BDMV root directory
    display_name: str
    title: Optional[str] = None
    year: Optional[str] = None
    kind: str = 'movie'                      # 'movie' or 'tv' -- forwarded to TMDB resolution
    external_ids: dict = field(default_factory=dict)  # {'tmdb': '603', 'imdb': 'tt0133093'} -- whatever
                                              # identifiers the source can supply; empty if it can't
                                              # (design/library-sync-pipeline-plan.md §3.1.1)
    audio_stream: int = 0
    playlist_name: Optional[str] = None      # BD only, forwarded to Session.extract()
    art_path: Optional[str] = None           # a local poster/cover file the source already has, if any
                                              # (§3.1.3)
    meta: dict = field(default_factory=dict) # extra BeqMetadata ctor kwargs the source can supply directly
                                              # -- merged in on top of, and losing to nothing from, whatever
                                              # TMDB resolution produces
    fingerprint: str = ''                    # source-specific change marker (§4.1) -- opaque to callers


class LibrarySource(Protocol):
    def list_items(self, **query) -> Iterable[LibraryItem]:
        ''' `query`'s accepted keyword arguments are entirely source-specific (e.g. a JRiver search-string
        query) -- there is no fixed cross-source query schema, matching how model/batch.py's existing
        FileSearch takes source-specific glob patterns today. '''
        ...
```

`pipeline/library/registry.py`:

```python
'''
A small in-process registry binding a library source name to an already-configured instance --
pipeline.designer.registry's exact pattern (design/api-headless-pipeline.md D8): the cheapest binding
first, no HTTP binding needed unless something actually requires one.
'''
from typing import Dict, List

from pipeline.library.source import LibrarySource

_registry: Dict[str, LibrarySource] = {}


def register_source(name: str, source: LibrarySource) -> None:
    '''
    :param name: a unique name for the source, e.g. 'jriver'.
    :param source: an already-configured LibrarySource instance -- any per-connection configuration (a
        JRiver server's ip/auth, say) is resolved by the caller *before* registering, the same way
        pipeline.designer.http_binding.http_designer(url) is called once to produce a ready callable
        before pipeline.designer.registry.register_designer() ever sees it.
    '''
    _registry[name] = source


def unregister_source(name: str) -> None:
    _registry.pop(name, None)


def get_source(name: str) -> LibrarySource:
    '''
    :raises KeyError: if no source is registered under that name.
    '''
    if name not in _registry:
        raise KeyError(f"No library source registered as '{name}' -- registered: {sorted(_registry.keys())}")
    return _registry[name]


def registered_sources() -> List[str]:
    return sorted(_registry.keys())
```

### C.4 Tests to add

New `src/test/python/test_pipeline_library_source.py` -- nothing else
in the codebase exercises this yet (chunks 5-7 will), so this chunk's
own tests are what prove the shape works, using a small fake source
defined in the test file itself:

```python
class _FakeSource:
    def __init__(self, items):
        self._items = items
    def list_items(self, **query):
        return [i for i in self._items if not query.get('title_contains') or query['title_contains'] in i.title]
```

- `test_library_item_defaults` -- construct a `LibraryItem` with only
  the three required fields, assert every optional field's default
  (`title=None`, `kind='movie'`, `external_ids=={}`, `meta=={}`, etc.).
- `test_register_and_get_source_round_trips` -- `register_source('fake',
  _FakeSource([...]))`, `get_source('fake')` returns the same instance,
  `list_items()` on it works.
- `test_get_source_unregistered_raises_keyerror` -- includes the
  registered-names list in the message (matching
  `get_designer()`'s exact error-message shape).
- `test_unregister_source_removes_it` -- register, unregister, assert
  `get_source()` now raises.
- `test_registered_sources_is_sorted` -- register a few out of
  alphabetical order, assert `registered_sources()` comes back sorted.
- `test_list_items_accepts_source_specific_query_kwargs` -- via
  `_FakeSource`, confirms `list_items(**query)`'s duck-typed,
  no-fixed-schema shape works as intended (e.g.
  `list_items(title_contains='Player')` filters as expected).
- A `pipeline_qt_boundary`-style AST-scan test confirming
  `pipeline/library/source.py`/`registry.py` import no `qtpy` --
  matching the per-module convention every other new `pipeline/`
  module in this plan has added one of (see chunk 2's
  `test_pipeline_publish_project_module_has_no_qtpy_import`).

### C.5 Acceptance checklist

- [x] `pipeline/library/__init__.py`, `source.py`, `registry.py` created.
- [x] All of C.4's tests pass; full suite still green.
- [x] No `qtpy` import in either new module.

---

## Appendix D -- Chunk 5 handoff spec: idempotent extract cache

Self-contained enough to implement without reading the rest of this
document, though §4.1 above has the same design plus the rationale.
Depends only on chunk 4 (Appendix C, already shipped -- commit
`7d0bac5`) for `LibraryItem`. Entirely within `pipeline/` (Qt-free).

### D.1 Why

`Session.extract()` always shells out to ffmpeg -- nothing skips
anything today. At library scale (hundreds of titles, run repeatedly
as new titles get added), re-extracting everything on every run is
wasteful and slow. This chunk adds a manifest-backed idempotency layer
in front of it: skip ffmpeg when the source hasn't changed (per
`LibraryItem.fingerprint`, or a local mtime/size fallback) and neither
the analysis config nor the extraction mode (mono vs. multichannel)
has changed since the last recorded run.

**Cross-chunk constraint discovered while writing this spec**: chunk 2
(`pipeline.review.publish_reviewed_queue()`, already shipped, commit
`407bd91`) hardcodes `<work_dir>/<entry.id>/mono.wav`,
`.../multichannel.wav`, and reads `.../manifest.json`'s flat
`channel_layout_name` key. This chunk's on-disk output **must** match
that exactly -- it was designed against this constraint, not
independently, so implementing it as literally specified below keeps
both chunks consistent. If you find any mismatch against the *actual*
current `pipeline/review.py` (not just this document), the shipped
code wins and this spec needs correcting, not the other way around --
confirm by reading `pipeline/review.py`'s `publish_reviewed_queue()`
and `_read_channel_layout_name()` before writing any code.

### D.2 Scope

**In scope:**
- A new, backward-compatible `Session.extract_with_layout()` in
  `pipeline/orchestrate.py`, which also lets a caller fix the output
  filename. `Session.extract()` itself becomes a thin wrapper over it
  (same signature, same return type, behaviourally identical) so every
  existing caller is unaffected.
- New `pipeline/library/extract_cache.py` -- `extract_if_needed()`
  plus its manifest read/write helpers.

**Explicitly out of scope:**
- `run_library()` (chunk 7) -- deciding *whether* to call
  `extract_if_needed()` once or twice per item (the
  `LibraryRunConfig.keep_multichannel` toggle) is that chunk's job;
  this chunk's `extract_if_needed()` only ever handles one `mono_mix`
  value per call, agnostic to how many times or in what order it's
  called for a given item.
- Anything about design (`design_if_needed()` is chunk 6).

### D.3 `Session` change (`pipeline/orchestrate.py`)

```python
@dataclass(frozen=True)
class ExtractResult:
    wav_path: str
    channel_layout_name: str  # model.ffmpeg's CHANNEL_LAYOUTS key, e.g. '5.1', or 'unknown'/a generic
                              # "<n> channels" string when ffmpeg's probe couldn't name it more precisely
```

Add to `Session`:

```python
def extract_with_layout(self, src: str, target_dir: str, audio_stream: int = 0, video_stream: int = -1,
                        mono_mix: bool = True, decimate: bool = True, playlist_name: Optional[str] = None,
                        output_file_name: Optional[str] = None) -> ExtractResult:
    '''
    Same as extract(), but (a) lets a caller fix the output filename instead of ffmpeg's auto-derived one
    (output_file_name is given *without* an extension -- Executor appends the format's own extension,
    '.wav' by default) and (b) also returns the source's detected channel layout name
    (Executor.channel_layout_name, the same one model/batch.py's ExtractCandidate.design() already reads
    off its own Executor) -- for a caller (the extract cache, §4.1) that needs to record it without a
    second, separate probe. extract() itself keeps returning a bare path unchanged -- every other existing
    caller has no use for the layout and a changed return type would break them.
    '''
    os.makedirs(target_dir, exist_ok=True)
    display_name = None
    duration_override_s = None
    if is_bdmv_root(src):
        resolved = resolve_main_title(src, playlist_name=playlist_name)
        src = resolved.ffmpeg_input
        display_name = resolved.display_name
        duration_override_s = resolved.playlist.duration_s
    executor = Executor(src, target_dir, mono_mix=mono_mix, decimate_audio=decimate,
                        decimate_fs=self.__config.target_fs, display_name=display_name,
                        duration_override_s=duration_override_s)
    if output_file_name is not None:
        executor.output_file_name = output_file_name
    executor.probe_file()
    if not executor.has_audio():
        raise ValueError(f"{src} has no audio stream to extract")
    executor.update_spec(audio_stream, video_stream, mono_mix)
    executor.run_sync()
    return ExtractResult(wav_path=executor.get_output_path(), channel_layout_name=executor.channel_layout_name)

def extract(self, src: str, target_dir: str, audio_stream: int = 0, video_stream: int = -1,
           mono_mix: bool = True, decimate: bool = True, playlist_name: Optional[str] = None) -> str:
    ''' Unchanged signature/behaviour/return type -- now a thin wrapper, so this and extract_with_layout()
    can never behaviourally diverge. '''
    return self.extract_with_layout(src, target_dir, audio_stream, video_stream, mono_mix, decimate,
                                    playlist_name).wav_path
```

This is a refactor of an existing, heavily-relied-upon method
(`design_and_queue()`, `batch_design()`, `model/batch.py`, and
`test_pipeline_acceptance.py`'s end-to-end test all call `extract()`
today) -- the full test suite passing unmodified for every one of
those existing callers is the acceptance bar for this part, not just
"the new tests pass".

### D.4 `pipeline/library/extract_cache.py`

```python
'''
Idempotent wrapper around Session.extract_with_layout() -- design/library-sync-pipeline-plan.md §4.1.
Skips ffmpeg entirely when the source hasn't changed (per LibraryItem.fingerprint, or a local mtime/size
fallback) and neither the analysis config nor the extraction mode (mono_mix) has changed since the last
run recorded in <target_dir>/manifest.json.
'''
import hashlib
import json
import os
import time
from typing import Tuple

from pipeline.config import AnalysisConfig
from pipeline.library.source import LibraryItem
from pipeline.orchestrate import Session

_MANIFEST_FILENAME = 'manifest.json'


def _source_fingerprint(item: LibraryItem) -> str:
    if item.fingerprint:
        return item.fingerprint
    stat = os.stat(item.source_path)
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def _params_hash(item: LibraryItem, config: AnalysisConfig, mono_mix: bool) -> str:
    payload = json.dumps({
        'audio_stream': item.audio_stream,
        'playlist_name': item.playlist_name,
        'mono_mix': mono_mix,
        'target_fs': config.target_fs,  # only actually varies the output when mono_mix is True -- see D.1
    }, sort_keys=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _read_manifest(target_dir: str) -> dict:
    path = os.path.join(target_dir, _MANIFEST_FILENAME)
    if os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def _write_manifest(target_dir: str, manifest: dict) -> None:
    os.makedirs(target_dir, exist_ok=True)
    with open(os.path.join(target_dir, _MANIFEST_FILENAME), 'w', encoding='utf-8') as f:
        json.dump(manifest, f)


def extract_if_needed(session: Session, item: LibraryItem, target_dir: str, config: AnalysisConfig,
                      mono_mix: bool, force: bool = False) -> Tuple[str, bool]:
    '''
    :param mono_mix: True for the mono-for-design extraction (decimated to config.target_fs, written to
        <target_dir>/mono.wav), False for the full-quality multichannel "kept" extraction (never decimated
        -- see §4.1 -- written to <target_dir>/multichannel.wav). A caller wanting both calls this twice,
        once with each value -- this function only ever handles one at a time.
    :return: (wav_path, cached) -- cached=True if ffmpeg was skipped because the manifest already recorded
        a matching (source_fingerprint, params_hash) and the wav file still exists on disk.
    '''
    prefix = 'mono' if mono_mix else 'multichannel'
    wav_path = os.path.join(target_dir, f"{prefix}.wav")
    manifest = _read_manifest(target_dir)
    fingerprint = _source_fingerprint(item)
    params_hash = _params_hash(item, config, mono_mix)

    if not force:
        if (manifest.get(f"{prefix}_source_fingerprint") == fingerprint
                and manifest.get(f"{prefix}_params_hash") == params_hash
                and os.path.isfile(wav_path)):
            return wav_path, True

    result = session.extract_with_layout(item.source_path, target_dir, audio_stream=item.audio_stream,
                                         mono_mix=mono_mix, decimate=mono_mix, playlist_name=item.playlist_name,
                                         output_file_name=prefix)
    manifest[f"{prefix}_source_fingerprint"] = fingerprint
    manifest[f"{prefix}_params_hash"] = params_hash
    manifest[f"{prefix}_extracted_at"] = time.time()
    if not mono_mix:
        # flat top-level key -- pipeline.review._read_channel_layout_name() (chunk 2, already shipped)
        # reads exactly this key, not a nested one; do not change this without also revisiting that code.
        manifest['channel_layout_name'] = result.channel_layout_name
    _write_manifest(target_dir, manifest)
    return result.wav_path, False
```

**Manifest shape**, concretely, for an item that had both extractions
run once then the multichannel one force-redone:

```json
{
  "mono_source_fingerprint": "1737000000000000000:483821",
  "mono_params_hash": "3f2b...",
  "mono_extracted_at": 1737000012.5,
  "multichannel_source_fingerprint": "1737000000000000000:483821",
  "multichannel_params_hash": "9ac1...",
  "multichannel_extracted_at": 1737000512.1,
  "channel_layout_name": "5.1"
}
```

### D.5 Tests to add

New `src/test/python/test_pipeline_library_extract_cache.py`. Needs a
real short audio/video fixture ffmpeg can probe+extract -- check
`test_pipeline_orchestrate.py`/`test_pipeline_acceptance.py` for
whatever fixture file(s) this repo's existing extract-related tests
already use and reuse the same one(s) rather than adding a new binary
fixture.

- `test_extract_if_needed_runs_ffmpeg_on_first_call` -- fresh
  `target_dir`, assert `cached is False`, `mono.wav` exists, and
  `manifest.json` now has `mono_source_fingerprint`/`mono_params_hash`.
- `test_extract_if_needed_skips_ffmpeg_on_a_repeat_call` -- call twice
  with identical `item`/`config`/`mono_mix`, assert the second call's
  `cached is True` and (via monkeypatching `Session.extract_with_layout`
  to raise if called, or counting calls) that ffmpeg genuinely didn't
  run again.
- `test_extract_if_needed_reextracts_when_params_hash_changes` --
  second call with a different `AnalysisConfig.target_fs`, assert
  `cached is False`.
- `test_extract_if_needed_reextracts_when_source_fingerprint_changes`
  -- second call with a `LibraryItem.fingerprint` changed (or, for the
  mtime/size fallback path, actually touch/modify the source file),
  assert `cached is False`.
- `test_extract_if_needed_force_always_reextracts` -- identical inputs
  twice, `force=True` on the second, assert `cached is False`.
- `test_extract_if_needed_mono_and_multichannel_are_independently_cached`
  -- call once with `mono_mix=True`, once with `mono_mix=False`, then
  repeat only the mono call again -- assert the repeat is cached and
  the multichannel manifest entries/file are untouched (mtime
  unchanged, or via a call-counting monkeypatch).
- `test_extract_if_needed_multichannel_records_channel_layout_name` --
  a multichannel fixture, assert `manifest.json`'s `channel_layout_name`
  matches the fixture's actual layout (e.g. `'5.1'`), and that a mono
  call never writes that key at all.
- `test_extract_if_needed_multichannel_does_not_decimate` -- compare
  the multichannel output's sample rate against the source's original
  rate (unchanged), vs. the mono output's rate (`config.target_fs`).
- `test_extract_with_layout_is_behaviourally_identical_to_extract` --
  call both on the same input (different `target_dir`s), assert the
  returned wav paths' contents are byte-identical and
  `extract()`'s path equals `extract_with_layout(...).wav_path`.
- Re-run the *existing* `test_pipeline_orchestrate.py` / relevant
  `test_pipeline_review.py` / `test_pipeline_acceptance.py` tests
  unmodified -- they must all still pass after `extract()` becomes a
  wrapper (this is a regression check, not a new test to write).

### D.6 Acceptance checklist

- [x] `Session.extract_with_layout()` added; `Session.extract()`
      refactored to call it, same signature/return type/behaviour.
- [x] `pipeline/library/extract_cache.py` created with
      `extract_if_needed()` and its manifest helpers.
- [x] Manifest uses the exact flat key names in D.4 (`channel_layout_name`
      at top level, not nested) -- confirmed compatible with the
      already-shipped `pipeline.review._read_channel_layout_name()`.
- [x] All new tests (D.5) pass; full suite still green, including
      every pre-existing `extract()` caller's tests unmodified.
- [x] `pipeline_qt_boundary`-style AST scan confirms
      `pipeline/library/extract_cache.py` imports no `qtpy`.

## 9. Open questions / risks

- **JRiver field configuration remains library-specific** -- the endpoint,
  JSON response mode, and `hamcws` adapter are established, but chunk 3
  still needs a real response fixture to confirm the configured names for
  external IDs, artwork, and browse-node children. Nothing outside chunk 8
  depends on that mapping.
- **Whether the library actually carries a TMDB/IMDB id at all is
  unknown and library-specific** -- depends entirely on which metadata
  plugin(s) the user has configured in JRiver, if any. The fuzzy
  `tmdb_lookup(title, year)` fallback (3.1.1) must keep working
  regardless -- id-based resolution is a quality improvement for
  whatever fraction of the library has one, not something the design
  can require.
- ~~No reviewer-facing way to correct a wrong id-based TMDB match~~ --
  resolved by 3.1.2's metadata editor (Chunk 1 / Appendix A); flagged
  here only as a reminder that Chunk 1 is what closes this, not an
  afterthought.
- **Whether the library exposes local poster art is unverified and
  library-specific** (3.1.3) -- same caveat as the TMDB/IMDB id
  question; the TMDB-fallback tier must work regardless, and now also
  needs its existing wiring gap (`publish_reviewed_queue()` never
  passing `poster_path`) actually closed, which Chunk 1 does.
- ~~**Stable `LibraryItem.id` for JRiver**~~ -- resolved: `jriver-<server
  hash>-<Key>` (§3.1). Whether JRiver's `Key` survives a library rescan/
  re-import is still unverified against a real server (chunk 3).
- **`force_design` on an `accepted`/`published` entry**: deliberately
  requires an explicit single-item status reset rather than a
  library-wide flag (see 4.2) -- worth confirming this friction is
  acceptable rather than surprising once someone actually hits it.
- **Cross-machine layout**: this assumes beqdesigner runs on a host
  that can both reach the JRiver server over HTTP *and* read the
  source files directly (ffmpeg needs local/mounted access to
  `source_path`) -- same assumption `model/batch.py`'s existing glob
  search already makes, not a new constraint, but worth stating since
  a "library source" abstraction might tempt someone to assume
  network-transparent file access that doesn't exist yet.
- **Regenerating output 1's project files only on an accept-time
  mismatch** (§3.3) means a reviewer who merely *previews* a
  non-top-pick candidate (digit-key picking in `ReviewQueueDialog`,
  without accepting) never sees that choice reflected in the on-disk
  `.beq` projects -- only the eventual `accepted` pick triggers a
  rewrite, and even then only if the project hasn't already been
  hand-edited (§3.3.1's hash gate). Consistent with "projects reflect a
  decision, not a preview," but worth confirming that's the expected
  mental model before chunk 2 ships.
- **The `pipeline_filter_hash` mechanism (§3.3.1) only inspects the
  master channel** of a multichannel project -- a human who `free()`s
  one slave from the master and edits it independently inside the
  interactive app produces a project this plan cannot detect as edited
  (the master's hash still matches). Since a multichannel edit can now
  be publish-authoritative too (§3.3.1's revised, two-sided
  resolution), this gap is no longer purely a personal-use concern: an
  independently-edited *slave* channel is invisible to this mechanism
  either way, but an edit to the *master* channel (the normal case for
  changing what gets published) is still correctly detected. Accepted
  as a known, narrow gap rather than solved here.
- **Project-file-as-published-source is a real behaviour change** once
  chunk 2 ships, and now a two-sided one (§3.3.1 revised): a title's
  published filter can diverge from every candidate
  `QueueEntry.candidates` ever recorded, from either project file, and
  the two projects can end up in outright conflict if both are
  independently edited to disagree -- a case this plan deliberately
  refuses to auto-resolve rather than silently guessing. Worth the user
  explicitly confirming this is the intended trust model before chunk 2
  ships; it's a meaningful shift from "the pipeline always publishes
  what it designed," and the conflict case in particular needs a real
  UI/CLI answer (deferred to the relevant chunk, not designed here).

## 10. Implementation status vs. this plan (reviewed 2026-09-19)

Verified against the code at `1ebaa4e`; the full suite passes (471 tests).
Everything in §8 marked Implemented is present and tested, except as listed
here. Items are ordered roughly by impact.

**Behaviour gaps -- designed above, not built**

1. **Library artwork tiers 2 and 3 (§3.1.3).** Nothing writes
   `QueueEntry.art_path` from `LibraryItem.art_path` or from TMDB's `poster`
   via `fetch_poster()`; library-driven entries publish chart-only unless a
   human sets artwork in the review dialog.
2. **`project_edit_preserved` reporting (§3.3.1).** A hash-gated skip of a
   human-edited project is silent in both `run_library()` and `sync_library()`.
3. **Write-back of the authoritative project into the other one (§3.3.1).**
   Not implemented; only conflict detection is.
4. **Sync errors are not shown in the GUI (§7).** `_SyncJob` results with
   `'error': 'project_conflict'` are counted as "published".

**Deviations that change idempotency/cost**

5. **`resolve_meta()` runs before the design cache check** (§3.1.1), so every
   rerun re-queries TMDB for every already-designed item, and a TMDB failure
   fails the item outright. Fix: resolve only when `design_if_needed()` will
   actually design (or when the entry has no `meta`), and degrade to
   `item.meta` on TMDB errors.
6. **Design fingerprint omits `keep_multichannel`, metadata and
   `audio_stream`** (§4.2) -- see there.
7. **Kept multichannel extraction is not gated on the source being
   multichannel** (§4.1); a mono source gets a redundant extraction.
8. **`kind` ignores `Media Type`/`Media Sub Type`** (§3.1).

**Still open / unverified**

9. **Chunk 3, the real-server spike, was never done** -- field aliases and
   `Browse/Children` shape are unverified; there is no sanitised fixture.
10. **GUI gaps (§7):** no library-view filter bar (status/name/year/type),
    no source picker or query field, no browse-node selector (raw integer
    spin box), `LIBRARY_IMAGE_OWNER`/`LIBRARY_IMAGE_REPO_NAME` defined but
    unwired, no `LIBRARY_SOURCE_DEFAULT`. First saved MCWS connection only.
11. **`pipeline.library.registry` has no production callers** (§3).
12. **Untested:** `INTERNAL` artwork handling in `jriver.py`; manual
    verification of the review dialog (A.8) is unconfirmed.
13. **Document hygiene (fixed in this review):** the header said "plan only,
    not started"; the manifest shape in §4.1 disagreed with Appendix D.4; the
    chunk table and every appendix checklist were stale.
