# Library sync plan -- source selection, shared JRiver connections and the live findings

> Part of the library sync plan -- **start at the index**: [`../library-sync-pipeline-plan.md`](../library-sync-pipeline-plan.md).
> Contains §11 (§11.1-§11.9). Section numbers are global across the plan; the index maps every `§` to its file.
> Status of what this file describes: **built** (see `status-and-open-items.md` §10 for the exceptions)

## 11. Source selection and shared JRiver connections (added 2026-09-19)

The user's request: the JRiver server configuration should be **shared** with
the JRiver filter manager rather than owned by Library Sync; it moves to its
own Preferences pane; and Library Sync then *picks* a library source -- the
raw filesystem (as Batch Extract does today), a configured JRiver server, or
future sources (Plex, ...) -- with the JRiver root node *chosen* from the
server's browse tree rather than typed. The root-node setting stays on the
Library Sync screen, not in Preferences.

### 11.1 Shared connections (chunk 11)

Storage is unchanged: `JRIVER_MCWS_CONNECTIONS = {'host:port': (auth,
secure)}` (`auth` is `None` or `(user, password)`; legacy 3-tuples are still
read). New: a Qt widget for managing that list (add, test, delete -- lifted
out of `MCWSDialog`) hosted on a new `Preferences -> JRiver` page, and a
small parser giving the rest of the app typed connections. `MCWSDialog` (the
filter manager's zone dialog) keeps only the saved-server list and zones, and
points at Preferences for adding/removing servers. Library Sync's JRiver
settings choose a server from the same list.

**As built (chunk 11):** `model/jriver/connections.py` holds
`SavedConnection`, `parse_connections()`/`load_connections()`/`save_connections()`
and `JRiverConnectionsWidget` (list, add form, Test, Add, Delete; Add is only
enabled after a passing Test, and any edit invalidates that pass). It is the
content of a new `jriverPage` in the Preferences tool box. `MCWSDialog` lost
its add/test/delete controls (`load_zone.ui`) and shows a pointer to
Preferences; it lists `load_connections()` and loads zones as before. Library
Sync now reads the list through `load_connections()` (still "first saved
server" until chunk 13). Behaviour changes worth knowing: the endpoint check
is `host:port` -- the old dialog only accepted a dotted-quad IP, which
rejected hostnames such as `media.local`.

**Follow-up (editing and async test):** selecting a saved server loads it
into the form, so it can be edited: the button becomes **Update**, a change
must still pass **Test** before it can be saved, changing `host:port` renames
the server (rather than adding a second), and **New** clears the selection to
add another. **Test** now runs off the UI thread (`_TestJob` on the global
`QThreadPool`) with a spinning icon and a "Testing..." label; the form, list
and buttons are disabled while it runs, and any exception (not only
`MCWSError`) is shown in the result box. `MCWSDialog`'s zone loading is still
synchronous.

**Follow-up (server aliases):** `/Alive` reports a `FriendlyName`, which
`MediaServer.friendly_name` now exposes (`None` until authenticated or if
blank). It is the server's visible name everywhere a server is listed --
`SavedConnection.label` is `Name (host:port) [user]`, or the old
`host:port [user]` while unnamed -- in the Preferences list, the filter
manager's dialog and Library Sync's server combo. Aliases are stored in their
own preference (`JRIVER_MCWS_ALIASES`, `{endpoint: name}`) rather than in
`JRIVER_MCWS_CONNECTIONS`, whose tuple shape is shared with older code and
already has a legacy 3-tuple form; deleting a server drops its alias. A name
is learned three ways: **Test** (shown as "Connected to X" and saved with
Add/Update; a re-test that reports no name keeps the old one),
`JRiverConnectionsWidget.refresh_aliases()` (called when the Preferences
dialog opens; a background job that asks only servers with no name yet and
silently skips unreachable ones), and the filter manager's dialog once
`get_zones()` has authenticated. A server renamed on its host keeps its old
alias until it is re-tested. Not live-tested: only Browse endpoints were
permitted against the real server, so `/Alive` parsing is covered by a local
fake.

### 11.2 Filesystem source (chunk 12)

`FilesystemLibrarySource(globs)` yields one `LibraryItem` per matching file or
BDMV root, using `FileSearch`'s glob semantics (a directory means `dir/*`,
`recursive=True`). Identity: `fs-<hash of the resolved path>` -- a filesystem
has no key that survives a rename, so a moved file is a new title (documented
limitation, unlike JRiver's `Key`). Fingerprint: left empty so the extract
cache's mtime/size fallback applies. `title`/`year` are not guessed from the
filename; metadata is for the reviewer (or a later resolver) to supply.

**As built (chunk 12):** `pipeline/library/filesystem.py::FilesystemLibrarySource(globs,
extensions=DEFAULT_MEDIA_EXTENSIONS)`. Two additions to `FileSearch`'s
semantics, both for unattended runs: (1) a matched *file* must have a media
extension (`extensions=None` disables this) -- Batch Extract accepts anything
and lets the probe reject it, but here every non-media file in a film folder
would be a failed item; (2) anything under a `BDMV` path component is skipped,
so a recursive glob yields the disc root once rather than the root plus its
clips. Results are sorted, deduplicated by id (so overlapping globs and
symlinks give one title), and a BD root's fingerprint is the stat of
`BDMV/index.bdmv` (a directory's own stat isn't meaningful). Nothing matching
is an empty library, not an error. CLI: `--source filesystem` with repeatable
`--glob`, or `sources.filesystem.globs` in the config file.

### 11.3 Source picker (chunk 13)

A source *kind* is a name, a label and a settings page that can build a
`LibrarySource` from what the user entered. The dialog holds a registry of
kinds and a `QStackedWidget`; adding Plex later means registering one more
kind, not editing the dialog. Built-in kinds: **Filesystem** (globs + a
folder picker) and **JRiver** (server chosen from the shared list, browse node).
The chosen kind and its settings persist in preferences
(`LIBRARY_SOURCE_DEFAULT`, ...). `pipeline.library.registry` remains the
headless registry; the GUI kind registry is separate because it also carries
widgets.

**As built (chunk 13):** `model/library_sources.py` --
`LibrarySourceKind(name, label, create_page)`, `register_source_kind()` /
`registered_source_kinds()`, an abstract `SourcePage` (`load(prefs)`,
`save(prefs)`, `build_source()` raising `ValueError` with a user-facing
message), and the two built-in pages. `LibrarySyncDialog` builds one page per
registered kind into a `QStackedWidget`, preselects `LIBRARY_SOURCE_DEFAULT`
(default `filesystem`), and on Run/Sync persists the kind and every page's
settings. New preferences: `LIBRARY_SOURCE_DEFAULT`, `LIBRARY_FILESYSTEM_GLOBS`
(a list), `LIBRARY_JRIVER_CONNECTION` (the chosen server's `host:port`);
`LIBRARY_JRIVER_BROWSE_NODE` is unchanged. The filesystem page takes globs
one per line with an Add folder button. Pages are built when the dialog
opens, so servers added in Preferences while it is open appear next time.

### 11.4 Browse-node picker (chunk 14)

A dialog with a lazily-expanding tree over `hamcws.MediaServer.browse_children()`
run off the UI thread; selecting a node fills the node id and shows its path
(`Video > Movies > Needs BEQ`). **Caveat -- unverified:** `browse_children()`
returns a `{Item name: Item text}` dict, and the exact response shape (which
is the display name, which the id, how a leaf is marked) has not been checked
against a real server (chunk 3). The parser is written defensively, and the
numeric field remains so a wrong guess never blocks a run.

**As built (chunk 14):** `pipeline/library/jriver.py::list_browse_children()`
(Qt-free; same running-loop guard as `list_items()`) returns
`BrowseNode(id, name)` values, reading hamcws's `{Item name: Item text}` dict
as `{display name: node id}` and skipping any entry whose text isn't an
integer. `model/browse_node_picker.py::JRiverBrowseNodePicker(parent, fetch,
current_id)` is a tree with a "Whole library (root)" node (id `-1`) at the
top; children load off the UI thread only when a node is first expanded
(a failed load is shown inline and retried on the next expand; a node found
to have no children loses its expander). `fetch` is injected, so the dialog
knows nothing about MCWS. The JRiver source page gets a **Choose...** button
(enabled once a server is selected) that opens the picker on the current id;
the choice fills the id spin box and a path label (`Video > Needs BEQ`),
persisted as `LIBRARY_JRIVER_BROWSE_PATH` and cleared if the id is then typed
by hand. The spin box stays as the fallback the caveat above requires.

**Still unverified:** the response shape. It is exercised only against a
local fake serving `<Item Name="Movies">21</Item>`-style XML; if a real
server returns something else the picker will show an empty or partial tree,
and the fix is confined to `_map_children()`. Capturing a real
`Browse/Children` response is now part of the chunk 3 spike.

### 11.5 Live findings against a real JRiver server (2026-09-19)

With the user's permission, a local server was queried with **Browse/Children
and Browse/Files only** (nothing else: no playback, no `/Alive`, no
`Library/Fields`); credentials and file paths were not printed or recorded.
What that established, and what it changed:

- **`Browse/Children` shape confirmed** for the root and two deeper levels:
  hamcws returns `{display name: node id (string)}` (e.g. `Video -> '3'`),
  and a leaf node returns `{}`. The picker's reading (§11.4) is right. One
  structural limit: the dict is keyed by name, so two *siblings with the same
  name* would collapse into one entry -- not seen, but not detectable.
- **Year:** requested as `Year`, returned as `Date (year)`. Every title had
  no year until fixed (commit following `cc53ffa`).
- **Kind:** `Media Sub Type` is `Movie` / `TV Show`. Twelve films in a 1283-title
  Movies node carry a `Series` value and one a stray `Season`; the old rule
  called them TV. Verified after the fix: 1283 movies / 1657 TV shows in the
  two nodes.
- **Paths are Windows paths** (`W:\...`) although the client is Linux: the
  server is a Windows host. JRiver treats this as a server-side bug, so the
  app carries a configurable **path mapping** instead (§11.6, built).
- **`Image File` is a bare file name** (1162 of 1283 films, 1533 of 1657
  shows), not an absolute path and never `INTERNAL` here, so it can only be
  found relative to the media file's folder (§11.6).
- **`IMDB`/`TheMovieDB` were absent** from every row of both nodes. With
  `Library/Fields` (later permitted) the cause is clear: they are not fields
  in this library at all. It defines 296 fields; the id-like ones are `IMDb ID`,
  `TMDb ID`, `TheMovieDB Movie ID`, `IMDb Series ID`, `TheMovieDB Series ID`
  and `TheTVDB Series ID`. Populated counts: films -- `IMDb ID` 1239 / 1283,
  `TheMovieDB Movie ID` 277, `TMDb ID` 5; shows -- `TheTVDB Series ID` 1587 /
  1657, `TheMovieDB Series ID` 896, `IMDb ID` 559 (episode level), `IMDb
  Series ID` 242. After the defaults changed (§11.7): 1241 films and 896
  shows get an id from the library; 42 and 761 fall back to the fuzzy search.
  Library/Fields also confirms `Date (year)` is the internal name of the field
  displayed as `Year`, and that a user's own fields differ from a default --
  hence the customisation.
- **`Date Modified` and `File Size`** are present on every row, so the extract
  cache uses JRiver's fingerprint rather than a local stat.
- **Path mapping, checked live** (then the share was mounted at `/media/films`
  with the one rule `W:\` -> `/media/films`): 933 of 1283 films and 1647 of
  1657 shows existed locally, and artwork resolved beside the media for 1159
  films and 1492 shows. That confirms the translation and the bare-`Image
  File` handling. Only `stat` and Browse/Files were used; nothing was opened.
- **Discs are reported as pseudo-files, not folders.** 346 of the missing
  films were `<disc>\BDMV\index.bluray;1`, which is not a file; the disc
  folder two levels up is a real BDMV root in all 346. `_disc_root()` now
  reports that folder (before path translation), which the pipeline already
  handles (it resolves the main title). After this, 1279 of 1283 films exist
  locally. **Not handled:** `BDMV\PLAYLIST\index.bluray;N` (4 shows -- names a
  playlist on a multi-episode disc and the mapping from `N` to a title is
  unknown, so guessing the main title would pick the wrong episode), which is
  passed through as reported and fails at extraction. A DVD's
  `VIDEO_TS\VIDEO_TS.dvd;N` (6 shows) and `index.bluray3d;N` were handled
  afterwards -- see §11.8.

### 11.6 Path mappings (chunk 15)

JRiver reports paths as its own host sees them, in Windows form (`W:\Films\x.mkv`,
or UNC), which ffmpeg on another machine cannot open. The user's decision: add
a mapping in our preferences rather than depend on a JRiver fix.

- **Rules.** `PathMapping(source, target)` in `pipeline/library/pathmap.py`
  (Qt-free). `translate_path()` picks the rule with the **longest source prefix**
  that contains the path; matching ignores case and treats `\` and `/` alike;
  a prefix only matches whole components (`W:\Film` does not claim
  `W:\Films\x.mkv`); a trailing separator on either side is irrelevant; a
  drive root (`W:\` or `W:`) and UNC prefixes work; a path no rule contains
  passes through unchanged, so a Windows host needs no rules at all.
- **Where they apply.** `JRiverLibrarySource(path_mappings=...)` translates
  `Filename` into `LibraryItem.source_path`. The id and fingerprint do not
  depend on it, so adding or changing a mapping never invalidates a cache.
  `Image File` -- a **bare file name** on most live titles -- is looked for
  beside the *translated* media file; an absolute one is translated too.
- **Storage and UI.** Per server, in `JRIVER_MCWS_PATH_MAPPINGS`
  (`{'host:port': [[server, local], ...]}`), loaded into
  `SavedConnection.path_mappings`. Preferences -> JRiver has a table for the
  selected server (Add/Remove); edits **save at once** and do not require a
  connection test. A half-typed row is shown but not saved until both cells are
  filled. Updating or renaming a server carries its mappings across; deleting
  it drops them. Library Sync's JRiver source is built with the chosen
  server's rules.
- **CLI.** `sources.jriver.path_mappings` in the config file
  (`[{from, to}, ...]`) or repeatable `--path-map SERVER=LOCAL`; flags replace
  the file's rules.
- **Not built.** No warning when a path is still Windows-style on a non-Windows
  host (an unmapped library fails per item at extraction); no folder picker on
  the local column; no UI for mappings on the filesystem source (not needed).

### 11.7 Configurable external-id fields (chunk 16)

Users keep the IMDb/TMDb ids in different fields (a plugin's, JRiver's own, a
custom one), so which JRiver field feeds each identifier is configuration.

**Backend (built).** `pipeline/library/jriver.py`:
- `DEFAULT_EXTERNAL_ID_FIELDS`, **per kind** -- films: `imdb` <- `IMDb ID`,
  `tmdb` <- `TheMovieDB Movie ID` then `TMDb ID`; TV: `imdb` <- `IMDb Series
  ID`, `tmdb` <- `TheMovieDB Series ID`. TV uses the *series* fields because a
  TV row's plain `IMDb ID` is the episode's, which TMDB's tv lookup cannot
  resolve. Several names per identifier are tried in order, first value wins;
  an unset numeric id (`0`) is ignored.
- `normalise_external_id_fields(config)` accepts `None` (defaults), a nested
  `{movie: {imdb: [...], tmdb: [...]}, tv: {...}}` (per-kind override; anything
  unmentioned keeps its default; `[]` switches an identifier off) or the flat
  `{imdb: [...], tmdb: [...]}` (both kinds). A field may be a name or a list.
  Unknown identifiers raise `ValueError`.
- `JRiverLibrarySource.requested_fields` = the fixed fields plus every
  configured id field, so a custom field is actually requested.
- `list_library_fields(host, port, ...)` (Qt-free, running-loop guard) returns
  `LibraryField(name, display_name, data_type)` from `Library/Fields`; `name`
  is the internal name a request and a reply use.
- The CLI's existing `sources.jriver.external_id_fields` takes either form.

**Editor and storage (built).**
- `SavedConnection.field_mappings` holds a server's **overrides only** (the
  nested form above), stored per server in `JRIVER_MCWS_FIELD_MAPPINGS`
  (`{endpoint: {kind: {identifier: [names]}}}`), so a default that improves
  later reaches everyone who never customised it. Updating or renaming a
  server carries it across; deleting the server drops it.
- Preferences -> JRiver has a **Metadata fields for the selected server**
  group (`JRiverFieldMappingsWidget`): a 2x2 grid of editable combos --
  films/TV by IMDb/TMDb -- each holding one or more field names, comma
  separated, first with a value wins. Edits **save at once**; an empty box
  switches that id off; typing a default back removes the override; a
  **Defaults** button clears them all.
- **Load fields from server** fetches `Library/Fields` on the thread pool
  (spinner, form disabled meanwhile, failure shown inline) and offers the
  `String`/`Integer` fields as choices (paths, dates, images and so on cannot
  hold an id): 230 of the live server's 296. Once loaded, any configured name
  the server does not define is flagged ("Not a field on this server: ...").
  Nothing is fetched until asked, so opening Preferences makes no request.
- The page is now inside a scroll area (it holds a list, a form and three
  editors).
- Library Sync's JRiver source is built with the chosen server's overrides
  (`external_id_fields`) and so requests exactly those fields.
- Live check (Library/Fields only): every default name exists on the real
  server and none would be flagged.

**Possible extension, not done:** 1587 of 1657 shows carry `TheTVDB Series ID`,
which TMDB's `/find` can resolve (`external_source=tvdb_id`). A `tvdb`
identifier would identify most of the shows that currently fall back to the
fuzzy search.

### 11.8 DVD-Video support (chunk 17)

Blu-ray rips were supported; DVD rips (a `VIDEO_TS` folder) were not, and JRiver
reports a DVD as `<disc>\VIDEO_TS\VIDEO_TS.dvd;N`.

**Reading the disc.** ffmpeg's `dvdvideo` demuxer (libdvdread/libdvdnav)
understands DVD titles, which are *program chains* that may play cells out of
order, so concatenating `VTS_nn_m.VOB` files (the Blu-ray approach) would not
reproduce a title and would mix a multi-episode disc together. So the pipeline
reads `-f dvdvideo -title N` against the **disc folder**. That needs an ffmpeg
built with libdvdread; without it the probe fails with a clear message ("This
ffmpeg cannot read DVDs..."). Encrypted (CSS) discs need a CSS library;
decrypted rips are unaffected. Checked live: the local ffmpeg 8.1 has the
demuxer.

**Choosing a title (`model/dvd.py`, pure Python).** `list_titles()` reads
`VIDEO_TS.IFO`'s title table (`TT_SRPT`) and, for each title, its title set's
`VTS_nn_0.IFO` (`VTS_PTT_SRPT` -> first program chain -> `VTS_PGCIT` playback
time, BCD h/m/s/frames at 25 or 29.97 fps). Titles are returned longest first;
`resolve_main_title(root, title_name=None)` takes the longest, or a title by
number, and returns a `ResolvedDvdTitle` shaped like the Blu-ray
`ResolvedTitle` (`playlist.duration_s`, `display_name`, plus `input_options`
`{'f': 'dvdvideo', 'title': N}`). Offsets follow libdvdread's `ifo_types.h`.
The parser was checked against a real disc: a 10.00 s stub title matched
ffprobe, and title 3 extracted through the whole pipeline to a wav of
1569.4 s, exactly the IFO's 1569.4 s, with 5.1 detected. Titles in a missing
or unreadable title set are skipped; case of `VIDEO_TS`/file names is ignored.

**Wiring.** `Executor(input_options=...)` applies the options to the probe and
both commands (they must precede `-i`). `Session.extract_with_layout()`
resolves a DVD root (or its `VIDEO_TS` folder) as it does a BDMV root, with
`playlist_name` as the **title number** (`'3'`). Batch Extract's
`ExtractCandidates.append()` accepts a DVD (using the disc folder for the
entry id even if `VIDEO_TS` itself matched). `FilesystemLibrarySource` yields
a DVD root once (fingerprint: stat of `VIDEO_TS.IFO`) and skips anything under
`VIDEO_TS`/`BDMV` (now case-insensitive). JRiver's `VIDEO_TS.dvd;N` and
`index.bluray3d;N` pseudo-files are reported as the disc folder before path
translation. Live: films missing locally 14 -> 5.

**Limits.**
- **Multi-episode discs.** The longest title is normally "play all" (the sample
  disc's title 1 is 4953.52 s = the three episodes, 1569.4 + 1597.0 + 1787.1
  s, summed). Extract one episode by naming its title, via
  `Session.extract(..., playlist_name='3')` or `LibraryItem.playlist_name`.
  Batch Extract only takes the main title.
- **JRiver cannot say which title an item is.** Its DVD entries are named
  "Chapter 1..6", share one path, and the fields inspected (`Duration`,
  `Playback Info`, `DVD Video Info`, `Playback Range`) do not identify a title
  or chapter. So every JRiver item on a disc resolves to that disc's *main*
  title: several items, one audio. Fixing that needs a way to map an item to a
  title (a per-item override, or de-duplicating by disc).
- **No interactive title picker** for DVD in the single-file Extract dialog
  (Blu-ray has `BdmvTitlePickerDialog`); `list_titles()` provides what one would
  need.
- **No end-to-end DVD test in the suite.** A disc libdvdread accepts needs full
  navigation tables; the tests build IFO fixtures (`src/test/python/dvd_fixtures.py`)
  for title selection and assert the ffmpeg command, and the real read was
  verified by hand as above.

### 11.9 TV seasons: per episode, or the whole season as one track (chunk 18)

The user's requirement: a config option to treat a **season as a single track**
or to produce **per-episode filters**, with the output metadata marked with the
episode or episodes in scope.

**What the catalogue understands** (read from `beqcatalogue/__init__.py`, not
assumed). *Structured season* --
`<beq_season id="<TMDB season id>"><number>1</number><episodes count="8">1,2,3</episodes></beq_season>`
-- where `episodes` lists the episodes the filter covers and `count` is the
season's total; the site calls the season **complete** when every one of
`count` is listed, and otherwise shows `S1E1-3`. Omitting `episodes` reads as
the whole season. The `id` attribute is **required**: `parse_season` throws
away the season on a missing one. *Legacy* -- plain `<beq_season>1</beq_season>`
with the episode in the note (`E3`, `E1-8`, `S1-E3`); a contiguous range only.

**Metadata (built, `pipeline/metadata.py`, `model/minidsp.py`).** `BeqMetadata`
gains `episodes: List[int]`, `season_id`, `season_episode_count`. `to_dict()`'s
`beq_season` is the structured dict when `season`, `season_id` and a positive
`season_episode_count` are all known (`structured_season`), else the plain
season text; in the plain case a *contiguous* episode range is written into the
note (`E3`, `E1-3`) -- but only into an **empty** note, and never for a gap,
neither of which the legacy form can express. The XML writer's `beq_season`
dict branch emits the structured element (`episodes` omitted when none given).
A show with no season is written exactly as before.

**Library items and TMDB (built).**
- `pipeline.metadata.tmdb_season_info(series_id, season, api_key)` reads
  `/tv/{id}/season/{n}`: the season's TMDB id and its episode count (`None`
  on a 404). `parse_episodes()` / `format_episodes()` convert `"1-3, 5"` <->
  `[1, 2, 3, 5]`.
- `LibraryItem` gains `season: Optional[str]` and `episodes: tuple[int, ...]`
  (one for an episode; several once a season is grouped into a single track).
- JRiver TV items are now **titled by their `Series`** (falling back to the
  name), where before the title was the episode's name -- which would have sent
  "Chapter 3" to TMDB as the show. `season`/`episodes` are set only from
  *numeric* `Season`/`Episode`. `display_name` is `Series S01E03 name`.
  `meta['season']` is no longer used. Live (Shows node): 84 series, 182
  series/season groups, season and episode on 1646 of 1657 items.
- `resolve_meta()` puts `season` and `episodes` (from the library; also what an
  item gets with no TMDB key, or when TMDB fails) in the metadata and, for a TV
  item with a season, asks TMDB for the season id/count. A failed or empty
  lookup keeps just the season and episodes, so the plain-season/note fallback
  applies. `item.meta` still wins over everything.
- The review dialog's metadata tab has an **Episodes** field (`1-3, 5`).
  Unlike the text fields a blank box is a real answer -- no episodes in scope --
  so it is written as `[]`; an unparseable entry is reported and nothing is
  saved.

**`tv_mode` (built; `pipeline/library/season.py`, `pipeline/library/run.py`).**
`LibraryRunConfig.tv_mode` is `episode` (the default; a filter per episode,
each with its own season/episode metadata) or `season`. Set by the Library
Sync dialog's **TV shows** combo (`LIBRARY_TV_MODE`), the CLI's `--tv-mode`, or
`run.tv_mode` in the config file; an unknown mode is rejected when the config
is built.

In `season` mode `plan_units()` replaces the numbered episodes of one series
and season with a single `SeasonGroup`, placed where its first episode was
(films, other items and anything without a series, season or a single
episode number pass through untouched; a second item for an episode already
present is ignored, first wins). `_run_season()` then:
1. extracts every episode as in episode mode -- each into its own work
   directory and cache, so **switching mode re-extracts nothing**;
2. joins their mono, decimated wavs in episode order into one
   `<work_dir>/<season id>/mono.wav` (`season_track_if_needed()`; streamed in
   blocks with `soundfile`, 24-bit kept; skipped when every episode wav is
   unchanged, rebuilt when one is added, dropped or re-extracted; episodes that
   disagree on rate or channel count are rejected);
3. designs that track **once** (`project_dir` is the season directory, so the
   `.beq` project is made from it) and queues one entry.

- **Season id** `some-show-s01-1a2b3c`: series slug, season number and a short
  hash of the title and season, so it is filesystem-safe and **independent of
  which episodes are present** -- the review entry and caches survive a season
  gaining an episode. (Two copies of a series with different capitalisation
  are one season.)
- **What is marked in scope.** The season item's `episodes` are the ones that
  **actually made it into the track**: an episode that will not extract is
  reported in `failed` and left out, rather than sinking the season, and the
  metadata then lists only the rest (so a later run that finds it re-designs,
  since the track fingerprint changed). A season with no extractable episode
  fails as one item.
- **Metadata** follows §11.9's rules: with a TMDB key, the structured season
  with all in-scope episodes (the catalogue then shows the season as
  complete only if they cover TMDB's count, else `S1E1-3`); without one, the
  library's season and episodes still go in, written as the plain season and an
  `E1-8` note where the range is contiguous.
- **Reporting.** `LibraryRunReport.seasons` maps a season id to its episode
  item ids; `on_item_done` ticks once per season; the dialog's summary says
  "N season(s) joined".
- **Limits.** Multichannel is not kept for a season (`keep_multichannel` is
  ignored -- there is no single multichannel file to link). A season is only as
  complete as the browse node: episodes elsewhere in the library are not
  found. The joined track is a plain concatenation with no gap or level match
  between episodes, which is fine for the long-term spectral average the
  design uses. Not run against the real library: the local JRiver server was not
  reachable when this was written, so the grouping was checked with unit tests
  (real audio for the joining), not live.
