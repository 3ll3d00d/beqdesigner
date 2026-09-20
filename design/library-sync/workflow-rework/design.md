# Library sync plan -- workflow rework: design

> Part of the library sync plan -- **start at the index**: [`../../library-sync-pipeline-plan.md`](../../library-sync-pipeline-plan.md).
> Contains §12 (§12.1-§12.12, §12.14-§12.15). Section numbers are global across the plan; the index maps every `§` to its file.
> Status of what this file describes: **design agreed 2026-09-19**; the backend (§12.4-§12.9 and §12.11: profile, discovery, states, publish/commit, revise, stage entry points, bulk accept) is **built through chunk 25**, the work list window with its actions and its settings drawer (§12.10, chunks 26a-26c) is built; the title page is **not built**; supersedes §7

## 12. Workflow, discovery and a work-list UI (designed 2026-09-19; backend built, UI **not built**)

Status: **the backend is built (chunks 19-25, see the chunk table in the index); §12.10's work list, its actions and its settings drawer are built (26a-26c), the rest of §12.10 (the title page) is design only**; §12.13 is the build order.
Every open decision raised while designing it was settled with the user on 2026-09-19 and is
recorded in §12.12. This section **supersedes §7's "as built" GUI** (Run tab + Review tab) and
absorbs T9 and T10 of §10.

### 12.1 Why

Library Sync as built (§7) is a form: one source, one set of paths, a "Run library" button, then a
Review tab. Reading the code and rendering the dialog offscreen (2026-09-19) found:

- The Review tab is **empty on open**: `ReviewQueueDialog` is only created in `__run_finished`, so
  reviewing an existing queue, or just syncing, needs a run first.
- The §3.3.1 workflow (open the `.beq` project, edit the filter, publish that) has **no GUI affordance**:
  nothing opens `<id>.mono.beq` or shows whether it was edited.
- When TMDB does not resolve (no key, or an error), `meta` holds only season fields, so the queue table
  shows the raw id (`jriver-3fa9c2-1234`); the library's own title/year are dropped.
- Two publish paths exist (the embedded review dialog's XML-only button, and Sync), and only one reads projects (T9).
- `Enter` is bound window-wide to Accept, so Enter in a metadata field accepts the entry
  (**verified at runtime, §12.14**); an unsaved metadata edit is lost on Accept; an accepted entry cannot be reopened.
- `report.failed` (`(id, "ExcType: message")`) is reduced to a count; progress is an indeterminate bar and an
  opaque id; no cancel. Failures are not remembered, so a bad item retries on every run.
- Nothing can say what work exists without doing it: `run_library()` lists, extracts and designs in one pass.

The deeper cause is that the UI models one action ("run"), while the real workflow has phases at
different cadences, each independently repeatable (§12.3). The design below starts from that.

### 12.2 Vocabulary

- **Catalogue profile** -- everything one catalogue needs: an ordered list of sources, the work/queue
  directories, the two repos, the designer and analysis settings, TMDB key, `tv_mode`, ignore rules.
- **Source item** -- what a `LibrarySource` yields (`LibraryItem`).
- **Title** -- one unit of work in the catalogue: a film, an episode, or (in `tv_mode='season'`) a season
  (the existing `SeasonGroup`). A title is owned by exactly one source item (§12.4).
- **Stage** -- `extract`, `design`, `review`, `publish`, `commit`. Each has its own state (§12.6).
- **Needs** -- the single next thing a title needs, derived from its stage states. Drives the work list.
- **Tier** -- `attention` > `human` > `machine` > `done`, the sort/visibility class of a `needs` value.

### 12.3 Phases and cadence

| Phase | Cadence | Triggered by | Unattended? |
|---|---|---|---|
| 1 Configuration | rare | human | n/a |
| 2 Discovery | on open, on demand, nightly | human or cron | yes |
| 3a Extract, 3b Design | batch, hours | human or cron | yes |
| 3c Review / refine | interactive sessions | human | **no** |
| 3d Publish (write to the local repos) | after a review session | human | no |
| 3e Commit (commit + push to the remotes) | deliberate, batched | human | no |

The review gate (§1) is preserved and extended: nothing from review onwards runs unattended.

### 12.4 Phase 1 -- configuration and the union of sources

**Profile (proposed).** The CLI config file's shape (`run:`, `sync:`, `sources.<name>:`, `designers:`),
extended with an **ordered** `sources:` list, an `ignore:` rule list, and the repo/work/queue paths. The GUI
reads and writes the same file; `QSettings` keeps only "which profile is current". One schema for GUI,
CLI and documentation. The profile is **durable** state.

**Priority.** Sources are listed in priority order. When two source items are the same title, the first
source wins.

**Clashes** (evaluated at discovery, §12.5):

- **Hard clash -- same media file.** Same path after path mapping, case-folding and normalising a BDMV/DVD
  root to its disc folder (`_disc_root()`). Deterministic and needs no disk read. The loser is **shadowed**:
  it is not a title; the winner records "also in <source>".
- **Soft clash -- same title, different file.** Same `tmdb`/`imdb` id, else title+year+kind. May be a real
  second entry (edition, audio track), so it is **flagged, never dropped**: a "Possible duplicate" badge and a
  filter in the work list. Both items remain titles.

**Sticky ownership.** A title's id is its work directory, queue entry and catalogue filename
(`<xml_dir>/<entry.id>.xml`, `pipeline/review.py`). Reordering priority must not change it, or an expensive
extraction is orphaned and a second XML published for the same film. So once an item id has been claimed for a
title it **keeps that id**; another source takes over only if the owner's item disappears. Claims are
**reconstructible** (every existing queue entry and work directory is a claim), so the discovery index (§12.5)
stays a disposable cache.

**Ignore rules.** `ignore:` entries match `LibraryItem` fields: `source`, `path` (prefix or glob), `title`
(regex), `year` (comparison), `kind`, `external_ids`. Examples: "path under `/films/Kids/**`", "kind = tv",
"year < 1960". Rules are evaluated at discovery; a matching item is **Done -> Ignored**, labelled with the rule,
and deleting the rule brings it back (the state is derived, not flagged). A **per-title ignore** is a separate
explicit flag with an optional reason, stored in the profile (not the cache). Prefer library-side curation
(a JRiver browse node) where the source has it; rules mainly serve filesystem sources and cross-source cases.

### 12.5 Phase 2 -- discovery

Discovery is its own operation. It reads and diffs, writes an index, and **never extracts or designs**.

**Inputs.** (1) each source's listing; (2) the outputs: extract manifests
(`<work_dir>/<id>/manifest.json`), queue entries, project hashes, and the state of both catalogue repos.

**Trust rules.**

- **Library sources (JRiver):** one bulk `Browse/Files` call; **no media file is read or stat'd**. The fingerprint
  (`Date Modified` + `File Size`) is the library's. The one disk touch there was `_local_art_path`
  (`os.path.isfile` per item); **built in chunk 24**: a listing now only names `LibraryItem.art_candidates` and `artwork.resolve_art()` checks
  them at design time, where artwork is actually resolved.
- **Filesystem source:** one `stat` per file (`BDMV/index.bdmv` / `VIDEO_TS.IFO` for discs). No file contents are read.
- Whether a JRiver file exists locally is **not** checked at discovery. A wrong path mapping surfaces as an
  extract failure with its reason (below) rather than costing a stat per title on every scan.

**"What would run" and "what runs" must not diverge.** The states are computed by pure functions split out of the
existing `*_if_needed()` wrappers -- `extract_status()` (manifest vs `source_fingerprint` + `params_hash`,
`extract_cache.py`) and `design_status()` (`design_fingerprint()` vs `QueueEntry.design_fingerprint`,
`design_cache.py`) -- which the wrappers then also call.

**Index (built, chunk 24: SQLite in `<work_dir>/library-index.sqlite`, stdlib `sqlite3`; `pipeline/library/index.py`).** One row per
title: catalogue id, owning source and item id, path, source fingerprint, metadata snapshot (title/year/kind/external ids), `last_seen`,
shadowed duplicates, per-stage state, `needs`/`tier`, **`state_since`** (when the title entered its current
state -- the queue JSON's mtime is not usable, since every edit changes it), and failure memory. The index is
a **cache**: deleting it costs a rescan, nothing else (claims are reconstructed, ignores live in the profile). The frozen schema is
in "Index schema" below.

**Refresh.** The GUI shows the cached index immediately (stale-while-revalidate) and rescans **per source, on
demand**, showing each source's "last scanned". A rescan highlights what is **new since last scan**. Cron
scans first. As built, `scan()` lists the sources **without holding the index's lock**, so readers (`titles()`, `summary()`,
`sources()`, `record_failure()`) see the last scan until the new one is written; two scans do not overlap. A source that lists
**nothing** when it listed some at the last scan (an unmounted share matches no files and raises nothing) is treated like a failed
listing -- its previous listing is kept and the scan reports it -- unless `scan(allow_empty=True)` / `scan --allow-empty`.
`refresh()` (after a run) takes the sources from the index's own `sources` table, not from the profile passed in, so an ad-hoc
profile with other source names cannot make every title look gone; an index with no recorded sources is left alone.

**Failure memory.** An extract/design failure is recorded with the fingerprint and params it failed against.
It is retried only on an explicit **Retry failed**, or when either changes. A transient failure (a NAS offline, a designer down) is therefore
sticky until then, so `run` prints a stderr warning ("N titles skipped: failed earlier ... use --retry-failed", exit status unchanged) whenever it
skipped any. (Today `LibraryRunReport.failed`
is not persisted.)

**Units.** `plan_units()` (tv_mode season grouping, `pipeline/library/season.py`) moves out of `run_library()`
into discovery, so the units the user sees are the units that get worked on.

**Gone.** An item that has left its source: if it has no outputs it is dropped; otherwise it is kept as **Done**
with a "gone from source" badge. Outputs are never deleted automatically.

**Superseded (tv_mode changed).** A row whose items are *still listed* but are now grouped under another row (episode rows after
`tv_mode` became `season`, or a season row after it went back) is not gone. With no queue entry it is dropped (its extraction serves the
new row); with one it is kept, re-read from its outputs so it still needs what it needs (a pending review stays in the human tier),
with `superseded by <new id>` appended to its detail (`ScanResult.superseded`; no schema change).

**Repo awareness.** Discovery reads the local XML repo. A title whose TMDB id already appears there but which
this profile did not publish is labelled **Already in catalogue** -- informational only; it stays in the list
(and can be ignored). Matching must use the XML's TMDB id, not the filename (ours are entry ids; others' are not).
The element is `<beq_metadata><beq_theMovieDB>`, and the XML has no movie/tv kind, so match on it together with
`<beq_season>` being non-empty (§12.14).

#### Index schema (FROZEN at chunk 24, `SCHEMA_VERSION = 1`)

Versioned by `PRAGMA user_version`. **Rule for any other version** (older, newer, or a file that is not a database): drop *our* tables (`meta`, `sources`,
`titles`, `failures`, `repo_xml`) and recreate -- the index is a cache, the next `scan` refills it, and `generation = 0` tells the caller
it has never been scanned. Nothing is ever migrated. A SQLite file that has none of our tables is somebody else's and is **refused**
(`IndexFileError`), never dropped; and `status` opens the index **read-only** (`LibraryIndex(path, readonly=True)`), so it never
creates, migrates or drops anything. A later chunk that needs a column bumps `SCHEMA_VERSION`; the cost is one rescan, but every
consumer of the columns below must then be revisited, which is why the list is meant to be complete. `test_the_frozen_schema_in_the_design_doc_is_the_one_in_the_code`
keeps this block identical to `index.SCHEMA`.

```sql
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
    external_ids          TEXT NOT NULL DEFAULT '{}',
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
```

What a work list needs, and where it is (chunks 26-27 read these; they never recompute them):

| A work list shows (§12.6, §12.10) | Column |
|---|---|
| title, year | `title`, `year` (the queue entry's metadata once it has one, else the library's) |
| source | `source` (the owning source's name in the profile); `also_in` lists the sources whose copy of the file was shadowed |
| Needs, and the strip's counts | `needs` (`attention` `review` `extract` `design` `publish` `commit` `done`), `tier` (`attention` `human` `machine` `done`); `LibraryIndex.summary().counts` |
| detail / reason text | `detail` (one line, e.g. `conf 0.62 - 3 candidates`, `extract failed: ...`, `gone from source`); `failure` is the remembered message |
| confidence of the top candidate, candidate count | `confidence`, `candidate_count` |
| Waiting (and the sort: tier, then oldest first) | `state_since` (epoch seconds); `LibraryIndex.titles()` orders by tier then `state_since` |
| new-since-scan marker | `first_seen_generation = meta.generation` (`TitleRow.is_new`, `summary().new`); the generation is bumped by every `scan()` and is 0 after a rebuild |
| flags | `ignored` (the rule or "ignored by you", empty if not), `shadowed_by` (the owner's id), `gone`, `duplicates` (JSON list of ids), `in_catalogue`; `TitleRow.flags` names them |
| per-stage state (title page, filters) | `extract_state`, `design_state`, `review_state`, `publish_state`, `commit_state` (vocabularies in `state.py`) |
| last scanned, per source | `sources.last_scanned` (last attempt), `last_ok`, `last_error`, `item_count` |
| what a row is | `unit` (`item` or `season`), `item_id`, `members` (a season's episode ids), `path`, `display_name`, `kind`, `season`, `episodes`, `external_ids` |

Notes on the columns that are not self-explanatory:

- **`id`** is the catalogue id -- the queue entry's, the work directory's, the XML's file name. For a source item it is `LibraryItem.id`;
  for a TV season in `tv_mode='season'` it is the season id, and the episodes are in `members` (they are not rows of their own).
- **`items`** is the JSON of every source `LibraryItem` the row consumed (one, or a season's episodes). It is what lets a source that is down,
  or one skipped by `scan(only=...)`, still take part in the merge: its last listing is re-read from here, so **one source being down
  never makes its titles vanish**. A `gone` row is not re-read.
- **Shadowed** items are rows too (`shadowed_by` set, `needs = done`), so the work list can show and count them; the owner's `also_in` names
  their source. **Ignored** titles are rows with `ignored` set, evaluated like any other, and `needs = done`.
- **`gone`** rows are kept only if the id still has outputs (a queue entry or a work directory); one with none is dropped and reported in
  `ScanResult.dropped`. Its other columns are as they were.
- **`entry_summary`, `digest_key`, `current_digest`, `conflict`** are caches keyed on the queue entry's mtime and size and on the artwork's
  and projects' stats, so a rescan re-reads only what changed (an entry carries the whole average curve). They are not for display.
- **`failures`** is separate from `titles` so a failure recorded by `run_library` survives a rescan and a title that was never scanned. A failure
  applies while the source fingerprint **and** `status.failure_key()` (the extract parameters, or the designer, analysis and coverage) are
  unchanged; a scan deletes one that no longer applies, and `clear_failure()` is *Retry failed*.
- **`repo_xml`** caches the XML repo's TMDB ids by relative path, mtime and size, so an unchanged file is not parsed again.
- **`state_since`** moves only when `needs` changes. A rebuild sets it to the rebuild time for every title (accepted, documented).
  A rebuild also sets `generation` to 0 and forgets the recorded `sources` and `last_scan_at` (even on a live index), so the next
  command that works from the index scans first.
- **`last_seen`** is bumped only for titles whose source was actually listed by that scan (not for a source that was down, one skipped
  by `only`, or a `refresh`).

### 12.6 The stage state machine

| Stage | States | Made stale by |
|---|---|---|
| Extract | none / current / stale / failed | source fingerprint; extract params (stream, decimation, playlist/title, keep-multichannel) |
| Design | none / current / stale / failed / **protected** (accepted or published) | extract change; designer, `AnalysisConfig`, coverage (existing `design_fingerprint`) |
| Review | pending / accepted / skipped / rejected | a redesign resets to pending, unless protected |
| Publish | not written / written / out of date | changed **published digest** (below) |
| Commit | uncommitted / committed / pushed | derived from git (§12.7) |

Orthogonal flags: **Ignored** (rule or per-title), **Shadowed**, **Gone**, **Possible duplicate**,
**Already in catalogue**.

**Needs and tier.** The first matching row, top down -- with one deliberate exception: `derive_needs` evaluates **Ignored, Shadowed and
Gone first** (a title that is not for this catalogue is `done` whatever its stages say), although the table lists them last:

| Tier | Needs | When |
|---|---|---|
| attention | *reason* | extract or design failed; `ProjectFilterConflict`; **source changed since accepted/published** (the fingerprint differs from the one recorded at accept) |
| human | review | design current, review pending (including a designer decline, which can only be skipped/rejected/redesigned); metadata incomplete |
| machine | extract | none, or stale |
| machine | design | extract current, design none or stale (not protected) |
| machine | publish | accepted and not written, or written and out of date |
| machine | commit | written but not committed, or committed but not pushed |
| done | -- | pushed; skipped, rejected; ignored; shadowed; gone |

**Published digest.** `publish_reviewed_queue()` records on the entry (additive fields) a hash of the publish
inputs -- the published filter (project filter hash, else chosen candidate), the metadata dict, the artwork
file's hash, the report spec and image owner/repo when they are set (not the default/unset, so older digests stay valid) -- plus
`published_at`. The designer is *not* in it (the filter it produced is). `ScanSettings` carries `image_owner`, `image_repo_name`
and `report_spec` (from `sync:`) and must be given what `publish` is given. "Out of date" means the current inputs hash differently. This
is what lets a metadata typo on a published title flow straight to Publish without a re-review.

**Done titles and staleness (deliberate).** A done title re-enters the list on its own **only** when its
*source* changed (attention). A **config** change (designer, analysis) must not: it would put every done title
in the list at once. Instead the UI shows one banner, "Settings changed since N titles were designed --
Revise...", which opens a bulk revise on that selection (§12.8).

### 12.7 Phase 3 -- doing work

**Stage entry points.** `run_library()` is split so each stage runs alone, over a **selection** of titles from
the index: `run_stages(profile, selection, through=...)`. `through` means "run every stage up to and including
this one that the title still needs", so "Design" extracts first when required and the user never picks
prerequisites. Review is human, so `through` stops at design for the machine tier, then at publish/commit only
for titles that are accepted/written.

**Cancel.** A cooperative cancel hook (`should_cancel()`), checked between titles; the run reports what
completed. Progress is determinate (the count is known from discovery) and names the title and stage.

**Publish (local write) and commit (push) become separate steps.** As built they are one: `commit_and_push()`
(`pipeline/publish/git.py`) writes, commits and pushes one file at a time, so a title costs two pushes (image,
then XML). Change:

- **Publish** renders the image and XML and **writes them into the repo working trees** (`<xml_dir>/<id>.xml`,
  `<image_dir>/<id>.png`), records the published digest, and sets `status='published'` (meaning *written locally*).
  The image raw URL needs only owner/repo/branch, not a completed push, so it is computed without pushing
  (split out of `push_image()`).
- **Commit** is one commit per repo, over exactly the published paths, then one push per repo. **Images repo first,
  then XML**, so a pushed XML never references a missing image. It commits only our paths (not other staged
  changes) and treats "nothing to commit" as success, not an error.
- **Committed/pushed is derived from git, not stored on the entry**: `git status --porcelain` for uncommitted and
  `git diff --name-only @{u}..HEAD` for unpushed, **one call per repo**, not per title. So a hand-made commit stays
  correct.
- The existing `sync` command remains as `publish` followed by `commit`.

**Reopening depends on the stage.** Written but uncommitted: revert the working-tree file(s) and reopen.
Committed or pushed: it is a **revision** -- the same catalogue path is rewritten and committed again (the path is
`<entry.id>`-derived, so it is stable; this is also why ownership is sticky, §12.4).

**One selection vocabulary, shared by the CLI and GUI** (and the documentation):

```
--needs {attention,extract,design,review,publish,commit,done}   a strip chip in the GUI
--source NAME    --match TEXT    --id ID    --new-since-scan
--through {extract,design,publish,commit}                       the action button in the GUI
```

CLI subcommands: `scan`, `status` (counts per needs), `run`, `publish`, `commit`, `sync` (= publish + commit), plus `revise` (§12.8) and `accept` (§12.9). As built (chunk 25): a `run` with any
selector works from the index; `--retry-failed` is the way to try a failed title again; `publish`/`sync --republish` writes published titles that are out of date. The Python entry points are `selection.Selection`,
`stages.run_stages()` and `bulk.accept_top_pick()`; see the chunk 25 notes in `implementation-order.md`.

### 12.8 Revising done titles

Any title can be sent back into the work list. **Revise...** (context menu, single or bulk on a selection in the
Done view) offers:

| Choice | Effect |
|---|---|
| Edit metadata or filter only | edit in place; the title becomes **Publish: out of date**; no re-review |
| Reopen for review | status back to pending; candidates, metadata, artwork and project kept |
| Redesign | clears the accepted/published protection and re-runs the designer (project edits are still preserved by the hash gate, §3.3.1) |
| Re-extract | e.g. another audio stream; invalidates everything downstream |

Each revise records a reviewer note. After commit/push, see §12.7 (a revision, same path). Automatic re-entry is
limited to source changes, as in §12.6.

### 12.9 Bulk accept

Allowed. Selector: `needs=review` and `candidates[0].confidence >= threshold` (a preference; default **0.90**).
Action: **Accept top pick for N titles**, behind a confirmation showing the count and an expandable list. Excluded and
reported: incomplete metadata (`validate()`), a decline reason, an edited project. Each entry gets a reviewer
note ("bulk accepted, confidence >= 0.90").

### 12.10 The screen

One window, **no tabs**. `LibrarySyncDialog` becomes a top-level window (`QMainWindow`), since it is now the
workspace rather than a dialog.

```
 Attention 4 | New 120 | Design 3 | Review 37 | Publish 12 | Commit 12 | Done 1,100 (hidden)
 [search...]  [source v]                                       last scan 09:14 · Rescan
 +-[ ] Title      Year  Source  Needs          Detail                          Waiting
 | [ ] Dune       2021  JRiver  ! Failed       file not found (path mapping?)  3d
 | [ ] Heat       1995  JRiver  ! Changed      source re-ripped since accepted 1d
 | [ ] Alien      1979  JRiver  * Review       conf 0.62 . 3 candidates        9d
 | [ ] Sicario    2015  FS      o Extract      new                             new
 +-
 [ Extract & design 120 ]   <- label follows the selection
```

- **The strip is summary and filter.** Default view: everything except Done. Done is one click away and never
  crowds the work.
- **Sort:** tier (attention, human, machine), then **oldest first** by `state_since` (user-changeable: title, confidence,
  source). New-since-scan rows are highlighted.
- **Selection.** Multi-select, select-all-in-filter (a chip then Select all), selection survives rescan. One
  action button, labelled with what it will do to how many titles; a mixed selection runs the eligible
  subset and states what was skipped ("3 of 125 skipped: already accepted"). The action is "run **through**
  stage X" (§12.7).
- **Running work is visible:** a row shows its stage while it runs; cancel stops after the current title.
- **Drill in** (double-click or Enter) opens the **in-window title page**, which replaces the table (no dialog nested in a
  window): Esc/breadcrumb returns to the table with selection, scroll and filter intact. It has **prev/next** over the
  current filtered and sorted list ("12 of 37 - oldest first"); **Accept and next** advances to the next title needing
  review, as `ReviewQueueDialog` does today.
- **Title page contents:** a header (poster, title, year, and a badge "Ready to publish" or the list of what is missing,
  from `validate()`); Candidates (list, commentary, chart); Metadata split into **Essentials** (title, year, audio types,
  edition, season/episodes, note, warning, TMDB id + Reload) and a collapsed **More**; Artwork; **Open project** (mono,
  multichannel) with a "modified since design" badge from `read_project_filter()`; Reopen/Revise. Accept is disabled or warns
  while metadata is invalid. Metadata saves on focus-out (or prompts on Accept), so an edit is never lost. Keyboard shortcuts
  apply to the table and candidate list, **not** to text fields (fixes the Enter problem of §12.1).
- **Settings** (work/queue dirs, repos, designer, TV mode, keep-multichannel, sources and priority, ignore rules) live behind
  a settings drawer with folder pickers and validation, persisted on change; an incomplete setup shows a banner.
- **One publish path.** The review dialog's own XML-only Publish button is retired when embedded; the standalone review
  entry point uses the same publish and commit code (closes T9).
- **JRiver page:** the browse path is primary; the numeric id sits under Advanced; with no servers configured the page is
  disabled with a link to Preferences > JRiver.
- **Wording:** "Run library" -> **Analyse**, "Explicit sync" -> **Publish**/**Commit**, "Review queue" -> **Review folder**.

### 12.11 Unattended runs

Cron covers `scan`, then `run --through design` on the machine tier. It never accepts, publishes or commits. `status`
prints the counts per needs, so a scheduled job can report what is waiting for a human.

### 12.12 Decisions (agreed 2026-09-19)

| # | Decision |
|---|---|
| 1 | Soft clashes (same title, different file) are **flagged, never dropped**; hard clashes (same file) go to the higher-priority source |
| 2 | Ownership is **sticky**, via claims reconstructible from queue entries and work dirs |
| 3 | Publish (local write) is **explicit and batched**, not automatic on Accept |
| 4 | "Already in the catalogue repo" is **informational**: a label, not removal |
| 5 | Commit is **one commit per repo per batch**, images repo first |
| 6 | Unattended runs cover discovery, extract and design only |
| 7 | **Ignore** is separate from Skip/Reject, **by rule as well as per title** |
| 8 | **Bulk accept** is allowed, with a confidence threshold and a confirmation |
| 9 | Default order within a tier is **oldest first** |
| 10 | **One window, no tabs**; drill-in is an in-window title page (recommended over a dialog) |
| 11 | Done titles can be **revised** (§12.8); only a *source* change re-enters one automatically |

### 12.13 Implementation order

Moved to [`implementation-order.md`](implementation-order.md) (chunks 19-28, dependencies, milestones, risks).

### 12.14 Verified facts (chunk 19 spike, 2026-09-19)

Each item was checked against the code at `HEAD` and, where it is behaviour, run. Tests that pin a finding a later chunk
must fix are `xfail(strict=True)`, so the chunk that fixes it cannot forget to remove the marker.

| Item | Finding | Consequence |
|---|---|---|
| Enter in a metadata field | **Confirmed.** `model/review.py` binds `Key_Return`/`Key_Enter` to Accept as window-context `QShortcut`s; a `QLineEdit` does not claim Return via `ShortcutOverride`, so Enter in `editionField` accepted `title-a`. Printable keys are safe: `A`/`S`/`R`/`1-9` typed into a field are text (`QLineEdit` claims them). Enter with focus on the queue table accepts, and must keep doing so. | Chunk 20 scopes only Return/Enter (and keeps the letter shortcuts, which need no change). Tests: `gui/test_review_dialog.py::test_enter_in_a_metadata_field_does_not_accept_the_entry` (strict xfail), `test_letter_and_digit_keys_typed_in_a_metadata_field_are_text_not_shortcuts` and `test_enter_on_the_queue_table_accepts_the_selected_entry` (both pass today). Key events reach a widget offscreen only after `activateWindow()` + `qtbot.waitActive()`. |
| TMDB id in the XML | `<beq_metadata><beq_theMovieDB>` (`BeqMetadata.to_dict()` key `beq_theMovieDB`), emitted by `to_beq_xml()`; empty (`<beq_theMovieDB />`) when unresolved. **The XML carries no movie/tv kind**, and TMDB numbers movies and series separately, so a film and a series can share an id. `<beq_season>` is non-empty for TV entries and is the only discriminator. | Chunk 24's "Already in catalogue" match is on `(theMovieDB, is-tv)` with `is-tv` = `beq_season` non-empty, never the bare id; an XML with no id can never match. |
| `git commit` with unchanged content | **Errors.** `commit_and_push()` -> `git commit` exits 1 ("nothing to commit, working tree clean", on *stdout*, so `CalledProcessError.stderr` is empty) and `subprocess.run(check=True)` raises. Changed content at the same path commits normally, so a revision (§12.7) works today. | Chunk 21's `commit_paths()` must treat "nothing to commit" as success. **Fixed in chunk 21**; `test_pipeline_publish_git.py::test_committing_unchanged_content_is_a_no_op_not_an_error` is now an ordinary test. |
| Foreign staged files | **Swept in.** `commit_and_push()` runs `git add <path>` then `git commit -m` with no pathspec, so anything already staged in the working tree (`other.txt` in the probe) lands in our commit. | Chunk 21 commits with an explicit pathspec (`git commit -- <paths>`). **Fixed in chunk 21**; `test_a_commit_contains_only_the_published_path` is now an ordinary test. |
| `QueueEntry` fields | `id, fs, meta, curve, candidates, decline_reason, decline_message, status, chosen_candidate_index, reviewer_note, art_path, art_overridden, design_fingerprint`. **No timestamp of any kind**; `status` is `pending`/`accepted`/`skipped`/`rejected`/`published`. Entries are `<queue_dir>/<id>.json`. | Confirms `state_since` cannot come from the entry and lives in the index (chunk 24). Chunks 21-22 add `published_digest`, `published_at`, `revision` (all optional). |
| Season title id | A season is one title with its own id, `season_item_id(title, season)` = `<slug>-sNN-<sha256(casefolded title|season)[:6]>` (`some-show-s01-60ad24`); members keep their own work dirs under their item ids, and only the season id gets a queue entry, project files and XML (`<xml_dir>/<season-id>.xml`). Unlike other ids it has **no source component**, and it depends on the **series title text**: `' some show '` / `'01'` give the same id, `'Some Show (2019)'` a different one. | Chunk 23's claim reconstruction must recognise season ids as a second id shape; two sources offering the same season already collapse to one id, which is wanted. A library title correction re-keys the season (orphaning its cache and entry) -- the sticky-claim rule (§12.4) must cover it, not just source-derived ids. |
| Catalogue filenames | `<xml_dir>/<entry.id>.xml` and `<image_dir>/<entry.id>.png` (`pipeline/review.py`), so `jriver-<hash>-<Key>.xml`, `fs-<hash>.xml` or the season id above. beqcatalogue globs `**/*.xml`, so opaque names are acceptable. A human-readable name would need its own stable-name rule and is out of scope. | -- |

### 12.15 Documentation outline (for chunk 28 / T1)

The user documentation follows the model, not the widgets: (1) **Concepts** -- titles, stages, needs and tiers, the review gate; (2)
**Set up a catalogue** -- profile, sources and priority, clashes, ignore rules, repos; (3) **Discover** -- what a scan reads and why it is
cheap, refresh, states; (4) **Do the work** -- selecting, `through`, bulk and single, cancel, failures and retry; (5) **Review** -- the title
page, projects, bulk accept; (6) **Publish and commit** -- what each writes and where, revisions; (7) **Revise a done title**; (8)
**Unattended (CLI/cron)** -- the shared selector vocabulary. `docs/ui/manage_mc.md` and `preferences.md` are updated in the same pass (T1).
