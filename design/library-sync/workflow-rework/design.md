# Library sync plan -- workflow rework: design

> Part of the library sync plan -- **start at the index**: [`../../library-sync-pipeline-plan.md`](../../library-sync-pipeline-plan.md).
> Contains §12 (§12.1-§12.12, §12.14-§12.15). Section numbers are global across the plan; the index maps every `§` to its file.
> Status of what this file describes: **design agreed 2026-09-19, not built**; supersedes §7

## 12. Workflow, discovery and a work-list UI (designed 2026-09-19, **not built**)

Status: **design only. Nothing in this section exists yet**; §12.13 is the build order.
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
  (`Date Modified` + `File Size`) is the library's. The one disk touch today is `_local_art_path`
  (`os.path.isfile` per item); it moves to design time, where artwork is actually resolved.
- **Filesystem source:** one `stat` per file (`BDMV/index.bdmv` / `VIDEO_TS.IFO` for discs). No file contents are read.
- Whether a JRiver file exists locally is **not** checked at discovery. A wrong path mapping surfaces as an
  extract failure with its reason (below) rather than costing a stat per title on every scan.

**"What would run" and "what runs" must not diverge.** The states are computed by pure functions split out of the
existing `*_if_needed()` wrappers -- `extract_status()` (manifest vs `source_fingerprint` + `params_hash`,
`extract_cache.py`) and `design_status()` (`design_fingerprint()` vs `QueueEntry.design_fingerprint`,
`design_cache.py`) -- which the wrappers then also call.

**Index (proposed: SQLite in `<work_dir>/`, stdlib `sqlite3`).** One row per title: catalogue id, owning source
and item id, path, source fingerprint, metadata snapshot (title/year/kind/external ids), `last_seen`,
shadowed duplicates, per-stage state, `needs`/`tier`, **`state_since`** (when the title entered its current
state -- the queue JSON's mtime is not usable, since every edit changes it), and failure memory. The index is
a **cache**: deleting it costs a rescan, nothing else (claims are reconstructed, ignores live in the profile).

**Refresh.** The GUI shows the cached index immediately (stale-while-revalidate) and rescans **per source, on
demand**, showing each source's "last scanned". A rescan highlights what is **new since last scan**. Cron
scans first.

**Failure memory.** An extract/design failure is recorded with the fingerprint and params it failed against.
It is retried only on an explicit **Retry failed**, or when either changes. (Today `LibraryRunReport.failed`
is not persisted.)

**Units.** `plan_units()` (tv_mode season grouping, `pipeline/library/season.py`) moves out of `run_library()`
into discovery, so the units the user sees are the units that get worked on.

**Gone.** An item that has left its source: if it has no outputs it is dropped; otherwise it is kept as **Done**
with a "gone from source" badge. Outputs are never deleted automatically.

**Repo awareness.** Discovery reads the local XML repo. A title whose TMDB id already appears there but which
this profile did not publish is labelled **Already in catalogue** -- informational only; it stays in the list
(and can be ignored). Matching must use the XML's TMDB id, not the filename (ours are entry ids; others' are not).
The element is `<beq_metadata><beq_theMovieDB>`, and the XML has no movie/tv kind, so match on it together with
`<beq_season>` being non-empty (§12.14).

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

**Needs and tier.** The first matching row, top down:

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
file's hash, the designer -- plus `published_at`. "Out of date" means the current inputs hash differently. This
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

CLI subcommands: `scan`, `status` (counts per needs), `run`, `publish`, `commit`, `sync` (= publish + commit).

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
