# Library sync plan -- workflow rework: implementation order

> Part of the library sync plan -- **start at the index**: [`../../library-sync-pipeline-plan.md`](../../library-sync-pipeline-plan.md).
> Contains §12.13. Section numbers are global across the plan; the index maps every `§` to its file.
> Status of what this file describes: chunks 19-25 built (M0-M2), 26-28 not started; the status of each chunk lives in the index

### 12.13 Implementation order (proposed)

Chunk numbers continue from §8 (chunks 1-18 are built). Each chunk is one commit, or a small handful, and leaves the
app and the test suite green.

**Principles**

1. **Value first.** Cheap fixes to what exists ship before any redesign.
2. **Headless before GUI.** Every backend capability lands with its CLI subcommand first, so it can be exercised end to end
   without Qt; the GUI is then a thin consumer.
3. **The current dialog keeps working until it is replaced.** `LibrarySyncDialog` stays reachable, and works, through every
   backend chunk. The new window is added beside it (chunk 26a), takes over the menu entry at 26b, and the old dialog is deleted at 27c.
4. **Additive data only.** New `QueueEntry` fields are optional with defaults; the index and profile are new files; the old CLI
   config file keeps loading. An existing queue directory or work directory is never migrated destructively.
5. **Freeze the index schema at chunk 24.** UI chunks build on it; changing it afterwards costs a rework of every consumer.

**Dependency graph**

Each line is "chunk <- what it needs". Nothing else is a prerequisite.

```
19 spike            <- nothing
20 quick fixes      <- 19   (only for the strict-xfail test)      ships alone
21 publish/commit   <- 19   (only for the no-diff commit finding)
22 revise backend   <- 21
23 profile + union  <- nothing
24 discovery index  <- 21, 23, 19   (19: the XML TMDB-id element)   schema FROZEN here
25 stages/selectors <- 24
26a work list (r/o) <- 24                                            (not 25)
26b actions         <- 26a, 25
26c settings        <- 26a                                           (independent of 26b and 27)
27a title page      <- 26a
27b metadata        <- 27a
27c project/revise  <- 27b, 22, 26b                                  retires the old dialog
28 docs             <- 27c
```

After 19, three tracks can run in parallel: **20**, **21-22** and **23**. 24 needs both 21 (publish/commit state) and 23 (the union).
26a can start against a fixture index as soon as 24's schema is frozen; it need not wait for 25, but 26b does.

**Suggested sequence for one developer** (a linear order that respects every dependency):

| Step | Chunk | Why here |
|---|---|---|
| 1 | 19 | tiny; unblocks the rest |
| 2 | 20 | ships value at once (M0) |
| 3 | 21 | highest-risk backend change (git); do it while nothing else is in flight |
| 4 | 22 | small, and completes M1 |
| 5 | 23 | independent of 21-22, so its position is free; it precedes 24 |
| 6 | 24 | the keystone: freeze the schema |
| 7 | 25 | completes M2; the CLI is now the whole workflow |
| 8 | 26a | first look at the new window |
| 9 | 26b | M3 |
| 10 | 26c | replaces the bootstrapped profile (see 26a) with an editor; before 27 so no one hand-writes YAML for the title page work |
| 11 | 27a, 27b, 27c | M4 |
| 12 | 28 | M5 |

Steps 2-5 can be reordered freely (they are independent); steps 6 onwards cannot.

**Milestones** (each is a point where something is usable and worth releasing)

| | After | What you can do |
|---|---|---|
| M0 | 20 | The existing dialog is materially better: preloaded review, readable titles, failures visible, no accidental accept |
| M1 | 22 | Two-stage publish/commit and revision, from the CLI; the existing dialog's Sync still works |
| M2 | 25 | The whole workflow headless: multi-source profile, `scan`, `status`, `run --needs ... --through ...`, bulk accept |
| M3 | 26b | The GUI shows what needs doing and does the machine work, publish and commit from a selection |
| M4 | 27c | Complete: review in the title page, projects, revise; the old dialog is gone |
| M5 | 28 | Documented |

#### The chunks

**19 -- Verification spike** (no product code; write-up + tests). Settles the §12.14 items that later chunks depend on.
- A `pytest-qt` test showing Enter in a metadata `QLineEdit` triggers Accept (`xfail(strict=True)` until chunk 20 fixes it).
- Read the XML writer/sample XML for the element carrying the TMDB id (chunk 24's repo awareness) and record its name (`beq_theMovieDB`; no kind element).
- Run `commit_and_push` with unchanged content against a temp repo; record whether `git commit` errors (chunk 21).
- List every `QueueEntry` field (timestamps?), and check how a season title's id flows through publish naming.
- **Done when:** §12.14 is rewritten as verified facts, and the strict-xfail test is committed. **Done 2026-09-19** -- the spike also found that `commit_and_push` sweeps in
  foreign staged files (a second strict xfail, `test_pipeline_publish_git.py`), and that the season id has no source component (§12.14); chunks 21, 23 and 24 carry both.

**20 -- Fixes that need no redesign** (`model/library_sync.py`, `model/review.py`, `pipeline/library/run.py`, `ui/review.ui`)
1. The Review tab is created on open and loads `DESIGNER_QUEUE_DIR`.
2. `_meta_source()` seeds `meta` with `title`/`year` from the item (`item.title or item.display_name`) on every path, including no-TMDB and
   TMDB-failure, so `QueueEntry.meta` always has a title; existing entries with none keep showing the id.
3. Return/Enter scoped to the table/candidate list, not text fields (turns 19's xfail into a pass; delete the marker). Letter/digit shortcuts already yield to text fields (§12.14).
4. A details view for `report.failed` and `report.meta_unresolved` (id, message), replacing the bare counts.
5. Accept with unsaved metadata edits prompts to save (or saves them).
6. **Reopen** on an *accepted* (not yet published) entry: status back to pending. This is only the trivial case, a status change in the dialog;
   chunk 22's `reopen_entry()` generalises it (written, committed and pushed entries, working-tree revert, revision counter) and 27c moves the dialog onto it.
- **Tests:** `gui/test_review_dialog.py` and `gui/test_library_sync_dialog.py` (both exist; extend them): open with an existing queue; a run with no TMDB key names
  its rows; Enter in a field does not accept.
- **Done when:** M0. Ships alone; nothing depends on it.
- **Done (2026-09-19); how it differs from the list above:**
  - (1) `LibrarySyncDialog` builds the `ReviewQueueDialog` in `__init__`, loads the queue directory if it exists, and reloads on `queueDirEdit.editingFinished`.
  - (2) `library_meta(item)` in `pipeline/library/library_metadata.py`, used by `run.py`'s no-key and TMDB-failure paths. The TMDB-success path already had a title.
  - (3) Only Return/Enter needed scoping: they are now `WidgetShortcut`s on the queue table and candidate list. Letter and digit shortcuts already yield to text fields (§12.14).
    Enter now does nothing on an already accepted or published entry, and Accept is disabled there (before, it would have set a published entry back to accepted).
  - (4) A **Details...** button (`detailsButton`, shown after a run that failed or had unresolved titles) opens a `QMessageBox` with the per-item reasons. **A run with
    problems now stays on the Run tab**: it used to jump to Review, which hid the status line, so the failure count was never seen.
  - (5) Accept with unsaved edits asks Save / Discard / Cancel; a failed save (unparseable episodes) does not go on to accept.
  - (6) `reopenButton`, enabled for `accepted` only; status back to `pending`, `chosen_candidate_index` cleared, same row kept.
  - Found and fixed on the way: saving metadata, artwork changes and Refresh jumped the selection to row 0 (`__reload_queue(keep_current=True)`), and the "Saved" label was
    cleared by the reload it preceded.
  - **Still open, for 27b:** editing metadata and then selecting another row, or pressing Skip/Reject, still discards the edits without asking (only Accept prompts).
  - Tests: the two dialog test files and `test_pipeline_library_run.py` (`library_meta`, and the no-TMDB-key run names its entry).

**21 -- Publish/commit split** (`pipeline/publish/git.py`, `pipeline/orchestrate.py`, `pipeline/review.py`, new `pipeline/library/commit.py`)
- `git.py`: `write_files()` (working tree only), `commit_paths()` (an explicit pathspec, so foreign staged files stay out; "nothing to commit" is success -- both are 19's strict xfails, remove the markers),
  `push()`, `image_url()` (owner/repo/branch without pushing), `repo_state()` (one `git status --porcelain` and one
  `git diff --name-only @{u}..HEAD` per repo -> uncommitted/unpushed path sets).
- `Session.publish()` gains a write-only mode; `publish_reviewed_queue(..., push=True)` keeps today's behaviour by default so the standalone review
  dialog is untouched until 27c. The library path (`sync_library`, CLI, `LibrarySyncDialog`) moves to the two-step.
- `QueueEntry` gains `published_digest` and `published_at` (schema doc updated); the digest inputs are as §12.6.
- `commit_catalogue()`: images repo first, then XML; one commit and push per repo; the message summarises the titles.
- CLI: `publish`, `commit`; `sync` = both. Every option needs `--help` text and a README entry (`test_pipeline_library_cli_docs.py` enforces it).
- **Tests** (temp bare remotes): image-then-XML order; a foreign staged file is not swept into our commit; no-diff commit is a no-op; `repo_state`
  classification; the digest changes on a metadata/artwork/filter edit and is otherwise stable; the resulting repo content equals today's `sync` output.
- **Risk:** git edge cases (detached HEAD, no upstream). `repo_state()` must degrade to "unknown", not raise.
- **Done (2026-09-19); how it differs from the list above:**
  - `git.py` also gained `write_files()`, `push()` and a `current_branch()` that works before the first commit (`symbolic-ref`, not `rev-parse`, which fails on an unborn branch).
    `commit_and_push()` is now those three, so `push_image`/`push_xml` and the standalone review dialog are unchanged. Both of chunk 19's strict xfails now pass, markers removed.
  - `commit_paths()` returns `None` for "nothing to commit". `repo_state()` returns `RepoState(uncommitted, unpushed)`, **each `None` when unknown** (not a repo; no upstream; detached HEAD),
    and `commit_catalogue()` treats unknown as "cannot tell": every present file is a candidate (recommitting an unchanged one is a no-op) and the repo is pushed.
  - New `pipeline/publish/catalogue.py` (`catalogue_paths()`, `publish_digest()`), shared by publish and commit. The digest covers the published filter, the metadata as published (before
    the image URLs are filled in), the artwork file's **content**, whether an image is made, and the mv offset. It does **not** include the designer name (the entry does not store it, and
    the filter it produced is in there) or the repo paths.
  - `QueueEntry.published_digest`/`published_at` (schema doc updated). They are recorded on **every** publish, including the default push path, but nothing reads them yet.
  - `pipeline/library/commit.py` `commit_catalogue()` and, in `sync.py`, `publish_library()`/`commit_library()`; `sync_library()` is both and still returns the publish results (now also `xml_commit`/`image_commit`).
  - CLI: `publish`, `commit` (with `--push/--no-push`), `sync` (which also takes `--push`). All three read the one `sync:` config section, and the docs tests now cover all four commands.
  - **Not done here, for 24/25 (both done: 24 factored the digest out, 25 republishes):** republishing a title that is already `published` and out of date.
  - **Found, not fixed here; fixed in 25:** a queue entry with invalid metadata (no title, say) made `publish_reviewed_queue()` *raise* `ValueError` mid-batch, aborting the rest, unlike `project_conflict`, which
    is returned as a per-entry `error`. It is now returned the same way (`{'id', 'error': 'invalid_metadata', 'problems': [...]}`).
  - Tests: `test_pipeline_library_commit.py` (temp bare remotes: one commit and one push per repo, images first, foreign staged and untracked files untouched, a failed push retried, a failed images push leaves the
    XML repo alone, a revision recommits the same path, the remote content equals the old per-file publish), plus the git and CLI test files. Writing them found a real bug: `commit` with nothing published
    and no upstream tried to push a repo with no commits.

**22 -- Revise backend** (`pipeline/review.py`, `pipeline/library/design_cache.py`, `extract_cache.py`)
- `reopen_entry()`: written-but-uncommitted -> delete or `git checkout` the working-tree files and set pending; committed/pushed -> pending with a
  `revision` counter (the same path is rewritten and recommitted).
- `redesign_entry()` clears the accepted/published protection deliberately (the hash gate still preserves project edits);
  `invalidate_extract()` drops a manifest's keys. A reviewer note records the reason. `QueueEntry.revision` (additive).
- CLI: `revise --id ID --to {review,design,extract}`.
- **Tests:** each path and each starting state, including that project edits survive a redesign and that a reopened, pushed entry publishes to the same path.
- **Done when:** M1.
- **Done (2026-09-19); how it differs from the list above:**
  - The functions live in a new **`pipeline/library/revise.py`** (`reopen_entry`, `redesign_entry`, `revise_entry`, `ReviseResult`), not in `pipeline/review.py`, which is already ~415 lines. `invalidate_extract()` is in
    `extract_cache.py` as planned; `invalidate_season_track()` is in `season.py`. `git.py` gained `is_committed()` and `discard_changes()`.
  - **State only.** None of these redesign, re-extract or republish; they leave the title needing that work, and the next `run` / `publish` + `commit` does it. `redesign_entry()` clears the design fingerprint
    (so the pending entry is stale) and `design_if_needed()` now carries `revision` across the redesign as well as the reviewer note.
  - **Revision rule.** `revision` increments only when a *published* entry whose XML is committed and **clean** is reopened. Reopening a revision that is written but not committed restores the committed version and
    does not count again; a never-committed entry has its files deleted and does not count. `published_digest`/`published_at` are cleared on every reopen (the digest of what the catalogue holds is not known
    once a revision was written over it).
  - A *published* entry needs `xml_repo` (and `images_repo` for its image) to be reopened, else `ValueError` before anything changes. Reopening a pending entry is a `ValueError`; skipped and rejected can be
    reopened at the backend (the dialog's Reopen button, chunk 20, still offers only accepted -- 27c moves it onto this).
  - `--to extract` forgets `mono_*`/`multichannel_*` manifest keys, not the wav files, and keeps `source_channel_count`/`channel_layout_name`. **A TV season's member episodes are not forgotten** (their directories are
    named by ids the season entry does not record); the joined season track is. A re-extract of a season therefore rebuilds the join from the members' existing audio unless a member's source also changed.
  - CLI: `revise --queue-dir --id ID [--id ...] --to {review,design,extract} [--reason] [--work-dir] [repos]`, in the shared `sync:` config section; one bad id is reported and does not stop the others.
  - Tests: `test_pipeline_library_revise.py` (real temp git repos; each rule mutation-checked) plus git and CLI tests.

**23 -- Profile, union, ignore rules** (new `pipeline/library/profile.py`, `union.py`, `ignore.py`)
- The profile schema over the existing config shape: ordered `sources`, `ignore`, repos, work/queue dirs. The old CLI config still loads unchanged.
- `union_items(profile) -> UnionResult(titles, shadowed, duplicates)`: hard-clash normalisation (path mapping, case-fold, separator, disc root),
  soft-clash detection, priority, **sticky claims** reconstructed from queue entries and work dirs (existing ids `jriver-<hash>-<Key>` / `fs-<hash>` are
  simply claims; nothing is renamed).
- `ignore.py`: the rule model, `evaluate()` and `explain()` (which rule matched); rules on source/path/title/year/kind/external ids; per-title flag.
- CLI: `run --profile PATH`.
- **Tests:** JRiver-mapped path == filesystem path; BDMV root vs its clip; case/separator variants; soft clash by tmdb, imdb and title+year; reordering priority keeps
  ids; an owner that vanishes hands over; each rule field; deleting a rule un-ignores; profile round-trip; the old config loads.
- **Done (2026-09-19); how it differs from the list above:**
  - Modules: `profile.py` (`Profile`, `SourceSpec`, `build_source`, `load_profile`/`save_profile`, `profile_from_config`, `Profile.to_config()` which keeps everything it does not manage), `ignore.py`
    (`IgnoreRule`, `evaluate`, `explain`), `union.py` (`union_of`, `union_items`, `UnionLibrarySource`, `Claims`, `reconstruct_claims`, `clash_key`). `union_of(listings, ignore=, ignored_titles=, claims=)` is **pure**
    over already-listed items, so chunk 24's scan can list each source itself (per-source errors, last-scanned) and then merge; `union_items(profile)` lists and merges and lets a listing error propagate.
  - The per-title ignore is the profile key `ignore_titles` (id -> reason, or a list of ids / `{id, reason}`). Ignore rules are evaluated *after* clash resolution, so a `source:` rule that ignores the owner
    ignores the file. Ignored titles are not reported as duplicates.
  - The profile is the CLI config file's shape: `sources:` is now a list, or (older) a mapping with `run.source` naming the one in use (a filesystem source may keep its globs in `run:`). `work_dir`/`queue_dir` come
    from `run:` then `sync:`; the four repo settings from `sync:`. `cli._source` and `_load_config` now delegate to `profile.py` (`build_source`, `read_config_file`).
  - **Sticky ownership as built:** the item whose id has a queue entry or work directory owns a clashing file whatever the order; unclaimed clashes go to the earlier source. **When the owner's item disappears the
    other item takes over under its own id** -- a dangling claim records no path, so it cannot be adopted; its old outputs are left behind. Chunk 24's index, which keeps a path per title, is where to improve that.
  - **Season ids** (§12.14): `Claims.season_id()` matches an existing season-shaped queue entry by (series TMDB id, season) then (title, season); `plan_units(..., season_id_for=)` uses it and **`run_library()` now always
    passes it**, so a corrected series title no longer re-keys a season. `season.is_season_id()` distinguishes season ids from source ids (an episode's own entry also carries a season in its metadata).
  - CLI: `run --profile FILE` (replaces `--config` and the source options; refused with `--config` or with no sources). README has a "One catalogue from several libraries" section.
  - **Not done:** a per-source timeout policy (a source that raises is handled in 24; a slow one is not), GUI editing of the profile (26c), and adopting a dangling claim (above).

**24 -- Discovery index and states** (new `pipeline/library/status.py`, `index.py`, `state.py`)
- Split `extract_status()` / `design_status()` out of the `*_if_needed()` wrappers (which then call them): a refactor with **no behaviour change**,
  proven by the existing suite plus equivalence tests.
- `state.derive_needs()`: the §12.6 table as a pure function. `index.py`: SQLite schema, `scan(profile)`, `rebuild_from_outputs()`, `state_since`,
  failure memory (`run_library` records failures against the fingerprint/params they failed on), repo awareness (XML scan by TMDB id, using chunk 19's finding).
- CLI: `scan`, `status [--json]`.
- **Tests:** one case per row of the needs table and per flag; index deletion -> rebuild yields the same states (`state_since` resets to the rebuild time --
  accepted and documented); a 5,000-item synthetic scan stays within a budget; a JRiver listing performs **no** `os.stat`/`isfile` on media (monkeypatched to fail).
- **Done when:** the schema is **frozen** and written into §12.5.
- **Done (2026-09-19); how it differs from the list above:**
  - Modules: `state.py` (pure: `StageStates`, `derive_needs()`, the vocabularies), `status.py` (the read half: `ScanSettings`, `Evaluator`, `EntryFacts`, `failure_key()`, `FailureMemory`), `index.py` (`LibraryIndex`,
    `SCHEMA`, `TitleRow`, `ScanResult`, `IndexSummary`) and a new `catalogue_scan.py` (the XML repo scan). **`extract_status()` and `design_status()` live in `extract_cache.py` / `design_cache.py`** beside the wrappers
    that now call them, not in `status.py`; each returns a small dataclass (`ExtractStatus`, `DesignStatus`) and both wrappers keep their behaviour (existing suite plus equivalence tests that walk every state
    through both). `design_fingerprint()`/`extract_status()` take the source fingerprint as an optional argument, so a scan that already has it does not `stat` twice; an **empty fingerprint means "unknown" and is not compared**
    (a filesystem item that cannot be stat'd keeps a recorded extraction as current rather than stale for ever).
  - **The schema is frozen in design.md §12.5** (`SCHEMA_VERSION = 1`, `PRAGMA user_version`; any other version, or a file that is not a database, is dropped and rescanned). A test keeps the doc's DDL identical to
    `index.SCHEMA`. It is complete for §12.10: title, year, source (`also_in`), needs/tier, `detail`, `confidence`, `candidate_count`, `state_since`, `first_seen_generation` (the new-since-scan marker),
    the five flag columns, the five stage states, `sources.last_scanned`/`last_ok`/`last_error`. Failure memory is its own table (`failures`), so a failure recorded by `run` survives a rescan.
  - **`QueueEntry.source_fingerprint`** (new, optional; schema doc updated) is the "fingerprint recorded at accept time": `design_if_needed()` records it when it designs, and a protected (accepted or published) entry is never
    redesigned, so it is the fingerprint the human accepted. An entry designed before this chunk has none, so a change to its source **cannot be detected** (no false alarms; the next redesign records it).
  - **The digest** is factored into `review.current_publish_digest(entry, meta_defaults=, work_dir=, has_image=)` (plus `publication_meta()` and `project_paths()`, now shared with `publish_reviewed_queue()`) and
    `project.preview_published_projects()` (what publishing would resolve, without writing: a missing or pipeline-pure project counts as holding the chosen candidate). Equivalence with what publish records is tested with
    real publishes, with and without projects and edits. The digest depends on `meta_defaults`, on whether an image repo is given and on `work_dir`, so **`ScanSettings` must be given what `publish` is given**; the CLI `scan` reads
    them from the config's `run:`/`sync:`. An entry published before digests were recorded has none and is treated as `written`, not out of date. A published title whose XML is missing from the repo is `out_of_date` ("its file is
    missing"), which is also how a later chunk republishes it.
  - **JRiver listing does no `stat`/`isfile`**: `_local_art_path` became `_art_candidates()`; a listing sets `LibraryItem.art_candidates` (new, default empty; absolute candidates only) and `artwork.resolve_art()` picks the first that
    exists at design time. The jriver tests that asserted `item.art_path` now assert `resolve_art(item, {}, None)`; new tests monkeypatch `os.stat`/`isfile`/`isdir`/`exists` to fail during a listing (one over real HTTP).
  - **Failure memory:** `run_library(..., index=)` records a failure (stage `extract` or `design`, message, the unit's source fingerprint, `failure_key()` of the settings) and forgets it on success; the CLI `run` opens the work
    directory's index for it (and carries on without one if it cannot be opened). A failure applies while both fingerprint and key are unchanged; a scan deletes one that no longer applies. **Chunk 25 reads it**:
    a title whose failure still applies is not tried again unless asked (`run_library(index=, retry_failed=)`, `run_stages(retry_failed=)`, `--retry-failed`). A season's failure is recorded against the season id;
    an episode that fails inside a season is now remembered against its own id too (25).
  - **`scan(profile, settings=None, only=, sources=, now=)`** lists each source itself and calls `union.union_of` (a source that raises keeps its last listing, from the `items` column, and is reported; `only` rescans named sources
    and merges against the others' last listings), then `plan_units()` (so the units the index shows are the units a `tv_mode='season'` run works on -- `run_library` still calls `plan_units` itself; chunk 25 makes it use the index's
    units), then reads outputs. A **season** is one row; its extract state aggregates its episodes (an episode never extracted is tolerated once some were, since a run leaves out one that will not extract); its design status compares
    the joined track's fingerprint. Shadowed and ignored items are rows (`done`). A `gone` row needs outputs (a queue entry or work directory) or it is dropped and reported. `reconstruct_claims()` now opens only season-shaped queue
    entries (an entry's file name is its id), which a 5,000-title scan needed. **5,000 items scan in about 0.25 s** (and a rescan the same) on the dev machine; the test's budget is 6 s.
  - **`rebuild_from_outputs(settings)`** recreates rows for every queue entry from the outputs alone, assuming sources unchanged; `state_since` restarts at the rebuild time, `source` is empty until the next scan, nothing is marked
    new, and a **title with no queue entry is not rebuilt** (extracted but undesigned, or new) -- it reappears at the next scan. Tested: scan, delete the file, rebuild -> identical states; a scan after a rebuild agrees and attaches sources.
  - **Repo awareness** parses `<beq_theMovieDB>` and `<beq_season>` from every `*.xml` in the local XML repo outside `.git` (cached by relative path, mtime and size in `repo_xml`); "Already in catalogue" is set when a
    matching `(tmdb, is-tv)` exists under a file stem that is not one of this profile's ids (queue entries, work dirs, titles). A title with no TMDB id (item or entry) can never match.
  - **Commit state** is from `repo_state()`, one call per repo, for the XML and the image path (the worse of the two). A repo with **no upstream** gives `unknown`, which is `needs = commit` ("cannot tell whether it is pushed"),
    exactly as `commit` itself behaves: such a title never reaches Done. A clone with an upstream (every real one) is fine; chunk 25 added the fallback to `<remote>/<branch>` in `repo_state()`, so a repo `push()` has pushed to is no longer unknown.
  - **CLI:** `scan` (`--profile --source NAME --from-outputs --work-dir --queue-dir --designer --coverage --keep-multichannel --tv-mode`, the repos, the analysis options; reads `run:` over `sync:`) and `status`
    (`--profile --work-dir --json`); README and help tests cover both. `scan --source NAME` names a *profile source* (the `--source` of chunk 25's selectors), unlike `run --source jriver|filesystem`, which names a kind.
    Exit 1 from `scan` if a source could not be listed, from `status` if there is no scanned index.
  - **Not done, for later chunks:** `Selection`/`--needs`/`--new-since-scan` (**done in 25**), an "edited project" fact for bulk accept's exclusions (**done in 25**, `project.edited_projects()`), adopting a dangling claim by path
    (the index has the path, but nothing uses it yet), a per-source timeout policy (a failure is recorded, a slow source still blocks the scan), and republishing a `published` entry (**done in 25**).
  - **For chunks 25-27:** open with `LibraryIndex(index_path(work_dir))` (thread-safe: one connection, an `RLock`); a scan's `ScanSettings` must match `run`/`publish` (see above); `titles()` is already in work-list order;
    `TitleRow.flags`, `is_new` and `summary()` are what the strip needs; `record_failure`/`clear_failure`/`failures()` are the failure API; `state.py`'s `NEEDS`/`TIER_OF_NEEDS` are the `--needs` vocabulary.
  - Tests: `test_pipeline_library_state.py` (one case per row of the table and per flag, pure), `test_pipeline_library_index.py` (each row and flag from real outputs and real temp git repos; source down, `only`, `state_since`, new marker,
    ordering, queries, rebuild equivalence, schema versioning, failure memory through `run_library`, seasons, the 5,000-item budget), `test_pipeline_review_digest.py`, `test_pipeline_library_catalogue_scan.py`, the extract/design
    equivalence tests, the jriver no-stat tests, and CLI tests. The new tests were mutation-checked (state_since, source-down, gone, failure clearing, digest cache, TV/film id, own ids, source-changed, fingerprint compare, rebuild, new marker).

**25 -- Stage entry points and selectors** (`pipeline/library/run.py`, new `selection.py`)
- `Selection` (needs, source, match, ids, new-since-scan) and `run_stages(profile, selection, through, should_cancel, on_progress)`; `Progress(done, total, title, stage)`;
  `plan_units()` moves into discovery (the index already calls it; `run_library` still does too); failures are read back through the index (chunk 24 records them; the artwork move to design time is done).
- `accept_top_pick(selection, threshold)` with an exclusion report (incomplete metadata, decline reason, edited project) and the reviewer note.
- CLI: `run --needs ... --source ... --match ... --through ...`; a `run` with no selector behaves as today.
- **Tests:** `through` semantics (design extracts first; a protected entry is never redesigned); cancel mid-run leaves consistent state; retry-failed only on an explicit request
  or a changed fingerprint; a parity table proving the CLI flags and the GUI strip map to the same `Selection`; bulk-accept exclusions.
- **Done when:** M2.
- **Done (2026-09-19); how it differs from the list above. This completes M2: the whole workflow (scan, run through design, bulk accept, run through publish and commit, status) runs headless, and
  `test_pipeline_library_cli.py::test_the_whole_workflow_runs_headless_scan_design_accept_publish_commit` does exactly that against real temp git repos.** The index schema is **unchanged** (still frozen, `SCHEMA_VERSION = 1`); no `QueueEntry` field was added.
  - **Modules.** `selection.py` (`Selection`, `selection_from_chip()`, `CHIPS`, `THROUGH`, `plan_stages()` -> `StagePlan`), `stages.py` (`run_stages()`, `Progress`, `PublishSettings`, `StagesReport`) and `bulk.py`
    (`plan_accept()`, `accept_top_pick()`, `AcceptPlan`/`AcceptReport`/`Exclusion`), all Qt-free. `run.py` gained `run_unit()` (one title's extract and design with its own failure boundary), which both `run_library()` and `run_stages()` call.
  - **`run_stages` works from the index, not from a listing.** The rows say what each title needs, and `LibraryIndex.units(ids)` rebuilds the `LibraryItem`/`SeasonGroup` a title stands for from the `items` column, so nothing re-lists a source and
    the units worked on are exactly the units the scan showed (the "plan_units moves into discovery" item: the run loop for a selection no longer calls `plan_units`; the legacy no-selector `run` still lists and does). A row rebuilt from outputs alone has
    no items and is skipped ("not in the last scan: scan again"). Afterwards `LibraryIndex.refresh(profile, settings)` (new; a scan of no source: no listing, **the generation, so the new-since-scan marker, and "last scanned" do not move**) re-reads every title's
    outputs, in a `finally`, so `needs` is current after a cancel or a failure too. **`refresh` must be given the profile the last scan used**: a profile with fewer sources would drop those sources' titles.
  - **Selection semantics.** Every field given is ANDed (`LibraryIndex.titles()` already did); `ids` empty is no constraint. `--needs` is repeatable. `selection_from_chip(chip, source=, match=, ids=)` is what a strip chip selects: **New** is `new_since_scan`, every
    other chip is `needs=(chip.lower(),)`; the combo, the search box and the row selection are `source`, `match` and `ids` on top. `CHIPS` has a chip per `needs` value (the design's sample strip has no Extract chip; 26a may merge Extract into New).
  - **What `through` does** (`plan_stages()`; the action button's label is `StagePlan.label`, e.g. "Extract & design 120 (3 of 123 skipped)"): needs *extract* -> extract, then design if `through` reaches it; needs *design* -> design; needs *publish* (accepted and not written, or
    written and out of date) -> publish, and commit too for `--through commit`; needs *commit* -> commit; needs *attention* because extract/design failed -> nothing unless `retry_failed`; needs *review*, a project conflict, a changed source and *done* are skipped
    with a reason. A title is **never taken past design without a person**: only an *accepted* one needs publish. A protected entry is never redesigned (it never needs design).
  - **Publish and commit inside `run_stages`.** One `publish_library(ids=, republish=True)` call for the whole selection (per-entry `on_entry`/`should_cancel` hooks added to `publish_reviewed_queue`), then one `commit_library(ids=)` for the selection (**one commit and one push per repo**
    for the whole selection; `commit_catalogue(ids=)` is new; the push sends the whole branch as before). A cancel before publish, between entries, or before commit stops there and **never commits**; a git failure while committing is reported (`commit_error`), what was committed stays.
  - **Progress.** `Progress(done, total, title, stage, id)`; `total` counts title-stages up front (an extract-and-design counts once, a publish once, a commit once -- so a title published and committed in one run counts twice), `stage` is `extract`, `design`, `publish` or `commit`
    (for commit, `title` is "N titles"), and one final event has `stage == ''` and `done == total` unless cancelled. It is called on the running thread.
  - **Cancel** is checked before each title and before each published entry; the title in hand finishes. `StagesReport` has `cancelled`, `attempted` and `not_run` and is what a GUI shows ("stopped after 40 of 120").
  - **Retry failed (closes the chunk 24 carry-over).** `status.failure_applies()` is the one rule (source fingerprint and `failure_key()` unchanged) used by discovery and by the run, so "shown as failed" and "not retried" cannot disagree. `run_library(index=)` and `run_unit()` skip such a title
    (`LibraryRunReport.failed_earlier`, new, `[id, message]`) unless `retry_failed`; a changed fingerprint or setting retries it without asking. An episode that fails inside a season is remembered against its own id and skipped the same way (a scan drops one that no longer applies); it
    does **not** change the season row's state, so it is visible in `index.failures()` and the report only. **The old dialog's `run_library` call passes no index, so `LibrarySyncDialog` still retries failures every time** (it is retired at 27c).
  - **Republish (closes the chunk 21/24 carry-over).** `publish_reviewed_queue(ids=, republish=True)` also takes each *published* entry whose XML is missing from the repo, or whose current digest differs from `published_digest` (an entry with no recorded digest is left alone, as the index does);
    the same path is rewritten, the entry stays `published`, `published_digest` and `published_at` are updated, `revision` is untouched, and the result carries `republished: True`. `commit` then commits it as a revision. It covers a missing XML, not a missing image. A published title with a project conflict
    is reported as `project_conflict` on every republish run (it is `attention` in the index anyway). CLI: `publish|sync --republish [--id ID]`, `commit --id ID`; `run --needs publish --through publish` selects them by the index.
  - **Per-title isolation in publish (closes the chunk 21 carry-over).** Incomplete metadata is validated up front, before the image or projects are written, and returned as `{'id', 'error': 'invalid_metadata', 'problems': [...]}`; `describe_publish_error()` spells it out; the entry is left as it was.
  - **`repo_state()` fallback (closes the chunk 24 wrinkle).** With no `@{upstream}`, `unpushed` is measured against `refs/remotes/<remote>/<branch>` if it exists (what `push()` updates); a never-pushed branch or a detached HEAD is still `None`. `commit_catalogue()` needed no change (it already pushes when
    unpushed is unknown, and is now precise afterwards). One existing test changed meaning and was updated: after a `push()` with no upstream the state is "nothing unpushed", not unknown.
  - **Edited-project fact.** `publish.project.edited_projects(mono_path, multichannel_path)` -> `None`/`mono`/`multichannel`/`both` (a project with no `pipeline_filter_hash` counts as edited; an unreadable one raises and bulk accept excludes the title). It is not an index column (the schema is frozen); bulk accept reads it.
  - **Bulk accept.** `plan_accept(index, selection, threshold=0.90, queue_dir=, meta_defaults=, work_dir=)` -> `AcceptPlan(eligible, excluded=[Exclusion(id, title, reason)], below_threshold, not_for_review)`; `accept_top_pick(...)` does it (candidate 0 chosen, `status='accepted'`, reviewer note `bulk accepted, confidence >= 0.90` appended to any note).
    Only titles the index calls *review* are considered; a top pick below the threshold is counted, not "excluded"; a title is judged from its **queue entry**, not the index row, so a stale index cannot accept what is no longer pending (`already <status>`). Exclusions: incomplete metadata (with `meta_defaults`), a decline, an edited or unreadable project (needs `work_dir`), no queue entry.
    The threshold is a parameter; **the preference that backs it is not added** (26b/27c add it to `model/preferences.py`, default `bulk.DEFAULT_ACCEPT_THRESHOLD`). `accept_top_pick` does not refresh the index (call `index.refresh()`).
  - **CLI.** `run` gained `--needs --match --id --new-since-scan --through --retry-failed` and the repository options and `--push/--no-push` (for `--through publish|commit`); with **no selector it behaves as before** (lists the source and runs everything; only the skip of remembered failures is new, and it needs no flag).
    Any selector, or `--through`, or `--source` with `--profile`, is *selector mode*: it works from the last scan (scanning first if there has never been one) and prints a `StagesReport` as JSON (exit 1 if anything failed, was refused or could not be committed). **`--source NAME` is a profile source's name in selector
    mode and a source *kind* without one** (a config with only `run.source: jriver` gives a one-source profile named `jriver`, so both readings agree). New `accept` subcommand (`--profile --source --match --id --new-since-scan --threshold --dry-run`, `--work-dir --queue-dir`, and the scan's settings options so the index is refreshed correctly). Every option has help and a README entry.
  - **Not done / open.** The `accept` threshold preference and its confirmation UI (26b/27c); a per-source timeout; adopting a dangling claim; `Selection` has no "everything but Done" convenience (the strip's default view is `include_done=False` on `LibraryIndex.titles()`, not a `Selection` field, so 26a
    should filter Done itself when it lists, and select-all-in-filter should pass the chip's `Selection`); `Progress` is per title-stage rather than per title; `run --through commit` commits the *selection's* published titles, not every uncommitted file (a whole-queue commit is still `commit`).
  - **For chunks 26-27 -- the Python API the GUI calls** (all Qt-free; run the blocking ones on a `QRunnable`, marshal `Progress` through a signal):
    1. Open `LibraryIndex(index_path(work_dir))` once (thread-safe); list with `index.titles(...)`/`Selection.rows(index)`, counts with `index.summary()`; rescan with `index.scan(profile, settings, only=[source])`.
    2. The action: `plan = plan_stages(selection.rows(index), through, retry_failed=)` for the label and the "N of M skipped" text (`plan.label`, `plan.skipped[i].reason`), then `run_stages(profile, selection, through, run_config=LibraryRunConfig(...), index=index, publish=PublishSettings.from_scan_settings(settings, ...),
       settings=settings, retry_failed=, should_cancel=, on_progress=)` -> `StagesReport` (`run.failed`, `run.failed_earlier`, `skipped`, `published`, `publish_errors`, `committed`, `commit_error`, `cancelled`, `not_run`, `counts`). `run_config`/`settings` must match what a scan and `publish` are given; `PublishSettings` is required for `publish`/`commit`.
    3. **Retry failed** is `retry_failed=True` on the same call (the failures panel is `index.failures()`, keyed by title id, plus `TitleRow.failure`); do not call `clear_failure()` yourself.
    4. **Bulk accept**: `plan_accept(...)` for the confirmation (count, `excluded` list) then `accept_top_pick(...)`, then `index.refresh(profile, settings)`. The strip chips are `selection_from_chip()`; the parity test (`test_pipeline_library_selection.py`) pins each to its CLI flag.
    5. **Republish** needs no new call: an out-of-date published title has `needs == 'publish'` and `publish_state == 'out_of_date'`, and is run by `run_stages(..., through='publish')`.
  - Tests: `test_pipeline_library_selection.py` (vocabulary, what a selection picks, `plan_stages` per kind of title and `through`, and **the CLI-flag / strip-chip parity table**), `test_pipeline_library_stages.py` (`through`, selection, refresh not counting as a scan, progress, cancel between titles and between published
    entries, retry, publish/republish/commit against real temp git repos, a rejected push), `test_pipeline_library_bulk.py`, `test_pipeline_review_republish.py` (invalid metadata isolation, `ids`, republish, hooks, `commit_catalogue(ids=)`), additions to the run, index, CLI (including the whole-workflow test), git and project test files. Mutation-checked: the
    failure skip, the fingerprint and key halves of `failure_applies`, the invalid-metadata isolation, the republish digest and missing-file checks, `through` never past design, the review skip, cancel, `republish`/`ids` reaching publish and commit, the accept threshold, incomplete-metadata and edited-project exclusions, the git fallback, refresh keeping the generation, `commit ids`, the season-episode memory, and the CLI's selector mode and scan-first.

**26a -- Work list, read-only** (new `model/worklist.py`; `ui/worklist.ui`, generated)
- A `QMainWindow`, the pipeline strip with counts, the table model plus proxy over the index, filters and search, source combo, sort (tier then `state_since` ascending),
  Done hidden by default, new-since-scan highlight, a Rescan button with "last scanned", an empty state pointing to settings.
- A new menu entry beside Library Sync; the old dialog is untouched. Tests use a fixture index.
- **Profile bootstrap.** The settings editor is 26c, so until then, when no profile file exists, the window **builds a profile from the existing
  preferences** (`LIBRARY_SOURCE_DEFAULT`, the per-kind settings, `LIBRARY_WORK_DIR`, `DESIGNER_QUEUE_DIR`, `LIBRARY_XML_REPO`, `LIBRARY_IMAGES_REPO`,
  `DESIGNER_DEFAULT`, `LIBRARY_TV_MODE`) -- a single source, no ignore rules -- so nobody has to hand-write YAML to use M3. 26c then writes a real profile file.

**26b -- Work list, actions** (`model/worklist.py`)
- Multi-select, select-all-in-filter, the action button labelled by what it will do (and what it skipped), the run on a `QRunnable` with determinate progress and Cancel, a
  running-row marker, the failures panel with **Retry failed**, Publish and Commit buttons with a confirmation naming the repos and a per-title result list.
- The Tools entry now opens the new window. **Done when:** M3.

**26c -- Settings drawer** (`model/worklist_settings.py`)
- Profile editing, sources with drag-to-reorder priority (reusing the `SourcePage` kinds), the ignore-rule editor with a live "would ignore N titles" preview and
  **Ignore titles like this...** from a row, folder pickers, validation, persistence on change, an incomplete-setup banner. Independent of 27.

**27a -- Title page core** (in-window stacked page)
- Breadcrumb and Esc, prev/next over the filtered list with "n of N", candidates/commentary/chart (reusing `MagnitudeModel`), Accept-and-next, skip/reject,
  keyboard scoped to the table and candidate list.

**27b -- Metadata and artwork**
- Essentials / More, the validity badge from `validate()`, autosave on focus-out, TMDB reload, artwork; edits allowed on done titles (they become Publish: out of date, per §12.6).

**27c -- Project, revise, retire**
- **Open project** (mono / multichannel; opened through the parent `BeqDesigner`'s project import) and the modified-since-design badge; Reopen/Revise using
  chunk 22; the bulk-accept UI; the "settings changed" banner.
- **Delete** `LibrarySyncDialog` and the embedded review dialog's own Publish button; the standalone "Review Batch Designs" entry becomes the title page over a queue
  directory. Update `AGENTS.md`'s map and `pipeline/README.md`. **Done when:** M4 (closes T9, T10).

**28 -- Documentation (T1)**, following the outline in §12.15; includes the updated `manage_mc.md`/`preferences.md` and new screenshots. **Done when:** M5.

#### Definition of done, every chunk

- Pipeline code: plain `pytest`, Qt-free (`test_pipeline_*.py`). Qt code: `pytest-qt` offscreen, with `import ui.beq` first (the circular-import note in `AGENTS.md`).
  `PYTHONPATH=./src/main/python QT_QPA_PLATFORM=offscreen uv run pytest src/test/python` green.
- `.ui` edits are regenerated with `ui/convert.sh` and both files committed; nothing generated is hand-edited. Lazy imports in `app.py` are kept.
- New CLI options carry help text and a README entry (`test_pipeline_library_cli_docs.py`).
- Additive schema changes update `docs/schema/review_queue.schema.json`; an existing queue and work directory load unchanged.
- **Per `AGENTS.md`, after every commit** this section and the §8 table are updated with the commit hash and any deviation, in the same commit where possible.

#### Risks, by chunk

| Chunk | Risk | Mitigation |
|---|---|---|
| 21 | git edge cases: detached HEAD, no upstream, a dirty tree | `repo_state()` degrades to "unknown"; commit only named paths; temp-remote tests |
| 23 | the claim rules must adopt existing ids without renaming them | claims are reconstructed from existing queue entries and work dirs; test against a fixture of real-shaped ids |
| 24 | index performance and schema churn | the 5,000-item budget test; freeze the schema at exit |
| 24 | `state_since` is lost on a rebuild | documented; only affects the oldest-first order after a rebuild |
| 26-27 | size: the UI is most of the work | split into 26a/b/c and 27a/b/c, each shippable behind the old dialog |
| 27c | opening a project from the work window needs a route to `BeqDesigner` | pass the parent's import callable in; test with a stub |
