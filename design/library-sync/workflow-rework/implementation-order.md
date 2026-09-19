# Library sync plan -- workflow rework: implementation order

> Part of the library sync plan -- **start at the index**: [`../../library-sync-pipeline-plan.md`](../../library-sync-pipeline-plan.md).
> Contains §12.13. Section numbers are global across the plan; the index maps every `§` to its file.
> Status of what this file describes: **not started**; the status of each chunk lives in the index

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
- Read the XML writer/sample XML for the element carrying the TMDB id (chunk 24's repo awareness) and record its name.
- Run `commit_and_push` with unchanged content against a temp repo; record whether `git commit` errors (chunk 21).
- List every `QueueEntry` field (timestamps?), and check how a season title's id flows through publish naming.
- **Done when:** §12.14 is rewritten as verified facts, and the strict-xfail test is committed.

**20 -- Fixes that need no redesign** (`model/library_sync.py`, `model/review.py`, `pipeline/library/run.py`, `ui/review.ui`)
1. The Review tab is created on open and loads `DESIGNER_QUEUE_DIR`.
2. `_meta_source()` seeds `meta` with `title`/`year` from the item (`item.title or item.display_name`) on every path, including no-TMDB and
   TMDB-failure, so `QueueEntry.meta` always has a title; existing entries with none keep showing the id.
3. Shortcuts scoped to the table/candidate list, not text fields (turns 19's xfail into a pass).
4. A details view for `report.failed` and `report.meta_unresolved` (id, message), replacing the bare counts.
5. Accept with unsaved metadata edits prompts to save (or saves them).
6. **Reopen** on an *accepted* (not yet published) entry: status back to pending. This is only the trivial case, a status change in the dialog;
   chunk 22's `reopen_entry()` generalises it (written, committed and pushed entries, working-tree revert, revision counter) and 27c moves the dialog onto it.
- **Tests:** `gui/test_review_dialog.py`, `gui/test_library_sync_dialog.py` (new): open with an existing queue; a run with no TMDB key names
  its rows; Enter in a field does not accept.
- **Done when:** M0. Ships alone; nothing depends on it.

**21 -- Publish/commit split** (`pipeline/publish/git.py`, `pipeline/orchestrate.py`, `pipeline/review.py`, new `pipeline/library/commit.py`)
- `git.py`: `write_files()` (working tree only), `commit_paths()` (only the named paths; "nothing to commit" is success),
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

**22 -- Revise backend** (`pipeline/review.py`, `pipeline/library/design_cache.py`, `extract_cache.py`)
- `reopen_entry()`: written-but-uncommitted -> delete or `git checkout` the working-tree files and set pending; committed/pushed -> pending with a
  `revision` counter (the same path is rewritten and recommitted).
- `redesign_entry()` clears the accepted/published protection deliberately (the hash gate still preserves project edits);
  `invalidate_extract()` drops a manifest's keys. A reviewer note records the reason. `QueueEntry.revision` (additive).
- CLI: `revise --id ID --to {review,design,extract}`.
- **Tests:** each path and each starting state, including that project edits survive a redesign and that a reopened, pushed entry publishes to the same path.
- **Done when:** M1.

**23 -- Profile, union, ignore rules** (new `pipeline/library/profile.py`, `union.py`, `ignore.py`)
- The profile schema over the existing config shape: ordered `sources`, `ignore`, repos, work/queue dirs. The old CLI config still loads unchanged.
- `union_items(profile) -> UnionResult(titles, shadowed, duplicates)`: hard-clash normalisation (path mapping, case-fold, separator, disc root),
  soft-clash detection, priority, **sticky claims** reconstructed from queue entries and work dirs (existing ids `jriver-<hash>-<Key>` / `fs-<hash>` are
  simply claims; nothing is renamed).
- `ignore.py`: the rule model, `evaluate()` and `explain()` (which rule matched); rules on source/path/title/year/kind/external ids; per-title flag.
- CLI: `run --profile PATH`.
- **Tests:** JRiver-mapped path == filesystem path; BDMV root vs its clip; case/separator variants; soft clash by tmdb, imdb and title+year; reordering priority keeps
  ids; an owner that vanishes hands over; each rule field; deleting a rule un-ignores; profile round-trip; the old config loads.

**24 -- Discovery index and states** (new `pipeline/library/status.py`, `index.py`, `state.py`)
- Split `extract_status()` / `design_status()` out of the `*_if_needed()` wrappers (which then call them): a refactor with **no behaviour change**,
  proven by the existing suite plus equivalence tests.
- `state.derive_needs()`: the §12.6 table as a pure function. `index.py`: SQLite schema, `scan(profile)`, `rebuild_from_outputs()`, `state_since`,
  failure memory (`run_library` records failures against the fingerprint/params they failed on), repo awareness (XML scan by TMDB id, using chunk 19's finding).
- CLI: `scan`, `status [--json]`.
- **Tests:** one case per row of the needs table and per flag; index deletion -> rebuild yields the same states (`state_since` resets to the rebuild time --
  accepted and documented); a 5,000-item synthetic scan stays within a budget; a JRiver listing performs **no** `os.stat`/`isfile` on media (monkeypatched to fail).
- **Done when:** the schema is **frozen** and written into §12.5.

**25 -- Stage entry points and selectors** (`pipeline/library/run.py`, new `selection.py`)
- `Selection` (needs, source, match, ids, new-since-scan) and `run_stages(profile, selection, through, should_cancel, on_progress)`; `Progress(done, total, title, stage)`;
  `plan_units()` moves into discovery; artwork lookup moves from listing to design; failures persisted through the index.
- `accept_top_pick(selection, threshold)` with an exclusion report (incomplete metadata, decline reason, edited project) and the reviewer note.
- CLI: `run --needs ... --source ... --match ... --through ...`; a `run` with no selector behaves as today.
- **Tests:** `through` semantics (design extracts first; a protected entry is never redesigned); cancel mid-run leaves consistent state; retry-failed only on an explicit request
  or a changed fingerprint; a parity table proving the CLI flags and the GUI strip map to the same `Selection`; bulk-accept exclusions.
- **Done when:** M2.

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
