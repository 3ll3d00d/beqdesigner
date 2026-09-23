# Library sync pipeline -- plan

Status: **chunks 1-2 and 4-28 are built; 1,902 tests passed in the 2026-09-21
repository sweep. Chunk 3 remains an external-evidence spike. The remaining
verification, product work and deliberate limits are classified in §10 and
scheduled in §13; do not infer that every historic T-item is unfinished.**
**§12 (added 2026-09-19) is the agreed design for the reworked workflow and
work-list UI: multi-source profiles, discovery, per-stage state,
publish/commit, revision and review. Chunks 19-28 are built through M5;
chunk 28’s documentation is commit `ecd8cef`. It supersedes §7's GUI. Its
implementation order is §12.13.**
**§14 is the planned follow-on for bounded parallel work-list runs, live
per-row progress and per-title run details. It is design only; no code has
been changed for it.**
Written 2026-09-17. Builds on the
headless pipeline in `pipeline/` (see `pipeline/README.md` and
`design/api-headless-pipeline.md`/`pipeline-implementation-plan.md`,
all shipped) and the existing GUI Batch Extract & Design workflow
(`model/batch.py`).

## How to use these documents (read this first)

This file is the **index**: the goal, what already exists, the status of every chunk, and where each part of
the design lives. It is usually all you need to decide what to do next. The design itself is split by topic under
`design/library-sync/`, so that a task reads one focused file rather than the whole plan.

1. Find your area in **Where things are** below and open **only that file**. `library-sync/archive/` holds the
   original specs of chunks that are built and shipped; read it only if you need that detail.
2. **Section numbers (`§3.3.1`, `§11.7`, ...) are global**, and code comments cite them as
   `design/library-sync-pipeline-plan.md §N`. The map below says which file holds a given `§`.
3. **This file holds status; the topic files hold design.** After every commit (`AGENTS.md`, "Working from a plan") update
   the chunk table in §8 below with the hash and status, plus the one topic file the chunk touched. Do not restate status inside topic files.
4. **Keep files under ~500 lines.** When a topic outgrows that, split it; do not append "because it is related". A chunk that
   needs a detailed spec gets its own file, `library-sync/chunks/NN-name.md`, linked from its §8 row -- not another appendix.

### Where things are

| `§` | File | What it holds | State |
|---|---|---|---|
| §1, §2, §8 | this file | goal, reuse table, chunk status | -- |
| §3, §3.1-§3.2 | [`library-sync/source-and-metadata.md`](library-sync/source-and-metadata.md) | `LibrarySource`, the JRiver adapter, metadata resolution, reviewer editor, artwork | built |
| §3.3, §3.3.1 | [`library-sync/local-artifacts-and-projects.md`](library-sync/local-artifacts-and-projects.md) | per-title `.beq` projects and why the project is what gets published | built |
| §4 | [`library-sync/idempotency.md`](library-sync/idempotency.md) | extract cache, design cache, sync | built |
| §5, §6, §7 | [`library-sync/orchestration-cli-gui.md`](library-sync/orchestration-cli-gui.md) | `run_library`, the CLI, and the GUI as first built (§7: superseded by §12.10, and its code deleted in chunk 27c) | built; §7 is history |
| §9, §10 | [`library-sync/status-and-open-items.md`](library-sync/status-and-open-items.md) | risks and the authoritative classification of historic T-items | current |
| §11 | [`library-sync/jriver-and-sources.md`](library-sync/jriver-and-sources.md) | shared JRiver connections, filesystem source, source/node pickers, live findings, path mappings, id fields, DVD, TV seasons | built |
| §12 (except §12.13) | [`library-sync/workflow-rework/design.md`](library-sync/workflow-rework/design.md) | the agreed workflow, discovery, state machine, revise, screen; **the frozen index schema (§12.5)** | discovery (chunk 24) and the stage entry points, selectors and bulk accept (chunk 25) built; the work list window (§12.10, chunk 26a, read-only), its actions (26b) and its settings drawer (26c) are built; the title page core (27a: candidates, chart, accept/skip/reject, prev/next) and its metadata and artwork (27b) are built (committed); the title page's projects, Reopen / Revise, bulk accept and the retirement of the old dialogs (27c) are built (committed); the user documentation (28, `docs/library/`) is built in `ecd8cef` |
| §12.13 | [`library-sync/workflow-rework/implementation-order.md`](library-sync/workflow-rework/implementation-order.md) | chunks 19-28: order, dependencies, milestones, risks | chunks 19-25 and 26a-27c built (M4: the title page with its metadata, projects, revise and bulk accept, the old dialogs retired); 28 (M5, the documentation) built in `ecd8cef` |
| §13 | [`library-sync/sweep-up.md`](library-sync/sweep-up.md) | follow-on completion plan: live verification, JRiver and disc gaps, path UX, TVDB, season fidelity, JSON-output completion, selected-audio metadata | done: 29, 34-36, 38, 41; in progress: 39-40; externally blocked: 30-33, 37 |
| §14 | [`library-sync/worklist-parallel-runs.md`](library-sync/worklist-parallel-runs.md) | stage-specific concurrency, per-row progress and details controls, and explicit action buttons | design agreed; implementation not started (chunk 42) |
| JSON output migration | [`library-sync/catalogue-json-output.md`](library-sync/catalogue-json-output.md) | BEQCatalogue filter-record output contract and migration scope | chunk 39 in progress |
| Appendix A-D | [`library-sync/archive/`](library-sync/archive/) | handoff specs for chunks 1, 2, 4, 5 | built, archival |

**Which file for which task**

| If you are... | Read |
|---|---|
| working on any chunk 19-28 | `workflow-rework/implementation-order.md`, then `workflow-rework/design.md` for the parts it cites |
| touching JRiver browsing, ids, paths or artwork | `jriver-and-sources.md`, then `source-and-metadata.md` |
| touching extract/design caching or the run loop | `idempotency.md`, `orchestration-cli-gui.md` |
| touching publish, projects or the review queue | `local-artifacts-and-projects.md` (and `workflow-rework/design.md` §12.6-§12.8 if it is chunk 21-22) |
| touching the GUI | `workflow-rework/design.md` §12.10 (target), `orchestration-cli-gui.md` §7 (the dialog the work list replaced: history) |
| looking for what is still unfinished | `status-and-open-items.md` §10 |
| working on a remaining item | `sweep-up.md`, then the topic file it names |
| implementing parallel work-list runs | `worklist-parallel-runs.md` (§14), then the existing §12.10 UI design |


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
| Human review UI | the Library Work List's title page, `model/worklist_title.py` (keyboard-first triage), also over a queue directory in the Review folder window, `model/worklist_review.py` (it was `model/review.py`'s `ReviewQueueDialog`, deleted at chunk 27c) |
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
| 8 | `pipeline/library/jriver.py`, built on `hamcws.MediaServer.browse_files()` and the configured browse-node id (§3.1), plus the `hamcws` dependency. If an id field exists, also `pipeline.metadata.tmdb_find_by_imdb_id()` + `pipeline/library/library_metadata.py::resolve_meta()` (§3.1.1). | 3, 4 | **Implemented -- commit `d870d6d`; `run_library()` calls `resolve_meta()` when `tmdb_api_key` is set. Artwork tiers 2/3 (§3.1.3) wired later, in `pipeline/library/artwork.py`** |
| 9 | `pipeline/library/cli.py` -- CLI entry point (§6). | 7, 8 | **Implemented -- commit `e23e03d`** |
| 10 | GUI: `model/library_sync.py`/`ui/library_sync.py` + `model/preferences.py` additions (§7), including the library-view filter bar (status + name/year/content-type, exact fields decided at UI design time), with a `pytest-qt` safety-net test before wiring, per this repo's established practice for touching a dialog. | 7, 8, 9 | **Implemented -- commit `9da7aea`; deferred: library-view filter bar, source picker/query field, browse-node selector** |
| 11 | Shared JRiver connections (§11.1): a `Preferences -> JRiver` pane owns add/test/delete of MCWS servers (`JRIVER_MCWS_CONNECTIONS`, unchanged storage); the JRiver filter manager's `MCWSDialog` and Library Sync both *pick from* that list instead of each managing their own. | 10 | **Implemented** -- `model/jriver/connections.py`, Preferences -> JRiver page, `MCWSDialog` trimmed to pick-only |
| 12 | `pipeline/library/filesystem.py` -- a Qt-free `FilesystemLibrarySource` (globs, BDMV roots) so "raw filesystem, as batch extract does" is a `LibrarySource` too (§11.2). CLI gains `--source filesystem`. | 4 | **Implemented** -- `pipeline/library/filesystem.py`; CLI `--source filesystem --glob ...` |
| 13 | Library Sync **source picker** (§11.3): a Source combo (Filesystem / JRiver servers / future kinds) over a per-kind settings page, via a small registry of source *kinds*; replaces the "first saved connection" logic and the hard-wired JRiver group. | 11, 12 | **Implemented** -- `model/library_sources.py`, `LibrarySyncDialog` source combo + stacked pages |
| 14 | JRiver **browse-node picker** (§11.4): a tree dialog over `Browse/Children` so the root node is chosen, not typed; the numeric field stays as a fallback. | 13 | **Implemented** -- `model/browse_node_picker.py`, `list_browse_children()`; response shape still unverified against a real server |
| 15 | Path mappings (§11.6): a per-server list of server-folder -> local-folder rules, edited in Preferences -> JRiver and applied when a JRiver source reads items (`pipeline/library/pathmap.py`, CLI `--path-map`). | 11, 13 | **Implemented** |
| 16 | Configurable external-id fields (§11.7): `Library/Fields` listing, per-kind defaults, and a per-server field mapping edited in Preferences -> JRiver. | 15 | **Implemented** -- `model/jriver/field_mappings.py`, per-server storage, Library Sync hand-off |
| 17 | DVD-Video rips (§11.8): `model/dvd.py` (title table + durations from the IFO files), read through ffmpeg's `dvdvideo` demuxer via new `Executor` input options; wired into `Session.extract`, Batch Extract, the filesystem source and JRiver's `VIDEO_TS.dvd;N` entries. | 15 | **Implemented**; multi-episode discs limited (see §11.8) |
| 18 | TV seasons (§11.9): metadata that marks the episodes a filter covers (built), TMDB season lookup + season/episodes on library items, and a `tv_mode` option -- one filter per episode, or the whole season as a single track. | 15 | **Implemented** -- `tv_mode` `episode` \| `season` |
| 19 | Workflow rework verification spike (§12.14): Enter-accepts confirmed, TMDB XML element, no-diff commit and foreign-staged-file findings, `QueueEntry` fields, season id shape. Tests only, no product code. | 18 | **Done -- commit `d4a33ce`** |
| 20 | Workflow rework: fixes to the current dialog that need no redesign (M0) -- review preloaded, entries named from the library, Enter scoped, failure details, unsaved-edit prompt, Reopen. | 19 | **Done -- commit `7197994`** |
| 21 | Workflow rework: publish/commit split -- `write_files`/`commit_paths`/`push`/`repo_state`, `commit_catalogue()`, `QueueEntry.published_digest`/`published_at`, CLI `publish`/`commit`/`sync`. (M1 needs 22 as well.) | 19 | **Done -- commit `d5a1708`** |
| 22 | Workflow rework: revise backend -- `reopen_entry`/`redesign_entry`/`revise_entry`, `invalidate_extract`, `QueueEntry.revision`, CLI `revise`. **Completes M1** (publish/commit + revision from the CLI). | 21 | **Done -- commit `58d318b`** |
| 23 | Workflow rework: profile (`profile.py`), ignore rules (`ignore.py`), the union of sources with hard/soft clashes and sticky ownership (`union.py`), season-id claims, CLI `run --profile`. | 19 | **Done -- commit `a05d20e`** |
| 24 | Workflow rework: discovery -- `state.derive_needs()` (the §12.6 table, pure), `status.py` (reads the outputs), `index.py` (SQLite index, **schema frozen in §12.5**, `scan`, `rebuild_from_outputs`, failure memory), `catalogue_scan.py` (XML repo awareness), `extract_status`/`design_status` split out of the wrappers, `current_publish_digest()`, `QueueEntry.source_fingerprint`, no `stat` in a JRiver listing, CLI `scan`/`status`. | 21, 23, 19 | **Done -- commit `5ce9de6`** |
| 25 | Workflow rework: stage entry points and selectors -- `selection.py` (`Selection`, the chips, `plan_stages()`), `stages.py` (`run_stages(... through ...)`, `Progress`, cooperative cancel), `bulk.py` (`plan_accept()`/`accept_top_pick()`), retry-failed (a remembered failure is no longer retried unasked, seasons' episodes included), republish of out-of-date published titles, per-title refusal of incomplete metadata in publish, `repo_state()` falling back to `<remote>/<branch>`, `LibraryIndex.units()`/`refresh()`, CLI `run --needs/--match/--id/--new-since-scan/--through/--retry-failed`, `accept`, `publish/sync --republish --id`, `commit --id`. **Completes M2** (the whole workflow headless). | 24 | **Done -- commit `20262db`** |
| 26a | Workflow rework: work list, read-only -- `model/worklist.py` (`WorkListWindow`), `worklist_model.py` (table model and proxy), `worklist_profile.py` (profile from a file or bootstrapped from the Library Sync preferences), `ui/worklist.ui`, a Tools menu entry beside Library Sync, `LIBRARY_PROFILE_PATH`. The pipeline strip with counts, search, source combo, tier-then-oldest sort, Done hidden, new-since-scan highlight, Rescan on a `QRunnable`, empty states, per-source errors. Runs nothing yet. | 24 | **Done -- commit `8c5abd1`** |
| 26b | Workflow rework: work list, actions -- `model/worklist_actions.py` (selection, the action button labelled by `plan_stages()`, Publish and Commit behind confirmations naming the repositories, Retry failed, failures panel, *Last run* results), `worklist_run.py` (`RunJob`, a `QRunnable` around `run_stages` with progress and cooperative Cancel; result wording), `worklist_confirm.py`, the running-row marker in `worklist_model.py`, `WORKLIST_PUSH`; the Tools menu opens the work list first and keeps the old dialog as "Library Sync (classic dialog)". **Completes M3.** Suite: 1428 passed. | 26a, 25 | **Done -- commit `dc2a5c3`** |
| 26c | Workflow rework: work-list settings drawer -- `model/worklist_settings.py` (`SettingsDrawer`), `worklist_sources.py`, `worklist_ignore.py`, `worklist_edit.py`; a real profile file edited on change (validated, debounced, atomic), first-save creates it and sets `LIBRARY_PROFILE_PATH`; sources with drag-to-reorder priority (the `SourcePage` kinds reused), ignore rules with a live "would ignore N titles" count and "Ignore titles like this...", the incomplete-setup and out-of-date banners, the profile's `designers:` registered by the window, the `WORKLIST_ACCEPT_THRESHOLD` preference; `save_profile()` atomic + validated. | 26a, 25 | **Done -- commit `873021b`** |
| 27a | Workflow rework: title page core -- `model/worklist_title.py` (`TitlePage`: header from the index row, candidates, commentary, chart, **Accept & next** / Skip / Reject over the queue entry, Previous / Next "n of N", Esc), `worklist_titles.py` (`WorkListTitles` mixin: double-click, Enter or *Open* stacks the page on `contentStack` over the listed titles; back with the selection, scroll and filters intact; the index is read again once, on a worker, when the page is left), `ui/worklisttitle.ui`, `openButton` and the `listHeader`/`listFooter` wrappers in `ui/worklist.ui`. Suite: 1695 passed. | 26a | **Done -- commit `b88ac1f`** (27a-27c share one commit) |
| 27b | Workflow rework: title page metadata and artwork -- `model/worklist_metadata.py` (`MetadataPanel`: Essentials / More, the validity badge from `validate()` through the index's own `metadata_problems`, autosave on focus-out and a `flush()` before the page moves, TMDB Reload and artwork download on the thread pool), `worklist_artwork.py`, `ui/worklistmetadata.ui`, a *Metadata* tab on `ui/worklisttitle.ui`; Accept is not offered while the metadata is incomplete; edits are allowed on every status and reach the index as `changed` (a published title becomes Publish: out of date). An independent review's 13 findings are fixed (a late TMDB answer, the API key in error text, a refused Accept not moving the keyboard, Skip/Reject behind an unsaveable edit, a blank field unsetting its key, the layout, the close order, ...; see the 27b block in `implementation-order.md`). Suite: 1821 passed. | 27a | **Done -- commit `b88ac1f`** (27a-27c share one commit) |
| 27c | Workflow rework: project, revise, retire -- *Open project* on the title page (through a callable `BeqDesigner` hands the window) with the "modified since design" badge (`worklist_projects.py`, `worklist_title_actions.py`), Reopen / Revise on the page and on the selected rows (`worklist_revise.py`), bulk accept over `WORKLIST_ACCEPT_THRESHOLD` and the "settings changed" banner (`worklist_bulk.py`, `pipeline/library/drift.py`), and the **Review folder** window (`worklist_review.py`, Tools > Review Folder...) which publishes through the work list's own `publish_library`/`commit_library`. **Deleted:** `LibrarySyncDialog`, `ReviewQueueDialog` (and its XML-only Publish button), their UI files and tests. **Completes M4; closes T9 and T10.** Suite: 1871 passed; **1900 after the independent review's fixes** (the folder window's interlocks, the redesign hold, truthful commit state, a re-read profile, Publish as the work list's path, bulk accept's account of what was not accepted, cheaper drift, `worklist_title_decide.py`; see `implementation-order.md`). | 27b, 26b, 22 | **Done -- commits `b88ac1f` (the title page, revise, bulk accept, banner) and `cf9561f` (the Review Folder window, the old dialogs retired)** |
| 28 | Workflow rework: documentation (§12.15) -- the user guide, `manage_mc.md` and `preferences.md`, new screenshots (T1). Detail in [`library-sync/workflow-rework/implementation-order.md`](library-sync/workflow-rework/implementation-order.md) (§12.13). | 27c | **Done (2026-09-21) -- commit `ecd8cef`**. `docs/library/` (9 pages), the updated `manage_mc.md`, `preferences.md`, `extract_audio.md`, `batch_extract.md`, the `mkdocs.yml` nav, 20 new or replaced screenshots; closes T1; **completes M5**. Docs only, so the suite is unchanged (1902 passed). |
| 29 | Sweep-up baseline -- make §10’s status current, move closed items out of the open list, and add the follow-on execution plan (§13). | 28 | **Done -- commit `791af5b`** |
| 30 | Live JRiver evidence -- capture sanitised `Browse/Files`/`Browse/Children` and `/Alive` fixtures; reproduce duplicate child names and `INTERNAL` artwork (T5-T6, chunk 3). | 29; access to a real JRiver server | **Not started — externally blocked** |
| 31 | Manual acceptance runbook and evidence -- exercise the GUI, a real JRiver source and season mode; record an end-to-end CLI run and sync (T2-T4). | 30; real designer, repositories and media | **Not started — externally blocked** |
| 32 | Disc title selection -- resolve or explicitly choose JRiver DVD titles and expose a single-file DVD picker; add fixture-backed coverage (T7). | 30 for JRiver shape; DVD fixture | **Not started** |
| 33 | Resolve JRiver Blu-ray playlist pseudo-paths, preserving a safe fallback for unknown playlist numbers (T8). | 30; playlist evidence | **Not started — evidence dependent** |
| 34 | Path-mapping guardrails -- flag unmapped Windows paths on non-Windows hosts and add a local-folder picker (T11). | 29 | **Implemented -- commit `f6d466d`** |
| 35 | TVDB IDs -- add optional `tvdb` external-ID mapping and TMDB lookup, retaining title/year fallback (T12). | 29 | **Implemented -- commit `4036bd5`** |
| 36 | Asynchronous MCWS zone loading (T13). | 29 | **Implemented -- commit `e87f4bb`** |
| 37 | Season fidelity spike and implementation decision: measure joined-season levels and multichannel feasibility, then either add a documented policy or retain the boundary (T15). | 31; representative media | **Not started — evidence dependent** |
| 38 | Close or promote intentional boundaries: confirm `registry` remains an extension seam (T14) and candidate reset on redesign remains correct (T16); remove them from the open-work list or create a separately approved feature plan. | 29 | **Implemented -- commit `323ac81`** |
| 39 | Complete the BEQCatalogue JSON-output migration: replace remaining `xml_*` settings, CLI flags and UI/docs terminology with filter-record names; migrate the legacy XML-specific review/commit/index tests; add an end-to-end BEQDesigner → BEQCatalogue → `CatalogueEntry` round trip; configure the production record repository in BEQCatalogue. | JSON publisher commit `d8f4219`; BEQCatalogue contract/reader `be708773b` | **In progress — core publisher landed; Settings now exposes two destination locations and derives git roots/subfolders; terminology, CLI, full round trip and onboarding remain** |
| 40 | JRiver Playback Info resolver: request the field and turn its selected container-stream data into an ffprobe-matched audio-stream ordinal. | sanitised Playback Info fixture | **In progress — `Playback Info` is requested and the resolver seam explicitly returns the established first audio stream; parsing and ffprobe reconciliation remain** |
| 41 | Selected-stream metadata and override: resync BEQ audio metadata from the resolved stream; let a reviewer choose a stream and safely re-extract/redesign when it changes. | 40 | **Implemented against chunk 40's first-stream resolver — commit `13e1519`: JRiver stream codec/channel descriptions drive automatic BEQ audio types; the title page persists an alternate stream, preserves a differing reviewer edit and invalidates extraction/design. A Revise > Redesign/Re-extract now hands straight into Extract & design after its row refreshes, so its label matches the outcome. ffprobe reconciliation remains part of chunk 40.** |
| 42 | Bounded parallel work-list runs with independent extract, design and publish limits; progress in each work-list row; and a live per-title details window showing execution steps and commands. | 25, 26a-26b | **Planned in §14 — design only; implementation not started** |
