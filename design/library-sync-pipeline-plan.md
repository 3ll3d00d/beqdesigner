# Library sync pipeline -- plan

Status: **chunks 1-2 and 4-10 implemented (2026-09-18); full suite green
(471 passed, 2026-09-19 review). Chunk 3 (real-server spike) is not done,
and a small set of design items in §3.1.3, §3.3.1 and §4 remain unbuilt --
see §10 "Implementation status vs. this plan" for the authoritative list.**
**§12 (added 2026-09-19) is an agreed *design only* for a reworked workflow and
work-list UI (multi-source catalogue profile, discovery index, per-stage state,
publish/commit split, revise); only chunks 19 (a tests-only spike), 20 (fixes to the current dialog) 21 (publish/commit split), 22 (revise backend), 23 (profile, union, ignore rules) and 24 (discovery index and states; the index schema is frozen in §12.5) are built, and it supersedes §7's GUI.
Its build order is §12.13.**
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
| §5, §6, §7 | [`library-sync/orchestration-cli-gui.md`](library-sync/orchestration-cli-gui.md) | `run_library`, the CLI, and the **current** GUI (§7 superseded by §12.10) | built |
| §9, §10 | [`library-sync/status-and-open-items.md`](library-sync/status-and-open-items.md) | risks, and the authoritative list of what is still open (T1-T16) | current |
| §11 | [`library-sync/jriver-and-sources.md`](library-sync/jriver-and-sources.md) | shared JRiver connections, filesystem source, source/node pickers, live findings, path mappings, id fields, DVD, TV seasons | built |
| §12 (except §12.13) | [`library-sync/workflow-rework/design.md`](library-sync/workflow-rework/design.md) | the agreed workflow, discovery, state machine, revise, screen; **the frozen index schema (§12.5)** | discovery built (chunk 24); the rest **design, not built** |
| §12.13 | [`library-sync/workflow-rework/implementation-order.md`](library-sync/workflow-rework/implementation-order.md) | chunks 19-28: order, dependencies, milestones, risks | chunks 19-24 built; 25-28 not started |
| Appendix A-D | [`library-sync/archive/`](library-sync/archive/) | handoff specs for chunks 1, 2, 4, 5 | built, archival |

**Which file for which task**

| If you are... | Read |
|---|---|
| working on any chunk 19-28 | `workflow-rework/implementation-order.md`, then `workflow-rework/design.md` for the parts it cites |
| touching JRiver browsing, ids, paths or artwork | `jriver-and-sources.md`, then `source-and-metadata.md` |
| touching extract/design caching or the run loop | `idempotency.md`, `orchestration-cli-gui.md` |
| touching publish, projects or the review queue | `local-artifacts-and-projects.md` (and `workflow-rework/design.md` §12.6-§12.8 if it is chunk 21-22) |
| touching the GUI | `workflow-rework/design.md` §12.10 (target), `orchestration-cli-gui.md` §7 (what exists) |
| looking for what is still unfinished | `status-and-open-items.md` §10 |


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
| 18 | TV seasons (§11.9): metadata that marks the episodes a filter covers (built), TMDB season lookup + season/episodes on library items, and a `tv_mode` option -- one filter per episode, or the whole season as a single track. | 15 **Implemented** -- `tv_mode` `episode` \| `season` |
| 19 | Workflow rework verification spike (§12.14): Enter-accepts confirmed, TMDB XML element, no-diff commit and foreign-staged-file findings, `QueueEntry` fields, season id shape. Tests only, no product code. | 18 | **Done -- commit `d4a33ce`** |
| 20 | Workflow rework: fixes to the current dialog that need no redesign (M0) -- review preloaded, entries named from the library, Enter scoped, failure details, unsaved-edit prompt, Reopen. | 19 | **Done -- commit `7197994`** |
| 21 | Workflow rework: publish/commit split -- `write_files`/`commit_paths`/`push`/`repo_state`, `commit_catalogue()`, `QueueEntry.published_digest`/`published_at`, CLI `publish`/`commit`/`sync`. (M1 needs 22 as well.) | 19 | **Done -- commit `d5a1708`** |
| 22 | Workflow rework: revise backend -- `reopen_entry`/`redesign_entry`/`revise_entry`, `invalidate_extract`, `QueueEntry.revision`, CLI `revise`. **Completes M1** (publish/commit + revision from the CLI). | 21 | **Done -- commit `58d318b`** |
| 23 | Workflow rework: profile (`profile.py`), ignore rules (`ignore.py`), the union of sources with hard/soft clashes and sticky ownership (`union.py`), season-id claims, CLI `run --profile`. | 19 | **Done -- commit `a05d20e`** |
| 24 | Workflow rework: discovery -- `state.derive_needs()` (the §12.6 table, pure), `status.py` (reads the outputs), `index.py` (SQLite index, **schema frozen in §12.5**, `scan`, `rebuild_from_outputs`, failure memory), `catalogue_scan.py` (XML repo awareness), `extract_status`/`design_status` split out of the wrappers, `current_publish_digest()`, `QueueEntry.source_fingerprint`, no `stat` in a JRiver listing, CLI `scan`/`status`. | 21, 23, 19 | **Done -- commit `5ce9de6`** |
| 25-28 | Workflow rework (§12): stage entry points + selectors (25), work-list UI (26a-c), title page (27a-c), documentation (28). Detail, dependencies, milestones and risks in [`library-sync/workflow-rework/implementation-order.md`](library-sync/workflow-rework/implementation-order.md) (§12.13); design in [`workflow-rework/design.md`](library-sync/workflow-rework/design.md). | 24 | **Not started** (design agreed 2026-09-19) |
