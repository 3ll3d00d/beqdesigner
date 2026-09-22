# Library sync plan -- open questions and implementation status

> Part of the library sync plan -- **start at the index**: [`../library-sync-pipeline-plan.md`](../library-sync-pipeline-plan.md).
> Contains §9, §10. Section numbers are global across the plan; the index maps every `§` to its file.
> Status of what this file describes: swept against `HEAD` on 2026-09-21 (1,902 tests passed); **§10 classifies every historic T-item and §13 schedules the remaining work**

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

## 10. Implementation status vs. this plan

The 2026-09-21 repository sweep verified that every implementation commit named
in §8 except chunk 3 is reachable from `HEAD`, that its claimed modules and
focused tests remain present, and that the complete offscreen suite passes
(**1,902 passed**). The follow-on order, ownership and external prerequisites
are in [`sweep-up.md`](sweep-up.md) §13. Historic T identifiers remain below so
links and commit notes stay meaningful.

**Behaviour gaps -- designed above (1-4 all now built)**

1. ~~Library artwork tiers 2 and 3 (§3.1.3)~~ -- fixed:
   `pipeline/library/artwork.py`, resolved in `design_if_needed()`. Entries
   designed before this change get a poster only on their next redesign.
2. ~~`project_edit_preserved` reporting (§3.3.1)~~ -- fixed: see §3.3.1
   (`LibraryRunReport.project_edit_preserved`, publish `edited_project`).
3. ~~Write-back of the authoritative project into the other one (§3.3.1)~~ --
   fixed: `align_projects()`, publish `projects_aligned`.
4. ~~Sync errors were not shown in the GUI (§7)~~ -- fixed: refused
   entries are no longer counted as published and are listed to the reviewer
   (Library Sync dialog and review dialog). Also fixed the review dialog's
   "Published N" message, which was overwritten by the queue summary at once.
   The later XML-only review-dialog Publish gap was T9; it closed when the
   dialog was retired in chunk 27c.

**Deviations that changed idempotency/cost -- fixed**

5. ~~`resolve_meta()` ran before the design cache check~~ -- fixed in
   `efb300f` (lazy callable; TMDB errors degrade to `item.meta`, reported in
   `meta_unresolved`).
6. ~~Design fingerprint omitted `keep_multichannel`/`audio_stream`~~ -- fixed
   in `e68511f` (metadata excluded on purpose; see §4.2).
7. ~~Kept multichannel extraction not gated on the source being
   multichannel~~ -- fixed in `1719151` via `source_channel_count`. Turned
   out to be masked by a pre-existing bug, also fixed (`dfcb7ff`): every
   mono-source extraction failed on an invalid `pan` filter.
8. ~~`kind` ignored `Media Type`/`Media Sub Type`~~ -- fixed, and verified live (below).
9. ~~Redesigning a `pending` entry discarded a reviewer's metadata and
   artwork edits~~ -- fixed (see the commit that follows `5c2dfdf`):
   `design_if_needed()` now keeps the existing entry's `meta` (fresh
   metadata only fills missing keys), `art_path`/`art_overridden` and
   `reviewer_note` across a redesign. `status` and the chosen candidate still
   reset to `pending`, since the candidates they refer to are replaced.

### TODO

T17. **JRiver's selected audio stream is not carried into BEQ metadata** --
the current library path selects its default first ffprobe audio stream, while
the metadata comes from broad JRiver fields rather than the selected stream.
Chunk 40 records the required work: parse JRiver `Playback Info`, match its
container stream id to ffprobe's global stream index, then persist the matched
codec/channel type as `QueueEntry.meta['audio_types']` without overwriting a
reviewer edit. The real `Playback Info` shape has now been observed; parsing,
sanitised fixture coverage and propagation are still **todo**.

### Closed

*Documentation*

T1. ~~**User documentation (mkdocs, `docs/`) does not cover any of the library
    work, and one page is now wrong.**~~ **Closed 2026-09-21 (chunk 28, commit `ecd8cef`).**
    `docs/library/` (overview, concepts, set up a catalogue, discover, do the work, review, publish and commit,
    revise, unattended) is the user guide, linked from `mkdocs.yml` and `docs/index.md`; `manage_mc.md` and its
    screenshots describe the server picker that moved to Preferences -> JRiver; `preferences.md` has the Designers and
    JRiver pages; `extract_audio.md` and `batch_extract.md` cover the design step, the Review Folder hand-over and
    DVD/Blu-ray discs. The CLI stays documented in `pipeline/README.md`, which `unattended.md` links to.
    `mkdocs build --strict` passes. **What the user docs still lack:** anything a person has checked in the running
    app (T2); a screenshot of a run in progress or of a dark palette; the older 2020 pages' dark screenshots; the
    `revise`/`accept`/`publish` commands beyond "not for a schedule"; and the profile keys the drawer does not edit
    (`meta_defaults`, `audio_types`, `commit_message`), which are left to the pipeline README. Found while writing
    them, left as they are (the list is in `implementation-order.md`, chunk 28): a stray progress bar and *Cancel*
    after a status message that is not a run, a TMDB key row that points at Preferences where there is no field, and
    the design/build differences in §12.4 and §12.10.

T9. ~~**The review dialog's own Publish button is XML-only and ignores `work_dir`**~~
    **Closed in chunk 27c (2026-09-21, commits `b88ac1f` and `cf9561f`):** the dialog and its button are deleted; the Review Folder window publishes through `publish_library()`/`commit_library()`, so it writes images, reads projects and refuses incomplete metadata per title (§12.10).

T10. ~~**Library view filter bar** (status / name / year / type) for the Run tab (§7).~~
    **Closed:** superseded by the work list (§12.10, chunks 26a-b); the Run tab it was for is deleted (27c).

T11. ~~**No warning for a path still in Windows form on a non-Windows host**~~
    **Closed in chunk 34:** JRiver items now carry a safe diagnostic before ffmpeg is called when a
    drive-letter or UNC path has no local mapping on a non-Windows host. The run result tells the person to add it in
    Preferences > JRiver, and that page's path-mapping table has a local-folder chooser. Local POSIX paths and native
    Windows behaviour are unchanged.

T12. ~~**`tvdb` identifier**~~ **Closed in chunk 35:** Preferences > JRiver now optionally maps a TVDB series-id
    field (empty by default). For TV titles a valid configured value resolves through TMDB's `tvdb_id` external-id
    lookup after the existing direct-TMDB and IMDb paths, before title/year search; missing or invalid values retain
    the existing fallback.

T13. ~~**`MCWSDialog`'s zone loading is still synchronous**~~ **Closed in chunk 36:** zone discovery now runs on a
    QRunnable. The current connection and zone controls indicate loading without blocking the UI; errors remain inline,
    and a response arriving after the dialog closes is discarded.

T14. ~~**`pipeline.library.registry` has no production callers**~~ **Accepted boundary in chunk 38:** it remains the
    deliberately uncalled, Qt-free in-process extension seam for a future Kodi/Plex source. Sources used by the CLI
    and work-list profile are constructed from persisted configuration; wiring the registry into those paths would add
    global mutable state without a second source to consume it.

T16. ~~**Redesigning a `pending` entry resets its status and chosen candidate**~~ **Accepted boundary in chunk 38:** a
    candidate selection is a decision about the old candidate list, which redesign replaces. Resetting to `pending`
    with no selection is therefore required; reviewer metadata, artwork and note remain preserved.

### Blocked — external verification required

T2. **The GUI has only been exercised by offscreen pytest-qt, never by a person
    in the running app**: Preferences -> JRiver (list, edit, async test,
    aliases, path-mapping table, field editor), and -- since the dialogs were
    replaced (chunks 26-27c) -- the Library Work List (strip, actions, settings
    drawer with its source picker, browse-node picker and ignore rules), the
    title page (candidates, metadata and artwork, Open project into the real main
    window, Reopen / Revise, bulk accept), the Review Folder window and its
    Publish / Commit against real repositories, and a real JRiver source, A.8.
T3. **`tv_mode='season'` has not been run against a real library** (the local
    server was unreachable); grouping and joining are unit-tested only (§11.9).
T4. **The CLI's `run` and `sync` have not been run end to end** against a real
    designer or repositories; the tests stub the run and publish steps (§6).
T5. **`/Alive`'s `FriendlyName`** is tested against a local fake only; it was
    outside the live-check permission (§11.1).
T6. **Chunk 3 (the sanitised real-server fixture) was never captured.** The
    endpoints and field names were verified live on one library (§11.5), but
    only the author's; `Browse/Children` keys by name, so two same-named
    siblings would collapse into one entry (not seen). `INTERNAL` artwork is
    handled but untested.

### Blocked — dependent product work

T7. **DVD, multi-episode discs (§11.8; chunk 32, blocked on chunk 30).** A JRiver item cannot say which title it
    is, so every JRiver item on a disc resolves to that disc's main title
    ("play all" for a multi-episode disc): several items, one audio. Needs a
    per-item title override or de-duplication by disc. Also: no DVD title picker
    in the single-file Extract dialog; Batch Extract takes only the main title;
    no end-to-end DVD test (a disc libdvdread accepts needs real navigation
    tables).
T8. **Blu-ray `BDMV\PLAYLIST\index.bluray;N` entries (chunk 33, blocked on chunk 30)** (4 shows) are passed
    through unresolved -- which title `N` names is unknown (§11.5).
### Blocked — evidence-based boundary decision

T15. **Season fidelity (chunk 37, blocked on chunk 31 and representative media).** Season mode does not keep
     multichannel, joins episodes with no level matching, and is only as complete as the browse node (§11.9).

### To do

Chunk 39 remains in progress: complete the JSON-output terminology/profile migration, source-record and aggregate
assertions, BEQDesigner → BEQCatalogue → `CatalogueEntry` fixture, and production record-repository onboarding. See
[`sweep-up.md`](sweep-up.md) §13.3 and `catalogue-json-output.md`.
