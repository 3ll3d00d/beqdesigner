# Library sync plan -- sweep-up

> Part of the library sync plan -- **start at the index**:
> [`../library-sync-pipeline-plan.md`](../library-sync-pipeline-plan.md).
> Contains §13. Status: chunk 29 is **built** in `791af5b`; chunks 30-38 turn
> the post-M5 audit into small, independently reviewable pieces.

## 13. Sweep-up after M5

### 13.1 Baseline and goal

The original delivery is complete through M5: chunks 1-2 and 4-28 are at
`HEAD`, and the 2026-09-21 full suite passed 1,902 tests. Completion is not
the same as production acceptance: chunk 3 and T2-T16 identified external
evidence, product gaps and deliberate boundaries that the original sequence
did not land.

This plan closes those items without reopening the work-list redesign. Its
definition of done is deliberately stricter:

1. each remaining product behaviour is implemented and covered by a focused
   automated test;
2. each live-system assertion has a sanitised fixture or a dated, reproducible
   manual record;
3. intentional boundaries are explicitly accepted or promoted to a separate
   feature plan, never left under an "open" heading; and
4. the full suite and the strict documentation build pass.

### 13.2 Evidence boundary

The checkout cannot create evidence for a real JRiver server, a real designer,
real catalogue repositories or representative DVD/season media. Chunks 30,
31 and the evidence portions of 32, 33 and 37 require their owner to provide
access or a sanitised capture. They are not implementation blockers for the
independent code chunks, but they block declaring the entire library workflow
production-verified.

Sanitised fixtures must remove hosts, credentials, usernames, absolute private
paths, artwork bytes and title metadata that is not needed for the shape under
test. Store only the smallest response that proves the mapping; explain its
origin and sanitisation in the fixture’s adjacent test/docstring.

### 13.3 Work order

```
29 plan/status baseline       <- nothing
30 JRiver evidence            <- real server access
31 manual acceptance          <- 30, real designer/repos/media
32 DVD title selection        <- 30 (JRiver evidence), DVD fixture
33 Blu-ray playlist mapping   <- 30 (playlist evidence)
34 path guardrails            <- 29
35 TVDB lookup                <- 29
36 async zones                <- 29
37 season fidelity decision   <- 31, representative media
38 boundary disposition       <- 29
```

Chunks 34-36 and 38 can proceed in parallel. Chunks 30 and 31 have no safe
substitute: a fake server can test the resulting code, not prove the MCWS
shape that a real server returns.

### 13.4 Chunks

**29 -- Audit baseline and plan hygiene**

- Make the index and status page describe the 1,902-test audit rather than the
  obsolete 471/810-test snapshots.
- Move T1, T9 and T10 to a closed section; keep their resolution and commits.
- Classify T2-T6 as external verification, T7/T8/T11-T13 as product work, and
  T14-T16 as boundaries requiring disposition.
- Add this section and the 29-38 rows to §8.
- **Done when:** the index, §10 and this plan agree on every T-item.

**30 -- JRiver evidence and fixture (original chunk 3; T5-T6)**

- Against a real, authorised JRiver server, capture one selected browse node’s
  `Browse/Files`, its `Browse/Children` tree and `/Alive`; record configured
  external-ID/artwork fields and one `INTERNAL` artwork case if available.
- Add fixtures for two same-named child nodes. Change the picker’s data model
  to retain distinct node IDs rather than keying solely by display name.
- Pin the `hamcws` mapping, field aliases and `FriendlyName` parsing with
  fixture-backed tests. Unknown or absent values must retain today’s fallback.
- **Done when:** the capture is sanitised, the duplicate-name test passes and
  the source/picker behaviour is proven against it.

**31 -- Manual acceptance record (T2-T4)**

- Add a versioned runbook (no credentials or private paths) covering the
  Preferences JRiver page; source/browse/ignore setup; work-list actions;
  title-page metadata/artwork/projects/revise/bulk accept; Review Folder;
  a season source; and Publish/Commit against disposable real repositories.
- Run `scan`, `run`, `publish`, `commit`/`sync` against an actual designer,
  ffmpeg and the repositories. Attach a dated, sanitised result record that
  names app version, OS, MC version and command exit codes.
- Defects found here return to a normal focused code chunk; the record never
  masks a failure.
- **Done when:** every T2-T4 assertion has an observed pass/fail result and
  any pass claims made in user docs are supported by the record.

**32 -- DVD title selection (T7)**

- Add a selection abstraction shared by Extract Audio, Batch Extract and the
  JRiver source: each title has a stable number, label and duration; "main
  title" remains the default for unattended runs.
- The single-file dialog offers a DVD picker like the existing Blu-ray picker.
  JRiver entries either retain their reported title number or deduplicate to
  the disc with an explicit, documented main-title fallback; choose only after
  chunk 30 reveals the real pseudo-path shape.
- Add IFO fixture tests, dialog tests and an ffmpeg command test. A real DVD
  navigation test belongs in chunk 31’s evidence record.
- **Done when:** an interactive user can select a DVD title and a library run
  cannot silently produce several identical filters for one disc.

**33 -- Blu-ray playlist pseudo-paths (T8)**

- Determine, from chunk 30’s capture and the BDMV playlist files, whether the
  `;N` value identifies a playlist that `resolve_main_title()` can select.
- If it does, preserve the requested playlist in `LibraryItem` and forward it
  to extraction. If not, show a stable warning/detail and use the documented
  main-title fallback; do not guess from `N`.
- Test the resolved and unresolved forms, path normalisation and unchanged
  ordinary BDMV roots.
- **Done when:** the four observed forms either select their intended playlist
  or report a truthful, actionable fallback.

**34 -- Path-mapping guardrails (T11)**

- In `pathmap.py`, detect an unmapped drive-letter/UNC path on a non-Windows
  host and expose a non-sensitive diagnostic on `LibraryItem`/the run result.
  Keep legitimate local POSIX paths and Windows-host behaviour unchanged.
- Add a folder chooser to the local side of Preferences > JRiver path mappings;
  do not make it mandatory, because mappings may point to a mount not visible
  while editing.
- Test path classification, UI persistence and an extraction failure that
  surfaces the mapping hint.
- **Done when:** a common unmapped-path failure tells the user where to fix it.

**35 -- Optional TVDB lookup (T12)**

- Extend the per-kind external-ID configuration with `tvdb`, request its
  fields, and call TMDB’s `/find` using `tvdb_id` for TV titles before fuzzy
  title/year fallback.
- Preserve the existing IMDb/TMDB priority, tolerate missing/invalid IDs and
  make the setting opt-in by default until chunk 31 proves it against a live
  library.
- Add pure mapping, HTTP-binding and metadata-resolution tests.
- **Done when:** a configured TVDB field can resolve a TV series without
  weakening today’s fallback path.

**36 -- Asynchronous zone loading (T13)**

- Move `MCWSDialog` zone retrieval onto the established `QRunnable` + signal
  pattern. Disable only the affected controls, show progress/failure inline,
  and reject a late response after close or server change.
- Keep zone ordering and upload/download semantics unchanged.
- Add pytest-qt tests with a held fake request, failure, close and late-result
  cases.
- **Done when:** an unreachable server never blocks the UI thread.

**37 -- Season-fidelity decision (T15)**

- Measure representative episode levels and channel layouts during chunk 31.
  Record whether joined mono tracks need gain matching and whether preserving
  multichannel audio is a real user need.
- If approved, write a separate, bounded design for the chosen policy before
  implementation: gain reference, clipping rule, manifest/fingerprint change,
  migration behaviour and project semantics. Do not add an untested automatic
  normaliser opportunistically.
- If not approved, document the observed limitation in the user guide and
  close T15 as an accepted boundary.
- **Done when:** a recorded evidence-based decision exists; implementation is
  only complete if the approved policy’s separate plan is complete.

**38 -- Intentional-boundary disposition (T14, T16)**

- Confirm whether the source registry stays as a deliberately uncalled plugin
  seam for future Kodi/Plex sources. If yes, document that role and close T14;
  if no, remove it and its tests in the same change.
- Confirm that redesigning a pending entry must clear its status and selected
  candidate because candidates have been replaced. If this is no longer wanted,
  design candidate identity/provenance before changing the behaviour.
- **Done when:** neither item remains presented as unfinished product work.

### 13.5 Completion reporting

After each chunk, update its §8 row with the commit hash and update the mapped
T-item in §10. A blocked external-evidence chunk stays **externally blocked**;
it is never marked complete based on a fake-server test. Once all product
chunks and dispositions are complete, §10 may say “implementation complete,
live acceptance pending” until chunks 30-31 supply their records.
