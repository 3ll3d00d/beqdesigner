# TODO — prioritized work

## Status

Reviewed on **2026-10-28**, against beqdesigner `0b37ffb` (the service runs
2.2.0-alpha.30) and beqforge 0.3.1 (design version 8, contract 1.6). This is the **only current backlog** for the
pipeline and library work. Items are unbuilt, partial, awaiting evidence, or
under observation; none is claimed complete. Architecture and contracts are
[design references](README.md); completed plans and delivery records live in
[archive/](archive/README.md). Archived open rows are historical, not extra
backlogs. The caller’s by-reference implementation and live acceptance are
complete; its three follow-ons are retained below as D2–D4.

Order is recommended execution priority: the service's throughput and job
model first (they shape every further library run), the people's work (review,
E2) beside them, then the library run's loose ends, disc behavior, test health,
and optional improvements/product decisions. A blocked item does not prevent starting the
next independent item. E1's fixture (`src/test/python/fixtures/jriver/`) is
the evidence J2 and the disc items start from.

When an item is completed, archive its result and evidence, update the lasting
design description if necessary, and remove it here in the same commit.
Do not keep a second status table in a design reference.

### Initial release milestone

**Definition:** the published Docker image, with a documented beqforge
designer beside it, can be deployed and left to extract and design an entire
catalogue unattended, and the results reviewed from the desktop app (or, since
2026-10-10, decided over HTTP). Publish and commit stay a person's decision;
the schedule never goes past design.

The milestone is the rows marked **R** in the table. Everything built for it
is [archived](archive/initial-release.md): R1–R9, C1, J2, W2, E1, E4 and D4's
release part (priorities 1–14). **One item remains, and it needs the
maintainer:** E2, the acceptance run on real media with the published images,
`ghcr.io/3ll3d00d/beqdesigner-pipeline:2.2.0-alpha.1` and
`ghcr.io/3ll3d00d/beqforge-designer:0.2.0`.

Two beqforge items are recorded in its own `TODO.md`, not here: #1 "Playback
contract" (the designer side of R8: the library now sends
`run.bass_management`, of which beqforge 0.2.0 uses only the crossover, as
the review guide says) and #13 "Residual reporting" (the reported fit error
and band do not match the fit's). Neither blocks the milestone. beqforge's F2
device check, optimiser, E10/E1 validation and steep-filter default are
outside it. A beqforge version bump invalidates its bundled optimiser seed
(see the R3 record), so its next release needs a seed rebuild.

| Priority | ID | R | Previous IDs | Status | Depends on |
|---|---|---|---|---|---|
| 1 | IG | | -- | Built 2026-10-28: a run keeps one prepared design waiting per pool slot. Left: measure titles an hour on the next library run after deploying it (the DV8 run made 93) | A release deployed, and a run of a few hundred titles |
| 2 | JE | | [jobs and executions](jobs-and-executions.md) | Design written 2026-10-27, not started; its four defaulted decisions (§ the build's API rename, cancel pauses a one-off, what can be edited, the migration) await the user | The user's answer on the four decisions |
| 3 | RV | | [library run](library-run-gaps.md#l1-finish-the-review-of-the-run-operational) | Operational: all 1,287 designs await a person's review; the DV8 redesign (1,051 titles, 2026-10-26 to 28) changed 728 and left 323 as they were | A person |
| 4 | E2 | R | chunk 31, T2-T4 | Waiting for manual acceptance; everything it exercises is built and published | Real media, a JRiver server and disposable repositories |
| 5 | L1 | | [library run](library-run-gaps.md#l1-finish-the-review-of-the-run-operational) | Declines through design version 6 are in the ledger (stage E, 270); the 16 still declined at version 8 are not | -- |
| 6 | L4 | | [library run](library-run-gaps.md#l4-designer-findings-to-hand-to-beqforge) | v5 and v6 handoffs sent (`/media/beq/triage/handoff-v5.md`, `handoff-v6.md`; beqforge TODO 31, 35); check the 7 findings listed there are each covered, and hand on what is not | L1's version 8 declines |
| 7 | L3 | | [design version](design-version.md), D4 | Built both sides, deployed (designers report version 8). Left: the work-list *out of date* filter and a drift-banner count | -- |
| 8 | WP | | [web parity](web-parity.md) | Built both sides; the designers run 0.3.1 (contract 1.5 and later) since 2026-10-26. Left: try an adjusted redesign in the browser live, then archive this row | -- |
| 9 | L6 | | [library run](library-run-gaps.md#l6-finish-what-the-run-started) | Not started; 78 possible duplicates flagged in the index | E2 |
| 10 | O2 | | former §9 reviewer questions | Behavior exists; acceptance decision missing | Real reviewer feedback |
| 11 | E3 | | chunk 32, T7 | Not started; the captured library (E1) has no DVDs | A DVD rip in a JRiver library, captured as E1 was |
| 12 | E5 | | chunk 37, T15 | Waiting for a product decision | E2; representative season media |
| 13 | T1 | | -- | Watch; probably recurred in CI on 2026-10-08, without a log | Recurrence with a stack dump |
| 14 | T2 | | -- | Watch; seen twice (last 2026-09-27) | Recurrence with its failure message |
| 14a | T3 | | -- | Watch; seen once | Recurrence |
| 14b | T4 | | -- | Watch; seen once | Recurrence with its failure message |
| 15 | D4 | | by-reference §5 | Release part done (each entry records its designer and build); the drift behavior is not started. The deployed designer reports its build as `unknown+src:66850800b23a` | Designer build identity policy (full drift behavior only) |
| 16 | D2 | | by-reference §5 | Planned; optional, not started | beqforge loader coordination |
| 17 | D3 | | by-reference §5 | Planned; optional, not started | Contract addition; beqforge coordination |
| 18 | O1 | | D6 | Optional idea; no implementation decision | Product decision |

E3 is outside the milestone only if E2's sample shows the documented DVD
main-title fallback is truthful for the catalogue's discs; if it silently
duplicates or picks the wrong title, it moves into it. (E4, Blu-ray, is done.)

## Work in priority order

### IG — Designer slots idle between designs

On the design version 8 redesign (2026-10-26 to 28) the service designed about 93 titles an hour with 3 designer
slots, so each title held a slot about 116 s, while the designers' own records give about 75 s of design per title
(beqforge's logs: 16-22 s idle per slot between requests). The gap is the caller's: a slot's next request is prepared
only after the previous result has been written. Prepare the next request while the current one designs, and write a
result after the slot's next request has gone, so each slot stays busy (beqforge's estimate: about 1.25x the
throughput, with nothing changed on the designers).

**Built 2026-10-28:** `run_stages` keeps twice the pool's capacity in hand when the designer is a pool, and the pool
(`pipeline.designer.pool`) still sends no endpoint more than its `parallel`: the extra designs prepare their request and
wait for a slot there. One routed straight to design that is still waiting when the run is cancelled, or its window
closes, gives up (`DesignNotStarted`, asked through `starting_only_while()` outside the pool's lock) and is left in
`not_run` with a `cancelled` event, not recorded as a failure; one extracted in the run is designed as before. Tests:
`test_pipeline_design_lookahead.py`. A side effect: the run's progress shows up to twice as many titles at the design
stage as there are slots, since a waiting one has started its stage.

**Done when** a library run's titles an hour, on a deployed release with this, is compared with the DV8 run's 93.

### JE — Jobs and executions

[jobs-and-executions.md](jobs-and-executions.md) (design, not built): a job is a definition (once or every N minutes,
a window; paused, resumed, edited, removed) apart from its executions (each continuous run: the log), any number of
recurring jobs replacing the one schedule, and statistics for each execution. Four decisions were taken as defaults
and await the user: the breaking API rename (`/v1/executions`), cancelling a one-off's execution pauses the job, what
can be edited, and the migration on upgrade. **Done when** its §7 holds.

### RV — Review the redesigned library (operational)

Every one of the 1,287 designs is at review. The design version 8 redesign (job ids and per-title results in
`/media/beq/triage/dv8-redesign-moved.json`) moved 728 of the 1,051 it targeted; the 323 that did not move, and the
236 left out as unaffected, are as a person last saw them. Nothing is accepted without a person (the work list's
*Accept top pick* asks first). **Done when** each title is accepted, rejected or deliberately left.

### L1, L3, L4, L6 — Gaps left by the first full library run

The run's results and each item's scope and completion criterion are in
[library-run-gaps.md](library-run-gaps.md) (plan, not built). Archive each item there and here when done.

### E2 — End-to-end acceptance record

The runbook is [`e2-runbook.md`](e2-runbook.md) (version 1, separate containers: no compose on
the test host), ending in the record template. Exercise the running app with a real JRiver
source: connection and source setup, browse/ignore rules, work-list actions,
title metadata/artwork/projects/revision, Review Folder, season mode, and
Publish/Commit against disposable real repositories. Run CLI `scan`, `run`,
`publish`, `commit`/`sync` with a real designer, ffmpeg, media and git repos.
Record app/OS/MC versions, commands, exit codes and observed pass/fail results
without private data. Return any discovered defect to a focused code change.

Also, for the initial release, deploy the published image and designer from
R3's compose example and run them, scheduled, over a sample of about 50
titles that covers every codec, channel layout, Blu-ray folder, DVD and TV
season present in the catalogue. Record per-title extract and design times
(R4's measurement and R9's estimate), work-directory growth (R5), stream
choices (J2/W2), disc fallbacks (E3/E4), a designer restart and a media mount
dropped mid-run (R1/R2), and review from the desktop while the run goes (R6:
the guide's layout, Review Folder on the shared queue folder during a run, the
work list with the service idle, and the index read over the chosen share).

Also record from 2026-10-08's work:
- **Re-extraction:** A Star Is Born, Arrietty and Bande A Part need *Revise >
  Re-extract*, since the logo-clip rule did not mark their extractions stale
  (E4). Confirm their extracted audio is the feature's.
- **Cover art:** posters for the 116 titles whose artwork is in MC's cover-art
  folder need a path mapping of `X:\JRiver Cover Art`. Confirm they show.
- **Compression:** the FLAC saving of a published title's multichannel audio
  on real soundtracks (R5).
- **Stream choice:** the titles where JRiver plays another audio stream (J2:
  Pacific Rim, The Godfather...) extract that stream.
- **Auto rescan:** the work list rescans by itself when opened more than 12
  hours after the last scan.

**Done when:** the GUI, season and CLI claims formerly called T2-T4 have
dated observed results, and the container sample run has a dated record with
no unexplained failure; a test stub alone is insufficient.

### E3 — DVD title selection

E1's capture has no DVDs, so first capture a JRiver library with a DVD rip
(`fixtures/jriver/capture.py`) to see how MC identifies its titles. Give Extract Audio a
DVD title picker and use one stable title number, label and duration across
that picker, Batch Extract and JRiver. Unattended runs may retain a documented
main-title default, but several JRiver entries for one disc must not silently
produce several copies of the same main-title audio. Cover IFO parsing,
dialog selection and ffmpeg command construction with fixtures; include a
real navigation run in E2's record.

**Done when:** a person can choose a DVD title and library runs cannot
silently duplicate a disc's main title for distinct episodes.

### E5 — Season fidelity decision

Measure levels and channel layouts in representative episodes during E2.
Decide whether joined mono tracks require level matching and whether a
season-wide multichannel project is needed. If so, write a bounded policy
for gain reference, clipping, fingerprints, migration and project semantics,
then implement it with tests. If not, state the observed limitation in the
user guide and close it as an accepted boundary.

**Done when:** the evidence and decision are recorded; an approved new
behavior is complete only after its implementation and tests land.

### T1 — Intermittent hang in the parallel suite

On 2026-09-26 two of three `pytest -n auto -v src/test/python` runs stopped
at about 99% with one xdist worker idle. Both times, the last test that
worker reported was
`gui/test_worklist_actions.py::test_titles_in_flight_together_share_one_status_and_a_finished_one_moves_before_the_run_ends`.
The worker never reported the next test in the file,
`test_a_test_that_ends_with_a_run_going_is_not_held_up_by_the_close_question`,
which closes the window while a held run is still going. The last test
printed by the run as a whole was an unrelated `gui/test_worklist_review.py`
test that had already passed. Could not reproduce: the file alone
(32 runs, 8 at once) and four further full parallel runs completed.
`faulthandler_timeout` prints nothing from an xdist worker, so if it
recurs, capture stacks with a throwaway plugin that calls
`faulthandler.dump_traceback_later(45, file=<per-worker file>)` around
each test (`-p <plugin>`), and fix what the dump shows. Close this if it
has not recurred after a few weeks of routine runs. **2026-10-08:** `main`
CI run `37799088419`'s `ubuntu-22.04` job hit the 30-minute limit (it
normally takes 6–12 minutes, and the next two runs did). A cancelled job's
log is not kept, so whether it was this hang is unknown.

### T2 — Intermittent ffmpeg-progress test failure

On 2026-09-26 one full `pytest -n auto` run failed
`test_pipeline_library_extract_cache.py::test_extract_if_needed_forwards_ffmpegs_time_progress`;
it had passed in the full runs just before, and the file passed 3 of 3 runs
alone afterwards. It failed once more on 2026-09-27 (during F3's full run);
two re-runs with `--tb=short` to capture it both passed. The failure
message was not captured. The suspected, unconfirmed cause: the test
extracts a 1 s synthetic wav and needs at least one progress report with
`out_time > 0`, which is racy under full parallel load. If it recurs, record the
assertion that failed; if the cause is confirmed, lengthen the source or
accept a run whose only report comes at the end, rather than retrying.

### T3 — Windows GUI wait timed out once

On 2026-10-08 `gui/test_worklist_window.py::test_closing_releases_the_index_and_a_scan_that_finishes_after_it_does_not_open_it_again`
failed on `windows-2025` only (run `37804977461`): `waitUntil timed out in 5000
milliseconds`. It passed on `windows-2022` in the same run and in every run
before. If it recurs, record which wait it was and lengthen it or wait on
the signal instead; close it after a few weeks without recurrence.

### T4 — Review folder's Open project test failed once

On 2026-10-10 one `pytest -n auto` run of the gui suite failed
`gui/test_worklist_review.py::test_open_project_needs_the_main_window_and_is_offered_through_the_callable`; the
failure message was not kept. It passed in the next three full runs and 15 of 15 repeats under `-n 8`. It does not
touch the code changed that day (W1, the decision write). If it recurs, record the assertion and fix what it shows;
close it after a few weeks without recurrence.

### WP — Web app parity

Every chunk in [web-parity.md](web-parity.md) is built on this side (2026-10-21), and beqforge's half of contract
1.5 shipped in 0.2.9; the deployed designers run 0.3.1. **Done when** an adjusted redesign has been made from the
browser against the deployed designers and web-app.md §1 no longer lists title-page work that stays in the desktop
app; then archive this row.

### D2 — Export a reusable request file

Write the by-reference request body beside the extracted audio as `<work_dir>/<id>/design-request.json`, allowing beqforge’s `design_beq.py` to reuse it without extraction. Agree the loader interface and test a round trip; neither `manifest.json` nor the work-directory layout becomes a contract.

**Done when:** the agreed behavior is documented and implemented with regression coverage.

### D3 — Designer records beside the title

Design an optional `record_path`, relative to the shared root and chosen by the caller. The designer writes its run record there, otherwise using `--record-dir`. Specify containment, write/error behavior and compatibility before implementing both sides with contract tests.

**Done when:** the agreed behavior is documented and implemented with regression coverage.

### O2 — Reviewer and project policy confirmation

Confirm three delivered choices with reviewers: `force_design` does not
reset accepted/published entries without an explicit revision; previewing a
different candidate does not rewrite project files until acceptance; and a
project edited only on a freed multichannel slave is outside the current
master-filter edit hash. Also confirm that refusing disagreeing edited
projects gives a usable recovery path. If these are acceptable, document
them as accepted boundaries in user guidance and close O2. If any needs a
different behavior, write its small, testable change separately. This is a
product decision, not evidence of a current implementation defect.

### O1 — Catalogue as input

Decide whether the pipeline should load a published catalogue BEQ and apply
it without extraction, design or metadata lookup. This was outside the
delivered pipeline's scope and has no approved implementation plan. If wanted,
specify the catalogue identity, target signal and output semantics before
building it; otherwise record that it is intentionally out of scope.
