# TODO — prioritized work

## Status

Reviewed on **2026-10-07**, against beqdesigner `b325752` and beqforge
`934dff8`. This is the **only current backlog** for the
pipeline and library work. Items are unbuilt, partial, awaiting evidence, or
under observation; none is claimed complete. Architecture and contracts are
[design references](README.md); completed plans and delivery records live in
[archive/](archive/README.md). Archived open rows are historical, not extra
backlogs. The caller’s by-reference implementation and live acceptance are
complete; its three follow-ons are retained below as D2–D4.

Order is recommended execution priority: first the **initial release**
milestone below, then disc behavior, test health, and optional
improvements/product decisions. A blocked item does not prevent starting the
next independent item. E1's fixture (`src/test/python/fixtures/jriver/`) is
the evidence J2 and the disc items start from.

When an item is completed, archive its result and evidence, update the lasting
design description if necessary, and remove it here in the same commit.
Do not keep a second status table in a design reference.

### Initial release milestone

**Definition:** the published Docker image, with a documented beqforge
designer beside it, can be deployed and left to extract and design an entire
catalogue unattended, and the results reviewed from the desktop app. Publish
and commit stay a person's decision in the app; the schedule never goes past
design.

The milestone is the rows marked **R** in the table: priorities 1–15 (R1–R9, J2, W2, E1 and D4's release part, now [archived](archive/initial-release.md), were 1–5 and 7–14). The
service and image themselves work (C1's amd64 smoke); what is missing is
behavior that only matters at catalogue scale and over days: the designer
deployment, disk use, remote review, stream
choice, and an acceptance run. Two beqforge items are recorded in its own
`TODO.md`, not here: #1 "Playback contract" (the designer side of R8, which
now sends `run.bass_management`) and
#13 "Residual reporting" (the reported fit error and band do not match the
fit's). Neither blocks the milestone, but R8 and the user guide must state
what they mean for a reviewer. beqforge's F2 device check, optimiser, E10/E1
validation and steep-filter default are outside it.

| Priority | ID | R | Previous IDs | Status | Depends on |
|---|---|---|---|---|---|
| 6 | C1 | R | chunk S5 | amd64 build and smoke verified; arm64 build and GHCR publish not yet run | A green `main`; the first tag push |
| 11 | D4 | | by-reference §5 | Release part done (each entry records its designer and build); the drift behavior is not started | Designer build identity policy (full drift behavior only) |
| 15 | E2 | R | chunk 31, T2-T4 | Waiting for manual acceptance | R1–R9, E1; real designer, media and disposable repositories |
| 16 | E3 | | chunk 32, T7 | Not started; the captured library has no DVDs | A DVD in a JRiver library; DVD fixture |
| 17 | E4 | | chunk 33, T8 | Premise not observed (E1); real playlist forms recorded | Product decision on `.mpls` entries and `BlurayPlaylist` |
| 18 | E5 | | chunk 37, T15 | Waiting for a product decision | E2; representative season media |
| 19 | T1 | | -- | Watch; not reproduced | Recurrence with a stack dump |
| 20 | T2 | | -- | Watch; seen twice | Recurrence with its failure message |
| 20a | T3 | | -- | Watch; seen once | Recurrence |
| 21 | R10 | | -- | Not started | -- |
| 22 | D2 | | by-reference §5 | Planned; optional, not started | beqforge loader coordination |
| 23 | D3 | | by-reference §5 | Planned; optional, not started | Contract addition; beqforge coordination |
| 24 | O2 | | former §9 reviewer questions | Behavior exists; acceptance decision missing | Real reviewer feedback |
| 25 | O1 | | D6 | Optional idea; no implementation decision | Product decision |

E3 and E4 are outside the milestone only if E2's sample shows the documented
main-title fallback is truthful for the catalogue's discs; if it silently
duplicates or picks the wrong playlist, they move into it.

## Work in priority order

### C1 — arm64 image and GHCR publish

On 2026-09-27 the push-CI job's two commands were run locally at `7db992d`
(Docker 29.8.1, linux/x86_64): `docker build -f docker/Dockerfile` succeeded,
including its `dvdvideo` demuxer check, and `docker/smoke.py` passed (ready,
run job through design, `succeeded`, queue entry present). The image runs as
`beq` and reports the `VERSION` file copied into it. What has not run is the
rest of `create-image.yaml`: the `linux/arm64` build under QEMU/buildx, the
`latest`-tag rule, and the push to GHCR. These are left to CI: the first
tag's `create-image.yaml` run exercises all of them (no local QEMU set-up).

The release workflow also smokes the image against the designer the compose
example pins, by reference (R3), before publishing.

**Done when:** the first tag's workflow run builds both architectures, passes
both smoke tests and publishes them to GHCR with the expected tags.

### E2 — End-to-end acceptance record

Write a versioned runbook, then exercise the running app with a real JRiver
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

**Done when:** the GUI, season and CLI claims formerly called T2-T4 have
dated observed results, and the container sample run has a dated record with
no unexplained failure; a test stub alone is insufficient.

### E3 — DVD title selection

Use E1 to determine how JRiver identifies DVD titles. Give Extract Audio a
DVD title picker and use one stable title number, label and duration across
that picker, Batch Extract and JRiver. Unattended runs may retain a documented
main-title default, but several JRiver entries for one disc must not silently
produce several copies of the same main-title audio. Cover IFO parsing,
dialog selection and ffmpeg command construction with fixtures; include a
real navigation run in E2's record.

**Done when:** a person can choose a DVD title and library runs cannot
silently duplicate a disc's main title for distinct episodes.

### E4 — Blu-ray playlist pseudo-paths

**E1 (2026-10-08) did not observe the premise:** the captured library has no
`BDMV\PLAYLIST\index.bluray;N` entries. Its pseudo-paths are all
`BDMV\index.bluray;1`. What it has instead: an entry that is a playlist file
(`...\BDMV\PLAYLIST\00305.mpls`, a short on another film's disc, which
`_disc_root` passes through as a file, so extraction opens the `.mpls`
itself); 13 `Playback Info` records naming a `BlurayPlaylist` (`00801.mpls`) for
an `index.bdmv` entry; and two disc folders listed by more than one entry.
Decide whether a `BlurayPlaylist` should choose the playlist extracted, and
how an `.mpls` entry is extracted, before the steps below. **J2's probe
(2026-10-08) found one disc to look at:** for "A Star Is Born" ffprobe sees
three AC-3 streams in the main title `model.bdmv` resolves to (the same
`00100.mpls` JRiver plays), where JRiver lists DTS-HD MA first. The extraction
may be reading a clip or track that is not the feature's.

Determine from E1 and BDMV files what `BDMV\PLAYLIST\index.bluray;N`
means. If `N` maps to a playlist, preserve it in `LibraryItem` and pass it
to extraction. Otherwise report the unresolved selection and use the
documented main-title fallback. Test known and unknown playlist numbers,
path normalisation and ordinary BDMV roots.

**Done when:** each observed pseudo-path either selects the intended
playlist or shows a truthful fallback.

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
has not recurred after a few weeks of routine runs.

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

### R10 — Configuration-folder hygiene

Two things write `library-profile.yaml` where they should not.
`worklist_settings.py` offers the file in `QStandardPaths`'
`AppConfigLocation`; run from source with no application name set, that is
`~/.config/app.py/`. And a gui test run left
`~/.config/pytest-qt-qapp/library-profile.yaml` in the real configuration
folder, so a test reaches the real location instead of a temp directory. Set
the application and organisation names before the location is read, and make
the gui tests redirect it (for example `QStandardPaths.setTestModeEnabled`
in `gui/conftest.py`).

**Done when:** a test pins the offered folder to the application's name and
the gui suite writes nothing under the real configuration folder.

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
