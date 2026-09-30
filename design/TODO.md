# TODO — prioritized work

## Status

Reviewed on **2026-09-30**. This is the **only current backlog** for the
pipeline and library work. Items are unbuilt, partial, awaiting evidence, or
under observation; none is claimed complete. Architecture and contracts are
[design references](README.md); completed plans and delivery records live in
[archive/](archive/README.md). Archived open rows are historical, not extra
backlogs. The caller’s by-reference implementation and live acceptance are
complete; its three follow-ons are retained below as D2–D4.

Order is recommended execution priority: stream correctness, external
acceptance and disc behavior, release
verification, test health, and optional improvements/product decisions. A
blocked item does not prevent starting the next independent item. E1 supplies
evidence needed by J2 and the disc items and may be captured alongside them.

When an item is completed, archive its result and evidence, update the lasting
design description if necessary, and remove it here in the same commit.
Do not keep a second status table in a design reference.

| Priority | ID | Previous IDs | Status | Depends on |
|---|---|---|---|---|
| 1 | J2 | chunk 40 | Not started beyond the seam: the resolver returns the first audio stream | Sanitised Playback Info and ffprobe evidence |
| 2 | W2 | chunk 45b | Partial in `62270b4`: codec, channels and stream count are requested | J2 for automatic stream selection; manual override already exists |
| 3 | E1 | chunk 3/30, T5-T6 | Waiting for authorised JRiver capture | Real MCWS server |
| 4 | E2 | chunk 31, T2-T4 | Waiting for manual acceptance | E1; real designer, media and disposable repositories |
| 5 | E3 | chunk 32, T7 | Not started; evidence dependent | E1; DVD fixture |
| 6 | E4 | chunk 33, T8 | Not started; evidence dependent | E1; Blu-ray playlist fixture |
| 7 | E5 | chunk 37, T15 | Waiting for a product decision | E2; representative season media |
| 8 | C1 | chunk S5 | amd64 build and smoke verified; arm64 build and GHCR publish not yet run | The first tag push |
| 9 | T1 | -- | Watch; not reproduced | Recurrence with a stack dump |
| 10 | T2 | -- | Watch; seen twice | Recurrence with its failure message |
| 11 | D4 | by-reference §5 | Planned; not started | Designer build identity policy |
| 12 | D2 | by-reference §5 | Planned; optional, not started | beqforge loader coordination |
| 13 | D3 | by-reference §5 | Planned; optional, not started | Contract addition; beqforge coordination |
| 14 | O2 | former §9 reviewer questions | Behavior exists; acceptance decision missing | Real reviewer feedback |
| 15 | O1 | D6 | Optional idea; no implementation decision | Product decision |

## Work in priority order

### J2 — Resolve JRiver's selected audio stream

`Playback Info` is requested but its resolver currently returns the first
audio stream. Capture its length-prefixed value alongside ffprobe streams
from the same file; the observed comma-separated fields cannot be assumed
to be audio-list ordinals. Parse the selected container stream and match it
to ffprobe's global `streams[].index`, then select the matching audio ordinal.
Keep the first-audio fallback only for absent or unparseable data and report
that fallback. Cover malformed, missing and non-audio selections.

**Done when:** fixture-backed tests show that JRiver's selected track is the
one extracted. The title page's manual override and metadata update are
already built; J2 supplies their automatic initial choice.

### W2 — Stream evidence and truthful stages

JRiver now requests per-stream codec, channel count and stream count, using
the first selected codec to fill automatic audio metadata. A rescan fills an
existing queue entry when its audio type is absent or still the prior automatic
value; a different reviewer value is kept. Still request
per-stream sample rate, bitrate, language and title fields and normalize
them into readable audio-list choices, retaining ordinals. If a source does
not supply a useful list, ffprobe only that title's
mapped local source on a worker; explain a missing/unplayable file with the
title and path. Show the chosen stream, actual channel count and Keep
multichannel setting before work and in Details. A true design-only cache hit
should enter design without an “Extracting” event or extract worker slot;
stale/missing audio must explicitly replan extraction. Keep selection and
cache invalidation in the existing index/queue path.

Use a short synthetic multistream/multichannel fixture to verify the chosen
stream's mono mix and diagnostic arrays share the analysis rate and frame
count; also cover mono-only behavior and the stage wording. The existing
single-stream multichannel regression does not establish this choice path.

**Done when:** a person can see and select the correct stream without an
ineffective rescan, and the run shows only the stages actually performed.

### E1 — JRiver response fixture

Capture, with authorisation, a selected node's `Browse/Files`, its
`Browse/Children` tree and `/Alive`, including configured external-ID and
artwork fields. Include two same-named child nodes and an `INTERNAL` artwork
case if present. Remove credentials, hosts, personal paths and unrelated
metadata. Use the fixture to verify field aliases, `FriendlyName`, stable
item identity, and the browse picker; change the picker so duplicate names
retain distinct node IDs. The current implementation has fake-server tests
and partial live observations, not a sanitised reusable fixture.

**Done when:** the fixture and adjacent tests prove the MCWS shape and the
duplicate-node behavior, with a dated account of capture and sanitisation.

### E2 — End-to-end acceptance record

Write a versioned runbook, then exercise the running app with a real JRiver
source: connection and source setup, browse/ignore rules, work-list actions,
title metadata/artwork/projects/revision, Review Folder, season mode, and
Publish/Commit against disposable real repositories. Run CLI `scan`, `run`,
`publish`, `commit`/`sync` with a real designer, ffmpeg, media and git repos.
Record app/OS/MC versions, commands, exit codes and observed pass/fail results
without private data. Return any discovered defect to a focused code change.

**Done when:** the GUI, season and CLI claims formerly called T2-T4 have
dated observed results; a test stub alone is insufficient.

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

### C1 — arm64 image and GHCR publish

On 2026-09-27 the push-CI job's two commands were run locally at `7db992d`
(Docker 29.8.1, linux/x86_64): `docker build -f docker/Dockerfile` succeeded,
including its `dvdvideo` demuxer check, and `docker/smoke.py` passed (ready,
run job through design, `succeeded`, queue entry present). The image runs as
`beq` and reports the `VERSION` file copied into it. What has not run is the
rest of `create-image.yaml`: the `linux/arm64` build under QEMU/buildx, the
`latest`-tag rule, and the push to GHCR. These are left to CI: the first
tag's `create-image.yaml` run exercises all of them (no local QEMU set-up).

**Done when:** the first tag's workflow run builds both architectures, passes
the smoke test and publishes them to GHCR with the expected tags.

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

### D4 — Surface designer revision drift

The design fingerprint does not include the designer build or startup parameters. Define how the caller detects and presents a changed designer revision, using the response’s existing `beqforge_revision` commentary where appropriate. Decide cache and redesign semantics, then test the changed-revision banner and unchanged-revision behavior.

**Done when:** the agreed behavior is documented and implemented with regression coverage.

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

