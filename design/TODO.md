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

The milestone is the rows marked **R** in the table: priorities 1–15 (R1–R4, R7 and E1, now [archived](archive/initial-release.md), were 1–5 and 9). The
service and image themselves work (C1's amd64 smoke); what is missing is
behavior that only matters at catalogue scale and over days: the designer
deployment, disk use, remote review, stream
choice, and an acceptance run. Two beqforge items are recorded in its own
`TODO.md`, not here: #1 "Playback contract" (the designer side of R8) and
#13 "Residual reporting" (the reported fit error and band do not match the
fit's). Neither blocks the milestone, but R8 and the user guide must state
what they mean for a reviewer. beqforge's F2 device check, optimiser, E10/E1
validation and steep-filter default are outside it.

| Priority | ID | R | Previous IDs | Status | Depends on |
|---|---|---|---|---|---|
| 6 | C1 | R | chunk S5 | amd64 build and smoke verified; arm64 build and GHCR publish not yet run | A green `main`; the first tag push |
| 7 | J2 | R | chunk 40 | Not started beyond the seam; Playback Info captured (E1), ffprobe of the same files missing | ffprobe of the fixture's titles (the media share mounted) |
| 8 | W2 | R | chunk 45b | Partial in `62270b4`: codec, channels and stream count are requested | J2 for automatic stream selection; manual override already exists |
| 10 | R8 | R | -- | Not started; the library never sends bass management | beqforge TODO #1 for the designer side |
| 11 | D4 | R | by-reference §5 | Planned; not started. Release needs only the per-entry record | Designer build identity policy (full drift behavior only) |
| 12 | R5 | R | -- | Not started; no retention or free-space check | Product decision on kept audio |
| 13 | R6 | R | -- | Not started; the supported layout is undefined | R3 |
| 14 | R9 | R | -- | Not started | -- |
| 15 | E2 | R | chunk 31, T2-T4 | Waiting for manual acceptance | R1–R9, E1; real designer, media and disposable repositories |
| 16 | E3 | | chunk 32, T7 | Not started; the captured library has no DVDs | A DVD in a JRiver library; DVD fixture |
| 17 | E4 | | chunk 33, T8 | Premise not observed (E1); real playlist forms recorded | Product decision on `.mpls` entries and `BlurayPlaylist` |
| 18 | E5 | | chunk 37, T15 | Waiting for a product decision | E2; representative season media |
| 19 | T1 | | -- | Watch; not reproduced | Recurrence with a stack dump |
| 20 | T2 | | -- | Watch; seen twice | Recurrence with its failure message |
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

### J2 — Resolve JRiver's selected audio stream

**Evidence so far (E1, 2026-10-08):** `src/test/python/fixtures/jriver/`
holds real `Playback Info` values. A `Streams` record's value is three
numbers (`0,1,3`, 205 of 211) or two (`0,1`), in records whose other names
vary and come in any order (see the fixture README). The capture machine did
not have the media share mounted, so the ffprobe half is missing: ffprobe the
fixture's titles (`ffprobe -show_streams -of json`), record their global
indices beside the rows, and only then decide what the numbers mean.

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

### R8 — Send bass management to the designer

The contract carries `bass_management`, and `design_if_needed` accepts it,
but the library's `_design` (`pipeline/library/run.py`) never passes it and
the profile has no place to declare it; Batch Design's `DesignJob` omits it
too. The designer's clipping and headroom advice is therefore made on an
assumed playback chain. Add a profile setting (crossover, LFE gain,
headroom type) and pass it on both paths. beqforge's own TODO #1 records that
it consumes only the crossover today; until that lands, the title page and
user guide must say which fields were modelled and which were assumed.

**Done when:** a request from the library and from Batch Design carries the
profile's bass management (tests over the request body), and the reviewer can
see whether the clipping figures used it.

### D4 — Surface designer revision drift

The design fingerprint does not include the designer build or startup
parameters, so a catalogue run lasting days can mix designer builds without
saying so. **For the release:** record the response's `beqforge_revision` (or
the designer's reported build) in each queue entry and show it in the title
page's Details; that is enough to tell which titles came from which build.
**After it:** define how the caller detects and presents a changed revision,
decide cache and redesign semantics, and test the changed-revision banner and
unchanged-revision behavior.

**Done when (release):** every newly designed entry records its designer
build, with a test. **Done when (full):** the agreed drift behavior is
documented and implemented with regression coverage.

### R5 — Work-directory size and retention

With `keep_multichannel: true` a title's work folder is 150–250 MB (observed
on 17 real titles: `multichannel.wav` is about 165 MB for a 2-hour 7.1 title,
`mono.wav` about 20 MB). A catalogue of 1,000 titles is about 200 GB, and
nothing prunes or compresses it. Decide what is kept after design, after
acceptance and after publish (for example, drop or compress `multichannel.wav`
once a title is published, keeping what revise and the multichannel project
need, or replan extraction on demand). Check free space before each
extraction and stop the run with a clear reason below a configurable floor,
rather than failing titles one by one (raise `pipeline.library.failure.Unavailable`,
so it is not remembered and counts towards R1's stop).

**Done when:** the retention policy is documented and implemented with tests,
and a run stops cleanly when the floor is reached.

### R6 — Reviewing the container's work from the desktop

The service guide says the work directory is exported to the reviewer, but
the supported layout is not defined. The profile's paths are container paths,
so the desktop needs its own profile naming the same work and queue folders
by its paths, and nothing pairs the two. Queue entries store absolute media
paths (for example `art_path: /media/films/...`). The SQLite index (default
journal mode) would be read over SMB or NFS while the container writes it, and
the work-directory lease is not atomic across machines. Choose one supported
layout, make queue entries free of host-specific paths (or translate them),
document the desktop profile and the mount, and verify concurrent reading of
the index across the chosen network filesystem.

**Done when:** the layout is in the user guide, queue entries open on the
desktop from a container-written work directory (tested with differing roots),
and E2 records the cross-machine review working while a run goes.

### R9 — Operating an unattended catalogue run

- **TMDB:** without `TMDB_API_KEY` the service designs with library metadata
  only and says nothing; the desktop has a built-in key, so this is
  surprising. Report it in `/ready` and `/v1/status`, and in the guide.
- **Progress:** at about two minutes or more per title, a catalogue takes
  days. Report the run's throughput and an estimate of the time remaining in
  `/v1/status` and the job's progress.
- **First run:** document how to seed a large catalogue (filtered batches by
  year or source, `scan` first, checking the first results before enabling
  the schedule) and what a restart mid-run does.

**Done when:** status and readiness report both, with tests, and the guide
has the first-run procedure.

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
dropped mid-run (R1/R2), and review from the desktop while the run goes (R6).

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
how an `.mpls` entry is extracted, before the steps below.

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
