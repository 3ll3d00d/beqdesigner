# Outstanding design work

This is the only design backlog: every item here is unbuilt or unverified at
`HEAD`. Built behavior is in [`implemented.md`](implemented.md) and, for the
service, [`pipeline-service.md`](pipeline-service.md). The library plan's old
chunk and T numbers are retained solely as traceable identifiers. Status as of
2026-09-27. A test using a fake JRiver server does not close an item that
requires evidence from a real one. When an item is done, its lasting behavior
moves to the implemented design and the item is removed from this file.

| ID | Previous IDs | Status | Depends on |
|---|---|---|---|
| E1 | chunk 3/30, T5-T6 | Waiting for authorised JRiver capture | Real MCWS server |
| E2 | chunk 31, T2-T4 | Waiting for manual acceptance | E1; real designer, media and disposable repositories |
| E3 | chunk 32, T7 | Not started; evidence dependent | E1; DVD fixture |
| E4 | chunk 33, T8 | Not started; evidence dependent | E1; Blu-ray playlist fixture |
| E5 | chunk 37, T15 | Waiting for a product decision | E2; representative season media |
| J2 | chunk 40 | Not started beyond the seam: the resolver returns the first audio stream | Sanitised Playback Info and ffprobe evidence |
| W1 | chunk 45a | Partially implemented: retry labels and failure/detail views missing | None |
| W2 | chunk 45b | Partial in `62270b4`: codec, channels and stream count are requested | J2 for automatic stream selection; manual override already exists |
| W3 | pipeline-service §5.1 gap | Not started | None |
| C1 | chunk S5 | amd64 build and smoke verified; arm64 build and GHCR publish not yet run | The first tag push |
| D1 | beqforge R2 | Design agreed with beqforge (`e96129a`), not built; parked behind beqforge's R2a (its own stage cache) | R2a's warm-request timings, or the designer moving host |
| O1 | D6 | Optional idea; no implementation decision | Product decision |
| O2 | former §9 reviewer questions | Behavior exists; acceptance decision missing | Real reviewer feedback |
| T1 | -- | Watch; not reproduced | Recurrence with a stack dump |
| T2 | -- | Watch; seen twice | Recurrence with its failure message |

## External evidence and disc behavior

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

## Selected-stream work

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

## Work list

### W1 — Retry, failures and title-page details

The current attempt appears in the row while it runs; unplanned progress is
rejected, exceptions are logged with tracebacks, and ffmpeg
command-preparation failures emit events. A decline is shown as designer
commentary on its title page. The title page shows no failure text, and the
Failures tab has one Reason column. Complete the visible workflow:

- Label Retry by the failed stage and distinguish the previous indexed
  failure from the active attempt on the row and title page. Keep the old
  failure visible until an index refresh confirms success.
- Show full persisted, multiline failure text in selectable, copyable views
  on the Failures tab and title page. Expose the existing per-title Run
  Details dialog there, including redacted ffmpeg commands and an explicit
  cache-hit message when no command ran.
- Test extract and design retry, another failure, success, cancellation,
  disjoint later runs, multiline copy and redaction. Update `docs/library/work.md`.

**Done when:** those user paths work in real-widget tests and the persisted
failure remains distinct from one-run event history.

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

### W3 — Review Folder honours the lease

The Review Folder window's *Publish accepted* and *Commit published* call
`publish_library`/`commit_library` without checking the work-directory lease,
so they can write the repositories while a service job, a CLI run or another
work list's run holds it. Make them wait or refuse as the work list's Publish
and Commit do ([pipeline-service.md §5.1](pipeline-service.md#51-the-work-directory-lease-and-joining-a-run)),
naming the holder, with a real-widget test over a held lease.

**Done when:** neither button writes while another process's fresh lease is
held, and a stale lease does not block them.

## Delivery

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

## Designer

### D1 — Designer requests by reference

beqforge's R2 asks for requests that name audio on a filesystem both sides
see. The design, and feedback that most of R2's benefit needs no contract
change, is in [`designer-by-reference.md`](designer-by-reference.md) (design,
not built). beqforge agreed it (`e96129a`, that doc's §6) and builds R2a, its
own stage cache, first. D1 is parked until R2a's warm-request timings show the
transfer matters, or the designer moves to another host. It is complete when
D1.1-D1.4 there are built, and the parity test shows a request by reference
decodes byte-identical to the same request inline.

## Test health

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

## Optional idea

### O1 — Catalogue as input

Decide whether the pipeline should load a published catalogue BEQ and apply
it without extraction, design or metadata lookup. This was outside the
delivered pipeline's scope and has no approved implementation plan. If wanted,
specify the catalogue identity, target signal and output semantics before
building it; otherwise record that it is intentionally out of scope.

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
