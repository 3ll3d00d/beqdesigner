# Outstanding design work

This is the only design backlog. Completed behavior is in
[`implemented.md`](implemented.md); the library plan's old chunk and T numbers
are retained here solely as traceable identifiers. Status reflects the
2026-09-25 design sweep. A test using a fake JRiver server does not close an
item that requires evidence from a real one.

| ID | Previous IDs | Status | Depends on |
|---|---|---|---|
| E1 | chunk 3/30, T5-T6 | Waiting for authorised JRiver capture | Real MCWS server |
| E2 | chunk 31, T2-T4 | Waiting for manual acceptance | E1; real designer, media and disposable repositories |
| E3 | chunk 32, T7 | Not started; evidence dependent | E1; DVD fixture |
| E4 | chunk 33, T8 | Not started; evidence dependent | E1; Blu-ray playlist fixture |
| E5 | chunk 37, T15 | Waiting for a product decision | E2; representative season media |
| J2 | chunk 40 | In progress; resolver is a first-stream stub | Sanitised Playback Info and ffprobe evidence |
| W1 | chunk 45a | Partially implemented | None |
| W2 | chunk 45b | Partial in `62270b4`: codec, channels and stream count are requested | J2 for automatic stream selection; manual override already exists |
| O1 | D6 | Optional idea; no implementation decision | Product decision |
| O2 | former §9 reviewer questions | Behavior exists; acceptance decision missing | Real reviewer feedback |
| S0-S7 | -- | S0-S2 done; S3-S7 not started; design in [`pipeline-service.md`](pipeline-service.md) | None (S3 next) |

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

## Work-list feedback

### W1 — Retry, failures and title-page details

The current attempt already appears in the row while it runs; unplanned
progress is rejected, exceptions are logged with tracebacks, and ffmpeg
command-preparation failures emit events. Complete the visible workflow:

- Label Retry by the failed stage and distinguish the previous indexed
  failure from the active attempt on the row and title page. Keep the old
  failure visible until an index refresh confirms success.
- Show full persisted, multiline failure text in selectable, copyable views
  on the Failures tab and title page. Expose the existing per-title Run
  Details dialog there, including redacted ffmpeg commands and an explicit
  cache-hit message when no command ran.
- Show a decline's reason and message once as designer commentary, without
  repeating the index detail in the pending state text.
- Test extract and design retry, another failure, success, cancellation,
  disjoint later runs, multiline copy, redaction and a decline without
  candidates. Update `docs/library/work.md`.

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
