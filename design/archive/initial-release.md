# Initial release milestone — completed items

Completed items from the initial release milestone in [TODO](../TODO.md),
in the order they were done. Each records its completion date, implementation commit and the
validation run.

## R7 — Windows CI failure in the publish aggregate test (completed)

Completed on 2026-10-08, in the commit "Read the publish aggregate test's
JSON as UTF-8 on every platform". Since `b325752`,
`test_pipeline_publish_readable.py::test_titles_in_several_letter_folders_share_one_aggregate_in_their_category_folder`
failed on `windows-2022` and `windows-2025` (CI run `36975470653`, its only
failure). The publisher was not at fault: `catalogue_json.aggregate()` encodes
`database.json` as UTF-8 explicitly and `git.write_files` writes bytes. The
test opened the file without `encoding=`, so Windows decoded it as cp1252 and
read `Élite` as `Ã‰lite`. The CI log showed the two the other way round
(`Élite` against `�lite`) because the cp1252 console output was displayed as
UTF-8. Every text `open()` in that test file now names `encoding='utf-8'`.
Windows desktop publishing is unaffected.

Validation: `PYTHONPATH=./src/main/python uv run pytest -q
src/test/python/test_pipeline_publish_readable.py`: **19 passed** (Linux). The
Windows result: `main` CI run `37807101885` (`29965c2`, 2026-10-08) passed on
all of Linux, macOS and Windows. The two runs before it found the Windows and
empty-`/media` problems fixed in `56ea53d` and `29965c2`. Turning `EncodingWarning` into an
error was not usable as a suite-wide guard: third-party imports raise it at
collection.

## R1 — Transient failures are not remembered (completed)

Completed on 2026-10-08, in the commit "Report unavailable dependencies
without remembering them as title failures". Before, `_remember_failure`
recorded every exception, so a designer outage, a dropped mount or a JRiver
that did not answer failed every title in a tick for good.

- `pipeline/library/failure.py` `unavailable_reason(error, source_path)`
  walks the error's cause chain. A dependency is unavailable on
  `requests.Timeout`/`ConnectionError`, an HTTP 5xx, a builtin
  `TimeoutError`/`ConnectionError`, a network or remote-filesystem errno
  (`ENOTCONN`, `ESTALE`, `EHOSTDOWN`, ...), `Unavailable`, or the new
  `http_binding.DesignerUnavailable` (`/health` unanswered, or no
  by-reference support). An HTTP 4xx in the chain is the title's own failure.
  For an extract failure, `missing_mount()` also calls it unavailable when the
  nearest existing folder is empty (a mount point with nothing mounted) or the
  path's root is absent (a disconnected drive or share). A missing file in a
  populated folder, or a POSIX path with only `/` above it, stays the title's
  failure.
- `run_unit`/`design_unit_work` share `_report_failure`, which puts an
  unavailable title in `LibraryRunReport.unavailable` and remembers nothing.
  A season episode that meets one ends the season the same way, without
  remembering the episode.
- `run_stages` and `run_library` count unavailable titles in a row
  (`UnavailableStreak`). A title that reached its dependencies, done or
  failed on its own account, resets the count. At
  `run.stop_after_unavailable` (default 3) the run stops like a cancel:
  titles in hand finish, nothing new starts, extracted titles waiting for
  design are not designed, publish/commit do not run, `StagesReport.stopped`
  says why, and the rest are in `not_run`. `cancelled` stays false.
- `StagesReport.failed` (the CLI exit status, the service job state) includes
  `unavailable` and `stopped`. The CLI warns about both, the service's
  `RunResult` carries them (OpenAPI regenerated), notifications list them,
  and the work list shows *Unavailable* and *Not run* lines with the reason.

Tests: `test_pipeline_library_failure.py` (the classification);
`test_pipeline_library_stages.py` (a designer timeout, connection error and
503, a JRiver connection error and a missing mount are not remembered and the
next unattended tick runs them; a 4xx is remembered and not retried; the stop
leaves the rest in `not_run` and the next tick runs them all; a title failure
ends a streak; config validation); `test_pipeline_library_run.py`
(`run_library`'s stop, an unavailable mount inside a season); the CLI golden,
setup and work-list config parsing, the work-list results wording and
notifications.

Validation: `PYTHONPATH=./src/main/python QT_QPA_PLATFORM=offscreen uv run
pytest -q -n auto src/test/python`: **2616 passed, 1 skipped** (the skip is the
Windows-only drive-letter test).

## R2 — Check the designer before a run (completed)

Completed on 2026-10-08, in the commit "Skip scheduled runs while the
designer does not answer, and report it". Before, `/ready` reported the
designer as ready once it was declared, and nothing asked whether it was up.

- `http_binding.check_designer(url, by_reference=)` asks `/health`. A
  `by_reference` designer must answer contract 1.2 with `shared_root`, as
  before. Any other designer counts as up for any answer, because only 1.2
  designers must serve `/health` (a 1.0 designer's 404, or the smoke stub's
  501, is up), and as down only for no answer or 502/503/504. Failures raise
  `DesignerUnavailable` (R1's marker).
- `setup.designer_endpoint()` resolves the run's designer exactly as
  `register_designers` does; `JobContext.designer_unavailable()` asks it.
  `pipeline/service/designer.py` `DesignerProbe` caches the answer for 30 s
  for `/ready` and `/v1/status`.
- `AutoScheduler(designer=, on_designer_down=)` asks before a tick through
  design, without holding its lock. A refused tick records
  `last_skip: designer unavailable: ...`, is retried after
  `min(interval_minutes, 5)` minutes, and `on_designer_down` fires once per
  outage. `Notifier.designer_unavailable` sends a `failed` notification with
  `job: null` (`Notification.job` is now optional) to targets that take
  scheduled jobs.
- A run job through design is refused before anything runs. Through publish
  or commit it is not, and its designs go to R1's `unavailable`.
- `/ready` adds `designer_reachable` with the new `Check.required: false`,
  so it never makes the service unready (the image's HEALTHCHECK uses
  `/health` anyway). `/v1/status` has `designer: DesignerStatus`. OpenAPI
  regenerated.

Tests: `test_pipeline_service_designer.py` (a real stub designer: what counts
as up and down, by-reference, the profile's designer is the one asked, the
manual designer, the cache; ticks skipped and notified once, resumed when it
is back, a new outage notified again, extract-only ticks not asking; a design
run job refused; `/ready` and `/v1/status`; the notification). The existing
service tests make their placeholder designer answer through an autouse
fixture, and the readiness expectations gained `required` and
`designer_reachable`.

Validation: `PYTHONPATH=./src/main/python QT_QPA_PLATFORM=offscreen uv run
pytest -q -n auto src/test/python`: **2635 passed, 1 skipped**.

## R4 — Designer timeouts against design parallelism (completed)

Completed on 2026-10-08, in the commit "Give a queued design request time
to wait behind the others". beqforge's server answers one request at a time,
so with `run.parallelism.design` above 1 a request waits behind the others.
The fixed 300 s timeout could then expire while the designer was still busy,
and before R1 the title was then remembered as failed.

Of the plan's two policies, this takes the queue-depth one:
`register_declared_designers(queue_depth=)` multiplies each declared
designer's `timeout` (default 300 s, one design) by
`run.parallelism.design`. Both registration paths pass it: `setup` for the
CLI and service, and the work list's `register_profile_designers`. The
default stays one design at a time, and the guides say more gains nothing
against one designer. Preferences designers (`http:NAME`) are registered at
app start-up, before any profile is read, and keep 300 s; the setup guide
says to declare the designer in the profile if you raise design parallelism.
A timeout that happens anyway is R1's `unavailable` ("timed out"), not a
design failure.

Tests: `test_pipeline_designer_queue_timeout.py` uses a single-threaded stub
taking 0.5 s per design, a declared timeout of 0.8 s and two requests at
once. With `design: 2` both answer, through the CLI/service registration and
through the work list's. With `design: 1` the queued one times out and is
classified as unavailable. 18 concurrent repetitions all passed.

Validation: `PYTHONPATH=./src/main/python QT_QPA_PLATFORM=offscreen uv run
pytest -q -n auto src/test/python`: **2642 passed, 1 skipped**. The timings
on the longest real titles remain to be recorded by E2.

## R3 — Deploy the designer beside the service (completed)

Completed on 2026-10-08. Built in beqforge `aaf0f21` and `62be0c0` (pushed) and beqdesigner's
"Run the designer beside the pipeline service in the compose example".

- beqforge: `packaging/designer/Dockerfile` runs `beqforge serve-designer` on
  `0.0.0.0:8420` as uid 1000, with `BEQFORGE_SHARED_ROOT=/work` and
  `BEQFORGE_CACHE_DIR=/cache`, and a `/health` HEALTHCHECK.
  `build-designer-image.yml` builds it and runs `packaging/designer/smoke.py`
  (health, one design inline, one by reference) on every change. On a
  `vX.Y.Z` tag that matches `beq_common.__version__`, it publishes
  linux/amd64 and linux/arm64 as `ghcr.io/3ll3d00d/beqforge-designer:<version>`
  (and `latest`). A manual run with `publish` publishes the current version's
  image from a later commit, and refuses unless the designer code is identical
  to that version's tag. README section "Running the designer in a container".
- beqdesigner: `docker/compose.example.yaml` runs `designer` beside
  `pipeline`, pinned to `beqforge-designer:0.2.0`, with the same `./work`
  mounted at `/work` in both, a `designer-cache` volume and the same user.
  `docker/smoke.py --designer-image IMAGE` runs the real designer on a
  private network, waits for `/v1/status` to report it reachable, designs a
  30 s six-channel title, and checks from the designer's own request log that
  the audio arrived by reference (`body 0.0 MB`). Push CI builds the designer
  from beqforge `main` for this; the release workflow pulls the pinned image.
  `docs/library/service.md` "The designer" covers the profile change
  (`designers:` with `by_reference: true`, not a Preferences `http:NAME`
  designer, the same name on the desktop) and design parallelism.

Evidence, 2026-10-08 (Docker 29.8.2, linux/x86_64): both images built from
the working trees. The designer smoke passed both requests. The pipeline
smoke passed with the stub, and with `--designer-image` (job succeeded, a
queue entry, designer log `body 0.0 MB`). `docker compose up` of the example,
with local builds tagged as its images, started both services; `/v1/status`
reported the designer reachable at `http://designer:8420/design`. With the
designer stopped it reported `reachable: false` with the reason, and `/ready`
stayed 200.

**Image release** (decided 2026-10-08): the designer code is unchanged since
beqforge `v0.2.0`, so its image was published as `0.2.0` by a manual run of
the image workflow (run `37770513918`, linux/amd64 and linux/arm64), which
checks that the code matches the tag. A version bump would have invalidated
beqforge's bundled optimiser seed: its cache identity includes
`beq_common.__version__`, and the uses are in `core.py`, which is itself
hashed. Decoupling that is a beqforge follow-up for the next optimiser change
that needs a seed rebuild anyway. The container's designer reports its build
as `unknown+src:<digest>` (no git in the image); D4 should decide whether to
bake a stamp.

After publishing, `docker/smoke.py --designer-image
ghcr.io/3ll3d00d/beqforge-designer:0.2.0` (the release workflow's step, the
image pulled from GHCR) passed by reference. The first beqdesigner tag runs
that step in CI (C1).

## E1 — JRiver response fixture (completed)

Completed on 2026-10-08, in the commit "Replay a real JRiver server's
responses, and keep same-named browse nodes apart". With the owner's
authorisation, `src/test/python/fixtures/jriver/capture.py` captured
`/Alive`, `Library/Fields`, `Browse/Children` (64 nodes, three levels) and
`Browse/Files` of the profile's browse node (1,291 rows) from JRiver MC
36.0.38. It then wrote a sanitised fixture: the server's name, access key and
GUID replaced; only the browse path's nodes; 132 rows covering 87 shapes;
`Description` dropped; a display's hardware id scrubbed. The dated account of
the capture and sanitisation, and everything the capture showed, is the
fixture's `README.md`.

`test_pipeline_library_jriver_fixture.py` replays it through a fake MCWS for
the real `JRiverLibrarySource`, `list_browse_children` and the app's
`MediaServer`. It covers: one request with every field; `Key` as a stable,
server-scoped id; `Year` answered as `Date (year)`; ids from the configured
fields and absent when unset; `w:\` and `W:\` both mapped; disc entries as
their folders; artwork beside the media; every listed audio stream a choice;
the browse tree in order; and `FriendlyName`.

The picker fix: `list_browse_children` read `Browse/Children` through hamcws,
which keys entries by name, so two nodes with the same name kept one id. It
now parses the XML in order (`_parse_children`). No same-named siblings exist
in the captured library, so the test repeats a real entry under another id;
on the old code it returns only one of them.

What the capture showed that moves other items: J2 has its `Playback Info`
evidence but not the ffprobe half (the media was not mounted); E3 has no DVD
in this library; E4's `PLAYLIST\index.bluray;N` premise was not observed, but
`.mpls` entries and `BlurayPlaylist` records were (both recorded in TODO).

Validation: the JRiver suites (`test_pipeline_library_jriver_fixture.py`,
`test_pipeline_library_jriver.py`, `test_jriver_mcws_friendly_name.py`,
`gui/test_browse_node_picker.py`): **95 passed**.

## W2 — Stream evidence and truthful stages (completed)

Completed on 2026-10-08 over four commits: "List each audio stream with its
rate, bitrate, language and title" (`9260b48`), "Design a title whose audio
is current without an extract stage" (`a01f5bb`), "Read a title's audio
streams from its file when the source lists none" (`b4af2e5`) and "Say which
stream a run extracts and what it found".

- **Streams described:** JRiver is asked for `Audio Sample Rate`, `Audio
  Bitrate` and `Audio Title` as well as codec, channels and language. They are
  kept per stream (`_STREAM_FIELDS`), and `pipeline/library/streams.py`
  `describe_stream()` says each one ("2: AC-3 5.1, French, 48 kHz, 640 kbps")
  in the title page's choice. `audio_types()` moved there from `jriver.py`, to
  be shared with ffprobe's streams.
- **No list from the source:** the choice ffprobes that title's mapped file on
  the thread pool (`Session.probe_audio_streams()`, which shares extraction's
  BD/DVD resolution), records the list (`LibraryIndex.set_audio_stream_details()`)
  and then asks. A missing or unreadable file is explained with the title and
  path.
- **Truthful stages:** a title that only needs design, with current audio
  (`run.cached_unit_work()`, the manifest and wavs only), goes straight to
  design: no extract event and no extract worker. A cancel drops it if it has
  not started. Audio gone or stale since the scan is extracted again with
  "Extracting again: <why>".
- **Said before and during work:** the extract stage starts with "Extracting
  audio stream 2: DTS-HD MA 5.1, English; multichannel kept" and ends with
  the channels found ("Extraction complete: 6 channels (5.1(side))"), both in
  Run Details. The multi-title confirmation says whether multichannel audio is
  kept.

Tests: `test_pipeline_library_streams.py` covers the wording, ffprobe's
streams, and a two-stream file (stereo at 100 Hz, then 5.1 at 40 Hz) probed
and extracted as stream 2, with six channels and 40 Hz in the mono mix, both
at the analysis rate with one frame count (the parity contract), and its
events. `test_pipeline_library_jriver_fixture.py` checks the per-stream
fields of the real capture. `test_pipeline_library_stages.py` covers the
design-only route, extraction again and the cancel.
`gui/test_worklist_stream_choice.py` covers the chooser's list, the probe and
a missing file. `test_pipeline_library_index.py` covers the recorded list.

Validation: `PYTHONPATH=./src/main/python QT_QPA_PLATFORM=offscreen uv run
pytest -q -n auto src/test/python`: **2676 passed, 1 skipped**. The automatic
initial choice is J2's.

## R8 — Send bass management to the designer (completed)

Completed on 2026-10-08, in the commit "Send the playback chain to the
designer and show the reviewer what was sent". The contract carried
`bass_management`, but neither the library nor Batch Design sent it.

- `pipeline/library/bass.py` validates a profile's `run.bass_management`
  against the contract's keys (`lpf_fs`, `lpf_position`, `headroom_type`,
  `clip_before`, `clip_after`), each optional with the contract's default; an
  unknown key or bad value is named. Leaving the section out sends `None`. The
  contract has no LFE gain, so the plan's "LFE gain" is not a setting.
- Library: `LibraryRunConfig.bass_management`, from `setup` (CLI and service)
  and the work list's `build_run_config`, reaches `design_if_needed`. Batch
  Design and Extract Audio's design step (`model/batch.py` `DesignJob`) send
  `from_preferences()`: the crossover and position the session was built with
  from Preferences, other keys defaulted.
- `QueueEntry.bass_management` records what was sent. The title page's
  commentary ends with "Bass management sent to the designer" ("none sent:
  the designer assumed its own playback chain" when absent). The review guide
  says beqforge 0.2.0 takes only the crossover and assumes the rest (its TODO
  #1), so its clipping figures are an estimate.
- Not in the design fingerprint: changing it does not mark designs stale, and
  the setup guide says to use *Revise...*. Fingerprinting it would need
  `ScanSettings`, the scan and the drift banner to carry it; that is left
  until beqforge consumes more than the crossover.

Tests: `test_pipeline_library_bass.py` (validation, wording, and a real
`run_library` design whose designer records the `DesignRequest`: the
profile's values sent and recorded in the entry, or `None`);
`gui/test_batch_design_bass.py` (`DesignJob` sends the session's crossover);
`gui/test_worklist_title.py` (the commentary's last line).

Validation: `PYTHONPATH=./src/main/python QT_QPA_PLATFORM=offscreen uv run
pytest -q -n auto src/test/python`: **2691 passed, 1 skipped**.

## D4, release part — record the designer build (completed)

Completed on 2026-10-08, in the commit "Record which designer and build
designed each entry". `QueueEntry` gains `designer` (the registered name) and
`designer_build`. The contract has no build field, so
`review.designer_build()` reads what the designer said: a build key in a
design's commentary (`beqforge_revision`, or a generic `designer_revision`,
`revision` or `build`), else the bracketed provenance closing a decline
message. `design_and_queue` records both, and the title page's commentary
ends with "Designed by: <designer>, <build>" (or "did not say which build").
Older entries show nothing. The drift behavior (detecting a changed build,
cache and redesign semantics) stays in TODO as D4.

Tests: `test_pipeline_review_designer_build.py` (each source of the build,
and a real `design_and_queue` recording both) and `gui/test_worklist_title.py`
(the wording). Validation: full suite **2697 passed, 1 skipped**.

## R5 — Work-directory size and retention (completed)

Completed on 2026-10-08, in the commit "Compress a published title's
multichannel audio and stop a run before the disk fills". The policy, decided
by the maintainer that day: after publish, compress `multichannel.wav`
losslessly to FLAC in place; keep everything else.

- `pipeline/library/retention.py`: `compress_multichannel()` streams the
  24-bit wav into `multichannel.flac` in blocks, checks the frame count,
  notes `multichannel_compressed` in the manifest and removes the wav. A float
  wav is left alone. `restore_multichannel()` reverses it exactly.
  `publish_library` compresses every title it published
  (`compress_published()`, which logs and never fails the publish).
- Restored before every read: `_run_item` (extraction with Keep
  multichannel), `_design_work` (a design from kept audio, including a
  designer reading WAVs by reference) and publish's project writing (a
  republish). `extract_status` counts the compressed file as the current
  extraction, and `project_paths` as a kept one, so a compressed title is
  neither extracted again nor seen as needing republishing.
- `run.min_free_gb` (default 10, 0 off): below it, extraction raises
  `OutOfSpace`, an R1 `Unavailable` recorded in `LibraryRunReport.halt`, which
  stops `run_stages` and `run_library` at once, saying why, with the rest in
  `not_run` and nothing remembered.

Tests: `test_pipeline_library_retention.py` covers a sample-exact round trip,
the float wav, the cache staying current, a real publish through `run_stages`
compressing the title without it needing publishing again, a failed
compression not failing the publish, a design restoring first, the floor,
the immediate stop and the validation. Validation: full suite
**2707 passed, 1 skipped**. The saving on real soundtracks is for E2 to
record.

## R6 — Reviewing the container's work from the desktop (completed)

Completed on 2026-10-08, in the commit "Let a desktop review a work
directory the container wrote, and say how". Queue entries recorded absolute
artwork paths (`/media/films/...`) that meant nothing on the desktop, and no
layout was defined.

- `artwork.resolve_art` copies library artwork into the title's own folder as
  `poster<ext>`, as it already did for a TMDB download. Every entry's poster is
  then inside the shared work directory.
- `review.entry_art_path()` / `resolve_art_path()`: the recorded path if it
  exists here, else the same file name in the title's folder under this
  machine's work directory. They are used by the publish digest, the report
  image, the status cache key and the Metadata tab (`MetadataPanel(work_dir=)`,
  from the title page's hooks). A title designed under one root and reviewed
  under another therefore shows its poster and keeps its publish digest.
  Project files embed their audio and need nothing.
- `docs/library/service.md`, "Reviewing from another machine": the supported
  layout. The work and queue folders are on the container host's own disk,
  shared to the desktop. The desktop has its own profile with the same sources,
  designer name and settings but its own paths. During a run, use Review
  Folder (the queue only). Use the work list, which writes the SQLite index,
  only when the service is idle, and start runs through the service.

Tests: `test_pipeline_review_portable.py` (art copied in; a work directory
copied to a new root with the old one deleted keeps its digest and finds its
poster); `gui/test_worklist_portable_art.py` (the Metadata tab shows it);
`test_pipeline_library_artwork.py` (the changed behavior). Validation: full
suite **2711 passed, 1 skipped**. The cross-machine run (an index read over the
share while the container writes, Review Folder during a run) is E2's to
record.

## R9 — Operating an unattended catalogue run (completed)

Completed on 2026-10-08, in the commit "Say whether TMDB is looked up, how
fast a run goes and how to start a catalogue".

- **TMDB:** `/ready` reports a `tmdb` check (`required: false`, so it never
  makes the service unready) whose detail says titles are designed with
  library metadata only when `TMDB_API_KEY` is unset. `/v1/status` has
  `tmdb: bool`.
- **Progress:** `Progress` gains `per_hour`, `remaining_seconds` and
  `estimated_finish`, worked out by `models.with_rate()` from the job's start
  and its done/total title-stages. They are filled only for a running job,
  once a title-stage is done (`job_model(now=)`), and appear in
  `/v1/status`'s `current_job` and every job read.
- **Guide:** `docs/library/service.md`, "A first run over a whole catalogue":
  ready check, scan, a small filtered batch checked by hand, then the
  schedule; what progress says; and what a restart mid-run does (the job
  `interrupted`, not resumed, nothing finished lost).

Tests: `test_pipeline_service_operating.py` (the rate arithmetic, running
jobs only, `/ready` and `/v1/status` with and without a key); the readiness
and job expectations in `test_pipeline_service_api.py`. OpenAPI regenerated.
Validation: full suite **2716 passed, 1 skipped**.

## J2 — Resolve JRiver's selected audio stream (completed)

Completed on 2026-10-08, in the commit "Extract the audio stream JRiver
plays, and keep stream choices across rescans", once the media share was
mounted. ffprobe of the fixture's 67 titles with a `Streams` record
(`src/test/python/fixtures/jriver/streams.json`, see its README) showed that
`Streams` is video, audio[, subtitle] as ffprobe's global indices. In 8
titles the audio is not the first stream, and JRiver's codec is at the
matching position.

- `streams.selected_streams()` parses Playback Info, and `audio_ordinal()`
  matches the audio index against a probe of the file. It trusts a selection
  only when its first stream is video and its second audio ("The Town",
  `1,2,19` on a disc, is refused).
- The JRiver source records `LibraryItem.selected_streams` and still lists
  stream 0, since listing never touches the disk. `run.resolve_selected_stream()`
  probes the file as extraction opens it (`Session.probe_streams()`) before
  extracting. The run then records the result (`LibraryIndex.resolve_audio_stream`,
  `audio_stream_source='source'`) and says "(as the library plays it)". When it
  falls back to the first stream it emits a `note` event saying why.
- **A bug found on the way:** a rescan replaced each title's listing, so a
  reviewer's stream choice was lost and the next run extracted the first
  stream again. `index.carried_choice()` keeps a reviewer's choice
  (`'manual'`) and a resolved one while the source selects the same streams.
  `select_audio_stream` marks choices as manual, and a resolved selection
  never replaces one.

Tests: `test_pipeline_library_selected_stream.py` covers parsing (real and
broken values), the rule over all 67 real titles, listing, the rescan
regression, carry-over and its limits, and a real file with video and two
audio streams resolved and extracted as JRiver's second audio stream (kept by
the next scan), plus a refused selection's fallback note. Found for E4:
"A Star Is Born"'s resolved main title shows only AC-3 where JRiver lists
DTS-HD MA.

## E4 — Blu-ray titles: the playlist played, the feature's streams (completed)

Completed on 2026-10-08, in the commit "Choose a Blu-ray's playlist as the
library plays it, and read its streams from the feature". It started from
"A Star Is Born" picking an AC-3 track (J2). Its playlist `00100` plays a
22 s logo clip with stereo AC-3, then the DTS-HD MA feature. The resolver
joined them (`concat:`), and ffmpeg takes a joined input's streams from its
first clip, so the feature was read as the logo's AC-3. Choosing by the
maintainer's suggestions (JRiver's duration, then its `BlurayPlaylist`), with
a fallback for sources that have neither:

- `model.bdmv.resolve_title` leaves out a clip under 2 minutes at either end
  of a title whose audio (ffprobe's codec and channels) differs from the
  longest clip's (`ResolvedTitle.dropped`; `duration_s` excludes it). This
  applies to every source and to Batch and Extract Audio.
- `resolve_main_title(playlist_name, duration_s, first_audio)` picks the
  named playlist (JRiver's `BlurayPlaylist`, or a playlist-file entry
  `PLAYLIST\00305.mpls`, which now opens as its disc with that playlist).
  Failing that, the playlist within 2 s or 0.2% of the library's duration.
  Two that close are told apart by the source's first audio codec ("Glory": a
  stereo AC-3 decoy 0.2 s from the TrueHD Atmos feature), then the closer.
  Otherwise the longest playlist, as before, which is all a source without
  hints gets. A named playlist missing a clip falls back to one of that
  length. A rip whose feature-length playlists all miss a clip ("RoboCop")
  fails, saying it looks incomplete, instead of extracting an unrelated short
  title. `LibraryItem` gains `duration_s`; `extract_cache.title_hint()` and
  `model.bdmv.TitleHint` carry duration and first codec to extraction and the
  probes.

Evidence, the 76 discs of the captured library with the media mounted: 10
choose a playlist other than the longest (7 named by JRiver, 3 by duration),
3 drop a logo clip (A Star Is Born, Arrietty, Bande A Part), RoboCop fails as
incomplete, and on all 74 where JRiver lists audio, ffprobe's first audio
stream of the resolved input is JRiver's first codec (Glory too, after the
tie-break). The `PLAYLIST\index.bluray;N` pseudo-path this item was opened
for was not seen.

Titles already extracted are not extracted again unless their playlist name
changed (it is in the cache key: the titles with a `BlurayPlaylist` will be).
A Star Is Born, Arrietty and Bande A Part need *Revise > Re-extract*.

Tests: `model/test_bdmv.py` covers the drop rules, duration and tie-break,
named and fallback, incomplete rips and codec families.
`test_pipeline_orchestrate_bdmv.py` probes a real disc built with ffmpeg:
an AC-3 stereo intro before a 5.1 feature, where the feature's 6 channels are
read. `test_pipeline_library_jriver_fixture.py` covers playlist, duration and
the playlist-file entry from the real rows. Full suite **2756 passed, 1
skipped**.

## C1 — arm64 image and GHCR publish (completed)

Completed on 2026-10-08 by the first release tag, `2.2.0-alpha.1` (commit
`3ec6b5d`), whose `create-image.yaml` run `37847667859` passed every step. It
built the image, extracted and designed through the stub designer, designed by
reference through the pinned `beqforge-designer:0.2.0`, then built and
published linux/amd64 and linux/arm64 with `docker/setup-buildx-action` and
`docker/login-action` v4 (their first use). Verified from GHCR afterwards:
`ghcr.io/3ll3d00d/beqdesigner-pipeline:2.2.0-alpha.1` has amd64 and arm64
manifests and no `latest` (the prerelease rule). The amd64 image reports
version `2.2.0-alpha.1`, labels revision `3ec6b5d`, and runs as `beq`. The
arm64 image was built and published but not run: no runner or local machine
had arm64 emulation.
