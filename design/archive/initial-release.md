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
Windows result is the next `main` CI run. Turning `EncodingWarning` into an
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
