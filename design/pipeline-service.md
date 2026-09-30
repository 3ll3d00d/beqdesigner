# Pipeline service — HTTP control plane, auto mode and Docker image

**Document type:** Architecture reference — delivered behavior.

This is the design of the pipeline service as it is built: `pipeline/service/`,
the Docker image in `docker/`, and the work-directory lease the service shares
with the work list and the CLI. The user guide is
[`docs/library/service.md`](../docs/library/service.md); the wire interface is
[`docs/schema/service.openapi.json`](../docs/schema/service.openapi.json). The
image and the Qt-free boundary it depends on are described in §10 below. Open work is in
[`TODO.md`](TODO.md) (W3, C1). Section numbers are cited by
source comments, so they are kept stable.

## 1. Goal

The headless library pipeline runs as a long-lived Docker container that:

1. **sits idle** and does work only when told to over HTTP;
2. optionally runs an **auto mode**: every *N* minutes, scan the sources and
   extract and design whatever is new, leaving each result in the review queue
   for a person (the same outcome as a nightly `scan` then `run --through design`);
3. lets a caller **select titles the way the work list does** -- needs,
   source, text search, plus year and kind -- and run a stage over them, e.g.
   "extract and design every movie from 2026";
4. publishes the HTTP interface as **OpenAPI 3.1** with strongly typed request
   and response bodies, served by the container itself together with an
   interactive "try it out" page.

Out of scope: reviewing titles over HTTP (it stays in the app), editing the
profile over HTTP, and any UI beyond the generated API page.

## 2. Reuse of the pipeline

The service adds no pipeline logic of its own. Every operation is an existing,
Qt-free call (`pipeline/service/work.py`):

| Service operation | Call |
|---|---|
| Scan | `LibraryIndex.scan(profile, settings, only=, allow_empty=)` |
| Counts / status | `pipeline.library.status` (what `cli status --json` prints) |
| List titles | `Selection.rows(index)` → `TitleRow` |
| Preview a run | `plan_stages(rows, through, retry_failed=)` → `StagePlan` (the work list's button label) |
| Run a stage | `run_stages(profile, selection, through, run_config=, index=, publish=, should_cancel=, on_progress=, on_event=, join=)` → `StagesReport` |
| Bulk accept (guarded, §6.4) | `plan_accept()` / `accept_top_pick()` |
| Config | `pipeline.library.profile` (the profile file the app's Settings drawer and the CLI's `--profile` read) and `pipeline/library/setup.py`, the CLI's profile-to-run resolution, shared so a service run is exactly what `run --profile` does |

The pipeline's invariants hold unchanged: review is a person's (nothing is taken
past design unless accepted); one bad title never stops a run; a failure is
remembered against the source fingerprint and settings and not retried
without `retry_failed`; a cancel leaves only whole titles done; publish and
commit are serialised.

## 3. Selection: year and kind

`Selection` carries two fields besides `needs`, `source`, `match`, `ids` and
`new_since_scan`, so the CLI, the work list and the API share one vocabulary:

| Field | Type | Meaning |
|---|---|---|
| `kind` | `'movie' \| 'tv' \| None` | `TitleRow.kind` equals it |
| `year` | year expression `\| None` | `2026`, `<1960`, `<=1960`, `>1999`, `>=1999`, `1990-1999` (inclusive) |

- The year expression is the same language as an ignore rule's `year`.
  `pipeline/library/year.py` holds the grammar, `YearRange` (`<1960` → max
  1959, `1990-1999` → 1990..1999), `year_matches()` and `YEAR_PATTERN` (the
  grammar as a JSON-schema pattern, held to the parser's answers by a test).
  The grammar is whole-string ASCII (`[0-9]`, `[ \t]`), so fullwidth digits or
  a trailing newline in an ignore rule are refused when the profile loads.
- A title with no numeric year never matches a year expression. A filesystem
  source records no year, so `year` selects its titles only once TMDB has
  named them.
- A reversed range is a `ValueError` when a `Selection` is constructed; in an
  ignore rule it loads and matches nothing.
- `LibraryIndex.titles()` takes `kind=` and `year=` and filters in SQL; the
  SQL test and `year_matches()` are checked against each other.
- `Selection.describe()` includes "movies", "year >=2020".
- CLI: `--kind {movie,tv}` and `--year EXPR` on `run` and `accept`.
- The work list has no Kind filter; its Year column filter works through `ids`.

## 4. Architecture

```
pipeline/service/                 # Qt-free: covered by test_pipeline_qt_boundary.py's AST scan
    __main__.py                   # python -m pipeline.service --profile ... --service-config ...
    config.py                     # ServiceConfig: listen, auth, guards, schedule defaults (§7)
    context.py                    # loads the profile per job; environment secrets
    jobs.py                       # JobManager: one worker, FIFO queue, cancel, progress, history, joining (§5)
    work.py                       # what each kind of job does (§2)
    lease.py                      # the work-directory lease (§5.1)
    scheduler.py                  # AutoScheduler: interval timer that submits jobs (§8)
    notify.py                     # outbound webhooks (§9)
    models.py                     # pydantic request/response models and enums (§6.5)
    api.py                        # FastAPI app factory; run as a module it prints the OpenAPI document
```

- **HTTP stack:** FastAPI + pydantic v2 + uvicorn. The OpenAPI 3.1 document
  is derived from the models and served at `/openapi.json`, with Swagger UI
  (`/docs`) and ReDoc (`/redoc`) as the "live test mode": the same process,
  routes and auth.
- **Dependencies** are the `service` dependency group in `pyproject.toml`; the
  desktop's Qt packages are the default `desktop` group, so `uv sync` still
  installs the app and the PyInstaller bundle does not carry the service.
- **Threads:** uvicorn's event loop answers requests; handlers never do
  pipeline work. Reads (status, titles, plan) open their own read-only
  `LibraryIndex` connection per request. Work goes to the `JobManager`'s
  single worker thread.
- **Profile reloads:** the profile and service config are read again at the
  start of each job, so an edit to the mounted file takes effect on the next
  job without a restart. A profile that fails to load fails that job with the
  loader's message; the service stays up.
- **Shutdown:** uvicorn re-raises the signal that stopped it, so the entry
  point handles SIGTERM itself: it stops accepting jobs, cancels the running
  one cooperatively, waits up to `shutdown_grace_seconds` (default 120, matched
  by the compose file's `stop_grace_period`) and exits 0.

## 5. Jobs

A **job** is one unit of queued work: `scan`, `run` (a selection through a
stage, optionally scanning first) or `accept`. One job runs at a time, because
the index, the queue directory and the git working trees each have a single
writer; further submissions queue in FIFO order, except that a run job may
join the run job in progress (§5.1).

| Field | Type |
|---|---|
| `id` | UUID |
| `kind` | `scan \| run \| accept` |
| `origin` | `api \| schedule` |
| `state` | `queued \| running \| succeeded \| failed \| cancelled \| interrupted` |
| `request` | the typed request that created it |
| `submitted_at`, `started_at`, `finished_at` | RFC 3339 |
| `progress` | `{done, total, title, stage, id}` from `stages.Progress` |
| `joined_to` | the job whose run it joined, if any |
| `result` | typed per kind (§6.5), present once finished |
| `error` | message, when the job itself failed (profile unreadable, index refused) |

- `failed` means the job raised or `StagesReport.failed` is true (a title
  failed, a publish was refused, git refused). A title-level failure is in the
  result; the job still reports every title it did.
- **Cancel** of a queued job removes it; of a running job sets the flag
  `run_stages(should_cancel=)` polls between titles. The job ends `cancelled`
  with `attempted`/`not_run` from the report.
- **Events:** `on_progress` and `on_event` feed a bounded per-job ring buffer,
  redacted with the work list's redaction (`model/execution_events.py`),
  readable afterwards and streamed live (§6.1).
- **History:** the last `history_limit` (default 200) finished jobs are kept in
  memory and written atomically to `<work_dir>/service/jobs.json`. A job found
  `running` at start-up is recorded `interrupted`; nothing resumes
  automatically (the stages are idempotent, so the next scan/run redoes what is
  still needed).

### 5.1 The work-directory lease and joining a run

A person reviews in the app, which reads the same index and queue. Reading
while a run goes is supported (SQLite readers; queue entries re-read as they
are written). Two *runs* at once are not: every run -- a service job, a work
list run, a CLI `run` -- holds the **work-directory lease**
(`<work_dir>/service/lease.json`: host, pid, job id, a heartbeat every 30 s,
stale after three missed beats; `run_lease()` names work-list and CLI runs
`worklist-…` and `cli-…`). A stale lease is ignored and taken over. Taking it is
not atomic across machines; it guards against the ordinary mistake, not as a
lock manager. The documented set-up is the work directory on the container
host's local disk, exported to the reviewer read-mostly.

While a run is in its machine phase (extract and design), more extract/design
work **joins it** instead of being refused:

- `run_stages(join=...)` polls a `JoinQueue` (`pipeline/library/join.py`) each
  time it looks for work, and every quarter second while titles are in hand.
  Each request is a selection planned `through` extract or design (never past
  it), minus titles already in the run; its titles go to the back of the queue
  and the total grows. When the machine phase ends the queue closes: an offer
  is refused, and what was offered but not taken is `report.not_joined`.
- Across processes, the **join inbox** `<work_dir>/service/join/`
  (`pipeline/library/inbox.py`, `handoff.py`) carries requests: a JSON file
  written atomically; the runner claims one by renaming it to `.taken`, and the
  poster withdraws one not yet claimed by renaming it to `.withdrawn`
  (whichever rename wins decides). Every run's `JoinQueue` claims from it.
- **Work list:** while its own run goes, the action button and *Retry failed*
  add to it; Publish, Commit and a bulk accept or revise wait and start in
  order when it ends, and Cancel drops them. While another process holds the
  lease, its extract/design work goes to that run's inbox and the window
  follows the index until the holder's run ends; Publish, Commit and the
  one-step accept-and-publish refuse. Work the holder did not take is run by
  the work list once the lease is free.
- **CLI `run`:** with a fresh lease held, it posts its selection to the inbox;
  once claimed it waits until the run ends (the lease is released), reports its
  titles from the index and exits 0 if none failed. If the lease is released
  before the request is claimed, it withdraws it and runs itself. Its own
  runs take the lease and serve the inbox. Publish and commit wait for the
  lease.
- **Service:** a run job submitted while a run job is in its machine phase
  joins it through the `JoinQueue` the running job registers
  (`JobControl.accept_joins()`): it is `running` at once with `joined_to` set,
  and ends with its host, sharing its result; if the host did not take it
  (`not_joined`) or failed, it goes back to the front of the queue. A job that
  finds the lease held by the work list or a CLI run hands its extract/design
  titles over the same way (`work.handed_off()`, its result read from the
  index) or, for anything else, waits for the lease; cancelled while it waits,
  it ends without running.

Review Folder refuses Publish/Commit while a fresh lease is held, naming its
holder. It checks before and after confirmation, then its worker takes and
holds the work-directory lease until the operation ends. Stale leases are
ignored and replaced as for other runs.

## 6. HTTP interface

All routes are under `/v1` except the health probes and the documentation.
Every request and response body is a named pydantic model in
`components.schemas`.

### 6.1 Routes

| Method & path | Body → response | Notes |
|---|---|---|
| `GET /health` | → `Health` | liveness and release; no auth |
| `GET /ready` | → `Readiness` | profile loads, work dir writable, ffmpeg/ffprobe found, profile's designer registered; 503 otherwise; no auth |
| `GET /v1/status` | → `ServiceStatus` | index counts (the `status --json` content), current job, queue length, schedule state, notification outcomes |
| `GET /v1/titles` | query `TitleFilter` + `limit`/`offset` → `TitlePage` | the work list's table; paged after the index query |
| `GET /v1/titles/{id}` | → `Title` | 404 if unknown |
| `POST /v1/plan` | `RunRequest` → `PlanPreview` | dry run: what would run and what would be skipped and why |
| `POST /v1/jobs/scan` | `ScanRequest` → 202 `Job` | |
| `POST /v1/jobs/run` | `RunRequest` → 202 `Job` | |
| `POST /v1/jobs/accept` | `AcceptRequest` → 202 `Job` | guarded (§6.4) |
| `GET /v1/jobs` | query `state`, `kind`, `limit` → `JobList` | newest first |
| `GET /v1/jobs/{id}` | → `Job` | |
| `POST /v1/jobs/{id}/cancel` | → `Job` | 409 if already finished |
| `GET /v1/jobs/{id}/events` | → `text/event-stream` of `JobEvent` | Server-Sent Events: backlog then live, ends with the job |
| `GET /v1/jobs/{id}/log` | → list of `JobEvent` | the events after the fact |
| `GET /v1/schedule` | → `Schedule` | |
| `PUT /v1/schedule` | `ScheduleUpdate` → `Schedule` | persisted (§8) |
| `POST /v1/schedule/trigger` | → 202 `Job` | one tick now; 409 while busy |
| `POST /v1/notify/test` | `NotifyTest` → `NotifyOutcome` | sample events to one target (§9) |
| `GET /openapi.json`, `/docs`, `/redoc` | | the interface and its try-it-out page |

Submitting returns `202` with the `Job` and `Location: /v1/jobs/{id}`. Errors
use one model, `Problem` (RFC 9457: `type`, `title`, `status`, `detail`, and
`errors` for field-level validation); FastAPI's own 422 responses are
re-shaped into it and its 422 schema is replaced.

### 6.2 The filter

`TitleFilter` is `Selection` over the wire, field for field:

```json
{"needs": ["extract", "design"], "new_since_scan": false, "source": "films",
 "match": "alien", "ids": [], "kind": "movie", "year": "2026"}
```

Every field is optional and they are ANDed; `{}` is every title. On
`GET /v1/titles` they are query parameters (`needs` repeated,
`?kind=movie&year=%3E%3D2020`); on the job routes the filter is the `filter`
member of the body. `include_done` (default false, like the work list's *All*
chip) applies to listing only. An unknown `source` is a 422 naming the
profile's sources.

### 6.3 Running a stage

```http
POST /v1/jobs/run
{"filter": {"kind": "movie", "year": "2026"}, "through": "design",
 "scan_first": true, "retry_failed": false}
```

`through` is `extract | design | publish | commit`, with `plan_stages`'
meaning: extract and design as needed; publish/commit only titles a person
accepted. `scan_first` (default true) lists the sources before selecting.
The finished job's `result` is a `RunResult` carrying the `StagesReport`
fields (`selected`, `extracted`, `cached`, `designed`, `design_cached`,
`failed[{id, message}]`, `failed_earlier`, `skipped[{id, title, reason}]`,
`published`, `publish_errors`, `committed`, `commit_error`, `cancelled`,
`attempted`, `not_run`, `counts`) as typed members.

### 6.4 Guards on writing to the repositories

`through: publish | commit` and `POST /v1/jobs/accept` write to, and may push,
the catalogue repositories. They are refused with 403 unless the service
config has `allow_repository_writes: true`, and `commit` is further refused
unless the profile's `sync:` names the repositories. The schedule can never go
past `design` (§8). The default container can therefore extract and design
only.

### 6.5 Typing

- Enums, each a named schema: `Needs`, `Through`, `AutoThrough`
  (`extract | design`), `Kind`, `JobKind`, `JobState`, `JobOrigin`, `Tier`,
  `Flag`.
- `Job` is a **discriminated union** on `kind`:
  `ScanJob{request: ScanRequest, result: ScanResult}`,
  `RunJob{request: RunRequest, result: RunResult}`,
  `AcceptJob{request: AcceptRequest, result: AcceptResult}`.
- `Title` mirrors `TitleRow` (states, `needs`, `tier`, `detail`, `flags`,
  `confidence`, `candidate_count`, `external_ids`, `is_new`), times as RFC 3339.
- `YearExpression` is a named string schema with `YEAR_PATTERN` and examples;
  the server parses it with `year.py`. A reversed range passes the pattern
  and is a 422 from the parser.
- Input models are `extra='forbid'`, so a misspelt field is a 422, not a
  silently wider selection.
- The models live only in `models.py` and convert to and from the dataclasses
  (`Selection`, `StagePlan`, `StagesReport`, `AcceptReport`, `TitleRow`); the
  dataclasses know nothing of pydantic. A test round-trips each conversion and
  fails if a dataclass gains a field its model does not carry.

### 6.6 Publishing the interface

- The document is generated from the app and committed as
  `docs/schema/service.openapi.json`; a test regenerates it and fails on any
  difference. It carries no release (`/health` does); `info.version` is the
  API's own semantic version (1.2.0), bumped by hand on a change.
- Swagger UI and ReDoc are served from a local copy when `--static-dir` /
  `BEQ_SERVICE_STATIC` names one -- the image fetches pinned packages at build
  -- and otherwise load from a CDN.
- The document declares the bearer scheme, so *Authorize* in Swagger UI
  authenticates every *Try it out* call.
- The user page is `docs/library/service.md`.

### 6.7 Authentication

A single bearer token from `BEQ_SERVICE_TOKEN` (or `BEQ_SERVICE_TOKEN_FILE`,
for Docker secrets), compared in constant time. Without one the service
refuses to start unless bound to `127.0.0.1` with `--no-auth`. `/health` and
`/ready` are unauthenticated; `/docs` and `/openapi.json` are readable without
a token (the calls made from them still need it). TLS is left to a reverse
proxy.

## 7. Configuration

Two files, both mounted read-only at `/config`:

- **`profile.yaml`** -- the catalogue profile shared with the app and the CLI.
  Paths in it are *container* paths (JRiver's `path_mappings` map the server's
  Windows paths to the container's mounts).
- **`service.yaml`** -- the service's own settings, separate so the app's
  Settings drawer, which rewrites the profile, never meets them:

```yaml
listen: {host: 0.0.0.0, port: 8080}
allow_repository_writes: false
history_limit: 200
shutdown_grace_seconds: 120
schedule:                        # the initial schedule; PUT /v1/schedule overrides it (§8)
  enabled: true
  interval_minutes: 60
  filter: {kind: movie}          # a TitleFilter; needs is always extract|design (§8)
  through: design                # extract | design
  retry_failed: false
notify: []                       # webhook targets (§9)
```

Secrets come from the environment or `*_FILE` variables rather than the YAML:
`BEQ_SERVICE_TOKEN`, `TMDB_API_KEY` (overrides `run.tmdb_api_key`),
`JRIVER_PASSWORD` (per source: `JRIVER_PASSWORD_<SOURCE>`), and a designer's
headers as JSON in `BEQ_DESIGNER_HEADERS_<NAME>`. Precedence is environment,
then `service.yaml`, then the profile.

## 8. Auto mode

`AutoScheduler` holds `{enabled, interval_minutes, filter, through,
retry_failed, next_run_at, last_run: {job_id, state, finished_at}}`.

- **A tick** submits one `run` job with `origin: schedule`, `scan_first: true`,
  `unattended: true` and the schedule's filter with `needs` forced to
  `[extract, design]`. It does not use `new_since_scan`: "new" is only the
  latest scan's, so a failed or cancelled tick would lose its titles, whereas
  "needs extract or design" is exactly what is still to do.
- `through` is `extract` or `design` only (`AutoThrough`). No automatic
  publish is offered.
- **Busy:** if any job is queued or running when a tick is due, the tick is
  skipped and recorded (`last_skip: busy`) through an atomic idle-only
  submission in `JobManager`; ticks never pile up. The next is
  `interval_minutes` after the *finish* of the last scheduled job.
- **Failures:** an unattended run does not retry a remembered failure
  (including a failed extraction) until the source or settings change or
  `retry_failed`; each tick reports them as `failed_earlier`.
- `PUT /v1/schedule` validates and writes `<work_dir>/service/schedule.json`
  atomically; at start-up that file, when present, wins over `service.yaml`.
  `enabled: false` pauses without losing the settings. The minimum interval is
  5 minutes.

## 9. Notifications

Targets, events, payload fields and setup are in the
[service user guide](../docs/library/service.md); the JSON payload is the
`Notification` schema in the OpenAPI `webhooks`. `notify.py` validates the
configured targets, reads URL/header secrets from the environment, and
delivers after a job's history is saved, on its own thread with up to three
attempts. JSON carries job, designed-title, failure and review-count details;
text, Slack and Discord carry a count summary. Redirects are refused; status
records only sanitised outcomes, never URLs or headers. `POST /v1/notify/test`
sends sample events. Delivery goes only to URLs configured in `service.yaml` or
the target's environment override.

## 10. Docker image

- **Files:** `docker/Dockerfile`, `docker/compose.example.yaml`,
  `docker/service.example.yaml`, `docker/smoke.py`, `.dockerignore`.
- **Base:** `python:3.13-slim` (the project's `requires-python`), with `ffmpeg`
  from Debian (including the `dvdvideo` demuxer the README asks for DVDs; the
  build asserts `ffmpeg -demuxers` lists it), `git` and `openssh-client`. No
  graphviz and no Qt or its X/GL libraries (§10.1).
- **Install:** multi-stage. A pinned uv binary runs
  `uv sync --frozen --no-dev --group service` into `/opt/venv`; the source is
  copied to `/app/src/main/python` with `PYTHONPATH` set to it. Swagger UI and
  ReDoc are fetched as pinned npm packages and served locally
  (`BEQ_SERVICE_STATIC`), so the try-it-out page works without internet access.
  The runtime stage has no uv.
- **Dependency groups:** the desktop's Qt packages are the default `desktop`
  group in `pyproject.toml`, so a normal `uv sync` installs the app, while the
  image installs only the runtime list and the `service` group.
- **User:** a non-root `beq` user; `PUID`/`PGID` build args (or `user:` in
  compose) so files written to the mounts belong to the host user.
- **Volumes:** `/config` (ro: `profile.yaml` and `service.yaml`), `/work` (work
  directory, index, service state), `/queue` (review queue), the media mounts at
  the paths the profile names (ro), and, only when repository writes are
  enabled, the repository clones and `/home/beq/.ssh` (ro: key and
  `known_hosts`) -- git uses the credentials mounted for the container user.
  Git identity comes from `GIT_AUTHOR_NAME`/`GIT_AUTHOR_EMAIL`/`GIT_COMMITTER_*`.
  The compose example shows the mounts and the token as a Docker secret.
- **Entrypoint:** `python -m pipeline.service --profile /config/profile.yaml
  --service-config /config/service.yaml`; `HEALTHCHECK` requests `/health`.
- **Smoke test:** `docker/smoke.py` starts the built image against a fixture
  profile with a filesystem source over a short synthetic six-channel WAV and
  a stub HTTP designer on the host, waits for `/ready`, submits a run of that
  title through `design`, polls the job to `succeeded` and asserts a queue
  entry exists. It runs in `.github/workflows/test.yaml` on every push (Linux
  only) and in `create-image.yaml` before publishing.
  `test_pipeline_service_docker.py` loads `docker/smoke.py` by path and runs
  the same fixture through the real local service, ffmpeg and HTTP designer
  without Docker.
- **Publishing:** on a pushed tag (the trigger `create-app.yaml` uses for the
  desktop release), `create-image.yaml` builds `linux/amd64` and `linux/arm64`,
  runs the smoke test, and pushes `ghcr.io/3ll3d00d/beqdesigner-pipeline:<tag>`
  with `GITHUB_TOKEN` (`packages: write`). `latest` is applied only to a tag
  without a pre-release suffix (`-alpha`/`-beta`/`-rc`). The image carries OCI
  labels (source, revision, version), and `src/main/python/VERSION` is written
  before the build so `/health` reports the release.

## 10.1 The Qt-free boundary

Nothing under `pipeline/` reaches `qtpy`, `PyQt6`, `qtawesome`, `pyqtgraph` or
`ui.*`, even indirectly. The `model/` modules the pipeline uses keep a Qt-free
core, with their Qt halves in sibling modules that import it:

| Qt-free core | Qt half |
|---|---|
| `model/preferences.py` | `model/preferences_dialog.py` |
| `model/limits.py` | `model/limits_dialog.py` |
| `model/minidsp.py` | `model/minidsp_qt.py` |
| `model/ffmpeg.py` (`Executor`, `run_sync()`) | `model/ffmpeg_qt.py` (`AudioExtractor`, which `Executor.execute()` imports for the GUI) |
| `model/signal.py` (`Signal`, `SignalData`, `AutoWavLoader`) | `model/signal_qt.py` (table models, dialogs, their loaders, the smoother) |
| `model/dsp_type.py` (a plain enum) | re-exported by `model/merge.py` |

`model.magnitude` imports its dialogs only where it shows them.
`model.xy.interp()` called without `smooth=` reads the desktop's smooth-graphs
preference; with no Qt available it uses that preference's default, so a
container run matches a desktop left at the default.

The ffmpeg progress bridge's UDP port is picked by the operating system, so two processes
extracting at once do not collide.

**Tests:** `test_qt_free_modules.py` imports every `pipeline/` module, and the
`model/` modules it uses, in a fresh interpreter with Qt blocked by a
`sys.meta_path` finder, and completes a whole `Session` run there;
`test_pipeline_qt_boundary.py` scans imports and checks no `QApplication` is
constructed. A module the pipeline starts to use is added to
`test_qt_free_modules.py`'s list.

## 11. Delivery history

Completed chunks and their commit hashes are in the
[archived delivery record](archive/pipeline-service/completion-record.md).
Current work is tracked only in [TODO](TODO.md).

## 12. Decisions

Settled on 2026-09-26 and 2026-09-27:

| # | Question | Decision |
|---|---|---|
| 1 | HTTP framework | FastAPI + pydantic v2; the pipeline's dataclasses stay pydantic-free |
| 2 | Repository writes over HTTP | Offered, off by default behind `allow_repository_writes` (§6.4); never from the schedule |
| 3 | Year filter shape | The ignore-rule expression string, one grammar for ignore rules, CLI and API (§3, §6.5) |
| 4 | Auth | One static bearer token (§6.7) |
| 5 | Notifications | Outbound webhooks (§9) |
| 6 | Image distribution | Built and pushed to GHCR by GitHub Actions when a tag is pushed (§10) |
| 7 | Qt in the image | None: the pipeline is Qt-free (§10.1) |
| 8 | Work asked for during a run | Extract/design joins the run in progress; the CLI hands its titles over and waits (§5.1) |
