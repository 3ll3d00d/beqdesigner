# Pipeline service — HTTP control plane, auto mode and Docker image

**Status: design, not built.** Every chunk below is *Not started*; see the
[index](library-sync-pipeline-plan.md). The Docker image and chunk S0 are in
[`pipeline-service/docker.md`](pipeline-service/docker.md). Branch: `pipeline-service`.

## 1. Goal

Run the headless library pipeline as a long-lived Docker container that:

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

## 2. What already exists and is reused unchanged

The service adds no pipeline logic of its own. Every operation is an existing,
Qt-free call:

| Service operation | Existing call |
|---|---|
| Scan | `LibraryIndex.scan(profile, settings, only=, allow_empty=)` |
| Counts / status | `pipeline.library.status` (what `cli status --json` prints) |
| List titles | `Selection.rows(index)` → `TitleRow` |
| Preview a run | `plan_stages(rows, through, retry_failed=)` → `StagePlan` (the work list's button label) |
| Run a stage | `run_stages(profile, selection, through, run_config=, index=, publish=, should_cancel=, on_progress=, on_event=)` → `StagesReport` |
| Bulk accept (guarded, §6.4) | `plan_accept()` / `accept_top_pick()` |
| Config | `pipeline.library.profile` (the same profile file the app's Settings drawer and the CLI's `--profile` read) and the CLI's option → `LibraryRunConfig`/`PublishSettings` builders |

Invariants carried over as they are: review is a person's (nothing is taken
past design unless accepted); one bad title never stops a run; a failure is
remembered against the source fingerprint and settings and not retried
without `retry_failed`; a cancel leaves only whole titles done; publish and
commit are serialised.

## 3. Selection: year and kind (chunk S1)

The work list's filters are the pipeline strip (a `needs` chip or *New*), the
source combo, the search box and the per-column filters (Title, Year, Source,
Needs, Detail). `Selection` today has `needs`, `source`, `match`, `ids` and
`new_since_scan`; the work list turns column filters into an explicit `ids`
list. A remote caller cannot build that list, so the two fields the example
needs become first-class **in `Selection` itself**, keeping one vocabulary for
the CLI, the work list and the API (`selection.py`'s docstring rule):

| Field | Type | Meaning |
|---|---|---|
| `kind` | `'movie' \| 'tv' \| None` | `TitleRow.kind` equals it |
| `year` | year expression `\| None` | the ignore-rule syntax: `2026`, `<1960`, `<=1960`, `>1999`, `>=1999`, `1990-1999` (inclusive) |

- The year expression is **the same language as an ignore rule's `year`**
  (decided, §11): one syntax for a person to learn across ignore rules, the
  CLI and the API. Its grammar and matching move from `ignore.py`
  (`_YEAR`, `_year_matches`) into one shared module, `pipeline/library/year.py`,
  with a `YearRange` parse (`<1960` → `max 1959`, `1990-1999` → `1990..1999`;
  a reversed range is an error) that both ignore rules and `Selection` use.
- A title with no numeric year never matches a year expression, as for an
  ignore rule.
- `Selection.__post_init__` parses `year` so a malformed expression is a
  `ValueError` at construction, not an empty match.
- `LibraryIndex.titles()` gains `kind=` and `year=` (a `YearRange`) and
  filters in SQL (`year GLOB '[0-9]*' AND CAST(year AS INTEGER) BETWEEN ? AND ?`,
  an open end omitted), so paging (§6.2) stays in the database.
- `Selection.describe()` adds "movies", "year >=2020".
- CLI: `--kind {movie,tv}` and `--year EXPR` on `run` and `accept`; the README
  option list and its docs test follow.
- The work list is unchanged in S1 (its Year column filter keeps working
  through `ids`). A Kind chip or combo is a possible follow-up, not required.

**Tests:** `test_pipeline_library_selection.py` (every expression form, open
ranges, reversed and malformed expressions, missing and non-numeric years, kind, AND with the existing fields, `describe()`);
index query tests for the SQL path including a batch of `ids`; CLI parse of
each `--year` form and its rejection of a malformed one; ignore-rule tests
still pass against the shared helper.

## 4. Architecture (chunks S2-S3)

```
pipeline/service/                 # Qt-free: covered by test_pipeline_qt_boundary.py's AST scan
    __main__.py                   # python -m pipeline.service --profile ... --service-config ...
    config.py                     # ServiceConfig: listen, auth, guards, schedule defaults (§7)
    jobs.py                       # JobManager: one worker, FIFO queue, cancel, progress, history (§5)
    scheduler.py                  # AutoScheduler: interval timer that submits jobs (§8)
    context.py                    # loads the profile per job; builds LibraryRunConfig / PublishSettings
    models.py                     # pydantic request/response models and enums (§6.5)
    api.py                        # FastAPI app factory: routes -> JobManager / index reads
```

- **HTTP stack:** FastAPI + pydantic v2 + uvicorn. FastAPI derives the
  OpenAPI 3.1 document from the pydantic models, serves it at `/openapi.json`
  and serves Swagger UI (`/docs`, with *Try it out* and *Authorize*) and ReDoc
  (`/redoc`). That is the "live test mode": the same process, the same
  routes, the same auth.
- **Dependencies** go in a new `service` dependency group in `pyproject.toml`
  (`fastapi`, `uvicorn[standard]`, `pydantic>=2`), not the app's runtime
  list, so the PyInstaller desktop bundle does not grow. The dev group gains
  them too (plus `httpx` for FastAPI's `TestClient`) so the suite and CI run
  the service tests.
- **Threads:** uvicorn's event loop answers requests; route handlers never do
  pipeline work. Reads (status, titles, plan) open their own read-only
  `LibraryIndex` connection per request, as the work list's run job does.
  Work goes to the `JobManager`'s single worker thread.
- **Profile reloads:** the profile and service config are read again at the
  start of each job (as the Review Folder window reads the library profile
  at each decision), so an edit to the mounted file takes effect on the next
  job without a restart. A profile that fails to load fails that job with
  the loader's message; the service stays up.

## 5. Jobs (chunk S2)

A **job** is one unit of queued work: `scan`, `run` (a selection through a
stage, optionally scanning first) or `accept`. One job runs at a time,
because the index, the queue directory and the git working trees each have a
single writer. Further submissions queue in FIFO order.

| Field | Type |
|---|---|
| `id` | UUID |
| `kind` | `scan \| run \| accept` |
| `origin` | `api \| schedule` |
| `state` | `queued \| running \| succeeded \| failed \| cancelled \| interrupted` |
| `request` | the typed request that created it |
| `submitted_at`, `started_at`, `finished_at` | RFC 3339 |
| `progress` | `{done, total, title, stage, id}` from `stages.Progress` |
| `result` | typed per kind (§6.5), present once finished |
| `error` | message, when the job itself failed (profile unreadable, index refused) |

- `failed` means the job raised or `StagesReport.failed` is true (a title
  failed, a publish was refused, git refused). A title-level failure is in
  the result; the job still reports every title it did.
- **Cancel** of a queued job removes it; of a running job sets the flag
  `run_stages(should_cancel=)` polls between titles. The job ends `cancelled`
  with `attempted`/`not_run` from the report.
- **Events:** `on_progress` and `on_event` feed a bounded per-job ring buffer
  (secrets redacted the way the work list's run details are), readable
  after the fact and streamed live (§6.2).
- **History:** the last *K* (default 200) finished jobs are kept in memory and
  written atomically to `<work_dir>/service/jobs.json`, so a restart shows
  them. A job found `running` at start-up is recorded `interrupted`; nothing
  is resumed automatically (the next scan/run redoes what is still needed,
  since the stages are idempotent).
- **Shutdown:** SIGTERM stops accepting jobs, cancels the running one
  cooperatively, waits up to a grace period (config, default 120 s, matched
  by the compose file's `stop_grace_period`), then exits.

### 5.1 Sharing a work directory with the desktop app

A person reviews in the app, which reads the same index and queue. Reading
while the service runs is supported (SQLite readers, queue entries re-read as
they are written, as bulk accept already does). Two *runs* at once are not:
the service takes a **work-directory lease** (`<work_dir>/service/lease.json`:
host, pid, job id, heartbeat every 30 s, stale after 3 missed beats) for the
duration of each job, and the work list's Run/Publish/Commit actions refuse
with "the pipeline service on HOST is running a job" while a fresh lease
exists. SQLite over a network share is a known hazard; the documented set-up
is the work directory on the container host's local disk, exported to the
reviewer read-mostly, not the other way round.

**Tests (S2):** job lifecycle and FIFO order with a fake `run_stages`;
cancel queued vs running; `failed` from a `StagesReport` with a failed title;
restart marks `interrupted`; history trimmed and atomically rewritten; the
lease written, heart-beaten, released, taken over when stale and honoured by
the work list's actions (a gui test).

## 6. HTTP interface (chunk S3)

All routes are under `/v1` except the health probes and the documentation.
Request and response bodies are JSON; every one is a named pydantic model, so
each appears in `components.schemas` of the OpenAPI document.

### 6.1 Routes

| Method & path | Body → response | Notes |
|---|---|---|
| `GET /health` | → `Health` | liveness; no auth |
| `GET /ready` | → `Readiness` | profile loads, work dir writable, ffmpeg/ffprobe found, designer named in the profile registered; 503 otherwise; no auth |
| `GET /v1/status` | → `ServiceStatus` | index counts (the `status --json` content), current job, queue length, schedule state |
| `GET /v1/titles` | query `TitleFilter` + `limit`/`offset` → `TitlePage` | the work list's table |
| `GET /v1/titles/{id}` | → `Title` | 404 if unknown |
| `POST /v1/plan` | `RunRequest` → `PlanPreview` | dry run: what would run and what would be skipped and why; changes nothing |
| `POST /v1/jobs/scan` | `ScanRequest` → 202 `Job` | |
| `POST /v1/jobs/run` | `RunRequest` → 202 `Job` | the filtered extract/design |
| `POST /v1/jobs/accept` | `AcceptRequest` → 202 `Job` | guarded (§6.4) |
| `GET /v1/jobs` | query `state`, `kind`, `limit` → `JobList` | newest first |
| `GET /v1/jobs/{id}` | → `Job` | |
| `POST /v1/jobs/{id}/cancel` | → `Job` | 409 if already finished |
| `GET /v1/jobs/{id}/events` | → `text/event-stream` of `JobEvent` | Server-Sent Events: backlog then live, ends when the job does |
| `GET /v1/schedule` | → `Schedule` | |
| `PUT /v1/schedule` | `ScheduleUpdate` → `Schedule` | persisted (§8) |
| `POST /v1/schedule/trigger` | → 202 `Job` | one tick now, whatever the timer |
| `POST /v1/notify/test` | `NotifyTest` → `NotifyOutcome` | send a sample to one target (§9) |
| `GET /openapi.json`, `/docs`, `/redoc` | | the published interface and its try-it-out page |

Submitting returns `202` with the `Job` and a `Location: /v1/jobs/{id}`
header. Errors use one model, `Problem` (RFC 9457: `type`, `title`, `status`,
`detail`, and `errors` for field-level validation), including FastAPI's own
422 validation responses, which are re-shaped into it.

### 6.2 The filter

`TitleFilter` is `Selection` over the wire, field for field:

```json
{
  "needs": ["extract", "design"],
  "new_since_scan": false,
  "source": "films",
  "match": "alien",
  "ids": [],
  "kind": "movie",
  "year": "2026"
}
```

Every field is optional and they are ANDed; `{}` is every title. On
`GET /v1/titles` the same fields are query parameters (`needs` repeated,
`?kind=movie&year=%3E%3D2020`, i.e. `year=>=2020` URL-encoded); on the job routes the filter is
the `filter` member of the body. `include_done` (default false, as the work
list's *All* chip) applies to listing only; a run skips done titles anyway.
An unknown `source` is a 422 naming the profile's sources, rather than an
empty match.

### 6.3 Running a stage

```http
POST /v1/jobs/run
{
  "filter": {"kind": "movie", "year": "2026"},
  "through": "design",
  "scan_first": true,
  "retry_failed": false
}
```

`through` is `extract | design | publish | commit`, with `plan_stages`'
meaning: extract and design as needed, and publish/commit only titles a
person accepted. `scan_first` lists the sources before selecting (default
true for API jobs, so "new content" is seen). The finished job's `result` is
a `RunResult`: the `StagesReport` fields (`selected`, `extracted`, `cached`,
`designed`, `design_cached`, `failed[{id, message}]`, `failed_earlier`,
`skipped[{id, title, reason}]`, `published`, `publish_errors`, `committed`,
`commit_error`, `cancelled`, `attempted`, `not_run`, `counts`) as typed
members rather than the CLI's tuples.

### 6.4 Guards on writing to the repositories

`through: publish | commit` and `POST /v1/jobs/accept` write to, and may push,
the catalogue repositories. They are refused with 403 unless the service
config has `allow_repository_writes: true`, and `commit` is further refused
unless the profile's `sync:` names the repositories. The automatic schedule
can never go past `design` (§8). The default container therefore can extract
and design only.

### 6.5 Typing

- Enums, each a named schema: `Needs`, `Through`, `AutoThrough`
  (`extract | design`), `Kind`, `JobKind`, `JobState`, `JobOrigin`, `Tier`,
  `Flag`.
- `Job` is generic over its request and result with a **discriminated union**
  on `kind`: `ScanJob{request: ScanRequest, result: ScanResult}`,
  `RunJob{request: RunRequest, result: RunResult}`,
  `AcceptJob{request: AcceptRequest, result: AcceptResult}`, so a generated
  client gets a concrete result type per kind.
- `Title` mirrors `TitleRow` (states, `needs`, `tier`, `detail`, `flags`,
  `confidence`, `candidate_count`, `external_ids`, `is_new`), with times as
  RFC 3339 rather than epoch floats.
- `YearExpression` is a named string schema carrying the grammar as its
  `pattern` (and examples), so Swagger UI and generated clients show and
  check it; the server parses it with the shared `year.py` so pattern and
  parser cannot disagree (a test checks the pattern against the parser over
  a table of good and bad inputs). A reversed range passes the pattern and
  is a 422 from the parser.
- Models are `extra='forbid'` on input, so a misspelt field is a 422, not a
  silently wider selection.
- The models live only in `pipeline/service/models.py` and convert to and
  from the dataclasses (`Selection`, `StagePlan`, `StagesReport`,
  `AcceptReport`, `TitleRow`); the dataclasses do not learn about pydantic.
  A test round-trips each conversion and fails if a dataclass gains a field
  its model does not carry.

### 6.6 Publishing the interface

- The document is generated from the app, not hand-written. It is also
  committed as `docs/schema/service.openapi.json` beside the other published
  schemas, and a test regenerates it and fails on any difference, so a
  change to the interface is a visible diff. `info.version` is the API's own
  semantic version (starting `1.0.0`), bumped by hand on a change.
- Swagger UI and ReDoc load their scripts from a CDN by default. The image
  vendors them (a pinned `swagger-ui-dist` copy under
  `pipeline/service/static/`) and serves them locally, so the try-it-out page
  works on a LAN without internet access.
- The document declares the bearer security scheme, so *Authorize* in
  Swagger UI takes the token and every *Try it out* call is authenticated.
- `docs/` gains a page (on readthedocs) describing the service, linking the
  schema and showing the curl form of §6.3.

### 6.7 Authentication

A single bearer token from `BEQ_SERVICE_TOKEN` (or `BEQ_SERVICE_TOKEN_FILE`,
for Docker secrets), compared in constant time. Without one the service
refuses to start unless it is bound to `127.0.0.1` and started with
`--no-auth`. `/health` and `/ready` are unauthenticated; `/docs` and
`/openapi.json` are readable without a token (the calls made from them still
need it). TLS is left to a reverse proxy; the compose example notes it.

**Tests (S3):** `TestClient` over the app with a fixture index and a fake
`JobManager`/`run_stages`: every route's happy path and its 4xx (unknown id,
unknown source, extra field, bad enum, guard 403, cancel of a finished job
409, missing/wrong token 401); query-string and body forms of the filter
select the same rows; the SSE stream delivers backlog then live events and
closes; the committed OpenAPI document matches the generated one; the
document validates as OpenAPI 3.1 (`openapi-spec-validator`, dev only);
secrets never appear in a job's events or `error`.

## 7. Configuration

Two files, both mounted read-only at `/config`:

- **`profile.yaml`** -- the existing catalogue profile, unchanged, shared
  with the app and the CLI. Paths in it are *container* paths (JRiver's
  `path_mappings` map the server's Windows paths to the container's mounts).
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

Secrets are taken from the environment or `*_FILE` variables rather than the
YAML: `BEQ_SERVICE_TOKEN`, `TMDB_API_KEY` (overrides `run.tmdb_api_key`),
`JRIVER_PASSWORD` (per-source: `JRIVER_PASSWORD_<SOURCE>`), and a designer's
headers via `BEQ_DESIGNER_HEADERS_<NAME>` as JSON. Precedence is environment,
then `service.yaml`, then the profile.

## 8. Auto mode (chunk S4)

The scheduler holds `{enabled, interval_minutes, filter, through,
retry_failed, next_run_at, last_run: {job_id, state, finished_at}}`.

- **A tick** submits one `run` job with `origin: schedule`, `scan_first:
  true`, and the schedule's filter with `needs` forced to
  `[extract, design]`. It deliberately does **not** use `new_since_scan`:
  "new" is only the latest scan's, so a tick that failed or was cancelled
  would lose its titles to the next scan's generation, whereas "needs
  extract or design" is exactly what is still to do, and the stages are
  idempotent.
- `through` is `extract` or `design` only (the `AutoThrough` enum). Reviewing
  stays a person's; an automatic publish is not offered.
- **Busy:** if any job is queued or running when a tick is due, the tick is
  skipped and recorded (`last_skip: busy`), not queued behind it -- ticks
  never pile up. The next is `interval_minutes` after the *finish* of the
  last scheduled job, so a run longer than the interval does not start the
  next straight away.
- **Failures** are remembered per title as today, so a title that fails
  every hour is tried once, and reported in each tick's result as
  `failed_earlier`, until the source or settings change or `retry_failed`.
- `PUT /v1/schedule` validates and writes `<work_dir>/service/schedule.json`
  atomically; on start-up that file, when present, wins over `service.yaml`.
  `enabled: false` pauses without losing the settings.
- The minimum interval is 5 minutes (a scan of a JRiver node is one request,
  but a filesystem source costs a `stat` per file).

**Tests (S4):** an injected clock drives ticks; interval measured from finish;
busy skip; `needs` forced whatever the filter says; `through: publish`
refused by the model; persisted schedule wins at start-up; pause/resume;
trigger now while idle and while busy (409 `Problem`).

## 9. Notifications (chunk S6)

Auto mode runs with no one watching, so the service tells someone when there
is something to look at, instead of relying on polling `GET /v1/status`.

```yaml
notify:                                   # service.yaml; zero or more targets
  - name: phone
    url: https://ntfy.example/beq          # or a Home Assistant, Slack or Discord webhook
    format: text                           # json | text | slack | discord
    events: [review_waiting, failed]       # default: all but job_finished
    origins: [schedule]                    # default: schedule only; add api for API jobs
```

- **Events**, each evaluated when a job finishes: `review_waiting` (the job
  designed titles that now need review, including declines), `failed` (a
  title failed, a publish was refused, git refused, or the job itself
  failed), `job_finished` (every job, for a log or a dashboard).
  `failed_earlier` titles are not news and never trigger `failed`, so a title
  that fails every hour notifies once.
- **Payload.** `json` POSTs a typed `Notification`: `{event, job: {id, kind,
  origin, state, started_at, finished_at}, designed: [{id, title, year, kind,
  confidence}], failed: [{id, title, message}], review_waiting: N,
  links: {job, docs}}` where `review_waiting` is the index's count after the
  job. It is declared in the OpenAPI document's top-level **`webhooks`**
  section (OpenAPI 3.1; FastAPI's `app.webhooks`), so the outbound shape is
  published and typed like the inbound one. `text` POSTs one plain-text line
  ("3 titles waiting for review, 1 failed", ntfy's native form); `slack` and
  `discord` wrap that line as `{"text": ...}` / `{"content": ...}`.
- **Secrets:** a target's headers come from `BEQ_NOTIFY_HEADERS_<NAME>`
  (JSON) and a URL containing a token may be given as
  `BEQ_NOTIFY_URL_<NAME>`; neither is ever echoed in status, events or logs.
- **Delivery** is on its own thread after the job is recorded: 10 s timeout,
  three attempts with backoff, then give up. A delivery never changes a
  job's state; the last outcome per target is in `ServiceStatus.notify`
  (`{name, last_event, last_attempt_at, ok, message}`).
- **`POST /v1/notify/test`** (`{target: name}` → the delivery outcome) sends
  a sample of each configured event so a target can be checked from Swagger UI.

**Tests (S6):** which events fire for each shape of `StagesReport` (designed,
declined, failed, failed_earlier only, cancelled, job error); origin and
event filters; each format's body against a local HTTP server; retries and
give-up with a server that fails then succeeds, and one that hangs; secrets
absent from status and the recorded message; the `webhooks` entry present in
the committed OpenAPI document; the test route.

## 10. Docker image (chunks S0, S5)

In its own file: [`pipeline-service/docker.md`](pipeline-service/docker.md) --
the image (§10 there), GHCR publishing on tag, and chunk S0, the Qt-free
extraction path the image depends on (§10.1 there).

## 11. Chunks

| Chunk | Content | Depends on | Status |
|---|---|---|---|
| S0 | Qt-free extraction path ([docker.md §10.1](pipeline-service/docker.md)): no `qtpy`/`PyQt6` reachable from `pipeline/` | -- | Done: `55c3425`, `5d136d3`, `579f542`, `4929e93` and the "S0 done" commit after it (see docker.md §10.1 "As built") |
| S1 | `Selection.kind` and `Selection.year` (expression), shared `year.py`, index SQL, CLI `--kind`/`--year` | -- | Not started |
| S2 | `pipeline/service`: config, per-job profile context, `JobManager`, history, work-dir lease (+ work list honours it) | S1 | Not started |
| S3 | FastAPI app, models, routes, auth, SSE, committed OpenAPI doc + drift test, vendored Swagger UI, `docs/` page | S2 | Not started |
| S4 | Auto scheduler and `/v1/schedule` | S3 | Not started |
| S5 | Docker image (no Qt), compose example, CI smoke job, GHCR publish on tag | S0, S3 (S4 for the schedule in the example) | Not started |
| S6 | Notifications: `notify` targets, events, typed payload in OpenAPI `webhooks`, test route | S4 | Not started |
| S7 | README "Pipeline service" section; `implemented.md` entry once built | S5, S6 | Not started |

Each chunk follows AGENTS.md: tests in the same commit, focused suite then
`uv run pytest src/test/python -n auto`, status here and in the index updated.

## 12. Decisions

Settled on 2026-09-26:

| # | Question | Decision |
|---|---|---|
| 1 | HTTP framework | FastAPI + pydantic v2; the pipeline's dataclasses stay pydantic-free |
| 2 | Repository writes over HTTP | Offered, off by default behind `allow_repository_writes` (§6.4); never from the schedule |
| 3 | Year filter shape | The ignore-rule expression string (`2026`, `>=2020`, `1990-1999`), one grammar shared by ignore rules, CLI and API (§3, §6.5) |
| 4 | Auth | One static bearer token (§6.7) |
| 5 | Notifications | In v1: outbound webhooks, chunk S6 (§9) |
| 6 | Image distribution | Built and pushed to GHCR by GitHub Actions when a tag is pushed ([docker.md](pipeline-service/docker.md)) |
| 7 | Qt in the image | Removed first, chunk S0 ([docker.md §10.1](pipeline-service/docker.md)) |
