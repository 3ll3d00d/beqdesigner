# Pipeline service — HTTP control plane, auto mode and Docker image

**Document type:** Architecture reference — delivered behavior.

This is the design of the pipeline service as it is built: `pipeline/service/`,
the Docker image in `docker/`, and the work-directory lease the service shares
with the work list and the CLI. The user guide is
[`docs/library/service.md`](../docs/library/service.md); the wire interface is
[`docs/schema/service.openapi.json`](../docs/schema/service.openapi.json). The
image and the Qt-free boundary it depends on are described in §10 below. Open work is in
[`TODO.md`](TODO.md) (its initial release milestone, and C1). Section numbers are cited by
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

A person may also decide a title over HTTP (§6.8): accept a design or reject the title, with the title page's rules.
Out of scope: editing metadata, revising and editing the profile over HTTP. The service also serves a browser app over
these routes at `/ui` ([web-app.md](web-app.md)).

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
without `retry_failed`, unless something the title depends on was unavailable
(`pipeline/library/failure.py`: a connection error, timeout or 5xx, a remote
filesystem errno, a media mount that is not there), which is reported as
`unavailable` and not remembered; `run.stop_after_unavailable` (default 3)
such titles in a row stop the run with `stopped` set and the rest in
`not_run`; a cancel leaves only whole titles done; publish and commit are
serialised.

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
    config.py                     # ServiceConfig: listen, auth, guards, the old schedule to migrate (§7)
    context.py                    # loads the profile per job; environment secrets
    jobbook.py                    # JobBook: the jobs, when each is due, the migration (§5, §8)
    executions.py                 # ExecutionQueue: one worker, FIFO queue, cancel, progress, history, joining (§5)
    statistics.py                 # how an execution ran, from its result and its titles' records (§5)
    work.py                       # what each kind of job does (§2)
    lease.py                      # the work-directory lease (§5.1)
    notify.py                     # outbound webhooks (§9)
    refresh.py                    # IndexRefresher: the index brought up to date after a decision (§6.8)
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
  `LibraryIndex` connection per request. Work goes to the `ExecutionQueue`'s
  single worker thread.
- **Profile reloads:** the profile and service config are read again at the
  start of each job, so an edit to the mounted file takes effect on the next
  job without a restart. A profile that fails to load fails that job with the
  loader's message; the service stays up.
- **Shutdown:** uvicorn re-raises the signal that stopped it, so the entry
  point handles SIGTERM itself: it stops accepting work, stops the running
  execution cooperatively, waits up to `shutdown_grace_seconds` (default 120, matched
  by the compose file's `stop_grace_period`) and exits 0.

## 5. Jobs and executions

A **job** is the definition of some work -- `scan`, `run` (a selection through a stage, optionally scanning first) or
`accept` -- done once or every N minutes, within its window; an **execution** is one continuous run of it. The model,
the rules and the interface are [jobs and executions](jobs-and-executions.md); the times of day are
[job windows](job-windows.md). One execution runs at a time, because the index, the queue directory and the git
working trees each have a single writer; further executions queue in FIFO order, except that a run may join the run in
progress (§5.1).

| Execution field | Type |
|---|---|
| `id`, `job_id`, `job_name`, `number` | UUID, its job, the job's name, which of its executions (1, 2, ...) |
| `kind` | `scan \| run \| accept` |
| `origin` | `api` (a one-off job) `\| schedule` (a recurring one) |
| `state` | `queued \| running \| succeeded \| failed \| cancelled \| stopped \| interrupted` |
| `stopped_because` | `window \| paused \| removed`, when `stopped` |
| `request` | the typed request it did |
| `submitted_at`, `started_at`, `finished_at` | RFC 3339 |
| `progress` | `{done, total, title, stage, id}` from `stages.Progress` |
| `joined_to` | the execution whose run it joined, if any |
| `result` | typed per kind (§6.5), present once finished |
| `error` | message, when it itself failed (profile unreadable, index refused) |
| `statistics` | how it ran (`GET /v1/executions/{id}` only): titles by outcome, designers, stage times |

- `failed` means it raised or `StagesReport.failed` is true (a title failed, a publish was refused, git refused). A
  title-level failure is in the result; it still reports every title it did.
- **Cancel** of a queued execution removes it; of a running one sets the flag `run_stages(should_cancel=)` polls
  between titles. It ends `cancelled` with `attempted`/`not_run` from the report, and a one-off job whose execution is
  cancelled is removed. A run told to stop (its window closed, its job paused or removed) ends `stopped`, keeping the
  titles it had not started (`remaining`) for its job's next execution.
- **Events:** `on_progress` and `on_event` feed a bounded per-execution ring buffer,
  redacted with the work list's redaction (`model/execution_events.py`),
  readable afterwards and streamed live (§6.1). ffmpeg reports every 0.5 s, so a
  title's ffmpeg updates are kept at most every 5 s (its first and last always),
  and the execution event that repeats each one is not kept; otherwise they push
  the title's own events out of the 500-event buffer within minutes. The same events also build a record per title
  (`Execution.titles`: queued, each stage's start and end, the `designer_call` the pool reports with its endpoint,
  release and version, how it ended), kept whole, which `statistics.py` makes the figures of.
- **History:** the jobs are in `<work_dir>/service/jobs.json` and the last `history_limit` (default 200) finished
  executions in `executions.json`, both written atomically. An execution found queued or running at start-up is recorded
  `interrupted`; its job, if a one-off with work left, starts another (the stages are idempotent, so it redoes only what
  is still needed). A job done or removed is forgotten once none of its executions is kept.

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
| `GET /ready` | → `Readiness` | profile loads, work dir writable, ffmpeg/ffprobe found, profile's designer registered; 503 otherwise; no auth. Also `designer_reachable` (`required: false`: reported, never unready) |
| `GET /v1/status` | → `ServiceStatus` | index counts (the `status --json` content), the running execution, how many wait, notification outcomes, `designer` (`DesignerStatus`: reachable, detail, checked_at, and each endpoint with its own `checked_at` and `designing`: the titles a run has out on it now, read from the run's `DesignerPool`, which `design_and_queue` tells through `pool.designing()`) |
| `GET /v1/titles` | query `TitleFilter` + `sort`/`descending` + `limit`/`offset` → `TitlePage` | the work list's table; sorted by a column (else work-list order; a missing year or confidence last), then paged |
| `GET /v1/titles/{id}` | → `Title` | 404 if unknown |
| `POST /v1/plan` | `RunRequest` → `PlanPreview` | dry run: what would run and what would be skipped and why |
| `GET /v1/jobs` | query `state`, `kind` → `JobList` | the jobs, newest first, each with its next and running execution |
| `POST /v1/jobs` | `JobCreate` (`kind`, `request`, `name`, `every`, `window`) → 201 `Job` | a one-off is started at once while its window is open; only a run through extract or design recurs (422); publish, commit and accept guarded (§6.4). A run with `redesign: true` and 1-500 `ids` (no other filter, through design) sends every one back (`revise_to` design, or extract; with one id and extract, `audio_stream` chooses the stream: `revise.choose_stream()`), all checked first, then designs them in one run; or with `out_of_date: true` (any other filter but ids) sends back every designed title out of date that is not accepted or published |
| `GET`/`PATCH`/`DELETE /v1/jobs/{id}` | → `Job` / `JobUpdate` → `Job` / → `Job` | one job (with its titles by outcome); its name and window, a recurring job's `every` and `request` (409 for a one-off's); remove it |
| `POST /v1/jobs/{id}/pause`, `/resume` | → `Job` | its running execution stops before its next title; 409 done or removed |
| `POST /v1/jobs/{id}/run` | → 202 `Execution` | a recurring job now; 409 while another execution runs or waits |
| `GET /v1/executions` | query `job`, `state`, `limit` → `ExecutionList` | the running one, each active job's `next`, and the history newest first |
| `GET /v1/executions/{id}` | → `Execution` | with its `statistics` |
| `GET /v1/executions/{id}/titles` | query `outcome` → `ExecutionTitles` | each title's outcome, stage times and designer |
| `POST /v1/executions/{id}/cancel` | → `Execution` | 409 if already finished; a one-off's job is removed |
| `GET /v1/executions/{id}/events` | → `text/event-stream` of `JobEvent` | Server-Sent Events: backlog then live, ends with the execution |
| `GET /v1/executions/{id}/log` | → list of `JobEvent` | the events after the fact |
| `POST /v1/notify/test` | `NotifyTest` → `NotifyOutcome` | sample events to one target (§9) |
| `GET /v1/titles/{id}/review` | → `Review` | §6.8; 404 unknown title, 409 no queue entry |
| `GET /v1/titles/{id}/chart` | query `candidate` or `candidates` (`all`, or indexes), `smoothing`, `responses`, `variant` (another design run, `variants.variant_view()`; its series carry `variant`) → `Chart` | measured curves, and after each design asked for (`ChartSeries.candidate`), with each one's response (`kind` filter) when asked; the main window's smoothing (`review_chart.SMOOTHING`) of all but the responses; 422 no such design, both `candidate` and `candidates`, an unknown smoothing or one a curve is too sparse for |
| `POST /v1/titles/{id}/decision` | `Decision` → `Review` | §6.8; 409 refused (why in `title`/`detail`), 422 invalid. An accept may name an earlier design run (`variant`): the digest is checked, the run made current, then accepted, and the current run put back if the accept is refused |
| `GET /v1/review/next` | query `after` + `TitleFilter` → `NextTitle` | 204 when no other title in the list waits |
| `GET /v1/metadata/fields` | → `MetadataField[]` | `metadata_edit.FIELD_SPECS`: the title page's fields, labels, tips and choices |
| `PATCH /v1/titles/{id}/metadata` | `MetadataEdit` → `Review` | `metadata_edit.changes_from_texts()` then `save_meta_changes()` (only the fields sent, merged into the entry on disk); 409 `In flight` or `Changed since it was read` (the `metadata_digest` differs), 422 not an edit; the index refreshed after |
| `POST /v1/titles/{id}/metadata/tmdb` | `TmdbLookup` → `TmdbProposal` | `tmdb_reload()`/`tmdb_fill()`, nothing saved; 409 no `TMDB_API_KEY`, 502 TMDB failed (the key redacted) |
| `GET /v1/titles/{id}/artwork` | → image | `entry_art_path()`; 404 none |
| `PUT /v1/titles/{id}/artwork` | `ArtworkChoice` → `Review` | `url` downloaded or `data_base64` stored in `<queue>/_art_cache/` (`artwork_edit`: png or jpeg only, 20 MB), `art_overridden`; 409 as metadata, 422 not an image |
| `DELETE /v1/titles/{id}/artwork` | query `digest` → `Review` | clears it; a later run may choose one again |
| `GET /v1/titles/{id}/heatmap.png` | query `candidate` → PNG | `spectrum.render_comparison()` with the default `HeatmapSpec` (the service has no Analyse Signal preferences); drawn on `visuals.RenderQueue` (202 + `Retry-After` until done), kept at `<work>/<id>/web/heatmap-<key>.png`, the key a digest of the stamps, filters and spec; a failed render a 422 once; 409 no work dir or extracted audio, 422 no such design |
| `GET /v1/titles/{id}/spectrogram` | query `candidate`, `filtered`, `max_hz` → bytes | `spectrum.spectrogram_grid()`: at most 2048 × 512 cells, each the loudest of its group, one byte over the 100 dB below the loudest (`SpectrogramGrid`, a 40-byte header); as the heatmap for the 202, cache and refusals |
| `GET /v1/titles/{id}/catalogue` | → `PublishedBeqs` | `catalogue_remote.CatalogueCache` (`<work>/catalogue/`, beqcatalogue's `compare/titles.json` and `compare/<key>.json`, a day's cache, conditional requests, the cached copy when down; `BEQ_CATALOGUE_URL`) and `catalogue_match` (`catalogue_target`, `same_track`); 502 unreachable with nothing cached |
| `GET /v1/titles/{id}/catalogue/{n}/chart` | query `smoothing` → `Chart` | `review_chart.published_curves()`: the BEQ's response (`filter`) and the measured curves under it; 404 no such entry, 422 smoothing |
| `POST /v1/titles/{id}/variants/{vid}/use` | `VariantChoice` (the Review's `digest`) → `Review` | `variants.use_variant()`: the earlier design run made current, the current one kept, the title pending; 409 in flight, changed, decided (reopen first) or no such run |
| `DELETE /v1/titles/{id}/variants/{vid}` | query `digest` → `Review` | `variants.discard_variant()`; 409 for the current run |
| `PUT /v1/titles/{id}/edited` | `EditedDesign` (the Review's `digest`, the `source` design, its filters) → `Review` | `variants.save_edited()`: a new current design run whose one design is the edit (`method` edited, `commentary.edited_from`), held to a designer's rules (`convert.specs_to_complete_filter`: types, frequency below fs/2, Q, gain, 10 sections) and realised at the entry's fs; the run's curves, designer and identity kept, its fit figures dropped; 409 in flight, changed, decided or no such design, 422 not a design |
| `GET /v1/titles/{id}/projects` | → `TitleProjects` | `projects.project_states()`/`badge()` (the desktop title page's): each project's file name, whether it exists, is flat, was edited since the design (`pipeline_filter_hash`), or could not be read |
| `POST /v1/revise` | `ReviseRequest` → `ReviseOutcome` | `adjustments` (contract 1.5, with design or extract) replace each title's `QueueEntry.adjustments` (ids `a1`…, `by` service, `at`, `reason`; `[]` none, absent kept), every redesign sending them again; `revise_entry()` for each id now (each refused alone: no such title, in flight, not designed, published without `allow_repository_writes`, already pending for review), the index refreshed; or with `run` (design or extract) one redesign run job, returned in `job` |
| `GET /v1/titles/{id}/streams` | → `TitleStreams` | the index's `audio_stream_details` in words (`describe_streams`), the one extracted and the preferred; a source that lists none has them read from the file now (`streams.probe_stream_details`, ffprobe: `probed`, not kept until one is chosen; the stream job keeps them, `index.set_audio_stream_details`); 409 a season, or the file not read |
| `PUT /v1/titles/{id}/stream` | `StreamChoice` → 202 `RunJob` | a redesign job to extract with `audio_stream`; 409 in flight or a season, 422 no such stream |
| `GET /openapi.json`, `/docs`, `/redoc` | | the interface and its try-it-out page |

Submitting returns `202` with the `Job` and `Location: /v1/jobs/{id}`. Errors
use one model, `Problem` (RFC 9457: `type`, `title`, `status`, `detail`, and
`errors` for field-level validation); FastAPI's own 422 responses are
re-shaped into it and its 422 schema is replaced.

A `JobEvent` is one of four shapes, chosen by `type` (a `oneOf` with a
discriminator in the schema; the SSE `event:` line is the same `type`). Each has
`seq`, `at` and `text` (what a person reads), only its own fields, and no nulls:

| `type` | Fields |
|---|---|
| `state` | `state` |
| `run_progress` | `done`, `total` (title-stages); `stage`, `title`, `title_id` while a title-stage is starting, absent once the run is over |
| `extract_progress` | `title`, `title_id`, `done_ms`; `total_ms` and `percent` when ffmpeg knows the length |
| `event` | `title_id`, `stage`, `kind` (a title's execution event; `text` is redacted) |

### 6.2 The filter

`TitleFilter` is `Selection` over the wire, field for field:

```json
{"needs": ["extract", "design"], "new_since_scan": false, "source": "films",
 "match": "alien", "ids": [], "kind": "movie", "year": "2026",
 "detail": "rolloff", "flags": ["Gone"]}
```

`detail` (text in the title's detail), `flags` (every one given), `declined`
(true: designs the designer declined, waiting for review; false: the rest) and
`out_of_date` (true: designs a redesign now could answer differently, by design
version and settings, [design-version.md](design-version.md); false: the rest)
are the titles table's; the CLI has no flag for them. `out_of_date` is judged
against the designer probe's recent answer on the read routes and asked again
when a job starts; while no designer that is up says its design version it is
a 409 (a job fails). Each title carries its `design_version`.

Every field is optional and they are ANDed; `{}` is every title. On
`GET /v1/titles` they are query parameters (`needs` repeated,
`?kind=movie&year=%3E%3D2020`); on the job routes the filter is the `filter`
member of the body. `include_done` (default false, like the work list's *All*
chip) applies to listing only. An unknown `source` is a 422 naming the
profile's sources.

### 6.3 Running a stage

```http
POST /v1/jobs
{"kind": "run", "request": {"filter": {"kind": "movie", "year": "2026"}, "through": "design",
                            "scan_first": true, "retry_failed": false}}
```

`through` is `extract | design | publish | commit`, with `plan_stages`'
meaning: extract and design as needed; publish/commit only titles a person
accepted. `scan_first` (default true) lists the sources before selecting.
A finished execution's `result` is a `RunResult` carrying the `StagesReport`
fields (`selected`, `extracted`, `cached`, `designed`, `design_cached`,
`failed[{id, message}]`, `failed_earlier`, `skipped[{id, title, reason}]`,
`published`, `publish_errors`, `committed`, `commit_error`, `cancelled`,
`attempted`, `not_run`, `counts`) as typed members.

### 6.4 Guards on writing to the repositories

`through: publish | commit` and a job of kind `accept` write to, and may push,
the catalogue repositories. They are refused with 403 unless the service
config has `allow_repository_writes: true`, and `commit` is further refused
unless the profile's `sync:` names the repositories. A recurring job can never go
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
  API's own semantic version (2.0.0 since jobs and executions replaced the schedule), bumped by hand on a change.
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
`/ready` are unauthenticated; `/docs`, `/openapi.json` and the browser app (`/ui` and its files) are readable without
a token (the calls made from them still need it). TLS is left to a reverse
proxy.

### 6.8 Review over HTTP

The four review routes (§6.1) decide one title with the app's own rules (`pipeline/library/decide.py`): the person's
decision needs only the token, and refuses a title a run has in hand or that changed since it was read. The design is
[review-over-http.md](review-over-http.md).

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
schedule:                        # an older service's schedule: made the recurring job at the first start (§8)
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

The app's Library Work List shares the profile and leaves its JRiver password out
the same way: a scan or run fills in a JRiver source that names a user and no
password from `JRIVER_PASSWORD[_<SOURCE>]`, else from the Preferences > JRiver
server with the same address and user (`model/worklist_profile.profile_with_secrets`).
That copy is never written back; the settings drawer edits the file as read.

## 8. Recurring jobs

A recurring job (`every` N minutes, `JobBook`) is how the service works unattended; any number of them may run on their
own intervals and windows ([jobs and executions](jobs-and-executions.md) §2.1).

- **An execution** of one is a `run` with `origin: schedule`, `unattended: true` and its filter, through `extract` or
  `design` only: a recurring job is refused anything else (no automatic publish). The migrated schedule's job has
  `scan_first: true` and `needs` forced to `[extract, design]`, as the schedule had: "needs extract or design" is
  exactly what is still to do, where `new_since_scan` would lose a failed or cancelled run's titles.
- **Due** `every` minutes after its last execution *finished*, within its window. **Busy:** if another execution runs
  or waits when it is due (one waiting for its window does not count), it is skipped and recorded (`last_skip: busy`)
  through an atomic idle-only submission in `ExecutionQueue`, and due again `every` minutes later; ticks never pile up.
- **Ends by itself:** a recurring job whose filter lists `ids` can never match more titles, so once an execution of
  those ids leaves none of them for later -- nothing `unavailable` or not reached, it was neither cancelled nor
  stopped, and nothing `failed` when `retry_failed` is on -- it is `done` and `ended` says why
  (`JobBook.__all_done`). A filter without ids keeps going.
- **Migration** (`jobbook.migrate`, at the first start of the release): `schedule.json` (else `service.yaml`'s
  `schedule`) becomes the recurring job *Extract & design new titles*, `active` if it was enabled and `paused` if not;
  the file is renamed `schedule.json.migrated`.
- **Designer down:** before a recurring execution through design, `DesignerProbe`
  (`pipeline/service/designer.py`) asks the profile's designer's `/health`
  (`http_binding.check_designer`; a `by_reference` designer must answer 1.2
  with `shared_root`, any other is down only for no answer or 502/503/504). A
  refused tick is skipped (`last_skip: designer unavailable: ...`), retried
  after `min(every, 5)` minutes, and `Notifier.designer_unavailable` sends
  one `failed` notification (with `job: null`) per outage. A run job through
  design is refused before anything runs; one through publish or commit is
  not, and its designs are reported `unavailable` (R1). `/ready` and
  `/v1/status` reuse an answer for 30 s.
- **Designer queue:** `register_declared_designers(queue_depth=)` multiplies
  each declared designer's `timeout` by `run.parallelism.design` (from
  `setup.register_designers` and the work list's
  `register_profile_designers`), because a one-at-a-time designer queues the
  other requests. Preferences designers are not scaled.
- **Failures:** an unattended run does not retry a remembered failure
  (including a failed extraction) until the source or settings change or
  `retry_failed`; each tick reports them as `failed_earlier`. A title that met
  an unavailable dependency is not remembered, so the next tick runs it again
  (`RunResult.unavailable`, and `stopped` when enough in a row ended the run);
  both make the job `failed` and are in the `failed` notification.

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
  A `web` stage (Node 24, native platform like the docs assets) builds the browser app from `src/main/web` with
  `npm ci && npm run build`; its `dist/` is `/app/ui`, served at `/ui` (`BEQ_SERVICE_UI`, `--ui-dir`;
  [web-app.md](web-app.md) §4). The runtime stage has no uv and no Node.
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
  entry exists. It also checks the image serves the browser app at `/ui/` (the page and the script it loads) and
  reads the designed title's `GET /v1/titles/{id}/review`. It runs in `.github/workflows/test.yaml` on every push (Linux
  only) and in `create-image.yaml` before publishing.
  `test_pipeline_service_docker.py` loads `docker/smoke.py` by path and runs
  the same fixture through the real local service, ffmpeg and HTTP designer
  without Docker.
- **Publishing:** on a pushed tag (the trigger `create-app.yaml` uses for the
  desktop release), `create-image.yaml` builds `linux/amd64` and `linux/arm64`,
  runs the smoke test, and pushes `ghcr.io/3ll3d00d/beqdesigner-pipeline:<tag>`
  with `GITHUB_TOKEN` (`packages: write`). `alpha` follows every release, so a
  container on `:alpha` moves on with a plain pull and recreate (Unraid's update);
  `latest` is applied only to a tag without a pre-release suffix
  (`-alpha`/`-beta`/`-rc`). The tags are `docker/image_tags.py`'s. The image carries OCI
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
| 9 | Reviewing over HTTP (2026-10-10, reversing §1's "it stays in the app") | Per-title accept/reject with the token alone, through the app's own rules (§6.8); metadata, revise and the profile stay in the app |
