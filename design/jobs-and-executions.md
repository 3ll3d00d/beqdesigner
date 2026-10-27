# Jobs and executions

**Design, not built** (TODO JE). Agreed with the user on 2026-10-27. It replaces the service's single schedule and its
one-record-per-request jobs with two things: a **job**, the definition of some work, and its **executions**, the runs
that do it. When built, the lasting behaviour moves to [pipeline service §5 and §8](pipeline-service.md) and
[job windows](job-windows.md), and this file is archived.

## 1. Why

The service conflated the two. The automatic schedule was a definition (what, how often) with no window and only one of
it; every submitted request was a record that was at once the definition (its filter, its window) and the execution
(its state, progress and log), and a windowed run's nightly parts were executions hidden inside it. So the main page
could not say "these are the jobs; this one is paused; this one runs tonight", and the Jobs page mixed queued
definitions with the log of what ran.

## 2. The model

### 2.1 A job

The definition of some work, kept until it is removed.

| Field | Meaning |
|---|---|
| `id`, `name` | A UUID, and a name a person gives it (a default from what it does: "Redesign 500 titles"). |
| `kind`, `request` | `scan`, `run` (through a stage, optionally a redesign) or `accept`, with the request as today. |
| `recurrence` | `once`, or `every` N minutes (at least 5). |
| `window` | As [job windows §2](job-windows.md#2-the-window): null for the default, `[]` any time, or ranges. |
| `state` | `active`, `paused` (by a person), `done` (a one-off with nothing left, or a recurring job that ended by itself), `removed`. |
| `next_at` | When its next execution may start: the window's next opening, the next interval, or null (paused, done, or waiting only for the worker). |
| `progress` | A one-off run: titles done of all it set out to do, over every execution so far. |
| `remaining` | A one-off run: the titles its executions have not yet started (what the next one does). |
| `last_execution`, `running_execution` | The ids, for the page. |

- **Every task is a job.** A scan, a run, a redesign or a bulk accept submitted for once is a one-off job; the
  schedule is a recurring job. There may be any number of recurring jobs, each with its own request, interval and
  window (an extract/design of new titles every 30 minutes, a nightly redesign of whatever is out of date).
- **A one-off job** is executed until it has done everything: an execution that stops because the window closed
  leaves `remaining`, and the next execution, when the window next opens, does those titles. When an execution ends
  with nothing remaining, or a scan or accept finishes, the job is `done`.
- **A recurring job** is executed every `interval` minutes after its last execution finished, within its window; a tick
  due while another execution runs or waits is skipped and taken at the next chance (as the schedule's *busy* was).
  As the schedule did, a run through design is skipped while the designer is down (retried within 5 minutes, one
  notification per outage), and a recurring job whose filter lists ids ends itself (`done`) once an execution leaves
  none of them to do. Recurring jobs are runs through extract or design only (no automatic publish, as before).
- **Pause** (a person): the job is `paused`. Its running execution, if any, stops before its next title (designs in
  progress finish) and ends `stopped` (§2.2); nothing more starts. **Resume**: `active` again; a one-off carries on with
  `remaining`, a recurring job's next execution is due at once (within its window).
- **Edit**: the name and window of any job not done; for a recurring job also its interval and request (filter,
  through, retry failed), which apply from its next execution. A one-off job's request is fixed once it has started
  (what is left is `remaining`). A running execution follows a window change at its next title, as now.
- **Run now** (a recurring job): one execution now, if nothing else is running or waiting to run.
- **Remove**: a job that is not running is removed (`removed`, kept in the history for its executions); a running one
  is paused first.

### 2.2 An execution

One continuous run of a job: from when it starts until it finishes, fails, is cancelled or stops.

| Field | Meaning |
|---|---|
| `id`, `job_id`, `number` | A UUID, its job, and which of the job's executions it is (1, 2, ...). |
| `request` | What it did: for the first execution of a one-off, the job's request; for a later one, the titles that were remaining. |
| `state` | `queued`, `running`, `succeeded`, `failed`, `cancelled`, `stopped`, `interrupted`. |
| `stopped_because` | `window` (its window closed) or `paused` (its job was). |
| `started_at`, `finished_at`, `progress`, `result`, `error`, the log | As a job has them today, each for this execution only. |

- An execution is a historic record once it ends. **Cancel** stops a running execution before its next title (it ends
  `cancelled`); a one-off job whose execution was cancelled is then `paused`, so it does not start again at once — the
  person resumes or removes it. A recurring job's next execution is due at its interval as usual.
- The **log** of executions (the Jobs page, renamed Executions) lists the running execution, the **next scheduled
  execution** of each active job (computed from `next_at`, not stored), and the history, newest first.
- One execution runs at a time, as now; joining the run in progress (§5.1) is between executions as it was between jobs.
- A notification is sent when an execution ends `succeeded` or `failed` (as for a job today), not when it stops; a
  one-off job being `done` is the last one's notification.

### 2.3 What an execution says about how it ran

Each execution keeps **statistics**, so its page answers "how did it go" at a glance, without reading the log:

- **Titles by outcome**: designed, declined by the designer, failed (extraction or design, with the reasons grouped
  and counted), unavailable, skipped (grouped by reason), already done (extraction or design cached), and not reached
  (stopped or cancelled); each count opens the list of titles.
- **Designers**: for each endpoint that designed for it -- its URL, release, design version and settings -- how many
  designs it made, declined and failed, and their times (mean, p50, p90, max). A run whose endpoints reported different
  releases or versions says so.
- **Timing**: wall time from start to end, titles an hour, and for each stage -- extraction, waiting for a design slot,
  the design at the designer -- mean, p90 and max, with the slowest titles listed.
- **A job** shows the same, summed over its executions (a one-off job's progress, a recurring job's last N runs).

The log's 500 kept events cannot hold this for a run of hundreds of titles, so it is gathered as the run goes into the
execution's own record (a small entry per title: its outcome, each stage's start and end, and the endpoint that
designed it), kept with the execution in `executions.json`, and the statistics are computed from those. The endpoint
comes from the designer pool, which reports each design call (title, endpoint, its release and version from its last
`/health`, start, end, outcome) to the run (`DesignerPool`, through `run_stages`' events), since neither the queue entry
nor today's events record which endpoint designed a title.

## 3. The service

- Internally the present `Job`/`JobManager` become `Execution`/`ExecutionQueue` (the queue, the worker, cancel, the log,
  history, joining, the lease); a new `JobBook` (`pipeline/service/jobbook.py`) keeps the jobs, decides when each is
  due, and submits its executions, replacing `AutoScheduler`. The window logic of [job windows](job-windows.md) moves
  from a job's pause and resume to "an execution stops when the window closes; the job is due when it opens".
- **Kept in the state directory**: `jobs.json` (every job, atomically written) and `executions.json` (the last
  `history_limit` executions). At start-up an execution found running is `interrupted`, as now, and its one-off job
  carries on with what it had not done.
- **Migration** (once, at the first start of the release): `schedule.json` becomes a recurring job named
  "Extract & design new titles" with its interval, filter, through and retry failed, `active` if the schedule was
  enabled and `paused` if not, and no window of its own; the old `jobs.json` becomes the execution history, each
  record an execution of a one-off job made for it (`done`, or for one still `queued` or `paused`, `active` with its
  `remaining` and window), so a windowed redesign in progress carries on across the upgrade.

## 4. Interface

The routes are renamed to the model: this is a breaking change to `/v1/jobs` (it lists definitions now), made while
the service is an alpha; scripts that submit work move from `POST /v1/jobs/run` to `POST /v1/jobs` with `kind: run`.

| Route | Does |
|---|---|
| `GET /v1/jobs` | The jobs (`?state=`), with each one's next execution and running execution. |
| `POST /v1/jobs` | Make a job: `{name?, kind, request, recurrence, window}`; a one-off is queued at once. |
| `GET`/`PATCH`/`DELETE /v1/jobs/{id}` | One job; edit it (§2.1); remove it. |
| `POST /v1/jobs/{id}/pause`, `/resume`, `/run` | Pause, resume, run now (recurring). |
| `GET /v1/executions` | The log: running, next scheduled, history (`?job=`, `?state=`). |
| `GET /v1/executions/{id}`, `/log`, `/events` | One execution (with its statistics, §2.3), its log, and following it (as a job's are today). |
| `GET /v1/executions/{id}/titles` | Its titles, each with its outcome, stage times and designer (`?outcome=`). |
| `POST /v1/executions/{id}/cancel` | Cancel a running or queued execution. |
| `POST /v1/plan` | Unchanged: what a run would do. |
| `POST /v1/revise` with `run` | Makes a one-off redesign job, and returns it with its first execution. |
| `GET`/`PUT /v1/job-window` | Unchanged: the default window. |
| removed | `/v1/jobs/scan`, `/v1/jobs/run`, `/v1/jobs/accept`, `/v1/jobs/{id}/window`, `/v1/schedule`, `/v1/schedule/trigger`. |

`/v1/status` reports `running_execution` and the count of executions waiting, in place of `current_job` and `queued`,
and no longer a `schedule`.

## 5. Browser

- **Status (the main page)**: its *Jobs* section lists the jobs, one row each: name, what it does, when (once, or every
  N min) and its window, state, next execution, and the running execution with its progress and *Cancel*. Each row has
  *Pause*/*Resume*, *Edit…*, *Run now* (recurring) and *Remove*; *New job…* above. The default window stays here.
- **Executions** (the Jobs tab, renamed): the log — the running execution, each active job's next scheduled
  execution, and every past execution, filterable by job and state, each row with its counts (designed, declined,
  failed) and wall time. An execution's page leads with its statistics (§2.3): the outcome counts, a table of designers
  (endpoint, release, version, designs, declined, failed, mean, p90, max), the stage times and the slowest titles;
  then its result lists, its log (live while it runs) and *Cancel*, with a link to its job. Live while it runs: the
  counts and times update as titles finish.
- **New job / Edit job**: what (scan, run, accept), the filter and preview as today, *When* (once / every N minutes),
  the window field, and a name.
- **Revise** with *now* makes a one-off redesign job, as above.

## 6. Steps

1. Rename `Job`/`JobManager` to `Execution`/`ExecutionQueue` internally, no behaviour change.
2. `JobBook`: jobs, recurrence, windows (an execution stops at the window's end; no more pause/resume inside one),
   pause/resume/edit/run now/remove, cancel's effect on a one-off, the designer-down skip and ending by itself,
   persistence, the migration. Replaces `AutoScheduler`. Tests with a fake execute and clock.
3. Statistics (§2.3): the pool's per-call reports, the per-title records, the figures, persistence; tests over a
   fake pool and the real `run_stages`.
4. The routes and models (§4), `/v1/status`, the notifier, the OpenAPI document and `schema.d.ts`, `docker/smoke.py`
   and the user guide.
5. The browser (§5).
6. The references (pipeline-service §5, §8, job windows, web-app), the archive.

## 7. Done when

The main page lists every job and pausing, resuming, editing or removing one does what §2.1 says; the Executions page
is the log of every execution, with each job's next one, and each execution's page says how it ran (titles by outcome,
designers with their versions and counts, stage times as mean, p90 and max); a windowed one-off run is one job whose nights are separate
executions; several recurring jobs run on their own intervals and windows; and an upgrade keeps the schedule as a
recurring job and a windowed redesign in progress as an active one-off job.
