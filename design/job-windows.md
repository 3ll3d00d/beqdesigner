# Job windows — the times of day a job may run

**Architecture reference.** Built 2026-10-26 (TODO JW, agreed with the user that day); the plan, its steps and the
commits are in the [archive](archive/job-windows.md). The rest of the job model is [pipeline service §5](pipeline-service.md#5-jobs).

## 1. Goal

A long run (a redesign of hundreds of titles takes hours on the designers) should use the machines only at the times a
person allows: overnight, say, and not while the house is watching a film. Each job may be given an **active window**,
the times of day it may run. Outside the window it does not start, and a run in progress **pauses** between titles and
carries on when the window next opens. A service-wide **default window** applies to every job given none.

## 2. The window

A window is a list of daily ranges, `"HH:MM-HH:MM"`, the same every day, in the **service's time zone** (the
container's `TZ`). `GET /v1/job-window` names the zone, and the browser shows ranges in it rather than converting.

- A range may cross midnight: `"22:00-06:00"` is open from 22:00 to 06:00 the next morning.
- Several ranges may be given (at most 24); the window is open when any of them is, and they may overlap.
- The start is inside the range and the end is not (`01:00-07:00` is open at 01:00, closed at 07:00).
  `"00:00-24:00"` is the whole day; a range whose start and end are the same is refused (ambiguous), as is anything
  that is not two times of day. Ranges are kept normalised (`1:00 - 7:00` is `01:00-07:00`).
- On a job, `window` is one of:
  - `null` or absent: **the default applies**, as it is when checked, so changing the default changes every such
    job, including one running or paused.
  - `[]`: **any time**, whatever the default says.
  - a list of ranges: that window.
- The default (`GET`/`PUT /v1/job-window`, kept in `<state_dir>/job-window.json`, written atomically) is a list of
  ranges; `[]`, the initial value, means any time. A file that cannot be read is logged and taken as any time.

`pipeline/service/window.py` holds the value (`ActiveWindow`: `of()`, `is_open(t)`, `next_open(t)`, `next_close(t)`,
`texts()`) and `time_zone_name()`. It reads the wall clock a minute at a time, in an injected zone (the tests use a
fake daylight-saving zone, so they need neither `tzdata` nor a real wait): a range whose start a daylight-saving change
skips opens at the first minute after the change that is inside it, and an hour the change repeats is open both times.

## 3. What a window does to a job

| Job | Outside its window |
|---|---|
| queued | It is not started; it stays `queued`, and `waiting_until` says when it next opens. |
| a running run | It stops taking titles before the next one (`JobControl.should_stop()`, which `run_stages`, the hand-off to another process's run, and the wait for the work directory's lease all poll). Titles in hand finish. It is then `paused` until its window opens, and carries on with the titles it had not started. |
| a running scan or accept | It finishes: both are short, and stopping either part-way leaves nothing worth resuming. |
| paused | It stays `paused`, with `waiting_until`. |

- **Others run meanwhile.** The worker takes the first queued or paused job whose window is open now, in queue
  order (`JobManager.__next`); while every one waits it sleeps until the soonest opening, looking again at least every
  30 s, and at once after a window or the default changes. A paused job goes back to its own place in the queue
  (`Job.order`), not to the end.
- **One job runs at a time**, as before. A paused job is not running: it holds neither the worker nor the work
  directory's lease, so another job, the work list or a command-line run can use them.
- **States.** `paused` is a state, not a finished one; `pausing` is an event while the part winds down. A paused job
  can be cancelled (it ends `cancelled` at once, with the merged report of what it did) and its window changed.
- **Resuming.** `jobs.resumed()`: a run carries on as a run of the titles it had not started (`not_run` of the part,
  from `work.remaining_titles`), through the same stage, with the same `retry_failed` and `unattended`, without
  scanning first. A redesign sends every title back before it designs any (that loop polls only a cancel, not the
  window), so it carries on as a plain run of what it had not designed. A part that stopped before it planned
  anything (waiting for the lease, or handed to the work list's run) carries on with its request as it was. A part that
  had started every title before the window closed, or that stopped for another reason (unavailable titles in a row),
  finishes the job as it would have.
- **One job, one report.** Its progress counts all its titles (`done` carries on from where the last part stopped:
  `Job.done_before`), its log is one log, and its result merges its parts' (`work.merge_outcomes`, over plain data so it
  works after a restart: lists joined, `attempted` once each, `selected` and `through` from the first part, `cancelled`,
  `stopped`, `not_run` and `counts` from the last). `parts` says how many times it started; `paused_at` lists each pause.
- **Joining a run (F5).** A run job joins the run in progress only while its own window is open. When the host pauses,
  the jobs that joined it go back to the queue, each to wait for its own window.
- **The schedule** has no window of its own: the runs it submits carry none, so the default applies. Its *busy* check
  (`submit(if_idle=True)`) counts only a job running or able to run now, so a windowed job waiting for the night does
  not stop the schedule's ticks.
- **Restart.** At a stop, a queued job whose window is open is cancelled as before, but a `paused` job, or a `queued`
  one whose window is closed, is left as it is; at the next start its request is read back from the history
  (`jobs.request_from_dict`) and it waits again. One that cannot be read back is recorded `interrupted`. A job that was
  `running` when the service stopped is cancelled (a graceful stop) or `interrupted` (otherwise), as before.

## 4. Interface

| Route | Change |
|---|---|
| `POST /v1/jobs/scan`, `/v1/jobs/run`, `/v1/jobs/accept`, `/v1/revise` (with `run`) | Optional `window`: `null`, `[]` or ranges (§2); 422 for a bad range, or a window on `/v1/revise` without `run`. |
| `PUT /v1/jobs/{id}/window` | `{window}` (required; `null` is the default): change a queued, running or paused job's window. 404 unknown, 409 finished, 422 bad. |
| `GET`/`PUT /v1/job-window` | The default: `{window, time_zone, open_now, next_change}` (when it next opens, or closes while open; null: never). |
| `Job` | Adds `window` (as given, also in `request`), `window_open` (now, with the default applied), `waiting_until` (a queued or paused job whose window is closed), `parts`, `paused_at`; `state` may be `paused`. |
| events | `state` events `pausing`, `paused` (with `resumes_at`) and `running` with `part` on a resume; a `window` event when a job's window is changed. |

A pause is not a notification; the job's finish is, once, as before.

## 5. Browser

`components/WindowField.tsx`:

- **New job** (scan and run) and **Revise** (with *now*): a *Runs* field: *When the default allows (…)*, *Any time*, or
  *Only between* with the ranges typed a line (or comma) each. The default sends no `window`.
- **Jobs**: a waiting or paused job says *waits for …* / *until …* by its state. **Job**: *Runs* (its window in words,
  and when it waits until), *Change…* while it has not finished, its parts and pauses; a queued or paused job is
  *Drop*ped, a running one *Cancel*led.
- **Status**: *Jobs run*: the default window with its time zone, open or closed now and until when; *Change…* edits it.
