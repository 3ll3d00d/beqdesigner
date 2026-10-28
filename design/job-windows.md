# Job windows — the times of day a job may run

**Architecture reference.** Built 2026-10-26 (TODO JW, agreed with the user that day); the plan, its steps and the
commits are in the [archive](archive/job-windows.md). On 2026-10-28 jobs and their executions were separated
([jobs and executions](jobs-and-executions.md), TODO JE): a run no longer pauses and resumes as one job, but stops,
and its job starts another execution when the window opens. This document says what a window does in that model.

## 1. Goal

A long run (a redesign of hundreds of titles takes hours on the designers) should use the machines only at the times a
person allows: overnight, say, and not while the house is watching a film. Each job may be given an **active window**,
the times of day it may run. Outside the window it does not start, and a run in progress **stops** between titles; its
job carries on with what was left when the window next opens. A service-wide **default window** applies to every job
given none.

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
    job, including one running.
  - `[]`: **any time**, whatever the default says.
  - a list of ranges: that window.
- The default (`GET`/`PUT /v1/job-window`, kept in `<state_dir>/job-window.json`, written atomically) is a list of
  ranges; `[]`, the initial value, means any time. A file that cannot be read is logged and taken as any time.

`pipeline/service/window.py` holds the value (`ActiveWindow`: `of()`, `is_open(t)`, `next_open(t)`, `next_close(t)`,
`texts()`) and `time_zone_name()`. It reads the wall clock a minute at a time, in an injected zone (the tests use a
fake daylight-saving zone, so they need neither `tzdata` nor a real wait): a range whose start a daylight-saving change
skips opens at the first minute after the change that is inside it, and an hour the change repeats is open both times.

## 3. What a window does

| | Outside its window |
|---|---|
| a one-off job | No execution is started (`JobBook.tick` starts one only while the window is open); `next_at` says when it will be. |
| a recurring job | It is not due until the window opens: `next_at` is its next interval moved to the window's next opening. |
| a queued execution | It waits, keeping its place in the queue (`ExecutionQueue.__next`); `waiting_until` says when its window opens. An execution waits like this only when it was queued behind another while its window was open. |
| a running run | It stops taking titles before the next one (`ExecutionControl.should_stop()`, which `run_stages`, the hand-off to another process's run, and the wait for the work directory's lease all poll). Titles being designed or extracted finish; one that only waits for a designer slot (prepared ahead, TODO IG) is not sent. It ends `stopped` (`stopped_because: window`), keeping the titles it had not started (`remaining`). |
| a running scan or accept | It finishes: both are short, and stopping either part-way leaves nothing worth resuming. |

- **Others run meanwhile.** The worker takes the first queued execution whose window is open now, in queue order;
  while every one waits it sleeps until the soonest opening, looking again at least every 30 s, and at once after a
  window or the default changes.
- **One execution runs at a time**, as before. A job waiting for its window holds neither the worker nor the work
  directory's lease, so another job, the work list or a command-line run can use them.
- **Carrying on.** A stopped run's job starts its next execution when the window opens, as a run of the titles it
  had not started (`executions.resumed()`: through the same stage, with the same `retry_failed` and `unattended`,
  without scanning first). A redesign sends every title back before it designs any (that loop polls only a cancel,
  not the window), so it carries on as a plain run of what it had not designed. An execution that stopped before it
  planned anything (waiting for the lease, or handed to the work list's run) leaves `remaining` unknown, and the next
  does the job's request again. One that had started every title before the window closed, or that stopped for another
  reason (unavailable titles in a row), ends as its result says, and the job is done.
- **Each night is an execution**, with its own progress, log and result; the job's progress counts all of them
  (`JobBook.progress`).
- **Joining a run (F5).** A run joins the run in progress only while its own window is open. When the host stops,
  the executions that joined it go back to the queue.
- **A recurring job** has its own window (null for the default). Its *busy* check (`submit(if_idle=True)`) counts
  only an execution running or able to run now, so one waiting for the night does not stop another job's ticks.
- **Restart.** At a stop, queued and running executions end `interrupted`; a one-off job with work left starts another
  at the next start, when its window is open.

## 4. Interface

| Route | |
|---|---|
| `POST /v1/jobs`, `PATCH /v1/jobs/{id}`, `/v1/revise` (with `run`) | `window`: `null` (the default), `[]` or ranges (§2); 422 for a bad range, or a window on `/v1/revise` without `run`. A changed window applies to the job's queued or running execution too. |
| `GET`/`PUT /v1/job-window` | The default: `{window, time_zone, open_now, next_change}` (when it next opens, or closes while open; null: never). |
| `Job` | `window` (as given) and `next_at` (when its next execution may start, the window applied). |
| `Execution` | `window` (its job's), `window_open` (now, with the default applied), `waiting_until` (a queued one whose window is closed), `stopped_because: window`. |
| events | a `state` event `stopping` (with `because`), then `stopped`; a `window` event when an execution's window is changed. |

A stop is not a notification; the execution that finishes the job is, once.

## 5. Browser

`components/WindowField.tsx`:

- **New job** (scan and run), **Edit…** and **Revise** (with *now*): a *Runs* field: *When the default allows (…)*, *Any time*, or
  *Only between* with the ranges typed a line (or comma) each. The default sends no `window`.
- **Status** lists each job with its window and next start; **a job's page** says *Runs* (its window in words) and
  *Next*, and *Edit…* changes the window while it is not done. **Executions**: a queued one says *waits for …*, a
  stopped one *its window closed*; a queued one is *Drop*ped, a running one *Cancel*led.
- **Status**: *Jobs run*: the default window with its time zone, open or closed now and until when; *Change…* edits it.
