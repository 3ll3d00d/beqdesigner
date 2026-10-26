# Job windows — the times of day a job may run

**Design, not built** (TODO JW). Agreed with the user on 2026-10-26. When it is built, the lasting behaviour moves to
[pipeline service §5](pipeline-service.md#5-jobs) and this file is archived.

## 1. Goal

A long run (a redesign of hundreds of titles takes hours on the designers) should only use the machines at the times
a person allows: overnight, say, and not while the house is watching a film. Each job may be given an **active window**,
the times of day it may run. Outside the window it does not start, and a run in progress **pauses** between titles and
carries on when the window next opens. A service-wide **default window** applies to every job given none.

## 2. The window

A window is a list of daily ranges, `"HH:MM-HH:MM"`, the same every day, in the **service's time zone** (the
container's `TZ`; `GET /v1/job-window` says which zone that is, so the browser shows it rather than converting).

- A range may cross midnight: `"22:00-06:00"` is open from 22:00 to 06:00 the next morning.
- Several ranges may be given; the window is open when any of them is. Overlapping ranges are allowed.
- The start is inside the range and the end is not (`01:00-07:00` is open at 01:00, closed at 07:00).
  `"00:00-24:00"` is the whole day; a range whose start and end are equal is refused (ambiguous).
- On a job, `window` is one of:
  - `null` or absent: **the default applies**, as it is when checked, so changing the default changes every such
    job, including one running or paused.
  - `[]`: **any time**, whatever the default says.
  - a list of ranges: that window.
- The default (`GET`/`PUT /v1/job-window`, kept in `<state_dir>/job-window.json`, written atomically like
  `schedule.json`) is a list of ranges; `[]` (the initial value) means any time, which is today's behaviour.

`pipeline/service/window.py` holds the value (`ActiveWindow`): parsing and validation, `is_open(t)`,
`next_open(t)` and `next_close(t)`, over the local time of an injected clock so the tests need no real time zone
or waiting. A daylight-saving change is handled by working in local wall-clock time: a range that names a time the
change skips opens at the first time after it; one it repeats is open once.

## 3. What a window does to a job

| Job | Outside its window |
|---|---|
| queued | It is not started; it stays `queued` with `waiting_until` (the next opening). |
| a running run | It stops taking titles at the next title boundary, like a cancel. Titles already in hand (an extraction, a design at the designer) finish. It is then `paused` until its window opens, and carries on with the titles it had not started. |
| a running scan or accept | It finishes: both are short, and stopping either part-way leaves nothing worth resuming. |
| paused | It stays `paused` with `resumes_at`. |

- **Others run meanwhile.** The worker takes the first queued or paused job whose window is open now, in queue
  order. A job waiting for its window keeps its place: it is taken before anything queued after it once its window
  opens and nothing is running. A paused job goes back to its place in the queue, not to the end.
- **One job runs at a time**, as now. A paused job is not running: it holds neither the worker nor the work
  directory's lease, so another job, the work list or a command-line run can use them.
- **States.** `paused` is a new state, not finished. A paused job can be cancelled (it then ends `cancelled` at
  once, with the report of what it did), and its window edited.
- **Resuming.** A run resumes as a run of the titles it had not started (`not_run` of the part it ran), through the
  same stage, with the same `retry_failed` and `unattended`, without scanning first. A redesign sends every title back
  before it designs any, so it pauses only after that, and resumes as a plain run of what it had not designed (the
  titles are already sent back; sending them back again would undo the ones it designed). A run handed to the work
  list's run (§5.1) withdraws its titles when it pauses and offers them again when it resumes.
- **One report.** A paused-and-resumed job is one job: its progress counts all its titles (`done` carries on from
  where the last part stopped), its log is one log, and its result merges the reports of its parts (lists joined,
  counts added; `cancelled` and `stopped` from the last part). The job's `parts` says how many parts it ran in, and
  `paused_at` lists each pause, for the log and the Job page.
- **Joining a run (F5).** A run job joins the run in progress only when its own window is open. When the host pauses,
  the jobs that joined it go back to the queue (as after a failure), each to wait for its own window.
- **The schedule** has no window of its own: the runs it submits carry none, so the default applies. Its *busy*
  check counts only a job running or able to run now: a job waiting for its window does not stop a tick, so a
  windowed redesign paused for the day does not hold up the schedule's extract/design of new titles. Those runs are
  joined to nothing that is paused, and run in their own (default) window.
- **Restart.** A job `paused`, or `queued` with a window that is closed, when the service stops is kept and carries
  on after the restart (its request is read back from the history; one that cannot be is recorded `interrupted` as
  now). A job that was `running` when the service stopped is still recorded `interrupted`: the stages are idempotent,
  so the next run does what it left. This keeps a windowed job through a container update.

## 4. Interface

| Route | Change |
|---|---|
| `POST /v1/jobs/scan`, `/v1/jobs/run`, `/v1/jobs/accept`, `/v1/revise` (with `run`) | Optional `window`: `null`, `[]` or ranges (§2). |
| `PUT /v1/jobs/{id}/window` | `{window}`: change a queued, running or paused job's window. 409 for a finished job; 422 for a bad range. A running run whose new window is closed pauses at its next title; a paused job whose new window is open is taken when the worker is free. |
| `GET`/`PUT /v1/job-window` | The default: `{window: [ranges], time_zone, open_now, next_change}`. |
| `Job` | Adds `window` (as given), `window_open` (now, with the default applied), `waiting_until` / `resumes_at` (or null), `parts`, `paused_at`; `state` gains `paused`. |
| events | `{type: state, state: paused, resumes_at}` and `{type: state, state: running, part}` on a resume. |

Notifications (§9): a pause is not a notification; the job's finish is, once, as now.

## 5. Browser

- **New job** and **Revise**: a *Runs* field: *Default (…)*, *Any time*, or ranges, entered as `01:00-07:00`
  lines.
- **Jobs** and **Job**: a paused or waiting job shows *Paused until 01:00* / *Waits for 01:00*; the Job page edits
  the window (the same field) and shows each part in its log.
- **Status**: the default window, with the time zone and when it next opens or closes; editable there.

## 6. Steps

1. `window.py` (`ActiveWindow`): parsing, `is_open`, `next_open`, `next_close`; tests over midnight, several ranges,
   the whole day, bad ranges and a daylight-saving change.
2. `JobManager`: the window on a job and the default, choosing the next runnable job, `queued` waiting, `paused`,
   resume with merged progress and result, window edits, cancel of a paused job, the schedule's busy check, keeping
   waiting and paused jobs across a restart. Tests with a fake `execute` and clock.
3. `work.py`: the pause polled by `run_stages` (and the hand-off) but not by the redesign's sending back; the resumed
   request. Tests through the real `run_stages` with a fake designer.
4. The routes and models, `service.openapi.json` and `schema.d.ts`; tests through the API.
5. The browser (§5), with its tests.
6. The references (pipeline-service §5, §6.1, §8; web-app), the TODO row archived.

## 7. Done when

A run given a window that closes part-way pauses between titles with nothing half done, frees the worker and the
lease for other jobs, and on the window's opening carries on to finish every title exactly once, with one progress,
one log and one merged result; a job waiting for its window does not hold up the queue or the schedule; the default
window applies to every job given none; and a paused job survives a service restart.
