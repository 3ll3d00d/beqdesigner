# Job windows (TODO JW) — plan and completion record

**Completed 2026-10-26.** The delivered behaviour is the reference [design/job-windows.md](../job-windows.md); this
records the request, the decisions and how it was built.

## Request and decisions

The user asked (2026-10-26) to give each job, including a running one, an active schedule: the times of day it may run.
The four decisions, the user's:

| Question | Chosen |
|---|---|
| A running job at its window's end | Pause between titles and resume when the window next opens, as the same job |
| The window's shape | Daily time ranges, the same every day (not per weekday), in the service's time zone |
| A queued job waiting for its window | Others run meanwhile; it keeps its place |
| Where a window is set | On each job when submitted; editable on a queued, running or paused job; a service-wide default (not on the automatic schedule, whose runs follow the default) |

Added on the way, as defaults: a paused job, or one waiting for its window, is kept across a restart (otherwise every
container update would lose it); a scan or accept running when its window closes finishes.

## Steps and commits

| Step | Commit | What |
|---|---|---|
| Design | `25811df` | The design (then "design, not built") and the TODO row |
| 1 | `63ff428` | `pipeline/service/window.py`: ranges, open/next open/next close, daylight saving; `test_pipeline_service_window.py` (38) |
| 2 | `48f3492` | `JobManager`: waiting, pausing, resuming as one job, the default, window edits, cancel of a paused job, the schedule's busy check, restart; `test_pipeline_service_job_windows.py` (12) |
| 3 | `7aadb61` | `work.py`: the pause polled by `run_stages`, the hand-off and the lease wait, not by a redesign's sending back; `remaining_titles`, `merge_outcomes`; `test_pipeline_service_work_windows.py` (4) |
| 4 | `c4fc3c2` | Routes, models, events, OpenAPI, `schema.d.ts`, the user guide; `test_pipeline_service_api_windows.py` (9) |
| 5 | `96bbd4a` | The browser: `WindowField.tsx`, New job, Revise, Jobs, Job, Status; `windows.test.tsx` (7) and a Revise test |
| 6 | this commit | The reference, pipeline-service §5/§8, web-app, the archive |

## Evidence

- Python: `PYTHONPATH=./src/main/python QT_QPA_PLATFORM=offscreen uv run pytest src/test/python -n auto`: 3636 passed,
  1 skipped (at step 4; step 5 changed only the browser).
- Web: `npx vitest run` 165 passed; `npm run typecheck`, `npm run lint`, `npm run build` clean (step 5).
- Deviations from the design as written: the `Job` model's `resumes_at` was folded into `waiting_until` (one field for
  a queued or paused job); a `pausing` state event and a `window` event were added; a running job at a graceful stop is
  cancelled, as before, not interrupted.
