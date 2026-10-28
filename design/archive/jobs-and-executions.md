# Jobs and executions (TODO JE) — plan and completion record

**Completed 2026-10-28.** The delivered behaviour is the reference [design/jobs-and-executions.md](../jobs-and-executions.md)
(its §6 says where the build differs from the design); [pipeline service §5 and §8](../pipeline-service.md) and
[job windows](../job-windows.md) were rewritten to match. This records the request, the decisions, the steps and the
evidence.

## Request and decisions

On 2026-10-26, after the first windowed redesign, the user asked for the jobs section of the main page to list each job,
and for the Jobs tab to be the log of executions: a job (a definition with a schedule and a window) is paused, resumed and
edited; an execution is a record, the next scheduled one, or the one running (which can be cancelled). They also asked
that each execution say how it ran: titles by outcome, the designers and their versions, and stage times. The design was
written on 2026-10-27 with four questions taken as defaults; the user settled them on 2026-10-28:

| Question | Chosen |
|---|---|
| Keep the old routes beside the new ones? | No: a breaking rename (`/v1/jobs` lists definitions, `/v1/executions` is the log); API 2.0.0 |
| What cancelling a one-off's execution does to the job | It **removes** the job (the design's default was to pause it) |
| What can be edited | Any job's name and window; a recurring job's interval and run; a one-off's request is fixed |
| Migration on upgrade | The schedule becomes a recurring job (paused if it was); each old job an execution; a paused windowed run carries on as a one-off |

## Steps and commits

| Step | Commit | What |
|---|---|---|
| 1 | `d1932df` | `Job`/`JobManager`/`JobControl` renamed `Execution`/`ExecutionQueue`/`ExecutionControl`, `jobs.py` to `executions.py`; no change in behaviour |
| 2 and 4 | `6e9e68c` | `jobbook.py` (`JobBook`, `migrate`), executions stop instead of pausing (`halt`, `stopped_because`, `remaining`), the scheduler removed; the routes, models, notifier, OpenAPI document, `schema.d.ts`, user guide and image smoke |
| 5 | `d8ff20a` | The browser: Status's jobs, a job's page, the Executions log and an execution's page |
| 3 | `0b114e1` | Statistics: the pool's `designer_call` events, `Execution.titles`, `statistics.py`, `GET /v1/executions/{id}` `statistics`, `/titles`, a job's `outcomes`, *How it ran* in the browser |
| 6 | this commit | The references (pipeline service, job windows, web app, review over HTTP, design version, E2 runbook, pipeline README), API 2.0.0, and this record |

Steps 2 and 4 went in together because the old routes could not stand over the new model; the browser (5) was built
before the statistics (3) so the app worked again as soon as possible.

## Evidence

- Python: `PYTHONPATH=./src/main/python QT_QPA_PLATFORM=offscreen uv run pytest -n auto src/test/python`: 3,665 passed,
  1 skipped (after step 3). New and rewritten tests: `test_pipeline_service_jobbook.py` (one-offs carried on over
  windows, cancel removes, pause/resume/remove, recurring intervals, busy and designer-down skips, ending by itself,
  persistence and restart, migration), `test_pipeline_service_execution_windows.py`, `test_pipeline_service_work_windows.py`
  (the real executor), `test_pipeline_service_api_windows.py`, `test_pipeline_service_statistics.py` (the figures, the
  pool's reports, the queue's records, the real `run_stages` over a fake pool, and HTTP), and the API tests.
- Browser: `npx tsc --noEmit`, `npm run lint` and `npx vitest run` (173 passed), `npm run build`.
- Not yet observed on the deployed service: the migration of the live `schedule.json` and history, and a night's
  windowed redesign as separate executions of one job. Check both after the next deploy (the E2 record).
