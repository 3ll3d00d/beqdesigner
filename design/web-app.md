# The browser app — the pipeline service's web front end

**Document type:** Architecture reference — delivered behavior (2026-10-10). The plan it was built from, with its
chunk records and evidence, is [archive/web-review.md](archive/web-review.md). The review routes it calls are
[review-over-http.md](review-over-http.md); the user guide is "Review in a browser" in
[`docs/library/service.md`](../docs/library/service.md); the developer guide is
[`src/main/web/README.md`](../src/main/web/README.md). Section numbers are cited by source comments and kept stable.

## 1. Goal

A browser app, served by the pipeline service itself, giving a basic view of what the desktop work list and title page
do: the workload and its status, starting and following jobs, and deciding each title's design. Publishing and
committing accepted titles are offered behind the service's `allow_repository_writes`. Metadata editing, TMDB and
artwork, Revise/Reopen, projects, the profile and Review Folder stay in the desktop app; a title whose metadata is
incomplete says what is missing and holds Accept back.

## 2. Decision rules are shared

The rules and the write of a decision are the Qt-free `pipeline/library/decide.py` (`decide()`, `decision_blocked()`,
`offered_digest()`, `next_waiting_id()`), and the curves a design is judged on `pipeline/library/review_chart.py`
(`chart_curves()`). The desktop title page (`model/worklist_title_decide.py`, `model/worklist_title_text.py`) and the
service's review routes both call them, so the two cannot apply different rules. Where they say where to do something,
the caller passes its own words: `redo`, `run_hint` (the service names `POST /v1/jobs/run`) and `metadata_hint` (the
title page says its Metadata tab, the service says the app). `decide()` refuses with a kind -- `changed`, `blocked`,
`metadata`, `override` or `invalid` -- that the service maps to 409s and a 422, and the title page to a reload or a
message. Tests: `test_pipeline_library_decide.py`.

## 3. The review routes

`GET /v1/titles/{id}/review`, `GET /v1/titles/{id}/chart`, `POST /v1/titles/{id}/decision` and `GET /v1/review/next`
(API 1.3.0), the in-flight rule, the index refresh after a decision and the status's `repository_writes` /
`repositories_configured`: [review-over-http.md](review-over-http.md).

## 4. The app

- **Where:** `src/main/web/` -- React 19, TypeScript, Vite; React Router (data router, `basename` `/ui`); TanStack
  Query; uPlot for the chart. The API types are generated from `docs/schema/service.openapi.json` by
  `openapi-typescript` into `src/api/schema.d.ts` (committed) and called through `openapi-fetch`; the web CI job
  regenerates them and fails on any difference, so the app and the contract cannot drift. Request shapes come from the
  same types: a form holds `Partial<TitleFilter>` and `cleanFilter()` makes the request's.
- **Versions** -- the npm registry's latest on 2026-10-10, pinned exactly in `package.json`, installing together with no
  peer conflict:

  | Package | Version | | Package | Version |
  |---|---|---|---|---|
  | react, react-dom, @types/react, @types/react-dom | 19.3.0 | | vitest | 5.0.3 |
  | react-router | 8.4.0 | | jsdom | 30.1.2 |
  | @tanstack/react-query | 5.104.1 | | @testing-library/react | 16.3.3 |
  | uplot | 1.6.32 | | @testing-library/dom | 10.4.2 |
  | openapi-fetch | 0.17.0 | | @testing-library/user-event | 14.6.7 |
  | vite | 8.3.4 | | @testing-library/jest-dom | 7.0.1 |
  | @vitejs/plugin-react | 6.1.2 | | msw | 3.0.3 |
  | typescript | **6.0.3** | | eslint | 10.12.0 |
  | openapi-typescript | 7.13.0 | | typescript-eslint | 8.71.1 |
  | | | | eslint-plugin-react-hooks | 7.1.1 |

  - **TypeScript 6.0.3, not 7.0.2.** 7.0 is the native (Go) compiler; its package exports only its version and
    `unstable/*`, not the compiler API type generators print through. On 2026-10-10 `openapi-typescript` 7.13.0 and
    `@hey-api/openapi-ts` 0.99.0 both crashed under 7.0.2 on this schema (`ts.factory` undefined), and
    `typescript-eslint` 8.71.1 declares `typescript <6.1.0`. `openapi-typescript`'s peer range still says `^5.x`, so
    `package.json` has `"overrides": {"openapi-typescript": {"typescript": "$typescript"}}`. Move to 7 when both support it.
  - **Node 24 LTS** (`.nvmrc`, `engines.node >=24`): `react-router` 8 needs Node ≥22.22 and `jsdom` 30 ≥22.22.2 or ≥24.15.
  - **uPlot** is used only in `src/components/MagnitudeChart.tsx`, so its planned 1.7 release changes that file alone;
    take it, when released, in a commit of its own.
  - **MSW 3** moved `http`/`HttpResponse` to `msw/http` and renamed `onUnhandledRequest` to `onUnhandledFrame` (the old
    name is ignored silently: an unmatched request would only warn).
  - Dependabot proposes npm bumps weekly (`.github/dependabot.yml`, directory `/src/main/web`).
- **Served:** `pipeline.service.api.serve_ui()` serves the build at `/ui` (`--ui-dir`, `BEQ_SERVICE_UI`): its files,
  `index.html` for any other path under it (the app routes in the browser), a 404 for a missing file with an extension,
  nothing outside the build (resolved paths are checked to stay inside it), hashed `assets/` cached for good and
  everything else `no-cache`. `/` redirects to `/ui/`, or to `/docs` without the app. Without `--ui-dir`, `/ui` is a 404
  Problem saying how to build it; a folder with no `index.html` stops the service at start-up with a usage error. The
  page and its files need no token; every `/v1` call does. Tests: `test_pipeline_service_ui.py`.
- **Auth:** sign-in checks the token with `GET /v1/status` and keeps it for the tab (`sessionStorage`) or, when asked,
  the device (`localStorage`), in memory when storage refuses (`src/api/token.ts`). Every call reads it from there; a 401
  from any call signs out, back to sign-in with the page asked for kept (`?next=`, a path inside the app only).
- **Live events:** `src/api/events.ts` reads `GET /v1/jobs/{id}/events` with `fetch` and a streamed body, because
  `EventSource` cannot send the `Authorization` header. It resumes a dropped connection with `Last-Event-ID` (giving up
  after five retries in a row), and ends at the job's final state event or a clean end of the stream, which the service sends only
  for a job that is over. A 401 or 404 is not retried. The service sends the stream with `X-Accel-Buffering: no`, so nginx in front passes
  each event on as it comes. `useJobEvents()` keeps the last 500 and refreshes the job, jobs,
  status and titles queries when the job ends.
- **Screens** (`src/pages/`):
  - *Status*: the pipeline strip (titles per `needs`, each linked to the Titles list filtered to it), the job running
    now with live progress (title-stages, the stage and title in hand, ffmpeg's percent, time to go), how many wait, the
    designer and whether it answers, the schedule (Pause/Resume over `PUT /v1/schedule`, Run now, the last run and skip),
    the sources and their last error. Polls `/v1/status` every 10 s.
  - *Jobs*: history newest first, by state. A job's page: its state and times, the result (counts; failures,
    unavailable, failed earlier, not published, designed and skipped titles, each linked to its review), why the job
    itself failed, its log (live while it runs) and Cancel (Drop for a queued job; "Stopping after the title in hand" for
    a running one).
  - *New job*: a scan of chosen sources, or a run: the filter form (needs, search, source, kind, year, new since the
    last scan), `through`, scan first, retry failed, *Preview* (`POST /v1/plan`: the label, what would run with its
    stages, what would be skipped and why; marked stale when the choices change) and *Run*. Publish and commit are
    disabled, with the reason, until `/v1/status` says `repository_writes` and `repositories_configured`. Bulk accept is
    not offered: the app decides titles one at a time.
  - *Titles*: the filter form over the address bar (`filterToQuery`/`filterFromQuery`, so a list can be linked to),
    *Include titles that need nothing*, a table of 100 a page (title, year, needs, detail, confidence, designs, flags)
    linked to each review with the filter kept, *Review next waiting*, and *Publish accepted…* / *Commit published…*:
    a plan of the filter with `needs` publish (or commit) says how many, an inline confirmation names the repositories,
    and a run job through publish (or commit) is submitted and followed. Disabled with the reason as on New job.
  - *Review*: the title, its status, what it needs; the designs (rejected ones listed apart, with how many reasons),
    commentary of the one highlighted (or the decline), the playback chain and designer, the metadata and what it lacks,
    and the chart (§4.1). *Accept design N*, *Reject* and *Skip*, with keys A, R, S, and 1–9 to highlight a design; the keys
    stand aside while a text field has focus. A blocked decision is disabled and says why. Accepting a rejected design
    asks first, with the designer's reasons, and sends `override_rejection`. A refused decision shows the service's
    reason and the title as it is now. After a decision, or Skip, `GET /v1/review/next` in the same filtered list opens
    the next waiting title; with none, the page says so and stays.
- **Look:** colour tokens on `:root` with a dark set under `prefers-color-scheme: dark`; layouts collapse to one column
  on a narrow window.
- **Tests:** Vitest + Testing Library + MSW, against fixtures typed by the generated schema (`src/test/fixtures.ts`, so a
  fixture that no longer fits the interface fails typecheck): the token, the client and the event stream; sign-in; each
  screen, including a decision's 409, the override, the keys and publishing's confirmation. uPlot is replaced by a
  recorder in `review.test.tsx` (jsdom has no canvas); the shared setup stubs `ResizeObserver` and `matchMedia`.

### 4.1 The chart

`MagnitudeChart` draws `GET /v1/titles/{id}/chart?candidate=N`: the measured average and peak curves, dashed when a design
is applied, and the same after the design, solid -- as the desktop title page draws them. The x axis is log frequency
(uPlot `distr: 3`), so 0 Hz is dropped, and every series is resampled onto the first's frequencies (`chartData()`).

## 5. Build and delivery

- `package.json` scripts: `dev` (Vite at `/ui/`, proxying `/v1` to `BEQ_SERVICE_URL`, default `http://127.0.0.1:8080`),
  `build` (typecheck, then Vite into `dist/`), `test`, `typecheck`, `lint`, `gen:api`. `package-lock.json` is committed;
  `node_modules/` is ignored (and `dist/`, by the root `.gitignore`). The root `.gitignore` also ignores any `lib/`: the
  app has none.
- **CI:** the `web` job of `.github/workflows/test.yaml` (Node from `.nvmrc`): `npm ci`, the generated-types check,
  typecheck, lint, test, build. The workflow also runs when `src/main/web/**` or the OpenAPI document changes.
- **Image:** a native-platform `web` stage (`node:24-alpine`, as the docs-assets stage) builds the app; the runtime has
  it at `/app/ui` with `BEQ_SERVICE_UI` set, and no Node. `docker/smoke.py` checks the image serves `/ui/` and the script
  it loads, and reads a designed title's review over HTTP. `test_pipeline_service_docker.py` holds every base image to
  ECR Public.
- **Desktop:** unaffected; the PyInstaller bundle carries neither the web source nor the service.
