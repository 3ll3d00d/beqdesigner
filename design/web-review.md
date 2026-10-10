# Web review — a browser front end for the pipeline service

**Document type:** Design, not built except W1–W3 (built: §2 here, [review-over-http.md](review-over-http.md), and the
scaffold, serving and build of §4–§5). Agreed on 2026-10-10; the chunks are W1–W6 in
[TODO](TODO.md). When a chunk lands, move what it built into [pipeline-service.md](pipeline-service.md) (or a
reference of its own) and shrink this file; delete it when W6 lands.

## 1. Goal

The pipeline service (`pipeline/service/`, [pipeline-service.md](pipeline-service.md)) gets a browser app, served by the
service itself, that gives a basic view of what the desktop work list and title page do:

1. **Workload and status** -- the index counts by what each title needs, the sources and their last scan, the designer's
   reachability, the schedule, the current job and the queue, followed live.
2. **Trigger jobs** -- scan; run a filtered selection through a stage, previewed first with `POST /v1/plan`; cancel; run or
   pause the schedule.
3. **Review** -- for one title: its candidates (and the designs the designer rejected), commentary, the chart of the
   measured curves before and after the highlighted candidate, and **Accept** / **Reject** / **Skip** (go to the next
   waiting title).
4. **Publish and commit** accepted titles, through the existing run job `through: publish | commit`, which stays refused
   unless `service.yaml` has `allow_repository_writes: true` (pipeline-service.md §6.4).

Out of scope (stays in the desktop app): editing metadata, TMDB reload and artwork, Revise/Reopen, opening projects,
editing the profile, Review Folder. A title whose metadata is incomplete shows what is missing and Accept is disabled
with that reason, as on the title page.

This reverses pipeline-service.md §1's "reviewing titles over HTTP ... stays in the app" and the initial release
milestone's "results reviewed from the desktop app". Both texts change when W2 lands.

## 2. Decision rules are shared, not copied

**Built (W1).** The rules for a decision used to live in the Qt mixin `model/worklist_title_decide.py`
(`TitleDecisions._decide`) and the pure helpers in `model/worklist_title_text.py`. The parts that do not touch widgets are
now the Qt-free `pipeline/library/decide.py` and `pipeline/library/review_chart.py`, which the title page calls and the
service will:

- `ACCEPTABLE`, `REJECTABLE`, `STATUS_WORDS`, `next_waiting_id()`, `sent_back()`, `REDO_IN_*` and `decision_blocked()`
  moved there unchanged, except that `decision_blocked()` takes `run_hint` (default "Run it from the work list.") so the
  service can say where a run is started; `model.worklist_title_text` re-exports them.
- `offered_digest(entry)` -- a stable hash of `entry.offered` (and the entry's `fs`), what the page "saw".
- `decide(queue_dir, title_id, decision, *, seen_digest, picked, row, running, revised, redo, run_hint, meta_defaults,
  override_rejection)` is the write half of `_decide`: read the entry again; refuse if its status is not in `ACCEPTABLE`/`REJECTABLE`, its
  digest differs from `seen_digest`, `picked` is out of range, `decision_blocked()` says so, the metadata is incomplete
  (accept), or `picked` is a rejected design and `override_rejection` is false; otherwise `update_entry()`. It returns
  the written entry or raises `DecisionRefused(reason, kind)`, where `kind` is `changed | blocked | metadata | override | invalid`
  (the service maps them to 409, 409, 409, 409 and 422; the title page shows a `blocked` reason and reloads for the rest).
  The title page still asks its own questions first (saving an edit, the override confirmation, its page-level
  `decision_blocked`), then calls `decide(..., override_rejection=True)`.
- `review_chart.chart_curves(entry, picked)` returns `ChartCurve(kind: average|peak, filtered, data: MagnitudeData)`,
  named for a legend; `chart_data` colours them for the desktop, and the service (W2) sends `x`/`y` as arrays.

`test_qt_free_modules.py` finds both modules on disk, and now names `model.iir`, `model.xy` and `model.codec`, which
`chart_curves` imports lazily. `test_pipeline_library_decide.py` covers each refusal of `decide()` and the curves without
Qt; the queue-entry fixture moved from `gui/` to `src/test/python/review_entry_fixture.py` so it can be shared.

## 3. HTTP additions (W2) — built

Delivered as API 1.3.0 and described in [review-over-http.md](review-over-http.md): `GET /v1/titles/{id}/review`,
`GET /v1/titles/{id}/chart`, `POST /v1/titles/{id}/decision`, `GET /v1/review/next`, `ServiceStatus.repository_writes`
and `repositories_configured`, the in-flight rule, and the index refresh after a decision. Two changes from the plan
above: `decide()` gained a `metadata` refusal kind (409 *Metadata incomplete*), so the service does not call incomplete
metadata a change; and `commit_configured` became `repositories_configured`, since publish needs the repository too.

## 4. The app (W3–W5)

- **Where:** `src/main/web/` -- React, TypeScript, Vite; React Router; TanStack Query for fetching and cache; uPlot for
  the chart (log-frequency x axis, a small canvas library). The API types are generated from
  `docs/schema/service.openapi.json` with `openapi-typescript` into `src/main/web/src/api/schema.d.ts` (committed) and
  called through `openapi-fetch`; a check in the web CI job regenerates the types and fails on any difference, so the app
  and the contract cannot drift.
- **Versions** (the npm registry's latest on 2026-10-10, each pinned exactly in `package.json`, checked to install together
  with no peer conflict):

  | Package | Version | | Package | Version |
  |---|---|---|---|---|
  | react, react-dom, @types/react, @types/react-dom | 19.3.0 | | vitest | 5.0.3 |
  | react-router | 8.4.0 | | jsdom | 30.1.2 |
  | @tanstack/react-query | 5.104.1 | | @testing-library/react | 16.3.3 |
  | uplot | 1.6.32 | | @testing-library/dom | 10.4.2 |
  | openapi-fetch | 0.17.0 | | @testing-library/user-event | 14.6.7 |
  | vite | 8.3.4 | | @testing-library/jest-dom | 7.0.1 |
  | @vitejs/plugin-react | 6.1.2 | | msw | 3.0.3 |
  | typescript | **6.0.3** (not 7.0.2, below) | | eslint | 10.12.0 |
  | openapi-typescript | 7.13.0 | | typescript-eslint | 8.71.1 |
  | | | | eslint-plugin-react-hooks | 7.1.1 (added in W3) |

  - **TypeScript 6.0.3, not 7.0.2.** 7.0 is the native (Go) compiler. Its package exports only its version and
    `unstable/*` APIs, not the compiler API that type generators print code through. Tried on 2026-10-10 against this
    schema: `openapi-typescript` 7.13.0 and `@hey-api/openapi-ts` 0.99.0 both crash under 7.0.2
    (`ts.factory` is undefined), and `typescript-eslint` 8.71.1 declares `typescript <6.1.0`. 6.0.3 is the newest
    release that every tool here runs on; `openapi-typescript` generates and type-checks the schema with it. Its peer
    range still says `^5.x`, so `package.json` has `"overrides": {"openapi-typescript": {"typescript": "$typescript"}}`.
    Move to 7 when `typescript-eslint` and `openapi-typescript` support it.
  - **Node 24 LTS** (24.21.0 on 2026-10-10). `react-router` 8 needs Node ≥22.22 and `jsdom` 30 needs ≥22.22.2 or
    ≥24.15, so the 22.18 found on the development machine is too old (`nvm install 24`). `package.json` sets
    `engines.node` to `>=24` and `.nvmrc` to `24`.
  - uPlot 1.6.32 (2025-03) is the latest on npm; a large 1.7 release is planned (no pre-release published as of
    2026-10-10). Keep uPlot behind the one chart component so the move to 1.7 touches only that file; take it when it is
    released, in a commit of its own.
  - W3 adds an `npm` entry (directory `/src/main/web`, weekly) to `.github/dependabot.yml`, which today covers only
    GitHub Actions and uv, so these are kept current by reviewed bumps that re-run the web CI job.
- **Served:** the service mounts the built `dist/` at `/ui` (`--ui-dir` / `BEQ_SERVICE_UI`; absent, `/ui` is a 404 that
  says how to build it) with an SPA fallback to `index.html`, and `/` redirects to `/ui/`. The page and its assets need no
  token; every `/v1` call does.
- **Auth in the browser:** a sign-in screen takes the token and keeps it in `sessionStorage` ("remember on this device"
  puts it in `localStorage`). A 401 returns to sign-in. Live events use `fetch` with a streamed body reader (EventSource
  cannot send the `Authorization` header), resuming with `Last-Event-ID`.
- **Screens:**
  - *Status* (W4): the pipeline strip (count per `needs`), sources, designer, schedule (Run now, Pause/Resume), current
    job with live progress (title, stage, ffmpeg percent), queue length; polls `/v1/status` every 10 s and follows the
    current job's events.
  - *Jobs* (W4): history newest first with state filter; a job's page shows its request, result counts, failures, and
    its log (live while running); Cancel. *New job*: scan, or run with a filter form (needs, source, kind, year, search)
    and `through`, showing `/v1/plan`'s preview before submitting.
  - *Titles* (W5): the filter form, a paged table (title, year, needs, detail, confidence, candidates, flags) linked to
    review.
  - *Review* (W5): header (title, year, status, position in the filtered list), candidate list (rejected designs
    separately, with their reasons), commentary, chart for the highlighted candidate, metadata problems, and Accept /
    Reject / Skip with keyboard shortcuts A / R / S. Accepting a rejected design asks first and sends
    `override_rejection`. A 409 shows its reason and reloads the title. After a decision it goes to `/v1/review/next`.
    *Publish accepted* / *Commit published* buttons submit run jobs through publish/commit for the filter's accepted
    titles, disabled with the reason when `repository_writes` or `repositories_configured` is false.
- **Tests:** Vitest + React Testing Library + MSW against fixtures shaped by the generated types: each screen's
  rendering, the decision flow including 409 and override, the token flow and the event-stream parser. The Python side
  tests the mount, fallback and redirect (`test_pipeline_service_ui.py`).

## 5. Build and delivery (W3, W6)

- `src/main/web/package.json` scripts: `dev` (Vite, proxying `/v1` to a local service), `build`, `test`, `typecheck`,
  `lint`, `gen:api`. `package-lock.json` is committed; `dist/` and `node_modules/` are ignored.
- **CI:** a `web` job in `.github/workflows/test.yaml` (Node 24, from `.nvmrc`): `npm ci`, generated-types check, typecheck, lint, test,
  build.
- **Image:** a native-platform `web` stage on `node:24-alpine` runs `npm ci && npm run build` (the docs-assets stage moves
  from `node:22-alpine` to 24 in the same commit, so the image uses one Node); the
  runtime copies `dist/` to `/app/ui` and sets `BEQ_SERVICE_UI=/app/ui`. `docker/smoke.py` also fetches `/ui/` and checks
  it is the app's HTML.
- **Desktop:** unaffected; the PyInstaller bundle carries neither the web source nor the service.
- **Docs (W6):** `docs/library/service.md` gains a "Review in a browser" section; pipeline-service.md takes §3–§5 of this
  file as delivered behavior, its §12 gains decision 9 (review over HTTP, per-title with the token), and the milestone
  text in TODO is corrected.

## 6. Chunks

| ID | What | Done when |
|---|---|---|
| W1 | **Built.** `pipeline/library/decide.py`; title page uses it | gui suite unchanged and green; `decide()` refusal tests; Qt-free list updated |
| W2 | **Built.** Review/chart/decision/next routes, in-flight set, status capabilities, API 1.3.0 | route tests for each 2xx/4xx incl. a race with a redesign; OpenAPI doc regenerated |
| W3 | **Built.** `src/main/web` scaffold, generated types, sign-in, API client and event stream, `/ui` mount, CI job, Docker stage | app builds in CI and the image; `/ui/` served; client and mount tests |
| W4 | Status and Jobs screens, new job with plan preview, cancel, schedule controls | component tests; manual check against a local service |
| W5 | Titles and Review screens, decisions, publish/commit buttons | component tests incl. 409 and override; manual review of a fixture queue |
| W6 | User guide, design references moved to delivered | docs reviewed |
