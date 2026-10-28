# The browser app — the pipeline service's web front end

**Document type:** Architecture reference — delivered behavior (2026-10-10). The plan it was built from, with its
chunk records and evidence, is [archive/web-review.md](archive/web-review.md). The review routes it calls are
[review-over-http.md](review-over-http.md); the user guide is "Review in a browser" in
[`docs/library/service.md`](../docs/library/service.md); the developer guide is
[`src/main/web/README.md`](../src/main/web/README.md). Section numbers are cited by source comments and kept stable.

## 1. Goal

A browser app, served by the pipeline service itself, giving a basic view of what the desktop work list and title page
do: the workload and its status, starting and following jobs, deciding each title's design, and editing its metadata
and artwork (with TMDB), sending titles back (revise) and choosing their audio stream. Publishing and committing
accepted titles are offered behind the service's `allow_repository_writes`. Everything the desktop title page does is
in the browser too: a design's filters are edited there (§4.7) rather than by opening its `.beq` project, whose state
the review page reports in the desktop badge's words, and a redesign can ask for adjustments (§4.6). Setup -- the
profile, sources and ignore rules -- and Review Folder are not title-page work, and stay in the desktop app.

## 2. Decision rules are shared

The rules and the write of a decision are the Qt-free `pipeline/library/decide.py` (`decide()`, `decision_blocked()`,
`offered_digest()`, `next_waiting_id()`), and the curves a design is judged on `pipeline/library/review_chart.py`
(`chart_curves()`). The desktop title page (`model/worklist_title_decide.py`, `model/worklist_title_text.py`) and the
service's review routes both call them, so the two cannot apply different rules. Where they say where to do something,
the caller passes its own words: `redo`, `run_hint` (the service names `POST /v1/jobs` with kind run) and `metadata_hint` (the
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
- **Live events:** `src/api/events.ts` reads `GET /v1/executions/{id}/events` with `fetch` and a streamed body, because
  `EventSource` cannot send the `Authorization` header. It resumes a dropped connection with `Last-Event-ID` (giving up
  after five retries in a row), and ends at the execution's final state event or a clean end of the stream, which the
  service sends only for one that is over. A 401 or 404 is not retried. The service sends the stream with
  `X-Accel-Buffering: no`, so nginx in front passes each event on as it comes. `useExecutionEvents()` keeps the last 500
  and refreshes the execution, executions, jobs, status and titles queries when it ends.
- **Screens** (`src/pages/`):
  - *Status*: the pipeline strip (titles per `needs`, each linked to the Titles list filtered to it), then two
    sections. *Jobs* is the work from both ends: as facts, the execution running now (linked to its page; started,
    estimated end with time to go, progress with the rate an hour, and activity: the stage and title in hand with
    ffmpeg's percent, or the titles out on the designer) and a progress bar, how many wait, and the default window;
    then a table of every job active or paused (design/jobs-and-executions.md §5: name and what it does, once or every
    N, its window, state and last skip, its running execution or next start) with *Pause*/*Resume*, *Run now* (a
    recurring job) and *Edit…* (its page); then *Scan sources* -- *Rescan sources* after the first scan -- which makes a
    one-off scan job and opens it, *New job…*, *Every execution*, and the sources with their last error.
    *Designers* is one table with a row per endpoint of the configured designer (designer, URL, up/down with when
    `/health` was last asked and why it is down, protocol -- the design version --, version -- the release its
    `/health` names, the build on hover --, max jobs, and what each is designing now from `designing`, linked to its
    review; *idle* when nothing is out; a designer with no endpoints is one row saying why) with the warning when the
    endpoints up would design differently. The scan action is also on Titles. Polls `/v1/status` every 10 s.
  - *A job* (`/jobs/:id`): its name, state (and why it ended), what it does, once or every N, its window, its next
    start, a one-off's progress over its executions and the titles left, its titles by outcome over the executions kept; its execution running now, followed live, with
    *Cancel* (which removes a one-off job, and says so); the executions before; and *Pause*/*Resume*, *Run now*,
    *Edit…* (name and window; a recurring job's interval, filter, through and retry failed) and *Remove…* (asked first).
    `/jobs` goes to Status.
  - *Executions* (`/executions`, the log): each active job's next start, then every execution kept, newest first,
    filtered by job and state, each with its job's name and which run of it, why it stopped, and what a run designed
    and failed. An execution's page: its state (and why it stopped), its job, what it was asked, its times and window,
    the result (counts; failures, unavailable, failed earlier, not published, designed and skipped titles, each linked to
    its review; a stopped run's titles left for its job's next run), *How it ran* (its time and titles an hour, its
    titles by outcome, each opening the titles from `GET /v1/executions/{id}/titles`, why titles failed or were
    skipped, each designer endpoint with its release, version, counts and times -- warning when they differ -- each
    stage's times, and the slowest titles), why it failed, its log (live while it runs) and
    Cancel (Drop for a queued one; "Stopping after the title in hand" for a running one). The window field on *New job*,
    *Edit…* and *Revise*, and the default on Status: [job windows](job-windows.md) §5.
  - *New job*: a name, then a scan of chosen sources, or a run: the filter form (needs, search, source, kind, year, new
    since the last scan), `through`, scan first, retry failed, *When* (once, or every N minutes -- only a run through
    extract or design recurs), the window, *Preview* (`POST /v1/plan`: the label, what would run with its stages, what
    would be skipped and why; marked stale when the choices change) and *Run* (*Make job* for a recurring one); either
    opens the job's page. Publish and commit are
    disabled, with the reason, until `/v1/status` says `repository_writes` and `repositories_configured`. Bulk accept is
    not offered: the app decides titles one at a time.
  - *Titles*: a table of 100 a page (title, year, kind, source, needs, detail, confidence, designs, flags) linked to
    each review with the filter and sort kept. Each heading sorts by its column (ascending, descending, then back to
    work-list order) and the row under the headings filters it (text, year, kind, source, needs, detail, flag; typed
    text is sent once typing pauses); both are the service's (`sort`/`descending`, the filter), so they hold across pages,
    and both live in the address bar (`filterToQuery`/`sortToQuery`, so a list can be linked to). *New since the last
    scan*, *Include titles that need nothing*, *Designer* (any, declined, not declined), *Design version* (any, out of
    date, not out of date), *Adjustments* and *Answers* (contract 1.5's asks: `adjusted`,
    `unmet`, read from the index's entry summary; a badge marks a title adjusted, with how many asks were not honoured) sit
    above it; how many design runs a title keeps is the review page's to show, and the *Designs* column counts the designs
    to choose from in all of them (`design_choices`), *Review next waiting*, *Redesign out of date…* (shown with *Out of date*: one
    redesign job by version after an inline confirmation) with, under it, *Check what a redesign would move*
    (`components/ImpactPanel.tsx`, contract 1.6, [redesign scope](redesign-scope.md)): the designer's answer by tier --
    will move, unknown (read as will move), likely, might, unaffected -- each title with its score and reasons, an answer
    for another design version than the designer's now marked stale; *Select those likely to move* / *the rest*, then
    *Redesign selected* (`/v1/revise` to design, in one run) or *Keep selected* (`/v1/reviewed`, each refusal named), and *Publish accepted…* / *Commit published…*:
    a plan of the filter with `needs` publish (or commit) says how many, an inline confirmation names the repositories,
    and a run job through publish (or commit) is submitted and followed. Disabled with the reason as on New job.
  - *Review*: the title, its status, what it needs; the designs (rejected ones listed apart, with how many reasons),
    commentary of the one highlighted (or the decline; full width under the designs and chart), the playback chain and
    designer, the metadata as one line under the heading (content type, audio types, genres by name) and what it lacks,
    the chart with its controls (§4.1) beside *Spectrogram*, *Heatmap* and *Published BEQs* tabs (§4.4, §4.5), *Edit metadata and artwork* (§4.2), *Revise…* and *Audio stream…* (§4.3), *Design runs* (§4.6) and *Edit filters…* (§4.7). *Accept design N*, *Reject* and *Skip*, with keys A, R, S, and 1–9 to highlight a design; the keys
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

`MagnitudeChart` draws `GET /v1/titles/{id}/chart`: the measured average and peak curves, dashed when a design is applied,
and the same after the design highlighted, solid -- as the desktop title page draws them -- or, with *All designs*,
after every design at once (`candidates=all`), one colour per design, the highlighted widest and peaks dotted. Each
design in the list also has a *compare* tick: the ones ticked, from any run, are drawn with the highlighted one
(`candidates=0,2`, and `/chart?variant=…&candidates=…` per other run), coloured by their number in the one list
(`numbered()`), so the same index in two runs is two colours; moving the highlight while comparing keeps the design
highlighted until then on the chart, and *Stop comparing* goes back to the one. Ticks are per title and digest, not kept. The view
(`src/chart/view.ts`, a `ChartView`) is kept per browser in `localStorage` (`beq.chartView`, each setting checked when
read, defaults when storage refuses) and changed by `ChartControls`:

- **Range:** *Bass* (1-160 Hz, `FREQUENCY_RANGE`, the desktop charts' default and the default view) and *Sub* (1-40 Hz);
  no *Full*, since the curves stop at 500 Hz; frequency and level limits typed (blank levels: fitted to what is
  drawn); *Log frequency* (uPlot's log distribution, labelled at 1, 2 and 5 of each decade or finer when zoomed into one, `logTicks()`; a range starting at or below 0 Hz starts at 1 Hz).
- **Zoom:** a drag selects a frequency range (and the levels, when dragged vertically too); the wheel zooms about the
  cursor (on the log scale when the axis is); a double click goes back to the preset frequencies and fitted levels.
- **Curves:** *Average*, *Peak*, *Before design*, *After design*; *Filter response* asks for each design's own response
  (`responses=true`) and draws it on a second level axis; *Smoothing* is sent to the service (`smoothing`), which applies
  the main window's (`review_chart.SMOOTHING`: none, 1/1-1/48 octave, Savitzky-Golay; `model.xy.smooth_octave`, a mean of power over each point's 1/N octave on the curve's own frequencies) after the design, never to a
  response.
- **Relative to:** every curve but a response is drawn minus the one chosen (`normalise()`), which becomes flat.

Every series is placed on the frequencies of the one with the most points in range, the others interpolated
(`chartData()`). *Reset view* restores the defaults.

### 4.2 Metadata and artwork

`MetadataPanel` edits a title as the desktop's Metadata tab does, over `PATCH /v1/titles/{id}/metadata`, the TMDB and
the artwork routes ([pipeline-service.md](pipeline-service.md) §6.1). Its fields come from `GET /v1/metadata/fields`
(Essentials, then *More* folded): text, the audio types as checkboxes over a comma-separated text (values not offered
are kept), language and source typed or picked. A field is saved when it is left, on Enter or *Save*: only the fields
typed, with the review's `metadata_digest`. A 409 *Changed since it was read* reads the review again and keeps what was
typed, to be saved against what is there now. *Revert* (which does not take the focus, so leaving the field does not
save it first) throws away what is unsaved. *Reload from TMDB* fills the form and the extras (genres, overview,
collection) unsaved; it is disabled when `/v1/status` says `tmdb` is false. The poster is fetched as a blob (an `<img>`
cannot send the token); upload (read as base64), download from an address, and clear answer with the review. Problems
`metadata_problems` lists are shown and their fields marked; a run with the title in hand disables the form.

### 4.4 Spectrogram and heatmap

The review's chart card has tabs: *Magnitude* (§4.1), *Spectrogram* and *Heatmap*, for the design highlighted. Both
drawings are rendered by the service off the request, so `renderedQuery()` reads a 202 as "pending" and asks again
every 2 s. `Spectrogram` decodes the grid (`src/chart/grid.ts`, `decodeGrid()`) and paints it pixel by pixel
(`paint()`: frequency across on the chart's range and log setting, time down; a perceptual scale over the 60 dB below the
loudest cell; *Difference* blue-white-red over ±20 dB from the grids before and after the design) into a canvas, with
`loudestByFrequency()` in a strip under it, and a colour bar of the scale (`scaleGradient()`).

*Heatmap* is drawn here too, from the same two grids (no extra render): the published image's panes, after the design on
the left and as measured on the right, 1-40 Hz across and time down, each cell within 60 dB of the loudest as a dot a cell
wide (at least 5 px) in the spectrogram's colours, quietest first (`byLevel()` groups the cells by level so each colour is
set once), on the page's background. *Published image…* fetches the PNG publish writes and links it.

Both are navigated the same way (`PlaneControls`, `usePlaneGestures`, `src/chart/plane.ts`): the frequency and time ranges
are typed (Hz; h:mm:ss, m:ss or seconds) and applied on Enter or leaving the box, a box that does not read is marked;
*＋*/*－* zoom about the middle, *Back* returns to the view before (a stack of 50), *Whole title* (or a double click)
resets; the line under them says how far in the view is. On the canvas a drag selects a box -- drawn as it is dragged,
labelled with the ranges it would zoom to, Escape drops it -- and zooms both axes to it (a side thinner than 4 px keeps
that axis); the wheel zooms both about the pointer; the pointer's frequency, time and level are read out under it. The
spectrogram's frequencies are the shared `ChartView`'s, so the magnitude chart follows; its times, and all of the
heatmap's view, are kept on the page.

The page follows the system's light or dark setting live: the colours are CSS variables under `prefers-color-scheme`,
and the canvases and the uPlot chart, which read them when they draw, draw again when it changes (`useColourScheme()`,
`src/theme.ts`).

### 4.5 Published BEQs

`PublishedBeqs` lists `GET /v1/titles/{id}/catalogue` (the same track only, unless *Include other tracks and editions*,
or every one when none is the same track) with each entry's catalogue page and the catalogue's compare page. Ticking
entries asks for each one's chart (`usePublishedCharts`) and draws them on the workbench (`ChartControls` and
`MagnitudeChart`, the same `ChartView`) beside the chart route's curves for the highlighted design; a published BEQ's
curves carry `candidate` `CATALOGUE_BASE` + its index and take colours of their own. Each ticked entry says its peak
boost and the RMS difference between our design's filtered average and the track under it over 5-40 Hz
(`comparison()`).

### 4.6 Adjustments and design runs

`ReviseDialog` (to design or extract) has *Ask the designer to change something*, enabled when `/v1/status` lists
`designer.adjustment_kinds` (the kinds every endpoint up lists; none from a designer before contract 1.5). Ticked, it
shows `AdjustmentsEditor`: a list of asks, each a kind with the parameter inputs its spec gives (number, string or
boolean, with unit and bounds checked by `askProblem()`; the designer still judges them) and an optional *Why*; *Add an
adjustment* and presets for the kinds listed (ignore the chart's frequency range, +1 dB/oct, a 15 dB cap, nothing
below 10 Hz), and *Pick a band on the chart*: the page then takes the magnitude chart's next drag as an `exclude_band`
(`ReviewPage`'s `pickBand`, a prompt above the chart) instead of a zoom, and the bands asked for are shaded on it
(`MagnitudeChart`'s `bands`, drawn in uPlot's `draw` hook). For one title it starts from the review's `adjustments`. The list goes as `ReviseRequest.adjustments`
(`[]` asks for none; unticked, each title keeps its own).

Every kept design run's designs are in the review's one list of designs (`runsOf()`, `offersOf()`): the current run
first, then the earlier ones newest first, each under a heading (its id, when, the build, declined) when there is more
than one, its candidates and then the ones its designer rejected, numbered on through the whole list -- the number is
what 1-9 and *Accept design N* say. The review carries each earlier run's designs in full (`VariantSummary.candidates`
and `rejected`, as the current run's). Highlighting one of an earlier run charts it from that run (`/chart?variant=…`),
and the spectrogram and heatmap draw it the same way (`variant` on both); with *All designs* (or a design of
theirs ticked to compare) the other runs' designs are drawn too (`useVariantCharts`), in the colour of their number. Accepting it sends
`Decision.variant`: the service checks the digest, makes that run current (the current one kept) and accepts, and puts
the current run back if the accept is refused. *Edit filters…* edits a design of the current run only.

`VariantsPanel` (*Design runs (N)*) lists the current design run (`variant_id`, `designed_at`, `adjusted_with` and each
ask's outcome, `outcomeText()`; what the next design is asked when it differs) and the earlier ones newest first, with
*Use this one* (disabled on an accepted or published title) and *Discard…* (asks first), which send the review's
`digest`; with two or more earlier runs, *Discard all earlier runs…* (asks first) forgets them all
(`DELETE /v1/titles/{id}/variants`, `variants.discard_variants()`), keeping the current one.

### 4.7 Editing a design

*Edit filters…* (not on an accepted or published title) opens `FilterEditor` on the highlighted design: its stored
filters as rows (`biquad.rowsOf()`; a design with any other kind of filter cannot be edited here), each type, frequency,
Q, gain and how many stacked, with add (a low shelf, a peak), reorder, remove and *Back to the design*. The edit is drawn
on the magnitude chart with the view's limits and zoom: the measured curves, the design it started from, the edit's
response and the curves under it, computed in the browser by `src/chart/biquad.ts` (the RBJ cookbook biquads of
`model/iir.py`, held to its values in `biquad.test.ts`). `rowProblem()` and the 10-section budget hold *Save as a new
design* back with the reason; saving sends `PUT /v1/titles/{id}/edited` and the review becomes the new design run. Under the review's heading,
`ProjectsLine` says `GET /v1/titles/{id}/projects` in the desktop title page's words (as designed, modified since design,
not written yet, unreadable), and the editor warns when a project was edited by hand since the design: publish takes
the project's filter, not the browser's edit.

## 5. Build and delivery

- **Chunks:** the review page (uPlot and every chart) is a route `lazy` import in `App.tsx`, built as its own chunk
  and loaded when a title is first opened; the rest of the app stays under Vite's 500 kB warning.
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

### 4.3 Revise and the audio stream

*Revise…* on the review page, and *Revise selected…* over the titles ticked on the Titles page (a checkbox per row, one
in the heading for the page, then *Select all N matching*, which collects the ids page by page up to the 500 a revise
takes, `useMatchingIds`; the selection is forgotten when the list changes), open `ReviseDialog`: *Reopen for
review*, *Redesign* or *Re-extract and redesign*, in the desktop's words (`worklist_revise.CHOICES`), a reason kept in
the note, and for the last two *Extract & design now* (on by default). It sends `POST /v1/revise`: with `run`, one
redesign job (`revise_to` design or extract) and the job page opens; without, the titles are sent back at once and the
page says how many were and why any were not. *Audio stream…* (`StreamPicker`) lists `GET /v1/titles/{id}/streams` and
`PUT /v1/titles/{id}/stream` starts the job that extracts and designs the one picked. Both are disabled while a run has
the title, and for a published title unless the service allows repository writes.

The redesign job waits for the work-directory lease
rather than joining another run, checks the designer before changing state, calls `revise_entry()` (or, for a stream,
`revise.choose_stream()`, shared with the desktop), refreshes the
index and runs the normal stages. Existing extraction, metadata, artwork and hand-edited projects are retained.
Published titles require `allow_repository_writes`; the existing revise rules revert uncommitted catalogue files
and preserve committed files for a subsequent revision. Publish and commit remain separate actions.
