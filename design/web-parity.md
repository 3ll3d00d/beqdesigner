# Web app parity — editing, visualisation, comparison and guided redesign

**Document type:** Plan — **design, not built** (drafted 2026-10-20). TODO row **WP**, chunks WP0–WP8 below, all
*Not started*. When a chunk lands, describe what was built in [web-app.md](web-app.md) (and
[review-over-http.md](review-over-http.md) / [pipeline-service.md](pipeline-service.md) for routes), archive the chunk's
record, and strike it here.

## 1. Goal

Today the browser app (`src/main/web`, [web-app.md](web-app.md)) decides titles and runs jobs; metadata, artwork,
TMDB, Revise/Reopen, stream choice, projects, the spectrum comparison and the published-BEQ lookup "stay in the desktop
app" (web-app.md §1). The goal is that a person can do **everything the desktop title page does from a browser**, and
some things it cannot:

1. **Edit** what the title page edits: metadata (Essentials / More), TMDB reload, artwork, audio stream, revise
   (review / design / extract), retry, and the chosen design's filters.
2. **See** what it shows, better: the magnitude chart with zoom, limits, log axis, smoothing, normalisation and series
   toggles; the spectrogram (the publication heatmap and an interactive one); every design on one chart.
3. **Compare** against other BEQs: published catalogue entries for the same title, overlaid on our chart and applied
   to our measured curves, and a redesign against the design it replaces.
4. **Redesign on request**, selectively: one title or a chosen set, with instructions such as *ignore 28–34 Hz*,
   *don't go below 8 Hz*, *cap the boost at 15 dB*, *tilt the goal +1 dB/oct* (a bass-boosted target), *only the
   parametric strategy* — whatever the designer says it supports.

Out of scope for this plan: the profile/Settings drawer, sources and ignore rules (setup, not review), Review Folder
(the service has no separate queue directories), and seasons beyond what the review page already shows. Each can be its
own row later.

## 2. Principles carried over

- **Rules are shared, not copied.** As `decide.py` / `review_chart.py` are today (web-app.md §2): every write the web
  app gains is a Qt-free function in `pipeline/library/` that the desktop module is rewired onto *in the same chunk*, so
  the title page and the service cannot disagree. Where the logic sits in a Qt module today (`model/worklist_metadata.py`
  `save_meta_changes`, `model/worklist_published.py` `published_catalogue`, `model/worklist_spectrum.py`), WP0 moves it.
- **Optimistic concurrency everywhere.** Every write sends back the digest it was read at (as `Decision` does with
  `offered_digest()`) and is refused with 409 *Changed since it was read* otherwise. Every write is refused while the
  title is in flight (review-over-http.md's rule).
- **Authority unchanged.** Writes that touch only the queue entry (metadata, artwork, revise to review, design requests) need
  only the token. Anything that touches the catalogue repositories (revising a published title, publish, commit) stays
  behind `allow_repository_writes`, as now.
- **Contract first.** New routes go into `docs/schema/service.openapi.json` and `src/api/schema.d.ts` in the chunk that
  serves them; the API minor version bumps per chunk.
- **Heavy work is a job or a cache, never a blocking request.** Spectrograms and catalogue responses are computed once,
  cached in the work directory keyed by what they depend on, and served from there.

## 3. Where parity stands

| Desktop title page | Web today | Chunk |
|---|---|---|
| Candidates, commentary, chart, Accept / Reject / Skip, keys | Yes | -- |
| Chart: limits dialog, zoom, smoothing, normalisation | Fixed linear 1–160 Hz, auto y | WP1 |
| *Spectrum comparison* tab (publication heatmap, filtered vs not) | No | WP2 |
| *Published BEQs* tab (catalogue matches, their images, our filter beside) | No | WP3 |
| *Metadata* tab, TMDB reload, artwork, autosave, badge | Read-only line; Accept held back | WP4 |
| Reopen / Revise (review, design, extract), retry, choose audio stream | *Redesign…* only (one id, design) | WP5 |
| Open project (hand-edit filters in the main window) | No | WP8 |
| Redesign on request | Neither | WP6, WP7 |

## 4. Chunks

### WP0 — Push the remaining title-page logic down (Qt-free)

No user-visible change; the desktop keeps working exactly as before.

- `pipeline/library/metadata_edit.py`: `field_texts`, `parse_audio_types`, `apply_changes`, `save_meta_changes`
  (returns the entry and whether it changed) and `metadata_digest(entry)`. `model/worklist_metadata.py` imports them.
- `pipeline/library/catalogue.py`: `published_catalogue(cache_dir, refresh)` (download, size cap, atomic replace) and
  `matching_entries` (from `model/worklist_catalogue.py`, already widget-free but under `model/`), plus
  `catalogue_filter(entry) -> CompleteFilter` built from the catalogue's filter list. `model/catalogue.py`'s
  `CatalogueEntry` is Qt-free in its data half; split it if it is not.
- `pipeline/library/spectrum.py`: `comparison_request()` / `render_comparison()` from `model/worklist_spectrum.py`
  (they already use only `pipeline.*` and `model.codec`).
- Add each module to `test_qt_free_modules.py`. Tests: the existing `gui/test_worklist_metadata*.py`,
  `gui/test_worklist_*published*`/spectrum tests still pass unchanged; new unit tests on each function.

### WP1 — The chart workbench

Mostly client side; one route change.

- **Route:** `GET /v1/titles/{id}/chart` gains `candidates=all|N[,M…]` (default: today's single `candidate`, so old
  clients are unaffected) and `smoothing=none|1/1…1/48|sg` computed server side with `model/xy.py`'s smoothing (the same
  code as the desktop, so a 1/6-octave curve is the same curve). Each series carries `kind`, `filtered`, `candidate`.
  Also returns each design's own filter response (`filter` series) so it can be drawn on a secondary axis, as the main
  window does.
- **`MagnitudeChart` becomes a workbench** (still uPlot, still only this component imports it):
  - drag to zoom x (and y with a modifier), double-click to reset; wheel zoom on x;
  - a limits popover: x min/max, y min/max, *log x* toggle, presets **Bass (1–160 Hz)**, **Sub (1–40 Hz)**,
    **Full (1–Nyquist)**, *Auto y*, mirroring `limits_dialog.py`'s full-range/bass buttons;
  - a legend that toggles each series; *Show*: average / peak, filtered / unfiltered, filter response (secondary y);
  - *Normalise to*: any visible series (plotted relative to it), as the main window's reference curve;
  - *Designs*: overlay all designs at once (one colour each, the highlighted one emphasised) rather than one at a time;
  - a crosshair read-out of every visible series at the cursor frequency;
  - *Reset view*. View settings persist per browser (`localStorage`, wrapped in try/catch), not per title.
- **Not** resampled onto the first series' frequencies any more when the x axis is log: uPlot's log distribution on
  the server's native grid. `chartData()`'s resampling is kept only where series grids differ.
- Tests: `chartData()` and the smoothing query unit-tested; the workbench's controls with Testing Library against the
  uPlot recorder (the scales/series it was given); the route's new parameters in `test_pipeline_service_review.py`.

### WP2 — Spectrogram

Two views, one data source.

- **Publication heatmap (parity):** `GET /v1/titles/{id}/heatmap.png?candidate=N` — exactly the image publish would
  write (`pipeline/publish/heatmap.py` `heatmap_for`, the spec from the profile/preferences), via WP0's
  `pipeline/library/spectrum.py`. Rendering takes seconds, so the first request starts it on the service's pool and
  answers 202 with `Retry-After`; the PNG is cached under `<work_dir>/<id>/web/heatmap-<digest>.png` keyed by the mono
  wav's stamp, the design's filter and the spec, and `ETag`ged.
- **Interactive spectrogram (beyond):** `GET /v1/titles/{id}/spectrogram?candidate=N&filtered=bool` returns the
  time×frequency grid the heatmap is drawn from (`model.heatmap.spectrogram_data`), quantised to uint8 dB steps over a
  stated range, as a compact binary body (`application/octet-stream` with a small JSON header, or JSON with base64 —
  decide in the chunk by size on a 3-hour title) at a bounded resolution (≤ 2048 time bins × ≤ 512 frequency bins, the
  service decimating). Same cache and 202 rule.
- **Client:** a `Spectrogram` component drawing to `<canvas>` with a colour map matching `beq_dark`, linked zoom with
  the magnitude chart's x (frequency) axis, time zoom by drag, *unfiltered | filtered | difference* toggle, and a
  max-over-time strip (the "where are the heavy hits" view of `model/analysis.py`'s `MaxSpectrumByTime`). Lazy: fetched
  only when its tab is opened.
- A **Visualise** area on the review page holds tabs *Magnitude*, *Spectrogram*, *Heatmap*, *Published BEQs* (WP3).
- Tests: cache key and 202/200 behaviour (`test_pipeline_service_spectrum.py`), the quantisation round trip, and the
  component's rendering against a fake canvas context.

### WP3 — Comparison with published BEQs

- **Service:** `GET /v1/titles/{id}/catalogue` — the matching catalogue entries (WP0's `matching_entries`, with the
  "other tracks / editions" switch as `include_other=true`), each with author, edition, audio, the catalogue page URL
  and its image URLs; `GET /v1/titles/{id}/catalogue/{entry}/chart` — that entry's filter response and our measured
  average/peak **filtered by their filter**, on the same grid as WP1's chart; `POST /v1/catalogue/refresh` — refetch
  `database.json` (a job; token only; the cache dir is the service's `--catalogue-dir`, default under the work dir).
  Images are proxied through the service (`/v1/catalogue/image?url=`, allow-listed to the catalogue's hosts, cached) so
  the browser needs no third-party access and the token-free page loads nothing external.
- **Client:** the *Published BEQs* tab lists matches; ticking one or more overlays, in WP1's workbench, their filter
  response (secondary axis) and *our track under their BEQ* beside *our track under our design*; their published
  image beside ours (WP2's heatmap). A difference trace (ours − theirs, filtered average) answers "how far apart are
  we" at a glance; a one-line summary per entry gives the max boost, the frequency it peaks at and the RMS difference
  across 5–40 Hz.
- Tests: matching and the catalogue filter build (unit), the image proxy's allow-list and caching (service), the
  overlay controls (web).

### WP4 — Metadata and artwork editing

Removes the biggest reason to go back to the desktop: a title whose metadata is incomplete can be fixed and accepted.

- **Service:** `PATCH /v1/titles/{id}/metadata` with `{digest, changes}` (WP0's `save_meta_changes`; refused in flight,
  409 on a changed digest, 422 with the field when a value is invalid); `Review` gains `metadata_digest` and the
  editable fields with their typed choices (content type, audio types, genres) so the form is built from data, not
  duplicated. `POST /v1/titles/{id}/metadata/tmdb` — TMDB reload (the service's `--tmdb-api-key` / `BEQ_TMDB_API_KEY`;
  the key redacted from any error, as the desktop's review fix does); returns proposed values, the client shows the
  difference and the person applies them through the PATCH. `PUT /v1/titles/{id}/artwork` — choose from TMDB posters,
  a URL (downloaded by the service with the desktop's image checks, `worklist_artwork.py` moved down in WP0), an upload
  (size- and type-checked), or clear; `GET /v1/titles/{id}/artwork` serves the current one.
- **Client:** a *Metadata* panel on the review page: Essentials and More, the missing fields highlighted (the same
  `metadata_problems()` list), autosave on blur with a visible saved/saving/failed state, *Discard* to the read values,
  and the poster with *Change…*. Accept stays held back until `metadata_problems` is empty, now with a link to the
  field instead of "use the desktop app".
- `metadata_hint` for the service changes from "the app" to the panel; web-app.md §1's "stay in the desktop app" list
  shrinks accordingly.
- Tests: route rules (digest, in flight, validation, TMDB key redaction), form round trip, autosave, the Accept gate.

### WP5 — Revise, retry and stream choice

- **Service:** `POST /v1/titles/{id}/revise` `{to: review|design|extract, reason, digest}` over
  `pipeline.library.revise` (the same function the desktop's `worklist_revise.py` calls); a published title needs
  `allow_repository_writes` and the profile's repositories, and the response says which catalogue files were reverted.
  `to: design|extract` can also queue the run (`run: true`), the existing single-title redesign job generalised.
  `GET /v1/titles/{id}/streams` (`describe_streams`) and `PUT /v1/titles/{id}/stream` `{audio_stream}` — choosing a
  stream sends the title back to extract, as the desktop's *Choose audio stream* does. Retry is `revise` on a failed
  title (or the existing `retry_failed` run flag).
- **Selective, many titles:** `POST /v1/revise` `{ids[], to, reason, run}` with a per-id result list (as
  `revise_titles()`/`ReviseOutcome`), and on the Titles page a **selection column** (checkbox per row, *select all
  matching the filter*) with *Revise…* acting on the selection. The existing *Redesign out of date…* becomes one preset
  of it.
- **Client:** the review page's *Redesign…* becomes *Revise…*: the three depths with the desktop's wording
  (`revise_text`, moved to the shared module if it is not already Qt-free), and a stream picker on the Playback line.
- Tests: route rules per depth and status, the bulk outcome, the Titles selection and confirmation.

### WP6 — Contract 1.5: design requests (both sides)

A general protocol by which a caller **asks** for changes to a design and the designer **answers each ask**: honoured,
adapted, declined or not understood, saying why. The designer stays free to react as it sees fit; the caller always
learns what was done with what it asked. This is the normal way a caller influences a design, not a carve-out.

It replaces the reading of designer-interface.md §2 "What is deliberately absent" as "nothing may cross in". The
concern that text protects is a designer *inferring* the caller's intent and shaping its answer silently; that stays
ruled out. Restated for 1.5: **a designer acts on no unstated preference; a stated request is an input it weighs
against the evidence and reports on.** §4's line is unchanged: a request can make a design weaker or narrower, and a
designer may decline one that would have it publish what its evidence does not support (an excluded band that held the
only rolloff, a boost where there is no content), saying so in the request's outcome.

- **Request (additive, optional):**

  ```python
  requests: list[DesignAsk] | None = None   # 1.5

  @dataclass(frozen=True)
  class DesignAsk:
      id: str               # the caller's, echoed in the outcome
      kind: str             # from the vocabulary below, or one a designer advertises; unknown kinds are allowed
      params: dict          # the kind's parameters
      text: str | None = None   # what the person said, for the designer's record and for kind 'note'
  ```

  A starting vocabulary (§2.x, extended by contract minor versions; a designer may advertise kinds of its own):

  | kind | params | meaning |
  |---|---|---|
  | `exclude_band` | `low_hz`, `high_hz` | do not treat this range as evidence or as a target |
  | `floor` | `hz` | restore nothing below this (the content edge) |
  | `ceiling` | `hz` | look for the rolloff no higher than this |
  | `max_boost` | `db` | cap the correction |
  | `goal_tilt` | `db_per_octave` | tilt the target (positive: a bass-boosted goal) |
  | `goal_tolerance` | `db` | how close the design must come to its goal |
  | `strategies` | `names` | only these strategies, in this order |
  | `max_sections` | `count` | biquad budget |
  | `exclude_times` | `ranges_s` | leave these stretches of the programme out of the analysis |
  | `note` | -- | free text; the designer may act on it, and must still answer it |

  beqforge already has internal dials for most of these (`DiagnoseParams.exclude_bands_hz`,
  `DesignParams.max_boost_db` / `lowest_frequency_hz` / `max_sections`, `--goal-tilt`, `--goal-tolerance`,
  `strategies`, `content_edge`); today they are server-wide settings, folded into `design_settings`.
- **Response:** `request_outcomes: list[{id, outcome: 'honoured' | 'adapted' | 'declined' | 'not_understood',
  applied: dict | None, reason: str | None}]` -- one per ask, `applied` the parameters actually used (after clamping,
  for `adapted`), `reason` required for anything but `honoured`. A 1.5 designer answers every ask; an ask it does not
  know is `not_understood`, never an error. A designer before 1.5 ignores `requests` (the 1.x rule: unknown request
  fields are ignored) and returns no outcomes, which the caller shows as *not acknowledged by this designer*.
  A declined ask does not decline the design: the design is the designer's best answer to the asks it did act on.
- **Discovery (advisory):** `GET /health` adds `request_kinds: {kind: {params: {name: {type, unit, min, max,
  default}}, description}}`. The caller builds its form from it, so a new kind needs no caller release, but discovery
  only informs: the caller may still send a kind the designer did not list (a `note`, or a kind newer than the health
  it read), and the outcome says what happened.
- **Who asks:** the protocol does not care. The caller records provenance on its side: a person on the review page, a
  bulk revise, or (later, if wanted) a profile default sent with every request.
- **Identity:** the asks are part of the request, like the audio; not part of `design_settings` (server-wide). The
  caller adds a canonical digest of the asks sent to the entry's `design_fingerprint`, so the design cache never serves
  an answer to different asks, and a redesign-by-version sends the stored asks again. A request with no asks must
  design exactly as 1.4 did (golden test both sides; no `design_version` bump unless a default changes).
- **Caller storage:** `QueueEntry.requests` (each ask with who, when and why) and `request_outcomes` (the designer's
  answers). `revise_entry(to='design', requests=…)` sets them; *Clear requests* removes them and marks the design stale.
  Kept across re-extraction.
- **Work:** designer-interface.md §2 (the restated rule, `requests`), §3 (`request_outcomes`), §4 (declining an ask vs
  declining the design), §7 (health) and the conformance spec; `pipeline/designer` (request, health, outcome parsing),
  the fingerprint, `QueueEntry`, the service models; beqforge implements the vocabulary it can and advertises it (a
  handoff in its own `TODO.md`/`plans/`).
- Tests: conformance (each outcome kind, an unknown kind answered `not_understood`, a pre-1.5 designer ignoring the
  field), fingerprint changes with asks and not without, cache separation, replay on redesign-by-version.

### WP7 — Guided redesign in the browser

- **Revise… with requests:** WP5's dialog gains a *Requests* section built from the designer's advertised kinds
  (WP6): numbers with their units and bounds, band lists, strategy checkboxes, and always a free-text note. Presets for
  the common asks (*ignore a band*, *bass boost +N dB/oct*, *cap the boost*, *higher content edge*) add those asks; the
  person sees the exact asks that will be sent.
- **Pick on the chart:** with the dialog open, drag on the magnitude chart (or the spectrogram) to add an excluded
  frequency band, or a time range on the spectrogram for `exclude_times_s`; the bands are shaded on both views.
- **Selective, in bulk:** from the Titles selection (WP5), the same asks sent for every selected title, or *per-title*
  asks queued one at a time from each review page and then **Run the titles with requests** as one job
  (`needs` design, filter `with_requests=true`).
- **See what it did:** the review page lists each ask with its outcome (honoured / adapted with what was used /
  declined or not understood with the designer's reason), and a
  **Previous design** comparison: before a guided (or any) redesign replaces the entry's designs, the service keeps the
  last set as `QueueEntry.previous` (designs, chosen index, asks and outcomes, designer identity, when) — one level of history —
  and WP1's workbench overlays it, dashed, against the new ones. *Restore previous* puts it back without a design run
  (a revise of its own, digest-checked).
- **Filter on the Titles page:** *With requests* (any, yes, no) and *Requests not honoured* (any ask declined, not understood or not acknowledged).
- Tests: the form from a capability fixture, chart band picking, bulk submit, previous-design comparison and restore.

### WP8 — Editing the chosen design's filters

The desktop does this by opening the `.beq` project in the main window; the browser edits the design directly.

- **Service:** `GET /v1/titles/{id}/projects` (`worklist_projects.py`'s states, moved down in WP0: mono/multichannel,
  *modified since design*), and `PUT /v1/titles/{id}/design/{n}/filters` `{digest, filters}` — an edited copy of a
  design, stored as a new offered design marked `edited_from: n` (the designer's original stays for comparison), its
  filters validated with the same rules as a designer's `BiquadSpec` (§5: types, ranges, stability, the device's slot
  budget) and realised at the entry's `fs`. Accepting it writes the projects through the existing hash gate, so a later
  redesign still respects hand edits.
- **Client:** a filter table (type, Fc, Q/S, gain; add, remove, reorder) beside the workbench with the response
  updated live (client-side biquad maths, unit-tested against `model/iir.py`'s values for each filter type), *Add low
  shelf* as the manual "bass boost", *Reset to design*, and *Save as edited design*.
- Tests: validation parity with `iir.py` (a shared fixture of filters and their responses), the route's rules, the
  editor.

## 5. Order and dependencies

```
WP0 ─┬─ WP2 ─┐
     ├─ WP3 ─┤
     ├─ WP4  │
     └─ WP5 ─┴─ WP7 ── WP8
WP1 ───────────┘
WP6 (both repos) ── WP7
```

Recommended order: **WP0, WP1, WP4, WP5** (parity on what blocks deciding from a browser), then **WP2, WP3**
(visualisation and comparison), then **WP6** with beqforge, **WP7**, **WP8**. WP6's contract text can be agreed early,
in parallel, since it needs beqforge's side.

## 6. Open questions

1. **The restated §2 rule** (WP6): agree the wording with beqforge before building either side.
2. **Which kinds first.** Proposed minimum: `exclude_band`, `floor`, `max_boost`, `goal_tilt`, `note`.
   `exclude_times` interacts with `coverage` (a complete programme with stretches excluded is no longer quite one);
   decide whether the designer reports it as an excerpt.
3. **Spectrogram transport** (binary vs JSON) and resolution: decide on a measured 3-hour title in WP2.
4. **Catalogue cache on the service** — its own download, or a mounted copy of the desktop's `BEQ_DOWNLOAD_DIR`.
5. **History depth** — one previous design set (proposed) or several.
