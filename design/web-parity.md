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
4. **Redesign with guidance**, selectively: one title or a chosen set, with instructions such as *ignore 28–34 Hz*,
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
- **Authority unchanged.** Writes that touch only the queue entry (metadata, artwork, revise to review, guidance) need
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
| Guided redesign | Neither | WP6, WP7 |

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

### WP6 — Contract 1.5: design guidance (both sides)

The contract today forbids exactly this (designer-interface.md §2 "What is deliberately absent", §4): *no headroom,
max-boost or device preference crosses in*. That rule exists so **unattended** designs are the evidence's, not a
caller's policy. Guidance keeps that rule for every unattended request and adds one deliberate exception: a
**person's instruction for one title**, recorded with the design, never sent by a scheduled or bulk run unless a person
asked for that title.

- **Request (additive, optional):**

  ```python
  guidance: dict | None = None   # 1.5: a person's instruction for this title; None for every unattended request
  # keys a designer may support, all optional:
  #   exclude_bands_hz: list[[low, high]]      ignore these frequency ranges as evidence and as targets
  #   lowest_frequency_hz: float               restore nothing below this (the content edge)
  #   highest_frequency_hz: float              look for the rolloff no higher than this
  #   max_boost_db: float                      cap the correction
  #   goal_tilt_db_per_octave: float           tilt the target (positive = a bass-boosted goal)
  #   goal_tolerance_db: float                 how close the design must come to its goal
  #   strategies: list[str]                    only these strategies, in this order
  #   max_sections: int                        biquad budget
  #   exclude_times_s: list[[start, end]]      leave these stretches of the programme out of the analysis
  #   note: str                                free text, for the designer's record only
  ```

  beqforge already has internal dials for most of these (`DiagnoseParams.exclude_bands_hz`,
  `DesignParams.max_boost_db` / `lowest_frequency_hz` / `max_sections`, `--goal-tilt`, `--goal-tolerance`,
  `strategies`, `content_edge`): today they are server-wide flags, folded into `design_settings`.
- **Capability discovery:** `GET /health` adds `guidance: {key: {type, min, max, default, unit, description}}` — the
  keys this designer honours. The caller offers only those (this is how the UI supports "any other option that might
  be supported" without a caller release per option), and refuses to send an unknown key (422 before any work).
- **Response:** `guidance_applied: dict` (what was honoured, after clamping) and `guidance_notes: list[str]` (what was
  clamped or ignored and why). A designer that does not support 1.5 ignores `guidance` (1.x rule: unknown request
  fields are ignored), returns no `guidance_applied`, and the caller says *guidance was not applied* on the review page.
- **Identity:** the request's guidance is part of the answer's input, not of `design_settings` (which stays
  server-wide). The caller adds a canonical digest of the sent guidance to the entry's `design_fingerprint`, so the
  design cache never serves an unguided design to a guided request or the reverse, and a redesign-by-version replays
  the stored guidance.
- **Rejections and declines:** guidance can make the designer reject or decline (e.g. a band excluded that held the
  only rolloff); it reports that as today. Guidance never overrides the designer's internal-evidence checks (§4's line
  holds: a cap may make a design weaker, not make an unsupported boost acceptable).
- **Caller storage:** `QueueEntry.guidance` (what the person asked, who, when, reason) and `guidance_applied` (what the
  designer said it did). `revise_entry(to='design', guidance=…)` sets it; *Clear guidance* removes it (and marks the
  design stale). Kept across re-extraction.
- **Work:** designer-interface.md §2/§3/§4/§7 and the conformance spec; `pipeline/designer` (request, health
  capabilities, response parsing), the fingerprint, `QueueEntry`, the service models; beqforge implements and
  advertises the keys (a beqforge handoff in its own `TODO.md`/`plans/`). Bump the beqforge `design_version` only if a
  default changes; a request with no guidance must design exactly as now (golden test both sides).
- Tests: contract conformance (a guided request, an unknown key, a 1.4 designer ignoring it), fingerprint changes with
  guidance and not without, cache separation, replay on redesign-by-version.

### WP7 — Guided redesign in the browser

- **Revise… with guidance:** WP5's dialog gains a *Guidance* section built from the designer's advertised keys
  (WP6): numbers with their units and bounds, band lists, strategy checkboxes, a note. Presets for the common asks
  (*ignore a band*, *bass boost +N dB/oct*, *cap the boost*, *higher content edge*) fill those keys; the person sees the
  exact guidance that will be sent.
- **Pick on the chart:** with the dialog open, drag on the magnitude chart (or the spectrogram) to add an excluded
  frequency band, or a time range on the spectrogram for `exclude_times_s`; the bands are shaded on both views.
- **Selective, in bulk:** from the Titles selection (WP5), one guidance applied to every selected title, or *per-title*
  guidance queued one at a time from each review page and then **Run the guided titles** as one job
  (`needs` design, filter `guided=true`).
- **See what it did:** the review page shows the guidance asked and applied (and the designer's notes), and a
  **Previous design** comparison: before a guided (or any) redesign replaces the entry's designs, the service keeps the
  last set as `QueueEntry.previous` (designs, chosen index, guidance, designer identity, when) — one level of history —
  and WP1's workbench overlays it, dashed, against the new ones. *Restore previous* puts it back without a design run
  (a revise of its own, digest-checked).
- **Filter on the Titles page:** *Guided* (any, yes, no) and *Guidance not applied* (designer ignored it).
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

1. **Guidance vs the contract's no-preference rule.** This plan treats a person's per-title instruction as a
   deliberate, recorded exception. Agree that with beqforge before WP6, or decide guidance is caller-side only (a
   post-design edit — WP8 — and caller-side exclusions on the request audio) and drop the contract change.
2. **Which guidance keys first.** Proposed minimum: `exclude_bands_hz`, `lowest_frequency_hz`, `max_boost_db`,
   `goal_tilt_db_per_octave`. `exclude_times_s` changes `coverage` semantics and may be better done caller-side by
   muting those stretches before the request.
3. **Spectrogram transport** (binary vs JSON) and resolution: decide on a measured 3-hour title in WP2.
4. **Catalogue cache on the service** — its own download, or a mounted copy of the desktop's `BEQ_DOWNLOAD_DIR`.
5. **History depth** — one previous design set (proposed) or several.
