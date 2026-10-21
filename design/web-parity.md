# Web app parity — editing, visualisation, comparison and adjusted redesign

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
4. **Redesign with adjustments**, selectively: one title or a chosen set, with instructions such as *ignore 28–34 Hz*,
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
- **Authority unchanged.** Writes that touch only the queue entry (metadata, artwork, revise to review, discarding a variant) need
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
| Redesign with adjustments, variants | Neither | WP6, WP7 |

## 4. Chunks

### WP0 — Push the remaining title-page logic down (Qt-free)

No user-visible change; the desktop keeps working exactly as before.

- `pipeline/library/metadata_edit.py`: `field_texts`, `parse_audio_types`, `apply_changes`, `save_meta_changes`
  (returns the entry and whether it changed) and `metadata_digest(entry)`. `model/worklist_metadata.py` imports them.
- `pipeline/library/catalogue.py`: `matching_entries` (from `model/worklist_catalogue.py`, already widget-free but
  under `model/`) and `catalogue_filter(filters) -> CompleteFilter` built from a catalogue filter list, plus the
  on-demand reader WP3 uses (below). The desktop's full `database.json` download stays where it is.
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
  stated range, as a binary body (`application/octet-stream`, sent compressed): a fixed little-endian header (magic,
  format version, time and frequency bin counts, first time and step, first frequency and step, dB floor and step)
  then the grid, one byte per cell, time-major. No JSON or base64: the grid is the whole payload, so it is the smallest
  form and decodes straight into a typed array for the canvas. Resolution is bounded (≤ 2048 time × ≤ 512 frequency
  bins, the service taking the max over each time bin so short hits survive): at most 1 MiB before compression. Same
  cache and 202 rule.
- **Client:** a `Spectrogram` component drawing to `<canvas>` with a colour map matching `beq_dark`, linked zoom with
  the magnitude chart's x (frequency) axis, time zoom by drag, *unfiltered | filtered | difference* toggle, and a
  max-over-time strip (the "where are the heavy hits" view of `model/analysis.py`'s `MaxSpectrumByTime`). Lazy: fetched
  only when its tab is opened.
- A **Visualise** area on the review page holds tabs *Magnitude*, *Spectrogram*, *Heatmap*, *Published BEQs* (WP3).
- Tests: cache key and 202/200 behaviour (`test_pipeline_service_spectrum.py`), the quantisation round trip, and the
  component's rendering against a fake canvas context.

### WP3 — Comparison with published BEQs

- **On demand, not the whole catalogue.** `database.json` is 38 MB (15,512 entries); the service never downloads it.
  beqcatalogue already publishes what a per-title lookup needs: `compare/titles.json` (1.1 MB: key, title, year,
  content type, authors; keys end in the TMDB id, e.g. `film-10-cloverfield-lane_333371`) and `compare/<key>.json`
  (about 1 KB: each author's entry with audio types, season/episode, edition and filters). The service fetches the
  index with `If-None-Match` at most once a day, finds the title by TMDB id (then title and year, WP0's matching), and
  fetches that title's file when a review asks, caching both under the work dir with their `ETag`s.
- **beqcatalogue change (small, own repo):** add each entry's `catalogue_url`, `images` and `underlying` to
  `compare/<key>.json` (written by `beqcatalogue/devices.py`), so the per-title file carries everything the panel shows.
  Until it does, the panel shows filters and links to the compare page, without images.
- **Routes:** `GET /v1/titles/{id}/catalogue` — the matching entries (with `include_other=true` for other tracks and
  editions), each with author, edition, audio and links; `GET /v1/titles/{id}/catalogue/{key}/{n}/chart` — that
  entry's filter response and our measured average/peak **filtered by their filter**, on the same grid as WP1's chart.
  Images are proxied through the service (`/v1/catalogue/image?url=`, allow-listed to the catalogue's image hosts,
  cached) so the page loads nothing from third parties.
- **Client:** the *Published BEQs* tab lists matches; ticking one or more overlays, in WP1's workbench, their filter
  response (secondary axis) and *our track under their BEQ* beside *our track under our design*; their published
  image beside ours (WP2's heatmap). A difference trace (ours − theirs, filtered average) answers "how far apart are
  we" at a glance; a one-line summary per entry gives the max boost, the frequency it peaks at and the RMS difference
  across 5–40 Hz.
- Tests: matching and the catalogue filter build (unit), the index/file fetch with ETags against a fake server, the
  image proxy's allow-list and caching (service), the
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

### WP6 — Contract 1.5: design adjustments (both sides)

A general protocol by which a caller **asks for adjustments** to a design and the designer **answers each one**:
honoured, adapted, declined or not understood, saying why. The designer stays free to react as it sees fit; the caller always
learns what was done with what it asked. This is the normal way a caller influences a design, not a carve-out.

It replaces the reading of designer-interface.md §2 "What is deliberately absent" as "nothing may cross in". The
concern that text protects is a designer *inferring* the caller's intent and shaping its answer silently; that stays
ruled out. Restated for 1.5: **a designer acts on no unstated preference; a stated adjustment is an input it weighs
against the evidence and reports on.** §4's line is unchanged: a request can make a design weaker or narrower, and a
designer may decline one that would have it publish what its evidence does not support (an excluded band that held the
only rolloff, a boost where there is no content), saying so in that adjustment's outcome.

- **Request (additive, optional):**

  ```python
  adjustments: list[DesignAdjustment] | None = None   # 1.5

  @dataclass(frozen=True)
  class DesignAdjustment:
      id: str               # the caller's, echoed in its outcome
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
- **Response:** `adjustment_outcomes: list[{id, outcome: 'honoured' | 'adapted' | 'declined' | 'not_understood',
  applied: dict | None, reason: str | None}]` -- one per adjustment, `applied` the parameters actually used (after clamping,
  for `adapted`), `reason` required for anything but `honoured`. A 1.5 designer answers every adjustment; one it does not
  know is `not_understood`, never an error. A designer before 1.5 ignores `adjustments` (the 1.x rule: unknown request
  fields are ignored) and returns no outcomes, which the caller shows as *not acknowledged by this designer*.
  A declined adjustment does not decline the design: the design is the designer's best answer to the adjustments it
  did act on.
- **Discovery (advisory):** `GET /health` adds `adjustment_kinds: {kind: {params: {name: {type, unit, min, max,
  default}}, description}}`. The caller builds its form from it, so a new kind needs no caller release, but discovery
  only informs: the caller may still send a kind the designer did not list (a `note`, or a kind newer than the health
  it read), and the outcome says what happened.
- **Who asks:** the protocol does not care. The caller records provenance on its side: a person on the review page, a
  bulk revise, or (later, if wanted) a profile default sent with every request.
- **Identity:** the adjustments are part of the request, like the audio; not part of `design_settings` (server-wide). The
  caller adds a canonical digest of the adjustments sent to the entry's `design_fingerprint`, so the design cache never serves
  an answer to different adjustments, and a redesign-by-version sends the variant's adjustments again. A request with no adjustments must
  design exactly as 1.4 did (golden test both sides; no `design_version` bump unless a default changes).
- **Caller storage:** see *Variants* in WP7: the adjustments sent and the designer's outcomes are stored with the
  variant they produced, permanently. Kept across re-extraction.
- **Work:** designer-interface.md §2 (the restated rule, `adjustments`), §3 (`adjustment_outcomes`), §4 (declining an adjustment vs
  declining the design), §7 (health) and the conformance spec; `pipeline/designer` (request, health, outcome parsing),
  the fingerprint, `QueueEntry`, the service models; beqforge implements the vocabulary it can and advertises it (a
  handoff in its own `TODO.md`/`plans/`).
- Tests: conformance (each outcome kind, an unknown kind answered `not_understood`, a pre-1.5 designer ignoring the
  field), fingerprint changes with adjustments and not without, cache separation, replay on redesign-by-version.

### WP7 — Adjusted redesign and variants in the browser

- **Variants.** A title holds **several design variants**, not one set of designs. A variant is one design run's
  result: its designs (candidates and rejected), the adjustments sent (each with who asked, when and why), the
  designer's outcome for each, the designer and its identity (1.4), the extraction it was designed from, and when.
  The first unadjusted design is variant 1; each redesign adds one rather than replacing what is there.
  `QueueEntry.variants` replaces the entry's single `candidates`/`rejected` (migrated: today's designs become variant 1
  with no adjustments), with `current_variant` naming the one the review page opens on; `offered`, the digest and the
  decision rules work within the current variant, and accepting records `(variant, design)`.
- **Persistent.** A variant's adjustments are never edited or dropped by the service: redesign-by-version, a re-extract
  or a revise to review all keep them. Re-running a variant (*Redesign this variant*) sends the same adjustments again
  and adds a new variant, so the earlier answer stays for comparison. A variant goes only when a person discards it
  (*Discard variant…*, digest-checked, refused for the accepted or published variant).
- **Pick between them.** The review page has a variant selector (number, adjustments in one line, designer, date, how
  many designs, the outcome summary) beside the design list; WP1's workbench overlays designs from any variants at
  once (one line style per variant), so a person compares across variants and accepts any design in any of them.
- **Revise… with adjustments:** WP5's dialog gains an *Adjustments* section built from the designer's advertised kinds
  (WP6): numbers with their units and bounds, band lists, strategy checkboxes, and always a free-text note. It starts
  from the current variant's adjustments (to tweak them) or empty. Presets for the common ones (*ignore a band*,
  *bass boost +N dB/oct*, *cap the boost*, *higher content edge*); the person sees exactly what will be sent.
- **Pick on the chart:** with the dialog open, drag on the magnitude chart (or the spectrogram) to add an excluded
  frequency band, or a time range on the spectrogram for `exclude_times`; the ranges are shaded on both views.
- **Selective, in bulk:** from the Titles selection (WP5), the same adjustments sent for every selected title, or
  per-title adjustments queued from each review page and then **Run the queued variants** as one job (`needs` design,
  filter `pending_variant=true`).
- **See what it did:** the variant lists each adjustment with its outcome (honoured / adapted with what was used /
  declined or not understood with the designer's reason).
- **Filter on the Titles page:** *Variants* (one, several), *Adjusted* (any, yes, no) and *Adjustments not honoured*
  (any declined, not understood or not acknowledged).
- Tests: the variants migration and its round trip, decisions within a variant, discard rules, replay keeping the
  adjustments, the form from a capability fixture, chart range picking, bulk submit, cross-variant overlay.

### WP8 — Editing the chosen design's filters

The desktop does this by opening the `.beq` project in the main window; the browser edits the design directly.

- **Service:** `GET /v1/titles/{id}/projects` (`worklist_projects.py`'s states, moved down in WP0: mono/multichannel,
  *modified since design*), and `PUT /v1/titles/{id}/design/{n}/filters` `{digest, filters}` — an edited copy of a
  design, stored as a new variant whose one design is marked `edited_from: (variant, n)` and whose adjustments record
  the edit (so it persists, is compared and discarded like any other variant), its
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
2. **Which adjustment kinds first.** Proposed minimum: `exclude_band`, `floor`, `max_boost`, `goal_tilt`, `note`.
   `exclude_times` interacts with `coverage` (a complete programme with stretches excluded is no longer quite one);
   decide whether the designer reports it as an excerpt.

Decided 2026-10-21: the spectrogram is a binary uint8 grid (WP2); the catalogue is read per title on demand from
beqcatalogue's `compare/` files (WP3); a title keeps every variant until a person discards it, and a variant's
adjustments persist (WP7).
