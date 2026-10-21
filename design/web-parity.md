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
  the title page and the service cannot disagree. Where the logic sat in a Qt module (`model/worklist_metadata.py`, `model/worklist_spectrum.py`), WP0 moved it.
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

### WP0 — Push the remaining title-page logic down (Qt-free) — done

Built 2026-10-21; record in [archive/web-parity-record.md](archive/web-parity-record.md#wp0). Moved:
`pipeline/library/metadata_edit.py`, `catalogue_match.py` (plus `catalogue_biquads`/`catalogue_filter`) and
`spectrum.py`. Left for the chunk that needs it: the artwork checks (WP4: they read images with Qt's `QImageReader`),
`worklist_revise.py`'s words (WP5) and `worklist_projects.py` (WP8, already Qt-free).

### WP1 — The chart workbench — done

Built 2026-10-21; described in [web-app.md §4.1](web-app.md#41-the-chart), record in
[archive/web-parity-record.md](archive/web-parity-record.md#wp1). Smoothing goes to 1/48 octave since the smoother was
replaced (below the WP8 record: it used to integrate bands and inflate the level).

### WP2 — Spectrogram — done

Built 2026-10-21; described in [web-app.md §4.4](web-app.md#44-spectrogram-and-heatmap), record in
[archive/web-parity-record.md](archive/web-parity-record.md#wp2). The binary grid was measured on a 3-hour synthetic
title: 2.4 s to compute, 314 KB. The service draws with the default `HeatmapSpec`, not a person's Analyse Signal
preferences (it has none).

### WP3 — Comparison with published BEQs — done

Built 2026-10-21; described in [web-app.md §4.5](web-app.md#45-published-beqs), record in
[archive/web-parity-record.md](archive/web-parity-record.md#wp3). Deviations: no images (the comparison files have
none; the plan's image proxy waits for them), and beqcatalogue writes a comparison entry only for a BEQ that matched
one of its device profiles, so a title may have published BEQs the tab does not list. Both are beqcatalogue changes
(`beqcatalogue/devices.py`: write every entry, with `images`).

### WP4 — Metadata and artwork editing — done

Built 2026-10-21; described in [web-app.md §4.2](web-app.md#42-metadata-and-artwork) and the route table of
[pipeline-service.md](pipeline-service.md), record in [archive/web-parity-record.md](archive/web-parity-record.md#wp4).
Choosing among TMDB's posters is not built: a poster's address can be pasted into *Download*.

### WP5 — Revise, retry and stream choice — done

Built 2026-10-21; described in [web-app.md §4.3](web-app.md#43-revise-and-the-audio-stream), record in
[archive/web-parity-record.md](archive/web-parity-record.md#wp5). Not built: *select all matching the filter* (the
selection is per page, up to 100 at a time), and reading a title's streams from its file when the source lists none
(the desktop's probe; the browser says to use the app). Retry is *Revise* to design or extract, or New job's *retry
failed*.

### WP6 — Contract 1.5: design adjustments (both sides) — caller side done

**Caller side built 2026-10-21** (record in [archive/web-parity-record.md](archive/web-parity-record.md#wp6-caller-side)):
the contract types and checks, the HTTP binding (sent only to a designer whose `/health` says 1.5), the queue entry's
`adjustments` and `adjustment_outcomes`, the design fingerprint, `POST /v1/revise`'s `adjustments`, and the kinds every
endpoint lists on `/v1/status`. **Left: beqforge** implements the kinds it can and lists them on `/health` (a handoff
in its own repo). Until it does, the browser has no kinds to offer and a design asked for adjustments is refused by
the binding with the designer's version.

**Contract agreed 2026-10-21** and written into [designer-interface.md](designer-interface.md) (§2 "Adjustments",
§3 "Adjustment outcomes", §7), the three `docs/schema/http_designer_*.schema.json` and the conformance spec's "1.5".
The first kinds are `exclude_band`, `floor`, `max_boost`, `goal_tilt` and `note`; `ceiling`, `goal_tolerance`,
`strategies`, `max_sections` and `exclude_times` wait (a designer may offer its own as `x-` kinds meanwhile). The
contract is normative; where the sketch below differs, the contract wins. Not built on either side.

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

### WP7 — Adjusted redesign and variants in the browser — done

Built 2026-10-21; described in [web-app.md §4.6](web-app.md#46-adjustments-and-design-runs) and
`pipeline/library/variants.py`, record in [archive/web-parity-record.md](archive/web-parity-record.md#wp7). Built
differently from the sketch: the entry's own design fields stay the **current** variant (so decisions, publishing,
the index and the desktop title page are unchanged) and `QueueEntry.variants` holds the others; a design is accepted
from the current variant, so taking an earlier design is *Use this one*, then Accept. `adjusted_with` (what a design was
made with) is kept apart from `adjustments` (what the next one is asked). The Titles filters on adjustments and
variants followed the same day (`adjusted`, `unmet`, `several_runs`). Not built: picking a band by dragging on the
chart (the *Ignore the chart's range* preset uses the zoomed range instead). Queuing per-title adjustments for one run is *Revise* without *Extract & design now*, then a run.

### WP8 — Editing the chosen design's filters — done

Built 2026-10-21; described in [web-app.md §4.7](web-app.md#47-editing-a-design), record in
[archive/web-parity-record.md](archive/web-parity-record.md#wp8). An edit is a new design run (WP7's variants) rather than
an extra design beside the designer's: it is reviewed and accepted like any other, and publish writes it through the
existing project rules. Not built: a projects route (the desktop's *modified since design* badge): an edit in the
browser does not touch the `.beq` projects.

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

1. **`exclude_times`** interacts with `coverage` (a complete programme with stretches excluded is no longer quite
   one): decide whether the designer reports it as an excerpt before adding the kind.

Decided 2026-10-21: the contract rule and the first adjustment kinds (WP6); the spectrogram is a binary uint8 grid (WP2); the catalogue is read per title on demand from
beqcatalogue's `compare/` files (WP3); a title keeps every variant until a person discards it, and a variant's
adjustments persist (WP7).
