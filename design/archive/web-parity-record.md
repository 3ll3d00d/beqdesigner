# Web app parity — delivery record

Completed chunks of [../web-parity.md](../web-parity.md) (TODO WP), with their commits and evidence. Lasting behavior is
described in the architecture references; this file is history.

## WP0

**2026-10-21.** The title page's remaining no-widget logic moved under `pipeline/library/`, so the service can share it:

- `metadata_edit.py`: the field list and vocabularies, `field_texts`, `changes_from_texts` (the panel's text-to-change
  rules, formerly `MetadataPanel.changes()`), `apply_changes`, `save_meta_changes`, the badge words, and the TMDB reload
  (`tmdb_reload`, raising `TmdbError` with the key redacted; `tmdb_fill`). `model/worklist_metadata.py` imports them.
- `catalogue_match.py` (was `model/worklist_catalogue.py`), plus `catalogue_biquads` (expands the `count` of
  beqcatalogue's `compare/` files) and `catalogue_filter`; `model.catalogue.CatalogueEntry.iir_filters` uses them.
- `spectrum.py`: `ComparisonRequest`, `comparison_request` (now takes a `spec` as well as desktop preferences) and
  `render_comparison`.

Evidence: new `test_pipeline_library_metadata_edit.py`, `test_pipeline_library_catalogue_match.py`,
`test_pipeline_library_spectrum.py`; the gui tests that patched the old module paths now patch the new ones.
`PYTHONPATH=./src/main/python QT_QPA_PLATFORM=offscreen uv run pytest -n auto src/test/python`: 3235 passed, 1 skipped.

## WP1

**2026-10-21.** The review chart became a workbench.

- Service (API 1.6.0): `GET /v1/titles/{id}/chart` takes `candidates` (`all`, or indexes), `smoothing` and `responses`;
  `ChartSeries` carries `candidate` and `kind` `filter`. Built on `review_chart.workbench_curves()`, which
  `chart_curves()` (the desktop's) now calls.
- Browser: `src/chart/view.ts` (the `ChartView`, kept in `localStorage`; visibility, normalising, the shared axis,
  zooming), `ChartControls`, and `MagnitudeChart` drawing the view with drag, wheel and double-click zoom and a second
  axis for responses.
- Found on the way, committed apart (`fa8e0f8`): two Titles filter changes before a redraw lost the first, the cause of
  the intermittent "filters from the row under the headings" failure.

Evidence: `test_pipeline_library_review_chart.py` and the chart tests in `test_pipeline_service_review.py`;
`src/chart/view.test.ts`, `src/pages/workbench.test.tsx`. Web: typecheck, lint, 88 tests (10 consecutive full runs) and
the build pass. Python: `pytest -n auto src/test/python`, 3257 passed, 1 skipped.

## WP4

**2026-10-21.** Metadata and artwork are edited in the browser.

- Shared (Qt-free): `metadata_edit.FIELD_SPECS` and `metadata_digest()`; `pipeline/library/artwork_edit.py` (the image
  checks now decode with Pillow instead of Qt's `QImageReader`; `store_image()` for an image sent; `set_artwork()`).
  `model/worklist_artwork.py` keeps only the download `QRunnable`.
- Service (API 1.7.0, `pipeline/service/edits.py`): `GET /v1/metadata/fields`, `PATCH /v1/titles/{id}/metadata`,
  `POST .../metadata/tmdb`, `GET`/`PUT`/`DELETE .../artwork`; `Review` carries `metadata_digest`, `metadata_texts` and
  `artwork`. A refused Accept now says the Metadata panel.
- Browser: `MetadataPanel` on the review page.

Evidence: `test_pipeline_service_edit.py` (21), `test_pipeline_library_artwork_edit.py` (9); `src/pages/metadata.test.tsx`
(11). Web: typecheck, lint, 99 tests (3 consecutive full runs), build. Python: `pytest -n auto src/test/python`, 3289
passed, 1 skipped.

## WP5

**2026-10-21.** Titles are sent back, and their audio stream chosen, from the browser.

- Shared: `revise.choose_stream()` (the desktop's *Choose audio stream* write, now also the service job's).
- Service (API 1.8.0): `POST /v1/revise`, `GET /v1/titles/{id}/streams`, `PUT /v1/titles/{id}/stream`; the redesign job
  takes `revise_to` (design or extract) and, for one title, `audio_stream`; a stream may be chosen before a title's
  first design.
- Browser: `ReviseDialog` (review page and Titles selection), `StreamPicker`, the Titles selection column.

Evidence: `test_pipeline_service_edit.py` (revise, streams), `test_pipeline_service_work.py` (re-extract, a person's
stream, request checks), `gui/test_worklist_stream_choice.py` unchanged; web `review.test.tsx` (revising one title,
choosing the stream, revising from the list). Web: typecheck, lint, 103 tests (8 consecutive full runs), build. Python:
`pytest -n auto src/test/python`, 3303 passed, 1 skipped. The metadata panel test now waits for the fields' own request.

## WP2

**2026-10-21.** A title's spectrogram and publication heatmap are drawn in the browser.

- Shared: `spectrum.spectrogram_grid()` and `SpectrogramGrid` (encode/decode).
- Service (API 1.9.0): `GET /v1/titles/{id}/heatmap.png`, `GET /v1/titles/{id}/spectrogram`, on
  `pipeline/service/visuals.py`'s `RenderQueue` (202 while drawing; files kept under `<work>/<id>/web/`).
- Browser: `src/chart/grid.ts`, `Spectrogram`, `HeatmapView`, the chart card's tabs. Looked at in Chrome against a
  running service with a synthetic 10-minute track: the steady tone and three bursts where expected, the low shelf's
  +4.5 dB in *Difference*, the heatmap PNG.

Evidence: `test_pipeline_service_visuals.py` (6), `test_pipeline_library_spectrum.py` (grid tests); web
`src/chart/grid.test.ts`, `src/pages/visuals.test.tsx`.

## WP3

**2026-10-21.** Published BEQs are compared in the browser, read per title on demand.

- Shared: `pipeline/library/catalogue_remote.py` (`CatalogueCache`, `same_track`), `review_chart.published_curves()`.
- Service (API 1.10.0, `pipeline/service/catalogue.py`): `GET /v1/titles/{id}/catalogue`,
  `GET /v1/titles/{id}/catalogue/{n}/chart`; `BEQ_CATALOGUE_URL`.
- Browser: `PublishedBeqs`, the fourth tab.

Evidence: `test_pipeline_library_catalogue_remote.py` (7, over `catalogue_fixture.FakeCatalogue`),
`test_pipeline_service_catalogue.py` (4); web `src/pages/published.test.tsx` (4).

## WP6 (caller side)

**2026-10-21.** Contract 1.5 on beqdesigner's side.

- `pipeline/designer/contract.py`: `DesignAdjustment`, `AdjustmentOutcome`, `DesignRequest.adjustments` (the request
  says 1.5 only with some), `DesignResponse.adjustment_outcomes`; `convert.validate_outcomes()`.
- `http_binding`: adjustments in the body; `check_takes_adjustments()` once before the first request with any (refused
  below 1.5); outcomes parsed; `reported_adjustment_kinds()`.
- `orchestrate.Session.design(adjustments=)`, outcomes on `Applied`/`Declined`; `QueueEntry.adjustments` and
  `adjustment_outcomes`; `design_and_queue(adjustments=)`; `design_cache`: part of the fingerprint (only when there are
  some), replayed by every redesign; `EntryFacts.adjustments` so the index judges them alike.
- Service (API 1.11.0): `ReviseRequest.adjustments`, `Review.adjustments`/`adjustment_outcomes`,
  `DesignerStatus.adjustment_kinds` (the kinds every endpoint up lists).

Evidence: `test_pipeline_designer_adjustments.py` (17), adjustment tests in `test_pipeline_service_edit.py`; the full
suite, 3349 passed, 1 skipped.

## WP7

**2026-10-21.** Adjusted redesigns and kept design runs.

- Shared: `pipeline/library/variants.py` (`kept_variants`, `next_variant_id`, `use_variant`, `discard_variant`,
  `variant_view`); `design_cache.design_if_needed` keeps the run a redesign replaces; `QueueEntry.variant_id`,
  `designed_at`, `variants`, `adjusted_with`.
- Service (API 1.12.0): `Review.variant_id`/`designed_at`/`variants`/`adjusted_with`; `/chart?variant=`;
  `POST .../variants/{vid}/use`, `DELETE .../variants/{vid}`.
- Browser: `AdjustmentsEditor` in `ReviseDialog`, `VariantsPanel`, other runs drawn dashed.
- Found by the variants tests: revise wrote the next design's adjustments onto the entry before the redesign, so the
  run it replaced would have been kept with the wrong ones; hence `adjusted_with`.

Evidence: `test_pipeline_library_variants.py` (6), variant tests in `test_pipeline_service_edit.py`; web
`src/pages/adjust.test.tsx` (6). Web: 121 tests (3 runs), build (it now passes Vite's 500 kB warning: 507 kB, 163 kB
gzipped). Python: 3360 passed, 1 skipped.

## WP8

**2026-10-21.** A design's filters are edited in the browser.

- Shared: `convert.specs_to_complete_filter()`, `variants.save_edited()`.
- Service (API 1.13.0): `PUT /v1/titles/{id}/edited`.
- Browser: `src/chart/biquad.ts` (held to `model/iir.py`'s values), `FilterEditor`, *Edit filters…*. Looked at in
  Chrome against a running service: the edit drawn live, saved as v2 with v1 kept.

Evidence: the edit tests in `test_pipeline_service_edit.py`; web `src/chart/biquad.test.ts`, `src/pages/editor.test.tsx`.
Web: 131 tests (3 runs). Python: 3366 passed, 1 skipped.

## Smoothing fixed (desktop and web)

**2026-10-21.** `model.xy.smooth_octave`, which the main window, the title page and the service's chart all use, called
the vendored `acoustics.smooth.fractional_octaves`: it summed the *amplitude* of every point in each 1/N-octave band
onto band centres from 5 Hz (10 Hz at 1/12 and finer), and `amplitude_to_db` then clipped 80 dB below the top. A flat
-20 dB spectrum on a 1 kHz Welch grid came out at +25 to +59 dB at 1/1, rising with frequency, and everything below the
first band was lost. It is now a fractional-octave *smoother*: each point the mean power over the 1/N octave centred on
it, weighted by the span each point stands for, on the curve's own frequencies, unclipped; a point alone in its window
(and 0 Hz) is unchanged. A flat spectrum keeps its level; on a real track the 5-120 Hz mean moves by under 2 dB at 1/1
and the peaks soften. 1/48, which the main window's menu already offered and the old code refused, now works, so the
service and the browser offer it too. The vendored `acoustics` module is left as it is (it is band integration, correct
for what it is).

Evidence: `test_xy_smoothing.py` (12); the review-chart and chart-route smoothing tests now hold the level and the
frequencies; the full suite, 3379 passed, 1 skipped.

## Finishing (21 October 2026)

- **Titles filters on adjustments and design runs:** `EntryFacts` keeps `adjusted`, `unmet` and `runs`, `TitleRow` reads
  them from the entry summary, `Selection`/`TitleFilter` filter on `adjusted`, `unmet` and `several_runs` (API 1.14.0),
  and `Title` carries them for the list's badges. Evidence: `test_pipeline_library_adjusted_filters.py` (7), the web
  test "filtering on adjustments and design runs".
- **A band picked by dragging on the chart:** `AdjustmentsEditor` *Pick a band on the chart*, the next drag on the
  magnitude chart becomes an `exclude_band`, the bands shaded (`MagnitudeChart` `bands`). Evidence: the web test "takes a
  band to ignore from a drag across the chart".
