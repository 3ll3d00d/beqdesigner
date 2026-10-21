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
