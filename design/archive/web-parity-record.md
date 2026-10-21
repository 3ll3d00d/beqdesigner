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
