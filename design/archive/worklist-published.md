# Title-page published BEQs — completed

Completed on 2026-09-30. Library Work List and Review Folder title pages have a
Published BEQs tab: matching authors and entries, catalogue metadata, image
selection and a single image pane, plus an explicit link to the catalogue page.
Saved metadata and the discovery row identify the title. TMDB identities take
precedence over normalized title/year fallback, movie and TV remain separate,
and audio/known edition/language/source and TV season/episode overlap narrow
results. Include other tracks / editions broadens the title’s publications.
Metadata matching does not assert identical audio.

Lookup is lazy, with background downloads, a shared Browse Catalogue database
cache, validated images cached by full URL, bounded requests and atomic database
replacement. Failed refreshes use the old cache. Image selection can retry a
failure. One image worker coalesces selections; stale images are discarded.
Metadata reload changes the matches. Browsing never writes a review decision,
changes the chosen candidate or imports catalogue filters. `ScaledImage` is
shared with the spectrum comparison and preserves status text on resize.

Validation with `PYTHONPATH=./src/main/python QT_QPA_PLATFORM=offscreen
UV_CACHE_DIR=/tmp/beq-uv-cache`:

- `uv run pytest src/test/python/gui/test_worklist_published.py -q --tb=short`: **13 passed**.
- Broader work-list selection: **287 passed**; two setup errors and one failure
  were caused by the sandbox denying local sockets in existing TMDB HTTP tests.
  The full suite below runs with local-socket access enabled.
- `uv run pytest src/test/python -n 4 --cov=./src/main/python --tb=short -q`:
  **2562 passed, 32 warnings**, 69% coverage. This includes all spectrum
  comparison, work-list, Review Folder, metadata and real HTTP regressions.
- `git diff --check` and focused module compilation: passed.
