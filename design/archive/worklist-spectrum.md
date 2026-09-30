# Title-page spectrum comparison — completed

Completed on 2026-09-30. The Library Work List and Review Folder title page now
have a Spectrum comparison tab. It renders the same PNG as publication, using
`heatmap_for`, the saved Analyse Signal preferences, 40 Hz panes, and the
profile’s analysis config. It previews the selected candidate with saved
project edits resolved by `preview_published_projects`; it writes no projects.

Rendering is lazy and runs on the thread pool. One worker and one cached image
bound each page’s work. Changed title/candidate requests coalesce, stale results
are discarded, and leaving the page invalidates a pending result. Refresh and
returning from project edits recheck file stamps/preferences. Missing audio,
corrupt projects and conflicting edits are explained on the tab.

Validation with `PYTHONPATH=./src/main/python QT_QPA_PLATFORM=offscreen
UV_CACHE_DIR=/tmp/beq-uv-cache`:

- `uv run pytest src/test/python/gui/test_worklist_spectrum.py -q --tb=short`: **11 passed**.
- `uv run pytest src/test/python/gui/test_worklist_spectrum.py src/test/python/gui/test_worklist_title.py src/test/python/gui/test_worklist_projects.py src/test/python/gui/test_worklist_review.py src/test/python/gui/test_worklist_review_fixes.py src/test/python/test_pipeline_publish_heatmap.py -q --tb=short`: **158 passed**, before adding two final focused cases (project conflict and incomplete setup/profile config).
- `git diff --check`: passed.

The pixel-parity test renders a short synthetic mono source through the real
GUI worker and publication renderer and compares every output pixel. Other
regressions cover candidate changes, saved mono/multichannel edits, errors and
recovery, request coalescing, leaving the page, settings and file invalidation.
