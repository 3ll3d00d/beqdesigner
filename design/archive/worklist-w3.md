# W3 — Review Folder lease guard (completed)

Completed on 2026-09-30; implementation commit `1923e28` (before the
completion-record amendment). Review Folder now refuses Publish and Commit when a
fresh work-directory lease is held, names the holder, and rechecks after
confirmation. Its worker acquires and holds its own lease until all writes
finish, releasing it before completion signals. Stale leases do not block it.
Current behavior is described in the service architecture reference.

Regression coverage uses real Review Folder widgets and temporary git repos:
both actions, service/CLI/work-list holders, stale takeover, lease ownership
during writes, and another holder appearing during confirmation.

Validation (prefix `PYTHONPATH=./src/main/python QT_QPA_PLATFORM=offscreen
UV_CACHE_DIR=/tmp/beq-uv-cache`):

- `uv run pytest src/test/python/gui/test_worklist_review_fixes.py src/test/python/gui/test_worklist_review.py -q`: **58 passed**.
- `uv run pytest src/test/python/gui/test_worklist_real_pipeline.py src/test/python/test_pipeline_service_lease.py -q`: **12 passed**.

The normal uv cache was read-only; the temporary cache allowed the same tests
to run without changing project dependencies.
