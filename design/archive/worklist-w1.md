# W1 — Retry and failure details (completed)

Completed on 2026-09-30; implementation commit `69253c9` (before the
completion-record amendment). Retry labels identify extraction, design, or a mixed
selection. The title page has a Failures tab with the complete redacted indexed
message, a separate current-attempt label, text selection and Copy failure.
Its Run Details button opens the same dialog as the table’s Details cell.
Persisted failures remain inspectable without retained events; current run
events carry redacted commands/output, and cache hits explicitly say no command
ran. The last run’s event cache is still expired when another run starts.

A retry snapshots the previous indexed failure, retaining it in the page and
the Detail tooltip even across an intermediate extraction-to-design index
refresh. The final refreshed result removes it on success, replaces it on
another failure, or retains the index’s failure on cancellation. Open Details
dialogs follow the refreshed result too.

**Clarification of the original W1 wording:** the separate Failures and Last
run panels were removed by the delivered work-list redesign before W1. They
are not restored. The existing main table remains the failure list, with a
Failures tab on each title page and a shared Run Details view. The user guide
now describes these actual controls and Retry’s selection/filter behavior.

Regression coverage: both failed stages, full multiline failure text beyond
the event-buffer limit, text selection and clipboard copy, secret redaction,
retry success/repeated failure/cancellation, intermediate refresh, disjoint
later runs, persisted failures without event history, and cache-hit details.
Existing action tests now assert stage-specific labels and the retained prior
failure tooltip. W3’s lease guard was committed separately as `ac3744e` before
this task.

## Validation

Commands use `PYTHONPATH=./src/main/python QT_QPA_PLATFORM=offscreen
UV_CACHE_DIR=/tmp/beq-uv-cache`:

- `uv run pytest src/test/python/gui/test_worklist_failures.py -q --tb=short`: **11 passed**.
- `uv run pytest src/test/python/gui/test_worklist_failures.py src/test/python/gui/test_worklist_actions.py src/test/python/gui/test_worklist_title.py src/test/python/gui/test_worklist_projects.py src/test/python/gui/test_worklist_window.py -q --tb=short`: **236 passed** (before the additional intermediate-refresh regression; the final full suite includes it).
- `uv run pytest src/test/python -n 4 --cov=./src/main/python --tb=short -q`: **2538 passed, 31 warnings**, **68% total coverage**, with local socket access.
- `git diff --check` and local design-document link/TODO-order checks: passed.

The initial sandboxed full-suite run was interrupted after socket-dependent
fixtures failed. A focused reproduction,
`uv run pytest src/test/python/test_pipeline_designer_http_binding.py -x -q --tb=short`,
failed at `HTTPServer(('127.0.0.1', 0), ...)` with
`PermissionError: [Errno 1] Operation not permitted`. The same full-suite command
passed with socket access. One attempted focused command named the nonexistent
`gui/test_worklist_model.py` and collected no tests; the corrected command above
uses `gui/test_worklist_window.py`.
