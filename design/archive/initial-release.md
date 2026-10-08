# Initial release milestone — completed items

Completed items from the initial release milestone in [TODO](../TODO.md),
in the order they were done. Each records its completion date, implementation commit and the
validation run.

## R7 — Windows CI failure in the publish aggregate test (completed)

Completed on 2026-10-08, in the commit "Read the publish aggregate test's
JSON as UTF-8 on every platform". Since `b325752`,
`test_pipeline_publish_readable.py::test_titles_in_several_letter_folders_share_one_aggregate_in_their_category_folder`
failed on `windows-2022` and `windows-2025` (CI run `36975470653`, its only
failure). The publisher was not at fault: `catalogue_json.aggregate()` encodes
`database.json` as UTF-8 explicitly and `git.write_files` writes bytes. The
test opened the file without `encoding=`, so Windows decoded it as cp1252 and
read `Élite` as `Ã‰lite`. The CI log showed the two the other way round
(`Élite` against `�lite`) because the cp1252 console output was displayed as
UTF-8. Every text `open()` in that test file now names `encoding='utf-8'`.
Windows desktop publishing is unaffected.

Validation: `PYTHONPATH=./src/main/python uv run pytest -q
src/test/python/test_pipeline_publish_readable.py`: **19 passed** (Linux). The
Windows result is the next `main` CI run. Turning `EncodingWarning` into an
error was not usable as a suite-wide guard: third-party imports raise it at
collection.

## R1 — Transient failures are not remembered (completed)

Completed on 2026-10-08, in the commit "Report unavailable dependencies
without remembering them as title failures". Before, `_remember_failure`
recorded every exception, so a designer outage, a dropped mount or a JRiver
that did not answer failed every title in a tick for good.

- `pipeline/library/failure.py` `unavailable_reason(error, source_path)`
  walks the error's cause chain. A dependency is unavailable on
  `requests.Timeout`/`ConnectionError`, an HTTP 5xx, a builtin
  `TimeoutError`/`ConnectionError`, a network or remote-filesystem errno
  (`ENOTCONN`, `ESTALE`, `EHOSTDOWN`, ...), `Unavailable`, or the new
  `http_binding.DesignerUnavailable` (`/health` unanswered, or no
  by-reference support). An HTTP 4xx in the chain is the title's own failure.
  For an extract failure, `missing_mount()` also calls it unavailable when the
  nearest existing folder is empty (a mount point with nothing mounted) or the
  path's root is absent (a disconnected drive or share). A missing file in a
  populated folder, or a POSIX path with only `/` above it, stays the title's
  failure.
- `run_unit`/`design_unit_work` share `_report_failure`, which puts an
  unavailable title in `LibraryRunReport.unavailable` and remembers nothing.
  A season episode that meets one ends the season the same way, without
  remembering the episode.
- `run_stages` and `run_library` count unavailable titles in a row
  (`UnavailableStreak`). A title that reached its dependencies, done or
  failed on its own account, resets the count. At
  `run.stop_after_unavailable` (default 3) the run stops like a cancel:
  titles in hand finish, nothing new starts, extracted titles waiting for
  design are not designed, publish/commit do not run, `StagesReport.stopped`
  says why, and the rest are in `not_run`. `cancelled` stays false.
- `StagesReport.failed` (the CLI exit status, the service job state) includes
  `unavailable` and `stopped`. The CLI warns about both, the service's
  `RunResult` carries them (OpenAPI regenerated), notifications list them,
  and the work list shows *Unavailable* and *Not run* lines with the reason.

Tests: `test_pipeline_library_failure.py` (the classification);
`test_pipeline_library_stages.py` (a designer timeout, connection error and
503, a JRiver connection error and a missing mount are not remembered and the
next unattended tick runs them; a 4xx is remembered and not retried; the stop
leaves the rest in `not_run` and the next tick runs them all; a title failure
ends a streak; config validation); `test_pipeline_library_run.py`
(`run_library`'s stop, an unavailable mount inside a season); the CLI golden,
setup and work-list config parsing, the work-list results wording and
notifications.

Validation: `PYTHONPATH=./src/main/python QT_QPA_PLATFORM=offscreen uv run
pytest -q -n auto src/test/python`: **2616 passed, 1 skipped** (the skip is the
Windows-only drive-letter test).
