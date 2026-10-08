# Initial release milestone — completed items

Completed items from the initial release milestone in [TODO](../TODO.md),
newest last. Each records its completion date, implementation commit and the
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
