# Implementation plan — headless BEQ pipeline

**Status: done.** All six phases below shipped. For the current architecture,
package layout, and workflow, see
[`src/main/python/pipeline/README.md`](../src/main/python/pipeline/README.md).
This file stays only as a phase index for code comments that cite it by
phase/item number (`design/pipeline-implementation-plan.md phase 4 (item 10)`,
etc.) — the design rationale itself lives in the README and in
`api-headless-pipeline.md`.

| Phase | Shipped | Items |
|---|---|---|
| 0 | `test_pipeline_publish_roundtrip.py` — the publish round-trip test, pinned before any refactor | — |
| 1 | `pipeline/config.py`, `filters.py`, `stats.py` + ffmpeg accessor + TMDB-key config + `beq_gain` fix | B1, B2, B5, B3, items 6, 7 |
| 2 | `pipeline/designer/{contract,registry,convert}.py`, `pipeline/metadata.py`, `pipeline/publish/xml.py` | 5, 12 |
| 3 | `pipeline/publish/{art,report}.py` | 8, 9 |
| 4 | `pipeline/publish/git.py` | 10 |
| 5 | `pipeline/orchestrate.py` (`Session`, `Applied`/`Declined`) | 11 |

`test_pipeline_acceptance.py` reproduces the `docs/workflow/beq.md` Ready
Player One example end to end and is the definition of done for the whole
plan; it, `test_pipeline_qt_boundary.py`, and the rest of
`src/test/python/test_pipeline_*.py` are green.

Nothing is left open from this plan — see the pipeline README's "Open / not
built" section for the one still-unconfirmed idea (D6, catalogue-as-input),
which was never part of this plan in the first place.
