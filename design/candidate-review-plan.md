# Implementation plan — candidate review workflow

**Status: done.** All four phases below shipped. For the current
`QueueEntry` workflow and GUI, see
[`src/main/python/pipeline/README.md`](../src/main/python/pipeline/README.md#batch-design--review)
and `docs/schema/review_queue.schema.json` (the published format). This file
stays only as a phase index for code comments that cite it by phase number.

| Phase | Shipped |
|---|---|
| 1 | `pipeline/review.py` — `QueueEntry`, `write_queue_entry`/`read_queue`/`read_entry`/`update_entry`, `batch_design()` |
| 2 | `apply_reviewed_entry()`, `publish_reviewed_queue()` in `pipeline/review.py` |
| 3 | `model/review.py` + `ui/review.py` — the `ReviewQueueDialog` (table, live chart, keyboard triage, "Publish accepted") |
| 4 | `docs/schema/review_queue.schema.json` |

Decisions made while planning, still true of the shipped code:
- One JSON file per title in a queue directory, not one manifest — safe for
  partial/concurrent writes and re-running a batch without clobbering
  already-reviewed entries.
- Skip leaves an entry `pending`; reject is permanent and, like a decline,
  nothing runs downstream of it.
- The candidate chart is a live widget (redraws on pick change), not a
  pre-rendered image.

Nothing is left open from this plan.
