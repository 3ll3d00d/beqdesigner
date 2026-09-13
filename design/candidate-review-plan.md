# Implementation plan — candidate review workflow

**Status:** planning → ready to execute. Builds on `designer-interface.md`'s
`DesignResponse.candidates` (a designer may now return several ranked filter
candidates per title, each with its own confidence/commentary — see that
document §3). This plan adds the missing piece: a way to run design over
*many* titles unattended, and a tool for a human to work through the results
and pick a candidate per title, fast, before anything is published.

Four phases, each its own commit (`Phase 1: ...` / `Phase 2: ...`, matching
`pipeline-implementation-plan.md`'s convention). No phase after the first
depends on GUI work from an earlier phase existing yet — 1 and 2 are
Qt-free and independently testable; 3 is the GUI; 4 is docs.

Decisions made while planning (flagged so they're easy to revisit):
- **One JSON file per title** in a queue directory, not one manifest file —
  safe for partial/concurrent batch writes and for re-running a batch
  without clobbering already-reviewed entries.
- **Skip vs. reject are different.** Skip leaves an entry `pending` (come
  back to it later); reject is permanent, human-made, and is otherwise
  treated like a decline downstream (nothing simulated or published).
- **The chart preview is a live widget** (`mpl.MplWidget` +
  `model.magnitude.MagnitudeModel`, the same combination `ui/filter.py`
  already uses for live filter preview), not a pre-rendered image — it
  redraws when the reviewer changes which candidate is selected for a row.

---

## Package layout

```
src/main/python/pipeline/
    review.py               # QueueEntry, queue read/write, batch_design() driver — Phase 1
    designer/
        convert.py           # + candidate_to_complete_filter() made public — Phase 2

src/main/python/model/
    review.py                # Qt glue: loads a queue dir, drives review.py, wraps candidates
                             # for the table model + live chart -- Phase 3

src/main/python/ui/
    review.py / review.ui    # ReviewQueueDialog -- Phase 3

src/test/python/
    test_pipeline_review.py          # QueueEntry round-trip, batch_design(), apply/publish -- Phase 1+2
    gui/test_review_dialog.py        # pytest-qt, headless -- Phase 3

docs/schema/
    review_queue.schema.json          # published spec for the queue entry format -- Phase 4
```

---

## Phase 1 — batch-design driver + review queue

**Ships:** `pipeline/review.py` (`QueueEntry`, `write_queue_entry`,
`read_queue`, `update_entry`, `batch_design`), `test_pipeline_review.py`
covering all of it. No GUI, no publish.

`QueueEntry` (frozen-ish dataclass, mutated only via `update_entry`, which
rewrites the file):

```python
@dataclass
class QueueEntry:
    id: str                       # stable slug, also the filename (id + '.json')
    fs: int                       # publish-target fs the candidates will be realised at
    meta: dict                    # BeqMetadata.to_dict()-shaped, or a partial dict if
                                  # TMDB lookup hasn't happened yet -- resolving metadata
                                  # at review time (not batch time) is deliberately allowed
    curve: dict                   # one MagnitudeData (avg), via model.codec.xydata_to_json --
                                  # enough to redraw the live preview without the original audio
    response: dict                # the full DesignResponse (candidates + decline fields), via
                                  # a small asdict-based (de)serializer in this module -- not
                                  # model/codec.py, which has no reason to know this shape
    status: str = 'pending'       # pending | accepted | skipped | rejected | published
    chosen_candidate_index: Optional[int] = None
    reviewer_note: Optional[str] = None
```

`write_queue_entry(dir, entry)` / `read_queue(dir) -> list[QueueEntry]` /
`update_entry(dir, id, **fields) -> QueueEntry` (read-modify-write one file;
raises if `id` isn't present rather than silently creating one — entries are
only ever created by `batch_design`).

`batch_design(items, designer, queue_dir, config=AnalysisConfig())` where
`items` is `Sequence[tuple[id, source_path, meta_dict_or_None]]`: for each,
runs `Session.extract`/`load`/`design` (reusing `Session` exactly as it
exists today) and writes one `QueueEntry` — status `pending` on an `Applied`
outcome (whatever `candidates` the designer returned), or a `QueueEntry`
with `response.decline_reason` set and status `pending` on a `Declined`
outcome (a decline still needs a human to see *why*, even though there's
nothing to pick). Never calls `set_filters`/`publish` — that's Phase 2, and
only for entries a human has since marked `accepted`.

**Tests:** `QueueEntry` round-trips through `write_queue_entry`/`read_queue`
with a multi-candidate response (commentary included); `batch_design` over
2-3 synthetic titles (reusing `test_pipeline_acceptance.py`'s synthetic-wav
helper) produces one queue file per title with the right status; a declined
title round-trips its `decline_reason`/`decline_message`.

---

## Phase 2 — applying a reviewed decision

**Ships:** `candidate_to_complete_filter` made public in
`designer/convert.py` (currently the private `_candidate_to_complete_filter`
helper `to_complete_filter`/`alternative_filters` already share — no new
logic, just a public name), plus `apply_reviewed_entry` and
`publish_reviewed_queue` in `pipeline/review.py`.

```python
def apply_reviewed_entry(entry: QueueEntry) -> Applied:
    '''
    entry.status must be 'accepted' and chosen_candidate_index must be set.
    Converts response.candidates[chosen_candidate_index] via
    candidate_to_complete_filter(), returning the same Applied shape
    Session.design() produces for an automatic top-pick -- so nothing
    downstream (set_filters, to_beq_xml, report, publish) can tell a human
    was involved.
    '''

def publish_reviewed_queue(queue_dir, xml_repo, meta_defaults=None, **publish_kwargs) -> list[dict]:
    '''
    For every 'accepted' entry: apply_reviewed_entry(), build BeqMetadata
    from entry.meta (filling gaps from meta_defaults), call the existing
    Session.report()/Session.publish() exactly as test_pipeline_acceptance.py
    does today, then update_entry(..., status='published'). Idempotent --
    already-'published' entries are skipped, so re-running after a partial
    failure only retries what didn't make it.
    '''
```

**Tests:** an `accepted` entry with `chosen_candidate_index=1` (not the top
one) publishes *that* candidate's filters, not `candidates[0]`'s — the one
behaviour this phase exists to prove. `publish_reviewed_queue` over a queue
with one `accepted`, one `pending`, one already-`published` entry only
touches the `accepted` one and marks it `published`; a second run is a
no-op.

---

## Phase 3 — the review GUI in beqd

**Ships:** `model/review.py`, `ui/review.py`/`.ui`, a menu action wiring it
into `app.py`, `gui/test_review_dialog.py` (pytest-qt, headless — see
`pytest_qt_gui_testing` conventions already established in this repo:
real widgets, real `QSettings` where needed, watch for the `ui.beq`/`app`
circular import).

A new top-level dialog, opened independently of the main window's loaded
signals (`File` or `Tools` → "Review Batch Designs…"), pointed at a queue
directory:

- **Table** (`QTableView` + a small model over `read_queue()`'s result):
  one row per entry — title, status, top candidate's confidence/method, or
  decline reason if declined. Sorted pending-first.
- **Detail pane**, populated from the selected row: every candidate listed
  (confidence, method, residual, `commentary` as a small key/value table),
  a picker defaulting to `candidates[0]`, and the **live chart** — an
  `mpl.MplWidget` driven by a `model.magnitude.MagnitudeModel`-style
  controller showing the entry's stored unfiltered curve plus the
  currently-picked candidate's filtered curve, redrawing whenever the
  picker selection changes (no re-render step, no PNG — the same live
  pattern `ui/filter.py` already uses).
- **Fast triage**, keyboard-first: Enter/A accepts the current pick and
  advances; digit keys change the pick before accepting; S skips (stays
  `pending`); R rejects. Advancing always selects the next *pending* row,
  not row 0 — same principle as this session's earlier signal-table delete
  fix (`app.py::deleteSignal`).
- **"Publish accepted"** action drives `publish_reviewed_queue` on a
  background thread via the existing `QThreadPool`/worker-signal pattern
  (`model/batch.py`, `model/ffmpeg.py`), with a progress bar and a
  completion summary (published / failed / nothing to do).

---

## Phase 4 — docs

**Ships:** `docs/schema/review_queue.schema.json` (same JSON-Schema
treatment `docs/schema/{filter,signal,project}.schema.json` already got for
the save formats), and a short pointer from `designer-interface.md`'s
revision history to this plan, so a reader following the contract's history
lands here too.
