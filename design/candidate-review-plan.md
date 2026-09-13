# Implementation plan — candidate review workflow

**Status:** Phases 1-4 done. Builds on `designer-interface.md`'s
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

`QueueEntry` (plain dataclass, mutated only via `update_entry`, which
rewrites the file). **Revised while implementing:** rather than a bespoke
serializer for the raw `DesignResponse`/`DesignCandidate`/`BiquadSpec`
shapes, each candidate's filters are realised into a `CompleteFilter` at
`entry.fs` immediately (`to_complete_filter`/`alternative_filters`, already
existing) and stored via `CompleteFilter.to_json()` — the same, already
tested/published format as `docs/schema/filter.schema.json`. This also
means Phase 2 needs no new public API from `designer/convert.py` (see
below):

```python
@dataclass
class CandidateSummary:
    filters: dict              # CompleteFilter.to_json(), already realised at entry.fs
    confidence: float
    method: str
    mv_adjust_db: float
    residual_db: Optional[float] = None
    residual_band_hz: Optional[tuple] = None
    commentary: Optional[dict] = None

@dataclass
class QueueEntry:
    id: str                       # stable slug, also the filename (id + '.json')
    fs: int                       # publish-target fs the candidates were realised at
    meta: dict                    # BeqMetadata.to_dict()-shaped, or a partial dict if
                                  # TMDB lookup hasn't happened yet -- resolving metadata
                                  # at review time (not batch time) is deliberately allowed
    curve: dict                   # one MagnitudeData (avg), via model.codec.xydata_to_json --
                                  # enough to redraw the live preview without the original audio
    candidates: List[CandidateSummary] = field(default_factory=list)  # empty on decline
    decline_reason: Optional[str] = None
    decline_message: Optional[str] = None
    status: str = 'pending'       # pending | accepted | skipped | rejected | published
    chosen_candidate_index: Optional[int] = None
    reviewer_note: Optional[str] = None
```

`write_queue_entry(dir, entry)` / `read_queue(dir) -> list[QueueEntry]` /
`read_entry(dir, id) -> QueueEntry` / `update_entry(dir, id, **fields) ->
QueueEntry` (read-modify-write one file; raises `FileNotFoundError` if `id`
isn't present rather than silently creating one — entries are only ever
created by `batch_design`).

`batch_design(items, designer, queue_dir, work_dir, config=AnalysisConfig())`
where `items` is `Sequence[tuple[id, source_path, meta_dict_or_None]]`: for
each, runs `Session.extract`/`load`/`design` (reusing `Session` exactly as
it exists today) and writes one `QueueEntry` — status `pending` either way,
`candidates` populated from an `Applied` outcome (`[primary] +
alternatives`, in rank order) or empty with `decline_reason`/
`decline_message` set from a `Declined` outcome (a decline still needs a
human to see *why*, even though there's nothing to pick). Never calls
`set_filters`/`publish` — that's Phase 2, and only for entries a human has
since marked `accepted`.

**Tests (`test_pipeline_review.py`, done — 11 tests):** `QueueEntry`
construction validation (accepted-without-index, out-of-range index,
invalid status); round-trip through `write_queue_entry`/`read_entry` with
multiple candidates and commentary; `read_queue`'s pending-first ordering;
`update_entry` rewriting a file and rejecting a missing id;
`batch_design` over two synthetic titles producing one pending entry each,
top-ranked-first; a declined title carrying its reason/message; the
per-module no-`qtpy`-import check every other `pipeline/` module gets.

---

## Phase 2 — applying a reviewed decision

**Ships:** `apply_reviewed_entry` and `publish_reviewed_queue` in
`pipeline/review.py`. No `designer/convert.py` changes needed after all —
see the Phase 1 revision above: each `CandidateSummary.filters` is already
a realised `CompleteFilter.to_json()` dict, so applying a pick is just
`model.codec.filter_from_json(chosen.filters)`, not a re-run of the
designer-contract conversion.

```python
def apply_reviewed_entry(entry: QueueEntry) -> CompleteFilter:
    '''
    entry.status must be 'accepted' (chosen_candidate_index in range is
    already enforced by QueueEntry construction/update_entry).
    filter_from_json(entry.candidates[entry.chosen_candidate_index].filters)
    -- so nothing downstream (set_filters, to_beq_xml, report, publish) can
    tell a human picked this over the top-ranked candidate.
    '''

def publish_reviewed_queue(queue_dir, xml_repo, meta_defaults=None, images_repo=None, image_owner=None,
                           image_repo_name=None, xml_dir='', image_dir='', report_spec=ReportSpec(),
                           config=AnalysisConfig()) -> list[dict]:
    '''
    For every 'accepted' entry: apply_reviewed_entry(), build BeqMetadata
    from entry.meta (a constructor-kwargs dict -- see the Phase 1 revision
    below) filling gaps from meta_defaults (defaulting `gain` to the chosen
    candidate's mv_adjust_db, catalogue-compat), a fresh report image from
    the entry's stored curve + chosen filter when images_repo is given
    (MagnitudeData.filter(complete_filter.get_transfer_function().get_magnitude())
    to get the filtered curve back from the stored unfiltered one), then
    Session.publish() -- the same sequence test_pipeline_acceptance.py
    exercises for an automatic top-pick. update_entry(..., status='published')
    on success. Idempotent -- anything not 'accepted' (including already-
    'published') is left alone, so re-running after a partial failure only
    retries what's still 'accepted'.
    '''
```

**Revised while implementing:** `QueueEntry.meta` turned out to need a
correction from Phase 1's own description -- it must be `BeqMetadata`
*constructor* kwargs (`{'title': ..., 'year': ..., ...}`), not
`to_dict()`'s `beq_`-prefixed XML-ready shape, since `BeqMetadata(**meta)`
has to reconstruct it here. Fixed in both `QueueEntry`'s docstring and
`batch_design`'s.

**Tests (`test_pipeline_review.py`, done — 8 more, 19 total):**
`apply_reviewed_entry` picks `candidates[1]`'s filters when that's what was
chosen, not `candidates[0]`'s, and raises on a non-`'accepted'` entry.
`publish_reviewed_queue`: XML-only publish against a real local bare+clone
repo (same pattern as `test_pipeline_acceptance.py`); `gain` defaulting
from the chosen candidate's `mv_adjust_db`; the image path (report PNG
pushed, `image_url` returned); a `pending` entry in the same queue is left
untouched; a second run against an already-`published` entry is a no-op.

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

**Implementation notes / scope cuts:**
- `.ui` authored by hand (this session has no Qt Designer GUI access) and
  compiled with `uv run pyuic6 review.ui -o review.py`, same as
  `readme.md`'s documented workflow — not hand-edited afterwards. Same for
  the `action_Review_Batch_Designs` menu entry added to `ui/beq.ui`
  (Tools menu) and regenerated into `ui/beq.py`.
- **"Publish accepted" only asks for the XML repo directory** (a
  `QFileDialog` directory picker, same UX as the queue-dir picker) and
  calls `publish_reviewed_queue` XML-only (`images_repo=None`) — no
  images-repo/owner/repo-name UI yet. `publish_reviewed_queue` already
  supports images; this is a scope cut in the dialog only, not the
  pipeline, and is additive to fill in later.
- `ReviewQueueDialog.load_queue_dir(path)` is a small public seam added
  beyond the original sketch so tests (and `app.py`, if it ever wants to
  pre-point the dialog at a queue) don't have to reach into private state.
- Confirmed the `ui.beq`/`app` circular import (documented in the
  `pytest_qt_gui_testing` memory) is pre-existing and unaffected by this
  change — a cold `import app` fails identically before and after; the
  real app (launching `app.py` as `__main__`) and every test file that
  does `import ui.beq` first are both unaffected.

**Tests (`gui/test_review_dialog.py`, done — 9 tests):** loading a queue
populates the table; selecting a row populates the candidate list
(defaulting to `candidates[0]`) and commentary table; a declined entry
shows its reason and disables Accept; a digit key changes the pick without
accepting; Accept writes the *chosen* candidate (not necessarily the top
one) and advances to the next pending row; Skip leaves the entry `pending`
and advances; Reject is permanent and distinct from Skip; advancing never
jumps back to row 0 regardless of where the reviewer was; the chart's data
provider returns the unfiltered + currently-picked-candidate curves.

---

## Phase 4 — docs

**Ships:** `docs/schema/review_queue.schema.json` (same JSON-Schema
treatment `docs/schema/{filter,signal,project}.schema.json` already got for
the save formats — `candidates[].filters` `$ref`s `common.schema.json`'s
existing `completeFilter` def, `curve` its `xyData` def), and a short
pointer from `designer-interface.md`'s revision history to this plan, so a
reader following the contract's history lands here too. Validated the same
way the save-format schemas were: generated real `QueueEntry`/
`CandidateSummary` instances (one multi-candidate, one declined) through
the actual dataclasses and checked both against the schema before
committing it — not hand-guessed.
