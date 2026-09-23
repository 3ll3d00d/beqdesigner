# Work-list run state follow-up

> **In progress.** Fix 44a is committed as `9761873`; fix 44b remains. This is
> §16 of the library sync plan, scheduled as chunk 44. Chunk 43 and its acceptance coverage are recorded in
> [§15](worklist-parallel-runs-remediation.md); this file records two defects
> found after that implementation.

## 16.1 Findings

1. `run_stages()` emits commit-wide `stage_started`, `stage_completed` and
   `failed` events with an empty title id. It separately emits title-scoped
   events for each committed title and duplicates shared git command events
   to those titles. `WorkListWindow._on_execution_event()` currently creates
   a row outcome for the empty id, so a two-title commit can show
   "2 queued · 1 active" before either title has started. A failed commit
   can also add a phantom failure to the aggregate count. The empty id has
   no visible table row or Details control.
2. Cancelling a run gives an undispatched title the row text "Cancelled".
   Starting a later run clears its Details history and `has_details` flag,
   but leaves that text in runtime row state. A title outside the new
   selection therefore still shows old progress with no corresponding
   details. `WorkListModel.set_rows()` also retains state for ids temporarily
   absent from the index, so clearing only currently listed rows can revive
   an older status when an id appears again.

## 16.2 Chunk 44 — Correct event identity and expire row state

- Treat only ids in the active run's planned title set as per-title outcomes.
  Commit-wide events can inform the shared run status, but must not create a
  row buffer, row state, or outcome under the empty id. Keep the existing
  title-scoped commit and command events available in each affected title's
  Details dialog. Apply the same rule to unexpected ids, so aggregate counts
  cannot exceed the selected title count.
- At the start of a new run, expire the previous generation's **entire**
  runtime row state: progress text, fraction, stage, active/queued flags and
  Details availability, including ids not currently listed. Then initialize
  queued state only for the new run's planned titles. Preserve index-backed
  title data, selection and filters.

## 16.3 Acceptance and verification

- Add a real-widget commit test driven by the actual `run_stages()` event
  sequence, including the empty-id stage events and shared git commands.
  Assert that aggregate counts total exactly the selected titles throughout
  success and failure, and that each title's Details contains its command
  events. A smaller slot-level test may isolate an unexpected id.
- Add a real-widget sequence: cancel a run with one title undispatched, start
  a disjoint run, and confirm the old row has no progress text, fraction or
  Details. Remove and reinsert a title through `set_rows()` between runs and
  check that its old state does not return.
- Run the focused work-list action and pipeline stage tests, then the broader
  relevant suite. Record the exact commands and results. The three existing
  commit-result assertion failures belong to the JSON-output migration;
  report them separately rather than treating a deselected run as clean.

Mark chunk 44 complete in the plan index with its commit hash only after the
regression tests and behavior changes are in the same commit.
