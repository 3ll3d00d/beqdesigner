# Parallel work-list run remediation

> **Implemented in chunk 43.** This is §15 of the library sync plan. Fixes
> landed in `51d3595` (43a), `39f8b32` (43b), and the current 43c commit. The
> shipped behavior and original contract are in
> [§14](worklist-parallel-runs.md); the current status is in the
> [plan index](../library-sync-pipeline-plan.md).

## 15.1 Goal and scope

Make parallel runs readable and predictable in the work-list UI. The review
found that the shared progress bar switches between titles, Details buttons
look available before they have events, live Details updates force the reader
to the end, the Needs cell is not notified when its running label changes,
and histories from unrelated older runs remain in memory. These are UI and
runtime-state fixes; extraction/design concurrency limits and the index
schema do not change.

The existing cancellation contract is that **a dispatched title finishes its
requested stages**, including design after extraction; Cancel prevents further
titles from being dispatched. §14.4's phrase "stops new dispatch immediately"
is ambiguous about the already-dispatched title's next stage. The user guide
and existing cancellation test both use the title as the safe boundary. This
chunk clarifies that wording and tests the boundary. A scheduler change is
needed only if the test shows that a title not dispatched before Cancel starts.

## 15.2 Remediation and acceptance

### 43a — Run identity, retained details, and row notification

- Keep one in-memory event-history generation: clear all prior title buffers
  and their `has_details` row state at the start of a new run, including rows
  not selected this time. Close any open dialog showing the previous run
  before clearing it. Keep the finished run's events until the next run or
  window close.
- Associate incoming signals with the `RunJob` that emitted them as well as
  the pipeline's run id. An event delivered late from a finished job must not
  initialize the next run's id, recreate an old buffer, or update a row.
- When `set_run_state()` changes the stage marker used by Needs, emit a model
  notification covering Needs and both run columns. Preserve index-backed
  sorting and filters.
- Add real-widget tests for two successive runs on disjoint title selections,
  a Details dialog open across the transition, a late event from the first
  job, and Needs updates on stage start/completion. Assert both the model data
  and the user-visible availability of Details.

### 43b — Aggregate progress and Details affordance

- Use the shared `runProgress` bar for completed **titles out of planned
  titles**. Derive it from unique terminal outcomes; a stage completion does
  not complete a title. Set its final value from `StagesReport`, including
  failed and cancelled titles. Keep ffmpeg's percentage exclusively in that
  title's row. The shared status text and count label can name the latest
  activity and aggregate queued/active/succeeded/failed/cancelled totals.
- Do not reset the shared bar when another extraction starts or allow
  interleaved ffmpeg packets to change its value. Review the existing
  `Progress` and `FfmpegProgress` slots so each updates only its intended
  display.
- Paint the row's Details button unmistakably disabled while `has_details`
  is false, and enable it as soon as the first event is retained. Keep mouse
  behavior aligned with its visual state; support keyboard activation when
  the Details cell has focus.
- Add a controlled two-title Qt test: hold one extraction at 80%, start the
  second, interleave their progress packets, and verify the shared bar never
  falls or adopts either ffmpeg percentage while both row bars remain
  independent. Test the Details button before and after the first event by
  rendering the delegate and attempting activation.

### 43c — Details reading position and documentation

- Append new event text without replacing the whole document on each update.
  Follow new output only when the reader is already at the bottom. Preserve
  scroll position and text selection while the reader is inspecting earlier
  events. When the bounded buffer trims old events, rebuild the visible text
  from that buffer and preserve the nearest available reading position; keep
  the existing visible trim notice and copy behavior.
- Add a real-widget test that scrolls up and selects text, delivers several
  live events, and verifies that position and selection remain usable. Check
  auto-follow at the bottom, close/reopen, trim behavior, and Copy all.
- Clarify §14.4 and `docs/library/work.md` about the title-level cancellation
  boundary, aggregate progress, and details expiring when **any** later run
  starts. Cover cancellation before dispatch,
  during extraction, and while a dispatched title waits for a design slot.
  Assert that no undispatched title starts, started titles finish, and
  `attempted`/`not_run` and UI counts agree.

## 15.3 Verification and completion

Each behavior change has a regression test. The offscreen Qt cases cover two
successive disjoint runs, an open Details dialog during the transition, a late
event from the old job, Needs notifications, aggregate and per-row progress,
the rendered Details delegate and its keyboard activation, live reading
position, auto-follow, trimming and Copy all. Pipeline cancellation cases cover
before dispatch, between titles and while dispatched work waits for a design
slot. The full pipeline stage suite passed: 36 tests. The final Qt action module
run had 50 tests: 47 passed, with the three existing commit-result JSON-output
migration assertions still failing (`test_commit_results_are_listed_per_title_and_per_repository`,
`test_a_push_only_commit_says_push_in_its_button_its_confirmation_and_its_outcome`,
and `test_the_results_show_each_new_publish_and_commit_result_shape`).

The chunk is complete when the UI contracts above have passing regression
coverage, the user guide matches the final behavior, and the plan index
records the implementation commits. Update the chunk status in the same
commit as each completed piece, as required by the plan's working convention.
