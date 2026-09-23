# Library work list -- bounded parallel runs

> Part of the library sync plan -- start at
> [`../library-sync-pipeline-plan.md`](../library-sync-pipeline-plan.md).
> Contains §14. This is the agreed design for chunk 42. Chunks 42a and 42b
> are built; 42c and 42d remain. The existing workflow and work-list contract are in
> [`workflow-rework/design.md`](workflow-rework/design.md) §12.7 and §12.10.

## 14. Bounded parallel work-list runs

### 14.1 Goal

Allow one work-list action to process several independent titles at once,
with the user controlling concurrency separately for each stage. Each active
title must show its current stage and progress in its table row. The user can
open that title's live details to see what the run is doing, including the
command lines sent to external tools such as ffmpeg and git and their
output/errors.

The work-list remains the single place to select and start work. Existing
action labels, selection rules, stage ordering for an individual title,
confirmation, cooperative cancellation, result reporting and index refresh
remain in force.

### 14.2 Current behavior and seams

- `model.worklist_actions.WorkListActions._launch()` creates one
  `model.worklist_run.RunJob`; that job calls
  `pipeline.library.stages.run_stages()` in one worker.
- `run_stages()` processes machine titles in selection order, then publish
  entries, then commits. It emits `Progress` at title-stage boundaries and
  `FfmpegProgress` during extraction. A commit is repository-wide and is not
  cancellable part way through.
- The table model already has a per-title running role, but it only stores a
  stage name and displays it in Needs. A single progress bar and status label
  live in the window's run panel. Those widgets cannot describe several
  simultaneous titles.
- `run_stages()` currently owns one SQLite index connection for the whole
  run, while extraction/design and publishing update shared queue/work
  directories. Concurrency must preserve the index and per-title
  idempotency contracts; adding threads around the current loop is not a
  safe implementation by itself.
- `model.ffmpeg.Executor` builds and executes ffmpeg commands. Git commands
  are issued in the publish/commit path. The detail stream must be emitted
  at these execution boundaries, not reconstructed from log text after the
  command has finished.

### 14.3 User-facing behavior

1. Add independent **Maximum concurrent** settings for **Extract** and
   **Design** to the Library Work List Settings drawer. Each defaults to `1`
   and allows values from `1` through `4`. Persist them under
   `run.parallelism.extract` and `run.parallelism.design`; older profiles
   without these keys continue to mean `1` for each stage. Validate values
   on load and edit. Publish and commit/push remain serialized: publishing
   regenerates shared catalogue aggregates and commits mutate shared working
   trees, so per-title publish concurrency is not a safe independent stage.
2. Give each work-list row exactly two run-status controls: a progress bar
   for its in-flight stage, and a **Details** button. The bar shows the
   current stage and a determinate percentage when available (ffmpeg), or a
   stage-level/indeterminate state otherwise. Queued titles are visibly
   queued; rows with no active or recent run have no run progress to show.
3. The **Details** button opens a title-scoped dialog showing progress so
   far. It is available while the title runs and after it finishes, until a
   later run replaces that title's retained events. The dialog shows a
   timestamped, ordered event stream: queued, stage started, external
   command started, command output/progress, stage completed, cancellation
   and errors. Commands are shown as argument lists in a copyable form, with
   stdout/stderr and exit status where available. New events append live;
   opening the dialog mid-run includes earlier events from that run. Closing
   it does not cancel work.
4. Keep explicit, separately labeled buttons for each available work-list
   action: **Extract & design** (labelled **Extract** when that is all the
   selected work needs), **Publish**, **Commit**, and **Retry failed**.
   Enable each button from the current selection, or from listed rows when
   nothing is selected, and keep its eligible/skipped counts visible. Do not
   replace these buttons with a generic command selector and Execute button.
   The buttons retain the existing confirmations and stage eligibility
   rules; per-title actions such as Accept/Skip/Reject remain on the title
   page.
5. The existing run status area summarizes aggregate counts (queued, active,
   succeeded, failed, cancelled) and retains Cancel. Cancel stops dispatching
   queued titles, lets each already-started title reach its current safe
   boundary, and retains the existing rule that a begun commit finishes.
   The user can still start only one work-list run at a time.
6. Publish and repository-wide commit/push remain serialized. A failure for
   one published title does not discard successful results for other titles.

### 14.4 Execution and state model

Introduce a run coordinator that owns the fixed selection and independent
stage queues. A title advances through its required stages in order, but
releases its current stage slot on completion and can wait in the next
stage's queue while other titles use that stage's capacity. For example, a
title can be designed while other titles are extracting, subject to the
separate design and extraction limits. Enforce each stage's configured
maximum independently; do not let a title-level worker pool accidentally
couple those limits. Aggregate per-title results into the existing
`StagesReport` contract. The global Qt thread pool must not be used as an
unbounded nested fan-out; stage permits are the actual concurrency bounds.

Before allowing `parallelism > 1`, establish and test these invariants:

- `LibraryIndex` uses one SQLite connection with `check_same_thread=False`
  and serializes its public database operations with an `RLock`. Workers may
  share the `LibraryIndex` object only through those methods; never pass its
  raw connection to a worker. Keep `refresh()` on the coordinator after all
  workers join, not concurrent with title work. Stage transitions and index
  updates must be ordered so a downstream stage never sees stale status from
  an upstream stage.
- A title's outputs have one writer. Detect overlapping work units, including
  season entries and member episode extracts, before dispatch. Either group
  conflicting titles into one serial lane or reject the run with an
  actionable explanation; never let two jobs overwrite the same audio,
  queue entry, project, or cache record.
- Preserve extract/design cache fingerprints and failure memory. A worker
  failure is recorded against the same fingerprint/settings key as the
  serial implementation, and one title's exception cannot terminate the
  coordinator or prevent other independent titles from finishing.
- Do not parallelize publish or repository commit/push. Publishing updates
  shared derived catalogue files as well as per-title records; git operations
  use shared working trees and retain the current one-commit-per-repository
  semantics.
- Cancellation stops new dispatch immediately, signals active jobs at their
  established safe boundaries, and reports queued/not-run titles distinctly
  from failed ones.

Progress and detail events carry the run id and title id. UI slots discard
events from a finished or superseded run. Events are delivered to the UI
through Qt signals; workers do not mutate widgets or the table model.

### 14.5 Per-title progress and details contract

Add a runtime row-state value separate from `TitleRow` (which remains the
discovery index snapshot). It contains run id, queue/active state, stage,
stage progress (current/total or fraction when known), a short status line,
and whether retained details exist. The row exposes exactly two run-status
controls: a progress bar for the in-flight task and a Details button. The
table model exposes state through roles; a delegate or compact widgets paint
the controls without changing the index schema. Sorting/filtering continue
to use index values. Work-list action buttons remain in the action area and
operate on the selection or listed rows, not as per-row command selectors.

Add a structured event type rather than passing formatted strings alone. At
minimum it carries run id, title id, timestamp, event kind, stage, message,
optional command argv, output stream, exit code and progress values. Retain
events only for the active/most recent run in memory; no persistent log
format or cross-session history is introduced by this chunk. Apply a
bounded per-title event buffer and visibly indicate when older output was
trimmed, so a verbose subprocess cannot grow memory without limit.

The display must distinguish the command actually executed from a friendly
summary. Keep argument boundaries when launching subprocesses and render a
copyable, shell-quoted form for inspection. Do not log credentials or
authentication headers; redact known secrets from command arguments and
output before sending them to the UI. Preserve useful paths and error text.

### 14.6 Delivery steps

**42a -- Execution safety and event seams**

- Trace shared writes and index operations for extraction, design, publish,
  season/member work, failure recording and refresh.
- Add Qt-free per-title run events and callbacks. Capture ffprobe/ffmpeg and
  git argv, command output, exit status and title/stage context; expose the
  event signal from `RunJob`. Extraction progress remains a separate,
  structured event. Existing `LibraryIndex` methods serialize their access;
  refresh stays on the run thread after title work.
- Add output-resource discovery: a season unit claims its own work/queue id
  and every member episode work id. The scheduler must reject or serialize
  any pair of units with an intersecting resource set.
- Done when tests cover event identity/payload, season/member resource
  overlap and existing per-title failure boundaries. This chunk adds no
  fan-out, so it cannot schedule conflicting outputs concurrently.
  **Done -- commit `817569c`; focused pipeline tests passed (57).**

**42b -- Stage scheduler and independent concurrency settings**

- Implement the coordinator, per-stage queues and permits, cancellation and
  aggregate `StagesReport` construction. A title's stage dependencies remain
  serial while independent titles can occupy different stages concurrently.
- Add `run.parallelism.extract` and `.design` to profile editing/loading and
  validation; each defaults to `1`, with a maximum of `4`. The run snapshots
  both limits at launch.
- Keep publish and commit/push serial. The current publisher regenerates a
  shared catalogue aggregate for each title, so a per-title publish limit
  would introduce competing writes without an isolated write boundary.
- Done when controlled tests demonstrate maximum active work never exceeds
  each extract/design setting, profiles default and persist both limits,
  stage overlap respects per-title dependencies, cancellation leaves queued
  work untouched, and independent failures do not stop other titles.
  **Done -- commit `0f62322`; focused scheduler and settings checks passed.
  Existing unrelated failures: the CLI commit assertion expects two paths
  while the current JSON publisher also writes `xml/database.json`; one
  settings assertion expects older path-validation wording.**

**42c -- Row progress and live title details**

- Replace the stage-only runtime marker with row state, a progress bar and a
  Details button in the table. Keep scan/rescan state and index
  data separate from run state.
- Add the per-title progress-details dialog, event history for the active run,
  command/output display, copy support, bounded buffering and redaction.
- Keep distinct buttons for each available action; update their counts and
  enabled state as selection, filters and run state change. Change the shared
  run panel to aggregate counts and preserve its cancel action. Ensure detail
  dialogs close cleanly when the work-list closes.
- Done when GUI tests cover multiple rows updating independently, opening
  details after events have arrived, live append, close/reopen, errors,
  cancellation, and UI cleanup after completion.

**42d -- Integration and acceptance**

- Exercise extraction/design with several synthetic independent titles and
  a conflicting season/member selection; prove outputs and index states
  match a serial run.
- Exercise publish failures alongside successful titles and verify publish
  and commit remain serialized, with commit/push once per repository.
- Document the setting, progress states, details view and cancellation
  behavior in `docs/library/`.
- Done when focused pipeline and GUI tests pass, the full suite passes, and
  the docs describe the shipped behavior.

### 14.7 Dependencies and limits

Chunk 42 depends on the existing stage planning and work-list actions
(chunks 25 and 26a-26b). It does not require an index schema migration or
change the meaning of `TitleRow.needs`. Every stage limit defaults to one
until the user raises it independently. If the safety analysis finds a class
of work that cannot be isolated safely (notably overlapping season/member
outputs or repository mutation), that class stays serialized behind the
coordinator rather than weakening cache, publish or idempotency guarantees.
