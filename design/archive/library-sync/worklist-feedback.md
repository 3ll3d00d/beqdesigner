# Work-list feedback, 2026-09-27 — design

> **Historical archive:** original path `design/worklist-feedback.md`. This is the design record of F1-F5, all built; it is not the current status. The maintained account is [implemented](../../implemented.md) and, for the lease and joining, [pipeline-service §5.1](../../pipeline-service.md#51-the-work-directory-lease-and-joining-a-run). Source comments cite `worklist-feedback.md` Fn by these IDs.

Five changes asked for after using the work list on a real library.

| ID | Change | Status |
|---|---|---|
| F1 | A *Working* chip for titles being extracted or designed | Built in `de45afe` |
| F2 | Write the `.beq` projects when a title is extracted | Built in `7420a54` |
| F3 | A decline is a flat (empty) filter a person can publish as "does not require BEQ" | Built in `49bbf34` |
| F4 | A failed extraction stays extractable; Revise is not offered for it | Built in `c58a4b4` |
| F5 | New work joins the run in progress (work list, CLI and service) | Built: F5a `3d38f76`, F5b `ae1a086`, F5c `dc527f5`, F5d (the service) `12d875e` |

Decisions taken with the user on 2026-09-27: the CLI hands its titles to a run
in progress and waits for them (F5); only runs a person starts retry a failed
extraction, the scheduler does not (F4); a published flat filter gets the note
"Does not require BEQ" unless the reviewer wrote one, and *Accept top pick*
never accepts one (F3).

## F1 — Working chip

A title in a run keeps the `needs` it had when the run began (a title extracted
through design is refreshed only when its design ends), so it sits under
*Extract* until it is done. The strip gets a **Working** chip, after
*Attention*, listing the titles the window's run has queued or is working on,
from the model's run state (not the index: the index says what a title needs,
not who is doing it). Its count is the run's titles not yet finished; the
*Needs* cell of a working title keeps saying the stage in hand. The run also
refreshes the index after a title's extraction, as it does after its design, so
its row moves from *Extract* to *Design* while it waits for a design slot.

## F2 — Projects at extraction

`design_and_queue()` writes the mono (and multichannel) `.beq` projects only for
an Applied design. The run now writes them as soon as a title is extracted,
with an empty filter, if they do not exist yet; design then overwrites them
through the existing hash gate (`write_title_projects_if_safe()`), so a project
a person has edited in between is left alone, as it is today. An existing
project is never replaced at extraction.

## F3 — Decline as a flat filter

The contract's decline stays as it is. The queue entry of a decline gets one
candidate with an empty filter (`method` `declined`, no confidence, the decline
laid out as its commentary), and keeps `decline_reason` and `decline_message`;
an entry written before this is given the same candidate when it is read. So a
declined title can be accepted, its project and chart show a flat response,
and publishing writes a record with `"filters": []` — the catalogue's existing
convention for a title with no BEQ — with the note "Does not require BEQ"
unless the reviewer wrote one. *Accept top pick* skips it (no confidence).

## F4 — Failed extraction

`derive_needs()` makes a failed extraction `extract` (detail "extract failed:
…") instead of `attention`. `plan_stages()` plans it, and `run_stages()` tries
it again, unless the run is **unattended** (the scheduler's), which keeps
skipping a remembered failure until the source or settings change. A failed
design stays `attention`. *Revise* is offered only when a selected title has
something to send back: an extraction or a queue entry.

## F5 — Joining the run in progress

Every run holds the work-directory lease (today only the service does), and a
run's machine phase takes extra titles while it lasts:

- `run_stages(join=...)` polls a `JoinQueue` (`pipeline/library/join.py`) each
  time it looks for work, and every quarter second while titles are in hand:
  each request is a selection planned `through` extract or design (never past
  it), minus titles already in the run; its titles go to the back of the queue
  and the total grows. When the machine phase ends the queue is closed: an offer
  is refused, and what was offered but not taken is in `report.not_joined`.
- The inbox is a directory, `<work_dir>/service/join/`: a request is a JSON file
  written atomically; the runner claims it by renaming it to `.taken`, and the
  poster withdraws one not yet claimed by renaming it to `.withdrawn` (whichever
  rename wins decides). This lets a run in one process take work from another.
- **Work list**: while its run is going, the action button and *Retry failed*
  stay enabled and add to it; Publish, Commit and a bulk accept or revise
  (which cannot join a machine phase) wait and start when it ends, in order.
- **CLI** `run`: if a fresh lease is held, it posts its selection to the inbox
  and waits: once the runner claims it, until the run ends (the lease is
  released), then it reports its titles from the index and exits (0 if none
  failed); if the lease is released first, it withdraws the request and runs
  itself. Its own runs take the lease
  and serve the inbox.
- **Service**: a submitted run job whose `through` is extract or design joins
  the running run job's machine phase when there is one: the new job is
  `running` at once and finishes when the run it joined does. Otherwise, or if
  the run has left its machine phase, it queues as today.
