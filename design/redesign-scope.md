# Redesign scope: which out-of-date designs to redesign (caller side)

**Status:** built on this side (TODO row RS), against a fake designer; live use waits for beqforge's `/impact`. The contract is
[`designer-interface.md`](designer-interface.md) 1.6 (§3 "The run record" and "Reviewed designs", §7.1 "Impact"),
agreed with beqforge on 2026-10-24. beqforge returns `run_record` from 0.2.11; its `/impact` is its TODO 34 and lands
after design version 7, so this side is built and tested against a fake designer first.

## Why

A beqforge release that bumps the design version makes every design out of date (§3), and until now the only answer
was to redesign the whole library: about 20 hours, for a change that may move a handful of titles. `/impact` lets the
designer say, per title, how likely its answer is to move and why; a person redesigns those and keeps the rest, which
are then stamped *reviewed at* the new identity and stop showing as out of date.

## What gets built, in order

1. **The run record is kept** (built).  `DesignResponse.run_record` (1.6) is decoded and validated (relative POSIX path, no
   backslash, no `..`), carried on `Applied`/`Declined`, and stored on the queue entry as `run_record` (written only
   when known, so an older reader never meets it) and with it into `variants` when the design becomes an earlier run.
2. **Reviewed at** (built; `pipeline.library.reviewed`). The queue entry gains `reviewed_at`: a list of `{design_version, design_settings, at, by}`, written
   only when non-empty. `DesignIdentity` carries the reviewed identities, and `is_out_of_date` is false against any of
   them as against the one it was designed at (§3). A redesign or an edit starts with none, and each design run keeps its
   own, so using an earlier run brings its reviews back. The index summary carries it, so the out-of-date filter, the drift banner and `/v1/status` follow it.
3. **An `/impact` client** (built; `pipeline.designer.impact`, tested against `fake_impact_designer`, a real HTTP
   server checking each request against the schema): `/health`'s `impact` read into `EndpointIdentity`; the endpoint
   to ask is one that is up, reports `impact: true` and the current identity (version and settings, never the build).
   Records go in requests of at most 500, one after another, with a 120 s timeout each; a design without a
   `run_record` is not sent and is `unknown`. The response is checked (one result per record, in order; tiers and
   scores in shape) and an answer out of shape is an error, not a guess.
4. **The service** (built; `pipeline.service.impact`). `POST /v1/impact` checks the out-of-date designs (or given ids)
   in the background -- not as a job, which would wait behind a run for hours and touches nothing -- one check at a
   time, its answer kept in `<state_dir>/impact.json` across restarts (one cut short by a restart says so): per title
   its tier, score, reasons and the changes listed, read most likely to move first. `GET /v1/impact` serves it. An
   answer for another identity than a redesign would use fails the check. `POST /v1/reviewed` stamps the given titles
   reviewed at the current identity, each refused with its reason (not out of date, in a run's hands, the designers
   disagree). Redesigning goes through the existing `/v1/revise` with the chosen ids. `Title.reviewed_at` shows a
   title's reviews.
5. **The web app** (built; `components/ImpactPanel.tsx`). On Titles, with *Out of date*: *Check what a redesign would
   move*, then the titles grouped by tier (unknown with will move), highest score first, each with its reasons; select
   and *Redesign selected* or *Keep selected*. An answer for another design version than the designer's now is marked
   stale. A title's reviews are on `/v1/titles` (`reviewed_at`); the review page does not show them yet.
6. **Scores are probabilities** (agreed with beqforge 2026-10-25, still 1.6, additive): a score is the designer's
   measured estimate that a title moves, so the panel shows the expected movers for the answer and for the titles
   selected (an unknown counts as one). Each change may say what counts as moved (`moved`) and what its scores were
   measured on (`basis`, movers of measured per outcome class), shown beside it; an unknown may give
   `unknown_reason` (a design with no run record is `no_record`).

## What it does not do

- Nothing is redesigned or stamped without a person choosing it (§7.1: advisory).
- Records written before beqforge 0.2.11 carry no identity and had no `run_record` on the entry. beqforge names the
  record in the design's commentary (an absolute path on the machine that designed it), so `pipeline.library.run_records`
  gives each current design `records/<that name>`, relative to the work directory (dry run by default, `--apply` to
  write; then rescan). beqforge recognises such a record by the release that wrote it; one from before design versions
  is `unknown` (`unknown_build`).
