# Redesign scope: which out-of-date designs to redesign (caller side)

**Status:** design, partly built (TODO row RS): steps 1-3 are built. The contract is
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
4. **The service.** An *impact* job kind over the out-of-date designs (or given ids), whose result is kept: per title
   its tier, score, reasons and the changes listed. `GET /v1/impact` serves the latest; `POST /v1/reviewed` stamps the
   given titles reviewed at the current identity (refused for a title that is not out of date, or when the designers
   disagree). Redesigning goes through the existing `/v1/revise` with the chosen ids.
5. **The web app.** On Titles, when designs are out of date: *Check what would move*, then the out-of-date titles
   grouped by tier (unknown with will_move), highest score first, each with its reasons; select and *Redesign* or
   *Keep* (stamp reviewed). The title page shows a design's reviewed-at identities.

## What it does not do

- Nothing is redesigned or stamped without a person choosing it (§7.1: advisory).
- Records written before beqforge 0.2.11 carry no identity and have no `run_record` on the entry: every current design
  is `unknown` until it has been designed once on a build that writes them.
