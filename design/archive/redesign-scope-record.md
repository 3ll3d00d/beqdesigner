# Redesign scope (TODO RS) — completion record

**Completed 2026-10-28.** The delivered behaviour is the reference [design/redesign-scope.md](../redesign-scope.md);
the contract is [designer-interface.md](../designer-interface.md) 1.6 (§3, §7.1). This records how it was built and
the live evidence.

## Build

The caller side was built on 2026-10-25 against a fake designer, in the five steps of the reference, plus scores as
probabilities (`915ff97`), the `run_record` backfill for pre-1.6 designs (`586cc5c`, `pipeline.library.run_records`),
the release smoke expecting a run record only from a 1.6 designer (`a025022`), and the index rereading a cached
summary that lacks a field this version keeps (`a0cb694`, released in 2.2.0-alpha.29).

## Live evidence

- **Backfill:** applied to all 1,287 queue entries (2026-10-25, 12:56-13:09), then a rescan.
- **Client check** against beqforge's `/impact` build (`impact-0b20b49`), 2026-10-25: 1,287 answers, no schema errors.
- **DV7 sample** (beqforge 0.2.12, 105 titles, 2026-10-25 15:30-17:31 UTC): moved 15 of 35 scored 0.47, 15 of 60
  scored 0.13, 0 of 10 unaffected. Per title in `/media/beq/triage/impact-step4-moved.json`.
- **Service check** on 2.2.0-alpha.30 with beqforge 0.3.1 (design version 8), 2026-10-26: the first `/v1/impact`
  answered only the 105 sampled titles (`no_record` for the rest, from cached summaries); after a rescan it answered
  all 1,287, none `no_record`, about 600 expected movers.
- **Cut-off, the user's (2026-10-26):** redesign every title with any chance of moving (1,051) and keep the 236 scored
  unaffected. A random 20 of those 236 were redesigned as a check (2026-10-27): none moved, all declined before and
  after.
- **The scoped redesign** (three jobs of at most 500 ids, window 23:30-05:30, 2026-10-26 to 28): all 1,051 at design
  version 8, 728 moved (69%) against about 600 expected; 16 still declined. Per title in
  `/media/beq/triage/dv8-redesign-moved.json`.

| Band (estimate) | Moved | Rate |
|---|---|---|
| likely 0.88, from v6 | 247 of 314 | 0.79 |
| likely 0.78, from v7 | 10 of 23 | 0.43 |
| likely 0.53 (declines) | 19 of 34 | 0.56 |
| possible 0.46, from v6 | 411 of 608 | 0.68 |
| possible 0.14, from v7 | 41 of 72 | 0.57 |
| unaffected (sample) | 0 of 20 | 0 |

The total was close but the bands were not: beqforge's version 8 scores were measured mostly on synthetic variants,
and on this library version 8 alone moves roughly half of the designed titles whatever their band. The figures went
back to beqforge. One request to `.29` (*The Little Sister*) hung to the 900 s timeout; run alone it designed in 80 s.

## Follow-ons

The throughput gap the run showed is TODO IG; reviewing the redesigned titles is TODO RV.
