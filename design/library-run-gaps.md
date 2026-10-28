# Library run: results and the plan for the remaining gaps

**Plan, partly built** (L1-L5 done, L6 not started). Written 2026-10-19 after the first unattended run over the whole JRiver library
(1,292 titles) with the pipeline service (2.2.0-alpha.12 to alpha.20) and the beqforge designer
(0.2.0 to 0.2.5). Status of each item is in [TODO](TODO.md) (rows L1-L6); this file says what was found
and what each item is. The decline-by-decline evidence is the triage ledger on the share
(`/media/beq/triage/ledger.jsonl`, `findings.md`, `STATUS.md`), not in this repository.

## 1. Results

| | Titles |
|---|---|
| In the library | 1,292 |
| Designed, at review | 1,281 |
| ... with a design to review | 1,008 (confidence quartiles 0.89 / 0.93 / 0.95; 710 at 0.9 or more; 3 below 0.6) |
| ... declined by the designer | 273 (21%) |
| Extraction failed | 9 |
| Published and committed | 2 |
| Flagged *Possible duplicate* | 78 |

Declines by reason: no qualifying loud events 91, no usable plateau 49, insufficient coherent bandwidth 42,
no publishable candidate 33, channel evidence unavailable 30 (every mono title, see L5), within programme
ripple 23, within goal tolerance 5.

**Review of the declines.** 220 of the 273 are in the ledger. Of the verdicts on titles still declined:
140 genuine, 9 genuine at low confidence, 9 the documented steep-filter limitation, 6 an infrasonic
reference (O2) with the right outcome, 32 mono (F3/F4), and 13 suspected designer flaws (below). 18 titles
reviewed as declines now have a design, through the fixes below. 53 declines are not yet reviewed (L1).

**Fixed during the run.**

- F7 (beqdesigner, alpha.14/15): with no stream chosen, extraction took the first audio stream, so stereo
  or dubbed tracks were designed where a lossless multichannel main track followed. Now the richest stream,
  in the language of the first lossless one. 115 titles moved; they re-extracted on their own.
- F2 part 1 (beqforge 0.2.2): diagnose lost a mix plateau touching the band top.
- F1 (beqforge 0.2.3): the texture yardstick counted an LFE knee as ripple. 4 of 5 known cases now design.
- Designer pool (alpha.17+): several designer endpoints, each with its own parallelism, a down one skipped.

**Requeued.** The designer build is deliberately not in the design fingerprint, so a designer upgrade
never redesigns on its own. Two manual redesign jobs bring every title up to the current behaviour:
the 71 declines from 0.2.1/0.2.2 (job 892b0046: 11 now design, 60 still decline) and every other title
designed by 0.2.0-0.2.2, 275 of them (job 28643fea, running). 0.2.4 and 0.2.5 changed no design
behaviour, so titles designed by 0.2.3 or later are current. Working that out needed each build's
source digest computed at each beqforge tag by hand (L3).

## 2. The gaps

### L1. Finish the review of the run (done)

Done 2026-10-28: [L1, L3, L4 record](archive/library-run-l1-l3-l4.md). Every decline in the queue is in the ledger against
the build that made it (stage F for the 44 at versions 7 and 8). A person's review of every design is RV in TODO.

### L2. Extraction failures (done)

Fixed and archived: [L2 record](archive/library-run-l2.md). Seven of the nine now extract (or will, run with
`retry_failed` on the next image); two are encrypted rips and one a file JRiver lists that is gone.

### L3. Know which designs a designer upgrade changes (extends D4) (done)

Done: [design-version.md](design-version.md) and contract 1.6's `/impact` on both sides, and the desktop work list's
*Design* filter and drift-banner count (2026-10-28). Record: [L1, L3, L4](archive/library-run-l1-l3-l4.md).

### L4. Designer findings to hand to beqforge (done)

Done 2026-10-28: every finding is in beqforge's `plans/library-review-improvements.md` (TODO 29) and handoffs v5/v6/v8.
Record and table: [L1, L3, L4](archive/library-run-l1-l3-l4.md).

### L5. Mono titles (done)

Decided and fixed: [L5 record](archive/library-run-l5.md). If the content is there it is recovered, whatever the
channel count; the library route now sends a mono title's one channel as its decomposition, as Batch Extract does.

### L6. Finish what the run started

- **E2.** The run exercised scan, extract and design unattended over the whole library; publish and
  commit ran on 2 titles only. Record the run in the E2 record and do the publish/commit part against the
  disposable repositories.
- **Possible duplicates (78)**: decide which copy each pair keeps (several L2 failures are the second copy).
- **Stale work directories**: titles whose default stream moved (F7) left their old `- audio N`
  directories on the share. A cleanup that lists directories no index row points at, then deletes on
  confirmation.

## 3. Order

L1, L3 and L4 are done; L6 is left.
