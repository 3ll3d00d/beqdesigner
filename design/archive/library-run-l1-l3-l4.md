# L1, L3, L4: the library run's review, design version, and findings handed on (done 2026-10-28)

From [the library run plan](../library-run-gaps.md).

## L1. Every decline reviewed against the build that made it

The queue held 252 declines on 2026-10-28: 208 at design version 6, already in the triage ledger as stage E
(handoff-v6), and 44 at versions 7 (8, beqforge 0.2.12) and 8 (36, beqforge 0.3.1: .10 container
`unknown+src:66850800b23a`, .29 frozen `v0.3.1+src:333a24c98384`). Those 44 were reviewed and written as stage F rows
(`/media/beq/triage/ledger.jsonl`, 716 rows): 43 give the same reason with the same failing checks as their stage E
review, and one is new, 22 Jump Street, a designed->declined flip on a near-miss cliff check (40.7 vs 35.5 dB/oct),
the same shape as Platoon's v6 flip. 19 titles that declined on v6 design on v8; 4 of them had been reviewed as genuine
declines (The Adventures of Priscilla, The Book Thief, The Manchurian Candidate, Bugonia), and their designs need a
person's look during RV.

The plan's "16 still declined at version 8" counted only the DV8 redesign's targets. The other 28 were titles the
redesign left out, whose v7/v8 designs came from earlier runs.

## L3. Which designs a designer upgrade changes

Built both sides earlier ([design-version.md](../design-version.md), contract 1.4, then 1.6's `/impact`). What was
left was the desktop half, done in `7e3698d`: the work list's *Design* filter (Any / Declined / Out of date) and the
out-of-date count on D4's drift banner, with *Show them*. The desktop asks its profile's designer through the same
`pipeline.library.setup.designer_identity()` the service uses, which moved there from `pipeline/service/context.py`.
Tests: `gui/test_worklist_design_filter.py` (filter, selection, banner, a pre-1.4 designer, the manual designer, a
designer that stops saying its version), `test_pipeline_library_setup.py::test_designer_identity_is_the_highest_version_among_the_endpoints_that_are_up`.
Full suite: 3680 passed, 1 skipped.

## L4. Designer findings handed to beqforge

Each of the plan's seven findings is in beqforge's `plans/library-review-improvements.md` (its TODO 29), and the ones
still open are carried in handoff-v5 and handoff-v6 (its TODO 31 and 35):

| Finding | Where in beqforge |
|---|---|
| 1. F2 part 2 (Before the Devil Knows You're Dead, Blink Twice) | plan §1 (wording fix released in 0.2.9); handoff-v6 §3 |
| 2. Texture retune (Carrie, Changeling, Mirai, Song of the Sea, Fame, Despicable Me 3) | TODO 26 (blocked on E10); plan's research tables |
| 3. F6, the 45 dB scene margin (Blade II, Do the Right Thing, Jason Bourne, Sunshine) | plan research 5; handoff-v6 §3 |
| 4. F5, a dialogue-band reference (A Real Pain, Eastern Promises) | plan's research; handoff-v5/-v6 |
| 5. O2 infrasonic reference; O1 DC offset (Desperately Seeking Susan) | plan §8: O2 built in 0.2.9; O1 built report-only (`beqforge/dc_offset.py`) |
| 6. A plateau that starts at the filter floor (So I Married an Axe Murderer) | plan §7; handoff-v6 §3 |
| 7. Steep filters (9 titles) | TODO 5 and 6 |

What the version 8 review found (section L1 above) is in `/media/beq/triage/handoff-v8.md`, sent to the beqforge session
on 2026-10-28 as evidence for its rows 32 (fit optimum selection) and 36 (whether a design is worth publishing). It
found no new kind of problem.
