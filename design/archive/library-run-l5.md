# L5: mono titles (decided and fixed 2026-10-19)

From [the library run plan](../library-run-gaps.md). Every mono title in the library declined
`channel_evidence_unavailable`: the library route sent no decomposition for a one-channel source (F3), though Batch
Extract / Design sends its one channel as `{"FC": ...}` (designer-interface.md §2), so the two routes disagreed.

**Decision (F4): if the content is there, recover it; the channel count is not a principle.** beqforge already works
that way: it has no channel-count rule and blocks only when no decomposition is sent (`if not material.channels`), so
given the one channel it applies the same content evidence (loud events, tracking, the support ceiling) as to a 5.1
title. The fix is `f775fa0` (first written as `ebca201` on 2026-10-16, held back for this decision).

**Evidence, beqforge v0.2.8, each title's mono channel sent as FC** (28 titles with audio on disk):

| Outcome | Titles |
|---|---|
| Designed | 7: Blazing Saddles, Bonnie and Clyde, Dog Day Afternoon, Missing, The Big Heat, Tootsie, White Heat |
| No qualifying loud events | 16: A Streetcar Named Desire, Adam's Rib, All the President's Men, Being There, Casablanca, Dark Passage, Day for Night, Double Indemnity, Mildred Pierce, Rififi, The Big Sleep, The Killers, The Lady from Shanghai, The Maltese Falcon, The Panic in Needle Park, The Philadelphia Story |
| No publishable candidate | 2: After Hours (a steep peak then rolloff), Badlands (tilt) |
| Insufficient coherent bandwidth | 2: Woman of the Year, Wuthering Heights |
| No usable plateau | 1: Network |

Missing, The Big Heat and The Lady from Shanghai also have a richer stream (F8) and move to it.

One design is not content recovery: Bonnie and Clyde's material has mastering hum notches (-10.9 dB at 60 Hz, -9.6 dB
at 120 Hz) and the design fills them (+13.2 dB at 62.5 Hz, +11.8 dB at 121 Hz), bringing the hum back. That is a
designer flaw that is not about mono (Butch Cassidy and the Sundance Kid, +20.9 dB on a -8.6 dB notch at 61 Hz; The
Birds, +7.9 dB on a -10.2 dB notch at 120 Hz), handed to beqforge as F11 in its TODO 29 proposal. White Heat's design
(+21.8 dB, Q 5.6 at 92 Hz beside a -10 dB notch at 100 Hz, and a shelf at 327 Hz) is worth a person's look in review.

After deploying, a mono entry's design inputs change (it now has a decomposition), so the schedule redesigns the
mono titles on its own; accepted and published entries are protected as always.

Tests: `test_pipeline_library_run.py` (the mono decomposition sent as FC; a season's joined track sends none).
