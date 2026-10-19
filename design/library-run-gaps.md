# Library run: results and the plan for the remaining gaps

**Plan, not built.** Written 2026-10-19 after the first unattended run over the whole JRiver library
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

### L1. Finish the review of the run (operational)

- Review the declines of job 28643fea as they arrive, the 53 unreviewed ones, and re-confirm the
  flaw-type verdicts of the 60 that declined again on 0.2.5 (a verdict reached on an older build is
  evidence, not a conclusion).
- The second designer (10.151.167.29, the Windows build) is offline until after this run, so the
  requeue designs on `.10` alone, which keeps a record of each. The Windows build kept no run record for
  any of its 432 titles (111 declines): beqforge keeps none unless told to. Handed to beqforge as its
  TODO 27 (records and a log file by default, `record_dir` on `/health`); until then start it with
  `--record-dir` on the share's `work\records`. A decline among those 432 that needs its evidence is
  redesigned once to get a record.
- No design carries the contract 1.3 fit fields yet: beqforge's fit reporting (`92407e8`) is unreleased.
  Once released, the fit shows on titles designed after it; a redesign only for the fit is not worth it.

**Done when** every decline in the queue is in the ledger, against the build that made it.

### L2. Extraction failures (beqdesigner bugs)

Nine titles failed extraction and are retried, and fail, on every scheduled run:

- `KeyError: 'codec_type'` (Elf): `model/ffmpeg.py:450` indexes every ffprobe stream's `codec_type`;
  a stream without one (an attachment or data stream) breaks the probe. Read it with `.get`.
- `ZeroDivisionError` (House Party, Le Samourai, Parasite): find the division (a zero duration or rate
  from the probe is the likely input) and fail with a reason that says so.
- `ffprobe error` / `ffmpeg error (see stderr output for detail)` (House Party, John Wick 3, The Dark
  Knight, Prizzi's Honor, Silkwood): the stderr the message points at is not kept anywhere a person
  can read it. Put its last lines in the failure detail, then decide per file (damaged rip, unsupported
  container, the duplicate copy of a title).

Each fix needs a regression test from a fixture probe (`test_ffmpeg.py`). **Done when** the nine either
extract or fail with a reason a person can act on.

### L3. Know which designs a designer upgrade changes (extends D4)

Today the only identity a design keeps is the designer build string, which differs between a frozen
build and a container of the same release (0.2.5 is `e1a2c24b1172+src:008f1c30d377` frozen and
`unknown+src:76b2edd7e060` in the container) and says nothing about whether design behaviour changed.

- Agree with beqforge a **design version** on `/health` and in each response: a number that changes only
  when the same request could get a different answer (beqforge already keys its optimiser seed this way,
  `1e8d6ee`). A contract addition (1.4), optional field.
- Keep it in each queue entry; let the titles filter and the work list select "designed before
  version N"; show the count in the drift banner D4 already describes.
- Never redesign on its own: accepted and published entries are protected as now, and a redesign stays a
  person's (or a scheduled job's explicit) choice.

**Done when** today's requeue is one filter and one job, with no hand-made version map.

### L4. Designer findings to hand to beqforge

beqforge belongs to its own session: each item becomes a handoff in its `plans/`, with the titles,
messages and repro, not an edit there. In order of titles affected:

1. **F2 part 2, still live on 0.2.5.** Before the Devil Knows You're Dead and Blink Twice still decline
   *no usable plateau* with "real bass content down to 200.0 Hz": an unmeasured tracking floor reported
   as a measurement, and the mix plateau still lost on these two.
2. **Texture retune (beqforge TODO 26).** Its condition, the end of the library review, is met. Cases:
   Carrie (plateau 8.1-66 Hz, judged 5-10 Hz), Changeling, Mirai, Song of the Sea (community BEQ +11 dB),
   Fame, and Despicable Me 3, which now fails on tilt instead.
3. **F6, the absolute 45 dB scene margin**, on dense or LFE-bed mixes: Blade II, Do the Right Thing,
   Jason Bourne; Sunshine a near miss. Community BEQs +16 to +28 dB on all of them.
4. **F5, a dialogue-band reference** hides LFE bass: A Real Pain, Eastern Promises.
5. **O2, an infrasonic plateau** taken as the reference (6 titles, outcome right so far; Carrie is
   close to it), and **O1, DC offset and its steps** (Desperately Seeking Susan: a correction would ship
   thumps). The user asked for O1 to be catered for even though rare.
6. **New: a plateau that starts at the filter floor** licenses nothing below it (So I Married an Axe
   Murderer; community BEQ 4x low shelf 23 Hz +8 dB).
7. **Steep filters**: 9 titles decline by documented design (beqforge TODO 5/6, `--content-edge`).

**Done when** each is a handoff in beqforge's `plans/` with a row in its TODO; their releases then go
through L3.

### L5. Mono titles (needs a decision)

All 30 mono titles decline *channel evidence unavailable* because the library route sends no
decomposition for a one-channel source (F3). The fix, `ebca201`, was left on no branch with its worktree; it is now kept on the local branch
`f3-mono-decomposition` (not pushed). It must not ship alone:
beqforge's content principle would then lift a vintage mono recording's own rolloff to flat, up to the
26 dB cap (F4, A Hard Day's Night). Decide one of:

- (a) keep the principle: send the channel, and mono titles get large corrections; or
- (b) decline one-channel sources on purpose, with their own reason code, in beqforge or the pipeline.

**Done when** the decision is recorded and the 30 titles either design or decline with that reason.

### L6. Finish what the run started

- **E2.** The run exercised scan, extract and design unattended over the whole library; publish and
  commit ran on 2 titles only. Record the run in the E2 record and do the publish/commit part against the
  disposable repositories.
- **Possible duplicates (78)**: decide which copy each pair keeps (several L2 failures are the second copy).
- **Stale work directories**: titles whose default stream moved (F7) left their old `- audio N`
  directories on the share. A cleanup that lists directories no index row points at, then deletes on
  confirmation.

## 3. Order

L2 first (small), L1 alongside as declines
arrive, L4's handoffs once L1's re-review settles which verdicts still hold, then L3 (contract work with
beqforge), then L6.
