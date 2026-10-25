# BEQ filter designer interface — v1.6

**Status:** CONTRACT. 1.6's run record path and `/impact` (§3, §7.1) are agreed
(2026-10-24) and written into the schemas; beqforge returns `run_record` from 0.2.11, and its
`/impact` is not built yet (its TODO 34). 1.5's adjustments (§2, §3) are agreed, written into the
schemas and built on this repo's side (2026-10-21: `pipeline/designer`, the queue entry, the
service), and on beqforge's from 0.2.9 (TODO WP, chunk WP6). Up to 1.4 it
is fully implemented on this repo's side, 1.1's `rejected`
included (reviewed, and overridable by a person, on the title page) — both bindings
(in-process callable and HTTP, `§7.1`) exist; see
[`src/main/python/pipeline/README.md`](../src/main/python/pipeline/README.md#the-designer-contract)
for where. 1.2's arrays by reference (§7.1) are implemented on this repo's side too. They
are sent only to a designer declared `by_reference: true` in a profile
([`archive/designer-by-reference.md`](archive/designer-by-reference.md)). This document remains the thing both sides build to and stays
authoritative for the field-level contract; it is not a plan. Delivered
caller behavior is in [`implemented.md`](implemented.md); open work is in
[`TODO.md`](TODO.md).

**Audience:** whoever implements a filter designer — beqanalyser first, but
nothing here is beqanalyser-specific. **Read this document alone; it does
not assume you have access to the beqdesigner repo.**

This is the formal caller/designer contract negotiated with the beqanalyser
project. It is the document to implement against and is kept in sync when
the contract changes. The delivered caller behavior is summarized in
[`implemented.md`](implemented.md); this document defines the fields precisely
enough to write a designer without reading the rest of the repository.

*Revised three times against external review before either side had written
code against v1.0 — once on shape (added `coverage`, `method`,
`channel_scope`, `residual_band_hz`, tightened `confidence`/`residual_db`'s
definitions, dropped a false provenance guarantee), once on a factual error
(`mv_adjust_db`'s sign was measured and found inverted — see §3), once to
let a single call return several ranked candidates instead of one answer
(`DesignResponse.candidates`, `DesignCandidate.commentary` — see §3). Still
`contract_version = "1.0"`; nothing has been implemented against any earlier
draft.*

*v1.1, additive (§7): `DesignResponse.rejected` and
`DesignCandidate.rejection_reasons` — designs the designer built and judged
unfit to publish, returned for a human to review, never to act on (§3,
"`rejected`"). Nothing in 1.0 changes meaning; a 1.0 caller ignores both
fields and behaves exactly as before.*

*The caller side of `candidates` — running design over many titles
unattended and letting a human pick one per title before anything
publishes — is described in [`implemented.md`](implemented.md).*

*v1.2, HTTP binding only and opt-in (§7.1, "Arrays by reference"): an
array may name a column of a WAV under a root both sides see, with the
SHA-256 of its decoded samples, instead of carrying the samples inline; and
`GET /health` reports whether a designer can take one. The data model (§2)
does not change, and a caller sends an array by reference only to a designer
it was configured to send them to, so a 1.0/1.1 designer never sees one.*

*v1.3, additive (§3, "How well the filters match"): five optional fields on
`DesignCandidate` that describe `residual_db` further — where the worst error
is and which way (`residual_at_hz`, `residual_direction`), the typical error
(`residual_rms_db`) and the share of the requested boost within a stated
tolerance (`residual_within_tolerance`, `residual_tolerance_db`). `residual_db`
and `residual_band_hz` are clarified, not changed: measured on the cascade as
published and as the device plays it, over the band actually measured. A 1.2
caller ignores the new fields; a designer may return them to any request.*

*v1.4, additive (§3, "Design identity"; §7.1): three optional fields on
`DesignResponse` and on `GET /health` -- `design_version`, an integer bumped
only by a change that can alter an answer; `design_settings`, a digest of the
designer's non-default settings that can; and `designer_build`, for people --
so a caller can tell which of its designs a designer upgrade could change.
`/health` may add `design_settings_detail`. A 1.3 caller ignores them.*

*v1.5, additive (§2, "Adjustments"; §3, "Adjustment outcomes"; §7.1): a
request may carry `adjustments` — changes a caller asks for, such as a band to
leave alone, a lowest frequency, a boost cap or a tilted goal — and a 1.5
designer answers each one in `adjustment_outcomes`: honoured, adapted,
declined or not understood, with why. The designer decides what to do with
each; the caller always learns what was done. `GET /health` may list the
`adjustment_kinds` a designer knows. §2's "What is deliberately absent" is
restated: a designer acts on no *unstated* preference. A request without
adjustments is unchanged and must be answered exactly as under 1.4.*

*v1.6, additive (§3, "Design identity"; §7.1, "Impact"): a response may name its run record
(`run_record`, a path relative to the designer's shared root), and a designer may answer
`POST /impact` with, for a batch of those records, how likely a design of the same request now
is to differ, and why. A caller uses it to show a person which out-of-date designs to redesign,
and records the ones they keep as *reviewed at* the new identity (§3). `GET /health` says
whether a designer answers it. Nothing about `/design`'s request changes.*

*A test checklist for whoever implements this contract, extracted from two
independent implementations' test suites (this repo's caller side and
beqforge's designer side), is `design/designer-conformance-tests.md`.*

---

## 1. What you are building

A pure function:

```python
def design(request: DesignRequest) -> DesignResponse:
    ...
```

One call, one title, one answer — but that answer may itself be a short,
ranked list of candidates (§3), not just a single filter. No session, no
callback, no streaming, no iteration protocol — "iterate until acceptable"
(the caller's step 4) is resolved by **you returning a confidence (per
candidate) and/or declining**, not by being called repeatedly with feedback.
If you want to try several internal candidates before answering, return the
ones worth showing a human, ranked — that no longer has to collapse to a
single filter before it reaches the caller. 1.5's `adjustments` (§2) do not
change this: a request with adjustments is a new, self-contained call, not a
reply to an earlier one, and you keep no state between them.

**Must be deterministic given the same input**, or as close as your method
allows — the caller may re-run it and compare. Any internal randomness
(e.g. an optimiser with random restarts) should be seeded so repeat calls on
identical input reproduce the same answer; the caller is not required to
tolerate different answers for the same audio.

**Must not raise for a normal "nothing to correct" outcome.** Declining is a
value (§4), not an exception. Reserve exceptions for actual bugs/crashes on
your side — the caller treats an exception as an implementation failure, not
a signal, and will not publish anything as a result of one either way, so
raising buys you nothing over declining and loses the reason code.

**Binding for v1: an in-process Python callable**, registered by name on the
caller's side. Nothing in the data shapes below assumes that — see §7 — but
that's the only binding v1 needs to support.

---

## 2. `DesignRequest` — what you receive

```python
from dataclasses import dataclass
from typing import Literal
from numpy import ndarray

@dataclass(frozen=True)
class DesignRequest:
    contract_version: str          # "1.0" — echo it back unchanged in the response
    fs: int                        # sample rate (Hz) of every array below
    mono_mix: ndarray              # 1-D float64, the primary signal — see below
    coverage: Literal['complete_programme', 'excerpt']   # see below
    channels: dict[str, ndarray] | None = None   # optional, diagnostic — see below
    bass_management: dict | None = None          # see below
    adjustments: list[DesignAdjustment] | None = None   # 1.5 — see "Adjustments" below
```

**`mono_mix`** is a full-band sum of all channels with the LFE channel
carried at +10 dB relative to the others (matching how LFE is authored
relative to the rest of the mix), sampled at `fs`. This is the signal to
design against by default.

**`coverage`** states what `mono_mix` actually spans, honestly, rather than
leaving it implicit. `'complete_programme'` means the entire title, start to
end, silence included — the case the production pipeline sends, and the one
your noise-floor/envelope statistics can trust as representative. `'excerpt'`
means anything less: a validation clip, a swept-corner test signal, a
loud-scenes-only sample. An excerpt's quiet passages (if any) are not
representative of the whole, and its scene mix is not necessarily
representative either — do not treat an excerpt's envelope statistics as
comparable to a complete-programme run, and say so via `decline_reason` if
your method depends on that assumption and `coverage` says it doesn't hold.

**`channels`**, when present, is a per-channel decomposition of the same
underlying audio at the same `fs` — keys are channel labels (`"L"`, `"R"`,
`"C"`, `"LFE"`, `"Ls"`, `"Rs"`, etc., channel-layout-dependent), values are
1-D float64 arrays **the same length as `mono_mix` and time-aligned with
it** — sample `i` of every array corresponds to the same instant. Provided
so you can tell a mastering-wide rolloff (present on every channel) apart
from an authoring decision on one channel (present on LFE alone, say) — a
diagnostic input, not a replacement for `mono_mix`. See `channel_scope` in
§3 for reporting what you conclude from it. A mono source's decomposition is
its one channel, sent as `{"FC": ...}` (ffmpeg's label for the mono layout)
rather than omitted: a designer cannot otherwise tell a mono title from a
caller that supplied no decomposition, and may decline the latter for want of
channel evidence.

**`fs` will typically be low** — a decimated analysis rate (1000 Hz in the
common case), not the source material's native rate. Whatever content exists
above `fs / 2` has already been discarded before you see it; there is no
higher-resolution version available through this call. If your method needs
more bandwidth than a given `fs` provides, decline (§4) rather than
guessing — the caller controls `fs` and can be asked to raise it.

**`bass_management`, added after implementation feedback:** the caller's
bass-management configuration, if it has one — `None` when there is none
(or it hasn't been decided). When present:

```python
{
    'lpf_fs': float,                         # crossover frequency (Hz)
    'lpf_position': 'Before' | 'After' | 'Off',  # relative to each channel's own filter
    'headroom_type': 'WCS' | '<numeric dB as a string>',  # see below
    'clip_before': bool,
    'clip_after': bool,
}
```

This exists because computing `gain_reduction_db` (§3) means building the
actual sub feed — mains and LFE summed, attenuated, and low-passed the way
the caller's own playback chain would do it — and guessing that
configuration is not a fact you can reconstruct from `mono_mix`/`channels`
alone; it's information the caller already has and you don't. It is not a
*preference*: the "no headroom/max-boost/device preference" line below
still holds, and this is a fact about the playback chain, not a knob you're
being asked to defer to. `headroom_type='WCS'` means the mains and LFE are
attenuated by the *worst-case-scenario coherent-summation* figure, computed
from `channels`' keys as `20·log10(n_mains)` combined with LFE at +10 dB:
`headroom_db = 20·log10(10^(20·log10(n_mains)/20) + 10^(10/20))`, `n_mains`
= every `channels` key except `"LFE"`. A numeric string instead means a
fixed headroom of `abs(float(headroom_type)) + 10` dB. Meaningful only
together with `channels` — without a per-channel decomposition there is no
sub feed to build regardless of what this says.

**No unstated preference (restated in 1.5).** Your answer must not be
shaped by what you guess the caller intends to do with it: there is no
implicit headroom, max-boost or device preference, and you must not infer
one — see `mv_adjust_db`/`gain_reduction_db` in §3 for where headroom
information flows, and it flows *out*. `bass_management` above is a fact
about the playback chain, not a preference. What a caller *does* want
changed it says explicitly, in `adjustments` below, and you answer each one;
that is the only way a preference reaches you, and it is always stated,
always answered and always recorded.

**`adjustments`, added in 1.5:** changes the caller asks for in this design
— typically a person's, after reviewing an earlier answer for the same
title. `None` or an empty list means none, and the request must then be
answered exactly as a 1.4 request would be.

```python
@dataclass(frozen=True)
class DesignAdjustment:
    id: str                    # the caller's, unique within the request; echoed in its outcome
    kind: str                  # a kind from the table below, or one a designer lists on /health
    params: dict               # the kind's parameters, by name; {} when it has none
    text: str | None = None    # what the person said, in their words: for your record, and the content of 'note'
```

| `kind` | `params` | asks you to |
|---|---|---|
| `exclude_band` | `low_hz`, `high_hz` (`0 < low_hz < high_hz`) | treat this frequency range as neither evidence of a rolloff nor something to correct: no boost is targeted inside it, and nothing measured inside it moves the design. Several may be sent. |
| `floor` | `hz` (> 0) | restore nothing below `hz`: the content edge is no lower than this |
| `max_boost` | `db` (> 0) | let the published cascade boost no more than `db` at any frequency |
| `goal_tilt` | `db_per_octave` | tilt the goal you design towards by this much per octave, rising towards the low end when positive (a "bass boost" goal), relative to the goal you would otherwise build |
| `note` | none | read `text`; act on it if you can, and answer it in either case |

How you treat an adjustment is yours to decide, against the evidence, and
you report what you did (§3, "Adjustment outcomes"). Rules:

- **Weigh, don't obey blindly.** §4's line is unchanged: an adjustment can
  make a design narrower or weaker; it never makes you publish what your
  evidence does not support. Decline an adjustment that would (a
  `max_boost` you can honour, a `floor` below where anything was measured
  that you cannot), and say why in its outcome.
- **A declined adjustment is not a declined design.** Answer with your best
  design given the adjustments you did act on. Decline the design (§4) only
  on the evidence, as always — though an adjustment may be why the evidence
  left is too thin (an `exclude_band` over the only rolloff), which you then
  say in `decline_message`.
- **Unknown kinds are not errors.** Answer them `not_understood`. A kind a
  designer defines for itself starts with `x-` (`x-beqforge-strategy`), so
  it never collides with one this contract adds later.
- **Bad parameters are not errors either**: a missing, mistyped or
  out-of-range parameter is answered `declined` with the reason. Only a body
  that is not this shape at all (an `adjustments` that is not a list, an
  entry without `id` or `kind`, two entries with one `id`) may be refused as
  an implementation failure.
- **Order carries no meaning**, and the same adjustments must give the same
  answer whatever their order (§1, determinism). Two of a kind where only one
  makes sense (two `floor`s) are yours to reconcile: honour or adapt one and
  decline the other, saying so.
- **Adjustments are part of the request**, like the audio: the same audio
  with the same adjustments must reproduce the same answer, and they never
  change `design_settings` (§3, which describes your server-wide settings).

**No guarantee of provenance.** `mono_mix`/`channels` may be real
theatrical/consumer audio, a known filter applied to known-clean material for
accuracy testing, or fully synthetic full-band material for false-positive
testing — production and validation traffic arrive through the same call, on
purpose, so the code path under test is the code path that ships. Do not
assume anything about how the audio was produced beyond what `coverage`
states.

---

## 3. `DesignResponse` — what you must return

Exactly one of two shapes. Populating fields from both, or neither, is a
contract violation the caller will reject. `rejected` (1.1) is not a third
shape: it may accompany either one, and never stands in for either.

```python
from dataclasses import dataclass
from typing import Literal

BiquadType = Literal['peaking_eq', 'low_shelf', 'high_shelf']
DesignMethod = Literal['exact', 'fitted', 'non_parametric']
ChannelScope = Literal['all_channels', 'lfe_only', 'mixed']

@dataclass(frozen=True)
class BiquadSpec:
    type: BiquadType
    freq_hz: float     # > 0
    gain_db: float      # signed; 0 is a no-op section
    q: float            # > 0

@dataclass(frozen=True)
class DesignCandidate:
    filters: list[BiquadSpec]
    confidence: float      # 0.0-1.0, ordinal in v1 — see below
    mv_adjust_db: float    # the cascade's implied gain; NOT a clipping-cost estimate — see below
    method: DesignMethod   # how `filters` was derived; see below

    # the actual clipping-cost figure; see below. None if not computed
    gain_reduction_db: float | None = None

    # fit-quality — see below for what "no exact target" now means for non_parametric
    residual_db: float | None = None            # max abs error, dB, over residual_band_hz
    residual_band_hz: tuple[float, float] | None = None

    # structured, human-facing notes about *this* candidate — see below
    commentary: dict[str, str] | None = None

    # optional either way: not published, surfaced to a human reviewer only
    fc_hz: float | None = None
    slope: float | None = None
    fc_uncertainty_hz: float | None = None
    slope_uncertainty: float | None = None
    channel_scope: ChannelScope | None = None    # see below; requires `channels` in the request

    # 1.1: why this design was judged unfit to publish. Non-empty on every
    # entry of `rejected`; None on every entry of `candidates` — see below
    rejection_reasons: list[str] | None = None

    # 1.3: how well `filters` match their target — see "How well the filters
    # match" below. Each None whenever residual_db is
    residual_at_hz: float | None = None          # where |error| == residual_db
    residual_direction: Literal['over', 'under'] | None = None
    residual_rms_db: float | None = None         # boost-weighted RMS error
    residual_within_tolerance: float | None = None   # share of the boost, [0, 1]
    residual_tolerance_db: float | None = None   # the tolerance that share uses

@dataclass(frozen=True)
class DesignResponse:
    contract_version: str            # echo the request's value

    # success: a non-empty list, best (most preferred) first — see below.
    # decline fields left None.
    candidates: list[DesignCandidate] | None = None

    # decline: these two populated, candidates left None
    decline_reason: str | None = None    # short stable code — see §4
    decline_message: str | None = None   # optional human-readable detail

    # 1.1, with either shape: designs built and judged unfit to publish, for
    # a human to review only — never applied or published. See below.
    rejected: list[DesignCandidate] | None = None

    # 1.4, with either shape: which answer this designer gives. See "Design
    # identity" below.
    design_version: int | None = None    # >= 0; bumped by a change that can alter an answer
    design_settings: str | None = None   # digest of the non-default settings that can
    designer_build: str | None = None    # the build, for people; never a key

    # 1.5, with either shape: one per request adjustment. See "Adjustment
    # outcomes" below.
    adjustment_outcomes: list[AdjustmentOutcome] | None = None

@dataclass(frozen=True)
class AdjustmentOutcome:
    id: str                       # the adjustment's id
    outcome: Literal['honoured', 'adapted', 'declined', 'not_understood']
    applied: dict | None = None   # the params actually used: required for 'adapted', optional for 'honoured', None otherwise
    reason: str | None = None     # for a person; required for every outcome but 'honoured'
```

**`candidates`** replaces what earlier drafts of this document expressed as
a single flat set of `filters`/`confidence`/`mv_adjust_db`/`method` fields on
`DesignResponse` itself. You may still return exactly one — a list of one
is the common case and nothing about validation treats it specially — but
when your method genuinely produces more than one plausible correction (a
strong "exact" alignment and a weaker "fitted" fallback, say, or two
`fc`/alignment hypotheses that fit about equally well), return them all
rather than picking silently. **Order matters: best (most preferred) first,
by non-increasing `confidence`** — the caller rejects a list that isn't
sorted that way, since "ranked" only means something if the order is
load-bearing. `candidates[0]` is the only one the caller ever acts on
automatically (simulated, written to XML, published); every other entry is
carried through only as far as a human reviewing the report, for comparison
— never applied, never published, and the caller does not pick among them
for you. If you don't have a genuine second opinion worth showing, don't
manufacture one just to populate the list — a single confident candidate is
still a complete, valid answer.

**`rejected`, added in 1.1.** Designs you built and then judged unfit to
publish — a fit that failed your own acceptance checks, say — returned so a
human reviewing the title can see what was tried and why it lost, and can
load one to look at or listen to. It is the other half of "return them all
rather than picking silently": `candidates` holds what you would publish,
`rejected` holds what you would not. Rules:

- **Review only.** The caller never applies, simulates for publication,
  writes to XML, or publishes a `rejected` entry, and never promotes one
  into `candidates` on its own. If a person chooses to publish one anyway,
  that is a human override of the designer's verdict and the caller records
  it as one — it is the caller's policy whether to allow that at all.
- **With either shape.** Alongside a success, it lists the designs that lost
  to `candidates`. Alongside a decline, it lists the designs whose failure
  *is* the decline (e.g. `no_publishable_candidate`); the decline stands,
  and `rejected` is its evidence. A decline that never built a design (an
  excerpt, no usable reference) has nothing to put here. `None`/absent when
  there is nothing to show; when present, a non-empty list — an empty list
  is rejected, as for `candidates`.
- **Every entry is a full `DesignCandidate`** and passes the same §5 checks
  (types, budget, finite values) and the same field rules above — a person
  may load its filters, so they must be loadable. `confidence`,
  `mv_adjust_db` and `method` are required as usual; `gain_reduction_db` is
  as useful here as anywhere.
- **`rejection_reasons`** is required and non-empty on every `rejected`
  entry: one plain-language string per failed check, e.g. `"introduces a
  cliff of 53 dB/oct at 17 Hz"`. Human-facing, never machine-parsed, like
  `decline_message`. It must be `None` on every entry of `candidates` — a
  candidate with reasons against it is a rejected design in the wrong list,
  and the caller rejects the response.
- **No ordering rule.** `confidence` measures evidence, not whether a design
  passed, so a rejected design can legitimately score higher than the
  accepted one; `rejected` is exempt from the non-increasing-`confidence`
  rule. Order it however helps a reviewer — nearest to acceptable first is
  a good default. Its entries do not count against `candidates`' budget and
  vice versa.

Why a separate list rather than a flag on entries in `candidates`: §7 lets a
caller ignore response fields it does not recognise, so a 1.0 caller would
drop a flag and could publish a rejected design as `candidates[0]`. A
separate list is invisible to such a caller, which then behaves exactly as it
did under 1.0 — safe by construction.

**`filters`** — see §5 for exactly what's allowed in a `BiquadSpec`. Order
in the list is not meaningful (biquad sections in a cascade commute); return
them in whatever order is natural to you.

**`confidence`, revised: an ordinal in v1, not a calibrated probability.**
Earlier drafts defined this as precisely P(a real rolloff was applied to
this programme). That's still the *intent* — whether there is something
here to correct at all, scoped to *this candidate*, fit quality kept out of
it (that's `residual_db`) — but there is no labelled corpus to calibrate a
probability against: `api-headless-pipeline.md §15.4` notes the existing
catalogue has zero negatives, which is exactly why a false positive is
expensive and undetectable, and exactly why nothing exists to fit a
probability to. Demanding calibration you cannot obtain would make the
field unimplementable honestly. So for v1, `confidence` only has to be
**monotone in your own evidence** — comparable *within* one `DesignResponse`
(which is what the ranking rule needs), not across titles or across
different designer implementations. It is also the field that rule sorts
on — since it's the only cross-candidate scalar this contract defines,
there is no separate "rank" field: position in `candidates` *is* the
ranking, and it must agree with `confidence` order. Calibrating it as a
true probability is deferred until a labelled corpus exists to calibrate
against.

**`commentary`** is free-form but structurally simple: a flat
`dict[str, str]`, both keys and values plain strings — no nesting, no
non-string values. It exists so you can explain *this specific candidate* to
a human reading the report (why it was ranked where it is, what's uncertain
about it, what would change your mind) in a form a report/GUI can render
directly (e.g. as a small key/value table) without parsing free text. It is
never machine-parsed by the caller — treat it exactly like `decline_message`
in that respect, just attached per-candidate and structured enough to
tabulate. Optional; omit it (leave `None`) rather than populating it with
nothing useful.

**`method`** says which of three genuinely different claims `filters`
represents, because `confidence` and `residual_db` cannot distinguish them
on their own:

- `'exact'` — the closed-form shelf decomposition of a matched-alignment
  exact inverse (`api-headless-pipeline.md §15.3`). Error is bilinear
  pre-warp only, ~1e-3 dB in the cases checked so far.
- `'fitted'` — a numerical fit (e.g. minimax) to an identified parametric
  target — a real rolloff was found and modelled, but alignment didn't
  allow the closed form, so this is an approximation of a known target.
  ~0.1 dB achievable at 3-4 sections in the cases checked so far.
  `residual_db`/`residual_band_hz` describe *this* fit's error.
  `fc_hz`/`slope` describe the target that was fitted to.
  Populate `residual_db`/`residual_band_hz` here.
- `'non_parametric'` — a correction curve produced without identifying a
  specific high-pass model to invert (no clean `fc`/alignment found, but
  enough signal to justify a correction anyway). This is a materially
  weaker claim than the other two — there is no *identified model* for
  `residual_db` to be measured against the way `'exact'`/`'fitted'` have
  one. **Revised:** that doesn't mean there's no target at all. If your
  method builds an explicit target curve (e.g. the measured deficit,
  clipped to what the evidence supports) and fits `filters` against *that*,
  populate `residual_db`/`residual_band_hz` the same way `'fitted'` does —
  just document, via `commentary`, that it's error against a constructed
  target, not a claim that the target itself is correct. A small residual
  here says "the fit matched what I built", not "the correction is right" —
  that distinction is exactly why this is worth reporting: it separates "the
  fit is bad" from "the evidence only licensed this much". Leave both
  `None` only when there genuinely is no target curve to measure against.

Do not infer `method` from `fc_hz is None` — leave `fc_hz`/`slope` as
optional diagnostics either way and populate `method` explicitly.

**`mv_adjust_db`, corrected: this is not the clipping-cost figure it
sounds like.** It is the cascade's implied gain — in practice, its peak
magnitude — reported so the field lines up with the catalogue's `mv_adjust`
(same quantity, §7 backward compatibility). Earlier drafts described it as
"what would master volume need to move by to keep this filter safe", which
reads as a whole-system attenuation cost. It is not one, for two reasons: a
BEQ runs post bass-management, on the sub channel only, so it costs no
master volume at all; and even read as "cost to the sub channel", cascade
peak magnitude is close to *inverted* against what actually constrains
publication. A +45 dB boost sitting where the programme has no content
costs nothing; an +18 dB one landing on real content can cost several dB.
Gating on this number rejects the cheap corrections and accepts the
expensive ones. **Use `gain_reduction_db` below for anything that needs the
actual cost** — this field is kept only for catalogue compatibility.

**`gain_reduction_db`, new.** The actual publication-blocking quantity: the
gain reduction the *filtered sub feed* needs to avoid clipping —
`min(20·log10(1/peak), 0)` of that feed, so always `<= 0`, `0` meaning no
reduction needed. "The filtered sub feed" means mains + LFE summed,
attenuated and low-passed per `bass_management` (§2) with `filters`
applied — buildable only when `bass_management` and `channels` were both
supplied; leave this `None` otherwise rather than approximating it from
`mono_mix` alone, which is not the same signal. This is the same
computation the caller already does internally (`pipeline/stats.py`'s
`signal_stats().headroom`, `min`'d with 0) — reporting it here means the
caller doesn't have to re-derive it, and a designer without `channels`
isn't forced to guess at a number it can't actually compute.

**Sign convention, pinned — corrected from an earlier draft, and pinned to
measured catalogue data rather than an assumption:** `mv_adjust_db` is
**positive when master volume should be turned down.** A filter that needs
15.9 dB of boost headroom is reported as `mv_adjust_db=+15.9`. This is the
opposite of what an earlier revision of this document said; that version
inferred the sign from `beq_gain` (`model/postbuilder.py:117`, sourced from
a signal's dB `offset`), which turned out to be a **different field** from
the catalogue's `mv_adjust` (sourced from `vals['mv']`,
`model/catalogue.py:465`) — related in purpose, not confirmed to share a
sign convention. Measured directly from the published catalogue instead:
3,505 entries carry a value, 81% positive, median +3.0 dB, range −10.5 to
+19.5 — consistent with "most authored corrections are boosts, and a boost
demands attenuation somewhere," and inconsistent with a
mostly-negative convention. A negative value is valid and means the
opposite — headroom to spare, master volume could go *up* — so treat this
as signed, not a pure magnitude. This is measured, not confirmed against
beqcatalogue's own documentation (out of both repos' control — the same gap
`api-headless-pipeline.md` D3 already tracks), so treat it as the best
current evidence rather than beyond dispute. Get the sign wrong and a
downstream player turns volume the wrong way.

**`residual_db`** is the **max absolute error, in dB**, of `filters`
against whatever target you fitted it to — an identified model's exact
target for `'exact'`/`'fitted'`, or a constructed target curve for
`'non_parametric'` (see above) — not RMS, a minimax/peak error, matching how
the worked examples below are reported. **`residual_band_hz`** is the
`(low_hz, high_hz)` band that error was measured over — a residual number
with no band is not comparable across titles with different knee
frequencies. Both `None` only when there's genuinely no target to measure
against. Measure `residual_db` on the cascade **as published and as the
device plays it** — the canonical rounding of the parameters you return, and
the coefficients quantised at the device rate — not on an optimiser's raw
parameters or its fit objective; and report in `residual_band_hz` the band you
actually measured, which may be wider than a nominal one.

**How well the filters match (1.3).** Five optional fields describe that
error further, so a reviewer can tell a worst case at one frequency from a
poor fit throughout:

- **`residual_at_hz`** — the frequency at which `|error|` equals
  `residual_db`, inside `residual_band_hz`; and **`residual_direction`** —
  `'over'` if the filters give more boost than the target there, `'under'` if
  less. Both present or both absent.
- **`residual_rms_db`** — the RMS error over `residual_band_hz`, weighted by
  the boost the target asks for at each frequency (`max(target_db, 0)`): bins
  where no boost is asked carry no weight. Never negative, never more than
  `residual_db`; `None` when the target asks for no boost in the band.
- **`residual_within_tolerance`** — the share, in `[0, 1]`, of that same
  requested boost where `|error| <= residual_tolerance_db`: "94% of the boost
  lands within ±0.5 dB of the target". Not a probability, and not a verdict.
  **`residual_tolerance_db`** — the tolerance used (finite, `> 0`); the two are
  present together or not at all.

Rules the caller enforces: with `residual_db` `None`, all five are `None`;
`residual_at_hz` lies within `residual_band_hz`; `0 <= residual_rms_db <=
residual_db` (plus a 1e-9 rounding allowance); `residual_rms_db` and
`residual_within_tolerance` are present together or not at all, since they
share one weighting (a target asking for no boost gives neither); every value
is finite.

**What they do not say.** These fields measure only whether the filters match
the target the designer built, never whether that target is right. A designer
may build a target smaller than the whole deficit (beqforge's is
evidence-priced: the measured low-end deficit, capped at what the title's
loud-versus-quiet contrast supports); how much of the deficit a target
restores is a separate question with no field in this contract — say it in
`commentary` if you have it. A small residual means "the filters do what was
asked", not "the correction is correct".

**`fc_hz` / `slope` / uncertainties** — diagnostic. Optional even on
success. Populate them if your method produces them; leave them `None` if
it doesn't. Never published to the output artifact — carried only as far as
a human-facing report for audit, per the caller's own design decision (see
`api-headless-pipeline.md` D7).

**`channel_scope`** — what the per-channel diagnostic (`channels` in the
request, if supplied) told you: `'all_channels'` if the rolloff shows up
consistently across channels (a mastering-wide effect — the case the
inversion filter is meant for), `'lfe_only'` if it's confined to the LFE
channel (an authoring decision, arguably not something to correct the same
way), `'mixed'` if the per-channel picture is inconsistent enough not to
call cleanly either way. `None` when `channels` wasn't supplied or wasn't
used. This is currently the only field carrying anything back out about
what `channels` showed — worth populating whenever you have an answer,
since it's otherwise invisible to the caller.

**Design identity (1.4).** Three optional fields, on a design and a decline
alike, returned whatever `contract_version` the request names:

- **`design_version`**, an integer of at least 0 that never decreases. Bump it
  with any change that can alter `candidates`, `rejected`, a decline or a 1.3
  fit figure for an unchanged request on the same machine. A version names one
  algorithm, not one answer: floating point differs between processors, and
  for an optimising strategy that can lead to a different optimum, so two
  designers on different hardware at the same version and settings may answer
  a title differently. A caller compares versions, never figures. A change to wording only
  (`commentary`, `decline_message`), speed, packaging or transport is not a
  bump; nor is adding a field that leaves the answer as it was (1.3's fit
  figures). 0 means "before versioning".
- **`design_settings`**, an opaque string, equal between two designers exactly
  when their server-wide settings that can change an answer are equal. Leave
  out every setting at its default (so all defaults is the empty set's digest,
  and a new flag whose default keeps today's answers changes nobody's digest)
  and every setting that cannot change an answer (workers, cores, caches, log
  and record locations). Absent means the designer has no such settings.
- **`designer_build`**, free text naming the build, for display. A caller keys
  nothing on it.

The pair (`design_version`, `design_settings`) is the design identity. A
caller keeps it with each design and selects as *out of date* those whose
identity differs from what the designers it can use now report on `/health`
(§7.1); it never redesigns on that alone.

**The run record (1.6).** Optional, on a design and a decline alike:

- **`run_record`**, the path of the record the designer kept of this run, relative to its
  shared root (§7.1, "Arrays by reference"): POSIX-style, no backslash, no `..` segment, never
  absolute. Absent when the designer has no shared root or keeps the record outside it. A
  caller stores it with the design and sends it back unchanged to `/impact` (§7.1); it never
  builds one, and never reads one out of `commentary`, which may name the same file as a path
  on the machine that wrote it.

**Reviewed designs (1.6).** A person may decide that a design out of date against an identity
should be kept rather than redesigned, usually after reading `/impact`'s answer for it (§7.1).
The caller then records that identity beside the design as *reviewed at*; a design may be
reviewed at several identities over time. A design counts as out of date only against an
identity that is neither the one it was designed at nor one it was reviewed at. Reviewing
changes nothing about the design itself: its filters, its `rejected` list and the identity it
was designed at stay as they were, and are what get published.

**Adjustment outcomes (1.5).** A 1.5 designer returns exactly one
`AdjustmentOutcome` per entry of the request's `adjustments`, on a design
and a decline alike, and `None` (or omits the field) when the request had
none:

- **`honoured`** — done as asked. `applied` may repeat the params.
- **`adapted`** — done in part or differently: a `max_boost` of 10 dB met at
  12 dB because 10 left the rolloff uncorrected, a band narrowed to where
  the evidence was. `applied` holds what was actually used; `reason` says why.
- **`declined`** — not acted on, with `reason`: it would have you publish
  what the evidence does not support, it contradicts another adjustment, or
  its params are missing or out of range.
- **`not_understood`** — a kind you do not know, or a `note` you could not act
  on; `reason` may say what you took it to mean.

The caller rejects as a contract violation an outcome naming an `id` the
request did not send, or one `id` twice. An adjustment with no outcome —
every one, from a designer before 1.5, which ignores the request field — is
shown to a person as *not acknowledged*: the caller cannot tell whether it
was acted on. `reason` is free text for a person, not machine-parsed.

---

## 4. Declining

Declining is success, not failure — "we correctly concluded nothing should
be applied" is exactly as valid an answer as returning a filter, and the
caller treats it as a first-class, non-error outcome. Populate:

- **`decline_reason`** — a short, stable, machine-usable string:
  `snake_case`, no punctuation beyond underscores, safe to aggregate across
  many runs (e.g. counted in a dashboard). Not standardised or enforced by
  the caller in v1 — pick codes you'll want to distinguish later.
  Suggested starting set, extend freely: `no_rolloff_detected`,
  `insufficient_coherent_bandwidth`, `scene_disagreement_too_high`,
  `unstable_fc_estimate`, `confidence_below_internal_threshold`,
  `analysis_rate_too_low` (§2's "decline rather than guessing" when `fs` is
  too low to trust — distinct from `insufficient_coherent_bandwidth`, which
  is about the content, not the request), `no_publishable_candidate` (1.1:
  designs were built and every one failed your own checks — list them in
  `rejected`, §3).
- **`decline_message`** — optional, free text, for a human reading the
  report. Not machine-parsed.

Leave `candidates` as `None` on decline — not an empty list, `None`; an
empty list is rejected the same as populating it, since "zero candidates" is
what a decline already means. A decline may still carry `rejected` (1.1,
§3): the designs that were tried and failed, as evidence for the decline —
it does not make the response any less a decline. The caller designs nothing downstream of a
decline and never publishes one on its own. BEQDesigner offers it to a
person as a single flat candidate (no filters); if they accept it, the title
is published as a record with an empty filter list and the note "Does not
require BEQ" — the catalogue's convention for a title that needs no
correction — so no report claims a correction was made.

**What's yours to decline on, versus what to report and let the caller
decide.** §2 keeps the input surface small on purpose — no unstated
headroom, max-boost or device preference crosses in, and a stated one comes
only as an adjustment you weigh — but that leaves a gap: what do you do when
a real decision point turns on exactly one of those things you weren't
given? Decline only on grounds internal to the evidence itself — no
rolloff detected, insufficient coherent bandwidth, an unstable estimate,
your own confidence floor (`confidence_below_internal_threshold` already
implies this). Whether a correction is *worth* publishing given how much of
the deficit it recovers, or whether a given amount of gain reduction is
acceptable, are calls that depend on playback configuration or publication
policy — the caller's decisions, not yours. Report the measurement (through
`confidence`, `residual_db`, `commentary`) and return the candidate; let the
caller decide what to do with it.

The same line holds for `rejected` (1.1, §3). A design belongs there only
for a reason that is internal to the evidence: it does not match its own
target, it introduces a cliff, or it corrects too little of the extent it
found. It does not belong there because it needs too much gain reduction or
boosts more than some device would like. A design whose only objection is a
matter of policy is a candidate: return it in `candidates` with the
measurement, and let the caller decide.

---

## 5. What makes a `BiquadSpec` valid

The caller enforces these; know them so you don't produce something that
gets rejected after you've done the work.

**Only three types.** `'peaking_eq'`, `'low_shelf'`, `'high_shelf'` — nothing
else survives the caller's publishing format. If your method's natural
output is a general biquad (a target section with independent zero/pole
locations, not a shelf), you likely already have what you need to convert
it: the exact inverse of a matched-alignment rolloff *is* a low-shelf
cascade in closed form —

```
zero at fz, pole at fp, shared Q  →  low_shelf(freq_hz = √(fz·fp),
                                                gain_db = 40·log10(fz/fp),
                                                q = <the shared Q>)
```

— an identity verified numerically to ~1e-3 dB against LR2/LR4/LR8 targets
during this contract's negotiation. Where alignment isn't
shared, a numerical minimax fit against your exact target closes the gap at
3-4 biquad sections to <0.1 dB error in the cases already checked. Either
route stays inside the budget below.

**Gain convention: RBJ, not a plain amplitude ratio.** `gain_db` must be
defined the way the caller's `PeakingEQ`/`LowShelf`/`HighShelf`
implementations define it: `A = 10 ** (gain_db / 40.0)` is the filter's
amplitude parameter (**not** `10 ** (gain_db / 20.0)`; the `/40` is
deliberate RBJ convention for shelving/peaking filters, not a typo). This
matches the `gain = 40·log10(fz/fp)` derivation used to verify the identity
above — if you used that derivation, you already match.

**No sample rate on a `BiquadSpec`, by design.** `freq_hz`/`gain_db`/`q`
fully determine an RBJ shelf/peaking filter independent of the rate it's
eventually realised at — the caller instantiates it against whatever `fs`
the *publish target* needs (commonly 48 kHz for the hardware this ships to),
which will not be the `fs` you analysed at (commonly 1 kHz, decimated). You
do not need to know or care about the publish-target rate.

**No `count` / stacking field.** If your method wants the same shelf applied
twice (a stacked shelf, in this format's terms), put the same `BiquadSpec`
in the list twice — don't try to express repetition yourself. Collapsing
repeated identical entries into a stacked-shelf representation is the
caller's concern, not yours.

**Budget: 10 biquad sections total, per candidate.** Every `BiquadSpec` in a
candidate's `filters` counts as one section (a `low_shelf`/`high_shelf`
counts once per list entry, same as `peaking_eq` — there is no multiplier).
The budget applies independently to each entry in `candidates` — a second
candidate does not shrink the first one's allowance. The cases worked
through so far land at 2-4 sections, nowhere near the limit; if your method
is producing close to 10, that's worth a second look on your side before it
ever reaches the caller.

**`freq_hz > 0`, `q > 0`.** `gain_db` may be any sign, including `0.0` (a
no-op section — allowed, though there's no reason to emit one).

---

## 6. Worked examples

**Success, single candidate** — matching a worked, numerically-verified case
— identifying an `LR4` rolloff at 25 Hz and inverting it against an `LR4`
protective filter at 10 Hz:

```python
DesignResponse(
    contract_version="1.0",
    candidates=[
        DesignCandidate(
            filters=[
                BiquadSpec(type='low_shelf', freq_hz=15.810, gain_db=15.918, q=0.7071),
                BiquadSpec(type='low_shelf', freq_hz=15.810, gain_db=15.918, q=0.7071),
            ],
            confidence=0.94,
            mv_adjust_db=15.918,           # cascade's implied gain -- catalogue compat only, not a cost figure
            gain_reduction_db=-1.2,         # illustrative -- the actual sub-feed clipping cost, usually << mv_adjust_db
            method='exact',                 # matched-alignment closed-form decomposition
            residual_db=0.0008, residual_band_hz=(5.0, 200.0),
            commentary={'alignment': 'LR4', 'knee_hz': '25.0'},
            fc_hz=25.0, slope=24.0,        # e.g. dB/octave, if that's your slope unit — say so
            fc_uncertainty_hz=0.6, slope_uncertainty=1.1,
            channel_scope='all_channels',   # rolloff was consistent across the `channels` diagnostic
        ),
    ],
)
```

**Success, two ranked candidates** — same title, but the alignment was
genuinely ambiguous between two plausible knees; both are worth showing a
human, best first:

```python
DesignResponse(
    contract_version="1.0",
    candidates=[
        DesignCandidate(
            filters=[BiquadSpec(type='low_shelf', freq_hz=15.810, gain_db=15.918, q=0.7071)] * 2,
            confidence=0.72,
            mv_adjust_db=15.918,
            method='exact',
            residual_db=0.0008, residual_band_hz=(5.0, 200.0),
            commentary={'alignment': 'LR4', 'knee_hz': '25.0',
                       'note': 'best fit, but a second knee near 32 Hz fits almost as well'},
        ),
        DesignCandidate(
            filters=[BiquadSpec(type='low_shelf', freq_hz=20.199, gain_db=12.304, q=0.7071)] * 2,
            confidence=0.61,
            mv_adjust_db=12.304,
            method='exact',
            residual_db=0.0011, residual_band_hz=(5.0, 200.0),
            commentary={'alignment': 'LR4', 'knee_hz': '32.0',
                       'note': 'alternative alignment; lower confidence, smaller correction'},
        ),
    ],
)
```

The caller applies/publishes `candidates[0]` only; `candidates[1]` is
carried through for a human reviewing the report to compare against.

**Decline**, insufficient bandwidth to trust the estimate:

```python
DesignResponse(
    contract_version="1.0",
    decline_reason="insufficient_coherent_bandwidth",
    decline_message="Coherent band above candidate knee spans <1 octave; "
                     "below the margin this method requires.",
)
```

**Decline with rejected designs (1.1)** — designs were built, and every one
failed the designer's own checks:

```python
DesignResponse(
    contract_version="1.1",
    decline_reason="no_publishable_candidate",
    decline_message="flatten: introduces a cliff; counterfactual: tilt outside tolerance",
    rejected=[
        DesignCandidate(
            filters=[BiquadSpec(type='low_shelf', freq_hz=26.87, gain_db=16.14, q=5.797)],
            confidence=0.95,
            mv_adjust_db=16.14,
            method='non_parametric',
            residual_db=13.86, residual_band_hz=(5.0, 200.0),
            commentary={'strategy': 'flatten'},
            rejection_reasons=[
                "tilts -14.4 dB/oct where -0.1 was intended",
                "corrected only down to 24.7 Hz; content continues to 16.7 Hz",
            ],
        ),
    ],
)
```

A success response carries `rejected` the same way, beside `candidates`. The
caller publishes nothing from `rejected` in either case.

---

## 7. Versioning and future bindings

**`contract_version`** is `"1.0"` for everything in this document except
what is marked 1.1 (`rejected`, `rejection_reasons`), which a 1.1 caller
sends as `"1.1"`. Echo the request's version back in the response unchanged
— it lets the caller notice a mismatch rather than silently misinterpret a
field. A designer may return the 1.1 fields to a `"1.0"` request: a 1.0
caller ignores them (below), which is exactly why they are a separate list.
1.2 changes only the HTTP wire (§7.1, "Arrays by reference"): a caller sends
`"1.2"` exactly when the body holds an array by reference, and `"1.1"`
otherwise, so a request with every array inline is unchanged. 1.3 adds
optional response fields only (§3, "How well the filters match"); a designer
may return them whatever version the request names, as it may the 1.1 fields. 1.4 adds the design identity
(§3, "Design identity") to responses and `/health`, on the same terms. 1.5
adds `adjustments` to the request: a caller sends `"1.5"` exactly when the
request carries at least one adjustment (and then whatever else the body
holds, arrays by reference included), and the version it sent before
otherwise, so a request without adjustments is unchanged. A designer may
return `adjustment_outcomes` to a `"1.5"` request only. 1.6 adds the optional response field
`run_record` (§3, "The run record"), returned whatever version the request names, and
`POST /impact` (§7.1, "Impact"), which a caller sends only to a designer that reports it on
`/health`; `/design`'s request is unchanged. Changes within `1.x`
will be additive-only (new optional fields, defaulting to `None`/absent);
you can ignore fields you don't recognise on the request side, and the
caller does the same for the response. A breaking change bumps to `2.0` and
this document gets a new version marker at the top.

**This document specifies data shapes, not a transport.** §1's in-process
Python callable is one binding; §7.1 below specifies a second, over HTTP.
The field names, types and validation rules above don't change between
them — only how they cross a process boundary does. Any future binding
(subprocess, message queue, whatever) follows the same rule: same fields,
different wire.

### 7.1 HTTP binding

Chosen over the process boundary staying Python-only because it's the more
deployment-flexible option — the designer can be local, on another
machine, written in any language, or shared across a team, without the
caller spawning or managing it. This repo's caller-side implementation is
`pipeline.designer.http_binding.http_designer(url)`.

**One POST per `design()` call**, request body:

```json
{
  "contract_version": "1.0",
  "fs": 1000,
  "coverage": "complete_programme",
  "mono_mix": {"dtype": "float64", "shape": [N], "data_base64": "..."},
  "channels": {"L": {"dtype": "float64", "shape": [N], "data_base64": "..."}, "...": "..."},
  "bass_management": {"lpf_fs": 80.0, "lpf_position": "Before", "headroom_type": "WCS",
                      "clip_before": false, "clip_after": false}
}
```

`channels`/`bass_management` are JSON `null` exactly when the corresponding
`DesignRequest` field is `None`. `adjustments` (1.5) is omitted when there are none,
and otherwise a JSON array of `{"id", "kind", "params", "text"}` objects
(`text` omitted or `null` when none); `adjustment_outcomes` comes back the
same way, as `{"id", "outcome", "applied", "reason"}`. **Arrays are base64-encoded raw
little-endian float64 bytes, not JSON number arrays** — at the common 1 kHz
decimated rate, a 2-hour `mono_mix` is ~58 MB as float64 (§8's own figure);
JSON's text encoding of that many floats bloats it 3-5x for no benefit on a
same-machine/LAN call, where decode simplicity matters more than the
payload being human-readable. `dtype` travels alongside rather than being
assumed, so a future widening (e.g. float32) isn't a silent breaking change;
`"float64"` is the only value v1 produces or accepts.

**Response body is `DesignResponse` as JSON, field-for-field** — success:
`{"contract_version": "1.0", "candidates": [...]}`, each candidate's
`filters` as `[{"type": "low_shelf", "freq_hz": ..., "gain_db": ..., "q": ...}, ...]`;
decline: `{"contract_version": "1.0", "decline_reason": "...", "decline_message": "..."}`.
Either may add `"rejected": [...]` (1.1), each entry a candidate object with
`"rejection_reasons": ["...", ...]`.
No binary encoding on this side at all — a `BiquadSpec` carries no
sample-rate-bound data by design (§5), so nothing in a response needs it.

**Arrays by reference (1.2).** Each array (`mono_mix`, and each value of
`channels`) may instead name one column of a WAV file, when caller and
designer see the same files:

```json
"mono_mix": {"dtype": "float64", "shape": [N],
             "file": {"path": "t_1234/mono.wav", "channel": 0},
             "sha256": "<hex SHA-256 of the decoded column's little-endian float64 bytes>"},
"channels": {"LFE": {"dtype": "float64", "shape": [N],
                     "file": {"path": "t_1234/multichannel.wav", "channel": 3},
                     "sha256": "..."}, "...": "..."}
```

It is another wire form for the same field (§7), not a new field: the
designer decodes it into exactly the array the inline form would have
carried, and nothing after decoding may tell the two apart. The rules:

- **One of `data_base64` or `file` per array, never both.** A request may mix
  them, and holds `"contract_version": "1.2"` when it has at least one `file`.
- **`path` is relative to a shared root** that each side configures for
  itself (the caller: its work directory; the designer: wherever it sees the
  same files). It is POSIX-style, never absolute, has no backslash and no
  `..` segment. The designer refuses a path whose real path, after
  resolving symlinks, lies outside its root's real path — a symlink under
  the root that points out of it included.
- **The file is a WAV** (RIFF WAVE, including `WAVE_FORMAT_EXTENSIBLE`,
  which ffmpeg writes for more than two channels): PCM of 16, 24 or 32
  bits, or IEEE float. The
  decoded array is defined by its arithmetic, not by a library: an integer
  sample `s` of `b` bits becomes `s / 2**(b-1)` as float64; a float sample
  is widened to float64. `channel` is the zero-based column and must be
  below the file's channel count. The file's sample rate must equal `fs` —
  a file at any other rate is refused, **never resampled** — and its frame
  count must equal `shape[0]`.
- **`sha256` is required with `file`**: the lower-case hex SHA-256 of the
  decoded column as a C-contiguous little-endian float64 1-D array of
  `shape[0]` elements, i.e. exactly the bytes `data_base64` would have
  carried. The designer checks it **before using the array**. It is a
  check, not a cache key the contract promises anything about: a designer
  keys its own caches however it likes.
- **Channel labels stay as `channels`' keys**, set by the caller. The
  designer never works them out from the file.
- **Errors.** `400` for a body that does not parse or fails the schema, as
  for any request. `422` for a well-formed `file` the designer cannot
  honour: a bad or escaping path, a missing or unreadable file, a file that
  is not a WAV of the kinds above, a channel out of range, a rate or frame
  count that differs, a digest that differs, or **no shared root configured
  on this designer**. The body is JSON and names the array (`mono_mix` or the
  channel label) and the reason. Both are implementation failures (below),
  never a decline. The caller does not retry inline: a `422` is a
  configuration fault (a wrong mount, a stale file) to show, not to hide.
- **`GET /health`**, on the same origin as the design endpoint, answers
  `200` with at least `{"contract_version": "1.2", "shared_root": true}`
  from a designer that can take arrays by reference. Other fields are
  allowed and ignored (beqforge also sends `"status": "ok"`, for monitors
  that already poll it). Optionally, `"parallel": N` (an integer 1-8) says how
  many designs the designer runs at once (beqforge 0.2.4 on, `--workers N`); a
  caller then sends it no more than N at a time (§7.2). Absent means 1. From
  1.4 it may also report `design_version`, `design_settings` and
  `designer_build` as on a response (§3, "Design identity"), and
  `design_settings_detail`, the non-default settings as an object, so a person
  can see why two designers disagree; a caller shows it and keys nothing on it.
  It may also report `designer_release`, the release it is as its version
  (`"0.2.10"`), for people: a build names a revision, not a release.
  From 1.5 it may list `adjustment_kinds`: each kind it knows (§2's and its
  own `x-` kinds), as `{kind: {"description": str, "params": {name: {"type":
  "number" | "integer" | "string" | "boolean", "unit": str, "minimum":
  number, "maximum": number, "default": any, "description": str}}}}`, every
  member but `description` optional. It is advisory: a caller builds its
  choices from it, but may still send a kind not listed, which is answered
  `not_understood`. A caller sends adjustments only to a designer whose
  `/health` reports `contract_version` 1.5 or later, or that it was
  configured to send them to.
  From 1.6 it may report `"impact": true` when the designer answers `POST /impact` (below).
  The caller reads it
  once, before its first request by reference to that designer, and sends
  none — failing the design with the answer in its message — when the
  version is below 1.2 or `shared_root` is not `true`.

**Impact (1.6).** `POST /impact`, on the same origin as the design endpoint, takes run records
that designer builds wrote and answers, for each one, how likely a design of the same request
by *this* designer is to differ from the one the record holds. It is advisory: a person
decides what to redesign, and the caller never redesigns or marks a design reviewed (§3) on
this answer alone.

Request:

```json
{
  "contract_version": "1.6",
  "records": [
    {"id": "jriver-18bd9f7c1e48-2015688", "path": "records/designer-0123456789abcdef-89abcdef-v6.run.json.gz"}
  ]
}
```

- **`id`** is the caller's own key, unique within the request, and is echoed back. The
  designer attaches no meaning to it.
- **`path`** is a design's `run_record` (§3), sent back unchanged: relative to the shared root,
  under "Arrays by reference"'s rules (POSIX-style, no backslash, no `..` segment, refused if its
  real path lies outside the root's). It is never an absolute path or a path from `commentary`.
- **At most 500 records per request.** A caller sends a library in several requests, one after
  another to one endpoint; `/impact` reads records, it does not design, so `parallel` (§7.2)
  does not apply. A request of 500 takes tens of seconds.

Response, `200`:

```json
{
  "contract_version": "1.6",
  "design_version": 7,
  "design_settings": "9cf04c30b0f0",
  "designer_build": "…",
  "changes": [
    {"design_version": 7, "id": "fit-choice", "summary": "the fitter chooses among near-equal fits by the judge's checks",
     "moved": "swapped designed/declined, changed strategy, or the cascade moved > 1 dB at 5-50 Hz",
     "basis": [{"class": "declined_after_fit", "measured": 34, "moved": 18}]}
  ],
  "results": [
    {"id": "jriver-18bd9f7c1e48-2015688", "tier": "likely", "score": 0.72, "from_design_version": 6,
     "reasons": [{"change": "fit-choice", "feature": "fit.choice",
                  "detail": "the fit has near-equal alternatives; its slot check failed by 0.08 dB"}]},
    {"id": "…", "tier": "unknown", "score": null, "from_design_version": null, "unknown_reason": "not_found",
     "reasons": [{"change": null, "feature": null, "detail": "record not found"}]}
  ]
}
```

- **`results`** holds one entry per request record, in request order.
- **`tier`** is one of:
  - `will_move`: certain, e.g. a recorded figure lies between a threshold's old and new values;
  - `likely`;
  - `possible`;
  - `unaffected`: no change since the record's version touches anything this design relied on;
  - `unknown`: the designer cannot say. That covers a record that is missing, unreadable,
    escaping the root or from another designer, one older than the designer's impact data,
    one written under different `design_settings`, or one that lacks what a change needs to
    be scored.

  `unknown` is a result, never left out, and a caller treats it as `will_move` unless a person
  decides otherwise.
- **`score`**, in `[0, 1]`, is an estimated probability that a design now differs from the
  record's, measured by the designer's own regression check (rerunning records at both
  versions), never invented: a sum over records is the expected number that move. Several
  changes combine as `1 − ∏(1 − p)`, not their maximum. A threshold change scores `1.0` for a
  recorded figure between its old and new values; a figure that cannot be scored makes the
  record `unknown`. It depends only on the record and the designer's build, so it is comparable
  across every response from one design identity, and orders records within a tier, highest
  first. It is `null` for `unknown`.
- **`unknown_reason`** (optional, only with `unknown`) says why as a code: `not_found`,
  `unreadable`, `outside_root`, `not_a_record`, `unknown_build` (written by a build the
  designer cannot place), `older_than_data`, `other_settings`, `newer_version`, `no_figure`. A
  caller tolerates a code it does not know; `reasons` stays the text a person reads.
- **`from_design_version`** is the version the record was designed at, read from the record.
  It is `null` when unknown.
- **`reasons`** is non-empty for every tier but `unaffected`. Each entry names the change (an
  `id` from `changes`, or `null`), the part of the design it reached (`feature`, the
  designer's own name, for display), and `detail` for a person. All of it is free text for
  display: never machine-parsed, like `commentary`.
- **`changes`** lists every change between the oldest `from_design_version` in the response
  and this designer's `design_version`, oldest first, so a person can read what moved. A
  designer states what counts as moved in each change's basis: optionally `moved`, its
  definition as text for display, and `basis`, what the scores were measured on, as
  `[{"class": "declined_after_fit", "measured": 34, "moved": 18}, ...]` (`class` is the
  designer's own name for an outcome class; `moved` is never more than `measured`). A
  change that can reach every design (a "redesign everything" change) is listed like any
  other; every record it reaches is then at least `likely`.
- The identity fields are this designer's, as on `/health` (§3, "Design identity").

Rules:

- **Same records, same build, same answer** (§1, determinism).
- **A record from this designer at this identity** is `unaffected`, with no reasons. It is not
  out of date.
- **Errors.** `400` for a body that does not parse or fails the schema, more than 500 records
  or two with one `id`. `422` when **no shared root is configured** on this designer, naming
  that as the reason. A bad individual record is never a `422`: it is that record's `unknown`,
  so one stale path cannot sink a batch. Both errors are implementation failures, as for
  `/design`.
- **Which designer to ask.** A caller sends `/impact` only to a designer whose `/health`
  reports `"impact": true` and the `design_version` and `design_settings` of the identity it
  would redesign with (§3). It keys nothing on `designer_build`.

**Errors are not declines.** A non-2xx status, a connection failure or
timeout, or a body that doesn't parse into a `DesignResponse` are all
implementation failures per §1 — the caller's HTTP binding raises rather
than inventing a decline on your behalf. A *well-formed* `DesignResponse`
that fails §3-§5 validation is unaffected by which binding produced it —
that check happens after either binding hands back a `DesignResponse`, not
as part of decoding one.

**Formal spec:** `docs/schema/http_designer_request.schema.json`,
`http_designer_response.schema.json` and (1.2) `http_designer_health.schema.json` (with 1.5's adjustments in all three), and (1.6)
`http_designer_impact_request.schema.json` and `http_designer_impact_response.schema.json` — the machine-checkable version of the
shapes above, meant for implementers in any language (unlike the rest of
this document, which only needs to match this repo's own Python).

### 7.2 Several endpoints (caller side)

Nothing changes for a designer: this is how a caller spreads its designs over several copies of one. A profile may
declare a designer as a list of endpoints, each with its own `by_reference` and `headers`, and optionally `parallel` (how
many designs it is sent at once, 1-8). Without `parallel`, an endpoint is sent as many as its `/health` reports
(`parallel`, §7.1), or one at a time if it does not say; the figure is read at every check, so a designer restarted
with more workers is used to the full within a minute. The designer's `timeout` and `headers` apply to all of them:

```yaml
designers:
  beqforge:
    timeout: 900
    endpoints:
      - {url: 'http://10.151.167.10:8420/design', by_reference: true}   # as many as its /health reports
      - {url: 'http://10.151.167.20:8420/design', parallel: 2}          # 2, whatever it reports
```

The caller (`pipeline.designer.pool`) registers them as one designer under its name, so a design's fingerprint (which
names the designer) does not depend on which endpoint answered. **Every endpoint must run the same designer build**, or
two designs of one title could differ by where they ran. A run keeps as many designs in hand as the endpoints that are up take together
(in place of `run.parallelism.design`, and following them as they change), and each design goes to the endpoint with the most room
that is up. The caller asks each endpoint's `/health` (§7.1) before using it and again every minute; one that does not
answer, or that a request cannot reach (no connection, or a gateway's 502/503/504), is left out until it answers again,
and that design is sent to another endpoint. Any other failure, including a design that outlives its timeout, is the
design's own: it is not sent elsewhere. Only when no endpoint is up is the design unavailable. A single `url` is one
endpoint taking `run.parallelism.design` designs at once, as before.

---

## 8. What the caller guarantees

- `mono_mix` (and `channels`, if present) is finite, decimated audio at the
  stated `fs`. **No guarantee about how it was produced** — see §2's "No
  guarantee of provenance": it may be unprocessed theatrical/consumer audio,
  known material with a known filter applied for accuracy testing, or fully
  synthetic material for false-positive testing, and `coverage` (§2) states
  whether it's the complete programme or an excerpt. Production and
  validation calls are indistinguishable on purpose — do not special-case
  either.
- **Size, so you can plan for it:** at the common-case decimated rate
  (1 kHz), a 2-hour `mono_mix` is ~58 MB as float64; a `channels` dict for
  an 8-channel source adds up to ~460 MB more at the same rate and
  precision. v1.0 mandates float64 for both (§2) for simplicity, but if
  memory becomes a real constraint on your side, float32 halves this with
  no meaningful loss at 1 kHz — flag it if you need that, since it would be
  a contract change (§7), not something to do unilaterally. Over HTTP,
  that is 931 MB of JSON for a 2.7-hour title with 8 channels (about 8 s
  to encode, send and decode on one host); arrays by reference (§7.1, 1.2)
  cut it to the hashing, about 2 s.
- You will not be asked to redesign against feedback from a previous call
  for the same title — each call is independent; there is no session state
  to leak between titles.
- A `DesignResponse` that violates §3-§5 (wrong shape, disallowed filter
  type, budget exceeded, non-finite values — in `rejected` entries as much
  as in `candidates`, or `rejection_reasons` in the wrong list) is rejected
  before anything downstream runs, and rejection is reported back as a build/integration
  problem, not silently coerced into something publishable.
