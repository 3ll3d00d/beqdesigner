# BEQ filter designer interface — v1.0

**Status:** CONTRACT. Ready to implement against. No code exists yet on
either side; this document is the thing both sides build to.

**Audience:** whoever implements a filter designer — beqanalyser first, but
nothing here is beqanalyser-specific. **Read this document alone; it does
not assume you have access to the beqdesigner repo.**

This is the formal version of the contract sketched in
[`api-headless-pipeline.md §2`](api-headless-pipeline.md) and negotiated with
the beqanalyser project (`api-headless-pipeline.md §15` has the negotiation
summary). Where the two documents disagree, **this document wins** — it is
the one meant to be implemented against, and will be kept in sync if the
contract changes. `api-headless-pipeline.md §2`/`§15` explain the *why*; this
explains the *what*, precisely enough to write code against without reading
the rest.

*Revised twice against external review before either side had written code
against v1.0 — once on shape (added `coverage`, `method`, `channel_scope`,
`residual_band_hz`, tightened `confidence`/`residual_db`'s definitions,
dropped a false provenance guarantee), once on a factual error
(`mv_adjust_db`'s sign was measured and found inverted — see §3). Still
`contract_version = "1.0"`; nothing has been implemented against any earlier
draft.*

---

## 1. What you are building

A pure function:

```python
def design(request: DesignRequest) -> DesignResponse:
    ...
```

One call, one title, one answer. No session, no callback, no streaming, no
iteration protocol — "iterate until acceptable" (the caller's step 4) is
resolved by **you returning a confidence and/or declining**, not by being
called repeatedly with feedback. If you want to try several internal
candidates before answering, that happens inside this one call.

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
§3 for reporting what you conclude from it.

**`fs` will typically be low** — a decimated analysis rate (1000 Hz in the
common case), not the source material's native rate. Whatever content exists
above `fs / 2` has already been discarded before you see it; there is no
higher-resolution version available through this call. If your method needs
more bandwidth than a given `fs` provides, decline (§4) rather than
guessing — the caller controls `fs` and can be asked to raise it.

**What is deliberately absent, and why:** no headroom/max-boost/device
preference of any kind. Your answer must not be shaped by what the caller
intends to do with it — see `mv_adjust_db` in §3 for where headroom
information flows, and it flows *out*, never in.

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
contract violation the caller will reject.

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
class DesignResponse:
    contract_version: str            # echo the request's value

    # --- success: filters/confidence/mv_adjust_db populated, decline fields left None ---
    filters: list[BiquadSpec] | None = None
    confidence: float | None = None      # 0.0-1.0 — P(a real rolloff was applied); see below
    mv_adjust_db: float | None = None    # implied headroom compensation; see below
    method: DesignMethod | None = None   # how `filters` was derived; see below

    # fit-quality — populated whenever `method` is 'exact' or 'fitted'; see below
    residual_db: float | None = None            # max abs error, dB, over residual_band_hz
    residual_band_hz: tuple[float, float] | None = None

    # decline: these two populated, all success fields above left None
    decline_reason: str | None = None    # short stable code — see §4
    decline_message: str | None = None   # optional human-readable detail

    # optional either way: not published, surfaced to a human reviewer only
    fc_hz: float | None = None
    slope: float | None = None
    fc_uncertainty_hz: float | None = None
    slope_uncertainty: float | None = None
    channel_scope: ChannelScope | None = None    # see below; requires `channels` in the request
```

**`filters`** — see §5 for exactly what's allowed in a `BiquadSpec`. Order
in the list is not meaningful (biquad sections in a cascade commute); return
them in whatever order is natural to you.

**`confidence`** is precisely **P(a real rolloff was applied to this
programme)** — not "is `fc` right", not "is the inversion right", just
whether there is something here to correct at all. This is the number that
maps onto false-positive rate, which `api-headless-pipeline.md §15.4`
identifies as the metric that matters most
(zero negatives in the existing catalogue makes a false positive both
expensive and undetectable downstream). Keep fit quality out of it —
that's `residual_db`.

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
  weaker claim than the other two — there is no "exact target" for
  `residual_db` to be measured against, so leave `residual_db`/
  `residual_band_hz` as `None` here and let `confidence` alone carry the
  caller's trust in the result.

Do not infer `method` from `fc_hz is None` — leave `fc_hz`/`slope` as
optional diagnostics either way and populate `method` explicitly.

**`mv_adjust_db`** — your model's implied headroom compensation ("what
would master volume need to move by to keep this filter safe" — the same
quantity as the catalogue's `mv_adjust` field). This is the *only*
headroom-shaped value that crosses the boundary, and it only goes one
direction: out. You do not receive a headroom constraint as an input (§2),
and this field is not read back as a constraint on a later call — there is
no later call.

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
against whatever exact target you fitted it to (a minimax/peak error, not
RMS — matching how the worked examples below are reported). **`residual_band_hz`** is the `(low_hz, high_hz)` band
that error was measured over — a residual number with no band is not
comparable across titles with different knee frequencies. Both `None` when
`method='non_parametric'` (no exact target exists to measure against).

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
  is about the content, not the request).
- **`decline_message`** — optional, free text, for a human reading the
  report. Not machine-parsed.

Leave every success-path field (`filters`, `confidence`, `mv_adjust_db`,
`method`, `residual_db`, `residual_band_hz`) as `None` on decline. The
caller runs nothing downstream of a decline — no filter is simulated, no
XML is written, no report claims a correction was made.

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

**Budget: 10 biquad sections total.** Every `BiquadSpec` in `filters` counts
as one section (a `low_shelf`/`high_shelf` counts once per list entry, same
as `peaking_eq` — there is no multiplier). The cases worked through so far
land at 2-4 sections, nowhere near the limit; if your method is producing
close to 10, that's worth a second look on your side before it ever reaches
the caller.

**`freq_hz > 0`, `q > 0`.** `gain_db` may be any sign, including `0.0` (a
no-op section — allowed, though there's no reason to emit one).

---

## 6. Worked examples

**Success**, matching a worked, numerically-verified case — identifying an
`LR4` rolloff at 25 Hz and inverting it against an `LR4` protective filter at
10 Hz:

```python
DesignResponse(
    contract_version="1.0",
    filters=[
        BiquadSpec(type='low_shelf', freq_hz=15.810, gain_db=15.918, q=0.7071),
        BiquadSpec(type='low_shelf', freq_hz=15.810, gain_db=15.918, q=0.7071),
    ],
    confidence=0.94,
    mv_adjust_db=15.918,           # positive: master volume should come down 15.918 dB
    method='exact',                 # matched-alignment closed-form decomposition
    residual_db=0.0008, residual_band_hz=(5.0, 200.0),
    fc_hz=25.0, slope=24.0,        # e.g. dB/octave, if that's your slope unit — say so
    fc_uncertainty_hz=0.6, slope_uncertainty=1.1,
    channel_scope='all_channels',   # rolloff was consistent across the `channels` diagnostic
)
```

**Decline**, insufficient bandwidth to trust the estimate:

```python
DesignResponse(
    contract_version="1.0",
    decline_reason="insufficient_coherent_bandwidth",
    decline_message="Coherent band above candidate knee spans <1 octave; "
                     "below the margin this method requires.",
)
```

---

## 7. Versioning and future bindings

**`contract_version`** is `"1.0"` for everything in this document. Echo the
request's version back in the response unchanged — it lets the caller notice
a mismatch rather than silently misinterpret a field. Changes within `1.x`
will be additive-only (new optional fields, defaulting to `None`/absent);
you can ignore fields you don't recognise on the request side, and the
caller does the same for the response. A breaking change bumps to `2.0` and
this document gets a new version marker at the top.

**This document specifies data shapes, not a transport.** v1 binds them as
an in-process Python callable (§1) because that's the cheapest path for both
sides to start — nothing here stops a subprocess or HTTP binding later that
serialises the same fields as JSON (numpy arrays becoming nested lists or a
base64/npy payload — not yet specified, because v1 doesn't need it). If that
ever becomes necessary, the field names, types and validation rules above
don't change; only how they cross a process boundary does.

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
  a contract change (§7), not something to do unilaterally.
- You will not be asked to redesign against feedback from a previous call
  for the same title — each call is independent; there is no session state
  to leak between titles.
- A `DesignResponse` that violates §3-§5 (wrong shape, disallowed filter
  type, budget exceeded, non-finite values) is rejected before anything
  downstream runs, and rejection is reported back as a build/integration
  problem, not silently coerced into something publishable.
