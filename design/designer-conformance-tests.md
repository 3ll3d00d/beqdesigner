# Designer conformance test spec — v1.0

**Status:** a checklist, not a contract. [`designer-interface.md`](designer-interface.md)
is the contract; this document is a test spec extracted from two independent
implementations of it — this repo's caller side
(`src/test/python/test_pipeline_designer_contract.py`,
`test_pipeline_designer_http_binding.py`) and beqforge's designer side
(`tests/test_design_designer.py`, `tests/test_design_designer_server.py`,
`beqforge/designer.py`). Comparing the two, the same ~30 behaviours got
tested independently on both sides, converging on the same list without
either project copying the other — that convergence is the evidence this is
the right list, not an artifact of one codebase's style. This document
exists so a third implementer doesn't have to rediscover it from scratch by
reading two other repos' test suites.

**Audience:** whoever implements a filter designer against
`designer-interface.md`. **Read that document first; this one assumes it and
will cite it by section (`§n`) throughout.** Like that document, nothing
here is beqanalyser- or beqforge-specific, and nothing assumes Python or
access to either repo — every check below is stated as a behaviour
(given/when/then), with a Python/pytest illustration where one helps, not as
a thing you must import.

**How to use this:** not everything applies to every implementer. §2-§4 are
pure functions of a response value — no audio, no fitting — and apply
regardless of language or binding; write these first, they're the cheapest
and catch the most contract violations. §5 only applies if you bind over
HTTP (§7.1). §6-§7 depend on your own internals but the *pattern* they
describe generalises. Section numbers below are this document's own
(`§2`-`§8`); references to the contract are written `interface §n`.

---

## 1. Why a response-shape checklist is most of this document

`design()` is a pure function (`interface §1`): one `DesignRequest` in, one
`DesignResponse` out, no session, no I/O on the happy path. That means the
overwhelming majority of the contract is checkable without ever running your
actual detection/fitting logic — construct a `DesignResponse` by hand (or
with a fake/monkeypatched internal result) and check it against `interface
§3`-`§5`'s rules directly. Both reference implementations structure their
tests this way (a `validate_response`/`ContractViolation` function, tested
in isolation from anything audio-shaped) and it is worth copying that
structure even if you copy nothing else from this document: it lets you test
the contract boundary in milliseconds, independent of how slow your real
detection is.

---

## 2. Response shape (`interface §3`)

| # | Given | Then | ref |
|---|---|---|---|
| 2.1 | a response with both `candidates` and `decline_reason` populated | rejected | §3 |
| 2.2 | a response with neither populated | rejected | §3 |
| 2.3 | a response with `candidates = []` (empty list, not `None`) | rejected — "zero candidates" is what a decline already means | §3, §4 |
| 2.4 | a response with one candidate | accepted | §3 |
| 2.5 | a response with several candidates whose `confidence` is non-increasing (`[0.9, 0.6, 0.6]` — ties allowed) | accepted | §3 |
| 2.6 | a response with candidates whose `confidence` is *not* non-increasing (`[0.5, 0.9]`) | rejected — "must be ordered best-first" | §3 |
| 2.7 | `response.contract_version` | equals the *request's* `contract_version`, unchanged | §7 |

```python
# illustration, not a requirement to use pytest
def test_candidates_not_ranked_best_first_rejected():
    response = make_response(candidates=[candidate(confidence=0.5), candidate(confidence=0.9)])
    with pytest.raises(ContractViolation):
        validate_response(response)
```

## 3. Candidate fields (`interface §3`)

| # | Given | Then | ref |
|---|---|---|---|
| 3.1 | `confidence = None` | rejected | §3 |
| 3.2 | `confidence` outside `[0.0, 1.0]` (e.g. `1.5`) | rejected | §3 |
| 3.3 | `confidence = 0.0` or `1.0` (boundary) | accepted | §3 |
| 3.4 | `mv_adjust_db` is `None`, `NaN`, or `±inf` | rejected — must be a finite number | §3 |
| 3.5 | `gain_reduction_db = None` (not computed) | accepted | §3 |
| 3.6 | `gain_reduction_db = 0.0` (no reduction needed) | accepted — `0` is the "cheap filter" case, not the same as absent | §3 |
| 3.7 | `gain_reduction_db` negative (e.g. `-4.37`) | accepted | §3 |
| 3.8 | `gain_reduction_db` positive (e.g. `+4.37`) | rejected — always `<= 0` by construction | §3 |
| 3.9 | `gain_reduction_db = NaN` | rejected | §3 |
| 3.10 | `method = 'non_parametric'` with `residual_db = None`, `residual_band_hz = None` | accepted — "no target to measure against" is a valid state for this method only | §3 |
| 3.11 | `method = 'non_parametric'` with `residual_db`/`residual_band_hz` populated (fit against a *constructed* target) | accepted | §3 |
| 3.12 | `commentary = {'k': 'v', ...}` (flat, string-to-string) | accepted | §3 |
| 3.13 | `commentary` with a non-string value (e.g. `{'knee_hz': 25.0}`) | rejected | §3 |
| 3.14 | `commentary` that isn't a dict at all (e.g. a plain string) | rejected | §3 |
| 3.15 | `filters = []` on a candidate | rejected — omit the candidate instead | §3, §5 |

## 4. Decline path (`interface §4`)

| # | Given | Then | ref |
|---|---|---|---|
| 4.1 | a decline with a non-empty `decline_reason` and `candidates = None` | accepted | §4 |
| 4.2 | `decline_reason = ''` (empty string) | rejected — must be non-empty | §4 |
| 4.3 | a decline | `candidates` is `None`, never `[]`, and nothing downstream (simulate/publish) runs | §4 |
| 4.4 | your code path that returns a decline | does **not** raise an exception to do it — declining is a normal return value, not an error | §1, §4 |
| 4.5 | (your own mapping layer) a "nothing to correct" internal result | maps to a decline with a **stable, machine-usable, `snake_case`** `decline_reason` — not a generic catch-all string, unless nothing more specific applies | §4 |

On 4.5: both reference implementations keep an internal table mapping their
own internal blocker/failure reasons to contract-level `decline_reason`
codes (beqforge's `_BLOCKER_CODES`, matched by substring against its own
internal notes; beqdesigner's caller side doesn't produce declines itself
but validates the *shape* of whatever a designer sends). If your internal
failure taxonomy is richer than the contract's suggested codes, keep your
own table and test that each of your internal failure modes lands on the
code you intend — a decline your dashboard can't distinguish from another
one is a decline whose `decline_reason` picked the wrong granularity.

## 5. Biquad validity (`interface §5`)

| # | Given | Then | ref |
|---|---|---|---|
| 5.1 | 11 `BiquadSpec` entries in one candidate's `filters` | rejected — budget is 10 per candidate | §5 |
| 5.2 | 10 entries | accepted | §5 |
| 5.3 | 10 entries in **each** of two candidates (20 total) | accepted — the budget applies per-candidate, not to the response as a whole | §5 |
| 5.4 | `type = 'all_pass'` (or any type outside the allowed three) | rejected — "not publishable" | §5 |
| 5.5 | `type` one of `'peaking_eq'`, `'low_shelf'`, `'high_shelf'` | accepted | §5 |
| 5.6 | `freq_hz = 0.0`, negative, or `+inf` | rejected in each case | §5 |
| 5.7 | `gain_db = NaN` | rejected | §5 |
| 5.8 | `gain_db` any finite value including `0.0` (sign unrestricted) | accepted | §5 |
| 5.9 | `q = 0.0` or `+inf` | rejected in each case | §5 |
| 5.10 | `q` any finite positive value | accepted | §5 |
| 5.11 | the same `BiquadSpec` (identical type/freq/gain/q) twice in one `filters` list | accepted, and **not** your job to collapse into a stacked/counted representation — that's the caller's concern (`interface §5`: "no count/stacking field") | §5 |

**Gain convention self-check (`interface §5`):** worth one concrete test
independent of the table above, since getting this wrong produces
values that pass every rule in §5 while still being numerically wrong by a
factor that a schema check can't catch. `interface §5`'s worked identity —
a zero at `fz`, pole at `fp`, shared `Q` inverts to
`low_shelf(freq_hz = √(fz·fp), gain_db = 40·log10(fz/fp), q = Q)` — is also
`interface §6`'s worked example: `fz=25`, `fp=10` ⇒
`freq_hz ≈ 15.810`, `gain_db ≈ 15.918`. If your method produces this kind of
exact-alignment inverse, reproduce that arithmetic in a test and compare
against those published numbers (or your own reference implementation's
filter coefficients, `A = 10**(gain_db/40)`, **not** `/20`) — a sign or
scale error here changes what a downstream player does to real master
volume (`interface §3`'s pinned sign convention on `mv_adjust_db`), so it's
worth catching with a number you can check by hand rather than trusting a
range check alone.

## 6. Mapping your internal result into a `DesignResponse` (your logic; pattern is generic)

This section is inherently implementation-specific — your internal
pipeline's result type is not this contract's business — but the *shape* of
how both reference implementations tested this mapping generalises:

- **Use hand-built or monkeypatched fakes of your own internal result type
  for most of these tests, not a real detection/fit run.** A real run is
  slow (beqforge's docstrings measure 40-80s for a full multi-strategy run
  against real-length material) and mapping-layer bugs (a field renamed on
  one side, a sign flipped, a clamp forgotten) don't need real audio to
  reproduce — they need one specific internal result value fed through your
  mapping function. Keep real runs for §7.
- **Test the defensive edges explicitly**, the same way beqforge tests its
  own mapper: an internal confidence score that comes back `NaN` (clamp to
  something in-range rather than propagate — beqforge clamps to `0.0`), a
  `gain_reduction_db` that's only computed/reported when the caller actually
  supplied enough information to compute it honestly (`interface §2`:
  `bass_management` present *and* a per-channel decomposition available —
  don't approximate it from `mono_mix` alone and report a number anyway).
- **One consistency test between your mapper and your validator.** Feed a
  real (non-monkeypatched, or at least not hand-constructed) mapped output
  through your own `validate_response` equivalent and assert it doesn't
  raise. This is not a tautology if the two are separately maintained code
  paths (as they are in both reference implementations) — it catches the
  mapper and the validator drifting apart from *each other*, which a test
  that only exercises the validator against hand-built fixtures cannot
  catch, since those fixtures are never wrong in the way a real code path
  can accidentally become wrong.
- **If you reimplement `validate_response`'s rules yourself** (likely, if
  you're not in this repo's Python codebase, or even if you are but in a
  separate process/language) — treat it as its own unit under test, checked
  against §2-§5 above directly, not just inline assertions scattered through
  the mapping function. beqforge's `validate_response` is a from-scratch
  reimplementation of this repo's `pipeline.designer.convert.validate_response`
  for exactly this reason (can't share code across the process/venv/repo
  boundary) and both are tested against the same table independently — do
  the same rather than trusting the mapping function alone to encode every
  rule correctly by construction.

## 7. HTTP wire format (`interface §7.1`) — only if you bind over HTTP

Skip this section entirely for an in-process-only binding.

| # | Given | Then | ref |
|---|---|---|---|
| 7.1 | a `POST` to your design path with a well-formed request body | `200`, body decodes as a valid `DesignResponse` per §2-§5 | §7.1 |
| 7.2 | a `POST` with a body that isn't valid JSON at all | a `4xx` with an error body, not a crash / hung connection / `5xx` | §7.1 |
| 7.3 | a `POST` missing a required `DesignRequest` field (e.g. no `fs`) | a `4xx` with an error body | §7.1 |
| 7.4 | a `GET`/`POST` to an unrecognised path | `404` | — (not contract-mandated, but good hygiene) |
| 7.5 | `mono_mix`/`channels` arrays | encoded as `{"dtype": "float64", "shape": [...], "data_base64": "..."}`, little-endian raw float64 bytes — not a JSON number array | §7.1 |
| 7.6 | an array with a `dtype` other than `"float64"` | rejected (`4xx`), not silently coerced | §7.1 |
| 7.7 | `channels` / `bass_management` absent from the request | your decoded `DesignRequest` has `channels = None` / `bass_management = None`, not an empty dict or a KeyError | §2, §7.1 |
| 7.8 | a **decline** response body | omits (or nulls) the `candidates` key; a **success** body omits (or nulls) `decline_reason`/`decline_message` | §3, §7.1 |
| 7.9 | your own mapping produces a `DesignResponse` that fails your own §2-§5 validation | the server responds `5xx` with the violation described — **never** silently reshaped into a decline and never sent as if valid. This is your bug to report loudly, not the caller's problem to guess at (`interface §1`: "the caller treats an exception as an implementation failure") | §1, §3-§5 |
| 7.10 | at least one of the tests above | runs against a **real socket** (real `http.server`/framework request handling), not only against your in-process `design()` function directly | — |

On 7.10: everything else in this document can be — and, for speed, mostly
should be — tested without a socket. But a transport-layer bug (a wrong
status code, a header the client library can't parse, `do_POST` never
actually reaching your `design()` call) is invisible to any test that calls
your Python/whatever function directly. Both reference implementations keep
exactly one test file that opens a real socket for this reason
(`test_pipeline_designer_http_binding.py` on the caller side,
`test_design_designer_server.py` on the designer side) and keep everything
else mocked/monkeypatched. Do the same rather than either extreme (every
test through a real socket — slow; no test through a real socket — leaves
the transport layer unverified).

**A concurrency note if your detection forks worker processes** (a
multi-section fit escalating to a process pool, say): serve one request at a
time. `interface §1` guarantees one synchronous call per title — there is no
concurrent request to lose by doing so — and forking from a
multi-threaded HTTP server risks a classic fork-deadlock (a lock held by
another thread at fork time is never released in the child). Not a
contract requirement, but worth a comment next to wherever you configure
your server's threading model, the way `tools/designer_server.py` does.

## 8. End-to-end smoke tests

Two real-audio cases are worth keeping regardless of how much of §6 you
covered with fakes — they're the only tests that exercise your actual
detection/fitting logic against something audio-shaped, and the only ones
that would catch, e.g., a real fit producing a contract-invalid result that
every hand-built fixture in §6 happened not to expose:

- **A "no rolloff, real decline" case.** Plain noise (or genuinely flat
  material), no per-channel decomposition if your method needs one to say
  anything — should decline quickly and for the *right* reason (§4.5's
  granularity point), not time out or fall through to a generic catch-all
  code.
- **A "known injected rolloff, real correction" case.** Synthetic material
  built by applying a *known* high-pass (a Butterworth/Linkwitz-Riley
  alignment at a known corner frequency you chose) to noise shaped with
  enough dynamic structure (loud/quiet segments) to resemble real programme
  material, rather than pure stationary noise. Assert on the result's
  *shape*, not exact numbers your method happens to produce today:
  `0.0 <= confidence <= 1.0`, `method` is one of the three allowed values,
  `1 <= len(filters) <= 10`, and — the strongest check — **run the result
  through your own `validate_response` and assert it doesn't raise**, so
  this test would fail the moment your real fitter starts producing
  something the rest of this document's rules reject, without having to
  duplicate every one of §2-§5's checks inline here.
- If you bind over HTTP, run at least the "known injected rolloff" case
  through the real socket per §7.10, not just via your in-process `design()`
  — it's the test most likely to be slow, so it's tempting to skip the
  transport hop for it specifically; don't, that's exactly the path a real
  caller takes.

Keep these two minimal and few — they're slow by nature (a real fit, even a
deliberately cheap single-strategy one, costs single-digit seconds at
least) — everything that doesn't need real audio to fail belongs in §2-§7
instead.

---

## Appendix: reference reading

Both were written independently against `designer-interface.md` alone, not
against each other — that's what makes the overlap between them evidence
for this checklist rather than a copy of one project's taste.

- **Caller side** (this repo) — `src/test/python/test_pipeline_designer_contract.py`
  (§2-§6 above), `src/test/python/test_pipeline_designer_http_binding.py`
  (§7 above), implementation in `src/main/python/pipeline/designer/`.
- **A designer implementation** (beqforge, a sibling repo) —
  `tests/test_design_designer.py` (§2-§6 above),
  `tests/test_design_designer_server.py` (§7-§8 above), implementation in
  `beqforge/designer.py` and `tools/designer_server.py`.
