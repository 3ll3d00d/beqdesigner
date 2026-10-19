# Design version: knowing which designs a designer upgrade changes

**Built 2026-10-19** on both sides (beqforge's in its working tree, to be released): the contract is now 1.4 in
[designer-interface.md](designer-interface.md) ("Design identity"). Agreed 2026-10-19 (TODO L3, extending D4), with
beqforge's four refinements the same day. The first beqforge release with it is 0.2.7, at version 4 (0.2.6 was tagged, never published). beqforge's all-defaults digest is `9cf04c30b0f0`; its detail keys are
strategies, exclude_bands_hz, goal_tolerance_db, goal_tilt_db_per_octave, content_edge, device_rate_hz,
coefficient_bits and integer_bits (strategy order kept: it can change the ranking).

**Caller side built** (2026-10-19): the response and health fields kept per entry, the backfill and the out-of-date
rule (`pipeline/designer/identity.py`); each endpoint's identity, the current one and the disagreement warning on
`/v1/status` and the Status page; `out_of_date` on the titles filter, the Titles page's *Design version* select and
*Redesign out of date…*, and a redesign job by version. Not built: the desktop work list's filter (it has no
`declined` either) and a count in D4's drift banner. Waiting for beqforge TODO 28; until then every title is
*unknown* and the filter is refused, as designed.

## Why

A queue entry keeps only the build string the designer happened to put in its commentary
(`review.designer_build()`): `e1a2c24b1172+src:008f1c30d377 (frozen build)` from the Windows exe and
`unknown+src:76b2edd7e060` from the container, both beqforge 0.2.5. Two builds of one release do not
compare equal, two releases that design identically (0.2.3, 0.2.4, 0.2.5) do not either, and nothing says
whether a newer designer would answer differently. After the first library run, choosing what to redesign
meant computing beqforge's source digest at each tag and reading its commit log by hand.

The design fingerprint deliberately leaves the build out, so an upgrade never redesigns anything on its own.
That stays.

## Contract 1.4 (additive)

On `DesignResponse`, optional, returned to a request of any version (as the 1.1 and 1.3 fields are):

```python
    # 1.4: the designer's identity as far as the answer is concerned
    design_version: int | None = None      # bumped by any change that can alter an answer to the same request
    design_settings: str | None = None     # digest of the designer's own settings that can alter an answer
    designer_build: str | None = None      # the build, for people: release, revision, packaging
```

- **`design_version`** is a non-negative integer that never decreases. A designer bumps it with any change
  that can alter `candidates`, `rejected`, a decline or a 1.3 fit figure for an unchanged request; a change to
  wording only (`commentary`, `decline_message`) or to speed, packaging or transport does not. A version names
  one algorithm, not one answer: floating point differs between processors and BLAS kernels, and for an
  optimising strategy that can grow into a different optimum, so the same version and settings on different
  hardware can design a title differently (beqforge 0.2.6's golden: 2 sections against 3, a 15.2 against a
  16.5 dB boost). A caller compares versions, never figures, and a redesign on another machine may move a
  title even when nothing is out of date. Adding a
  field that leaves the answer as it was (1.3's fit figures) is not a bump. 0 means "before versioning".
- **`design_settings`** is an opaque string, equal between two designers exactly when the server-wide
  settings that can change an answer are equal (beqforge: strategies, exclusions, goal tolerance and tilt,
  content edge, device rate, coefficient and integer bits). Settings that cannot (workers, cores, cache,
  record, log and shared-root) are left out, and so is every setting at its default: all defaults is the
  empty set's digest, so a new flag whose default keeps today's answers changes nobody's digest (changing a
  default that alters answers is a `design_version` bump). Always sent by beqforge; absent means a designer
  has no such settings.
- **`designer_build`** replaces the commentary key a caller reads today; free text, for display.
- `GET /health` reports the same three fields, so a caller knows them before it sends anything, and may add
  **`design_settings_detail`**, the non-default settings as an object, so a person can see why two live
  designers disagree. A caller shows it; it never keys anything on it.

A caller keys nothing on `designer_build`. The pair (`design_version`, `design_settings`) is the
**design identity**.

## Caller behaviour

1. Each queue entry keeps the identity it was designed with (and `designer_build`). A pre-1.4 entry gets one
   from a backfill table of the builds this library has seen, mapped to the versions beqforge declares for
   its past releases: before 0.2.0 = 0, 0.2.0 = 1, 0.2.1 = 2, 0.2.2 = 3, 0.2.3 to 0.2.5 = 4 (confirmed by
   beqforge against its tags and source digests, 2026-10-19). A build not in the table has none. Backfilled
   entries get the default settings' digest only when the run's profile shows the designer ran with defaults.
2. The pool reads each endpoint's identity from `/health` as it does `parallel`. The **current identity** is
   that of the highest `design_version` among endpoints that are up. Endpoints that are up and disagree are
   shown on the status page as a warning, with each one's `design_settings_detail`, since a redesign would
   answer differently by endpoint.
3. `Selection` gains **out of date**: an entry whose identity differs from the current one, or that has none.
   The titles page, the work list filters and `POST /v1/jobs/run` take it like `declined`; the drift banner
   D4 describes counts it.
4. Nothing redesigns by itself. Accepted and published entries are protected as today; a redesign stays a
   person's choice or an explicit scheduled job.

## Done when

Today's requeue is one filter and one job: select *out of date* (minus whatever a person excludes, such as
the mono titles), redesign, with no version map built by hand. Tests: the identity carried from response to
entry; the backfill; the pool's current identity and the mixed-pool warning; the selection over the index;
a pre-1.4 designer (no fields) leaving every entry *unknown* and nothing else changed.
