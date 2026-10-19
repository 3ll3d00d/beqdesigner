# Design version: knowing which designs a designer upgrade changes

**Design, not built.** Agreed 2026-10-19 (TODO L3, extending D4). The designer side is beqforge's TODO 28
(`plans/design-version.md` there). Once both sides are built this becomes contract 1.4 in
[designer-interface.md](designer-interface.md) and the caller behaviour moves to [implemented.md](implemented.md).

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

- **`design_version`** is a positive integer that never decreases. A designer bumps it with any change that
  can alter `candidates`, `rejected`, a decline or a 1.3 fit figure for an unchanged request; a change to
  wording only (`commentary`, `decline_message`) or to speed, packaging or transport does not. Two builds that
  report the same number answer every request the same way, whatever their packaging.
- **`design_settings`** is an opaque string, equal between two designers exactly when the server-wide
  settings that can change an answer are equal (beqforge: strategies, exclusions, goal tolerance and tilt,
  content edge, device realisation). Settings that cannot (workers, cores, cache, record and log
  directories) are left out. Absent means the designer has none.
- **`designer_build`** replaces the commentary key a caller reads today; free text, for display.
- `GET /health` reports the same three fields, so a caller knows them before it sends anything.

A caller keys nothing on `designer_build`. The pair (`design_version`, `design_settings`) is the
**design identity**.

## Caller behaviour

1. Each queue entry keeps the identity it was designed with (and `designer_build`). A pre-1.4 entry gets one
   from a backfill table of the builds this library has seen, mapped to the versions beqforge declares for
   its past releases (beqforge's handoff asks for that list); a build not in the table has none.
2. The pool reads each endpoint's identity from `/health` as it does `parallel`. The **current identity** is
   that of the highest `design_version` among endpoints that are up. Endpoints that are up and disagree are
   shown on the status page as a warning, since a redesign would answer differently by endpoint.
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
