# Implementation plan — headless BEQ pipeline

**Status:** planning → ready to execute. Builds directly on `api-headless-pipeline.md`
(the design) and `designer-interface.md` (the designer contract) — read those
for *why*; this is *what ships, in what order, in what PR*. No code written yet.

This turns `api-headless-pipeline.md §12`/`§13`'s item table and phase list
into concrete PRs: files, dependencies, tests, and a definition of done per
PR. The item numbers (B1, B2, B3, B5, items 5-12) are the same ones from that
document's `§12` table — cross-referenced, not renumbered.

---

## Package layout

One new Qt-free package, per `§10`:

```
src/main/python/pipeline/
    __init__.py
    config.py              # AnalysisConfig — B1's replacement for import-time Preferences
    filters.py             # FilterSpec, create_filter()  — B3
    stats.py                # signal_stats()  — B5
    metadata.py            # BeqMetadata, tmdb_lookup(), validation  — item 5
    designer/
        __init__.py
        contract.py         # DesignRequest, DesignResponse, BiquadSpec — designer-interface.md §2/§3, item 12
        registry.py         # register_designer()/get_designer() — D8's in-process callable registry
        convert.py           # DesignResponse -> validated CompleteFilter — the §5 validation rules
    publish/
        __init__.py
        xml.py               # to_beq_xml() — thin wrapper over the existing HDXmlParser path
        art.py                # poster fetch — item 8
        report.py            # headless report renderer — item 9
        git.py                # publisher: place files, commit, push/PR — item 10
    orchestrate.py          # Session façade (§9) + Applied/Declined outcome — item 11

src/test/python/
    test_pipeline_publish_roundtrip.py   # Phase 0 — written first, against today's code
    test_pipeline_designer_contract.py   # designer contract validation, using a fake in-process designer
    test_pipeline_qt_boundary.py         # AST scan: no qtpy import anywhere under pipeline/
    test_pipeline_acceptance.py          # the Ready Player One worked example, end to end
```

Two rules from `§10`, unchanged: **`pipeline/` never imports `qtpy`**, and
**dependencies point inward** — the GUI comes to call into `pipeline/`, not
the other way round. Existing Qt-free modules (`model/iir.py`,
`model/codec.py`, `acoustics/`, most of `model/jriver/`) stay where they are;
only `model/xy.py` changes in place (B1).

---

## Phase 0 — pin the contract

**Ships:** `test_pipeline_publish_roundtrip.py` only. No production code.

Build a filter set and metadata by hand, write beqcatalogue XML via the
*existing* `HDXmlParser`, read it back with `xml_to_filt`
(`model/minidsp.py:257`), assert filters and metadata survive — the RP1
numbers from `§11.1` (`LowShelf fc=18 q=0.7 gain=4.5 count=5`, 6 biquads
against a 10 budget). This is the safety net every later phase runs against;
it must be green before Phase 1 starts touching the code it exercises.

**Depends on:** nothing. **Blocks:** everything — do this first.

---

## Phase 1 — make the existing pipeline callable

Items **B1, B2, B5, B3**, plus items **6** and **7**. All refactors of
existing code; the GUI stays the caller throughout, so there is no GUI
regression risk *from adding pipeline/* — the risk is entirely in changing
the code pipeline/ pulls out of.

**Order within the phase matters, and differs from `§12`'s table order:**

**1a. B1 — `model/xy.py:15`.** Remove the import-time `Preferences(QSettings(...))`
construction and the process-global `singleton` assignment; make smoothing
take an explicit config value instead. Add `pipeline/config.py`:
`AnalysisConfig` (`target_fs`, `resolution`, avg/peak windows — the fields
`§1`'s "Configuration" section names as workflow inputs). Add a small
GUI-side adapter that builds an `AnalysisConfig` from `QSettings` so existing
call sites keep working. **This must land first** — `model/iir.py` imports
`model/xy.py`, so every later item that touches filters or signals is
currently reading real desktop settings on import; nothing else in this
package should be built on top of that.
*Test:* import `model.iir` with `DISPLAY`/`QT_QPA_PLATFORM` unset and no
`QApplication` — assert no `Preferences`/`QSettings` object is constructed.
*Risk:* `model/xy.py` smoothing is used by the GUI today; needs a manual
regression pass on the waveform/spectrum views, not just the headless test.

**1b. B2 — ffmpeg accessor.** Public property for the built command +
`run_sync()` on `Executor` (`model/ffmpeg.py`); `execute()` becomes a thin
GUI-side wrapper over it. *Test:* the headless 5.1→mono→1kHz extraction
already verified manually in `§1` — turn it into a real test fixture (a
small synthetic multichannel wav, checked in or generated in the test).

**1c. B5 — `signal_stats()`.** Extract from `waveform.py:486` (+2
duplicates) into `pipeline/stats.py`: `signal_stats(signal) -> Stats(peak,
rms, crest, headroom, fs)`. Update the three call sites to use it.
*Test:* unit test against a synthetic signal with a known peak/RMS.

**1d. B3 — `FilterSpec`/`create_filter()`.** New `pipeline/filters.py`.
`FilterSpec` accepts either `q` or `s` for shelves (per `§2`'s S/Q duality
note) plus `count`, reports both back. `create_filter(spec, fs) ->
CompleteFilter`. Model it on `filter_from_json` (`model/codec.py:169`) —
closest existing relative — not on `create_shaping_filter`/
`create_pass_filter` (`model/filter.py:817`/`:855`), which are Qt-widget-bound
and not reusable. Update the GUI's filter-creation dialog to call through
this instead of its own widget-dispatch logic.
*Test:* round-trip every filter type through `FilterSpec` → `create_filter`
→ back to spec fields; assert `q`/`s` conversion matches `q_to_s`/`s_to_q`
(`model/iir.py:560`-`582`).

**1e. Item 6 — TMDB key as configuration.** Move the hardcoded key
(`§5`) out of source into config (env var or a config file entry), same
mechanism `AnalysisConfig`/the GUI adapter already establishes for B1.
Small, independent, can land any time in this phase.

**1f. Item 7 — `beq_gain` derivation, per D2.** Delete the dead
`__find_gain` (`model/postbuilder.py:343`); keep the signal-offset-based
population (`:117`) as the single implementation; add a publish-time check
that rejects a `Gain` filter reaching XML write rather than silently
producing a `beq_gain`-less entry.
*Test:* a `CompleteFilter` containing a `Gain` filter fails `to_beq_xml()`
loudly (Phase 2, but the validation rule is written here).

**End of Phase 1, verify:** extract → load → build a `FilterSpec` list by
hand → `create_filter` → `signal_stats` on the filtered signal, scripted,
headless, no `QApplication` instantiated anywhere in the call chain. This
isn't a checked-in test yet (no XML/report to assert against) — run it by
hand as a milestone check before starting Phase 2.

**Depends on:** Phase 0 green. **Blocks:** Phase 2 (needs B3, item 7),
Phase 3's report (needs B5 for headroom numbers in the table).

---

## Phase 2 — metadata, designer contract, and XML

Items **5** and **12**, run together — the designer contract is what
actually produces the `FilterSpec` list this phase publishes, so building
metadata/XML without it would just be re-testing Phase 0.

**2a. Item 12 — designer contract.** `pipeline/designer/contract.py`:
`DesignRequest`, `DesignResponse`, `BiquadSpec` exactly as specified in
`designer-interface.md §2`/`§3` — same field names, same types, same
`Literal` enums (`method`, `channel_scope`, `BiquadType`). This is a direct
transcription, not a design decision — any deviation from the formal
contract is a bug. `pipeline/designer/registry.py`:
`register_designer(name, callable)` / `get_designer(name)`, the in-process
callable registry `D8` recommends. `pipeline/designer/convert.py`: validates
a `DesignResponse` against `designer-interface.md §5` (only the three
publishable types, budget, `q`>0/`freq_hz`>0, exactly-one-of-success-or-decline)
and converts `filters: list[BiquadSpec]` into a `CompleteFilter` via
`create_filter` (item 1d) — this is the one place `BiquadSpec`'s RBJ gain
convention (`A = 10**(gain_db/40)`) has to be honoured exactly, since it's
what makes a `low_shelf`/`high_shelf` `BiquadSpec` land on the same
coefficients as `model/iir.py`'s `LowShelf`/`HighShelf`.
*Test:* `test_pipeline_designer_contract.py` — a **fake in-process
designer** (a plain Python function matching the contract signature,
returning canned `DesignResponse` objects) exercises every rejection path:
wrong filter type, budget exceeded, both/neither of success-decline
populated, non-finite values, `method='non_parametric'` with a `residual_db`
set (should be rejected — `§3` says both are `None` for that method). This
test needs no real designer and should not wait for beqanalyser to ship
anything.

**2b. Item 5 — `BeqMetadata` + `tmdb_lookup()`.** New
`pipeline/metadata.py`, extracted from `postbuilder.py`'s widget-bound TMDB
population (`:216`) and `__build_metadata` (`:293`, all 24 fields). Headless
TMDB client call, dataclass instead of Qt fields, validated (required fields
present, genres well-formed) rather than assumed correct.
*Test:* mock the TMDB HTTP call; assert `BeqMetadata` fields populate
correctly and validation rejects an incomplete response.

**2c. Wire to XML.** `pipeline/publish/xml.py`: `to_beq_xml(filters, meta) ->
str`, a thin wrapper over the existing `HDXmlParser`/`flat24hd.xml` path
(`§1`'s headline finding 1 — this side already works). Re-run the Phase 0
round-trip test through this new code path instead of the ad hoc script;
both must agree.
*Test:* Phase 0's test, now exercised via `to_beq_xml`/`convert.py` end to
end from a fake designer's `DesignResponse` through to XML.

**End of Phase 2, verify:** title + year + a `DesignResponse` (real or
faked) → publishable beqcatalogue XML string, headlessly.

**Depends on:** Phase 1 (B3, item 7). **Blocks:** Phase 5 (orchestrator
needs both metadata and the designer contract wired).

---

## Phase 3 — art and report

Items **8** and **9**. Independent of Phase 2 — can run in parallel; the two
only meet at the publisher in Phase 4.

**Before this phase starts:** resolve D4's follow-on (`§14`) — fixed output
width vs poster-driven width, and whether the filter table sits inside the
chart axes or below it. Both are one-line answers, not open design
questions, but they're inputs `report.py`'s signature needs on day one, not
something to discover mid-implementation.

**3a. Item 8 — poster fetch.** `pipeline/publish/art.py`: TMDB
`poster_path` → downloaded image file. Small; the download-from-URL helper
already exists per `§6`, this is packaging it headlessly.

**3b. Item 9 — report renderer.** `pipeline/publish/report.py`: the Agg
canvas shim (`§6` — matplotlib figure without a Qt-embedded canvas), filter
table parameters lifted off widgets into plain arguments, fixed size/layout
per the D4 follow-on answer. This is a spec-driven render, not a port of the
dialog's layout code (`§6`'s headline finding 4) — build against the size/
layout spec, not against what the dialog currently draws.
*Test:* deferred golden-image comparison per `§11.4`, until size/DPI are
settled (which this phase itself settles) — write the golden test at the
*end* of this phase, not before.

**Depends on:** Phase 1 (B5, for headroom numbers in the table) for 3b; item
8 is independent. **Blocks:** Phase 4.

---

## Phase 4 — publish

Item **10**. `pipeline/publish/git.py`: commit + push the XML and the
report image into two separate target repos, per D3's resolution (`§14`) —
no PR step, no beqcatalogue-side layout to guess at:

- **XML repo** (small, the one beqcatalogue actually clones): write the
  `to_beq_xml()` output somewhere under the configured subdirectory —
  `extract_from_repo()` globs `**/*.xml` recursively, so path/filename
  within that subdirectory is this module's own choice, not a beqcatalogue
  constraint. Commit, push to the default branch.
- **Images repo** (separate, per-user decision to keep the XML repo small):
  commit the `render_report()`/`compose_with_poster()` PNG, push, then
  build the GitHub raw-content URL for the pushed path
  (`raw.githubusercontent.com/<owner>/<images-repo>/<branch>/<path>`) and
  set that as `BeqMetadata.spectrum_url`/`.pva_url` *before* `to_beq_xml()`
  runs — the XML write has to happen after the image push, not before,
  since it needs the resulting URL.
- The XML repo (only) needs a copy of beqcatalogue's
  `.github/workflows/trigger.yaml` (fires a `repository_dispatch` at
  `3ll3d00d/beqcatalogue` on push) — that's a one-time repo-setup step, not
  something this module writes per publish.

**Still open: D5** (CLI vs service) — decides whether git credentials are
the invoking user's or a bot's, which this module needs to know before it
can commit anything. Everything else about this phase no longer depends on
external information.

**Also out of scope for this module, one-time and out-of-band:** adding the
new XML repo to beqcatalogue's hardcoded `repo_configs` list
(`beqcatalogue/__init__.py`) and `update_inputs.sh`'s parallel arrays — a
manual edit to the beqcatalogue project itself, not pipeline code.

**Depends on:** Phase 2 (XML), Phase 3 (report, art), D5. **Blocks:**
nothing downstream except Phase 5's end-to-end acceptance test.

---

## Phase 5 — orchestration

Item **11**. `pipeline/orchestrate.py`: the `Session` façade from `§9`,
composing everything above — `extract`/`load`, `design` (via the designer
registry), `curves`/`stats`, `to_beq_xml`, `report`, `publish`.

Deliberately last, per `§13`: the seams are only clear once every piece
exists independently; wiring them earlier tends to freeze the wrong
boundaries.

**What this phase actually adds**, beyond composition:

- **`Applied`/`Declined` outcome handling** (`§15.4`) — `session.design()`
  returns one or the other; the orchestrator must branch on it explicitly,
  and a `Declined` outcome runs nothing downstream (no simulate, no XML, no
  publish). This is new logic, not just wiring.
- **D7's provenance threading** — a designer's `confidence`, `method`,
  `fc_hz`/`slope`/uncertainties, `residual_db` reach the report (for human
  review before publish) but never the published XML (the catalogue format
  has no field for them, and no consumer other than this pipeline needs
  them).

**Acceptance test** — `test_pipeline_acceptance.py`: script the Ready Player
One example from `docs/workflow/beq.md` end to end through `Session`, using
a fake designer that returns the documented filter values, and assert the
documented numbers reproduce at every stage (filter values, headroom,
published XML, report). This is the closest thing to a full spec test for
the pipeline (`§11.2`), and it's the definition of done below.

**Depends on:** Phases 1-4. **Blocks:** nothing — this is the last phase.

---

## Definition of done

A single scripted call — no GUI, no `QApplication`, no manual steps —
that:

1. extracts a title's audio to mono (optionally decimated),
2. gets a filter from a registered designer (or accepts a hand-built
   `FilterSpec` list),
3. simulates headroom impact and reports the numbers,
4. on `Declined`, stops cleanly with a reason and does nothing else,
5. on success, produces a beqcatalogue-format XML, a report image, and (once
   D3 is resolved) a commit/PR to the target repo,

reproduces the Ready Player One numbers from `docs/workflow/beq.md`, and
`test_pipeline_qt_boundary.py` (AST scan, no `qtpy` under `pipeline/`, plus
a runtime check that `QApplication.instance()` stays `None` through the
whole run) stays green throughout.

## Carried-forward risks

- **D3** (beqcatalogue conventions) — resolved (`§14`); no longer a risk.
- **D5** (CLI vs service) should be decided before Phase 4 for the same
  reason; Phases 1-3 are transport-neutral by construction and don't need an
  answer.
- **D6** (catalogue-as-input, via `CatalogueEntry.iir_filters()`) is not on
  this plan at all — smaller, separate job, worth confirming whether it's
  also wanted, but doesn't block anything above.
- **B1's regression risk** (1a) is the one item in Phase 1 that touches
  live GUI behaviour, not just adds a headless path alongside it — budget a
  manual pass on the waveform/spectrum views after that PR, not just the
  automated test.
