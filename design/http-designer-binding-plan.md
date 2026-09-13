# Implementation plan — HTTP designer binding

**Status:** all four phases done. Answers "how does a random user plug
their own filter-design implementation into beqd" — today the answer is
"they can't": `pipeline.designer.registry.register_designer()` is in-process
Python only, beqd ships as a frozen PyInstaller binary with no Python
exposed to end users, and the GUI has no action that even calls
`pipeline.review.batch_design()` in the first place. `designer-interface.md`
§7 already names the intended fix — *"nothing here stops a subprocess or
HTTP binding later that serialises the same fields as JSON... only how they
cross a process boundary does [change]"* — this plan builds the HTTP half of
that, chosen over a subprocess binding for deployment flexibility (a
designer can be local, on another machine, written in any language, or
shared across a team, without beqd needing to spawn or manage it).

Four phases, each its own commit, matching `candidate-review-plan.md`'s
convention:

1. **Binding** (Qt-free) — the wire format + an `http_designer(url)` factory
   that produces a normal `DesignerCallable`. This alone already answers the
   original question for a technical user: stand up any HTTP server
   speaking the wire format, `register_designer('name', http_designer(url))`
   in a script, done.
2. **Docs** — formalise §7 (no longer "not yet specified") + a published
   JSON Schema pair for the wire format, since the whole point of this
   binding is language-agnostic interop, unlike the Python-dataclass-shaped
   contract the rest of `designer-interface.md` documents.
3. **GUI: Designers settings** — a durable, named list of HTTP designer
   endpoints a non-technical user configures once (Tools → Designers),
   registered automatically on every app startup.
4. **GUI: Run Batch Design** — the dialog that's been missing this whole
   time: pick source files, pick a configured designer, pick a queue
   directory, run `batch_design()` with progress — the step that currently
   only exists as "write a Python script."

No phase after the first depends on GUI work existing yet; 1-2 are Qt-free.

---

## Phase 1 — the HTTP binding

**Ships:** `pipeline/designer/http_binding.py` (Qt-free), tests.

**Wire format.** One POST per `design()` call. Request body:

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

`channels`/`bass_management` are `null` when the `DesignRequest` field is
`None`. **Arrays are base64-encoded raw little-endian float64 bytes, not
JSON number arrays** — a 2-hour `mono_mix` at the common 1 kHz decimated
rate is ~58 MB as float64 (`designer-interface.md` §8's own figure); JSON's
text encoding of that many floats bloats it 3-5x for no benefit on a
same-machine/LAN call, where decode simplicity matters more than being
human-readable. `dtype`/`shape` travel alongside so the decoder doesn't have
to guess; `dtype` is always `"float64"` for v1 (matching `DesignRequest`'s
existing float64 mandate) but named explicitly rather than assumed, so it
isn't a silent breaking change if that ever needs to widen.

Response body: `DesignResponse` as JSON, field-for-field —
`{"contract_version": "1.0", "candidates": [...]}` with each candidate's
`filters` as `[{"type": "low_shelf", "freq_hz": ..., "gain_db": ..., "q": ...}, ...]`,
or `{"contract_version": "1.0", "decline_reason": "...", "decline_message": "..."}`.
No arrays appear in a response — `BiquadSpec` carries no sample-rate-bound
data (§5: "No sample rate on a `BiquadSpec`, by design") — so the response
side needs no binary encoding at all, only the request side does.

**Error handling, per §1's existing rule** ("the caller treats an exception
as an implementation failure, not a signal"): a non-2xx status, a connection
failure/timeout, or a response body that doesn't parse into a `DesignResponse`
all raise `HttpDesignerError` (new, `RuntimeError` subclass) rather than
being coerced into a decline. A *well-formed* `DesignResponse` that fails
`§3-§5` validation still goes through the existing
`pipeline.designer.convert.validate_response` path unchanged — this binding
only has to produce a `DesignResponse`, not validate it; that boundary
doesn't move.

```python
def http_designer(url: str, timeout: float = 300.0, headers: dict | None = None) -> DesignerCallable:
    '''
    :return: a callable matching DesignerCallable (Callable[[DesignRequest], DesignResponse]) --
        register it the same way as any in-process one: register_designer(name, http_designer(url)).
    '''
```

`headers` is there for a bearer token or similar against a
non-localhost/shared designer — no larger auth framework, just pass-through.

**Tests:** one real round trip against a minimal `http.server` spun up in a
background thread (proves the actual JSON+base64 encode/decode works, not
just mocked); `requests`-level failure modes (connection refused, timeout,
non-2xx, malformed JSON body, a body that parses but doesn't match
`DesignResponse`'s shape) each raise `HttpDesignerError`; a decline response
round-trips; a multi-candidate response with `commentary`/`gain_reduction_db`/
`bass_management`-shaped request all round-trip losslessly (array values
included, via `np.allclose` after decode).

---

## Phase 2 — docs

**Ships:** `designer-interface.md` §7 rewritten with the wire format above
(no longer "not yet specified, because v1 doesn't need it" — it's needed
now), plus `docs/schema/http_designer_request.schema.json` and
`http_designer_response.schema.json` under the same published-spec
treatment as the save formats — unlike the rest of the contract (which only
this repo's Python needs to match), the HTTP wire format is explicitly for
implementers in any language, which is exactly what JSON Schema is for.

---

## Phase 3 — GUI: Designers settings

**Ships:** `model/designers.py` (a small dialog + persistence + a
`register_configured_designers(preferences)` helper), a new preference key,
a `Tools → Designers…` menu action, and a call to
`register_configured_designers` during app startup (before the main window
is usable) so every configured endpoint is registered before anything could
call `batch_design()`.

A durable list of `{name, url, headers}` entries. **Revised while
implementing:** stored as a plain `list[dict]` preference value (new
`DESIGNER_HTTP_ENDPOINTS` key, `TYPES[...] = list`), the same pattern
`JRIVER_MCWS_CONNECTIONS` already uses for a dict-shaped preference in this
codebase — `QSettings`'s own variant handling round-trips it fine, so no
manual `json.dumps`/`json.loads` of the whole list was needed after all
(each row's `headers` cell is still edited as JSON text in the table, since
that's a free-form object a plain table cell can't structure any other
way). A simple table (add/remove row, name/url/headers-as-JSON-text
columns), each registered under a `http:`-prefixed name so re-registering
(on save, or at every app startup) cleanly replaces rather than
accumulates. Save validates (name+URL required, headers must parse as a
JSON object, names unique) before writing/re-registering — the dialog
stays open with a message box on a validation problem, matching this
repo's existing modal-validation convention rather than closing and
silently discarding.

**Tests (`gui/test_designers_dialog.py`, done — 8 tests):** starts empty;
loads existing configured entries; add+save persists to the preference and
registers under the `http:` prefix; headers JSON round-trips; invalid
headers JSON and duplicate names each block the save (preference
untouched, nothing (re)registered); removing a row; saving a changed list
unregisters the stale entries and registers the new ones.

---

## Phase 4 — GUI: Run Batch Design

**Ships:** `model/batch_design.py` + `ui/batch_design.py`/`.ui`, a
`Tools → Batch Design…` menu action.

**Ships in `pipeline/review.py` too:** `batch_design()` gains an optional
`on_item_done: Callable[[str], None] | None = None` parameter, invoked
after each item's queue entry is written — plain callable, still Qt-free;
the GUI wraps it to emit a Qt signal for a progress bar. No other change to
that function.

The dialog: a source-file picker (multi-select, reusing the simplest of
`BatchExtractDialog`'s patterns rather than its full glob-search machinery —
scope note below), a designer combobox populated from
`pipeline.designer.registry.registered_designers()` (so it only ever offers
what Phase 3 already configured, or whatever a technical user registered by
hand), queue-dir/work-dir pickers, and a Run button that drives
`batch_design()` on a background thread (`QThreadPool`, the same pattern as
`model/batch.py`/`model/ffmpeg.py`/Phase 3's own review-dialog publish
action) with a progress bar driven by `on_item_done` and a completion
summary. On completion, offers to open the queue directly in
`ReviewQueueDialog` (Phase 3 of `candidate-review-plan.md`) — closing the
loop from "ran a batch" to "review it" in one flow.

**Scope note:** file selection is a plain multi-select `QFileDialog`, not
`BatchExtractDialog`'s recursive glob-search-with-per-file-probe UI — that
dialog solves a different problem (picking audio streams out of arbitrary
containers before extraction); here, each selected file becomes one
`batch_design()` item with an id derived from its filename, and per-title
metadata is left to be resolved later at review time (already supported --
`candidate-review-plan.md` Phase 1's `meta=None` case) rather than adding a
metadata-entry UI to this dialog too. Built as sketched, with one added
guard: duplicate filename stems across selected files are rejected up
front (queue entry ids collide otherwise) rather than silently
overwriting one entry with another.

**Tests (`gui/test_batch_design_dialog.py`, done — 7 tests, real
`QThreadPool` job, no mocking it, same pattern as
`test_ffmpeg_execute.py`):** the designer combo lists what's registered;
running with no files, or without both directories, is rejected;
duplicate filename stems are rejected; removing a selected file; a
successful run against a real synthetic wav + a fake registered designer
writes exactly one pending queue entry with the right id and candidate
data; the progress bar's range matches the item count and reaches it on
completion.
