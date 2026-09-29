# Designer requests by reference — design, not built

**Status:** design, not built. Answers beqforge's R2 ("a shared-filesystem mode
with beqdesigner", beqforge `IMPROVEMENT_PLAN.md`, `3d5db14`), which asks for
part of the change here, since [`designer-interface.md`](designer-interface.md)
is this repo's contract. Tracked as [D1](outstanding.md#d1--designer-requests-by-reference).
Nothing here changes the contract until a chunk below is built and
`designer-interface.md` is revised with it.

## 1. What R2 asks, and what it already has

R2 wants three things: (a) a request that names audio on a shared filesystem
instead of carrying it; (b) one cache both sides can use; (c) a redesign of a
title already analysed that skips extraction and analysis.

Checked against both repos at `19dbae1` / beqforge `3d5db14`:

- **Extraction is already cached on this side.** `pipeline/library/extract_cache.py`
  keeps `<work_dir>/<id>/mono.wav` and `multichannel.wav` (PCM s24, already at
  `target_fs`) under `manifest.json`. A redesign here never runs ffmpeg again
  unless the source or the extraction settings changed.
- **The slow part of a redesign is on beqforge's side.** Its server builds a
  `Material` from the request and runs `pipeline.run` with no stage cache, so
  `diagnose`/`extract`/`identify` (21-100 s) repeat on every call.
- **beqforge's stage cache is already keyed by content, not by path.**
  `cache.material_fingerprint` hashes the samples (0.25 s for 488 MB). An
  inline request carries the same samples every time, so it would hit that
  cache as well as a request by path would.

## 2. Feedback on R2: split it

**R2a — the server uses its stage cache (beqforge only, no contract change).**
This alone gives (c). It needs R2's own fixes (b) (atomic write-then-rename,
entries never changed in place) and (c) (per-stage digests baked into the
frozen build), plus C3 (the parametric key covers too little). Ship this
first, then measure what a warm inline request still costs.

**R2b — requests by reference (contract 1.2, this doc).** This saves only the
wire cost: base64 encoding, JSON and transfer, and decoding on the server. For
an 8-channel, 2-hour title at 1 kHz that is about 520 MB of float64, or about
690 MB of base64 in one POST. On localhost that may be a few seconds, which is
small next to a cold analysis. The caller still loads the arrays either way
(it builds projects from them), so its memory does not shrink. Build R2b only
if R2a's timings show the transfer matters, or once the designer runs on
another host and the wire becomes the bottleneck.

**Two points beqforge needs if R2b goes ahead:**

1. `material_fingerprint` hashes `material.name`. The server uses the fixed
   `"designer-request"`, but `material.load()` names a `.npz` after its file
   stem. A by-path loader that followed that pattern would give a request by
   path and the same request inline different keys. That would break R2's own
   rule ("a request by path and one inline produce identical records") and
   its rule that a renamed file still hits. Keep the name fixed for designer
   requests, or leave it out of the key.
2. There should be no shared **stage** cache across the boundary.
   beqdesigner never reads beqforge's stage cache, and beqforge should treat
   `manifest.json` as private to beqdesigner. What both sides share is the
   audio, named in each request (§3). For the CLI's `tools/extract.py`
   duplication, see §5.

## 3. Wire proposal (contract 1.2, HTTP binding only)

The data model does not change: `DesignRequest.mono_mix` and `channels` stay
arrays (§2 of the interface). What changes is §7.1's encoding, which gains a
second way to send an array. Under §7's own rule, this is another wire form
for the same fields. The in-process binding ignores it.

```json
"mono_mix": {"dtype": "float64", "shape": [N],
             "file": {"path": "t_1234/mono.wav", "channel": 0},
             "sha256": "<hex of the array's little-endian float64 bytes>"},
"channels": {"LFE": {"dtype": "float64", "shape": [N],
                     "file": {"path": "t_1234/multichannel.wav", "channel": 3},
                     "sha256": "..."}, "...": "..."}
```

The rules:

- **One of `data_base64` or `file` per array, never both.** A request may mix
  them.
- **`path` is relative to a shared root** that each side configures for
  itself: the caller's `work_dir`, and wherever the designer mounts it. It is
  POSIX-style, cannot be absolute, and may not contain `..`. The designer
  refuses a path that resolves outside its root. Absolute paths would break
  between hosts and containers (the same problem `pathmap.py` solves for
  JRiver). A path outside the root would let anyone who can POST to the
  designer read files through it.
- **The file format is WAV**, and the designer decodes it as
  `soundfile.read(path, dtype='float64', always_2d=True)[:, channel]`. That is
  exactly what `model.signal.read_wav_data` gives the caller, and `resample`
  does nothing because the file is already at `fs`. The file's sample rate
  must equal the request's `fs`, and `shape` its frame count.
- **`sha256` is required with `file`.** It is the digest of the decoded
  float64 little-endian bytes, which the caller computes from the array it
  would otherwise have sent inline. The designer decodes, hashes and
  compares. This is what makes a request by path identical to one sent
  inline. It also catches a file read mid-write: ffmpeg writes
  `multichannel.wav` in place, not by write-then-rename. And it hands
  beqforge its cache key without hashing again.
- **Channel labels stay as `channels`' keys**, set by the caller as today
  (`get_channel_name` over the recorded layout). The designer never works
  them out from the file.
- **Errors**: the designer answers **422** if it cannot resolve a path, finds
  a mismatch in format, rate or shape, or gets a different digest. The body
  says which. That is an implementation failure (§7.1), never a decline.
- **Versioning.** Request and response carry `"1.2"`. A 1.0/1.1 designer does
  not know `file` and fails on the missing `data_base64`, so the caller sends
  by reference **only to a designer configured for it** (§4). The change is
  additive for the response and opt-in for the request.

## 4. Caller implementation plan

The steps are in order, and each is its own commit with its tests (AGENTS.md).

**D1.1 — contract text and schema.** Revise `designer-interface.md` to 1.2
(§7.1, §8's size note) and add the array's `oneOf` to
`docs/schema/http_designer_request.schema.json`. In
`designer-conformance-tests.md`, add rows for: a mixed request, a bad digest,
a path escaping the root, an absolute path, a rate or shape mismatch, and
422 not being treated as a decline. Tests: the request schema accepts both
forms and rejects `file` and `data_base64` together, a missing `sha256`, and
an absolute path.

**D1.2 — the binding.** Change `http_designer(url, ..., shared_root=None)`.
When `shared_root` is set and the caller supplies where each array came from,
encode it as `file` + `sha256`; otherwise send it inline as today. How the
paths reach the binding without changing `DesignRequest`: `Session.design`
gains an optional `sources: AudioSources` (mono path; multichannel path and
the label-to-column map), and passes it on only to designers registered as
by-reference (`pipeline/designer/registry.py` records the capability).
In-process designers are called as today. Tests: a path under the root is
sent relative and one outside it goes inline; the digest equals the digest
of what would have gone inline; a 422 raises `HttpDesignerError`.

**D1.3 — sources from the library and batch paths.** `pipeline.review.design_and_queue`
and the library `stages` already hold the wav paths they loaded. Thread them
through as `AudioSources`. Batch Extract / Design goes through the same call,
which keeps the extraction/design parity contract (AGENTS.md). Test (under
the parity rule): a short synthetic multichannel source goes through the
library path to a fake by-reference designer, which decodes each `file` as
§3 says; its arrays are byte-identical to the inline encoding of the same
request, with the same `fs` and frame count for mono and every channel.

**D1.4 — configuration.** The profile's `designers:` entry gains
`by_reference: true` (the service's `BEQ_DESIGNER_*` environment variables and
the work-list Settings follow `register_declared_designers`). The shared root
is always the run's `work_dir`, so there is no second path to get wrong.
Tests: parsing the profile, and that a designer without the flag never gets
`file`.

**Fallback, decided:** none. A 422 is a configuration error (wrong mount,
stale file) and should show as a failed design with its message. Retrying
inline would hide a mount that is always wrong, and would quietly bring back
the cost this change removes.

## 5. Optional follow-ons (not part of D1)

- **A request file beside the audio.** With D1 built, the caller can write
  the by-reference request body to `<work_dir>/<id>/design-request.json` at
  almost no cost. beqforge's `tools/replay.py` and CLI could run from it
  instead of from their own `tools/extract.py` output. That gives R2's "the
  CLI shares the extraction" without making `manifest.json` or the folder
  layout a contract.
- **Records beside the title.** beqforge's R1 `record_dir` could point into
  the same title folder, so a queue entry and the run record behind it sit
  together.
- **Designer-side drift.** `design_fingerprint` does not cover the designer's
  build or its start-up parameters, so the "settings changed since N titles
  were designed" banner cannot see a new beqforge build. The response already
  names its build (`beqforge_revision` in commentary). Surfacing a revision
  that changed would be a separate caller item, and it matters more once a
  redesign is cheap.
