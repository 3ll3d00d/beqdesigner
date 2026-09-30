# Designer requests by reference

**Status:** built on this repo's side (D1.1-D1.4, 2026-09-30), unparked
when beqforge started on its side of R2b (its R2a is done, `7951cc0`). A
designer declared `by_reference: true` in a profile is sent the extracted
WAVs by path; every other designer is sent every array inline, as before.
Agreed with beqforge (§6, their `e96129a`). Answers beqforge's R2 ("a
shared-filesystem mode with beqdesigner", beqforge `IMPROVEMENT_PLAN.md`,
`3d5db14`), which asks for
part of the change here, since [`designer-interface.md`](designer-interface.md)
is this repo's contract. Recorded as built in [`implemented.md`](implemented.md#design-and-review) (formerly outstanding item D1).
The contract itself is `designer-interface.md` v1.2, §7.1 "Arrays by
reference". Where this document and the contract differ, the contract wins.

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
frozen build). C3, a correct parametric key, is also a prerequisite, and
beqforge has already built it (`a2db373`). Ship this first, then measure what
a warm inline request still costs.

**R2b — requests by reference (contract 1.2, this doc).** This saves only the
wire cost: base64 encoding, JSON and transfer, and decoding on the server. For
an 8-channel, 2-hour title at 1 kHz that is about 520 MB of float64, or about
690 MB of base64 in one POST. On localhost that may be a few seconds, which is
small next to a cold analysis. The caller still loads the arrays either way
(it builds projects from them), so its memory does not shrink. Build R2b only
if R2a's timings show the transfer matters, or once the designer runs on
another host and the wire becomes the bottleneck.

**Measured, 2026-09-29 (the caller's half of R2b's evidence).** *One Battle
After Another*: 2.70 h at 1 kHz, mono plus 8 channels, 931.6 MB of JSON
body. This is the largest title in the local library. The caller used
beqdesigner's real `http_binding` and posted to localhost. The server was a
stub that runs beqforge's own `request_from_json` and then declines. Times
are three runs, which agreed within 0.2 s:

| Step | Inline (1.1) | By reference (§3) |
|---|---|---|
| Caller: base64 of every array | 1.42 s | -- |
| Caller: `json.dumps` | 2.8 s | -- |
| Caller: SHA-256 of every array | -- | 0.68 s |
| Server: read the body | 0.45 s | -- |
| Server: `json.loads` | 1.47 s | -- |
| Server: base64 decode and checks | 2.2 s | -- |
| Server: read both WAVs (`soundfile`) | -- | 0.54-0.65 s |
| Server: SHA-256 check | -- | 0.87 s |
| **Total per request** | **8.3-8.5 s** | **about 2.2 s** |

Loading the WAVs on the caller took 0.9 s. Both forms pay that, so it is
left out of the table. A request by reference would save about 6 s per
request on this title, and less on shorter ones. Against a cold design
(40-80 s) that is 8-15%, which is not worth a contract change. Against a
warm R2a request it could be most of the time left. So R2a's warm timings
still decide, and this table is the other half of that comparison. Peak
memory was not measured. The inline server holds a 0.9 GB body and its
parsed strings before the arrays exist.

beqforge accepted the split: R2a is its priority 9, and R2b is parked
behind R2a's timings (§6 Q1).

**Two points beqforge needed if R2b goes ahead (both agreed, §6):**

1. `material_fingerprint` hashed `material.name`, so a by-path loader that
   named material after its file would have keyed a request by path apart
   from the same request inline. beqforge takes the name out of the key in
   R2a, and the by-path loader also keeps `name="designer-request"` (Q3).
2. There is no shared **stage** cache across the boundary. beqdesigner never
   reads beqforge's stage cache, and beqforge never reads `manifest.json` or
   relies on the `work_dir` layout. What both sides share is the audio,
   named in each request (§3). R2's "one cache both sides use" is withdrawn
   (Q2).

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

The rules (beqforge's amendments from §6 Q5 are folded in):

- **One of `data_base64` or `file` per array, never both.** A request may mix
  them.
- **`path` is relative to a shared root** that each side configures for
  itself: the caller's `work_dir`, and wherever the designer mounts it
  (beqforge: `--shared-root`). It is POSIX-style, cannot be absolute, and may
  not contain `..`. The designer refuses a path whose **real path** (after
  resolving symlinks) lies outside its root's real path, so a symlink under
  the root that points out of it is refused too. Absolute paths would break
  between hosts and containers (the same problem `pathmap.py` solves for
  JRiver). A path outside the root would let anyone who can POST to the
  designer read files through it.
- **The file format is WAV** (PCM s16/s24/s32 or float). The decoded array
  is defined by its arithmetic, not by a library: an integer sample `s` of
  `b` bits becomes `s / 2**(b-1)` as float64, and a float sample is widened
  to float64. That is exactly what `model.signal.read_wav_data` gives the
  caller (`soundfile.read(path, dtype='float64', always_2d=True)[:, channel]`),
  and `resample` does nothing because the file is already at `fs`. beqforge
  has no WAV reader today (it has no `soundfile`, and its CLI reads `.npz`).
  `scipy.io.wavfile` returns s24 left-justified in int32, and dividing that by
  `2**31` gives the same values exactly, because the scaling is a power of
  two. The per-array digest catches any decoder that differs. `channel` must be
  `0 <= channel < ` the file's channel count. The file's sample rate must
  equal the request's `fs` (a file at any other rate is refused, **never
  resampled**), and `shape` must be `[frames]`.
- **`sha256` is required with `file`.** It is the hex SHA-256 of exactly
  these bytes: the selected column after decoding, as a C-contiguous
  little-endian float64 1-D array of `shape[0]` elements (in numpy,
  `np.ascontiguousarray(col, dtype="<f8").tobytes()`). The caller computes it
  from the array it would otherwise have sent inline. The designer decodes,
  hashes and compares **before using the array**. This is what makes a
  request by path identical to one sent inline. It also catches a file read
  mid-write: ffmpeg writes `multichannel.wav` in place, not by
  write-then-rename. It is a check only, not the designer's cache key:
  beqforge keeps hashing the samples in one pass (Q4), so the contract does
  not pin its cache format.
- **Channel labels stay as `channels`' keys**, set by the caller as today
  (`get_channel_name` over the recorded layout). The designer never works
  them out from the file.
- **Errors.** **400** for a body that does not parse or fails the schema, as
  today. **422** for a well-formed `file` the designer cannot honour: a bad
  or escaping path, a file that is not a readable WAV, a channel out of
  range, a rate or shape mismatch, a different digest, or **no shared root
  configured on this server**. The JSON body names the array (`mono_mix` or
  the channel label) and the reason. Both are implementation failures
  (§7.1), never a decline.
- **Capability check.** `GET /health` on the designer's origin, which today
  is a liveness check outside the contract, becomes part of 1.2. It answers
  `{"contract_version": "1.2", "shared_root": true|false}`. A caller with a
  designer registered as by-reference checks it at registration, so a
  mismatched server shows up there and not on its first title.
- **Versioning.** Request and response carry `"1.2"`. A 1.0/1.1 designer does
  not know `file` and fails on the missing `data_base64`, so the caller sends
  by reference **only to a designer configured for it** (§4). The change is
  additive for the response and opt-in for the request.

## 4. Caller implementation plan

The steps are in order, and each is its own commit with its tests (AGENTS.md).
None starts until R2b is unparked (Status).

**D1.1 — contract text and schema. Done.** `designer-interface.md` is now
v1.2. Its §7.1 gains "Arrays by reference", and the caller sends `"1.2"`
only when a body holds a `file`, so inline requests are unchanged. The
request schema's array is a `oneOf` of inline and `file`. The new
`http_designer_health.schema.json` covers `/health`. Conformance rows are
7.11-7.22. Tests are in `test_designer_http_schemas.py`. As first planned: Revise `designer-interface.md` to 1.2
(§7.1, §8's size note, `GET /health`) and add the array's `oneOf` to
`docs/schema/http_designer_request.schema.json`, plus a small
`http_designer_health.schema.json`. In `designer-conformance-tests.md`, add
rows for: a mixed request; a bad digest; a path escaping the root, lexically
and through a symlink; an absolute path; a channel out of range; a rate or
shape mismatch; a `file` sent to a server with no shared root; 400 against
422; the 422 body naming the array; `/health`'s answer; and 422 not being
treated as a decline. Tests: the request schema accepts both forms and
rejects `file` and `data_base64` together, a missing `sha256`, and an
absolute path; the health schema.

**D1.2 — the binding. Done.** Where the code differs from the plan below:

- A by-reference binding checks `/health` once, just before its first request
  by reference, not at registration. The work list registers designers
  whenever it reads the profile, on the GUI thread, and a network call there
  would block. A mismatched server therefore shows up as a failed first
  design whose message quotes the `/health` answer, and nothing is sent to it.
- An array also goes inline when its WAV's header shows it cannot be the
  array unchanged: another rate, another length, a missing column, or a
  format other than WAV. This covers a resampled or trimmed load, and makes a
  422 mean a file that changed or a wrong mount.
- The registry flag is `register_designer(..., takes_sources=True)`, and
  sources live in `pipeline/designer/sources.py`.
- Tests are in `test_pipeline_designer_by_reference.py`, against a fake
  by-reference designer on a real socket.

As first planned: change `http_designer(url, ..., shared_root=None)`.
When `shared_root` is set and the caller supplies where each array came from,
encode it as `file` + `sha256`; otherwise send it inline as today. How the
paths reach the binding without changing `DesignRequest`: `Session.design`
gains an optional `sources: AudioSources` (mono path; multichannel path and
the label-to-column map), and passes it on only to designers registered as
by-reference (`pipeline/designer/registry.py` records the capability).
In-process designers are called as today. Tests: a path under the root is
sent relative and one outside it goes inline; the digest equals SHA-256 of
the exact bytes §3 names, and of what would have gone inline; a 400 or 422
raises `HttpDesignerError` carrying the body's array and reason.

**D1.3 — sources from the library and batch paths. Done.**
`design_and_queue` works out the sources itself
(`pipeline.designer.sources.sources_for`): the mono WAV it loads, and the
multichannel WAV that `channels` was split from. Library runs already pass
that path, and batch's `DesignJob` now passes it too (it writes no project
from it, having no `project_dir`). The channels get no sources unless there
is one label for every column of the file, so a label can never name
another column's samples. Found while building this: ffmpeg writes
`WAVE_FORMAT_EXTENSIBLE` (libsndfile calls it `WAVEX`) for more than two
channels. The binding accepts it, and the contract and conformance row 7.19
now name it, since a designer's decoder must read it too. Tests: the library
parity test in `test_pipeline_designer_by_reference.py` (real ffmpeg; every
array goes by reference and decodes byte-identical to the request, at one
rate and frame count), and
`gui/test_batch_extract_design.py::test_design_names_the_wavs_it_loaded_for_a_designer_that_takes_arrays_by_reference`.
As first planned: `pipeline.review.design_and_queue`
and the library `stages` already hold the wav paths they loaded. Thread them
through as `AudioSources`. Batch Extract / Design goes through the same call,
which keeps the extraction/design parity contract (AGENTS.md). Test (under
the parity rule): a short synthetic multichannel source goes through the
library path to a fake by-reference designer, which decodes each `file` as
§3 says; its arrays are byte-identical to the inline encoding of the same
request, with the same `fs` and frame count for mono and every channel.

**D1.4 — configuration. Done.** A `designers:` entry takes
`by_reference: true`, and the run's `work_dir` is its shared root:
`register_declared_designers(..., shared_root=)` from `run_config_from_values`
(the CLI and the service) and from `register_profile_designers` (the work
list). A `by_reference` entry with no work directory, or a non-boolean
value, is refused with a message. The service's
`BEQ_DESIGNER_HEADERS_<NAME>` keeps the flag. GUI Preferences' designers stay
inline. User docs: `docs/library/unattended.md` and the pipeline README.
Tests are in `test_pipeline_designer_by_reference.py` (parsing, a plain
designer never gets `file`, the run config, the service) and
`gui/test_worklist_profile.py`. As first planned: the profile's `designers:` entry gains
`by_reference: true` (the service's `BEQ_DESIGNER_*` environment variables and
the work-list Settings follow `register_declared_designers`). The shared root
is always the run's `work_dir`, so there is no second path to get wrong.
Tests: parsing the profile, and that a designer without the flag never gets
`file`. The `/health` check is in the binding (D1.2).

**Fallback, decided (agreed, §6 Q5):** none. A 422 is a configuration error
(wrong mount, stale file) and should show as a failed design with its message. Retrying
inline would hide a mount that is always wrong, and would quietly bring back
the cost this change removes.

## 5. Optional follow-ons (not part of D1; beqforge said yes to both, after R2b)

- **A request file beside the audio.** With D1 built, the caller can write
  the by-reference request body to `<work_dir>/<id>/design-request.json` at
  almost no cost. beqforge would read it with a new loader for
  `design_beq.py`, next to its `.npz` loader. `replay.py` works from a record
  and needs no audio. The CLI then stops extracting again a title
  beqdesigner already has. `tools/extract.py` stays for use without
  beqdesigner. `manifest.json` and the folder layout do not become a
  contract.
- **Records beside the title.** beqforge would rather not learn the
  `work_dir` layout, so the request gains an optional `record_path`,
  relative to the shared root and chosen by the caller. The server writes
  its run record there, and otherwise falls back to its `--record-dir`.
  This is a contract addition of its own, designed when it is wanted.
- **Designer-side drift.** `design_fingerprint` does not cover the designer's
  build or its start-up parameters, so the "settings changed since N titles
  were designed" banner cannot see a new beqforge build. The response already
  names its build (`beqforge_revision` in commentary). Surfacing a revision
  that changed would be a separate caller item, and it matters more once a
  redesign is cheap. beqforge agrees this is the caller's to do.

## 6. Answers from beqforge

Answered in beqforge's `IMPROVEMENT_PLAN.md`, Progress, "R2 split" (`e96129a`,
2026-09-29). Its rows are R2a (priority 9, open) and R2b (parked).

- **Q1, the split: accepted.** R2a removes the redesign cost with no contract
  change. What it leaves unsolved is only the wire, which is R2b's whole
  case, and R2a's warm timings settle it (it reports transfer and decode
  separately).
- **Q2, no shared stage cache: agreed.** Only the audio is shared, and R2's
  shared cache is withdrawn. Atomic writes stay in R2a, because beqforge's
  server and CLI may share a cache directory.
- **Q3, the name: taken out of the key** in R2a. The by-path loader also
  keeps `name="designer-request"`, so records stay as they are today.
- **Q4, digests as the key: no.** beqforge keeps hashing the samples in one
  pass, which costs about 0.25 s against a 21-100 s hit. The per-array
  `sha256` stays a check (§3).
- **Q5, the wire rules: accepted, with amendments**, all now in §3-§4: the
  exact digested bytes; containment checked with `realpath`; channel range;
  a rate mismatch refused and never resampled; 400 against 422, with "no
  shared root" as a 422 and the body naming the array; `GET /health`
  advertising the version and shared root.
- **Q6, the follow-ons: yes to both, optional, after R2b.** Adjusted in §5:
  `design_beq.py`, not `replay.py`, and `record_path` in place of pointing
  `record_dir` at the layout.
