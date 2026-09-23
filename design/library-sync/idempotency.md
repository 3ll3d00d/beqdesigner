# Library sync plan -- idempotency (extract cache, design cache, sync)

> Part of the library sync plan -- **start at the index**: [`../library-sync-pipeline-plan.md`](../library-sync-pipeline-plan.md).
> Contains §4. Section numbers are global across the plan; the index maps every `§` to its file.
> Status of what this file describes: **built** (see `status-and-open-items.md` §10 for the exceptions)

## 4. Idempotency

Two independent caches, one per stage -- extraction is expensive
(ffmpeg decode of a whole film) and design is comparatively cheap but
still shouldn't silently re-run and perturb an in-review queue entry
underneath a reviewer.

### 4.1 Extract cache

New `pipeline/library/extract_cache.py`.

- One manifest file per item, `<work_dir>/<item.id>/manifest.json`, with
  **flat, prefixed keys** (this replaces the single-record shape drafted
  earlier; Appendix D.4 has the exact shape): `mono_source_fingerprint`,
  `mono_params_hash`, `mono_extracted_at`, the same three with a
  `multichannel_` prefix, and a top-level `channel_layout_name`. That last
  key is read by chunk 2's `publish_reviewed_queue()` regeneration step (and
  by `run_library()` via `extract_cache.read_channel_layout_name()`) to
  correctly label a multichannel project's channels
  (`model.ffmpeg.get_channel_name()`). It is only written by the multichannel
  ("kept") extraction; absent (or the whole manifest missing) degrades to
  `'unknown'`, which `get_channel_name()` already handles sanely by channel
  count. There is no stored `wav_path` -- it is always
  `<target_dir>/{mono,multichannel}.wav`.
- `params_hash` = hash of everything that changes ffmpeg's output for
  this item (audio_stream, mono_mix, decimate/target_fs,
  playlist_name) -- a config change must invalidate the cache even if
  the source file hasn't changed.
- `source_fingerprint` = `item.fingerprint` if the source supplies one
  (JRiver's own modified-date/size are more trustworthy than a local
  stat, since the source is the library's system of record), else
  fall back to `(os.path.getmtime, os.path.getsize)` of `source_path`.
- `extract_if_needed(session, item, target_dir, config, mono_mix, force=False) ->
  (wav_path, cached: bool)`: compares the current
  `(source_fingerprint, params_hash)` against the manifest; matches
  and `wav_path` still exists on disk -> skip ffmpeg, return the
  cached path; otherwise runs `session.extract()` (existing, unchanged)
  and rewrites the manifest. `force=True` always re-extracts.
- Called **twice per item** whenever `LibraryRunConfig.keep_multichannel`
  is set **and the source isn't known to be mono** -- either extraction
  records the source's channel count as the manifest's flat
  `source_channel_count` key (`ExtractResult.channel_count`), and
  `run_library()` skips the kept pass when it is `1`; an unrecorded count
  still extracts and lets `load_channels()` decide (commit `1719151`). This
  only became reachable once mono-source extraction stopped failing
  (`dfcb7ff`: `Executor` built an invalid `pan=mono|c0=pan=mono|c0=c0`
  filter for a mono source) -- once with
  `mono_mix=True` into `<work_dir>/<item.id>/mono.wav` (design always
  needs this one -- `Session.design()`'s `mono_mix`) and once with
  `mono_mix=False` into `.../multichannel.wav` (the "kept" file, and
  the source for output 1's multichannel `.beq` project, §3.3) --
  mirrors `model/batch.py`'s existing "Mix to Mono?" toggle and its
  `ExtractCandidate.design()` docstring exactly ("a candidate whose kept
  file is multichannel gets a second, mono-only extraction made just
  for the design step"). Each call gets its own manifest entry (keyed
  by `mono_mix` as part of `params_hash`), so the two are independently
  cacheable -- redoing one doesn't force redoing the other.
- **Both extractions use the analysis sample rate.** Mono and kept
  multichannel outputs are decimated to `AnalysisConfig.target_fs`, as
  in Batch Extract / Design. This keeps channel diagnostics on the same
  sample grid as the mono design input. The kept multichannel WAV is an
  internal work artifact; multichannel `.beq` projects are built from
  analysis signals at this same rate.
- **Fixed output filenames, not ffmpeg's auto-derived ones.**
  `pipeline.review.publish_reviewed_queue()` (chunk 2, already shipped
  -- commit `407bd91`) hardcodes `<project_dir>/mono.wav` and
  `.../multichannel.wav` when reading `work_dir` -- so this chunk's
  extractions must land at exactly those paths, not whatever filename
  `Session.extract()` would otherwise derive from the source's own
  name. See Appendix D for the small, backward-compatible addition to
  `Session` this requires.

### 4.2 Design cache

`design_and_queue()` currently always overwrites
`<entry_id>.json`. Add a thin wrapper,
`pipeline/library/design_cache.py::design_if_needed()`:

- Skip re-designing when a queue entry for this id already exists
  *and* its recorded fingerprint (new, additive `QueueEntry.design_fingerprint:
  Optional[str] = None` field -- small, backward-compatible schema
  addition to `docs/schema/review_queue.schema.json`, existing entries
  without it just never match and get one designed on next touch)
  matches `hash(source_fingerprint, designer_name, AnalysisConfig,
  coverage)` for the current run.
- **Never** silently redesigns an entry whose `status` is `accepted`
  or `published` -- a rerun must not clobber a human's decision (or an
  already-published result) out from under them. `force=True` redesigns
  `pending`/`skipped`/`rejected` entries regardless of fingerprint match
  ("told to redo"); redoing an `accepted`/`published` entry requires
  the caller to explicitly reset its status first (a deliberate,
  single-item action, not something a library-wide `--force` flag does
  by accident).
- **Fingerprint scope (as built, commit `e68511f`):** the four inputs
  above plus `item.audio_stream`, `item.playlist_name`, and whether a
  multichannel extraction feeds the design -- the optional three are only
  hashed when set, so fingerprints recorded before they existed still match
  an unchanged default run. Metadata is **deliberately excluded**: a design
  does not depend on it, and a redesign would replace the entry a reviewer
  may have edited. Toggling `keep_multichannel` on a source that really is
  multichannel therefore redesigns a `pending` entry (and writes its
  multichannel project). A redesign keeps the entry's metadata, artwork and
  reviewer note (the new metadata only fills keys the entry lacks); only the
  candidates, fingerprint and status are replaced.
- Whenever this actually (re)designs (not on a skip), output 1's local
  `.beq` project file(s) get written for the top-pick candidate too.
  This lives in core `pipeline.review.design_and_queue()` itself, not
  in this wrapper (§3.3: it benefits `model/batch.py`'s existing manual
  flow equally, so it belongs in the shared, already-shipped module) --
  `design_and_queue()` gains optional `multichannel_wav_path`/
  `channel_layout_name`/`project_dir` parameters and calls
  `pipeline.publish.project.write_title_projects()` internally
  whenever `project_dir` is given (`None`, the default, skips it --
  backward compatible for every existing caller). `design_if_needed()`
  just threads `project_dir=os.path.join(work_dir, item.id)` through on
  a run/redesign, and passes nothing through on a skip -- a skip leaves
  existing project files untouched, same as it leaves the existing
  `QueueEntry` untouched.

### 4.3 Sync

Idempotency for the publish step itself is unchanged in shape --
`publish_reviewed_queue()` already is: only `accepted` entries publish,
each is marked `published` on success, so a rerun after a partial
failure (e.g. a git push failure partway through the queue) only
retries what's still `accepted`. **What changes (§3.3.1) is *which*
filter gets published**: no longer unconditionally
`entry.candidates[chosen_candidate_index].filters`, but the mono
project's *current* filter (`read_project_filter()`), falling back to
the old candidate-based path only when no project file is available.
This makes "publish" pick up a human's project-file edit even though
nothing about the queue entry's own status/fields changed -- a
re-publish is no longer purely a function of `QueueEntry`, it also
depends on live state in a second file. Still idempotent in the sense
that mattered before (accepted-but-already-published entries are
skipped either way); just worth being explicit that the *content*
published for a given entry can now change between two runs with an
identical queue directory, if the project file changed in between.
Project-file regeneration itself (the hash-gated overwrite in §3.3.1)
is idempotent for the same reason it always was: a pipeline-pure
project regenerates identically every time; a human-edited one is
never touched at all.

This plan adds `pipeline/library/sync.py::sync_library()` purely as a
same-shaped verb alongside `run_library()` for CLI/GUI symmetry -- it
is a documented, near-literal call-through to
`pipeline.review.publish_reviewed_queue()`, no new logic.
