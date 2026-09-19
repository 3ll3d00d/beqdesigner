# Library sync plan -- archived handoff spec

> Part of the library sync plan -- **start at the index**: [`../../library-sync-pipeline-plan.md`](../../library-sync-pipeline-plan.md).
> Contains Appendix D (chunk 5). Section numbers are global across the plan; the index maps every `§` to its file.
> Status of what this file describes: **built and shipped** -- kept as the record of the original spec; read only if you need the detail

## Appendix D -- Chunk 5 handoff spec: idempotent extract cache

Self-contained enough to implement without reading the rest of this
document, though §4.1 above has the same design plus the rationale.
Depends only on chunk 4 (Appendix C, already shipped -- commit
`7d0bac5`) for `LibraryItem`. Entirely within `pipeline/` (Qt-free).

### D.1 Why

`Session.extract()` always shells out to ffmpeg -- nothing skips
anything today. At library scale (hundreds of titles, run repeatedly
as new titles get added), re-extracting everything on every run is
wasteful and slow. This chunk adds a manifest-backed idempotency layer
in front of it: skip ffmpeg when the source hasn't changed (per
`LibraryItem.fingerprint`, or a local mtime/size fallback) and neither
the analysis config nor the extraction mode (mono vs. multichannel)
has changed since the last recorded run.

**Cross-chunk constraint discovered while writing this spec**: chunk 2
(`pipeline.review.publish_reviewed_queue()`, already shipped, commit
`407bd91`) hardcodes `<work_dir>/<entry.id>/mono.wav`,
`.../multichannel.wav`, and reads `.../manifest.json`'s flat
`channel_layout_name` key. This chunk's on-disk output **must** match
that exactly -- it was designed against this constraint, not
independently, so implementing it as literally specified below keeps
both chunks consistent. If you find any mismatch against the *actual*
current `pipeline/review.py` (not just this document), the shipped
code wins and this spec needs correcting, not the other way around --
confirm by reading `pipeline/review.py`'s `publish_reviewed_queue()`
and `_read_channel_layout_name()` before writing any code.

### D.2 Scope

**In scope:**
- A new, backward-compatible `Session.extract_with_layout()` in
  `pipeline/orchestrate.py`, which also lets a caller fix the output
  filename. `Session.extract()` itself becomes a thin wrapper over it
  (same signature, same return type, behaviourally identical) so every
  existing caller is unaffected.
- New `pipeline/library/extract_cache.py` -- `extract_if_needed()`
  plus its manifest read/write helpers.

**Explicitly out of scope:**
- `run_library()` (chunk 7) -- deciding *whether* to call
  `extract_if_needed()` once or twice per item (the
  `LibraryRunConfig.keep_multichannel` toggle) is that chunk's job;
  this chunk's `extract_if_needed()` only ever handles one `mono_mix`
  value per call, agnostic to how many times or in what order it's
  called for a given item.
- Anything about design (`design_if_needed()` is chunk 6).

### D.3 `Session` change (`pipeline/orchestrate.py`)

```python
@dataclass(frozen=True)
class ExtractResult:
    wav_path: str
    channel_layout_name: str  # model.ffmpeg's CHANNEL_LAYOUTS key, e.g. '5.1', or 'unknown'/a generic
                              # "<n> channels" string when ffmpeg's probe couldn't name it more precisely
    channel_count: int = 0    # added with fix 7 -- the source stream's channel count; 0 if unknown
```

Add to `Session`:

```python
def extract_with_layout(self, src: str, target_dir: str, audio_stream: int = 0, video_stream: int = -1,
                        mono_mix: bool = True, decimate: bool = True, playlist_name: Optional[str] = None,
                        output_file_name: Optional[str] = None) -> ExtractResult:
    '''
    Same as extract(), but (a) lets a caller fix the output filename instead of ffmpeg's auto-derived one
    (output_file_name is given *without* an extension -- Executor appends the format's own extension,
    '.wav' by default) and (b) also returns the source's detected channel layout name
    (Executor.channel_layout_name, the same one model/batch.py's ExtractCandidate.design() already reads
    off its own Executor) -- for a caller (the extract cache, §4.1) that needs to record it without a
    second, separate probe. extract() itself keeps returning a bare path unchanged -- every other existing
    caller has no use for the layout and a changed return type would break them.
    '''
    os.makedirs(target_dir, exist_ok=True)
    display_name = None
    duration_override_s = None
    if is_bdmv_root(src):
        resolved = resolve_main_title(src, playlist_name=playlist_name)
        src = resolved.ffmpeg_input
        display_name = resolved.display_name
        duration_override_s = resolved.playlist.duration_s
    executor = Executor(src, target_dir, mono_mix=mono_mix, decimate_audio=decimate,
                        decimate_fs=self.__config.target_fs, display_name=display_name,
                        duration_override_s=duration_override_s)
    if output_file_name is not None:
        executor.output_file_name = output_file_name
    executor.probe_file()
    if not executor.has_audio():
        raise ValueError(f"{src} has no audio stream to extract")
    executor.update_spec(audio_stream, video_stream, mono_mix)
    executor.run_sync()
    return ExtractResult(wav_path=executor.get_output_path(), channel_layout_name=executor.channel_layout_name)

def extract(self, src: str, target_dir: str, audio_stream: int = 0, video_stream: int = -1,
           mono_mix: bool = True, decimate: bool = True, playlist_name: Optional[str] = None) -> str:
    ''' Unchanged signature/behaviour/return type -- now a thin wrapper, so this and extract_with_layout()
    can never behaviourally diverge. '''
    return self.extract_with_layout(src, target_dir, audio_stream, video_stream, mono_mix, decimate,
                                    playlist_name).wav_path
```

This is a refactor of an existing, heavily-relied-upon method
(`design_and_queue()`, `batch_design()`, `model/batch.py`, and
`test_pipeline_acceptance.py`'s end-to-end test all call `extract()`
today) -- the full test suite passing unmodified for every one of
those existing callers is the acceptance bar for this part, not just
"the new tests pass".

### D.4 `pipeline/library/extract_cache.py`

```python
'''
Idempotent wrapper around Session.extract_with_layout() -- design/library-sync-pipeline-plan.md §4.1.
Skips ffmpeg entirely when the source hasn't changed (per LibraryItem.fingerprint, or a local mtime/size
fallback) and neither the analysis config nor the extraction mode (mono_mix) has changed since the last
run recorded in <target_dir>/manifest.json.
'''
import hashlib
import json
import os
import time
from typing import Tuple

from pipeline.config import AnalysisConfig
from pipeline.library.source import LibraryItem
from pipeline.orchestrate import Session

_MANIFEST_FILENAME = 'manifest.json'


def _source_fingerprint(item: LibraryItem) -> str:
    if item.fingerprint:
        return item.fingerprint
    stat = os.stat(item.source_path)
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def _params_hash(item: LibraryItem, config: AnalysisConfig, mono_mix: bool) -> str:
    payload = json.dumps({
        'audio_stream': item.audio_stream,
        'playlist_name': item.playlist_name,
        'mono_mix': mono_mix,
        'target_fs': config.target_fs,  # only actually varies the output when mono_mix is True -- see D.1
    }, sort_keys=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _read_manifest(target_dir: str) -> dict:
    path = os.path.join(target_dir, _MANIFEST_FILENAME)
    if os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def _write_manifest(target_dir: str, manifest: dict) -> None:
    os.makedirs(target_dir, exist_ok=True)
    with open(os.path.join(target_dir, _MANIFEST_FILENAME), 'w', encoding='utf-8') as f:
        json.dump(manifest, f)


def extract_if_needed(session: Session, item: LibraryItem, target_dir: str, config: AnalysisConfig,
                      mono_mix: bool, force: bool = False) -> Tuple[str, bool]:
    '''
    :param mono_mix: True for the mono-for-design extraction (decimated to config.target_fs, written to
        <target_dir>/mono.wav), False for the full-quality multichannel "kept" extraction (never decimated
        -- see §4.1 -- written to <target_dir>/multichannel.wav). A caller wanting both calls this twice,
        once with each value -- this function only ever handles one at a time.
    :return: (wav_path, cached) -- cached=True if ffmpeg was skipped because the manifest already recorded
        a matching (source_fingerprint, params_hash) and the wav file still exists on disk.
    '''
    prefix = 'mono' if mono_mix else 'multichannel'
    wav_path = os.path.join(target_dir, f"{prefix}.wav")
    manifest = _read_manifest(target_dir)
    fingerprint = _source_fingerprint(item)
    params_hash = _params_hash(item, config, mono_mix)

    if not force:
        if (manifest.get(f"{prefix}_source_fingerprint") == fingerprint
                and manifest.get(f"{prefix}_params_hash") == params_hash
                and os.path.isfile(wav_path)):
            return wav_path, True

    result = session.extract_with_layout(item.source_path, target_dir, audio_stream=item.audio_stream,
                                         mono_mix=mono_mix, decimate=mono_mix, playlist_name=item.playlist_name,
                                         output_file_name=prefix)
    manifest[f"{prefix}_source_fingerprint"] = fingerprint
    manifest[f"{prefix}_params_hash"] = params_hash
    manifest[f"{prefix}_extracted_at"] = time.time()
    if not mono_mix:
        # flat top-level key -- pipeline.review._read_channel_layout_name() (chunk 2, already shipped)
        # reads exactly this key, not a nested one; do not change this without also revisiting that code.
        manifest['channel_layout_name'] = result.channel_layout_name
    _write_manifest(target_dir, manifest)
    return result.wav_path, False
```

**Manifest shape**, concretely, for an item that had both extractions
run once then the multichannel one force-redone:

```json
{
  "mono_source_fingerprint": "1737000000000000000:483821",
  "mono_params_hash": "3f2b...",
  "mono_extracted_at": 1737000012.5,
  "multichannel_source_fingerprint": "1737000000000000000:483821",
  "multichannel_params_hash": "9ac1...",
  "multichannel_extracted_at": 1737000512.1,
  "channel_layout_name": "5.1",
  "source_channel_count": 6
}
```

### D.5 Tests to add

New `src/test/python/test_pipeline_library_extract_cache.py`. Needs a
real short audio/video fixture ffmpeg can probe+extract -- check
`test_pipeline_orchestrate.py`/`test_pipeline_acceptance.py` for
whatever fixture file(s) this repo's existing extract-related tests
already use and reuse the same one(s) rather than adding a new binary
fixture.

- `test_extract_if_needed_runs_ffmpeg_on_first_call` -- fresh
  `target_dir`, assert `cached is False`, `mono.wav` exists, and
  `manifest.json` now has `mono_source_fingerprint`/`mono_params_hash`.
- `test_extract_if_needed_skips_ffmpeg_on_a_repeat_call` -- call twice
  with identical `item`/`config`/`mono_mix`, assert the second call's
  `cached is True` and (via monkeypatching `Session.extract_with_layout`
  to raise if called, or counting calls) that ffmpeg genuinely didn't
  run again.
- `test_extract_if_needed_reextracts_when_params_hash_changes` --
  second call with a different `AnalysisConfig.target_fs`, assert
  `cached is False`.
- `test_extract_if_needed_reextracts_when_source_fingerprint_changes`
  -- second call with a `LibraryItem.fingerprint` changed (or, for the
  mtime/size fallback path, actually touch/modify the source file),
  assert `cached is False`.
- `test_extract_if_needed_force_always_reextracts` -- identical inputs
  twice, `force=True` on the second, assert `cached is False`.
- `test_extract_if_needed_mono_and_multichannel_are_independently_cached`
  -- call once with `mono_mix=True`, once with `mono_mix=False`, then
  repeat only the mono call again -- assert the repeat is cached and
  the multichannel manifest entries/file are untouched (mtime
  unchanged, or via a call-counting monkeypatch).
- `test_extract_if_needed_multichannel_records_channel_layout_name` --
  a multichannel fixture, assert `manifest.json`'s `channel_layout_name`
  matches the fixture's actual layout (e.g. `'5.1'`), and that a mono
  call never writes that key at all.
- `test_extract_if_needed_multichannel_does_not_decimate` -- compare
  the multichannel output's sample rate against the source's original
  rate (unchanged), vs. the mono output's rate (`config.target_fs`).
- `test_extract_with_layout_is_behaviourally_identical_to_extract` --
  call both on the same input (different `target_dir`s), assert the
  returned wav paths' contents are byte-identical and
  `extract()`'s path equals `extract_with_layout(...).wav_path`.
- Re-run the *existing* `test_pipeline_orchestrate.py` / relevant
  `test_pipeline_review.py` / `test_pipeline_acceptance.py` tests
  unmodified -- they must all still pass after `extract()` becomes a
  wrapper (this is a regression check, not a new test to write).

### D.6 Acceptance checklist

- [x] `Session.extract_with_layout()` added; `Session.extract()`
      refactored to call it, same signature/return type/behaviour.
- [x] `pipeline/library/extract_cache.py` created with
      `extract_if_needed()` and its manifest helpers.
- [x] Manifest uses the exact flat key names in D.4 (`channel_layout_name`
      at top level, not nested) -- confirmed compatible with the
      already-shipped `pipeline.review._read_channel_layout_name()`.
- [x] All new tests (D.5) pass; full suite still green, including
      every pre-existing `extract()` caller's tests unmodified.
- [x] `pipeline_qt_boundary`-style AST scan confirms
      `pipeline/library/extract_cache.py` imports no `qtpy`.
