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
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from pipeline.config import AnalysisConfig
from pipeline.library.source import LibraryItem
from pipeline.orchestrate import Session

_MANIFEST_FILENAME = 'manifest.json'


def source_fingerprint(item: LibraryItem) -> str:
    if item.fingerprint:
        return item.fingerprint
    stat = os.stat(item.source_path)
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def extract_params_hash(item: LibraryItem, config: AnalysisConfig, mono_mix: bool) -> str:
    payload = json.dumps({
        'audio_stream': item.audio_stream,
        'playlist_name': item.playlist_name,
        'mono_mix': mono_mix,
        # Both the design downmix and kept multichannel extraction are
        # decimated to this rate, matching model/batch.py's extraction path.
        'target_fs': config.target_fs,
        'decimate': True,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def read_manifest(target_dir: str) -> dict:
    path = os.path.join(target_dir, _MANIFEST_FILENAME)
    if os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def _write_manifest(target_dir: str, manifest: dict) -> None:
    os.makedirs(target_dir, exist_ok=True)
    with open(os.path.join(target_dir, _MANIFEST_FILENAME), 'w', encoding='utf-8') as f:
        json.dump(manifest, f)


def read_channel_layout_name(target_dir: str) -> str:
    '''Return the kept extraction's recorded layout, or the safe fallback.'''
    return read_manifest(target_dir).get('channel_layout_name', 'unknown')


def read_source_channel_count(target_dir: str) -> Optional[int]:
    '''
    Return the source's channel count as recorded by the last extraction, or None if it was never recorded
    (an extraction that predates this key, or a probe that couldn't tell). None means "unknown", not "mono".
    '''
    return read_manifest(target_dir).get('source_channel_count') or None


def invalidate_extract(target_dir: str) -> bool:
    '''
    Forgets what was extracted into `target_dir`, so the next extract_if_needed() runs ffmpeg again even though the
    source and settings look unchanged. The wav files stay (a failed re-extract leaves the old audio in place); what is
    dropped is the manifest's record of them. What describes the *source* (its channel count) is kept.
    :return: True if there was anything recorded to forget.
    '''
    manifest = read_manifest(target_dir)
    kept = {k: v for k, v in manifest.items() if not k.startswith(('mono_', 'multichannel_'))}
    if kept == manifest:
        return False
    _write_manifest(target_dir, kept)
    return True


@dataclass(frozen=True)
class ExtractStatus:
    '''
    Whether one extraction (mono or multichannel) is up to date -- what extract_if_needed() decides before it runs
    ffmpeg, without running anything.
    :param state: `current` (recorded against this source fingerprint and these parameters, and the wav is still
        there), `stale` (recorded against something else) or `none` (never recorded, or the wav is gone).
    '''
    state: str
    wav_path: str
    fingerprint: str
    params_hash: str

    @property
    def current(self) -> bool:
        return self.state == 'current'


def extract_status(item: LibraryItem, target_dir: str, config: AnalysisConfig, mono_mix: bool, *,
                   fingerprint: Optional[str] = None, manifest: Optional[dict] = None) -> ExtractStatus:
    '''
    The pure half of extract_if_needed(): reads the manifest and the wav's existence, changes nothing.
    :param fingerprint: the item's source fingerprint if the caller already has it (a discovery scan, which must
        tolerate a source it cannot stat); default source_fingerprint(item), which may raise OSError for a source
        that has none of its own. An **empty** fingerprint means "unknown": it is not compared, so a recorded
        extraction whose parameters match stays `current`.
    :param manifest: the manifest already read, to save a second read.
    '''
    prefix = 'mono' if mono_mix else 'multichannel'
    wav_path = os.path.join(target_dir, f"{prefix}.wav")
    if manifest is None:
        manifest = read_manifest(target_dir)
    if fingerprint is None:
        fingerprint = source_fingerprint(item)
    params_hash = extract_params_hash(item, config, mono_mix)
    recorded = manifest.get(f"{prefix}_source_fingerprint")
    if recorded is None or not os.path.isfile(wav_path):
        state = 'none'
    elif (fingerprint == '' or recorded == fingerprint) and manifest.get(f"{prefix}_params_hash") == params_hash:
        state = 'current'
    else:
        state = 'stale'
    return ExtractStatus(state, wav_path, fingerprint, params_hash)


def extract_if_needed(session: Session, item: LibraryItem, target_dir: str, config: AnalysisConfig,
                      mono_mix: bool, force: bool = False,
                      on_progress: Optional[Callable[[int, int], None]] = None) -> Tuple[str, bool]:
    '''
    :param mono_mix: True for the mono-for-design extraction, False for the kept multichannel extraction.
        Both are decimated to config.target_fs, matching Batch Extract / Design. A caller wanting both calls
        this twice, once with each value -- this function only ever handles one at a time.
    :return: (wav_path, cached) -- cached=True if ffmpeg was skipped because the manifest already recorded
        a matching (source_fingerprint, params_hash) and the wav file still exists on disk (see extract_status()).
    '''
    prefix = 'mono' if mono_mix else 'multichannel'
    manifest = read_manifest(target_dir)
    status = extract_status(item, target_dir, config, mono_mix, manifest=manifest)
    fingerprint, params_hash = status.fingerprint, status.params_hash

    if not force and status.current:
        return status.wav_path, True

    result = session.extract_with_layout(item.source_path, target_dir, audio_stream=item.audio_stream,
                                         mono_mix=mono_mix, decimate=True, playlist_name=item.playlist_name,
                                         output_file_name=prefix, on_progress=on_progress)
    manifest[f"{prefix}_source_fingerprint"] = fingerprint
    manifest[f"{prefix}_params_hash"] = params_hash
    manifest[f"{prefix}_extracted_at"] = time.time()
    if result.channel_count:
        manifest['source_channel_count'] = result.channel_count  # describes the source, whichever mode ran
    if not mono_mix:
        # flat top-level key -- pipeline.review._read_channel_layout_name() (chunk 2, already shipped)
        # reads exactly this key, not a nested one; do not change this without also revisiting that code.
        manifest['channel_layout_name'] = result.channel_layout_name
    _write_manifest(target_dir, manifest)
    return result.wav_path, False
