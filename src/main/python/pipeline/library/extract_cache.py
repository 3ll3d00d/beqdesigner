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
