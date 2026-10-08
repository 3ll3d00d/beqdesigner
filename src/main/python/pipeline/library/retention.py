'''
What a title's work folder keeps, and room for the next one (TODO R5).

With Keep multichannel, a title's folder holds `multichannel.wav`, the per-channel audio at the analysis rate: about
165 MB for a two-hour 7.1 title, so a 1,000-title catalogue is about 200 GB. Once a title is published, that file is
compressed in place, losslessly, to `multichannel.flac` (decided 2026-10-08). Everything that reads it first restores the
wav (`restore_multichannel()`): a redesign, a revise from the kept audio, a republish writing the multichannel project,
and a designer that takes arrays by reference, which reads WAV only. The extract cache treats the compressed file as the
extraction it is, so a title is not extracted again because of it. `mono.wav` and the `.beq` projects (which embed their
samples) are kept as they are.

Separately, a run stops before an extraction would fill the disk: below `run.min_free_gb` free in the work directory,
extraction raises OutOfSpace, an unavailable dependency (pipeline.library.failure) that ends the run at once.
'''
import logging
import os
import shutil
from typing import Optional

import soundfile as sf

from pipeline.library.failure import Unavailable

logger = logging.getLogger('library_retention')

WAV, FLAC = 'multichannel.wav', 'multichannel.flac'
COMPRESSED_KEY = 'multichannel_compressed'   # the manifest's note that the kept wav is now FLAC_SUBTYPE in FLAC
DEFAULT_MIN_FREE_GB = 10.0
_BLOCK = 1 << 18   # frames converted at a time: a long title is never read whole
_FLAC_SUBTYPES = {'PCM_16', 'PCM_24', 'PCM_S8'}


class OutOfSpace(Unavailable):
    ''' The work directory is below the run's free-space floor: no title can be extracted, so the run stops now. '''


def min_free_gb(value=None) -> float:
    ''' Validate the optional `run.min_free_gb` profile value (0 turns the check off). '''
    if value is None:
        return DEFAULT_MIN_FREE_GB
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError('run.min_free_gb must be a number of gigabytes, 0 or more')
    return float(value)


def check_free_space(work_dir: str, floor_gb: float) -> None:
    ''' :raises OutOfSpace: if the disk holding `work_dir` has less than `floor_gb` free. '''
    if floor_gb <= 0:
        return
    path = work_dir
    while path and not os.path.isdir(path):   # the work directory may not exist yet: its disk does
        parent = os.path.dirname(path)
        if parent == path:
            return
        path = parent
    free = shutil.disk_usage(path).free / 1e9
    if free < floor_gb:
        raise OutOfSpace(f'only {free:.1f} GB is free where the work directory is, below run.min_free_gb '
                         f'({floor_gb:g} GB): free some space or lower it')


def _convert(source: str, target: str, fmt: str, subtype: str) -> int:
    ''' Copy every frame of `source` into `target` (written beside it, then moved into place). :return: frames. '''
    partial = target + '.partial'
    frames = 0
    with sf.SoundFile(source) as reader, sf.SoundFile(partial, 'w', reader.samplerate, reader.channels,
                                                      format=fmt, subtype=subtype) as writer:
        for block in reader.blocks(blocksize=_BLOCK, dtype='int32', always_2d=True):
            writer.write(block)
            frames += len(block)
        expected = reader.frames
    if frames != expected or sf.info(partial).frames != expected:
        os.remove(partial)
        raise ValueError(f'{os.path.basename(target)} would hold {frames} of {expected} frames')
    os.replace(partial, target)
    return frames


def compress_multichannel(target_dir: str) -> int:
    '''
    Replace a title's `multichannel.wav` with a lossless `multichannel.flac`. Nothing happens to a folder without the wav,
    or whose wav FLAC cannot hold exactly (a float wav).
    :return: the bytes saved (0 if nothing was done).
    '''
    from pipeline.library.extract_cache import _write_manifest, read_manifest
    wav, flac = os.path.join(target_dir, WAV), os.path.join(target_dir, FLAC)
    if not os.path.isfile(wav):
        return 0
    subtype = sf.info(wav).subtype
    if subtype not in _FLAC_SUBTYPES:
        logger.info('Not compressing %s: FLAC cannot hold %s samples exactly', wav, subtype)
        return 0
    before = os.path.getsize(wav)
    _convert(wav, flac, 'FLAC', subtype)
    manifest = read_manifest(target_dir)
    manifest[COMPRESSED_KEY] = subtype
    _write_manifest(target_dir, manifest)
    os.remove(wav)
    return before - os.path.getsize(flac)


def restore_multichannel(target_dir: str) -> bool:
    ''' Put a compressed `multichannel.wav` back, as it was, before anything reads it. :return: True if it was restored. '''
    from pipeline.library.extract_cache import _write_manifest, read_manifest
    wav, flac = os.path.join(target_dir, WAV), os.path.join(target_dir, FLAC)
    if os.path.isfile(wav) or not os.path.isfile(flac):
        return False
    manifest = read_manifest(target_dir)
    _convert(flac, wav, 'WAV', manifest.get(COMPRESSED_KEY) or sf.info(flac).subtype)
    manifest.pop(COMPRESSED_KEY, None)
    _write_manifest(target_dir, manifest)
    os.remove(flac)
    return True


def is_compressed(target_dir: str) -> bool:
    return os.path.isfile(os.path.join(target_dir, FLAC)) and not os.path.isfile(os.path.join(target_dir, WAV))


def compress_published(work_dir: Optional[str], ids) -> int:
    ''' After a publish: compress each published title's kept multichannel audio. Never fails the publish. '''
    if not work_dir:
        return 0
    from pipeline.library.workdir import entry_directory
    saved = 0
    for title_id in ids:
        try:
            saved += compress_multichannel(entry_directory(work_dir, title_id))
        except Exception as error:   # the audio is still there, uncompressed: a convenience was missed, nothing lost
            logger.warning('Could not compress the multichannel audio of %s: %s', title_id, error)
    return saved
