'''Idempotent wrapper around pipeline.review.design_and_queue().'''
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Callable, Optional, Union

from pipeline.config import AnalysisConfig
from pipeline.designer.contract import Coverage
from pipeline.library.extract_cache import source_fingerprint
from pipeline.library.source import LibraryItem
from pipeline.orchestrate import Session
from pipeline.review import QueueEntry, design_and_queue, read_entry, update_entry


# Either the metadata itself, or a callable producing it. A callable is only invoked when a design actually
# runs, so an expensive resolution (a TMDB round-trip) is skipped for every cache hit and protected entry.
MetaSource = Union[dict, Callable[[], dict], None]


@dataclass(frozen=True)
class DesignCacheResult:
    entry: QueueEntry
    designed: bool
    protected: bool = False


def design_fingerprint(item: LibraryItem, designer: str, config: AnalysisConfig, coverage: Coverage,
                       multichannel: bool = False) -> str:
    '''
    Stable hash of every library-run input that can change a design result.

    Metadata (title, TMDB id, artwork) is deliberately absent: a design does not depend on it, and a redesign
    would replace the entry a reviewer may already have edited. The optional inputs are only hashed when set,
    so a fingerprint recorded before they existed still matches an unchanged default run.

    :param multichannel: True when a multichannel extraction feeds this design (DesignRequest.channels and the
        multichannel project), as opposed to a mono-only design.
    '''
    payload = {
        'source_fingerprint': source_fingerprint(item),
        'designer': designer,
        'config': asdict(config),
        'coverage': coverage,
    }
    if item.audio_stream:
        payload['audio_stream'] = item.audio_stream
    if item.playlist_name:
        payload['playlist_name'] = item.playlist_name
    if multichannel:
        payload['multichannel'] = True
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


def design_if_needed(session: Session, item: LibraryItem, wav_path: str, designer: str, queue_dir: str,
                     config: AnalysisConfig, coverage: Coverage = 'complete_programme', *, force: bool = False,
                     meta: MetaSource = None, bass_management: Optional[dict] = None,
                     channels: Optional[dict] = None, multichannel_wav_path: Optional[str] = None,
                     channel_layout_name: str = 'unknown', project_dir: Optional[str] = None) -> DesignCacheResult:
    '''
    Design only when no compatible queue entry already exists.

    Accepted and published entries are protected from every redesign path,
    including force. A caller must explicitly reset their status before a
    library-wide rerun can replace them.

    Redesigning a pending/skipped/rejected entry replaces its candidates but keeps everything a human may have
    set on it: its metadata (the freshly resolved `meta` only fills keys the entry lacks), its artwork and its
    reviewer note. A design does not depend on any of those, so nothing about them is stale.

    :param meta: the entry's metadata, or a zero-argument callable returning it, called only if this call
        actually designs.
    '''
    fingerprint = design_fingerprint(item, designer, config, coverage,
                                     multichannel=multichannel_wav_path is not None)
    try:
        existing = read_entry(queue_dir, item.id)
    except FileNotFoundError:
        existing = None

    if existing is not None:
        if existing.status in {'accepted', 'published'}:
            return DesignCacheResult(existing, designed=False, protected=True)
        if not force and existing.design_fingerprint == fingerprint:
            return DesignCacheResult(existing, designed=False)

    if callable(meta):
        meta = meta()
    if existing is not None:
        meta = {**(meta or {}), **existing.meta}
    entry = design_and_queue(
        session, item.id, wav_path, designer, queue_dir, meta=meta, coverage=coverage,
        bass_management=bass_management, channels=channels, multichannel_wav_path=multichannel_wav_path,
        channel_layout_name=channel_layout_name, project_dir=project_dir,
    )
    kept = {'art_path': existing.art_path, 'art_overridden': existing.art_overridden,
            'reviewer_note': existing.reviewer_note} if existing is not None else {}
    entry = update_entry(queue_dir, entry.id, design_fingerprint=fingerprint, **kept)
    return DesignCacheResult(entry, designed=True)
