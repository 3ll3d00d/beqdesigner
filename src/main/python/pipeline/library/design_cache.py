'''Idempotent wrapper around pipeline.review.design_and_queue().'''
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Optional

from pipeline.config import AnalysisConfig
from pipeline.designer.contract import Coverage
from pipeline.library.extract_cache import source_fingerprint
from pipeline.library.source import LibraryItem
from pipeline.orchestrate import Session
from pipeline.review import QueueEntry, design_and_queue, read_entry, update_entry


@dataclass(frozen=True)
class DesignCacheResult:
    entry: QueueEntry
    designed: bool
    protected: bool = False


def design_fingerprint(item: LibraryItem, designer: str, config: AnalysisConfig, coverage: Coverage) -> str:
    '''Stable hash of every library-run input that can change a design result.'''
    payload = {
        'source_fingerprint': source_fingerprint(item),
        'designer': designer,
        'config': asdict(config),
        'coverage': coverage,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


def design_if_needed(session: Session, item: LibraryItem, wav_path: str, designer: str, queue_dir: str,
                     config: AnalysisConfig, coverage: Coverage = 'complete_programme', *, force: bool = False,
                     meta: Optional[dict] = None, bass_management: Optional[dict] = None,
                     channels: Optional[dict] = None, multichannel_wav_path: Optional[str] = None,
                     channel_layout_name: str = 'unknown', project_dir: Optional[str] = None) -> DesignCacheResult:
    '''
    Design only when no compatible queue entry already exists.

    Accepted and published entries are protected from every redesign path,
    including force. A caller must explicitly reset their status before a
    library-wide rerun can replace them.
    '''
    fingerprint = design_fingerprint(item, designer, config, coverage)
    try:
        existing = read_entry(queue_dir, item.id)
    except FileNotFoundError:
        existing = None

    if existing is not None:
        if existing.status in {'accepted', 'published'}:
            return DesignCacheResult(existing, designed=False, protected=True)
        if not force and existing.design_fingerprint == fingerprint:
            return DesignCacheResult(existing, designed=False)

    entry = design_and_queue(
        session, item.id, wav_path, designer, queue_dir, meta=meta, coverage=coverage,
        bass_management=bass_management, channels=channels, multichannel_wav_path=multichannel_wav_path,
        channel_layout_name=channel_layout_name, project_dir=project_dir,
    )
    entry = update_entry(queue_dir, entry.id, design_fingerprint=fingerprint)
    return DesignCacheResult(entry, designed=True)
