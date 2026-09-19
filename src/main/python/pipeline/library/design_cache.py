'''Idempotent wrapper around pipeline.review.design_and_queue().'''
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Callable, Optional, Union

from pipeline.config import AnalysisConfig
from pipeline.designer.contract import Coverage
from pipeline.library.artwork import resolve_art
from pipeline.library.extract_cache import source_fingerprint
from pipeline.library.source import LibraryItem
from pipeline.orchestrate import Session
from pipeline.review import QueueEntry, design_and_queue, read_entry, update_entry


# Either the metadata itself, or a callable producing it. A callable is only invoked when a design actually
# runs, so an expensive resolution (a TMDB round-trip) is skipped for every cache hit and protected entry.
MetaSource = Union[dict, Callable[[], dict], None]

# accepted and published entries are a human's decision and are never redesigned, whatever asks
PROTECTED_STATUSES = frozenset({'accepted', 'published'})


@dataclass(frozen=True)
class DesignCacheResult:
    entry: QueueEntry
    designed: bool
    protected: bool = False
    projects: Optional[dict] = None  # write_title_projects_if_safe()'s result when a design wrote projects

    @property
    def project_edit_preserved(self) -> bool:
        ''' True if a human-edited `.beq` project was left in place rather than overwritten by this design. '''
        return self.projects is not None and any(written is False for written in self.projects.values())


def design_fingerprint(item: LibraryItem, designer: str, config: AnalysisConfig, coverage: Coverage,
                       multichannel: bool = False, *, source: Optional[str] = None) -> str:
    '''
    Stable hash of every library-run input that can change a design result.

    Metadata (title, TMDB id, artwork) is deliberately absent: a design does not depend on it, and a redesign
    would replace the entry a reviewer may already have edited. The optional inputs are only hashed when set,
    so a fingerprint recorded before they existed still matches an unchanged default run.

    :param multichannel: True when a multichannel extraction feeds this design (DesignRequest.channels and the
        multichannel project), as opposed to a mono-only design.
    '''
    payload = {
        'source_fingerprint': source_fingerprint(item) if source is None else source,
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


@dataclass(frozen=True)
class DesignStatus:
    '''
    Whether a title's design is up to date -- what design_if_needed() decides before it runs the designer.
    :param state: `none` (no queue entry), `protected` (accepted or published: never redesigned), `current` (the
        entry was designed against exactly these inputs) or `stale` (against something else, or it predates
        the fingerprint).
    '''
    state: str
    fingerprint: str
    entry: Optional[QueueEntry] = None

    @property
    def protected(self) -> bool:
        return self.state == 'protected'

    @property
    def current(self) -> bool:
        return self.state == 'current'


def design_status(item: LibraryItem, existing: Optional[QueueEntry], designer: str, config: AnalysisConfig,
                  coverage: Coverage = 'complete_programme', *, multichannel: bool = False,
                  source: Optional[str] = None) -> DesignStatus:
    '''
    The pure half of design_if_needed(): compares `existing` (the title's queue entry, or None) with the fingerprint
    a design run now would record. Reads and changes nothing.
    :param multichannel: as design_fingerprint().
    :param source: the item's source fingerprint if the caller already has it, as design_fingerprint().
    '''
    fingerprint = design_fingerprint(item, designer, config, coverage, multichannel=multichannel, source=source)
    if existing is None:
        return DesignStatus('none', fingerprint)
    if existing.status in PROTECTED_STATUSES:
        return DesignStatus('protected', fingerprint, existing)
    return DesignStatus('current' if existing.design_fingerprint == fingerprint else 'stale', fingerprint, existing)


def design_if_needed(session: Session, item: LibraryItem, wav_path: str, designer: str, queue_dir: str,
                     config: AnalysisConfig, coverage: Coverage = 'complete_programme', *, force: bool = False,
                     meta: MetaSource = None, bass_management: Optional[dict] = None,
                     channels: Optional[dict] = None, multichannel_wav_path: Optional[str] = None,
                     channel_layout_name: str = 'unknown', project_dir: Optional[str] = None) -> DesignCacheResult:
    '''
    Design only when no compatible queue entry already exists (see design_status()).

    Accepted and published entries are protected from every redesign path,
    including force. A caller must explicitly reset their status before a
    library-wide rerun can replace them.

    Redesigning a pending/skipped/rejected entry replaces its candidates but keeps everything a human may have
    set on it: its metadata (the freshly resolved `meta` only fills keys the entry lacks), its artwork and its
    reviewer note. A design does not depend on any of those, so nothing about them is stale.

    Artwork is resolved here, once, when a design runs (library art, then TMDB's poster -- see
    pipeline.library.artwork), and only if the entry has none: a reviewer's choice, or an automatic one whose
    file still exists, is kept. A cache hit never touches artwork, so a reviewer's Clear stays cleared until
    the entry is next redesigned.

    :param meta: the entry's metadata, or a zero-argument callable returning it, called only if this call
        actually designs.
    '''
    try:
        existing = read_entry(queue_dir, item.id)
    except FileNotFoundError:
        existing = None
    status = design_status(item, existing, designer, config, coverage, multichannel=multichannel_wav_path is not None)
    fingerprint = status.fingerprint

    if status.protected:
        return DesignCacheResult(existing, designed=False, protected=True)
    if not force and status.current:
        return DesignCacheResult(existing, designed=False)

    if callable(meta):
        meta = meta()
    if existing is not None:
        meta = {**(meta or {}), **existing.meta}
    projects: dict = {}
    entry = design_and_queue(
        session, item.id, wav_path, designer, queue_dir, meta=meta, coverage=coverage,
        bass_management=bass_management, channels=channels, multichannel_wav_path=multichannel_wav_path,
        channel_layout_name=channel_layout_name, project_dir=project_dir, on_projects=projects.update,
    )
    art_path = existing.art_path if existing is not None else None
    art_overridden = existing.art_overridden if existing is not None else False
    if not (art_overridden or (art_path and os.path.isfile(art_path))):
        art_path = resolve_art(item, meta or {}, project_dir)
    kept = {'reviewer_note': existing.reviewer_note, 'revision': existing.revision} if existing is not None else {}
    entry = update_entry(queue_dir, entry.id, design_fingerprint=fingerprint, source_fingerprint=source_fingerprint(item),
                         art_path=art_path,
                         art_overridden=art_overridden, **kept)
    return DesignCacheResult(entry, designed=True, projects=projects or None)
