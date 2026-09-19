'''Composition of a library source with the extract and design caches.'''
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import requests

from pipeline.config import AnalysisConfig
from pipeline.designer.contract import Coverage
from pipeline.library.design_cache import design_if_needed
from pipeline.library.extract_cache import extract_if_needed, read_channel_layout_name, \
    read_source_channel_count
from pipeline.library.index import LibraryIndex
from pipeline.library.library_metadata import library_meta, resolve_meta
from pipeline.library.season import DEFAULT_TV_MODE, SeasonGroup, plan_units, season_track_if_needed, with_extracted
from pipeline.library.source import LibraryItem, LibrarySource
from pipeline.library.status import failure_applies, failure_key, safe_fingerprint, unit_fingerprint
from pipeline.library.union import reconstruct_claims
from pipeline.orchestrate import Session

logger = logging.getLogger('library_run')


@dataclass(frozen=True)
class LibraryRunConfig:
    work_dir: str
    queue_dir: str
    designer: str
    config: AnalysisConfig = field(default_factory=AnalysisConfig)
    coverage: Coverage = 'complete_programme'
    keep_multichannel: bool = False
    force_extract: bool = False
    force_design: bool = False
    tmdb_api_key: Optional[str] = None
    audio_types: Sequence[str] = ()
    # 'episode': a filter per TV episode. 'season': each series' season is joined into one track, designed once and
    # published for the whole season (see pipeline.library.season). Multichannel is not kept for a season.
    tv_mode: str = DEFAULT_TV_MODE

    def __post_init__(self):
        plan_units([], self.tv_mode)  # rejects an unknown mode up front, not on the first TV item


@dataclass(frozen=True)
class LibraryRunReport:
    extracted: list[str] = field(default_factory=list)
    cached: list[str] = field(default_factory=list)
    designed: list[str] = field(default_factory=list)
    design_cached: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    # not tried: it failed before, against this same source and these same settings (see run_library(retry_failed=))
    failed_earlier: list[tuple[str, str]] = field(default_factory=list)
    meta_unresolved: list[tuple[str, str]] = field(default_factory=list)  # designed with item.meta only
    project_edit_preserved: list[str] = field(default_factory=list)  # a human-edited .beq project was kept
    seasons: dict[str, list[str]] = field(default_factory=dict)  # tv_mode='season': season id -> its episodes' ids


def _meta_source(item: LibraryItem, run_config: LibraryRunConfig, report: LibraryRunReport):
    '''
    Metadata for design_if_needed(): resolved lazily, so only an item that is actually designed pays for the
    TMDB lookup. A TMDB failure must not lose the (expensive) extraction and design, so it degrades to
    whatever the library itself supplied (always including a title, so a queue row is never just an id) and is recorded in report.meta_unresolved for a reviewer to fix.
    '''
    if not run_config.tmdb_api_key:
        return {**library_meta(item), **item.meta}

    def resolve() -> dict:
        try:
            return resolve_meta(item, run_config.tmdb_api_key, run_config.audio_types)
        except requests.RequestException as error:
            logger.warning('Unable to resolve TMDB metadata for %s: %s', item.id, error)
            report.meta_unresolved.append((item.id, f'{type(error).__name__}: {error}'))
            return {**library_meta(item), **item.meta}

    return resolve


@contextmanager
def _stage(name: str):
    ''' Tags an exception with the stage it came from, so run_library() can remember what failed and where. '''
    try:
        yield
    except Exception as error:
        if not hasattr(error, 'library_stage'):
            error.library_stage = name
        raise


def _remember_failure(index: LibraryIndex, unit, run_config: 'LibraryRunConfig', error: Exception) -> None:
    '''
    Records a failure against the source fingerprint and settings it happened with (design.md §12.5), so discovery
    can say "failed" and stop suggesting a retry until one of them changes. Never lets the index sink the run.
    '''
    item = unit.item if isinstance(unit, SeasonGroup) else unit
    _record(index, item, unit_fingerprint(unit), getattr(error, 'library_stage', 'extract'), error, run_config)


def _record(index: LibraryIndex, item: LibraryItem, fingerprint: str, stage: str, error: Exception,
            run_config: 'LibraryRunConfig') -> None:
    try:
        index.record_failure(
            item.id, stage, f'{type(error).__name__}: {error}', fingerprint,
            failure_key(stage, item, config=run_config.config, designer=run_config.designer,
                        coverage=run_config.coverage, keep_multichannel=run_config.keep_multichannel))
    except Exception as index_error:
        logger.warning('could not record the failure of %s in the index: %s', item.id, index_error)


def _failed_before(index: Optional[LibraryIndex], item: LibraryItem, fingerprint: str,
                   run_config: 'LibraryRunConfig') -> Optional[str]:
    ''' The remembered message if this title failed against exactly this source and these settings, else None. '''
    if index is None:
        return None
    try:
        memory = index.failure(item.id)
    except Exception as index_error:  # remembering is a convenience: without it the title is simply tried
        logger.warning('could not read the failure of %s from the index: %s', item.id, index_error)
        return None
    if memory is not None and failure_applies(memory, item, fingerprint, config=run_config.config,
                                              designer=run_config.designer, coverage=run_config.coverage,
                                              keep_multichannel=run_config.keep_multichannel):
        return memory.message
    return None


def _run_item(session: Session, item: LibraryItem, run_config: LibraryRunConfig, report: LibraryRunReport,
              through: str = 'design', on_stage: Optional[Callable[[str, str], None]] = None) -> None:
    item_dir = os.path.join(run_config.work_dir, item.id)
    if on_stage is not None:
        on_stage(item.id, 'extract')
    with _stage('extract'):
        mono_path, mono_cached = extract_if_needed(
            session, item, item_dir, run_config.config, mono_mix=True, force=run_config.force_extract)
        multichannel_path = None
        channel_layout_name = 'unknown'
        channels = None
        extraction_cached = mono_cached

        # a source known to be mono has nothing to keep, so skip the second (full-length) ffmpeg pass; an
        # unknown channel count still extracts, and load_channels() below decides
        if run_config.keep_multichannel and read_source_channel_count(item_dir) != 1:
            kept_path, kept_cached = extract_if_needed(
                session, item, item_dir, run_config.config, mono_mix=False, force=run_config.force_extract)
            extraction_cached = mono_cached and kept_cached
            channel_layout_name = read_channel_layout_name(item_dir)
            kept_channels = session.load_channels(kept_path, channel_layout_name)
            if kept_channels:
                multichannel_path = kept_path
                channels = kept_channels

    if extraction_cached:
        report.cached.append(item.id)
    else:
        report.extracted.append(item.id)
    if through == 'extract':
        return

    if on_stage is not None:
        on_stage(item.id, 'design')
    _design(session, item, mono_path, run_config, report, item_dir, channels=channels,
            multichannel_path=multichannel_path, channel_layout_name=channel_layout_name)


def _run_season(session: Session, group: SeasonGroup, run_config: LibraryRunConfig, report: LibraryRunReport,
                through: str = 'design', on_stage: Optional[Callable[[str, str], None]] = None,
                index: Optional[LibraryIndex] = None, retry_failed: bool = False) -> None:
    '''
    Extract every episode (each cached as in episode mode, so switching mode re-extracts nothing), join them into
    one track and design that. An episode that will not extract is reported and left out -- the season is then
    marked with the episodes that really went into it -- rather than sinking the whole season.

    With an `index`, such an episode's failure is remembered against its own id (as a title's is) and it is not
    tried again until its source or the settings change, or `retry_failed`; it is then in `report.failed_earlier`.
    '''
    if on_stage is not None:
        on_stage(group.item.id, 'extract')
    with _stage('extract'):
        track_path, fingerprint, item, group_dir = _extract_season(session, group, run_config, report, index,
                                                                   retry_failed)
    if through == 'extract':
        return
    if on_stage is not None:
        on_stage(group.item.id, 'design')
    _design(session, item, track_path, run_config, report, group_dir)


def _extract_season(session: Session, group: SeasonGroup, run_config: LibraryRunConfig, report: LibraryRunReport,
                    index: Optional[LibraryIndex] = None, retry_failed: bool = False):
    member_wavs = []
    for member in group.members:
        fingerprint = safe_fingerprint(member)
        if not retry_failed:
            remembered = _failed_before(index, member, fingerprint, run_config)
            if remembered is not None:
                report.failed_earlier.append((member.id, remembered))
                continue
        try:
            wav_path, cached = extract_if_needed(
                session, member, os.path.join(run_config.work_dir, member.id), run_config.config, mono_mix=True,
                force=run_config.force_extract)
        except Exception as error:
            report.failed.append((member.id, f'{type(error).__name__}: {error}'))
            if index is not None:
                _record(index, member, fingerprint, 'extract', error, run_config)
            continue
        if index is not None:
            index.clear_failure(member.id)
        member_wavs.append((member.episodes[0], wav_path))
        (report.cached if cached else report.extracted).append(member.id)
    if not member_wavs:
        raise ValueError(f'none of the {len(group.members)} episodes could be extracted')

    group_dir = os.path.join(run_config.work_dir, group.item.id)
    track_path, fingerprint, _ = season_track_if_needed(member_wavs, group_dir, force=run_config.force_extract)
    item = with_extracted(group, [episode for episode, _ in sorted(member_wavs)], fingerprint)
    report.seasons[item.id] = [m.id for m in group.members if m.episodes[0] in item.episodes]
    return track_path, fingerprint, item, group_dir


def _design(session: Session, item: LibraryItem, wav_path: str, run_config: LibraryRunConfig,
            report: LibraryRunReport, project_dir: str, channels=None, multichannel_path=None,
            channel_layout_name: str = 'unknown') -> None:
    with _stage('design'):
        result = design_if_needed(
            session, item, wav_path, run_config.designer, run_config.queue_dir, run_config.config,
            coverage=run_config.coverage, force=run_config.force_design,
            meta=_meta_source(item, run_config, report), channels=channels,
            multichannel_wav_path=multichannel_path, channel_layout_name=channel_layout_name,
            project_dir=project_dir,
        )
    if result.designed:
        report.designed.append(item.id)
        if result.project_edit_preserved:
            report.project_edit_preserved.append(item.id)
    else:
        report.design_cached.append(item.id)


def run_unit(session: Session, unit, run_config: LibraryRunConfig, report: LibraryRunReport,
             index: Optional[LibraryIndex] = None, *, retry_failed: bool = False, through: str = 'design',
             on_stage: Optional[Callable[[str, str], None]] = None) -> None:
    '''
    Extract and design one title (an item, or a TV season as a SeasonGroup) with its own failure boundary: an error is
    reported in `report.failed` and remembered in `index`, never raised. With an `index`, a title that failed before
    against the same source and settings is not tried again (it goes in `report.failed_earlier`) unless `retry_failed`.

    :param through: 'extract' stops after the audio is extracted; 'design' (the default) also designs.
    :param on_stage: called with (title id, 'extract' | 'design') as each stage starts.
    '''
    if through not in ('extract', 'design'):
        raise ValueError(f"through must be 'extract' or 'design', got {through!r}")
    item = unit.item if isinstance(unit, SeasonGroup) else unit
    if not retry_failed:
        remembered = _failed_before(index, item, unit_fingerprint(unit), run_config)
        if remembered is not None:
            report.failed_earlier.append((item.id, remembered))
            return
    try:
        if isinstance(unit, SeasonGroup):
            _run_season(session, unit, run_config, report, through, on_stage, index, retry_failed)
        else:
            _run_item(session, unit, run_config, report, through, on_stage)
        if index is not None:
            index.clear_failure(item.id)
    except Exception as error:
        report.failed.append((item.id, f'{type(error).__name__}: {error}'))
        if index is not None:
            _remember_failure(index, unit, run_config, error)


def run_library(source: LibrarySource, run_config: LibraryRunConfig,
                on_item_done: Optional[Callable[[str], None]] = None, index: Optional[LibraryIndex] = None, *,
                retry_failed: bool = False, through: str = 'design', **source_query) -> LibraryRunReport:
    '''Run source -> cached extraction -> cached design without publishing.

    Each item has an independent failure boundary so one bad input cannot
    prevent other titles in a large library from being designed.

    :param index: the discovery index. If given, a title that fails is remembered there against the source
        fingerprint and settings it failed with (so discovery can call it *failed*), and a title that works has any
        earlier failure forgotten. A title whose remembered failure still applies -- same source, same settings -- is
        **not tried again**: it is reported in `failed_earlier`, so a nightly run does not repeat a failure every night.
    :param retry_failed: try those titles again anyway (the CLI's `--retry-failed`, the GUI's *Retry failed*).
    :param through: 'design' (the default) or, to stop after the audio is extracted, 'extract'.
    '''
    session = Session(run_config.config)
    report = LibraryRunReport()
    claims = reconstruct_claims(run_config.work_dir, run_config.queue_dir)  # a season keeps the id it already has
    for unit in plan_units(list(source.list_items(**source_query)), run_config.tv_mode, claims.season_id):
        item = unit.item if isinstance(unit, SeasonGroup) else unit
        try:
            run_unit(session, unit, run_config, report, index, retry_failed=retry_failed, through=through)
        finally:
            if on_item_done is not None:
                on_item_done(item.id)
    return report
