'''Composition of a library source with the extract and design caches.'''
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Callable, Mapping, Optional, Sequence

import requests

from pipeline.config import AnalysisConfig
from pipeline.designer.contract import Coverage
from pipeline.library.design_cache import design_if_needed
from model.execution_events import emit_execution_event, event_scope
from pipeline.library.failure import Unavailable, unavailable_reason
from pipeline.library.disks import DiskLimit
from pipeline.library.retention import DEFAULT_MIN_FREE_GB, OutOfSpace, check_free_space, min_free_gb, \
    restore_multichannel
from pipeline.library.streams import audio_ordinal, channels_found, stream_choice
from pipeline.library.extract_cache import extract_if_needed, title_hint, extract_status, mono_from_multichannel_if_needed, \
    read_channel_layout_name, read_source_channel_count
from pipeline.library.index import LibraryIndex, with_audio_stream
from pipeline.library.library_metadata import library_meta, resolve_meta
from pipeline.library.season import DEFAULT_TV_MODE, SeasonGroup, plan_units, season_track_if_needed, with_extracted
from pipeline.library.source import LibraryItem, LibrarySource
from pipeline.library.status import failure_applies, failure_key, safe_fingerprint, season_source_fingerprint, \
    unit_fingerprint
from pipeline.library.union import reconstruct_claims
from pipeline.library.workdir import item_directory
from pipeline.metadata import redact
from pipeline.orchestrate import Session
from pipeline.review import project_name

logger = logging.getLogger('library_run')

# extraction mostly reads (a whole remux for its audio): with run.disks it is the number of disks kept busy. A designer is
# one service, so a handful of designs at once is plenty
MAX_STAGE_PARALLELISM = {'extract': 16, 'design': 4}
DEFAULT_STOP_AFTER_UNAVAILABLE = 3
_RUN_STAGES = ('extract', 'design')


def stage_parallelism(value=None) -> dict[str, int]:
    """Validate the optional ``run.parallelism`` profile mapping; older profiles mean one per stage."""
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError('run.parallelism must be a mapping of stage names to integers')
    unknown = set(value) - set(_RUN_STAGES)
    if unknown:
        raise ValueError(f'unknown run.parallelism stage(s): {", ".join(sorted(map(str, unknown)))}')
    result = {}
    for stage in _RUN_STAGES:
        count = value.get(stage, 1)
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MAX_STAGE_PARALLELISM[stage]:
            raise ValueError(f'run.parallelism.{stage} must be an integer from 1 to {MAX_STAGE_PARALLELISM[stage]}')
        result[stage] = count
    return result


def stop_after_unavailable(value=None) -> int:
    """Validate the optional ``run.stop_after_unavailable`` profile value: how many titles in a row may meet an
    unavailable dependency (pipeline.library.failure) before a run stops."""
    if value is None:
        return DEFAULT_STOP_AFTER_UNAVAILABLE
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError('run.stop_after_unavailable must be a whole number of titles, 1 or more')
    return value


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
    extract_parallelism: int = 1
    design_parallelism: int = 1
    # a run stops after this many titles in a row met an unavailable dependency (a designer or media mount that is
    # not there), rather than failing the rest of its selection one by one
    stop_after_unavailable: int = DEFAULT_STOP_AFTER_UNAVAILABLE
    min_free_gb: float = DEFAULT_MIN_FREE_GB   # below this free in the work directory, extraction stops the run (R5)
    # the playback chain sent to the designer (pipeline.library.bass, `run.bass_management`); None sends none
    bass_management: Optional[dict] = None
    # read each disk one (or per_disk) title at a time (pipeline.library.disks, `run.disks`); None does not look
    disks: Optional[DiskLimit] = None

    def __post_init__(self):
        plan_units([], self.tv_mode)  # rejects an unknown mode up front, not on the first TV item
        stage_parallelism({'extract': self.extract_parallelism, 'design': self.design_parallelism})
        stop_after_unavailable(self.stop_after_unavailable)
        min_free_gb(self.min_free_gb)


@dataclass(frozen=True)
class LibraryRunReport:
    extracted: list[str] = field(default_factory=list)
    cached: list[str] = field(default_factory=list)
    designed: list[str] = field(default_factory=list)
    design_cached: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    # not tried: it failed before, against this same source and these same settings (see run_library(retry_failed=))
    failed_earlier: list[tuple[str, str]] = field(default_factory=list)
    # not done because something it depends on was unavailable (pipeline.library.failure): not remembered, so the next
    # run tries it again
    unavailable: list[tuple[str, str]] = field(default_factory=list)
    halt: list[str] = field(default_factory=list)   # why no further title can be done now (the disk is full): stop at once
    meta_unresolved: list[tuple[str, str]] = field(default_factory=list)  # designed with item.meta only
    project_edit_preserved: list[str] = field(default_factory=list)  # a human-edited .beq project was kept
    seasons: dict[str, list[str]] = field(default_factory=dict)  # tv_mode='season': season id -> its episodes' ids
    stopped: str = ''   # run_library(): why it stopped before the end of its titles, if it did


@dataclass(frozen=True)
class UnitWork:
    """The extraction outputs a design stage needs; safe to hand between stage workers."""

    unit: object
    item: LibraryItem
    wav_path: str
    project_dir: str
    multichannel_wav_path: Optional[str] = None
    channel_layout_name: str = 'unknown'
    recorded_source: Optional[str] = None


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
            # requests puts the whole URL, `?api_key=...`, in the text of an HTTPError or a ConnectionError
            reason = redact(str(error), run_config.tmdb_api_key)
            logger.warning('Unable to resolve TMDB metadata for %s: %s', item.id, reason)
            report.meta_unresolved.append((item.id, f'{type(error).__name__}: {reason}'))
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


def _report_failure(report: LibraryRunReport, index: Optional[LibraryIndex], unit, run_config: 'LibraryRunConfig',
                    error: Exception) -> None:
    '''
    A title's failure: reported, and remembered in `index` unless something the title depends on was unavailable
    (pipeline.library.failure), when it is reported in `report.unavailable` and the next run tries it again.
    '''
    item = unit.item if isinstance(unit, SeasonGroup) else unit
    message = f'{type(error).__name__}: {error}'
    in_extract = getattr(error, 'library_stage', 'extract') == 'extract'
    # a path the source already says is wrong (an unmapped JRiver drive) is that, not storage that is not there
    readable = in_extract and not isinstance(unit, SeasonGroup) and not item.source_path_problem
    reason = unavailable_reason(error, item.source_path if readable else None)
    if reason is not None:
        report.unavailable.append((item.id, message))
        if isinstance(error, OutOfSpace):
            report.halt.append(str(error))
        emit_execution_event('failed', message=f'Unavailable ({reason}), so it is tried again next run: {message}')
        return
    report.failed.append((item.id, message))
    emit_execution_event('failed', message=message)
    if index is not None:
        _remember_failure(index, unit, run_config, error)


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
              through: str = 'design', on_stage: Optional[Callable[[str, str], None]] = None,
              on_extract_progress: Optional[Callable[[str, int, int], None]] = None) -> UnitWork:
    if item.source_path_problem:
        raise ValueError(item.source_path_problem)
    item_dir = item_directory(run_config.work_dir, item, create=True)
    with _stage('extract'):
        check_free_space(run_config.work_dir, run_config.min_free_gb)
        if run_config.keep_multichannel:
            restore_multichannel(item_dir)   # a published title's, compressed (R5): the cache checks the wav itself
    if on_stage is not None:
        on_stage(item.id, 'extract')
    with event_scope(title_id=item.id, stage='extract'):
        emit_execution_event('stage_started', message='Extracting ' + stream_choice(
            item.audio_stream_details, item.audio_stream, run_config.keep_multichannel, item.audio_stream_source))
        with _stage('extract'):
            progress = ({'on_progress': lambda position, total: on_extract_progress(item.id, position, total)}
                        if on_extract_progress is not None else {})
            multichannel_path = None
            channel_layout_name = 'unknown'
            known_mono = (run_config.keep_multichannel and read_source_channel_count(item_dir) == 1 and
                          extract_status(item, item_dir, run_config.config, True).current)
            if run_config.keep_multichannel and not known_mono:
                kept_path, kept_cached = extract_if_needed(
                    session, item, item_dir, run_config.config, mono_mix=False, force=run_config.force_extract,
                    **progress)
                mono_path, mono_cached = mono_from_multichannel_if_needed(
                    session, item, item_dir, run_config.config, force=run_config.force_extract)
                extraction_cached = mono_cached and kept_cached
                channel_layout_name = read_channel_layout_name(item_dir)
                if read_source_channel_count(item_dir) != 1:
                    multichannel_path = kept_path
            else:
                mono_path, extraction_cached = extract_if_needed(
                    session, item, item_dir, run_config.config, mono_mix=True, force=run_config.force_extract,
                    **progress)
        found = channels_found(read_source_channel_count(item_dir), channel_layout_name)
        emit_execution_event('stage_completed', message=('Extraction complete' if not extraction_cached else
                                                          'Extraction cache hit') + (f': {found}' if found else ''))

    if extraction_cached:
        report.cached.append(item.id)
    else:
        report.extracted.append(item.id)
    work = UnitWork(item, item, mono_path, item_dir, multichannel_path, channel_layout_name)
    _write_projects(session, work)
    if through == 'extract':
        return work

    if on_stage is not None:
        on_stage(item.id, 'design')
    with event_scope(title_id=item.id, stage='design'):
        emit_execution_event('stage_started', message='Designing filter')
        _design_work(session, work, run_config, report)
        emit_execution_event('stage_completed', message='Design complete')
    return work


def _write_projects(session: Session, work: 'UnitWork') -> None:
    '''
    design/archive/library-sync/worklist-feedback.md F2: the title's `.beq` projects, flat, as soon as it is extracted
    (only those not written yet). A project is a convenience: failing to write one is logged, and never fails the
    extraction.
    '''
    from pipeline.publish.project import write_missing_projects
    name = project_name(work.project_dir, work.item.id)
    multichannel = work.multichannel_wav_path
    try:
        write_missing_projects(session, work.wav_path, os.path.join(work.project_dir, f'{name}.mono.beq'),
                               multichannel_wav_path=multichannel, channel_layout_name=work.channel_layout_name,
                               multichannel_out_path=os.path.join(work.project_dir, f'{name}.multichannel.beq')
                               if multichannel else None)
    except Exception as error:
        logger.warning('could not write the projects of %s after extracting it: %s', work.item.id, error, exc_info=True)


def _run_season(session: Session, group: SeasonGroup, run_config: LibraryRunConfig, report: LibraryRunReport,
                through: str = 'design', on_stage: Optional[Callable[[str, str], None]] = None,
                index: Optional[LibraryIndex] = None, retry_failed: bool = False,
                on_extract_progress: Optional[Callable[[str, int, int], None]] = None) -> UnitWork:
    '''
    Extract every episode (each cached as in episode mode, so switching mode re-extracts nothing), join them into
    one track and design that. An episode that will not extract is reported and left out -- the season is then
    marked with the episodes that really went into it -- rather than sinking the whole season.

    With an `index`, such an episode's failure is remembered against its own id (as a title's is) and it is not
    tried again until its source or the settings change, or `retry_failed`; it is then in `report.failed_earlier`.
    '''
    if on_stage is not None:
        on_stage(group.item.id, 'extract')
    with event_scope(title_id=group.item.id, stage='extract'):
        emit_execution_event('stage_started', message='Extracting season episodes')
        with _stage('extract'):
            track_path, fingerprint, item, group_dir = _extract_season(session, group, run_config, report, index,
                                                                       retry_failed, on_extract_progress)
        emit_execution_event('stage_completed', message='Season extraction complete')
    work = UnitWork(group, item, track_path, group_dir, recorded_source=season_source_fingerprint(group) or None)
    _write_projects(session, work)
    if through == 'extract':
        return work
    if on_stage is not None:
        on_stage(group.item.id, 'design')
    with event_scope(title_id=group.item.id, stage='design'):
        emit_execution_event('stage_started', message='Designing season filter')
        _design_work(session, work, run_config, report)
        emit_execution_event('stage_completed', message='Season design complete')
    return work


def _extract_season(session: Session, group: SeasonGroup, run_config: LibraryRunConfig, report: LibraryRunReport,
                    index: Optional[LibraryIndex] = None, retry_failed: bool = False,
                    on_extract_progress: Optional[Callable[[str, int, int], None]] = None):
    member_wavs = []
    for member in group.members:
        fingerprint = safe_fingerprint(member)
        if not retry_failed:
            remembered = _failed_before(index, member, fingerprint, run_config)
            if remembered is not None:
                report.failed_earlier.append((member.id, remembered))
                continue
        try:
            check_free_space(run_config.work_dir, run_config.min_free_gb)
            progress = ({'on_progress': lambda position, total: on_extract_progress(group.item.id, position, total)}
                        if on_extract_progress is not None else {})
            wav_path, cached = extract_if_needed(
                session, member, item_directory(run_config.work_dir, member, create=True), run_config.config, mono_mix=True,
                force=run_config.force_extract, **progress)
        except Exception as error:
            reason = unavailable_reason(error, member.source_path)
            if reason is not None:   # the rest of the season would meet it too: nothing about the episode is remembered
                raise Unavailable(f'{member.display_name}: {reason}') from error
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

    group_dir = item_directory(run_config.work_dir, group.item, create=True)
    track_path, fingerprint, _ = season_track_if_needed(member_wavs, group_dir, force=run_config.force_extract)
    item = with_extracted(group, [episode for episode, _ in sorted(member_wavs)], fingerprint)
    report.seasons[item.id] = [m.id for m in group.members if m.episodes[0] in item.episodes]
    return track_path, fingerprint, item, group_dir


def _design_work(session: Session, work: UnitWork, run_config: LibraryRunConfig,
                 report: LibraryRunReport) -> None:
    channels = None
    multichannel_path = work.multichannel_wav_path
    if multichannel_path:
        restore_multichannel(os.path.dirname(multichannel_path))   # compressed once published (R5)
        channels = session.load_channels(multichannel_path, work.channel_layout_name)
        if not channels:
            multichannel_path = None
    _design(session, work.item, work.wav_path, run_config, report, work.project_dir, channels=channels,
            multichannel_path=multichannel_path, channel_layout_name=work.channel_layout_name,
            recorded_source=work.recorded_source)


def _design(session: Session, item: LibraryItem, wav_path: str, run_config: LibraryRunConfig,
            report: LibraryRunReport, project_dir: str, channels=None, multichannel_path=None,
            channel_layout_name: str = 'unknown', recorded_source=None) -> None:
    with _stage('design'):
        result = design_if_needed(
            session, item, wav_path, run_config.designer, run_config.queue_dir, run_config.config,
            coverage=run_config.coverage, force=run_config.force_design,
            meta=_meta_source(item, run_config, report), channels=channels,
            bass_management=run_config.bass_management,
            multichannel_wav_path=multichannel_path, channel_layout_name=channel_layout_name,
            project_dir=project_dir, recorded_source=recorded_source,
        )
    if result.designed:
        report.designed.append(item.id)
        if result.project_edit_preserved:
            report.project_edit_preserved.append(item.id)
    else:
        report.design_cached.append(item.id)


def resolve_selected_stream(session: Session, item: LibraryItem) -> tuple[LibraryItem, str]:
    '''
    The audio stream the source plays, as this file numbers its audio streams (J2): the source's selection
    (`selected_streams`, ffprobe global indices) matched against a probe of the file, as extraction opens it. A reviewer's
    choice, or one already resolved, is left alone.
    :return: (the item, '') -- or (the item unchanged, why the selection was not used, for the run to say): the first
        audio stream is then extracted, as before.
    '''
    if item.audio_stream_source in ('manual', 'source') or not item.selected_streams:
        return item, ''
    if item.source_path_problem or not os.path.exists(item.source_path):
        return item, ''   # extraction says what is wrong with the path
    said = ','.join(str(n) for n in item.selected_streams)
    try:
        streams = session.probe_streams(item.source_path, item.playlist_name, title_hint(item))
    except Exception as error:
        return item, f"the file could not be probed for the source's stream selection ({type(error).__name__}), so " \
                     f"the first audio stream is used"
    ordinal, why = audio_ordinal(streams, item.selected_streams)
    if ordinal is None:
        return item, f"the source's stream selection ({said}) does not match the file: {why}; the first audio stream " \
                     f"is used"
    return with_audio_stream(item, ordinal, 'source'), ''


def run_unit(session: Session, unit, run_config: LibraryRunConfig, report: LibraryRunReport,
             index: Optional[LibraryIndex] = None, *, retry_failed: bool = False, through: str = 'design',
             on_stage: Optional[Callable[[str, str], None]] = None,
             on_extract_progress: Optional[Callable[[str, int, int], None]] = None) -> Optional[UnitWork]:
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
            work = _run_season(session, unit, run_config, report, through, on_stage, index, retry_failed,
                               on_extract_progress)
        else:
            unit, fallback = resolve_selected_stream(session, unit)
            if fallback:
                with event_scope(title_id=item.id, stage='extract'):
                    emit_execution_event('note', message=fallback[:1].upper() + fallback[1:])
            elif unit.audio_stream_source == 'source' and index is not None:
                index.resolve_audio_stream(item.id, unit.audio_stream)   # kept by later scans (carried_choice)
            work = _run_item(session, unit, run_config, report, through, on_stage, on_extract_progress)
        if index is not None:
            index.clear_failure(item.id)
        return work
    except Exception as error:
        logger.exception('Library extraction failed for %s (%s)', item.display_name, item.id)
        _report_failure(report, index, unit, run_config, error)
        return None


def cached_unit_work(unit, run_config: LibraryRunConfig) -> tuple[Optional[UnitWork], str]:
    '''
    What a design needs of a title whose extraction is already current, read without extracting anything: a title that
    only needs design then goes straight to design (no extract stage, no extract worker). Reads the work folder's
    manifest and checks the wavs are there; never runs ffmpeg and never touches the source.
    :return: (the work, '') -- or (None, why it must be extracted again) when the audio is missing or out of date, or
        the unit is a season (its joined track is rebuilt by the extract stage) or extraction is forced.
    '''
    if isinstance(unit, SeasonGroup):
        return None, 'a season is joined by the extract stage'
    if run_config.force_extract:
        return None, 'extraction is forced'
    item = unit
    item_dir = item_directory(run_config.work_dir, item)
    try:
        mono = extract_status(item, item_dir, run_config.config, True)
    except OSError as error:   # no fingerprint of its own, and the source cannot be read: let extraction say why
        return None, f'the source cannot be read ({error.strerror or error})'
    if not mono.current:
        return None, 'the extracted audio is missing' if mono.state == 'none' else 'the extracted audio is out of date'
    multichannel, layout = None, 'unknown'
    if run_config.keep_multichannel and read_source_channel_count(item_dir) != 1:
        kept = extract_status(item, item_dir, run_config.config, False, fingerprint=mono.fingerprint)
        if not kept.current:
            return None, 'the kept multichannel audio is missing' if kept.state == 'none' else \
                'the kept multichannel audio is out of date'
        multichannel, layout = kept.wav_path, read_channel_layout_name(item_dir)
    return UnitWork(item, item, mono.wav_path, item_dir, multichannel, layout), ''


def design_unit_work(work: UnitWork, run_config: LibraryRunConfig,
                     index: Optional[LibraryIndex] = None,
                     on_stage: Optional[Callable[[str, str], None]] = None) -> LibraryRunReport:
    """Run only the design stage for an extraction already completed by ``run_unit(..., through='extract')``."""
    report = LibraryRunReport()
    unit = work.unit
    item = work.item
    try:
        with event_scope(title_id=item.id, stage='design'):
            if on_stage is not None:
                on_stage(item.id, 'design')
            emit_execution_event('stage_started', message='Designing filter')
            _design_work(Session(run_config.config), work, run_config, report)
            emit_execution_event('stage_completed', message='Design complete')
    except Exception as error:
        logger.exception('Library design failed for %s (%s)', item.display_name, item.id)
        if not hasattr(error, 'library_stage'):
            error.library_stage = 'design'
        with event_scope(title_id=item.id, stage='design'):
            _report_failure(report, index, unit, run_config, error)
    else:
        if index is not None:
            index.clear_failure(item.id)
    return report


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
    in_a_row = UnavailableStreak(run_config.stop_after_unavailable)
    for unit in plan_units(list(source.list_items(**source_query)), run_config.tv_mode, claims.season_id):
        item = unit.item if isinstance(unit, SeasonGroup) else unit
        before = (len(report.unavailable), len(report.failed_earlier))
        try:
            run_unit(session, unit, run_config, report, index, retry_failed=retry_failed, through=through)
        finally:
            if on_item_done is not None:
                on_item_done(item.id)
        if report.halt:
            logger.warning('Library run stopped: %s', report.halt[-1])
            return replace(report, stopped=f'stopped: {report.halt[-1]}')
        if len(report.unavailable) > before[0]:
            in_a_row.unavailable(report.unavailable[-1][1])
        elif len(report.failed_earlier) == before[1]:
            in_a_row.reached()
        if in_a_row.reason:
            logger.warning('Library run stopped: %s', in_a_row.reason)
            return replace(report, stopped=in_a_row.reason)
    return report


class UnavailableStreak:
    '''
    Counts the titles in a row whose dependencies were unavailable; once there are `limit`, `reason` says why the run
    stops. A title that reached its dependencies (done, or failed on its own account) ends the streak.
    '''
    def __init__(self, limit: int):
        self.limit = limit
        self.count = 0
        self.reason = ''

    def unavailable(self, message: str) -> None:
        self.count += 1
        if self.count >= self.limit and not self.reason:
            self.reason = (f'stopped after {self.count} titles in a row could not be worked on because something they '
                           f'depend on was unavailable; the rest are left for the next run (last: {message})')

    def reached(self) -> None:
        self.count = 0
