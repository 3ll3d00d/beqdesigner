'''Composition of a library source with the extract and design caches.'''
import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import requests

from pipeline.config import AnalysisConfig
from pipeline.designer.contract import Coverage
from pipeline.library.design_cache import design_if_needed
from pipeline.library.extract_cache import extract_if_needed, read_channel_layout_name, \
    read_source_channel_count
from pipeline.library.library_metadata import resolve_meta
from pipeline.library.source import LibraryItem, LibrarySource
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


@dataclass(frozen=True)
class LibraryRunReport:
    extracted: list[str] = field(default_factory=list)
    cached: list[str] = field(default_factory=list)
    designed: list[str] = field(default_factory=list)
    design_cached: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    meta_unresolved: list[tuple[str, str]] = field(default_factory=list)  # designed with item.meta only
    project_edit_preserved: list[str] = field(default_factory=list)  # a human-edited .beq project was kept


def _meta_source(item: LibraryItem, run_config: LibraryRunConfig, report: LibraryRunReport):
    '''
    Metadata for design_if_needed(): resolved lazily, so only an item that is actually designed pays for the
    TMDB lookup. A TMDB failure must not lose the (expensive) extraction and design, so it degrades to
    whatever the library itself supplied and is recorded in report.meta_unresolved for a reviewer to fix.
    '''
    if not run_config.tmdb_api_key:
        return dict(item.meta)

    def resolve() -> dict:
        try:
            return resolve_meta(item, run_config.tmdb_api_key, run_config.audio_types)
        except requests.RequestException as error:
            logger.warning('Unable to resolve TMDB metadata for %s: %s', item.id, error)
            report.meta_unresolved.append((item.id, f'{type(error).__name__}: {error}'))
            return dict(item.meta)

    return resolve


def run_library(source: LibrarySource, run_config: LibraryRunConfig,
                on_item_done: Optional[Callable[[str], None]] = None,
                **source_query) -> LibraryRunReport:
    '''Run source -> cached extraction -> cached design without publishing.

    Each item has an independent failure boundary so one bad input cannot
    prevent other titles in a large library from being designed.
    '''
    session = Session(run_config.config)
    report = LibraryRunReport()
    for item in source.list_items(**source_query):
        try:
            item_dir = os.path.join(run_config.work_dir, item.id)
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

            result = design_if_needed(
                session, item, mono_path, run_config.designer, run_config.queue_dir, run_config.config,
                coverage=run_config.coverage, force=run_config.force_design,
                meta=_meta_source(item, run_config, report), channels=channels,
                multichannel_wav_path=multichannel_path, channel_layout_name=channel_layout_name,
                project_dir=item_dir,
            )
            if result.designed:
                report.designed.append(item.id)
                if result.project_edit_preserved:
                    report.project_edit_preserved.append(item.id)
            else:
                report.design_cached.append(item.id)
        except Exception as error:
            report.failed.append((item.id, f'{type(error).__name__}: {error}'))
        finally:
            if on_item_done is not None:
                on_item_done(item.id)
    return report
