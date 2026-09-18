'''Composition of a library source with the extract and design caches.'''
import os
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from pipeline.config import AnalysisConfig
from pipeline.designer.contract import Coverage
from pipeline.library.design_cache import design_if_needed
from pipeline.library.extract_cache import extract_if_needed, read_channel_layout_name
from pipeline.library.library_metadata import resolve_meta
from pipeline.library.source import LibrarySource
from pipeline.orchestrate import Session


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

            if run_config.keep_multichannel:
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

            meta = dict(item.meta)
            if run_config.tmdb_api_key:
                meta = resolve_meta(item, run_config.tmdb_api_key, run_config.audio_types)
            result = design_if_needed(
                session, item, mono_path, run_config.designer, run_config.queue_dir, run_config.config,
                coverage=run_config.coverage, force=run_config.force_design, meta=meta, channels=channels,
                multichannel_wav_path=multichannel_path, channel_layout_name=channel_layout_name,
                project_dir=item_dir,
            )
            if result.designed:
                report.designed.append(item.id)
            else:
                report.design_cached.append(item.id)
        except Exception as error:
            report.failed.append((item.id, f'{type(error).__name__}: {error}'))
        finally:
            if on_item_done is not None:
                on_item_done(item.id)
    return report
