'''
pipeline/orchestrate.py: the Session facade -- design/pipeline-implementation-
plan.md phase 5 (item 11), composing every piece built in phases 0-4 per
design/archive/api-headless-pipeline.md §9.

Beyond composition, this module owns two things nothing built so far needed:

- **Applied/Declined outcome handling (§15.4).** `Session.design()` returns
  one or the other, explicitly -- `Declined` is a value, not an exception,
  and nothing downstream (set_filters, stats, to_beq_xml, publish) should
  run on it. "We correctly concluded nothing should be applied" is a
  first-class result, not an error, matching the catalogue's own framing
  (15,208 positives and zero negatives -- a false positive is the expensive
  failure).
- **D7's provenance threading.** A designer's confidence/method/fc_hz/
  slope/uncertainties/residual_db/commentary/channel_scope travel with an
  `Applied` outcome so a report can show *why* a filter was accepted -- but
  `to_beq_xml()` only ever receives `Applied.filters`, never the outcome
  itself: the catalogue XML format has no field for any of this, and no
  consumer other than a human reviewing the report needs it. A designer may
  offer several ranked candidates (design/designer-interface.md §3); only
  the top-ranked one is reflected in `Applied`'s own fields (and the only
  one ever simulated/published), the rest travel as `Applied.alternatives`
  for the same human-reviewing-the-report purpose. Designs the designer built
  and judged unfit to publish (1.1's `rejected`) travel as `.rejected` on
  either outcome, for a human to review -- never simulated or published here.

Scope trim: no `Session.fit()`/`optimise_filters()` wrapper -- no phase
built a Qt-free extraction of that GUI feature, and nothing in this plan's
acceptance criteria needs it. `Session.load()`/`set_filters()`/`stats()`/
`curves()` still assume a single-channel (mono) signal, matching this
pipeline's mono_mix=True scope throughout (design/archive/api-headless-pipeline.md:
"this pipeline always targets one device") -- a multi-channel file will
load via `load()` but nothing past that point handles the multi-channel
result. `design()` is the one exception: `load_channels()` decomposes a
multichannel wav (pure decomposition, no mixing) into the named per-channel
arrays `design()`'s optional `channels` param forwards as
`DesignRequest.channels` (design/designer-interface.md §2) -- a diagnostic
supplement to `mono_mix`, not a second primary signal, so it doesn't need
`load()`/`SingleChannelSignalData` to grow multi-channel awareness itself.

Qt-boundary note: `Session.load()` reuses `model.signal.AutoWavLoader` and
`extract()` uses `model.ffmpeg.Executor.run_sync()`. Both modules are Qt-free
(their dialogs and Qt models are in `model.signal_qt` / `model.ffmpeg_qt`),
so a Session runs where Qt is not installed at all --
test_qt_free_modules.py makes a whole run with every Qt package blocked.
'''
import json
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Union

from model.bdmv import bdmv_root_of, resolve_main_title
from model.dvd import dvd_root, resolve_main_title as resolve_main_dvd_title
from model.ffmpeg import Executor
from model.iir import CompleteFilter
from model.preferences import BASS_MANAGEMENT_LPF_FS, BASS_MANAGEMENT_LPF_POSITION, DEFAULT_PREFS
from model.signal import AutoWavLoader, SingleChannelSignalData
from model.xy import MagnitudeData

from pipeline.config import AnalysisConfig
from pipeline.designer.contract import ChannelScope, Coverage, build_request
from pipeline.designer.convert import alternative_filters, rejected_filters, to_complete_filter
from pipeline.designer.registry import get_designer, takes_sources
from pipeline.designer.manual import MANUAL_DESIGNER, manual_response
from pipeline.designer.sources import AudioSources
from pipeline.filters import FilterSpec, create_filter
from pipeline.metadata import BeqMetadata, tmdb_lookup
from pipeline.publish.catalogue import aggregate_path
from pipeline.publish.catalogue_json import aggregate, filter_record
from pipeline.publish.git import RepoTarget, commit_paths, fs_path, image_url as git_image_url, push as push_repo, push_image, write_files
from pipeline.publish.report import ReportSpec, render_report
from pipeline.publish.xml import to_beq_xml as render_beq_xml
from pipeline.stats import Stats, signal_stats

_CURVE_INDEX = {'avg': 0, 'peak': 1, 'median': 2}


@dataclass(frozen=True)
class AlternativeDesign:
    '''
    A lower-ranked candidate the designer considered but did not lead with --
    design/designer-interface.md §3's DesignResponse.candidates[1:]. Carried
    through so a report can show it alongside the applied filter; never
    simulated or published automatically.
    '''
    filters: CompleteFilter
    confidence: float
    method: str
    mv_adjust_db: Optional[float] = None  # NOT a clipping-cost estimate -- see gain_reduction_db
    gain_reduction_db: Optional[float] = None
    commentary: Optional[dict] = None
    residual_db: Optional[float] = None
    residual_band_hz: Optional[tuple] = None
    channel_scope: Optional[ChannelScope] = None
    # a rejected design (design/designer-interface.md 1.1, §3 "rejected"): why the designer judged it unfit to publish
    rejection_reasons: tuple = ()


def _design_of(candidate, complete_filter: CompleteFilter) -> AlternativeDesign:
    ''' A validated DesignCandidate, realised as `complete_filter`, as the outcome carries it. '''
    return AlternativeDesign(filters=complete_filter, confidence=candidate.confidence, method=candidate.method,
                             mv_adjust_db=candidate.mv_adjust_db, gain_reduction_db=candidate.gain_reduction_db,
                             commentary=candidate.commentary,
                             residual_db=candidate.residual_db, residual_band_hz=candidate.residual_band_hz,
                             channel_scope=candidate.channel_scope,
                             rejection_reasons=tuple(candidate.rejection_reasons or ()))


@dataclass(frozen=True)
class Applied:
    ''' A designer produced a publishable filter -- design/archive/api-headless-pipeline.md §15.4. '''
    filters: CompleteFilter
    confidence: float
    method: str
    mv_adjust_db: Optional[float] = None  # NOT a clipping-cost estimate -- see gain_reduction_db
    gain_reduction_db: Optional[float] = None
    fc_hz: Optional[float] = None
    slope: Optional[float] = None
    fc_uncertainty_hz: Optional[float] = None
    slope_uncertainty: Optional[float] = None
    residual_db: Optional[float] = None
    residual_band_hz: Optional[tuple] = None
    commentary: Optional[dict] = None
    channel_scope: Optional[ChannelScope] = None
    alternatives: tuple = ()  # tuple[AlternativeDesign, ...], lower-ranked candidates, best-first
    rejected: tuple = ()      # tuple[AlternativeDesign, ...] with rejection_reasons: judged unfit to publish (1.1)


@dataclass(frozen=True)
class Declined:
    ''' The designer concluded nothing should be applied -- a first-class result, not an error. '''
    reason: str
    message: Optional[str] = None
    rejected: tuple = ()      # tuple[AlternativeDesign, ...]: the designs whose failure is the decline (1.1), as evidence


DesignOutcome = Union[Applied, Declined]


@dataclass(frozen=True)
class ExtractResult:
    '''
    Session.extract_with_layout()'s return value -- design/archive/library-sync-pipeline-plan.md Appendix D.3.
    '''
    wav_path: str
    channel_layout_name: str  # model.ffmpeg's CHANNEL_LAYOUTS key, e.g. '5.1', or 'unknown'/a generic
                              # "<n> channels" string when ffmpeg's probe couldn't name it more precisely
    channel_count: int = 0    # the source stream's channel count; 0 if the probe couldn't determine it
    mono_mix_spec: Optional[str] = None  # the same pan coefficients Executor uses for a mono extraction


class _ConfigPreferences:
    '''
    Maps AnalysisConfig onto the `.get(key)` interface model.signal's
    AutoWavLoader/readWav/Signal expect, in place of a real Preferences/
    QSettings object (same pattern as pipeline/publish/report.py's
    _SpecPreferences).
    '''
    def __init__(self, config: AnalysisConfig, bm_lpf_fs: int, bm_lpf_position: str):
        self.__values = {
            'analysis/target_fs': config.target_fs,
            'analysis/resolution': config.resolution,
            'analysis/avg_window': config.avg_window,
            'analysis/peak_window': config.peak_window,
            'bm/fs': bm_lpf_fs,
            'bm/type': bm_lpf_position,
        }

    def get(self, key, default_if_unset=True):
        return self.__values.get(key)


def read_records(xml_repo: RepoTarget, record_dir: str) -> dict:
    ''':return: {relative path: record} of every individual JSON record beneath `record_dir` (not `database.json`).'''
    records = {}
    root = xml_repo.local_path
    scan_root = fs_path(xml_repo, record_dir) if record_dir else root
    if os.path.isdir(scan_root):
        for folder, dirs, names in os.walk(scan_root):
            dirs[:] = [name for name in dirs if name != '.git']
            for name in names:
                if not name.endswith('.json') or name == 'database.json':
                    continue
                path = os.path.join(folder, name)
                relative = os.path.relpath(path, root).replace(os.sep, '/')
                try:
                    with open(path, encoding='utf-8') as handle:
                        candidate = json.load(handle)
                    if isinstance(candidate, dict):
                        records[relative] = candidate
                except (OSError, ValueError):
                    continue
    return records


def write_aggregate_for(xml_repo: RepoTarget, record_dir: str) -> str:
    '''
    Writes `record_dir`'s derived `database.json` from the records beside it, once. A batch of publishes does this at
    its end instead of once per title (which read every record in the directory again each time).
    :return: the aggregate's relative path.
    '''
    relative = aggregate_path(record_dir)
    write_files(xml_repo, {relative: aggregate(read_records(xml_repo, record_dir))})
    return relative


class Session:
    '''
    Session-scoped facade over one signal at a time -- signals are
    expensive to load/analyse, filters are cheap to iterate, so this holds
    the loaded signal steady across design/set_filters/curves/stats calls
    rather than re-extracting or re-analysing on every step
    (design/archive/api-headless-pipeline.md §9).
    '''
    def __init__(self, config: AnalysisConfig = AnalysisConfig(), *,
                 bm_lpf_fs: int = DEFAULT_PREFS[BASS_MANAGEMENT_LPF_FS],
                 bm_lpf_position: str = DEFAULT_PREFS[BASS_MANAGEMENT_LPF_POSITION]):
        self.__config = config
        self.__preferences = _ConfigPreferences(config, bm_lpf_fs, bm_lpf_position)

    @property
    def preferences(self):
        '''The session's analysis and bass-management settings for model signal objects.'''
        return self.__preferences

    def extract_with_layout(self, src: str, target_dir: str, audio_stream: int = 0, video_stream: int = -1,
                            mono_mix: bool = True, decimate: bool = True, playlist_name: Optional[str] = None,
                            output_file_name: Optional[str] = None,
                            on_progress: Optional[Callable[[int, int], None]] = None) -> ExtractResult:
        '''
        Same as extract(), but (a) lets a caller fix the output filename instead of ffmpeg's auto-derived one
        (output_file_name is given *without* an extension -- Executor appends the format's own extension,
        '.wav' by default) and (b) also returns the source's detected channel layout name
        (Executor.channel_layout_name, the same one model/batch.py's ExtractCandidate.design() already reads
        off its own Executor) -- for a caller (the extract cache, design/archive/library-sync-pipeline-plan.md §4.1)
        that needs to record it without a second, separate probe. extract() itself keeps returning a bare path
        unchanged -- every other existing caller has no use for the layout and a changed return type would
        break them.
        :raises ValueError: if src has no audio stream, or (BD input) no matching/parseable title is found.
        '''
        os.makedirs(target_dir, exist_ok=True)
        display_name = None
        duration_override_s = None
        input_options = None
        bdmv_root = bdmv_root_of(src)
        if bdmv_root is not None:
            resolved = resolve_main_title(bdmv_root, playlist_name=playlist_name)
            src = resolved.ffmpeg_input
            display_name = resolved.display_name
            duration_override_s = resolved.playlist.extraction_duration_s
        elif dvd_root(src) is not None:
            resolved = resolve_main_dvd_title(src, title_name=playlist_name)
            src = resolved.ffmpeg_input
            display_name = resolved.display_name
            duration_override_s = resolved.playlist.duration_s
            input_options = resolved.input_options
        executor = Executor(src, target_dir, mono_mix=mono_mix, decimate_audio=decimate,
                            decimate_fs=self.__config.target_fs, display_name=display_name,
                            duration_override_s=duration_override_s, input_options=input_options)
        if on_progress is not None:
            def report_ffmpeg_progress(key, value):
                if key == 'progress' and value == 'end' and executor.duration_micros:
                    # finished: older ffmpeg (6.1) reports out_time 0 throughout for a filter_complex output, even here
                    on_progress(executor.duration_micros, executor.duration_micros)
                    return
                if key != 'out_time_ms' or value in (None, 'N/A'):
                    return
                try:
                    on_progress(int(value), executor.duration_micros)
                except ValueError:
                    return
            executor.progress_handler = report_ffmpeg_progress
        if output_file_name is not None:
            executor.output_file_name = output_file_name
        executor.probe_file()
        if not executor.has_audio():
            raise ValueError(f"{src} has no audio stream to extract")
        executor.update_spec(audio_stream, video_stream, mono_mix)
        executor.run_sync()
        return ExtractResult(wav_path=executor.get_output_path(), channel_layout_name=executor.channel_layout_name,
                             channel_count=executor.channel_count, mono_mix_spec=executor.mono_mix_spec)

    def extract(self, src: str, target_dir: str, audio_stream: int = 0, video_stream: int = -1,
               mono_mix: bool = True, decimate: bool = True, playlist_name: Optional[str] = None) -> str:
        '''
        Runs ffmpeg synchronously (model.ffmpeg.Executor.run_sync, B2). If src is a BD disc rip folder (a BDMV
        structure) rather than a single container file, it is first resolved to a concrete ffmpeg input -- the
        main feature (longest playlist) unless playlist_name names a specific one (its BDMV/PLAYLIST/*.mpls
        basename, e.g. '00800'). A DVD rip (a VIDEO_TS folder, or the folder holding one) is resolved the same
        way from its own title table, read through ffmpeg's dvdvideo demuxer; playlist_name is then a title
        number such as '3'. Note that a multi-episode DVD's longest title is usually "play all", so name the
        title to extract one episode. There is no interactive title picker here (unlike ui/extract.py's
        BdmvTitlePickerDialog) since this path is headless/unattended by design.
        Unchanged signature/behaviour/return type -- now a thin wrapper over extract_with_layout(), so this
        and extract_with_layout() can never behaviourally diverge.
        :return: the path to the extracted wav.
        :raises ValueError: if src has no audio stream, or (BD input) no matching/parseable title is found.
        '''
        return self.extract_with_layout(src, target_dir, audio_stream, video_stream, mono_mix, decimate,
                                        playlist_name).wav_path

    def load(self, path: str, name: Optional[str] = None, decimate: bool = True,
            offset: float = 0.0) -> SingleChannelSignalData:
        ''' Loads a (mono) wav file -- model.signal.AutoWavLoader, backed by AnalysisConfig instead of QSettings. '''
        loader = AutoWavLoader(self.__preferences)
        loader.load(path)
        default_name = name or os.path.splitext(os.path.basename(path))[0]
        return loader.auto_load(lambda idx, count: default_name, decimate=decimate, offset=offset)

    def load_channels(self, path: str, channel_layout_name: str = 'unknown', decimate: bool = True) -> dict:
        '''
        Decomposes a *multichannel* wav into named per-channel arrays -- design/designer-interface.md
        §2's DesignRequest.channels input ("a per-channel decomposition of the same underlying audio...
        the same length as mono_mix and time-aligned with it... a diagnostic input, not a replacement for
        mono_mix"). Pure decomposition, no downmix/gain-staging math -- that stays load()/extract()'s job
        (mono_mix must match the ffmpeg mono downmix everywhere else in this app, not be reimplemented
        here). Labels come from model.ffmpeg.get_channel_name/CHANNEL_LAYOUTS -- the same layout-dependent
        FL/FR/FC/LFE/... labels the GUI already assigns a multichannel extraction's per-channel signals.
        :param channel_layout_name: the source's ffmpeg channel layout name (Executor.channel_layout_name,
            populated once update_spec() has run), e.g. '5.1'.
        :return: {label: ndarray}, or {} if path is actually mono (nothing to decompose).
        '''
        from model.ffmpeg import get_channel_name
        loader = AutoWavLoader(self.__preferences)
        loader.load(path)
        channel_count = loader.info.channels
        if channel_count <= 1:
            return {}
        channels = {}
        for idx in range(channel_count):
            label = get_channel_name(None, idx, channel_count, channel_layout_name=channel_layout_name)
            loader.prepare(channel=idx + 1, name=label, channel_count=channel_count, decimate=decimate)
            channels[label] = loader.get_signal(idx + 1, label).signal.samples
        return channels

    def load_channel_signals(self, path: str, name: Optional[str] = None,
                             channel_layout_name: str = 'unknown',
                             decimate: bool = True) -> List[SingleChannelSignalData]:
        '''
        Loads every channel of a (possibly multichannel) wav as its own SingleChannelSignalData, named
        "<name>_<channel-label>" (model.ffmpeg.get_channel_name -- same labelling load_channels() uses) --
        ready for set_filters()/enslave(), unlike load_channels()'s raw decomposed arrays. A single-element
        list if path is actually mono.

        Calls AutoWavLoader.prepare()/get_signal() directly, once per channel, so callers can
        link the filter before wrapping the channels in a bass-managed project.
        '''
        from model.ffmpeg import get_channel_name
        default_name = name or os.path.splitext(os.path.basename(path))[0]
        loader = AutoWavLoader(self.__preferences)
        loader.load(path)
        channel_count = loader.info.channels
        signals = []
        for idx in range(channel_count):
            channel_name = get_channel_name(default_name, idx, channel_count, channel_layout_name=channel_layout_name)
            loader.prepare(channel=idx + 1, name=channel_name, channel_count=channel_count, decimate=decimate)
            signals.append(loader.get_signal(idx + 1, channel_name))
        return signals

    def design(self, sig: SingleChannelSignalData, designer: str, coverage: Coverage = 'complete_programme',
              bass_management: Optional[dict] = None, channels: Optional[dict] = None,
              sources: Optional[AudioSources] = None) -> DesignOutcome:
        '''
        Invokes a registered designer (pipeline.designer.registry) with a
        DesignRequest built from sig, validates and converts its response
        (pipeline.designer.convert), and returns the outcome as a value.
        Does not mutate sig -- call set_filters(sig, outcome.filters) on an
        Applied outcome to actually apply it. A designer may return several
        ranked candidates (design/designer-interface.md §3); only the
        top-ranked one becomes Applied.filters (the only one ever simulated/
        published automatically) -- the rest travel along as
        Applied.alternatives, for a human reviewing the report. Designs the
        designer judged unfit to publish (1.1's `rejected`) travel as
        `.rejected` on either outcome -- a decline's are its evidence.
        :param bass_management: this session's bass-management configuration
            (design/designer-interface.md §2), if any -- passed straight
            through to the designer as DesignRequest.bass_management. None
            (the default) if there is none to report; this session does not
            infer one from sig, which is always single-channel.
        :param channels: DesignRequest.channels (design/designer-interface.md §2) -- a per-channel
            diagnostic decomposition of the same signal sig was mixed down from, if the caller has one
            (see load_channels()). Optional; None if the source was mono to begin with, or the caller
            chose not to supply it.
        :param sources: the WAV columns mono_mix and channels were loaded from (pipeline.designer.sources), if known --
            passed on only to a designer registered as taking them, which may then send the arrays by reference
            (design/designer-interface.md §7.1, 1.2). Every other designer gets the request alone.
        '''
        request = build_request(mono_mix=sig.signal.samples, fs=sig.signal.fs, coverage=coverage,
                                channels=channels, bass_management=bass_management)
        if designer == MANUAL_DESIGNER:
            response = manual_response()
        elif sources is not None and takes_sources(designer):
            response = get_designer(designer)(request, sources=sources)
        else:
            response = get_designer(designer)(request)
        fs = sig.signal.fs
        rejected = tuple(_design_of(candidate, rejected_filter)
                         for candidate, rejected_filter in zip(response.rejected or [], rejected_filters(response, fs)))
        if response.decline_reason is not None:
            return Declined(reason=response.decline_reason, message=response.decline_message, rejected=rejected)
        primary = response.candidates[0]
        complete_filter = to_complete_filter(response, fs=fs)
        alternatives = tuple(_design_of(candidate, alt_filter)
                             for candidate, alt_filter in zip(response.candidates[1:], alternative_filters(response, fs=fs)))
        return Applied(filters=complete_filter, confidence=primary.confidence, method=primary.method,
                       mv_adjust_db=primary.mv_adjust_db, gain_reduction_db=primary.gain_reduction_db,
                       fc_hz=primary.fc_hz, slope=primary.slope,
                       fc_uncertainty_hz=primary.fc_uncertainty_hz, slope_uncertainty=primary.slope_uncertainty,
                       residual_db=primary.residual_db, residual_band_hz=primary.residual_band_hz,
                       commentary=primary.commentary, channel_scope=primary.channel_scope,
                       alternatives=alternatives, rejected=rejected)

    def set_filters(self, sig: SingleChannelSignalData,
                    filters: Union[CompleteFilter, Sequence[FilterSpec]]) -> CompleteFilter:
        '''
        Applies a filter to sig -- either an Applied outcome's CompleteFilter
        (the designer path) or a hand-built list[FilterSpec] (the manual
        path, §9) -- both go through the same pipeline.filters.create_filter.
        '''
        if isinstance(filters, CompleteFilter):
            complete_filter = filters
        else:
            biquads = [create_filter(spec, sig.signal.fs) for spec in filters]
            complete_filter = CompleteFilter(fs=sig.signal.fs, filters=biquads)
        sig.filter = complete_filter
        return complete_filter

    def curves(self, sig: SingleChannelSignalData, kind: str = 'avg', filtered: bool = False) -> MagnitudeData:
        ''' :param kind: 'avg', 'peak' or 'median'. '''
        source = sig.current_filtered if filtered else sig.current_unfiltered
        return source[_CURVE_INDEX[kind]]

    def stats(self, sig: SingleChannelSignalData, filtered: bool = False) -> Stats:
        ''' peak/rms/crest/headroom (B5) -- of the raw signal, or the filtered signal if filtered=True. '''
        if filtered:
            filtered_signal = sig.filter_signal(filt=True, clip=False)
            return signal_stats(filtered_signal.samples, filtered_signal.fs)
        return signal_stats(sig.signal.samples, sig.signal.fs)

    def tmdb(self, title: str, year: str, api_key: str, kind: str = 'movie',
            audio_types: Optional[Sequence[str]] = None) -> BeqMetadata:
        return tmdb_lookup(title, year, api_key, kind=kind, audio_types=list(audio_types or []))

    def to_beq_xml(self, filters, meta: BeqMetadata) -> str:
        return render_beq_xml(filters, meta)

    def to_catalogue_record(self, filters, meta: BeqMetadata, existing: Optional[dict] = None) -> dict:
        '''Render the version-1 BEQCatalogue source record, not a project file.'''
        return filter_record(filters, meta, existing=existing)

    def report(self, curves: Sequence[MagnitudeData], filters, meta: Optional[BeqMetadata] = None,
              poster_path: Optional[str] = None, spec: ReportSpec = ReportSpec(),
              mv_offset: Optional[float] = None) -> bytes:
        title = meta.title if meta is not None else ''
        return render_report(curves, list(filters), poster_path=poster_path, spec=spec, title=title,
                             mv_offset=mv_offset)

    def publish(self, filters, meta: BeqMetadata, xml_repo: RepoTarget, xml_relative_path: str,
               images_repo: Optional[RepoTarget] = None, image_relative_path: Optional[str] = None,
               image_png: Optional[bytes] = None, image_owner: Optional[str] = None,
               image_repo_name: Optional[str] = None, push: bool = True, write_aggregate: bool = True,
               heatmap_png: Optional[bytes] = None, heatmap_relative_path: Optional[str] = None,
               record_dir: Optional[str] = None) -> dict:
        '''
        Sequences the image-then-record publish order pipeline.publish.git
        requires: the report image goes in first (if given) so its raw URL
        can be written into meta.spectrum_url/.pva_url *before* the XML is
        rendered -- beqcatalogue never reads an image out of the filter repo
        itself (design/archive/api-headless-pipeline.md §6, D3).
        :param push: True (the default) commits and pushes each file as it is written. False only **writes** the
            image and JSON record into the repos' working trees -- the image URL needs no push (it is built from the
            remote's owner, repo and branch) -- leaving pipeline.library.commit to commit and push a whole batch.
        :param write_aggregate: False leaves the directory's `database.json` alone (with push=False only): a batch
            writes it once at the end with write_aggregate_for() rather than reading every record again per title.
        :param heatmap_png/heatmap_relative_path: the heatmap image, written beside the report image; its URL is the
            record's second image (`spectrum_url`) where the report image is the first (`pva_url`).
        :param record_dir: the folder whose `database.json` aggregates this record: by default the record's own folder;
            a record in a letter folder names the category folder above it.
        :return: {'record': the rendered object, 'filter_commit': its commit sha (push only),
            'image_url': the image's raw URL, if an image was published}.
        '''
        result = {}
        if image_png is not None:
            if images_repo is None or image_relative_path is None:
                raise ValueError("image_png given without images_repo/image_relative_path")
            if push:
                image_url = push_image(image_png, images_repo, image_relative_path, owner=image_owner,
                                       repo_name=image_repo_name)
            else:
                write_files(images_repo, {image_relative_path: image_png})
                image_url = git_image_url(images_repo, image_relative_path, image_owner, image_repo_name)
            meta.spectrum_url = image_url
            meta.pva_url = image_url
            result['image_url'] = image_url
            if heatmap_png is not None and heatmap_relative_path is not None:
                if push:
                    meta.spectrum_url = push_image(heatmap_png, images_repo, heatmap_relative_path, owner=image_owner,
                                                   repo_name=image_repo_name)
                else:
                    write_files(images_repo, {heatmap_relative_path: heatmap_png})
                    meta.spectrum_url = git_image_url(images_repo, heatmap_relative_path, image_owner, image_repo_name)
                result['heatmap_url'] = meta.spectrum_url

        existing = None
        try:
            with open(fs_path(xml_repo, xml_relative_path), encoding='utf-8') as handle:
                existing = json.load(handle)
        except (OSError, ValueError):
            pass
        record = self.to_catalogue_record(filters, meta, existing)
        record_bytes = (json.dumps(record, indent=2, ensure_ascii=False) + '\n').encode('utf-8')
        if record_dir is None:
            record_dir = os.path.dirname(xml_relative_path).replace(os.sep, '/')
        result['record'] = record
        files = {xml_relative_path: record_bytes}
        if push or write_aggregate:
            files[aggregate_path(record_dir)] = aggregate(
                {**read_records(xml_repo, record_dir), xml_relative_path: record})
        write_files(xml_repo, files)
        if push:
            result['filter_commit'] = commit_paths(xml_repo, list(files), f'Publish BEQ filter: {meta.title}')
            push_repo(xml_repo)
        return result
