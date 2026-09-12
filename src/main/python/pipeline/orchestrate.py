'''
pipeline/orchestrate.py: the Session facade -- design/pipeline-implementation-
plan.md phase 5 (item 11), composing every piece built in phases 0-4 per
design/api-headless-pipeline.md §9.

Beyond composition, this module owns two things nothing built so far needed:

- **Applied/Declined outcome handling (§15.4).** `Session.design()` returns
  one or the other, explicitly -- `Declined` is a value, not an exception,
  and nothing downstream (set_filters, stats, to_beq_xml, publish) should
  run on it. "We correctly concluded nothing should be applied" is a
  first-class result, not an error, matching the catalogue's own framing
  (15,208 positives and zero negatives -- a false positive is the expensive
  failure).
- **D7's provenance threading.** A designer's confidence/method/fc_hz/
  slope/uncertainties/residual_db travel with an `Applied` outcome so a
  report can show *why* a filter was accepted -- but `to_beq_xml()` only
  ever receives `Applied.filters`, never the outcome itself: the catalogue
  XML format has no field for any of this, and no consumer other than a
  human reviewing the report needs it.

Scope trim: no `Session.fit()`/`optimise_filters()` wrapper -- no phase
built a Qt-free extraction of that GUI feature, and nothing in this plan's
acceptance criteria needs it. `Session.load()` assumes a single-channel
(mono) signal, matching this pipeline's mono_mix=True scope throughout
(design/api-headless-pipeline.md: "this pipeline always targets one
device"); a multi-channel file will load via the same call but nothing
past that point is designed to handle the multi-channel result.

Qt-boundary note: `Session.load()` reuses `model.signal.AutoWavLoader`,
which -- unlike every module built from scratch in phases 0-4 -- imports
qtpy.QtCore/QtWidgets and Qt-Designer dialog UI modules directly at the top
of model/signal.py (for its interactive dialogs, none of which this class
ever touches). This is the same kind of deliberate, disclosed reuse as
phase 3's report.py + model/magnitude.py, one level deeper: importing this
module still constructs no QApplication and needs no DISPLAY (verified by
test_pipeline_orchestrate.py), which is the guarantee design/api-headless-
pipeline.md §11's Qt-boundary test actually checks -- "pipeline/ never
imports qtpy" is enforced by AST-scanning this package's own literal
imports (none of which are qtpy), not the transitive closure of every
model/ module it reuses.
'''
import os
from dataclasses import dataclass
from typing import Optional, Sequence, Union

from model.bdmv import is_bdmv_root, resolve_main_title
from model.ffmpeg import Executor
from model.iir import CompleteFilter
from model.signal import AutoWavLoader, SingleChannelSignalData
from model.xy import MagnitudeData

from pipeline.config import AnalysisConfig
from pipeline.designer.contract import Coverage, build_request
from pipeline.designer.convert import to_complete_filter
from pipeline.designer.registry import get_designer
from pipeline.filters import FilterSpec, create_filter
from pipeline.metadata import BeqMetadata, tmdb_lookup
from pipeline.publish.git import RepoTarget, push_image, push_xml
from pipeline.publish.report import ReportSpec, render_report
from pipeline.publish.xml import to_beq_xml as render_beq_xml
from pipeline.stats import Stats, signal_stats

_CURVE_INDEX = {'avg': 0, 'peak': 1, 'median': 2}


@dataclass(frozen=True)
class Applied:
    ''' A designer produced a publishable filter -- design/api-headless-pipeline.md §15.4. '''
    filters: CompleteFilter
    confidence: float
    method: str
    mv_adjust_db: Optional[float] = None
    fc_hz: Optional[float] = None
    slope: Optional[float] = None
    fc_uncertainty_hz: Optional[float] = None
    slope_uncertainty: Optional[float] = None
    residual_db: Optional[float] = None
    residual_band_hz: Optional[tuple] = None


@dataclass(frozen=True)
class Declined:
    ''' The designer concluded nothing should be applied -- a first-class result, not an error. '''
    reason: str
    message: Optional[str] = None


DesignOutcome = Union[Applied, Declined]


class _ConfigPreferences:
    '''
    Maps AnalysisConfig onto the `.get(key)` interface model.signal's
    AutoWavLoader/readWav/Signal expect, in place of a real Preferences/
    QSettings object (same pattern as pipeline/publish/report.py's
    _SpecPreferences).
    '''
    def __init__(self, config: AnalysisConfig):
        self.__values = {
            'analysis/target_fs': config.target_fs,
            'analysis/resolution': config.resolution,
            'analysis/avg_window': config.avg_window,
            'analysis/peak_window': config.peak_window,
        }

    def get(self, key, default_if_unset=True):
        return self.__values.get(key)


class Session:
    '''
    Session-scoped facade over one signal at a time -- signals are
    expensive to load/analyse, filters are cheap to iterate, so this holds
    the loaded signal steady across design/set_filters/curves/stats calls
    rather than re-extracting or re-analysing on every step
    (design/api-headless-pipeline.md §9).
    '''
    def __init__(self, config: AnalysisConfig = AnalysisConfig()):
        self.__config = config
        self.__preferences = _ConfigPreferences(config)

    def extract(self, src: str, target_dir: str, audio_stream: int = 0, video_stream: int = -1,
               mono_mix: bool = True, decimate: bool = True, playlist_name: Optional[str] = None) -> str:
        '''
        Runs ffmpeg synchronously (model.ffmpeg.Executor.run_sync, B2). If src is a BD disc rip folder (a BDMV
        structure) rather than a single container file, it is first resolved to a concrete ffmpeg input -- the
        main feature (longest playlist) unless playlist_name names a specific one (its BDMV/PLAYLIST/*.mpls
        basename, e.g. '00800'). There is no interactive title picker here (unlike ui/extract.py's
        BdmvTitlePickerDialog) since this path is headless/unattended by design.
        :return: the path to the extracted wav.
        :raises ValueError: if src has no audio stream, or (BD input) no matching/parseable title is found.
        '''
        os.makedirs(target_dir, exist_ok=True)
        display_name = None
        duration_override_s = None
        if is_bdmv_root(src):
            resolved = resolve_main_title(src, playlist_name=playlist_name)
            src = resolved.ffmpeg_input
            display_name = resolved.display_name
            duration_override_s = resolved.playlist.duration_s
        executor = Executor(src, target_dir, mono_mix=mono_mix, decimate_audio=decimate,
                            decimate_fs=self.__config.target_fs, display_name=display_name,
                            duration_override_s=duration_override_s)
        executor.probe_file()
        if not executor.has_audio():
            raise ValueError(f"{src} has no audio stream to extract")
        executor.update_spec(audio_stream, video_stream, mono_mix)
        executor.run_sync()
        return executor.get_output_path()

    def load(self, path: str, name: Optional[str] = None, decimate: bool = True,
            offset: float = 0.0) -> SingleChannelSignalData:
        ''' Loads a (mono) wav file -- model.signal.AutoWavLoader, backed by AnalysisConfig instead of QSettings. '''
        loader = AutoWavLoader(self.__preferences)
        loader.load(path)
        default_name = name or os.path.splitext(os.path.basename(path))[0]
        return loader.auto_load(lambda idx, count: default_name, decimate=decimate, offset=offset)

    def design(self, sig: SingleChannelSignalData, designer: str,
              coverage: Coverage = 'complete_programme') -> DesignOutcome:
        '''
        Invokes a registered designer (pipeline.designer.registry) with a
        DesignRequest built from sig, validates and converts its response
        (pipeline.designer.convert), and returns the outcome as a value.
        Does not mutate sig -- call set_filters(sig, outcome.filters) on an
        Applied outcome to actually apply it.
        '''
        designer_fn = get_designer(designer)
        request = build_request(mono_mix=sig.signal.samples, fs=sig.signal.fs, coverage=coverage)
        response = designer_fn(request)
        if response.decline_reason is not None:
            return Declined(reason=response.decline_reason, message=response.decline_message)
        complete_filter = to_complete_filter(response, fs=sig.signal.fs)
        return Applied(filters=complete_filter, confidence=response.confidence, method=response.method,
                       mv_adjust_db=response.mv_adjust_db, fc_hz=response.fc_hz, slope=response.slope,
                       fc_uncertainty_hz=response.fc_uncertainty_hz, slope_uncertainty=response.slope_uncertainty,
                       residual_db=response.residual_db, residual_band_hz=response.residual_band_hz)

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

    def report(self, curves: Sequence[MagnitudeData], filters, meta: Optional[BeqMetadata] = None,
              poster_path: Optional[str] = None, spec: ReportSpec = ReportSpec(),
              mv_offset: Optional[float] = None) -> bytes:
        title = meta.title if meta is not None else ''
        return render_report(curves, list(filters), poster_path=poster_path, spec=spec, title=title,
                             mv_offset=mv_offset)

    def publish(self, filters, meta: BeqMetadata, xml_repo: RepoTarget, xml_relative_path: str,
               images_repo: Optional[RepoTarget] = None, image_relative_path: Optional[str] = None,
               image_png: Optional[bytes] = None, image_owner: Optional[str] = None,
               image_repo_name: Optional[str] = None) -> dict:
        '''
        Sequences the image-then-XML publish order pipeline.publish.git
        requires: pushes the report image first (if given) so its raw URL
        can be written into meta.spectrum_url/.pva_url *before* the XML is
        rendered -- beqcatalogue never reads an image out of the XML repo
        itself (design/api-headless-pipeline.md §6, D3).
        :return: {'xml': the rendered XML, 'xml_commit': its commit sha,
            'image_url': the pushed image's raw URL, if an image was pushed}.
        '''
        result = {}
        if image_png is not None:
            if images_repo is None or image_relative_path is None:
                raise ValueError("image_png given without images_repo/image_relative_path")
            image_url = push_image(image_png, images_repo, image_relative_path, owner=image_owner,
                                   repo_name=image_repo_name)
            meta.spectrum_url = image_url
            meta.pva_url = image_url
            result['image_url'] = image_url

        xml = self.to_beq_xml(filters, meta)
        result['xml'] = xml
        result['xml_commit'] = push_xml(xml, xml_repo, xml_relative_path)
        return result
