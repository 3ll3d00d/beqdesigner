import os
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

from pipeline.designer.http_binding import http_designer
from pipeline.designer.registry import register_designer, registered_designers, unregister_designer

if TYPE_CHECKING:  # Qt-free at runtime: the headless pipeline imports this module (the dialog is model.preferences_dialog)
    from qtpy.QtCore import QSettings

X_RESOLUTION = 32769

WINDOWS = ['barthann', 'bartlett', 'blackman', 'blackmanharris', 'bohman', 'boxcar', 'cosine', 'flattop', 'hamming',
           'hann', 'nuttall', 'parzen', 'triang', 'tukey']

SPECTROGRAM_FLAT = 'spectrogram (flat)'
SPECTROGRAM_CONTOURED = 'spectrogram (contoured)'
ELLIPSE = 'ellipse'
POINT = 'point'

SHOW_ALL_FILTERS = 'Individual'
SHOW_COMBINED_FILTER = 'Combined'
SHOW_NO_FILTERS = 'None'
SHOW_FILTER_OPTIONS = [SHOW_ALL_FILTERS, SHOW_COMBINED_FILTER, SHOW_NO_FILTERS]

SHOW_ALL_SIGNALS = 'All'
SHOW_PEAK_MEDIAN = 'Peak & Median'
SHOW_PEAK_AVERAGE = 'Peak & Average'
SHOW_MEDIAN_AVERAGE = 'Average & Median'
SHOW_PEAK = 'Peak'
SHOW_AVERAGE = 'Average'
SHOW_MEDIAN = 'Median'
SHOW_SIGNAL_OPTIONS = [
    SHOW_ALL_SIGNALS,
    SHOW_PEAK_MEDIAN,
    SHOW_PEAK_AVERAGE,
    SHOW_MEDIAN_AVERAGE,
    SHOW_PEAK,
    SHOW_AVERAGE,
    SHOW_MEDIAN
]
SHOWING_AVERAGE = [SHOW_ALL_SIGNALS, SHOW_PEAK_AVERAGE, SHOW_MEDIAN_AVERAGE, SHOW_AVERAGE]
SHOWING_MEDIAN = [SHOW_ALL_SIGNALS, SHOW_PEAK_MEDIAN, SHOW_MEDIAN_AVERAGE, SHOW_MEDIAN]
SHOWING_PEAK = [SHOW_ALL_SIGNALS, SHOW_PEAK_MEDIAN, SHOW_PEAK_AVERAGE, SHOW_PEAK]

SHOW_ALL_FILTERED_SIGNALS = 'Both'
SHOW_FILTERED_ONLY = 'Yes'
SHOW_UNFILTERED_ONLY = 'No'
SHOW_FILTERED_SIGNAL_OPTIONS = [SHOW_ALL_FILTERED_SIGNALS, SHOW_FILTERED_ONLY, SHOW_UNFILTERED_ONLY]

BM_LPF_BEFORE = 'Before'
BM_LPF_AFTER = 'After'
BM_LPF_OFF = 'Off'
BM_LPF_OPTIONS = [BM_LPF_BEFORE, BM_LPF_AFTER, BM_LPF_OFF]

COMPRESS_FORMAT_NATIVE = 'Native'
COMPRESS_FORMAT_FLAC = 'FLAC'
COMPRESS_FORMAT_EAC3 = 'EAC3'
COMPRESS_FORMAT_AC3 = 'AC3'
COMPRESS_FORMAT_OPTIONS = [COMPRESS_FORMAT_NATIVE, COMPRESS_FORMAT_FLAC]

APP_FONT_SIZE = 'app/font_size'

EXTRACTION_OUTPUT_DIR = 'extraction/output_dir'
EXTRACTION_NOTIFICATION_SOUND = 'extraction/notification_sound'
EXTRACTION_BATCH_FILTER = 'extraction/batch_filter'
EXTRACTION_MIX_MONO = 'extraction/mix_to_mono'
EXTRACTION_DECIMATE = 'extraction/decimate'
EXTRACTION_INCLUDE_ORIGINAL = 'extraction/include_original'
EXTRACTION_INCLUDE_SUBTITLES = 'extraction/include_subtitles'
EXTRACTION_COMPRESS = 'extraction/compress'
EXTRACTION_COMPRESS_FORMAT = 'extraction/compress_format'
EXTRACTION_GEOMETRY = 'extraction/geometry'

DESIGNER_HTTP_ENDPOINTS = 'designers/http_endpoints'
DESIGNER_QUEUE_DIR = 'designers/queue_dir'
DESIGNER_DEFAULT = 'designers/default'
LIBRARY_WORK_DIR = 'library/work_dir'
LIBRARY_XML_REPO = 'library/xml_repo'
LIBRARY_FILTER_REPO = 'library/filter_repo'
LIBRARY_IMAGES_REPO = 'library/images_repo'
LIBRARY_JRIVER_BROWSE_NODE = 'library/jriver_browse_node'
LIBRARY_JRIVER_BROWSE_PATH = 'library/jriver_browse_path'
LIBRARY_SOURCE_DEFAULT = 'library/source'
LIBRARY_TV_MODE = 'library/tv_mode'
LIBRARY_FILESYSTEM_GLOBS = 'library/filesystem_globs'
LIBRARY_JRIVER_CONNECTION = 'library/jriver_connection'
LIBRARY_PROFILE_PATH = 'library/profile_path'
WORKLIST_GEOMETRY = 'library/worklist_geometry'
WORKLIST_ACCEPT_THRESHOLD = 'library/accept_threshold'   # bulk accept takes the top pick at or above this confidence (27c)
WORKLIST_PUSH = 'library/worklist_push'   # the work list's Commit pushes each repository (the confirmation's checkbox)

# distinguishes preference-configured designers from ad-hoc/in-process ones (e.g. registered by a test or a
# script), so re-registering on every startup or Preferences-save replaces cleanly rather than accumulating
_REGISTERED_DESIGNER_PREFIX = 'http:'


def register_configured_designers(preferences):
    ''' Registers every endpoint in DESIGNER_HTTP_ENDPOINTS -- call once at app startup. '''
    for entry in preferences.get(DESIGNER_HTTP_ENDPOINTS):
        name = f"{_REGISTERED_DESIGNER_PREFIX}{entry['name']}"
        register_designer(name, http_designer(entry['url'], headers=entry.get('headers') or None))


def _unregister_configured_designers():
    for name in list(registered_designers()):
        if name.startswith(_REGISTERED_DESIGNER_PREFIX):
            unregister_designer(name)

ANALYSIS_RESOLUTION = 'analysis/resolution'
ANALYSIS_RESOLUTION_DEFAULT = 1.0
ANALYSIS_TARGET_FS = 'analysis/target_fs'
ANALYSIS_WINDOW_DEFAULT = 'Default'
ANALYSIS_AVG_WINDOW = 'analysis/avg_window'
ANALYSIS_PEAK_WINDOW = 'analysis/peak_window'

AUDIO_ANALYSIS_MARKER_SIZE = 'audio/marker_size'
AUDIO_ANALYSIS_MARKER_TYPE = 'audio/marker_type'
AUDIO_ANALYSIS_ELLIPSE_WIDTH = 'audio/ellipse_width'
AUDIO_ANALYSIS_ELLIPSE_HEIGHT = 'audio/ellipse_height'
AUDIO_ANALYIS_MIN_FREQ = 'audio/min_freq'
AUDIO_ANALYIS_MAX_UNFILTERED_FREQ = 'audio/max_unfiltered_freq'
AUDIO_ANALYIS_MAX_FILTERED_FREQ = 'audio/max_filtered_freq'
AUDIO_ANALYSIS_COLOUR_MAX = 'audio/colour_max'
AUDIO_ANALYSIS_COLOUR_MIN = 'audio/colour_min'
AUDIO_ANALYSIS_SIGNAL_MIN = 'audio/signal_min'
AUDIO_ANALYSIS_GEOMETRY = 'audio/geometry'

BINARIES_GROUP = 'binaries'
BINARIES_FFPROBE = f"{BINARIES_GROUP}/ffprobe"
BINARIES_FFMPEG = f"{BINARIES_GROUP}/ffmpeg"
BINARIES_MINIDSP_RS = f"{BINARIES_GROUP}/minidsprs"

FILTERS_PRESET_x = 'filters/preset_%d'
FILTERS_DEFAULT_Q = 'filters/defaults/q'
FILTERS_DEFAULT_FREQ = 'filters/defaults/freq'
FILTERS_DEFAULT_HS_Q = 'filters/defaults/hs_q'
FILTERS_DEFAULT_HS_FREQ = 'filters/defaults/hs_freq'
FILTERS_DEFAULT_PEAK_Q = 'filters/defaults/peak_q'
FILTERS_DEFAULT_PEAK_FREQ = 'filters/defaults/peak_freq'
FILTERS_GEOMETRY = 'filters/geometry'
FILTERS_GEOMETRY_SMALL = 'filters/geometry_small'

SCREEN_GEOMETRY = 'screen/geometry'
SCREEN_WINDOW_STATE = 'screen/window_state'

STYLE_MATPLOTLIB_THEME_DEFAULT = 'beq_dark'
STYLE_MATPLOTLIB_THEME = 'style/matplotlib_theme'
STYLE_IMAGE_FORMAT_DEFAULT = 'style/image_format'

DISPLAY_SHOW_LEGEND = 'display/show_legend'
DISPLAY_SHOW_FILTERS = 'display/show_filters'
DISPLAY_SHOW_SIGNALS = 'display/show_signals'
DISPLAY_SHOW_FILTERED_SIGNALS = 'display/show_filtered_signals'
DISPLAY_FREQ_STEP = 'display/freq_step'
DISPLAY_Q_STEP = 'display/q_step'
DISPLAY_S_STEP = 'display/s_step'
DISPLAY_GAIN_STEP = 'display/gain_step'
DISPLAY_LINE_STYLE = 'display/line_style'
DISPLAY_SMOOTH_PRECALC = 'display/precalc_smooth'
DISPLAY_SMOOTH_GRAPHS = 'display/smooth_graphs'

GRAPH_X_AXIS_SCALE = 'graph/x_axis'
GRAPH_X_MIN = 'graph/x_min'
GRAPH_X_MAX = 'graph/x_max'
GRAPH_EXPAND_Y = 'graph/expand_y'

REPORT_GROUP = 'report'
REPORT_TITLE_FONT_SIZE = 'report/title_font_size'
REPORT_IMAGE_ALPHA = 'report/image/alpha'
REPORT_IMAGE_WIDTH = 'report/image/width'
REPORT_IMAGE_HEIGHT = 'report/image/height'
REPORT_FILTER_ROW_HEIGHT_MULTIPLIER = 'report/filter/row_height'
REPORT_FILTER_X0 = 'report/filter/x0'
REPORT_FILTER_X1 = 'report/filter/x1'
REPORT_FILTER_Y0 = 'report/filter/y0'
REPORT_FILTER_Y1 = 'report/filter/y1'
REPORT_FILTER_FONT_SIZE = 'report/filter/font_size'
REPORT_FILTER_SHOW_HEADER = 'report/filter/show_header'
REPORT_LAYOUT_MAJOR_RATIO = 'report/layout/major_ratio'
REPORT_LAYOUT_MINOR_RATIO = 'report/layout/minor_ratio'
REPORT_LAYOUT_SPLIT_DIRECTION = 'report/layout/split_direction'
REPORT_LAYOUT_WSPACE = 'report/layout/wspace'
REPORT_LAYOUT_HSPACE = 'report/layout/hspace'
REPORT_LAYOUT_TYPE = 'report/layout/type'
REPORT_CHART_GRID_ALPHA = 'report/chart/grid_alpha'
REPORT_CHART_SHOW_LEGEND = 'report/chart/show_legend'
REPORT_CHART_LIMITS_X0 = 'report/chart/limits_x0'
REPORT_CHART_LIMITS_X1 = 'report/chart/limits_x1'
REPORT_CHART_LIMITS_Y0 = 'report/chart/limits_y0'
REPORT_CHART_LIMITS_Y1 = 'report/chart/limits_y1'
REPORT_CHART_LIMITS_X_SCALE = 'report/chart/limits_x_scale'
REPORT_GEOMETRY = 'report/geometry'

LOGGING_LEVEL = 'logging/level'

SYSTEM_CHECK_FOR_UPDATES = 'system/check_for_updates'
SYSTEM_CHECK_FOR_BETA_UPDATES = 'system/check_for_beta_updates'

BEQ_DOWNLOAD_DIR = 'beq/directory'
BEQ_MERGE_DIR = 'beq/merge_dir'
BEQ_CONFIG_FILE = 'beq/config_file'
BEQ_EXTRA_DIR = 'beq/extra_dir'
BEQ_MINIDSP_TYPE = 'beq/minidsp_type'
BEQ_OUTPUT_CHANNELS = 'beq/output_channels'
BEQ_OUTPUT_MODE = 'beq/output_mode'

TMDB_API_KEY = 'tmdb/api_key'

BIQUAD_EXPORT_FS = 'biquad/fs'
BIQUAD_EXPORT_MAX = 'biquad/max'
BIQUAD_EXPORT_DEVICE = 'biquad/device'

BASS_MANAGEMENT_LPF_FS = 'bm/fs'
BASS_MANAGEMENT_LPF_POSITION = 'bm/type'

HTP1_ADDRESS = 'htp1/address'
HTP1_AUTOSYNC = 'htp1/autosync'
HTP1_SYNC_GEOMETRY = 'htp1/geometry'
HTP1_GRAPH_X_MIN = 'htp1/x_min'
HTP1_GRAPH_X_MAX = 'htp1/x_max'

JRIVER_GEOMETRY = 'jriver/geometry'
JRIVER_GRAPH_X_MIN = 'jriver/x_min'
JRIVER_GRAPH_X_MAX = 'jriver/x_max'
JRIVER_DSP_DIR = 'jriver/dsp_dir'
JRIVER_MCWS_CONNECTIONS = 'jriver/mcws'
JRIVER_MCWS_ALIASES = 'jriver/mcws_aliases'
JRIVER_MCWS_PATH_MAPPINGS = 'jriver/mcws_path_mappings'
JRIVER_MCWS_FIELD_MAPPINGS = 'jriver/mcws_field_mappings'

GEQ_GEOMETRY = 'geq/geometry'
GEQ_GRAPH_X_MIN = 'geq/x_min'
GEQ_GRAPH_X_MAX = 'geq/x_max'

XO_GEOMETRY = 'geq/geometry'
XO_GRAPH_X_MIN = 'geq/x_min'
XO_GRAPH_X_MAX = 'geq/x_max'

IMPULSE_GEOMETRY = 'impulse/geometry'
IMPULSE_GRAPH_X_MIN = 'impulse/x_min'
IMPULSE_GRAPH_X_MAX = 'impulse/x_max'

POST_GEOMETRY = 'post/geometry'

MINIDSP_RS_OPTIONS = 'minidsp/rs_options'

DEFAULT_PREFS = {
    DESIGNER_HTTP_ENDPOINTS: [],
    DESIGNER_QUEUE_DIR: '',
    DESIGNER_DEFAULT: '',
    LIBRARY_WORK_DIR: '',
    LIBRARY_XML_REPO: '',
    LIBRARY_FILTER_REPO: '',
    LIBRARY_IMAGES_REPO: '',
    LIBRARY_JRIVER_BROWSE_NODE: -1,
    LIBRARY_JRIVER_BROWSE_PATH: '',
    LIBRARY_SOURCE_DEFAULT: 'filesystem',
    LIBRARY_TV_MODE: 'episode',
    LIBRARY_FILESYSTEM_GLOBS: [],
    LIBRARY_JRIVER_CONNECTION: '',
    LIBRARY_PROFILE_PATH: '',
    WORKLIST_PUSH: True,
    WORKLIST_ACCEPT_THRESHOLD: 0.90,   # pipeline.library.bulk.DEFAULT_ACCEPT_THRESHOLD (a test keeps them equal)
    ANALYSIS_RESOLUTION: ANALYSIS_RESOLUTION_DEFAULT,
    ANALYSIS_TARGET_FS: 1000,
    ANALYSIS_AVG_WINDOW: ANALYSIS_WINDOW_DEFAULT,
    ANALYSIS_PEAK_WINDOW: ANALYSIS_WINDOW_DEFAULT,
    AUDIO_ANALYSIS_MARKER_SIZE: 1,
    AUDIO_ANALYSIS_MARKER_TYPE: POINT,
    AUDIO_ANALYSIS_ELLIPSE_WIDTH: 3.0,
    AUDIO_ANALYSIS_ELLIPSE_HEIGHT: 1.0,
    AUDIO_ANALYIS_MIN_FREQ: 1,
    AUDIO_ANALYIS_MAX_UNFILTERED_FREQ: 160,
    AUDIO_ANALYIS_MAX_FILTERED_FREQ: 40,
    AUDIO_ANALYSIS_COLOUR_MAX: -10,
    AUDIO_ANALYSIS_COLOUR_MIN: -70,
    AUDIO_ANALYSIS_SIGNAL_MIN: -70.0,
    BASS_MANAGEMENT_LPF_FS: 80,
    BASS_MANAGEMENT_LPF_POSITION: BM_LPF_BEFORE,
    BEQ_DOWNLOAD_DIR: os.path.join(os.path.expanduser('~'), '.beq'),
    BEQ_MERGE_DIR: os.path.join(os.path.expanduser('~'), 'beq_minidsp'),
    TMDB_API_KEY: os.environ.get('BEQDESIGNER_TMDB_API_KEY', '5e23b4412adb55e7cca19cfb9d0196b6'),
    BIQUAD_EXPORT_FS: '48000',
    BIQUAD_EXPORT_MAX: 10,
    BIQUAD_EXPORT_DEVICE: 'Minidsp 2x4HD',
    STYLE_MATPLOTLIB_THEME: STYLE_MATPLOTLIB_THEME_DEFAULT,
    DISPLAY_SHOW_SIGNALS: SHOW_PEAK_AVERAGE,
    DISPLAY_SHOW_LEGEND: True,
    DISPLAY_SHOW_FILTERS: SHOW_ALL_FILTERS,
    DISPLAY_FREQ_STEP: '1',
    DISPLAY_Q_STEP: '0.1',
    DISPLAY_S_STEP: '0.1',
    DISPLAY_GAIN_STEP: '0.1',
    DISPLAY_LINE_STYLE: True,
    DISPLAY_SMOOTH_GRAPHS: True,
    DISPLAY_SMOOTH_PRECALC: False,
    EXTRACTION_OUTPUT_DIR: os.path.expanduser('~'),
    EXTRACTION_MIX_MONO: False,
    EXTRACTION_COMPRESS: False,
    EXTRACTION_COMPRESS_FORMAT: COMPRESS_FORMAT_NATIVE,
    EXTRACTION_DECIMATE: False,
    EXTRACTION_INCLUDE_ORIGINAL: False,
    EXTRACTION_INCLUDE_SUBTITLES: False,
    FILTERS_DEFAULT_FREQ: 20.0,
    FILTERS_DEFAULT_Q: 0.707,
    FILTERS_DEFAULT_HS_FREQ: 80.0,
    FILTERS_DEFAULT_HS_Q: 0.707,
    FILTERS_DEFAULT_PEAK_FREQ: 20.0,
    FILTERS_DEFAULT_PEAK_Q: 1.000,
    GEQ_GRAPH_X_MIN: 10,
    GEQ_GRAPH_X_MAX: 20000,
    GRAPH_X_AXIS_SCALE: 'log',
    GRAPH_X_MIN: 1,
    GRAPH_X_MAX: 160,
    GRAPH_EXPAND_Y: False,
    HTP1_ADDRESS: '127.0.0.1:80',
    HTP1_AUTOSYNC: False,
    HTP1_GRAPH_X_MIN: 10,
    HTP1_GRAPH_X_MAX: 20000,
    IMPULSE_GRAPH_X_MIN: -20,
    IMPULSE_GRAPH_X_MAX: 50,
    JRIVER_GRAPH_X_MIN: 10,
    JRIVER_GRAPH_X_MAX: 20000,
    JRIVER_DSP_DIR: str(Path.home()),
    JRIVER_MCWS_CONNECTIONS: {},
    JRIVER_MCWS_ALIASES: {},
    JRIVER_MCWS_PATH_MAPPINGS: {},
    JRIVER_MCWS_FIELD_MAPPINGS: {},
    REPORT_FILTER_ROW_HEIGHT_MULTIPLIER: 1.2,
    REPORT_TITLE_FONT_SIZE: 36,
    REPORT_IMAGE_ALPHA: 1.0,
    REPORT_FILTER_X0: 0.748,
    REPORT_FILTER_X1: 1.0,
    REPORT_FILTER_Y0: 0.75,
    REPORT_FILTER_Y1: 1.0,
    REPORT_LAYOUT_MAJOR_RATIO: 1.0,
    REPORT_LAYOUT_MINOR_RATIO: 2.0,
    REPORT_LAYOUT_SPLIT_DIRECTION: 'Vertical',
    REPORT_LAYOUT_TYPE: 'Image | Chart',
    REPORT_CHART_GRID_ALPHA: 0.5,
    REPORT_CHART_SHOW_LEGEND: False,
    REPORT_CHART_LIMITS_X0: 1,
    REPORT_CHART_LIMITS_X1: 160,
    REPORT_CHART_LIMITS_X_SCALE: 'linear',
    REPORT_FILTER_SHOW_HEADER: True,
    REPORT_FILTER_FONT_SIZE: matplotlib.rcParams['font.size'],
    REPORT_LAYOUT_HSPACE: matplotlib.rcParams['figure.subplot.hspace'],
    REPORT_LAYOUT_WSPACE: matplotlib.rcParams['figure.subplot.wspace'],
    STYLE_IMAGE_FORMAT_DEFAULT: 'png',
    SYSTEM_CHECK_FOR_UPDATES: True,
    SYSTEM_CHECK_FOR_BETA_UPDATES: False,
    XO_GRAPH_X_MIN: 10,
    XO_GRAPH_X_MAX: 20000,
}

TYPES = {
    DESIGNER_HTTP_ENDPOINTS: list,
    LIBRARY_FILESYSTEM_GLOBS: list,
    WORKLIST_PUSH: bool,
    WORKLIST_ACCEPT_THRESHOLD: float,
    ANALYSIS_RESOLUTION: float,
    ANALYSIS_TARGET_FS: int,
    APP_FONT_SIZE: float,
    AUDIO_ANALYSIS_MARKER_SIZE: int,
    AUDIO_ANALYSIS_ELLIPSE_WIDTH: float,
    AUDIO_ANALYSIS_ELLIPSE_HEIGHT: float,
    AUDIO_ANALYIS_MIN_FREQ: int,
    AUDIO_ANALYIS_MAX_UNFILTERED_FREQ: int,
    AUDIO_ANALYIS_MAX_FILTERED_FREQ: int,
    AUDIO_ANALYSIS_COLOUR_MAX: int,
    AUDIO_ANALYSIS_COLOUR_MIN: int,
    AUDIO_ANALYSIS_SIGNAL_MIN: float,
    BASS_MANAGEMENT_LPF_FS: int,
    BIQUAD_EXPORT_MAX: int,
    DISPLAY_SHOW_LEGEND: bool,
    DISPLAY_LINE_STYLE: bool,
    DISPLAY_SMOOTH_PRECALC: bool,
    DISPLAY_SMOOTH_GRAPHS: bool,
    EXTRACTION_MIX_MONO: bool,
    EXTRACTION_COMPRESS: bool,
    EXTRACTION_DECIMATE: bool,
    EXTRACTION_INCLUDE_ORIGINAL: bool,
    EXTRACTION_INCLUDE_SUBTITLES: bool,
    FILTERS_DEFAULT_FREQ: int,
    FILTERS_DEFAULT_Q: float,
    FILTERS_DEFAULT_HS_FREQ: int,
    FILTERS_DEFAULT_HS_Q: float,
    FILTERS_DEFAULT_PEAK_FREQ: int,
    FILTERS_DEFAULT_PEAK_Q: float,
    GEQ_GRAPH_X_MIN: int,
    GEQ_GRAPH_X_MAX: int,
    GRAPH_X_MIN: int,
    GRAPH_X_MAX: int,
    GRAPH_EXPAND_Y: bool,
    HTP1_AUTOSYNC: bool,
    JRIVER_GRAPH_X_MIN: int,
    JRIVER_GRAPH_X_MAX: int,
    JRIVER_MCWS_CONNECTIONS: dict,
    JRIVER_MCWS_ALIASES: dict,
    JRIVER_MCWS_PATH_MAPPINGS: dict,
    JRIVER_MCWS_FIELD_MAPPINGS: dict,
    REPORT_FILTER_ROW_HEIGHT_MULTIPLIER: float,
    REPORT_TITLE_FONT_SIZE: int,
    REPORT_IMAGE_ALPHA: float,
    REPORT_FILTER_X0: float,
    REPORT_FILTER_X1: float,
    REPORT_FILTER_Y0: float,
    REPORT_FILTER_Y1: float,
    REPORT_LAYOUT_MAJOR_RATIO: float,
    REPORT_LAYOUT_MINOR_RATIO: float,
    REPORT_LAYOUT_HSPACE: float,
    REPORT_LAYOUT_WSPACE: float,
    REPORT_CHART_GRID_ALPHA: float,
    REPORT_CHART_SHOW_LEGEND: bool,
    REPORT_CHART_LIMITS_X0: int,
    REPORT_CHART_LIMITS_X1: int,
    REPORT_FILTER_SHOW_HEADER: bool,
    REPORT_FILTER_FONT_SIZE: int,
    SYSTEM_CHECK_FOR_UPDATES: bool,
    SYSTEM_CHECK_FOR_BETA_UPDATES: bool,
    XO_GRAPH_X_MIN: int,
    XO_GRAPH_X_MAX: int,
}

COLOUR_INTERVALS = [x / 255 for x in range(36, 250, 24)] + [1.0]
# keep peak green, avg red, median blue and filters cyan
AVG_SPECLAB_COLOURS = [(x, 0.0, 0.0) for x in COLOUR_INTERVALS[::-1]]
PEAK_SPECLAB_COLOURS = [(0.0, x, 0.0) for x in COLOUR_INTERVALS[::-1]]
MEDIAN_SPECLAB_COLOURS = [(0.0, 0.0, x) for x in COLOUR_INTERVALS[::-1]]
FILTER_COLOURS = [(0.0, x, x) for x in COLOUR_INTERVALS[::-1]]

singleton = None


def get_avg_colour(idx):
    if singleton is None or singleton.get(DISPLAY_LINE_STYLE) is True:
        return AVG_SPECLAB_COLOURS[idx % len(AVG_SPECLAB_COLOURS)]
    else:
        colours = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
        return colours[idx % len(colours)]


def get_peak_colour(idx):
    if singleton is None or singleton.get(DISPLAY_LINE_STYLE) is True:
        return PEAK_SPECLAB_COLOURS[idx % len(PEAK_SPECLAB_COLOURS)]
    else:
        colours = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
        return colours[idx % len(colours)]


def get_median_colour(idx):
    if singleton is None or singleton.get(DISPLAY_LINE_STYLE) is True:
        return MEDIAN_SPECLAB_COLOURS[idx % len(MEDIAN_SPECLAB_COLOURS)]
    else:
        colours = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
        return colours[idx % len(colours)]


def get_filter_colour(idx):
    colours = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
    return colours[idx % len(colours)]
    # return FILTER_COLOURS[idx % len(FILTER_COLOURS)]


class Preferences:
    def __init__(self, settings: 'QSettings'):
        self.__settings = settings
        global singleton
        singleton = self

    def has(self, key):
        '''
        checks for existence of a value.
        :param key: the key.
        :return: True if we have a value.
        '''
        return self.get(key) is not None

    def get(self, key, default_if_unset=True):
        '''
        Gets the value, if any.
        :param key: the settings key.
        :param default_if_unset: if true, return a default value.
        :return: the value.
        '''
        default_value = DEFAULT_PREFS.get(key, None) if default_if_unset is True else None
        value_type = TYPES.get(key, None)
        if value_type is not None:
            return self.__settings.value(key, defaultValue=default_value, type=value_type)
        else:
            return self.__settings.value(key, defaultValue=default_value)

    def get_all(self, prefix):
        '''
        Get all values with the given prefix.
        :param prefix: the prefix.
        :return: the values, if any.
        '''
        self.__settings.beginGroup(prefix)
        try:
            return set(filter(None.__ne__, [self.__settings.value(x) for x in self.__settings.childKeys()]))
        finally:
            self.__settings.endGroup()

    def set(self, key, value):
        '''
        sets a new value.
        :param key: the key.
        :param value:  the value.
        '''
        if value is None:
            self.__settings.remove(key)
        else:
            self.__settings.setValue(key, value)
        self.__settings.sync()

    def clear_all(self, prefix):
        ''' clears all under the given group '''
        self.__settings.beginGroup(prefix)
        self.__settings.remove('')
        self.__settings.endGroup()

    def clear(self, key):
        '''
        Removes the stored value.
        :param key: the key.
        '''
        self.set(key, None)

    def reset(self):
        '''
        Resets all preferences.
        '''
        self.__settings.clear()
