"""
A title's spectrum comparison -- the publication heatmap of one design, filtered beside unfiltered, rendered exactly as
publish would without writing any project or repository (no widgets): the title page's *Spectrum comparison* tab
(`model.worklist_spectrum`) and the pipeline service's heatmap route (design/web-parity.md WP2).
"""
import json
import os
from dataclasses import dataclass
from typing import Optional

from model.codec import filter_from_json
from pipeline.config import AnalysisConfig
from pipeline.orchestrate import Session
from pipeline.publish.heatmap import HeatmapSpec, heatmap_for, spec_from_preferences
from pipeline.publish.project import ProjectFilterConflict, preview_published_projects
from pipeline.review import project_paths


@dataclass(frozen=True)
class ComparisonRequest:
    title_id: str
    title: str
    mono_wav: str
    mono_project: str
    multichannel_project: Optional[str]
    filters: str
    config: AnalysisConfig
    spec: HeatmapSpec
    files: tuple


def _stamp(path):
    try:
        stat = os.stat(path) if path else None
        return (stat.st_mtime_ns, stat.st_size) if stat else None
    except OSError:
        return None


def comparison_request(work_dir, entry, picked, title, config, preferences=None, spec=None) -> ComparisonRequest:
    """
    What to render for one design of a title. The heatmap's look is `spec`, else the desktop's `preferences`' one.
    :raises ValueError: (a sentence for a person) when there is no design, work directory or extracted mono audio.
    """
    if entry is None or not 0 <= picked < len(entry.offered):
        raise ValueError('No design to compare yet. Extract and design this title first.')
    if not work_dir:
        raise ValueError('No work directory is set. The comparison needs the extracted mono audio.')
    directory, mono, multichannel, _ = project_paths(work_dir, entry.id)
    wav = os.path.join(directory, 'mono.wav')
    if not os.path.isfile(wav):
        raise ValueError('No extracted mono audio is available for this title. Extract it again to view the comparison.')
    return ComparisonRequest(entry.id, title, wav, mono, multichannel,
                             json.dumps(entry.offered[picked].filters, sort_keys=True), config,
                             spec if spec is not None else spec_from_preferences(preferences), tuple(_stamp(p) for p in (wav, mono, multichannel)))


def render_comparison(request: ComparisonRequest) -> tuple[bytes, str]:
    """Publication's loading, project resolution and renderer, without writing any projects or repositories."""
    candidate = filter_from_json(json.loads(request.filters))
    try:
        published = preview_published_projects(request.mono_project, request.multichannel_project, candidate)
    except ProjectFilterConflict as error:
        raise ValueError('The edited mono and multichannel projects disagree. Resolve their filters before comparing.') from error
    signal = Session(request.config).load(request.mono_wav)
    png = heatmap_for(signal, published.filter, request.spec, title=request.title)
    source = f'Saved {published.edited_side} project edits' if published.edited_side else 'Selected design'
    return png, f'{source}. Filtered on the left; unfiltered on the right. Both panes reach 40 Hz.'


# --- the interactive spectrogram's data (design/web-parity.md WP2) ------------------------------------------------------

GRID_MAGIC = b'BEQS'
GRID_VERSION = 1
GRID_HEADER = '<4sHHII6f'   # magic, version, reserved, time bins, frequency bins, t0, dt, f0, df, dB floor, dB step
GRID_RANGE_DB = 100.0       # the levels kept below the loudest cell; quieter cells are 0 ("nothing")
MAX_TIME_BINS = 2048
MAX_FREQ_BINS = 512


@dataclass(frozen=True)
class SpectrogramGrid:
    ''' A spectrogram reduced for a browser: one byte per cell, time-major, each cell the loudest it stands for. '''
    t0: float
    dt: float
    f0: float
    df: float
    floor_db: float     # the level of cell value 1 (0 is below the floor)
    step_db: float      # dB per cell value
    times: int
    freqs: int
    cells: bytes        # times x freqs, time-major

    def encode(self) -> bytes:
        import struct
        return struct.pack(GRID_HEADER, GRID_MAGIC, GRID_VERSION, 0, self.times, self.freqs, self.t0, self.dt, self.f0,
                           self.df, self.floor_db, self.step_db) + self.cells

    @classmethod
    def decode(cls, data: bytes) -> 'SpectrogramGrid':
        import struct
        size = struct.calcsize(GRID_HEADER)
        magic, version, _, times, freqs, t0, dt, f0, df, floor_db, step_db = struct.unpack(GRID_HEADER, data[:size])
        if magic != GRID_MAGIC or version != GRID_VERSION:
            raise ValueError('not a version 1 spectrogram grid')
        return cls(t0, dt, f0, df, floor_db, step_db, times, freqs, bytes(data[size:size + times * freqs]))


def _max_pool(values, axis: int, most: int):
    ''' `values` reduced along `axis` to at most `most` groups, each the loudest of its run. :return: (values, run length). '''
    import numpy as np
    count = values.shape[axis]
    run = max(1, -(-count // most))
    if run == 1:
        return values, 1
    padded = count + (-count % run)
    pad = [(0, 0)] * values.ndim
    pad[axis] = (0, padded - count)
    values = np.pad(values, pad, constant_values=-np.inf)
    shape = list(values.shape)
    shape[axis:axis + 1] = [padded // run, run]
    return values.reshape(shape).max(axis=axis + 1), run


def spectrogram_grid(request: ComparisonRequest, filtered: bool, max_hz: float = 160.0,
                     max_times: int = MAX_TIME_BINS, max_freqs: int = MAX_FREQ_BINS) -> SpectrogramGrid:
    '''
    The spectrogram of a title's mono track (as the publication heatmap computes it, `model.heatmap.spectrogram_data` at the
    spec's resolution), after the design when `filtered`, from 0 Hz to `max_hz`, reduced to at most `max_times` x
    `max_freqs` cells by keeping the loudest of each group -- so a short hit is never averaged away -- and quantised to a
    byte over the GRID_RANGE_DB below the loudest cell.
    '''
    import numpy as np
    from model.heatmap import resolution_shift, spectrogram_data
    candidate = filter_from_json(json.loads(request.filters))
    published = preview_published_projects(request.mono_project, request.multichannel_project, candidate) \
        if filtered else None
    signal = Session(request.config).load(request.mono_wav)
    raw = signal.signal
    if published is not None:
        sos = published.filter.resample(signal.fs, copy_listener=False).get_sos()
        raw = raw.sosfilter(sos) if len(sos) > 0 else raw
    data = spectrogram_data(raw, resolution_shift(request.spec.resolution_multiplier))
    keep = data.f <= max_hz
    f, sxx = data.f[keep], data.sxx[keep]                 # (frequency, time)
    sxx, f_run = _max_pool(sxx, 0, max_freqs)
    sxx, t_run = _max_pool(sxx, 1, max_times)
    t = data.t
    dt = float(t[1] - t[0]) * t_run if t.size > 1 else 1.0
    df = float(f[1] - f[0]) * f_run if f.size > 1 else 1.0
    finite = sxx[np.isfinite(sxx)]
    top = float(finite.max()) if finite.size else 0.0
    floor = top - GRID_RANGE_DB
    step = GRID_RANGE_DB / 254
    cells = np.where(np.isfinite(sxx), np.clip(np.round((sxx - floor) / step) + 1, 0, 255), 0)
    cells[sxx < floor] = 0
    return SpectrogramGrid(float(t[0]) if t.size else 0.0, dt, float(f[0]) if f.size else 0.0, df, floor, step,
                           sxx.shape[1], sxx.shape[0], cells.astype(np.uint8).T.tobytes())
