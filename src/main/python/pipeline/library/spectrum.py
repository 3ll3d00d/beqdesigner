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
