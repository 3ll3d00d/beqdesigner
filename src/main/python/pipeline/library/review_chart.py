'''
The curves a reviewer judges a title's designs on (design/web-app.md §2): the measured average and peak mono mix, and the
same after the chosen design. Shared by the desktop title page (`model.worklist_title_text.chart_data`, which colours them)
and the service's chart route (which sends them as arrays). Qt-free.
'''
from dataclasses import dataclass
from typing import List, Optional

from pipeline.review import QueueEntry


@dataclass
class ChartCurve:
    kind: str          # 'average' or 'peak'
    filtered: bool     # after the chosen design
    data: object       # model.xy.MagnitudeData, named for a legend


def track_name(entry: QueueEntry) -> str:
    ''' The source audio stream the curves are of, as a legend says it (rather than the pipeline's transient signal name). '''
    return 'audio track (all channels mixed)' if entry.audio_stream is None \
        else f'audio track {entry.audio_stream + 1} (all channels mixed)'


def chart_curves(entry: Optional[QueueEntry], picked: Optional[int]) -> List[ChartCurve]:
    '''
    The average curve (and peak, which old queue entries do not have), then -- when `picked` is a design in `entry.offered` --
    each filtered by it. A declined title's one candidate is flat, so its filtered curves are what was measured: what the
    decline was judged on.
    '''
    from model.codec import filter_from_json, xydata_from_json
    if entry is None or not entry.curve:
        return []
    track = track_name(entry)
    sources = [('average', xydata_from_json(entry.curve))]
    if entry.peak_curve:
        sources.append(('peak', xydata_from_json(entry.peak_curve)))
    result = []
    for kind, source in sources:
        source.override_name(f'{kind.capitalize()} {track}')
        result.append(ChartCurve(kind, False, source))
    if picked is not None and 0 <= picked < len(entry.offered):
        response = filter_from_json(entry.offered[picked].filters).get_transfer_function().get_magnitude()
        for kind, source in sources:
            filtered = source.filter(response)
            filtered.override_name(f'Filtered {kind} {track}')
            result.append(ChartCurve(kind, True, filtered))
    return result
