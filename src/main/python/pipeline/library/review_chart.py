'''
The curves a reviewer judges a title's designs on (design/web-app.md §2): the measured average and peak mono mix, and the
same after the chosen design. Shared by the desktop title page (`model.worklist_title_text.chart_data`, which colours them)
and the service's chart route (which sends them as arrays). Qt-free.
'''
from dataclasses import dataclass
from typing import List, Optional, Sequence

from pipeline.review import QueueEntry


SMOOTHING = (None, 1, 2, 3, 6, 12, 24, 'SG')   # the main window's smoothing box: none, 1/N octave, Savitzky-Golay


@dataclass
class ChartCurve:
    kind: str          # 'average', 'peak', or 'filter' (a design's own response)
    filtered: bool     # after a design
    data: object       # model.xy.MagnitudeData, named for a legend
    candidate: Optional[int] = None   # the design it is after (or is the response of); None for the measured curves


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
    return workbench_curves(entry, [picked] if picked is not None else [])


def workbench_curves(entry: Optional[QueueEntry], designs: Sequence[int], smoothing=None,
                     responses: bool = False) -> List[ChartCurve]:
    '''
    The measured curves, then for each design in `designs` (indexes into `entry.offered`; any out of range are left out)
    the same curves after it -- and, with `responses`, the design's own magnitude response (kind 'filter').
    :param smoothing: one of SMOOTHING: the measured and filtered curves are smoothed (the filter applied first, as the main
        window does), the responses never are.
    :raises ValueError: for a smoothing not in SMOOTHING, or one a curve has too few points for.
    '''
    from model.codec import filter_from_json, xydata_from_json
    if smoothing not in SMOOTHING:
        raise ValueError(f'{smoothing!r} is not a smoothing: one of {", ".join(str(s) for s in SMOOTHING[1:])} or none')
    if entry is None or not entry.curve:
        return []
    track = track_name(entry)
    sources = [('average', xydata_from_json(entry.curve))]
    if entry.peak_curve:
        sources.append(('peak', xydata_from_json(entry.peak_curve)))

    def named(data, name):
        if smoothing is not None:
            try:
                data = data.smooth(smoothing)
            except ValueError as error:   # Savitzky-Golay's window is longer than a sparse curve
                raise ValueError(f'This curve cannot be smoothed so ({smoothing}): {error}') from None
        data.override_name(name)
        return data

    result = [ChartCurve(kind, False, named(source, f'{kind.capitalize()} {track}')) for kind, source in sources]
    several = len(designs) > 1
    for picked in designs:
        if not 0 <= picked < len(entry.offered):
            continue
        response = filter_from_json(entry.offered[picked].filters).get_transfer_function().get_magnitude()
        suffix = f' (design {picked + 1})' if several else ''
        for kind, source in sources:
            result.append(ChartCurve(kind, True, named(source.filter(response), f'Filtered {kind} {track}{suffix}'),
                                     picked))
        if responses:
            response.override_name(f'Design {picked + 1}')
            result.append(ChartCurve('filter', True, response, picked))
    return result
