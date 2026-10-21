'''
What a title's audio looks like, over HTTP (design/web-parity.md WP2): the publication heatmap of a design, and the
spectrogram grid the browser draws, both from `pipeline.library.spectrum`.

Rendering takes seconds, so it never holds a request: the first asks for it on a small pool of its own and is answered 202
(`Retry-After`); the result is kept in the title's work folder (`<work>/<id>/web/`) under a name made from everything it
depends on -- the mono track's and projects' stamps, the design's filters, the heatmap spec and the request -- so a
redesign, re-extraction or edited project is a new file, and asking again serves the file. A render that fails is
answered once with why (422), then may be asked for again.
'''
import hashlib
import json
import os
import tempfile
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Annotated, Callable, Dict, Optional

from fastapi import APIRouter, Query
from fastapi.responses import Response

from pipeline.config import AnalysisConfig
from pipeline.library.spectrum import comparison_request, render_comparison, spectrogram_grid
from pipeline.library.workdir import entry_directory
from pipeline.publish.heatmap import HeatmapSpec

RETRY_SECONDS = 2


class Pending(Exception):
    ''' Being rendered: ask again. '''


class RenderQueue:
    ''' Renders on a pool of its own, one at a time per file, and keeps each result as a file. '''

    def __init__(self, workers: int = 2):
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='visuals')
        self._running: Dict[str, Future] = {}
        self._lock = threading.Lock()

    def get(self, path: str, make: Callable[[], bytes]) -> bytes:
        '''
        :return: the file's bytes once made.
        :raises Pending: while it is being made (it is started if it is not).
        :raises ValueError: (once) with why making it failed.
        '''
        with self._lock:
            future = self._running.get(path)
            if future is not None and future.done():
                del self._running[path]
                error = future.exception()
                if error is not None:
                    raise ValueError(str(error)) from error
            elif future is not None:
                raise Pending()
            if os.path.isfile(path):
                with open(path, 'rb') as f:
                    return f.read()
            self._running[path] = self._pool.submit(self._make, path, make)
        raise Pending()

    @staticmethod
    def _make(path: str, make: Callable[[], bytes]) -> None:
        data = make()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix='.render.', dir=os.path.dirname(path))
        with os.fdopen(handle, 'wb') as f:
            f.write(data)
        os.replace(temporary, path)

    def stop(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


@dataclass(frozen=True)
class VisualRoutes:
    context: Callable
    read_index: Callable
    review_row: Callable
    review_entry: Callable
    entry_title: Callable
    problem: type
    responses: Callable
    queue: RenderQueue


def _key(request, *extra) -> str:
    seen = json.dumps([request.files, request.filters, asdict(request.spec), *extra], sort_keys=True, default=str)
    return hashlib.sha256(seen.encode('utf-8')).hexdigest()[:24]


def register(v1: APIRouter, deps: VisualRoutes) -> None:
    Problem = deps.problem
    pending = {202: {'description': 'Being rendered: ask again after `Retry-After` seconds.'}}

    def comparison(title_id: str, candidate: Optional[int]):
        ctx = deps.context()
        settings = ctx.scan_settings()
        with deps.read_index(ctx) as index:
            row = deps.review_row(index, title_id)
        entry = deps.review_entry(settings, title_id)
        picked = entry.chosen_candidate_index if candidate is None else candidate
        picked = 0 if picked is None else picked
        try:
            request = comparison_request(settings.work_dir, entry, picked, deps.entry_title(entry, row), AnalysisConfig(),
                                         spec=HeatmapSpec())
        except ValueError as error:
            raise Problem(409 if 'design' not in str(error) else 422, 'Nothing to draw', str(error))
        return settings, request

    def served(path: str, make: Callable[[], bytes], media_type: str) -> Response:
        try:
            data = deps.queue.get(path, make)
        except Pending:
            return Response(status_code=202, headers={'Retry-After': str(RETRY_SECONDS), 'Cache-Control': 'no-store'})
        except ValueError as error:
            raise Problem(422, 'Not drawn', str(error))
        return Response(data, media_type=media_type, headers={'Cache-Control': 'private, max-age=3600'})

    @v1.get('/titles/{title_id}/heatmap.png', tags=['review'], response_class=Response,
            summary="The publication heatmap of a design: filtered on the left, as it would be published",
            responses={200: {'content': {'image/png': {}}}, **pending, **deps.responses(404, 409, 422, 503)})
    def heatmap(title_id: str, candidate: Annotated[Optional[int], Query(ge=0, description='A design; default the '
                                                                                         'chosen one, else the first.')] = None):
        settings, request = comparison(title_id, candidate)
        path = os.path.join(entry_directory(settings.work_dir, title_id), 'web', f'heatmap-{_key(request)}.png')
        return served(path, lambda: render_comparison(request)[0], 'image/png')

    @v1.get('/titles/{title_id}/spectrogram', tags=['review'], response_class=Response,
            summary="A title's spectrogram as a grid of bytes, for drawing (design/web-parity.md WP2)",
            responses={200: {'content': {'application/octet-stream': {}},
                             'description': 'A 40-byte little-endian header -- magic `BEQS`, version (u16), reserved '
                                            '(u16), time bins (u32), frequency bins (u32), then as f32: first time (s), '
                                            'time step (s), first frequency (Hz), frequency step (Hz), the level of cell '
                                            'value 1 (dB), dB per value -- then one byte per cell, time-major; 0 is '
                                            'below the floor.'},
                       **pending, **deps.responses(404, 409, 422, 503)})
    def spectrogram(title_id: str,
                    candidate: Annotated[Optional[int], Query(ge=0, description='The design applied when filtered; '
                                                                                'default the chosen one, else the first.')] = None,
                    filtered: Annotated[bool, Query(description='After the design (else as measured).')] = False,
                    max_hz: Annotated[float, Query(gt=0, le=500, description='The highest frequency kept.')] = 160.0):
        settings, request = comparison(title_id, candidate)
        name = f'spectrogram-{_key(request, filtered if filtered else None, max_hz)}.bin'
        path = os.path.join(entry_directory(settings.work_dir, title_id), 'web', name)
        return served(path, lambda: spectrogram_grid(request, filtered, max_hz).encode(), 'application/octet-stream')
