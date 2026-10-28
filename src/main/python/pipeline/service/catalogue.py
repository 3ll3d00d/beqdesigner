'''
Published BEQs for a title, over HTTP (design/web-parity.md WP3): the catalogue's entries for it, read on demand
(`pipeline.library.catalogue_remote`, cached under `<work dir>/catalogue/`), and any one of them charted against the
title's own measured curves.
'''
import os
from dataclasses import dataclass
from typing import Annotated, Callable, Optional

from fastapi import APIRouter, Query

from pipeline.library.catalogue_match import catalogue_target
from pipeline.library.catalogue_remote import CATALOGUE_URL, CatalogueCache, CatalogueUnavailable, same_track
from pipeline.library.review_chart import published_curves
from pipeline.service import models


@dataclass(frozen=True)
class CatalogueRoutes:
    context: Callable
    read_index: Callable
    review_row: Callable
    review_entry: Callable
    parse_smoothing: Callable
    problem: type
    responses: Callable
    base_url: str = CATALOGUE_URL
    get: Optional[Callable] = None   # requests.get unless given


_TYPES = {'LowShelf': 'low_shelf', 'HighShelf': 'high_shelf', 'PeakingEQ': 'peaking_eq'}


def published_filters(filters) -> list:
    ''' A catalogue entry's filters (`{type, freq, gain, q, count}`) as the interface lists them; any it cannot say, left out. '''
    out = []
    for f in filters or ():
        try:
            out.append(models.PublishedFilter(type=_TYPES[f['type']], freq_hz=float(f['freq']), gain_db=float(f['gain']),
                                              q=float(f['q']), count=max(1, int(f.get('count') or 1))))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def register(v1: APIRouter, deps: CatalogueRoutes) -> None:
    Problem = deps.problem

    def matched(title_id: str):
        ctx = deps.context()
        settings = ctx.scan_settings()
        with deps.read_index(ctx) as index:
            row = deps.review_row(index, title_id)
        entry = deps.review_entry(settings, title_id)
        target = catalogue_target(entry, row, settings.meta_defaults)
        cache = CatalogueCache(os.path.join(settings.work_dir, 'catalogue'), deps.base_url, deps.get)
        try:
            return entry, target, cache.find(target)
        except CatalogueUnavailable as error:
            raise Problem(502, 'Catalogue unavailable', str(error))

    @v1.get('/titles/{title_id}/catalogue', response_model=models.PublishedBeqs, tags=['review'],
            summary='The BEQs the catalogue has published for a title', responses=deps.responses(404, 409, 502, 503))
    def published(title_id: str) -> models.PublishedBeqs:
        _, target, found = matched(title_id)
        if not target.usable:
            return models.PublishedBeqs(entries=[], message='Give the title a TMDB id, or a title and year, to look it up.')
        if found is None:
            return models.PublishedBeqs(entries=[], message='The catalogue has no BEQ for this title.')
        entries = [models.PublishedBeq(index=e.index, author=e.author, audio_types=list(e.audio_types), edition=e.edition,
                                       season=e.season, episodes=e.episodes, same_track=same_track(target, e),
                                       catalogue_url=e.catalogue_url or None, filter_count=len(e.filters),
                                       filters=published_filters(e.filters))
                   for e in found.entries]
        return models.PublishedBeqs(key=found.key, matched_by=found.matched_by, entries=entries,
                                    compare_url=f'{deps.base_url.rstrip("/")}/compare/?t={found.key}',
                                    message='' if entries else 'The catalogue lists the title but no BEQ to compare.')

    @v1.get('/titles/{title_id}/catalogue/{index}/chart', response_model=models.Chart, tags=['review'],
            summary="A published BEQ's response, and the title's measured curves under it",
            responses=deps.responses(404, 409, 422, 502, 503))
    def published_chart(title_id: str, index: int,
                        smoothing: Annotated[Optional[str], Query(description='As the chart route.')] = None) -> models.Chart:
        entry, _, found = matched(title_id)
        chosen = found.entries[index] if found is not None and 0 <= index < len(found.entries) else None
        if chosen is None:
            raise Problem(404, 'No such published BEQ', f'the catalogue lists no BEQ {index} for {title_id}')
        smooth = deps.parse_smoothing(smoothing)
        try:
            curves = published_curves(entry, chosen.filters, chosen.author, smooth)
        except (ValueError, KeyError, TypeError) as error:
            raise Problem(422, 'Cannot chart it', str(error))
        return models.Chart(candidate=None, candidates=[], smoothing=None if smooth is None else str(smooth).lower(),
                            series=[models.ChartSeries(name=c.data.name, kind=c.kind, filtered=c.filtered,
                                                       x=c.data.x.tolist(), y=c.data.y.tolist()) for c in curves])
