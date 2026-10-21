'''
Editing a title over HTTP (design/web-parity.md WP4): its metadata, a TMDB reload, and its artwork -- the title page's
Metadata tab, with the same rules (`pipeline.library.metadata_edit`, `pipeline.library.artwork_edit`).

Every write names the `metadata_digest` it was made against and is refused (409) if the metadata or artwork changed since,
or while a run has the title in hand (a design in flight writes a new entry over whatever is there). Only the entry is
written: no repository. The index learns of an edit by reading the outputs again, as after a decision.
'''
import base64
import binascii
import os
from dataclasses import dataclass
from typing import Annotated, Callable

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse

from pipeline.library.artwork_edit import ArtworkError, download_artwork, set_artwork, store_image
from pipeline.library.metadata_edit import FIELD_SPECS, TmdbError, changes_from_texts, metadata_digest, save_meta_changes, \
    tmdb_fill, tmdb_reload
from pipeline.metadata import BeqMetadata
from pipeline.review import entry_art_path, read_entry
from pipeline.service import models

RUNNING = 'A run is working on this title now: it cannot be edited until the run has finished with it.'
CHANGED = 'Changed since it was read'
TV_KINDS = ('tv', 'season', 'episode')


@dataclass(frozen=True)
class EditRoutes:
    ''' What the routes need of the app: its helpers, so they read and refuse as the review routes do. '''
    context: Callable
    read_index: Callable
    review_row: Callable
    review_entry: Callable
    in_flight: Callable
    review_model: Callable
    refresh: Callable[[], None]
    problem: type     # ServiceProblem
    responses: Callable


def register(v1: APIRouter, deps: EditRoutes) -> None:
    Problem, responses = deps.problem, deps.responses

    def editable(title_id: str, digest: str):
        ''' (ctx, settings, row, entry) of a title that can be written now, as it was read. '''
        ctx = deps.context()
        settings = ctx.scan_settings()
        with deps.read_index(ctx) as index:
            row = deps.review_row(index, title_id)
        entry = deps.review_entry(settings, title_id)
        if deps.in_flight(ctx, row):
            raise Problem(409, 'In flight', RUNNING)
        if digest != metadata_digest(entry):
            raise Problem(409, CHANGED, f'{title_id}: its metadata or artwork changed since it was read; read it again')
        return ctx, settings, row, entry

    def written(ctx, settings, row, title_id: str):
        deps.refresh()
        return deps.review_model(ctx, settings, row, read_entry(settings.queue_dir, title_id))

    @v1.get('/metadata/fields', response_model=list[models.MetadataField], tags=['edit'],
            summary='The metadata fields a title page edits, in order, with their labels and choices')
    def fields() -> list[models.MetadataField]:
        return [models.MetadataField(key=f.key, label=f.label, group=f.group, tip=f.tip, kind=f.kind,
                                     choices=list(f.choices)) for f in FIELD_SPECS]

    @v1.patch('/titles/{title_id}/metadata', response_model=models.Review, tags=['edit'],
              summary="Change a title's metadata (refused if it changed since it was read, or a run has the title)",
              responses=responses(404, 409, 422, 503))
    def edit_metadata(title_id: str, body: models.MetadataEdit) -> models.Review:
        ctx, settings, row, _ = editable(title_id, body.digest)
        try:
            changes = changes_from_texts(body.texts, body.extras)
        except ValueError as error:
            raise Problem(422, 'Not a metadata edit', str(error))
        try:
            save_meta_changes(settings.queue_dir, title_id, changes)
        except FileNotFoundError:
            raise Problem(409, 'Not designed', f'{title_id} has no queue entry any more')
        except OSError as error:
            raise Problem(503, 'Not saved', f'{title_id}: {type(error).__name__}: {error}')
        return written(ctx, settings, row, title_id)

    @v1.post('/titles/{title_id}/metadata/tmdb', response_model=models.TmdbProposal, tags=['edit'],
             summary='Look a title up on TMDB: what it would fill (nothing is saved)',
             responses=responses(404, 409, 502, 503))
    def tmdb(title_id: str, body: models.TmdbLookup) -> models.TmdbProposal:
        ctx = deps.context()
        settings = ctx.scan_settings()
        with deps.read_index(ctx) as index:
            row = deps.review_row(index, title_id)
        entry = deps.review_entry(settings, title_id)
        key = ctx.values.get('tmdb_api_key')
        if not key:
            raise Problem(409, 'No TMDB key', 'TMDB_API_KEY is not set on the service: TMDB cannot be asked')
        meta = entry.meta
        tmdb_id = (body.tmdb_id if body.tmdb_id is not None else str(meta.get('the_movie_db') or '')).strip()
        title = (body.title if body.title is not None else str(meta.get('title') or row.title or '')).strip()
        year = (body.year if body.year is not None else str(meta.get('year') or row.year or '')).strip()
        season = body.season if body.season is not None else str(meta.get('season') or '')
        if not tmdb_id and not title:
            raise Problem(422, 'Nothing to look up', 'give a title (and year), or a TMDB id')
        kind = 'tv' if season.strip() or row.kind in TV_KINDS else 'movie'
        try:
            found: BeqMetadata = tmdb_reload(key, tmdb_id, title, year, kind)
        except TmdbError as error:
            raise Problem(502, 'TMDB lookup failed', str(error))
        texts, extras = tmdb_fill(found)
        if not texts:
            return models.TmdbProposal(found=False, texts={}, extras={}, message=(
                f'TMDB has no match for "{title}"' + (f' ({year})' if year else '') + '. Nothing was changed.'))
        return models.TmdbProposal(found=True, texts=texts, extras=extras,
                                   message='Filled in from TMDB. Nothing is saved until you save the edit.')

    @v1.get('/titles/{title_id}/artwork', tags=['edit'], summary="A title's poster image",
            responses={200: {'content': {'image/png': {}, 'image/jpeg': {}}}, **responses(404, 409, 503)},
            response_class=FileResponse)
    def artwork(title_id: str):
        ctx = deps.context()
        settings = ctx.scan_settings()
        entry = deps.review_entry(settings, title_id)
        path = entry_art_path(entry, settings.work_dir or None)
        if not path or not os.path.isfile(path):
            raise Problem(404, 'No artwork', f'{title_id} has no poster' + (' file here' if path else ''))
        return FileResponse(path, headers={'Cache-Control': 'no-cache'})

    @v1.put('/titles/{title_id}/artwork', response_model=models.Review, tags=['edit'],
            summary="Set a title's poster: an image downloaded from a URL, or one sent",
            responses=responses(404, 409, 422, 503))
    def choose_artwork(title_id: str, body: models.ArtworkChoice) -> models.Review:
        if (body.url is None) == (body.data_base64 is None):
            raise Problem(422, 'One image', 'give a url or data_base64, not both or neither')
        ctx, settings, row, _ = editable(title_id, body.digest)
        try:
            if body.url is not None:
                path = download_artwork(body.url, settings.queue_dir, title_id)
            else:
                try:
                    data = base64.b64decode(body.data_base64, validate=True)
                except (binascii.Error, ValueError):
                    raise ArtworkError('that is not base64') from None
                path = store_image(data, settings.queue_dir, title_id)
        except ArtworkError as error:
            raise Problem(422, 'Not used', f'Not used: {error}.')
        set_artwork(settings.queue_dir, title_id, path)
        return written(ctx, settings, row, title_id)

    @v1.delete('/titles/{title_id}/artwork', response_model=models.Review, tags=['edit'],
               summary="Clear a title's poster (a later run may choose one again)", responses=responses(404, 409, 503))
    def clear_artwork(title_id: str, digest: Annotated[str, Query(description="The Review's metadata_digest.")]):
        ctx, settings, row, _ = editable(title_id, digest)
        set_artwork(settings.queue_dir, title_id, None)
        return written(ctx, settings, row, title_id)
