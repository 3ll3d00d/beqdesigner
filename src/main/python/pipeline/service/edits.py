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
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Annotated, Callable

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse, JSONResponse

from pipeline.library.artwork_edit import ArtworkError, download_artwork, set_artwork, store_image
from pipeline.library.metadata_edit import FIELD_SPECS, TmdbError, changes_from_texts, metadata_digest, save_meta_changes, \
    tmdb_fill, tmdb_reload
from pipeline.library.decide import offered_digest
from pipeline.library.revise import revise_entry
from pipeline.designer.contract import BiquadSpec
from pipeline.designer.convert import ContractViolation
from pipeline.library.variants import VariantRefused, discard_variant, save_edited, use_variant
from pipeline.library.selection import Selection
from pipeline.library.streams import describe_streams, preferred_stream, probe_stream_details
from pipeline.metadata import BeqMetadata
from pipeline.publish.git import RepoTarget
from pipeline.review import entry_art_path, read_entry, update_entry
from pipeline.service import models
from pipeline.service.jobs import RunRequest

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
    submit: Callable          # the job manager's: a RunRequest in, the queued Job out
    repository_writes: bool   # the service config's allow_repository_writes


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

    # --- revise and the audio stream (WP5), with adjustments (WP6) ---------------------------------------------------

    def stored_adjustments(asks, reason: str) -> list:
        ''' What a person asked, as the entry keeps it: an id each, when, and why (the service knows no one by name). '''
        at = datetime.now(timezone.utc).isoformat(timespec='seconds')
        return [{'id': f'a{i + 1}', 'kind': a.kind, 'params': dict(a.params), 'text': a.text, 'by': 'service', 'at': at,
                 'reason': reason or None} for i, a in enumerate(asks)]

    def repositories(ctx) -> dict:
        settings, _ = ctx.stage_settings('design')
        return dict(xml_repo=RepoTarget(settings.xml_repo) if settings.xml_repo else None,
                    images_repo=RepoTarget(settings.images_repo) if settings.images_repo else None,
                    xml_dir=settings.xml_dir, image_dir=settings.image_dir,
                    category_folders=settings.category_folders, meta_defaults=settings.meta_defaults)

    @v1.post('/revise', response_model=models.ReviseOutcome, tags=['edit'],
             summary='Send titles back to review, design or extract: now, or in one run job',
             responses=responses(409, 422, 503))
    def revise(body: models.ReviseRequest) -> models.ReviseOutcome:
        ctx = deps.context()
        settings = ctx.scan_settings()
        if body.to == 'review' and (body.run or body.adjustments is not None):
            raise Problem(422, 'Nothing to run' if body.run else 'Not for review',
                          'a title sent back to review waits for a person: no run, and no adjustments to ask for')
        asked = None if body.adjustments is None else stored_adjustments(body.adjustments, body.reason)
        with deps.read_index(ctx) as index:
            rows = {row.id: row for row in index.titles(ids=list(body.ids))}
        revised, failed, ready = [], [], []
        where = None
        for title_id in body.ids:
            row = rows.get(title_id)
            try:
                if row is None:
                    raise ValueError('no such title')
                if deps.in_flight(ctx, row):
                    raise ValueError(RUNNING)
                entry = read_entry(settings.queue_dir, title_id)
                if entry.status == 'published' and not deps.repository_writes:
                    raise ValueError('it is published: sending it back needs allow_repository_writes in service config')
                if body.run:   # the job sends it back; what it is asked to change is kept on it first
                    if asked is not None:
                        update_entry(settings.queue_dir, title_id, adjustments=asked)
                    ready.append(title_id)
                    continue
                where = where if where is not None else repositories(ctx)
                result = revise_entry(settings.queue_dir, title_id, body.to, body.reason, work_dir=settings.work_dir,
                                      adjustments=asked, **where)
            except FileNotFoundError:
                failed.append(models.ReviseFailure(id=title_id, reason='not designed yet: nothing to send back'))
            except (ValueError, OSError) as error:
                failed.append(models.ReviseFailure(id=title_id, reason=str(error)))
            else:
                revised.append(models.RevisedTitle(id=title_id, reverted=result.reverted,
                                                   extract_invalidated=result.extract_invalidated))
        if body.run:
            job = deps.submit(RunRequest(Selection(ids=tuple(ready)), 'design', scan_first=False, retry_failed=True,
                                         redesign=True, revise_to=body.to)) if ready else None
            return models.ReviseOutcome(revised=[], failed=failed, job=models.job_model(job) if job else None)
        if revised:
            deps.refresh()
        return models.ReviseOutcome(revised=revised, failed=failed)

    def listed_or_probed(item) -> tuple:
        ''' The streams the source listed; else read from the file (ffprobe) now -- kept only once one is chosen. '''
        if item.audio_stream_details:
            return item.audio_stream_details, False
        try:
            return probe_stream_details(item), True
        except ValueError as error:
            raise Problem(409, 'Streams not read', f'The source lists no audio streams and the file could not be read: '
                                                   f'{error}')

    @v1.get('/titles/{title_id}/streams', response_model=models.TitleStreams, tags=['edit'],
            summary="A title's audio streams, and the one extracted", responses=responses(404, 409, 503))
    def streams(title_id: str) -> models.TitleStreams:
        ctx = deps.context()
        with deps.read_index(ctx) as index:
            deps.review_row(index, title_id)
            item = index.units([title_id]).get(title_id)
        if item is None or not hasattr(item, 'audio_stream_details'):
            raise Problem(409, 'A season', 'a season is a file per episode: choose an audio stream for each episode')
        details, probed = listed_or_probed(item)
        preferred = preferred_stream(details) if details else 0
        return models.TitleStreams(chosen=item.audio_stream, source=item.audio_stream_source or '', probed=probed, streams=[
            models.AudioStream(index=i, text=text, preferred=i == preferred)
            for i, text in enumerate(describe_streams(details))])

    @v1.put('/titles/{title_id}/stream', status_code=202, response_model=models.RunJob, tags=['edit'],
            summary='Extract and design a title from another audio stream: a run job', responses=responses(404, 409, 422, 503))
    def choose(title_id: str, body: models.StreamChoice):
        ctx = deps.context()
        with deps.read_index(ctx) as index:
            row = deps.review_row(index, title_id)
            item = index.units([title_id]).get(title_id)
        if deps.in_flight(ctx, row):
            raise Problem(409, 'In flight', RUNNING)
        if item is None or not hasattr(item, 'audio_stream_details'):
            raise Problem(409, 'A season', 'a season is a file per episode: choose an audio stream for each episode')
        details, _ = listed_or_probed(item)
        if not 0 <= body.audio_stream < len(details):
            raise Problem(422, 'No such stream', f'{title_id} has {len(details)} audio stream(s)')
        job = deps.submit(RunRequest(Selection(ids=(title_id,)), 'design', scan_first=False, retry_failed=True,
                                     redesign=True, revise_to='extract', audio_stream=body.audio_stream))
        return JSONResponse(models.job_model(job).model_dump(mode='json'), status_code=202,
                            headers={'Location': f'/v1/jobs/{job.id}'})

    # --- variants (WP7) --------------------------------------------------------------------------------------------

    def as_looked_at(title_id: str, digest: str):
        ctx = deps.context()
        settings = ctx.scan_settings()
        with deps.read_index(ctx) as index:
            row = deps.review_row(index, title_id)
        entry = deps.review_entry(settings, title_id)
        if deps.in_flight(ctx, row):
            raise Problem(409, 'In flight', RUNNING)
        if digest != offered_digest(entry):
            raise Problem(409, CHANGED, f'{title_id}: its designs changed since they were read; read them again')
        return ctx, settings, row

    @v1.post('/titles/{title_id}/variants/{variant_id}/use', response_model=models.Review, tags=['edit'],
             summary='Make an earlier design run of a title its current one again (the current one is kept)',
             responses=responses(404, 409, 503))
    def use(title_id: str, variant_id: str, body: models.VariantChoice) -> models.Review:
        ctx, settings, row = as_looked_at(title_id, body.digest)
        try:
            use_variant(settings.queue_dir, title_id, variant_id)
        except VariantRefused as refused:
            raise Problem(409, 'Not now', str(refused))
        return written(ctx, settings, row, title_id)

    @v1.delete('/titles/{title_id}/variants/{variant_id}', response_model=models.Review, tags=['edit'],
               summary='Discard one of a title\'s earlier design runs', responses=responses(404, 409, 503))
    def discard(title_id: str, variant_id: str,
                digest: Annotated[str, Query(description="The Review's digest: what was looked at.")]) -> models.Review:
        ctx, settings, row = as_looked_at(title_id, digest)
        try:
            discard_variant(settings.queue_dir, title_id, variant_id)
        except VariantRefused as refused:
            raise Problem(409, 'Not now', str(refused))
        return written(ctx, settings, row, title_id)

    # --- editing a design's filters (WP8) ----------------------------------------------------------------------------

    @v1.put('/titles/{title_id}/edited', response_model=models.Review, tags=['edit'],
            summary="Save a person's edit of one of a title's designs: a new design run (the current one is kept)",
            responses=responses(404, 409, 422, 503))
    def edited(title_id: str, body: models.EditedDesign) -> models.Review:
        ctx, settings, row = as_looked_at(title_id, body.digest)
        specs = [BiquadSpec(type=f.type, freq_hz=f.freq_hz, gain_db=f.gain_db, q=f.q)
                 for f in body.filters for _ in range(f.count)]
        try:
            save_edited(settings.queue_dir, title_id, body.source, specs)
        except VariantRefused as refused:
            raise Problem(409, 'Not now', str(refused))
        except ContractViolation as violation:
            raise Problem(422, 'Not a design', str(violation))
        return written(ctx, settings, row, title_id)
