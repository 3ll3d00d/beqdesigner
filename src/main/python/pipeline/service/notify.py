'''Outbound webhooks for completed service jobs (design/pipeline-service.md §9).'''
import json
import logging
import os
import queue
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional
from urllib.parse import urlsplit

from model.execution_events import redact_text
from pipeline.library.index import LibraryIndex, index_path
from pipeline.service import models
from pipeline.service.config import secret
from pipeline.service.context import env_name, load_context
from pipeline.service.jobs import Job, JobManager

logger = logging.getLogger('pipeline_service_notify')
EVENTS = frozenset(event.value for event in models.NotifyEvent)
ORIGINS = frozenset(origin.value for origin in models.JobOrigin)
FORMATS = frozenset(('json', 'text', 'slack', 'discord'))


@dataclass(frozen=True)
class Target:
    name: str
    url: str = field(repr=False)
    format: str = 'json'
    events: frozenset[str] = field(default_factory=lambda: frozenset(('review_waiting', 'failed')))
    origins: frozenset[str] = field(default_factory=lambda: frozenset(('schedule',)))
    headers: Mapping[str, str] = field(default_factory=dict, repr=False)


def targets_from_config(values, env: Mapping[str, str]) -> tuple[Target, ...]:
    found, used = [], set()
    for value in values:
        name = value.get('name')
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', name):
            raise ValueError('notify target name must start with a letter and use letters, numbers, _ or -')
        key = env_name(name)
        if key in used:
            raise ValueError(f'notify target {name}: duplicate environment name')
        used.add(key)
        unknown = set(value) - {'name', 'url', 'format', 'events', 'origins'}
        if unknown:
            raise ValueError(f'notify target {name}: unknown keys {", ".join(sorted(unknown))}')
        url = secret(env, f'BEQ_NOTIFY_URL_{key}') or value.get('url')
        try:
            valid_url = isinstance(url, str) and urlsplit(url).scheme in ('http', 'https') and bool(urlsplit(url).netloc)
        except ValueError:
            valid_url = False
        if not valid_url:
            raise ValueError(f'notify target {name}: url must be HTTP or HTTPS (or set BEQ_NOTIFY_URL_{key})')
        format_ = value.get('format', 'json')
        if format_ not in FORMATS:
            raise ValueError(f'notify target {name}: format must be json, text, slack or discord')
        events = value.get('events', ['review_waiting', 'failed'])
        origins = value.get('origins', ['schedule'])
        if not isinstance(events, list) or any(not isinstance(event, str) or event not in EVENTS for event in events):
            raise ValueError(f'notify target {name}: unknown events')
        if not isinstance(origins, list) or any(not isinstance(origin, str) or origin not in ORIGINS for origin in origins):
            raise ValueError(f'notify target {name}: unknown origins')
        raw_headers = secret(env, f'BEQ_NOTIFY_HEADERS_{key}')
        try:
            headers = json.loads(raw_headers) if raw_headers else {}
        except ValueError:
            raise ValueError(f'notify target {name}: BEQ_NOTIFY_HEADERS_{key} must be a JSON object') from None
        if not isinstance(headers, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                                 for k, v in headers.items()):
            raise ValueError(f'notify target {name}: BEQ_NOTIFY_HEADERS_{key} must be a string-to-string JSON object')
        found.append(Target(name, url, format_, frozenset(events), frozenset(origins), headers))
    return tuple(found)


def _index_details(profile_path: str, env: Mapping[str, str], ids: list[str]):
    try:
        ctx = load_context(profile_path, env)
        with LibraryIndex(index_path(ctx.work_dir), readonly=True) as index:
            return ({row.id: row for row in index.titles(ids=ids)} if ids else {},
                    index.summary().counts.get('review', 0))
    except (OSError, ValueError):
        logger.warning('notification index unavailable')
        return {}, None


def _parts(job: Job):
    result = models.job_model(job).result if job.result is not None else None
    designed, failed, count = [], [], 0
    if isinstance(result, models.RunResult):
        designed = result.designed
        failed.extend((error.id, error.message) for error in result.failed)
        failed.extend((error.id, error.error) for error in result.publish_errors)
        if result.commit_error:
            failed.append(('commit', result.commit_error))
        if result.scan:
            failed.extend(result.scan.errors.items())
        count = result.counts.get('review', 0)
    elif isinstance(result, models.ScanResult):
        failed.extend(result.errors.items())
        count = result.counts.get('review', 0)
    if job.error:
        failed.append((job.id, job.error))
    return designed, failed, count


def events_for(job: Job) -> list[models.NotifyEvent]:
    designed, failed, _ = _parts(job)
    events = []
    if designed:
        events.append(models.NotifyEvent.review_waiting)
    if failed or job.state == 'failed':
        events.append(models.NotifyEvent.failed)
    events.append(models.NotifyEvent.job_finished)
    return events


def notification(job: Job, event: models.NotifyEvent, resolve: Callable) -> models.Notification:
    designed, failures, reported_count = _parts(job)
    rows, indexed_count = resolve(designed)
    titles = []
    for id_ in designed:
        row = rows.get(id_)
        titles.append(models.NotifyTitle(id=id_, title=(row.title or row.display_name or id_) if row else id_,
                                        year=(row.year or '') if row else '', kind=row.kind if row else 'movie',
                                        confidence=row.confidence if row else None))
    failed = [models.NotifyFailure(id=id_, title=(rows[id_].title or rows[id_].display_name or id_)
                                   if id_ in rows else id_, message=redact_text(message))
              for id_, message in failures]
    return models.Notification(event=event,
        job=models.NotifyJob(id=job.id, kind=job.kind, origin=job.origin, state=job.state,
                             started_at=models.timestamp(job.started_at), finished_at=models.timestamp(job.finished_at)),
        designed=titles, failed=failed, review_waiting=indexed_count if indexed_count is not None else reported_count,
        links=models.NotifyLinks(job=f'/v1/jobs/{job.id}', docs='/docs'))


def _body(target: Target, note: models.Notification) -> tuple[bytes, str]:
    if target.format == 'json':
        return note.model_dump_json().encode(), 'application/json'
    line = f'{note.review_waiting} titles waiting for review, {len(note.failed)} failed'
    if target.format == 'text':
        return line.encode(), 'text/plain; charset=utf-8'
    return json.dumps({'text' if target.format == 'slack' else 'content': line}).encode(), 'application/json'


def _send(target: Target, body: bytes, content_type: str) -> None:
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, request, response, code, message, headers, url):
            return None  # a configured destination cannot forward the title payload to another URL

    request = urllib.request.Request(target.url, data=body,
                                     headers={'Content-Type': content_type, **target.headers}, method='POST')
    with urllib.request.build_opener(NoRedirect()).open(request, timeout=10) as response:
        response.read(1024)


class Notifier:
    def __init__(self, manager: JobManager, profile_path: str, values=(), env: Optional[Mapping[str, str]] = None,
                 *, resolve: Optional[Callable] = None, send: Callable = _send,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.time,
                 start: bool = True):
        self.env = os.environ if env is None else env
        self.targets = targets_from_config(values, self.env)
        self.resolve = resolve or (lambda ids: _index_details(profile_path, self.env, ids))
        self.send, self.sleep, self.clock = send, sleep, clock
        self.outcome: dict[str, models.NotifyOutcome] = {}
        self.lock = threading.RLock()
        self.queue: queue.Queue = queue.Queue()
        self.unsubscribe = manager.subscribe_completed(self.queue.put) if self.targets else lambda: None
        self.thread = None
        if start and self.targets:
            self.thread = threading.Thread(target=self._loop, name='pipeline-service-notify', daemon=True)
            self.thread.start()

    def outcomes(self) -> list[models.NotifyOutcome]:
        with self.lock:
            return [self.outcome.get(target.name, models.NotifyOutcome(name=target.name)) for target in self.targets]

    def _deliver(self, target: Target, note: models.Notification) -> models.NotifyOutcome:
        body, content_type = _body(target, note)
        message = 'delivery failed'
        for attempt in range(3):
            if attempt:
                self.sleep(attempt)
            try:
                self.send(target, body, content_type)
                result = models.NotifyOutcome(name=target.name, last_event=note.event,
                                              last_attempt_at=models.timestamp(self.clock()), ok=True,
                                              message='delivered')
                break
            except urllib.error.HTTPError as error:
                message = f'HTTP {error.code}'
            except Exception as error:
                message = f'{type(error).__name__}; delivery failed'
        else:
            result = models.NotifyOutcome(name=target.name, last_event=note.event,
                                          last_attempt_at=models.timestamp(self.clock()), ok=False, message=message)
        with self.lock:
            self.outcome[target.name] = result
        return result

    def _for_job(self, job: Job) -> None:
        for event in events_for(job):
            for target in self.targets:
                if job.origin in target.origins and event.value in target.events:
                    self._deliver(target, notification(job, event, self.resolve))

    def _loop(self) -> None:
        while True:
            job = self.queue.get()
            if job is None:
                return
            try:
                self._for_job(job)
            except Exception:
                logger.warning('notification preparation failed for job %s', job.id)

    def test(self, name: str) -> models.NotifyOutcome:
        target = next((target for target in self.targets if target.name == name), None)
        if target is None:
            raise KeyError(name)
        now = models.timestamp(self.clock())
        outcome = models.NotifyOutcome(name=name)
        for event in models.NotifyEvent:
            if event.value in target.events:
                note = models.Notification(event=event,
                    job=models.NotifyJob(id='sample', kind='run', origin='schedule', state='succeeded',
                                         started_at=now, finished_at=now),
                    designed=[models.NotifyTitle(id='sample', title='Sample title', year='2026', kind='movie',
                                                 confidence=0.9)],
                    failed=[models.NotifyFailure(id='sample', title='Sample title', message='Sample failure')],
                    review_waiting=1, links=models.NotifyLinks(job='/v1/jobs/sample', docs='/docs'))
                outcome = self._deliver(target, note)
        return outcome

    def stop(self, grace_seconds: float = 10) -> None:
        self.unsubscribe()
        if self.thread:
            self.queue.put(None)
            self.thread.join(timeout=grace_seconds)
