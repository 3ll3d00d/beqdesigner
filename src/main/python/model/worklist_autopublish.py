'''
One step from *Accept* to the repository: accepting a title writes its files into the catalogue repositories' working trees
and commits them, locally. Only the push is left to a person (the *Commit*/*Push* button), so nothing leaves the machine until
they say.

`publish_and_commit_locally()` is the work and has no widgets; `WorkListAutoPublish` is the mixin that runs it on a
worker after a decision on the title page or a bulk accept, one job at a time (two writers to a git working tree would
collide on its index lock), and waits for a publish or commit run in flight, which uses the same repositories.
'''
import logging
from typing import Dict, List, Optional

from qtpy.QtCore import QObject, QRunnable, QThreadPool, Signal

from model.worklist_folder_state import split_for_publish
from model.worklist_run import build_publish_settings, publish_problem
from pipeline.library.stages import PublishSettings
from pipeline.library.sync import commit_library, publish_library
from pipeline.review import split_publish_results, describe_publish_error

logger = logging.getLogger('worklist')


class AutoPublishOutcome:
    ''' What one job did: `results` are publish_library()'s, `committed` a CatalogueCommit (None if nothing was written). '''

    def __init__(self, ids: List[str], results: List[dict], committed, commit_error: str = ''):
        self.ids = ids
        self.results = results
        self.committed = committed
        self.commit_error = commit_error
        self.published, self.errors = split_publish_results(results)

    @property
    def ok(self) -> bool:
        return not self.errors and not self.commit_error

    def describe(self, titles: Optional[Dict[str, str]] = None) -> str:
        ''' One line for the status bar. '''
        def name(title_id: str) -> str:
            return (titles or {}).get(title_id) or title_id
        if self.ok:
            count = len(self.published)
            if count == 1:
                return f'Published {name(self.published[0]["id"])} (committed locally; push when ready)'
            return f'Published {count:,} titles (committed locally; push when ready)'
        problems = [f'{name(e["id"])}: {describe_publish_error(e)}' for e in self.errors]
        if self.commit_error:
            problems.append(f'the local commit failed ({self.commit_error})')
        return 'Accepted, but not fully published: ' + '; '.join(problems) + '. Publish tries again.'


def publish_and_commit_locally(ids: List[str], queue_dir: str, work_dir: Optional[str], settings: PublishSettings,
                               config) -> AutoPublishOutcome:
    '''
    Writes every accepted entry of `ids` into the repositories, then commits what was written (locally: `push=False`, one
    commit per repository, images first). An entry that cannot be published is reported and keeps its status, so *Publish*
    retries it; the others are not held up.
    '''
    with_projects, without, refused = split_for_publish(work_dir, ids)
    results: List[dict] = list(refused)
    for chosen, where in ((with_projects, work_dir), (without, None)):
        if chosen:
            results += publish_library(
                queue_dir, settings.xml_repo, meta_defaults=settings.meta_defaults, images_repo=settings.images_repo,
                image_owner=settings.image_owner, image_repo_name=settings.image_repo_name, xml_dir=settings.xml_dir,
                image_dir=settings.image_dir, category_folders=settings.category_folders,
                report_spec=settings.report_spec, heatmap_spec=settings.heatmap_spec, config=config, work_dir=where,
                ids=chosen)
    published, _ = split_publish_results(results)
    committed, error = None, ''
    if published:
        try:
            committed = commit_library(
                queue_dir, settings.xml_repo, images_repo=settings.images_repo, xml_dir=settings.xml_dir,
                image_dir=settings.image_dir, push=False, ids=[r['id'] for r in published],
                category_folders=settings.category_folders, meta_defaults=settings.meta_defaults)
        except Exception as failure:   # the files are written and the entries published: the Commit button finishes it
            logger.warning('local commit failed: %s', failure, exc_info=True)
            error = f'{type(failure).__name__}: {failure}'
    return AutoPublishOutcome(list(ids), results, committed, error)


class _AutoSignals(QObject):
    finished = Signal(object)   # AutoPublishOutcome
    errored = Signal(str)


class _AutoPublishJob(QRunnable):
    def __init__(self, ids, queue_dir, work_dir, settings, config):
        super().__init__()
        self.signals = _AutoSignals()
        self._args = (ids, queue_dir, work_dir, settings, config)

    def run(self):
        try:
            outcome = publish_and_commit_locally(*self._args)
        except Exception as error:
            logger.exception('Publishing accepted titles failed')
            self.signals.errored.emit(f'{type(error).__name__}: {error}')
            return
        self.signals.finished.emit(outcome)


class WorkListAutoPublish:
    '''
    The mixin: it uses `_setup`, `_preferences`, `_job` (a run in flight), `_index_dirty`, `_title_open`, `statusBar`,
    `_model` and `_sync_index_if_dirty()` of `WorkListWindow`, and its signal `auto_published`.
    '''

    def _init_autopublish(self) -> None:
        self._auto_pool = QThreadPool(self)
        self._auto_pool.setMaxThreadCount(1)   # one at a time: they share a git working tree
        self._auto_pending: List[str] = []
        self._auto_job: Optional[_AutoPublishJob] = None

    @property
    def auto_publishing(self) -> bool:
        return self._auto_job is not None

    def auto_publish(self, ids) -> bool:
        '''
        Writes and commits (locally) these accepted titles' files. Queued behind one already going, and behind a run of the
        window's own (which uses the same repositories): it starts when that ends.
        :return: False if nothing was queued (no repository is set up: the titles stay accepted, and Publish is there).
        '''
        ids = [i for i in ids if i]
        if not ids:
            return False
        problem = publish_problem(self._setup)
        if problem:
            self.statusBar.showMessage(f'Accepted, not published: {problem}', 15000)
            return False
        self._auto_pending.extend(i for i in ids if i not in self._auto_pending)
        self._start_auto_publish()
        return True

    def _start_auto_publish(self) -> None:
        if self._auto_job is not None or not self._auto_pending or self._job is not None:
            return
        try:
            settings = build_publish_settings(self._setup, push=False, preferences=self._preferences)
            config = self._setup.settings.config
        except Exception as error:
            logger.exception('Could not prepare publishing')
            self.statusBar.showMessage(f'Accepted, not published: {error}', 15000)
            self._auto_pending = []
            return
        ids, self._auto_pending = self._auto_pending, []
        job = _AutoPublishJob(ids, self._setup.settings.queue_dir, self._setup.settings.work_dir or None, settings,
                              config)
        job.signals.finished.connect(self._on_auto_published)
        job.signals.errored.connect(self._on_auto_publish_failed)
        self._auto_job = job
        count = len(ids)
        self.statusBar.showMessage(f'Publishing {count:,} accepted title{"" if count == 1 else "s"}...')
        self._auto_pool.start(job)

    def _auto_publish_done(self) -> None:
        self._auto_job = None
        self._index_dirty = True   # the entries are 'published' now, and the index has not read them
        if not self._title_open:
            self._sync_index_if_dirty()
        self._start_auto_publish()  # accepted while it went on

    def _on_auto_published(self, outcome: AutoPublishOutcome) -> None:
        titles = {row.id: row.title or row.display_name for row in self._model.rows}
        self.statusBar.showMessage(outcome.describe(titles), 20000)
        self._auto_publish_done()
        self.auto_published.emit(outcome)

    def _on_auto_publish_failed(self, message: str) -> None:
        self.statusBar.showMessage(f'Accepted, but publishing failed: {message}. Publish tries again.', 20000)
        self._auto_publish_done()
