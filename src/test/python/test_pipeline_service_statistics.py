'''
What an execution says about how it ran (design/jobs-and-executions.md §2.3): the designer pool reports each call, the
queue keeps a record per title from the run's events, and pipeline/service/statistics.py makes the outcomes, the
designers and the stage times of them -- over plain records, over the real run_stages with a fake pool, and over HTTP.
'''
import json
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from model.execution_events import ExecutionEvent, event_scope, execution_event_context
from pipeline.designer.pool import DesignerPool, PoolEndpoint, designing
from pipeline.designer.registry import register_designer, unregister_designer
from pipeline.library.run import LibraryRunConfig, LibraryRunReport, UnitWork
from pipeline.library.selection import Selection
from pipeline.library.stages import StagesReport, run_stages
from pipeline.service.api import create_app
from pipeline.service.config import ServiceConfig
from pipeline.service.executions import Execution, ExecutionQueue, RunRequest
from pipeline.service.statistics import figures, statistics, title_outcomes
from pipeline.service.work import RunOutcome, job_failed
from service_http import make_job
from test_pipeline_library_index import DESIGNER, _extracted, _item, _profile, _scan, env  # noqa: F401 (a fixture)

TOKEN = 'test-token'
AUTH = {'Authorization': f'Bearer {TOKEN}'}


def _until(condition, timeout=10.0):
    deadline = time.time() + timeout
    while not condition():
        assert time.time() < deadline, 'timed out'
        time.sleep(0.005)


# --- the figures, over plain records ----------------------------------------------------------------------------------

def test_figures_are_the_mean_median_90th_percentile_and_max():
    found = figures([10, 20, 30, 40, 50, 60, 70, 80, 90, 1000])
    assert (found.count, found.mean, found.p50, found.p90, found.max) == (10, 145.0, 50, 90, 1000)
    assert figures([]) is None and figures([None, -1]) is None


def _record(**times):
    return {'title': times.pop('title', 'T'), **times}


def test_each_title_is_given_one_outcome_and_its_stage_times():
    result = RunOutcome(None, StagesReport(
        'design', 7, LibraryRunReport(designed=['a', 'b'], design_cached=['c'], failed=[('d', 'bad rip\nmore')],
                                      unavailable=[('e', 'designer down')], failed_earlier=[('f', 'gone')]),
        attempted=['a', 'b', 'c', 'd', 'e'], not_run=['g'], cancelled=True))
    records = {
        'a': _record(title='Alien', extract_start=0, extract_end=40, design_queued=40, sent=55, answered=130, ended=134,
                     endpoint='http://one', release='0.3.1', version='8', call='designed'),
        'b': _record(title='Brazil', design_queued=0, sent=5, answered=8, ended=9, endpoint='http://two', release='0.3.1',
                     version='8', call='declined'),
    }
    outcomes = {t.id: t for t in title_outcomes(result, records)}

    assert {k: v.outcome for k, v in outcomes.items()} == {'a': 'designed', 'b': 'declined', 'c': 'already_done',
                                                         'd': 'failed', 'e': 'unavailable', 'f': 'skipped',
                                                         'g': 'not_reached'}
    alien = outcomes['a']
    assert (alien.title, alien.extract_seconds, alien.waiting_seconds, alien.designer_seconds, alien.writing_seconds) == \
        ('Alien', 40, 15, 75, 4)
    assert (alien.endpoint, alien.release, alien.version) == ('http://one', '0.3.1', '8')
    assert outcomes['d'].reason == 'bad rip\nmore' and outcomes['f'].reason == 'failed before: gone'


def test_the_statistics_count_outcomes_group_reasons_and_figure_each_designer():
    result = {'report': {'through': 'design', 'attempted': ['a', 'b', 'c', 'd'],
                         'run': {'designed': ['a', 'b', 'c'], 'failed': [{'id': 'd', 'message': 'bad rip'}]}}}
    records = {'a': _record(sent=0, answered=60, ended=61, endpoint='http://one', release='0.3.1', version='8',
                            call='designed'),
               'b': _record(sent=0, answered=120, ended=121, endpoint='http://one', release='0.3.1', version='8',
                            call='declined'),
               'c': _record(sent=0, answered=90, ended=92, endpoint='http://two', release='0.2.12', version='7',
                            call='designed')}
    execution = Execution('x', 'run', 'api', RunRequest(), state='succeeded', started_at=1000.0, finished_at=1000.0 + 3600,
                          result=result, titles=records)

    found = statistics(execution)

    assert found.outcomes == {'designed': 2, 'declined': 1, 'failed': 1}
    assert found.failures == [('bad rip', 1)] and found.skips == []
    one, two = found.designers
    assert (one.endpoint, one.release, one.version, one.designed, one.declined, one.failed) == \
        ('http://one', '0.3.1', '8', 1, 1, 0)
    assert (one.seconds.mean, one.seconds.max) == (90.0, 120)
    assert (two.endpoint, two.designed) == ('http://two', 1) and found.designers_differ
    assert (found.wall_seconds, found.per_hour) == (3600.0, 4.0)
    assert set(found.stages) == {'designer', 'writing'} and found.stages['designer'].count == 3
    assert [t.id for t in found.slowest] == ['b', 'c', 'a']


def test_a_running_execution_is_figured_from_what_its_titles_have_reported_so_far():
    records = {'a': _record(ended=5, end='title_completed', call='designed', sent=1, answered=4),
               'b': _record(ended=6, end='failed', message='boom'),
               'c': _record(design_queued=3)}
    execution = Execution('x', 'run', 'api', RunRequest(), state='running', started_at=0.0, titles=records)
    found = statistics(execution)
    assert found.outcomes == {'designed': 1, 'failed': 1, 'not_reached': 1} and found.failures == [('boom', 1)]
    assert found.wall_seconds == 6.0


# --- the pool reports each call, and the queue keeps it with the title ----------------------------------------------

class Endpoint:
    ''' A designer endpoint whose /health names a release and design version, and which declines the titles asked. '''

    def __init__(self, url, decline=()):
        self.url, self.decline = url, set(decline)

    def health(self):
        return {'status': 'ok', 'design_version': 8, 'designer_release': '0.3.1', 'designer_build': '0.3.1+src:abc'}

    def call(self, request):
        return SimpleNamespace(decline_reason='no_rolloff_detected' if request in self.decline else None)

    def endpoint(self):
        return PoolEndpoint(self.url, self.call, self.health, parallel=1)


def test_the_pool_reports_each_call_it_makes_inside_a_run():
    events = []
    pool = DesignerPool('d', [Endpoint('http://one', decline={'b'}).endpoint()])
    with execution_event_context('run', events.append):
        for title in ('a', 'b'):
            with event_scope(title_id=title, stage='design'), designing(title):
                pool(title)
    pool('c')   # outside a run: nothing to report to

    calls = [(e.title_id, e.stage, e.kind, dict(e.fields)) for e in events]
    assert [(t, s, k, f['outcome'], f['endpoint'], f['release'], f['version']) for t, s, k, f in calls] == [
        ('a', 'design', 'designer_call', 'designed', 'http://one', '0.3.1', '8'),
        ('b', 'design', 'designer_call', 'declined', 'http://one', '0.3.1', '8')]
    assert all(float(f['answered']) >= float(f['sent']) > 0 for *_, f in calls)


def test_the_queue_keeps_a_record_of_each_title_from_the_runs_events(tmp_path):
    def execute(execution, control):
        for kind, title_id, stage, message, fields in (
                ('queued', 'a', '', 'Alien', ()), ('stage_queued', 'a', 'design', 'Waiting for design slot', ()),
                ('stage_started', 'a', 'design', 'Designing filter', ()),
                ('designer_call', 'a', 'design', 'designed on http://one',
                 (('endpoint', 'http://one'), ('release', '0.3.1'), ('version', '8'), ('sent', '10.5'),
                  ('answered', '70.25'), ('outcome', 'designed'))),
                ('stage_completed', 'a', 'design', 'Design complete', ()),
                ('title_completed', 'a', '', 'Alien', ())):
            control.event(ExecutionEvent('r', title_id, stage, kind, 100.0, message, fields=fields))
        return RunOutcome(None, StagesReport('design', 1, LibraryRunReport(designed=['a']), attempted=['a']))

    queue = ExecutionQueue(execute, state_dir=str(tmp_path))
    try:
        execution = queue.submit(RunRequest())
        _until(lambda: execution.finished)
        record = execution.titles['a']
        assert record == {'queued': 100.0, 'title': 'Alien', 'design_queued': 100.0, 'design_start': 100.0,
                          'endpoint': 'http://one', 'release': '0.3.1', 'version': '8', 'sent': 10.5, 'answered': 70.25,
                          'call': 'designed', 'design_end': 100.0, 'ended': 100.0, 'end': 'title_completed',
                          'message': 'Alien'}
    finally:
        queue.stop(5)
    # read once the queue has stopped writing (Windows: its os.replace may be swapping the file in as a test opens it)
    saved = json.loads((tmp_path / 'executions.json').read_text())
    assert saved[0]['titles']['a']['endpoint'] == 'http://one'   # kept with the execution, for the history
    again = ExecutionQueue(lambda e, c: None, state_dir=str(tmp_path))
    try:
        assert statistics(again.get(execution.id)).designers[0].designed == 1
    finally:
        again.stop(5)


# --- over the real run_stages --------------------------------------------------------------------------------------

def test_a_run_over_a_pool_says_which_endpoint_designed_each_title_and_how_long_it_took(env, monkeypatch):
    from model.execution_events import emit_execution_event
    from pipeline.library import stages
    items = [_item(n) for n in ('a', 'b', 'c')]
    for item in items:
        _extracted(env, item)
    _scan(env, *items)
    pool = DesignerPool(DESIGNER, [Endpoint('http://one', decline={'fs-b'}).endpoint(),
                                   Endpoint('http://two', decline={'fs-b'}).endpoint()])
    register_designer(DESIGNER, pool)
    monkeypatch.setattr(stages, 'cached_unit_work', lambda unit, config: (UnitWork(unit, unit, 'mono.wav', unit.id), ''))

    def design(work, config, index, on_stage=None):   # as design_unit_work does, around the designer
        emit_execution_event('stage_started', message='Designing filter')
        pool(work.item.id)
        emit_execution_event('stage_completed', message='Design complete')
        return LibraryRunReport(designed=[work.item.id])
    monkeypatch.setattr(stages, 'design_unit_work', design)
    config = LibraryRunConfig(work_dir=env.work, queue_dir=env.queue, designer=DESIGNER)

    def execute(execution, control):
        return RunOutcome(None, run_stages(_profile(env), Selection(ids=('fs-a', 'fs-b', 'fs-c')), 'design',
                                           run_config=config, index=env.index, settings=env.settings, refresh=False,
                                           should_cancel=control.should_stop, on_progress=control.progress,
                                           on_event=control.event))
    queue = ExecutionQueue(execute)
    try:
        execution = queue.submit(RunRequest(Selection(ids=('fs-a', 'fs-b', 'fs-c'))))
        _until(lambda: execution.finished)
        assert execution.state == 'succeeded', execution.error
        found = statistics(execution)
        assert found.outcomes == {'designed': 2, 'declined': 1}
        assert sum(d.designed + d.declined for d in found.designers) == 3
        assert {d.endpoint for d in found.designers} <= {'http://one', 'http://two'}
        assert all((d.release, d.version) == ('0.3.1', '8') for d in found.designers) and not found.designers_differ
        assert {'waiting', 'designer', 'writing'} <= set(found.stages) and found.stages['designer'].count == 3
        titles = {t.id: t for t in title_outcomes(execution.result, execution.titles)}
        assert titles['fs-a'].title == 'Film a' and titles['fs-b'].outcome == 'declined'
    finally:
        queue.stop(5)
        unregister_designer(DESIGNER)


# --- over HTTP ---------------------------------------------------------------------------------------------------------

@pytest.fixture
def profile(tmp_path):
    config = {'sources': [{'name': 'disk', 'kind': 'filesystem', 'globs': [str(tmp_path / 'media')]}],
              'designers': {'rolloff': 'http://127.0.0.1:9/design'},
              'run': {'work_dir': str(tmp_path / 'work'), 'queue_dir': str(tmp_path / 'queue'), 'designer': 'rolloff'}}
    path = tmp_path / 'profile.json'
    path.write_text(json.dumps(config))
    return path


def test_an_execution_says_how_it_ran_and_lists_its_titles_by_outcome_and_its_job_sums_them(profile, monkeypatch):
    monkeypatch.setattr('pipeline.library.setup.check_designer', lambda *args, **kwargs: None)
    release = threading.Event()

    def execute(execution, control):
        for title_id, outcome in (('a', 'designed'), ('b', 'declined')):
            control.event(ExecutionEvent('r', title_id, '', 'queued', 1.0, title_id.upper()))
            control.event(ExecutionEvent('r', title_id, 'design', 'designer_call', 2.0, '', fields=(
                ('endpoint', 'http://one'), ('release', '0.3.1'), ('version', '8'), ('sent', '2'), ('answered', '32'),
                ('outcome', outcome))))
            control.event(ExecutionEvent('r', title_id, '', 'title_completed', 33.0, title_id.upper()))
        assert release.wait(5)
        return RunOutcome(None, StagesReport('design', 3, LibraryRunReport(designed=['a', 'b'], failed=[('c', 'bad rip')]),
                                             attempted=['a', 'b', 'c']))

    queue = ExecutionQueue(execute, failed=job_failed)
    app = create_app(queue, ServiceConfig(profile_path=str(profile), token=TOKEN), env={})
    try:
        with TestClient(app) as client:
            job = make_job(client, AUTH, 'run').json()
            execution_id = job['last_execution']
            _until(lambda: client.get(f'/v1/executions/{execution_id}', headers=AUTH).json()['statistics']['outcomes']
                   == {'designed': 1, 'declined': 1})   # live: what its titles have said so far
            release.set()
            _until(lambda: client.get(f'/v1/executions/{execution_id}', headers=AUTH).json()['state'] == 'failed')
            stats = client.get(f'/v1/executions/{execution_id}', headers=AUTH).json()['statistics']
            assert stats['outcomes'] == {'designed': 1, 'declined': 1, 'failed': 1}
            assert stats['failures'] == [{'reason': 'bad rip', 'count': 1}]
            assert stats['designers'] == [{'endpoint': 'http://one', 'release': '0.3.1', 'version': '8', 'designed': 1,
                                           'declined': 1, 'failed': 0,
                                           'seconds': {'count': 2, 'mean': 30.0, 'p50': 30.0, 'p90': 30.0, 'max': 30.0}}]
            assert stats['stages']['designer']['mean'] == 30.0
            listed = client.get('/v1/executions', headers=AUTH).json()['executions'][0]
            assert listed['statistics'] is None   # the log stays light

            failed = client.get(f'/v1/executions/{execution_id}/titles', params={'outcome': 'failed'},
                                headers=AUTH).json()['titles']
            assert [(t['id'], t['reason']) for t in failed] == [('c', 'bad rip')]
            every = client.get(f'/v1/executions/{execution_id}/titles', headers=AUTH).json()['titles']
            assert [(t['id'], t['title'], t['outcome'], t['designer_seconds']) for t in every] == [
                ('a', 'A', 'designed', 30.0), ('b', 'B', 'declined', 30.0), ('c', 'c', 'failed', None)]
            assert client.get(f"/v1/jobs/{job['id']}", headers=AUTH).json()['outcomes'] == \
                {'designed': 1, 'declined': 1, 'failed': 1}
            assert client.get('/v1/executions/nope/titles', headers=AUTH).status_code == 404
    finally:
        release.set()
        queue.stop(5)
