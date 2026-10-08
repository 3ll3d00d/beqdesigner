'''
TODO R9: operating an unattended catalogue run -- a run that has no TMDB key says so, and a running job says how fast it
goes and when it should end.
'''
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from pipeline.service import models
from pipeline.service.api import create_app
from pipeline.service.config import ServiceConfig
from pipeline.service.jobs import JobManager, RunRequest

TOKEN = 'test-token'


def test_a_running_jobs_rate_and_estimate_come_from_how_far_it_has_got():
    progress = models.Progress(done=10, total=30, title='Heat', stage='design')

    rated = models.with_rate(progress, started_at=1000.0, now=4600.0)   # an hour for the first ten

    assert rated.per_hour == 10 and rated.remaining_seconds == 7200
    assert rated.estimated_finish == datetime.fromtimestamp(11800, tz=timezone.utc)
    assert models.with_rate(progress.model_copy(update={'done': 0}), 1000.0, 4600.0).per_hour is None   # nothing done
    assert models.with_rate(None, 1000.0, 4600.0) is None


@pytest.mark.parametrize('state, rated', [('running', True), ('succeeded', False)])
def test_only_a_running_job_is_estimated(state, rated):
    from pipeline.library.stages import Progress
    job = SimpleNamespace(id='j', origin='schedule', state=state, submitted_at=900.0, started_at=1000.0,
                          finished_at=None if state == 'running' else 5000.0, kind='run', request=RunRequest(),
                          progress=Progress(10, 30, 'Heat', 'design'), error='', result=None, joined_to=None)

    model = models.job_model(job, now=4600.0)

    assert (model.progress.per_hour is not None) is rated and (model.progress.estimated_finish is not None) is rated


@pytest.fixture
def profile(tmp_path):
    (tmp_path / 'work').mkdir()
    config = {'sources': [{'name': 'disk', 'kind': 'filesystem', 'globs': [str(tmp_path / 'media')]}],
              'run': {'work_dir': str(tmp_path / 'work'), 'queue_dir': str(tmp_path / 'queue'), 'designer': 'manual'}}
    path = tmp_path / 'profile.json'
    path.write_text(json.dumps(config))
    return str(path)


@pytest.mark.parametrize('env, configured', [({}, False), ({'TMDB_API_KEY': 'k'}, True)])
def test_ready_and_status_say_whether_tmdb_is_looked_up_without_making_it_required(profile, env, configured):
    manager = JobManager(lambda job, control: None)
    app = create_app(manager, ServiceConfig(profile_path=profile, token=TOKEN), env=env)
    client = TestClient(app)
    try:
        checks = {c['name']: c for c in client.get('/ready').json()['checks']}
        status = client.get('/v1/status', headers={'Authorization': f'Bearer {TOKEN}'}).json()
    finally:
        manager.stop(1)

    assert checks['tmdb']['ok'] is configured and checks['tmdb']['required'] is False
    assert status['tmdb'] is configured
