'''
pipeline/service/lease.py (design/pipeline-service.md §5.1): a service job holds the work directory, so the work list does not
start a second run over the same index, queue and repositories; a dead service's lease goes stale and is taken over.
'''
import json
import os
import time

import pytest

from pipeline.service.lease import STALE_AFTER_SECONDS, LeaseHeld, LeaseHolder, WorkDirLease, lease_path, read_lease


def _write(work_dir, **fields):
    holder = {'host': 'nas', 'pid': 42, 'job_id': 'other-job', 'heartbeat_at': time.time(), **fields}
    os.makedirs(os.path.dirname(lease_path(str(work_dir))), exist_ok=True)
    with open(lease_path(str(work_dir)), 'w') as f:
        json.dump(holder, f)


def test_a_job_holds_the_lease_while_it_runs_and_gives_it_up_after(tmp_path):
    with WorkDirLease(str(tmp_path), 'job-1', host='box', pid=7):
        holder = read_lease(str(tmp_path))
        assert (holder.host, holder.pid, holder.job_id) == ('box', 7, 'job-1')
        assert 'the pipeline service on box is running a job (job-1)' in holder.describe()

    assert read_lease(str(tmp_path)) is None and not os.path.exists(lease_path(str(tmp_path)))


def test_a_fresh_lease_of_another_process_is_refused_and_a_stale_one_taken_over(tmp_path):
    _write(tmp_path)
    with pytest.raises(LeaseHeld, match='the pipeline service on nas is running a job'):
        with WorkDirLease(str(tmp_path), 'job-2', host='box', pid=7):
            pass

    _write(tmp_path, heartbeat_at=time.time() - STALE_AFTER_SECONDS - 1)
    assert read_lease(str(tmp_path)) is None
    with WorkDirLease(str(tmp_path), 'job-2', host='box', pid=7):
        assert read_lease(str(tmp_path)).job_id == 'job-2'


def test_the_heartbeat_keeps_the_lease_fresh(tmp_path):
    with WorkDirLease(str(tmp_path), 'job', heartbeat_seconds=0.01):
        first = read_lease(str(tmp_path)).heartbeat_at
        deadline = time.time() + 5
        while read_lease(str(tmp_path)).heartbeat_at == first:
            assert time.time() < deadline, 'no heartbeat'
            time.sleep(0.01)


def test_leaving_never_removes_a_lease_someone_else_took_over(tmp_path):
    with WorkDirLease(str(tmp_path), 'job', host='box', pid=7):
        _write(tmp_path, job_id='theirs')   # our lease went stale and was taken over

    assert read_lease(str(tmp_path)).job_id == 'theirs'


def test_no_work_directory_or_a_damaged_lease_is_no_lease(tmp_path):
    assert read_lease(None) is None and read_lease('') is None
    os.makedirs(os.path.dirname(lease_path(str(tmp_path))))
    with open(lease_path(str(tmp_path)), 'w') as f:
        f.write('{not json')

    assert read_lease(str(tmp_path)) is None


def test_a_service_job_waits_while_another_holds_the_work_directory_and_holds_it_while_it_runs(tmp_path, monkeypatch):
    ''' design/archive/library-sync/worklist-feedback.md F5: it no longer fails; a scan (or a publish) waits for the run
        in progress to end. '''
    import os
    from pipeline.service import work
    from pipeline.service.jobs import JobControl, Job, ScanRequest
    from pipeline.service.lease import lease_path
    work_dir = tmp_path / 'work'

    class Context:
        def __init__(self):
            self.work_dir = str(work_dir)
    seen, slept = [], []
    monkeypatch.setattr(work, 'scan', lambda context, request: seen.append(read_lease(str(work_dir))) or 'scanned')

    def sleep(seconds):   # the other run ends while this one waits
        slept.append(seconds)
        os.unlink(lease_path(str(work_dir)))

    execute = work.executor('profile.yaml', env={}, load=lambda path, env: Context(), sleep=sleep)
    job = Job('job-9', 'scan', 'api', ScanRequest())

    assert execute(job, JobControl(None, job)) == 'scanned' and seen[0].job_id == 'job-9' and slept == []
    assert read_lease(str(work_dir)) is None
    _write(work_dir)
    assert execute(job, JobControl(None, job)) == 'scanned' and len(slept) == 1 and seen[1].job_id == 'job-9'


def test_a_job_cancelled_while_it_waits_for_the_work_directory_ends_without_running(tmp_path, monkeypatch):
    from pipeline.service import work
    from pipeline.service.jobs import JobControl, Job, ScanRequest
    work_dir = tmp_path / 'work'

    class Context:
        def __init__(self):
            self.work_dir = str(work_dir)
    monkeypatch.setattr(work, 'scan', lambda context, request: pytest.fail('ran while another run held it'))
    job = Job('job-9', 'scan', 'api', ScanRequest())
    execute = work.executor('profile.yaml', env={}, load=lambda path, env: Context(),
                            sleep=lambda seconds: setattr(job, 'cancel_requested', True))
    _write(work_dir)

    assert execute(job, JobControl(None, job)) is None and job.cancel_requested


def test_describe_names_the_short_job_id():
    assert '(12345678)' in LeaseHolder('h', 1, '12345678-aaaa-bbbb', 0.0).describe()
