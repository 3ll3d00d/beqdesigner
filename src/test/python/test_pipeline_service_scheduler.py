'''Automatic schedule, persistence and job submission.'''
import json
import threading
import time

import pytest
from pydantic import ValidationError

from pipeline.service.jobs import JobManager, ScanRequest
from pipeline.service.models import ScheduleUpdate
from pipeline.service.scheduler import AutoScheduler


def _until(predicate):
    deadline = time.time() + 5
    while not predicate():
        assert time.time() < deadline
        time.sleep(0.005)


def test_tick_forces_needs_and_waits_from_finish(tmp_path):
    now = [1000.0]
    release = threading.Event()
    calls = []

    def execute(job, control):
        calls.append(job)
        release.wait(5)
        return None

    manager = JobManager(execute, clock=lambda: now[0])
    scheduler = AutoScheduler(manager, str(tmp_path),
                              {'enabled': True, 'interval_minutes': 5,
                               'filter': {'needs': ['publish'], 'new_since_scan': True, 'kind': 'movie'},
                               'through': 'extract'}, clock=lambda: now[0], start=False)
    try:
        now[0] = 1300
        scheduler.tick()
        _until(lambda: len(calls) == 1)
        job = calls[0]
        assert job.origin == 'schedule' and job.request.through == 'extract' and job.request.scan_first
        assert job.request.selection.needs == ('extract', 'design')
        # design/archive/library-sync/worklist-feedback.md F4: a failed extraction is not repeated every tick
        assert job.request.unattended
        assert not job.request.selection.new_since_scan and job.request.selection.kind == 'movie'
        now[0] = 1700
        scheduler.tick()
        assert len(calls) == 1
        release.set()
        _until(lambda: job.finished)
        assert scheduler.snapshot()['next_run_at'] == 2000
        now[0] = 1999
        scheduler.tick()
        assert len(calls) == 1
        now[0] = 2000
        scheduler.tick()
        _until(lambda: len(calls) == 2)
    finally:
        release.set()
        scheduler.stop()
        manager.stop(5)


def test_busy_skip_and_trigger_do_not_queue_ticks(tmp_path):
    now = [1000.0]
    release = threading.Event()
    manager = JobManager(lambda job, control: release.wait(5), clock=lambda: now[0])
    scheduler = AutoScheduler(manager, str(tmp_path), {'enabled': True, 'interval_minutes': 5},
                              clock=lambda: now[0], start=False)
    try:
        other = manager.submit(ScanRequest())
        now[0] = 1300
        scheduler.tick()
        assert scheduler.snapshot()['last_skip'] == 'busy'
        assert scheduler.snapshot()['next_run_at'] == 1600
        assert scheduler.trigger() is None
        assert [job.id for job in manager.jobs()] == [other.id]
        release.set()
        _until(lambda: other.finished)
        assert scheduler.trigger() is not None
    finally:
        release.set()
        scheduler.stop()
        manager.stop(5)


def test_saved_schedule_wins_and_pause_resume(tmp_path):
    manager = JobManager(lambda job, control: None)
    first = AutoScheduler(manager, str(tmp_path), {'enabled': True, 'interval_minutes': 7}, start=False)
    try:
        paused = first.update(ScheduleUpdate(enabled=False, interval_minutes=9, through='extract'))
        assert paused['next_run_at'] is None
        assert json.loads((tmp_path / 'schedule.json').read_text())['interval_minutes'] == 9
        second = AutoScheduler(manager, str(tmp_path), {'enabled': True, 'interval_minutes': 7}, start=False)
        try:
            assert second.settings.interval_minutes == 9 and not second.settings.enabled
            resumed = second.update(ScheduleUpdate(enabled=True, interval_minutes=9))
            assert resumed['next_run_at'] is not None
        finally:
            second.stop()
    finally:
        first.stop()
        manager.stop(5)


@pytest.mark.parametrize('data', [{'interval_minutes': 4}, {'through': 'publish'}, {'through': 'commit'}])
def test_schedule_rejects_unsafe_settings(data):
    with pytest.raises(ValidationError):
        ScheduleUpdate.model_validate(data)
