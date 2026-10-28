'''
TODO IG: a run keeps one design more in hand per designer slot than the pool has slots, so the next title's request is
prepared, and the last one's result written, while a slot designs. The pool still sends no endpoint more than its
`parallel` at once, and a design waiting for a slot gives up (DesignNotStarted) once its run stops wanting it.
'''
import threading

import pytest

from pipeline.designer.pool import DesignNotStarted, DesignerPool, not_started, starting_only_while
from pipeline.designer.registry import register_designer, unregister_designer
from pipeline.library.run import LibraryRunConfig, LibraryRunReport, UnitWork, design_unit_work
from pipeline.library.selection import Selection
from pipeline.library.stages import run_stages
from test_pipeline_designer_pool import Clock, FakeEndpoint, _request
from test_pipeline_library_index import DESIGNER, _extracted, _item, _needs, _profile, _scan, env  # noqa: F401 (a fixture)


# --- the pool ------------------------------------------------------------------------------------------------------

def _waiting_design(pool, may_start, outcome):
    ''' A design sent through `pool` under `may_start`, on its own thread; `outcome` gets what it ended with. '''
    def design():
        try:
            with starting_only_while(may_start):
                outcome.append(pool(_request()))
        except Exception as error:
            outcome.append(error)
    thread = threading.Thread(target=design)
    thread.start()
    return thread


def test_a_design_waiting_for_a_slot_is_not_sent_once_its_run_stops_wanting_it():
    release = threading.Event()
    one = FakeEndpoint('http://one', hold=release)
    pool = DesignerPool('d', [one.endpoint(parallel=1)], clock=Clock())
    wanted = {'second': True}
    first, second = [], []
    holder = _waiting_design(pool, lambda: True, first)
    assert _until(lambda: one.active == 1)
    waiter = _waiting_design(pool, lambda: wanted['second'], second)

    wanted['second'] = False
    release.set()
    holder.join(5)
    waiter.join(5)

    assert len(one.designed) == 1                  # only the first was sent
    assert isinstance(second[0], DesignNotStarted)
    assert all(s.in_use == 0 for s in pool.states())


def test_a_design_waiting_for_a_slot_is_sent_as_soon_as_one_comes_free():
    release = threading.Event()
    one = FakeEndpoint('http://one', hold=release)
    pool = DesignerPool('d', [one.endpoint(parallel=1)], clock=Clock())
    first, second = [], []
    holder = _waiting_design(pool, lambda: True, first)
    assert _until(lambda: one.active == 1)
    waiter = _waiting_design(pool, lambda: True, second)
    release.set()
    holder.join(5)
    waiter.join(5)

    assert len(one.designed) == 2 and one.most_at_once == 1


def test_the_runs_check_is_asked_without_the_pools_lock_held():
    ''' The service's check takes the job manager's lock, and its status route reads the pool under that lock. '''
    one = FakeEndpoint('http://one')
    pool = DesignerPool('d', [one.endpoint(parallel=1)], clock=Clock())
    read_from_elsewhere = []

    def may_start():
        reader = threading.Thread(target=lambda: read_from_elsewhere.append(pool.states()))
        reader.start()
        reader.join(2)
        return not reader.is_alive()

    with starting_only_while(may_start):
        pool(_request())

    assert read_from_elsewhere and len(one.designed) == 1


def test_not_started_is_found_through_what_it_was_raised_from():
    try:
        try:
            raise DesignNotStarted('stopped')
        except DesignNotStarted as error:
            raise RuntimeError('design failed') from error
    except RuntimeError as wrapped:
        assert not_started(wrapped)
    assert not not_started(RuntimeError('design failed')) and not not_started(None)


def test_design_unit_work_lets_a_design_not_started_through_without_reporting_a_failure(env, monkeypatch):
    item = _item('a')
    _scan(env, item)

    def design(*args, **kwargs):
        raise DesignNotStarted('stopped')

    monkeypatch.setattr('pipeline.library.run.design_if_needed', design)
    config = LibraryRunConfig(work_dir=env.work, queue_dir=env.queue, designer=DESIGNER)

    with pytest.raises(DesignNotStarted):
        design_unit_work(UnitWork(item, item, 'mono.wav', env.work), config, env.index)

    assert env.index.failure('fs-a') is None


# --- the run -------------------------------------------------------------------------------------------------------

class Slot:
    ''' A one-slot pool registered as the run's designer, and a design_unit_work that goes through it. '''

    def __init__(self):
        self.release = threading.Event()
        self.endpoint = FakeEndpoint('http://one', hold=self.release)
        self.pool = DesignerPool(DESIGNER, [self.endpoint.endpoint(parallel=1)], clock=Clock())
        self.prepared, self.written = [], []
        self.lock = threading.Lock()

    def design(self, work, config, index, on_stage=None):
        with self.lock:
            self.prepared.append(work.item.id)
        self.pool(_request())
        with self.lock:
            self.written.append(work.item.id)
        return LibraryRunReport(designed=[work.item.id])


@pytest.fixture
def slot(monkeypatch):
    from pipeline.library import stages
    slot = Slot()
    register_designer(DESIGNER, slot.pool)
    monkeypatch.setattr(stages, 'design_unit_work', slot.design)
    yield slot
    slot.release.set()
    unregister_designer(DESIGNER)


def _straight_to_design(env, monkeypatch, *names):
    ''' Titles whose extraction is current, so each goes straight to design. '''
    from pipeline.library import stages
    items = [_item(name) for name in names]
    for item in items:
        _extracted(env, item)
    _scan(env, *items)
    assert all(_needs(env, item.id)[0] == 'design' for item in items)
    monkeypatch.setattr(stages, 'cached_unit_work', lambda unit, config: (UnitWork(unit, unit, 'mono.wav', unit.id), ''))


def _until(condition, seconds=5.0):
    done = threading.Event()
    for _ in range(int(seconds / 0.01)):
        if condition():
            return True
        done.wait(0.01)
    return condition()


def _run(env, ids, **kwargs):
    config = LibraryRunConfig(work_dir=env.work, queue_dir=env.queue, designer=DESIGNER)
    result = {}
    thread = threading.Thread(target=lambda: result.setdefault('report', run_stages(
        _profile(env), Selection(ids=ids), 'design', run_config=config, index=env.index, settings=env.settings,
        refresh=False, **kwargs)))
    thread.start()
    return thread, result


def test_the_next_title_is_prepared_while_the_slot_designs_and_no_endpoint_gets_more_than_its_parallel(env, slot,
                                                                                                      monkeypatch):
    _straight_to_design(env, monkeypatch, 'a', 'b', 'c')

    thread, result = _run(env, ('fs-a', 'fs-b', 'fs-c'))

    assert _until(lambda: slot.endpoint.active == 1 and len(slot.prepared) == 2), \
        'the second title was not prepared while the first designed'
    slot.release.set()
    thread.join(10)

    assert sorted(result['report'].run.designed) == ['fs-a', 'fs-b', 'fs-c']
    assert slot.endpoint.most_at_once == 1


def test_a_title_waiting_for_a_slot_when_the_run_is_cancelled_is_left_for_the_next_run(env, slot, monkeypatch):
    _straight_to_design(env, monkeypatch, 'a', 'b')
    cancel = threading.Event()
    events = []

    thread, result = _run(env, ('fs-a', 'fs-b'), should_cancel=cancel.is_set, on_event=events.append)
    assert _until(lambda: slot.endpoint.active == 1 and len(slot.prepared) == 2)
    cancel.set()
    slot.release.set()
    thread.join(10)
    report = result['report']

    assert len(report.run.designed) == 1 and len(slot.endpoint.designed) == 1
    waiting = report.not_run
    assert len(waiting) == 1 and not report.run.failed and report.cancelled
    assert env.index.failure(waiting[0]) is None
    assert any(e.kind == 'cancelled' and e.title_id == waiting[0] for e in events)


def test_a_title_extracted_in_the_run_is_still_designed_after_a_cancel(env, slot, monkeypatch):
    ''' As before the lookahead: what the run extracted is in hand, and designed whatever happens. '''
    from pipeline.library import stages
    _scan(env, _item('a'), _item('b'))
    cancel = threading.Event()

    def extract(session, unit, config, local_report, index, **kwargs):
        return UnitWork(unit, unit, 'mono.wav', unit.id)

    monkeypatch.setattr(stages, 'run_unit', extract)
    config = LibraryRunConfig(work_dir=env.work, queue_dir=env.queue, designer=DESIGNER, extract_parallelism=2)
    result = {}
    thread = threading.Thread(target=lambda: result.setdefault('report', run_stages(
        _profile(env), Selection(ids=('fs-a', 'fs-b')), 'design', run_config=config, index=env.index,
        settings=env.settings, refresh=False, should_cancel=cancel.is_set)))
    thread.start()
    assert _until(lambda: slot.endpoint.active == 1 and len(slot.prepared) == 2)
    cancel.set()
    slot.release.set()
    thread.join(10)

    assert sorted(result['report'].run.designed) == ['fs-a', 'fs-b']
