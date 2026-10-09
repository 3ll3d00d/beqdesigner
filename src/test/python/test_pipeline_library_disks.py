"""Extraction reads each physical disk one title at a time (run.disks): pipeline.library.disks and run_stages."""
import os
import threading
import time
from collections import Counter

import pytest

from pipeline.library import disks as disks_module
from pipeline.library.disks import DiskLimit, DiskSlots, disk_limit, disks_of
from pipeline.library.run import LibraryRunReport, UnitWork
from pipeline.library.selection import Selection
from pipeline.library.setup import run_config_from_values
from pipeline.library.stages import run_stages
from test_pipeline_library_index import _item, _scan, env  # noqa: F401 (env is a fixture)
from test_pipeline_library_stages import DESIGNER, _profile, _run_config


def _located(monkeypatch, where):
    ''' os.getxattr as Unraid's shfs answers it: `where` maps a path to its disk(s); anything else has no attribute. '''
    asked = []

    def getxattr(path, name):
        asked.append((path, name))
        if path not in where:
            raise OSError(61, 'No data available')
        return where[path].encode()
    monkeypatch.setattr(os, 'getxattr', getxattr, raising=False)   # Windows and macOS have none
    return asked


# --- the profile setting -----------------------------------------------------------------------------------------------

def test_run_disks_names_the_attribute_and_how_many_titles_may_read_a_disk_at_once():
    assert disk_limit() is None
    assert disk_limit({'xattr': 'system.LOCATION'}) == DiskLimit('system.LOCATION', 1)
    assert disk_limit({'xattr': ' user.mergerfs.basepath ', 'per_disk': 2}) == DiskLimit('user.mergerfs.basepath', 2)


@pytest.mark.parametrize('value', ['system.LOCATION', {}, {'xattr': ''}, {'xattr': 3}, {'xattr': 'a', 'per_disk': 0},
                                   {'xattr': 'a', 'per_disk': True}, {'xattr': 'a', 'per_disk': 1.5},
                                   {'xattr': 'a', 'spare': 1}])
def test_a_bad_run_disks_is_refused(value):
    with pytest.raises(ValueError, match='run.disks'):
        disk_limit(value)


def test_the_cli_and_service_take_run_disks_from_the_profile(tmp_path):
    values = {'work_dir': str(tmp_path), 'queue_dir': str(tmp_path), 'designer': 'manual',
              'disks': {'xattr': 'system.LOCATION', 'per_disk': 1}}
    assert run_config_from_values(values, {}).disks == DiskLimit('system.LOCATION', 1)
    assert run_config_from_values({**values, 'disks': None}, {}).disks is None


def test_the_work_list_takes_run_disks_from_the_profile(tmp_path):
    from types import SimpleNamespace
    from model.worklist_run import build_run_config
    from pipeline.config import AnalysisConfig
    settings = SimpleNamespace(work_dir=str(tmp_path), queue_dir=str(tmp_path), designer='manual',
                               config=AnalysisConfig(), coverage='complete_programme', keep_multichannel=False,
                               tv_mode='episode')
    profile = SimpleNamespace(config={'run': {'disks': {'xattr': 'system.LOCATION'}}})
    config = build_run_config(SimpleNamespace(settings=settings, profile=profile), {})
    assert config.disks == DiskLimit('system.LOCATION', 1)


# --- where a title is ------------------------------------------------------------------------------------------------

def test_a_title_is_on_the_disks_its_files_say_and_on_none_when_they_do_not(monkeypatch):
    asked = _located(monkeypatch, {'/films/a.mkv': 'disk2', '/films/s/e1.mkv': 'disk1', '/films/s/e2.mkv': 'disk1,disk3'})

    assert disks_of(['/films/a.mkv'], 'system.LOCATION') == {'disk2'}
    assert disks_of(['/films/s/e1.mkv', '/films/s/e2.mkv'], 'system.LOCATION') == {'disk1', 'disk3'}
    assert disks_of(['/films/elsewhere.mkv'], 'system.LOCATION') == frozenset()
    assert asked[0] == ('/films/a.mkv', 'system.LOCATION')


def test_without_xattrs_on_the_platform_no_title_has_a_disk(monkeypatch):
    monkeypatch.delattr(disks_module.os, 'getxattr', raising=False)
    assert disks_of(['/films/a.mkv'], 'system.LOCATION') == frozenset()


def test_a_title_may_start_only_when_every_disk_it_reads_has_room():
    slots = DiskSlots(per_disk=1)
    slots.take(frozenset({'disk1'}))

    assert not slots.free(frozenset({'disk1'})) and not slots.free(frozenset({'disk1', 'disk2'}))
    assert slots.free(frozenset({'disk2'})) and slots.free(frozenset())
    assert slots.busy(frozenset({'disk1', 'disk2'})) == {'disk1'}
    slots.give_back(frozenset({'disk1'}))
    assert slots.free(frozenset({'disk1', 'disk2'}))


# --- the run -----------------------------------------------------------------------------------------------------------

def _counting_extract(where):
    ''' A fake extraction that records the most titles read from each disk at once, and from any disk. '''
    lock = threading.Lock()
    reading, most = Counter(), Counter()

    def extract(session, unit, config, local_report, index, **kwargs):
        item = unit.item if hasattr(unit, 'item') else unit
        disk = where.get(item.source_path, '?')
        with lock:
            reading[disk] += 1
            reading['all'] += 1
            most[disk] = max(most[disk], reading[disk])
            most['all'] = max(most['all'], reading['all'])
        time.sleep(0.05)
        with lock:
            reading[disk] -= 1
            reading['all'] -= 1
        return UnitWork(unit, item, 'mono.wav', item.id)
    return extract, most


def _design(work, config, index, on_stage=None):
    return LibraryRunReport(designed=[work.item.id])


def test_each_disk_is_read_by_one_extraction_at_a_time_while_other_disks_are_read_alongside(env, monkeypatch):
    from pipeline.library import stages
    where = {'/films/a.mkv': 'disk1', '/films/b.mkv': 'disk1', '/films/c.mkv': 'disk2', '/films/d.mkv': 'disk2'}
    _located(monkeypatch, where)
    _scan(env, *(_item(name) for name in 'abcde'))   # e: no attribute, so not held back by disk
    extract, most = _counting_extract(where)
    monkeypatch.setattr(stages, 'run_unit', extract)
    monkeypatch.setattr(stages, 'design_unit_work', _design)
    events = []

    report = run_stages(_profile(env), Selection(ids=tuple(f'fs-{n}' for n in 'abcde')), 'design',
                        run_config=_run_config(env, extract_parallelism=4, disks=DiskLimit('system.LOCATION', 1)),
                        index=env.index, settings=env.settings, on_event=events.append)

    assert sorted(report.run.designed) == [f'fs-{n}' for n in 'abcde']
    assert most['disk1'] == most['disk2'] == 1
    assert most['all'] == 3   # disk1, disk2 and the unknown one together
    waiting = {e.title_id: e.message for e in events if e.kind == 'stage_queued' and e.stage == 'extract'}
    assert waiting == {'fs-b': 'Waiting to read disk1: another title is being extracted from it',
                       'fs-d': 'Waiting to read disk2: another title is being extracted from it'}


def test_per_disk_lets_that_many_titles_read_one_disk(env, monkeypatch):
    from pipeline.library import stages
    where = {f'/films/{n}.mkv': 'disk1' for n in 'abcd'}
    _located(monkeypatch, where)
    _scan(env, *(_item(name) for name in 'abcd'))
    extract, most = _counting_extract(where)
    monkeypatch.setattr(stages, 'run_unit', extract)
    monkeypatch.setattr(stages, 'design_unit_work', _design)

    report = run_stages(_profile(env), Selection(ids=tuple(f'fs-{n}' for n in 'abcd')), 'design',
                        run_config=_run_config(env, extract_parallelism=4, disks=DiskLimit('system.LOCATION', 2)),
                        index=env.index, settings=env.settings)

    assert len(report.run.designed) == 4 and most['disk1'] == 2


def test_a_failed_extraction_frees_its_disk_for_the_next_title(env, monkeypatch):
    from pipeline.library import stages
    where = {'/films/a.mkv': 'disk1', '/films/b.mkv': 'disk1'}
    _located(monkeypatch, where)
    _scan(env, _item('a'), _item('b'))
    extract, _ = _counting_extract(where)

    def failing(session, unit, config, local_report, index, **kwargs):
        if unit.id == 'fs-a':
            raise ValueError('bad rip')
        return extract(session, unit, config, local_report, index, **kwargs)
    monkeypatch.setattr(stages, 'run_unit', failing)
    monkeypatch.setattr(stages, 'design_unit_work', _design)

    report = run_stages(_profile(env), Selection(ids=('fs-a', 'fs-b')), 'design',
                        run_config=_run_config(env, extract_parallelism=2, disks=DiskLimit('system.LOCATION', 1)),
                        index=env.index, settings=env.settings)

    assert report.run.failed == [('fs-a', 'ValueError: bad rip')] and report.run.designed == ['fs-b']


def test_without_run_disks_no_disk_is_looked_up(env, monkeypatch):
    from pipeline.library import stages
    asked = _located(monkeypatch, {})
    _scan(env, _item('a'), _item('b'))
    extract, most = _counting_extract({})
    monkeypatch.setattr(stages, 'run_unit', extract)
    monkeypatch.setattr(stages, 'design_unit_work', _design)

    run_stages(_profile(env), Selection(ids=('fs-a', 'fs-b')), 'design', run_config=_run_config(env, extract_parallelism=2),
               index=env.index, settings=env.settings)

    assert asked == [] and most['all'] == 2


def test_a_season_reads_every_disk_its_episodes_are_on(env, monkeypatch):
    from pipeline.library import stages
    from pipeline.library.season import SeasonGroup
    from pipeline.library.status import ScanSettings
    where = {'/tv/show/e1.mkv': 'disk1', '/tv/show/e2.mkv': 'disk2', '/films/film.mkv': 'disk2'}
    _located(monkeypatch, where)
    episodes = [_item(f'show-e{n}', title='Some Show', kind='tv', season='1', episodes=(n,), source_path=f'/tv/show/e{n}.mkv')
                for n in (1, 2)]
    settings = ScanSettings(work_dir=env.work, queue_dir=env.queue, designer=DESIGNER, tv_mode='season')
    _scan(env, _item('film'), *episodes, settings=settings)
    lock, reading, most = threading.Lock(), Counter(), Counter()

    def extract(session, unit, config, local_report, index, **kwargs):
        members = unit.members if isinstance(unit, SeasonGroup) else (unit,)
        disks = {where[m.source_path] for m in members}
        with lock:
            reading.update(disks)
            most.update({d: 0 for d in disks})
            for d in disks:
                most[d] = max(most[d], reading[d])
        time.sleep(0.05)
        with lock:
            reading.subtract(disks)
        item = unit.item if isinstance(unit, SeasonGroup) else unit
        return UnitWork(unit, item, 'mono.wav', item.id)
    monkeypatch.setattr(stages, 'run_unit', extract)
    monkeypatch.setattr(stages, 'design_unit_work', _design)
    ids = tuple(row.id for row in env.index.titles())

    report = run_stages(_profile(env), Selection(ids=ids), 'design',
                        run_config=_run_config(env, extract_parallelism=2, tv_mode='season',
                                               disks=DiskLimit('system.LOCATION', 1)),
                        index=env.index, settings=settings)

    assert len(report.run.designed) == 2 and most['disk2'] == 1   # the film and the season never read disk2 together
