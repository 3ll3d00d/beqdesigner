'''
pipeline/library/run_records.py -- giving designs made before contract 1.6 the run record their commentary names
(design/redesign-scope.md), over real queue entries (review_entry_fixture).
'''
import json
import os

import pytest

from pipeline.library.run_records import backfill, main, named_records, record_for, summary
from pipeline.review import read_entry
from review_entry_fixture import write_entry

IMAGE = '/work/records/designer-d56656b9cf3c6498.run.json.gz'
EXE = '\\\\ZALAGA-UNRAID\\beq\\work\\records\\designer-0018f164720928a6.run.json.gz'
V7 = '/work/records/designer-0018f164720928a6-89abcdef-v7.run.json.gz'


@pytest.fixture
def dirs(tmp_path):
    queue, work = tmp_path / 'queue', tmp_path / 'work'
    (work / 'records').mkdir(parents=True)
    queue.mkdir()
    return str(queue), str(work)


def _entry(queue, entry_id, record=None, *, decline=False, variant=None, **fields):
    ''' A real entry, then its stored form edited as a 0.2.10 design leaves it: the record named in its commentary. '''
    write_entry(queue, entry_id, decline=decline, rejected=1 if decline else 0)
    path = os.path.join(queue, f'{entry_id}.json')
    with open(path) as f:
        stored = json.load(f)
    if record:
        target = stored['rejected'][0] if decline else stored['candidates'][0]
        target['commentary']['run_record'] = record
    if variant:
        stored['variants'] = [{'candidates': [{'commentary': {'run_record': variant}}]}]
    stored.update(design_version=6, **fields)
    with open(path, 'w') as f:
        json.dump(stored, f)
    return stored


def _record(work, name):
    open(os.path.join(work, 'records', os.path.basename(name.replace('\\', '/'))), 'w').close()


@pytest.mark.parametrize('named, expected', [
    (IMAGE, 'records/designer-d56656b9cf3c6498.run.json.gz'),
    (EXE, 'records/designer-0018f164720928a6.run.json.gz'),   # the exe names it as a Windows share path
    (V7, 'records/designer-0018f164720928a6-89abcdef-v7.run.json.gz'),
])
def test_the_record_named_in_commentary_becomes_a_path_relative_to_the_work_directory(dirs, named, expected):
    queue, work = dirs
    _record(work, named)
    found = record_for(_entry(queue, 'a', named), work)
    assert (found.run_record, found.why) == (expected, 'given its run record')


def test_a_decline_names_its_record_too(dirs):
    queue, work = dirs
    _record(work, IMAGE)
    assert record_for(_entry(queue, 'a', IMAGE, decline=True), work).run_record == \
        'records/designer-d56656b9cf3c6498.run.json.gz'


def test_an_earlier_runs_record_is_not_the_current_designs(dirs):
    queue, work = dirs
    _record(work, EXE)
    stored = _entry(queue, 'a', variant=EXE)
    assert named_records(stored) == [] and record_for(stored, work).why == 'names no run record'


@pytest.mark.parametrize('make, why', [
    (lambda q: _entry(q, 'a', IMAGE, run_record='records/x.run.json.gz'), 'already has a run record'),
    (lambda q: _entry(q, 'a', IMAGE), 'is not in'),
    (lambda q: _entry(q, 'a'), 'names no run record'),
    (lambda q: _entry(q, 'a', IMAGE, decline_message=f'see {EXE}'), 'names 2 run records'),
])
def test_an_entry_it_cannot_be_sure_of_is_left_as_it_is_and_says_why(dirs, make, why):
    queue, work = dirs
    found = record_for(make(queue), work)
    assert found.run_record is None and why in found.why


def test_a_dry_run_writes_nothing_and_apply_writes_only_the_run_record(dirs):
    queue, work = dirs
    _record(work, IMAGE)
    _entry(queue, 'a', IMAGE)
    _entry(queue, 'b')
    before = read_entry(queue, 'a')

    assert [f.run_record for f in backfill(queue, work)] == ['records/designer-d56656b9cf3c6498.run.json.gz', None]
    assert read_entry(queue, 'a').run_record is None

    found = backfill(queue, work, apply=True)
    after = read_entry(queue, 'a')
    assert after.run_record == 'records/designer-d56656b9cf3c6498.run.json.gz'
    assert after == before.__class__(**{**before.__dict__, 'run_record': after.run_record})   # nothing else changed
    assert read_entry(queue, 'b').run_record is None
    assert summary(found) == {'given its run record': 1, 'names no run record': 1}
    assert backfill(queue, work, apply=True)[0].why == 'already has a run record'   # a second pass changes nothing


def test_the_command_says_what_it_would_do(dirs, capsys):
    queue, work = dirs
    _record(work, IMAGE)
    _entry(queue, 'a', IMAGE)
    assert main(['--queue-dir', queue, '--work-dir', work, '--list']) == 0
    out = capsys.readouterr().out
    assert 'a\trecords/designer-d56656b9cf3c6498.run.json.gz\tgiven its run record' in out
    assert 'dry run: nothing written; 1 entries' in out
    assert read_entry(queue, 'a').run_record is None
