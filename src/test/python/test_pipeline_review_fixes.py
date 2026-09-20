'''
Review fixes for the publish path (chunks 21/22/25): one bad entry never aborts a batch (unbuildable metadata, any
other failure), the queue file is written atomically, a report style change makes a title out of date, bulk accept
does not overwrite a concurrent edit, and one bad title cannot abort run_stages. Real temp git repos.
'''
import json
import os
import sqlite3

import pytest

from pipeline.library.bulk import accept_top_pick
from pipeline.library.selection import Selection
from pipeline.library.stages import PublishSettings, run_stages
from pipeline.library.status import ScanSettings
from pipeline.orchestrate import Session
from pipeline.publish.report import ReportSpec
from pipeline.review import (current_publish_digest, publish_reviewed_queue, read_entry, read_queue, split_publish_results,
                             update_entry, write_queue_entry)
from test_pipeline_library_commit import IMAGES_NAME, OWNER, _publish, _queue_entry, repos  # noqa: F401 (a fixture)
from test_pipeline_library_index import DESIGNER, _entry, _extracted, _item, _needs, _scan, env  # noqa: F401
from test_pipeline_library_stages import (_accepted, _go, _publish_settings, _run_config, work)  # noqa: F401


# --- 9: metadata that cannot even be built ------------------------------------------------------------------------

_BAD_META = {
    'missing title': ({'year': '2018', 'audio_types': ['Atmos']}, 'title is required'),
    'null title': ({'title': None, 'year': '2018', 'audio_types': ['Atmos']}, 'title is required'),
    'null year': ({'title': 'X', 'year': None, 'audio_types': ['Atmos']}, 'year is required'),
    'unknown key': ({'title': 'X', 'year': '2018', 'audio_types': ['Atmos'], 'bogus': 1}, "unknown field 'bogus'"),
    'null list': ({'title': 'X', 'year': '2018', 'audio_types': None}, 'audio_types must not be null'),
}


@pytest.mark.parametrize('meta, problem', _BAD_META.values(), ids=list(_BAD_META))
def test_unbuildable_metadata_is_refused_per_entry_and_the_next_entry_still_publishes(tmp_path, repos, meta, problem):
    xml, _, _, _ = repos
    queue_dir = str(tmp_path / 'queue')
    _queue_entry(queue_dir, 'a', 'Alien')
    _queue_entry(queue_dir, 'b', 'Bad')
    _queue_entry(queue_dir, 'c', 'Cube')
    update_entry(queue_dir, 'b', meta=meta)

    results = publish_reviewed_queue(queue_dir, xml, xml_dir='xml', push=False)

    published, refused = split_publish_results(results)
    assert [r['id'] for r in published] == ['a', 'c']
    assert [(r['id'], r['error']) for r in refused] == [('b', 'invalid_metadata')]
    assert problem in refused[0]['problems']
    assert read_entry(queue_dir, 'c').status == 'published'


def test_the_republish_selection_does_not_raise_on_a_published_entry_whose_metadata_broke(tmp_path, repos):
    xml, _, _, _ = repos
    queue_dir = str(tmp_path / 'queue')
    _queue_entry(queue_dir, 'a', 'Alien')
    _queue_entry(queue_dir, 'b', 'Bad')
    _queue_entry(queue_dir, 'c', 'Cube')
    publish_reviewed_queue(queue_dir, xml, xml_dir='xml', push=False)
    update_entry(queue_dir, 'a', meta={'title': 'Alien 2', 'year': '2018', 'audio_types': ['Atmos']})  # out of date
    update_entry(queue_dir, 'b', meta={'title': None, 'year': '2018', 'bogus': True})  # cannot even be built
    update_entry(queue_dir, 'c', meta={'title': 'Cube 2', 'year': '2018', 'audio_types': ['Atmos']})

    results = publish_reviewed_queue(queue_dir, xml, xml_dir='xml', push=False, republish=True)

    assert [(r['id'], r.get('error')) for r in results] == [('a', None), ('b', 'invalid_metadata'), ('c', None)]


def test_any_other_failure_of_one_entry_is_that_entrys_result_not_the_batchs(tmp_path, repos, monkeypatch):
    xml, _, _, _ = repos
    queue_dir = str(tmp_path / 'queue')
    for entry_id, title in (('a', 'Alien'), ('b', 'Bad'), ('c', 'Cube')):
        _queue_entry(queue_dir, entry_id, title)
    real = Session.publish

    def publish(self, filters, meta, *args, **kwargs):
        if meta.title == 'Bad':
            raise FileNotFoundError('poster.jpg')
        return real(self, filters, meta, *args, **kwargs)

    monkeypatch.setattr(Session, 'publish', publish)

    results = publish_reviewed_queue(queue_dir, xml, xml_dir='xml', push=False)

    assert [r['id'] for r in results] == ['a', 'b', 'c']
    assert results[1]['error'] == 'publish_failed' and 'poster.jpg' in results[1]['message']
    assert read_entry(queue_dir, 'b').status == 'accepted' and read_entry(queue_dir, 'c').status == 'published'


# --- 10: atomic queue writes ----------------------------------------------------------------------------------------

def test_the_queue_file_is_replaced_not_truncated_and_leaves_no_temporary_file(tmp_path, monkeypatch):
    queue_dir = str(tmp_path / 'queue')
    _queue_entry(queue_dir, 'a', 'Alien')
    entry = read_entry(queue_dir, 'a')

    def broken_dump(obj, f, *args, **kwargs):
        f.write('{"half": ')
        raise RuntimeError('disk full')

    monkeypatch.setattr('pipeline.review.json.dump', broken_dump)
    with pytest.raises(RuntimeError):
        write_queue_entry(queue_dir, entry)
    monkeypatch.undo()

    assert read_entry(queue_dir, 'a').meta['title'] == 'Alien'   # the old entry is intact
    assert os.listdir(queue_dir) == ['a.json']                     # and nothing was left behind
    assert [e.id for e in read_queue(queue_dir)] == ['a']


# --- 10: bulk accept re-checks what it planned ------------------------------------------------------------------------

def test_bulk_accept_skips_an_entry_that_changed_between_the_plan_and_the_write(env, monkeypatch):
    from pipeline.library import bulk
    a, b = _item('a'), _item('b')
    for item in (a, b):
        _extracted(env, item)
        _entry(env, item, confidence=0.95)
    _scan(env, a, b)
    real = bulk.plan_accept

    def plan_then_a_person_edits(*args, **kwargs):
        plan = real(*args, **kwargs)
        update_entry(env.queue, 'fs-a', status='skipped')                  # a GUI edit, after the plan read it
        update_entry(env.queue, 'fs-b', meta={'title': '', 'year': ''})    # ... and one that made it ineligible
        return plan

    monkeypatch.setattr(bulk, 'plan_accept', plan_then_a_person_edits)

    report = accept_top_pick(env.index, Selection(), queue_dir=env.queue, work_dir=env.work)

    assert report.accepted == []
    assert sorted(e.id for e in report.excluded) == ['fs-a', 'fs-b']
    assert all('changed while accepting' in e.reason for e in report.excluded)
    assert read_entry(env.queue, 'fs-a').status == 'skipped'   # the person's edit stands


def test_bulk_accept_of_an_unchanged_entry_still_works_and_the_plan_json_has_no_snapshot(env):
    from dataclasses import asdict
    from pipeline.library.bulk import plan_accept
    a = _item('a')
    _extracted(env, a)
    _entry(env, a, confidence=0.95)
    _scan(env, a)

    assert 'seen' not in json.dumps(asdict(plan_accept(env.index, Selection(), queue_dir=env.queue)))
    assert accept_top_pick(env.index, Selection(), queue_dir=env.queue, work_dir=env.work).accepted == ['fs-a']


# --- 7: the report style is in the digest --------------------------------------------------------------------------

def test_a_changed_report_style_changes_the_digest_only_when_an_image_is_published(tmp_path):
    queue_dir = str(tmp_path / 'queue')
    _queue_entry(queue_dir, 'a', 'Alien')
    entry = read_entry(queue_dir, 'a')
    wide = ReportSpec(width_px=2000)

    default_image = current_publish_digest(entry, has_image=True)
    assert current_publish_digest(entry, has_image=True, report_spec=wide) != default_image
    assert current_publish_digest(entry, has_image=True, report_spec=ReportSpec()) == default_image  # unchanged default
    assert current_publish_digest(entry, has_image=False, report_spec=wide) == current_publish_digest(entry)


def test_republish_takes_a_title_whose_report_style_changed(tmp_path, repos):
    xml, _, images, _ = repos
    queue_dir, _ = _publish(tmp_path, repos, ('a', 'Alien'))
    kwargs = dict(images_repo=images, xml_dir='xml', image_dir='img', image_owner=OWNER, image_repo_name=IMAGES_NAME,
                  push=False)

    assert publish_reviewed_queue(queue_dir, xml, republish=True, **kwargs) == []
    results = publish_reviewed_queue(queue_dir, xml, republish=True, report_spec=ReportSpec(width_px=1200), **kwargs)

    assert [(r['id'], r['republished']) for r in results] == [('a', True)]


# --- 11: one bad title cannot abort run_stages ---------------------------------------------------------------------

def test_a_title_whose_listing_cannot_be_rebuilt_fails_alone(env, work):
    _scan(env, _item('a'), _item('b'), _item('c'))
    connection = sqlite3.connect(os.path.join(env.work, 'library-index.sqlite'))
    connection.execute("UPDATE titles SET items = 'not json' WHERE id = 'fs-b'")
    connection.commit()
    connection.close()

    report = _go(env, Selection(), 'design')

    assert sorted(report.run.designed) == ['fs-a', 'fs-c']
    assert [f[0] for f in report.run.failed] == ['fs-b'] and 'JSONDecodeError' in report.run.failed[0][1]
    assert report.failed


def test_a_publish_exception_keeps_the_results_already_recorded(env, work, repos, monkeypatch):
    (a, b, c), settings = _accepted(env, repos, 'a', 'b', 'c')
    _scan(env, a, b, c, settings=settings)
    real = Session.publish

    def publish(self, filters, meta, *args, **kwargs):
        if meta.title == 'Film b':
            raise OSError('poster file is gone')
        return real(self, filters, meta, *args, **kwargs)

    monkeypatch.setattr(Session, 'publish', publish)

    report = _go(env, Selection(needs=('publish',)), 'publish', settings=settings, publish=_publish_settings(repos))

    assert sorted(r['id'] for r in report.published) == ['fs-a', 'fs-c']
    assert [(e['id'], e['error']) for e in report.publish_errors] == [('fs-b', 'publish_failed')]
    assert report.failed


def test_an_unexpected_failure_before_any_entry_is_reported_not_raised(env, work, repos, monkeypatch):
    (a,), settings = _accepted(env, repos, 'a')
    _scan(env, a, settings=settings)
    monkeypatch.setattr('pipeline.library.stages.publish_library', lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError('queue unreadable')))

    report = _go(env, Selection(needs=('publish',)), 'publish', settings=settings, publish=_publish_settings(repos))

    assert report.publish_errors[0]['error'] == 'publish_failed' and 'queue unreadable' in report.publish_errors[0]['message']


# --- 13: nits -------------------------------------------------------------------------------------------------------

def test_not_run_is_sorted(env, work):
    items = [_item(n) for n in ('c', 'a', 'b')]
    _scan(env, *items)

    report = _go(env, Selection(), 'design', should_cancel=lambda: True)

    assert report.cancelled and report.not_run == sorted(report.not_run) == ['fs-a', 'fs-b', 'fs-c']


def test_an_accepted_title_held_at_review_for_incomplete_metadata_is_skipped_with_that_reason(env, work):
    item = _item('a')
    _extracted(env, item)
    _entry(env, item, status='accepted')
    update_entry(env.queue, 'fs-a', meta={'title': '', 'year': '', 'audio_types': []})
    _scan(env, item)

    report = _go(env, Selection(), 'design')

    (skipped,) = report.skipped
    assert 'title is required' in skipped.reason, skipped.reason
