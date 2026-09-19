'''
Bulk accept (design.md §12.9): accept the designer's top pick for confident titles waiting for review, leaving out --
and reporting -- incomplete metadata, a decline and an edited project, with a reviewer note on each accepted title.
'''
import os

from pipeline.library.bulk import DEFAULT_ACCEPT_THRESHOLD, accept_note, accept_top_pick, plan_accept
from pipeline.library.selection import Selection
from pipeline.orchestrate import Session
from pipeline.config import AnalysisConfig
from pipeline.publish.project import write_mono_project
from pipeline.review import read_entry, update_entry
from test_pipeline_library_index import FakeSource, _entry, _extracted, _item, _needs, _profile, _scan, env  # noqa: F401
from test_pipeline_publish_project import _HUMAN_FILTER, _PIPELINE_FILTER, _hand_edit_filter, _write_mono_wav


def _title(env, name, confidence=0.95, **entry_fields):
    item = _item(name)
    _extracted(env, item)
    _entry(env, item, confidence=confidence, **entry_fields)
    return item


def _accept(env, selection=Selection(), threshold=DEFAULT_ACCEPT_THRESHOLD, **kwargs):
    return accept_top_pick(env.index, selection, threshold, queue_dir=env.queue, work_dir=env.work, **kwargs)


def _plan(env, selection=Selection(), threshold=DEFAULT_ACCEPT_THRESHOLD, **kwargs):
    return plan_accept(env.index, selection, threshold, queue_dir=env.queue, work_dir=env.work, **kwargs)


def _project(env, item, edited):
    ''' The mono .beq project a design writes, hand-edited if `edited`. '''
    wav = os.path.join(env.work, item.id, 'mono.wav')
    _write_mono_wav(wav)
    path = os.path.join(env.work, item.id, f'{item.id}.mono.beq')
    write_mono_project(Session(AnalysisConfig()), wav, _PIPELINE_FILTER, path)
    if edited:
        _hand_edit_filter(path, _HUMAN_FILTER)
    return path


def test_the_default_threshold_and_note_are_the_designs():
    assert DEFAULT_ACCEPT_THRESHOLD == 0.90
    assert accept_note(0.9) == 'bulk accepted, confidence >= 0.90' and accept_note(0.86) == 'bulk accepted, confidence >= 0.86'


def test_a_confident_title_is_accepted_with_its_top_pick_and_a_reviewer_note(env):
    _scan(env, _title(env, 'a', confidence=0.95))

    report = _accept(env)

    entry = read_entry(env.queue, 'fs-a')
    assert report.accepted == ['fs-a'] and report.excluded == []
    assert (entry.status, entry.chosen_candidate_index) == ('accepted', 0)
    assert entry.reviewer_note == 'bulk accepted, confidence >= 0.90' == report.note


def test_the_note_is_added_to_a_reviewers_own(env):
    _scan(env, _title(env, 'a', reviewer_note='looks right to me'))

    _accept(env)

    assert read_entry(env.queue, 'fs-a').reviewer_note == 'looks right to me\nbulk accepted, confidence >= 0.90'


def test_the_threshold_is_inclusive_and_a_less_confident_title_is_left_for_a_person(env):
    _scan(env, _title(env, 'exact', confidence=0.90), _title(env, 'below', confidence=0.8999), _title(env, 'high', 0.99))

    report = _accept(env)

    assert sorted(report.accepted) == ['fs-exact', 'fs-high']
    assert report.below_threshold == 1 and report.excluded == []  # counted, not an exclusion
    assert read_entry(env.queue, 'fs-below').status == 'pending'


def test_a_lower_threshold_takes_more(env):
    _scan(env, _title(env, 'a', confidence=0.7))

    assert _accept(env, threshold=0.75).accepted == []
    assert _accept(env, threshold=0.7).accepted == ['fs-a']


def test_incomplete_metadata_is_left_out_and_reported(env):
    _scan(env, _title(env, 'a'), _title(env, 'b'))
    update_entry(env.queue, 'fs-a', meta={'title': 'Film a', 'year': '', 'audio_types': []})
    _scan(env, _item('a'), _item('b'))

    report = _accept(env)

    assert report.accepted == ['fs-b']
    (excluded,) = report.excluded
    assert excluded.id == 'fs-a' and excluded.title == 'Film a'
    assert 'metadata incomplete' in excluded.reason and 'year is required' in excluded.reason
    assert read_entry(env.queue, 'fs-a').status == 'pending'


def test_metadata_the_publish_defaults_supply_is_not_incomplete(env):
    _scan(env, _title(env, 'a'))
    update_entry(env.queue, 'fs-a', meta={'title': 'Film a', 'year': '2001'})  # no audio types of its own
    _scan(env, _item('a'))

    assert _accept(env).excluded and _accept(env).accepted == []
    assert _accept(env, meta_defaults={'audio_types': ['Atmos']}).accepted == ['fs-a']


def test_a_designer_decline_is_left_out_and_reported(env):
    item = _item('a')
    _extracted(env, item)
    _entry(env, item, candidates=0, decline_reason='no_signal', decline_message='nothing below 20 Hz')
    _scan(env, item)

    report = _accept(env)

    assert report.accepted == []
    assert report.excluded[0].reason == 'designer declined: nothing below 20 Hz'


def test_an_edited_project_is_left_out_and_reported(env):
    edited, pure = _title(env, 'edited'), _title(env, 'pure')
    _project(env, edited, edited=True)
    _project(env, pure, edited=False)
    _scan(env, edited, pure)

    report = _accept(env)

    assert report.accepted == ['fs-pure']
    assert report.excluded[0].id == 'fs-edited'
    assert 'mono project was edited' in report.excluded[0].reason
    assert read_entry(env.queue, 'fs-edited').status == 'pending'


def test_a_project_that_cannot_be_read_is_left_out_not_accepted_blind(env):
    item = _title(env, 'a')
    path = _project(env, item, edited=False)
    with open(path, 'wb') as f:
        f.write(b'not a gzip file')
    _scan(env, item)

    report = _accept(env)

    assert report.accepted == [] and 'cannot be read' in report.excluded[0].reason


def test_without_a_work_directory_projects_are_not_looked_for(env):
    item = _title(env, 'a')
    _project(env, item, edited=True)
    _scan(env, item)

    report = accept_top_pick(env.index, Selection(), queue_dir=env.queue)

    assert report.accepted == ['fs-a']


def test_only_titles_waiting_for_review_are_considered(env):
    todo, extracted = _item('todo'), _item('extracted')
    _extracted(env, extracted)
    _scan(env, todo, extracted, _title(env, 'ready'))

    report = _accept(env)

    assert report.accepted == ['fs-ready'] and report.not_for_review == 2


def test_the_selection_narrows_what_is_accepted(env):
    a = _title(env, 'alpha')
    _title(env, 'beta')
    _scan(env, sources={'films': FakeSource([a]), 'disk': FakeSource([_item('beta', source_path='/disk/b.mkv')])})

    assert _plan(env, Selection(source='disk')).eligible == ['fs-beta']
    assert _plan(env, Selection(match='alp')).eligible == ['fs-alpha']
    assert _plan(env, Selection(needs=('extract',))).eligible == []  # not review: nothing to accept
    assert _plan(env, Selection(ids=('fs-alpha',))).eligible == ['fs-alpha']

    assert _accept(env, Selection(source='disk')).accepted == ['fs-beta']
    assert read_entry(env.queue, 'fs-alpha').status == 'pending'  # only the selection was touched


def test_plan_accept_changes_nothing_and_says_what_accept_would_do(env):
    _scan(env, _title(env, 'a'), _title(env, 'b', confidence=0.5))
    before = read_entry(env.queue, 'fs-a')

    plan = _plan(env)

    assert plan.eligible == ['fs-a'] and plan.below_threshold == 1
    assert read_entry(env.queue, 'fs-a') == before


def test_a_stale_index_cannot_make_it_accept_what_is_no_longer_pending(env):
    _scan(env, _title(env, 'a'))
    update_entry(env.queue, 'fs-a', status='rejected')  # a person did this after the scan

    report = _accept(env)

    assert report.accepted == [] and report.excluded[0].reason == 'already rejected'
    assert read_entry(env.queue, 'fs-a').status == 'rejected'


def test_a_title_with_no_queue_entry_is_reported(env):
    _scan(env, _title(env, 'a'))
    os.remove(os.path.join(env.queue, 'fs-a.json'))

    report = _accept(env)

    assert report.accepted == [] and report.excluded[0].reason == 'it has no queue entry'


def test_after_a_refresh_an_accepted_title_needs_publish(env):
    item = _title(env, 'a')
    profile = _profile(env)
    _scan(env, item)
    assert _needs(env, 'fs-a')[0] == 'review'

    _accept(env)
    env.index.refresh(profile, env.settings)

    assert _needs(env, 'fs-a') == ('publish', 'accepted, not written to the repository')
