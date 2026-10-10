'''
Deciding a title's review without the title page (design/web-app.md §2): `pipeline.library.decide`, which the desktop page and
the pipeline service's review routes share, and the chart curves of `pipeline.library.review_chart`. Each refusal writes nothing.
'''
import os
from types import SimpleNamespace

import pytest

from pipeline.library.decide import DecisionRefused, decide, offered_digest
from pipeline.library.review_chart import chart_curves
from pipeline.review import read_entry, update_entry
from review_entry_fixture import write_entry


def _row(needs='review', extract_state='done', design_state='done', detail=''):
    ''' The parts of a TitleRow `decision_blocked` reads. '''
    return SimpleNamespace(needs=needs, extract_state=extract_state, design_state=design_state, detail=detail)


def _decide(queue_dir, decision='accept', picked=0, entry_id='a', **kwargs):
    seen = kwargs.pop('seen_digest', None) or offered_digest(read_entry(queue_dir, entry_id))
    return decide(queue_dir, entry_id, decision, seen_digest=seen, picked=picked, **kwargs)


def _unchanged(queue_dir, entry_id='a'):
    ''' The file as written by the fixture, by modification time and content. '''
    path = os.path.join(queue_dir, f'{entry_id}.json')
    return os.path.getmtime(path), open(path, encoding='utf-8').read()


@pytest.mark.parametrize('status', ['pending', 'skipped'])
def test_accept_writes_the_chosen_design(tmp_path, status):
    q = str(tmp_path)
    write_entry(q, 'a', status=status)

    written = _decide(q, picked=1, row=_row())

    assert (written.status, written.chosen_candidate_index) == ('accepted', 1)
    assert read_entry(q, 'a').status == 'accepted'


def test_reject_writes_rejected_and_needs_no_pick(tmp_path):
    q = str(tmp_path)
    write_entry(q, 'a')
    assert _decide(q, 'reject', picked=None).status == 'rejected'


@pytest.mark.parametrize('status', ['accepted', 'rejected', 'published'])
@pytest.mark.parametrize('decision', ['accept', 'reject'])
def test_a_title_already_decided_is_not_decided_again(tmp_path, status, decision):
    q = str(tmp_path)
    write_entry(q, 'a', status=status)
    before = _unchanged(q)

    with pytest.raises(DecisionRefused) as refused:
        _decide(q, decision)

    assert refused.value.kind == 'changed'
    assert refused.value.reason == f'Not changed: this title is {status} now.'
    assert _unchanged(q) == before


def test_a_redesign_since_the_title_was_seen_is_refused(tmp_path):
    q = str(tmp_path)
    seen = offered_digest(write_entry(q, 'a'))
    write_entry(q, 'a', reverse=True)      # the same number of candidates, not the same ones

    with pytest.raises(DecisionRefused) as refused:
        _decide(q, seen_digest=seen)

    assert refused.value.kind == 'changed' and 'changed while it was open' in refused.value.reason
    assert read_entry(q, 'a').status == 'pending'


def test_the_digest_is_of_the_designs_only_and_survives_the_file(tmp_path):
    q = str(tmp_path)
    written = write_entry(q, 'a')
    digest = offered_digest(written)

    assert offered_digest(read_entry(q, 'a')) == digest   # JSON turns tuples into lists: the digest does not care
    update_entry(q, 'a', reviewer_note='looked at it', meta={'title': 'other'})
    assert offered_digest(read_entry(q, 'a')) == digest
    assert offered_digest(write_entry(q, 'a', reverse=True)) != digest
    assert offered_digest(write_entry(q, 'a', rejected=1)) != digest


@pytest.mark.parametrize('decision, picked, words', [
    ('accept', None, 'no design was chosen'),
    ('accept', 5, 'there is no design 6'),
    ('accept', -1, 'there is no design 0'),
    ('approve', 0, 'unknown decision'),
])
def test_an_invalid_request_is_refused(tmp_path, decision, picked, words):
    q = str(tmp_path)
    write_entry(q, 'a')
    with pytest.raises(DecisionRefused) as refused:
        _decide(q, decision, picked=picked)
    assert refused.value.kind == 'invalid' and words in refused.value.reason
    assert read_entry(q, 'a').status == 'pending'


@pytest.mark.parametrize('decision', ['accept', 'reject'])
def test_nothing_is_decided_on_a_title_a_run_is_working_on_or_whose_last_stage_failed(tmp_path, decision):
    q = str(tmp_path)
    write_entry(q, 'a')
    with pytest.raises(DecisionRefused) as running:
        _decide(q, decision, row=_row(), running=True)
    with pytest.raises(DecisionRefused) as failed:
        _decide(q, decision, row=_row(needs='attention', design_state='failed', detail='designer said no'))

    assert running.value.kind == failed.value.kind == 'blocked'
    assert 'A run is working on this title now' in running.value.reason
    assert 'the last design failed (designer said no)' in failed.value.reason
    assert read_entry(q, 'a').status == 'pending'


def test_an_out_of_date_design_is_not_accepted_but_can_be_rejected_and_the_hint_says_where_to_run_it(tmp_path):
    q = str(tmp_path)
    write_entry(q, 'a')
    with pytest.raises(DecisionRefused) as refused:
        _decide(q, row=_row(needs='design', detail='settings changed'), run_hint='Run it from Jobs.')

    assert refused.value.kind == 'blocked'
    assert refused.value.reason == 'Not offered: this title needs design first (settings changed). Run it from Jobs.'
    assert _decide(q, 'reject', row=_row(needs='design')).status == 'rejected'


def test_a_title_sent_back_on_the_page_is_not_accepted(tmp_path):
    q = str(tmp_path)
    write_entry(q, 'a')
    with pytest.raises(DecisionRefused) as refused:
        _decide(q, revised='design')
    assert refused.value.kind == 'blocked' and 'sent back for redesign' in refused.value.reason


def test_incomplete_metadata_holds_accept_back_but_not_reject(tmp_path):
    q = str(tmp_path)
    write_entry(q, 'a', meta={'title': 'a'})     # what a library run leaves: no year, no audio types

    with pytest.raises(DecisionRefused) as refused:
        _decide(q)

    assert refused.value.kind == 'metadata' and refused.value.reason.startswith('Not accepted: the metadata is not complete')
    assert _decide(q, 'reject', picked=None).status == 'rejected'


def test_metadata_defaults_complete_what_the_entry_leaves_out(tmp_path):
    q = str(tmp_path)
    write_entry(q, 'a', meta={'title': 'a', 'year': '2001'})
    assert _decide(q, meta_defaults={'audio_types': ['DTS-HD MA 5.1']}).status == 'accepted'


def test_a_design_the_designer_rejected_is_accepted_only_with_the_override(tmp_path):
    q = str(tmp_path)
    write_entry(q, 'a', count=2, rejected=1)

    with pytest.raises(DecisionRefused) as refused:
        _decide(q, picked=2)
    assert refused.value.kind == 'override'
    assert read_entry(q, 'a').status == 'pending'

    written = _decide(q, picked=2, override_rejection=True)
    assert (written.status, written.chosen_candidate_index, written.overrides_rejection) == ('accepted', 2, True)


def test_a_title_with_no_entry_is_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        decide(str(tmp_path), 'nothing', 'reject', seen_digest='x')


# --- the chart ---------------------------------------------------------------------------------------------------------

def test_chart_curves_are_the_measured_curves_then_each_after_the_chosen_design(tmp_path):
    q = str(tmp_path)
    entry = write_entry(q, 'a')

    curves = chart_curves(entry, 0)

    assert [(c.kind, c.filtered) for c in curves] == [('average', False), ('peak', False), ('average', True), ('peak', True)]
    assert [c.data.name for c in curves] == ['Average audio track (all channels mixed)',
                                             'Peak audio track (all channels mixed)',
                                             'Filtered average audio track (all channels mixed)',
                                             'Filtered peak audio track (all channels mixed)']
    assert (curves[0].data.y == 0).all() and curves[2].data.y.any()   # a 4.5 dB low shelf lifts the bottom
    assert [(c.kind, c.filtered) for c in chart_curves(entry, None)] == [('average', False), ('peak', False)]
    assert [(c.kind, c.filtered) for c in chart_curves(entry, 9)] == [('average', False), ('peak', False)]
    assert chart_curves(None, 0) == []
