'''
Rejected designs in the review queue (designer contract 1.1, design/outstanding.md W4): the designs a designer judged unfit to
publish are kept on the queue entry with their reasons, beside the candidates of a success and the flat candidate of a
decline; nothing picks one automatically; a person's pick of one is an override the entry records; and publishing uses
whichever design was picked.
'''
import json
import os

import pytest

from pipeline.designer.contract import BiquadSpec, DesignCandidate, DesignResponse
from pipeline.designer.registry import register_designer, unregister_designer
from pipeline.orchestrate import Session
from pipeline.review import QueueEntry, apply_reviewed_entry, design_and_queue, publish_reviewed_queue, read_entry, \
    update_entry
from test_pipeline_library_commit import repos  # noqa: F401 (a fixture)
from test_pipeline_orchestrate import _write_mono_wav

SUCCESS, DECLINE = 'test.rejected.queue.success', 'test.rejected.queue.decline'
REASONS = ['introduces a cliff of 53 dB/oct at 17 Hz', 'corrected only down to 24.7 Hz']
META = {'title': 'Heat', 'year': '1995', 'audio_types': ['Atmos']}


def _rejected_design(freq_hz=26.87, mv=16.14):
    return DesignCandidate(filters=[BiquadSpec(type='low_shelf', freq_hz=freq_hz, gain_db=16.14, q=5.797)],
                           confidence=0.95, mv_adjust_db=mv, method='non_parametric',
                           commentary={'strategy': 'flatten'}, rejection_reasons=list(REASONS))


@pytest.fixture(autouse=True)
def _designers():
    accepted = DesignCandidate(filters=[BiquadSpec(type='low_shelf', freq_hz=18.0, gain_db=4.5, q=0.7)],
                               confidence=0.6, mv_adjust_db=4.0, method='fitted')
    register_designer(SUCCESS, lambda request: DesignResponse(
        contract_version='1.1', candidates=[accepted], rejected=[_rejected_design(), _rejected_design(30.0, 2.5)]))
    register_designer(DECLINE, lambda request: DesignResponse(
        contract_version='1.1', decline_reason='no_publishable_candidate', decline_message='every design failed',
        rejected=[_rejected_design()]))
    yield
    unregister_designer(SUCCESS)
    unregister_designer(DECLINE)


def _queued(tmp_path, designer=SUCCESS, entry_id='heat'):
    wav = str(tmp_path / 'mono.wav')
    _write_mono_wav(wav)
    queue_dir = str(tmp_path / 'queue')
    design_and_queue(Session(), entry_id, wav, designer, queue_dir, meta=dict(META))
    return queue_dir, entry_id


def test_rejected_designs_are_kept_on_the_entry_with_their_reasons(tmp_path):
    queue_dir, entry_id = _queued(tmp_path)

    entry = read_entry(queue_dir, entry_id)

    assert len(entry.candidates) == 1 and entry.candidates[0].rejection_reasons is None
    assert [r.rejection_reasons for r in entry.rejected] == [REASONS, REASONS]
    assert entry.rejected[0].commentary == {'strategy': 'flatten'} and entry.rejected[0].confidence == 0.95
    assert entry.rejected[0].filters['filters'][0]['fc'] == 26.87
    assert len(entry.offered) == 3 and entry.chosen is None and not entry.overrides_rejection


def test_a_decline_keeps_its_rejected_designs_as_evidence_beside_its_flat_candidate(tmp_path):
    queue_dir, entry_id = _queued(tmp_path, DECLINE)

    entry = read_entry(queue_dir, entry_id)

    assert entry.declined and len(entry.candidates) == 1 and entry.candidates[0].filters['filters'] == []
    assert [r.rejection_reasons for r in entry.rejected] == [REASONS]


def test_an_entry_written_before_1_1_reads_with_no_rejected_designs(tmp_path):
    queue_dir, entry_id = _queued(tmp_path)
    path = os.path.join(queue_dir, f'{entry_id}.json')
    with open(path) as f:
        data = json.load(f)
    del data['rejected']
    for candidate in data['candidates']:
        del candidate['rejection_reasons']
    with open(path, 'w') as f:
        json.dump(data, f)

    entry = read_entry(queue_dir, entry_id)

    assert entry.rejected == [] and entry.offered == entry.candidates


def test_picking_a_rejected_design_is_recorded_as_an_override(tmp_path):
    queue_dir, entry_id = _queued(tmp_path)

    entry = update_entry(queue_dir, entry_id, status='accepted', chosen_candidate_index=2)

    assert entry.overrides_rejection and entry.chosen is entry.rejected[1]
    assert read_entry(queue_dir, entry_id).overrides_rejection   # it is what the file says, not only in memory
    assert not update_entry(queue_dir, entry_id, chosen_candidate_index=0).overrides_rejection
    with pytest.raises(ValueError, match='out of range for 1 candidate.* and 2 rejected'):
        update_entry(queue_dir, entry_id, chosen_candidate_index=3)


def test_the_picked_rejected_design_is_what_is_applied_and_published(tmp_path, repos):  # noqa: F811
    queue_dir, entry_id = _queued(tmp_path)
    update_entry(queue_dir, entry_id, status='accepted', chosen_candidate_index=1)
    assert apply_reviewed_entry(read_entry(queue_dir, entry_id)).filters[0].freq == 26.87

    (result,) = publish_reviewed_queue(queue_dir, repos[0], xml_dir='xml', push=False, heatmap_spec=None)

    assert not result.get('error'), result
    assert [f['freq'] for f in result['record']['filters']] == [26.87]
    assert result['record']['mv'] == '+16.14'    # the gain is the picked design's, as for any candidate
    assert read_entry(queue_dir, entry_id).status == 'published'
    assert read_entry(queue_dir, entry_id).overrides_rejection


def test_a_declined_titles_rejected_design_can_be_published_by_a_person(tmp_path, repos):  # noqa: F811
    queue_dir, entry_id = _queued(tmp_path, DECLINE)
    update_entry(queue_dir, entry_id, status='accepted', chosen_candidate_index=1)

    (result,) = publish_reviewed_queue(queue_dir, repos[0], xml_dir='xml', push=False, heatmap_spec=None)

    assert [f['freq'] for f in result['record']['filters']] == [26.87]
    assert result['record'].get('note', '') != 'Does not require BEQ'   # it has a filter now


def test_a_queue_entry_refuses_a_pick_past_every_design():
    with pytest.raises(ValueError, match='out of range'):
        QueueEntry(id='x', fs=48000, meta={}, curve={}, status='accepted', chosen_candidate_index=0)
