'''
Phase 1 of design/candidate-review-plan.md: QueueEntry persistence and the
batch_design() driver, exercised without any GUI -- the review dialog
(Phase 3) is just a reader/writer of the same queue.
'''
import wave

import numpy as np
import pytest

from pipeline.designer.contract import BiquadSpec, DesignCandidate, DesignResponse
from pipeline.designer.registry import register_designer, unregister_designer
from pipeline.review import CandidateSummary, QueueEntry, batch_design, read_entry, read_queue, update_entry, \
    write_queue_entry

DESIGNER_NAME = 'test.review'
DECLINE_DESIGNER_NAME = 'test.review.decline'


def _multi_candidate_designer(request):
    return DesignResponse(contract_version='1.0', candidates=[
        DesignCandidate(filters=[BiquadSpec(type='low_shelf', freq_hz=18.0, gain_db=4.5, q=0.7)],
                        confidence=0.9, mv_adjust_db=4.0, gain_reduction_db=-1.5, method='fitted',
                        commentary={'note': 'primary'}),
        DesignCandidate(filters=[BiquadSpec(type='peaking_eq', freq_hz=40.0, gain_db=-3.0, q=2.0)],
                        confidence=0.4, mv_adjust_db=1.0, gain_reduction_db=0.0, method='fitted',
                        commentary={'note': 'alternative'}),
    ])


def _decline_designer(request):
    return DesignResponse(contract_version='1.0', decline_reason='no_rolloff_detected',
                          decline_message='nothing to correct')


@pytest.fixture(autouse=True)
def _designers():
    register_designer(DESIGNER_NAME, _multi_candidate_designer)
    register_designer(DECLINE_DESIGNER_NAME, _decline_designer)
    yield
    unregister_designer(DESIGNER_NAME)
    unregister_designer(DECLINE_DESIGNER_NAME)


def _write_synthetic_wav(path, fs=48000, duration_s=0.25, channel_values=(1000, 2000, 3000, 4000, 5000, 6000)):
    n_frames = int(fs * duration_s)
    frame = np.array(channel_values, dtype=np.int16)
    data = np.tile(frame, (n_frames, 1)).astype('<i2').tobytes()
    with wave.open(path, 'wb') as w:
        w.setnchannels(len(channel_values))
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(data)


# --- QueueEntry construction/validation -----------------------------------

def _candidate(**overrides):
    defaults = dict(filters={'_type': 'CompleteFilter', 'fs': 1000, 'description': 'designer', 'filters': []},
                    confidence=0.9, method='fitted', mv_adjust_db=4.0)
    defaults.update(overrides)
    return CandidateSummary(**defaults)


def test_accepted_entry_requires_chosen_candidate_index():
    with pytest.raises(ValueError, match='chosen_candidate_index'):
        QueueEntry(id='t1', fs=1000, meta={}, curve={}, candidates=[_candidate()], status='accepted')


def test_accepted_entry_index_out_of_range_rejected():
    with pytest.raises(ValueError, match='out of range'):
        QueueEntry(id='t1', fs=1000, meta={}, curve={}, candidates=[_candidate()], status='accepted',
                  chosen_candidate_index=1)


def test_invalid_status_rejected():
    with pytest.raises(ValueError, match='status'):
        QueueEntry(id='t1', fs=1000, meta={}, curve={}, status='not-a-status')


# --- persistence -----------------------------------------------------------

def test_queue_entry_round_trips_multiple_candidates_and_commentary(tmp_path):
    entry = QueueEntry(
        id='ready-player-one', fs=48000, meta={'beq_title': 'Ready Player One'},
        curve={'_type': 'MagnitudeData', 'name': 'avg', 'description': '', 'x': [1.0, 2.0], 'y': [0.1, 0.2],
              'colour': None, 'linestyle': '-'},
        candidates=[
            _candidate(confidence=0.9, commentary={'alignment': 'LR4', 'knee_hz': '25.0'}),
            _candidate(confidence=0.4, commentary={'note': 'alternative'}),
        ],
    )
    write_queue_entry(str(tmp_path), entry)

    read_back = read_entry(str(tmp_path), 'ready-player-one')

    assert read_back == entry
    assert [c.confidence for c in read_back.candidates] == [0.9, 0.4]
    assert read_back.candidates[0].commentary == {'alignment': 'LR4', 'knee_hz': '25.0'}


def test_read_entry_missing_id_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_entry(str(tmp_path), 'does-not-exist')


def test_read_queue_sorts_pending_first(tmp_path):
    write_queue_entry(str(tmp_path), QueueEntry(id='b', fs=1000, meta={}, curve={}, status='pending'))
    write_queue_entry(str(tmp_path), QueueEntry(id='a', fs=1000, meta={}, curve={},
                                                candidates=[_candidate()], status='accepted',
                                                chosen_candidate_index=0))
    write_queue_entry(str(tmp_path), QueueEntry(id='c', fs=1000, meta={}, curve={}, status='pending'))

    ids = [e.id for e in read_queue(str(tmp_path))]

    assert ids == ['b', 'c', 'a']  # both pending entries (alphabetical), then the accepted one


def test_update_entry_rewrites_the_file(tmp_path):
    write_queue_entry(str(tmp_path), QueueEntry(id='t1', fs=1000, meta={}, curve={},
                                                candidates=[_candidate(), _candidate()], status='pending'))

    updated = update_entry(str(tmp_path), 't1', status='accepted', chosen_candidate_index=1)

    assert updated.status == 'accepted'
    assert updated.chosen_candidate_index == 1
    assert read_entry(str(tmp_path), 't1') == updated


def test_update_entry_missing_id_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        update_entry(str(tmp_path), 'does-not-exist', status='skipped')


# --- batch_design ------------------------------------------------------------

def test_batch_design_writes_one_pending_entry_per_title(tmp_path):
    source_wav = str(tmp_path / 'source.wav')
    _write_synthetic_wav(source_wav)
    queue_dir = str(tmp_path / 'queue')
    work_dir = str(tmp_path / 'work')

    written = batch_design(
        [('title-one', source_wav, {'beq_title': 'Title One'}), ('title-two', source_wav, None)],
        DESIGNER_NAME, queue_dir, work_dir)

    assert written == ['title-one', 'title-two']
    entries = {e.id: e for e in read_queue(queue_dir)}
    assert set(entries) == {'title-one', 'title-two'}
    for entry in entries.values():
        assert entry.status == 'pending'
        assert len(entry.candidates) == 2
        assert entry.candidates[0].confidence == 0.9  # top-ranked first
        assert entry.candidates[0].gain_reduction_db == -1.5
        assert entry.candidates[1].confidence == 0.4
        assert entry.candidates[1].gain_reduction_db == 0.0
        assert entry.curve['_type'] == 'MagnitudeData'
    assert entries['title-one'].meta == {'beq_title': 'Title One'}
    assert entries['title-two'].meta == {}


def test_batch_design_declined_title_carries_the_reason(tmp_path):
    source_wav = str(tmp_path / 'source.wav')
    _write_synthetic_wav(source_wav)
    queue_dir = str(tmp_path / 'queue')
    work_dir = str(tmp_path / 'work')

    batch_design([('declined-title', source_wav, None)], DECLINE_DESIGNER_NAME, queue_dir, work_dir)

    entry = read_entry(queue_dir, 'declined-title')
    assert entry.status == 'pending'
    assert entry.candidates == []
    assert entry.decline_reason == 'no_rolloff_detected'
    assert entry.decline_message == 'nothing to correct'


def test_pipeline_review_module_has_no_qtpy_import():
    import ast
    import pathlib
    source = (pathlib.Path(__file__).parent / '..' / '..' / 'main' / 'python' / 'pipeline' / 'review.py').resolve()
    tree = ast.parse(source.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(n.name.startswith('qtpy') for n in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith('qtpy')
