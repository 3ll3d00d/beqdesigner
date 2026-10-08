'''TODO D4 (release part): each queue entry records which designer, and which build of it, designed it.'''
import numpy as np
import pytest
import soundfile as sf

from pipeline.config import AnalysisConfig
from pipeline.designer.contract import BiquadSpec, DesignCandidate, DesignResponse
from pipeline.designer.registry import register_designer
from pipeline.orchestrate import Session
from pipeline.review import CandidateSummary, QueueEntry, design_and_queue, designer_build, read_entry


def _candidate(**commentary):
    return CandidateSummary(filters=[], confidence=0.9, method='m', mv_adjust_db=0.0, commentary=commentary)


@pytest.mark.parametrize('entry, build', [
    (QueueEntry(id='a', fs=1000, meta={}, curve={}, candidates=[_candidate(beqforge_revision='v0.2.0+src:abc')]),
     'v0.2.0+src:abc'),
    (QueueEntry(id='a', fs=1000, meta={}, curve={}, decline_reason='no_rolloff_detected',
                decline_message='within ripple | found: plateau [beqforge v0.2.0+src:abc]'), 'beqforge v0.2.0+src:abc'),
    (QueueEntry(id='a', fs=1000, meta={}, curve={}, candidates=[_candidate(revision='r7')]), 'r7'),
    (QueueEntry(id='a', fs=1000, meta={}, curve={}, candidates=[_candidate(found='plateau')]), None),
])
def test_the_build_is_what_the_designer_said_it_is(entry, build):
    assert designer_build(entry) == build


def test_a_new_design_records_its_designer_and_build(tmp_path):
    register_designer('stamped', lambda request: DesignResponse(
        '1.0', candidates=[DesignCandidate(filters=[BiquadSpec('peaking_eq', 20.0, 3.0, 1.0)], confidence=0.8,
                                           mv_adjust_db=1.0, method='test',
                                           commentary={'found': 'a plateau', 'beqforge_revision': 'v9+src:d4'})]))
    wav = tmp_path / 'mono.wav'
    sf.write(str(wav), np.random.default_rng(2).normal(0, 0.05, (1000 * 5, 1)), 1000, subtype='PCM_24')

    design_and_queue(Session(AnalysisConfig(target_fs=1000)), 'one', str(wav), 'stamped', str(tmp_path / 'queue'))

    entry = read_entry(str(tmp_path / 'queue'), 'one')
    assert (entry.designer, entry.designer_build) == ('stamped', 'v9+src:d4')
