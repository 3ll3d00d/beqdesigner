'''
Contract 1.6's run record (design/designer-interface.md §3, "The run record"; design/redesign-scope.md step 1): a
response may name the designer's record of the run, relative to its shared root; the caller validates it, keeps it with
the design (and with it when the design becomes an earlier run), and never builds or parses one.
'''
import json

import numpy as np
import pytest
import soundfile as sf

from pipeline.designer.contract import BiquadSpec, DesignCandidate, DesignResponse
from pipeline.designer.convert import ContractViolation, validate_response
from pipeline.designer.http_binding import _response_from_json
from pipeline.review import QueueEntry, _entry_from_dict, _entry_to_dict, write_queue_entry

RECORD = 'records/designer-0123456789abcdef-89abcdef-v6.run.json.gz'
CANDIDATE = {'filters': [{'type': 'low_shelf', 'freq_hz': 20.0, 'gain_db': 6.0, 'q': 0.7}], 'confidence': 0.9,
             'mv_adjust_db': 6.0, 'method': 'fitted'}


def _designed(**fields):
    return DesignResponse('1.5', candidates=[DesignCandidate(filters=[BiquadSpec('low_shelf', 20.0, 6.0, 0.7)],
                                                             confidence=0.9, mv_adjust_db=6.0, method='fitted')],
                          **fields)


def test_a_1_6_response_names_its_run_record_on_either_shape():
    designed = _response_from_json({'contract_version': '1.5', 'candidates': [CANDIDATE], 'run_record': RECORD})
    declined = _response_from_json({'contract_version': '1.5', 'decline_reason': 'no_usable_plateau',
                                    'run_record': RECORD})
    older = _response_from_json({'contract_version': '1.5', 'candidates': [CANDIDATE]})

    assert designed.run_record == RECORD and declined.run_record == RECORD and older.run_record is None
    for response in (designed, declined, older):
        validate_response(response)


@pytest.mark.parametrize('path', ['/work/records/x.run.json.gz', '\\\\ZALAGA-UNRAID\\beq\\work\\records\\x.run.json.gz',
                                  'records\\x.run.json.gz', '../records/x.gz', 'records/../../x.gz', 'records/..',
                                  'C:/records/x.gz', '', '  ', 3])
def test_a_run_record_that_is_not_relative_to_the_shared_root_is_refused(path):
    with pytest.raises(ContractViolation, match='run_record'):
        validate_response(_designed(run_record=path))


@pytest.mark.parametrize('path', [RECORD, 'x.run.json.gz', 'records/a..b.run.json.gz'])
def test_a_relative_run_record_is_accepted(path):
    validate_response(_designed(run_record=path))


def _entry(**fields):
    return QueueEntry(id='t', fs=1000, meta={}, curve={'x': [1.0]}, **fields)


def test_the_run_record_is_written_only_when_known_so_an_older_reader_never_meets_it():
    assert 'run_record' not in _entry_to_dict(_entry())
    written = json.loads(json.dumps(_entry_to_dict(_entry(run_record=RECORD))))
    assert written['run_record'] == RECORD
    assert _entry_from_dict(written).run_record == RECORD


@pytest.mark.parametrize('declines', [False, True])
def test_the_run_record_reaches_the_queue_entry(tmp_path, declines):
    from pipeline.designer.registry import register_designer, unregister_designer
    from pipeline.orchestrate import Session
    from pipeline.review import design_and_queue, read_entry

    def designer(request):
        if declines:
            return DesignResponse('1.5', decline_reason='no_usable_plateau', design_version=6, run_record=RECORD)
        return _designed(design_version=6, run_record=RECORD)

    wav = str(tmp_path / 'mono.wav')
    sf.write(wav, np.random.default_rng(0).normal(0, 0.05, (1000 * 5, 1)), 1000, subtype='PCM_24')
    register_designer('test.run_record', designer)
    try:
        design_and_queue(Session(), 't', wav, 'test.run_record', str(tmp_path / 'queue'))
    finally:
        unregister_designer('test.run_record')

    assert read_entry(str(tmp_path / 'queue'), 't').run_record == RECORD


def test_an_earlier_design_run_keeps_its_run_record_and_using_it_brings_it_back(tmp_path):
    from pipeline.library.variants import kept_variants, use_variant
    from pipeline.review import CandidateSummary, read_entry
    queue = str(tmp_path)
    candidate = CandidateSummary(filters=[], confidence=0.9, method='fitted', mv_adjust_db=0.0)
    first = _entry(candidates=[candidate], variant_id='v1', run_record='records/first-v5.run.json.gz')
    second = _entry(candidates=[candidate], variant_id='v2', run_record=RECORD, variants=kept_variants(first))
    write_queue_entry(queue, second)

    assert second.variants[0]['run_record'] == 'records/first-v5.run.json.gz'
    swapped = use_variant(queue, 't', 'v1')
    assert swapped.run_record == 'records/first-v5.run.json.gz'
    assert read_entry(queue, 't').variants[-1]['run_record'] == RECORD


def test_an_earlier_run_kept_before_1_6_has_no_run_record(tmp_path):
    from pipeline.library.variants import use_variant
    from pipeline.review import CandidateSummary
    candidate = CandidateSummary(filters=[], confidence=0.9, method='fitted', mv_adjust_db=0.0)
    old_run = {'variant_id': 'v1', 'fs': 1000, 'curve': {'x': [1.0]}, 'candidates': [], 'rejected': []}
    write_queue_entry(str(tmp_path), _entry(candidates=[candidate], variant_id='v2', run_record=RECORD,
                                            variants=[old_run]))
    assert use_variant(str(tmp_path), 't', 'v1').run_record is None


def test_a_design_a_person_edited_has_no_run_record(tmp_path):
    from pipeline.library.variants import save_edited
    from pipeline.review import CandidateSummary
    candidate = CandidateSummary(filters=[{'_type': 'LowShelf', 'fs': 1000, 'fc': 20.0, 'gain': 6.0, 'q': 0.7}],
                                 confidence=0.9, method='fitted', mv_adjust_db=6.0)
    write_queue_entry(str(tmp_path), _entry(candidates=[candidate], variant_id='v1', run_record=RECORD))
    edited = save_edited(str(tmp_path), 't', 0, [BiquadSpec('low_shelf', 20.0, 3.0, 0.7)])
    assert edited.run_record is None and edited.variants[0]['run_record'] == RECORD
