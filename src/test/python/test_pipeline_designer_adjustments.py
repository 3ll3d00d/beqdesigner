'''
Contract 1.5's adjustments on the caller's side (design/designer-interface.md §2 "Adjustments", §3 "Adjustment outcomes",
§7.1; design/web-parity.md WP6): sent only when there are some, and then as 1.5; sent over HTTP only to a designer whose
/health says 1.5; each answer checked against what was sent; the adjustments and answers kept on the queue entry; and the
adjustments part of the design's fingerprint, so a redesign asks the same again and an answer to other adjustments is
never taken for this one.
'''
import json
import os

import pytest
from jsonschema import Draft202012Validator

from pipeline.config import AnalysisConfig
from pipeline.designer import http_binding
from pipeline.designer.contract import AdjustmentOutcome, BiquadSpec, DesignAdjustment, DesignCandidate, DesignResponse, \
    build_request
from pipeline.designer.convert import ContractViolation, validate_outcomes
from pipeline.designer.http_binding import DesignerUnavailable, _request_to_json, http_designer, \
    reported_adjustment_kinds
from pipeline.designer.registry import register_designer, unregister_designer
from pipeline.library.design_cache import design_fingerprint, design_if_needed
from pipeline.library.source import LibraryItem
from pipeline.orchestrate import Session
from pipeline.review import design_and_queue, read_entry, update_entry
from test_pipeline_orchestrate import _write_mono_wav

import numpy as np

SCHEMA = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'docs', 'schema', 'http_designer_request.schema.json')
BAND = DesignAdjustment('a1', 'exclude_band', {'low_hz': 28.0, 'high_hz': 34.0})
TILT = DesignAdjustment('a2', 'goal_tilt', {'db_per_octave': 1.0}, 'a little more weight')
DESIGN = DesignCandidate(filters=[BiquadSpec(type='low_shelf', freq_hz=18.0, gain_db=4.5, q=0.7)], confidence=0.6,
                         mv_adjust_db=4.0, method='fitted')


def _request(*adjustments):
    return build_request(np.zeros(8), 1000, adjustments=adjustments)


# --- the request ----------------------------------------------------------------------------------------------------------

def test_a_request_without_adjustments_is_as_it_was_and_one_with_them_says_1_5():
    assert _request().contract_version == '1.1' and _request().adjustments is None
    assert 'adjustments' not in _request_to_json(_request())
    body = _request_to_json(_request(BAND, TILT))
    assert body['contract_version'] == '1.5'
    assert body['adjustments'] == [{'id': 'a1', 'kind': 'exclude_band', 'params': {'low_hz': 28.0, 'high_hz': 34.0},
                                    'text': None},
                                   {'id': 'a2', 'kind': 'goal_tilt', 'params': {'db_per_octave': 1.0},
                                    'text': 'a little more weight'}]
    with open(SCHEMA) as f:
        Draft202012Validator(json.load(f)).validate(body)


@pytest.mark.parametrize('adjustments', [(BAND, DesignAdjustment('a1', 'floor', {'hz': 8})),
                                         (DesignAdjustment('', 'floor', {}),), (DesignAdjustment('x', 'floor', [8]),)])
def test_adjustments_need_an_id_each_once_a_kind_and_params(adjustments):
    with pytest.raises(ValueError):
        _request(*adjustments)


# --- the answers ----------------------------------------------------------------------------------------------------------

def _answered(*outcomes):
    return DesignResponse(contract_version='1.5', candidates=[DESIGN], adjustment_outcomes=list(outcomes))


def test_answers_that_keep_to_the_contract_pass():
    validate_outcomes(['a1', 'a2'], _answered(AdjustmentOutcome('a1', 'honoured'),
                                               AdjustmentOutcome('a2', 'adapted', {'db_per_octave': 0.5}, 'capped')))
    validate_outcomes(['a1', 'a2'], _answered(AdjustmentOutcome('a1', 'declined', reason='no evidence there')))
    validate_outcomes(['a1'], DesignResponse(contract_version='1.1', candidates=[DESIGN]))   # a 1.4 designer: none


@pytest.mark.parametrize('outcome, says', [
    (AdjustmentOutcome('zz', 'honoured'), 'did not send'),
    (AdjustmentOutcome('a1', 'ignored', reason='x'), 'must be one of'),
    (AdjustmentOutcome('a1', 'declined'), 'must say why'),
    (AdjustmentOutcome('a1', 'adapted', reason='x'), 'what was applied'),
    (AdjustmentOutcome('a1', 'not_understood', {'x': 1}, 'x'), 'nothing was applied')])
def test_an_answer_that_breaks_the_contract_is_refused(outcome, says):
    with pytest.raises(ContractViolation, match=says):
        validate_outcomes(['a1'], _answered(outcome))


def test_an_adjustment_answered_twice_is_refused():
    with pytest.raises(ContractViolation, match='a second time'):
        validate_outcomes(['a1'], _answered(AdjustmentOutcome('a1', 'honoured'), AdjustmentOutcome('a1', 'honoured')))


# --- over HTTP ------------------------------------------------------------------------------------------------------------

class _Answer:
    def __init__(self, body, status=200):
        self._body, self.status_code, self.ok = body, status, status < 400

    def raise_for_status(self):
        if not self.ok:
            import requests
            raise requests.HTTPError(str(self.status_code))

    def json(self):
        return self._body


def test_adjustments_go_only_to_a_designer_that_says_1_5_and_its_answers_come_back(monkeypatch):
    health, posted = {'contract_version': '1.4', 'shared_root': False}, []
    monkeypatch.setattr(http_binding.requests, 'get', lambda url, **kw: _Answer(health))

    def post(url, json=None, **kw):
        posted.append(json)
        return _Answer({'contract_version': '1.5', 'candidates': [{
            'filters': [{'type': 'low_shelf', 'freq_hz': 18.0, 'gain_db': 4.5, 'q': 0.7}], 'confidence': 0.6,
            'mv_adjust_db': 4.0, 'method': 'fitted'}],
            'adjustment_outcomes': [{'id': 'a1', 'outcome': 'adapted', 'applied': {'low_hz': 29, 'high_hz': 33},
                                     'reason': 'narrowed to where the hum is'}]})
    monkeypatch.setattr(http_binding.requests, 'post', post)
    with pytest.raises(DesignerUnavailable, match='does not take adjustments'):
        http_designer('http://designer/design')(_request(BAND))
    assert posted == []
    assert http_designer('http://designer/design')(_request()).candidates   # none asked: sent as before
    health['contract_version'] = '1.5'
    response = http_designer('http://designer/design')(_request(BAND))
    assert posted[-1]['adjustments'][0]['id'] == 'a1'
    assert response.adjustment_outcomes == [AdjustmentOutcome('a1', 'adapted', {'low_hz': 29, 'high_hz': 33},
                                                              'narrowed to where the hum is')]


def test_the_kinds_a_designer_lists_are_read_from_its_health():
    kinds = {'floor': {'description': 'restore nothing below', 'params': {'hz': {'type': 'number'}}},
             'broken': 'not an object', 'nameless': {'params': {}}}
    assert set(reported_adjustment_kinds({'contract_version': '1.5', 'adjustment_kinds': kinds})) == {'floor'}
    assert reported_adjustment_kinds({'contract_version': '1.4', 'adjustment_kinds': kinds}) == {}
    assert reported_adjustment_kinds(None) == {}


# --- kept on the entry, and part of the design ---------------------------------------------------------------------------

ADJUSTED = 'test.adjustments.answers'
OLD = 'test.adjustments.before_1_5'


@pytest.fixture(autouse=True)
def _designers():
    asked = []

    def answers(request):
        asked.append(request.adjustments)
        return DesignResponse(contract_version=request.contract_version, candidates=[DESIGN], adjustment_outcomes=[
            AdjustmentOutcome(a.id, 'honoured') if a.kind != 'note' else AdjustmentOutcome(a.id, 'not_understood', reason='?')
            for a in request.adjustments or ()] or None)
    register_designer(ADJUSTED, answers)
    register_designer(OLD, lambda request: DesignResponse(contract_version='1.1', candidates=[DESIGN]))
    yield asked
    unregister_designer(ADJUSTED)
    unregister_designer(OLD)


STORED = [{'id': 'a1', 'kind': 'exclude_band', 'params': {'low_hz': 28.0, 'high_hz': 34.0}, 'text': None, 'by': 'web',
           'at': '2026-10-21T10:00:00Z', 'reason': 'hum'},
          {'id': 'a2', 'kind': 'note', 'params': {}, 'text': 'the logo is not the film'}]


def test_the_adjustments_and_answers_are_kept_on_the_entry(tmp_path, _designers):
    wav = str(tmp_path / 'mono.wav')
    _write_mono_wav(wav)
    entry = design_and_queue(Session(), 'heat', wav, ADJUSTED, str(tmp_path / 'q'), adjustments=STORED)
    assert [a.kind for a in _designers[-1]] == ['exclude_band', 'note']           # the bookkeeping is not sent
    assert entry.adjustments == STORED and read_entry(str(tmp_path / 'q'), 'heat').adjustments == STORED
    assert entry.adjustment_outcomes == [{'id': 'a1', 'outcome': 'honoured', 'applied': None, 'reason': None},
                                         {'id': 'a2', 'outcome': 'not_understood', 'applied': None, 'reason': '?'}]
    old = design_and_queue(Session(), 'old', wav, OLD, str(tmp_path / 'q'), adjustments=STORED)
    assert old.adjustments == STORED and old.adjustment_outcomes == []           # not acknowledged


def test_adjustments_are_part_of_the_fingerprint_and_none_leaves_it_as_it_was(tmp_path):
    item = LibraryItem(id='t', source_path='/m/t.mkv', display_name='t', fingerprint='s')
    plain = design_fingerprint(item, 'd', AnalysisConfig(), 'complete_programme')
    assert design_fingerprint(item, 'd', AnalysisConfig(), 'complete_programme', adjustments=[]) == plain
    adjusted = design_fingerprint(item, 'd', AnalysisConfig(), 'complete_programme', adjustments=STORED)
    assert adjusted != plain
    reordered = [dict(STORED[1], by='someone else'), STORED[0]]                  # order and bookkeeping do not count
    assert design_fingerprint(item, 'd', AnalysisConfig(), 'complete_programme', adjustments=reordered) == adjusted


def test_a_redesign_asks_the_same_adjustments_again_and_is_then_current(tmp_path, _designers, monkeypatch):
    wav = str(tmp_path / 'mono.wav')
    _write_mono_wav(wav)
    queue = str(tmp_path / 'q')
    source = tmp_path / 'source.mkv'
    source.write_bytes(b'x')
    item = LibraryItem(id='heat', source_path=str(source), display_name='Heat', fingerprint='s')
    design_if_needed(Session(), item, wav, ADJUSTED, queue, AnalysisConfig())
    update_entry(queue, 'heat', adjustments=STORED, design_fingerprint=None)    # what a revise with adjustments leaves
    result = design_if_needed(Session(), item, wav, ADJUSTED, queue, AnalysisConfig())
    assert result.designed and [a.id for a in _designers[-1]] == ['a1', 'a2']
    assert read_entry(queue, 'heat').adjustments == STORED
    again = design_if_needed(Session(), item, wav, ADJUSTED, queue, AnalysisConfig())
    assert not again.designed                                                     # current, with its adjustments


def test_the_kinds_offered_are_those_every_endpoint_up_lists():
    from pipeline.service.designer import EndpointState, shared_kinds
    floor = {'description': 'restore nothing below'}
    tilt = {'description': 'tilt the goal'}
    both = EndpointState('http://a', 1, True, 'a', None, {'floor': floor, 'goal_tilt': tilt})
    one = EndpointState('http://b', 1, True, 'b', None, {'floor': floor})
    assert shared_kinds([both, one]) == {'floor': floor}
    assert shared_kinds([both]) == {'floor': floor, 'goal_tilt': tilt}
    assert shared_kinds([both, EndpointState('http://c', 1, True, 'c')]) == {} and shared_kinds([]) == {}
