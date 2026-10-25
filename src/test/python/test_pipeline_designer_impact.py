'''
pipeline/designer/impact.py -- contract 1.6's /impact client (design/designer-interface.md §7.1 "Impact";
design/redesign-scope.md step 3), against a fake designer over real HTTP (fake_impact_designer).
'''
import pytest

from fake_impact_designer import CHANGE, FakeImpactDesigner, response_errors
from pipeline.designer.identity import CurrentIdentity, EndpointIdentity, identity_from_health
from pipeline.designer.impact import MAX_RECORDS, ImpactError, ask_impact, impact_endpoint, impact_url, no_record

V7 = CurrentIdentity(7, 's0')


def _records(n, prefix='records/r'):
    return [(f't{i}', f'{prefix}{i}-v6.run.json.gz') for i in range(n)]


def test_impact_is_asked_on_the_design_endpoints_origin():
    assert impact_url('http://10.151.167.10:8420/design') == 'http://10.151.167.10:8420/impact'


def test_health_says_whether_a_designer_answers_impact():
    assert identity_from_health('u', {'design_version': 7, 'impact': True}).impact is True
    assert identity_from_health('u', {'design_version': 7}).impact is False
    assert identity_from_health('u', {'design_version': 7, 'impact': 'yes'}).impact is False


@pytest.mark.parametrize('endpoints, current, chosen', [
    ([EndpointIdentity('a', 7, 's0', impact=True)], V7, 'a'),
    ([EndpointIdentity('a', 7, 's0'), EndpointIdentity('b', 7, 's0', impact=True)], V7, 'b'),
    ([EndpointIdentity('a', 7, 's1', impact=True)], V7, None),   # other settings: its answer is not this identity's
    ([EndpointIdentity('a', 6, 's0', impact=True)], V7, None),
    ([EndpointIdentity('a', 7, 's0', build='x', impact=True)], V7, 'a'),   # the build is never compared
    ([EndpointIdentity('a', 7, 's0', impact=True)], CurrentIdentity(7, 's0', disagreement=('x', 'y')), None),
    ([None, EndpointIdentity('a', 7, 's0', impact=True)], V7, 'a'),
    ([EndpointIdentity('a', 7, 's0', impact=True)], None, None),
])
def test_the_designer_asked_answers_impact_at_the_identity_a_redesign_would_use(endpoints, current, chosen):
    found = impact_endpoint(endpoints, current)
    assert (found.url if found else None) == chosen


def test_every_record_is_answered_in_the_order_asked_with_its_reasons():
    tiers = {'records/r1-v6.run.json.gz': 'will_move', 'records/r2-v6.run.json.gz': 'likely',
             'records/r3-v6.run.json.gz': 'possible'}
    with FakeImpactDesigner(tiers=tiers) as designer:
        answer = ask_impact(designer.url, _records(4))

    assert [r.id for r in answer.results] == ['t0', 't1', 't2', 't3']
    assert [r.tier for r in answer.results] == ['unaffected', 'will_move', 'likely', 'possible']
    assert answer.results[1].reasons[0].change == 'fit-choice' and answer.results[0].reasons == ()
    assert (answer.design_version, answer.design_settings) == (7, 's0')
    assert [c.id for c in answer.changes] == [CHANGE['id']]
    assert designer.asked[0]['records'][1] == {'id': 't1', 'path': 'records/r1-v6.run.json.gz'}


def test_a_library_goes_in_requests_of_at_most_500_one_after_another_and_comes_back_as_one():
    seen = []
    with FakeImpactDesigner() as designer:
        answer = ask_impact(designer.url, _records(1201), on_batch=lambda done, of: seen.append((done, of)))
    assert [len(b['records']) for b in designer.asked] == [MAX_RECORDS, MAX_RECORDS, 201]
    assert len(answer.results) == 1201 and len(answer.changes) == 1
    assert seen == [(500, 1201), (1000, 1201), (1201, 1201)]


def test_a_record_the_designer_cannot_read_is_unknown_not_an_error():
    with FakeImpactDesigner(held={'records/r0-v6.run.json.gz'}) as designer:
        answer = ask_impact(designer.url, _records(2))
    assert [r.tier for r in answer.results] == ['unaffected', 'unknown']
    assert answer.results[1].score is None and answer.results[1].reasons[0].detail == 'record not found'
    assert (answer.results[0].unknown_reason, answer.results[1].unknown_reason) == (None, 'not_found')


def test_the_fakes_answer_is_in_the_contracts_shape():
    tiers = {'records/r1-v6.run.json.gz': 'likely'}
    with FakeImpactDesigner(tiers=tiers, held={'records/r0-v6.run.json.gz', 'records/r1-v6.run.json.gz'}) as designer:
        answer = designer.answer({'records': [{'id': i, 'path': p} for i, p in _records(3)]})
    assert response_errors(answer) == []
    assert response_errors({**answer, 'results': [{**answer['results'][1], 'unknown_reason': 'not_found'}]})


def test_a_change_says_what_counts_as_moved_and_what_its_scores_were_measured_on():
    with FakeImpactDesigner() as designer:
        answer = ask_impact(designer.url, _records(1))
    change = answer.changes[0]
    assert change.moved == CHANGE['moved']
    assert [(b.cls, b.measured, b.moved) for b in change.basis] == [('declined_after_fit', 34, 18),
                                                                      ('designed_all_passing', 15, 2)]


def test_a_change_without_a_basis_is_still_in_shape():
    with FakeImpactDesigner() as designer:
        designer.mangle = lambda a: {**a, 'changes': [{k: CHANGE[k] for k in ('design_version', 'id', 'summary')}]}
        change = ask_impact(designer.url, _records(1)).changes[0]
    assert (change.moved, change.basis) == (None, ())


def test_the_scores_sum_to_the_expected_movers_an_unknown_counting_as_one():
    tiers = {'records/r0-v6.run.json.gz': 'likely', 'records/r1-v6.run.json.gz': 'possible'}
    with FakeImpactDesigner(tiers=tiers, held=set(tiers)) as designer:
        answer = ask_impact(designer.url, _records(3))
    assert sum(r.expected for r in answer.results) == pytest.approx(0.7 + 0.2 + 1.0)
    assert no_record('t').expected == 1.0


def test_a_person_reads_will_move_and_unknown_first_then_by_score():
    tiers = {'records/r0-v6.run.json.gz': 'possible', 'records/r1-v6.run.json.gz': 'likely',
             'records/r2-v6.run.json.gz': {'tier': 'likely', 'score': 0.9, 'from_design_version': 6,
                                           'reasons': [{'change': None, 'feature': None, 'detail': 'x'}]},
             'records/r3-v6.run.json.gz': 'will_move'}
    with FakeImpactDesigner(tiers=tiers, held=set(tiers) - {'records/r4-v6.run.json.gz'}) as designer:
        answer = ask_impact(designer.url, _records(5))
    assert [r.id for r in answer.ranked()] == ['t3', 't4', 't2', 't1', 't0']


def test_a_design_with_no_run_record_is_unknown_without_asking():
    result = no_record('t9')
    assert (result.tier, result.score, result.rank, result.unknown_reason) == ('unknown', None, 0, 'no_record')


@pytest.mark.parametrize('mangle, says', [
    (lambda a: {**a, 'results': a['results'][::-1]}, 'order asked'),
    (lambda a: {**a, 'results': a['results'][:1]}, 'one result per record'),
    (lambda a: {**a, 'contract_version': '1.5'}, 'contract'),
    (lambda a: {**a, 'design_version': None}, 'design_version'),
    (lambda a: {**a, 'changes': None}, 'changes'),
    (lambda a: {**a, 'changes': [{'id': 'x'}]}, 'change is out of shape'),
    (lambda a: {**a, 'results': [{**r, 'tier': 'maybe'} for r in a['results']]}, 'tier'),
    (lambda a: {**a, 'results': [{**r, 'score': 1.5} for r in a['results']]}, 'score'),
    (lambda a: {**a, 'results': [{**r, 'score': None} for r in a['results']]}, 'score'),
    (lambda a: {**a, 'results': [{**r, 'tier': 'unknown', 'score': 0.5} for r in a['results']]}, 'unknown but scored'),
    (lambda a: {**a, 'results': [{**r, 'tier': 'likely', 'reasons': []} for r in a['results']]}, 'no reasons'),
    (lambda a: {**a, 'results': [{**r, 'tier': 'likely', 'reasons': [{'change': 'x'}]} for r in a['results']]},
     'reason out of shape'),
    (lambda a: {**a, 'results': [{**r, 'from_design_version': -1} for r in a['results']]}, 'from_design_version'),
    (lambda a: {**a, 'results': [{**r, 'unknown_reason': 'not_found'} for r in a['results']]}, 'unknown_reason'),
    (lambda a: {**a, 'results': [{**r, 'tier': 'unknown', 'score': None, 'unknown_reason': '',
                                  'reasons': [{'change': None, 'feature': None, 'detail': 'x'}]} for r in a['results']]},
     'unknown_reason'),
    (lambda a: {**a, 'changes': [{**CHANGE, 'moved': 3}]}, 'has moved'),
    (lambda a: {**a, 'changes': [{**CHANGE, 'basis': {}}]}, 'has basis'),
    (lambda a: {**a, 'changes': [{**CHANGE, 'basis': [{'class': 'x', 'measured': 3, 'moved': 4}]}]}, 'basis out of'),
    (lambda a: {**a, 'changes': [{**CHANGE, 'basis': [{'class': 'x', 'measured': True, 'moved': 0}]}]}, 'basis out of'),
    (lambda a: {**a, 'changes': [{**CHANGE, 'basis': [{'measured': 3, 'moved': 1}]}]}, 'basis out of'),
])
def test_an_answer_out_of_the_contracts_shape_is_an_error_not_a_guess(mangle, says):
    with FakeImpactDesigner() as designer:
        designer.mangle = mangle
        with pytest.raises(ImpactError, match=says):
            ask_impact(designer.url, _records(2))


def test_a_refusal_says_the_designers_reason():
    with FakeImpactDesigner() as designer:
        designer.status = 422
        with pytest.raises(ImpactError, match='422.*no shared root'):
            ask_impact(designer.url, _records(1))


def test_an_identity_that_changes_between_requests_is_an_error():
    with FakeImpactDesigner() as designer:
        calls = []

        def mangle(answer):
            calls.append(1)
            return {**answer, 'design_version': 7 + len(calls) - 1}
        designer.mangle = mangle
        with pytest.raises(ImpactError, match='identity changed'):
            ask_impact(designer.url, _records(MAX_RECORDS + 1))


def test_a_designer_that_cannot_be_reached_is_an_error():
    with pytest.raises(ImpactError, match='failed'):
        ask_impact('http://127.0.0.1:9/design', _records(1), timeout=2)


@pytest.mark.parametrize('records', [[], [('a', 'r1'), ('a', 'r2')]])
def test_no_records_or_a_shared_id_is_refused_before_asking(records):
    with pytest.raises(ImpactError):
        ask_impact('http://127.0.0.1:9/design', records)
